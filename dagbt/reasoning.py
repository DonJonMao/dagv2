"""Bounded, source-preserving evidence reasoning adapted from BridgeTree 16809bd.

This protocol applies to the fusion planner/evidence/solver only. Original DAG
and final reader requests keep their existing transport and token accounting.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json

from .budget import BudgetExceeded
from .evidence_protocol import EvidenceJSONError, parse_evidence_object
from .model_runtime import (REASONING_MARKER, count_evidence_request_tokens,
                            evidence_token_accounting, estimate_evidence_tokens)
from .transport import call_reservation


class InputOverflow(ValueError):
    pass


class ProtocolError(ValueError):
    def __init__(self, message, *, category="schema", metadata=None, response_ref=None):
        super().__init__(message)
        self.category = category
        self.metadata = deepcopy(metadata or {})
        self.response_ref = response_ref


class OutputTruncated(ProtocolError):
    def __init__(self, message="Provider explicitly truncated reasoning output", **kwargs):
        super().__init__(message, category="output_truncated", **kwargs)


class RefusalError(ProtocolError):
    def __init__(self, message="Provider refused reasoning response", **kwargs):
        super().__init__(message, category="refusal", **kwargs)


class EmptyResponseError(ProtocolError):
    def __init__(self, message="Reasoning response has no text content", **kwargs):
        super().__init__(message, category="empty_response", **kwargs)


class Reasoner:
    def __init__(self, calls, tokenizer, config, settings, ledger, event):
        self.calls, self.tokenizer, self.config = calls, tokenizer, config
        self.settings, self.ledger, self.event = settings, ledger, event
        self.sequence = 0
        self.requests = []
        self.last_response = None

    @property
    def url(self):
        return self.config["llm_base_url"].rstrip("/") + "/chat/completions"

    def _payload(self, operation, system, data, schema=None):
        mode = self.settings.get("response_format", "plain")
        if mode not in {"plain", "json_object", "json_schema"}:
            raise ValueError("response_format must be plain, json_object, or json_schema")
        schema = deepcopy(schema) if schema is not None else {"type": "object"}
        if not isinstance(schema, Mapping) or schema.get("type") != "object":
            raise ValueError("Reasoning response schema must describe an object")
        # In plain/object mode the schema remains a prompt instruction, never
        # an invented provider-side constrained decoding guarantee.
        schema_text = json.dumps(schema, ensure_ascii=False, sort_keys=True, allow_nan=False)
        if mode != "json_schema":
            system += "\n\nReturn one complete JSON object matching this schema:\n" + schema_text
        payload = {
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": json.dumps(data, ensure_ascii=False, allow_nan=False)}],
            "max_tokens": self.settings["reasoning_output_tokens"],
            REASONING_MARKER: True,
        }
        if mode == "json_object":
            payload["response_format"] = {"type": "json_object"}
        elif mode == "json_schema":
            payload["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "dagbt_evidence", "strict": True, "schema": dict(schema)}}
        return payload

    def estimate(self, operation, system, data, schema=None):
        """Estimated input tokens for exact wire JSON, including schema fields."""
        payload = self._payload(operation, system, data, schema)
        return count_evidence_request_tokens((operation, str(self.sequence + 1)), self.url, payload, self.config)

    def _preflight(self, operation, count, *, reserve=None, extra_reserve=0):
        output = self.settings["reasoning_output_tokens"]
        margin = self.settings.get("input_margin", 256)
        if count + output + 8 + margin > self.settings["context_tokens"]:
            raise InputOverflow(f'{operation}: {count}+{output}+8+{margin} exceeds '
                                f'{self.settings["context_tokens"]} (estimated tokens)')
        reserved = call_reservation(self.settings, operation, reserve, extra_reserve)
        if self.ledger.remaining("llm") <= reserved:
            raise BudgetExceeded("llm", operation, 1, 0)

    def request(self, operation, system, data, validate=None, schema=None, *, reserve=None, extra_reserve=0):
        """One metered request and strict parse; local mapping owns its recovery.

        Transport keeps its finite physical HTTP retry policy. This method
        never retries a model response or changes its requested output mode.
        ``last_response``/``requests`` retain raw output and provider metadata.
        """
        payload = self._payload(operation, system, data, schema)
        count = count_evidence_request_tokens((operation, str(self.sequence + 1)), self.url, payload, self.config)
        self._preflight(operation, count, reserve=reserve, extra_reserve=extra_reserve)
        self.sequence += 1
        stage = (operation, str(self.sequence))
        record = {"operation": operation, "call_index": self.sequence, "messages": deepcopy(payload["messages"]),
                  "logical_call_index": self.sequence,
                  "response_format": self.settings.get("response_format", "plain"),
                  "input_tokens_local": count, "output_token_reserve": payload["max_tokens"],
                  "input_margin": self.settings.get("input_margin", 256), **evidence_token_accounting()}
        self.requests.append(record)
        self.last_response = record
        self.event({"event": "reasoning_request_started", "operation": operation,
                    "logical_call_index": self.sequence,
                    "accounting_note": "One logical model request; Transport meters each physical retry against llm budget"})
        try:
            # Old explicit offline injection clients retain their three-arg
            # contract unless this request needs additional reservations.
            options = {} if reserve is None and not extra_reserve else {
                "reserve": reserve, "extra_reserve": extra_reserve}
            response = self.calls.get(stage, self.url, payload, **options)
        except BaseException as exc:
            record.update(error_type=type(exc).__name__, failure_category="service_or_budget")
            raise
        body = response.get("response") if isinstance(response, Mapping) else None
        body = body if isinstance(body, Mapping) else {}
        choices = body.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], Mapping) else {}
        message = choice.get("message") if isinstance(choice.get("message"), Mapping) else {}
        raw = message.get("content")
        metadata = {"finish_reason": choice.get("finish_reason"), "refusal": message.get("refusal"),
                    "usage": body.get("usage") if isinstance(body.get("usage"), Mapping) else None,
                    "response_id": body.get("id"), "protocol": record["response_format"]}
        response_ref = response.get("response_ref") if isinstance(response, Mapping) else None
        record.update(raw_output=raw, response_metadata=metadata, response_ref=response_ref,
                      output_tokens_estimate=estimate_evidence_tokens(raw) if isinstance(raw, str) else 0)
        self.event({"event": "reasoning_response", **deepcopy(record), **metadata})
        context = {"metadata": metadata, "response_ref": response_ref}
        try:
            if metadata["finish_reason"] == "length":
                raise OutputTruncated(**context)
            if metadata["refusal"] or metadata["finish_reason"] == "content_filter":
                raise RefusalError(**context)
            if not isinstance(raw, str) or not raw.strip():
                raise EmptyResponseError(**context)
            value = parse_evidence_object(raw)
            result = validate(value) if validate is not None else value
        except (ValueError, KeyError, TypeError) as error:
            if isinstance(error, ProtocolError):
                exc = error
                exc.metadata = deepcopy(metadata)
                exc.response_ref = response_ref
            else:
                category = error.category if isinstance(error, EvidenceJSONError) else "schema"
                exc = ProtocolError(str(error), category=category, **context)
            record.update(validation_status="invalid", validation_error=str(exc), failure_category=exc.category)
            self.event({"event": "protocol_error", "operation": operation, "error": str(exc),
                        "failure_category": exc.category, "response_ref": response_ref, **metadata})
            raise exc from error if exc is not error else None
        record["validation_status"] = "valid"
        return result

    def repair_reservation(self, operation, reserve_repairs=None):
        """Reserve global repairs for the final legal selection and coverage."""
        if reserve_repairs is None:
            reserve_repairs = (0 if operation.startswith(("select", "reader")) else
                               self.settings.get("reserved_selection_repairs", 0))
        if isinstance(reserve_repairs, bool) or not isinstance(reserve_repairs, int) or reserve_repairs < 0:
            raise ValueError("reserve_repairs must be a nonnegative integer")
        return reserve_repairs

    def json(self, operation, system, data, validate, schema=None, *, reserve=None, extra_reserve=0,
             repair_builder=None, reserve_repairs=None):
        """Bounded local repair, preserving original evidence and allowed IDs."""
        original = deepcopy(data)
        current = deepcopy(data)
        local_repairs = 0
        repair_reserve = self.repair_reservation(operation, reserve_repairs)
        while True:
            try:
                return self.request(operation, system, current, validate, schema,
                                    reserve=reserve, extra_reserve=extra_reserve)
            except (OutputTruncated, RefusalError):
                # Mapping can split a truncated batch itself. Refusal is not
                # treated as a broken JSON enum and never causes blind repair.
                raise
            except ProtocolError as exc:
                if (local_repairs >= self.settings.get("max_repairs_per_request", 2)
                        or self.ledger.remaining("json_repairs") <= repair_reserve):
                    self.event({"event": "reasoning_repair_exhausted", "operation": operation,
                                "error_type": type(exc).__name__, "error": str(exc),
                                "failure_category": exc.category, "local_repairs": local_repairs,
                                "reserved_repairs": repair_reserve,
                                "remaining_repairs": self.ledger.remaining("json_repairs")})
                    raise
                feedback = ("Use only original source evidence and allowed IDs. "
                            "Correct this validation error: " + str(exc))[:1200]
                state = {"original_data": deepcopy(original), "error": str(exc),
                         "error_type": type(exc).__name__, "failure_category": exc.category,
                         "repair_index": local_repairs + 1}
                current = (repair_builder(state) if repair_builder is not None else
                           deepcopy(original) if isinstance(original, dict) else {"original_input": deepcopy(original)})
                if not isinstance(current, dict):
                    raise TypeError("repair_builder must return a payload object")
                current = deepcopy(current)
                while True:
                    current["validation_feedback"] = feedback
                    count = self.estimate(operation, system, current, schema)
                    try:
                        self._preflight(operation, count, reserve=reserve, extra_reserve=extra_reserve)
                    except InputOverflow as overflow:
                        if len(feedback) <= 80:
                            self.event({"event": "reasoning_repair_preflight_failed", "operation": operation,
                                        "error_type": type(overflow).__name__, "error": str(overflow),
                                        "failure_category": "repair_input_budget",
                                        "input_tokens_local": count, "original_error": str(exc),
                                        "local_repairs": local_repairs})
                            raise overflow from exc
                        feedback = feedback[:max(80, len(feedback) // 2)]
                        continue
                    break
                self.event({"event": "reasoning_repair_prepared", "operation": operation,
                            "repair_index": local_repairs + 1, "input_tokens_local": count,
                            "scoped_builder": repair_builder is not None,
                            "reserved_repairs": repair_reserve})
                # Count only repairs that have passed local budget checks.
                self.ledger.reserve("json_repairs", operation)
                local_repairs += 1
