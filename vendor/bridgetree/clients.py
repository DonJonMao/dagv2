from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Protocol, Sequence

import numpy as np

from .config import EmbeddingConfig, EndpointConfig, GeneratorConfig
from .diagnostic_identity import (
    canonical_request_payload, deployment_fingerprint, execution_hash, request_hash,
    validate_provider_params, validate_request_param_credentials,
)
from .request_audit import (
    TransportBudgetExceeded, current_audit_scope, document_descriptors,
    emit_request_event, logical_request_scope, new_request_id, request_audit_scope,
)
from .information import StateBasisProvider
from .math_utils import normalize_rows
from .types import (
    ContextPlan,
    Memory,
    _strict_float_value,
    _strict_int_value,
    context_plan_hash,
)

# Shared reader instruction for every retrieval method.  It is deliberately
# question-time aware and does not encode a keyword-specific latest-wins
# branch; historical state and reasons for change remain available when the
# question asks for them.
GENERATOR_SYSTEM_PROMPT = (
    "Answer the user using only relevant personal memories. Assess evidence "
    "for the time or period asked in the question. Earlier statements may be "
    "necessary for historical states or reasons for change. Distinguish user "
    "statements from assistant suggestions. Do not mention retrieval internals."
)
GENERATOR_USER_TEMPLATE = (
    "User query:\n{query}\n\nRetrieved personal memories (chronological):\n{context}"
    "\n\nAnswer options:\n{answer_options}\nReturn the best option label and a concise answer."
)


def estimate_tokens(text: str) -> int:
    """Deterministic tokenizer-independent accounting used for method matching."""
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def context_token_count(memories: Sequence[Memory]) -> int:
    return sum(estimate_tokens(memory.text) for memory in memories)


def fit_context_budget(memories: Sequence[Memory], token_budget: int) -> List[Memory]:
    if isinstance(memories, (str, bytes)) or not isinstance(memories, Sequence):
        raise ValueError("memories must be a sequence")
    token_budget = _strict_int_value(token_budget, "context token budget", nonnegative=True)
    selected: List[Memory] = []
    used = 0
    for memory in memories:
        tokens = estimate_tokens(memory.text)
        if selected and used + tokens > token_budget:
            continue
        if not selected and tokens > token_budget:
            continue
        selected.append(memory)
        used += tokens
    return selected


def build_generation_messages(
    query: str,
    memories: Sequence[Memory],
    answer_options: str = "",
) -> List[Dict[str, str]]:
    if not isinstance(query, str) or not isinstance(answer_options, str):
        raise ValueError("query and answer_options must be strings")
    if isinstance(memories, (str, bytes)) or not isinstance(memories, Sequence):
        raise ValueError("memories must be a sequence")
    context = "\n\n".join(
        f"[Memory {index}; source={memory.source_id}; time={memory.timestamp}]\n{memory.text}"
        for index, memory in enumerate(memories, start=1)
    )
    user_content = GENERATOR_USER_TEMPLATE.format(
        query=query,
        context=context,
        answer_options=answer_options,
    )
    return [
        {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


class ContextPlanError(ValueError):
    """Raised when an exact selected context cannot satisfy its budget."""


def build_context_plan(
    query: str,
    memories: Sequence[Memory],
    answer_options: str = "",
    *,
    token_budget: int | None = None,
    strict: bool = True,
    selected_ids: Sequence[str] | None = None,
    model: str = "",
    endpoint: str = "",
    temperature: float = 0.0,
    max_tokens: int | None = None,
    response_format: Any | None = None,
    request_params: Mapping[str, Any] | None = None,
    generator_config: GeneratorConfig | None = None,
) -> ContextPlan:
    """Build the one authoritative generation request without truncation."""

    # Validate the control flags before resolving a config.  In particular,
    # ``strict=1`` must not silently opt a caller into a different budget
    # contract, and NumPy booleans should not be accepted as a wire-level
    # boolean by accident.
    if not isinstance(strict, bool):
        raise ValueError("ContextPlan strict must be boolean")
    if isinstance(memories, (str, bytes)) or not isinstance(memories, Sequence):
        raise ValueError("ContextPlan memories must be a sequence")
    if not isinstance(query, str) or not isinstance(answer_options, str):
        raise ValueError("ContextPlan query and answer_options must be strings")
    if request_params is not None:
        # Explicit parameters retain the legacy allowance for canonical
        # fields with identical values, but never for nested credentials.
        validate_request_param_credentials(request_params)

    # Callers may provide a GeneratorConfig directly; explicit keyword values
    # remain useful for lightweight tests and custom HTTP adapters.  Resolve
    # this before hashing so the plan identity is exactly the final request
    # identity rather than a later client-side mutation.
    if generator_config is not None:
        def config_value(name: str, default: Any = None) -> Any:
            if isinstance(generator_config, Mapping):
                return generator_config.get(name, default)
            return getattr(generator_config, name, default)

        model_value = config_value("model", model)
        endpoint_value = config_value("endpoint", endpoint)
        if not isinstance(model_value, str) or not isinstance(endpoint_value, str):
            raise ValueError("GeneratorConfig model and endpoint must be strings")
        model = model_value
        endpoint = endpoint_value
        configured_params = validate_provider_params(config_value("provider_request_params", {}) or {})
        if request_params is not None:
            for key, value in request_params.items():
                if key in configured_params and configured_params[key] != value:
                    raise ValueError("explicit request_params conflict with configured provider parameters")
                configured_params[key] = value
        request_params = configured_params
        temperature = _strict_float_value(
            config_value("temperature", temperature),
            "ContextPlan temperature",
            nonnegative=True,
        )
        if max_tokens is None:
            max_tokens = _strict_int_value(
                config_value("max_tokens", 0),
                "ContextPlan max_tokens",
                nonnegative=True,
            )
        if token_budget is None:
            configured_budget = config_value("context_token_budget", None)
            token_budget = (
                None
                if configured_budget is None
                else _strict_int_value(
                    configured_budget,
                    "ContextPlan token_budget",
                    nonnegative=True,
                )
            )
    # Apply the same strict contract to explicit arguments.  Avoid ``int`` /
    # ``float`` coercions here: they turn booleans, fractions, and NaN into
    # plausible-looking request fields that later audits cannot distinguish
    # from the caller's intended values.
    temperature = _strict_float_value(
        temperature,
        "ContextPlan temperature",
        nonnegative=True,
    )
    max_tokens = (
        None
        if max_tokens is None
        else _strict_int_value(max_tokens, "ContextPlan max_tokens", nonnegative=True)
    )
    token_budget = (
        None
        if token_budget is None
        else _strict_int_value(token_budget, "ContextPlan token_budget", nonnegative=True)
    )

    # The reader contract always receives chronology, even if a selector
    # returns IDs in greedy order.  Ties use memory_id so the same selected
    # set produces one canonical request across methods and processes.
    def chronology_key(memory: Memory) -> tuple[float, str]:
        try:
            stamp = float(memory.timestamp)
            if not np.isfinite(stamp):
                stamp = float("inf")
        except (TypeError, ValueError):
            stamp = float("inf")
        return stamp, str(memory.memory_id)

    ordered = sorted(list(memories), key=chronology_key)
    chronological_ids = tuple(memory.memory_id for memory in ordered)
    if len(set(chronological_ids)) != len(chronological_ids):
        raise ValueError("ContextPlan memory IDs must be unique")
    if selected_ids is not None:
        supplied_ids = tuple(str(value) for value in selected_ids)
        if len(supplied_ids) != len(set(supplied_ids)):
            raise ValueError("ContextPlan selected_ids must be unique")
        if set(supplied_ids) != set(chronological_ids):
            raise ValueError(
                "ContextPlan selected_ids must contain exactly the memories represented in the request"
            )
    messages = tuple(build_generation_messages(query, ordered, answer_options))
    serialized_context = messages[1]["content"] if len(messages) > 1 else ""
    # This is a deterministic estimate, not a claim about a service tokenizer.
    token_count = sum(estimate_tokens(message["content"]) for message in messages)
    budget_status = "within_budget" if token_budget is None or token_count <= token_budget else "exceeds_budget"
    if strict and budget_status == "exceeds_budget":
        raise ContextPlanError(
            f"selected context uses {token_count} estimated tokens, exceeding budget {token_budget}"
        )
    prompt_hash = generation_prompt_hash()
    request = {
        # ``endpoint`` is transport provenance.  It is retained in the frozen
        # plan/hash but stripped by GeneratorClient before sending JSON.
        "endpoint": str(endpoint),
        "model": str(model),
        "messages": [dict(message) for message in messages],
        "temperature": temperature,
        "max_tokens": 0 if max_tokens is None else max_tokens,
    }
    if response_format is not None:
        request["response_format"] = response_format
    if request_params is not None:
        if not isinstance(request_params, Mapping):
            raise ValueError("ContextPlan request_params must be a mapping")
        # Custom parameters are part of the exact request and may not replace
        # the canonical fields above.  Rejecting collisions prevents a plan
        # hash from describing a value that the HTTP client later overwrites.
        for key, value in request_params.items():
            key = str(key)
            if key in request and key not in {"endpoint"} and request[key] != value:
                raise ValueError(f"request_params conflicts with ContextPlan field {key}")
            request[key] = value
    effective_selected = tuple(
        str(value) for value in (chronological_ids if selected_ids is None else selected_ids)
    )
    context_hash = context_plan_hash(
        selected_ids=effective_selected,
        chronological_ids=chronological_ids,
        serialized_context=serialized_context,
        messages=messages,
        token_count=token_count,
        token_count_is_estimate=True,
        budget=token_budget,
        budget_status=budget_status,
        prompt_hash=prompt_hash,
        request=request,
    )
    return ContextPlan(
        selected_ids=effective_selected,
        chronological_ids=chronological_ids,
        serialized_context=serialized_context,
        messages=messages,
        token_count=token_count,
        token_count_is_estimate=True,
        budget=token_budget,
        budget_status=budget_status,
        prompt_hash=prompt_hash,
        context_hash=context_hash,
        request=request,
    )


def generation_prompt_hash() -> str:
    payload = GENERATOR_SYSTEM_PROMPT + "\n" + GENERATOR_USER_TEMPLATE
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_query(self, text: str, instruction: str | None = None) -> np.ndarray: ...

    def encode_queries(self, texts: Sequence[str], instruction: str | None = None) -> np.ndarray: ...


_HTTP_MAX_TRANSPORT_ATTEMPTS = 4
_HTTP_RETRY_BACKOFF_BASE_SECONDS = 0.5
_HTTP_RETRY_BACKOFF_CAP_SECONDS = 4.0
_HTTP_RETRY_AFTER_CAP_SECONDS = 30.0
_HTTP_SPLIT_CHILD_MAX_TRANSPORT_ATTEMPTS = 1


class HTTPTransportError(RuntimeError):
    """A JSON request failed without exposing response bodies or credentials."""

    def __init__(
        self,
        *,
        attempts: int,
        status_code: int | None,
        retryable: bool,
        cause_type: str,
        retry_budget_exhausted: bool = False,
        error_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.attempts = int(attempts)
        self.status_code = status_code
        self.retryable = bool(retryable)
        self.cause_type = str(cause_type)
        self.retry_budget_exhausted = retry_budget_exhausted
        self.error_metadata = dict(error_metadata or {})
        detail = (
            f"HTTP status {status_code}"
            if status_code is not None
            else self.cause_type
        )
        noun = "attempt" if self.attempts == 1 else "attempts"
        super().__init__(
            f"request failed after {self.attempts} transport {noun}: {detail}"
        )

    @property
    def batch_reducible(self) -> bool:
        """Whether reducing a pointwise batch can plausibly recover the call."""

        if (self.error_metadata.get("service_retryable") is False
                or self.error_metadata.get("server_execution_unknown")):
            return False
        # Unknown legacy 413/422/500 may be batch-dependent. An explicit
        # single-input capacity/format error cannot be repaired by splitting.
        return self.status_code in {413, 422, 500}


def _http_error_metadata(exc: urllib.error.HTTPError) -> dict[str, Any]:
    """Bounded structural response summary; never retain arbitrary echoed text."""
    try:
        raw = exc.read(65537) if exc.fp is not None else b""
    except (OSError, http.client.HTTPException):
        raw = b""
    finally:
        if exc.fp is not None:
            exc.close()
    result = _server_response_metadata(None, exc.headers)
    server_id = result.get("server_request_id")
    if not isinstance(server_id, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", server_id):
        result["server_request_id"] = None
    result.update(response_body_sha256=hashlib.sha256(raw[:65536]).hexdigest(),
                  response_body_truncated=len(raw) > 65536,
                  capacity_verification="unverified", actual_input_tokens=None)
    try:
        body = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return result
    detail = body.get("detail", body.get("error", body)) if isinstance(body, dict) else {}
    if not isinstance(detail, dict):
        return result
    if type(detail.get("retryable")) is bool:
        result["service_retryable"] = detail["retryable"]
    for source, target in (("code", "service_error_code"), ("reason", "service_error_reason"),
                           ("error_type", "service_error_type"),
                           ("upstream_error_code", "upstream_error_code"),
                           ("upstream_error_type", "upstream_error_type"),
                           ("upstream_reason", "upstream_reason"),
                           ("upstream_request_id", "upstream_request_id"),
                           ("proxy_request_id", "proxy_request_id"),
                           ("token_count_source", "server_token_source")):
        value = detail.get(source)
        if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value):
            result[target] = value
    for source, target in (("input_tokens", "actual_input_tokens"),
                           ("document_index", "document_index"),
                           ("reserved_output_tokens", "reserved_output_tokens"),
                           ("effective_max_model_len", "effective_max_model_len")):
        value = detail.get(source)
        if type(value) is int and value >= 0:
            result[target] = value
    if result.get("actual_input_tokens") is not None and result.get("server_token_source"):
        result["capacity_verification"] = "server_reported"
    return result


def _retryable_http_status(status_code: int) -> bool:
    return status_code in {408, 425, 429} or 500 <= status_code <= 599


def _retry_delay_seconds(failed_attempt: int, error: Exception) -> float:
    delay = min(
        _HTTP_RETRY_BACKOFF_CAP_SECONDS,
        _HTTP_RETRY_BACKOFF_BASE_SECONDS * (2 ** (failed_attempt - 1)),
    )
    if isinstance(error, urllib.error.HTTPError) and error.headers is not None:
        raw_retry_after = error.headers.get("Retry-After")
        try:
            retry_after = float(raw_retry_after)
        except (TypeError, ValueError, OverflowError):
            retry_after = 0.0
        if np.isfinite(retry_after) and retry_after > 0.0:
            delay = max(delay, min(retry_after, _HTTP_RETRY_AFTER_CAP_SECONDS))
    return delay


def _post_json(
    url: str,
    payload: Dict[str, Any],
    timeout: float,
    headers: Dict[str, str] | None = None,
    *,
    on_attempt: Callable[[int], None] | None = None,
    max_attempts: int = _HTTP_MAX_TRANSPORT_ATTEMPTS,
) -> Any:
    # Existing call signatures/return values remain unchanged.  A scorer or
    # service client owns the logical scope; split children stay inside it.
    if not current_audit_scope().metadata.get("logical_call_id"):
        operation = "reranker" if "documents" in payload else "generation" if "messages" in payload else "embedding"
        documents = payload.get("documents", payload.get("input", []))
        descriptors = document_descriptors(documents) if isinstance(documents, list) and all(isinstance(x, str) for x in documents) else None
        with logical_request_scope(operation, payload, documents=descriptors):
            return _post_json_transport(url, payload, timeout, headers, on_attempt=on_attempt, max_attempts=max_attempts)
    return _post_json_transport(url, payload, timeout, headers, on_attempt=on_attempt, max_attempts=max_attempts)


def _server_response_metadata(response: Any, headers: Any = None) -> dict[str, Any]:
    raw = response if isinstance(response, Mapping) else {}
    usage = raw.get("usage") if isinstance(raw.get("usage"), Mapping) else {}
    def count(*keys):
        for key in keys:
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
        return None
    server_id = None
    if headers is not None:
        server_id = headers.get("x-request-id") or headers.get("request-id")
    return {
        "server_request_id": server_id,
        "server_reported_model": raw.get("model") if isinstance(raw.get("model"), str) else None,
        "server_reported_input_tokens": count("prompt_tokens", "input_tokens"),
        "server_reported_output_tokens": count("completion_tokens", "output_tokens"),
        "server_reported_total_tokens": count("total_tokens"),
        "server_token_source": "response.usage" if usage else None,
    }


def _post_json_transport(
    url: str, payload: Dict[str, Any], timeout: float,
    headers: Dict[str, str] | None = None, *,
    on_attempt: Callable[[int], None] | None = None,
    max_attempts: int = _HTTP_MAX_TRANSPORT_ATTEMPTS,
) -> Any:
    max_attempts = _strict_int_value(
        max_attempts, "HTTP max_attempts", positive=True
    )
    request_headers = {"Content-Type": "application/json"}
    request_headers.update(headers or {})
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=request_headers,
        method="POST",
    )
    # Match datacenter's `curl --noproxy '*'`: these model IPs are private
    # service routes and must not be sent through an ambient HTTP(S) proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    audit = current_audit_scope()
    metadata = audit.metadata
    request_id = metadata.get("request_id") or new_request_id()
    indices = metadata.get("original_document_indices")
    descriptors = metadata.get("documents", [])
    if indices is not None:
        descriptors = [item for item in descriptors if item.get("document_index") in indices]
    physical = {
        "request_id": request_id, "request_hash": request_hash(payload),
        "execution_hash": execution_hash(payload, str(metadata.get("deployment_fingerprint", ""))),
        "documents": descriptors, "batch_documents": len(payload.get("documents", payload.get("input", [])))
        if isinstance(payload.get("documents", payload.get("input", [])), list) else None,
        "server_reported_input_tokens": None, "server_reported_output_tokens": None,
        "server_reported_total_tokens": None, "server_token_source": None,
    }
    emit_request_event("http_request_started", **physical)
    for attempt_index in range(max_attempts):
        attempt = attempt_index + 1
        try:
            budget_used = None if audit.budget is None else audit.budget.consume()
        except TransportBudgetExceeded:
            emit_request_event("http_budget_exhausted", **physical, transport_attempt=attempt,
                               transport_budget_used=audit.budget.used,
                               transport_budget_max=audit.budget.max_attempts)
            raise
        # Outside the transport try block: audit failures must not trigger an
        # HTTP retry.  This durable start is the resume budget reservation.
        emit_request_event("http_attempt_started", **physical, transport_attempt=attempt,
                           transport_budget_used=budget_used,
                           transport_budget_max=None if audit.budget is None else audit.budget.max_attempts)
        if on_attempt is not None:
            on_attempt(attempt)
        started = time.perf_counter()
        try:
            with opener.open(request, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
                response_metadata = _server_response_metadata(result, getattr(response, "headers", None))
                status_code = getattr(response, "status", 200)
        except urllib.error.HTTPError as exc:
            retryable = _retryable_http_status(exc.code)
            error_metadata = _http_error_metadata(exc)
            # Service declarations may stop retries, never turn a 4xx into
            # a retryable failure merely by saying retryable=true.
            if error_metadata.get("service_retryable") is False:
                retryable = False
            timeout_unknown = "documents" in payload and (
                exc.code in {408, 504} or error_metadata.get("service_error_type") == "TimeoutError"
            )
            error_metadata["server_execution_unknown"] = timeout_unknown
            emit_request_event("http_attempt_failed", **{**physical, **error_metadata},
                               transport_attempt=attempt, status_code=exc.code,
                               cause_type=type(exc).__name__, retryable=retryable,
                               batch_reducible=exc.code in {413, 422, 500} and error_metadata.get("service_retryable") is not False and not timeout_unknown,
                               retry_budget_exhausted=not retryable or timeout_unknown or attempt == max_attempts,
                               elapsed_ms=(time.perf_counter() - started) * 1000)
            if not retryable or timeout_unknown or attempt == max_attempts:
                raise HTTPTransportError(
                    attempts=attempt,
                    status_code=exc.code,
                    retryable=retryable,
                    cause_type=type(exc).__name__,
                    retry_budget_exhausted=True,
                    error_metadata=error_metadata,
                ) from exc
            time.sleep(_retry_delay_seconds(attempt, exc))
        except (
            OSError,
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
            UnicodeDecodeError,
            http.client.HTTPException,
        ) as exc:
            invalid_response = "documents" in payload and isinstance(exc, (json.JSONDecodeError, UnicodeDecodeError))
            timeout_unknown = "documents" in payload and (
                isinstance(exc, TimeoutError) or
                isinstance(getattr(exc, "reason", None), TimeoutError)
            )
            # A timed-out scoring request may still run on the backend.
            # End this call instead of immediately duplicating the work.
            retryable = not invalid_response
            exhausted = invalid_response or timeout_unknown or attempt == max_attempts
            error_metadata = {"server_execution_unknown": timeout_unknown,
                              "capacity_verification": "unverified",
                              "actual_input_tokens": None}
            emit_request_event("http_attempt_failed", **physical, transport_attempt=attempt,
                               **error_metadata, retry_budget_exhausted=exhausted,
                               status_code=None, cause_type=type(exc).__name__, retryable=retryable,
                               batch_reducible=False, elapsed_ms=(time.perf_counter() - started) * 1000)
            if exhausted:
                raise HTTPTransportError(
                    attempts=attempt,
                    status_code=None,
                    retryable=retryable,
                    cause_type=type(exc).__name__,
                    retry_budget_exhausted=True,
                    error_metadata=error_metadata,
                ) from exc
            time.sleep(_retry_delay_seconds(attempt, exc))
        else:
            # Again outside the network try: a completed request must not be
            # resent merely because recording its result failed.
            emit_request_event("http_attempt_completed", **{**physical, **response_metadata},
                               transport_attempt=attempt, status_code=status_code,
                               elapsed_ms=(time.perf_counter() - started) * 1000)
            return result
    raise AssertionError("unreachable HTTP retry state")


class RemoteEmbeddingClient:
    def __init__(self, config: EmbeddingConfig):
        self.config = config

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise ValueError("embedding texts must be a sequence of strings")
        if any(not isinstance(text, str) for text in texts):
            raise ValueError("embedding texts must be a sequence of strings")
        if not texts:
            return np.empty((0, 0), dtype=np.float64)
        rows: List[List[float]] = []
        batch_size = _strict_int_value(self.config.batch_size, "embedding batch_size", positive=True)
        expected_dimension: int | None = None
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            payload = {"model": self.config.model, "input": batch}
            with logical_request_scope("embedding", payload, deployment_fingerprint(self.config),
                                       documents=document_descriptors(batch, estimated_tokens=[estimate_tokens(x) for x in batch])):
                response = _post_json(self.config.endpoint, payload, self.config.timeout_seconds)
            data = response.get("data") if isinstance(response, dict) else None
            if not isinstance(data, list):
                raise ValueError("embedding response must contain a data list")
            if len(data) != len(batch):
                raise ValueError(
                    f"embedding response count mismatch for batch: expected {len(batch)}, got {len(data)}"
                )
            ordered: list[tuple[int, Any]] = []
            seen_indices: set[int] = set()
            for item in data:
                if not isinstance(item, Mapping) or "index" not in item or "embedding" not in item:
                    raise ValueError("embedding response items must contain index and embedding")
                index = _strict_int_value(item["index"], "embedding response index", nonnegative=True)
                if index >= len(batch) or index in seen_indices:
                    raise ValueError("embedding response contains an unknown or duplicate index")
                vector = item["embedding"]
                if isinstance(vector, (str, bytes)) or not isinstance(vector, Sequence):
                    raise ValueError("embedding response vector must be a sequence")
                try:
                    values = [float(value) for value in vector]
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError("embedding response vector must contain numeric values") from exc
                if not values or not np.all(np.isfinite(values)):
                    raise ValueError("embedding response vector must be non-empty and finite")
                if expected_dimension is None:
                    expected_dimension = len(values)
                elif len(values) != expected_dimension:
                    raise ValueError("embedding response vectors have inconsistent dimensions")
                seen_indices.add(index)
                ordered.append((index, values))
            if seen_indices != set(range(len(batch))):
                raise ValueError("embedding response indices must cover the complete batch")
            rows.extend(vector for _index, vector in sorted(ordered, key=lambda pair: pair[0]))
        if len(rows) != len(texts):
            raise ValueError(f"embedding response count mismatch: expected {len(texts)}, got {len(rows)}")
        return normalize_rows(np.asarray(rows, dtype=np.float64))

    def encode_queries(self, texts: Sequence[str], instruction: str | None = None) -> np.ndarray:
        if instruction is not None and not isinstance(instruction, str):
            raise ValueError("embedding instruction must be a string or None")
        prefix = self.config.query_instruction if instruction is None else instruction
        return self.encode([prefix + text for text in texts])

    def encode_query(self, text: str, instruction: str | None = None) -> np.ndarray:
        return self.encode_queries([text], instruction=instruction)[0]


class LocalSentenceTransformerEmbedder:
    def __init__(self, model_path: str, device: str = "cpu", query_instruction: str = ""):
        if not model_path:
            raise ValueError("local embedding backend requires local_model_path")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("install the local-models extra to use local embeddings") from exc
        self.model = SentenceTransformer(model_path, device=device)
        self.query_instruction = query_instruction

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise ValueError("embedding texts must be a sequence of strings")
        if any(not isinstance(text, str) for text in texts):
            raise ValueError("embedding texts must be a sequence of strings")
        if not texts:
            return np.empty((0, 0), dtype=np.float64)
        vectors = self.model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)
        matrix = np.asarray(vectors, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != len(texts) or matrix.shape[1] == 0:
            raise ValueError("local embedding model returned an invalid vector matrix")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("local embedding model returned non-finite vectors")
        return normalize_rows(matrix)

    def encode_queries(self, texts: Sequence[str], instruction: str | None = None) -> np.ndarray:
        if instruction is not None and not isinstance(instruction, str):
            raise ValueError("embedding instruction must be a string or None")
        prefix = self.query_instruction if instruction is None else instruction
        return self.encode([prefix + text for text in texts])

    def encode_query(self, text: str, instruction: str | None = None) -> np.ndarray:
        return self.encode_queries([text], instruction=instruction)[0]


def build_embedder(config: EmbeddingConfig, device: str = "cpu") -> Embedder:
    if config.backend == "remote":
        return RemoteEmbeddingClient(config)
    return LocalSentenceTransformerEmbedder(
        config.local_model_path or config.model,
        device=device,
        query_instruction=config.query_instruction,
    )


@dataclass(frozen=True)
class RerankItem:
    index: int
    score: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "index", _strict_int_value(self.index, "rerank index", nonnegative=True))
        score = _strict_float_value(self.score, "rerank score")
        object.__setattr__(self, "score", score)


def _response_declares_truncation(value: Any) -> bool:
    """Return true only for an explicit positive backend truncation flag.

    Different OpenAI-compatible reranker servers expose this bit in slightly
    different envelopes (top-level, ``meta``/``usage``, or per result).  A
    full-set score must never be cached when any of those envelopes says the
    input was shortened.  We intentionally do not infer truncation from token
    counts or from a field describing the configured truncation strategy.
    """

    flag_names = {
        "truncated",
        "is_truncated",
        "was_truncated",
        "input_truncated",
        "inputs_truncated",
        "document_truncated",
        "documents_truncated",
    }
    reason_names = {"finish_reason", "termination_reason"}
    truncation_reasons = {"length", "max_tokens", "truncated", "input_length"}

    def positive_flag(flag: Any) -> bool:
        if isinstance(flag, (bool, np.bool_)):
            return bool(flag)
        if isinstance(flag, str):
            return flag.strip().lower() in {"true", "yes", "1"}
        if isinstance(flag, (int, np.integer)) and not isinstance(flag, bool):
            return int(flag) == 1
        if isinstance(flag, Mapping):
            return any(positive_flag(child) for child in flag.values())
        if isinstance(flag, (list, tuple)):
            return any(positive_flag(child) for child in flag)
        return False

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).strip().lower().replace("-", "_")
            if key in flag_names and positive_flag(child):
                return True
            if (
                key in reason_names
                and isinstance(child, str)
                and child.strip().lower().replace("-", "_") in truncation_reasons
            ):
                return True
            if isinstance(child, (Mapping, list, tuple)) and _response_declares_truncation(child):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_response_declares_truncation(item) for item in value)
    return False


def build_rerank_payload(config: Any, query: str, documents: Sequence[str], top_n: int) -> Dict[str, Any]:
    """One canonical provider payload shared by execution and offline plans."""
    payload: Dict[str, Any] = {"query": query, "documents": list(documents),
                               "top_n": min(top_n, len(documents)), "return_documents": False}
    model = getattr(config, "model", "")
    if model:
        payload["model"] = model
    return payload


class RerankerClient:
    def __init__(self, config: EndpointConfig):
        self.config = config
        raw_space = str(getattr(config, "score_space", "unit_interval")).strip().lower().replace("-", "_")
        self.score_space = {
            "probability": "unit_interval",
            "probabilities": "unit_interval",
            "unit": "unit_interval",
            "sigmoid": "unit_interval",
            "logit": "logit_difference",
            "logit_diff": "logit_difference",
            "raw_logit_difference": "logit_difference",
        }.get(raw_space, raw_space)
        if self.score_space not in {"unit_interval", "logit_difference"}:
            raise ValueError("reranker score_space must be unit_interval or logit_difference")
        self.score_contract = str(getattr(config, "score_contract", "pointwise")).strip().lower()
        if self.score_contract not in {"pointwise", "listwise"}:
            raise ValueError("reranker score_contract must be pointwise or listwise")
        # These counters distinguish one caller-visible scoring operation from
        # the extra transport work caused by retrying a failed pointwise batch.
        # Search/set budgets remain the caller's responsibility and are never
        # charged again when this client subdivides a request.
        self._logical_calls = 0
        self._logical_documents = 0
        self._batch_requests = 0
        self._batch_documents = 0
        self._transport_attempts = 0
        self._transport_document_attempts = 0
        self._failed_batch_requests = 0
        self._split_events = 0
        self._split_recovered_calls = 0
        self._failed_calls = 0
        self._capacity_contract: dict[str, Any] | None = None

    def verify_capacity_contract(self) -> dict[str, Any]:
        """Verify server enforcement, without claiming local exact token counts."""
        expected = getattr(self.config, "required_max_model_len", None)
        if expected is None:
            return {"capacity_verification": "unverified", "actual_input_tokens": None}
        if self._capacity_contract is None:
            from urllib.parse import urlsplit, urlunsplit
            parts = urlsplit(self.config.endpoint)
            url = urlunsplit((parts.scheme, parts.netloc, "/health", "", ""))
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(url, timeout=self.config.timeout_seconds) as response:
                health = json.loads(response.read())
            fields = ("declared_max_model_len", "backend_max_model_len", "effective_max_model_len")
            if (not isinstance(health, dict) or any(health.get(k) != expected for k in fields)
                    or health.get("reserved_output_tokens") != 1
                    or health.get("truncation_policy") != "reject_without_truncation"
                    or health.get("token_count_source") != "submitted_prompt_token_ids"):
                raise ValueError("reranker service capacity contract does not match frozen config")
            self._capacity_contract = {k: health[k] for k in fields}
            self._capacity_contract.update(
                capacity_verification="server_enforced", actual_input_tokens=None,
                reserved_output_tokens=1, server_token_source="submitted_prompt_token_ids",
                truncation_policy="reject_without_truncation")
        return dict(self._capacity_contract)

    @property
    def model_fingerprint(self) -> str:
        # Resolve unknown identity in the current run scope, not the process
        # scope in which a reusable client happened to be constructed.
        return deployment_fingerprint(self.config)

    @property
    def transport_stats(self) -> Dict[str, int]:
        """Return cumulative logical, batch, and actual HTTP-attempt counts.

        ``logical_documents`` counts each input document once per public
        ``rerank`` call. ``batch_documents`` counts it again when a failed
        batch is divided, while ``transport_document_attempts`` additionally
        includes exponential-backoff attempts made by :func:`_post_json`.
        """

        return {
            "logical_calls": self._logical_calls,
            "logical_documents": self._logical_documents,
            "batch_requests": self._batch_requests,
            "batch_documents": self._batch_documents,
            "transport_attempts": self._transport_attempts,
            "transport_document_attempts": self._transport_document_attempts,
            "failed_batch_requests": self._failed_batch_requests,
            "split_events": self._split_events,
            "split_recovered_calls": self._split_recovered_calls,
            "failed_calls": self._failed_calls,
        }

    def _rerank_batch(
        self,
        query: str,
        documents: Sequence[str],
        top_n: int,
        *,
        max_transport_attempts: int | None = None,
    ) -> List[RerankItem]:
        """Issue and validate one physical batch without changing indices."""

        payload = build_rerank_payload(self.config, query, documents, top_n)
        self._batch_requests += 1
        self._batch_documents += len(documents)

        def record_attempt(_attempt: int) -> None:
            self._transport_attempts += 1
            self._transport_document_attempts += len(documents)

        post_kwargs: Dict[str, Any] = {"on_attempt": record_attempt}
        if max_transport_attempts is not None:
            post_kwargs["max_attempts"] = max_transport_attempts
        try:
            response = _post_json(
                self.config.endpoint,
                payload,
                self.config.timeout_seconds,
                **post_kwargs,
            )
        except HTTPTransportError:
            self._failed_batch_requests += 1
            raise
        if _response_declares_truncation(response):
            raise ValueError("rerank backend explicitly reported input truncation")
        if isinstance(response, dict):
            raw_results = response.get("results", response.get("data"))
        elif isinstance(response, list):
            raw_results = response
        else:
            raw_results = None
        if not isinstance(raw_results, list):
            raise ValueError("rerank response must contain results or data")
        items = []
        seen: set[int] = set()
        for item in raw_results:
            if isinstance(item, Mapping):
                if "index" not in item:
                    raise ValueError("rerank response item is missing index")
                position = _strict_int_value(item["index"], "rerank response index", nonnegative=True)
                raw_score = item.get("relevance_score", item.get("score"))
            else:
                try:
                    position = _strict_int_value(item.index, "rerank response index", nonnegative=True)
                    raw_score = item.score
                except (AttributeError, TypeError, ValueError, OverflowError) as exc:
                    raise ValueError("rerank response item must contain index and score") from exc
            if raw_score is None:
                raise ValueError("rerank response item is missing score")
            score = _strict_float_value(raw_score, "rerank response score")
            if position >= len(documents) or position in seen:
                raise ValueError("rerank response contains an unknown or duplicate index")
            if not np.isfinite(score):
                raise ValueError("rerank response contains a non-finite score")
            if self.score_space == "unit_interval" and not 0.0 <= score <= 1.0:
                raise ValueError("unit-interval rerank score is outside [0, 1]")
            seen.add(position)
            items.append(RerankItem(index=position, score=score))
        if top_n >= len(documents) and seen != set(range(len(documents))):
            raise ValueError("rerank_all response must cover every document")
        return sorted(items, key=lambda item: (-item.score, item.index))[:top_n]

    def _rerank_pointwise_with_splits(
        self,
        query: str,
        documents: Sequence[str],
        top_n: int,
    ) -> tuple[List[RerankItem], bool]:
        """Bisect exhausted 5xx/oversized batches and restore original indices."""

        width = getattr(self.config, "max_batch_documents", None) or len(documents)
        pending: list[tuple[int, List[str], int | None, str, str | None, int]] = [
            (offset, list(documents[offset:offset + width]), None, new_request_id(), None, 0)
            for offset in reversed(range(0, len(documents), width))
        ]
        combined: list[RerankItem] = []
        split_used = False
        while pending:
            offset, batch, max_transport_attempts, request_id, parent_id, split_depth = pending.pop()
            try:
                with request_audit_scope({"request_id": request_id, "parent_request_id": parent_id,
                                          "split_depth": split_depth,
                                          "original_document_indices": list(range(offset, offset + len(batch)))}):
                    local_items = self._rerank_batch(
                        query, batch, min(top_n, len(batch)),
                        max_transport_attempts=max_transport_attempts,
                    )
            except HTTPTransportError as exc:
                # Splitting is valid only because this client enforces a
                # pointwise score contract. A singleton proves the failure is
                # not recoverable by changing batch composition.
                if not exc.batch_reducible or len(batch) == 1:
                    raise
                midpoint = len(batch) // 2
                self._split_events += 1
                split_used = True
                # LIFO with right first processes the left/original prefix
                # first and aborts along the first irrecoverable branch rather
                # than creating a full retry storm during a persistent outage.
                # The parent already exhausted its normal transport retry
                # budget. Probe each degraded child once so a persistent 500
                # cannot multiply four retries at every bisection depth.
                child_attempts = _HTTP_SPLIT_CHILD_MAX_TRANSPORT_ATTEMPTS
                left_id, right_id = new_request_id(), new_request_id()
                emit_request_event("http_request_split", request_id=request_id,
                                   parent_request_id=parent_id, split_depth=split_depth,
                                   children=[{"request_id": left_id, "original_document_indices": list(range(offset, offset + midpoint))},
                                             {"request_id": right_id, "original_document_indices": list(range(offset + midpoint, offset + len(batch)))}])
                pending.append(
                    (offset + midpoint, batch[midpoint:], child_attempts, right_id, request_id, split_depth + 1)
                )
                pending.append((offset, batch[:midpoint], child_attempts, left_id, request_id, split_depth + 1))
                continue
            combined.extend(
                RerankItem(index=offset + item.index, score=item.score)
                for item in local_items
            )
        return sorted(combined, key=lambda item: (-item.score, item.index))[:top_n], split_used

    def rerank(self, query: str, documents: Sequence[str], top_n: int) -> List[RerankItem]:
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence) or any(not isinstance(x, str) for x in documents):
            raise ValueError("documents must be a sequence of strings")
        top_n = _strict_int_value(top_n, "top_n", nonnegative=True)
        payload = build_rerank_payload(self.config, query, documents, top_n)
        with logical_request_scope("reranker", payload, deployment_fingerprint(self.config),
                                   documents=document_descriptors(documents, estimated_tokens=[estimate_tokens(query) + estimate_tokens(x) for x in documents])):
            contract = self.verify_capacity_contract() if documents and top_n else {}
            with request_audit_scope(contract):
                emit_request_event("reranker_capacity_contract", **contract)
                return self._rerank_impl(query, documents, top_n)

    def _rerank_impl(self, query: str, documents: Sequence[str], top_n: int) -> List[RerankItem]:
        if not isinstance(query, str):
            raise ValueError("rerank query must be a string")
        if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence):
            raise ValueError("rerank documents must be a sequence of strings")
        if any(not isinstance(document, str) for document in documents):
            raise ValueError("rerank documents must be a sequence of strings")
        top_n = _strict_int_value(top_n, "top_n", nonnegative=True)
        if self.score_contract != "pointwise":
            raise ValueError("listwise reranker contracts cannot be used as pointwise scores")
        self._logical_calls += 1
        self._logical_documents += len(documents)
        if not documents or top_n == 0:
            return []
        try:
            items, split_used = self._rerank_pointwise_with_splits(
                query,
                list(documents),
                min(top_n, len(documents)),
            )
        except Exception:
            self._failed_calls += 1
            raise
        if split_used:
            self._split_recovered_calls += 1
        return items

    def rerank_all(self, query: str, documents: Sequence[str]) -> List[RerankItem]:
        """Return a complete deterministic ranking so callers can cache once and slice later."""
        return self.rerank(query, documents, len(documents))


class GeneratorClient:
    def __init__(self, config: GeneratorConfig):
        self.config = config

    def complete_messages(
        self, messages: Sequence[Mapping[str, str]], *, operation: str, max_tokens: int
    ) -> str:
        """Frozen chat inference for evidence reasoning, distinct from the reader.

        The method owns evidence schemas and input budgets. This transport
        sends exactly its complete messages and preserves physical request
        accounting, provider parameters, and model identity. It never injects
        reader options, prompts, or cached answers.
        """
        if not isinstance(operation, str) or not re.fullmatch(r"evidence_[a-z_]+", operation):
            raise ValueError("structured reasoning operation must use the evidence_ prefix")
        if isinstance(messages, (str, bytes)) or not messages:
            raise ValueError("evidence messages must be a non-empty sequence")
        wire_messages = []
        for message in messages:
            if not isinstance(message, Mapping) or set(message) != {"role", "content"}:
                raise ValueError("evidence messages require exactly role and content")
            if message["role"] not in {"system", "user", "assistant"} or not isinstance(message["content"], str):
                raise ValueError("invalid evidence message")
            wire_messages.append(dict(message))
        payload = {
            "model": self.config.model,
            "messages": wire_messages,
            "temperature": self.config.temperature,
            "max_tokens": _strict_int_value(max_tokens, "evidence max_tokens", positive=True),
            **dict(self.config.provider_request_params),
        }
        estimate = estimate_tokens(json.dumps(wire_messages, ensure_ascii=False))
        with request_audit_scope({"stage": operation, "estimated_input_tokens": estimate,
                                  "token_count_is_estimate": True,
                                  "token_estimator_id": "regex_word_or_punctuation_v1"}):
            with logical_request_scope(operation, canonical_request_payload(payload),
                                       deployment_fingerprint(self.config)):
                response = _post_json(
                    self.config.endpoint, payload, self.config.timeout_seconds,
                    headers={"Authorization": f"Bearer {self.config.resolved_api_key()}"},
                )
        choices = response.get("choices") if isinstance(response, dict) else None
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
            raise ValueError("evidence chat response has no choices")
        message = choices[0].get("message")
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str) or not content.strip():
            raise ValueError("evidence chat response has no text content")
        return content.strip()

    def answer(self, query: str, memories: Sequence[Memory], answer_options: str = "") -> str:
        """Build and send one exact generation plan.

        This convenience API is kept for older callers, but it deliberately
        does not perform a second context-budget pass.  The selected memory
        sequence is frozen into a :class:`ContextPlan`; an over-budget
        selection is an explicit protocol error rather than an instruction to
        silently drop memories.
        """
        if not isinstance(query, str) or not isinstance(answer_options, str):
            raise ValueError("generator query and answer_options must be strings")
        plan = build_context_plan(
            query,
            memories,
            answer_options,
            token_budget=self.config.context_token_budget,
            strict=True,
            generator_config=self.config,
        )
        return self.answer_plan(plan)

    def answer_plan(self, plan: ContextPlan) -> str:
        """Send exactly the messages in a frozen :class:`ContextPlan`."""

        with request_audit_scope({"estimated_input_tokens": plan.token_count,
                                  "token_count_is_estimate": True,
                                  "token_estimator_id": "regex_word_or_punctuation_v1"}):
            with logical_request_scope("generation", canonical_request_payload(plan.request_dict()),
                                       deployment_fingerprint(self.config)):
                return self._answer_plan_impl(plan)

    def _answer_plan_impl(self, plan: ContextPlan) -> str:

        if not plan.within_budget:
            raise ContextPlanError("cannot send a ContextPlan that exceeds its declared budget")
        if plan.prompt_hash != generation_prompt_hash():
            raise ContextPlanError("ContextPlan prompt hash differs from GeneratorClient prompt contract")
        request_payload = plan.request_dict()
        # ``ContextPlan`` is the sole source of truth.  Endpoint is transport
        # metadata (and therefore not part of the provider JSON), but every
        # other field must already be final in the plan.  In particular,
        # ``max_tokens=0`` is an explicit value, not a sentinel to replace
        # with the client's configuration.
        declared_endpoint = request_payload.pop("endpoint", None)
        declared_endpoint_hash = request_payload.pop("endpoint_sha256", None)
        configured_endpoint_hash = hashlib.sha256(
            str(self.config.endpoint).encode("utf-8")
        ).hexdigest()
        if declared_endpoint not in (None, "", self.config.endpoint, configured_endpoint_hash):
            raise ContextPlanError("ContextPlan endpoint differs from GeneratorClient configuration")
        if declared_endpoint_hash not in (None, "", configured_endpoint_hash):
            raise ContextPlanError("ContextPlan endpoint identity differs from GeneratorClient configuration")
        declared_model = request_payload.get("model")
        if declared_model != self.config.model:
            raise ContextPlanError("ContextPlan model differs from GeneratorClient configuration")
        declared_temperature = request_payload.get("temperature")
        try:
            declared_temperature = _strict_float_value(
                declared_temperature, "ContextPlan request temperature", nonnegative=True
            )
            configured_temperature = _strict_float_value(
                self.config.temperature, "generator temperature", nonnegative=True
            )
        except ValueError as exc:
            raise ContextPlanError("ContextPlan temperature is invalid") from exc
        if declared_temperature != configured_temperature:
            raise ContextPlanError("ContextPlan temperature differs from GeneratorClient configuration")
        declared_max_tokens = request_payload.get("max_tokens")
        try:
            declared_max_tokens = _strict_int_value(
                declared_max_tokens, "ContextPlan request max_tokens", nonnegative=True
            )
            configured_max_tokens = _strict_int_value(
                self.config.max_tokens, "generator max_tokens", positive=True
            )
        except ValueError as exc:
            raise ContextPlanError("ContextPlan max_tokens is invalid") from exc
        if declared_max_tokens != configured_max_tokens:
            raise ContextPlanError("ContextPlan max_tokens differs from GeneratorClient configuration")
        if request_payload.get("messages") != [dict(message) for message in plan.messages]:
            raise ContextPlanError("ContextPlan messages differ from its declared request")
        for key, value in self.config.provider_request_params.items():
            if request_payload.get(key) != value:
                raise ContextPlanError("ContextPlan provider parameters differ from GeneratorClient configuration")
        response = _post_json(
            self.config.endpoint,
            request_payload,
            self.config.timeout_seconds,
            headers={"Authorization": f"Bearer {self.config.resolved_api_key()}"},
        )
        choices = response.get("choices") if isinstance(response, dict) else None
        if not choices:
            raise ValueError("chat response has no choices")
        return str(choices[0]["message"]["content"]).strip()


# Keep an immutable reference to the package implementation.  Offline
# evaluators in the research codebase historically monkeypatch
# ``GeneratorClient.answer`` with a deterministic adapter; comparing against
# this saved function lets the cache preserve that explicit compatibility
# hook while the normal client always sends the frozen ContextPlan directly.
_DEFAULT_GENERATOR_ANSWER = GeneratorClient.answer


class StateEmbeddingCache:
    """Content-addressed cache for frozen option/path state embeddings.

    The key intentionally includes every identity that can change a state
    coordinate system; this prevents accidental reuse across models, options,
    queries, or posterior path orderings.
    """

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root is not None else None
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)
        self._memory: dict[str, np.ndarray] = {}
        self.hits = 0
        self.misses = 0

    def key_for(
        self,
        *,
        endpoint: str = "",
        model: str = "",
        embedding_fingerprint: str = "",
        query_hash: str = "",
        ordered_path_ids: Sequence[str] = (),
        options_hash: str = "",
    ) -> str:
        payload = {
            "schema": 1,
            "endpoint": str(endpoint),
            "model": str(model),
            "embedding_fingerprint": str(embedding_fingerprint),
            "query_hash": str(query_hash),
            "ordered_path_ids": list(ordered_path_ids),
            "options_hash": str(options_hash),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

    def get(self, key: str) -> np.ndarray | None:
        if key in self._memory:
            self.hits += 1
            return self._memory[key].copy()
        if self.root is None:
            return None
        path = self.root / f"{key}.npy"
        if not path.is_file():
            return None
        try:
            value = np.load(path, allow_pickle=False)
            result = np.asarray(value, dtype=np.float64)
            if result.ndim != 2 or not np.all(np.isfinite(result)):
                return None
            self._memory[key] = result.copy()
            self.hits += 1
            return result
        except (OSError, ValueError):
            return None

    def put(self, key: str, value: np.ndarray) -> None:
        result = np.asarray(value, dtype=np.float64).copy()
        if result.ndim != 2 or not np.all(np.isfinite(result)):
            raise ValueError("state embedding cache values must be a finite 2-D matrix")
        self._memory[key] = result
        if self.root is None:
            return
        path = self.root / f"{key}.npy"
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, result, allow_pickle=False)
        temporary.replace(path)

    def get_or_encode(self, key: str, encode) -> tuple[np.ndarray, bool]:
        cached = self.get(key)
        if cached is not None:
            return cached, True
        value = np.asarray(encode(), dtype=np.float64)
        self.misses += 1
        self.put(key, value)
        return value, False

    make_key = key_for
    get_or_build = get_or_encode

    @property
    def stats(self) -> dict[str, int]:
        return {"hits": int(self.hits), "misses": int(self.misses)}


class GenerationCache:
    """Deterministic text cache keyed by the complete generation request."""

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root is not None else None
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)
        self._memory: dict[str, str] = {}
        self.hits = 0
        self.misses = 0

    def key_for(
        self,
        query: str,
        ordered_ids: Sequence[str],
        serialized_context: str,
        *,
        generator: str = "",
        endpoint: str = "",
        temperature: float = 0.0,
        max_tokens: int = 0,
        prompt_hash: str | None = None,
        answer_options: str = "",
    ) -> str:
        payload = {
            "schema": 1,
            "prompt_hash": prompt_hash or generation_prompt_hash(),
            "query": query,
            "ordered_ids": list(ordered_ids),
            "serialized_context": serialized_context,
            "generator": str(generator),
            "endpoint": str(endpoint),
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "answer_options": answer_options,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

    def get(self, key: str) -> str | None:
        if key in self._memory:
            self.hits += 1
            return self._memory[key]
        if self.root is None:
            return None
        path = self.root / f"{key}.json"
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            response = str(value["response"])
            self._memory[key] = response
            self.hits += 1
            return response
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None

    def put(self, key: str, response: str) -> None:
        value = str(response)
        self._memory[key] = value
        if self.root is None:
            return
        path = self.root / f"{key}.json"
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps({"response": value}, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(path)

    def answer(
        self,
        client: GeneratorClient,
        query: str,
        memories: Sequence[Memory],
        answer_options: str = "",
        *,
        selected_ids: Sequence[str] | None = None,
    ) -> tuple[str, bool]:
        """Cache a convenience call by first freezing its exact plan.

        ``answer`` remains a compatibility wrapper for callers that have not
        yet materialized a plan.  It never truncates or reorders the supplied
        selection; the resulting plan is the same object used for the cache
        key and for transport.
        """
        plan = build_context_plan(
            query,
            memories,
            answer_options,
            token_budget=client.config.context_token_budget,
            strict=True,
            selected_ids=selected_ids,
            generator_config=client.config,
        )
        # Keep the historical overridable ``GeneratorClient.answer`` hook
        # usable for local/offline evaluators.  With the package
        # implementation the exact frozen plan is sent directly; only an
        # explicitly overridden adapter is called through the compatibility
        # API.  Reconstruct the same chronological reader order that the
        # frozen plan carries; the selector order remains available through
        # ``selected_ids`` and does not change what the reader sees.
        if selected_ids is None:
            request_memories = list(memories)
        else:
            by_id = {str(memory.memory_id): memory for memory in memories}
            request_memories = [by_id[identifier] for identifier in plan.chronological_ids]
        cached = self._cache_key_for_plan(plan, client)
        cached_value = self.get(cached)
        if cached_value is not None:
            return cached_value, True
        answer_hook = getattr(client, "answer", None)
        answer_impl = getattr(answer_hook, "__func__", answer_hook)
        if answer_impl is _DEFAULT_GENERATOR_ANSWER:
            response = client.answer_plan(plan)
        else:
            response = answer_hook(query, request_memories, answer_options)
        self.misses += 1
        self.put(cached, response)
        return response, False

    def answer_plan(self, client: GeneratorClient, plan: ContextPlan) -> tuple[str, bool]:
        """Cache and send the exact frozen request represented by ``plan``."""

        key = self._cache_key_for_plan(plan, client)
        cached = self.get(key)
        if cached is not None:
            return cached, True
        response = client.answer_plan(plan)
        self.misses += 1
        self.put(key, response)
        return response, False

    @staticmethod
    def _cache_key_for_plan(plan: ContextPlan, client: GeneratorClient) -> str:
        # Cache identity follows the request actually sent to the model.  The
        # provenance ``context_hash`` intentionally includes selector order,
        # but that order is not part of the chronological reader messages and
        # must not split an otherwise identical generation request.
        request = canonical_request_payload(plan.request_dict())
        payload = {
            "schema": 3,
            "request": request,
            "deployment_fingerprint": deployment_fingerprint(client.config),
            # The endpoint is transport identity rather than JSON payload;
            # hash it so cache files do not expose internal service URLs.
            "endpoint_sha256": hashlib.sha256(str(client.config.endpoint).encode("utf-8")).hexdigest(),
        }
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    make_key = key_for
    get_or_generate = answer

    @property
    def stats(self) -> dict[str, int]:
        return {"hits": int(self.hits), "misses": int(self.misses)}


__all__ = [
    "Embedder",
    "RemoteEmbeddingClient",
    "LocalSentenceTransformerEmbedder",
    "RerankItem",
    "RerankerClient",
    "GeneratorClient",
    "StateBasisProvider",
    "StateEmbeddingCache",
    "GenerationCache",
    "estimate_tokens",
    "context_token_count",
    "fit_context_budget",
    "ContextPlan",
    "ContextPlanError",
    "build_context_plan",
    "build_generation_messages",
    "generation_prompt_hash",
]
