"""Task-local, payload-free request events and a physical transport ceiling.

Scope metadata never enters provider payloads.  Retry and split attempts all
consume the same optional budget.  A durable attempt-start is a conservative
reservation on resume: a crash may consume a slot without sending a request.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable, Mapping
import uuid

from .diagnostic_identity import execution_hash, request_hash


class TransportBudgetExceeded(RuntimeError):
    """A diagnostic physical-call ceiling was reached; never retry as HTTP."""


class AuditWriteError(RuntimeError):
    """Request audit persistence failed; never retry as a transport failure."""


class TransportBudget:
    def __init__(self, max_attempts: int, *, used: int = 0):
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 0:
            raise ValueError("transport max_attempts must be a nonnegative integer")
        if isinstance(used, bool) or not isinstance(used, int) or not 0 <= used <= max_attempts:
            raise ValueError("transport used must be between zero and max_attempts")
        self.max_attempts = max_attempts
        self.used = used
        self._lock = threading.Lock()

    def consume(self) -> int:
        with self._lock:
            if self.used >= self.max_attempts:
                raise TransportBudgetExceeded("physical transport attempt budget exhausted")
            self.used += 1
            return self.used


_FIELDS = {
    "service_retryable", "service_error_code", "service_error_reason", "service_error_type",
    "upstream_error_code", "upstream_error_type", "upstream_reason",
    "upstream_request_id", "proxy_request_id", "response_body_sha256", "response_body_truncated",
    "retry_budget_exhausted", "server_execution_unknown", "capacity_verification",
    "actual_input_tokens", "reserved_output_tokens", "effective_max_model_len",
    "declared_max_model_len", "backend_max_model_len", "truncation_policy",
    "schema_version", "event", "event_id", "at_epoch", "run_identity", "cache_scope",
    "task_id", "persona_id", "question_id", "method_id", "execution_attempt", "task_attempt", "attempt",
    "stage", "phase", "operation", "logical_call_id", "request_id", "parent_request_id", "split_depth",
    "original_document_indices", "document_index", "documents", "document_hash", "set_ids",
    "estimated_input_tokens", "token_count_is_estimate", "token_estimator_id", "batch_documents",
    "request_hash", "logical_request_hash", "execution_hash", "deployment_fingerprint",
    "transport_attempt", "transport_budget_used", "transport_budget_max", "status_code", "cause_type",
    "elapsed_ms", "retryable", "batch_reducible", "server_request_id", "server_reported_model",
    "server_reported_input_tokens", "server_reported_output_tokens", "server_reported_total_tokens",
    "server_token_source", "children", "probe_attempt", "diagnostic_id", "repetition", "scope",
    "score_reason", "objective_semantics", "utility_validation_id",
    "item_id", "trial_id", "subset_id", "condition_id", "repeat_index", "root_tie_seed", "root_tie_break",
}


def _safe_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe_value(item) for key, item in value.items() if str(key) in _FIELDS}
    if isinstance(value, (tuple, list)):
        return [_safe_value(item) for item in value]
    if isinstance(value, str):
        if re.search(r"https?://|Bearer\s|sk-[A-Za-z0-9_-]{8,}", value, re.I):
            return "[redacted]"
        return value[:512]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return type(value).__name__


class JsonlAuditSink:
    def __init__(self, path: str | Path, *, durable: bool = True):
        self.path = Path(path)
        self.durable = durable
        self._lock = threading.Lock()

    def __call__(self, event: Mapping[str, Any]) -> None:
        try:
            line = json.dumps(_safe_value(event), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as stream:
                    stream.write(line)
                    stream.flush()
                    if self.durable:
                        os.fsync(stream.fileno())
        except (OSError, ValueError, TypeError) as exc:
            raise AuditWriteError("could not persist request audit event") from exc


@dataclass(frozen=True)
class _AuditScope:
    metadata: Mapping[str, Any]
    sink: Callable[[Mapping[str, Any]], None] | None = None
    budget: TransportBudget | None = None


_CURRENT: ContextVar[_AuditScope] = ContextVar("bridgetree_request_audit", default=_AuditScope({}))
_PROCESS_SCOPE = uuid.uuid4().hex


def current_audit_scope() -> _AuditScope:
    return _CURRENT.get()


def cache_scope_identity() -> str:
    context = _CURRENT.get().metadata
    return str(context.get("cache_scope") or context.get("run_identity") or _PROCESS_SCOPE)


def new_request_id() -> str:
    return uuid.uuid4().hex


@contextmanager
def request_audit_scope(metadata: Mapping[str, Any] | None = None, *, sink=None, budget=None):
    parent = _CURRENT.get()
    token = _CURRENT.set(_AuditScope(
        {**parent.metadata, **_safe_value(metadata or {})},
        parent.sink if sink is None else sink,
        parent.budget if budget is None else budget,
    ))
    try:
        yield _CURRENT.get()
    finally:
        _CURRENT.reset(token)


def emit_request_event(event: str, **fields: Any) -> None:
    scope = _CURRENT.get()
    if scope.sink is not None:
        value = _safe_value({**scope.metadata, **fields, "schema_version": 1,
                             "event": event, "event_id": new_request_id(), "at_epoch": time.time()})
        try:
            scope.sink(value)
        except AuditWriteError:
            raise
        except Exception as exc:
            raise AuditWriteError("request audit sink failed") from exc


def document_descriptors(documents, *, set_ids=None, estimated_tokens=None):
    return [{"document_index": index, "document_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
             "set_ids": None if set_ids is None else list(set_ids[index]),
             "estimated_input_tokens": None if estimated_tokens is None else estimated_tokens[index],
             "token_count_is_estimate": True, "token_estimator_id": "regex_word_or_punctuation_v1"}
            for index, text in enumerate(documents)]


@contextmanager
def logical_request_scope(operation, payload, deployment_fingerprint="", *, documents=None):
    current = _CURRENT.get().metadata
    digest = request_hash(payload)
    if (current.get("logical_call_id") and current.get("logical_request_hash") == digest
            and current.get("operation") == operation
            and current.get("deployment_fingerprint") == deployment_fingerprint):
        yield current["logical_call_id"]
        return
    metadata = {"operation": operation, "logical_call_id": new_request_id(),
                "logical_request_hash": digest, "request_hash": digest,
                "deployment_fingerprint": deployment_fingerprint,
                "execution_hash": execution_hash(payload, deployment_fingerprint),
                "server_reported_input_tokens": None, "server_reported_output_tokens": None,
                "server_reported_total_tokens": None, "server_token_source": None}
    if documents is not None:
        metadata["documents"] = documents
    with request_audit_scope(metadata):
        started = time.perf_counter()
        emit_request_event("model_call_started")
        try:
            yield metadata["logical_call_id"]
        except BaseException as exc:
            emit_request_event("model_call_failed", cause_type=type(exc).__name__,
                               status_code=getattr(exc, "status_code", None),
                               elapsed_ms=(time.perf_counter() - started) * 1000)
            raise
        else:
            emit_request_event("model_call_completed", elapsed_ms=(time.perf_counter() - started) * 1000)
