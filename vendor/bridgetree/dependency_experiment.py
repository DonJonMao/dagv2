"""End-to-end execution and persistence for dependency retrieval experiments.

The search mathematics lives in :mod:`bridgetree.dependency_scoring` and
:mod:`bridgetree.dependency_search`.  This module owns the boundaries around
that mathematics: validating and freezing the complete dataset/task plan
before any service call, ensuring labels are used only after generation,
atomically persisting authoritative per-task outcomes, and producing
denominator-safe metrics that can be rebuilt during resume.

The public runner deliberately accepts explicit service objects.  Small
deterministic fakes therefore exercise the same task executor and persistence
path as deployment without introducing a second experimental pipeline.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import threading
import time
import urllib.error
from contextlib import suppress
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

import numpy as np

from .clients import (
    GeneratorClient,
    RerankerClient,
    build_context_plan,
    build_embedder,
    generation_prompt_hash,
)
from .dependency_config import DependencyRunConfig, load_dependency_config
from .dependency_retrieval import DEPENDENCY_PROPOSAL_INSTRUCTION, DependencyRetriever
from .diagnostic_observability import ModuleEventRecorder, observation_scope
from .diagnostic_identity import deployment_fingerprint, deployment_public, execution_hash, request_hash
from .request_audit import JsonlAuditSink, request_audit_scope
from .experiment import _visible_memory_records
from .metrics import answer_accuracy, answer_parse_failed, extract_option_label
from .personamem import (
    PERSONAMEM_REVISION,
    PERSONAMEM_SOURCE_SHA256,
    PersonaMemExample,
    file_sha256,
    iter_examples,
    load_shared_contexts,
    messages_to_memories,
)
from .types import ContextPlan, Memory

RUN_SCHEMA_VERSION = 1
TASK_SCHEMA_VERSION = 1
OUTCOME_SCHEMA_VERSION = 1
DEFAULT_METHODS = (
    "dense",
    "dense_rerank",
    "activation",
    "context_marginal",
    "activation_fixed_pool",
)
MODULE_LOGS = (
    "planner",
    "scheduler",
    "target",
    "archive",
    "evidence",
    "feedback",
    "scoring",
    "activation",
    "proposal",
    "state",
    "stop",
    "selection",
    "context",
    "cost",
    "effectiveness",
    "requests",
)
INTERACTION_SIGNAL_CAVEAT = (
    "positive interaction signals are measured reranker effects, not causal proof"
)
INFRASTRUCTURE_MARKERS = (
    "connection refused",
    "connection reset",
    "connection aborted",
    "connection error",
    "service unavailable",
    "temporarily unavailable",
    "timed out",
    "timeout",
    "request failed",
    "name or service not known",
    "network is unreachable",
    "remote end closed",
    "broken pipe",
)
_ACTIVE_HEARTBEATS_LOCK = threading.Lock()
_ACTIVE_HEARTBEATS: dict[Path, "_Heartbeat"] = {}
_TEMPORAL_CUTOFF_METADATA_KEYS = {
    "query_time",
    "query_date",
    "cutoff",
    "time",
    "value",
    "timestamp",
    "datetime",
    "date",
    "event_time",
    "observed",
    "observed_start",
    "observed_end",
    "event_start",
    "event_end",
    "start",
    "end",
    "validity",
    "time_source",
    "timezone",
    "tz",
    "utc_offset",
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "second",
    "microsecond",
    "fold",
}


class RunIdentityMismatch(ValueError):
    """Raised when resume would mix results from different run identities."""


class FrozenPlanError(ValueError):
    """Raised when an existing task plan is missing, changed, or malformed."""


class PointwiseProtocolMismatch(ValueError):
    """Raised with the complete measured numerical probe report attached."""

    def __init__(self, report: Mapping[str, Any]):
        self.report = dict(report)
        super().__init__(
            "reranker pointwise scores depend on batch composition/order: "
            f"max_abs={self.report.get('max_absolute_deviation')}, "
            f"max_rel={self.report.get('max_relative_deviation')}"
        )


class DependencyServiceBundle(Protocol):
    """Documentation protocol for explicit fake/deployment service bundles."""

    embedder: Any
    reranker: Any
    generator: Any


class _CountingServiceProxy:
    """Count logical adapter invocations while preserving the service API."""

    def __init__(
        self,
        delegate: Any,
        counters: dict[str, int],
        method_kinds: Mapping[str, str],
    ) -> None:
        self._delegate = delegate
        self._counters = counters
        self._method_kinds = dict(method_kinds)

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._delegate, name)
        kind = self._method_kinds.get(name)
        if kind is None or not callable(value):
            return value

        def counted(*args: Any, **kwargs: Any) -> Any:
            self._counters[kind] = self._counters.get(kind, 0) + 1
            return value(*args, **kwargs)

        return counted

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not callable(self._delegate):
            raise TypeError(f"{type(self._delegate).__name__} is not callable")
        kind = self._method_kinds.get("__call__", "generator_adapter_invocations")
        self._counters[kind] = self._counters.get(kind, 0) + 1
        return self._delegate(*args, **kwargs)


def _counter_delta(after: Mapping[str, int], before: Mapping[str, int]) -> dict[str, int]:
    return {
        key: int(after.get(key, 0)) - int(before.get(key, 0))
        for key in sorted(set(after) | set(before))
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("run artifacts cannot contain non-finite numbers")
        return value
    public = getattr(value, "public_dict", None)
    if callable(public):
        return _jsonable(public())
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def _safe_query_cutoff_metadata(value: Any) -> Any:
    """Keep temporal cutoff fields while excluding arbitrary answer metadata."""

    if isinstance(value, Mapping):
        safe: dict[str, Any] = {}
        for raw_key, child in sorted(value.items(), key=lambda item: str(item[0])):
            key = str(raw_key)
            normalized = key.strip().lower().replace("-", "_")
            if normalized not in _TEMPORAL_CUTOFF_METADATA_KEYS:
                continue
            safe[key] = _safe_query_cutoff_metadata(child)
        return safe
    if isinstance(value, (list, tuple)):
        return [_safe_query_cutoff_metadata(child) for child in value]
    return _jsonable(value)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value: Any) -> None:
    payload = (json.dumps(
        _jsonable(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n").encode("utf-8")
    _atomic_bytes(path, payload)


def _atomic_jsonl(path: Path, rows: Iterable[Any]) -> None:
    payload = "".join(
        json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        for row in rows
    ).encode("utf-8")
    _atomic_bytes(path, payload)


def _append_jsonl(path: Path, row: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def _append_text(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip("\n") + "\n")
        handle.flush()


def _emit_runtime_event(
    root: Path,
    event: Mapping[str, Any],
    *,
    module_name: str | None = None,
) -> None:
    """Persist one safe structured event and mirror it to captured stdout."""

    value = dict(event)
    if module_name is not None:
        if module_name not in MODULE_LOGS:
            raise ValueError(f"unknown module log: {module_name}")
        _append_jsonl(root / "modules" / f"{module_name}.jsonl", value)
    _append_jsonl(root / "events.jsonl", value)
    line = json.dumps(
        _jsonable(value), ensure_ascii=False, sort_keys=True, allow_nan=False
    )
    _append_text(root / "train.log", line)
    print(line, flush=True)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise FrozenPlanError(f"{path} line {line_number} is not an object")
            rows.append(dict(value))
    return rows


def _recoverable_partial_preparation(
    root: Path, identity: Mapping[str, str]
) -> bool:
    """Recognize only an empty or identity-matched pre-execution skeleton."""

    if not root.is_dir():
        return False
    entries = list(root.iterdir())
    if not entries:
        return True
    if all(entry.name == ".background_worker_ready.json" for entry in entries):
        return True
    marker_path = root / "preparation_identity.json"
    if not marker_path.is_file():
        return False
    try:
        marker = _read_json(marker_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if not isinstance(marker, Mapping):
        return False
    if marker.get("status") != "preparing" or marker.get("identity") != dict(
        identity
    ):
        return False
    allowed_root = {
        ".background_worker_ready.json",
        "preparation_identity.json",
        "planned_tasks.jsonl",
        "resolved_config.json",
        "events.jsonl",
        "predictions.jsonl",
        "failures.jsonl",
        "service_probes.jsonl",
        "outcomes",
        "reports",
        "visible_memories",
        "candidate_pool",
        "modules",
    }
    if any(entry.name not in allowed_root for entry in entries):
        return False
    for name in ("events.jsonl", "predictions.jsonl", "failures.jsonl", "service_probes.jsonl"):
        path = root / name
        if path.exists() and (not path.is_file() or path.stat().st_size != 0):
            return False
    for name in ("outcomes", "reports", "visible_memories", "candidate_pool"):
        directory = root / name
        if directory.exists() and (
            not directory.is_dir() or any(directory.iterdir())
        ):
            return False
    modules = root / "modules"
    if modules.exists():
        if not modules.is_dir():
            return False
        for path in modules.iterdir():
            if (
                not path.is_file()
                or path.name not in {f"{name}.jsonl" for name in MODULE_LOGS}
                or path.stat().st_size != 0
            ):
                return False
    return not (root / "service_probe.json").exists() and not (
        root / "run_manifest.json"
    ).exists()


def _public_config(config: DependencyRunConfig) -> dict[str, Any]:
    """Persist service identities while excluding endpoints and credentials."""

    sensitive = {
        "api_key", "api_key_env", "authorization", "token", "password", "secret"
    }

    def redact(value: Any, key: str = "") -> Any:
        normalized = key.strip().lower()
        if normalized in sensitive:
            return None
        if normalized == "endpoint":
            return {
                "sha256": hashlib.sha256(str(value).encode("utf-8")).hexdigest()
            }
        if isinstance(value, Mapping):
            return {
                str(child_key): redact(child, str(child_key))
                for child_key, child in value.items()
                if str(child_key).strip().lower() not in sensitive
            }
        if isinstance(value, (list, tuple)):
            return [redact(child) for child in value]
        return value

    return redact(config.resolved_dict())


def _source_code_hash() -> str:
    # This is the same package identity used by persisted protocol manifests.
    # It includes untracked source files and remains available in server
    # bundles that do not contain .git.
    from .protocol import _source_package_snapshot

    value = _source_package_snapshot()
    if not value:
        raise RuntimeError("could not compute source package identity")
    return str(value)


@dataclass(frozen=True)
class DependencyDataset:
    examples: tuple[PersonaMemExample, ...]
    dataset_revision: str
    split: str
    source_sha256: Mapping[str, str]
    processed_sha256: Mapping[str, str]
    question_id_sha256: str
    data_hash: str
    manifest: Mapping[str, Any] = field(default_factory=dict)
    synthetic: bool = False

    def public_dict(self) -> dict[str, Any]:
        return {
            "dataset_revision": self.dataset_revision,
            "split": self.split,
            "questions": len(self.examples),
            "source_sha256": dict(self.source_sha256),
            "processed_sha256": dict(self.processed_sha256),
            "question_id_sha256": self.question_id_sha256,
            "data_hash": self.data_hash,
            "synthetic": self.synthetic,
        }


def _question_identity(example: PersonaMemExample) -> dict[str, Any]:
    return {
        "persona_id": str(example.persona_id),
        "question_id": str(example.question_id),
        "question_type": str(example.question_type),
        "topic": str(example.topic),
        "shared_context_id": str(example.shared_context_id),
        "end_index": int(example.end_index),
        "query": str(example.query),
        "correct_answer": str(example.correct_answer),
        "all_options": str(example.all_options),
        "query_time": _jsonable(example.query_time),
        "metadata": _jsonable(example.metadata),
        "visible_messages": _jsonable(example.messages),
    }


def _validate_examples(examples: Sequence[PersonaMemExample]) -> None:
    if not examples:
        raise ValueError("dependency experiment requires a non-empty question set")
    seen: set[str] = set()
    for index, example in enumerate(examples):
        question_id = str(example.question_id)
        if not question_id:
            raise ValueError(f"question {index} has an empty question_id")
        if question_id in seen:
            raise ValueError(f"duplicate question_id: {question_id}")
        seen.add(question_id)
        if not str(example.persona_id) or not str(example.shared_context_id):
            raise ValueError(f"question {question_id} has an empty source identity")
        if not isinstance(example.end_index, int) or isinstance(example.end_index, bool):
            raise ValueError(f"question {question_id} has an invalid visibility cutoff")
        if example.end_index < 0:
            raise ValueError(f"question {question_id} has a negative visibility cutoff")
        if len(example.messages) != example.end_index:
            raise ValueError(
                f"question {question_id} messages do not match end_index visibility cutoff"
            )
        if not str(example.query):
            raise ValueError(f"question {question_id} has an empty query")


def _synthetic_dataset(examples: Sequence[PersonaMemExample], split: str) -> DependencyDataset:
    rows = tuple(examples)
    _validate_examples(rows)
    identities = [_question_identity(example) for example in rows]
    question_hash = hashlib.sha256(
        "\n".join(str(example.question_id) for example in rows).encode("utf-8")
    ).hexdigest()
    data_hash = _sha256_json({"schema": 1, "examples": identities})
    return DependencyDataset(
        examples=rows,
        dataset_revision=f"synthetic:{data_hash}",
        split=str(split),
        source_sha256={},
        processed_sha256={},
        question_id_sha256=question_hash,
        data_hash=data_hash,
        manifest={},
        synthetic=True,
    )


def load_dependency_dataset(
    config: DependencyRunConfig,
    *,
    examples: Sequence[PersonaMemExample] | None = None,
    require_full_32k: bool | None = None,
) -> DependencyDataset:
    """Load and verify the exact raw/processed PersonaMem task source.

    Passing ``examples`` is an explicit synthetic/offline-test boundary.  It
    never happens implicitly when production files are absent and therefore
    cannot become a silent development subsample.
    """

    if examples is not None:
        if require_full_32k:
            raise ValueError("explicit examples cannot satisfy a full PersonaMem-32k preflight")
        return _synthetic_dataset(examples, config.data.split)

    full_required = True if require_full_32k is None else bool(require_full_32k)
    split = str(config.data.split)
    raw_root = Path(config.data.raw_dir)
    processed_root = Path(config.data.processed_dir) / split
    question_path = raw_root / f"questions_{split}.csv"
    context_path = raw_root / f"shared_contexts_{split}.jsonl"
    manifest_path = processed_root / "manifest.json"
    processed_queries_path = processed_root / "queries.jsonl"
    processed_contexts_path = processed_root / "contexts.jsonl"
    required = (
        question_path,
        context_path,
        manifest_path,
        processed_queries_path,
        processed_contexts_path,
    )
    for path in required:
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"PersonaMem data file is missing or empty: {path}")

    manifest = _read_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise ValueError("prepared PersonaMem manifest must be an object")
    if str(manifest.get("revision", "")) != PERSONAMEM_REVISION:
        raise ValueError("prepared PersonaMem revision does not match the pinned revision")
    if str(manifest.get("split", "")) != split:
        raise ValueError("prepared PersonaMem split does not match configuration")

    source_hashes = {
        question_path.name: file_sha256(question_path),
        context_path.name: file_sha256(context_path),
    }
    for name, digest in source_hashes.items():
        if manifest.get("source_sha256", {}).get(name) != digest:
            raise ValueError(f"PersonaMem source checksum mismatch: {name}")
    if full_required:
        if split != "32k":
            raise ValueError("the full dependency experiment requires data.split=32k")
        if source_hashes != PERSONAMEM_SOURCE_SHA256["32k"]:
            raise ValueError("PersonaMem 32k source does not match pinned official checksums")

    processed_hashes = {
        processed_queries_path.name: file_sha256(processed_queries_path),
        processed_contexts_path.name: file_sha256(processed_contexts_path),
    }
    for name, digest in processed_hashes.items():
        if manifest.get("outputs", {}).get(name) != digest:
            raise ValueError(f"PersonaMem processed checksum mismatch: {name}")

    raw_contexts = load_shared_contexts(context_path)
    examples_value = tuple(iter_examples(question_path, context_path))
    if len(examples_value) != int(manifest.get("questions", -1)):
        raise ValueError("parsed question count does not match prepared manifest")
    if full_required and len(examples_value) != 589:
        raise ValueError(
            f"full PersonaMem 32k requires 589 questions, found {len(examples_value)}"
        )
    _validate_examples(examples_value)
    for example in examples_value:
        context = raw_contexts.get(str(example.shared_context_id))
        if context is None:
            raise ValueError(f"question {example.question_id} references an unknown context")
        if example.end_index > len(context):
            raise ValueError(f"question {example.question_id} visibility cutoff exceeds its context")
        if list(example.messages) != list(context[: example.end_index]):
            raise ValueError(f"question {example.question_id} visible messages do not match raw context")

    processed_queries = _read_jsonl(processed_queries_path)
    if len(processed_queries) != len(examples_value):
        raise ValueError("processed and raw question counts differ")
    raw_by_id = {str(example.question_id): example for example in examples_value}
    processed_ids: list[str] = []
    for row in processed_queries:
        question_id = str(row.get("question_id", ""))
        if not question_id or question_id in processed_ids:
            raise ValueError("processed questions contain an empty or duplicate question_id")
        processed_ids.append(question_id)
        raw = raw_by_id.get(question_id)
        if raw is None:
            raise ValueError(f"processed question is absent from raw data: {question_id}")
        try:
            cutoff = int(row.get("end_index_in_shared_context"))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"processed question {question_id} has an invalid cutoff") from exc
        if (
            str(row.get("persona_id", "")) != str(raw.persona_id)
            or str(row.get("shared_context_id", "")) != str(raw.shared_context_id)
            or cutoff != raw.end_index
            or str(row.get("user_question_or_message", "")) != str(raw.query)
        ):
            raise ValueError(f"processed/raw question identity mismatch: {question_id}")
    if set(processed_ids) != set(raw_by_id):
        raise ValueError("processed questions do not cover all raw question identities")

    processed_context_ids: set[str] = set()
    for row in _read_jsonl(processed_contexts_path):
        context_id = str(row.get("shared_context_id", ""))
        messages = row.get("messages")
        if not context_id or context_id in processed_context_ids:
            raise ValueError("processed contexts contain an empty or duplicate identity")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"processed context is empty: {context_id}")
        processed_context_ids.add(context_id)
    missing_contexts = {str(item.shared_context_id) for item in examples_value} - processed_context_ids
    if missing_contexts:
        raise ValueError(f"processed contexts miss referenced IDs: {sorted(missing_contexts)}")

    question_hash = hashlib.sha256(
        "\n".join(str(example.question_id) for example in examples_value).encode("utf-8")
    ).hexdigest()
    data_identity = {
        "schema": 1,
        "revision": PERSONAMEM_REVISION,
        "split": split,
        "source_sha256": source_hashes,
        "processed_sha256": processed_hashes,
        "question_id_sha256": question_hash,
        "questions": [_question_identity(example) for example in examples_value],
    }
    return DependencyDataset(
        examples=examples_value,
        dataset_revision=PERSONAMEM_REVISION,
        split=split,
        source_sha256=source_hashes,
        processed_sha256=processed_hashes,
        question_id_sha256=question_hash,
        data_hash=_sha256_json(data_identity),
        manifest=dict(manifest),
        synthetic=False,
    )


def _load_protocol_manifest(
    value: str | Path | Mapping[str, Any] | None,
    dataset: DependencyDataset,
) -> tuple[dict[str, str], dict[str, Any], str]:
    """Validate an optional authoritative role map without inventing a split."""

    if value is None:
        report = {
            "status": "not_defined",
            "reason": "no authoritative protocol manifest supplied",
            "roles": {},
        }
        return {}, report, "not_defined"
    if isinstance(value, Mapping):
        manifest = dict(value)
    else:
        path = Path(value)
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"protocol manifest is missing or empty: {path}")
        loaded = _read_json(path)
        if not isinstance(loaded, Mapping):
            raise ValueError("protocol manifest must be an object")
        manifest = dict(loaded)
    if manifest.get("frozen") is not True:
        raise ValueError("protocol manifest must be frozen")
    if str(manifest.get("dataset_revision", "")) != dataset.dataset_revision:
        raise ValueError("protocol manifest dataset revision mismatch")
    if str(manifest.get("split", "")) != dataset.split:
        raise ValueError("protocol manifest split mismatch")

    roles = manifest.get("roles")
    if not isinstance(roles, Mapping):
        raise ValueError("protocol manifest has no roles mapping")

    def role_ids(*names: str) -> tuple[str, ...]:
        record: Mapping[str, Any] | None = None
        for name in names:
            item = roles.get(name, manifest.get(name))
            if isinstance(item, Mapping):
                record = item
                break
        if record is None:
            return ()
        raw = record.get("question_ids", record.get("ids", ()))
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            raise ValueError(f"protocol role {names[0]} question_ids must be a sequence")
        result = tuple(str(item) for item in raw)
        if any(not item for item in result) or len(result) != len(set(result)):
            raise ValueError(f"protocol role {names[0]} has empty or duplicate question IDs")
        declared_hash = record.get("question_id_sha256", record.get("question_hash"))
        actual_hash = hashlib.sha256("\n".join(result).encode("utf-8")).hexdigest()
        if declared_hash not in (None, "", actual_hash):
            raise ValueError(f"protocol role {names[0]} question hash mismatch")
        declared_count = record.get("queries")
        if declared_count is not None and int(declared_count) != len(result):
            raise ValueError(f"protocol role {names[0]} question count mismatch")
        return result

    seen = role_ids("development-seen", "development_seen", "development")
    confirmation = role_ids("confirmatory-test", "confirmatory_test", "confirmation")
    full = role_ids("full-benchmark", "full_benchmark", "full")
    dataset_ids = {str(example.question_id) for example in dataset.examples}
    if not seen or not confirmation:
        raise ValueError("protocol manifest must define seen and confirmation roles")
    if set(seen).intersection(confirmation):
        raise ValueError("protocol seen and confirmation roles overlap")
    if set(seen).union(confirmation) != dataset_ids:
        raise ValueError("protocol seen/confirmation roles do not cover the dataset exactly")
    if full and set(full) != dataset_ids:
        raise ValueError("protocol full role does not cover the dataset exactly")
    role_by_question = {question_id: "seen" for question_id in seen}
    role_by_question.update({question_id: "confirmation" for question_id in confirmation})
    identity = _sha256_json(manifest)
    return role_by_question, {
        "status": "defined",
        "identity": identity,
        "seen_questions": len(seen),
        "confirmation_questions": len(confirmation),
        "roles": {
            "seen": "development-seen",
            "confirmation": "confirmatory-test",
        },
    }, identity


@dataclass(frozen=True)
class DependencyTask:
    dataset_revision: str
    split: str
    persona_id: str
    question_id: str
    method_id: str
    config_hash: str
    data_hash: str
    source_code_hash: str
    protocol_hash: str
    protocol_role: str = "all_data"
    schema_version: int = TASK_SCHEMA_VERSION

    @property
    def key(self) -> tuple[str, ...]:
        return (
            self.dataset_revision,
            self.split,
            self.persona_id,
            self.question_id,
            self.method_id,
            self.config_hash,
            self.data_hash,
            self.source_code_hash,
            self.protocol_hash,
            self.protocol_role,
        )

    @property
    def task_id(self) -> str:
        return hashlib.sha256("\0".join(self.key).encode("utf-8")).hexdigest()

    def public_dict(self) -> dict[str, Any]:
        return {**asdict(self), "task_id": self.task_id}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DependencyTask":
        fields = set(cls.__dataclass_fields__)
        kwargs = {key: value[key] for key in fields if key in value}
        task = cls(**kwargs)
        declared = value.get("task_id")
        if declared not in (None, task.task_id):
            raise FrozenPlanError("planned task_id does not match task identity")
        return task


def build_dependency_plan(
    dataset: DependencyDataset,
    *,
    config_hash: str,
    source_code_hash: str,
    protocol_hash: str,
    methods: Sequence[str],
    role_by_question: Mapping[str, str] | None = None,
) -> list[DependencyTask]:
    if not methods or len(methods) != len(set(methods)):
        raise ValueError("dependency methods must be non-empty and unique")
    roles = dict(role_by_question or {})
    tasks: list[DependencyTask] = []
    for example in dataset.examples:
        role = roles.get(str(example.question_id), "all_data")
        for method in methods:
            tasks.append(
                DependencyTask(
                    dataset_revision=dataset.dataset_revision,
                    split=dataset.split,
                    persona_id=str(example.persona_id),
                    question_id=str(example.question_id),
                    method_id=str(method),
                    config_hash=config_hash,
                    data_hash=dataset.data_hash,
                    source_code_hash=source_code_hash,
                    protocol_hash=protocol_hash,
                    protocol_role=role,
                )
            )
    if not tasks:
        raise ValueError("dependency task plan cannot be empty")
    return tasks


@dataclass(frozen=True)
class DependencyOutcome:
    task: DependencyTask
    status: str
    attempt: int
    started_at_epoch: float
    finished_at_epoch: float
    prediction: str | None = None
    predicted_label: str | None = None
    correct: bool | None = None
    parse_failed: bool | None = None
    selected_ids: tuple[str, ...] = ()
    context_hash: str | None = None
    error_type: str | None = None
    error: str | None = None
    infrastructure_failure: bool = False
    costs: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = OUTCOME_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.status not in {"success", "error"}:
            raise ValueError("outcome status must be success or error")
        if not isinstance(self.attempt, int) or isinstance(self.attempt, bool) or self.attempt <= 0:
            raise ValueError("outcome attempt must be a positive integer")
        if self.status == "success":
            if self.prediction is None or self.correct is None or self.error is not None:
                raise ValueError("successful outcomes require prediction/correct and no error")
        else:
            if not self.error:
                raise ValueError("error outcomes require an error message")

    def public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task": self.task.public_dict(),
            "status": self.status,
            "attempt": self.attempt,
            "started_at_epoch": self.started_at_epoch,
            "finished_at_epoch": self.finished_at_epoch,
            "prediction": self.prediction,
            "predicted_label": self.predicted_label,
            "correct": self.correct,
            "parse_failed": self.parse_failed,
            "selected_ids": list(self.selected_ids),
            "context_hash": self.context_hash,
            "error_type": self.error_type,
            "error": self.error,
            "infrastructure_failure": self.infrastructure_failure,
            "costs": dict(self.costs),
            "diagnostics": dict(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DependencyOutcome":
        task_value = value.get("task")
        if not isinstance(task_value, Mapping):
            raise ValueError("outcome has no task identity")
        return cls(
            task=DependencyTask.from_dict(task_value),
            status=str(value.get("status", "")),
            attempt=int(value.get("attempt", 0)),
            started_at_epoch=float(value.get("started_at_epoch", 0.0)),
            finished_at_epoch=float(value.get("finished_at_epoch", 0.0)),
            prediction=value.get("prediction"),
            predicted_label=value.get("predicted_label"),
            correct=value.get("correct"),
            parse_failed=value.get("parse_failed"),
            selected_ids=tuple(str(item) for item in value.get("selected_ids", ())),
            context_hash=value.get("context_hash"),
            error_type=value.get("error_type"),
            error=value.get("error"),
            infrastructure_failure=bool(value.get("infrastructure_failure", False)),
            costs=dict(value.get("costs", {})),
            diagnostics=dict(value.get("diagnostics", {})),
            schema_version=int(value.get("schema_version", OUTCOME_SCHEMA_VERSION)),
        )


def _mapping_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        return [dict(value)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [dict(item) for item in value if isinstance(item, Mapping)]
    return []


def _nonnegative_count(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return 0


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _selection_rows(selection: Mapping[str, Any]) -> list[dict[str, Any]]:
    steps = _mapping_rows(selection.get("steps"))
    if steps:
        return steps
    rows: list[dict[str, Any]] = []
    for round_value in _mapping_rows(selection.get("rounds")):
        rows.extend(_mapping_rows(round_value.get("comparisons")))
    return rows


def _module_effectiveness_event(
    task: DependencyTask,
    outcome: DependencyOutcome,
    artifacts: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build a compact, label-safe audit view from one authoritative attempt."""

    artifact_value = dict(artifacts or {})
    evidence_method = task.method_id == "evidence_bridge"
    visible = artifact_value.get("visible_memories")
    visible_value = dict(visible) if isinstance(visible, Mapping) else {}
    candidate = artifact_value.get("candidate_pool")
    candidate_value = dict(candidate) if isinstance(candidate, Mapping) else {}
    retrieval = candidate_value.get("retrieval")
    retrieval_value = dict(retrieval) if isinstance(retrieval, Mapping) else {}
    search = candidate_value.get("search")
    search_value = dict(search) if isinstance(search, Mapping) else {}
    selection = candidate_value.get("selection")
    selection_value = dict(selection) if isinstance(selection, Mapping) else {}
    baseline_rows = _mapping_rows(candidate_value.get("baseline_selection"))
    module_events = artifact_value.get("module_events")
    module_event_value = (
        dict(module_events) if isinstance(module_events, Mapping) else {}
    )

    proposal_batches = _mapping_rows(retrieval_value.get("proposal_batches"))
    proposed_ids: set[str] = set()
    conditional_proposed_ids: set[str] = set()
    returned_hits = 0
    initial_dense_ids: set[str] = set()
    for batch in proposal_batches:
        candidate_ids = [
            str(identifier)
            for identifier in batch.get("candidate_ids", ())
            if str(identifier)
        ]
        if not candidate_ids:
            candidate_ids = [
                str(hit.get("memory_id"))
                for hit in _mapping_rows(batch.get("hits"))
                if hit.get("memory_id") not in (None, "")
            ]
        proposed_ids.update(candidate_ids)
        if batch.get("stage") == "conditional":
            conditional_proposed_ids.update(candidate_ids)
        returned_hits += len(candidate_ids)
        if batch.get("stage") == "initial_dense":
            initial_dense_ids.update(candidate_ids)
    initial_pool_ids = {
        str(identifier)
        for identifier in retrieval_value.get("initial_candidate_ids", ())
        if str(identifier)
    }
    if not initial_pool_ids:
        initial_pool_ids = initial_dense_ids

    costs = dict(outcome.costs)
    scorer_cost = costs.get("set_scorer")
    scorer_value = dict(scorer_cost) if isinstance(scorer_cost, Mapping) else {}
    reranker_transport_raw = scorer_value.get("reranker_transport")
    reranker_transport = (
        {
            str(key): _nonnegative_count(value)
            for key, value in reranker_transport_raw.items()
            if isinstance(key, str)
            and not isinstance(value, bool)
            and isinstance(value, (int, np.integer))
        }
        if isinstance(reranker_transport_raw, Mapping)
        else {}
    )
    cache_hits = _nonnegative_count(
        scorer_value.get("cache_hits", costs.get("cache_hits", 0))
    )
    reranker_samples = _nonnegative_count(scorer_value.get("reranker_samples", 0))
    cache_lookups = cache_hits + reranker_samples
    scoring_rows = _mapping_rows(module_event_value.get("scoring"))

    activations = _mapping_rows(search_value.get("activations"))
    if not activations:
        activations = _mapping_rows(module_event_value.get("activation"))
    signals = [
        signal
        for signal in (_finite_number(row.get("signal")) for row in activations)
        if signal is not None
    ]
    positive_signals = [signal for signal in signals if signal > 0.0]
    states = _mapping_rows(search_value.get("states"))
    if not states:
        states = _mapping_rows(module_event_value.get("state"))
    bundles = _mapping_rows(search_value.get("bundles"))
    skipped_measurements = _mapping_rows(search_value.get("skipped_measurements"))
    skipped_pair_measurements = sum(
        isinstance(row.get("group_ids"), Sequence)
        and not isinstance(row.get("group_ids"), (str, bytes))
        and len(row["group_ids"]) == 2
        for row in skipped_measurements
    )
    premise_depths = [
        len(row.get("premise_ids", ()))
        for row in activations
        if isinstance(row.get("premise_ids", ()), Sequence)
        and not isinstance(row.get("premise_ids"), (str, bytes))
    ]
    for row in states:
        state = row.get("state")
        if isinstance(state, Mapping):
            premise_ids = state.get("premise_ids", ())
            if isinstance(premise_ids, Sequence) and not isinstance(
                premise_ids, (str, bytes)
            ):
                premise_depths.append(len(premise_ids))
    bundle_sizes = [
        len(row.get("memory_ids", ()))
        for row in bundles
        if isinstance(row.get("memory_ids", ()), Sequence)
        and not isinstance(row.get("memory_ids"), (str, bytes))
    ]

    dynamic_selection = bool(selection_value)
    selection_rounds = _mapping_rows(selection_value.get("rounds"))
    comparisons = (
        _selection_rows(selection_value) if dynamic_selection else baseline_rows
    )
    if not comparisons:
        comparisons = [
            row
            for row in _mapping_rows(module_event_value.get("selection"))
            if "accepted" in row
        ]
    marginals = [
        marginal
        for marginal in (_finite_number(row.get("marginal")) for row in comparisons)
        if marginal is not None
    ]
    accepted_marginals = [
        marginal
        for row in comparisons
        if row.get("accepted") is True
        for marginal in [_finite_number(row.get("marginal"))]
        if marginal is not None
    ]
    feasible_comparisons = sum(
        row.get("feasible") is not False
        if dynamic_selection
        else row.get("accepted") is True
        for row in comparisons
    )
    accepted_decisions = sum(row.get("accepted") is True for row in comparisons)
    diagnostics = dict(outcome.diagnostics)
    selection_stop = diagnostics.get("selection_stop_reason")
    stop_value = selection_value.get("stop")
    if selection_stop is None and isinstance(stop_value, Mapping):
        selection_stop = stop_value.get("reason")
    if selection_stop is None and not dynamic_selection and outcome.status == "success":
        selection_stop = "ranked_candidates_exhausted"
    search_stop = diagnostics.get("search_stop_reason")
    if search_stop is None:
        search_stop = search_value.get("stop_reason")
    context_plan = artifact_value.get("context_plan")
    context_value = dict(context_plan) if isinstance(context_plan, Mapping) else {}
    adapter_invocations = costs.get("adapter_invocations")
    adapter_value = (
        dict(adapter_invocations) if isinstance(adapter_invocations, Mapping) else {}
    )
    embedding_invocations = _nonnegative_count(
        adapter_value.get("embedding_adapter_invocations", 0)
    )
    memory_embedding_invocations = _nonnegative_count(
        costs.get("memory_embedding_adapter_invocations", 0)
    )

    elapsed_ms = _finite_number(costs.get("elapsed_ms"))
    if elapsed_ms is None:
        elapsed_ms = max(
            0.0,
            (outcome.finished_at_epoch - outcome.started_at_epoch) * 1000.0,
        )
    selected_count = len(outcome.selected_ids)
    if not selected_count:
        selected_count = len(
            {
                str(identifier)
                for identifier in artifact_value.get("selected_ids", ())
                if str(identifier)
            }
        )

    return {
        "schema_version": 1,
        "event": "module_effectiveness",
        "at_epoch": outcome.finished_at_epoch,
        "task_id": task.task_id,
        "persona_id": task.persona_id,
        "question_id": task.question_id,
        "method_id": task.method_id,
        "protocol_role": task.protocol_role,
        "attempt": outcome.attempt,
        "status": outcome.status,
        "partial": outcome.status != "success",
        "authoritative_at_write": True,
        "evidence_bridge": diagnostics.get("evidence_bridge_summary") if evidence_method else None,
        "retrieval": {
            "available": bool(retrieval_value),
            "visible_memory_count": len(visible_value.get("visible_memory_ids", ())),
            "initial_dense_candidate_count": len(initial_dense_ids),
            "initial_pool_unique_candidate_count": len(initial_pool_ids),
            "proposal_batch_count_total": len(proposal_batches),
            "conditional_proposal_batch_count": sum(
                batch.get("stage") == "conditional" for batch in proposal_batches
            ),
            "returned_candidate_hits_total": returned_hits,
            "unique_proposed_candidate_count": len(proposed_ids),
            "unique_candidates_discovered_outside_initial_pool": len(
                conditional_proposed_ids.difference(initial_pool_ids)
            ),
            "ann_calls_total": _nonnegative_count(
                costs.get("ann_calls", retrieval_value.get("ann_calls", 0))
            ),
            "max_ann_calls": retrieval_value.get("max_ann_calls"),
            "remaining_ann_calls": retrieval_value.get("remaining_ann_calls"),
            "ann_budget_exhausted_batch_count": sum(
                batch.get("stop_reason") == "ann_budget_exhausted"
                for batch in proposal_batches
            ),
            "fixed_pool": bool(retrieval_value.get("fixed_pool", False)),
            "proposal_edges_are_dependency_claims": False,
        },
        "scoring": {
            "available": bool(scorer_value or scoring_rows),
            "score_space": diagnostics.get("score_space"),
            "score_contract": diagnostics.get("score_contract"),
            "logical_unique_sets_charged_total": _nonnegative_count(
                scorer_value.get("scored_sets", costs.get("scored_sets", 0))
            ),
            "completed_set_score_event_count": len(scoring_rows),
            "reranker_logical_adapter_invocations_total": _nonnegative_count(
                scorer_value.get(
                    "reranker_adapter_requests",
                    costs.get("reranker_adapter_requests", 0),
                )
            ),
            "reranker_samples_attempted_total": reranker_samples,
            # Logical search/set budgets above deliberately exclude physical
            # HTTP retries and pointwise batch subdivision. This nested view
            # exposes recovery work without changing experiment accounting.
            "reranker_transport": reranker_transport,
            "cache_hit_events_total": cache_hits,
            "persistent_cache_hit_events_total": _nonnegative_count(
                scorer_value.get("persistent_cache_hits", 0)
            ),
            "memory_cache_hit_events_total": _nonnegative_count(
                scorer_value.get("memory_cache_hits", 0)
            ),
            "cache_lookup_events_total": cache_lookups,
            "cache_hit_event_rate": (
                cache_hits / cache_lookups if cache_lookups else None
            ),
            "logical_input_tokens_estimate": scorer_value.get(
                "logical_input_tokens_estimate"
            ),
            "reranker_elapsed_ms": scorer_value.get("reranker_elapsed_ms"),
            "token_count_is_estimate": bool(
                scorer_value.get("token_count_is_estimate", True)
            ),
        },
        "dependency_search": {
            "available": bool(search_value),
            "enabled": task.method_id not in {"dense", "dense_rerank"},
            "signal_kind": search_value.get("signal_kind"),
            "initial_target_count": len(search_value.get("initial_target_ids", ())),
            "interaction_measurement_count": len(activations),
            "singleton_measurement_count": sum(
                row.get("group_kind") == "singleton" for row in activations
            ),
            "pair_measurement_completed_count": sum(
                row.get("group_kind") == "pair" for row in activations
            ),
            "pair_measurement_attempt_count": sum(
                row.get("group_kind") == "pair" for row in activations
            ) + skipped_pair_measurements,
            "skipped_measurement_count": len(skipped_measurements),
            "positive_pair_signal_count": sum(
                row.get("group_kind") == "pair"
                and (_finite_number(row.get("signal")) or 0.0) > 0.0
                for row in activations
            ),
            "positive_signal_count": len(positive_signals),
            "accepted_signal_count": sum(
                row.get("accepted") is True for row in activations
            ),
            "queued_successor_count": sum(
                row.get("queued") is True for row in activations
            ),
            "max_signal": max(signals) if signals else None,
            "max_positive_signal": max(positive_signals) if positive_signals else None,
            "visited_state_count": len(states),
            "expanded_state_count": sum(
                row.get("event") == "expanded" for row in states
            ),
            "positive_successor_count_total": sum(
                _nonnegative_count(row.get("positive_successors", 0))
                for row in states
            ),
            "states_with_positive_successors": sum(
                _nonnegative_count(row.get("positive_successors", 0)) > 0
                for row in states
            ),
            "max_premise_count": max(premise_depths) if premise_depths else 0,
            "archived_bundle_count": len(bundles),
            "max_archived_bundle_size": max(bundle_sizes) if bundle_sizes else 0,
            "ann_calls_delta": search_value.get("ann_calls"),
            "logical_unique_sets_charged_delta": search_value.get("scored_sets"),
            "search_stop_reason": search_stop,
            "global_certificate": bool(search_value.get("global_certificate", False)),
            "interpretation": INTERACTION_SIGNAL_CAVEAT,
        },
        "selection": {
            "available": bool(
                dynamic_selection or baseline_rows or outcome.status == "success"
            ),
            "mode": (
                "evidence_requirement_set_selection" if evidence_method
                else "dynamic_bundle_marginal" if dynamic_selection else "baseline_rank_order"
            ),
            "round_count": len(selection_rounds),
            "complete_round_count": sum(
                row.get("complete") is True for row in selection_rounds
            ),
            "incomplete_round_count": sum(
                row.get("complete") is False for row in selection_rounds
            ),
            "comparison_count": None if evidence_method else len(comparisons),
            "feasible_comparison_count": None if evidence_method else feasible_comparisons,
            "positive_marginal_count": None if evidence_method else sum(value > 0.0 for value in marginals),
            "accepted_decision_count": None if evidence_method else accepted_decisions,
            "accepted_bundle_count": accepted_decisions if dynamic_selection and not evidence_method else None,
            "accepted_baseline_memory_count": (
                accepted_decisions if not dynamic_selection else None
            ),
            "final_memory_count": selected_count,
            "max_accepted_marginal": (
                max(accepted_marginals) if accepted_marginals else None
            ),
            "accepted_marginal_sequence": None if evidence_method else accepted_marginals,
            "logical_unique_sets_charged_delta": selection_value.get(
                "scored_sets"
            ),
            "selection_stop_reason": selection_stop,
            "generator_input_tokens_estimate": costs.get(
                "generator_input_tokens_estimate"
            ),
            "token_count_is_estimate": bool(
                costs.get("token_count_is_estimate", True)
            ),
            "context_budget": context_value.get("budget"),
            "context_budget_status": context_value.get("budget_status"),
            "context_within_budget": (
                context_value.get("budget_status") == "within_budget"
                if context_value
                else None
            ),
        },
        "generation_evaluation": {
            "generator_calls": _nonnegative_count(costs.get("generator_calls", 0)),
            "generator_adapter_invocations_total": _nonnegative_count(
                costs.get("generator_calls", 0)
            ),
            "predicted_label": outcome.predicted_label,
            "context_hash": outcome.context_hash or artifact_value.get("context_hash"),
            "correct": outcome.correct if outcome.status == "success" else None,
            "parse_failed": (
                outcome.parse_failed if outcome.status == "success" else None
            ),
            "evaluation_performed": outcome.status == "success",
            "evaluation_only_after_generation": True,
        },
        "cost_accounting": {
            "elapsed_ms": elapsed_ms,
            "elapsed_ms_scope": costs.get("elapsed_ms_scope", "task_executor"),
            "memory_vector_cache_hit": costs.get("memory_vector_cache_hit"),
            "memory_embedding_adapter_invocations": memory_embedding_invocations,
            "memory_embedding_samples": _nonnegative_count(
                costs.get("memory_embedding_samples", 0)
            ),
            "query_or_proposal_embedding_adapter_invocations": (
                max(0, embedding_invocations - memory_embedding_invocations)
                if retrieval_value
                else None
            ),
            "unclassified_embedding_adapter_invocations": (
                0 if retrieval_value else embedding_invocations
            ),
            "adapter_invocations": adapter_value,
            "scorer_logical_input_tokens_estimate": scorer_value.get(
                "logical_input_tokens_estimate"
            ),
            "generator_input_tokens_estimate": costs.get(
                "generator_input_tokens_estimate"
            ),
            "reranker_elapsed_ms": scorer_value.get("reranker_elapsed_ms"),
            "shared_question_embedding_cost_is_method_order_dependent": True,
            "stage_deltas_must_not_be_added_to_task_totals": True,
        },
        "error": (
            None
            if outcome.status == "success"
            else {
                "error_type": outcome.error_type,
                "infrastructure_failure": outcome.infrastructure_failure,
            }
        ),
    }


def _outcome_path(root: Path, task: DependencyTask) -> Path:
    return root / "outcomes" / f"{task.task_id}.json"


def _read_outcome(root: Path, task: DependencyTask) -> DependencyOutcome | None:
    path = _outcome_path(root, task)
    if not path.is_file():
        return None
    value = _read_json(path)
    if not isinstance(value, Mapping):
        raise ValueError(f"outcome is not an object: {path}")
    result = DependencyOutcome.from_dict(value)
    if result.task.key != task.key:
        raise RunIdentityMismatch(f"outcome task identity differs from frozen plan: {path}")
    return result


def load_authoritative_outcomes(
    root: str | Path, tasks: Sequence[DependencyTask]
) -> dict[str, DependencyOutcome]:
    result: dict[str, DependencyOutcome] = {}
    for task in tasks:
        outcome = _read_outcome(Path(root), task)
        if outcome is not None:
            result[task.task_id] = outcome
    return result


def _metric_row(
    tasks: Sequence[DependencyTask],
    outcomes: Mapping[str, DependencyOutcome],
    *,
    method_id: str,
    role: str,
) -> dict[str, Any]:
    scoped = [
        task for task in tasks
        if (method_id == "all" or task.method_id == method_id)
        and (role == "all_data" or task.protocol_role == role)
    ]
    expected = len(scoped)
    observed = [outcomes[task.task_id] for task in scoped if task.task_id in outcomes]
    successful = [outcome for outcome in observed if outcome.status == "success"]
    failures = [outcome for outcome in observed if outcome.status == "error"]
    correct = sum(outcome.correct is True for outcome in successful)
    parse_failures = sum(outcome.parse_failed is True for outcome in successful)
    completed = len(observed)
    return {
        "method_id": method_id,
        "role": role,
        "expected_tasks": expected,
        "completed_tasks": completed,
        "successful_tasks": len(successful),
        "failed_tasks": len(failures),
        "pending_tasks": max(0, expected - completed),
        "correct": correct,
        "incorrect": len(successful) - correct,
        "incorrect_successful_outputs": len(successful) - correct,
        "parse_failures": parse_failures,
        "coverage": completed / expected if expected else 1.0,
        # Failures remain in the denominator.  This field is useful while a
        # run is pending but must not be described as final accuracy.
        "denominator_accuracy": correct / expected if expected else None,
        "accuracy_lower_bound": correct / expected if expected else None,
        "successful_accuracy": correct / len(successful) if successful else None,
        "final_accuracy": correct / expected if expected and completed == expected else None,
        "status": (
            "complete" if expected and completed == expected
            else "pending" if expected
            else "not_defined"
        ),
    }


def summarize_dependency_outcomes(
    tasks: Sequence[DependencyTask],
    outcomes: Mapping[str, DependencyOutcome] | Sequence[DependencyOutcome],
) -> dict[str, Any]:
    by_id = (
        dict(outcomes)
        if isinstance(outcomes, Mapping)
        else {outcome.task.task_id: outcome for outcome in outcomes}
    )
    aggregate = _metric_row(tasks, by_id, method_id="all", role="all_data")
    methods = []
    for method in dict.fromkeys(task.method_id for task in tasks):
        methods.append(_metric_row(tasks, by_id, method_id=method, role="all_data"))
    return {
        **{key: value for key, value in aggregate.items() if key not in {"method_id", "role"}},
        "methods": methods,
        "optimizer_steps": 0,
        "weights_updated": False,
    }


def _method_metrics_event(
    tasks: Sequence[DependencyTask],
    outcomes: Mapping[str, DependencyOutcome],
    *,
    attempt: int,
    tasks_attempted_this_attempt: int,
    unique_tasks_attempted_this_attempt: int,
    checkpoint: str,
    tasks_skipped_this_attempt: int = 0,
) -> dict[str, Any]:
    summary = summarize_dependency_outcomes(tasks, outcomes)
    method_ids = list(dict.fromkeys(task.method_id for task in tasks))
    completed_questions_by_method = {
        method_id: {
            task.question_id
            for task in tasks
            if task.method_id == method_id and task.task_id in outcomes
        }
        for method_id in method_ids
    }
    completed_sets = list(completed_questions_by_method.values())
    common_completed = set.intersection(*completed_sets) if completed_sets else set()
    completed_sets_equal = all(
        values == completed_sets[0] for values in completed_sets[1:]
    ) if completed_sets else True
    planned_questions = {task.question_id for task in tasks}
    aggregate = {
        key: value
        for key, value in summary.items()
        if key not in {"methods", "optimizer_steps", "weights_updated"}
    }
    return {
        "schema_version": 1,
        "event": "method_metrics",
        "at_epoch": time.time(),
        "attempt": attempt,
        "checkpoint": checkpoint,
        "tasks_attempted_this_attempt": tasks_attempted_this_attempt,
        "unique_tasks_attempted_this_attempt": unique_tasks_attempted_this_attempt,
        "task_retries_this_attempt": max(
            0,
            tasks_attempted_this_attempt - unique_tasks_attempted_this_attempt,
        ),
        "tasks_skipped_this_attempt": tasks_skipped_this_attempt,
        "tasks_attempted_unit": "executor_invocation",
        "task_unit": "method_x_question",
        "progress_cadence_unit": "unique_method_x_question",
        "interval_resets_on_resume": True,
        "aggregate_scope": "all_method_task_micro",
        "metrics_scope": "authoritative_outcomes_only",
        "metrics_are_not_a_module_cost_ranking": True,
        "planned_common_question_count": len(planned_questions),
        "common_completed_question_count": len(common_completed),
        "completed_method_question_sets_equal": completed_sets_equal,
        "method_question_sets_comparable": (
            completed_sets_equal and common_completed == planned_questions
        ),
        "aggregate": aggregate,
        "methods": summary["methods"],
        "optimizer_steps": 0,
        "weights_updated": False,
    }


def _memory_public(memory: Memory) -> dict[str, Any]:
    return {
        "memory_id": str(memory.memory_id),
        "text": str(memory.text),
        "timestamp": float(memory.timestamp),
        "source_id": str(memory.source_id),
        "metadata": _jsonable(memory.metadata),
    }


def _generation_plan_for_ids(
    config: DependencyRunConfig,
    example: PersonaMemExample,
    records: Mapping[str, Memory],
    ids: Sequence[str],
    *,
    strict: bool,
) -> ContextPlan:
    normalized = tuple(str(identifier) for identifier in ids)
    if len(normalized) != len(set(normalized)):
        raise ValueError("selected generation IDs must be unique")
    unknown = set(normalized).difference(records)
    if unknown:
        raise ValueError(f"selected generation IDs are not visible: {sorted(unknown)}")
    return build_context_plan(
        example.query,
        [records[identifier] for identifier in normalized],
        example.all_options,
        token_budget=config.models.generator.context_token_budget,
        strict=strict,
        selected_ids=normalized,
        generator_config=config.models.generator,
    )


def _baseline_context(
    config: DependencyRunConfig,
    example: PersonaMemExample,
    records: Mapping[str, Memory],
    ranked_ids: Sequence[str],
) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    """Admit baseline memories in rank order under the complete prompt budget."""

    selected: list[str] = []
    decisions: list[dict[str, Any]] = []
    for identifier in ranked_ids:
        candidate = str(identifier)
        if candidate in selected:
            continue
        trial = tuple((*selected, candidate))
        plan = _generation_plan_for_ids(config, example, records, trial, strict=False)
        accepted = bool(plan.within_budget)
        decisions.append(
            {
                "event": "baseline_context_candidate",
                "candidate_id": candidate,
                "selected_before": list(selected),
                "estimated_input_tokens": plan.token_count,
                "token_count_is_estimate": plan.token_count_is_estimate,
                "budget": plan.budget,
                "accepted": accepted,
                "detail": "within_budget" if accepted else "generator_input_capacity",
            }
        )
        if accepted:
            selected.append(candidate)
    return tuple(selected), decisions


class DependencyTaskExecutor:
    """Execute one frozen method×question task using real or fake services.

    Correct labels are deliberately not passed to any helper that can invoke
    a model.  They are read only after ``_answer`` has returned.
    """

    def __init__(
        self,
        config: DependencyRunConfig,
        *,
        embedder: Any | None = None,
        reranker: Any | None = None,
        generator: Any | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        if not isinstance(config, DependencyRunConfig):
            raise TypeError("DependencyTaskExecutor requires DependencyRunConfig")
        self.config = config
        raw_embedder = embedder if embedder is not None else build_embedder(
            config.models.embedding, device=config.runtime.device
        )
        raw_reranker = reranker if reranker is not None else RerankerClient(
            config.models.reranker
        )
        raw_generator = generator if generator is not None else GeneratorClient(
            config.models.generator
        )
        self._adapter_counters: dict[str, int] = {
            "embedding_adapter_invocations": 0,
            "reranker_adapter_invocations": 0,
            "generator_adapter_invocations": 0,
        }
        self.embedder = _CountingServiceProxy(
            raw_embedder,
            self._adapter_counters,
            {
                "encode": "embedding_adapter_invocations",
                "encode_query": "embedding_adapter_invocations",
                "encode_queries": "embedding_adapter_invocations",
            },
        )
        self.reranker = _CountingServiceProxy(
            raw_reranker,
            self._adapter_counters,
            {
                "rerank": "reranker_adapter_invocations",
                "rerank_all": "reranker_adapter_invocations",
            },
        )
        self.generator = _CountingServiceProxy(
            raw_generator,
            self._adapter_counters,
            {
                "answer": "generator_adapter_invocations",
                "answer_plan": "generator_adapter_invocations",
                "complete_messages": "evidence_adapter_invocations",
                "__call__": "generator_adapter_invocations",
            },
        )
        self.cache_dir = Path(
            cache_dir if cache_dir is not None else config.runtime.cache_dir
        ) / "dependency_set_scores"
        self._memory_vector_cache: dict[str, np.ndarray] = {}
        self.evidence_snapshot_sink = None

    @property
    def adapter_cost(self) -> dict[str, int]:
        return dict(self._adapter_counters)

    def visible_memories(self, example: PersonaMemExample) -> tuple[Memory, ...]:
        # The CSV loader already slices messages at end_index.  The semantic
        # visibility helper then applies any explicit time cutoff before a
        # document is embedded or inserted into a scoring cache.
        memories = messages_to_memories(
            example.messages,
            source_prefix=str(example.question_id),
            include_system_persona=self.config.data.include_system_persona,
            memory_granularity=self.config.data.memory_granularity,
        )
        visible = tuple(_visible_memory_records(example, memories))
        if not visible:
            raise ValueError("no visible memories remain after segmentation and cutoff")
        ids = tuple(str(memory.memory_id) for memory in visible)
        if len(ids) != len(set(ids)):
            raise ValueError("visible memory IDs are not unique")
        return visible

    def _retriever(
        self,
        example: PersonaMemExample,
        memories: Sequence[Memory],
        *,
        fixed_pool: bool,
    ) -> tuple[DependencyRetriever, bool]:
        settings = self.config.dependency
        vector_key = _sha256_json(
            {
                "embedding": {
                    "model": self.config.models.embedding.model,
                    "endpoint": self.config.models.embedding.endpoint,
                    "backend": self.config.models.embedding.backend,
                    "deployment_fingerprint": deployment_fingerprint(self.config.models.embedding),
                },
                "records": [_memory_public(memory) for memory in memories],
            }
        )
        vector_cache_hit = vector_key in self._memory_vector_cache
        if vector_cache_hit:
            memory_vectors = self._memory_vector_cache[vector_key]
        else:
            memory_vectors = np.asarray(
                self.embedder.encode([memory.text for memory in memories]), dtype=np.float32
            )
            if (
                memory_vectors.ndim != 2
                or memory_vectors.shape[0] != len(memories)
                or memory_vectors.shape[1] == 0
                or not np.all(np.isfinite(memory_vectors))
            ):
                raise ValueError("embedder returned invalid visible-memory vectors")
            memory_vectors = np.array(memory_vectors, copy=True)
            memory_vectors.setflags(write=False)
            self._memory_vector_cache[vector_key] = memory_vectors
        retriever = DependencyRetriever(
            example.query,
            memories,
            self.embedder,
            memory_vectors=memory_vectors,
            initial_width=settings.initial_width,
            initial_expansion_width=settings.initial_expansion_width,
            proposal_width=settings.proposal_width,
            max_ann_calls=settings.max_ann_calls,
            fixed_pool=fixed_pool,
            query_instruction=self.config.models.embedding.query_instruction,
            proposal_instruction=DEPENDENCY_PROPOSAL_INSTRUCTION,
        )
        return retriever, vector_cache_hit

    def _scorer(
        self,
        task: DependencyTask,
        example: PersonaMemExample,
        records: Mapping[str, Memory],
    ) -> Any:
        from .dependency_scoring import SetReranker

        settings = self.config.dependency
        return SetReranker(
            example.query,
            records,
            self.reranker,
            cache_dir=self.cache_dir,
            cache_identity={
                # Deliberately contains no whole-dataset hash/revision.  A
                # synthetic data identity can encode labels/options, neither
                # of which may affect search caches.  SetReranker already
                # fingerprints the original query, complete visible records,
                # cutoff, template and backend contract.
                "experiment_protocol": "dependency-set-v1",
                "question_id": task.question_id,
            },
            query_identity={
                "persona_id": task.persona_id,
                "question_id": task.question_id,
            },
            visible_history_cutoff={
                "end_index": example.end_index,
                "query_time": example.query_time,
                "query_metadata": _safe_query_cutoff_metadata(example.metadata),
            },
            batch_size=settings.reranker_batch_size,
            max_input_tokens=settings.reranker_max_input_tokens,
            set_budget=None,
            score_space=self.config.models.reranker.score_space,
            score_contract=self.config.models.reranker.score_contract,
        )

    def _answer(
        self,
        plan: ContextPlan,
        example: PersonaMemExample,
        selected_memories: Sequence[Memory],
    ) -> str:
        answer_plan = getattr(self.generator, "answer_plan", None)
        if callable(answer_plan):
            response = answer_plan(plan)
        else:
            answer = getattr(self.generator, "answer", None)
            if not callable(answer):
                if callable(self.generator):
                    answer = self.generator
                else:
                    raise TypeError("generator must expose answer_plan or answer")
            # This compatibility path is intentionally supplied only public
            # query/options and the already frozen exact memory union.
            try:
                response = answer(example.query, selected_memories, example.all_options)
            except TypeError as exc:
                try:
                    response = answer(plan)
                except TypeError:
                    raise exc from None
        if not isinstance(response, str) or not response.strip():
            raise ValueError("generator returned an empty or non-string answer")
        return response.strip()

    def execute(self, task: DependencyTask, example: PersonaMemExample) -> dict[str, Any]:
        self._last_partial_artifacts: dict[str, Any] = {}
        self._active_evidence_selector = None
        try:
            # PR1: inherits an optional durable sink/physical budget from the
            # runner without changing the executor's public service API.
            with request_audit_scope({"task_id": task.task_id, "question_id": task.question_id,
                                      "persona_id": task.persona_id, "method_id": task.method_id}):
                result = self._execute(task, example)
                self._attach_evidence_details(result)
                return result
        except BaseException as exc:
            # Preserve the original exception type for callers and transport
            # classification, while attaching already-realized retrieval,
            # scoring and context artifacts for failure persistence.
            self._attach_evidence_details(self._last_partial_artifacts)
            with suppress(AttributeError, TypeError):
                exc.dependency_partial_artifacts = self._last_partial_artifacts
            raise

    def _attach_evidence_details(self, artifacts: dict[str, Any]) -> None:
        """Publish method traces for both successful and interrupted tasks."""
        selector = self._active_evidence_selector
        if selector is None:
            return
        from .evidence_diagnostics import evidence_bridge_summary

        candidate = artifacts.setdefault("candidate_pool", {})
        value = candidate.get("selection")
        if not isinstance(value, Mapping):
            value = selector.partial_public_dict()
        value = dict(value or {})
        candidate["evidence_selection"] = value
        candidate["selection"] = value
        costs = artifacts.setdefault("costs", {})
        costs["evidence_reasoning"] = dict(value.get("costs", {}))
        costs["evidence_calls"] = costs.get("adapter_invocations", {}).get("evidence_adapter_invocations", 0)
        modules = artifacts.setdefault("module_events", {})
        search = candidate.get("search") or {}
        modules["scheduler"] = list(search.get("scheduler_events", ()))
        modules["target"] = list(search.get("target_events", ()))
        modules["archive"] = [{"event": "measured_set_archived", **dict(row)}
                              for row in search.get("measured_sets", ()) if isinstance(row, Mapping)]
        events = value.get("events", value.get("steps", ()))
        for module in ("planner", "evidence", "feedback"):
            modules[module] = [dict(event) for event in events
                               if isinstance(event, Mapping) and event.get("module") == module]
        artifacts.setdefault("diagnostics", {})["evidence_bridge_summary"] = evidence_bridge_summary(
            candidate, costs, artifacts.get("selected_ids", ())
        )

    def _execute(self, task: DependencyTask, example: PersonaMemExample) -> dict[str, Any]:
        if task.question_id != str(example.question_id) or task.persona_id != str(example.persona_id):
            raise ValueError("task and example identities differ")
        if task.method_id not in self.config.methods:
            raise ValueError(f"task method is not configured: {task.method_id}")

        started = time.perf_counter()
        adapter_before = self.adapter_cost
        memories = self.visible_memories(example)
        records = {str(memory.memory_id): memory for memory in memories}
        visible_artifact = {
            "schema_version": 1,
            "persona_id": task.persona_id,
            "question_id": task.question_id,
            "shared_context_id": str(example.shared_context_id),
            "end_index": example.end_index,
            "query_time": _jsonable(example.query_time),
            "visible_memory_ids": list(records),
            "visible_memories": [_memory_public(memory) for memory in memories],
        }
        self._last_partial_artifacts = {"visible_memories": visible_artifact}
        method = task.method_id
        fixed_pool = method == "activation_fixed_pool"
        with request_audit_scope({"stage": "retrieval"}):
            retriever, memory_vector_cache_hit = self._retriever(
                example, memories, fixed_pool=fixed_pool
            )
        scorer: Any | None = None
        archive: Any | None = None
        selection: Any | None = None
        partial_search: dict[str, Any] | None = None
        partial_selection: dict[str, Any] | None = None
        selection_events: list[dict[str, Any]] = []

        def checkpoint(stage: str, plan_value: ContextPlan | None = None) -> None:
            scorer_cost_value = {} if scorer is None else dict(scorer.cost)
            adapter_value = _counter_delta(self.adapter_cost, adapter_before)
            checkpoint_costs = {
                "ann_calls": retriever.ann_calls,
                "scored_sets": int(scorer_cost_value.get("scored_sets", 0)),
                "reranker_adapter_requests": int(
                    scorer_cost_value.get("reranker_adapter_requests", 0)
                ),
                "cache_hits": int(scorer_cost_value.get("cache_hits", 0)),
                "memory_vector_cache_hit": memory_vector_cache_hit,
                # These are logical adapter invocations/samples, not a claim
                # about internal HTTP batching or transport retry counts.
                "memory_embedding_adapter_invocations": (
                    0 if memory_vector_cache_hit else 1
                ),
                "memory_embedding_samples": 0 if memory_vector_cache_hit else len(memories),
                "adapter_invocations": adapter_value,
                "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                "token_count_is_estimate": True,
                "set_scorer": scorer_cost_value,
            }
            search_value = (
                archive.public_dict() if archive is not None else partial_search
            )
            selection_value = (
                selection.public_dict() if selection is not None else partial_selection
            )
            state_values = (
                [] if search_value is None else list(search_value.get("states", ()))
            )
            activation_values = (
                []
                if search_value is None
                else list(search_value.get("activations", ()))
            )
            stop_values: list[dict[str, Any]] = []
            if search_value is not None and isinstance(
                search_value.get("stop"), Mapping
            ):
                stop_values.append(dict(search_value["stop"]))
            if selection_value is not None and isinstance(
                selection_value.get("stop"), Mapping
            ):
                stop_values.append(dict(selection_value["stop"]))
            selection_values = list(selection_events)
            if not selection_values and selection_value is not None:
                raw_steps = selection_value.get("steps")
                if isinstance(raw_steps, Sequence) and not isinstance(
                    raw_steps, (str, bytes)
                ):
                    selection_values.extend(
                        dict(item) for item in raw_steps if isinstance(item, Mapping)
                    )
                else:
                    for round_value in selection_value.get("rounds", ()):
                        if not isinstance(round_value, Mapping):
                            continue
                        comparisons = round_value.get("comparisons", ())
                        if isinstance(comparisons, Sequence) and not isinstance(
                            comparisons, (str, bytes)
                        ):
                            selection_values.extend(
                                dict(item)
                                for item in comparisons
                                if isinstance(item, Mapping)
                            )
                if isinstance(selection_value.get("stop"), Mapping):
                    selection_values.append(dict(selection_value["stop"]))
            context_values = [] if plan_value is None else [{
                "event": "final_context",
                "context_plan": plan_value.public_dict(),
                "generation_status": "pending",
            }]
            selected_checkpoint_ids = (
                tuple(plan_value.selected_ids)
                if plan_value is not None
                else tuple(
                    str(identifier)
                    for identifier in (
                        ()
                        if selection_value is None
                        else selection_value.get("selected_ids", ())
                    )
                )
            )
            self._last_partial_artifacts = {
                "visible_memories": visible_artifact,
                "candidate_pool": {
                    "schema_version": 1,
                    "task_id": task.task_id,
                    "method_id": method,
                    "checkpoint_stage": stage,
                    "retrieval": retriever.public_dict(),
                    "search": search_value,
                    "selection": selection_value,
                    "baseline_selection": (
                        selection_events if search_value is None else None
                    ),
                },
                "module_events": {
                    "scoring": [] if scorer is None else list(scorer.events),
                    "activation": activation_values,
                    "proposal": [
                        batch.public_dict() for batch in retriever.proposal_batches
                    ],
                    "state": state_values,
                    "stop": stop_values,
                    "selection": selection_values,
                    "context": context_values,
                    "cost": [
                        {
                            "event": "task_cost_checkpoint",
                            "checkpoint_stage": stage,
                            **checkpoint_costs,
                        }
                    ],
                },
                "context_plan": None if plan_value is None else plan_value.public_dict(),
                "selected_ids": selected_checkpoint_ids,
                "context_hash": None if plan_value is None else plan_value.context_hash,
                "costs": checkpoint_costs,
                "diagnostics": {
                    "method_id": method,
                    "fixed_pool": fixed_pool,
                    "score_space": self.config.models.reranker.score_space,
                    "score_contract": self.config.models.reranker.score_contract,
                    "checkpoint_stage": stage,
                    "search_stop_reason": (
                        None
                        if search_value is None
                        else search_value.get("stop_reason")
                    ),
                    "selection_stop_reason": (
                        None
                        if selection_value is None
                        else (
                            selection_value.get("stop", {}).get("reason")
                            if isinstance(selection_value.get("stop"), Mapping)
                            else None
                        )
                    ),
                },
            }

        checkpoint("retriever_initialized")

        if method == "dense":
            try:
                with request_audit_scope({"stage": "retrieval"}):
                    dense = retriever.retrieve_dense()
            finally:
                checkpoint("dense_retrieval")
            selected_ids, selection_events = _baseline_context(
                self.config, example, records, dense.ids
            )
            checkpoint("dense_selection")
        elif method == "dense_rerank":
            try:
                with request_audit_scope({"stage": "retrieval"}):
                    dense = retriever.retrieve_dense()
            finally:
                checkpoint("dense_retrieval")
            scorer = self._scorer(task, example, records)
            scorer.set_budget(self.config.dependency.max_scored_sets)
            singleton_sets = [(identifier,) for identifier in dense.ids]
            try:
                singleton_scores = scorer.score_sets(
                    singleton_sets, reason="dense_rerank"
                )
            finally:
                checkpoint("dense_rerank_scoring")
            ranked = [
                identifier
                for _negative, identifier in sorted(
                    (-float(score), str(ids[0]))
                    for ids, score in zip(singleton_sets, singleton_scores)
                )
            ]
            selected_ids, selection_events = _baseline_context(
                self.config, example, records, ranked
            )
            checkpoint("dense_rerank_selection")
        elif method == "evidence_bridge":
            from .diagnostic_observability import observe
            from .evidence_search import EvidenceBridgeSearcher
            from .evidence_selection import EvidenceSelector

            settings = self.config.evidence_bridge

            def evidence_feasible(ids: tuple[str, ...]) -> dict[str, Any]:
                reader_plan = _generation_plan_for_ids(self.config, example, records, ids, strict=False)
                return {
                    "feasible": reader_plan.within_budget,
                    "reason": "within_budget" if reader_plan.within_budget else "generator_input_capacity",
                    "token_count": reader_plan.token_count,
                    "budget": self.config.models.generator.context_token_budget,
                    "context_hash": reader_plan.context_hash,
                }

            def record_evidence_event(event):
                observe(event["module"], event)
                # A durable full method snapshot retains raw model responses
                # and source quotations even if the worker is later killed.
                # The compact live event stream intentionally contains IDs
                # and counters instead of arbitrary text.
                if callable(self.evidence_snapshot_sink) and self._active_evidence_selector is not None:
                    self.evidence_snapshot_sink(event, self._active_evidence_selector.partial_public_dict())

            selector = EvidenceSelector(
                self.generator, settings.selection, generation_feasible=evidence_feasible,
                event_sink=record_evidence_event,
            )
            self._active_evidence_selector = selector
            try:
                with request_audit_scope({"stage": "evidence_planning"}):
                    requirements = selector.plan(example.query)
                retriever.set_information_needs(requirements)
                # Initial expansion shares the search part of the ANN cap;
                # it must not consume the later evidence-gap reservation.
                search_ann_cap = max(0, self.config.dependency.max_ann_calls - settings.gap_ann_calls)
                total_ann_cap = retriever.max_ann_calls
                try:
                    retriever.max_ann_calls = search_ann_cap
                    with request_audit_scope({"stage": "retrieval"}):
                        initial_pool = retriever.build_initial_pool(expand=True)
                finally:
                    retriever.max_ann_calls = total_ann_cap
                    checkpoint("evidence_initial_pool")
                scorer = self._scorer(task, example, records)
                searcher = EvidenceBridgeSearcher(
                    scorer, retriever, settings=settings.search,
                    max_scored_sets=self.config.dependency.max_scored_sets,
                    pair_rescue_width=self.config.dependency.pair_rescue_width,
                    max_ann_calls=search_ann_cap,
                )
                try:
                    with request_audit_scope({"stage": "evidence_search"}):
                        archive = searcher.run(initial_pool)
                except BaseException as exc:
                    partial_search = searcher.partial_public_dict(
                        stop_reason="interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "execution_error",
                        detail=type(exc).__name__,
                    )
                    raise
                finally:
                    checkpoint("evidence_search")

                candidate_ids = list(initial_pool.candidate_ids)
                for batch in retriever.proposal_batches:
                    for identifier in batch.ids:
                        if identifier not in candidate_ids:
                            candidate_ids.append(identifier)
                seen = set(candidate_ids)
                gap_calls = 0

                def expand_missing(missing_requirements, selected_ids):
                    nonlocal gap_calls
                    if gap_calls >= settings.gap_ann_calls or retriever.remaining_ann_calls == 0:
                        return ()
                    gap_calls += 1
                    with request_audit_scope({"stage": "evidence_gap"}):
                        batch = retriever.retrieve_missing(
                            missing_requirements, exclude=seen, source_memory_ids=selected_ids,
                            width=settings.gap_proposal_width,
                        )
                    seen.update(batch.ids)
                    return batch.ids

                with request_audit_scope({"stage": "evidence_selection"}):
                    selection = selector.select(
                        example.query, records, candidate_ids,
                        requirements=requirements, expand=expand_missing,
                    )
                selected_ids = tuple(selection.selected_ids)
                selection_events = [step.public_dict() for step in selection.steps]
                selection_events.append(selection.stop.public_dict())
            except BaseException as exc:
                partial_selection = selector.partial_public_dict(
                    stop_reason="interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "execution_error",
                    detail=type(exc).__name__,
                )
                checkpoint("evidence_method_failed")
                terminal_event = {
                    "module": "evidence", "event": "evidence_method_failed",
                    "stop_reason": partial_selection["stop_reason"], "error_type": type(exc).__name__,
                }
                observe("evidence", terminal_event)
                if callable(self.evidence_snapshot_sink):
                    self.evidence_snapshot_sink(terminal_event, partial_selection)
                raise
            checkpoint("evidence_selection")
        else:
            from .dependency_search import DependencySearcher, DynamicBundleSelector

            try:
                with request_audit_scope({"stage": "retrieval"}):
                    initial_pool = retriever.build_initial_pool(expand=True)
            finally:
                checkpoint("initial_pool_retrieval")
            scorer = self._scorer(task, example, records)
            signal = "context_marginal" if method == "context_marginal" else "activation"
            pair_width = (
                0 if method == "activation_no_pairs"
                else self.config.dependency.pair_rescue_width
            )
            searcher = DependencySearcher(
                scorer,
                retriever,
                signal=signal,
                pair_rescue_width=pair_width,
                fixed_pool=fixed_pool,
                max_scored_sets=self.config.dependency.max_scored_sets,
                root_tie_break=self.config.dependency.root_tie_break,
                root_tie_seed=self.config.dependency.root_tie_seed,
            )
            try:
                with request_audit_scope({"stage": "dependency_search"}):
                    archive = searcher.run(initial_pool)
            except Exception as exc:
                snapshot = searcher.partial_public_dict(
                    stop_reason="execution_error",
                    detail=f"{type(exc).__name__}: {exc}",
                )
                partial_search = None if snapshot is None else dict(snapshot)
                checkpoint("dependency_search")
                raise
            checkpoint("dependency_search")

            def generation_feasible(ids: tuple[str, ...]) -> dict[str, Any]:
                plan = _generation_plan_for_ids(
                    self.config, example, records, ids, strict=False
                )
                return {
                    "feasible": plan.within_budget,
                    "reason": (
                        "within_budget" if plan.within_budget
                        else "complete_generator_request_exceeds_budget"
                    ),
                }

            selector = DynamicBundleSelector(
                scorer,
                max_selection_sets=self.config.dependency.max_selection_sets,
                generation_feasible=generation_feasible,
            )
            bundles: Any = archive
            if method == "activation_singleton_selection":
                bundles = tuple(bundle for bundle in archive.bundles if len(bundle.memory_ids) == 1)
            try:
                selection = selector.select(bundles)
            except Exception as exc:
                snapshot = selector.partial_public_dict(
                    stop_reason="execution_error",
                    detail=f"{type(exc).__name__}: {exc}",
                )
                partial_selection = None if snapshot is None else dict(snapshot)
                checkpoint("bundle_selection")
                raise
            checkpoint("bundle_selection")
            selected_ids = tuple(selection.selected_ids)
            selection_events = [step.public_dict() for step in selection.steps]
            selection_events.append(selection.stop.public_dict())
            checkpoint("selection_recorded")

        try:
            plan = _generation_plan_for_ids(
                self.config, example, records, selected_ids, strict=True
            )
        except Exception:
            checkpoint("context_plan_failed")
            raise
        checkpoint("context_plan_frozen", plan)
        generator_fingerprint = deployment_fingerprint(self.config.models.generator)
        generation_identity = {
            "request_hash": request_hash(plan.request_dict()),
            "execution_hash": execution_hash(plan.request_dict(), generator_fingerprint),
            "deployment_fingerprint": generator_fingerprint,
        }
        selected_memories = [records[identifier] for identifier in plan.selected_ids]
        scoring_events = [] if scorer is None else list(scorer.events)
        activation_events = (
            [] if archive is None
            else [record.public_dict() for record in archive.activations]
        )
        state_events = (
            [] if archive is None
            else [record.public_dict() for record in archive.state_records]
        )
        stop_events: list[dict[str, Any]] = []
        if archive is not None:
            stop_events.append({"event": "search_stop", "reason": archive.stop_reason})
        if selection is not None:
            stop_events.append(selection.stop.public_dict())
        candidate_artifact = {
            "schema_version": 1,
            "task_id": task.task_id,
            "method_id": method,
            "retrieval": retriever.public_dict(),
            "search": None if archive is None else archive.public_dict(),
            "selection": None if selection is None else selection.public_dict(),
            "baseline_selection": selection_events if archive is None else None,
        }
        context_event = {
            "event": "final_context",
            "context_plan": plan.public_dict(),
            "generation_status": "pending",
            **generation_identity,
        }
        scorer_cost = {} if scorer is None else dict(scorer.cost)
        pre_generation_adapter = _counter_delta(self.adapter_cost, adapter_before)
        partial_costs = {
            "ann_calls": retriever.ann_calls,
            "scored_sets": int(scorer_cost.get("scored_sets", 0)),
            "reranker_adapter_requests": int(
                scorer_cost.get("reranker_adapter_requests", 0)
            ),
            "cache_hits": int(scorer_cost.get("cache_hits", 0)),
            "memory_vector_cache_hit": memory_vector_cache_hit,
            "memory_embedding_adapter_invocations": 0 if memory_vector_cache_hit else 1,
            "memory_embedding_samples": 0 if memory_vector_cache_hit else len(memories),
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            "token_count_is_estimate": True,
            "generator_input_tokens_estimate": plan.token_count,
            "set_scorer": scorer_cost,
            "adapter_invocations": pre_generation_adapter,
        }
        partial_module_events = {
            "scoring": scoring_events,
            "activation": activation_events,
            "proposal": [batch.public_dict() for batch in retriever.proposal_batches],
            "state": state_events,
            "stop": stop_events,
            "selection": selection_events,
            "context": [context_event],
            "cost": [{"event": "task_cost_partial", **partial_costs}],
        }
        partial_diagnostics = {
            **generation_identity,
            "method_id": method,
            "fixed_pool": fixed_pool,
            "score_space": self.config.models.reranker.score_space,
            "score_contract": self.config.models.reranker.score_contract,
            "search_stop_reason": None if archive is None else archive.stop_reason,
            "selection_stop_reason": None if selection is None else selection.stop_reason,
        }
        self._last_partial_artifacts = {
            "visible_memories": visible_artifact,
            "candidate_pool": candidate_artifact,
            "module_events": partial_module_events,
            "context_plan": plan.public_dict(),
            "selected_ids": tuple(plan.selected_ids),
            "context_hash": plan.context_hash,
            "costs": partial_costs,
            "diagnostics": partial_diagnostics,
        }
        # This is the only generator call and it consumes the exact frozen
        # ContextPlan used by feasibility and persisted below.
        with request_audit_scope({"stage": "generation"}):
            prediction = self._answer(plan, example, selected_memories)

        # Evaluation begins only after all model requests have completed.
        accuracy = answer_accuracy(prediction, example.correct_answer)
        parse_failure = answer_parse_failed(prediction)
        scorer_cost = {} if scorer is None else dict(scorer.cost)
        adapter_invocations = _counter_delta(self.adapter_cost, adapter_before)
        costs = {
            "ann_calls": retriever.ann_calls,
            "scored_sets": int(scorer_cost.get("scored_sets", 0)),
            "reranker_adapter_requests": int(
                scorer_cost.get("reranker_adapter_requests", 0)
            ),
            "cache_hits": int(scorer_cost.get("cache_hits", 0)),
            "generator_calls": int(
                adapter_invocations.get("generator_adapter_invocations", 0)
            ),
            "memory_vector_cache_hit": memory_vector_cache_hit,
            "memory_embedding_adapter_invocations": 0 if memory_vector_cache_hit else 1,
            "memory_embedding_samples": 0 if memory_vector_cache_hit else len(memories),
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            "token_count_is_estimate": True,
            "generator_input_tokens_estimate": plan.token_count,
            "set_scorer": scorer_cost,
            "adapter_invocations": adapter_invocations,
        }
        scoring_events = [] if scorer is None else list(scorer.events)
        activation_events = (
            [] if archive is None
            else [record.public_dict() for record in archive.activations]
        )
        state_events = (
            [] if archive is None
            else [record.public_dict() for record in archive.state_records]
        )
        stop_events: list[dict[str, Any]] = []
        if archive is not None:
            stop_events.append(
                {"event": "search_stop", "reason": archive.stop_reason}
            )
        if selection is not None:
            stop_events.append(selection.stop.public_dict())
        candidate_artifact = {
            "schema_version": 1,
            "task_id": task.task_id,
            "method_id": method,
            "retrieval": retriever.public_dict(),
            "search": None if archive is None else archive.public_dict(),
            "selection": None if selection is None else selection.public_dict(),
            "baseline_selection": selection_events if archive is None else None,
        }
        context_event = {
            "event": "final_context",
            "context_plan": plan.public_dict(),
            "generation_status": "completed",
            **generation_identity,
        }
        return {
            "status": "success",
            "prediction": prediction,
            "predicted_label": extract_option_label(prediction),
            "correct": bool(accuracy),
            "parse_failed": bool(parse_failure),
            "selected_ids": tuple(plan.selected_ids),
            "context_hash": plan.context_hash,
            "context_plan": plan.public_dict(),
            "costs": costs,
            "visible_memories": visible_artifact,
            "candidate_pool": candidate_artifact,
            "module_events": {
                "scoring": scoring_events,
                "activation": activation_events,
                "proposal": [batch.public_dict() for batch in retriever.proposal_batches],
                "state": state_events,
                "stop": stop_events,
                "selection": selection_events,
                "context": [context_event],
                "cost": [{"event": "task_cost", **costs}],
            },
            "diagnostics": {
                **generation_identity,
                "method_id": method,
                "fixed_pool": fixed_pool,
                "score_space": self.config.models.reranker.score_space,
                "score_contract": self.config.models.reranker.score_contract,
                "search_stop_reason": None if archive is None else archive.stop_reason,
                "selection_stop_reason": None if selection is None else selection.stop_reason,
            },
        }


def _is_infrastructure_failure(exc: BaseException) -> bool:
    if isinstance(exc, (ConnectionError, TimeoutError, urllib.error.URLError)):
        return True
    # OSError is broad, but at this stage data/files have already passed
    # preflight; an OSError raised by a service adapter is transport-level.
    if isinstance(exc, OSError):
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in INFRASTRUCTURE_MARKERS)


def _is_retryable_infrastructure_failure(exc: BaseException) -> bool:
    """Respect a transport adapter's explicit retry decision when present."""

    if getattr(exc, "retry_budget_exhausted", False):
        return False
    declared = getattr(exc, "retryable", None)
    if isinstance(declared, bool):
        return declared
    return _is_infrastructure_failure(exc)


def _infrastructure_retry_delay(
    config: DependencyRunConfig,
    *,
    failed_attempts: int,
) -> float:
    """Return the bounded delay after a consecutive infrastructure failure."""

    if failed_attempts <= 0:
        raise ValueError("failed_attempts must be positive")
    execution = config.execution
    delay = float(execution.infrastructure_retry_initial_seconds)
    for _ in range(failed_attempts - 1):
        delay = min(
            float(execution.infrastructure_retry_max_seconds),
            delay * float(execution.infrastructure_retry_multiplier),
        )
        if delay >= float(execution.infrastructure_retry_max_seconds):
            break
    return delay


def _wait_for_infrastructure_retry(
    root: Path,
    *,
    event_context: Mapping[str, Any],
    delay_seconds: float,
) -> None:
    """Record a retry wait on every observable log and remain signal-safe.

    ``chain_worker`` converts SIGTERM into ``KeyboardInterrupt`` in the main
    thread.  Python's ``time.sleep`` is interruptible by that exception, so an
    operator stop during this wait is finalized by the runner's existing
    interruption guard instead of being delayed until the full backoff ends.
    """

    scheduled_at = time.time()
    _emit_runtime_event(
        root,
        {
            "schema_version": 1,
            "event": "infrastructure_retry_scheduled",
            "at_epoch": scheduled_at,
            "retry_at_epoch": scheduled_at + float(delay_seconds),
            "delay_seconds": float(delay_seconds),
            "circuit_state": "closed",
            **dict(event_context),
        },
    )
    time.sleep(float(delay_seconds))
    _emit_runtime_event(
        root,
        {
            "schema_version": 1,
            "event": "infrastructure_retry_wait_completed",
            "at_epoch": time.time(),
            "delay_seconds": float(delay_seconds),
            "circuit_state": "half_open",
            **dict(event_context),
        },
    )


class _Heartbeat:
    def __init__(self, root: Path, interval: float):
        self.root = root
        self.interval = float(interval)
        self.started_at = time.time()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {
            "active": True,
            "active_task": None,
            "completed_tasks": 0,
        }
        self._thread: threading.Thread | None = None

    def _write(self) -> None:
        with self._lock:
            state = dict(self._state)
        now = time.time()
        _atomic_json(
            self.root / "heartbeat.json",
            {
                "schema_version": 1,
                **state,
                "updated_at_epoch": now,
                "elapsed_seconds": max(0.0, now - self.started_at),
                "optimizer_steps": 0,
                "weights_updated": False,
            },
        )

    def start(self) -> None:
        key = self.root.resolve()
        with _ACTIVE_HEARTBEATS_LOCK:
            existing = _ACTIVE_HEARTBEATS.get(key)
            if existing is not None and existing is not self:
                raise RuntimeError(f"heartbeat already active for run directory: {key}")
            _ACTIVE_HEARTBEATS[key] = self
        try:
            self._write()
        except BaseException:
            with _ACTIVE_HEARTBEATS_LOCK:
                if _ACTIVE_HEARTBEATS.get(key) is self:
                    _ACTIVE_HEARTBEATS.pop(key, None)
            raise

        def loop() -> None:
            while not self._stop.wait(self.interval):
                self._write()

        self._thread = threading.Thread(
            target=loop, name="dependency-heartbeat", daemon=True
        )
        self._thread.start()

    def update(self, *, task: DependencyTask | None, completed_tasks: int) -> None:
        with self._lock:
            self._state["active_task"] = None if task is None else {
                "task_id": task.task_id,
                "persona_id": task.persona_id,
                "question_id": task.question_id,
                "method_id": task.method_id,
            }
            self._state["completed_tasks"] = int(completed_tasks)
        self._write()

    def close(self, *, completed_tasks: int) -> None:
        try:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=max(1.0, min(self.interval * 2, 5.0)))
            with self._lock:
                self._state.update(
                    {
                        "active": False,
                        "active_task": None,
                        "completed_tasks": int(completed_tasks),
                    }
                )
            self._write()
        finally:
            key = self.root.resolve()
            with _ACTIVE_HEARTBEATS_LOCK:
                if _ACTIVE_HEARTBEATS.get(key) is self:
                    _ACTIVE_HEARTBEATS.pop(key, None)


def _close_active_heartbeat(root: Path, *, completed_tasks: int) -> None:
    with _ACTIVE_HEARTBEATS_LOCK:
        heartbeat = _ACTIVE_HEARTBEATS.get(root.resolve())
    if heartbeat is not None:
        with suppress(Exception):
            heartbeat.close(completed_tasks=completed_tasks)


@dataclass(frozen=True)
class _PreparedRun:
    config: DependencyRunConfig
    root: Path
    dataset: DependencyDataset
    tasks: tuple[DependencyTask, ...]
    examples_by_id: Mapping[str, PersonaMemExample]
    identity: Mapping[str, str]
    protocol_report: Mapping[str, Any]
    manifest: Mapping[str, Any]


def _resolve_config(config: str | Path | DependencyRunConfig) -> DependencyRunConfig:
    if isinstance(config, DependencyRunConfig):
        return config
    return load_dependency_config(config)


def _safe_artifact_name(value: str) -> str:
    raw = str(value)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    readable = "".join(character if character.isalnum() or character in "-_" else "_" for character in raw)
    readable = readable[:80] or "item"
    return f"{readable}-{digest}"


def _task_plan_hash(tasks: Sequence[DependencyTask]) -> str:
    return _sha256_json([task.public_dict() for task in tasks])


def _write_metrics_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = sorted({str(key) for row in rows for key in row})
    if not fieldnames:
        _atomic_bytes(path, b"")
        return
    # csv.writer needs a text stream; build the complete value in memory so
    # replacement remains atomic.
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key) for key in fieldnames})
    _atomic_bytes(path, buffer.getvalue().encode("utf-8"))


def _write_current_effectiveness(
    root: Path,
    tasks: Sequence[DependencyTask],
    outcomes: Mapping[str, DependencyOutcome],
) -> None:
    """Materialize one retry-safe effectiveness row per current outcome."""

    rows: list[dict[str, Any]] = []
    for task in tasks:
        outcome = outcomes.get(task.task_id)
        if outcome is None:
            continue
        effectiveness = outcome.diagnostics.get("module_effectiveness")
        if not isinstance(effectiveness, Mapping):
            raise ValueError(
                f"outcome {task.task_id} omitted module_effectiveness diagnostics"
            )
        if (
            effectiveness.get("task_id") != task.task_id
            or effectiveness.get("attempt") != outcome.attempt
            or effectiveness.get("status") != outcome.status
        ):
            raise ValueError(
                f"outcome {task.task_id} has mismatched module_effectiveness identity"
            )
        rows.append(
            {
                **dict(effectiveness),
                "authoritative_current": True,
            }
        )
    _atomic_jsonl(root / "modules" / "effectiveness.current.jsonl", rows)


def _write_run_views(
    root: Path,
    tasks: Sequence[DependencyTask],
    outcomes: Mapping[str, DependencyOutcome],
    protocol_report: Mapping[str, Any],
    *,
    completion_status: str,
    inference_complete: bool,
) -> dict[str, Any]:
    summary = summarize_dependency_outcomes(tasks, outcomes)
    _write_current_effectiveness(root, tasks, outcomes)
    methods = tuple(dict.fromkeys(task.method_id for task in tasks))
    roles = ["all_data"]
    if protocol_report.get("status") == "defined":
        roles.extend(("seen", "confirmation"))
    metric_rows = [
        _metric_row(tasks, outcomes, method_id=method, role=role)
        for role in roles
        for method in methods
    ]
    _write_metrics_csv(root / "metrics.csv", metric_rows)

    def report_for(role: str) -> dict[str, Any]:
        rows = [row for row in metric_rows if row["role"] == role]
        aggregate = _metric_row(tasks, outcomes, method_id="all", role=role)
        return {
            "schema_version": 1,
            "status": aggregate["status"],
            "role": role,
            "aggregate": aggregate,
            "methods": rows,
            "inference_complete": inference_complete,
            "optimizer_steps": 0,
            "weights_updated": False,
        }

    _atomic_json(root / "reports" / "all_data_report.json", report_for("all_data"))
    if protocol_report.get("status") == "defined":
        _atomic_json(root / "reports" / "seen_report.json", report_for("seen"))
        _atomic_json(
            root / "reports" / "confirmation_report.json", report_for("confirmation")
        )
    else:
        undefined = {
            "schema_version": 1,
            "status": "not_defined",
            "inference_complete": inference_complete,
            "reason": "no authoritative protocol manifest supplied",
        }
        _atomic_json(root / "reports" / "seen_report.json", {**undefined, "role": "seen"})
        _atomic_json(
            root / "reports" / "confirmation_report.json",
            {**undefined, "role": "confirmation"},
        )

    progress = {
        "schema_version": 1,
        "status": completion_status,
        "inference_complete": inference_complete,
        "expected_tasks": summary["expected_tasks"],
        "completed_tasks": summary["completed_tasks"],
        "successful_tasks": summary["successful_tasks"],
        "failed_tasks": summary["failed_tasks"],
        "pending_tasks": summary["pending_tasks"],
        "coverage": summary["coverage"],
        "optimizer_steps": 0,
        "weights_updated": False,
        "updated_at_epoch": time.time(),
    }
    _atomic_json(root / "progress.json", progress)
    _atomic_json(root / "summary.json", summary)
    _atomic_json(
        root / "completion.json",
        {
            **progress,
            "status": completion_status,
            "complete": bool(inference_complete),
        },
    )
    return {"summary": summary, "progress": progress, "metrics": metric_rows}


def _prepare_run(
    config: str | Path | DependencyRunConfig,
    output_dir: str | Path,
    *,
    protocol_manifest: str | Path | Mapping[str, Any] | None,
    examples: Sequence[PersonaMemExample] | None,
    require_full_32k: bool | None,
    resume: bool,
) -> _PreparedRun:
    resolved = _resolve_config(config)
    dataset = load_dependency_dataset(
        resolved, examples=examples, require_full_32k=require_full_32k
    )
    role_by_question, protocol_report, protocol_hash = _load_protocol_manifest(
        protocol_manifest, dataset
    )
    source_hash = _source_code_hash()
    config_hash = resolved.config_hash()
    tasks = tuple(
        build_dependency_plan(
            dataset,
            config_hash=config_hash,
            source_code_hash=source_hash,
            protocol_hash=protocol_hash,
            methods=resolved.methods,
            role_by_question=role_by_question,
        )
    )
    plan_hash = _task_plan_hash(tasks)
    identity = {
        "config_hash": config_hash,
        "data_hash": dataset.data_hash,
        "source_code_hash": source_hash,
        "protocol_hash": protocol_hash,
        "task_plan_hash": plan_hash,
        "generation_prompt_hash": generation_prompt_hash(),
    }
    identity["run_identity"] = _sha256_json(identity)

    root = Path(output_dir).expanduser().resolve()
    manifest_path = root / "run_manifest.json"
    recovering_partial_preparation = False
    if resume and not manifest_path.is_file():
        recovering_partial_preparation = _recoverable_partial_preparation(
            root, identity
        )
        if not recovering_partial_preparation:
            raise FileNotFoundError(f"cannot resume without run_manifest.json: {root}")

    existing_manifest: dict[str, Any] | None = None
    if manifest_path.is_file():
        loaded = _read_json(manifest_path)
        if not isinstance(loaded, Mapping):
            raise RunIdentityMismatch("existing run manifest is not an object")
        existing_manifest = dict(loaded)
        recorded_identity = existing_manifest.get("identity")
        if not isinstance(recorded_identity, Mapping):
            raise RunIdentityMismatch("existing run manifest has no identity block")
        mismatches = {
            key: {"recorded": recorded_identity.get(key), "current": value}
            for key, value in identity.items()
            if recorded_identity.get(key) != value
        }
        if mismatches:
            names = ", ".join(sorted(mismatches))
            raise RunIdentityMismatch(f"resume identity mismatch: {names}")
        plan_path = root / "planned_tasks.jsonl"
        if not plan_path.is_file():
            raise FrozenPlanError("existing run has no frozen planned_tasks.jsonl")
        loaded_tasks = tuple(DependencyTask.from_dict(row) for row in _read_jsonl(plan_path))
        if [task.public_dict() for task in loaded_tasks] != [task.public_dict() for task in tasks]:
            raise FrozenPlanError("existing frozen task plan differs from current complete plan")
        status = str(existing_manifest.get("status", ""))
        if not resume and status not in {"prepared", "preflight_complete"}:
            raise FileExistsError(
                f"run directory already contains an execution ({status}); use resume=True"
            )
    else:
        if resume and not recovering_partial_preparation:
            raise FileNotFoundError(f"cannot resume missing run directory: {root}")
        root.mkdir(parents=True, exist_ok=True)
        preparation_path = root / "preparation_identity.json"
        _atomic_json(
            preparation_path,
            {
                "schema_version": 1,
                "status": "preparing",
                "identity": identity,
            },
        )
        for directory in ("outcomes", "reports", "visible_memories", "candidate_pool", "modules"):
            (root / directory).mkdir(parents=True, exist_ok=True)
        _atomic_jsonl(root / "planned_tasks.jsonl", (task.public_dict() for task in tasks))
        for name in (
            "events.jsonl",
            "predictions.jsonl",
            "failures.jsonl",
            "service_probes.jsonl",
        ):
            _atomic_bytes(root / name, b"")
        for name in MODULE_LOGS:
            _atomic_bytes(root / "modules" / f"{name}.jsonl", b"")
        _atomic_json(
            root / "resolved_config.json",
            {
                "schema_version": 1,
                "config": _public_config(resolved),
                "config_hash": config_hash,
                "identity": identity,
            },
        )
        created = time.time()
        existing_manifest = {
            "schema_version": RUN_SCHEMA_VERSION,
            "status": "prepared",
            "mode": "train_free",
            "created_at_epoch": created,
            "updated_at_epoch": created,
            "identity": identity,
            "dataset": dataset.public_dict(),
            "protocol": dict(protocol_report),
            "methods": list(resolved.methods),
            "expected_tasks": len(tasks),
            "optimizer_steps": 0,
            "weights_updated": False,
            "execution_attempts": 0,
        }
        _atomic_json(manifest_path, existing_manifest)
        _atomic_json(
            preparation_path,
            {
                "schema_version": 1,
                "status": "prepared",
                "identity": identity,
            },
        )
        _append_text(
            root / "train.log",
            "mode=train_free optimizer_steps=0 weights_updated=false status=prepared",
        )
        _append_jsonl(
            root / "events.jsonl",
            {
                "event": "task_plan_frozen",
                "at_epoch": created,
                "expected_tasks": len(tasks),
                "task_plan_hash": plan_hash,
                "model_calls": 0,
            },
        )

    return _PreparedRun(
        config=resolved,
        root=root,
        dataset=dataset,
        tasks=tasks,
        examples_by_id={str(example.question_id): example for example in dataset.examples},
        identity=identity,
        protocol_report=protocol_report,
        manifest=existing_manifest or {},
    )


def preflight_dependency_run(
    config: str | Path | DependencyRunConfig,
    output_dir: str | Path,
    *,
    protocol_manifest: str | Path | Mapping[str, Any] | None = None,
    examples: Sequence[PersonaMemExample] | None = None,
    require_full_32k: bool | None = None,
) -> dict[str, Any]:
    """Freeze and verify the complete plan without constructing services."""

    prepared = _prepare_run(
        config,
        output_dir,
        protocol_manifest=protocol_manifest,
        examples=examples,
        require_full_32k=require_full_32k,
        resume=False,
    )
    outcomes = load_authoritative_outcomes(prepared.root, prepared.tasks)
    # A preflight is a data/identity result, never a claim that pending model
    # inference has completed.
    views = _write_run_views(
        prepared.root,
        prepared.tasks,
        outcomes,
        prepared.protocol_report,
        completion_status="preflight_complete",
        inference_complete=False,
    )
    manifest = {
        **dict(prepared.manifest),
        "status": "preflight_complete",
        "updated_at_epoch": time.time(),
        "preflight": {
            "status": "passed",
            "questions": len(prepared.dataset.examples),
            "expected_tasks": len(prepared.tasks),
            "model_calls": 0,
        },
    }
    _atomic_json(prepared.root / "run_manifest.json", manifest)
    _append_text(
        prepared.root / "train.log",
        (
            "mode=train_free status=preflight_complete "
            f"questions={len(prepared.dataset.examples)} tasks={len(prepared.tasks)} "
            "model_calls=0 optimizer_steps=0 weights_updated=false"
        ),
    )
    _append_jsonl(
        prepared.root / "events.jsonl",
        {
            "event": "preflight_complete",
            "at_epoch": time.time(),
            "questions": len(prepared.dataset.examples),
            "expected_tasks": len(prepared.tasks),
            "model_calls": 0,
            "inference_complete": False,
        },
    )
    return {
        "status": "preflight_complete",
        "run_dir": str(prepared.root),
        "dataset": prepared.dataset.public_dict(),
        "expected_tasks": len(prepared.tasks),
        "methods": list(prepared.config.methods),
        "identity": dict(prepared.identity),
        "protocol": dict(prepared.protocol_report),
        "model_calls": 0,
        "inference_complete": False,
        "summary": views["summary"],
    }


def _pointwise_service_probe(
    executor: DependencyTaskExecutor,
    task: DependencyTask,
    example: PersonaMemExample,
) -> dict[str, Any]:
    """Run one numerical batch-composition probe outside all task budgets."""

    from .dependency_scoring import SetReranker, probe_pointwise_consistency

    memories = executor.visible_memories(example)
    records = {str(memory.memory_id): memory for memory in memories}
    # Constructing a set scorer freezes exactly the serialization contract but
    # does not score anything.  The protocol helper talks directly to the
    # reranker, so these calls cannot consume a task's logical search quota.
    serializer = SetReranker(
        example.query,
        records,
        executor.reranker,
        cache_dir=None,
        cache_identity={"experiment_protocol": "dependency-probe-v1"},
        query_identity={"persona_id": task.persona_id, "question_id": task.question_id},
        visible_history_cutoff={
            "end_index": example.end_index,
            "query_time": example.query_time,
            "query_metadata": _safe_query_cutoff_metadata(example.metadata),
        },
        batch_size=executor.config.dependency.reranker_batch_size,
        max_input_tokens=executor.config.dependency.reranker_max_input_tokens,
        set_budget=None,
        score_space=executor.config.models.reranker.score_space,
        score_contract=executor.config.models.reranker.score_contract,
    )
    ids = tuple(records)
    first = serializer.serialize_set(ids[:1])
    # Tiny deterministic tests may expose only one memory.  A distinct real-
    # memory serialization is preferred; with one record, use the same
    # serialization plus a harmless protocol marker.  The probe tests numeric
    # batch independence, not semantic correctness.
    second = (
        serializer.serialize_set(ids[1:2])
        if len(ids) >= 2
        else first + "\n[Pointwise protocol duplicate-composition check]"
    )
    report = probe_pointwise_consistency(
        executor.reranker,
        example.query,
        [first, second],
        score_space=executor.config.models.reranker.score_space,
        rtol=1e-5,
        atol=1e-6,
        raise_on_mismatch=False,
    )
    result = {
        "schema_version": 1,
        "status": "passed" if report.consistent else "failed",
        "query_id": task.question_id,
        "query_is_original": True,
        "search_budget_consumed": 0,
        **report.public_dict(),
    }
    if not report.consistent:
        raise PointwiseProtocolMismatch(result)
    return result


def _write_visible_artifact(root: Path, value: Mapping[str, Any]) -> None:
    question_id = str(value.get("question_id", ""))
    if not question_id:
        raise ValueError("visible-memory artifact has no question_id")
    path = root / "visible_memories" / f"{_safe_artifact_name(question_id)}.json"
    if path.is_file():
        existing = _read_json(path)
        if _canonical_json(existing) != _canonical_json(value):
            raise RunIdentityMismatch(
                f"visible memory snapshot changed across methods: {question_id}"
            )
        return
    _atomic_json(path, value)


def _persist_task_artifacts(
    prepared: _PreparedRun,
    task: DependencyTask,
    artifacts: Mapping[str, Any],
    *,
    attempt: int,
    partial: bool,
    partial_status: str = "failed_after_partial_execution",
) -> None:
    visible = artifacts.get("visible_memories")
    if isinstance(visible, Mapping):
        _write_visible_artifact(prepared.root, visible)
    candidate_pool = artifacts.get("candidate_pool")
    if isinstance(candidate_pool, Mapping):
        value = dict(candidate_pool)
        if partial:
            value["task_status"] = partial_status
            attempt_field = (
                "interrupted_attempt"
                if partial_status == "interrupted_after_partial_execution"
                else "failed_attempt"
            )
            value[attempt_field] = attempt
        _atomic_json(
            prepared.root / "candidate_pool" / f"{task.task_id}.json", value
        )
    module_events = artifacts.get("module_events", {})
    if not isinstance(module_events, Mapping):
        raise ValueError("task executor module_events must be a mapping")
    for module_name in MODULE_LOGS:
        raw_events = module_events.get(module_name, ())
        if isinstance(raw_events, Mapping):
            raw_events = (raw_events,)
        if isinstance(raw_events, (str, bytes)) or not isinstance(raw_events, Iterable):
            raise ValueError(f"module_events.{module_name} must be iterable")
        for event in raw_events:
            if not isinstance(event, Mapping):
                raise ValueError(f"module_events.{module_name} contains a non-object")
            _append_jsonl(
                prepared.root / "modules" / f"{module_name}.jsonl",
                {
                    "schema_version": 1,
                    "task_id": task.task_id,
                    "persona_id": task.persona_id,
                    "question_id": task.question_id,
                    "method_id": task.method_id,
                    "attempt": attempt,
                    "partial": partial,
                    "record_kind": "task_snapshot",
                    **dict(event),
                },
            )


def _persist_success_attempt(
    prepared: _PreparedRun,
    task: DependencyTask,
    result: Mapping[str, Any],
    *,
    attempt: int,
    started_at: float,
) -> DependencyOutcome:
    if result.get("status") != "success":
        raise ValueError("task executor did not return status=success")
    prediction = result.get("prediction")
    if not isinstance(prediction, str) or not prediction:
        raise ValueError("task executor returned no prediction")
    selected_ids = tuple(str(item) for item in result.get("selected_ids", ()))
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("task executor selected duplicate memory IDs")
    visible = result.get("visible_memories")
    candidate_pool = result.get("candidate_pool")
    module_events = result.get("module_events", {})
    if not isinstance(visible, Mapping) or not isinstance(candidate_pool, Mapping):
        raise ValueError("task executor omitted execution artifacts")
    if not isinstance(module_events, Mapping):
        raise ValueError("task executor module_events must be a mapping")

    finished = time.time()
    prediction_event = {
        "schema_version": 1,
        "event": "prediction",
        "attempt": attempt,
        "at_epoch": finished,
        "task": task.public_dict(),
        "prediction": prediction,
        "predicted_label": result.get("predicted_label"),
        "correct": bool(result.get("correct")),
        "parse_failed": bool(result.get("parse_failed")),
        "selected_ids": list(selected_ids),
        "context_hash": result.get("context_hash"),
        "costs": dict(result.get("costs", {})),
    }
    # Event history is append-only.  It is written before the authoritative
    # outcome: if the process dies in between, resume safely retries and the
    # history truthfully records both attempts.
    _append_jsonl(prepared.root / "predictions.jsonl", prediction_event)
    _persist_task_artifacts(
        prepared, task, result, attempt=attempt, partial=False
    )

    outcome = DependencyOutcome(
        task=task,
        status="success",
        attempt=attempt,
        started_at_epoch=started_at,
        finished_at_epoch=finished,
        prediction=prediction,
        predicted_label=str(result.get("predicted_label") or ""),
        correct=bool(result.get("correct")),
        parse_failed=bool(result.get("parse_failed")),
        selected_ids=selected_ids,
        context_hash=(
            None if result.get("context_hash") is None else str(result.get("context_hash"))
        ),
        costs=dict(result.get("costs", {})),
        diagnostics=dict(result.get("diagnostics", {})),
    )
    effectiveness = _module_effectiveness_event(task, outcome, result)
    outcome = replace(
        outcome,
        diagnostics={
            **dict(outcome.diagnostics),
            "module_effectiveness": effectiveness,
        },
    )
    _atomic_json(_outcome_path(prepared.root, task), outcome.public_dict())
    _append_jsonl(
        prepared.root / "events.jsonl",
        {
            "event": "task_success",
            "at_epoch": finished,
            "task_id": task.task_id,
            "question_id": task.question_id,
            "method_id": task.method_id,
            "attempt": attempt,
        },
    )
    return outcome


def _persist_failure_attempt(
    prepared: _PreparedRun,
    task: DependencyTask,
    exc: BaseException,
    *,
    attempt: int,
    started_at: float,
    costs: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, Any] | None = None,
) -> DependencyOutcome:
    finished = time.time()
    infrastructure = _is_infrastructure_failure(exc)
    retryable = bool(
        infrastructure and _is_retryable_infrastructure_failure(exc)
    )
    resolved_costs = dict(costs or {})
    resolved_costs["elapsed_ms"] = max(0.0, (finished - started_at) * 1000.0)
    resolved_costs["elapsed_ms_scope"] = "complete_failed_task_attempt"
    diagnostics = (
        dict(artifacts.get("diagnostics", {}))
        if isinstance(artifacts, Mapping) and isinstance(artifacts.get("diagnostics"), Mapping)
        else {}
    )
    if task.method_id == "evidence_bridge" and isinstance(artifacts, Mapping):
        from .evidence_diagnostics import evidence_bridge_summary
        diagnostics["evidence_bridge_summary"] = evidence_bridge_summary(
            artifacts.get("candidate_pool", {}), resolved_costs, artifacts.get("selected_ids", ())
        )
    error_event = {
        "schema_version": 1,
        "event": "task_failure",
        "attempt": attempt,
        "at_epoch": finished,
        "task": task.public_dict(),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "infrastructure_failure": infrastructure,
        "retryable": retryable,
        "retry_budget_exhausted": getattr(exc, "retry_budget_exhausted", False),
        "transport_error": getattr(exc, "error_metadata", {}),
        "costs": resolved_costs,
    }
    _append_jsonl(prepared.root / "failures.jsonl", error_event)
    _append_jsonl(
        prepared.root / "modules" / "cost.jsonl",
        {
            "schema_version": 1,
            "event": "failed_task_cost",
            "task_id": task.task_id,
            "persona_id": task.persona_id,
            "question_id": task.question_id,
            "method_id": task.method_id,
            "attempt": attempt,
            "partial": True,
            "costs": resolved_costs,
        },
    )
    _append_jsonl(
        prepared.root / "modules" / "stop.jsonl",
        {
            "schema_version": 1,
            "event": "task_failure_stop",
            "task_id": task.task_id,
            "persona_id": task.persona_id,
            "question_id": task.question_id,
            "method_id": task.method_id,
            "attempt": attempt,
            "reason": "infrastructure_failure" if infrastructure else "task_error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "infrastructure_failure": infrastructure,
            "retryable": retryable,
        },
    )
    outcome = DependencyOutcome(
        task=task,
        status="error",
        attempt=attempt,
        started_at_epoch=started_at,
        finished_at_epoch=finished,
        error_type=type(exc).__name__,
        error=str(exc),
        infrastructure_failure=infrastructure,
        costs=resolved_costs,
        diagnostics=diagnostics,
    )
    effectiveness = _module_effectiveness_event(task, outcome, artifacts)
    outcome = replace(
        outcome,
        diagnostics={
            **dict(outcome.diagnostics),
            "module_effectiveness": effectiveness,
        },
    )
    _atomic_json(_outcome_path(prepared.root, task), outcome.public_dict())
    _append_jsonl(prepared.root / "events.jsonl", error_event)
    return outcome


def _finalize_manifest(
    prepared: _PreparedRun,
    *,
    status: str,
    attempt_number: int,
    summary: Mapping[str, Any],
    stop_reason: str | None = None,
) -> dict[str, Any]:
    value = {
        **dict(prepared.manifest),
        "status": status,
        "updated_at_epoch": time.time(),
        "execution_attempts": attempt_number,
        "last_attempt": {
            "number": attempt_number,
            "status": status,
            "stop_reason": stop_reason,
            "completed_tasks": summary.get("completed_tasks"),
            "successful_tasks": summary.get("successful_tasks"),
            "failed_tasks": summary.get("failed_tasks"),
            "pending_tasks": summary.get("pending_tasks"),
        },
    }
    _atomic_json(prepared.root / "run_manifest.json", value)
    return value


def _finalize_interrupted_attempt(
    prepared: _PreparedRun,
    outcomes: Mapping[str, DependencyOutcome],
    *,
    attempt_number: int,
    tasks_attempted: int,
    stage: str,
    exc: KeyboardInterrupt,
    adapter_invocations: Mapping[str, int],
    active_task: DependencyTask | None = None,
    unique_tasks_attempted: int | None = None,
    tasks_skipped: int = 0,
) -> dict[str, Any]:
    """Persist a resumable operator interruption before returning control."""

    # An interrupt can land after an atomic outcome replace but before the
    # persistence helper returns.  Disk outcomes are authoritative, so always
    # reload them before deriving coverage and terminal status.
    authoritative_outcomes = load_authoritative_outcomes(
        prepared.root, prepared.tasks
    )
    summary = summarize_dependency_outcomes(
        prepared.tasks, authoritative_outcomes
    )
    pending = int(summary["pending_tasks"])
    failures = int(summary["failed_tasks"])
    if pending:
        resulting_status = "interrupted"
        inference_complete = False
        stop_reason = "execution_interrupted"
    elif failures:
        resulting_status = "completed_with_failures"
        inference_complete = True
        stop_reason = None
    else:
        resulting_status = "completed"
        inference_complete = True
        stop_reason = "fully_persisted_before_interruption"

    message = str(exc).strip() or "execution interrupted by operator"
    unique_attempted = (
        tasks_attempted
        if unique_tasks_attempted is None
        else int(unique_tasks_attempted)
    )
    event: dict[str, Any] = {
        "schema_version": 1,
        "event": "execution_interrupted",
        "status": "interrupted",
        "at_epoch": time.time(),
        "attempt": attempt_number,
        "stage": stage,
        "tasks_attempted_this_attempt": tasks_attempted,
        "unique_tasks_attempted_this_attempt": unique_attempted,
        "task_retries_this_attempt": max(0, tasks_attempted - unique_attempted),
        "tasks_skipped_this_attempt": tasks_skipped,
        "error_type": type(exc).__name__,
        "error": message,
        "infrastructure_failure": False,
        "adapter_invocations": dict(adapter_invocations),
        "resulting_status": resulting_status,
    }
    if active_task is not None:
        event.update(
            {
                "task_id": active_task.task_id,
                "persona_id": active_task.persona_id,
                "question_id": active_task.question_id,
                "method_id": active_task.method_id,
            }
        )
        committed = authoritative_outcomes.get(active_task.task_id)
        if committed is not None:
            event["authoritative_outcome_status"] = committed.status
    _append_jsonl(prepared.root / "failures.jsonl", event)
    _append_jsonl(prepared.root / "events.jsonl", event)
    views = _write_run_views(
        prepared.root,
        prepared.tasks,
        authoritative_outcomes,
        prepared.protocol_report,
        completion_status=resulting_status,
        inference_complete=inference_complete,
    )
    _finalize_manifest(
        prepared,
        status=resulting_status,
        attempt_number=attempt_number,
        summary=views["summary"],
        stop_reason=stop_reason,
    )
    _append_text(
        prepared.root / "train.log",
        (
            f"mode=train_free status={resulting_status} attempt={attempt_number} "
            f"stage={stage} completed={views['summary']['completed_tasks']} "
            f"failed={views['summary']['failed_tasks']} "
            f"pending={views['summary']['pending_tasks']} "
            "optimizer_steps=0 weights_updated=false"
        ),
    )
    return {
        "status": resulting_status,
        "run_dir": str(prepared.root),
        "summary": views["summary"],
        "stop_reason": stop_reason,
        "tasks_attempted_this_attempt": tasks_attempted,
        "unique_tasks_attempted_this_attempt": unique_attempted,
        "task_retries_this_attempt": max(0, tasks_attempted - unique_attempted),
        "tasks_skipped_this_attempt": tasks_skipped,
        "model_calls_this_attempt": sum(adapter_invocations.values()),
        "adapter_invocations_this_attempt": dict(adapter_invocations),
        "resume_noop": False,
    }


def _run_dependency_experiment_impl(
    config: str | Path | DependencyRunConfig,
    output_dir: str | Path,
    *,
    resume: bool = False,
    preflight_only: bool = False,
    protocol_manifest: str | Path | Mapping[str, Any] | None = None,
    embedder: Any | None = None,
    reranker: Any | None = None,
    generator: Any | None = None,
    examples: Sequence[PersonaMemExample] | None = None,
    require_full_32k: bool | None = None,
) -> dict[str, Any]:
    """Run or resume every frozen dependency method×question task.

    No service object is constructed during ``preflight_only``.  On resume,
    all identities and the frozen plan are checked first; if every outcome is
    already successful, the function returns before constructing clients or
    running the numerical protocol probe. Tasks that exhaust their transport
    retry budget retain an error outcome while later tasks continue. A failed
    service probe still stops execution before any task is attempted.
    """

    if not isinstance(resume, bool) or not isinstance(preflight_only, bool):
        raise ValueError("resume and preflight_only must be boolean")
    if preflight_only:
        if resume:
            raise ValueError("preflight_only cannot be combined with resume")
        return preflight_dependency_run(
            config,
            output_dir,
            protocol_manifest=protocol_manifest,
            examples=examples,
            require_full_32k=require_full_32k,
        )

    prepared = _prepare_run(
        config,
        output_dir,
        protocol_manifest=protocol_manifest,
        examples=examples,
        require_full_32k=require_full_32k,
        resume=resume,
    )
    outcomes = load_authoritative_outcomes(prepared.root, prepared.tasks)
    all_successful = all(
        task.task_id in outcomes and outcomes[task.task_id].status == "success"
        for task in prepared.tasks
    )
    previous_attempts = int(prepared.manifest.get("execution_attempts", 0))
    if all_successful:
        # The fully-successful zero-call path is intentionally before service
        # construction and before the once-per-attempt reranker probe.
        views = _write_run_views(
            prepared.root,
            prepared.tasks,
            outcomes,
            prepared.protocol_report,
            completion_status="completed",
            inference_complete=True,
        )
        _finalize_manifest(
            prepared,
            status="completed",
            attempt_number=previous_attempts,
            summary=views["summary"],
            stop_reason="fully_successful_resume_noop",
        )
        _append_jsonl(
            prepared.root / "events.jsonl",
            {
                "event": "resume_noop",
                "at_epoch": time.time(),
                "reason": "all frozen tasks already successful",
                "model_calls": 0,
            },
        )
        return {
            "status": "completed",
            "run_dir": str(prepared.root),
            "summary": views["summary"],
            "model_calls_this_attempt": 0,
            "adapter_invocations_this_attempt": {
                "embedding_adapter_invocations": 0,
                "reranker_adapter_invocations": 0,
                "generator_adapter_invocations": 0,
            },
            "tasks_attempted_this_attempt": 0,
            "unique_tasks_attempted_this_attempt": 0,
            "task_retries_this_attempt": 0,
            "tasks_skipped_this_attempt": 0,
            "resume_noop": True,
        }

    attempt_number = previous_attempts + 1
    # PR1: deployment metadata and streamed request events are separate from
    # legacy module checkpoints.  An outer diagnostic scope may additionally
    # impose one shared hard physical transport ceiling.
    audit_sink = JsonlAuditSink(prepared.root / "modules" / "requests.jsonl")
    audit_run = {"run_identity": prepared.manifest["identity"]["run_identity"],
                 "execution_attempt": attempt_number}
    with request_audit_scope(audit_run):
        service_identities = {
            name: deployment_public(getattr(prepared.config.models, name))
            for name in ("embedding", "reranker", "generator")
        }
    running_manifest = {
        **dict(prepared.manifest),
        "status": "running",
        "updated_at_epoch": time.time(),
        "execution_attempts": attempt_number,
        "deployment_identities": service_identities,
    }
    _atomic_json(prepared.root / "run_manifest.json", running_manifest)
    _append_text(
        prepared.root / "train.log",
        (
            f"mode=train_free status=running attempt={attempt_number} "
            "optimizer_steps=0 weights_updated=false"
        ),
    )
    _append_jsonl(
        prepared.root / "events.jsonl",
        {
            "event": "execution_attempt_started",
            "at_epoch": time.time(),
            "attempt": attempt_number,
            "resume": resume,
            "optimizer_steps": 0,
            "weights_updated": False,
        },
    )

    # All model-bearing objects are intentionally created after task plan
    # persistence and the successful-resume early return above.  Initialization
    # is nevertheless part of this recorded attempt: a local model/import
    # failure or an inability to start the heartbeat must not strand the
    # authoritative manifest in ``running`` with no explanation.
    executor: DependencyTaskExecutor | None = None
    heartbeat: _Heartbeat | None = None
    zero_adapter_invocations = {
        "embedding_adapter_invocations": 0,
        "reranker_adapter_invocations": 0,
        "generator_adapter_invocations": 0,
    }
    try:
        executor = DependencyTaskExecutor(
            prepared.config,
            embedder=embedder,
            reranker=reranker,
            generator=generator,
            cache_dir=prepared.config.runtime.cache_dir,
        )
        heartbeat = _Heartbeat(
            prepared.root, prepared.config.execution.heartbeat_seconds
        )
        heartbeat.start()
    except KeyboardInterrupt as exc:
        adapter_invocations = (
            zero_adapter_invocations
            if executor is None
            else _counter_delta(executor.adapter_cost, zero_adapter_invocations)
        )
        if heartbeat is not None:
            with suppress(Exception):
                heartbeat.close(completed_tasks=len(outcomes))
        return _finalize_interrupted_attempt(
            prepared,
            outcomes,
            attempt_number=attempt_number,
            tasks_attempted=0,
            stage="execution_initialization",
            exc=exc,
            adapter_invocations=adapter_invocations,
        )
    except Exception as exc:
        infrastructure = _is_infrastructure_failure(exc)
        adapter_invocations = (
            zero_adapter_invocations
            if executor is None
            else _counter_delta(executor.adapter_cost, zero_adapter_invocations)
        )
        failure = {
            "schema_version": 1,
            "event": "execution_initialization_failure",
            "status": "failed",
            "at_epoch": time.time(),
            "attempt": attempt_number,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "infrastructure_failure": infrastructure,
            "adapter_invocations": adapter_invocations,
        }
        _append_jsonl(prepared.root / "failures.jsonl", failure)
        _append_jsonl(prepared.root / "events.jsonl", failure)
        views = _write_run_views(
            prepared.root,
            prepared.tasks,
            outcomes,
            prepared.protocol_report,
            completion_status="interrupted",
            inference_complete=False,
        )
        _finalize_manifest(
            prepared,
            status="interrupted",
            attempt_number=attempt_number,
            summary=views["summary"],
            stop_reason="execution_initialization_failure",
        )
        if heartbeat is not None:
            with suppress(Exception):
                heartbeat.close(completed_tasks=len(outcomes))
        if infrastructure:
            return {
                "status": "interrupted",
                "run_dir": str(prepared.root),
                "summary": views["summary"],
                "stop_reason": "execution_initialization_failure",
                "model_calls_this_attempt": sum(adapter_invocations.values()),
                "adapter_invocations_this_attempt": adapter_invocations,
                "tasks_attempted_this_attempt": 0,
                "unique_tasks_attempted_this_attempt": 0,
                "task_retries_this_attempt": 0,
                "tasks_skipped_this_attempt": 0,
            }
        raise

    assert executor is not None and heartbeat is not None
    attempt_adapter_start = executor.adapter_cost
    first_pending = next(
        task
        for task in prepared.tasks
        if task.task_id not in outcomes or outcomes[task.task_id].status != "success"
    )
    probe_stage = "service_probe"
    terminal_probe_error: Exception | None = None
    terminal_probe_infrastructure = False
    probe_attempt = 0
    try:
        while True:
            probe_attempt += 1
            probe_adapter_start = executor.adapter_cost
            try:
                with request_audit_scope({**audit_run, "stage": "service_probe", "probe_attempt": probe_attempt}, sink=audit_sink):
                    probe = _pointwise_service_probe(
                        executor,
                        first_pending,
                        prepared.examples_by_id[first_pending.question_id],
                    )
            except Exception as exc:
                infrastructure = _is_infrastructure_failure(exc)
                retryable = bool(
                    infrastructure and _is_retryable_infrastructure_failure(exc)
                )
                probe_adapter_invocations = _counter_delta(
                    executor.adapter_cost, probe_adapter_start
                )
                measured = (
                    dict(exc.report)
                    if isinstance(exc, PointwiseProtocolMismatch)
                    else {}
                )
                failure = {
                    **measured,
                    "schema_version": 1,
                    "event": "service_probe_failure",
                    "status": "failed",
                    "at_epoch": time.time(),
                    # Keep the historical spelling for compatibility while
                    # making the nested attempt identity explicit.
                    "attempt": attempt_number,
                    "execution_attempt": attempt_number,
                    "probe_attempt": probe_attempt,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "infrastructure_failure": infrastructure,
                    "retryable": retryable,
                    "adapter_invocations": probe_adapter_invocations,
                }
                _append_jsonl(
                    prepared.root / "service_probes.jsonl",
                    failure,
                )
                _atomic_json(prepared.root / "service_probe.json", failure)
                _append_jsonl(prepared.root / "failures.jsonl", failure)
                _append_jsonl(prepared.root / "events.jsonl", failure)
                _emit_runtime_event(
                    prepared.root,
                    {
                        "schema_version": 1,
                        "event": "service_probe_failed",
                        "at_epoch": time.time(),
                        "attempt": attempt_number,
                        "execution_attempt": attempt_number,
                        "probe_attempt": probe_attempt,
                        "consistent": measured.get("consistent"),
                        "score_space": measured.get("score_space"),
                        "score_contract": measured.get("score_contract"),
                        "compared_values": measured.get("compared_values"),
                        "max_absolute_deviation": measured.get(
                            "max_absolute_deviation"
                        ),
                        "max_relative_deviation": measured.get(
                            "max_relative_deviation"
                        ),
                        "rtol": measured.get("rtol"),
                        "atol": measured.get("atol"),
                        "error_type": type(exc).__name__,
                        "infrastructure_failure": infrastructure,
                        "retryable": retryable,
                        "adapter_invocations": probe_adapter_invocations,
                        "search_budget_consumed": 0,
                    },
                )
                if (
                    retryable
                    and probe_attempt
                    < prepared.config.execution.infrastructure_task_max_attempts
                ):
                    delay = _infrastructure_retry_delay(
                        prepared.config, failed_attempts=probe_attempt
                    )
                    probe_stage = "service_probe_retry_wait"
                    _wait_for_infrastructure_retry(
                        prepared.root,
                        event_context={
                            "scope": "service_probe",
                            "execution_attempt": attempt_number,
                            "failed_attempt": probe_attempt,
                            "next_attempt": probe_attempt + 1,
                            "max_attempts": (
                                prepared.config.execution.infrastructure_task_max_attempts
                            ),
                            "attempts_remaining": (
                                prepared.config.execution.infrastructure_task_max_attempts
                                - probe_attempt
                            ),
                            "error_type": type(exc).__name__,
                            "retryable": retryable,
                        },
                        delay_seconds=delay,
                    )
                    probe_stage = "service_probe"
                    continue
                terminal_probe_error = exc
                terminal_probe_infrastructure = infrastructure
                if infrastructure:
                    _emit_runtime_event(
                        prepared.root,
                        {
                            "schema_version": 1,
                            "event": "infrastructure_circuit_opened",
                            "at_epoch": time.time(),
                            "scope": "service_probe",
                            "execution_attempt": attempt_number,
                            "failed_attempts": probe_attempt,
                            "max_attempts": (
                                prepared.config.execution.infrastructure_task_max_attempts
                            ),
                            "reason": (
                                "retry_budget_exhausted"
                                if retryable
                                else "non_retryable_infrastructure_failure"
                            ),
                            "error_type": type(exc).__name__,
                            "retryable": retryable,
                            "circuit_state": "open",
                        },
                    )
                break

            probe = {
                **probe,
                "execution_attempt": attempt_number,
                "probe_attempt": probe_attempt,
                "executed_at_epoch": time.time(),
            }
            _append_jsonl(prepared.root / "service_probes.jsonl", probe)
            _atomic_json(prepared.root / "service_probe.json", probe)
            _emit_runtime_event(
                prepared.root,
                {
                    "schema_version": 1,
                    "event": "service_probe_passed",
                    "at_epoch": time.time(),
                    "attempt": attempt_number,
                    "execution_attempt": attempt_number,
                    "probe_attempt": probe_attempt,
                    "consistent": probe["consistent"],
                    "score_space": probe["score_space"],
                    "score_contract": probe["score_contract"],
                    "compared_values": probe["compared_values"],
                    "max_absolute_deviation": probe["max_absolute_deviation"],
                    "max_relative_deviation": probe["max_relative_deviation"],
                    "rtol": probe["rtol"],
                    "atol": probe["atol"],
                    "reranker_adapter_requests": probe[
                        "reranker_adapter_requests"
                    ],
                    "search_budget_consumed": 0,
                },
            )
            if probe_attempt > 1:
                _emit_runtime_event(
                    prepared.root,
                    {
                        "schema_version": 1,
                        "event": "infrastructure_retry_recovered",
                        "at_epoch": time.time(),
                        "scope": "service_probe",
                        "execution_attempt": attempt_number,
                        "successful_attempt": probe_attempt,
                        "prior_infrastructure_failures": probe_attempt - 1,
                        "circuit_state": "closed",
                    },
                )
            break
    except KeyboardInterrupt as exc:
        adapter_invocations = _counter_delta(
            executor.adapter_cost, attempt_adapter_start
        )
        with suppress(Exception):
            heartbeat.close(completed_tasks=len(outcomes))
        return _finalize_interrupted_attempt(
            prepared,
            outcomes,
            attempt_number=attempt_number,
            tasks_attempted=0,
            stage=probe_stage,
            exc=exc,
            adapter_invocations=adapter_invocations,
        )

    if terminal_probe_error is not None:
        views = _write_run_views(
            prepared.root,
            prepared.tasks,
            outcomes,
            prepared.protocol_report,
            completion_status="interrupted",
            inference_complete=False,
        )
        _finalize_manifest(
            prepared,
            status="interrupted",
            attempt_number=attempt_number,
            summary=views["summary"],
            stop_reason="service_probe_failure",
        )
        heartbeat.close(completed_tasks=len(outcomes))
        if terminal_probe_infrastructure:
            adapter_invocations = _counter_delta(
                executor.adapter_cost, attempt_adapter_start
            )
            return {
                "status": "interrupted",
                "run_dir": str(prepared.root),
                "summary": views["summary"],
                "stop_reason": "service_probe_failure",
                "model_calls_this_attempt": sum(adapter_invocations.values()),
                "adapter_invocations_this_attempt": adapter_invocations,
                "tasks_attempted_this_attempt": 0,
                "unique_tasks_attempted_this_attempt": 0,
                "task_retries_this_attempt": 0,
                "tasks_skipped_this_attempt": 0,
            }
        raise terminal_probe_error
    # ``tasks_attempted`` counts executor invocations, including in-process
    # retries. Keep a separate unique count so operational cost is explicit.
    tasks_attempted = 0
    unique_tasks_attempted = 0
    tasks_skipped = 0
    active_task: DependencyTask | None = None
    interruption: KeyboardInterrupt | None = None
    interruption_stage = "task_execution"
    try:
        # A failed task is retried only within its bounded attempt budget for
        # this run. Once skipped, advance through the plan; an explicit resume
        # can retry errors later while preserving all successful outcomes.
        tasks_to_execute = [
            task for task in prepared.tasks
            if task.task_id not in outcomes or outcomes[task.task_id].status != "success"
        ]
        for task_index, task in enumerate(tasks_to_execute):
            previous = outcomes.get(task.task_id)
            unique_tasks_attempted += 1
            task_attempt = 1 if previous is None else previous.attempt + 1
            infrastructure_failures = 0
            active_task = task
            while True:
                tasks_attempted += 1
                interruption_stage = "task_execution"
                heartbeat.update(task=task, completed_tasks=len(outcomes))
                started_at = time.time()
                task_adapter_start = executor.adapter_cost
                retryable_infrastructure = False
                try:
                    recorder = None
                    if task.method_id == "evidence_bridge":
                        recorder = ModuleEventRecorder(
                            prepared.root, run_identity=audit_run["run_identity"],
                            metadata={"record_kind": "live", "task_attempt": task_attempt},
                        )

                        def persist_evidence_snapshot(event, snapshot):
                            _atomic_json(prepared.root / "evidence_live" / f"{task.task_id}.json", {
                                "task": task.public_dict(), "attempt": task_attempt,
                                "event": event["event"], "at_epoch": time.time(),
                                "evidence_selection": snapshot,
                            })

                        executor.evidence_snapshot_sink = persist_evidence_snapshot
                    with request_audit_scope({**audit_run, "task_attempt": task_attempt, "stage": "task_execution"}, sink=audit_sink), observation_scope(recorder):
                        result = executor.execute(
                            task, prepared.examples_by_id[task.question_id]
                        )
                    outcome = _persist_success_attempt(
                        prepared,
                        task,
                        result,
                        attempt=task_attempt,
                        started_at=started_at,
                    )
                except Exception as exc:
                    retryable_infrastructure = bool(
                        _is_infrastructure_failure(exc)
                        and _is_retryable_infrastructure_failure(exc)
                    )
                    partial = getattr(exc, "dependency_partial_artifacts", {})
                    if isinstance(partial, Mapping) and partial:
                        _persist_task_artifacts(
                            prepared,
                            task,
                            partial,
                            attempt=task_attempt,
                            partial=True,
                        )
                    task_adapter_cost = _counter_delta(
                        executor.adapter_cost, task_adapter_start
                    )
                    partial_costs = (
                        dict(partial.get("costs", {}))
                        if isinstance(partial, Mapping)
                        and isinstance(partial.get("costs"), Mapping)
                        else {}
                    )
                    partial_costs["adapter_invocations"] = task_adapter_cost
                    partial_costs["generator_calls"] = int(
                        task_adapter_cost.get("generator_adapter_invocations", 0)
                    )
                    outcome = _persist_failure_attempt(
                        prepared,
                        task,
                        exc,
                        attempt=task_attempt,
                        started_at=started_at,
                        costs=partial_costs,
                        artifacts=(
                            partial if isinstance(partial, Mapping) else None
                        ),
                    )
                outcomes[task.task_id] = outcome
                effectiveness = outcome.diagnostics.get("module_effectiveness")
                if not isinstance(effectiveness, Mapping):
                    raise ValueError(
                        "authoritative outcome omitted module_effectiveness diagnostics"
                    )
                _emit_runtime_event(
                    prepared.root,
                    effectiveness,
                    module_name="effectiveness",
                )
                # Make the retry-safe current view match a retry transition
                # immediately. Avoid rewriting the full materialized view on
                # every ordinary success; periodic run views own that path.
                if outcome.infrastructure_failure or infrastructure_failures:
                    _write_current_effectiveness(
                        prepared.root, prepared.tasks, outcomes
                    )

                if not outcome.infrastructure_failure:
                    if outcome.status == "success" and infrastructure_failures:
                        _emit_runtime_event(
                            prepared.root,
                            {
                                "schema_version": 1,
                                "event": "infrastructure_retry_recovered",
                                "at_epoch": time.time(),
                                "scope": "task",
                                "execution_attempt": attempt_number,
                                "task_id": task.task_id,
                                "persona_id": task.persona_id,
                                "question_id": task.question_id,
                                "method_id": task.method_id,
                                "successful_task_attempt": task_attempt,
                                "prior_infrastructure_failures": (
                                    infrastructure_failures
                                ),
                                "circuit_state": "closed",
                            },
                        )
                    heartbeat.update(task=None, completed_tasks=len(outcomes))
                    active_task = None
                    break

                infrastructure_failures += 1
                maximum_attempts = (
                    prepared.config.execution.infrastructure_task_max_attempts
                )
                if (
                    not retryable_infrastructure
                    or infrastructure_failures >= maximum_attempts
                ):
                    tasks_skipped += 1
                    next_task = (
                        tasks_to_execute[task_index + 1]
                        if task_index + 1 < len(tasks_to_execute)
                        else None
                    )
                    _emit_runtime_event(
                        prepared.root,
                        {
                            "schema_version": 1,
                            "event": "infrastructure_task_skipped",
                            "at_epoch": time.time(),
                            "scope": "task",
                            "execution_attempt": attempt_number,
                            "task_id": task.task_id,
                            "persona_id": task.persona_id,
                            "question_id": task.question_id,
                            "method_id": task.method_id,
                            "failed_task_attempt": task_attempt,
                            "failed_attempts": infrastructure_failures,
                            "max_attempts": maximum_attempts,
                            "reason": (
                                "retry_budget_exhausted"
                                if retryable_infrastructure
                                else "non_retryable_infrastructure_failure"
                            ),
                            "error_type": outcome.error_type,
                            "error": outcome.error,
                            "retryable": retryable_infrastructure,
                            "circuit_state": "closed",
                            "action": "continue_to_next_task",
                            "next_task_id": None if next_task is None else next_task.task_id,
                            "next_question_id": None if next_task is None else next_task.question_id,
                            "next_method_id": None if next_task is None else next_task.method_id,
                            "tasks_skipped_this_attempt": tasks_skipped,
                        },
                        module_name="stop",
                    )
                    # Publish the failed outcome and remaining work immediately
                    # even when the periodic reporting interval has not elapsed.
                    _write_run_views(
                        prepared.root,
                        prepared.tasks,
                        outcomes,
                        prepared.protocol_report,
                        completion_status="running",
                        inference_complete=False,
                    )
                    heartbeat.update(task=None, completed_tasks=len(outcomes))
                    active_task = None
                    break

                delay = _infrastructure_retry_delay(
                    prepared.config,
                    failed_attempts=infrastructure_failures,
                )
                interruption_stage = "task_infrastructure_retry_wait"
                _wait_for_infrastructure_retry(
                    prepared.root,
                    event_context={
                        "scope": "task",
                        "execution_attempt": attempt_number,
                        "task_id": task.task_id,
                        "persona_id": task.persona_id,
                        "question_id": task.question_id,
                        "method_id": task.method_id,
                        "failed_attempt": infrastructure_failures,
                        "failed_task_attempt": task_attempt,
                        "next_attempt": infrastructure_failures + 1,
                        "next_task_attempt": task_attempt + 1,
                        "max_attempts": maximum_attempts,
                        "attempts_remaining": (
                            maximum_attempts - infrastructure_failures
                        ),
                        "error_type": outcome.error_type,
                        "retryable": retryable_infrastructure,
                    },
                    delay_seconds=delay,
                )
                task_attempt = outcome.attempt + 1

            if (
                unique_tasks_attempted
                % prepared.config.execution.log_every_questions
                == 0
            ):
                progress = summarize_dependency_outcomes(prepared.tasks, outcomes)
                line = {
                    "event": "inference_progress",
                    "attempt": attempt_number,
                    "tasks_attempted_this_attempt": tasks_attempted,
                    "unique_tasks_attempted_this_attempt": unique_tasks_attempted,
                    "task_retries_this_attempt": max(
                        0, tasks_attempted - unique_tasks_attempted
                    ),
                    "tasks_skipped_this_attempt": tasks_skipped,
                    "tasks_attempted_unit": "executor_invocation",
                    "unique_tasks_attempted_unit": "method_x_question",
                    "progress_cadence_unit": "unique_method_x_question",
                    "completed_tasks": progress["completed_tasks"],
                    "successful_tasks": progress["successful_tasks"],
                    "failed_tasks": progress["failed_tasks"],
                    "pending_tasks": progress["pending_tasks"],
                    "optimizer_steps": 0,
                    "weights_updated": False,
                }
                print(json.dumps(line, ensure_ascii=False, sort_keys=True), flush=True)
                _append_text(
                    prepared.root / "train.log",
                    " ".join(f"{key}={value}" for key, value in line.items()),
                )
            if (
                unique_tasks_attempted
                % prepared.config.execution.evaluate_every_questions
                == 0
            ):
                _emit_runtime_event(
                    prepared.root,
                    _method_metrics_event(
                        prepared.tasks,
                        outcomes,
                        attempt=attempt_number,
                        tasks_attempted_this_attempt=tasks_attempted,
                        unique_tasks_attempted_this_attempt=(
                            unique_tasks_attempted
                        ),
                        checkpoint="periodic",
                        tasks_skipped_this_attempt=tasks_skipped,
                    ),
                    module_name="effectiveness",
                )
                _write_run_views(
                    prepared.root,
                    prepared.tasks,
                    outcomes,
                    prepared.protocol_report,
                    completion_status="running",
                    inference_complete=False,
                )
    except KeyboardInterrupt as exc:
        partial = getattr(exc, "dependency_partial_artifacts", {})
        if active_task is not None and isinstance(partial, Mapping) and partial:
            with suppress(Exception):
                _persist_task_artifacts(
                    prepared,
                    active_task,
                    partial,
                    attempt=(
                        1
                        if outcomes.get(active_task.task_id) is None
                        else outcomes[active_task.task_id].attempt + 1
                    ),
                    partial=True,
                    partial_status="interrupted_after_partial_execution",
                )
        interruption = exc
    finally:
        heartbeat.close(completed_tasks=len(outcomes))

    if interruption is not None:
        adapter_invocations = _counter_delta(
            executor.adapter_cost, attempt_adapter_start
        )
        return _finalize_interrupted_attempt(
            prepared,
            outcomes,
            attempt_number=attempt_number,
            tasks_attempted=tasks_attempted,
            stage=interruption_stage,
            exc=interruption,
            adapter_invocations=adapter_invocations,
            active_task=active_task,
            unique_tasks_attempted=unique_tasks_attempted,
            tasks_skipped=tasks_skipped,
        )

    summary_now = summarize_dependency_outcomes(prepared.tasks, outcomes)
    pending = int(summary_now["pending_tasks"])
    failures = int(summary_now["failed_tasks"])
    if (
        unique_tasks_attempted
        and unique_tasks_attempted
        % prepared.config.execution.evaluate_every_questions
        != 0
    ):
        _emit_runtime_event(
            prepared.root,
            _method_metrics_event(
                prepared.tasks,
                outcomes,
                attempt=attempt_number,
                tasks_attempted_this_attempt=tasks_attempted,
                unique_tasks_attempted_this_attempt=unique_tasks_attempted,
                checkpoint="attempt_end",
                tasks_skipped_this_attempt=tasks_skipped,
            ),
            module_name="effectiveness",
        )
    if pending:
        status = "interrupted"
        inference_complete = False
    elif failures:
        status = "completed_with_failures"
        inference_complete = True
    else:
        status = "completed"
        inference_complete = True
    views = _write_run_views(
        prepared.root,
        prepared.tasks,
        outcomes,
        prepared.protocol_report,
        completion_status=status,
        inference_complete=inference_complete,
    )
    _finalize_manifest(
        prepared,
        status=status,
        attempt_number=attempt_number,
        summary=views["summary"],
        stop_reason=None,
    )
    _append_text(
        prepared.root / "train.log",
        (
            f"mode=train_free status={status} attempt={attempt_number} "
            f"completed={views['summary']['completed_tasks']} "
            f"failed={views['summary']['failed_tasks']} "
            f"pending={views['summary']['pending_tasks']} "
            f"tasks_skipped_this_attempt={tasks_skipped} "
            "optimizer_steps=0 weights_updated=false"
        ),
    )
    _append_jsonl(
        prepared.root / "events.jsonl",
        {
            "event": "execution_attempt_finished",
            "at_epoch": time.time(),
            "attempt": attempt_number,
            "status": status,
            "tasks_attempted": tasks_attempted,
            "unique_tasks_attempted": unique_tasks_attempted,
            "task_retries": max(0, tasks_attempted - unique_tasks_attempted),
            "infrastructure_stop": None,
            "tasks_skipped_this_attempt": tasks_skipped,
            "completed_tasks": views["summary"]["completed_tasks"],
            "pending_tasks": views["summary"]["pending_tasks"],
        },
    )
    adapter_invocations = _counter_delta(executor.adapter_cost, attempt_adapter_start)
    return {
        "status": status,
        "run_dir": str(prepared.root),
        "summary": views["summary"],
        "stop_reason": None,
        "tasks_attempted_this_attempt": tasks_attempted,
        "unique_tasks_attempted_this_attempt": unique_tasks_attempted,
        "task_retries_this_attempt": max(
            0, tasks_attempted - unique_tasks_attempted
        ),
        "tasks_skipped_this_attempt": tasks_skipped,
        "model_calls_this_attempt": sum(adapter_invocations.values()),
        "adapter_invocations_this_attempt": adapter_invocations,
        "resume_noop": False,
    }


def _recover_outer_interruption(
    config: str | Path | DependencyRunConfig,
    output_dir: str | Path,
    *,
    resume: bool,
    preflight_only: bool,
    protocol_manifest: str | Path | Mapping[str, Any] | None,
    examples: Sequence[PersonaMemExample] | None,
    require_full_32k: bool | None,
    exc: KeyboardInterrupt,
) -> dict[str, Any]:
    """Recover interruption windows outside the attempt's phase guards.

    Recovery is attempted only after both identity manifest and frozen plan
    exist atomically.  Otherwise the original interrupt is re-raised and a
    later non-resume invocation may safely prepare that incomplete directory.
    """

    root = Path(output_dir).expanduser().resolve()
    if not (root / "run_manifest.json").is_file() or not (
        root / "planned_tasks.jsonl"
    ).is_file():
        raise exc
    try:
        prepared = _prepare_run(
            config,
            root,
            protocol_manifest=protocol_manifest,
            examples=examples,
            require_full_32k=require_full_32k,
            resume=True,
        )
        outcomes = load_authoritative_outcomes(prepared.root, prepared.tasks)
    except Exception:
        raise exc from None
    _close_active_heartbeat(prepared.root, completed_tasks=len(outcomes))

    message = str(exc).strip() or "execution interrupted by operator"
    attempt_number = int(prepared.manifest.get("execution_attempts", 0))
    event = {
        "schema_version": 1,
        "event": "outer_interruption_recovered",
        "at_epoch": time.time(),
        "attempt": attempt_number,
        "error_type": type(exc).__name__,
        "error": message,
        "resume": resume,
        "preflight_only": preflight_only,
    }

    if preflight_only:
        if attempt_number != 0 or str(prepared.manifest.get("status", "")) not in {
            "prepared",
            "preflight_complete",
        }:
            raise exc
        views = _write_run_views(
            prepared.root,
            prepared.tasks,
            outcomes,
            prepared.protocol_report,
            completion_status="preflight_complete",
            inference_complete=False,
        )
        manifest = {
            **dict(prepared.manifest),
            "status": "preflight_complete",
            "updated_at_epoch": time.time(),
            "preflight": {
                "status": "passed",
                "questions": len(prepared.dataset.examples),
                "expected_tasks": len(prepared.tasks),
                "model_calls": 0,
                "recovered_after_interruption": True,
            },
        }
        _atomic_json(prepared.root / "run_manifest.json", manifest)
        _append_jsonl(
            prepared.root / "events.jsonl",
            {**event, "resulting_status": "preflight_complete"},
        )
        return {
            "status": "preflight_complete",
            "run_dir": str(prepared.root),
            "dataset": prepared.dataset.public_dict(),
            "expected_tasks": len(prepared.tasks),
            "methods": list(prepared.config.methods),
            "identity": dict(prepared.identity),
            "protocol": dict(prepared.protocol_report),
            "model_calls": 0,
            "inference_complete": False,
            "summary": views["summary"],
            "recovered_after_interruption": True,
        }

    summary = summarize_dependency_outcomes(prepared.tasks, outcomes)
    pending = int(summary["pending_tasks"])
    failures = int(summary["failed_tasks"])
    if pending:
        status = "interrupted"
        inference_complete = False
        stop_reason = "execution_interrupted"
    elif failures:
        status = "completed_with_failures"
        inference_complete = True
        stop_reason = None
    else:
        status = "completed"
        inference_complete = True
        stop_reason = "fully_persisted_before_interruption"
    views = _write_run_views(
        prepared.root,
        prepared.tasks,
        outcomes,
        prepared.protocol_report,
        completion_status=status,
        inference_complete=inference_complete,
    )
    _finalize_manifest(
        prepared,
        status=status,
        attempt_number=attempt_number,
        summary=views["summary"],
        stop_reason=stop_reason,
    )
    terminal_event = {**event, "resulting_status": status}
    _append_jsonl(prepared.root / "events.jsonl", terminal_event)
    if status == "interrupted":
        _append_jsonl(
            prepared.root / "failures.jsonl",
            {**terminal_event, "status": "interrupted"},
        )
    return {
        "status": status,
        "run_dir": str(prepared.root),
        "summary": views["summary"],
        "stop_reason": stop_reason,
        "model_calls_this_attempt": 0 if resume and not pending and not failures else None,
        "tasks_attempted_this_attempt": None,
        "unique_tasks_attempted_this_attempt": None,
        "task_retries_this_attempt": None,
        "tasks_skipped_this_attempt": None,
        "resume_noop": bool(resume and not pending and not failures),
        "recovered_after_interruption": True,
    }


def run_dependency_experiment(
    config: str | Path | DependencyRunConfig,
    output_dir: str | Path,
    *,
    resume: bool = False,
    preflight_only: bool = False,
    protocol_manifest: str | Path | Mapping[str, Any] | None = None,
    embedder: Any | None = None,
    reranker: Any | None = None,
    generator: Any | None = None,
    examples: Sequence[PersonaMemExample] | None = None,
    require_full_32k: bool | None = None,
) -> dict[str, Any]:
    """Run the dependency experiment with a final atomic interruption guard."""

    try:
        return _run_dependency_experiment_impl(
            config,
            output_dir,
            resume=resume,
            preflight_only=preflight_only,
            protocol_manifest=protocol_manifest,
            embedder=embedder,
            reranker=reranker,
            generator=generator,
            examples=examples,
            require_full_32k=require_full_32k,
        )
    except KeyboardInterrupt as exc:
        return _recover_outer_interruption(
            config,
            output_dir,
            resume=resume,
            preflight_only=preflight_only,
            protocol_manifest=protocol_manifest,
            examples=examples,
            require_full_32k=require_full_32k,
            exc=exc,
        )


# Descriptive aliases used by worker/CLI integrations and older drafts.
run_dependency_chain = run_dependency_experiment
run_experiment = run_dependency_experiment
preflight = preflight_dependency_run


__all__ = [
    "DEFAULT_METHODS",
    "DependencyDataset",
    "DependencyOutcome",
    "DependencyTask",
    "DependencyTaskExecutor",
    "FrozenPlanError",
    "PointwiseProtocolMismatch",
    "RunIdentityMismatch",
    "build_dependency_plan",
    "load_authoritative_outcomes",
    "load_dependency_dataset",
    "preflight_dependency_run",
    "run_dependency_chain",
    "run_dependency_experiment",
    "summarize_dependency_outcomes",
]
