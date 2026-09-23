"""Canonical frozen-reranker utilities for conditional dependency search.

For one instance, the query and visible memory bank are immutable.  Every
score is therefore a pointwise utility ``R_q(S)`` of a canonical real-memory
set.  Persistent cache hits still consume the same logical unique-set budget
as cold scores; the cache changes network cost, never the explored frontier.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .clients import _response_declares_truncation, build_rerank_payload, estimate_tokens
from .diagnostic_identity import deployment_fingerprint
from .diagnostic_observability import observe
from .request_audit import current_audit_scope, document_descriptors, logical_request_scope, request_audit_scope
from .types import Memory

EMPTY_SET_SERIALIZATION = "Personal memories: [No personal memories supplied.]"
SET_SERIALIZATION_TEMPLATE_VERSION = "dependency-set-v1"
RERANKER_TRANSPORT_STAT_KEYS = (
    "logical_calls",
    "logical_documents",
    "batch_requests",
    "batch_documents",
    "transport_attempts",
    "transport_document_attempts",
    "failed_batch_requests",
    "split_events",
    "split_recovered_calls",
    "failed_calls",
)


class SetScoringError(ValueError):
    """Base class for explicit set-scoring contract failures."""


class SetBudgetExceeded(SetScoringError):
    """A complete logical scoring request does not fit its unique-set quota."""

    def __init__(self, *, required: int, remaining: int | None, limit: int | None):
        self.required = int(required)
        self.remaining = remaining
        self.limit = limit
        super().__init__(
            "set scoring budget exhausted: "
            f"request needs {self.required} new unique set(s), "
            f"remaining={self.remaining}, limit={self.limit}"
        )


class SetInputTooLong(SetScoringError):
    """At least one exact reranker input exceeds the configured capacity."""

    def __init__(self, ids: Sequence[str], *, estimated_tokens: int, limit: int):
        self.ids = tuple(ids)
        self.estimated_tokens = int(estimated_tokens)
        self.limit = int(limit)
        super().__init__(
            f"reranker input capacity exceeded for set {self.ids!r}: estimated at "
            f"{self.estimated_tokens} tokens, exceeding limit {self.limit}"
        )


class SetResponseError(SetScoringError):
    """The reranker response is incomplete, ambiguous, or non-finite."""


class SetCacheError(SetScoringError):
    """A persistent set-score entry is malformed or has the wrong identity."""


# Compatibility names make stop-reason handling readable in callers.
ScoringBudgetError = SetBudgetExceeded
InputCapacityError = SetInputTooLong
RerankerProtocolError = SetResponseError


def _strict_int(value: Any, name: str, *, positive: bool = False, nonnegative: bool = False) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, Integral):
        result = int(value)
    elif isinstance(value, Real):
        numeric = float(value)
        if not math.isfinite(numeric) or numeric != float(int(numeric)):
            raise ValueError(f"{name} must be an integer")
        result = int(numeric)
    elif isinstance(value, str):
        text = value.strip()
        try:
            numeric = float(text)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be an integer") from exc
        if not text or not math.isfinite(numeric) or numeric != float(int(numeric)):
            raise ValueError(f"{name} must be an integer")
        result = int(numeric)
    else:
        raise ValueError(f"{name} must be an integer")
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _strict_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _jsonable(value: Any, path: str = "value") -> Any:
    """Return deterministic JSON-shaped provenance or fail explicitly."""

    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist(), path)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return _jsonable(value.item(), path)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string key")
            result[key] = _jsonable(child, f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_jsonable(child, f"{path}[{index}]") for index, child in enumerate(value)]
    raise ValueError(f"{path} contains a non-JSON value of type {type(value).__name__}")


def _canonical_score_space(value: Any) -> str:
    raw = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "probability": "unit_interval",
        "probabilities": "unit_interval",
        "unit": "unit_interval",
        "unitinterval": "unit_interval",
        "sigmoid": "unit_interval",
        "logit": "logit_difference",
        "logit_diff": "logit_difference",
        "raw_logit_difference": "logit_difference",
    }
    result = aliases.get(raw, raw)
    if result not in {"unit_interval", "logit_difference"}:
        raise ValueError("score_space must be unit_interval or logit_difference")
    return result


@dataclass(frozen=True)
class CanonicalMemoryRecord:
    memory_id: str
    text: str
    timestamp: float
    source_id: str
    metadata: Mapping[str, Any]

    @classmethod
    def from_value(cls, identifier: str, value: Memory | Mapping[str, Any]) -> "CanonicalMemoryRecord":
        if isinstance(value, Memory):
            raw_id = value.memory_id
            text = value.text
            timestamp = value.timestamp
            source_id = value.source_id
            metadata = value.metadata
        elif isinstance(value, Mapping):
            raw_id = value.get("memory_id", identifier)
            text = value.get("text")
            timestamp = value.get("timestamp")
            source_id = value.get("source_id")
            metadata = value.get("metadata", {})
        else:
            raise ValueError("scoring records must be Memory objects or mappings")
        memory_id = str(raw_id)
        if not memory_id or memory_id != identifier:
            raise ValueError("record mapping keys must exactly match memory_id")
        if not isinstance(text, str):
            raise ValueError(f"record {memory_id} text must be a string")
        timestamp_value = _strict_float(timestamp, f"record {memory_id} timestamp")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(f"record {memory_id} source_id must be a non-empty string")
        metadata_value = _jsonable(metadata, f"record {memory_id} metadata")
        if not isinstance(metadata_value, Mapping):
            raise ValueError(f"record {memory_id} metadata must be a mapping")
        # Round-tripping severs references to mutable caller-owned metadata.
        snapshot = json.loads(json.dumps(metadata_value, ensure_ascii=False, sort_keys=True))
        return cls(memory_id, text, timestamp_value, source_id, snapshot)

    def identity_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "text": self.text,
            "timestamp": self.timestamp,
            "source_id": self.source_id,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PreparedSet:
    ids: tuple[str, ...]
    document: str
    estimated_input_tokens: int
    cache_key: str


@dataclass(frozen=True)
class ScorePreflight:
    requested_sets: int
    distinct_sets: int
    new_unique_sets: int
    remaining_before: int | None
    budget_limit: int | None
    max_estimated_input_tokens: int
    prepared: tuple[PreparedSet, ...]

    @property
    def fits(self) -> bool:
        return True


@dataclass(frozen=True)
class ActivationScores:
    premise_ids: tuple[str, ...]
    target_id: str
    group_ids: tuple[str, ...]
    p: float
    pe: float
    pg: float
    pge: float

    @property
    def activation(self) -> float:
        return self.pge - self.pg - self.pe + self.p

    @property
    def context_marginal(self) -> float:
        return self.pge - self.pe

    # Exact spellings used in persisted activation logs.
    @property
    def P(self) -> float:  # noqa: N802
        return self.p

    @property
    def Pe(self) -> float:  # noqa: N802
        return self.pe

    @property
    def PG(self) -> float:  # noqa: N802
        return self.pg

    @property
    def PGe(self) -> float:  # noqa: N802
        return self.pge

    def public_dict(self) -> dict[str, Any]:
        return {
            "premise_ids": list(self.premise_ids),
            "target_id": self.target_id,
            "group_ids": list(self.group_ids),
            "P": self.p,
            "Pe": self.pe,
            "PG": self.pg,
            "PGe": self.pge,
            "activation": self.activation,
            "context_marginal": self.context_marginal,
        }


@dataclass(frozen=True)
class PointwiseProtocolReport:
    consistent: bool
    score_space: str
    score_contract: str
    rtol: float
    atol: float
    compared_values: int
    max_absolute_deviation: float
    max_relative_deviation: float
    adapter_requests: int

    def public_dict(self) -> dict[str, Any]:
        return {
            "consistent": self.consistent,
            "score_space": self.score_space,
            "score_contract": self.score_contract,
            "rtol": self.rtol,
            "atol": self.atol,
            "compared_values": self.compared_values,
            "max_absolute_deviation": self.max_absolute_deviation,
            "max_relative_deviation": self.max_relative_deviation,
            "reranker_adapter_requests": self.adapter_requests,
        }


def _truncation_reported(value: Any) -> bool:
    """Use the exact shared explicit-positive truncation contract.

    Directly injected rerankers can bypass :class:`RerankerClient` and return
    their raw response envelope here.  Reusing the client predicate keeps the
    two paths identical and deliberately avoids guessing from token counts,
    finish reasons, or a configured truncation strategy.
    """

    return _response_declares_truncation(value)


def _response_items(raw: Any) -> Sequence[Any]:
    if _truncation_reported(raw):
        raise SetResponseError("reranker explicitly reported input truncation")
    items = raw.get("results", raw.get("data", raw.get("items"))) if isinstance(raw, Mapping) else raw
    if isinstance(items, (str, bytes)) or not isinstance(items, Sequence):
        raise SetResponseError("reranker response must be a sequence of indexed scores")
    return items


def _restore_indexed_scores(raw: Any, count: int, score_space: str) -> list[float]:
    items = _response_items(raw)
    if len(items) != count:
        raise SetResponseError(
            f"reranker returned {len(items)} item(s) for {count} document(s)"
        )
    restored: list[float | None] = [None] * count
    for item in items:
        if _truncation_reported(item):
            raise SetResponseError("reranker explicitly reported input truncation")
        if isinstance(item, Mapping):
            if "index" not in item:
                raise SetResponseError("reranker response item is missing index")
            raw_index = item["index"]
            raw_score = item.get("relevance_score", item.get("score"))
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            raw_index, raw_score = item
        else:
            try:
                raw_index, raw_score = item.index, item.score
            except AttributeError as exc:
                raise SetResponseError("reranker response item must contain index and score") from exc
        try:
            index = _strict_int(raw_index, "reranker response index", nonnegative=True)
        except ValueError as exc:
            raise SetResponseError(str(exc)) from exc
        if raw_score is None:
            raise SetResponseError("reranker response item is missing score")
        try:
            score = _strict_float(raw_score, "reranker response score")
        except ValueError as exc:
            raise SetResponseError(str(exc)) from exc
        if index >= count or restored[index] is not None:
            raise SetResponseError("reranker response contains an unknown or duplicate index")
        if score_space == "unit_interval" and not 0.0 <= score <= 1.0:
            raise SetResponseError("unit-interval reranker score is outside [0, 1]")
        restored[index] = score
    if any(value is None for value in restored):
        raise SetResponseError("reranker response omits one or more document indices")
    return [float(value) for value in restored if value is not None]


def _invoke_reranker(reranker: Any, query: str, documents: Sequence[str], score_space: str) -> list[float]:
    rerank_all = getattr(reranker, "rerank_all", None)
    rerank = getattr(reranker, "rerank", None)
    if callable(rerank_all):
        raw = rerank_all(query, list(documents))
    elif callable(rerank):
        raw = rerank(query, list(documents), len(documents))
    elif callable(reranker):
        raw = reranker(query, list(documents))
    else:
        raise SetResponseError("reranker must provide rerank_all or rerank")
    return _restore_indexed_scores(raw, len(documents), score_space)


class SetReranker:
    """Pointwise scorer for canonical sets from one visible history bank."""

    def __init__(
        self,
        query: str,
        records: Mapping[str, Memory | Mapping[str, Any]] | None,
        reranker: Any | None = None,
        *,
        client: Any | None = None,
        memories: Mapping[str, Memory | Mapping[str, Any]] | None = None,
        cache_dir: str | Path | None = None,
        cache_identity: Mapping[str, Any] | None = None,
        query_identity: Any = None,
        visible_history_cutoff: Any = None,
        cutoff: Any = None,
        batch_size: int = 32,
        max_input_tokens: int = 8192,
        set_budget: int | None = None,
        max_scored_sets: int | None = None,
        score_space: str | None = None,
        score_contract: str | None = None,
        template_version: str = SET_SERIALIZATION_TEMPLATE_VERSION,
    ):
        if not isinstance(query, str) or not query:
            raise ValueError("set scorer query must be a non-empty string")
        if records is not None and memories is not None:
            raise ValueError("records and memories cannot both be supplied")
        raw_records = records if records is not None else memories
        if not isinstance(raw_records, Mapping):
            raise ValueError("set scorer records must be a mapping")
        if reranker is not None and client is not None and reranker is not client:
            raise ValueError("reranker and client disagree")
        self.reranker = reranker if reranker is not None else client
        if self.reranker is None:
            raise ValueError("set scorer requires a reranker client")
        self.query = query

        snapshots: dict[str, CanonicalMemoryRecord] = {}
        for raw_identifier, value in raw_records.items():
            identifier = str(raw_identifier)
            if not identifier or identifier in snapshots:
                raise ValueError("record IDs must remain unique non-empty strings")
            snapshots[identifier] = CanonicalMemoryRecord.from_value(identifier, value)
        self.records = snapshots

        self.batch_size = _strict_int(batch_size, "reranker batch_size", positive=True)
        self.max_input_tokens = _strict_int(
            max_input_tokens, "reranker max input tokens", positive=True
        )
        if set_budget is not None and max_scored_sets is not None and set_budget != max_scored_sets:
            raise ValueError("set_budget and max_scored_sets disagree")
        initial_budget = max_scored_sets if set_budget is None else set_budget
        self._set_budget = (
            None
            if initial_budget is None
            else _strict_int(initial_budget, "set budget", nonnegative=True)
        )
        config = getattr(self.reranker, "config", None)
        declared_space = (
            score_space
            if score_space is not None
            else getattr(self.reranker, "score_space", getattr(config, "score_space", "unit_interval"))
        )
        self.score_space = _canonical_score_space(declared_space)
        declared_contract = (
            score_contract
            if score_contract is not None
            else getattr(self.reranker, "score_contract", getattr(config, "score_contract", "pointwise"))
        )
        self.score_contract = str(declared_contract).strip().lower()
        if self.score_contract != "pointwise":
            raise ValueError("dependency set scoring requires a pointwise reranker contract")
        if not isinstance(template_version, str) or not template_version.strip():
            raise ValueError("set serialization template_version must be non-empty")
        self.template_version = template_version

        if visible_history_cutoff is not None and cutoff is not None and visible_history_cutoff != cutoff:
            raise ValueError("visible_history_cutoff and cutoff disagree")
        effective_cutoff = visible_history_cutoff if visible_history_cutoff is not None else cutoff
        explicit_identity = _jsonable(cache_identity or {}, "cache_identity")
        backend_identity = {
            "endpoint": str(getattr(config, "endpoint", getattr(self.reranker, "endpoint", ""))),
            "model": str(getattr(config, "model", getattr(self.reranker, "model", ""))),
            "model_fingerprint": str(
                getattr(
                    self.reranker,
                    "model_fingerprint",
                    getattr(config, "model_fingerprint", getattr(config, "model", "")),
                )
            ),
            "task_instruction": str(getattr(config, "task_instruction", "")),
            "score_space": self.score_space,
            "score_contract": self.score_contract,
            "deployment_fingerprint": deployment_fingerprint(config),
        }
        namespace_payload = {
            "schema": 1,
            "template_version": self.template_version,
            "query": self.query,
            "query_identity": _jsonable(query_identity, "query_identity"),
            "visible_history_cutoff": _jsonable(effective_cutoff, "visible_history_cutoff"),
            "visible_records": [
                record.identity_dict()
                for record in sorted(self.records.values(), key=lambda item: item.memory_id)
            ],
            "backend": backend_identity,
            "cache_identity": explicit_identity,
        }
        encoded_namespace = json.dumps(
            namespace_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        self.namespace_hash = hashlib.sha256(encoded_namespace.encode("utf-8")).hexdigest()
        self.cache_identity_hash = self.namespace_hash
        self.cache_dir = None if cache_dir is None else Path(cache_dir).expanduser()
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._prepared: dict[tuple[str, ...], PreparedSet] = {}
        self._scores: dict[str, float] = {}
        self._logical_seen: set[str] = set()
        self._memory_cache_hits = 0
        self._persistent_cache_hits = 0
        self._client_requests = 0
        self._client_samples = 0
        self._client_elapsed_ms = 0.0
        self._logical_input_tokens = 0
        self._transport_stats_start = self._transport_stats_snapshot()
        self.events: list[dict[str, Any]] = []

    def _transport_stats_snapshot(self) -> dict[str, int]:
        """Read optional production-client transport counters safely."""

        try:
            raw = getattr(self.reranker, "transport_stats", None)
        except (AttributeError, TypeError, ValueError):
            return {}
        if not isinstance(raw, Mapping):
            return {}
        result: dict[str, int] = {}
        for key in RERANKER_TRANSPORT_STAT_KEYS:
            value = raw.get(key)
            if isinstance(value, bool) or not isinstance(value, Integral):
                continue
            numeric = int(value)
            if numeric >= 0:
                result[key] = numeric
        return result

    def _transport_stats_delta(self) -> dict[str, int]:
        current = self._transport_stats_snapshot()
        return {
            key: max(0, current.get(key, 0) - self._transport_stats_start.get(key, 0))
            for key in RERANKER_TRANSPORT_STAT_KEYS
            if key in current or key in self._transport_stats_start
        }

    def _canonical_ids(self, memory_ids: Iterable[str]) -> tuple[str, ...]:
        if isinstance(memory_ids, (str, bytes)):
            raise ValueError("memory IDs must be an iterable of IDs, not a string")
        try:
            supplied = list(memory_ids)
        except TypeError as exc:
            raise ValueError("memory IDs must be iterable") from exc
        normalized: set[str] = set()
        for raw_identifier in supplied:
            if not isinstance(raw_identifier, str) or not raw_identifier:
                raise ValueError("memory IDs must be non-empty strings")
            if raw_identifier not in self.records:
                raise KeyError(f"unknown visible memory ID: {raw_identifier}")
            normalized.add(raw_identifier)
        return tuple(
            sorted(
                normalized,
                key=lambda identifier: (
                    self.records[identifier].timestamp,
                    self.records[identifier].memory_id,
                ),
            )
        )

    def canonical_ids(self, memory_ids: Iterable[str]) -> tuple[str, ...]:
        return self._canonical_ids(memory_ids)

    def serialize_set(self, memory_ids: Iterable[str]) -> str:
        ids = self._canonical_ids(memory_ids)
        if not ids:
            return EMPTY_SET_SERIALIZATION
        blocks: list[str] = []
        for identifier in ids:
            record = self.records[identifier]
            metadata = record.metadata
            roles = metadata.get("roles", []) if isinstance(metadata, Mapping) else []
            if isinstance(roles, str):
                roles = [roles]
            time_metadata = metadata.get("time", {}) if isinstance(metadata, Mapping) else {}
            source_indices = (
                metadata.get("source_message_indices", []) if isinstance(metadata, Mapping) else []
            )
            header = {
                "memory_id": record.memory_id,
                "roles": _jsonable(roles, f"record {identifier} roles"),
                "source_id": record.source_id,
                "source_message_indices": _jsonable(
                    source_indices, f"record {identifier} source_message_indices"
                ),
                "time": _jsonable(time_metadata, f"record {identifier} time"),
                "timestamp": record.timestamp,
            }
            blocks.append(
                "[Memory "
                + json.dumps(header, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "]\n"
                + record.text
            )
        return "Personal memories:\n" + "\n\n".join(blocks)

    # Concise alias used in mathematical tests and audit tooling.
    serialize = serialize_set

    def _prepare(self, memory_ids: Iterable[str]) -> PreparedSet:
        ids = self._canonical_ids(memory_ids)
        cached = self._prepared.get(ids)
        if cached is not None:
            return cached
        document = self.serialize_set(ids)
        token_count = estimate_tokens(self.query) + estimate_tokens(document)
        set_payload = json.dumps(
            {"namespace": self.namespace_hash, "ids": list(ids), "document": document},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        cache_key = hashlib.sha256(set_payload.encode("utf-8")).hexdigest()
        prepared = PreparedSet(ids, document, token_count, cache_key)
        self._prepared[ids] = prepared
        return prepared

    def prepare_set(self, memory_ids: Iterable[str]) -> PreparedSet:
        return self._prepare(memory_ids)

    def estimated_input_tokens(self, memory_ids: Iterable[str]) -> int:
        return self._prepare(memory_ids).estimated_input_tokens

    def feasible(self, memory_ids: Iterable[str]) -> bool:
        """Whether the set fits the frozen *estimated* experiment input budget.

        This is not a physical tokenizer check. The reranker service validates
        the complete template and reserved output against its runtime capacity
        before scoring. An over-capacity input fails explicitly, without
        truncation or silently changing the search's feasibility rule.
        """

        return self._prepare(memory_ids).estimated_input_tokens <= self.max_input_tokens

    input_feasible = feasible

    def _preflight_prepared(self, prepared: Sequence[PreparedSet]) -> ScorePreflight:
        for item in prepared:
            if item.estimated_input_tokens > self.max_input_tokens:
                raise SetInputTooLong(
                    item.ids,
                    estimated_tokens=item.estimated_input_tokens,
                    limit=self.max_input_tokens,
                )
        distinct: dict[str, PreparedSet] = {}
        for item in prepared:
            distinct.setdefault(item.cache_key, item)
        new_keys = [key for key in distinct if key not in self._logical_seen]
        remaining = self.remaining_budget
        if remaining is not None and len(new_keys) > remaining:
            raise SetBudgetExceeded(
                required=len(new_keys),
                remaining=remaining,
                limit=self._set_budget,
            )
        return ScorePreflight(
            requested_sets=len(prepared),
            distinct_sets=len(distinct),
            new_unique_sets=len(new_keys),
            remaining_before=remaining,
            budget_limit=self._set_budget,
            max_estimated_input_tokens=max(
                (item.estimated_input_tokens for item in prepared), default=0
            ),
            prepared=tuple(prepared),
        )

    def preflight(self, sets: Sequence[Iterable[str]]) -> ScorePreflight:
        if isinstance(sets, (str, bytes)) or not isinstance(sets, Sequence):
            raise ValueError("sets must be a sequence of memory-ID iterables")
        return self._preflight_prepared([self._prepare(ids) for ids in sets])

    preflight_sets = preflight

    def can_score(self, memory_ids: Iterable[str]) -> bool:
        try:
            self._preflight_prepared([self._prepare(memory_ids)])
        except (SetBudgetExceeded, SetInputTooLong):
            return False
        return True

    def can_score_sets(self, sets: Sequence[Iterable[str]]) -> bool:
        try:
            self.preflight(sets)
        except (SetBudgetExceeded, SetInputTooLong):
            return False
        return True

    @property
    def scored_sets(self) -> int:
        return len(self._logical_seen)

    def measured_sets_snapshot(self) -> tuple[dict[str, Any], ...]:
        """Return actually available scores, including a partially failed batch.

        Logical charging precedes transport and therefore is deliberately not
        evidence that a score exists.  This public view exposes only completed
        scores, including persistent-cache hits, without performing I/O or
        changing the task's logical budget.  A caller must still require all
        four members before computing an activation.
        """
        return tuple(
            {
                "ids": list(prepared.ids),
                "score": self._scores[prepared.cache_key],
                "cache_key": prepared.cache_key,
                "estimated_input_tokens": prepared.estimated_input_tokens,
                "objective_semantics": "legacy_query_relevance",
            }
            for prepared in sorted(self._prepared.values(), key=lambda item: (len(item.ids), item.ids))
            if prepared.cache_key in self._logical_seen and prepared.cache_key in self._scores
        )

    @property
    def budget_limit(self) -> int | None:
        return self._set_budget

    @property
    def max_scored_sets(self) -> int | None:
        return self._set_budget

    @property
    def remaining_budget(self) -> int | None:
        if self._set_budget is None:
            return None
        return self._set_budget - self.scored_sets

    def set_budget(self, limit: int | None) -> None:
        """Set a cumulative logical-set cap without invalidating consumption."""

        normalized = None if limit is None else _strict_int(limit, "set budget", nonnegative=True)
        if normalized is not None and normalized < self.scored_sets:
            raise ValueError(
                f"set budget cannot be below already scored sets ({self.scored_sets})"
            )
        self._set_budget = normalized

    set_budget_limit = set_budget
    update_set_budget = set_budget

    def _cache_path(self, key: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / self.namespace_hash[:2] / self.namespace_hash / f"{key}.json"

    def _validate_score(self, value: Any, name: str) -> float:
        try:
            score = _strict_float(value, name)
        except ValueError as exc:
            raise SetResponseError(str(exc)) from exc
        if self.score_space == "unit_interval" and not 0.0 <= score <= 1.0:
            raise SetResponseError("unit-interval reranker score is outside [0, 1]")
        return score

    def _load_persistent(self, prepared: PreparedSet) -> float | None:
        path = self._cache_path(prepared.cache_key)
        if path is None or not path.is_file():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SetCacheError(f"cannot read set score cache entry: {path}") from exc
        if not isinstance(raw, Mapping):
            raise SetCacheError(f"set score cache entry is not an object: {path}")
        expected = {
            "schema": 1,
            "namespace_hash": self.namespace_hash,
            "cache_key": prepared.cache_key,
            "score_space": self.score_space,
            "score_contract": self.score_contract,
        }
        if any(raw.get(key) != value for key, value in expected.items()):
            raise SetCacheError(f"set score cache identity mismatch: {path}")
        try:
            score = self._validate_score(raw.get("score"), "cached set score")
        except SetResponseError as exc:
            raise SetCacheError(str(exc)) from exc
        return score

    def _store_persistent(self, prepared: PreparedSet, score: float) -> None:
        path = self._cache_path(prepared.cache_key)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": 1,
            "namespace_hash": self.namespace_hash,
            "cache_key": prepared.cache_key,
            "score_space": self.score_space,
            "score_contract": self.score_contract,
            "ids": list(prepared.ids),
            "score": score,
        }
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
        )
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(path)
        finally:
            try:
                if temporary.exists():
                    temporary.unlink()
            except OSError:
                pass

    def score_sets(self, sets: Sequence[Iterable[str]], *, reason: str = "search") -> list[float]:
        """Score sets in caller order after one all-or-nothing preflight.

        The backend may sort a batch by score.  Returned ``index`` values are
        therefore validated and used to restore document order before scores
        are associated with canonical set keys.
        """

        if not isinstance(reason, str) or not reason:
            raise ValueError("score reason must be a non-empty string")
        report = self.preflight(sets)
        prepared = list(report.prepared)
        distinct: dict[str, PreparedSet] = {}
        for item in prepared:
            distinct.setdefault(item.cache_key, item)

        # Consume logical quota before cache/network work.  A warm run and a
        # cold run consequently follow the same frontier, including failures.
        for key, item in distinct.items():
            if key not in self._logical_seen:
                self._logical_seen.add(key)
                self._logical_input_tokens += item.estimated_input_tokens

        missing: list[PreparedSet] = []
        source_by_key: dict[str, str] = {}
        for key, item in distinct.items():
            if key in self._scores:
                self._memory_cache_hits += 1
                source_by_key[key] = "memory_cache"
                observe("scoring", "cache_lookup", ids=item.ids, cache_key=key,
                        reason=reason, source="memory_cache", cache_hit=True,
                        score=self._scores[key], estimated_input_tokens=item.estimated_input_tokens)
                continue
            cached_score = self._load_persistent(item)
            if cached_score is not None:
                self._scores[key] = cached_score
                self._persistent_cache_hits += 1
                source_by_key[key] = "persistent_cache"
                observe("scoring", "cache_lookup", ids=item.ids, cache_key=key,
                        reason=reason, source="persistent_cache", cache_hit=True,
                        score=cached_score, estimated_input_tokens=item.estimated_input_tokens)
            else:
                missing.append(item)
                observe("scoring", "cache_lookup", ids=item.ids, cache_key=key,
                        reason=reason, source="cache_miss", cache_hit=False,
                        estimated_input_tokens=item.estimated_input_tokens)

        for offset in range(0, len(missing), self.batch_size):
            batch = missing[offset : offset + self.batch_size]
            documents = [item.document for item in batch]
            self._client_requests += 1
            self._client_samples += len(batch)
            observe("scoring", "score_batch_started", reason=reason,
                    batch_index=offset // self.batch_size, batch_documents=len(batch),
                    logical_call_index=self._client_requests)
            started = time.perf_counter()
            try:
                config = getattr(self.reranker, "config", None)
                payload = build_rerank_payload(config, self.query, documents, len(documents))
                descriptors = document_descriptors(
                    documents, set_ids=[item.ids for item in batch],
                    estimated_tokens=[item.estimated_input_tokens for item in batch],
                )
                audit_stage = {
                    "selection": "selection", "selection_round": "selection",
                    "activation": "dependency_search", "search": "dependency_search",
                    "dense_rerank": "dense_rerank", "diagnostic_fresh_all": "score",
                }.get(reason, current_audit_scope().metadata.get("stage", "score"))
                with request_audit_scope({"stage": audit_stage, "score_reason": reason,
                                          "objective_semantics": "legacy_query_relevance",
                                          "utility_validation_id": None}):
                    with logical_request_scope("reranker", payload, deployment_fingerprint(config), documents=descriptors):
                        scores = _invoke_reranker(
                            self.reranker, self.query, documents, self.score_space
                        )
            finally:
                self._client_elapsed_ms += (time.perf_counter() - started) * 1000.0
            for item, score in zip(batch, scores):
                validated = self._validate_score(score, "set score")
                self._scores[item.cache_key] = validated
                source_by_key[item.cache_key] = "reranker"
                self._store_persistent(item, validated)
                observe("scoring", "set_score_available", reason=reason, ids=item.ids,
                        cache_key=item.cache_key, score=validated, source="reranker",
                        estimated_input_tokens=item.estimated_input_tokens,
                        token_count_is_estimate=True, objective_semantics="legacy_query_relevance",
                        score_contract=self.score_contract, score_space=self.score_space,
                        utility_validation_id=None)
            observe("scoring", "score_batch_completed", reason=reason,
                    batch_index=offset // self.batch_size, batch_documents=len(batch),
                    logical_call_index=self._client_requests)

        scores_in_order = [self._scores[item.cache_key] for item in prepared]
        for key, item in distinct.items():
            self.events.append(
                {
                    "event": "set_score",
                    "reason": reason,
                    "ids": list(item.ids),
                    "cache_key": key,
                    "score": self._scores[key],
                    "source": source_by_key.get(key, "memory_cache"),
                    "estimated_input_tokens": item.estimated_input_tokens,
                    "token_count_is_estimate": True,
                    "objective_semantics": "legacy_query_relevance",
                    "utility_validation_id": None,
                }
            )
            observe("scoring", self.events[-1])
        return scores_in_order

    score_many = score_sets

    def score_set(self, memory_ids: Iterable[str], *, reason: str = "search") -> float:
        return self.score_sets([memory_ids], reason=reason)[0]

    score = score_set

    def score_activation(
        self,
        target_id: str,
        premise_ids: Iterable[str] = (),
        group_ids: Iterable[str] = (),
        *,
        reason: str = "activation",
    ) -> ActivationScores:
        premises = self._canonical_ids(premise_ids)
        group = self._canonical_ids(group_ids)
        if not isinstance(target_id, str) or target_id not in self.records:
            raise KeyError(f"unknown visible target memory ID: {target_id}")
        if not group:
            raise ValueError("activation group cannot be empty")
        if target_id in premises or target_id in group:
            raise ValueError("target cannot also be a premise or group member")
        if set(premises) & set(group):
            raise ValueError("activation group cannot contain an existing premise")
        pe = (*premises, target_id)
        pg = (*premises, *group)
        pge = (*premises, *group, target_id)
        p_score, pe_score, pg_score, pge_score = self.score_sets(
            [premises, pe, pg, pge], reason=reason
        )
        return ActivationScores(
            premise_ids=premises,
            target_id=target_id,
            group_ids=group,
            p=p_score,
            pe=pe_score,
            pg=pg_score,
            pge=pge_score,
        )

    activation = score_activation

    @property
    def reranker_adapter_requests(self) -> int:
        return self._client_requests

    @property
    def client_requests(self) -> int:
        return self._client_requests

    @property
    def cache_hits(self) -> int:
        return self._memory_cache_hits + self._persistent_cache_hits

    @property
    def persistent_cache_hits(self) -> int:
        return self._persistent_cache_hits

    @property
    def memory_cache_hits(self) -> int:
        return self._memory_cache_hits

    @property
    def cost(self) -> dict[str, Any]:
        return {
            "objective_semantics": "legacy_query_relevance",
            "utility_validation_id": None,
            "scored_sets": self.scored_sets,
            "set_budget": self._set_budget,
            "remaining_set_budget": self.remaining_budget,
            "reranker_adapter_requests": self._client_requests,
            "reranker_samples": self._client_samples,
            "persistent_cache_hits": self._persistent_cache_hits,
            "memory_cache_hits": self._memory_cache_hits,
            "cache_hits": self.cache_hits,
            "reranker_elapsed_ms": self._client_elapsed_ms,
            # Optional physical-transport accounting from RerankerClient.
            # Logical set/search budgets above are deliberately unchanged by
            # HTTP retries or pointwise batch subdivision.
            "reranker_transport": self._transport_stats_delta(),
            "logical_input_tokens_estimate": self._logical_input_tokens,
            "token_count_is_estimate": True,
        }

    def cost_dict(self) -> dict[str, Any]:
        return dict(self.cost)


# Naming used by search code that talks about a generic scorer.
SetScorer = SetReranker


def probe_pointwise_consistency(
    reranker: Any,
    query: str,
    documents: Sequence[str],
    *,
    score_space: str | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-6,
    raise_on_mismatch: bool = True,
) -> PointwiseProtocolReport:
    """Compare singleton, mixed, and reversed batches outside search budgets."""

    if not isinstance(query, str) or not query:
        raise ValueError("protocol probe query must be a non-empty string")
    if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence):
        raise ValueError("protocol probe documents must be a sequence")
    if len(documents) < 2 or any(not isinstance(document, str) for document in documents):
        raise ValueError("protocol probe requires at least two string documents")
    if isinstance(rtol, bool) or isinstance(atol, bool):
        raise ValueError("protocol tolerances must be numeric")
    rtol_value = _strict_float(rtol, "protocol rtol")
    atol_value = _strict_float(atol, "protocol atol")
    if rtol_value < 0.0 or atol_value < 0.0:
        raise ValueError("protocol tolerances must be non-negative")
    config = getattr(reranker, "config", None)
    contract = str(
        getattr(reranker, "score_contract", getattr(config, "score_contract", "pointwise"))
    ).strip().lower()
    if contract != "pointwise":
        raise ValueError("dependency set scoring requires a pointwise reranker contract")
    space = _canonical_score_space(
        score_space
        if score_space is not None
        else getattr(reranker, "score_space", getattr(config, "score_space", "unit_interval"))
    )

    probe_documents = [EMPTY_SET_SERIALIZATION, documents[0], documents[1]]
    reference = [
        _invoke_reranker(reranker, query, [document], space)[0]
        for document in probe_documents
    ]
    mixed = _invoke_reranker(reranker, query, probe_documents, space)
    reversed_scores = _invoke_reranker(reranker, query, list(reversed(probe_documents)), space)
    restored_reverse = list(reversed(reversed_scores))
    comparisons = list(zip(reference, mixed)) + list(zip(reference, restored_reverse))
    absolute = [abs(left - right) for left, right in comparisons]
    relative = [
        abs(left - right) / max(abs(left), abs(right), np.finfo(float).tiny)
        for left, right in comparisons
    ]
    consistent = all(
        math.isclose(left, right, rel_tol=rtol_value, abs_tol=atol_value)
        for left, right in comparisons
    )
    report = PointwiseProtocolReport(
        consistent=consistent,
        score_space=space,
        score_contract=contract,
        rtol=rtol_value,
        atol=atol_value,
        compared_values=len(comparisons),
        max_absolute_deviation=max(absolute, default=0.0),
        max_relative_deviation=max(relative, default=0.0),
        adapter_requests=5,
    )
    if not consistent and raise_on_mismatch:
        raise SetResponseError(
            "reranker pointwise scores depend on batch composition/order: "
            f"max_abs={report.max_absolute_deviation}, "
            f"max_rel={report.max_relative_deviation}"
        )
    return report


validate_pointwise_protocol = probe_pointwise_consistency
probe_score_protocol = probe_pointwise_consistency


__all__ = [
    "EMPTY_SET_SERIALIZATION",
    "SET_SERIALIZATION_TEMPLATE_VERSION",
    "ActivationScores",
    "CanonicalMemoryRecord",
    "InputCapacityError",
    "PointwiseProtocolReport",
    "PreparedSet",
    "RerankerProtocolError",
    "ScorePreflight",
    "ScoringBudgetError",
    "SetBudgetExceeded",
    "SetCacheError",
    "SetInputTooLong",
    "SetResponseError",
    "SetReranker",
    "SetScorer",
    "SetScoringError",
    "probe_pointwise_consistency",
    "probe_score_protocol",
    "validate_pointwise_protocol",
]
