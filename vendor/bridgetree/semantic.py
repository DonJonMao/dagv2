"""Semantic-path v1 execution.

The historical :mod:`bridgetree.retriever` implementation is intentionally
kept for strict legacy profiles.  This module supplies the revised execution
chain: discover a finite proposal DAG, freeze it, compute sparse path
provenance, score every candidate with a declared pointwise quality contract,
and only then run a fixed PSD selector.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .budget import CostTracker, SearchBudget, _strict_int
from .config import RetrievalConfig
from .index import ExactInnerProductIndex, build_index
from .information import (
    FrozenListwiseSelector,
    PureRerankSelector,
    QualityRecord,
    SemanticFeatureProvider,
    SemanticPathLogDetSelector,
    validate_quality_records,
)
from .math_utils import nonnegative_cosine, normalize, normalize_rows
from .measure import (
    angular_navigation_affinity,
    display_path_hypotheses,
    propagate_frozen_graph,
    shuffle_frozen_graph,
)
from .ranking import (
    DEFAULT_FINAL_RERANK_INSTRUCTION,
    build_bridge_embedding_text,
    format_memory_document,
)
from .temporal import build_time_mark, query_time_mark
from .types import (
    ContextPlan,
    FrozenProposalGraph,
    InformationAtom,
    Memory,
    PathHypothesis,
    RetrievalResult,
    SelectionStep,
    TemporalMark,
    TreeNode,
)

SEMANTIC_PROFILE = "semantic_path_v1"

# The task-conditioned representation is a separately identified operator,
# rather than an implicit variation of the cached memory embedding.  Keep the
# instruction stable so it participates in both the embedding request and its
# content-addressed cache key.
SEMANTIC_QUERY_CONDITIONED_INSTRUCTION = (
    "Encode the recorded personal evidence in the context of the current task. "
    "Preserve distinctions between user statements and assistant suggestions, "
    "and between historical and current evidence."
)

_SENSITIVE_METADATA_KEYS = {
    "answer",
    "correct_answer",
    "gold",
    "gold_answer",
    "label",
    "target",
    "solution",
    "oracle",
}


def _safe_semantic_metadata(value: Any, *, key: str = "") -> Any:
    """Remove answer-bearing metadata before it reaches a representation.

    PersonaMem's public question fields are already separated from the gold
    answer, but programmatic callers may attach arbitrary metadata.  The
    query-conditioned encoder must never receive a label/oracle field merely
    because it happened to be nested in that metadata.
    """

    normalized_key = str(key).strip().lower().replace("-", "_")
    if normalized_key in _SENSITIVE_METADATA_KEYS or any(
        token in normalized_key for token in ("gold", "answer", "label", "target", "oracle")
    ):
        return "[redacted]"
    if isinstance(value, Mapping):
        return {
            str(child_key): _safe_semantic_metadata(child_value, key=str(child_key))
            for child_key, child_value in sorted(value.items(), key=lambda item: str(item[0]))
            if str(child_key).strip().lower().replace("-", "_") not in _SENSITIVE_METADATA_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_safe_semantic_metadata(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def build_query_conditioned_representation_text(
    query: str,
    answer_options: str,
    memory: Memory,
    *,
    query_cutoff: Any = None,
    query_metadata: Mapping[str, Any] | None = None,
    embedding_model: str = "",
    embedding_fingerprint: str = "",
    instruction: str = SEMANTIC_QUERY_CONDITIONED_INSTRUCTION,
    cache_schema: str = "semantic_query_conditioned_v1",
) -> str:
    """Build the frozen S3 representation input.

    Every field that can change the task-conditioned vector is rendered in a
    deterministic envelope: task/options, original memory text, role and time
    provenance, cutoff, and model/instruction/cache identity.  The gold answer
    is intentionally not an accepted argument and answer-bearing metadata is
    redacted for defensive protocol hygiene.
    """

    if not isinstance(query, str) or not isinstance(answer_options, str):
        raise ValueError("query and answer_options must be strings")
    if not isinstance(memory, Memory):
        raise ValueError("query-conditioned representation requires a Memory")
    safe_query_metadata = _safe_semantic_metadata(dict(query_metadata or {}))
    safe_memory_metadata = _safe_semantic_metadata(memory.metadata if isinstance(memory.metadata, Mapping) else {})
    # ``_memory_mark`` validates the observation envelope used by the semantic
    # path.  Reusing its public form keeps the representation and provenance
    # schemas in lockstep.
    time_mark = _memory_mark(memory).public_dict()
    roles = safe_memory_metadata.get("roles", []) if isinstance(safe_memory_metadata, Mapping) else []
    payload = {
        "schema": str(cache_schema),
        "embedding_model": str(embedding_model),
        "embedding_fingerprint": str(embedding_fingerprint),
        "instruction": str(instruction),
        "query_cutoff": _safe_semantic_metadata(query_cutoff),
        "query_metadata": safe_query_metadata,
        "memory_id": str(memory.memory_id),
        "source_id": str(memory.source_id),
        "roles": roles,
        "time": time_mark,
    }
    metadata_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return (
        f"[Representation protocol: {cache_schema}]\n"
        f"[Embedding model: {embedding_model}]\n"
        f"[Embedding fingerprint: {embedding_fingerprint}]\n"
        f"[Representation instruction: {instruction}]\n"
        f"[Representation metadata: {metadata_json}]\n\n"
        f"Task:\n{query}\n\n"
        f"Options:\n{answer_options}\n\n"
        f"Memory:\n{memory.text}"
    )


def rho_squared_quality_records(
    measure: Any,
    *,
    legacy_rho: Mapping[str, float] | None = None,
    scorer_fingerprint: str = "legacy-rho2",
) -> dict[str, QualityRecord]:
    """Return the old executor's explicit ``legacy_rho[j] ** 2`` table.

    ``legacy_rho`` is intentionally separate from the frozen DAG's random
    path access quality ``h``.  A frozen graph measure alone is not an old
    executor trace, so callers that do not provide an explicit trace are
    rejected instead of silently relabelling ``h`` as legacy ``rho``.
    """

    graph = getattr(measure, "graph", None)
    if graph is None:
        raise ValueError("rho² quality requires a frozen graph measure")
    if legacy_rho is None:
        legacy_rho = getattr(measure, "legacy_rho", None)
    if not isinstance(legacy_rho, Mapping):
        raise ValueError(
            "rho² quality requires an explicit legacy_rho mapping from an old executor trace; "
            "frozen path access quality h is not a legacy trace"
        )
    normalized_rho = {str(key): float(value) for key, value in legacy_rho.items()}
    missing = sorted(set(graph.memory_ids) - set(normalized_rho))
    unknown = sorted(set(normalized_rho) - set(graph.memory_ids))
    if missing or unknown:
        raise ValueError(f"legacy_rho must cover exactly the frozen pool (missing={missing}, unknown={unknown})")
    records: dict[str, QualityRecord] = {}
    for raw_identifier in graph.memory_ids:
        identifier = str(raw_identifier)
        value = normalized_rho[identifier]
        if not np.isfinite(value) or value < -1e-10 or value > 1.0 + 1e-10:
            raise ValueError(f"legacy rho for {identifier} is outside [0, 1]")
        rho = min(1.0, max(0.0, value))
        squared = rho * rho
        input_hash = hashlib.sha256(
            json.dumps(
                {"graph_hash": graph.graph_hash, "memory_id": identifier, "legacy_rho": rho},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        records[identifier] = QualityRecord.from_raw(
            identifier,
            squared,
            "unit_interval",
            scorer_fingerprint=scorer_fingerprint,
            input_hash=input_hash,
        )
    return records


def legacy_trace_from_result(result: Any) -> tuple[dict[str, float], dict[str, str | None]]:
    """Extract the first-arrival rho/parent trace from a legacy result.

    This adapter deliberately reads the old executor's recorded node fields;
    it never recomputes a widest path from the new posterior DAG.
    """
    # Newer legacy runners may persist the trace as two explicit mappings;
    # accept that representation first so a result object does not need to
    # reconstruct it from serialized TreeNode instances.
    explicit = getattr(result, "legacy_trace", None)
    if isinstance(explicit, Mapping):
        raw_rho = explicit.get("rho", explicit.get("legacy_rho"))
        raw_parent = explicit.get("parent_id", explicit.get("legacy_parent_id", {}))
        if isinstance(raw_rho, Mapping) and isinstance(raw_parent, Mapping):
            rho = {str(identifier): float(value) for identifier, value in raw_rho.items()}
            parents = {
                str(identifier): (None if parent is None else str(parent))
                for identifier, parent in raw_parent.items()
            }
            return rho, parents
    raw_rho = getattr(result, "legacy_rho", None)
    raw_parent = getattr(result, "legacy_parent_id", None)
    if isinstance(raw_rho, Mapping) and isinstance(raw_parent, Mapping):
        return (
            {str(identifier): float(value) for identifier, value in raw_rho.items()},
            {
                str(identifier): (None if parent is None else str(parent))
                for identifier, parent in raw_parent.items()
            },
        )
    nodes = getattr(result, "nodes", None)
    if not isinstance(nodes, Mapping) or not nodes:
        raise ValueError("legacy result does not contain a node trace")
    rho: dict[str, float] = {}
    parents: dict[str, str | None] = {}
    for raw_id, node in nodes.items():
        identifier = str(raw_id)
        value = getattr(node, "reachability", None)
        if value is None:
            raise ValueError(f"legacy node {identifier} has no reachability")
        rho[identifier] = float(value)
        parent = getattr(node, "parent_id", None)
        parents[identifier] = None if parent is None else str(parent)
    return rho, parents


@dataclass(frozen=True)
class ProposalExposure:
    parent_id: str
    candidate_id: str
    layer: int
    rank: int
    score: float


def _memory_mark(memory: Memory) -> TemporalMark:
    """Decode one memory's temporal envelope without silently repairing it.

    Observation indices are protocol integers.  A malformed annotation is a
    data/provenance error and must be surfaced to the caller; falling back to a
    truncated timestamp would make two different records indistinguishable in
    a confirmatory run.
    """
    if not isinstance(memory, Memory):
        raise ValueError("memory temporal mark requires a Memory record")
    metadata = memory.metadata if isinstance(memory.metadata, Mapping) else {}
    if "time" in metadata:
        raw = metadata["time"]
        if not isinstance(raw, Mapping):
            raise ValueError("memory time metadata must be a mapping")
    else:
        # Accept the flat temporal envelope used by a few programmatic
        # callers in addition to PersonaMem's nested ``time`` field.
        raw = metadata

    def observed_integer(value: Any, name: str) -> int:
        # ``TemporalMark`` itself owns the final type, but doing the strict
        # check here prevents ``int(1.5)`` and booleans from being accepted.
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"memory {name} must be an integer")
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"memory {name} must be an integer") from exc
        if not np.isfinite(numeric) or numeric != float(int(numeric)):
            raise ValueError(f"memory {name} must be an integer")
        return int(numeric)

    observed_start = observed_integer(raw.get("observed_start", memory.timestamp), "observed_start")
    observed_end = observed_integer(raw.get("observed_end", memory.timestamp), "observed_end")
    event_start = raw.get("event_start", raw.get("start"))
    event_end = raw.get("event_end", raw.get("end"))

    def event_value(value: Any, name: str) -> str | None:
        if value is None:
            return None
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"memory {name} must be a date-like value or None")
        if isinstance(value, str):
            if not value.strip():
                raise ValueError(f"memory {name} must be a non-empty value")
            return value
        # Numeric event epochs are emitted by ``messages_to_memories`` for
        # explicitly annotated numeric dates.  The semantic provenance type
        # stores event values textually, so canonicalize them rather than
        # rejecting a valid annotation at the retrieval boundary.
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"memory {name} must be a date-like value or None") from exc
        if not np.isfinite(numeric):
            raise ValueError(f"memory {name} must be finite when present")
        return str(numeric)

    event_start = event_value(event_start, "event_start")
    event_end = event_value(event_end, "event_end")
    validity = raw.get(
        "validity",
        "instant" if event_start is not None or event_end is not None else "unknown",
    )
    source = raw.get(
        "time_source",
        "explicit" if event_start is not None or event_end is not None else "message_index",
    )
    if not isinstance(validity, str) or not validity.strip():
        raise ValueError("memory time validity must be a non-empty string")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("memory time_source must be a non-empty string")
    try:
        return TemporalMark(
            observed_start,
            observed_end,
            event_start,
            event_end,
            validity,
            source,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid temporal metadata for memory {memory.memory_id}") from exc


def _visible_records(
    memories: Sequence[Memory],
    memory_vectors: np.ndarray,
    query_cutoff: Any = None,
    query_metadata: Mapping[str, Any] | None = None,
) -> tuple[list[Memory], np.ndarray]:
    """Apply a query visibility envelope without inventing calendar time.

    Numeric cutoffs are PersonaMem observation indices and therefore compare
    against ``observed_end``.  Date-like/structured cutoffs compare against an
    explicitly annotated event endpoint when one exists.  A memory with no
    comparable event annotation is retained: it is already inside the source
    observation envelope and an unknown date must not be silently interpreted
    as either a future or a past event.
    """

    if len(memories) != len(memory_vectors):
        raise ValueError("memories and memory_vectors must have equal length")
    if query_cutoff is None and query_metadata is None:
        return list(memories), np.asarray(memory_vectors)

    # Keep a scalar numeric cutoff on the observation scale.  This branch is
    # intentionally checked before the generic temporal parser, whose
    # ``TimeMark`` representation carries both observed and event endpoints.
    # Query metadata often contains ordinary task fields (and PersonaMem
    # stores ``query_time`` there as a string); those fields must not turn an
    # observation-index cutoff into an event-date cutoff and thereby admit
    # future records.
    numeric_cutoff: float | None = None
    if not isinstance(query_cutoff, (Mapping, tuple, list)):
        try:
            if not isinstance(query_cutoff, (bool, np.bool_)):
                candidate = float(query_cutoff)
                if np.isfinite(candidate):
                    numeric_cutoff = candidate
        except (TypeError, ValueError):
            numeric_cutoff = None

    def has_explicit_event_fields(value: Any) -> bool:
        if not isinstance(value, Mapping):
            return False
        raw_value: Any = value
        if isinstance(raw_value.get("metadata"), Mapping):
            raw_value = {**dict(raw_value.get("metadata", {})), **dict(raw_value)}
        nested_time = raw_value.get("time")
        if isinstance(nested_time, Mapping):
            raw_value = {**dict(raw_value), **dict(nested_time)}
        event_keys = {
            "event_start",
            "event_end",
            "event_time",
            "date",
            "datetime",
            "query_date",
        }
        if any(key in raw_value and raw_value.get(key) is not None for key in event_keys):
            return True
        # A non-numeric scalar under ``time`` is an explicit date-like value;
        # numeric ``query_time``/``cutoff`` values remain observation indices.
        scalar_time = raw_value.get("time")
        if scalar_time is not None and not isinstance(scalar_time, Mapping):
            try:
                float(scalar_time)
            except (TypeError, ValueError):
                return True
        return False

    def metadata_observation_cutoff(value: Any) -> float | None:
        if not isinstance(value, Mapping):
            return None
        raw_value: Any = value
        if isinstance(raw_value.get("metadata"), Mapping):
            raw_value = {**dict(raw_value.get("metadata", {})), **dict(raw_value)}
        # PersonaMem's ``query_time`` and generic ``cutoff`` aliases denote
        # an observed message boundary when they are numeric.  Date-like
        # strings are left to ``query_time_mark`` as explicit event values.
        for key in ("query_time", "cutoff"):
            candidate = raw_value.get(key)
            if candidate is None or isinstance(candidate, (bool, np.bool_)):
                continue
            try:
                numeric = float(candidate)
            except (TypeError, ValueError, OverflowError):
                continue
            if np.isfinite(numeric):
                return numeric
        nested_time = raw_value.get("time")
        if isinstance(nested_time, Mapping) and not any(
            nested_time.get(key) is not None
            for key in ("event_start", "event_end", "event_time", "date", "datetime")
        ):
            candidate = nested_time.get("observed_end", nested_time.get("observed"))
            if isinstance(candidate, Mapping):
                candidate = candidate.get("observed_end", candidate.get("end"))
            try:
                numeric = float(candidate)
            except (TypeError, ValueError, OverflowError):
                return None
            if np.isfinite(numeric):
                return numeric
        return None

    explicit_event_metadata = has_explicit_event_fields(query_metadata)
    if numeric_cutoff is None and query_cutoff is None and not explicit_event_metadata:
        numeric_cutoff = metadata_observation_cutoff(query_metadata)
    query_mark = query_time_mark(query_cutoff, query_metadata)
    if numeric_cutoff is not None and not explicit_event_metadata:
        cutoff_end = numeric_cutoff
        explicit_event_cutoff = False
    else:
        if numeric_cutoff is None and query_mark.unavailable:
            # An opaque/date-less cutoff does not provide a safe comparison key.
            return list(memories), np.asarray(memory_vectors)
        cutoff_end = query_mark.end
        explicit_event_cutoff = (
            query_mark.event_start is not None or query_mark.event_end is not None
        )
    if cutoff_end is None:
        # An opaque/date-less cutoff does not provide a safe comparison key.
        return list(memories), np.asarray(memory_vectors)
    keep: list[int] = []
    for index, memory in enumerate(memories):
        mark = _memory_mark(memory)
        if explicit_event_cutoff:
            # ``build_time_mark`` canonicalizes ISO dates and numeric event
            # annotations.  Unknown event time remains visible rather than
            # being guessed from an observation index.
            event_mark = build_time_mark(memory.metadata, timestamp=memory.timestamp)
            event_end = event_mark.end if not event_mark.unavailable else None
            if event_end is None or float(event_end) <= float(cutoff_end):
                keep.append(index)
        elif float(mark.observed_end) <= float(cutoff_end):
            keep.append(index)
    return [memories[index] for index in keep], np.asarray(memory_vectors)[keep]


def _call_index_search(
    tracker: CostTracker,
    index: ExactInnerProductIndex,
    query: np.ndarray,
    top_k: int,
    exclude: set[str],
):
    method = getattr(tracker, "search_proposal", None)
    if callable(method):
        hits = method(index, query, top_k, exclude=exclude)
    else:
        # Compatibility with a custom tracker supplied by an older integration.
        hits = tracker.search_core(index, query, top_k, exclude=exclude)
    try:
        normalized_hits = list(hits)
    except TypeError as exc:
        raise ValueError("proposal index must return an iterable of (memory_id, score) pairs") from exc
    if len(normalized_hits) > int(top_k):
        # A backend returning more rows than requested invalidates both the
        # advertised ANN cost and the deterministic proposal width.  Do not
        # silently slice it: callers need to fix the adapter or its accounting.
        raise ValueError("index returned more candidates than requested")
    checked: list[tuple[str, float]] = []
    for item in normalized_hits:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError("proposal index returned a malformed candidate")
        raw_identifier, raw_score = item
        if isinstance(raw_identifier, (bool, np.bool_)):
            raise ValueError("proposal index candidate IDs must be non-empty strings")
        identifier = str(raw_identifier)
        if not identifier:
            raise ValueError("proposal index candidate IDs must be non-empty strings")
        if isinstance(raw_score, (bool, np.bool_)):
            raise ValueError("proposal index scores must be finite numbers")
        try:
            score = float(raw_score)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("proposal index scores must be finite numbers") from exc
        if not np.isfinite(score):
            raise ValueError("proposal index scores must be finite numbers")
        checked.append((identifier, score))
    return checked


def _strict_positive_int(value: Any, name: str) -> int:
    return _strict_int(value, name, positive=True)


def _normalize_initial_hits(
    hits: Sequence[tuple[str, float]] | Any,
) -> list[tuple[str, float]]:
    """Validate an injected ANN response without deduplicating or slicing it."""

    if isinstance(hits, (str, bytes)):
        raise ValueError("initial_hits must be a sequence of candidate pairs")
    try:
        raw = list(hits)
    except TypeError as exc:
        raise ValueError("initial_hits must be a sequence of candidate pairs") from exc
    normalized: list[tuple[str, float]] = []
    for item in raw:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError("initial_hits must contain (memory_id, score) pairs")
        raw_identifier, raw_score = item
        if isinstance(raw_identifier, (bool, np.bool_)):
            raise ValueError("initial_hits candidate IDs must be non-empty strings")
        identifier = str(raw_identifier)
        if not identifier:
            raise ValueError("initial_hits candidate IDs must be non-empty strings")
        if isinstance(raw_score, (bool, np.bool_)):
            raise ValueError("initial_hits scores must be finite numbers")
        try:
            score = float(raw_score)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("initial_hits scores must be finite numbers") from exc
        if not np.isfinite(score):
            raise ValueError("initial_hits scores must be finite numbers")
        # Deliberately preserve duplicate IDs and excluded IDs.  They are raw
        # physical exposure and must be charged/audited before admission.
        normalized.append((identifier, score))
    return normalized


def _invoke_proposal_query_provider(
    provider: Any,
    texts: Sequence[str],
    *,
    query: str,
    instruction: str = "",
) -> np.ndarray:
    """Resolve a bridge/query-anchor embedding provider without guessing twice.

    Providers used by the repository expose one of ``encode_queries``,
    ``encode`` or ``encode_query``.  Signature inspection keeps a provider's
    own ``TypeError`` visible instead of retrying it with a different payload.
    The returned rows are validated and normalized by the caller.
    """

    if provider is None:
        raise ValueError("proposal query provider is not configured")
    def invoke(method: Callable[..., Any], positional: tuple[Any, ...], optional: Mapping[str, Any]) -> Any:
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            # Signature inspection is unavailable for a few extension-backed
            # callables.  In that case use the minimal positional contract;
            # importantly, do not catch a TypeError raised by the provider's
            # own body and retry it with a different semantic request.
            return method(*positional)
        parameters = signature.parameters
        accepts_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
        )
        accepted = {
            key: value
            for key, value in optional.items()
            if accepts_var_kwargs or key in parameters
        }
        return method(*positional, **accepted)

    method = getattr(provider, "encode_queries", None)
    if callable(method):
        values = invoke(method, (list(texts),), {"instruction": instruction, "purpose": "bridge_query"})
    else:
        method = getattr(provider, "encode", None)
        if callable(method):
            values = invoke(method, (list(texts),), {"instruction": instruction, "purpose": "bridge_query"})
        else:
            method = getattr(provider, "encode_query", None)
            if callable(method):
                values = [
                    invoke(method, (text,), {"instruction": instruction, "purpose": "bridge_query"})
                    for text in texts
                ]
            elif callable(provider):
                values = invoke(provider, (list(texts),), {"query": query, "instruction": instruction})
            else:
                raise ValueError("proposal query provider must expose encode/encode_queries/encode_query")
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        if len(texts) != 1:
            raise ValueError("proposal query provider returned one vector for multiple anchors")
        matrix = matrix[None, :]
    if matrix.ndim != 2 or matrix.shape[0] != len(texts) or not np.all(np.isfinite(matrix)):
        raise ValueError("proposal query provider returned an invalid vector matrix")
    if matrix.shape[1] == 0 or np.any(np.linalg.norm(matrix, axis=1) <= 1e-12):
        raise ValueError("proposal query provider returned a zero vector")
    return normalize_rows(matrix)


def discover_frozen_graph(
    memories: Sequence[Memory],
    memory_vectors: np.ndarray,
    query_vector: np.ndarray,
    *,
    initial_width: int = 12,
    branch_width: int = 4,
    proposal_width: int | None = None,
    max_depth: int = 2,
    budget: SearchBudget | None = None,
    tracker: CostTracker | None = None,
    index: ExactInnerProductIndex | None = None,
    relation_mode: str = "angular",
    proposal_mode: str = "real_member_query_anchor",
    excluded_candidate_ids: Sequence[str] = (),
    cutoff: Any = None,
    query_metadata: Mapping[str, Any] | None = None,
    initial_hits: Sequence[tuple[str, float]] | None = None,
    initial_hits_accounted: bool = False,
    proposal_query_provider: Any = None,
    proposal_vectors: Mapping[str, np.ndarray] | None = None,
    proposal_query_instruction: str = "",
    query_text: str = "",
) -> tuple[FrozenProposalGraph, dict[str, Any], CostTracker, ExactInnerProductIndex]:
    """Build and freeze a layer-synchronous real-member proposal DAG.

    Every parent in a layer issues its proposal before the next layer is
    admitted.  Candidates are de-duplicated only at admission, so a repeated
    candidate retains all same-layer parent exposures in the graph metadata.
    """

    if proposal_mode not in {
        "real_member_query_anchor",
        "offline_q_plus_anchor",
        "real_member_vector",
        "round_robin",
        "dense",
    }:
        raise ValueError(f"unsupported semantic proposal mode: {proposal_mode}")
    if relation_mode not in {"angular", "cosine"}:
        raise ValueError("semantic relation_mode must be angular or cosine")
    initial_width = _strict_positive_int(initial_width, "initial_width")
    branch_width = _strict_positive_int(branch_width, "branch_width")
    proposal_width = (
        None if proposal_width is None else _strict_positive_int(proposal_width, "proposal_width")
    )
    depth_limit = _strict_positive_int(max_depth, "max_depth")
    # Discovery itself is a visibility boundary.  Callers that pre-freeze a
    # graph (the semantic matrix) must not be able to expose memories beyond
    # the question's observation/event cutoff merely by bypassing
    # ``semantic_retrieve``.
    visible, visible_vectors = _visible_records(
        memories,
        memory_vectors,
        cutoff,
        query_metadata=query_metadata,
    )
    vectors = normalize_rows(np.asarray(visible_vectors, dtype=np.float64)).astype(np.float32)
    if len(visible) != len(vectors):
        raise ValueError("memories and memory_vectors must have equal length")
    ids = [str(memory.memory_id) for memory in visible]
    if len(ids) != len(set(ids)):
        raise ValueError("memory ids must be unique")
    # Graph/index identifiers are canonical strings.  Normalize the lookup
    # namespace as well so numeric IDs from dataframe-style callers remain
    # valid after an ANN result is returned.
    memory_by_id = {str(memory.memory_id): memory for memory in visible}
    search_budget = budget or SearchBudget(
        max_unique_nodes=max(1, min(len(ids), initial_width + branch_width * max(0, depth_limit - 1))),
    )
    tracker = tracker or CostTracker(search_budget)
    index_ids = list(getattr(index, "ids", ())) if index is not None else []
    if index is None or index_ids != ids:
        index = build_index("exact", ids, vectors)
    excluded = {str(value) for value in excluded_candidate_ids}
    width = initial_width
    if initial_hits is None:
        first_hits = _call_index_search(tracker, index, normalize(query_vector), width, excluded)
    else:
        # Treat an injected response exactly like a physical ANN response:
        # validate it, account every raw row, and only then derive the unique
        # anchor set.  Previously this path deduplicated/truncated first,
        # under-reporting both duplicate and over-cap exposure.
        first_hits = _normalize_initial_hits(initial_hits)
        unknown = [identifier for identifier, _score in first_hits if identifier not in memory_by_id]
        if unknown:
            raise ValueError(f"initial_hits contain unknown memory IDs: {unknown}")
        if not initial_hits_accounted:
            tracker.record_candidate_exposure(
                len(first_hits),
                identifiers=[identifier for identifier, _score in first_hits],
            )
    # Admission is a separate operation from exposure.  The returned order is
    # part of the deterministic protocol, but does not affect later members'
    # same-layer visibility because all proposals are collected in a batch.
    anchor_ids: list[str] = []
    for identifier, _score in first_hits:
        if identifier in excluded or identifier in anchor_ids:
            continue
        anchor_ids.append(identifier)
    # Do not truncate the observed first-hop response before admission.  The
    # tracker records every over-cap exposure explicitly while admitting only
    # the deterministic prefix allowed by the unique-node budget.
    # A tracker can already have spent part of the shared budget (for
    # example, an adapter may account an injected first-hop request before
    # calling us).  Only IDs actually present after admission belong to the
    # frozen graph; otherwise the graph could advertise more unique nodes
    # than the cost record permits.
    anchor_candidates = list(anchor_ids)
    tracker.admit_nodes(anchor_candidates)
    anchor_ids = [identifier for identifier in anchor_candidates if identifier in tracker.visited_ids]
    layers: list[tuple[str, ...]] = [tuple(anchor_ids)] if anchor_ids else []
    admitted: set[str] = set(anchor_ids)
    exposures: list[ProposalExposure] = []
    back_edges: list[dict[str, Any]] = []
    parent_sources: dict[str, set[str]] = {identifier: set() for identifier in anchor_ids}
    edge_pairs: set[tuple[str, str]] = set()
    proposal_limit = proposal_width if proposal_width is not None else branch_width

    # ``real_member_query_anchor`` is intentionally a query-conditioned
    # proposal: the anchor remains a real member, while the search vector
    # represents the pair (current request, anchor).  Callers may provide a
    # batch embedding provider or precomputed per-anchor vectors.  In offline
    # mode we use a deterministic normalized q+anchor bridge; this is an
    # explicit mathematical adapter, never mislabeled as a service call.
    proposal_vector_map: dict[str, np.ndarray] = {}
    proposal_embedding_source = "not_applicable"
    proposal_embedding_queries = 0
    proposal_embedding_ms = 0.0
    if proposal_mode in {"real_member_query_anchor", "offline_q_plus_anchor"}:
        if proposal_vectors is not None:
            if not isinstance(proposal_vectors, Mapping):
                raise ValueError("proposal_vectors must be a mapping from anchor ID to vector")
            normalized_keys: set[str] = set()
            for raw_id, raw_vector in proposal_vectors.items():
                identifier = str(raw_id)
                if identifier in normalized_keys:
                    raise ValueError("proposal_vectors contain duplicate anchor IDs")
                normalized_keys.add(identifier)
                if identifier not in memory_by_id:
                    raise ValueError(f"proposal_vectors contain unknown anchor ID: {identifier}")
                value = np.asarray(raw_vector, dtype=np.float64).reshape(-1)
                if value.size == 0 or not np.all(np.isfinite(value)) or np.linalg.norm(value) <= 1e-12:
                    raise ValueError(f"proposal vector for {identifier} is invalid")
                if value.shape != np.asarray(query_vector).reshape(-1).shape:
                    raise ValueError(f"proposal vector for {identifier} has the wrong dimension")
                proposal_vector_map[identifier] = normalize(value)
            proposal_embedding_source = "precomputed"
        elif proposal_mode == "real_member_query_anchor" and proposal_query_provider is not None:
            # Providers are called layer-by-layer below because the set of
            # real members is itself discovered synchronously.  The marker is
            # retained in the graph config for audit/caching identity.
            proposal_embedding_source = "provider"
        else:
            proposal_embedding_source = "deterministic_q_plus_anchor"

    for layer_number in range(1, max(1, depth_limit)):
        current = list(layers[layer_number - 1]) if layer_number - 1 < len(layers) else []
        if not current:
            break
        # The exclusion set is fixed at the start of the layer.  In
        # particular, a candidate exposed by parent A remains visible to
        # parent B, preserving duplicate parent evidence.
        layer_exclude = set(admitted) | excluded
        layer_exposures: list[ProposalExposure] = []
        provider_parent_ids: list[str] = []
        provider_texts: list[str] = []
        for _parent_position, parent_id in enumerate(current):
            if not tracker.can_propose():
                break
            if proposal_mode in {"real_member_query_anchor", "offline_q_plus_anchor"}:
                if parent_id in proposal_vector_map:
                    proposal_query = proposal_vector_map[parent_id]
                elif proposal_mode == "real_member_query_anchor" and proposal_query_provider is not None:
                    provider_parent_ids.append(parent_id)
                    provider_texts.append(build_bridge_embedding_text(str(query_text), memory_by_id[parent_id]))
                    # The actual provider call is batched after collecting the
                    # whole synchronous layer; search happens below.
                    continue
                else:
                    query_unit = normalize(query_vector)
                    anchor_unit = normalize(index.vector(parent_id))
                    bridge = query_unit + anchor_unit
                    # Opposite query/anchor directions have no finite sum;
                    # retaining the query direction is a deterministic,
                    # explicitly recorded fallback rather than an arbitrary
                    # zero vector passed to the ANN backend.
                    proposal_query = normalize(bridge) if np.linalg.norm(bridge) > 1e-12 else query_unit
            else:
                # ``real_member_vector`` and the explicit round-robin mode
                # intentionally use the real member representation.  Dense
                # mode is the flat query expansion control.
                proposal_query = normalize(query_vector) if proposal_mode == "dense" else index.vector(parent_id)
            hits = _call_index_search(tracker, index, proposal_query, proposal_limit, layer_exclude)
            for rank, (candidate_id, score) in enumerate(hits):
                candidate_id = str(candidate_id)
                item = ProposalExposure(parent_id, candidate_id, layer_number + 1, rank, float(score))
                layer_exposures.append(item)
                exposures.append(item)
                parent_sources.setdefault(candidate_id, set()).add(parent_id)
        if provider_parent_ids and tracker.can_propose():
            started = time.perf_counter()
            # ``query_vector`` is only used for deterministic provider
            # adapters that want the raw query alongside bridge text; the
            # text itself carries the user query and anchor evidence.
            provider_matrix = _invoke_proposal_query_provider(
                proposal_query_provider,
                provider_texts,
                query=str(query_text),
                instruction=proposal_query_instruction,
            )
            proposal_embedding_ms += (time.perf_counter() - started) * 1000.0
            proposal_embedding_queries += len(provider_parent_ids)
            for parent_id, proposal_query in zip(provider_parent_ids, provider_matrix):
                hits = _call_index_search(tracker, index, proposal_query, proposal_limit, layer_exclude)
                for rank, (candidate_id, score) in enumerate(hits):
                    candidate_id = str(candidate_id)
                    item = ProposalExposure(parent_id, candidate_id, layer_number + 1, rank, float(score))
                    layer_exposures.append(item)
                    exposures.append(item)
                    parent_sources.setdefault(candidate_id, set()).add(parent_id)
        if not layer_exposures:
            break
        # First occurrence order is protocol-defined (member position, ANN
        # rank, ID), independent of network return timing.
        first_by_candidate: dict[str, ProposalExposure] = {}
        ordered_exposures = sorted(
            layer_exposures,
            key=lambda value: (current.index(value.parent_id), value.rank, value.candidate_id),
        )
        for item in ordered_exposures:
            first_by_candidate.setdefault(item.candidate_id, item)
        unseen = [
            item.candidate_id
            for item in sorted(
                first_by_candidate.values(),
                key=lambda value: (current.index(value.parent_id), value.rank, value.candidate_id),
            )
            if item.candidate_id not in admitted and item.candidate_id not in excluded
        ]
        # Pass the complete deterministic exposure list to the tracker.  It
        # admits the allowed prefix and counts every remaining candidate as
        # an ``unadmitted_exposure``; slicing here would lose that audit fact.
        before_layer = set(tracker.visited_ids)
        tracker.admit_nodes(unseen)
        # ``admit_nodes`` can be more restrictive when a shared tracker has
        # already consumed capacity; honor the actual newly admitted set.
        admitted_this_layer = [
            identifier
            for identifier in unseen
            if identifier in tracker.visited_ids and identifier not in before_layer
        ]
        admitted.update(admitted_this_layer)
        for candidate_id in admitted_this_layer:
            for item in layer_exposures:
                if item.candidate_id != candidate_id:
                    continue
                edge_pairs.add((item.parent_id, candidate_id))
        for item in layer_exposures:
            if item.candidate_id in admitted and item.candidate_id not in admitted_this_layer:
                back_edges.append({
                    "parent_id": item.parent_id,
                    "candidate_id": item.candidate_id,
                    "layer": item.layer,
                    "rank": item.rank,
                    "reason": "already_admitted_or_earlier_layer",
                })
        if not admitted_this_layer:
            break
        layers.append(tuple(admitted_this_layer))
        if len(admitted) >= search_budget.max_unique_nodes:
            # We still performed the complete current layer's exposures; no
            # later layer can admit a node, but duplicate evidence remains
            # accounted for above.
            break

    # Compute relation weights only on observed proposal edges.  This is the
    # central distinction from the superseded full-bank rank matrix.
    edge_weights: list[tuple[str, str, float]] = []
    for parent_id, candidate_id in sorted(edge_pairs):
        left = index.vector(parent_id)
        right = index.vector(candidate_id)
        weight = (
            angular_navigation_affinity(left, right)
            if relation_mode == "angular"
            else nonnegative_cosine(left, right)
        )
        edge_weights.append((parent_id, candidate_id, float(weight)))
    root_vectors = {identifier: index.vector(identifier) for identifier in anchor_ids}
    root_raw = {
        identifier: (
            angular_navigation_affinity(normalize(query_vector), vector)
            if relation_mode == "angular"
            else nonnegative_cosine(normalize(query_vector), vector)
        )
        for identifier, vector in root_vectors.items()
    }
    root_total = float(sum(root_raw.values()))
    if root_raw and root_total > 0.0:
        root_raw = {identifier: value / root_total for identifier, value in root_raw.items()}
    elif root_raw:
        root_raw = {identifier: 1.0 / len(root_raw) for identifier in root_raw}
    ordered_ids = tuple(sorted(admitted))
    # Keep layer order in the graph while memory IDs use a canonical order for
    # hashes and matrix indexing.  Re-map layers accordingly.
    layer_order = [identifier for layer in layers for identifier in layer if identifier in admitted]
    if set(layer_order) != set(ordered_ids):
        layer_order.extend(identifier for identifier in ordered_ids if identifier not in layer_order)
    layers_for_graph = tuple(tuple(layer) for layer in layers if layer)
    # Ensure every graph ID is covered exactly once even when a custom tracker
    # admitted an ID outside the local anchor order.
    covered = {identifier for layer in layers_for_graph for identifier in layer}
    if covered != set(ordered_ids):
        layers_for_graph = layers_for_graph + (
            tuple(identifier for identifier in ordered_ids if identifier not in covered),
        )
    graph = FrozenProposalGraph(
        memory_ids=ordered_ids,
        edges=tuple(sorted(edge_pairs)),
        edge_weights=tuple(edge_weights),
        root_mass=tuple(float(root_raw.get(identifier, 0.0)) for identifier in ordered_ids),
        layers=layers_for_graph,
        parent_sources=tuple(
            (identifier, tuple(sorted(parent_sources.get(identifier, set()))))
            for identifier in ordered_ids
        ),
        proposal_records=tuple(
            (item.parent_id, item.candidate_id, item.layer, item.rank, item.score)
            for item in exposures
            if item.parent_id in admitted and item.candidate_id in admitted
        ),
        cutoff=cutoff,
        proposal_config={
            "proposal_mode": proposal_mode,
            "relation_mode": relation_mode,
            "initial_width": initial_width,
            "branch_width": branch_width,
            "proposal_width": proposal_limit,
            "max_depth": depth_limit,
            # The visible bank is a set-like identity.  Canonical sorting is
            # essential: reordering caller input must not produce a new graph
            # hash or invalidate a cached frozen quality table.
            "visible_ids": sorted(ids),
            "query_anchor_source": proposal_embedding_source,
            "query_anchor_instruction": str(proposal_query_instruction),
            # Keep the complete paid exposure stream in the frozen graph
            # metadata.  ``proposal_records`` is intentionally restricted to
            # valid graph edges; these audit-only rows also retain candidates
            # that lost the unique-node admission race.
            "raw_anchor_exposures": [
                [str(identifier), float(score)] for identifier, score in first_hits
            ],
            "raw_proposal_records": [
                [item.parent_id, item.candidate_id, item.layer, item.rank, item.score]
                for item in exposures
            ],
            "proposal_exposure_budget": tracker.budget.max_candidate_exposure,
            "unique_node_budget": tracker.budget.max_unique_nodes,
        },
        domain_scope="proposal_domain",
    )
    raw_rows = [
        (str(identifier), float(score)) for identifier, score in first_hits
    ] + [
        (item.candidate_id, float(item.score)) for item in exposures
    ]
    unadmitted_raw_rows = [
        row for row in raw_rows if row[0] not in admitted and row[0] not in excluded
    ]
    unadmitted_unique_ids = sorted({row[0] for row in unadmitted_raw_rows})
    accepted_raw_rows = [row for row in raw_rows if row[0] in admitted]
    diagnostics = {
        "graph_hash": graph.graph_hash,
        "layers": [list(layer) for layer in graph.layers],
        "proposal_records": [list(record) for record in graph.proposal_records],
        "parent_sources": {candidate: list(parents) for candidate, parents in graph.parent_sources},
        "back_edge_exposures": back_edges,
        "raw_exposure": len(exposures) + len(first_hits),
        "raw_exposure_count": len(raw_rows),
        "raw_exposure_rows": [list(row) for row in raw_rows],
        "accepted_raw_exposure_count": len(accepted_raw_rows),
        "unadmitted_raw_row_count": len(unadmitted_raw_rows),
        "unadmitted_unique_count": len(unadmitted_unique_ids),
        "unadmitted_unique_ids": unadmitted_unique_ids,
        "excluded_exposure_count": sum(1 for row in raw_rows if row[0] in excluded),
        "proposal_exposure_budget": tracker.budget.max_candidate_exposure,
        "unique_node_budget": tracker.budget.max_unique_nodes,
        "unique_admissions": len(graph.memory_ids),
        "duplicate_parent_edges": sum(max(0, len(parents) - 1) for parents in parent_sources.values()),
        "relation_representation": "observed_edges_sparse",
        "proposal_mode": proposal_mode,
        "relation_mode": relation_mode,
        "cutoff": cutoff,
        "visible_bank_ids": ids,
        "proposal_domain_ids": list(graph.memory_ids),
        "query_anchor_source": proposal_embedding_source,
        "query_anchor_queries": int(proposal_embedding_queries),
        "query_anchor_embedding_ms": float(proposal_embedding_ms),
        "query_anchor_vectors_precomputed": len(proposal_vector_map),
        "unadmitted_exposure_count": int(getattr(tracker, "unadmitted_exposure_count", 0)),
    }
    if proposal_embedding_queries:
        tracker.record_bridge_embedding(proposal_embedding_queries, proposal_embedding_ms)
    tracker.record_duplicate_parent_edges(diagnostics["duplicate_parent_edges"])
    return graph, diagnostics, tracker, index


def _quality_from_provider(
    quality_provider: Any,
    query: Any,
    records: Mapping[str, Memory],
    *,
    score_space: str,
    scorer_fingerprint: str,
) -> dict[str, QualityRecord]:
    if quality_provider is None:
        raise ValueError("semantic_path_v1 requires a frozen quality provider")
    expected_ids = tuple(str(identifier) for identifier in records)
    if not expected_ids:
        return {}

    method = getattr(quality_provider, "score_all", None)
    if not callable(method):
        method = getattr(quality_provider, "score", None)
    if not callable(method) and callable(quality_provider):
        method = quality_provider
    if callable(method):
        # Prefer signature inspection to catching every TypeError.  A
        # TypeError raised *inside* a scorer is a real provider failure and
        # must not trigger a second, differently shaped request.
        try:
            signature = inspect.signature(method)
            positional = [
                parameter
                for parameter in signature.parameters.values()
                if parameter.kind
                in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
            has_varargs = any(
                parameter.kind == inspect.Parameter.VAR_POSITIONAL
                for parameter in signature.parameters.values()
            )
        except (TypeError, ValueError):
            positional, has_varargs = [], True
        if has_varargs or len(positional) >= 2:
            raw = method(query, records)
        elif len(positional) == 1:
            raw = method(records)
        else:
            raw = method()
    else:
        raw = quality_provider

    return _normalize_quality_output(
        raw,
        expected_ids,
        score_space=score_space,
        scorer_fingerprint=scorer_fingerprint,
    )


def _normalize_quality_output(
    raw: Any,
    expected_ids: Sequence[str],
    *,
    score_space: str,
    scorer_fingerprint: str,
) -> dict[str, QualityRecord]:
    """Normalize mapping, aligned sequence, and record-object scorer output."""
    expected = tuple(str(identifier) for identifier in expected_ids)
    if not expected:
        return {}
    if isinstance(raw, Mapping) and not set(str(key) for key in raw).intersection(set(expected)):
        # Some adapters wrap the actual table in ``scores``/``results``.
        for wrapper in ("scores", "results", "data", "records"):
            candidate = raw.get(wrapper)
            if candidate is not None:
                raw = candidate
                break
    if isinstance(raw, Mapping):
        # Accept the common ``{id: score}`` form as well as
        # ``{id: {score/raw_score/value, ...}}`` records.  Normalizing this
        # at the adapter boundary keeps the immutable QualityRecord contract
        # identical for mapping, sequence, and HTTP-style providers.
        table: dict[str, Any] = {}
        for raw_id, item in raw.items():
            identifier = str(raw_id)
            if identifier in table:
                raise ValueError("quality provider mapping contains duplicate IDs after string normalization")
            if isinstance(item, Mapping):
                item_id = item.get("memory_id", item.get("id", identifier))
                if str(item_id) != identifier:
                    raise ValueError("quality provider mapping item ID disagrees with its key")
                if "raw_score" in item:
                    table[identifier] = QualityRecord.from_raw(
                        identifier,
                        item["raw_score"],
                        item.get("score_space", score_space),
                        scorer_fingerprint=item.get("scorer_fingerprint", scorer_fingerprint),
                        input_hash=item.get("input_hash", ""),
                    )
                elif "score" in item or "value" in item:
                    table[identifier] = QualityRecord.from_raw(
                        identifier,
                        item.get("score", item.get("value")),
                        item.get("score_space", score_space),
                        scorer_fingerprint=item.get("scorer_fingerprint", scorer_fingerprint),
                        input_hash=item.get("input_hash", ""),
                    )
                else:
                    raise ValueError(f"quality provider record {identifier} has no score")
            else:
                table[identifier] = item
        return validate_quality_records(
            table,
            expected,
            score_space=score_space,
            scorer_fingerprint=scorer_fingerprint,
        )
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        # A sequence of ``(memory_id, score)`` pairs is an explicit keyed
        # response; all other sequences are positional and must align exactly
        # with the requested IDs.
        keyed_pairs = all(
            isinstance(item, (tuple, list))
            and len(item) == 2
            and isinstance(item[0], (str, int, np.integer))
            for item in raw
        )
        if keyed_pairs and len(raw) != len(expected):
            raise ValueError("quality provider keyed sequence must cover every requested ID")
        if len(raw) != len(expected):
            raise ValueError("quality provider sequence must align with every requested ID")
        table: dict[str, Any] = {}
        iterator = (
            ((str(item[0]), item[1]) for item in raw)
            if keyed_pairs
            else ((identifier, item) for identifier, item in zip(expected, raw))
        )
        for identifier, item in iterator:
            if keyed_pairs and identifier not in expected:
                raise ValueError("quality provider sequence contains an unknown ID")
            if identifier in table:
                raise ValueError("quality provider sequence contains duplicate IDs")
            if isinstance(item, QualityRecord):
                if item.memory_id != identifier:
                    raise ValueError("quality provider sequence item ID does not match its key")
                table[identifier] = item
            elif isinstance(item, Mapping):
                item_id = item.get("memory_id", item.get("id", identifier))
                if str(item_id) != identifier:
                    raise ValueError("quality provider sequence item ID does not match its position")
                if "raw_score" in item:
                    table[identifier] = QualityRecord.from_raw(
                        identifier,
                        item["raw_score"],
                        item.get("score_space", score_space),
                        scorer_fingerprint=item.get("scorer_fingerprint", scorer_fingerprint),
                        input_hash=item.get("input_hash", ""),
                    )
                elif "score" in item or "value" in item:
                    table[identifier] = QualityRecord.from_raw(
                        identifier,
                        item.get("score", item.get("value")),
                        item.get("score_space", score_space),
                        scorer_fingerprint=item.get("scorer_fingerprint", scorer_fingerprint),
                        input_hash=item.get("input_hash", ""),
                    )
                else:
                    raise ValueError("quality provider sequence item has no score")
            else:
                table[identifier] = item
        return validate_quality_records(
            table,
            expected,
            score_space=score_space,
            scorer_fingerprint=scorer_fingerprint,
        )
    if isinstance(raw, (int, float, np.integer, np.floating)) and not isinstance(raw, bool):
        # A scalar is an explicit constant adapter, useful for deterministic
        # smoke tests.  It is never inferred from a missing/failed provider.
        table = {
            identifier: QualityRecord.from_raw(
                identifier,
                float(raw),
                score_space,
                scorer_fingerprint=scorer_fingerprint,
            )
            for identifier in expected
        }
        return validate_quality_records(
            table,
            expected,
            score_space=score_space,
            scorer_fingerprint=scorer_fingerprint,
        )
    raise ValueError("quality provider must return an ID mapping or aligned sequence")


def reranker_quality_records(
    reranker: Any,
    query: str,
    records: Mapping[str, Memory],
    *,
    answer_options: str = "",
    instruction: str | None = None,
    include_time_metadata: bool = True,
    score_space: str = "unit_interval",
    scorer_fingerprint: str = "",
    query_cutoff: Any = None,
    query_metadata: Mapping[str, Any] | None = None,
    use_answer_options: bool = True,
) -> dict[str, QualityRecord]:
    """Score a frozen pool pointwise with strict response validation."""

    if reranker is None:
        raise ValueError("reranker is required for frozen_reranker quality")
    if not isinstance(query, str):
        raise ValueError("reranker query must be a string")
    if not isinstance(include_time_metadata, bool) or not isinstance(use_answer_options, bool):
        raise ValueError("reranker formatting flags must be boolean")
    declared_contract = str(getattr(reranker, "score_contract", "pointwise")).strip().lower()
    if declared_contract == "listwise":
        raise ValueError("listwise reranker output cannot be adapted as pointwise quality")
    if declared_contract not in {"pointwise", ""}:
        raise ValueError(f"unsupported reranker score contract: {declared_contract}")
    if not isinstance(records, Mapping):
        sequence_records = list(records)
        records = {}
        for item in sequence_records:
            identifier = str(getattr(item, "memory_id", ""))
            if not identifier:
                raise ValueError("reranker records must expose memory_id")
            if identifier in records:
                raise ValueError("reranker records must have unique IDs")
            records[identifier] = item
    normalized_records: dict[str, Any] = {}
    for raw_key, value in records.items():
        identifier = str(raw_key)
        if identifier in normalized_records:
            raise ValueError("reranker records must have unique IDs after string normalization")
        normalized_records[identifier] = value
    records = normalized_records
    if not records:
        return {}
    # Compose one task query for every candidate.  In particular, options are
    # public evidence for the reranker, while the gold answer is never part of
    # this string.  A raw ``query``-only call would make semantic quality
    # depend on whether a caller happened to pre-compose an Example object.
    task_instruction = (instruction or DEFAULT_FINAL_RERANK_INSTRUCTION).strip()
    effective_fingerprint = str(
        scorer_fingerprint
        or getattr(reranker, "model_fingerprint", "")
        or getattr(reranker, "model", "")
        or getattr(getattr(reranker, "config", None), "model", "")
        or getattr(getattr(reranker, "config", None), "endpoint", "")
    )
    sections = [task_instruction, f"Current request:\n{query}"]
    if use_answer_options and str(answer_options).strip():
        try:
            from .personamem import parse_options

            options = parse_options(answer_options)
            option_lines = []
            for index, option in enumerate(options):
                cleaned = str(option).strip()
                # Match the shared ranking formatter without exposing the
                # correct-answer label or the original ``(a)`` marker.
                cleaned = cleaned.lstrip("([{ ")
                if ")" in cleaned[:3] or "]" in cleaned[:3]:
                    cleaned = cleaned[2:].strip()
                option_lines.append(
                    f"{chr(ord('A') + index)}. {cleaned}" if index < 26 else f"{index + 1}. {cleaned}"
                )
            if option_lines:
                sections.append("Candidate answers:\n" + "\n".join(option_lines))
        except Exception:
            # Malformed options are still part of the caller's explicit input;
            # retain them verbatim rather than silently dropping a task field.
            sections.append(f"Candidate answers:\n{answer_options}")
    rank_query = "\n\n".join(sections)
    # Pointwise quality is a per-memory contract.  Canonicalizing the request
    # order makes the adapter invariant to mapping insertion order and keeps
    # service batches/cache keys reproducible across architectures.
    ids = sorted(records)
    timestamps: list[float] = []
    for memory in records.values():
        try:
            timestamp = float(memory.timestamp)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("reranker records must have finite timestamps") from exc
        if not np.isfinite(timestamp):
            raise ValueError("reranker records must have finite timestamps")
        timestamps.append(timestamp)
    max_timestamp = max(timestamps, default=0.0)
    documents = [
        format_memory_document(records[identifier], max_timestamp, include_time_metadata=include_time_metadata)
        for identifier in ids
    ]
    if hasattr(reranker, "rerank_all") and callable(reranker.rerank_all):
        items = reranker.rerank_all(rank_query, documents)
    elif hasattr(reranker, "rerank") and callable(reranker.rerank):
        items = reranker.rerank(rank_query, documents, len(documents))
    elif callable(reranker):
        items = reranker(rank_query, documents)
    else:
        raise ValueError("reranker must expose rerank/rerank_all or be callable")
    if not isinstance(items, Sequence):
        raise ValueError("reranker response must be a sequence")
    seen: set[int] = set()
    raw_scores: dict[str, float] = {}
    for item in items:
        if isinstance(item, Mapping):
            if "index" not in item:
                raise ValueError("reranker response item is missing index")
            position = _strict_int(item["index"], "reranker response index", nonnegative=True)
            raw_score = item.get("score", item.get("relevance_score"))
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            position, raw_score = item[0], item[1]
        else:
            try:
                position = _strict_int(item.index, "reranker response index", nonnegative=True)
                raw_score = item.score
            except (AttributeError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError("reranker response contains an invalid item") from exc
        if raw_score is None:
            raise ValueError("reranker response item is missing score")
        try:
            position = _strict_int(position, "reranker response index", nonnegative=True)
            score = float(raw_score)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("reranker response contains an invalid item") from exc
        if isinstance(raw_score, (bool, np.bool_)) or not np.isfinite(score):
            raise ValueError("reranker response contains a non-finite score")
        if position < 0 or position >= len(ids) or position in seen:
            raise ValueError("reranker response contains an unknown or duplicate index")
        if not np.isfinite(score):
            raise ValueError("reranker response contains a non-finite score")
        seen.add(position)
        raw_scores[ids[position]] = score
    if seen != set(range(len(ids))):
        missing = sorted(set(range(len(ids))) - seen)
        raise ValueError(f"reranker response omitted document indices: {missing}")
    def input_hash(memory: Memory) -> str:
        metadata = memory.metadata if isinstance(memory.metadata, Mapping) else {}
        payload = {
            "query": query,
            "rank_query": rank_query,
            "options": answer_options,
            "cutoff": query_cutoff,
            # Metadata participates in cache identity only after the same
            # defensive redaction used by query-conditioned representations;
            # replacing a gold label must not create a different retrieval
            # request or quality-cache key.
            "query_metadata": _safe_semantic_metadata(dict(query_metadata or {})),
            "memory_id": memory.memory_id,
            "memory_text": memory.text,
            "timestamp": memory.timestamp,
            "source_id": memory.source_id,
            "metadata": _safe_semantic_metadata(metadata),
            "include_time_metadata": bool(include_time_metadata),
            "score_space": score_space,
            "scorer_fingerprint": effective_fingerprint,
            "task_instruction": task_instruction,
        }
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    return {
        identifier: QualityRecord.from_raw(
            identifier,
            raw_scores[identifier],
            score_space,
            scorer_fingerprint=effective_fingerprint,
            input_hash=input_hash(records[identifier]),
        )
        for identifier in ids
    }


def _fallback_direct_quality(
    query_vector: np.ndarray,
    index: ExactInnerProductIndex,
    ids: Sequence[str],
) -> dict[str, QualityRecord]:
    # This fallback is available only for explicitly non-reranker semantic
    # experiments and is recorded as such by the caller.
    return {
        identifier: QualityRecord.from_raw(
            identifier,
            nonnegative_cosine(query_vector, index.vector(identifier)),
            "unit_interval",
        )
        for identifier in ids
    }


def _make_context_plan(
    query: str,
    selected: Sequence[Memory],
    answer_options: str = "",
    *,
    token_budget: int | None = None,
    strict: bool = True,
    selected_ids: Sequence[str] | None = None,
    generator_config: Any | None = None,
) -> ContextPlan:
    from .clients import build_context_plan

    return build_context_plan(
        query,
        selected,
        answer_options,
        token_budget=token_budget,
        strict=strict,
        selected_ids=selected_ids,
        generator_config=generator_config,
    )


def semantic_retrieve(
    query: str,
    query_vector: np.ndarray,
    memories: Sequence[Memory],
    memory_vectors: np.ndarray,
    config: RetrievalConfig,
    *,
    index: ExactInnerProductIndex | None = None,
    budget: SearchBudget | None = None,
    cost_tracker: CostTracker | None = None,
    index_build_ms: float = 0.0,
    quality_provider: Any = None,
    quality_records: Mapping[str, QualityRecord | float] | None = None,
    reranker: Any = None,
    representation_provider: Any = None,
    answer_options: str = "",
    query_cutoff: Any = None,
    query_metadata: Mapping[str, Any] | None = None,
    listwise_selector: Callable[..., Sequence[str]] | None = None,
    context_token_budget: int | None = None,
    generator_config: Any | None = None,
    frozen_graph: FrozenProposalGraph | None = None,
    proposal_query_provider: Any = None,
    proposal_vectors: Mapping[str, np.ndarray] | None = None,
    proposal_query_instruction: str = "",
    representation_fingerprint: str = "",
    legacy_rho: Mapping[str, float] | None = None,
    legacy_parent_id: Mapping[str, str | None] | None = None,
    shuffle_seed: int = 42,
) -> RetrievalResult:
    """Run the frozen semantic-path chain and return auditable provenance."""

    visible, visible_vectors = _visible_records(
        memories,
        memory_vectors,
        query_cutoff,
        query_metadata=query_metadata,
    )
    search_budget = budget or SearchBudget.from_config(config)
    tracker = cost_tracker or CostTracker(search_budget)
    tracker.index_build_ms += float(index_build_ms)
    if not visible:
        tracker.set_stop_reason("insufficient_candidates")
        return RetrievalResult(
            query, [], [], {}, [], [], [], [], tracker, False, [], [],
            visible_bank=(), proposal_domain=(), selected_context=(), certificate_status="unavailable",
        )
    if frozen_graph is None:
        if (
            config.proposal_mode == "real_member_query_anchor"
            and proposal_query_provider is None
            and proposal_vectors is None
        ):
            raise ValueError(
                "proposal_mode=real_member_query_anchor requires an explicit proposal_query_provider "
                "or proposal_vectors"
            )
        graph, graph_diagnostics, tracker, index = discover_frozen_graph(
            visible,
            visible_vectors,
            query_vector,
            initial_width=config.initial_width,
            branch_width=config.branch_width,
            proposal_width=config.proposal_width,
            max_depth=config.max_depth,
            budget=search_budget,
            tracker=tracker,
            index=index,
            relation_mode=config.relation_mode,
            proposal_mode=config.proposal_mode,
            cutoff=query_cutoff,
            query_metadata=query_metadata,
            proposal_query_provider=proposal_query_provider,
            proposal_vectors=proposal_vectors,
            proposal_query_instruction=proposal_query_instruction,
            query_text=query,
        )
    else:
        graph = frozen_graph
        graph_ids = set(graph.memory_ids)
        visible_ids = {str(memory.memory_id) for memory in visible}
        if not graph_ids.issubset(visible_ids):
            raise ValueError("frozen_graph contains IDs outside the visible memory bank")
        # A reused frozen graph is an explicit protocol boundary.  Its IDs,
        # edges and provenance are not recomputed or augmented during
        # selection; only a compact diagnostic records the reuse.
        graph_diagnostics = {
            "graph_hash": graph.graph_hash,
            "graph_reused": True,
            "layers": [list(layer) for layer in graph.layers],
            "visible_bank_ids": [memory.memory_id for memory in visible],
            "proposal_domain_ids": list(graph.memory_ids),
            "relation_representation": "sparse_observed_edges",
            "proposal_config": dict(graph.proposal_config),
            "query_anchor_source": graph.proposal_config.get("query_anchor_source", "unknown"),
        }
        if index is None:
            index = build_index("exact", [memory.memory_id for memory in visible], visible_vectors)
    shuffle_effective_nodes = 0
    graph_has_shuffle = isinstance(graph.proposal_config, Mapping) and "shuffle_seed" in graph.proposal_config
    if config.path_mode == "shuffle" and not graph_has_shuffle:
        # S2-shuffle owns a separately identified frozen structure.  Pointwise
        # qualities, records and embeddings remain keyed by the original IDs;
        # only the within-layer source assignment is migrated.
        graph = shuffle_frozen_graph(graph, seed=shuffle_seed)
        # Count the actual within-layer identity intervention.  This remains
        # meaningful for an edgeless/root-only graph, where comparing edge
        # tuples would incorrectly report zero despite a non-identity
        # permutation.
        shuffle_effective_nodes = int(
            graph.proposal_config.get("shuffle_effective_nodes", 0)
            if isinstance(graph.proposal_config, Mapping)
            else 0
        )
        graph_diagnostics["shuffle_seed"] = int(shuffle_seed)
        graph_diagnostics["shuffle_effective_nodes"] = int(shuffle_effective_nodes)
    elif config.path_mode == "shuffle" and graph_has_shuffle:
        graph_diagnostics["shuffle_seed"] = int(graph.proposal_config["shuffle_seed"])
        shuffle_effective_nodes = int(graph.proposal_config.get("shuffle_effective_nodes", 0))
        graph_diagnostics["shuffle_effective_nodes"] = shuffle_effective_nodes
    measure = propagate_frozen_graph(graph)
    records = {
        str(memory.memory_id): memory
        for memory in visible
        if str(memory.memory_id) in graph.memory_ids
    }
    if quality_records is not None:
        qualities = validate_quality_records(
            quality_records,
            graph.memory_ids,
            score_space=config.quality_score_space,
            scorer_fingerprint=config.scorer_fingerprint,
        )
    elif reranker is not None or config.quality_mode == "frozen_reranker":
        if reranker is None and quality_provider is None:
            raise ValueError("frozen_reranker quality requires reranker or quality_provider")
        if reranker is not None:
            started = time.perf_counter()
            qualities = reranker_quality_records(
                reranker,
                query,
                records,
                answer_options=answer_options,
                score_space=config.quality_score_space,
                scorer_fingerprint=config.scorer_fingerprint,
                query_cutoff=query_cutoff,
                query_metadata=query_metadata,
            )
            tracker.record_rerank(len(records), (time.perf_counter() - started) * 1000.0)
        else:
            qualities = _quality_from_provider(
                quality_provider,
                query,
                records,
                score_space=config.quality_score_space,
                scorer_fingerprint=config.scorer_fingerprint,
            )
    elif quality_provider is not None:
        qualities = _quality_from_provider(
            quality_provider,
            query,
            records,
            score_space=config.quality_score_space,
            scorer_fingerprint=config.scorer_fingerprint,
        )
    else:
        # Only an explicitly declared direct/cosine (or the internal
        # constant compatibility adapter) may use the deterministic geometric
        # fallback.  A declared mapping/pointwise reranker contract without
        # its provider is a protocol error; treating it as quality zero or a
        # cosine score would make a missing service indistinguishable from a
        # valid experiment.
        if config.quality_mode not in {"direct_cosine", "constant", "rho"}:
            raise ValueError(
                f"quality_mode={config.quality_mode!r} requires quality_records, quality_provider, or reranker"
            )
        if config.quality_mode == "rho":
            # ``rho`` is the legacy graph-access quality.  Keep it distinct
            # from direct query cosine: L0/L1 use r_j=rho_j² so the semantic
            # feature's sqrt scale is exactly the old rho scale.
            qualities = rho_squared_quality_records(
                measure,
                legacy_rho=legacy_rho,
                scorer_fingerprint=config.scorer_fingerprint or "legacy-rho2",
            )
        else:
            qualities = _fallback_direct_quality(normalize(query_vector), index, graph.memory_ids)
    representation_mode = "query_conditioned" if config.feature_mode in {"query_conditioned"} else "cached_memory"
    if representation_mode == "query_conditioned" and representation_provider is None:
        # Query-conditioned features are a distinct representation contract;
        # silently substituting cached memory vectors would make the reported
        # S3/Semantic row numerically equal to the cached-memory row while
        # claiming a different architecture.
        raise ValueError(
            "feature_mode=query_conditioned requires an explicit representation_provider"
        )
    if representation_provider is None:
        # Stage-one semantic-path v1 uses the already cached unit memory
        # embeddings.  Supplying them as an explicit mapping keeps the
        # provider interface strict while avoiding a second embedding call.
        representation_provider = {
            identifier: np.asarray(index.vector(identifier), dtype=np.float64).copy()
            for identifier in graph.memory_ids
        }
    feature_provider = SemanticFeatureProvider(
        graph,
        measure if config.path_mode not in {"none"} else None,
        qualities,
        records,
        representation_provider,
        query=query,
        path_mode=config.path_mode,
        representation_mode=representation_mode,
        scorer_fingerprint=config.scorer_fingerprint,
        representation_fingerprint=representation_fingerprint,
        legacy_parent_id=legacy_parent_id,
        shuffle_seed=shuffle_seed,
    )
    selection_mode = config.selection_mode
    if config.profile in {"semantic", SEMANTIC_PROFILE} and selection_mode not in {"pure_rerank", "frozen_listwise"}:
        selection_mode = "semantic_path_logdet"
    if selection_mode == "pure_rerank":
        selection = PureRerankSelector().select(qualities, config.context_size)
    elif selection_mode == "frozen_listwise":
        if listwise_selector is None:
            raise ValueError("frozen_listwise selection requires listwise_selector")
        selection = FrozenListwiseSelector(listwise_selector).select(query, records, config.context_size)
    else:
        certificate_mode = config.certificate_mode
        if certificate_mode == "off" and config.stop_mode == "certificate_or_budget":
            certificate_mode = "lazy"
        selector = SemanticPathLogDetSelector(
            certificate_mode=certificate_mode,
            tie_tolerance=config.tie_tolerance,
            max_materializations=None,
        )
        selection = selector.select(feature_provider, config.context_size)
    selected_ids = list(selection.selected_ids)
    # Materialization costs include the actual point features and any ancestor
    # representations needed by their frozen scatter.
    tracker.record_feature_materialization(feature_provider.materialization_count)
    tracker.record_feature_materialization(feature_provider.ancestor_materialization_count, ancestor=True)
    if selection.diagnostics.get("residual_certification_gap"):
        gaps = selection.diagnostics["residual_certification_gap"]
        tracker.residual_certification_gap = float(max(gaps, default=0.0))
    if selection.diagnostics.get("certificate_domain") == "frozen_pool":
        tracker.record_bound(len(graph.memory_ids))

    # Keep the complete frozen transition/path representation alongside the
    # selector's lazily materialized semantic features.  The selector only
    # needs a matrix for its objective; a RetrievalResult, however, is also a
    # provenance boundary and must let an auditor replay every positive
    # parent/path relation without reconstructing it from the graph.
    graph_positions = {identifier: position for position, identifier in enumerate(graph.memory_ids)}
    # The semantic path is sparse by construction.  Keep the legacy dense
    # compatibility field only for small graphs; materializing an N×N array
    # for a large frozen pool defeats the sparse transition contract and can
    # exhaust memory before generation even starts.
    dense_transition_limit = 2048
    transition_matrix: np.ndarray | None = None
    if len(graph.memory_ids) <= dense_transition_limit:
        transition_matrix = np.zeros(
            (len(graph.memory_ids), len(graph.memory_ids)),
            dtype=np.float64,
        )
        for parent_id, row in measure.transition.items():
            parent_position = graph_positions[parent_id]
            for child_id, probability in row.items():
                child_position = graph_positions[child_id]
                transition_matrix[parent_position, child_position] = float(probability)
        transition_matrix.setflags(write=False)
    graph_diagnostics["dense_transition_materialized"] = transition_matrix is not None
    graph_diagnostics["dense_transition_limit"] = dense_transition_limit

    # The frozen measure (h/gamma/w) is the production ancestry object.  Path
    # hypotheses are display provenance only: enumerate every path for a
    # small DAG, but switch to a bounded representative/local-parent path when
    # the dynamic count would exceed the cap.  This keeps a highly branching graph from
    # turning serialization into an exponential operation while preserving
    # every parent posterior and every ancestor occupancy weight exactly.
    path_hypotheses: dict[str, tuple[PathHypothesis, ...]] = {}
    path_hypothesis_limit = int(getattr(config, "max_path_hypotheses", 128))
    if path_hypothesis_limit <= 0:
        raise ValueError("max_path_hypotheses must be positive")
    truncated_path_ids: list[str] = []
    for candidate_id in graph.memory_ids:
        paths, truncated = display_path_hypotheses(
            measure,
            candidate_id,
            max_paths=path_hypothesis_limit,
            branch_id="frozen_graph",
        )
        path_hypotheses[candidate_id] = paths
        if truncated:
            truncated_path_ids.append(candidate_id)

    memory_by_id = {str(memory.memory_id): memory for memory in visible}
    layer_by_id = {
        identifier: layer_number
        for layer_number, layer in enumerate(graph.layers, start=1)
        for identifier in layer
    }
    parent_posteriors = {
        identifier: dict(measure.parent_posterior.get(identifier, {}))
        for identifier in graph.memory_ids
    }
    nodes: dict[str, TreeNode] = {}
    semantic_atoms = {}
    for identifier in graph.memory_ids:
        vector = np.asarray(index.vector(identifier), dtype=np.float32)
        parent_map = parent_posteriors.get(identifier, {})
        parent_id = min(parent_map, key=lambda key: (-parent_map[key], key)) if parent_map else None
        atom = feature_provider._atoms.get(identifier)
        if atom is not None:
            semantic_atoms[identifier] = atom
            innovation = np.asarray(atom.feature, dtype=np.float64)
        else:
            innovation = np.zeros(vector.shape[0], dtype=np.float64)
        direct = nonnegative_cosine(normalize(query_vector), vector)
        column = graph_positions[identifier]
        ancestor_map = {
            ancestor_id: float(weight)
            for ancestor_id, weight in zip(graph.memory_ids, measure.ancestor_occupancy[:, column])
            if float(weight) > 0.0
        }
        nodes[identifier] = TreeNode(
            memory=memory_by_id[identifier],
            vector=vector,
            parent_id=parent_id,
            depth=layer_by_id.get(identifier, 1),
            direct_score=direct,
            reachability=float(measure.access_quality.get(identifier, 0.0)),
            innovation=innovation,
            bridge_lift=max(0.0, float(measure.access_quality.get(identifier, 0.0)) - direct),
            discovery_order=graph.memory_ids.index(identifier),
            parent_posterior=parent_map,
            path_hypotheses=path_hypotheses.get(identifier, ()),
            transition_support=float(measure.access_quality.get(identifier, 0.0)),
            semantic_quality=float(qualities[identifier].value),
            temporal_mark=_memory_mark(memory_by_id[identifier]),
            proposal_sources=tuple(
                dict(graph.parent_sources).get(identifier, ())
            ),
            layer=layer_by_id.get(identifier, 1),
            ancestor_occupancy=ancestor_map,
        )
    # Keep the historical dense InformationAtom compatibility field only for
    # genuinely small vectors.  The production semantic chain stores rank-one
    # features; rebuilding one d-by-d outer product per candidate during
    # result assembly would defeat the low-rank implementation at embedding
    # dimensions used by the service.
    compatibility_dimension = int(next(iter(semantic_atoms.values())).feature.size) if semantic_atoms else 0
    information_atoms: dict[str, InformationAtom] = {}
    if compatibility_dimension <= 256:
        information_atoms = {
            identifier: InformationAtom(
                candidate_id=identifier,
                path_ids=atom.path_ids,
                matrix=atom.matrix,
                trace=float(atom.norm_sq),
                support=float(measure.access_quality.get(identifier, 0.0)),
            )
            for identifier, atom in semantic_atoms.items()
        }
    edges = [(parent, child) for parent, child in graph.edges]
    selection_steps = []
    certified_rows = list(selection.diagnostics.get("certified", ()))
    gaps = list(selection.diagnostics.get("residual_certification_gap", ()))
    for position, (identifier, margin) in enumerate(zip(selected_ids, selection.margins), start=1):
        upper = feature_provider.quality_upper(identifier)
        cert = bool(certified_rows[position - 1]) if position - 1 < len(certified_rows) else False
        gap = float(gaps[position - 1]) if position - 1 < len(gaps) else 0.0
        selection_steps.append(SelectionStep(position, identifier, float(margin), upper, gap, cert))
    cert_requested = config.certificate_mode != "off" or config.stop_mode == "certificate_or_budget"
    cert_status = "not_requested"
    if cert_requested:
        cert_status = (
            "available"
            if selection_steps and all(step.certified for step in selection_steps)
            else "unavailable"
        )
    if cert_status == "available":
        tracker.set_stop_reason("certificate")
    elif len(graph.memory_ids) < len(visible):
        budget_exhausted = (
            len(graph.memory_ids) >= search_budget.max_unique_nodes
            or tracker.remaining_unique_nodes <= 0
            or tracker.remaining_ann_calls == 0
            or tracker.remaining_candidate_exposure == 0
        )
        tracker.set_stop_reason("search_budget" if budget_exhausted else "max_depth")
    else:
        tracker.set_stop_reason("frontier_empty")
    chronological_ids = sorted(selected_ids, key=lambda identifier: (memory_by_id[identifier].timestamp, identifier))
    selected = [memory_by_id[identifier] for identifier in chronological_ids]
    context_plan: ContextPlan | None = None
    context_plan_error: dict[str, Any] | None = None
    try:
        context_plan = _make_context_plan(
            query,
            selected,
            answer_options,
            token_budget=context_token_budget,
            strict=config.context_strict,
            selected_ids=selected_ids,
            generator_config=generator_config,
        )
        tracker.final_context_count = len(context_plan.chronological_ids)
        tracker.final_context_tokens = context_plan.token_count
    except Exception as exc:
        # A strict semantic run must not return a seemingly successful result
        # whose eventual reader request contains a different context.  Keep a
        # diagnostic for non-strict exploratory callers, but fail the main
        # contract explicitly so matrix runners record a failed cell.
        context_plan_error = {"type": type(exc).__name__, "message": str(exc)}
        graph_diagnostics["context_plan_error"] = context_plan_error
        if config.context_strict:
            from .clients import ContextPlanError

            if isinstance(exc, ContextPlanError):
                raise
            raise ContextPlanError(f"unable to build exact ContextPlan: {exc}") from exc
    diagnostics = {
        **graph_diagnostics,
        "graph": graph.public_dict(),
        "graph_hash": graph.graph_hash,
        "propagation": measure.public_dict(),
        "quality": {identifier: record.public_dict() for identifier, record in qualities.items()},
        "quality_contract": config.quality_score_space,
        "semantic_atoms": {identifier: atom.public_dict() for identifier, atom in semantic_atoms.items()},
        "selection": dict(selection.diagnostics),
        "selection_ids": selected_ids,
        "chronological_ids": chronological_ids,
        "visible_bank": list(identifier for identifier in (memory.memory_id for memory in visible)),
        "proposal_domain": list(graph.memory_ids),
        "selected_context": chronological_ids,
        "transition_representation": "sparse_observed_edges",
        "transition_sparse": {
            parent: dict(row) for parent, row in measure.transition.items()
        },
        "parent_posterior": {
            identifier: dict(values) for identifier, values in measure.parent_posterior.items()
        },
        "path_hypotheses": {
            identifier: [path.public_dict() for path in paths]
            for identifier, paths in path_hypotheses.items()
        },
        "path_hypotheses_complete": not truncated_path_ids,
        "path_hypotheses_truncated_ids": truncated_path_ids,
        "path_hypothesis_limit": path_hypothesis_limit,
        "context_plan": None if context_plan is None else context_plan.public_dict(),
        "context_status": (
            "within_budget"
            if context_plan is not None and context_plan.budget_status == "within_budget"
            else "over_budget"
            if context_plan is not None
            else "failed"
        ),
    }
    return RetrievalResult(
        query=query,
        selected=selected,
        selected_in_greedy_order=selected_ids,
        nodes=nodes,
        edges=edges,
        all_branches=[],
        remaining_branches=[],
        selection_steps=selection_steps,
        cost_tracker=tracker,
        budget_frozen=(
            len(graph.memory_ids) < len(visible)
            and (
                len(graph.memory_ids) >= search_budget.max_unique_nodes
                or tracker.remaining_unique_nodes <= 0
                or tracker.remaining_ann_calls == 0
                or tracker.remaining_candidate_exposure == 0
            )
        ),
        cluster_radii=[],
        cluster_stabilities=[],
        cluster_member_counts=[],
        clustering_ms=0.0,
        first_arrival_semantics="semantic_path_v1",
        transition=transition_matrix,
        path_hypotheses=path_hypotheses,
        information_atoms=information_atoms,
        domains={"frozen_pool": graph.memory_ids},
        bounds={
            "frozen_pool": {
                "U": max(
                    (feature_provider.quality_upper(identifier) for identifier in graph.memory_ids),
                    default=0.0,
                )
            }
        },
        certificate_status=cert_status,
        diagnostics=diagnostics,
        context_hash=None if context_plan is None else context_plan.context_hash,
        visible_bank=tuple(memory.memory_id for memory in visible),
        proposal_domain=graph.memory_ids,
        selected_context=tuple(chronological_ids),
        frozen_graph=graph,
        semantic_atoms=semantic_atoms,
        context_plan=context_plan,
        residual_certification_gaps=tuple(gaps),
        transition_sparse={
            str(parent): {str(child): float(probability) for child, probability in row.items()}
            for parent, row in measure.transition.items()
        },
    )


# Friendly aliases for notebooks and hidden integration clients.
build_proposal_graph = discover_frozen_graph
run_semantic_path = semantic_retrieve
SemanticRetriever = semantic_retrieve


__all__ = [
    "SEMANTIC_PROFILE",
    "SEMANTIC_QUERY_CONDITIONED_INSTRUCTION",
    "build_query_conditioned_representation_text",
    "rho_squared_quality_records",
    "legacy_trace_from_result",
    "ProposalExposure",
    "discover_frozen_graph",
    "build_proposal_graph",
    "reranker_quality_records",
    "semantic_retrieve",
    "run_semantic_path",
    "SemanticRetriever",
]
