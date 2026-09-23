from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .budget import CostSnapshot, CostTracker, SearchBudget


def _strict_int_value(value: Any, name: str, *, positive: bool = False, nonnegative: bool = False) -> int:
    """Validate an integer-valued serialized field without bool coercion."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, Integral):
        result = int(value)
    elif isinstance(value, Real):
        numeric = float(value)
        if not np.isfinite(numeric) or numeric != float(int(numeric)):
            raise ValueError(f"{name} must be an integer")
        result = int(numeric)
    elif isinstance(value, str):
        text = value.strip()
        try:
            numeric = float(text)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be an integer") from exc
        if not text or not np.isfinite(numeric) or numeric != float(int(numeric)):
            raise ValueError(f"{name} must be an integer")
        result = int(numeric)
    else:
        raise ValueError(f"{name} must be an integer")
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _strict_float_value(value: Any, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(result) or (nonnegative and result < 0.0):
        raise ValueError(f"{name} must be finite" + (" and non-negative" if nonnegative else ""))
    return result


def _jsonable(value: Any) -> Any:
    """Recursively convert NumPy/scalar containers for provenance JSON."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _assert_jsonable(value: Any, path: str = "request") -> None:
    """Reject values that a frozen HTTP request cannot represent in JSON.

    ``json.dumps(..., default=str)`` is useful for diagnostic hashes but is
    unsafe for a request contract: an arbitrary object could hash one way and
    fail (or be stringified differently) at transport time.  Normalize NumPy
    containers first, then accept only JSON primitives/containers and finite
    numbers.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string key")
            _assert_jsonable(child, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_jsonable(child, f"{path}[{index}]")
        return
    raise ValueError(f"{path} contains a non-JSON value of type {type(value).__name__}")


def _freeze_value(value: Any) -> Any:
    """Copy nested provenance values into immutable containers.

    ``frozen=True`` only protects dataclass attributes; a nested list/dict can
    otherwise still be mutated after a graph or context has been certified.
    Keep this helper deliberately JSON-shaped so public serialization remains
    straightforward and non-JSON provider metadata is represented by a
    stable string only when hashing is requested.
    """
    if isinstance(value, np.ndarray):
        array = np.array(value, copy=True)
        array.setflags(write=False)
        return array
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_value(item) for item in value)
    return value


_SENSITIVE_REQUEST_KEYS = {
    "api_key",
    "api_key_env",
    "authorization",
    "token",
    "password",
    "secret",
}


def _safe_request_value(value: Any, *, key: str | None = None) -> Any:
    """Return a JSON-safe request identity with secrets/endpoints redacted.

    Context plans retain the transport endpoint in memory so a client can
    verify that a plan is sent to the intended service.  Persisted/public
    artifacts must not leak that endpoint (or credentials), so endpoint
    values are represented by a stable SHA-256 identity.  The same
    canonicalization is used by :func:`context_plan_hash`, allowing an audit
    to recompute a hash from the redacted public request.
    """

    normalized_key = str(key).lower() if key is not None else ""
    if normalized_key in _SENSITIVE_REQUEST_KEYS:
        return "[redacted]"
    if normalized_key == "endpoint":
        text = str(value)
        # Public plans already contain the digest.  Avoid hashing it twice so
        # a persisted plan remains independently auditable.
        if re.fullmatch(r"[0-9a-fA-F]{64}", text):
            return text.lower()
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    if isinstance(value, Mapping):
        return {
            str(child_key): _safe_request_value(child_value, key=str(child_key))
            for child_key, child_value in value.items()
            if str(child_key).lower() not in _SENSITIVE_REQUEST_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_safe_request_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_safe_request_value(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _context_plan_payload(
    *,
    selected_ids: Sequence[str],
    chronological_ids: Sequence[str],
    serialized_context: str,
    messages: Sequence[Mapping[str, Any]],
    token_count: int,
    token_count_is_estimate: bool,
    budget: int | None,
    budget_status: str,
    prompt_hash: str,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the canonical, request-complete ContextPlan identity payload.

    The hash deliberately contains the exact messages and every request
    parameter, rather than only an unordered set of memory IDs.  This helper
    is shared by construction and validation so a plan cannot carry a hash
    for one request while sending another one.
    """

    if isinstance(selected_ids, (str, bytes)) or not isinstance(selected_ids, Sequence):
        raise ValueError("ContextPlan selected_ids must be a sequence")
    if isinstance(chronological_ids, (str, bytes)) or not isinstance(chronological_ids, Sequence):
        raise ValueError("ContextPlan chronological_ids must be a sequence")
    selected: list[str] = []
    for value in selected_ids:
        if not isinstance(value, str) or not value:
            raise ValueError("ContextPlan selected_ids must contain non-empty strings")
        selected.append(value)
    chronological: list[str] = []
    for value in chronological_ids:
        if not isinstance(value, str) or not value:
            raise ValueError("ContextPlan chronological_ids must contain non-empty strings")
        chronological.append(value)
    if len(selected) != len(set(selected)) or len(chronological) != len(set(chronological)):
        raise ValueError("ContextPlan IDs must be unique")
    if set(selected) != set(chronological):
        raise ValueError("chronological_ids must contain exactly selected_ids")
    if not isinstance(serialized_context, str):
        raise ValueError("ContextPlan serialized_context must be a string")
    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
        raise ValueError("ContextPlan messages must be a sequence")
    normalized_messages: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, Mapping) or "role" not in message or "content" not in message:
            raise ValueError("ContextPlan messages must contain role and content")
        if not isinstance(message["role"], str) or not isinstance(message["content"], str):
            raise ValueError("ContextPlan message role and content must be strings")
        normalized_messages.append({"role": message["role"], "content": message["content"]})
    token_count = _strict_int_value(token_count, "ContextPlan token_count", nonnegative=True)
    if not isinstance(token_count_is_estimate, bool):
        raise ValueError("ContextPlan token_count_is_estimate must be boolean")
    budget = None if budget is None else _strict_int_value(budget, "ContextPlan budget", nonnegative=True)
    if budget_status not in {"within_budget", "exceeds_budget", "unknown"}:
        raise ValueError("ContextPlan budget_status is invalid")
    if not isinstance(prompt_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", prompt_hash):
        raise ValueError("ContextPlan prompt_hash must be a SHA-256 hex digest")
    if not isinstance(request, Mapping):
        raise ValueError("ContextPlan request must be a mapping")
    normalized_request = _jsonable(dict(request))
    _assert_jsonable(normalized_request)
    normalized_serialized_context = str(serialized_context)
    if normalized_messages:
        expected_context = (
            normalized_messages[1]["content"]
            if len(normalized_messages) > 1
            else normalized_messages[0]["content"]
        )
    else:
        expected_context = ""
    if normalized_serialized_context != expected_context:
        raise ValueError("ContextPlan serialized_context does not match its messages")
    return {
        "schema": 3,
        "selected_ids": selected,
        "chronological_ids": chronological,
        "serialized_context": normalized_serialized_context,
        "messages": normalized_messages,
        "token_count": token_count,
        "token_count_is_estimate": token_count_is_estimate,
        "budget": budget,
        "budget_status": budget_status,
        "prompt_hash": prompt_hash.lower(),
        "request": _safe_request_value(normalized_request),
    }


def context_plan_hash(
    *,
    selected_ids: Sequence[str],
    chronological_ids: Sequence[str],
    serialized_context: str,
    messages: Sequence[Mapping[str, Any]],
    token_count: int,
    token_count_is_estimate: bool,
    budget: int | None,
    budget_status: str,
    prompt_hash: str,
    request: Mapping[str, Any],
) -> str:
    """Hash the canonical identity of a frozen context request."""

    payload = _context_plan_payload(
        selected_ids=selected_ids,
        chronological_ids=chronological_ids,
        serialized_context=serialized_context,
        messages=messages,
        token_count=token_count,
        token_count_is_estimate=token_count_is_estimate,
        budget=budget,
        budget_status=budget_status,
        prompt_hash=prompt_hash,
        request=request,
    )
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TemporalMark:
    """Immutable provenance for observation and (optional) event time.

    ``observed_*`` refers to the source-message order and is therefore always
    available for PersonaMem records.  ``event_*`` is deliberately kept as a
    textual value: the data set may contain dates, periods, or an explicit
    unknown marker and the retrieval graph must not invent a calendar scale
    from message indices.
    """

    observed_start: int
    observed_end: int
    event_start: str | None = None
    event_end: str | None = None
    validity: str = "unknown"
    time_source: str = "message_index"

    def __post_init__(self) -> None:
        for name in ("observed_start", "observed_end"):
            value = getattr(self, name)
            if isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
            try:
                integer = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be an integer") from exc
            if float(value) != float(integer):
                raise ValueError(f"{name} must be an integer")
            object.__setattr__(self, name, integer)
        if self.observed_start > self.observed_end:
            raise ValueError("observed_start must be <= observed_end")
        for name in ("event_start", "event_end"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{name} must be a string or None")
        if not str(self.validity).strip() or not str(self.time_source).strip():
            raise ValueError("validity and time_source must be non-empty")

    @property
    def unknown(self) -> bool:
        return str(self.validity).strip().lower() in {"unknown", "unavailable", ""}

    # The semantic protocol and the older ``temporal.TimeMark`` adapter use
    # slightly different names for these convenience properties.  Keeping
    # them on the canonical record means a mark can be passed through either
    # API without being mistaken for an opaque metadata object.
    @property
    def unavailable(self) -> bool:
        return self.unknown

    @property
    def is_unknown(self) -> bool:
        return self.unknown

    @property
    def is_durative(self) -> bool:
        return str(self.validity).strip().lower() == "durative"

    @property
    def start(self) -> str | int | None:
        if self.event_start is not None:
            return self.event_start
        return self.observed_start

    @property
    def end(self) -> str | int | None:
        if self.event_end is not None:
            return self.event_end
        return self.observed_end

    @property
    def points(self) -> tuple[Any, ...]:
        values = (
            (self.event_start, self.event_end)
            if self.event_start is not None or self.event_end is not None
            else (self.observed_start, self.observed_end)
        )
        return tuple(dict.fromkeys(value for value in values if value is not None))

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any] | None, timestamp: Any = None) -> "TemporalMark":
        """Parse a temporal metadata envelope without importing it eagerly.

        ``temporal.py`` owns the numeric overlap adapter.  The local import
        keeps this data type usable on its own while allowing callers that
        only know about the semantic type to use the same constructor.
        """
        from .temporal import build_time_mark

        mark = build_time_mark(metadata or {}, timestamp=timestamp)
        return cls(
            int(mark.observed_start or 0),
            int(mark.observed_end or mark.observed_start or 0),
            None if mark.event_start is None else str(mark.event_start),
            None if mark.event_end is None else str(mark.event_end),
            mark.validity,
            mark.time_source,
        )

    def public_dict(self) -> Dict[str, Any]:
        return {
            "observed_start": self.observed_start,
            "observed_end": self.observed_end,
            "event_start": self.event_start,
            "event_end": self.event_end,
            "validity": self.validity,
            "time_source": self.time_source,
        }


def _canonical_score_space(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "probability": "unit_interval",
        "probabilities": "unit_interval",
        "unit": "unit_interval",
        "unitinterval": "unit_interval",
        "sigmoid": "unit_interval",
        "logit": "logit_difference",
        "logit_diff": "logit_difference",
        "logitdifference": "logit_difference",
        "raw_logit_difference": "logit_difference",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"unit_interval", "logit_difference"}:
        raise ValueError("score_space must be unit_interval or logit_difference")
    return normalized


def _sigmoid(value: float) -> float:
    """Numerically stable logistic transform for declared logit scores."""
    return (
        float(1.0 / (1.0 + np.exp(-value)))
        if value >= 0.0
        else float(np.exp(value) / (1.0 + np.exp(value)))
    )


@dataclass(frozen=True)
class QualityRecord:
    """A frozen, explicitly contracted pointwise semantic quality score."""

    memory_id: str
    raw_score: float
    value: float
    score_space: str
    scorer_fingerprint: str = ""
    input_hash: str = ""

    def __post_init__(self) -> None:
        memory_id = str(self.memory_id)
        if not memory_id:
            raise ValueError("quality memory_id cannot be empty")
        object.__setattr__(self, "memory_id", memory_id)
        score_space = _canonical_score_space(self.score_space)
        object.__setattr__(self, "score_space", score_space)
        raw = _strict_float_value(self.raw_score, "quality raw score")
        value = _strict_float_value(self.value, "quality value")
        if value < 0.0 or value > 1.0:
            raise ValueError("quality value must lie in [0, 1]")
        expected = raw if score_space == "unit_interval" else _sigmoid(raw)
        # A QualityRecord is a frozen contract, not an arbitrary pair of
        # numbers.  Requiring the adapted value to agree with the declared
        # raw score catches accidental double-sigmoid, pool normalization, or
        # hand-written provenance that would otherwise look valid.
        if not np.isclose(value, expected, atol=1e-10, rtol=1e-10):
            raise ValueError("quality value does not match the declared score space")
        object.__setattr__(self, "raw_score", raw)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "scorer_fingerprint", str(self.scorer_fingerprint))
        object.__setattr__(self, "input_hash", str(self.input_hash))

    @classmethod
    def from_raw(
        cls,
        memory_id: str,
        raw_score: float,
        score_space: str,
        *,
        scorer_fingerprint: str = "",
        input_hash: str = "",
    ) -> "QualityRecord":
        space = _canonical_score_space(score_space)
        raw = _strict_float_value(raw_score, "quality raw score")
        if space == "unit_interval":
            value = raw
            if value < 0.0 or value > 1.0:
                raise ValueError("unit_interval quality score is outside [0, 1]")
        else:
            # Stable sigmoid without clipping the declared raw score.
            value = _sigmoid(raw)
        return cls(
            str(memory_id),
            raw,
            value,
            space,
            scorer_fingerprint=scorer_fingerprint,
            input_hash=input_hash,
        )

    def public_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "raw_score": self.raw_score,
            "value": self.value,
            "score_space": self.score_space,
            "scorer_fingerprint": self.scorer_fingerprint,
            "input_hash": self.input_hash,
        }


@dataclass(frozen=True)
class FrozenProposalGraph:
    """Copy-on-construction finite proposal DAG used by semantic selection.

    Edges and weights are sparse tuples; no full-bank transition matrix is
    required.  ``parent_sources`` and ``proposal_records`` retain duplicate
    same-layer exposures for audit even when a candidate is admitted once.
    """

    memory_ids: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    root_mass: tuple[float, ...] | Mapping[str, float] = ()
    graph_hash: str = ""
    domain_scope: str = "proposal_domain"
    edge_weights: tuple[tuple[str, str, float], ...] = ()
    layers: tuple[tuple[str, ...], ...] = ()
    parent_sources: tuple[tuple[str, tuple[str, ...]], ...] = ()
    proposal_records: tuple[tuple[str, str, int, int, float], ...] = ()
    cutoff: Any = None
    proposal_config: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        # IDs are a semantic set.  Canonicalizing their order makes a frozen
        # graph/hash independent of dictionary, ANN, or caller input order;
        # root masses supplied positionally are remapped below before sorting.
        if isinstance(self.memory_ids, (str, bytes)):
            raise ValueError("frozen graph memory_ids must be a sequence")
        raw_ids = tuple(str(value) for value in self.memory_ids)
        if any(not value for value in raw_ids):
            raise ValueError("frozen graph memory_ids must contain non-empty IDs")
        ids = tuple(sorted(raw_ids))
        if len(raw_ids) != len(set(raw_ids)):
            raise ValueError("frozen graph memory_ids must be unique")
        object.__setattr__(self, "memory_ids", ids)
        id_set = set(ids)
        normalized_edges: list[tuple[str, str]] = []
        seen_edges: set[tuple[str, str]] = set()
        for raw_edge in self.edges:
            if not isinstance(raw_edge, (tuple, list)) or len(raw_edge) != 2:
                raise ValueError("graph edges must contain (parent, child) pairs")
            raw_left, raw_right = raw_edge
            edge = (str(raw_left), str(raw_right))
            if edge[0] not in id_set or edge[1] not in id_set:
                raise ValueError("graph edges must reference declared memory IDs")
            if edge[0] == edge[1]:
                raise ValueError("graph cannot contain self edges")
            if edge not in seen_edges:
                normalized_edges.append(edge)
                seen_edges.add(edge)
        normalized_edges = sorted(set(normalized_edges))
        object.__setattr__(self, "edges", tuple(normalized_edges))
        raw_edge_weights = self.edge_weights.items() if isinstance(self.edge_weights, Mapping) else self.edge_weights
        weights: dict[tuple[str, str], float] = {}
        for item in raw_edge_weights:
            if isinstance(self.edge_weights, Mapping):
                # Accept ``{(parent, child): weight}`` as a convenience API in
                # addition to the serialized triple sequence.
                edge_key, edge_value = item
                if not isinstance(edge_key, (tuple, list)) or len(edge_key) != 2:
                    raise ValueError("edge_weights mapping keys must be (parent, child)")
                item = (edge_key[0], edge_key[1], edge_value)
                if not isinstance(item, (tuple, list)) or len(item) != 3:
                    raise ValueError("edge_weights entries must be (parent, child, weight)")
            left, right = str(item[0]), str(item[1])
            raw_weight = _strict_float_value(item[2], "edge weight", nonnegative=True)
            if (left, right) not in seen_edges:
                raise ValueError("edge weight references an undeclared edge")
            if not np.isfinite(raw_weight) or raw_weight < 0.0:
                raise ValueError("edge weights must be finite and non-negative")
            if (left, right) in weights:
                raise ValueError("edge_weights cannot contain duplicate edges")
            weights[(left, right)] = raw_weight
        normalized_weights = tuple(
            (left, right, weights.get((left, right), 0.0))
            for left, right in normalized_edges
        )
        object.__setattr__(self, "edge_weights", normalized_weights)
        raw_root = self.root_mass
        if isinstance(raw_root, Mapping):
            root_map: dict[str, float] = {}
            for raw_key, raw_value in raw_root.items():
                key = str(raw_key)
                if key in root_map:
                    raise ValueError("root_mass keys must remain unique after string normalization")
                root_map[key] = _strict_float_value(raw_value, "root mass", nonnegative=True)
            unknown = set(root_map) - id_set
            if unknown:
                raise ValueError(f"root mass references unknown IDs: {sorted(unknown)}")
            root = tuple(root_map.get(identifier, 0.0) for identifier in ids)
        else:
            raw_root_values = tuple(
                _strict_float_value(value, "root mass", nonnegative=True) for value in raw_root
            )
            if len(raw_root_values) not in {0, len(raw_ids)}:
                raise ValueError("root_mass must have one value per memory ID")
            if raw_root_values:
                raw_by_id = dict(zip(raw_ids, raw_root_values))
                root = tuple(raw_by_id[identifier] for identifier in ids)
            else:
                root = tuple(0.0 for _ in ids)
        if any(not np.isfinite(value) or value < 0.0 for value in root):
            raise ValueError("root mass must be finite and non-negative")
        object.__setattr__(self, "root_mass", root)
        # Preserve layer depth, but canonicalize member order within each
        # layer.  This retains the protocol's synchronous semantics while
        # making equivalent input permutations hash-identical.
        raw_layers = self.layers.items() if isinstance(self.layers, Mapping) else self.layers
        if isinstance(self.layers, Mapping):
            # A depth -> IDs mapping is useful when reconstructing a graph
            # from JSON; preserve numeric/key order while canonicalizing IDs
            # within each layer.
            raw_layers = [item[1] for item in sorted(self.layers.items(), key=lambda item: str(item[0]))]
        normalized_layers = tuple(
            tuple(sorted(str(value) for value in layer)) for layer in (raw_layers or ())
        )
        layer_seen: set[str] = set()
        for layer in normalized_layers:
            if len(set(layer)) != len(layer) or any(value not in id_set for value in layer):
                raise ValueError("layers must contain unique declared memory IDs")
            if layer_seen.intersection(layer):
                raise ValueError("a memory may occur in only one graph layer")
            layer_seen.update(layer)
        if normalized_layers and layer_seen != id_set:
            raise ValueError("layers must cover every graph memory ID")
        if normalized_layers:
            layer_index = {
                identifier: depth
                for depth, layer in enumerate(normalized_layers)
                for identifier in layer
            }
            for parent, child in normalized_edges:
                if layer_index[parent] >= layer_index[child]:
                    raise ValueError("graph edges must point from an earlier to a later layer")
        else:
            # A frozen proposal graph is a DAG even when callers omit an
            # explicit layer annotation.  Reject cycles at the type boundary
            # rather than allowing propagation to depend on an arbitrary ID
            # order later on.
            indegree = {identifier: 0 for identifier in ids}
            children: dict[str, list[str]] = {identifier: [] for identifier in ids}
            for parent, child in normalized_edges:
                indegree[child] += 1
                children[parent].append(child)
            queue = sorted(identifier for identifier, degree in indegree.items() if degree == 0)
            visited = 0
            while queue:
                current = queue.pop(0)
                visited += 1
                for child in sorted(children[current]):
                    indegree[child] -= 1
                    if indegree[child] == 0:
                        queue.append(child)
                        queue.sort()
            if visited != len(ids):
                raise ValueError("frozen graph edges must form an acyclic graph")
        object.__setattr__(self, "layers", normalized_layers)
        raw_sources = self.parent_sources.items() if isinstance(self.parent_sources, Mapping) else self.parent_sources
        sources = []
        source_keys: set[str] = set()
        try:
            source_items = tuple(raw_sources or ())
        except TypeError as exc:
            raise ValueError("parent_sources must be a mapping or sequence") from exc
        for item in source_items:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError("parent_sources entries must be (candidate, parents)")
            candidate, parents = item
            candidate_id = str(candidate)
            if candidate_id not in id_set:
                raise ValueError("parent source candidate is not in graph")
            if candidate_id in source_keys:
                raise ValueError("parent source IDs must remain unique after string normalization")
            source_keys.add(candidate_id)
            if isinstance(parents, (str, bytes)):
                raise ValueError("parent source parents must be a sequence of IDs")
            try:
                parent_ids = tuple(str(value) for value in parents)
            except TypeError as exc:
                raise ValueError("parent source parents must be a sequence of IDs") from exc
            if (
                len(set(parent_ids)) != len(parent_ids)
                or any(not value for value in parent_ids)
                or any(value not in id_set for value in parent_ids)
                or candidate_id in parent_ids
            ):
                raise ValueError("parent source IDs must be unique graph IDs")
            sources.append((candidate_id, tuple(sorted(parent_ids))))
        object.__setattr__(self, "parent_sources", tuple(sorted(sources)))
        source_pairs = {
            (parent, candidate)
            for candidate, parents in sources
            for parent in parents
        }
        if source_pairs - set(normalized_edges):
            raise ValueError("parent_sources reference edges absent from the frozen graph")
        records = []
        if isinstance(self.proposal_records, (str, bytes)):
            raise ValueError("proposal_records must be a sequence")
        try:
            raw_records = tuple(self.proposal_records)
        except TypeError as exc:
            raise ValueError("proposal_records must be a sequence") from exc
        for record in raw_records:
            if not isinstance(record, (tuple, list)) or len(record) != 5:
                raise ValueError("proposal_records entries must be (parent, candidate, layer, rank, score)")
            parent, candidate = str(record[0]), str(record[1])
            score = _strict_float_value(record[4], "proposal score")
            layer = _strict_int_value(record[2], "proposal layer", positive=True)
            rank = _strict_int_value(record[3], "proposal rank", nonnegative=True)
            if parent not in id_set or candidate not in id_set:
                raise ValueError("proposal record references unknown ID")
            if (parent, candidate) not in seen_edges:
                raise ValueError("proposal record references an undeclared graph edge")
            if normalized_layers:
                layer_index = {
                    identifier: depth
                    for depth, layer in enumerate(normalized_layers)
                    for identifier in layer
                }
                if layer_index[parent] >= layer_index[candidate] or layer != layer_index[candidate] + 1:
                    raise ValueError("proposal record layer does not match graph layers")
            records.append((parent, candidate, layer, rank, score))
        # Proposal exposures are a multiset (duplicate parent/rank records are
        # meaningful), so sort rather than deduplicate them.
        canonical_records = tuple(
            sorted(records, key=lambda item: (item[2], item[0], item[3], item[1], item[4]))
        )
        object.__setattr__(self, "proposal_records", canonical_records)
        object.__setattr__(self, "proposal_config", _freeze_value(dict(self.proposal_config)))
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, Integral)
            or self.schema_version <= 0
        ):
            raise ValueError("schema_version must be a positive integer")
        domain_scope = str(self.domain_scope)
        if not domain_scope.strip():
            raise ValueError("graph domain_scope must be non-empty")
        object.__setattr__(self, "domain_scope", domain_scope)
        # Always derive the identity from the canonical frozen payload.  A
        # caller-supplied hash is useful as a consistency assertion, but must
        # never be allowed to certify a graph whose edges/order/provenance do
        # not match that hash.
        payload = {
            "memory_ids": ids,
            "edges": normalized_edges,
            "edge_weights": normalized_weights,
            "root_mass": root,
            "layers": normalized_layers,
            "parent_sources": tuple(sorted(sources)),
            "proposal_records": canonical_records,
            "domain_scope": domain_scope,
            "cutoff": _jsonable(self.cutoff),
            "proposal_config": _jsonable(dict(self.proposal_config)),
            "schema_version": self.schema_version,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        computed_hash = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        supplied_hash = str(self.graph_hash or "")
        if supplied_hash and supplied_hash != computed_hash:
            raise ValueError("graph_hash does not match the canonical frozen graph")
        object.__setattr__(self, "graph_hash", computed_hash)

    @property
    def edge_weight_map(self) -> dict[tuple[str, str], float]:
        return {(left, right): float(weight) for left, right, weight in self.edge_weights}

    @property
    def parent_map(self) -> dict[str, tuple[str, ...]]:
        result: dict[str, list[str]] = {identifier: [] for identifier in self.memory_ids}
        for left, right in self.edges:
            result.setdefault(right, []).append(left)
        return {key: tuple(sorted(value)) for key, value in result.items()}

    @property
    def children_map(self) -> dict[str, tuple[str, ...]]:
        result: dict[str, list[str]] = {identifier: [] for identifier in self.memory_ids}
        for left, right in self.edges:
            result.setdefault(left, []).append(right)
        return {key: tuple(sorted(value)) for key, value in result.items()}

    @property
    def root_ids(self) -> tuple[str, ...]:
        # A declared zero-weight exposure is retained for provenance but is
        # not a navigation transition.  With synchronous layers, only the
        # first layer is the source distribution; an isolated later-layer
        # candidate is unreachable rather than an implicit root.  Unlayered
        # graphs retain the structural-root interpretation.
        if self.layers and self.layers[0]:
            return tuple(self.layers[0])
        positive_parents = {identifier: False for identifier in self.memory_ids}
        for _parent, child, weight in self.edge_weights:
            if float(weight) > 0.0:
                positive_parents[child] = True
        return tuple(identifier for identifier in self.memory_ids if not positive_parents[identifier])

    def public_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "memory_ids": list(self.memory_ids),
            "edges": [list(edge) for edge in self.edges],
            "edge_weights": [list(edge) for edge in self.edge_weights],
            "root_mass": list(self.root_mass),
            "root_ids": list(self.root_ids),
            "graph_hash": self.graph_hash,
            "domain_scope": self.domain_scope,
            "layers": [list(layer) for layer in self.layers],
            "parent_sources": {candidate: list(parents) for candidate, parents in self.parent_sources},
            "proposal_records": [list(record) for record in self.proposal_records],
            "cutoff": _jsonable(self.cutoff),
            "proposal_config": _jsonable(dict(self.proposal_config)),
        }

    def freeze(self) -> "FrozenProposalGraph":
        return self


@dataclass(frozen=True)
class SemanticAtom:
    """Frozen rank-one PSD semantic feature for one memory."""

    memory_id: str
    graph_hash: str
    quality: float
    feature: np.ndarray
    feature_hash: str = ""
    path_ids: tuple[str, ...] = ()
    representation_hash: str = ""

    def __post_init__(self) -> None:
        identifier = str(self.memory_id)
        if not identifier:
            raise ValueError("semantic atom memory_id cannot be empty")
        object.__setattr__(self, "memory_id", identifier)
        quality = _strict_float_value(self.quality, "semantic atom quality")
        if quality < 0.0 or quality > 1.0:
            raise ValueError("semantic atom quality must lie in [0, 1]")
        object.__setattr__(self, "quality", quality)
        vector = np.array(self.feature, dtype=np.float64, copy=True).reshape(-1)
        if vector.size == 0 or not np.all(np.isfinite(vector)):
            raise ValueError("semantic atom feature must be a non-empty finite vector")
        vector.setflags(write=False)
        object.__setattr__(self, "feature", vector)
        object.__setattr__(self, "graph_hash", str(self.graph_hash))
        object.__setattr__(self, "path_ids", tuple(str(value) for value in self.path_ids))
        object.__setattr__(self, "representation_hash", str(self.representation_hash))
        computed_hash = hashlib.sha256(vector.tobytes()).hexdigest()
        feature_hash = str(self.feature_hash)
        if feature_hash and feature_hash != computed_hash:
            raise ValueError("semantic atom feature_hash does not match its feature")
        if not feature_hash:
            feature_hash = computed_hash
        object.__setattr__(self, "feature_hash", feature_hash)

    @property
    def matrix(self) -> np.ndarray:
        return np.outer(np.asarray(self.feature), np.asarray(self.feature))

    @property
    def norm_sq(self) -> float:
        return float(np.dot(self.feature, self.feature))

    def public_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "graph_hash": self.graph_hash,
            "quality": self.quality,
            "feature": self.feature.tolist(),
            "feature_hash": self.feature_hash,
            "path_ids": list(self.path_ids),
            "representation_hash": self.representation_hash,
        }


@dataclass(frozen=True)
class ContextPlan:
    """The sole source of truth for the context sent to a generator."""

    selected_ids: tuple[str, ...]
    chronological_ids: tuple[str, ...]
    serialized_context: str
    messages: tuple[Mapping[str, str], ...]
    token_count: int
    token_count_is_estimate: bool
    budget: int | None
    budget_status: str
    prompt_hash: str
    context_hash: str
    request: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.selected_ids, (str, bytes)) or not isinstance(self.selected_ids, Sequence):
            raise ValueError("ContextPlan selected_ids must be a sequence")
        if isinstance(self.chronological_ids, (str, bytes)) or not isinstance(self.chronological_ids, Sequence):
            raise ValueError("ContextPlan chronological_ids must be a sequence")
        selected_values = []
        for value in self.selected_ids:
            if not isinstance(value, str) or not value:
                raise ValueError("ContextPlan selected_ids must contain non-empty strings")
            selected_values.append(value)
        chronological_values = []
        for value in self.chronological_ids:
            if not isinstance(value, str) or not value:
                raise ValueError("ContextPlan chronological_ids must contain non-empty strings")
            chronological_values.append(value)
        selected = tuple(selected_values)
        chronological = tuple(chronological_values)
        if len(selected) != len(set(selected)) or len(chronological) != len(set(chronological)):
            raise ValueError("ContextPlan IDs must be unique")
        if set(selected) != set(chronological):
            raise ValueError("chronological_ids must contain exactly selected_ids")
        object.__setattr__(self, "selected_ids", selected)
        object.__setattr__(self, "chronological_ids", chronological)
        if not isinstance(self.token_count_is_estimate, bool):
            raise ValueError("ContextPlan token_count_is_estimate must be boolean")
        token_count = _strict_int_value(self.token_count, "ContextPlan token_count", nonnegative=True)
        object.__setattr__(self, "token_count", token_count)
        budget = None if self.budget is None else _strict_int_value(self.budget, "ContextPlan budget", nonnegative=True)
        object.__setattr__(self, "budget", budget)
        if not isinstance(self.serialized_context, str):
            raise ValueError("ContextPlan serialized_context must be a string")
        serialized_context = self.serialized_context
        object.__setattr__(self, "serialized_context", serialized_context)
        if isinstance(self.messages, (str, bytes)) or not isinstance(self.messages, Sequence):
            raise ValueError("ContextPlan messages must be a sequence")
        normalized_messages = []
        for message in self.messages:
            if not isinstance(message, Mapping) or "role" not in message or "content" not in message:
                raise ValueError("ContextPlan messages must contain role and content")
            if not isinstance(message["role"], str) or not isinstance(message["content"], str):
                raise ValueError("ContextPlan message role and content must be strings")
            normalized_messages.append(
                MappingProxyType({"role": message["role"], "content": message["content"]})
            )
        object.__setattr__(self, "messages", tuple(normalized_messages))
        expected_context = (
            normalized_messages[1]["content"]
            if len(normalized_messages) > 1
            else normalized_messages[0]["content"]
            if normalized_messages
            else ""
        )
        if serialized_context != expected_context:
            raise ValueError("ContextPlan serialized_context does not match its messages")
        if (
            not isinstance(self.budget_status, str)
            or self.budget_status not in {"within_budget", "exceeds_budget", "unknown"}
        ):
            raise ValueError("ContextPlan budget_status is invalid")
        # A measured token count is part of the frozen contract.  For the
        # deterministic estimate used by this package, reject a tampered
        # count at the boundary; callers that use a service tokenizer can set
        # ``token_count_is_estimate=False`` and retain their authoritative
        # count instead.
        if self.token_count_is_estimate:
            estimated = sum(
                len(re.findall(r"\w+|[^\w\s]", message["content"], flags=re.UNICODE))
                for message in normalized_messages
            )
            if int(self.token_count) != estimated:
                raise ValueError("ContextPlan token_count does not match its messages")
        if self.budget_status != "unknown":
            expected_status = (
                "within_budget"
                if self.budget is None or int(self.token_count) <= int(self.budget)
                else "exceeds_budget"
            )
            if self.budget_status != expected_status:
                raise ValueError("ContextPlan budget_status does not match token_count and budget")
        # Freeze the complete nested HTTP payload.  A shallow
        # ``MappingProxyType(dict(...))`` still leaves message lists and extra
        # request options mutable after a context hash has been recorded.
        if isinstance(self.request, Mapping):
            try:
                request_jsonable = _jsonable(dict(self.request))
                _assert_jsonable(request_jsonable)
            except (TypeError, ValueError) as exc:
                raise ValueError("ContextPlan request is not JSON-serializable") from exc
            frozen_request = _freeze_value(request_jsonable)
        else:
            frozen_request = None
        if frozen_request is None or not isinstance(frozen_request, Mapping):
            raise ValueError("ContextPlan request must be a mapping")
        # The request must carry the exact messages that are hashed and sent.
        # Older hand-built plans sometimes omitted this field; accepting one
        # would force the generator to synthesize a payload at send time and
        # is precisely the silent context mutation this type is meant to
        # prevent.
        if "messages" not in frozen_request:
            raise ValueError("ContextPlan request must include messages")
        def _find_sensitive_keys(value: Any, path: str = "") -> list[str]:
            found: list[str] = []
            if isinstance(value, Mapping):
                for raw_key, child in value.items():
                    key = str(raw_key)
                    child_path = f"{path}.{key}" if path else key
                    if key.lower() in _SENSITIVE_REQUEST_KEYS:
                        found.append(child_path)
                    else:
                        found.extend(_find_sensitive_keys(child, child_path))
            elif isinstance(value, (tuple, list)):
                for index, child in enumerate(value):
                    found.extend(_find_sensitive_keys(child, f"{path}[{index}]"))
            return found

        sensitive_request_keys = _find_sensitive_keys(frozen_request)
        if sensitive_request_keys:
            raise ValueError("ContextPlan request must not contain credentials")
        request_messages = frozen_request["messages"]
        if not isinstance(request_messages, (tuple, list)):
            raise ValueError("ContextPlan request messages must be a sequence")
        normalized_request_messages = []
        for item in request_messages:
            if not isinstance(item, Mapping) or "role" not in item or "content" not in item:
                raise ValueError("ContextPlan request messages must contain role and content")
            if not isinstance(item["role"], str) or not isinstance(item["content"], str):
                raise ValueError("ContextPlan request message role and content must be strings")
            normalized_request_messages.append(
                MappingProxyType({"role": item["role"], "content": item["content"]})
            )
        normalized_request_messages = tuple(normalized_request_messages)
        if tuple(dict(item) for item in normalized_request_messages) != tuple(
            dict(item) for item in normalized_messages
        ):
            raise ValueError("ContextPlan request messages must equal plan messages")
        # Preserve the immutable normalized representation in the request.
        mutable_request = dict(frozen_request)
        mutable_request["messages"] = normalized_request_messages
        if "model" not in mutable_request:
            raise ValueError("ContextPlan request must include model")
        if "temperature" not in mutable_request:
            raise ValueError("ContextPlan request must include temperature")
        if "max_tokens" not in mutable_request:
            raise ValueError("ContextPlan request must include max_tokens")
        if not isinstance(mutable_request["model"], str):
            raise ValueError("ContextPlan request model must be a string")
        if "endpoint" in mutable_request and not isinstance(mutable_request["endpoint"], str):
            raise ValueError("ContextPlan request endpoint must be a string")
        try:
            temperature = _strict_float_value(
                mutable_request["temperature"], "ContextPlan request temperature", nonnegative=True
            )
            max_tokens = _strict_int_value(
                mutable_request["max_tokens"], "ContextPlan request max_tokens", nonnegative=True
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("ContextPlan request decoding fields are invalid") from exc
        mutable_request["temperature"] = temperature
        mutable_request["max_tokens"] = max_tokens
        frozen_request = _freeze_value(mutable_request)
        # Prompt/context hashes are fixed-width identities.  The prompt hash
        # may refer to a caller-supplied prompt version, so validate its
        # shape here and verify the context hash against the complete payload
        # below rather than forcing the package's default prompt hash.
        if not isinstance(self.prompt_hash, str) or not isinstance(self.context_hash, str):
            raise ValueError("ContextPlan hashes must be strings")
        prompt_hash = self.prompt_hash.lower()
        context_hash = self.context_hash.lower()
        if not re.fullmatch(r"[0-9a-fA-F]{64}", prompt_hash):
            raise ValueError("ContextPlan prompt_hash must be a SHA-256 hex digest")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", context_hash):
            raise ValueError("ContextPlan context_hash must be a SHA-256 hex digest")
        object.__setattr__(self, "prompt_hash", prompt_hash)
        object.__setattr__(self, "context_hash", context_hash)
        expected_hash = context_plan_hash(
            selected_ids=selected,
            chronological_ids=chronological,
            serialized_context=self.serialized_context,
            messages=normalized_messages,
            token_count=int(self.token_count),
            token_count_is_estimate=self.token_count_is_estimate,
            budget=self.budget,
            budget_status=str(self.budget_status),
            prompt_hash=prompt_hash,
            request=frozen_request,
        )
        if context_hash != expected_hash:
            raise ValueError("ContextPlan context_hash does not match its content and request")
        object.__setattr__(self, "request", frozen_request)

    @property
    def within_budget(self) -> bool:
        return self.budget is None or self.token_count <= self.budget

    @property
    def http_payload(self) -> Mapping[str, Any]:
        """Return the immutable payload that should be sent to the generator."""
        return self.request

    def request_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly copy safe for transport/logging.

        The generator client strips the endpoint before posting; returning a
        hashed endpoint here prevents an adapter or debug logger from
        accidentally persisting a private URL.
        """
        return _safe_request_value(dict(self.request))

    def public_dict(self) -> Dict[str, Any]:
        return {
            "selected_ids": list(self.selected_ids),
            "chronological_ids": list(self.chronological_ids),
            "serialized_context": self.serialized_context,
            "messages": [dict(message) for message in self.messages],
            "token_count": self.token_count,
            "token_count_is_estimate": self.token_count_is_estimate,
            "budget": self.budget,
            "budget_status": self.budget_status,
            "prompt_hash": self.prompt_hash,
            "context_hash": self.context_hash,
            # Do not persist raw transport endpoints or credentials.  The
            # redacted request still contains the exact messages/decoding
            # fields and hashes to the same ContextPlan identity.
            "request": _safe_request_value(dict(self.request)),
        }

    @classmethod
    def from_public_dict(cls, value: Mapping[str, Any]) -> "ContextPlan":
        """Restore a frozen plan; never repair, reformat or silently drop fields."""
        if not isinstance(value, Mapping):
            raise ValueError("ContextPlan snapshot must be a mapping")
        expected = set(cls.__dataclass_fields__)
        if set(value) != expected:
            raise ValueError("ContextPlan snapshot fields differ from the frozen contract")
        # __post_init__ rechecks messages, token accounting, payload and hashes.
        return cls(**dict(value))


@dataclass(frozen=True)
class Memory:
    memory_id: str
    text: str
    timestamp: float
    source_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def time_metadata(self) -> Dict[str, Any]:
        """Normalized temporal metadata (empty for legacy memories)."""
        value = self.metadata.get("time", {})
        return dict(value) if isinstance(value, Mapping) else {}


@dataclass(frozen=True)
class PathHypothesis:
    """A posterior-supported path through real memories."""

    path_ids: Tuple[str, ...]
    parent_posterior: Dict[str, float]
    branch_id: str
    posterior: float
    support: float

    def __post_init__(self) -> None:
        path_ids = tuple(str(value) for value in self.path_ids)
        if not path_ids:
            raise ValueError("path hypothesis must contain at least one memory")
        if len(set(path_ids)) != len(path_ids):
            raise ValueError("path hypothesis cannot contain cyclic/duplicate memory IDs")
        posterior = _strict_float_value(self.posterior, "path posterior", nonnegative=True)
        support = _strict_float_value(self.support, "path support", nonnegative=True)
        if posterior < 0.0:
            raise ValueError("path posterior must be finite and non-negative")
        if support < 0.0:
            raise ValueError("path support must be finite and non-negative")
        if not isinstance(self.parent_posterior, Mapping):
            raise ValueError("parent posterior must be a mapping")
        parent_map: dict[str, float] = {}
        for raw_parent, raw_value in self.parent_posterior.items():
            parent_id = str(raw_parent)
            if parent_id in parent_map:
                raise ValueError("parent IDs must remain unique after string normalization")
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError("parent posterior values must be finite and non-negative") from exc
            if value < 0.0 or not np.isfinite(value):
                raise ValueError("parent posterior values must be finite and non-negative")
            parent_map[parent_id] = value
        # A non-empty parent map is a conditional posterior, rather than an
        # arbitrary collection of edge scores.  Requiring normalization at
        # the type boundary catches malformed hand-built/serialized paths;
        # internal constructors already normalize maps before instantiation.
        if parent_map and not np.isclose(sum(parent_map.values()), 1.0, atol=1e-8, rtol=1e-8):
            raise ValueError("parent posterior must sum to one")
        object.__setattr__(self, "path_ids", path_ids)
        object.__setattr__(self, "parent_posterior", MappingProxyType(parent_map))
        object.__setattr__(self, "branch_id", str(self.branch_id))
        object.__setattr__(self, "posterior", posterior)
        object.__setattr__(self, "support", support)

    def public_dict(self) -> Dict[str, Any]:
        return {
            "path_ids": list(self.path_ids),
            "parent_posterior": {str(key): float(value) for key, value in self.parent_posterior.items()},
            "branch_id": self.branch_id,
            "posterior": float(self.posterior),
            "support": float(self.support),
        }

@dataclass(frozen=True)
class InformationAtom:
    """One candidate/path PSD information contribution."""

    candidate_id: str
    path_ids: Tuple[str, ...]
    matrix: np.ndarray
    trace: float
    support: float

    def __post_init__(self) -> None:
        candidate_id = str(self.candidate_id)
        if not candidate_id:
            raise ValueError("information atom candidate_id cannot be empty")
        matrix = np.asarray(self.matrix, dtype=np.float64).copy()
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("information atom matrix must be square")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("information atom matrix must be finite")
        trace = _strict_float_value(self.trace, "information atom trace", nonnegative=True)
        if trace < -1e-12:
            raise ValueError("information atom trace must be finite and non-negative")
        support = _strict_float_value(self.support, "information atom support", nonnegative=True)
        if support < 0.0:
            raise ValueError("information atom support must be finite and non-negative")
        actual_trace = float(np.trace((matrix + matrix.T) * 0.5))
        if abs(actual_trace - trace) > 1e-8 * max(1.0, abs(actual_trace)):
            raise ValueError("information atom trace does not match its matrix")
        # Information atoms are PSD by definition.  Keep the check here as
        # well as in the numerical constructor so deserialized or manually
        # assembled provenance cannot silently claim an indefinite atom.
        if matrix.size:
            minimum = float(np.min(np.linalg.eigvalsh((matrix + matrix.T) * 0.5)))
            scale = max(1.0, float(np.max(np.abs(matrix))))
            if minimum < -1e-10 * scale:
                raise ValueError("information atom matrix must be positive semidefinite")
        matrix.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "path_ids", tuple(str(value) for value in self.path_ids))
        object.__setattr__(self, "trace", trace)
        object.__setattr__(self, "support", support)

    def public_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "path_ids": list(self.path_ids),
            "matrix": np.asarray(self.matrix, dtype=np.float64).tolist(),
            "trace": float(self.trace),
            "support": float(self.support),
        }

    @property
    def min_eigenvalue(self) -> float:
        matrix = np.asarray(self.matrix, dtype=np.float64)
        return float(np.min(np.linalg.eigvalsh((matrix + matrix.T) * 0.5))) if matrix.size else 0.0

    @property
    def is_psd(self) -> bool:
        return self.min_eigenvalue >= -1e-10


@dataclass
class TreeNode:
    memory: Memory
    vector: np.ndarray
    parent_id: Optional[str]
    depth: int
    direct_score: float
    reachability: float
    innovation: np.ndarray
    bridge_lift: float
    discovery_order: int
    # Trailing defaults preserve every legacy constructor while exposing the
    # complete posterior/path representation used by M=true.
    parent_posterior: Dict[str, float] = field(default_factory=dict)
    path_hypotheses: Tuple[PathHypothesis, ...] = ()
    transition_support: float = 0.0
    # Semantic-path fields are intentionally separate from ``reachability``.
    # The latter remains the legacy navigation/capacity quantity; a frozen
    # task scorer owns ``semantic_quality`` and may assign a high value to a
    # deep candidate regardless of its discovery mass.
    semantic_quality: float | None = None
    temporal_mark: TemporalMark | None = None
    proposal_sources: Tuple[str, ...] = ()
    layer: int | None = None
    # Expected ancestor occupancy for this candidate.  This is the compact
    # DP representation of M=true provenance; it avoids forcing callers to
    # enumerate an exponential set of complete paths.
    ancestor_occupancy: Dict[str, float] = field(default_factory=dict)

    @property
    def paths(self) -> Tuple[PathHypothesis, ...]:
        """Compatibility alias for the posterior-supported paths."""
        return self.path_hypotheses

    def public_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory.memory_id,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "direct_score": self.direct_score,
            "reachability": self.reachability,
            "bridge_lift": self.bridge_lift,
            "innovation_norm_sq": float(np.dot(self.innovation, self.innovation)),
            "discovery_order": self.discovery_order,
            "parent_posterior": {str(key): float(value) for key, value in self.parent_posterior.items()},
            "path_hypotheses": [path.public_dict() for path in self.path_hypotheses],
            "transition_support": float(self.transition_support),
            "semantic_quality": None if self.semantic_quality is None else float(self.semantic_quality),
            "temporal_mark": None if self.temporal_mark is None else self.temporal_mark.public_dict(),
            "proposal_sources": list(self.proposal_sources),
            "layer": self.layer,
            "ancestor_occupancy": {
                str(key): float(value) for key, value in self.ancestor_occupancy.items()
            },
        }


@dataclass
class Branch:
    branch_id: str
    member_ids: Tuple[str, ...]
    probe: np.ndarray
    path_upper_bound: float
    marginal_upper_bound: float
    radius_radians: float
    depth: int
    creation_order: int
    member_mass: Dict[str, float] = field(default_factory=dict)
    pi: Dict[str, float] = field(default_factory=dict)
    support: Dict[str, float] = field(default_factory=dict)
    domain_ids: Tuple[str, ...] = ()
    support_upper: float | None = None
    bound_ingredients: Dict[str, Any] = field(default_factory=dict)
    zero_mass_uniformized: bool = False

    @property
    def mass(self) -> Dict[str, float]:
        """Raw member masses (``a_i``) used to construct ``pi``."""
        return self.member_mass

    @property
    def sB(self) -> Dict[str, float]:
        """Propagated branch support ``s_B(j)``."""
        return self.support

    def public_dict(self) -> Dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "member_ids": list(self.member_ids),
            "path_upper_bound": self.path_upper_bound,
            "marginal_upper_bound": self.marginal_upper_bound,
            "radius_radians": self.radius_radians,
            "depth": self.depth,
            "creation_order": self.creation_order,
            "member_mass": {str(key): float(value) for key, value in self.member_mass.items()},
            "pi": {str(key): float(value) for key, value in self.pi.items()},
            "support": {str(key): float(value) for key, value in self.support.items()},
            "domain_ids": list(self.domain_ids),
            "support_upper": None if self.support_upper is None else float(self.support_upper),
            "bound_ingredients": _jsonable(self.bound_ingredients),
            "zero_mass_uniformized": self.zero_mass_uniformized,
        }


@dataclass(frozen=True)
class SelectionStep:
    step: int
    memory_id: str
    discovered_best_margin: float
    unseen_upper_bound: float
    epsilon: float
    certified: bool


@dataclass
class RetrievalResult:
    query: str
    selected: List[Memory]
    selected_in_greedy_order: List[str]
    nodes: Dict[str, TreeNode]
    edges: List[Tuple[Optional[str], str]]
    all_branches: List[Branch]
    remaining_branches: List[Branch]
    selection_steps: List[SelectionStep]
    cost_tracker: CostTracker
    budget_frozen: bool
    cluster_radii: List[float]
    cluster_stabilities: List[float]
    cluster_member_counts: List[int] = field(default_factory=list)
    clustering_ms: float = 0.0
    first_arrival_semantics: str = "deterministic_first_arrival"
    transition: np.ndarray | None = None
    path_hypotheses: Dict[str, Tuple[PathHypothesis, ...]] = field(default_factory=dict)
    information_atoms: Dict[str, InformationAtom] = field(default_factory=dict)
    domains: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    bounds: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    certificate_status: str = "not_requested"
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    context_hash: str | None = None
    # Semantic execution provenance.  Defaults keep positional construction of
    # legacy results source-compatible.
    visible_bank: Tuple[str, ...] = ()
    proposal_domain: Tuple[str, ...] = ()
    selected_context: Tuple[str, ...] = ()
    frozen_graph: FrozenProposalGraph | None = None
    semantic_atoms: Dict[str, SemanticAtom] = field(default_factory=dict)
    context_plan: ContextPlan | None = None
    residual_certification_gaps: Tuple[float, ...] = ()
    # ID-addressed sparse transition for semantic-path consumers.  The
    # legacy ``transition`` ndarray remains available for old readers; new
    # code should use this field so a large frozen graph does not require a
    # dense N×N representation merely to inspect provenance.
    transition_sparse: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    @property
    def cost(self) -> CostSnapshot:
        return self.cost_tracker.snapshot()

    @property
    def ann_calls(self) -> int:
        return self.cost.ann_calls_core

    @property
    def visited_nodes(self) -> int:
        return len(self.nodes)

    @property
    def stop_reason(self) -> str:
        return self.cost.stop_reason

    @property
    def posterior_paths(self) -> Dict[str, Tuple[PathHypothesis, ...]]:
        """Explicit alias for the formal M=true path representation."""
        return self.path_hypotheses

    @property
    def paths(self) -> Dict[str, Tuple[PathHypothesis, ...]]:
        """Short alias used by the mathematical specification and notebooks."""
        return self.path_hypotheses

    @property
    def transition_matrix(self) -> np.ndarray | None:
        """Alias for the stored row-stochastic transition matrix."""
        return self.transition

    @property
    def mass(self) -> Dict[str, Dict[str, float]]:
        """Branch propagated masses (``s_B``) keyed by branch ID."""
        return {branch.branch_id: dict(branch.support) for branch in self.all_branches}

    def diagnostic_summary(self, gold_ids: List[str] | None = None) -> Dict[str, Any]:
        depth: Dict[str, Dict[str, float]] = {}
        independent_gold = set(gold_ids or ())
        for level in sorted({node.depth for node in self.nodes.values()}):
            nodes = [node for node in self.nodes.values() if node.depth == level]
            root_cosines = [node.direct_score for node in nodes]
            values = {
                "nodes": float(len(nodes)),
                "mean_root_cosine": float(np.mean(root_cosines)),
                "min_root_cosine": float(min(root_cosines)),
            }
            if gold_ids is not None:
                values["gold_rate"] = sum(node.memory.memory_id in independent_gold for node in nodes) / len(nodes)
            depth[str(level)] = values
        parent_child = [
            float(np.dot(node.vector, self.nodes[node.parent_id].vector))
            for node in self.nodes.values()
            if node.parent_id is not None
        ]
        selected = set(self.selected_in_greedy_order)
        navigation_parents = set()
        for memory_id in selected:
            parent_id = self.nodes[memory_id].parent_id
            while parent_id is not None:
                navigation_parents.add(parent_id)
                parent_id = self.nodes[parent_id].parent_id
        navigation_only = navigation_parents - selected
        selected_deep = sum(self.nodes[memory_id].depth > 1 for memory_id in selected)
        return {
            "tree_semantics": self.first_arrival_semantics,
            "depth": depth,
            "mean_parent_child_cosine": float(np.mean(parent_child)) if parent_child else None,
            "min_parent_child_cosine": float(min(parent_child)) if parent_child else None,
            "selected_deep_node_rate": selected_deep / len(selected) if selected else 0.0,
            "navigation_only_parent_rate": (
                len(navigation_only) / len(navigation_parents) if navigation_parents else 0.0
            ),
            "duplicate_proposal_rate": (
                self.cost.duplicate_proposals / self.cost.proposal_count if self.cost.proposal_count else 0.0
            ),
            "raw_return_exposure": self.cost.raw_return_exposure,
            "unique_admissions": self.cost.unique_admissions,
            "unadmitted_exposure_count": self.cost.unadmitted_exposure_count,
            "new_unique_candidates_per_ann": self.cost.new_unique_candidates_per_ann,
            "new_unique_candidates_by_ann": list(self.cost.new_unique_candidates_by_ann),
            "actual_cluster_count": len(self.cluster_member_counts),
            "cluster_member_counts": self.cluster_member_counts,
            "clustering_ms": self.clustering_ms,
        }

    @property
    def certified(self) -> bool:
        return bool(self.selection_steps) and all(step.certified for step in self.selection_steps)

    @property
    def posterior_error(self) -> float:
        k = len(self.selection_steps)
        if k == 0:
            return 0.0
        return float(sum(((1.0 - 1.0 / k) ** (k - step.step)) * step.epsilon for step in self.selection_steps))

    def to_dict(self, include_text: bool = True) -> Dict[str, Any]:
        selected = []
        for memory in self.selected:
            item = asdict(memory)
            if not include_text:
                item.pop("text", None)
            selected.append(item)
        return {
            "provenance_schema": 2,
            "query": self.query,
            "selected": selected,
            "selected_in_greedy_order": self.selected_in_greedy_order,
            "greedy_ids": list(self.selected_in_greedy_order),
            "chronological_ids": [memory.memory_id for memory in self.selected],
            "nodes": [node.public_dict() for node in self.nodes.values()],
            "edges": self.edges,
            "all_branches": [branch.public_dict() for branch in self.all_branches],
            "remaining_branches": [branch.public_dict() for branch in self.remaining_branches],
            "selection_steps": [asdict(step) for step in self.selection_steps],
            "ann_calls": self.ann_calls,
            "ann_calls_core": self.cost.ann_calls_core,
            "ann_calls_diagnostic": self.cost.ann_calls_diagnostic,
            "visited_nodes": self.visited_nodes,
            "budget_frozen": self.budget_frozen,
            "stop_reason": self.stop_reason,
            "cost": self.cost.to_dict(),
            "certified": self.certified,
            "posterior_error": self.posterior_error,
            "cluster_radii": self.cluster_radii,
            "cluster_stabilities": self.cluster_stabilities,
            "cluster_member_counts": self.cluster_member_counts,
            "clustering_ms": self.clustering_ms,
            "tree_semantics": self.first_arrival_semantics,
            "diagnostic": self.diagnostic_summary(),
            "transition": None if self.transition is None else np.asarray(self.transition).tolist(),
            "transition_matrix": None if self.transition is None else np.asarray(self.transition).tolist(),
            "transition_shape": None if self.transition is None else list(np.asarray(self.transition).shape),
            "transition_sparse": {
                str(parent): {str(child): float(probability) for child, probability in row.items()}
                for parent, row in self.transition_sparse.items()
            },
            "mass": self.mass,
            "paths": {
                str(memory_id): [path.public_dict() for path in paths]
                for memory_id, paths in self.path_hypotheses.items()
            },
            # Explicit alias used by newer consumers; both names are emitted
            # so old JSON readers remain valid.
            "posterior_paths": {
                str(memory_id): [path.public_dict() for path in paths]
                for memory_id, paths in self.path_hypotheses.items()
            },
            "information_atoms": {
                str(memory_id): atom.public_dict() for memory_id, atom in self.information_atoms.items()
            },
            "domains": {str(key): list(value) for key, value in self.domains.items()},
            "bounds": _jsonable(self.bounds),
            "certificate_status": self.certificate_status,
            "context_hash": self.context_hash,
            "visible_bank": list(self.visible_bank),
            "proposal_domain": list(self.proposal_domain),
            "selected_context": list(self.selected_context),
            "frozen_graph": None if self.frozen_graph is None else self.frozen_graph.public_dict(),
            "semantic_atoms": {
                str(memory_id): atom.public_dict() for memory_id, atom in self.semantic_atoms.items()
            },
            "context_plan": None if self.context_plan is None else self.context_plan.public_dict(),
            "residual_certification_gaps": list(self.residual_certification_gaps),
            "tmic_diagnostics": _jsonable(self.diagnostics),
        }


@dataclass
class GuidedCandidatePool:
    """Traceable candidate support for reranker-guided BridgeTree methods."""

    dense_ids: List[str]
    anchor_ids: List[str]
    bridge_raw_ids: List[str]
    bridge_kept_ids: List[str]
    candidate_ids: List[str]
    parent_by_bridge_id: Dict[str, str]
    branch_by_bridge_id: Dict[str, str]
    rerank_scores: Dict[str, float]
    tree_result: RetrievalResult | None = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def empty_retrieval_result(query: str, budget: SearchBudget) -> RetrievalResult:
    tracker = CostTracker(budget)
    tracker.set_stop_reason("insufficient_candidates")
    return RetrievalResult(query, [], [], {}, [], [], [], [], tracker, False, [], [])
