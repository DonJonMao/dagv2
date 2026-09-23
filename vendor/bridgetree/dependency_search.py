"""Fixed-target conditional-activation search and dynamic bundle selection.

The search state is exactly ``(target_id, canonical_premise_ids)``.  Targets
are never replaced by newly accepted premises, and a proposal candidate is
excluded only from its current state.  Consequently the same memory can be
measured again under a different premise set, where its interaction may be
different.
"""

from __future__ import annotations

import heapq
import inspect
import itertools
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np

from .dependency_retrieval import DependencyRetriever, InitialCandidatePool, ProposalBatch
from .diagnostic_observability import observe
from .root_tie_diagnostics import root_tie_order, summarize_root_tie_run, validate_root_tie_settings


class SetScorer(Protocol):
    """Bound-query scorer protocol used by the search and selector."""

    @property
    def scored_sets(self) -> int: ...

    def preflight(self, sets: Sequence[Sequence[str]]) -> Any: ...

    def score_sets(self, sets: Sequence[Sequence[str]]) -> Sequence[float]: ...

    def score_set(self, ids: Sequence[str]) -> float: ...

    def feasible(self, ids: Sequence[str]) -> Any: ...

    def set_budget(self, limit: int | None) -> Any: ...


class SearchConfigurationError(ValueError):
    """Raised when a conditional-search contract is internally inconsistent."""


class _ScoringStop(RuntimeError):
    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _canonical_ids(values: Iterable[str]) -> tuple[str, ...]:
    ids = tuple(sorted({str(value) for value in values}))
    if any(not value for value in ids):
        raise ValueError("memory IDs must be non-empty strings")
    return ids


def _stable_unique(values: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw)
        if not value:
            raise ValueError("memory IDs must be non-empty strings")
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


def _measurement_sources(
    group_ids: Sequence[str],
    proposal_ids: set[str],
    proposal_source: str,
    root_candidate_ids: set[str],
) -> tuple[str, ...]:
    """Return the exact retrieval sources represented by a measured group."""

    sources: list[str] = []
    if any(identifier in proposal_ids for identifier in group_ids):
        sources.append(proposal_source)
    if any(identifier in root_candidate_ids for identifier in group_ids):
        sources.append("initial_pool")
    return tuple(sources)


def _scored_sets(scorer: Any) -> int:
    raw = getattr(scorer, "scored_sets", 0)
    value = raw() if callable(raw) else raw
    if isinstance(value, bool):
        raise ValueError("scorer.scored_sets must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("scorer.scored_sets must be a non-negative integer") from exc
    if result < 0 or float(value) != float(result):
        raise ValueError("scorer.scored_sets must be a non-negative integer")
    return result


def _set_cumulative_budget(scorer: Any, limit: int | None) -> None:
    """Set a cumulative set cap without assuming one concrete scorer class."""

    if limit is not None:
        limit = _nonnegative_int(limit, "set budget")
        if limit < _scored_sets(scorer):
            raise ValueError("set budget cannot be lower than already scored sets")
    for name in ("set_budget", "set_budget_limit", "set_cumulative_budget"):
        method = getattr(scorer, name, None)
        if callable(method):
            method(limit)
            return
    # Some injected fakes expose a writable cumulative limit instead of a
    # method.  Do not manufacture an attribute on an opaque production
    # scorer, but honor an explicitly declared one.
    for name in ("set_budget", "set_budget_limit"):
        if hasattr(scorer, name) and not callable(getattr(scorer, name)):
            setattr(scorer, name, limit)
            return


def _resource_reason(exc: BaseException) -> str | None:
    # A physical HTTP ceiling or failed audit write is not an algorithmic
    # set-budget stop. Preserve the error for the diagnostic task ledger.
    from .request_audit import AuditWriteError, TransportBudgetExceeded
    if isinstance(exc, (AuditWriteError, TransportBudgetExceeded)):
        return None
    text = f"{exc.__class__.__name__}: {exc}".lower().replace("-", "_")
    if any(word in text for word in ("budget", "quota", "setlimit", "set_limit")):
        return "score_budget_exhausted"
    if any(
        word in text
        for word in (
            "input_capacity",
            "input capacity",
            "inputtoolong",
            "setinputtoolong",
            "too_long",
            "too long",
            "token limit",
            "max_input",
            "context length",
        )
    ):
        return "input_capacity"
    return None


def _preflight(scorer: Any, sets: Sequence[tuple[str, ...]]) -> None:
    method = getattr(scorer, "preflight", None)
    if not callable(method):
        return
    try:
        result = method(sets)
    except Exception as exc:
        reason = _resource_reason(exc)
        if reason is None:
            raise
        raise _ScoringStop(reason, str(exc)) from exc
    if result is None or result is True:
        return
    reason = ""
    feasible: bool | None = None
    if isinstance(result, bool):
        feasible = result
    elif isinstance(result, Mapping):
        for name in ("feasible", "ok", "allowed"):
            if name in result:
                feasible = bool(result[name])
                break
        reason = str(result.get("reason", result.get("detail", "")))
    else:
        for name in ("feasible", "ok", "allowed"):
            if hasattr(result, name):
                feasible = bool(getattr(result, name))
                break
        reason = str(getattr(result, "reason", getattr(result, "detail", "")))
    if feasible is False:
        normalized = _resource_reason(RuntimeError(reason))
        raise _ScoringStop(normalized or "input_capacity", reason or "scorer preflight rejected request")


def _score_value(value: Any) -> float:
    if hasattr(value, "score"):
        value = value.score
    elif hasattr(value, "value"):
        value = value.value
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("set scorer returned a non-finite score")
    return result


def _accepts_reason_keyword(method: Callable[..., Any]) -> bool:
    try:
        parameters = inspect.signature(method).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == "reason" or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _score_sets(
    scorer: Any,
    sets: Sequence[tuple[str, ...]],
    *,
    reason: str,
) -> tuple[float, ...]:
    """Score in request order and reject incomplete/ambiguous responses."""

    method = getattr(scorer, "score_sets", None)
    try:
        if callable(method):
            raw = method(sets, reason=reason) if _accepts_reason_keyword(method) else method(sets)
            if isinstance(raw, Mapping):
                values: list[Any] = []
                for ids in sets:
                    if ids in raw:
                        values.append(raw[ids])
                    elif frozenset(ids) in raw:
                        values.append(raw[frozenset(ids)])
                    else:
                        raise ValueError("set scorer mapping omitted a requested set")
            else:
                if isinstance(raw, (str, bytes)):
                    raise ValueError("set scorer response must be a score sequence")
                try:
                    values = list(raw)
                except TypeError as exc:
                    raise ValueError("set scorer response must be a score sequence") from exc
        else:
            single = getattr(scorer, "score_set", None)
            if not callable(single):
                raise TypeError("set scorer must expose score_sets or score_set")
            values = [
                single(ids, reason=reason) if _accepts_reason_keyword(single) else single(ids)
                for ids in sets
            ]
    except Exception as exc:
        reason = _resource_reason(exc)
        if reason is None:
            raise
        raise _ScoringStop(reason, str(exc)) from exc
    if len(values) != len(sets):
        raise ValueError(
            f"set scorer returned {len(values)} scores for {len(sets)} requested sets"
        )
    return tuple(_score_value(value) for value in values)


def _preflight_and_score(
    scorer: Any,
    sets: Sequence[tuple[str, ...]],
    *,
    reason: str,
) -> tuple[float, ...]:
    # The complete comparison/round is checked before any network work.  A
    # production scorer additionally deduplicates/cache-checks these sets.
    _preflight(scorer, sets)
    return _score_sets(scorer, sets, reason=reason)


@dataclass(frozen=True)
class DependencyState:
    target_id: str
    premise_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        target = str(self.target_id)
        premises = _canonical_ids(self.premise_ids)
        if not target:
            raise ValueError("state target_id cannot be empty")
        if target in premises:
            raise ValueError("state target cannot also be a premise")
        object.__setattr__(self, "target_id", target)
        object.__setattr__(self, "premise_ids", premises)

    @property
    def identity(self) -> tuple[str, tuple[str, ...]]:
        return self.target_id, self.premise_ids

    @property
    def bundle_ids(self) -> tuple[str, ...]:
        return _canonical_ids((*self.premise_ids, self.target_id))

    def successor(self, group_ids: Sequence[str]) -> "DependencyState":
        group = _canonical_ids(group_ids)
        if self.target_id in group or set(group).intersection(self.premise_ids):
            raise ValueError("successor group must be new premises for the fixed target")
        return DependencyState(self.target_id, _canonical_ids((*self.premise_ids, *group)))

    def public_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "premise_ids": list(self.premise_ids),
            "identity": [self.target_id, list(self.premise_ids)],
            "bundle_ids": list(self.bundle_ids),
        }


@dataclass(frozen=True)
class ActivationRecord:
    """One measured four-term singleton or finite-group interaction."""

    target_id: str
    premise_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    score_P: float
    score_Pe: float
    score_PG: float
    score_PGe: float
    activation: float
    context_marginal: float
    signal_kind: str
    signal: float
    accepted: bool
    queued: bool
    proposal_sources: tuple[str, ...] = ()

    @property
    def P(self) -> float:
        return self.score_P

    @property
    def Pe(self) -> float:
        return self.score_Pe

    @property
    def PG(self) -> float:
        return self.score_PG

    @property
    def PGe(self) -> float:
        return self.score_PGe

    @property
    def group_kind(self) -> str:
        return "singleton" if len(self.group_ids) == 1 else "pair"

    @property
    def successor(self) -> DependencyState:
        return DependencyState(
            self.target_id,
            _canonical_ids((*self.premise_ids, *self.group_ids)),
        )

    def public_dict(self) -> dict[str, Any]:
        p_ids = self.premise_ids
        pe_ids = _canonical_ids((*self.premise_ids, self.target_id))
        pg_ids = _canonical_ids((*self.premise_ids, *self.group_ids))
        pge_ids = _canonical_ids((*self.premise_ids, *self.group_ids, self.target_id))
        return {
            "target_id": self.target_id,
            "premise_ids": list(self.premise_ids),
            "group_ids": list(self.group_ids),
            "group_kind": self.group_kind,
            "sets": {
                "P": list(p_ids),
                "Pe": list(pe_ids),
                "PG": list(pg_ids),
                "PGe": list(pge_ids),
            },
            # These exact field names make the four-term calculation directly
            # reproducible from the persisted activation log.
            "P": self.score_P,
            "Pe": self.score_Pe,
            "PG": self.score_PG,
            "PGe": self.score_PGe,
            "activation": self.activation,
            "target_marginal_before": self.score_Pe - self.score_P,
            "target_marginal_after": self.score_PGe - self.score_PG,
            "context_marginal": self.context_marginal,
            "signal_kind": self.signal_kind,
            "signal": self.signal,
            "accepted": self.accepted,
            "queued": self.queued,
            "successor": self.successor.public_dict(),
            "proposal_sources": list(self.proposal_sources),
        }


@dataclass(frozen=True)
class SkippedMeasurement:
    state: DependencyState
    group_ids: tuple[str, ...]
    reason: str
    detail: str

    def public_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.public_dict(),
            "group_ids": list(self.group_ids),
            "reason": self.reason,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class SearchStateRecord:
    state: DependencyState
    priority: float
    event: str
    detail: str
    proposed_count: int
    singleton_tests: int
    pair_tests: int
    positive_successors: int
    new_successors: int
    ann_call_index: int | None = None

    def __post_init__(self) -> None:
        if self.event not in {"expanded", "proposal_exhausted", "budget_stopped"}:
            raise ValueError("invalid dependency state event")

    def public_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.public_dict(),
            "priority": self.priority,
            "event": self.event,
            "detail": self.detail,
            "proposed_count": self.proposed_count,
            "singleton_tests": self.singleton_tests,
            "pair_tests": self.pair_tests,
            "positive_successors": self.positive_successors,
            "new_successors": self.new_successors,
            "ann_call_index": self.ann_call_index,
        }


@dataclass(frozen=True)
class EvidenceBundle:
    memory_ids: tuple[str, ...]
    archive_reasons: tuple[str, ...] = ()
    target_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        ids = _canonical_ids(self.memory_ids)
        if not ids:
            raise ValueError("an evidence bundle cannot be empty")
        object.__setattr__(self, "memory_ids", ids)
        object.__setattr__(self, "archive_reasons", tuple(sorted(set(self.archive_reasons))))
        object.__setattr__(self, "target_ids", _canonical_ids(self.target_ids))

    @property
    def ids(self) -> tuple[str, ...]:
        return self.memory_ids

    def public_dict(self) -> dict[str, Any]:
        return {
            "memory_ids": list(self.memory_ids),
            "archive_reasons": list(self.archive_reasons),
            "target_ids": list(self.target_ids),
        }


@dataclass(frozen=True)
class SearchArchive:
    initial_target_ids: tuple[str, ...]
    bundles: tuple[EvidenceBundle, ...]
    activations: tuple[ActivationRecord, ...]
    state_records: tuple[SearchStateRecord, ...]
    skipped_measurements: tuple[SkippedMeasurement, ...]
    proposal_batches: tuple[ProposalBatch, ...]
    stop_reason: str
    initial_ann_calls: int
    final_ann_calls: int
    initial_scored_sets: int
    final_scored_sets: int
    signal_kind: str
    fixed_pool: bool
    pair_rescue_width: int
    global_certificate: bool = False

    @property
    def bundle_ids(self) -> tuple[tuple[str, ...], ...]:
        return tuple(bundle.memory_ids for bundle in self.bundles)

    @property
    def expanded_states(self) -> tuple[DependencyState, ...]:
        return tuple(record.state for record in self.state_records)

    def public_dict(self) -> dict[str, Any]:
        resource_reasons = {
            "ann_budget_exhausted",
            "score_budget_exhausted",
            "input_capacity",
        }
        return {
            "initial_target_ids": list(self.initial_target_ids),
            "bundles": [bundle.public_dict() for bundle in self.bundles],
            "activations": [record.public_dict() for record in self.activations],
            "states": [record.public_dict() for record in self.state_records],
            "skipped_measurements": [record.public_dict() for record in self.skipped_measurements],
            "proposal_batches": [batch.public_dict() for batch in self.proposal_batches],
            "stop_reason": self.stop_reason,
            "stop": {
                "event": (
                    "budget_stopped" if self.stop_reason in resource_reasons else "search_stop"
                ),
                "reason": self.stop_reason,
                "global_certificate": False,
            },
            "initial_ann_calls": self.initial_ann_calls,
            "final_ann_calls": self.final_ann_calls,
            "ann_calls": self.final_ann_calls - self.initial_ann_calls,
            "initial_scored_sets": self.initial_scored_sets,
            "final_scored_sets": self.final_scored_sets,
            "scored_sets": self.final_scored_sets - self.initial_scored_sets,
            "signal_kind": self.signal_kind,
            "fixed_pool": self.fixed_pool,
            "pair_rescue_width": self.pair_rescue_width,
            "global_certificate": self.global_certificate,
        }


class DependencySearcher:
    """Best-first fixed-target search controlled by measured positive signal."""

    def __init__(
        self,
        scorer: SetScorer,
        retriever: DependencyRetriever,
        *,
        signal: str = "activation",
        pair_rescue_width: int = 4,
        fixed_pool: bool | None = None,
        max_scored_sets: int | None = None,
        root_tie_break: str = "legacy_lexical",
        root_tie_seed: int | None = None,
    ) -> None:
        normalized = str(signal).strip().lower().replace("-", "_")
        aliases = {"context": "context_marginal", "marginal": "context_marginal"}
        normalized = aliases.get(normalized, normalized)
        if normalized not in {"activation", "context_marginal"}:
            raise ValueError("signal must be activation or context_marginal")
        if fixed_pool is not None and not isinstance(fixed_pool, bool):
            raise ValueError("fixed_pool must be boolean or None")
        self.scorer = scorer
        self.retriever = retriever
        self.signal = normalized
        self.pair_rescue_width = _nonnegative_int(pair_rescue_width, "pair_rescue_width")
        self.fixed_pool = retriever.fixed_pool if fixed_pool is None else fixed_pool
        self.max_scored_sets = (
            None if max_scored_sets is None else _nonnegative_int(max_scored_sets, "max_scored_sets")
        )
        self._partial_progress: dict[str, Any] | None = None
        self.root_tie_break, self.root_tie_seed = validate_root_tie_settings(root_tie_break, root_tie_seed)
        self._root_tie_trace: list[dict[str, Any]] = []
        self._root_tie_final: tuple[int, int, str] | None = None

    def root_tie_diagnostics(self, selected_ids: Sequence[str] | None = None) -> dict[str, Any] | None:
        """Return a separate trace without changing legacy archive serialization.

        A final selection must be supplied explicitly; it is never inferred
        from visited states. Search counters are frozen before selection.
        """
        progress = self._partial_progress
        if progress is None:
            return None
        final = self._root_tie_final
        return summarize_root_tie_run(
            mode=self.root_tie_break, seed=self.root_tie_seed,
            initial_ids=progress["initial_ids"], state_observations=self._root_tie_trace,
            initial_ann_calls=progress["initial_ann_calls"],
            final_ann_calls=final[0] if final else int(getattr(self.retriever, "ann_calls", 0)),
            initial_scored_sets=progress["initial_scored_sets"],
            final_scored_sets=final[1] if final else _scored_sets(self.scorer),
            bundle_ids=progress["bundle_reasons"],
            proposed_ids=(identifier for batch in progress["proposal_batches"] for identifier in batch.ids),
            search_complete=final is not None, stop_reason=final[2] if final else None,
            selected_ids=selected_ids,
        )

    def partial_public_dict(
        self,
        *,
        stop_reason: str = "execution_error",
        detail: str = "search raised before producing a complete archive",
    ) -> dict[str, Any] | None:
        """Snapshot measured progress after an exceptional search exit.

        The lists and archive maps are registered before model work begins and
        updated in place.  The experiment runner can therefore persist every
        completed activation/state even when a later backend call raises.
        This is diagnostic state only and never resumes the search frontier.
        """

        progress = self._partial_progress
        if progress is None:
            return None
        bundle_reasons = progress["bundle_reasons"]
        bundle_targets = progress["bundle_targets"]
        bundles = [
            EvidenceBundle(
                memory_ids=ids,
                archive_reasons=tuple(bundle_reasons[ids]),
                target_ids=tuple(bundle_targets.get(ids, ())),
            ).public_dict()
            for ids in sorted(bundle_reasons, key=lambda value: (len(value), value))
        ]
        initial_ann_calls = int(progress["initial_ann_calls"])
        initial_scored_sets = int(progress["initial_scored_sets"])
        final_ann_calls = int(getattr(self.retriever, "ann_calls", 0))
        final_scored_sets = _scored_sets(self.scorer)
        return {
            "partial": True,
            "initial_target_ids": list(progress["initial_ids"]),
            "bundles": bundles,
            "activations": [record.public_dict() for record in progress["activations"]],
            "states": [record.public_dict() for record in progress["state_records"]],
            "skipped_measurements": [
                record.public_dict() for record in progress["skipped"]
            ],
            "proposal_batches": [
                batch.public_dict() for batch in progress["proposal_batches"]
            ],
            "stop_reason": stop_reason,
            "stop": {
                "event": "execution_error",
                "reason": stop_reason,
                "detail": detail,
                "global_certificate": False,
            },
            "initial_ann_calls": initial_ann_calls,
            "final_ann_calls": final_ann_calls,
            "ann_calls": final_ann_calls - initial_ann_calls,
            "initial_scored_sets": initial_scored_sets,
            "final_scored_sets": final_scored_sets,
            "scored_sets": final_scored_sets - initial_scored_sets,
            "signal_kind": self.signal,
            "fixed_pool": self.fixed_pool,
            "pair_rescue_width": self.pair_rescue_width,
            "global_certificate": False,
        }

    def _measure(
        self,
        state: DependencyState,
        group_ids: Sequence[str],
        proposal_sources: Sequence[str],
    ) -> ActivationRecord:
        group = _canonical_ids(group_ids)
        if not group or len(group) > 2:
            raise SearchConfigurationError("activation groups must be non-empty singletons or pairs")
        if state.target_id in group or set(group).intersection(state.premise_ids):
            raise SearchConfigurationError("activation group overlaps the current state")
        p = state.premise_ids
        pe = _canonical_ids((*p, state.target_id))
        pg = _canonical_ids((*p, *group))
        pge = _canonical_ids((*p, *group, state.target_id))
        score_p, score_pe, score_pg, score_pge = _preflight_and_score(
            self.scorer,
            (p, pe, pg, pge),
            reason="activation",
        )
        activation = score_pge - score_pg - score_pe + score_p
        context_marginal = score_pge - score_pe
        signal = activation if self.signal == "activation" else context_marginal
        return ActivationRecord(
            target_id=state.target_id,
            premise_ids=state.premise_ids,
            group_ids=group,
            score_P=score_p,
            score_Pe=score_pe,
            score_PG=score_pg,
            score_PGe=score_pge,
            activation=activation,
            context_marginal=context_marginal,
            signal_kind=self.signal,
            signal=signal,
            accepted=signal > 0.0,
            queued=False,
            proposal_sources=tuple(proposal_sources),
        )

    def run(
        self,
        initial_pool: InitialCandidatePool | Sequence[str] | None = None,
    ) -> SearchArchive:
        if initial_pool is None:
            initial_pool = self.retriever.build_initial_pool(expand=True)
        if isinstance(initial_pool, InitialCandidatePool):
            initial_ids = initial_pool.candidate_ids
        elif isinstance(initial_pool, (str, bytes)) or not isinstance(initial_pool, Sequence):
            raise ValueError("initial_pool must contain memory IDs")
        else:
            initial_ids = _stable_unique(str(value) for value in initial_pool)
        initial_ids = _stable_unique(initial_ids)
        known_bank = set(getattr(self.retriever, "visible_ids", ()))
        if known_bank:
            unknown = set(initial_ids).difference(known_bank)
            if unknown:
                raise ValueError(f"initial pool contains unknown memory IDs: {sorted(unknown)}")

        initial_ann_calls = int(getattr(self.retriever, "ann_calls", 0))
        initial_scored_sets = _scored_sets(self.scorer)
        if self.max_scored_sets is not None:
            _set_cumulative_budget(self.scorer, initial_scored_sets + self.max_scored_sets)

        bundle_reasons: dict[tuple[str, ...], set[str]] = {}
        bundle_targets: dict[tuple[str, ...], set[str]] = {}

        def archive_bundle(ids: Iterable[str], reason: str, target_id: str | None = None) -> None:
            key = _canonical_ids(ids)
            if not key:
                return
            bundle_reasons.setdefault(key, set()).add(str(reason))
            if target_id is not None:
                bundle_targets.setdefault(key, set()).add(str(target_id))

        activations: list[ActivationRecord] = []
        state_records: list[SearchStateRecord] = []
        skipped: list[SkippedMeasurement] = []
        proposal_batches: list[ProposalBatch] = []
        frontier: list[tuple[float, int, str, tuple[str, ...], DependencyState]] = []
        known_states: set[tuple[str, tuple[str, ...]]] = set()
        expanded_states: set[tuple[str, tuple[str, ...]]] = set()
        capacity_limited = False
        self._root_tie_trace = []
        self._root_tie_final = None
        root_ranks = {identifier: rank for rank, identifier in enumerate(root_tie_order(
            initial_ids, mode=self.root_tie_break, seed=self.root_tie_seed,
        ))}
        self._partial_progress = {
            "initial_ids": tuple(initial_ids),
            "initial_ann_calls": initial_ann_calls,
            "initial_scored_sets": initial_scored_sets,
            "bundle_reasons": bundle_reasons,
            "bundle_targets": bundle_targets,
            "activations": activations,
            "state_records": state_records,
            "skipped": skipped,
            "proposal_batches": proposal_batches,
        }

        for target_id in sorted(initial_ids):
            state = DependencyState(target_id)
            known_states.add(state.identity)
            archive_bundle((target_id,), "initial_target_singleton", target_id)
            root_rank = root_ranks[target_id] if self.root_tie_break == "seeded_hash" else 0
            heapq.heappush(frontier, (-0.0, root_rank, state.target_id, state.premise_ids, state))

        stop_reason = "finite_frontier_exhausted"
        if not frontier:
            stop_reason = "no_initial_candidates"

        while frontier:
            negative_priority, _tie_key, _target_key, _premise_key, state = heapq.heappop(frontier)
            if state.identity in expanded_states:
                continue
            expanded_states.add(state.identity)
            priority = -negative_priority
            archive_bundle(state.bundle_ids, "visited_state", state.target_id)
            observation = {
                "pop_index": len(self._root_tie_trace), "target_id": state.target_id,
                "premise_ids": list(state.premise_ids), "premise_depth": len(state.premise_ids),
                "priority": priority, "is_zero_priority_root": priority == 0.0 and not state.premise_ids,
                "root_tie_rank": root_ranks[state.target_id] if not state.premise_ids else None,
                "ann_calls_before": int(getattr(self.retriever, "ann_calls", 0)),
                "scored_sets_before": _scored_sets(self.scorer), "completed": False,
            }
            self._root_tie_trace.append(observation)
            observe("state", observation, event="state_popped")

            proposal = self.retriever.propose(
                state.target_id,
                state.premise_ids,
                fixed_pool=self.fixed_pool,
            )
            proposal_batches.append(proposal)
            # Conditional hits are evaluated first.  The root additionally
            # sees the entire initial pool, with stable first-source wins.
            root_candidates = initial_ids if not state.premise_ids else ()
            candidates = tuple(
                identifier
                for identifier in _stable_unique((*proposal.ids, *root_candidates))
                if identifier != state.target_id and identifier not in state.premise_ids
            )
            for candidate_id in candidates:
                archive_bundle((candidate_id,), "independent_candidate_singleton")

            singleton_tests = 0
            pair_tests = 0
            positive_successors = 0
            new_successors = 0
            already_seen = 0
            input_blocked = False
            score_blocked = False
            any_positive_singleton = False
            proposal_ids = set(proposal.ids)
            root_candidate_ids = set(root_candidates)

            for candidate_id in candidates:
                successor = state.successor((candidate_id,))
                try:
                    record = self._measure(
                        state,
                        (candidate_id,),
                        _measurement_sources(
                            (candidate_id,),
                            proposal_ids,
                            proposal.probe_id,
                            root_candidate_ids,
                        ),
                    )
                except _ScoringStop as exc:
                    skipped.append(SkippedMeasurement(state, (candidate_id,), exc.reason, exc.detail))
                    observe("activation", skipped[-1], event="measurement_skipped")
                    if exc.reason == "score_budget_exhausted":
                        score_blocked = True
                        break
                    input_blocked = True
                    capacity_limited = True
                    continue
                singleton_tests += 1
                if record.accepted:
                    any_positive_singleton = True
                    positive_successors += 1
                    if successor.identity in known_states:
                        already_seen += 1
                    else:
                        known_states.add(successor.identity)
                        archive_bundle(successor.bundle_ids, "positive_successor", state.target_id)
                        heapq.heappush(
                            frontier,
                            (-record.signal, 0, successor.target_id, successor.premise_ids, successor),
                        )
                        new_successors += 1
                        record = ActivationRecord(**{**record.__dict__, "queued": True})
                activations.append(record)
                observe("activation", record, event="activation_measured")

            # Pair rescue is deliberately finite.  Prefer the current new ANN
            # batch; if it is empty (for example in a tiny fake bank), fall
            # back to the candidates already available at this state.
            if (
                not any_positive_singleton
                and not score_blocked
                and self.pair_rescue_width >= 2
            ):
                rescue_source = proposal.ids if proposal.ids else candidates
                rescue_ids = _stable_unique(rescue_source)[: self.pair_rescue_width]
                for left, right in itertools.combinations(rescue_ids, 2):
                    successor = state.successor((left, right))
                    try:
                        record = self._measure(
                            state,
                            (left, right),
                            _measurement_sources(
                                (left, right),
                                proposal_ids,
                                proposal.probe_id,
                                root_candidate_ids,
                            ),
                        )
                    except _ScoringStop as exc:
                        skipped.append(SkippedMeasurement(state, (left, right), exc.reason, exc.detail))
                        observe("activation", skipped[-1], event="measurement_skipped")
                        if exc.reason == "score_budget_exhausted":
                            score_blocked = True
                            break
                        input_blocked = True
                        capacity_limited = True
                        continue
                    pair_tests += 1
                    if record.accepted:
                        positive_successors += 1
                        if successor.identity in known_states:
                            already_seen += 1
                        else:
                            known_states.add(successor.identity)
                            archive_bundle(
                                successor.bundle_ids,
                                "positive_pair_successor",
                                state.target_id,
                            )
                            heapq.heappush(
                                frontier,
                                (-record.signal, 0, successor.target_id, successor.premise_ids, successor),
                            )
                            new_successors += 1
                            record = ActivationRecord(**{**record.__dict__, "queued": True})
                    activations.append(record)
                    observe("activation", record, event="activation_measured")
                # A budget break in the pair loop is global and handled below.

            resource_detail: str | None = None
            if score_blocked:
                event = "budget_stopped"
                detail = "score_budget_exhausted"
                resource_detail = detail
            elif proposal.budget_exhausted:
                event = "budget_stopped"
                detail = "ann_budget_exhausted"
                resource_detail = detail
            elif new_successors:
                event = "expanded"
                detail = (
                    "positive_successors_queued_with_input_capacity_skips"
                    if input_blocked
                    else "positive_successors_queued"
                )
            elif input_blocked:
                event = "budget_stopped"
                detail = "input_capacity"
            elif not candidates:
                event = "proposal_exhausted"
                detail = "no_candidates"
            elif positive_successors or already_seen == len(candidates):
                event = "proposal_exhausted"
                detail = "successors_already_seen"
            else:
                event = "proposal_exhausted"
                detail = "no_positive_signal"
            state_records.append(
                SearchStateRecord(
                    state=state,
                    priority=priority,
                    event=event,
                    detail=detail,
                    proposed_count=len(candidates),
                    singleton_tests=singleton_tests,
                    pair_tests=pair_tests,
                    positive_successors=positive_successors,
                    new_successors=new_successors,
                    ann_call_index=proposal.ann_call_index,
                )
            )
            observation.update({
                "ann_calls_after": int(getattr(self.retriever, "ann_calls", 0)),
                "scored_sets_after": _scored_sets(self.scorer), "completed": True,
            })
            observe("state", state_records[-1], event="state_completed", completed=True,
                    state_event=state_records[-1].event,
                    pop_index=observation["pop_index"],
                    ann_calls_after=observation["ann_calls_after"],
                    scored_sets_after=observation["scored_sets_after"])
            if resource_detail is not None:
                stop_reason = resource_detail
                break

        if stop_reason == "finite_frontier_exhausted" and capacity_limited:
            # The finite frontier drained, but at least one proposed
            # measurement was never available at full input length.  Preserve
            # that incompleteness at the archive level as well as in the
            # per-state/skipped records.
            stop_reason = "input_capacity"

        frozen_bundles = tuple(
            EvidenceBundle(
                memory_ids=ids,
                archive_reasons=tuple(bundle_reasons[ids]),
                target_ids=tuple(bundle_targets.get(ids, ())),
            )
            for ids in sorted(bundle_reasons, key=lambda value: (len(value), value))
        )
        self._root_tie_final = (int(getattr(self.retriever, "ann_calls", 0)), _scored_sets(self.scorer), stop_reason)
        observe("stop", "search_stop", reason=stop_reason, signal_kind=self.signal,
                visited_state_count=len(state_records), archived_bundle_count=len(frozen_bundles),
                frontier_count=len(frontier), global_certificate=False,
                initial_ann_calls=initial_ann_calls, final_ann_calls=self._root_tie_final[0],
                initial_scored_sets=initial_scored_sets, final_scored_sets=self._root_tie_final[1])
        return SearchArchive(
            initial_target_ids=tuple(sorted(initial_ids)),
            bundles=frozen_bundles,
            activations=tuple(activations),
            state_records=tuple(state_records),
            skipped_measurements=tuple(skipped),
            proposal_batches=tuple(proposal_batches),
            stop_reason=stop_reason,
            initial_ann_calls=initial_ann_calls,
            final_ann_calls=int(getattr(self.retriever, "ann_calls", 0)),
            initial_scored_sets=initial_scored_sets,
            final_scored_sets=_scored_sets(self.scorer),
            signal_kind=self.signal,
            fixed_pool=self.fixed_pool,
            pair_rescue_width=self.pair_rescue_width,
        )

    search = run


def _bundle_ids(bundle: EvidenceBundle | Sequence[str]) -> tuple[str, ...]:
    if isinstance(bundle, EvidenceBundle):
        return bundle.memory_ids
    if isinstance(bundle, (str, bytes)) or not isinstance(bundle, Sequence):
        raise ValueError("bundle archive entries must contain memory IDs")
    return _canonical_ids(bundle)


def _feasibility_result(result: Any) -> tuple[bool, str]:
    if result is None:
        return True, ""
    if isinstance(result, bool):
        return result, "" if result else "infeasible"
    if isinstance(result, Mapping):
        value = result.get("feasible", result.get("within_budget", result.get("ok", True)))
        return bool(value), str(result.get("reason", result.get("detail", "")))
    for name in ("feasible", "within_budget", "ok"):
        if hasattr(result, name):
            value = getattr(result, name)
            value = value() if callable(value) else value
            return bool(value), str(getattr(result, "reason", getattr(result, "detail", "")))
    return bool(result), ""


@dataclass(frozen=True)
class SelectionComparison:
    round_index: int
    current_ids: tuple[str, ...]
    bundle_ids: tuple[str, ...]
    union_ids: tuple[str, ...]
    added_ids: tuple[str, ...]
    feasible: bool
    base_score: float | None
    combined_score: float | None
    marginal: float | None
    accepted: bool
    detail: str = ""

    def public_dict(self) -> dict[str, Any]:
        return {
            "event": "selection_step",
            "round": self.round_index,
            "current_ids": list(self.current_ids),
            "bundle_ids": list(self.bundle_ids),
            "union_ids": list(self.union_ids),
            "added_ids": list(self.added_ids),
            "feasible": self.feasible,
            "base_score": self.base_score,
            "combined_score": self.combined_score,
            "marginal": self.marginal,
            "accepted": self.accepted,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class SelectionRound:
    round_index: int
    current_ids: tuple[str, ...]
    comparisons: tuple[SelectionComparison, ...]
    accepted_bundle_ids: tuple[str, ...] | None
    selected_ids_after: tuple[str, ...]
    complete: bool

    def public_dict(self) -> dict[str, Any]:
        return {
            "round": self.round_index,
            "current_ids": list(self.current_ids),
            "comparisons": [comparison.public_dict() for comparison in self.comparisons],
            "accepted_bundle_ids": (
                None if self.accepted_bundle_ids is None else list(self.accepted_bundle_ids)
            ),
            "selected_ids_after": list(self.selected_ids_after),
            "complete": self.complete,
        }


@dataclass(frozen=True)
class SelectionStop:
    reason: str
    round_index: int
    detail: str = ""

    def public_dict(self) -> dict[str, Any]:
        return {
            "event": "selection_stop",
            "reason": self.reason,
            "round": self.round_index,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class SelectionResult:
    selected_ids: tuple[str, ...]
    rounds: tuple[SelectionRound, ...]
    stop: SelectionStop
    frozen_bundle_ids: tuple[tuple[str, ...], ...]
    initial_scored_sets: int
    final_scored_sets: int

    @property
    def stop_reason(self) -> str:
        return self.stop.reason

    @property
    def steps(self) -> tuple[SelectionComparison, ...]:
        return tuple(comparison for round_record in self.rounds for comparison in round_record.comparisons)

    def public_dict(self) -> dict[str, Any]:
        return {
            "selected_ids": list(self.selected_ids),
            "rounds": [round_record.public_dict() for round_record in self.rounds],
            "stop": self.stop.public_dict(),
            "frozen_bundle_ids": [list(ids) for ids in self.frozen_bundle_ids],
            "initial_scored_sets": self.initial_scored_sets,
            "final_scored_sets": self.final_scored_sets,
            "scored_sets": self.final_scored_sets - self.initial_scored_sets,
        }


class DynamicBundleSelector:
    """Greedily choose bundles by their *current* complete-set marginal.

    The candidate archive is normalized and frozen at entry.  Every feasible
    candidate in a round is preflighted/scored before a winner can be
    committed, so an incomplete round can never silently choose from a scored
    prefix.
    """

    def __init__(
        self,
        scorer: SetScorer,
        *,
        max_selection_sets: int | None = 512,
        reranker_feasible: Callable[[tuple[str, ...]], Any] | None = None,
        generation_feasible: Callable[[tuple[str, ...]], Any] | None = None,
    ) -> None:
        self.scorer = scorer
        self.max_selection_sets = (
            None
            if max_selection_sets is None
            else _nonnegative_int(max_selection_sets, "max_selection_sets")
        )
        self.reranker_feasible = reranker_feasible
        self.generation_feasible = generation_feasible
        self._partial_progress: dict[str, Any] | None = None

    def partial_public_dict(
        self,
        *,
        stop_reason: str = "execution_error",
        detail: str = "selection raised before producing a complete result",
    ) -> dict[str, Any] | None:
        """Snapshot completed selection rounds after an exceptional exit."""

        progress = self._partial_progress
        if progress is None:
            return None
        rounds = progress["rounds"]
        initial_scored_sets = int(progress["initial_scored_sets"])
        final_scored_sets = _scored_sets(self.scorer)
        return {
            "partial": True,
            "selected_ids": list(progress["selected_ids"]),
            "rounds": [round_record.public_dict() for round_record in rounds],
            "steps": [
                comparison.public_dict()
                for round_record in rounds
                for comparison in round_record.comparisons
            ],
            "stop": {
                "event": "selection_stop",
                "reason": stop_reason,
                "round": len(rounds),
                "detail": detail,
            },
            "frozen_bundle_ids": [list(ids) for ids in progress["frozen"]],
            "initial_scored_sets": initial_scored_sets,
            "final_scored_sets": final_scored_sets,
            "scored_sets": final_scored_sets - initial_scored_sets,
        }

    def _feasible(self, ids: tuple[str, ...]) -> tuple[bool, str]:
        checks: list[tuple[str, Callable[[tuple[str, ...]], Any]]] = []
        if self.reranker_feasible is not None:
            checks.append(("reranker", self.reranker_feasible))
        else:
            scorer_check = getattr(self.scorer, "feasible", None)
            if callable(scorer_check):
                checks.append(("reranker", scorer_check))
        if self.generation_feasible is not None:
            checks.append(("generator", self.generation_feasible))
        for label, check in checks:
            feasible, detail = _feasibility_result(check(ids))
            if not feasible:
                return False, f"{label}_input_capacity" + (f": {detail}" if detail else "")
        return True, ""

    def select(
        self,
        bundles: SearchArchive | Sequence[EvidenceBundle | Sequence[str]],
    ) -> SelectionResult:
        raw_bundles: Sequence[EvidenceBundle | Sequence[str]]
        if isinstance(bundles, SearchArchive):
            raw_bundles = bundles.bundles
        elif isinstance(bundles, (str, bytes)) or not isinstance(bundles, Sequence):
            raise ValueError("bundles must be a frozen search archive or sequence")
        else:
            raw_bundles = bundles
        normalized = {_bundle_ids(bundle) for bundle in raw_bundles}
        normalized.discard(())
        frozen = tuple(sorted(normalized, key=lambda ids: (len(ids), ids)))

        initial_scored_sets = _scored_sets(self.scorer)
        if self.max_selection_sets is not None:
            _set_cumulative_budget(
                self.scorer,
                initial_scored_sets + self.max_selection_sets,
            )
        selected: tuple[str, ...] = ()
        rounds: list[SelectionRound] = []
        stop = SelectionStop("archive_exhausted", 0)
        self._partial_progress = {
            "frozen": frozen,
            "rounds": rounds,
            "selected_ids": selected,
            "initial_scored_sets": initial_scored_sets,
        }

        round_index = 0
        while True:
            candidates: list[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = []
            infeasible_comparisons: list[SelectionComparison] = []
            had_additional_bundle = False
            for bundle_ids in frozen:
                union_ids = _canonical_ids((*selected, *bundle_ids))
                added_ids = tuple(identifier for identifier in union_ids if identifier not in selected)
                if not added_ids:
                    continue
                had_additional_bundle = True
                feasible, detail = self._feasible(union_ids)
                if not feasible:
                    infeasible_comparisons.append(
                        SelectionComparison(
                            round_index,
                            selected,
                            bundle_ids,
                            union_ids,
                            added_ids,
                            False,
                            None,
                            None,
                            None,
                            False,
                            detail,
                        )
                    )
                    continue
                candidates.append((bundle_ids, union_ids, added_ids))

            if not candidates:
                reason = "input_capacity" if had_additional_bundle else "archive_exhausted"
                stop = SelectionStop(reason, round_index, "no feasible bundle remains")
                if infeasible_comparisons:
                    rounds.append(
                        SelectionRound(
                            round_index,
                            selected,
                            tuple(infeasible_comparisons),
                            None,
                            selected,
                            True,
                        )
                    )
                    observe("selection", rounds[-1], event="selection_round")
                break

            unique_sets = _stable_unique_set_sequence((selected, *(item[1] for item in candidates)))
            try:
                scores = _preflight_and_score(self.scorer, unique_sets, reason="selection")
            except _ScoringStop as exc:
                # Preserve all feasibility decisions, but mark every otherwise
                # feasible candidate as unscored.  No candidate is committed.
                incomplete = list(infeasible_comparisons)
                incomplete.extend(
                    SelectionComparison(
                        round_index,
                        selected,
                        bundle_ids,
                        union_ids,
                        added_ids,
                        True,
                        None,
                        None,
                        None,
                        False,
                        "not_scored_incomplete_round",
                    )
                    for bundle_ids, union_ids, added_ids in candidates
                )
                rounds.append(
                    SelectionRound(
                        round_index,
                        selected,
                        tuple(sorted(incomplete, key=lambda item: item.bundle_ids)),
                        None,
                        selected,
                        False,
                    )
                )
                observe("selection", rounds[-1], event="selection_round")
                stop = SelectionStop(exc.reason, round_index, exc.detail)
                break
            score_by_set = dict(zip(unique_sets, scores))
            base_score = score_by_set[selected]
            ranked: list[tuple[float, int, tuple[str, ...], tuple[str, ...], float]] = []
            for bundle_ids, union_ids, added_ids in candidates:
                combined = score_by_set[union_ids]
                marginal = combined - base_score
                ranked.append((-marginal, len(added_ids), added_ids, bundle_ids, combined))
            ranked.sort()
            best = ranked[0]
            best_marginal = -best[0]
            best_bundle = best[3]
            best_union = _canonical_ids((*selected, *best_bundle))

            comparisons = list(infeasible_comparisons)
            for bundle_ids, union_ids, added_ids in candidates:
                combined = score_by_set[union_ids]
                marginal = combined - base_score
                comparisons.append(
                    SelectionComparison(
                        round_index,
                        selected,
                        bundle_ids,
                        union_ids,
                        added_ids,
                        True,
                        base_score,
                        combined,
                        marginal,
                        best_marginal > 0.0 and bundle_ids == best_bundle,
                    )
                )
            comparisons.sort(key=lambda item: item.bundle_ids)
            if best_marginal <= 0.0:
                rounds.append(
                    SelectionRound(
                        round_index,
                        selected,
                        tuple(comparisons),
                        None,
                        selected,
                        True,
                    )
                )
                observe("selection", rounds[-1], event="selection_round")
                stop = SelectionStop("no_positive_marginal", round_index)
                break
            rounds.append(
                SelectionRound(
                    round_index,
                    selected,
                    tuple(comparisons),
                    best_bundle,
                    best_union,
                    True,
                )
            )
            observe("selection", rounds[-1], event="selection_round")
            selected = best_union
            self._partial_progress["selected_ids"] = selected
            round_index += 1

        result = SelectionResult(
            selected_ids=selected,
            rounds=tuple(rounds),
            stop=stop,
            frozen_bundle_ids=frozen,
            initial_scored_sets=initial_scored_sets,
            final_scored_sets=_scored_sets(self.scorer),
        )
        observe("stop", stop, selected_ids=selected,
                initial_scored_sets=initial_scored_sets, final_scored_sets=result.final_scored_sets)
        return result

    run = select


def _stable_unique_set_sequence(values: Iterable[Sequence[str]]) -> tuple[tuple[str, ...], ...]:
    result: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for value in values:
        canonical = _canonical_ids(value)
        if canonical not in seen:
            seen.add(canonical)
            result.append(canonical)
    return tuple(result)


def search_dependencies(
    scorer: SetScorer,
    retriever: DependencyRetriever,
    initial_pool: InitialCandidatePool | Sequence[str] | None = None,
    **kwargs: Any,
) -> SearchArchive:
    return DependencySearcher(scorer, retriever, **kwargs).run(initial_pool)


def select_bundles(
    scorer: SetScorer,
    bundles: SearchArchive | Sequence[EvidenceBundle | Sequence[str]],
    **kwargs: Any,
) -> SelectionResult:
    return DynamicBundleSelector(scorer, **kwargs).select(bundles)


# Short aliases keep experiment adapters readable without maintaining a
# second implementation.
DependencySearch = DependencySearcher
BundleSelector = DynamicBundleSelector


__all__ = [
    "ActivationRecord",
    "BundleSelector",
    "DependencySearch",
    "DependencySearcher",
    "DependencyState",
    "DynamicBundleSelector",
    "EvidenceBundle",
    "SearchArchive",
    "SearchConfigurationError",
    "SearchStateRecord",
    "SelectionComparison",
    "SelectionResult",
    "SelectionRound",
    "SelectionStop",
    "SetScorer",
    "SkippedMeasurement",
    "search_dependencies",
    "select_bundles",
]
