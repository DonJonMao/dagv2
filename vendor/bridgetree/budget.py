from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from numbers import Integral, Real
from typing import Any, Dict, Iterable

import numpy as np

from .config import RetrievalConfig

STOP_REASONS = {
    "frontier_empty",
    "max_depth",
    "search_budget",
    "certificate",
    "insufficient_candidates",
    "certificate_unavailable",
}


def _strict_int(value: Any, name: str, *, positive: bool = False, nonnegative: bool = False) -> int:
    """Validate a protocol integer without silently truncating values."""
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


def _strict_nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite non-negative number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite non-negative number") from exc
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


@dataclass(frozen=True)
class SearchBudget:
    max_unique_nodes: int
    max_ann_calls: int | None = None
    max_candidate_exposure: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_unique_nodes",
            _strict_int(self.max_unique_nodes, "max_unique_nodes", positive=True),
        )
        if self.max_ann_calls is not None:
            object.__setattr__(
                self,
                "max_ann_calls",
                _strict_int(self.max_ann_calls, "max_ann_calls", positive=True),
            )
        if self.max_candidate_exposure is not None:
            object.__setattr__(
                self,
                "max_candidate_exposure",
                _strict_int(self.max_candidate_exposure, "max_candidate_exposure", positive=True),
            )

    @classmethod
    def from_config(cls, config: RetrievalConfig) -> "SearchBudget":
        return cls(config.search_budget, config.max_ann_calls, config.max_candidate_exposure)

    def validate(self) -> None:
        _strict_int(self.max_unique_nodes, "max_unique_nodes", positive=True)
        if self.max_ann_calls is not None:
            _strict_int(self.max_ann_calls, "max_ann_calls", positive=True)
        if self.max_candidate_exposure is not None:
            _strict_int(self.max_candidate_exposure, "max_candidate_exposure", positive=True)


@dataclass(frozen=True)
class CostSnapshot:
    ann_calls_core: int = 0
    ann_calls_diagnostic: int = 0
    candidates_returned: int = 0
    candidates_returned_diagnostic: int = 0
    unique_visited_nodes: int = 0
    index_build_ms: float = 0.0
    retrieval_core_ms: float = 0.0
    diagnostic_ms: float = 0.0
    rerank_calls: int = 0
    rerank_documents: int = 0
    rerank_ms: float = 0.0
    bridge_embedding_calls: int = 0
    bridge_embedding_queries: int = 0
    bridge_embedding_ms: float = 0.0
    generation_ms: float = 0.0
    final_context_count: int = 0
    final_context_tokens: int = 0
    stop_reason: str = "insufficient_candidates"
    duplicate_proposals: int = 0
    proposal_count: int = 0
    new_unique_candidates_per_ann: float = 0.0
    new_unique_candidates_by_ann: tuple[int, ...] = ()
    # TMIC accounting is additive; legacy fields above remain canonical for
    # old result readers.
    proposal_ann_calls: int = 0
    candidate_exposure: int = 0
    state_embedding_calls: int = 0
    state_embedding_queries: int = 0
    state_embedding_ms: float = 0.0
    transition_exact_ops: int = 0
    bound_ops: int = 0
    cache_hits: int = 0
    # Semantic-path accounting keeps physical proposal exposure separate from
    # unique-node admission.  The legacy fields above remain authoritative for
    # old readers; these are additive aliases/diagnostics.
    raw_return_exposure: int = 0
    unique_admissions: int = 0
    duplicate_parent_edges: int = 0
    proposal_exposure_budget: int | None = None
    unique_node_budget: int = 0
    materialized_features: int = 0
    ancestor_materializations: int = 0
    residual_certification_gap: float = 0.0
    # Raw candidates that were exposed but could not be admitted because the
    # independent unique-node budget was already full.
    unadmitted_exposure_count: int = 0
    # Matrix runs may share discovery/quality work across architectures.  The
    # ordinary counters above remain per-row logical work; these fields expose
    # the inherited physical work without charging it five times.
    shared_ann_calls: int = 0
    shared_candidate_exposure: int = 0
    shared_unique_admissions: int = 0
    shared_discovery_ms: float = 0.0
    shared_rerank_calls: int = 0
    shared_rerank_documents: int = 0
    shared_rerank_ms: float = 0.0
    shared_state_embedding_calls: int = 0
    shared_state_embedding_queries: int = 0
    shared_state_embedding_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def ann_calls(self) -> int:
        return self.ann_calls_core

    @property
    def proposal_calls(self) -> int:
        return self.proposal_ann_calls


@dataclass
class CostTracker:
    budget: SearchBudget
    ann_calls_core: int = 0
    ann_calls_diagnostic: int = 0
    candidates_returned: int = 0
    candidates_returned_diagnostic: int = 0
    index_build_ms: float = 0.0
    retrieval_core_ms: float = 0.0
    diagnostic_ms: float = 0.0
    rerank_calls: int = 0
    rerank_documents: int = 0
    rerank_ms: float = 0.0
    bridge_embedding_calls: int = 0
    bridge_embedding_queries: int = 0
    bridge_embedding_ms: float = 0.0
    generation_ms: float = 0.0
    final_context_count: int = 0
    final_context_tokens: int = 0
    stop_reason: str = "insufficient_candidates"
    duplicate_proposals: int = 0
    proposal_count: int = 0
    _visited: set[str] = field(default_factory=set)
    _expansion_yields: list[int] = field(default_factory=list)
    proposal_ann_calls: int = 0
    state_embedding_calls: int = 0
    state_embedding_queries: int = 0
    state_embedding_ms: float = 0.0
    transition_exact_ops: int = 0
    bound_ops: int = 0
    cache_hits: int = 0
    raw_return_exposure: int = 0
    unique_admissions: int = 0
    duplicate_parent_edges: int = 0
    materialized_features: int = 0
    ancestor_materializations: int = 0
    residual_certification_gap: float = 0.0
    unadmitted_exposure_count: int = 0
    shared_ann_calls: int = 0
    shared_candidate_exposure: int = 0
    shared_unique_admissions: int = 0
    shared_discovery_ms: float = 0.0
    shared_rerank_calls: int = 0
    shared_rerank_documents: int = 0
    shared_rerank_ms: float = 0.0
    shared_state_embedding_calls: int = 0
    shared_state_embedding_queries: int = 0
    shared_state_embedding_ms: float = 0.0

    def __post_init__(self) -> None:
        self.budget.validate()
        self._exposed: set[str] = set()

    @property
    def remaining_ann_calls(self) -> int | None:
        if self.budget.max_ann_calls is None:
            return None
        return max(0, self.budget.max_ann_calls - self.ann_calls_core)

    @property
    def remaining_unique_nodes(self) -> int:
        return max(0, self.budget.max_unique_nodes - len(self._visited))

    @property
    def remaining_candidate_exposure(self) -> int | None:
        if self.budget.max_candidate_exposure is None:
            return None
        return max(0, self.budget.max_candidate_exposure - self.candidates_returned)

    @property
    def cost_unique_count(self) -> int:
        return len(self._visited)

    @property
    def visited_ids(self) -> frozenset[str]:
        """Immutable view used by adapters when admitting pre-fetched hits."""
        return frozenset(self._visited)

    def can_search_core(self) -> bool:
        return (
            self.remaining_unique_nodes > 0
            and (self.remaining_ann_calls is None or self.remaining_ann_calls > 0)
            and (self.remaining_candidate_exposure is None or self.remaining_candidate_exposure > 0)
        )

    def can_propose(self) -> bool:
        """Whether another paid proposal may be issued.

        Unlike :meth:`can_search_core`, this deliberately ignores remaining
        unique-node capacity.  A semantic layer may need to expose duplicate
        candidates from additional real parents after the admission budget is
        full; conflating the two budgets loses that provenance.
        """
        return (
            (self.remaining_ann_calls is None or self.remaining_ann_calls > 0)
            and (self.remaining_candidate_exposure is None or self.remaining_candidate_exposure > 0)
        )

    def search_proposal(
        self,
        index,
        query: np.ndarray,
        top_k: int,
        exclude: Iterable[str] = (),
    ):
        """Issue a proposal query without admitting returned IDs.

        The returned list is raw exposure.  Callers must explicitly invoke
        :meth:`admit_nodes` after the complete layer has been collected.
        """
        requested = _strict_int(top_k, "top_k", nonnegative=True)
        if requested <= 0 or not self.can_propose():
            return []
        if self.remaining_candidate_exposure is not None:
            requested = min(requested, self.remaining_candidate_exposure)
        if requested <= 0:
            return []
        started = time.perf_counter()
        hits = index.search(
            query,
            requested,
            exclude=exclude,
            max_backend_calls=self.remaining_ann_calls,
        )
        self.retrieval_core_ms += (time.perf_counter() - started) * 1000.0
        stats = getattr(index, "last_search_stats", None)
        backend_calls = len(stats.backend_request_sizes) if stats is not None else 1
        backend_calls = max(1, backend_calls) if stats is None or hits else backend_calls
        if self.remaining_ann_calls is not None and backend_calls > self.remaining_ann_calls:
            raise ValueError("index used more ANN calls than the configured budget")
        self.ann_calls_core += backend_calls
        self.proposal_ann_calls += backend_calls
        hits = list(hits)
        if len(hits) > requested:
            # Truncation would under-report raw exposure and make an invalid
            # backend look budget-compliant.
            raise ValueError("index returned more candidates than requested")
        identifiers = [str(memory_id) for memory_id, _score in hits]
        previously_exposed = set(self._exposed)
        self.candidates_returned += len(identifiers)
        self.raw_return_exposure += len(identifiers)
        self.proposal_count += len(identifiers)
        seen_in_response: set[str] = set()
        duplicate = 0
        for identifier in identifiers:
            if identifier in previously_exposed or identifier in seen_in_response:
                duplicate += 1
            seen_in_response.add(identifier)
        self.duplicate_proposals += duplicate
        self._exposed.update(identifiers)
        # Semantic callers care about the number of genuinely new IDs per
        # physical request, while duplicates remain fully chargeable.
        if backend_calls:
            new_ids = set(identifiers) - previously_exposed
            self._expansion_yields.extend([len(new_ids)] + [0] * (backend_calls - 1))
        return hits

    def admit_nodes(self, ids: Iterable[str]) -> int:
        """Admit unique proposal IDs under the independent node budget."""
        added = 0
        for value in ids:
            identifier = str(value)
            if identifier in self._visited:
                continue
            if self.remaining_unique_nodes <= 0:
                self.unadmitted_exposure_count += 1
                continue
            self._visited.add(identifier)
            self._exposed.add(identifier)
            added += 1
        self.unique_admissions += added
        return added

    def record_duplicate_parent_edges(self, count: int = 1) -> None:
        count = _strict_int(count, "duplicate parent edges", nonnegative=True)
        self.duplicate_parent_edges += count

    def record_feature_materialization(self, count: int = 1, *, ancestor: bool = False) -> None:
        count = _strict_int(count, "feature materializations", nonnegative=True)
        if ancestor:
            self.ancestor_materializations += count
        else:
            self.materialized_features += count

    def search_core(self, index, query: np.ndarray, top_k: int, exclude: Iterable[str] = ()):
        requested_input = _strict_int(top_k, "top_k", nonnegative=True)
        if not self.can_search_core() or requested_input <= 0:
            return []
        top_k = min(requested_input, self.remaining_unique_nodes)
        if self.remaining_candidate_exposure is not None:
            top_k = min(top_k, self.remaining_candidate_exposure)
        if top_k <= 0:
            return []
        started = time.perf_counter()
        hits = index.search(
            query,
            top_k,
            exclude=exclude,
            max_backend_calls=self.remaining_ann_calls,
        )
        self.retrieval_core_ms += (time.perf_counter() - started) * 1000.0
        stats = getattr(index, "last_search_stats", None)
        backend_calls = len(stats.backend_request_sizes) if stats is not None else 1
        if self.remaining_ann_calls is not None and backend_calls > self.remaining_ann_calls:
            raise ValueError("index used more ANN calls than the configured budget")
        self.ann_calls_core += backend_calls
        # Every backend request is a proposal call.  M=true may make several
        # deterministic member proposals; callers can add those explicitly,
        # while this count still reflects the physical ANN work.
        self.proposal_ann_calls += backend_calls
        hits = list(hits)
        if len(hits) > top_k:
            raise ValueError("index returned more candidates than requested")
        self.candidates_returned += len(hits)
        self.raw_return_exposure += len(hits)
        ids = [str(memory_id) for memory_id, _score in hits]
        previously_exposed = set(self._exposed)
        seen_in_response: set[str] = set()
        duplicate_count = 0
        for memory_id in ids:
            if memory_id in previously_exposed or memory_id in seen_in_response:
                duplicate_count += 1
            seen_in_response.add(memory_id)
        new_ids = set(ids) - previously_exposed
        self.duplicate_proposals += duplicate_count
        self.proposal_count += len(ids)
        self._exposed.update(ids)
        self.admit_nodes(ids)
        if backend_calls:
            self._expansion_yields.extend([len(new_ids)] + [0] * (backend_calls - 1))
        return hits

    @property
    def candidate_exposure(self) -> int:
        return self.candidates_returned

    @property
    def exposure_count(self) -> int:
        return self.candidates_returned

    def record_proposal_ann(self, calls: int = 1) -> None:
        calls = _strict_int(calls, "proposal ANN calls", nonnegative=True)
        if self.budget.max_ann_calls is not None and self.ann_calls_core + calls > self.budget.max_ann_calls:
            raise ValueError("proposal ANN calls exceed the configured core ANN budget")
        # This method is the accounting hook for ANN adapters which do not
        # call ``search_core`` themselves.  Keep the physical core-call
        # counter and the proposal-specific counter in lockstep so budget
        # checks cannot be bypassed by an alternate adapter.
        self.ann_calls_core += calls
        self.proposal_ann_calls += calls
        # A custom adapter does not provide the returned IDs here, so record a
        # conservative zero-yield entry for each physical request.  Callers
        # that have IDs should use ``search_core`` (which replaces this with
        # the observed yield); keeping the vector length aligned still makes
        # the per-ANN diagnostic auditable.
        self._expansion_yields.extend([0] * calls)

    def record_candidate_exposure(
        self,
        count: int = 1,
        *,
        identifiers: Iterable[str] | None = None,
        enforce_budget: bool = True,
    ) -> None:
        """Record externally observed proposals (for custom ANN adapters).

        ``search_core`` records physical returned candidates automatically;
        this method is for adapters that perform an ANN request outside the
        built-in index and keeps the same budget/cost vocabulary.  Diagnostic
        full-pool methods may pass ``enforce_budget=False`` to report their
        deliberately exhaustive exposure; ordinary search adapters should
        retain the default hard-cap behavior.
        """
        count = _strict_int(count, "candidate exposure", nonnegative=True)
        normalized_ids = None if identifiers is None else [str(value) for value in identifiers]
        if normalized_ids is not None and len(normalized_ids) != count:
            raise ValueError("candidate exposure count must match identifiers")
        if enforce_budget and (
            self.budget.max_candidate_exposure is not None
            and self.candidates_returned + count > self.budget.max_candidate_exposure
        ):
            raise ValueError("candidate exposure exceeds the configured budget")
        self.candidates_returned += count
        self.raw_return_exposure += count
        self.proposal_count += count
        if normalized_ids is not None:
            previously_exposed = set(self._exposed)
            seen_in_response: set[str] = set()
            duplicate_count = 0
            for identifier in normalized_ids:
                if identifier in previously_exposed or identifier in seen_in_response:
                    duplicate_count += 1
                seen_in_response.add(identifier)
            self.duplicate_proposals += duplicate_count
            self._exposed.update(normalized_ids)

    def record_state_embedding(self, query_count: int, elapsed_ms: float = 0.0) -> None:
        query_count = _strict_int(query_count, "state embedding query count", nonnegative=True)
        elapsed_ms = _strict_nonnegative_float(elapsed_ms, "state embedding elapsed time")
        self.state_embedding_calls += 1
        self.state_embedding_queries += query_count
        self.state_embedding_ms += elapsed_ms

    def record_transition(self, operations: int = 1) -> None:
        operations = _strict_int(operations, "transition operations", nonnegative=True)
        self.transition_exact_ops += operations

    def record_bound(self, operations: int = 1) -> None:
        operations = _strict_int(operations, "bound operations", nonnegative=True)
        self.bound_ops += operations

    def record_cache_hit(self, count: int = 1) -> None:
        count = _strict_int(count, "cache hits", nonnegative=True)
        self.cache_hits += count

    def inherit_shared_cost(self, snapshot: CostSnapshot) -> None:
        """Attach physical work performed once for a shared matrix pool.

        These diagnostics are intentionally separate from the row's logical
        counters.  A matrix consumer can therefore report either per-method
        incremental cost or the true physical total without ambiguity.
        """

        self.shared_ann_calls = int(snapshot.ann_calls_core)
        self.shared_candidate_exposure = int(snapshot.candidate_exposure)
        self.shared_unique_admissions = int(snapshot.unique_admissions)
        self.shared_discovery_ms = float(snapshot.retrieval_core_ms)
        self.shared_rerank_calls = int(snapshot.rerank_calls)
        self.shared_rerank_documents = int(snapshot.rerank_documents)
        self.shared_rerank_ms = float(snapshot.rerank_ms)
        self.shared_state_embedding_calls = int(snapshot.state_embedding_calls)
        self.shared_state_embedding_queries = int(snapshot.state_embedding_queries)
        self.shared_state_embedding_ms = float(snapshot.state_embedding_ms)
        self.unadmitted_exposure_count += int(snapshot.unadmitted_exposure_count)

    def search_diagnostic(self, index, query: np.ndarray, top_k: int, exclude: Iterable[str] = ()):
        requested = _strict_int(top_k, "top_k", nonnegative=True)
        if requested <= 0:
            return []
        started = time.perf_counter()
        hits = index.search(query, requested, exclude=exclude)
        self.diagnostic_ms += (time.perf_counter() - started) * 1000.0
        stats = getattr(index, "last_search_stats", None)
        backend_calls = len(stats.backend_request_sizes) if stats is not None else 1
        hits = list(hits)
        if len(hits) > requested:
            raise ValueError("index returned more diagnostic candidates than requested")
        self.ann_calls_diagnostic += backend_calls
        self.candidates_returned_diagnostic += len(hits)
        return hits

    def mark_visited(self, ids: Iterable[str]) -> int:
        """Mark externally supplied IDs while honoring the node budget.

        Initial-hit adapters and full-pool baselines use this method instead
        of ``search_core``.  Silently inserting an unbounded list here would
        make the advertised ``max_unique_nodes`` budget method-dependent, so
        new IDs are accepted only until the remaining capacity is exhausted.
        The returned count is useful to callers that need to record truncation.
        """
        added = 0
        for value in ids:
            identifier = str(value)
            if identifier in self._visited:
                continue
            if self.remaining_unique_nodes <= 0:
                self.unadmitted_exposure_count += 1
                continue
            self._visited.add(identifier)
            added += 1
            self._exposed.add(identifier)
        self.unique_admissions += added
        return added

    def record_rerank(self, document_count: int, elapsed_ms: float) -> None:
        document_count = _strict_int(document_count, "rerank document count", nonnegative=True)
        elapsed_ms = _strict_nonnegative_float(elapsed_ms, "rerank elapsed time")
        self.rerank_calls += 1
        self.rerank_documents += document_count
        self.rerank_ms += elapsed_ms

    def record_bridge_embedding(self, query_count: int, elapsed_ms: float) -> None:
        query_count = _strict_int(query_count, "bridge embedding query count", nonnegative=True)
        elapsed_ms = _strict_nonnegative_float(elapsed_ms, "bridge embedding elapsed time")
        self.bridge_embedding_calls += 1
        self.bridge_embedding_queries += query_count
        self.bridge_embedding_ms += elapsed_ms

    def set_stop_reason(self, reason: str) -> None:
        if reason not in STOP_REASONS:
            raise ValueError(f"unsupported stop reason: {reason}")
        self.stop_reason = reason

    def snapshot(self) -> CostSnapshot:
        mean_yield = sum(self._expansion_yields) / len(self._expansion_yields) if self._expansion_yields else 0.0
        return CostSnapshot(
            ann_calls_core=self.ann_calls_core,
            ann_calls_diagnostic=self.ann_calls_diagnostic,
            candidates_returned=self.candidates_returned,
            candidates_returned_diagnostic=self.candidates_returned_diagnostic,
            unique_visited_nodes=len(self._visited),
            index_build_ms=self.index_build_ms,
            retrieval_core_ms=self.retrieval_core_ms,
            diagnostic_ms=self.diagnostic_ms,
            rerank_calls=self.rerank_calls,
            rerank_documents=self.rerank_documents,
            rerank_ms=self.rerank_ms,
            bridge_embedding_calls=self.bridge_embedding_calls,
            bridge_embedding_queries=self.bridge_embedding_queries,
            bridge_embedding_ms=self.bridge_embedding_ms,
            generation_ms=self.generation_ms,
            final_context_count=self.final_context_count,
            final_context_tokens=self.final_context_tokens,
            stop_reason=self.stop_reason,
            duplicate_proposals=self.duplicate_proposals,
            proposal_count=self.proposal_count,
            new_unique_candidates_per_ann=mean_yield,
            new_unique_candidates_by_ann=tuple(self._expansion_yields),
            proposal_ann_calls=self.proposal_ann_calls,
            candidate_exposure=self.candidates_returned,
            state_embedding_calls=self.state_embedding_calls,
            state_embedding_queries=self.state_embedding_queries,
            state_embedding_ms=self.state_embedding_ms,
            transition_exact_ops=self.transition_exact_ops,
            bound_ops=self.bound_ops,
            cache_hits=self.cache_hits,
            raw_return_exposure=self.raw_return_exposure,
            unique_admissions=self.unique_admissions,
            duplicate_parent_edges=self.duplicate_parent_edges,
            proposal_exposure_budget=self.budget.max_candidate_exposure,
            unique_node_budget=self.budget.max_unique_nodes,
            materialized_features=self.materialized_features,
            ancestor_materializations=self.ancestor_materializations,
            residual_certification_gap=float(self.residual_certification_gap),
            unadmitted_exposure_count=self.unadmitted_exposure_count,
            shared_ann_calls=self.shared_ann_calls,
            shared_candidate_exposure=self.shared_candidate_exposure,
            shared_unique_admissions=self.shared_unique_admissions,
            shared_discovery_ms=self.shared_discovery_ms,
            shared_rerank_calls=self.shared_rerank_calls,
            shared_rerank_documents=self.shared_rerank_documents,
            shared_rerank_ms=self.shared_rerank_ms,
            shared_state_embedding_calls=self.shared_state_embedding_calls,
            shared_state_embedding_queries=self.shared_state_embedding_queries,
            shared_state_embedding_ms=self.shared_state_embedding_ms,
        )
