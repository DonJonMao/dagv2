from __future__ import annotations

import hashlib
import heapq
import json
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .budget import CostTracker, SearchBudget
from .clustering import cluster_siblings
from .config import RetrievalConfig
from .index import ExactInnerProductIndex, build_index
from .information import (
    InformationObjective,
    StateBasisProvider,
    aggregate_path_atoms,
    information_atom_from_feature,
)
from .math_utils import (
    angular_distance,
    logdet_marginal,
    nonnegative_cosine,
    normalize,
    normalize_rows,
    path_conditioned_innovation,
)
from .measure import branch_measure, parent_posterior
from .temporal import TransitionCache, bank_hash, build_time_marks, build_transition_matrix, time_marks_hash
from .types import Branch, InformationAtom, Memory, PathHypothesis, RetrievalResult, SelectionStep, TreeNode


def _prepare_injected_hits(
    initial_hits: Sequence[Tuple[str, float]],
    *,
    memory_by_id: Mapping[str, Memory],
    tracker: CostTracker,
    excluded: set[str],
    initial_hits_accounted: bool,
) -> list[tuple[str, float]]:
    """Account and admit an externally supplied ANN response exactly once.

    The full response is raw exposure (including duplicates and excluded
    IDs); only the first occurrence of each eligible ID enters the frozen
    graph, subject to the independent unique-node budget.
    """

    from .semantic import _normalize_initial_hits

    raw_hits = _normalize_initial_hits(initial_hits)
    unknown = [memory_id for memory_id, _score in raw_hits if memory_id not in memory_by_id]
    if unknown:
        raise ValueError(f"initial_hits contain unknown memory ids: {unknown}")
    if not initial_hits_accounted:
        tracker.record_candidate_exposure(
            len(raw_hits), identifiers=[memory_id for memory_id, _score in raw_hits]
        )
    unique_hits: list[tuple[str, float]] = []
    seen: set[str] = set()
    for memory_id, score in raw_hits:
        if memory_id in excluded or memory_id in seen:
            continue
        seen.add(memory_id)
        unique_hits.append((memory_id, score))
    tracker.mark_visited(memory_id for memory_id, _score in unique_hits)
    return [
        (memory_id, score)
        for memory_id, score in unique_hits
        if memory_id in tracker.visited_ids
    ]


class BridgeTreeRetriever:
    """One deterministic first-arrival implementation controlled by runtime config."""

    def __init__(self, config: RetrievalConfig):
        config.validate()
        self.config = config

    def _anchor_reachability(self, reachability: float, direct_score: float) -> float:
        weight = self.config.root_anchor_weight
        return reachability * ((1.0 - weight) + weight * direct_score)

    def retrieve(
        self,
        query: str,
        query_vector: np.ndarray,
        memories: Sequence[Memory],
        memory_vectors: np.ndarray,
        *,
        index: ExactInnerProductIndex | None = None,
        budget: SearchBudget | None = None,
        cost_tracker: CostTracker | None = None,
        index_build_ms: float = 0.0,
        initial_hits: Sequence[Tuple[str, float]] | None = None,
        excluded_candidate_ids: Sequence[str] = (),
        answer_options: Sequence[str] | str | None = None,
        option_embeddings: np.ndarray | None = None,
        state_basis_provider: StateBasisProvider | None = None,
        query_cutoff: Any = None,
        query_metadata: Mapping[str, Any] | None = None,
        transition_cache: TransitionCache | None = None,
        state_embedding_cache: Any | None = None,
        force_tmic: bool = False,
        initial_hits_accounted: bool = False,
        quality_provider: Any | None = None,
        quality_records: Mapping[str, Any] | None = None,
        representation_provider: Any | None = None,
        reranker: Any | None = None,
        listwise_selector: Any | None = None,
        context_token_budget: int | None = None,
        generator_config: Any | None = None,
        representation_fingerprint: str = "",
        proposal_query_provider: Any | None = None,
        bridge_query_instruction: str | None = None,
    ) -> RetrievalResult:
        # Apply the question visibility envelope before any ANN or reranker
        # path.  PersonaMem normally slices by observation index already,
        # but explicit event/date cutoffs can make an earlier observed message
        # future evidence; no retrieval method may inspect it.  A supplied
        # index over the unfiltered bank is invalid and is rebuilt below.
        query_metadata_value = query_metadata if query_metadata else None
        if query_cutoff is not None or query_metadata_value is not None:
            from .semantic import _visible_records

            visible_memories, visible_vectors = _visible_records(
                memories,
                memory_vectors,
                query_cutoff,
                query_metadata=query_metadata_value,
            )
            visible_ids = [str(memory.memory_id) for memory in visible_memories]
            index_ids = list(getattr(index, "ids", ())) if index is not None else []
            if index is not None and index_ids != visible_ids:
                index = None
                index_build_ms = 0.0
            memories = visible_memories
            memory_vectors = visible_vectors
        # The revised named semantic profile has a separate frozen execution
        # chain.  Keeping the dispatch here makes legacy_core/legacy_path and
        # the historical TMIC switch combinations byte-compatible while
        # allowing callers to opt into semantic_path_v1 without constructing a
        # third legacy executor.
        if getattr(self.config, "profile", "legacy_core") in {"semantic", "semantic_path_v1"} and not force_tmic:
            from .semantic import semantic_retrieve

            return semantic_retrieve(
                query,
                query_vector,
                memories,
                memory_vectors,
                self.config,
                index=index,
                budget=budget,
                cost_tracker=cost_tracker,
                index_build_ms=index_build_ms,
                quality_provider=quality_provider,
                quality_records=quality_records,
                reranker=reranker,
                representation_provider=representation_provider,
                answer_options=(
                    answer_options
                    if isinstance(answer_options, str)
                    else ""
                    if answer_options is None
                    else str(answer_options)
                ),
                query_cutoff=query_cutoff,
                query_metadata=query_metadata,
                listwise_selector=listwise_selector,
                proposal_query_provider=proposal_query_provider,
                proposal_query_instruction=bridge_query_instruction or "",
                context_token_budget=context_token_budget,
                generator_config=generator_config,
                representation_fingerprint=representation_fingerprint,
            )
        if len(memories) != len(memory_vectors):
            raise ValueError("memories and memory_vectors must have equal length")
        if force_tmic or any(
            (
                self.config.temporal_measure,
                self.config.measure_propagation,
                self.config.state_information,
                self.config.information_certificate,
            )
        ):
            return self._retrieve_tmic(
                query,
                query_vector,
                memories,
                memory_vectors,
                index=index,
                budget=budget,
                cost_tracker=cost_tracker,
                index_build_ms=index_build_ms,
                initial_hits=initial_hits,
                excluded_candidate_ids=excluded_candidate_ids,
                answer_options=answer_options,
                option_embeddings=option_embeddings,
                state_basis_provider=state_basis_provider,
                query_cutoff=query_cutoff,
                query_metadata=query_metadata,
                transition_cache=transition_cache,
                state_embedding_cache=state_embedding_cache,
                initial_hits_accounted=initial_hits_accounted,
                context_token_budget=context_token_budget,
                generator_config=generator_config,
            )
        search_budget = budget or SearchBudget.from_config(self.config)
        tracker = cost_tracker or CostTracker(search_budget)
        tracker.index_build_ms += index_build_ms
        if not memories:
            tracker.set_stop_reason("insufficient_candidates")
            return RetrievalResult(query, [], [], {}, [], [], [], [], tracker, False, [], [])

        query_vector = normalize(query_vector).astype(np.float32)
        memory_vectors = normalize_rows(memory_vectors).astype(np.float32)
        ids = [memory.memory_id for memory in memories]
        if len(set(ids)) != len(ids):
            raise ValueError("memory ids must be unique")
        memory_by_id = {memory.memory_id: memory for memory in memories}
        if index is None:
            index_started = time.perf_counter()
            index = build_index(
                self.config.index_backend,
                ids,
                memory_vectors,
                exclusion_margin=self.config.faiss_exclusion_margin,
            )
            tracker.index_build_ms += (time.perf_counter() - index_started) * 1000.0

        previous_core_ms = tracker.retrieval_core_ms
        core_started = time.perf_counter()
        direct_scores: Dict[str, float] = {}

        def direct_score(memory_id: str) -> float:
            if memory_id not in direct_scores:
                direct_scores[memory_id] = nonnegative_cosine(query_vector, index.vector(memory_id))
            return direct_scores[memory_id]

        first_width = min(self.config.initial_width, search_budget.max_unique_nodes, len(memories))
        if initial_hits is None:
            first_hits = tracker.search_core(index, query_vector, first_width)
        else:
            first_hits = _prepare_injected_hits(
                initial_hits,
                memory_by_id=memory_by_id,
                tracker=tracker,
                excluded=set(),
                initial_hits_accounted=initial_hits_accounted,
            )
        nodes: Dict[str, TreeNode] = {}
        edges: List[Tuple[Optional[str], str]] = []
        for discovery_order, (memory_id, _score) in enumerate(first_hits):
            score = direct_score(memory_id)
            reachability = self._anchor_reachability(score, score)
            vector = index.vector(memory_id)
            unit_vector = np.asarray(vector, dtype=np.float64)
            unit_vector /= max(1.0, float(np.linalg.norm(unit_vector)))
            nodes[memory_id] = TreeNode(
                memory=memory_by_id[memory_id],
                vector=vector,
                parent_id=None,
                depth=1,
                direct_score=score,
                reachability=reachability,
                innovation=reachability * unit_vector,
                bridge_lift=0.0,
                discovery_order=discovery_order,
            )
            edges.append((None, memory_id))

        frontier: List[Tuple[Tuple[float, ...], str, Branch]] = []
        all_branches: List[Branch] = []
        cluster_radii: List[float] = []
        cluster_member_counts: List[int] = []
        branch_audit_candidates: Dict[str, List[str]] = {}
        branch_counter = 0
        clustering_ms = 0.0

        def push_sibling_branches(sibling_ids: Sequence[str], depth: int) -> None:
            nonlocal branch_counter, clustering_ms
            if not sibling_ids:
                return
            ordered = sorted(sibling_ids)
            sibling_vectors = np.vstack([nodes[memory_id].vector for memory_id in ordered])
            reaches = [nodes[memory_id].reachability for memory_id in ordered]
            clustering_started = time.perf_counter()
            clusters = cluster_siblings(
                sibling_vectors,
                reaches,
                mode=self.config.cluster_mode,
                fixed_count=self.config.cluster_count,
                max_clusters=self.config.max_clusters,
                min_cluster_size=self.config.min_cluster_size,
            )
            clustering_ms += (time.perf_counter() - clustering_started) * 1000.0
            for cluster in clusters:
                member_ids = tuple(ordered[position] for position in cluster.member_positions)
                path_upper = max(nodes[memory_id].reachability for memory_id in member_ids)
                branch = Branch(
                    branch_id=f"b{branch_counter:08d}",
                    member_ids=member_ids,
                    probe=cluster.probe,
                    path_upper_bound=path_upper,
                    marginal_upper_bound=float(np.log1p(path_upper**2)),
                    radius_radians=cluster.radius_radians,
                    depth=depth,
                    creation_order=branch_counter,
                )
                branch_counter += 1
                cluster_radii.append(cluster.radius_radians)
                cluster_member_counts.append(len(member_ids))
                all_branches.append(branch)
                if self.config.search_order == "best_first":
                    priority = (-branch.path_upper_bound, float(branch.creation_order))
                else:
                    priority = (float(branch.depth), float(branch.creation_order))
                heapq.heappush(frontier, (priority, branch.branch_id, branch))

        push_sibling_branches(list(nodes), depth=1)
        selected_ids: List[str] = []
        selection_steps: List[SelectionStep] = []
        frozen = False
        stopped_by_budget = False
        stopped_by_depth = False
        target_count = min(self.config.context_size, len(memories))

        while len(selected_ids) < target_count:
            available = [memory_id for memory_id in nodes if memory_id not in selected_ids]
            if not available:
                if frontier and not frozen:
                    expanded = self._expand_one(
                        frontier,
                        index,
                        query_vector,
                        memory_by_id,
                        direct_scores,
                        nodes,
                        edges,
                        push_sibling_branches,
                        len(nodes),
                        branch_audit_candidates,
                        search_budget,
                        tracker,
                        excluded_candidate_ids,
                    )
                    if expanded:
                        continue
                break

            selected_nodes = [nodes[memory_id] for memory_id in selected_ids]
            margins = {memory_id: self._selection_score(nodes[memory_id], selected_nodes) for memory_id in available}
            best_id = min(available, key=lambda memory_id: (-margins[memory_id], memory_id))
            best_margin = margins[best_id]
            if len(nodes) == len(memories):
                frontier.clear()
            unseen_upper = max((item[2].marginal_upper_bound for item in frontier), default=0.0)
            certificate_allowed = self.config.stop_mode == "certificate_or_budget" and self.config.selection_mode in {
                "rho_logdet",
                "path_logdet",
            }
            certified = certificate_allowed and best_margin + self.config.tie_tolerance >= unseen_upper
            if certified:
                selected_ids.append(best_id)
                selection_steps.append(SelectionStep(len(selected_ids), best_id, best_margin, unseen_upper, 0.0, True))
                continue

            node_budget_reached = len(nodes) >= min(search_budget.max_unique_nodes, len(memories))
            core_search_blocked = not tracker.can_search_core()
            expandable = any(item[2].depth < self.config.max_depth for item in frontier)
            if frozen or node_budget_reached or core_search_blocked or not expandable or not frontier:
                frozen = True
                stopped_by_budget = stopped_by_budget or (
                    len(nodes) < len(memories) and (node_budget_reached or core_search_blocked)
                )
                stopped_by_depth = stopped_by_depth or (bool(frontier) and not expandable)
                epsilon = max(0.0, unseen_upper - best_margin)
                selected_ids.append(best_id)
                selection_steps.append(
                    SelectionStep(len(selected_ids), best_id, best_margin, unseen_upper, epsilon, False)
                )
                continue

            self._expand_one(
                frontier,
                index,
                query_vector,
                memory_by_id,
                direct_scores,
                nodes,
                edges,
                push_sibling_branches,
                len(nodes),
                branch_audit_candidates,
                search_budget,
                tracker,
                excluded_candidate_ids,
            )

        tracker.retrieval_core_ms = previous_core_ms + (time.perf_counter() - core_started) * 1000.0
        if len(selected_ids) < target_count:
            tracker.set_stop_reason("insufficient_candidates")
        elif selection_steps and all(step.certified for step in selection_steps):
            tracker.set_stop_reason("certificate")
        elif stopped_by_budget:
            tracker.set_stop_reason("search_budget")
        elif stopped_by_depth:
            tracker.set_stop_reason("max_depth")
        else:
            tracker.set_stop_reason("frontier_empty")

        diagnostic_started = time.perf_counter()
        from .metrics import branch_ranking_stability

        cluster_stabilities = []
        if self.config.diagnostic_level == "light":
            for branch in all_branches:
                candidates = branch_audit_candidates.get(branch.branch_id)
                if candidates:
                    cluster_stabilities.append(branch_ranking_stability(branch, index, candidates))
        elif self.config.diagnostic_level == "full":
            for branch in all_branches:
                hits = tracker.search_diagnostic(index, branch.probe, min(32, len(memories)))
                cluster_stabilities.append(
                    branch_ranking_stability(branch, index, [memory_id for memory_id, _score in hits])
                )
        if self.config.diagnostic_level != "off":
            tracker.diagnostic_ms = (time.perf_counter() - diagnostic_started) * 1000.0

        chronological_ids = sorted(
            selected_ids,
            key=lambda memory_id: (nodes[memory_id].memory.timestamp, memory_id),
        )
        remaining = [item[2] for item in sorted(frontier)]
        return RetrievalResult(
            query=query,
            selected=[memory_by_id[memory_id] for memory_id in chronological_ids],
            selected_in_greedy_order=selected_ids,
            nodes=nodes,
            edges=edges,
            all_branches=all_branches,
            remaining_branches=remaining,
            selection_steps=selection_steps,
            cost_tracker=tracker,
            budget_frozen=stopped_by_budget,
            cluster_radii=cluster_radii,
            cluster_stabilities=cluster_stabilities,
            cluster_member_counts=cluster_member_counts,
            clustering_ms=clustering_ms,
        )

    def _selection_score(self, node: TreeNode, selected_nodes: Sequence[TreeNode]) -> float:
        if self.config.selection_mode == "rho_topk":
            return node.reachability
        if self.config.selection_mode == "mmr":
            redundancy = max(
                (nonnegative_cosine(node.vector, selected.vector) for selected in selected_nodes),
                default=0.0,
            )
            return self.config.mmr_lambda * node.reachability - (1.0 - self.config.mmr_lambda) * redundancy
        return logdet_marginal(node.innovation, [selected.innovation for selected in selected_nodes])

    def _retrieve_tmic(
        self,
        query: str,
        query_vector: np.ndarray,
        memories: Sequence[Memory],
        memory_vectors: np.ndarray,
        *,
        index: ExactInnerProductIndex | None = None,
        budget: SearchBudget | None = None,
        cost_tracker: CostTracker | None = None,
        index_build_ms: float = 0.0,
        initial_hits: Sequence[Tuple[str, float]] | None = None,
        excluded_candidate_ids: Sequence[str] = (),
        answer_options: Sequence[str] | str | None = None,
        option_embeddings: np.ndarray | None = None,
        state_basis_provider: StateBasisProvider | None = None,
        query_cutoff: Any = None,
        query_metadata: Mapping[str, Any] | None = None,
        transition_cache: TransitionCache | None = None,
        state_embedding_cache: Any | None = None,
        initial_hits_accounted: bool = False,
        context_token_budget: int | None = None,
        generator_config: Any | None = None,
    ) -> RetrievalResult:
        """Unified opt-in T/M/I/C execution path.

        The legacy branch above is intentionally untouched: with all four
        switches disabled it remains byte-for-byte compatible in its ranking,
        first-arrival parent choice, and old JSON fields.  This method uses
        only real memories for proposals and keeps every posterior-supported
        parent/path in the new fields.
        """
        config = self.config
        search_budget = budget or SearchBudget.from_config(config)
        tracker = cost_tracker or CostTracker(search_budget)
        tracker.index_build_ms += index_build_ms
        if not memories:
            tracker.set_stop_reason("insufficient_candidates")
            certificate_requested = config.information_certificate or config.stop_mode == "certificate_or_budget"
            return RetrievalResult(
                query, [], [], {}, [], [], [], [], tracker, False, [], [], transition=np.empty((0, 0)),
                certificate_status="certificate_unavailable" if certificate_requested else "not_requested",
            )

        query_value = normalize(query_vector).astype(np.float32)
        vectors = normalize_rows(np.asarray(memory_vectors)).astype(np.float32)
        ids = [memory.memory_id for memory in memories]
        if len(set(ids)) != len(ids):
            raise ValueError("memory ids must be unique")
        memory_by_id = {memory.memory_id: memory for memory in memories}
        if index is None:
            started = time.perf_counter()
            index = build_index(
                config.index_backend,
                ids,
                vectors,
                exclusion_margin=config.faiss_exclusion_margin,
            )
            tracker.index_build_ms += (time.perf_counter() - started) * 1000.0

        marks_for_hash = build_time_marks(memories)
        qhash_payload = json.dumps(
            {"query": query, "query_cutoff": query_cutoff, "query_metadata": query_metadata},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        qhash = hashlib.sha256(qhash_payload.encode("utf-8")).hexdigest()
        thash = time_marks_hash(marks_for_hash) if isinstance(marks_for_hash, Mapping) else ""
        transition_key = TransitionCache.key_for(
            qhash,
            bank_hash(vectors, ids),
            config.temporal_measure,
            thash,
        )
        cached_transition = transition_cache.get(transition_key) if transition_cache is not None else None
        if cached_transition is not None:
            transition, transition_diagnostic = cached_transition
            tracker.record_cache_hit()
        else:
            transition, transition_diagnostic = build_transition_matrix(
                vectors,
                memories,
                query_cutoff=query_cutoff,
                query=query_metadata,
                temporal_measure=config.temporal_measure,
                ids=ids,
                return_diagnostics=True,
            )
            if transition_cache is not None:
                transition_cache.put(transition_key, transition, transition_diagnostic)
        # ``transition_exact_ops`` is a logical query operation, not a wall
        # clock counter.  Cache hits still consume the same declared
        # transition computation in the protocol; only physical time is
        # avoided.
        tracker.record_transition(len(ids) * len(ids))
        excluded = set(str(value) for value in excluded_candidate_ids)
        direct_scores: Dict[str, float] = {}

        def direct_score(memory_id: str) -> float:
            if memory_id not in direct_scores:
                direct_scores[memory_id] = nonnegative_cosine(query_value, index.vector(memory_id))
            return direct_scores[memory_id]

        first_width = min(config.initial_width, search_budget.max_unique_nodes, len(ids))
        if initial_hits is None:
            first_hits = tracker.search_core(index, query_value, first_width, exclude=excluded)
        else:
            first_hits = _prepare_injected_hits(
                initial_hits,
                memory_by_id=memory_by_id,
                tracker=tracker,
                excluded=excluded,
                initial_hits_accounted=initial_hits_accounted,
            )

        nodes: Dict[str, TreeNode] = {}
        edges: List[Tuple[Optional[str], str]] = []
        proposal_branch_ids: Dict[str, set[str]] = {}
        parent_weights: Dict[str, Dict[str, float]] = {}
        node_support: Dict[str, float] = {}
        node_path_branches: Dict[str, set[str]] = {}
        dropped_zero_support: list[Dict[str, Any]] = []
        root_mass: Dict[str, float] = {}
        root_scores: list[float] = []
        # Discovery order is a topological key for posterior paths.  It must
        # advance only when a node is actually admitted; proposal offsets are
        # not suitable because zero-support proposals may be dropped.
        next_discovery_order = 0
        for discovery_order, (memory_id, _score) in enumerate(first_hits):
            score = direct_score(memory_id)
            vector = np.asarray(index.vector(memory_id), dtype=np.float32)
            node_support[memory_id] = score
            root_scores.append(max(0.0, score))
            nodes[memory_id] = TreeNode(
                memory=memory_by_id[memory_id],
                vector=vector,
                parent_id=None,
                depth=1,
                direct_score=score,
                reachability=score,
                innovation=score * vector.astype(np.float64),
                bridge_lift=0.0,
                discovery_order=discovery_order,
                parent_posterior={},
                # Rebuilt from the normalized root mass below once all
                # first-hop anchors are known.
                path_hypotheses=(),
                transition_support=score,
            )
            edges.append((None, memory_id))
            next_discovery_order = discovery_order + 1
        root_total = float(sum(root_scores))
        if root_total > 0.0:
            root_mass = {
                memory_id: float(max(0.0, nodes[memory_id].direct_score) / root_total)
                for memory_id in nodes
            }
        elif nodes:
            # The only first-hop uniformization is the all-zero mass case.
            uniform = 1.0 / len(nodes)
            root_mass = {memory_id: uniform for memory_id in nodes}

        frontier: List[Tuple[Tuple[float, ...], str, Branch]] = []
        all_branches: List[Branch] = []
        cluster_radii: List[float] = []
        cluster_member_counts: List[int] = []
        branch_audit_candidates: Dict[str, List[str]] = {}
        branch_counter = 0
        clustering_ms = 0.0

        def push_sibling_branches(sibling_ids: Sequence[str], depth: int) -> None:
            nonlocal branch_counter, clustering_ms
            ordered = sorted(set(sibling_ids))
            if not ordered:
                return
            sibling_vectors = np.vstack([nodes[memory_id].vector for memory_id in ordered])
            reaches = [nodes[memory_id].reachability for memory_id in ordered]
            started = time.perf_counter()
            clusters = cluster_siblings(
                sibling_vectors,
                reaches,
                mode=config.cluster_mode,
                fixed_count=config.cluster_count,
                max_clusters=config.max_clusters,
                min_cluster_size=config.min_cluster_size,
            )
            clustering_ms += (time.perf_counter() - started) * 1000.0
            for cluster in clusters:
                members = tuple(ordered[position] for position in cluster.member_positions)
                masses = {memory_id: max(0.0, nodes[memory_id].reachability) for memory_id in members}
                measured = branch_measure(
                    members,
                    masses,
                    transition,
                    candidate_ids=ids,
                )
                branch_id = f"b{branch_counter:08d}"
                branch_counter += 1
                support_values = dict(measured.support)
                support_upper = max(support_values.values(), default=0.0)
                path_upper = max((nodes[memory_id].reachability for memory_id in members), default=0.0)
                branch = Branch(
                    branch_id=branch_id,
                    member_ids=members,
                    probe=np.asarray(cluster.probe, dtype=np.float64),
                    path_upper_bound=float(path_upper),
                    marginal_upper_bound=float(np.log1p(path_upper**2)),
                    radius_radians=cluster.radius_radians,
                    depth=depth,
                    creation_order=branch_counter - 1,
                    member_mass=measured.member_mass,
                    pi=measured.pi,
                    support=support_values,
                    support_upper=float(support_upper),
                    bound_ingredients={"version": "tmic-uninitialized"},
                    zero_mass_uniformized=measured.zero_mass_uniformized,
                )
                cluster_radii.append(cluster.radius_radians)
                cluster_member_counts.append(len(members))
                all_branches.append(branch)
                # A0's all-disabled path is a strict R1 compatibility row:
                # retain the historical path-upper branch ordering.  Once T
                # or M is enabled, the propagated support is the formal
                # frontier key prescribed by TMIC.
                priority_score = (
                    path_upper
                    if not config.temporal_measure and not config.measure_propagation
                    else support_upper
                )
                priority = (
                    (-float(priority_score), float(branch.creation_order))
                    if config.search_order == "best_first"
                    else (float(depth), float(branch.creation_order))
                )
                heapq.heappush(frontier, (priority, branch.branch_id, branch))

        if nodes:
            push_sibling_branches(list(nodes), depth=1)

        def edge_probability(parent_id: str, child_id: str) -> float:
            """Read the formal transition probability for a real edge."""
            row = index_of_member.get(parent_id)
            col = index_of_member.get(child_id)
            if row is None or col is None:
                return 0.0
            value = float(transition[row, col])
            return value if np.isfinite(value) and value > 0.0 else 0.0

        def recompute_paths() -> Dict[str, Tuple[PathHypothesis, ...]]:
            paths_by_id: Dict[str, Tuple[PathHypothesis, ...]] = {}

            def expand(
                memory_id: str,
                visiting: tuple[str, ...] = (),
            ) -> list[tuple[tuple[str, ...], float, dict[str, float], float]]:
                # Alternate posterior parents can be shallower than the MAP
                # parent, so the formal path length is not implied by
                # ``TreeNode.depth``.  Enforce the configured depth here as
                # well as the cycle guard.
                if (
                    memory_id in visiting
                    or memory_id not in nodes
                    or len(visiting) >= config.max_depth
                ):
                    return []
                node = nodes[memory_id]
                if not node.parent_posterior:
                    return [
                        (
                            (memory_id,),
                            float(root_mass.get(memory_id, 1.0)),
                            {},
                            float(node_support.get(memory_id, node.reachability)),
                        )
                    ]
                output: list[tuple[tuple[str, ...], float, dict[str, float], float]] = []
                for parent_id, probability in sorted(node.parent_posterior.items()):
                    if probability <= 0.0 or parent_id == memory_id:
                        continue
                    edge = edge_probability(parent_id, memory_id)
                    if edge <= 0.0:
                        # ``parent_posterior`` is a conditional *support*
                        # map, not another path-transition factor.  Formal
                        # path mass is root_mass * product(P_q); a parent with
                        # no positive transition edge cannot contribute.  The
                        # old fallback multiplied by the conditional
                        # posterior as well and therefore double-counted edge
                        # mass (and could manufacture paths through a zero
                        # transition).
                        continue
                    for parent_path, parent_weight, _parent_edges, parent_support in expand(
                        parent_id,
                        visiting + (memory_id,),
                    ):
                        output.append(
                            (
                                parent_path + (memory_id,),
                                float(parent_weight * edge),
                                _parent_edges,
                                float(parent_support * edge),
                            )
                        )
                return output

            for node in sorted(
                nodes.values(),
                key=lambda item: (item.depth, item.discovery_order, item.memory.memory_id),
            ):
                memory_id = node.memory.memory_id
                candidates = expand(memory_id)
                if not candidates:
                    # A singleton is a valid path only for a root node.  Do
                    # not turn a cyclic, truncated, or otherwise malformed
                    # ancestry map into an implicit root merely to keep the
                    # posterior normalized.
                    if not node.parent_posterior:
                        candidates = [
                            (
                                (memory_id,),
                                float(root_mass.get(memory_id, 1.0)),
                                {},
                                float(node_support.get(memory_id, node.reachability)),
                            )
                        ]
                    else:
                        paths_by_id[memory_id] = ()
                        node.path_hypotheses = ()
                        continue
                total = sum(max(0.0, item[1]) for item in candidates)
                uniformized_paths = total <= 0.0
                if uniformized_paths:
                    total = float(len(candidates))
                branch_name = ",".join(sorted(node_path_branches.get(memory_id, set())))
                immediate_parent_map = {
                    str(parent): float(probability)
                    for parent, probability in sorted(node.parent_posterior.items())
                    if probability > 0.0
                }
                normalized = tuple(
                    PathHypothesis(
                        item[0],
                        immediate_parent_map,
                        branch_name,
                        (1.0 / total) if uniformized_paths else max(0.0, item[1]) / total,
                        float(max(0.0, item[3])),
                    )
                    for item in sorted(candidates, key=lambda item: (-item[1], item[0], branch_name))
                )
                paths_by_id[memory_id] = normalized
                node.path_hypotheses = normalized
            return paths_by_id

        def add_parent_posteriors(candidate_id: str, branch: Branch) -> Dict[str, float]:
            if not config.measure_propagation:
                # M=0 is the deterministic first-arrival compatibility mode:
                # one real parent is selected by the same bottleneck rule as
                # R1.  T may replace the edge relation with P_q, but no
                # posterior mass is split or accumulated across branches.
                available_members = [member for member in branch.member_ids if member in nodes]
                if not available_members:
                    return {}
                def edge_value(member_id: str) -> float:
                    if config.temporal_measure:
                        row = index_of_member.get(member_id)
                        col = index_of_member.get(candidate_id)
                        if row is not None and col is not None:
                            return float(transition[row, col])
                    return nonnegative_cosine(nodes[member_id].vector, index.vector(candidate_id))
                edge_values = {member: edge_value(member) for member in available_members}
                # Under T=true a zero transition is an absent temporal edge;
                # an ANN exposure alone must not manufacture a path mass.
                # T=0 retains the legacy zero-cosine first-arrival behavior.
                if config.temporal_measure and not any(value > 0.0 for value in edge_values.values()):
                    return {}
                chosen = min(
                    available_members,
                    key=lambda member: (-min(nodes[member].reachability, edge_values[member]), member),
                )
                return {chosen: 1.0}
            local = parent_posterior(
                candidate_id,
                branch.member_ids,
                branch.pi,
                transition,
                candidate_ids=ids,
            )
            if not local:
                return {}
            branch_mass = max(0.0, float(branch.support.get(candidate_id, 0.0)))
            if branch_mass <= 0.0:
                # ``local`` and ``branch.support`` are derived from the same
                # non-negative transition terms.  A zero aggregate therefore
                # cannot contribute a posterior.
                return {}
            target = parent_weights.setdefault(candidate_id, {})
            for parent_id, probability in local.items():
                # Keep every positive parent posterior exposed by the branch.
                # Discovery order is a compatibility/display detail, not a
                # probabilistic filter: an alternate parent can be discovered
                # after the candidate (or arrive through another branch).
                # Self edges are the only immediate exclusion; recursive path
                # enumeration has its own cycle guard.
                if parent_id == candidate_id:
                    continue
                target[parent_id] = target.get(parent_id, 0.0) + branch_mass * probability
            total = sum(target.values())
            return {key: value / total for key, value in target.items() if value > 0.0} if total > 0 else {}

        def recompute_parent_posteriors() -> None:
            """Merge every positive parent mass exposed by every real branch.

            ANN exclusion prevents an already discovered ID from being
            returned a second time.  Posterior semantics must nevertheless
            account for all branches that could have generated that ID, so we
            recompute from the stored exact supports after each expansion
            instead of relying on first-arrival proposals.
            """
            if not config.measure_propagation:
                return
            for candidate_id, node in nodes.items():
                if node.parent_id is None:
                    continue  # first-hop anchors remain roots for compatibility
                merged: Dict[str, float] = {}
                for branch in all_branches:
                    local = parent_posterior(
                        candidate_id,
                        branch.member_ids,
                        branch.pi,
                        transition,
                        candidate_ids=ids,
                    )
                    branch_mass = max(0.0, float(branch.support.get(candidate_id, 0.0)))
                    for parent_id, probability in local.items():
                        # Do not discard valid positive alternate parents
                        # based on arrival order.  The MAP ``parent_id`` is
                        # retained solely for legacy navigation; the complete
                        # posterior map is auditable and cycle-safe below.
                        if parent_id == candidate_id:
                            continue
                        merged[parent_id] = merged.get(parent_id, 0.0) + branch_mass * probability
                total = sum(merged.values())
                node.parent_posterior = (
                    {key: value / total for key, value in sorted(merged.items()) if value > 0.0}
                    if total > 0.0
                    else node.parent_posterior
                )

        def expand_one(branch: Branch) -> bool:
            nonlocal next_discovery_order
            if branch.depth >= config.max_depth or not tracker.can_search_core():
                return False
            capacity = min(config.branch_width, search_budget.max_unique_nodes - len(nodes))
            if capacity <= 0:
                return False
            proposals: list[Tuple[str, float, str]] = []
            if config.measure_propagation:
                # Deterministic round-robin over *real* member vectors.  The
                # compressed probe is retained solely as a diagnostic.
                members = tuple(sorted(branch.member_ids))
                proposed: set[str] = set(nodes) | excluded
                round_index = 0
                empty_rounds = 0
                while len(proposals) < capacity and members and tracker.can_search_core():
                    member_id = members[round_index % len(members)]
                    round_index += 1
                    hits = tracker.search_core(
                        index,
                        index.vector(member_id),
                        1,
                        exclude=proposed,
                    )
                    branch_audit_candidates.setdefault(branch.branch_id, []).extend(
                        memory_id for memory_id, _score in hits
                    )
                    for memory_id, score in hits:
                        proposed.add(memory_id)
                        proposals.append((memory_id, score, member_id))
                    if hits:
                        empty_rounds = 0
                    else:
                        empty_rounds += 1
                    # Some custom/external index implementations can return
                    # an empty list before every bank ID appears in the
                    # exclusion set.  Stop after a full no-progress round so
                    # deterministic round-robin cannot spin forever.
                    if not hits and (len(proposed) >= len(ids) or empty_rounds >= len(members)):
                        break
            else:
                hits = tracker.search_core(
                    index,
                    branch.probe,
                    capacity,
                    exclude=set(nodes) | excluded,
                )
                branch_audit_candidates[branch.branch_id] = [memory_id for memory_id, _score in hits]
                proposals = [
                    (memory_id, score, branch.member_ids[0] if branch.member_ids else "")
                    for memory_id, score in hits
                ]
            if not proposals:
                return False
            new_children_by_parent: Dict[str, List[str]] = {}
            for candidate_id, _proposal_score, proposer in proposals:
                proposal_branch_ids.setdefault(candidate_id, set()).add(branch.branch_id)
                node_path_branches.setdefault(candidate_id, set()).add(branch.branch_id)
                if candidate_id in nodes:
                    # It may still contribute a new positive parent/path mass;
                    # never overwrite the first-arrival compatibility parent.
                    posterior = add_parent_posteriors(candidate_id, branch)
                    if posterior:
                        nodes[candidate_id].parent_posterior = posterior
                    continue
                posterior = add_parent_posteriors(candidate_id, branch)
                if not posterior:
                    # A zero temporal mass cannot establish a formal parent;
                    # M=true must not manufacture mass from the proposing
                    # member.  M=false has already returned a deterministic
                    # parent map above and therefore never reaches this
                    # fallback.
                    posterior = {}
                if not posterior:
                    dropped_zero_support.append(
                        {
                            "candidate_id": candidate_id,
                            "branch_id": branch.branch_id,
                            "proposer_id": proposer,
                            "reason": "zero_support_parent_posterior",
                        }
                    )
                    continue
                map_parent = min(posterior, key=lambda key: (-posterior[key], key))
                if config.temporal_measure:
                    parent_position = index_of_member.get(map_parent)
                    candidate_position = index_of_member.get(candidate_id)
                    edge_similarity = (
                        float(transition[parent_position, candidate_position])
                        if parent_position is not None and candidate_position is not None
                        else 0.0
                    )
                else:
                    edge_similarity = nonnegative_cosine(nodes[map_parent].vector, index.vector(candidate_id))
                raw_reachability = min(nodes[map_parent].reachability, edge_similarity)
                support = float(branch.support.get(candidate_id, raw_reachability))
                reachability = max(0.0, support if config.measure_propagation else raw_reachability)
                candidate_direct = direct_scores.setdefault(
                    candidate_id,
                    nonnegative_cosine(query_value, index.vector(candidate_id)),
                )
                vector = np.asarray(index.vector(candidate_id), dtype=np.float32)
                # Fill a legacy-compatible feature now; the final I=1 atom is
                # assembled after all paths/ancestors are known.
                ancestor_vectors = []
                current: Optional[str] = map_parent
                while current is not None:
                    ancestor_vectors.append(nodes[current].reachability * nodes[current].vector)
                    current = nodes[current].parent_id
                ancestor_vectors.reverse()
                innovation = path_conditioned_innovation(vector, reachability, ancestor_vectors)
                node_support[candidate_id] = support
                nodes[candidate_id] = TreeNode(
                    memory=memory_by_id[candidate_id],
                    vector=vector,
                    parent_id=map_parent,
                    depth=nodes[map_parent].depth + 1,
                    direct_score=candidate_direct,
                    reachability=reachability,
                    innovation=innovation,
                    bridge_lift=max(0.0, raw_reachability - candidate_direct),
                    discovery_order=next_discovery_order,
                    parent_posterior=posterior,
                    path_hypotheses=(),
                    transition_support=support,
                )
                next_discovery_order += 1
                edges.append((map_parent, candidate_id))
                new_children_by_parent.setdefault(map_parent, []).append(candidate_id)
            for child_ids in new_children_by_parent.values():
                push_sibling_branches(child_ids, depth=nodes[child_ids[0]].depth)
            recompute_parent_posteriors()
            return True

        # Domains and bounds are rebuilt after each discovery, so C never
        # reasons about a stale candidate partition.
        domains_valid = False
        bounds_by_branch: Dict[str, Dict[str, Any]] = {}

        def rebuild_domains_and_bounds() -> bool:
            nonlocal domains_valid
            domains_valid = False
            if config.certificate_domain != "exact_partition" or config.index_backend != "exact":
                for branch in all_branches:
                    branch.domain_ids = ()
                    branch.bound_ingredients = {"valid": False, "reason": "domain_or_backend_unavailable"}
                    bounds_by_branch[branch.branch_id] = branch.bound_ingredients
                return False
            if not all_branches:
                return False
            eligible = [memory_id for memory_id in ids if memory_id not in excluded]
            assigned: Dict[str, str] = {}
            for memory_id in eligible:
                choices = []
                for branch in all_branches:
                    support = float(branch.support.get(memory_id, 0.0))
                    angle = angular_distance(branch.probe, index.vector(memory_id))
                    choices.append((-support, angle, branch.creation_order, branch.branch_id))
                if choices:
                    assigned[memory_id] = min(choices)[3]
            for branch in all_branches:
                domain = tuple(memory_id for memory_id in eligible if assigned.get(memory_id) == branch.branch_id)
                branch.domain_ids = domain
                domain_payload = json.dumps(list(domain), separators=(",", ":"), ensure_ascii=False)
                domain_hash = hashlib.sha256(domain_payload.encode("utf-8")).hexdigest()
                # Conservative finite-domain support envelope.  Unknown time
                # marks use kappa=1 (neutral), never a recency weight.
                kappa = {}
                for member_id in branch.member_ids:
                    values = [
                        float(transition[index_of_member[member_id], ids.index(candidate)])
                        for candidate in domain
                    ]
                    kappa[member_id] = max(values, default=0.0)
                smax = float(
                    sum(
                        branch.pi.get(member_id, 0.0) * kappa.get(member_id, 0.0)
                        for member_id in branch.member_ids
                    )
                )
                if config.temporal_measure and transition_diagnostic.get("time_unavailable"):
                    smax = max(smax, float(sum(branch.pi.values())))
                rho_upper = float(
                    max((nodes[mid].reachability for mid in branch.member_ids if mid in nodes), default=0.0)
                )
                upper = rho_upper**2 if not config.state_information else 4.0 * smax**2
                ingredients = {
                    "version": "tmic-bound-v1",
                    "domain": "exact_partition",
                    "domain_hash": domain_hash,
                    "domain_ids": list(domain),
                    "probe": np.asarray(branch.probe, dtype=np.float64).tolist(),
                    "angular_radius_radians": float(branch.radius_radians),
                    "time_envelope": "neutral_unknown" if transition_diagnostic.get("time_unavailable") else "observed",
                    "kappa_by_member": kappa,
                    "smax": smax,
                    "rho_upper": rho_upper,
                    "U": upper,
                    "bound_type": "state_trace" if config.state_information else "legacy_rho",
                    "unexposed_count": sum(memory_id not in nodes for memory_id in domain),
                    "valid": True,
                }
                branch.support_upper = float(smax if config.state_information else rho_upper)
                branch.marginal_upper_bound = upper
                branch.bound_ingredients = ingredients
                bounds_by_branch[branch.branch_id] = ingredients
            domains_valid = (
                set(assigned) == set(eligible)
                and len(assigned) == sum(len(branch.domain_ids) for branch in all_branches)
                and len({memory_id for branch in all_branches for memory_id in branch.domain_ids})
                == len(eligible)
            )
            return domains_valid

        # Helper map avoids repeatedly searching member positions in bounds.
        index_of_member = {memory_id: position for position, memory_id in enumerate(ids)}

        def build_atoms() -> tuple[
            Dict[str, InformationAtom],
            Dict[str, Tuple[PathHypothesis, ...]],
            StateBasisProvider | None,
        ]:
            paths = recompute_paths()
            provider = state_basis_provider
            basis = None
            basis_diagnostic: Dict[str, Any] = {}
            if config.state_information:
                provider = provider or StateBasisProvider(
                    config.state_basis_mode,
                    model_fingerprint="retriever-local",
                )
                options = answer_options
                # The retriever itself has no model client.  If callers supply
                # option embeddings we use them; otherwise identity is an
                # explicit, recorded fallback rather than synthetic semantics.
                if option_embeddings is not None:
                    basis = provider.build(option_embeddings, options=options)
                else:
                    basis = provider.build(dimension=vectors.shape[1])
                basis_diagnostic.update(provider.diagnostics)
            atoms: Dict[str, InformationAtom] = {}
            # Parent posteriors are constrained by discovery order; process
            # atoms in that topological order so every ancestor atom is
            # available when applying C_A^{-1/2} in an I=true path.
            ordered = sorted(nodes.values(), key=lambda item: (item.discovery_order, item.memory.memory_id))
            for node in ordered:
                memory_id = node.memory.memory_id
                node_paths = paths.get(memory_id, ())
                if config.state_information:
                    vectors_by_id = {key: other.vector for key, other in nodes.items()}
                    atoms_by_id = {key: atom.matrix for key, atom in atoms.items()}
                    atom = aggregate_path_atoms(
                        memory_id,
                        node_paths,
                        node.vector,
                        basis=basis,
                        ancestor_vectors_by_id=vectors_by_id,
                        ancestor_atoms_by_id=atoms_by_id,
                        state_information=True,
                    )
                else:
                    # I=0: recover R1's path-conditioned phi, then expose
                    # exactly phi phi^T as the matrix atom.
                    path_atoms = []
                    for path in node_paths or (PathHypothesis((memory_id,), {}, "", 1.0, node.reachability),):
                        ancestors = [nodes[item] for item in path.path_ids[:-1] if item in nodes]
                        weighted = [ancestor.reachability * ancestor.vector for ancestor in ancestors]
                        # I=0 retains the legacy path-conditioned geometry,
                        # but its candidate scale is the formal support of
                        # this particular path.  Using the aggregate node
                        # reachability here would erase the distinction
                        # between alternate M=true paths.
                        # With M disabled, A0 must be an element-wise
                        # regression of the legacy R1 path geometry.  The
                        # transition-derived path support is still retained
                        # in provenance, but it is not allowed to rescale the
                        # compatibility atom.  M=true, by contrast, uses the
                        # formal path-specific propagated support and can
                        # therefore distinguish alternate member paths.
                        path_support = float(
                            node.reachability if not config.measure_propagation else path.support
                        )
                        phi = path_conditioned_innovation(node.vector, path_support, weighted)
                        path_atoms.append(
                            InformationAtom(
                                memory_id,
                                path.path_ids,
                                np.outer(phi, phi),
                                float(np.dot(phi, phi)),
                                path_support,
                            )
                        )
                    probabilities = np.asarray([max(0.0, path.posterior) for path in node_paths], dtype=np.float64)
                    if not path_atoms:
                        atom = information_atom_from_feature(memory_id, node.innovation)
                    else:
                        if probabilities.sum() <= 0.0:
                            probabilities = np.ones(len(path_atoms), dtype=np.float64)
                        probabilities /= probabilities.sum()
                        matrix = sum(
                            (weight * item.matrix for weight, item in zip(probabilities, path_atoms)),
                            start=np.zeros_like(path_atoms[0].matrix),
                        )
                        atom = InformationAtom(
                            memory_id,
                            path_atoms[0].path_ids,
                            matrix,
                            float(np.trace(matrix)),
                            float(sum(weight * item.support for weight, item in zip(probabilities, path_atoms))),
                        )
                atoms[memory_id] = atom
                # Keep ``innovation`` as the legacy vector feature for old
                # diagnostics/ablations.  TMIC selection consumes the explicit
                # PSD matrix in ``information_atoms`` and must not collapse a
                # rank-r state matrix into an arbitrary vector.
            if config.state_information and provider is not None:
                transition_diagnostic["state_basis"] = provider.diagnostics
            return atoms, paths, provider

        def objective_scores(atoms: Mapping[str, InformationAtom], selected_ids: Sequence[str]) -> Dict[str, float]:
            if config.selection_mode in {"rho_topk", "mmr"}:
                selected_nodes = [nodes[memory_id] for memory_id in selected_ids]
                return {
                    memory_id: self._selection_score(nodes[memory_id], selected_nodes)
                    for memory_id in nodes
                    if memory_id not in selected_ids
                }
            objective = InformationObjective(
                next(iter(atoms.values())).matrix.shape[0] if atoms else vectors.shape[1],
                tie_tolerance=config.tie_tolerance,
            )
            for memory_id, atom in atoms.items():
                objective.add(memory_id, atom)
            return {
                memory_id: objective.marginal(memory_id, selected_ids)
                for memory_id in nodes
                if memory_id not in selected_ids
            }

        atoms, paths, provider = build_atoms()
        # ``information_certificate`` controls whether finite-domain bounds
        # are computed and recorded.  ``stop_mode`` independently controls
        # whether a valid bound is allowed to terminate expansion.  Keeping
        # these separate matters for the useful diagnostic combination
        # C=true, stop_mode=budget: bounds are auditable, but retrieval still
        # follows the requested budget rather than stopping early.
        certificate_requested = config.information_certificate or config.stop_mode == "certificate_or_budget"
        certificate_stopping = config.stop_mode == "certificate_or_budget"
        if certificate_requested:
            rebuild_domains_and_bounds()
        else:
            # Keep the provenance schema stable while ensuring C-specific
            # work/cost is genuinely disabled for A0--A3.
            domains_valid = False
            for branch in all_branches:
                branch.domain_ids = ()
                branch.bound_ingredients = {"valid": False, "reason": "certificate_disabled"}
            bounds_by_branch = {branch.branch_id: branch.bound_ingredients for branch in all_branches}
        selected_ids: List[str] = []
        selection_steps: List[SelectionStep] = []
        selection_bound_diagnostics: List[Dict[str, Any]] = []
        bound_evaluations: List[Dict[str, Any]] = []
        frozen = False
        stopped_by_budget = False
        stopped_by_depth = False
        eligible_count = sum(memory_id not in excluded for memory_id in ids)
        target_count = min(config.context_size, eligible_count)
        certificate_status = "not_requested"
        if certificate_requested:
            certificate_status = "available" if domains_valid else "certificate_unavailable"

        while len(selected_ids) < target_count:
            available = [memory_id for memory_id in nodes if memory_id not in selected_ids]
            if not available:
                if frontier and not frozen:
                    _priority, _branch_id, branch = heapq.heappop(frontier)
                    if expand_one(branch):
                        atoms, paths, provider = build_atoms()
                        if certificate_requested:
                            rebuild_domains_and_bounds()
                        if certificate_requested and domains_valid:
                            certificate_status = "available"
                        continue
                break
            margins = objective_scores(atoms, selected_ids)
            best_id = min(available, key=lambda memory_id: (-margins[memory_id], memory_id))
            best_margin = float(margins[best_id])
            unseen_ids = [memory_id for memory_id in ids if memory_id not in nodes and memory_id not in excluded]
            unseen_upper = max(
                (
                    float(branch.marginal_upper_bound)
                    for branch in all_branches
                    if any(memory_id in unseen_ids for memory_id in branch.domain_ids)
                ),
                default=0.0,
            )
            if certificate_requested:
                tracker.record_bound(len(all_branches))
            certified = bool(
                certificate_stopping
                and certificate_requested
                and domains_valid
                and best_margin + config.tie_tolerance >= unseen_upper
            )
            bound_row = {
                "step": len(selected_ids) + 1,
                "best_discovered_id": best_id,
                "best_discovered_margin": best_margin,
                "frontier_upper_bound": unseen_upper,
                "gap": max(0.0, unseen_upper - best_margin),
                "bound_valid": bool(domains_valid),
                "certified": certified,
            }
            bound_evaluations.append(bound_row)
            if certified:
                selected_ids.append(best_id)
                selection_bound_diagnostics.append(bound_row)
                selection_steps.append(SelectionStep(len(selected_ids), best_id, best_margin, unseen_upper, 0.0, True))
                continue
            node_budget_reached = len(nodes) >= min(search_budget.max_unique_nodes, len(memories))
            core_blocked = not tracker.can_search_core()
            expandable = any(item[2].depth < config.max_depth for item in frontier)
            if frozen or node_budget_reached or core_blocked or not expandable or not frontier:
                frozen = True
                stopped_by_budget = stopped_by_budget or (
                    len(nodes) < len(memories) and (node_budget_reached or core_blocked)
                )
                stopped_by_depth = stopped_by_depth or (bool(frontier) and not expandable)
                epsilon = max(0.0, unseen_upper - best_margin)
                selected_ids.append(best_id)
                selection_bound_diagnostics.append(bound_row)
                selection_steps.append(
                    SelectionStep(len(selected_ids), best_id, best_margin, unseen_upper, epsilon, False)
                )
                continue
            _priority, _branch_id, branch = heapq.heappop(frontier)
            if not expand_one(branch):
                continue
            atoms, paths, provider = build_atoms()
            if certificate_requested:
                rebuild_domains_and_bounds()
            if certificate_requested and not domains_valid:
                certificate_status = "certificate_unavailable"

        if len(nodes) >= eligible_count:
            frontier.clear()
        tracker.retrieval_core_ms += 0.0  # keep explicit accounting field initialized
        if len(selected_ids) < target_count:
            tracker.set_stop_reason("insufficient_candidates")
        elif selection_steps and all(step.certified for step in selection_steps):
            tracker.set_stop_reason("certificate")
        elif stopped_by_budget:
            tracker.set_stop_reason("search_budget")
        elif stopped_by_depth:
            tracker.set_stop_reason("max_depth")
        else:
            tracker.set_stop_reason("frontier_empty")

        chronological_ids = sorted(selected_ids, key=lambda memory_id: (nodes[memory_id].memory.timestamp, memory_id))
        remaining = [item[2] for item in sorted(frontier)]
        # When a caller supplies the generator contract, freeze the exact
        # request here.  Downstream generation and persistence must consume
        # this plan rather than applying a second, lossy token-budget pass.
        # Calls that use the historical retriever API without a generator
        # contract retain the legacy ID-only context hash for compatibility.
        context_plan = None
        context_plan_error: dict[str, Any] | None = None
        if context_token_budget is not None or generator_config is not None:
            from .clients import build_context_plan

            selected_memories = [memory_by_id[memory_id] for memory_id in chronological_ids]
            try:
                context_plan = build_context_plan(
                    query,
                    selected_memories,
                    "" if answer_options is None else (
                        answer_options
                        if isinstance(answer_options, str)
                        else "\n".join(str(value) for value in answer_options)
                    ),
                    token_budget=context_token_budget,
                    strict=config.context_strict,
                    selected_ids=selected_ids,
                    generator_config=generator_config,
                )
                context_hash = context_plan.context_hash
                tracker.final_context_count = len(context_plan.chronological_ids)
                tracker.final_context_tokens = context_plan.token_count
            except Exception as exc:
                # Preserve the retrieval object for diagnosis, but make the
                # failure explicit.  Generation callers can then reject the
                # result instead of silently truncating selected memories.
                context_plan_error = {"type": type(exc).__name__, "message": str(exc)}
                # Do not publish an ID-only hash for a request that failed to
                # freeze: such a hash would look like a valid context identity
                # while the exact messages/decoding fields are unknown.
                context_hash = None
        else:
            context_hash = hashlib.sha256(
                json.dumps(
                    {"query": query, "ids": chronological_ids},
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
        diagnostics = {
            "temporal": transition_diagnostic,
            "proposal_branch_ids": {key: sorted(value) for key, value in proposal_branch_ids.items()},
            "dropped_zero_support_proposals": dropped_zero_support,
            "parent_posterior": {key: dict(node.parent_posterior) for key, node in nodes.items()},
            "root_mass": dict(root_mass),
            "path_hypotheses": {
                key: [path.public_dict() for path in value] for key, value in paths.items()
            },
            "path_posterior_sums": {
                key: float(sum(path.posterior for path in value)) for key, value in paths.items()
            },
            "state_basis": provider.diagnostics if provider is not None else {"state_basis_fallback": "not_requested"},
            "certificate_status": certificate_status,
            "domains_valid": domains_valid,
            "bounds": bounds_by_branch,
            "selection_bounds": selection_bound_diagnostics,
            "bound_evaluations": bound_evaluations,
            "certified_rate": (
                sum(bool(step.certified) for step in selection_steps) / len(selection_steps)
                if selection_steps else 0.0
            ),
            "bound_tightness": (
                sum(
                    float(item["best_discovered_margin"]) / max(float(item["frontier_upper_bound"]), 1e-12)
                    for item in selection_bound_diagnostics
                ) / len(selection_bound_diagnostics)
                if selection_bound_diagnostics else None
            ),
            "greedy_ids": list(selected_ids),
            "chronological_ids": chronological_ids,
            "memory_ids": list(ids),
            "excluded_candidate_ids": sorted(excluded),
            "context_hash": context_hash,
            "context_plan": None if context_plan is None else context_plan.public_dict(),
        }
        if context_plan_error is not None:
            diagnostics["context_plan_error"] = context_plan_error
        return RetrievalResult(
            query=query,
            selected=[memory_by_id[memory_id] for memory_id in chronological_ids],
            selected_in_greedy_order=selected_ids,
            nodes=nodes,
            edges=edges,
            all_branches=all_branches,
            remaining_branches=remaining,
            selection_steps=selection_steps,
            cost_tracker=tracker,
            budget_frozen=stopped_by_budget,
            cluster_radii=cluster_radii,
            cluster_stabilities=[],
            cluster_member_counts=cluster_member_counts,
            clustering_ms=clustering_ms,
            first_arrival_semantics=(
                "posterior_paths" if config.measure_propagation else "deterministic_first_arrival"
            ),
            transition=transition,
            path_hypotheses=paths,
            information_atoms=atoms,
            domains={branch.branch_id: branch.domain_ids for branch in all_branches},
            bounds=bounds_by_branch,
            certificate_status=certificate_status,
            diagnostics=diagnostics,
            context_hash=context_hash,
            context_plan=context_plan,
        )

    def _expand_one(
        self,
        frontier: List[Tuple[Tuple[float, ...], str, Branch]],
        index: ExactInnerProductIndex,
        query_vector: np.ndarray,
        memory_by_id: Dict[str, Memory],
        direct_scores: Dict[str, float],
        nodes: Dict[str, TreeNode],
        edges: List[Tuple[Optional[str], str]],
        push_sibling_branches,
        discovery_order: int,
        branch_audit_candidates: Dict[str, List[str]],
        budget: SearchBudget,
        tracker: CostTracker,
        excluded_candidate_ids: Sequence[str],
    ) -> bool:
        _priority, _branch_id, branch = heapq.heappop(frontier)
        if branch.depth >= self.config.max_depth or not tracker.can_search_core():
            return False
        capacity = min(self.config.branch_width, budget.max_unique_nodes - len(nodes))
        if capacity <= 0:
            return False
        candidates = tracker.search_core(
            index,
            branch.probe,
            capacity,
            exclude=set(nodes) | set(excluded_candidate_ids),
        )
        branch_audit_candidates[branch.branch_id] = [memory_id for memory_id, _score in candidates]
        children_by_parent: Dict[str, List[str]] = {}
        for offset, (candidate_id, _probe_score) in enumerate(candidates):
            if candidate_id in nodes:
                continue
            parent_id = min(
                branch.member_ids,
                key=lambda memory_id: (
                    -min(
                        nodes[memory_id].reachability,
                        nonnegative_cosine(nodes[memory_id].vector, index.vector(candidate_id)),
                    ),
                    memory_id,
                ),
            )
            edge_similarity = nonnegative_cosine(nodes[parent_id].vector, index.vector(candidate_id))
            raw_reachability = min(nodes[parent_id].reachability, edge_similarity)
            candidate_direct_score = direct_scores.setdefault(
                candidate_id,
                nonnegative_cosine(query_vector, index.vector(candidate_id)),
            )
            reachability = self._anchor_reachability(raw_reachability, candidate_direct_score)
            ancestor_vectors = []
            current: Optional[str] = parent_id
            while current is not None:
                ancestor = nodes[current]
                ancestor_vectors.append(ancestor.reachability * ancestor.vector)
                current = ancestor.parent_id
            ancestor_vectors.reverse()
            vector = index.vector(candidate_id)
            if self.config.feature_mode == "path_conditioned":
                innovation = path_conditioned_innovation(vector, reachability, ancestor_vectors)
            else:
                unit_vector = np.asarray(vector, dtype=np.float64)
                unit_vector /= max(1.0, float(np.linalg.norm(unit_vector)))
                innovation = reachability * unit_vector
            nodes[candidate_id] = TreeNode(
                memory=memory_by_id[candidate_id],
                vector=vector,
                parent_id=parent_id,
                depth=nodes[parent_id].depth + 1,
                direct_score=candidate_direct_score,
                reachability=reachability,
                innovation=innovation,
                bridge_lift=max(0.0, raw_reachability - candidate_direct_score),
                discovery_order=discovery_order + offset,
            )
            edges.append((parent_id, candidate_id))
            children_by_parent.setdefault(parent_id, []).append(candidate_id)
        for child_ids in children_by_parent.values():
            push_sibling_branches(child_ids, depth=nodes[child_ids[0]].depth)
        return bool(candidates)
