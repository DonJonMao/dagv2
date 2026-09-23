from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np

from .budget import CostSnapshot, CostTracker, SearchBudget
from .clustering import spherical_kmeans
from .index import ExactInnerProductIndex, build_index
from .math_utils import normalize, normalize_rows


@dataclass
class BaselineResult:
    selected_ids: List[str]
    cost_tracker: CostTracker
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def ann_calls(self) -> int:
        return self.cost_tracker.ann_calls_core

    @property
    def cost(self) -> CostSnapshot:
        return self.cost_tracker.snapshot()


def _runtime(
    ids: Sequence[str],
    vectors: np.ndarray,
    index: ExactInnerProductIndex | None,
    budget: SearchBudget | None,
    cost_tracker: CostTracker | None,
    index_backend: str,
    exclusion_margin: int,
) -> tuple[ExactInnerProductIndex, SearchBudget, CostTracker, float]:
    current_budget = budget or SearchBudget(max_unique_nodes=len(ids))
    tracker = cost_tracker or CostTracker(current_budget)
    index_build_ms = 0.0
    if index is None:
        started = time.perf_counter()
        index = build_index(index_backend, ids, vectors, exclusion_margin=exclusion_margin)
        index_build_ms = (time.perf_counter() - started) * 1000.0
        tracker.index_build_ms += index_build_ms
    return index, current_budget, tracker, index_build_ms


def _finish(tracker: CostTracker, started: float, selected_count: int, target_count: int, total: int) -> None:
    tracker.retrieval_core_ms = (time.perf_counter() - started) * 1000.0
    if selected_count < target_count:
        tracker.set_stop_reason("insufficient_candidates")
    elif not tracker.can_search_core() and tracker.cost_unique_count < total:
        tracker.set_stop_reason("search_budget")
    else:
        tracker.set_stop_reason("frontier_empty")


def dense_retrieval(
    ids: Sequence[str],
    vectors: np.ndarray,
    query_vector: np.ndarray,
    top_k: int,
    *,
    index: ExactInnerProductIndex | None = None,
    budget: SearchBudget | None = None,
    cost_tracker: CostTracker | None = None,
    index_backend: str = "exact",
    exclusion_margin: int = 32,
) -> BaselineResult:
    index, current_budget, tracker, _build_ms = _runtime(
        ids, vectors, index, budget, cost_tracker, index_backend, exclusion_margin
    )
    started = time.perf_counter()
    count = min(top_k, current_budget.max_unique_nodes)
    hits = tracker.search_core(index, query_vector, count)
    selected = [memory_id for memory_id, _ in hits]
    _finish(tracker, started, len(selected), min(top_k, len(ids)), len(ids))
    return BaselineResult(selected, tracker)


def _rf_entropy(scores: np.ndarray) -> float:
    scaled = 20.0 * scores
    probabilities = np.exp(scaled - scaled.max())
    probabilities /= probabilities.sum() + 1e-12
    return float(-(probabilities * np.log(probabilities + 1e-12)).mean())


def rfmem_route(scores: Sequence[float], entropy_threshold: float = 0.2) -> Tuple[str, Dict[str, Any]]:
    values = np.asarray(scores, dtype=np.float64)
    if len(values) == 0:
        return "slow", {"entropy": 0.0, "mean_score": 0.0, "rule": "no_hits"}
    mean_score = float(values.mean())
    entropy = _rf_entropy(np.sort(values)[::-1])
    if mean_score >= 0.60:
        mode, rule = "fast", "mean_high"
    elif mean_score <= 0.30:
        mode, rule = "slow", "mean_low"
    elif entropy < entropy_threshold:
        mode, rule = "fast", "low_entropy"
    else:
        mode, rule = "slow", "high_entropy"
    return mode, {"entropy": entropy, "mean_score": mean_score, "rule": rule}


@dataclass
class _RFBranch:
    query: np.ndarray
    score_sum: float
    hits: List[Tuple[str, float]]


def _recollection_search(
    ids: Sequence[str],
    index: ExactInnerProductIndex,
    query_vector: np.ndarray,
    top_k: int,
    budget: SearchBudget,
    tracker: CostTracker,
    depth: int | None,
    beam_width: int,
    fanout: int,
    alpha: float,
    threshold: float,
    mmr_lambda: float,
) -> List[str]:
    q0 = normalize(query_vector).astype(np.float32)
    beam = [_RFBranch(q0, 0.0, [])]
    seen: Set[str] = set()
    results: List[str] = []
    for level in range(depth or top_k):
        next_branches: List[_RFBranch] = []
        for branch in beam:
            if not tracker.can_search_core() or len(seen) >= budget.max_unique_nodes:
                break
            request = min((beam_width + level) * fanout, budget.max_unique_nodes - len(seen))
            raw = tracker.search_core(index, branch.query, request, exclude=seen)
            candidates = [(memory_id, score) for memory_id, score in raw if score >= threshold]
            if not candidates:
                continue
            candidate_vectors = np.vstack([index.vector(memory_id) for memory_id, _ in candidates])
            selected: List[Tuple[str, float]] = []
            for position, (memory_id, score) in enumerate(candidates):
                similarities = np.dot(candidate_vectors, candidate_vectors[position])
                max_similarity = max(
                    (float(value) for other, value in enumerate(similarities) if other != position),
                    default=0.0,
                )
                selected.append((memory_id, mmr_lambda * score - (1.0 - mmr_lambda) * max_similarity))
            count = min(beam_width, len(selected))
            labels = spherical_kmeans(candidate_vectors, count)
            for cluster_index in range(count):
                positions = np.flatnonzero(labels == cluster_index)
                group = [selected[int(position)] for position in positions]
                centroid = normalize(candidate_vectors[positions].mean(axis=0))
                next_query = normalize(alpha * branch.query + (1.0 - alpha) * centroid + q0)
                next_branches.append(_RFBranch(next_query, branch.score_sum + sum(score for _, score in group), group))
        if not next_branches:
            continue
        beam = sorted(
            next_branches,
            key=lambda branch: (-branch.score_sum, tuple(item[0] for item in branch.hits)),
        )[:beam_width]
        for branch in beam:
            for memory_id, _score in branch.hits:
                if memory_id not in seen:
                    seen.add(memory_id)
                    results.append(memory_id)
        if len(results) >= top_k or not tracker.can_search_core():
            break
    return results[:top_k]


def rfmem_recollection(
    ids: Sequence[str],
    vectors: np.ndarray,
    query_vector: np.ndarray,
    top_k: int,
    depth: int | None = None,
    beam_width: int = 4,
    fanout: int = 3,
    alpha: float = 0.8,
    threshold: float = 0.3,
    mmr_lambda: float = 0.95,
    *,
    index: ExactInnerProductIndex | None = None,
    budget: SearchBudget | None = None,
    cost_tracker: CostTracker | None = None,
    index_backend: str = "exact",
    exclusion_margin: int = 32,
) -> BaselineResult:
    index, current_budget, tracker, _build_ms = _runtime(
        ids, vectors, index, budget, cost_tracker, index_backend, exclusion_margin
    )
    started = time.perf_counter()
    results = _recollection_search(
        ids,
        index,
        query_vector,
        top_k,
        current_budget,
        tracker,
        depth,
        beam_width,
        fanout,
        alpha,
        threshold,
        mmr_lambda,
    )
    _finish(tracker, started, len(results), min(top_k, len(ids)), len(ids))
    return BaselineResult(results, tracker)


def rfmem(
    ids: Sequence[str],
    vectors: np.ndarray,
    query_vector: np.ndarray,
    top_k: int,
    entropy_threshold: float = 0.2,
    *,
    index: ExactInnerProductIndex | None = None,
    budget: SearchBudget | None = None,
    cost_tracker: CostTracker | None = None,
    index_backend: str = "exact",
    exclusion_margin: int = 32,
) -> BaselineResult:
    index, current_budget, tracker, _build_ms = _runtime(
        ids, vectors, index, budget, cost_tracker, index_backend, exclusion_margin
    )
    started = time.perf_counter()
    probe = tracker.search_core(index, query_vector, min(10, current_budget.max_unique_nodes, len(ids)))
    mode, diagnostics = rfmem_route([score for _, score in probe], entropy_threshold)
    if mode == "fast":
        hits = list(probe[:top_k])
        if len(hits) < min(top_k, len(ids)) and tracker.can_search_core():
            seen_probe = {memory_id for memory_id, _score in probe}
            hits.extend(
                tracker.search_core(
                    index,
                    query_vector,
                    top_k - len(hits),
                    exclude=seen_probe,
                )
            )
        selected = [memory_id for memory_id, score in hits[:top_k] if score >= 0.3]
    else:
        selected = _recollection_search(
            ids,
            index,
            query_vector,
            top_k,
            current_budget,
            tracker,
            None,
            4,
            3,
            0.8,
            0.3,
            0.95,
        )
    _finish(tracker, started, len(selected), min(top_k, len(ids)), len(ids))
    return BaselineResult(selected, tracker, {**diagnostics, "route": mode})


def cluster_prf(
    ids: Sequence[str],
    vectors: np.ndarray,
    query_vector: np.ndarray,
    top_k: int,
    first_width: int = 12,
    *,
    index: ExactInnerProductIndex | None = None,
    budget: SearchBudget | None = None,
    cost_tracker: CostTracker | None = None,
    index_backend: str = "exact",
    exclusion_margin: int = 32,
) -> BaselineResult:
    vectors = normalize_rows(vectors).astype(np.float32)
    index, current_budget, tracker, _build_ms = _runtime(
        ids, vectors, index, budget, cost_tracker, index_backend, exclusion_margin
    )
    started = time.perf_counter()
    initial = tracker.search_core(index, query_vector, min(first_width, current_budget.max_unique_nodes, len(ids)))
    if not initial:
        _finish(tracker, started, 0, min(top_k, len(ids)), len(ids))
        return BaselineResult([], tracker)
    initial_ids = [memory_id for memory_id, _ in initial]
    initial_vectors = np.vstack([index.vector(memory_id) for memory_id in initial_ids])
    count = min(len(initial_ids), max(1, int(np.ceil(np.sqrt(len(initial_ids))))))
    labels = spherical_kmeans(initial_vectors, count)
    scores: Dict[str, float] = {memory_id: score for memory_id, score in initial}
    for cluster_index in range(count):
        if not tracker.can_search_core() or tracker.cost_unique_count >= current_budget.max_unique_nodes:
            break
        positions = np.flatnonzero(labels == cluster_index)
        probe = normalize(initial_vectors[positions].mean(axis=0))
        remaining = current_budget.max_unique_nodes - tracker.cost_unique_count
        for memory_id, score in tracker.search_core(index, probe, min(top_k, remaining)):
            scores[memory_id] = max(scores.get(memory_id, -1.0), score)
    selected = sorted(scores, key=lambda memory_id: (-scores[memory_id], memory_id))[:top_k]
    _finish(tracker, started, len(selected), min(top_k, len(ids)), len(ids))
    return BaselineResult(selected, tracker)
