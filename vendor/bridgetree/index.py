from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .math_utils import normalize, normalize_rows


@dataclass(frozen=True)
class SearchStats:
    requested_top_k: int
    returned: int
    backend_request_sizes: tuple[int, ...]


class ExactInnerProductIndex:
    """Deterministic exact float32 index for validation and small banks."""

    def __init__(self, ids: Sequence[str], vectors: np.ndarray, exclusion_margin: int = 32):
        if len(ids) != len(vectors):
            raise ValueError("ids and vectors must have equal length")
        if len(set(ids)) != len(ids):
            raise ValueError("memory ids must be unique")
        self.ids = list(ids)
        self.vectors = normalize_rows(np.asarray(vectors, dtype=np.float32)).astype(np.float32)
        self.position = {memory_id: index for index, memory_id in enumerate(self.ids)}
        self.exclusion_margin = exclusion_margin
        self.last_search_stats = SearchStats(0, 0, ())

    def vector(self, memory_id: str) -> np.ndarray:
        return self.vectors[self.position[memory_id]]

    def search(
        self,
        query: np.ndarray,
        top_k: int,
        exclude: Iterable[str] = (),
        max_backend_calls: int | None = None,
    ) -> List[Tuple[str, float]]:
        if top_k <= 0:
            self.last_search_stats = SearchStats(top_k, 0, ())
            return []
        query_vector = normalize(query).astype(np.float32)
        excluded = set(exclude)
        available = np.asarray(
            [index for index, memory_id in enumerate(self.ids) if memory_id not in excluded],
            dtype=np.int64,
        )
        if len(available) == 0:
            self.last_search_stats = SearchStats(top_k, 0, ())
            return []
        scores = np.dot(self.vectors, query_vector)
        count = min(top_k, len(available))
        if count < len(available):
            local_scores = scores[available]
            local_partition = np.argpartition(-local_scores, count - 1)[:count]
            cutoff = float(np.min(local_scores[local_partition]))
            stronger = [int(index) for index in available if float(scores[index]) > cutoff]
            tied = sorted(
                (int(index) for index in available if float(scores[index]) == cutoff),
                key=lambda index: self.ids[index],
            )
            candidate_positions = stronger + tied[: count - len(stronger)]
        else:
            candidate_positions = [int(index) for index in available]
        ranked = sorted(
            ((self.ids[index], float(scores[index])) for index in candidate_positions),
            key=lambda item: (-item[1], item[0]),
        )[:count]
        self.last_search_stats = SearchStats(top_k, len(ranked), (count,))
        return ranked


class FaissInnerProductIndex(ExactInnerProductIndex):
    """Exact FlatIP with adaptive exclusion over-fetch rather than full-bank search."""

    def __init__(self, ids: Sequence[str], vectors: np.ndarray, exclusion_margin: int = 32):
        super().__init__(ids, vectors, exclusion_margin=exclusion_margin)
        try:
            import faiss  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("faiss backend requested but faiss is not installed") from exc
        self._faiss = faiss.IndexFlatIP(self.vectors.shape[1])
        self._faiss.add(self.vectors)

    def search(
        self,
        query: np.ndarray,
        top_k: int,
        exclude: Iterable[str] = (),
        max_backend_calls: int | None = None,
    ) -> List[Tuple[str, float]]:
        if top_k <= 0:
            self.last_search_stats = SearchStats(top_k, 0, ())
            return []
        excluded = set(exclude)
        target = min(top_k, len(self.ids) - sum(memory_id in excluded for memory_id in self.ids))
        if target <= 0:
            self.last_search_stats = SearchStats(top_k, 0, ())
            return []
        query_vector = normalize(query).astype(np.float32)
        request_size = min(len(self.ids), max(target + self.exclusion_margin, target))
        request_sizes: list[int] = []
        results: List[Tuple[str, float]] = []
        while True:
            request_sizes.append(request_size)
            scores, positions = self._faiss.search(query_vector[None, :], request_size)
            results = []
            for position in positions[0]:
                if position < 0:
                    continue
                memory_id = self.ids[int(position)]
                if memory_id in excluded:
                    continue
                score = float(np.dot(self.vectors[int(position)], query_vector))
                results.append((memory_id, score))
            ranked = sorted(results, key=lambda item: (-item[1], item[0]))
            enough = len(ranked) >= target
            tie_may_be_truncated = False
            if enough and request_size < len(self.ids):
                kth_score = ranked[target - 1][1]
                last_backend_score = float(scores[0][-1])
                tie_may_be_truncated = np.isclose(kth_score, last_backend_score, atol=1e-7, rtol=0.0)
            call_limit_reached = max_backend_calls is not None and len(request_sizes) >= max_backend_calls
            if (enough and not tie_may_be_truncated) or request_size == len(self.ids) or call_limit_reached:
                results = ranked[:target]
                break
            request_size = min(len(self.ids), max(request_size + 1, request_size * 2))
        self.last_search_stats = SearchStats(top_k, len(results), tuple(request_sizes))
        return results


def build_index(
    backend: str,
    ids: Sequence[str],
    vectors: np.ndarray,
    exclusion_margin: int = 32,
) -> ExactInnerProductIndex:
    if backend == "exact":
        return ExactInnerProductIndex(ids, vectors, exclusion_margin=exclusion_margin)
    if backend == "faiss":
        return FaissInnerProductIndex(ids, vectors, exclusion_margin=exclusion_margin)
    raise ValueError(f"unknown index backend: {backend}")
