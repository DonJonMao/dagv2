from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import List, Sequence

import numpy as np

from .math_utils import effective_rank, normalize


@dataclass(frozen=True)
class ClusterResult:
    member_positions: tuple[int, ...]
    probe: np.ndarray
    radius_radians: float


def _initial_centers(vectors: np.ndarray, count: int) -> np.ndarray:
    chosen = [0]
    while len(chosen) < count:
        similarity = np.dot(vectors, vectors[chosen].T)
        nearest_distance = 1.0 - np.max(similarity, axis=1)
        nearest_distance[chosen] = -1.0
        chosen.append(int(np.argmax(nearest_distance)))
    return vectors[chosen].copy()


def spherical_kmeans(vectors: np.ndarray, count: int, max_iterations: int = 100) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if count < 1 or count > len(vectors):
        raise ValueError("cluster count must be in [1, number of vectors]")
    if count == 1:
        return np.zeros(len(vectors), dtype=np.int64)
    centers = _initial_centers(vectors, count)
    labels = np.full(len(vectors), -1, dtype=np.int64)
    for _ in range(max_iterations):
        new_labels = np.argmax(np.dot(vectors, centers.T), axis=1).astype(np.int64)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        empty_clusters = [index for index in range(count) if not np.any(labels == index)]
        if empty_clusters:
            assigned_similarity = np.max(np.dot(vectors, centers.T), axis=1)
            candidate_order = sorted(range(len(vectors)), key=lambda index: (assigned_similarity[index], index))
            claimed: set[int] = set()
            for cluster_index in empty_clusters:
                eligible = [
                    index
                    for index in candidate_order
                    if index not in claimed and np.count_nonzero(labels == labels[index]) > 1
                ]
                if not eligible:
                    eligible = [index for index in candidate_order if index not in claimed]
                replacement = eligible[0]
                claimed.add(replacement)
                labels[replacement] = cluster_index
        for cluster_index in range(count):
            positions = np.flatnonzero(labels == cluster_index)
            summed = vectors[positions].sum(axis=0)
            if np.linalg.norm(summed) <= 1e-12:
                centers[cluster_index] = vectors[int(positions[0])]
            else:
                centers[cluster_index] = normalize(summed)
    return labels


def _merge_small_clusters(vectors: np.ndarray, labels: np.ndarray, min_cluster_size: int) -> np.ndarray:
    if min_cluster_size <= 1:
        return labels
    counts = {int(label): int(np.count_nonzero(labels == label)) for label in np.unique(labels)}
    valid = [label for label, count in counts.items() if count >= min_cluster_size]
    if not valid:
        return np.zeros(len(labels), dtype=np.int64)
    centers = {}
    for label in valid:
        positions = np.flatnonzero(labels == label)
        summed = vectors[positions].sum(axis=0)
        centers[label] = normalize(summed) if np.linalg.norm(summed) > 1e-12 else vectors[int(positions[0])]
    merged = labels.copy()
    for label in sorted(set(counts) - set(valid)):
        positions = np.flatnonzero(merged == label)
        target = min(
            valid,
            key=lambda candidate: (
                -float(np.mean(np.dot(vectors[positions], centers[candidate]))),
                candidate,
            ),
        )
        merged[positions] = target
    relabel = {label: index for index, label in enumerate(sorted(int(value) for value in np.unique(merged)))}
    return np.asarray([relabel[int(label)] for label in merged], dtype=np.int64)


def cluster_siblings(
    vectors: np.ndarray,
    reachabilities: Sequence[float],
    mode: str = "effective_rank",
    fixed_count: int = 4,
    max_clusters: int = 8,
    min_cluster_size: int = 1,
    disable_compression: bool | None = None,
) -> List[ClusterResult]:
    """Cluster sibling directions under one of the runtime-selectable modes."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if len(vectors) == 0:
        return []
    if disable_compression is not None:  # compatibility with the pre-runtime API
        mode = "none" if disable_compression else "effective_rank"
    if mode not in {"none", "fixed", "effective_rank"}:
        raise ValueError("cluster mode must be none, fixed, or effective_rank")
    capacity = max(1, len(vectors) // max(1, min_cluster_size))
    if mode == "none":
        count = len(vectors)
    elif mode == "fixed":
        count = min(len(vectors), capacity, fixed_count)
    else:
        count = min(len(vectors), capacity, max_clusters, max(1, int(ceil(effective_rank(vectors)))))
    labels = spherical_kmeans(vectors, count)
    labels = _merge_small_clusters(vectors, labels, min_cluster_size)
    results: List[ClusterResult] = []
    weights = np.asarray(reachabilities, dtype=np.float64)
    for cluster_index in sorted(int(value) for value in np.unique(labels)):
        positions = np.flatnonzero(labels == cluster_index)
        cluster_vectors = vectors[positions]
        weighted_sum = (weights[positions, None] * cluster_vectors).sum(axis=0)
        if np.linalg.norm(weighted_sum) <= 1e-12:
            gram = np.dot(cluster_vectors, cluster_vectors.T)
            medoid_local = int(np.argmax(gram.sum(axis=1)))
            probe = cluster_vectors[medoid_local].copy()
        else:
            probe = normalize(weighted_sum)
        angles = np.arccos(np.clip(np.dot(cluster_vectors, probe), -1.0, 1.0))
        results.append(
            ClusterResult(
                member_positions=tuple(int(position) for position in positions),
                probe=probe,
                radius_radians=float(np.max(angles)),
            )
        )
    return results
