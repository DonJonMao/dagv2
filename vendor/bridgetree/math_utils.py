from __future__ import annotations

from numbers import Real
from typing import Iterable, Sequence

import numpy as np


def normalize(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    if value.ndim != 1 or value.size == 0:
        raise ValueError("expected a non-empty one-dimensional vector")
    if isinstance(eps, (bool, np.bool_)) or not isinstance(eps, Real):
        raise ValueError("eps must be a finite non-negative number")
    eps = float(eps)
    if not np.isfinite(eps) or eps < 0.0:
        raise ValueError("eps must be a finite non-negative number")
    if not np.all(np.isfinite(value)):
        raise ValueError("cannot normalize a non-finite vector")
    norm = float(np.linalg.norm(value))
    if norm <= eps:
        raise ValueError("cannot normalize a zero vector")
    return value / norm


def normalize_rows(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim != 2:
        raise ValueError("expected a two-dimensional matrix")
    if isinstance(eps, (bool, np.bool_)) or not isinstance(eps, Real):
        raise ValueError("eps must be a finite non-negative number")
    eps = float(eps)
    if not np.isfinite(eps) or eps < 0.0:
        raise ValueError("eps must be a finite non-negative number")
    if not np.all(np.isfinite(value)):
        raise ValueError("embedding matrix contains non-finite values")
    norms = np.linalg.norm(value, axis=1, keepdims=True)
    if np.any(norms <= eps):
        raise ValueError("embedding matrix contains a zero vector")
    return value / norms


def nonnegative_cosine(left: np.ndarray, right: np.ndarray) -> float:
    lhs = np.asarray(left, dtype=np.float64)
    rhs = np.asarray(right, dtype=np.float64)
    if lhs.ndim != 1 or rhs.ndim != 1 or lhs.shape != rhs.shape or lhs.size == 0:
        raise ValueError("cosine vectors must be non-empty one-dimensional vectors with equal dimensions")
    if not np.all(np.isfinite(lhs)) or not np.all(np.isfinite(rhs)):
        raise ValueError("cosine vectors must be finite")
    lhs_norm = float(np.linalg.norm(lhs))
    rhs_norm = float(np.linalg.norm(rhs))
    if lhs_norm <= 1e-12 or rhs_norm <= 1e-12:
        return 0.0
    return max(0.0, float(np.clip(np.dot(lhs, rhs) / (lhs_norm * rhs_norm), -1.0, 1.0)))


def angular_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Unit-sphere distance in turns (the specified ``acos/pi`` metric)."""
    left_value = np.asarray(left, dtype=np.float64)
    right_value = np.asarray(right, dtype=np.float64)
    if left_value.ndim != 1 or right_value.ndim != 1 or left_value.shape != right_value.shape or left_value.size == 0:
        raise ValueError("angular distance vectors must be non-empty one-dimensional vectors with equal dimensions")
    if not np.all(np.isfinite(left_value)) or not np.all(np.isfinite(right_value)):
        raise ValueError("angular distance vectors must be finite")
    left_norm = np.linalg.norm(left_value)
    right_norm = np.linalg.norm(right_value)
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        raise ValueError("angular distance requires non-zero vectors")
    cosine = float(np.dot(left_value, right_value) / (left_norm * right_norm))
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)) / np.pi)


def angular_rank_kernel(vectors: np.ndarray, ids: Sequence[str] | None = None) -> np.ndarray:
    """Build the no-bandwidth empirical angular-rank kernel.

    For each row, distances are ordered by ``(distance, memory_id)``.  The
    explicit ID tie-break makes equal embeddings deterministic while retaining
    the rank formula from the TMIC specification.
    """
    raw = np.asarray(vectors, dtype=np.float64)
    # ``np.asarray([])`` is a useful empty-bank spelling in callers that do
    # not know the embedding dimension yet.  Preserve the historical empty
    # kernel rather than reporting a misleading dimensionality error.
    if raw.size == 0:
        if ids is not None and len(ids) != 0:
            raise ValueError("ids and vectors must have equal length")
        return np.empty((0, 0), dtype=np.float64)
    value = normalize_rows(raw)
    count = len(value)
    if count == 0:
        return np.empty((0, 0), dtype=np.float64)
    labels = [str(item) for item in (ids if ids is not None else range(count))]
    if len(labels) != count:
        raise ValueError("ids and vectors must have equal length")
    if len(set(labels)) != len(labels):
        raise ValueError("ids must be unique")
    distances = np.arccos(np.clip(value @ value.T, -1.0, 1.0)) / np.pi
    kernel = np.zeros((count, count), dtype=np.float64)
    for row in range(count):
        order = sorted(range(count), key=lambda col: (float(distances[row, col]), labels[col]))
        for rank, col in enumerate(order, start=1):
            kernel[row, col] = (count - rank + 1) / count
    return kernel


def effective_rank(vectors: np.ndarray, eps: float = 1e-12) -> float:
    """Participation-ratio effective rank of the sibling Gram matrix."""
    value = np.asarray(vectors, dtype=np.float64)
    if (
        isinstance(eps, (bool, np.bool_))
        or not isinstance(eps, Real)
        or not np.isfinite(float(eps))
        or float(eps) < 0.0
    ):
        raise ValueError("effective-rank eps must be finite and non-negative")
    if value.size == 0:
        if value.ndim not in {1, 2}:
            raise ValueError("effective-rank vectors must be a finite two-dimensional matrix")
        return 0.0
    if value.ndim != 2 or not np.all(np.isfinite(value)):
        raise ValueError("effective-rank vectors must be a finite two-dimensional matrix")
    gram = np.dot(value, value.T)
    eigenvalues = np.maximum(np.linalg.eigvalsh(gram), 0.0)
    denominator = float(np.dot(eigenvalues, eigenvalues))
    if denominator <= eps:
        return 1.0
    return float(eigenvalues.sum() ** 2 / denominator)


def path_conditioned_innovation(
    vector: np.ndarray,
    reachability: float,
    ancestor_weighted_vectors: Sequence[np.ndarray],
) -> np.ndarray:
    """Compute (I + B_A B_A^T)^(-1/2) (rho * z) by a thin SVD.

    The implementation is Eq. (58) of the design and never forms a d-by-d
    inverse square root.
    """
    if isinstance(reachability, (bool, np.bool_)):
        raise ValueError("reachability must be finite and non-negative")
    try:
        reachability_value = float(reachability)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("reachability must be finite and non-negative") from exc
    if not np.isfinite(reachability_value) or not 0.0 <= reachability_value <= 1.0:
        raise ValueError("reachability must be finite and lie in [0, 1]")
    candidate = np.asarray(vector, dtype=np.float64)
    if candidate.ndim != 1 or candidate.size == 0 or not np.all(np.isfinite(candidate)):
        raise ValueError("vector must be a finite non-empty one-dimensional vector")
    weighted = reachability_value * candidate
    if not ancestor_weighted_vectors:
        return weighted.copy()
    columns_list = []
    for item in ancestor_weighted_vectors:
        value = np.asarray(item, dtype=np.float64)
        if value.ndim != 1 or value.shape != candidate.shape or not np.all(np.isfinite(value)):
            raise ValueError("ancestor vectors must be finite one-dimensional vectors matching vector")
        columns_list.append(value)
    columns = np.column_stack(columns_list)
    u, singular, _ = np.linalg.svd(columns, full_matrices=False)
    projection = np.dot(u.T, weighted)
    factor = 1.0 / np.sqrt(1.0 + singular**2) - 1.0
    return weighted + np.dot(u, factor * projection)


def factor_conditioned_innovation(
    vector: np.ndarray,
    quality: float,
    ancestor_vectors: Sequence[np.ndarray] = (),
    ancestor_weights: Sequence[float] = (),
) -> np.ndarray:
    """Return ``sqrt(r) (I + B B.T)^(-1/2) v`` from a thin factor.

    ``B[:, a] = sqrt(w[a] * r[a]) * v[a]``.  Keeping the factor as a
    ``d x |A|`` matrix avoids materialising the embedding-dimensional scatter
    matrix; the formula below is the corresponding low-rank update applied to
    one vector.
    """
    candidate = np.asarray(vector, dtype=np.float64).reshape(-1)
    if candidate.ndim != 1 or candidate.size == 0 or not np.all(np.isfinite(candidate)):
        raise ValueError("vector must be a finite non-empty one-dimensional vector")
    norm = float(np.linalg.norm(candidate))
    if norm <= 1e-12:
        raise ValueError("vector must be non-zero")
    candidate = candidate / norm
    r = float(quality)
    if not np.isfinite(r) or r < 0.0 or r > 1.0:
        raise ValueError("quality must lie in [0, 1]")
    vectors = list(ancestor_vectors)
    weights = list(ancestor_weights)
    if len(weights) == 0:
        weights = [1.0] * len(vectors)
    if len(vectors) != len(weights):
        raise ValueError("ancestor vectors and weights must have equal length")
    columns: list[np.ndarray] = []
    for value, raw_weight in zip(vectors, weights):
        ancestor = np.asarray(value, dtype=np.float64).reshape(-1)
        if ancestor.shape != candidate.shape or not np.all(np.isfinite(ancestor)):
            raise ValueError("ancestor vectors must match vector and be finite")
        ancestor_norm = float(np.linalg.norm(ancestor))
        if ancestor_norm <= 1e-12:
            raise ValueError("ancestor vectors must be non-zero")
        weight = float(raw_weight)
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError("ancestor weights must be finite and non-negative")
        coefficient = np.sqrt(weight)
        columns.append(coefficient * (ancestor / ancestor_norm))
    if not columns:
        return np.sqrt(r) * candidate
    B = np.column_stack(columns)
    U, singular, _ = np.linalg.svd(B, full_matrices=False)
    correction = (1.0 / np.sqrt(1.0 + singular * singular) - 1.0) * (U.T @ candidate)
    return np.sqrt(r) * (candidate + U @ correction)


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("expected a square matrix")
    if not np.all(np.isfinite(value)):
        raise ValueError("matrix must be finite")
    return (value + value.T) * 0.5


def psd_eigendecomposition(matrix: np.ndarray, tolerance: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Return a symmetric PSD eigendecomposition with numerical clipping."""
    raw = np.asarray(matrix, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1] or not np.all(np.isfinite(raw)):
        raise ValueError("matrix must be a finite square matrix")
    if isinstance(tolerance, (bool, np.bool_)) or not isinstance(tolerance, Real):
        raise ValueError("tolerance must be finite and non-negative")
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    values, vectors = np.linalg.eigh(symmetrize(raw))
    scale = max(1.0, float(np.max(np.abs(values))) if len(values) else 1.0)
    if np.min(values, initial=0.0) < -tolerance * scale:
        raise ValueError("matrix is not positive semidefinite")
    values = np.where(values < 0.0, 0.0, values)
    return values, vectors


def psd_inverse_sqrt(matrix: np.ndarray, tolerance: float = 1e-12) -> np.ndarray:
    """Compute a stable PSD inverse square root without forming an inverse."""
    values, vectors = psd_eigendecomposition(matrix, tolerance=tolerance)
    safe = np.where(values > tolerance, values ** -0.5, 0.0)
    return (vectors * safe) @ vectors.T


def stable_logdet(matrix: np.ndarray, tolerance: float = 1e-12) -> float:
    """Log determinant via Cholesky, with an eig/SVD-safe fallback."""
    value = symmetrize(matrix)
    try:
        chol = np.linalg.cholesky(value)
        return float(2.0 * np.log(np.diag(chol)).sum())
    except np.linalg.LinAlgError:
        eigenvalues, _vectors = psd_eigendecomposition(value, tolerance=tolerance)
        if np.any(eigenvalues <= tolerance):
            raise FloatingPointError("matrix is not positive definite") from None
        return float(np.log(eigenvalues).sum())


def logdet_matrix_marginal(candidate: np.ndarray, base: np.ndarray | None = None) -> float:
    """Stable ``logdet(A+Q)-logdet(A)`` for PSD matrices."""
    raw_q = np.asarray(candidate, dtype=np.float64)
    if raw_q.ndim == 1:
        if raw_q.size == 0 or not np.all(np.isfinite(raw_q)):
            raise ValueError("candidate must be a finite non-empty vector or square matrix")
        raw_q = np.outer(raw_q, raw_q)
    if raw_q.ndim != 2 or raw_q.shape[0] != raw_q.shape[1] or not np.all(np.isfinite(raw_q)):
        raise ValueError("candidate must be a finite square matrix or vector")
    q = symmetrize(raw_q)
    if not is_psd(q):
        raise ValueError("candidate must be positive semidefinite")
    if base is None:
        base = np.eye(q.shape[0], dtype=np.float64)
    a_raw = np.asarray(base, dtype=np.float64)
    if a_raw.ndim != 2 or a_raw.shape[0] != a_raw.shape[1] or not np.all(np.isfinite(a_raw)):
        raise ValueError("base must be a finite square matrix")
    a = symmetrize(a_raw)
    if not np.all(np.linalg.eigvalsh(a) > 0.0):
        raise ValueError("base must be positive definite")
    if q.shape != a.shape:
        raise ValueError("candidate and base matrices must have equal shape")
    value = float(stable_logdet(a + q) - stable_logdet(a))
    # The exact quantity is non-negative for PSD Q; suppress only floating
    # point noise around zero, while preserving a useful error for genuinely
    # invalid matrices in the stable_logdet calls above.
    return max(0.0, value) if value > -1e-12 else value


# Descriptive aliases used by the TMIC notes and downstream audit scripts.
def matrix_marginal(candidate: np.ndarray, base: np.ndarray | None = None) -> float:
    return logdet_matrix_marginal(candidate, base)


def psd_min_eigenvalue(matrix: np.ndarray) -> float:
    values, _vectors = np.linalg.eigh(symmetrize(np.asarray(matrix, dtype=np.float64)))
    return float(np.min(values)) if len(values) else 0.0


def is_psd(matrix: np.ndarray, tolerance: float = 1e-10) -> bool:
    if isinstance(tolerance, (bool, np.bool_)) or not isinstance(tolerance, Real):
        raise ValueError("PSD tolerance must be finite and non-negative")
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("PSD tolerance must be finite and non-negative")
    value = symmetrize(np.asarray(matrix, dtype=np.float64))
    eigenvalues = np.linalg.eigvalsh(value)
    minimum = float(np.min(eigenvalues)) if len(eigenvalues) else 0.0
    scale = max(1.0, float(np.max(np.abs(eigenvalues))) if len(eigenvalues) else 1.0)
    return minimum >= -tolerance * scale


def logdet_value(features: Iterable[np.ndarray]) -> float:
    rows = [np.asarray(feature, dtype=np.float64) for feature in features]
    if not rows:
        return 0.0
    if any(row.ndim != 1 or row.size == 0 or not np.all(np.isfinite(row)) for row in rows):
        raise ValueError("features must be finite non-empty one-dimensional vectors")
    dimension = rows[0].shape
    if any(row.shape != dimension for row in rows):
        raise ValueError("feature dimensions must match")
    phi = np.vstack(rows)
    sign, value = np.linalg.slogdet(np.eye(len(phi), dtype=np.float64) + np.dot(phi, phi.T))
    if sign <= 0:
        raise FloatingPointError("I + Phi Phi^T must be positive definite")
    return float(value)


def logdet_marginal(candidate: np.ndarray, selected_features: Sequence[np.ndarray]) -> float:
    """Exact Eq. (41), evaluated in k-space with the Woodbury identity."""
    phi = np.asarray(candidate, dtype=np.float64)
    if phi.ndim != 1 or phi.size == 0 or not np.all(np.isfinite(phi)):
        raise ValueError("candidate feature must be a finite non-empty vector")
    norm_sq = float(np.dot(phi, phi))
    if not selected_features:
        return float(np.log1p(max(0.0, norm_sq)))
    selected_rows = [np.asarray(feature, dtype=np.float64) for feature in selected_features]
    if any(row.ndim != 1 or row.shape != phi.shape or not np.all(np.isfinite(row)) for row in selected_rows):
        raise ValueError("selected feature dimensions must match and values must be finite")
    selected = np.vstack(selected_rows)
    cross = np.dot(selected, phi)
    small = np.eye(len(selected), dtype=np.float64) + np.dot(selected, selected.T)
    conditional_sq = norm_sq - float(np.dot(cross, np.linalg.solve(small, cross)))
    scale = max(1.0, norm_sq)
    if conditional_sq < -1e-10 * scale:
        raise FloatingPointError("conditional feature energy is materially negative")
    return float(np.log1p(max(0.0, conditional_sq)))


def posterior_error(epsilons: Sequence[float]) -> float:
    if isinstance(epsilons, (str, bytes)):
        raise ValueError("posterior error terms must be a finite sequence")
    k = len(epsilons)
    if k == 0:
        return 0.0
    values = []
    for epsilon in epsilons:
        if isinstance(epsilon, (bool, np.bool_)):
            raise ValueError("posterior error terms must be finite and non-negative")
        value = float(epsilon)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("posterior error terms must be finite and non-negative")
        values.append(value)
    return float(sum(((1.0 - 1.0 / k) ** (k - j)) * eps for j, eps in enumerate(values, start=1)))
