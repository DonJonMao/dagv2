"""Query-conditioned PSD information geometry (I operator).

This module supplies a frozen state basis and a log-determinant objective.  It
does not estimate answer labels or train a utility model: option embeddings are
used only to choose an orthogonal coordinate system for a deterministic
linear-Gaussian/Fisher-style surrogate.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np

from .math_utils import (
    factor_conditioned_innovation,
    is_psd,
    logdet_matrix_marginal,
    normalize,
    psd_eigendecomposition,
    psd_inverse_sqrt,
    stable_logdet,
    symmetrize,
)
from .measure import FrozenGraphMeasure
from .types import (
    FrozenProposalGraph,
    InformationAtom,
    PathHypothesis,
    QualityRecord,
    SemanticAtom,
)


def _strict_int(value: Any, name: str, *, positive: bool = False, nonnegative: bool = False) -> int:
    """Validate an integer protocol field without bool/fraction truncation."""
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


def _strict_float(value: Any, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(result) or (nonnegative and result < 0.0):
        raise ValueError(f"{name} must be finite" + (" and non-negative" if nonnegative else ""))
    return result


@dataclass(frozen=True)
class StateBasis:
    basis: np.ndarray
    mode: str
    fallback: str | None
    rank: int
    options_hash: str
    model_fingerprint: str

    def __post_init__(self) -> None:
        value = np.asarray(self.basis, dtype=np.float64).copy()
        if value.ndim != 2 or value.size == 0 or not np.all(np.isfinite(value)):
            raise ValueError("state basis must be a finite non-empty matrix")
        if value.shape[1] <= 0 or value.shape[1] > value.shape[0]:
            raise ValueError("state basis rank must be in [1, embedding_dimension]")
        rank = _strict_int(self.rank, "state basis rank", positive=True)
        if rank != int(value.shape[1]):
            raise ValueError("state basis rank does not match its matrix")
        gram = value.T @ value
        if not np.allclose(gram, np.eye(value.shape[1]), atol=1e-8, rtol=1e-8):
            raise ValueError("state basis columns must be orthonormal")
        value.setflags(write=False)
        object.__setattr__(self, "basis", value)
        object.__setattr__(self, "mode", str(self.mode))
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "options_hash", str(self.options_hash))
        object.__setattr__(self, "model_fingerprint", str(self.model_fingerprint))

    def public_dict(self) -> dict[str, Any]:
        return {
            "basis": np.asarray(self.basis, dtype=np.float64).tolist(),
            "mode": self.mode,
            "fallback": self.fallback,
            # Explicit spelling used by the protocol/audit schema; retain
            # ``fallback`` for older consumers.
            "state_basis_fallback": self.fallback or "none",
            "rank": self.rank,
            "options_hash": self.options_hash,
            "model_fingerprint": self.model_fingerprint,
        }


def _options_hash(options: Sequence[Any] | None) -> str:
    payload = json.dumps(list(options or ()), ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize_options(options: Sequence[Any] | str | None) -> list[Any]:
    if options is None:
        return []
    if isinstance(options, str):
        if not options.strip():
            return []
        try:
            parsed = ast.literal_eval(options)
        except (SyntaxError, ValueError):
            return [options]
        values = list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]
    else:
        # A mapping/set has no stable answer-option order and accepting it via
        # ``list(...)`` makes the frozen basis depend on an incidental
        # iteration order.  Require a real sequence for deterministic
        # provenance; strings were handled above as a legacy literal format.
        if isinstance(options, (Mapping, set, frozenset)):
            raise ValueError("options must be a sequence or string")
        try:
            values = list(options)
        except TypeError as exc:
            raise ValueError("options must be a sequence or string") from exc
    if any(not isinstance(value, str) for value in values):
        raise ValueError("answer options must be strings")
    return values


class StateBasisProvider:
    """Build a frozen orthogonal basis from answer-option embeddings.

    The provider accepts precomputed embeddings (preferred for train-free
    reproducibility) or an ``embed`` callable.  ``fit``/``build``/``provide``
    are aliases to make integration with older clients painless; all return
    the basis matrix directly.
    """

    def __init__(
        self,
        mode: str = "option_contrast",
        *,
        model_fingerprint: str = "",
        endpoint: str = "",
        tolerance: float = 1e-12,
    ) -> None:
        if mode not in {"option_contrast", "identity"}:
            raise ValueError("state basis mode must be option_contrast or identity")
        self.mode = mode
        self.model_fingerprint = str(model_fingerprint or endpoint)
        self.endpoint = str(endpoint)
        self.tolerance = _strict_float(tolerance, "state basis tolerance", nonnegative=True)
        self._state: StateBasis | None = None
        self._frozen = False
        # Keep a private digest of the exact embedding response used to build
        # the basis.  Option text alone is not sufficient provenance when an
        # embedding service changes its output for the same options.
        self._embedding_hash: str | None = None

    @property
    def state(self) -> StateBasis | None:
        return self._state

    @property
    def basis(self) -> np.ndarray | None:
        """Read-only convenience alias for the frozen basis matrix."""
        return None if self._state is None else self._state.basis.copy()

    @property
    def fallback(self) -> str | None:
        return self._state.fallback if self._state is not None else "uninitialized"

    @property
    def frozen(self) -> bool:
        return self._frozen

    @property
    def options_hash(self) -> str:
        return self._state.options_hash if self._state is not None else ""

    @property
    def diagnostics(self) -> dict[str, Any]:
        if self._state is None:
            return {"state_basis_fallback": "uninitialized"}
        return {**self._state.public_dict(), "frozen": bool(self._frozen)}

    def build(
        self,
        option_embeddings: np.ndarray | Sequence[Sequence[float]] | None = None,
        *,
        options: Sequence[str] | str | None = None,
        embed: Callable[[Sequence[str]], np.ndarray] | None = None,
        dimension: int | None = None,
        options_hash: str | None = None,
    ) -> np.ndarray:
        if dimension is not None:
            dimension = _strict_int(dimension, "state basis dimension", positive=True)
        if self._frozen and self._state is not None:
            # A frozen provider is deliberately immutable.  Idempotent calls
            # with the same inputs return a copy, while an explicitly
            # different query/options embedding is rejected instead of being
            # silently ignored (which could make a provenance record claim a
            # basis for the wrong question).
            state_dimension = int(self._state.basis.shape[0])
            if dimension is not None and dimension != state_dimension:
                raise ValueError("state basis is frozen with a different dimension")
            normalized_options = _normalize_options(options) if options is not None else None
            if option_embeddings is not None:
                values = np.asarray(option_embeddings, dtype=np.float64)
                if values.ndim != 2 or not np.all(np.isfinite(values)):
                    raise ValueError("option embeddings must be a finite two-dimensional matrix")
                if values.shape[1] != state_dimension:
                    raise ValueError("state basis is frozen with a different embedding dimension")
                if normalized_options is not None and len(normalized_options) != len(values):
                    raise ValueError("options and option embeddings must have equal length")
                embedding_payload = json.dumps(
                    values.tolist(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                candidate_embedding_hash = hashlib.sha256(embedding_payload.encode("utf-8")).hexdigest()
                if self._embedding_hash is not None and candidate_embedding_hash != self._embedding_hash:
                    raise ValueError("state basis is frozen for different option embeddings")
                candidate_hash = (
                    options_hash
                    if options_hash is not None
                    else (
                        _options_hash(normalized_options)
                        if normalized_options is not None and normalized_options
                        else _options_hash(values.tolist())
                    )
                )
                if candidate_hash != self._state.options_hash:
                    raise ValueError("state basis is frozen for different answer options")
            elif options is not None or options_hash is not None:
                candidate_hash = options_hash if options_hash is not None else _options_hash(normalized_options or [])
                if candidate_hash != self._state.options_hash:
                    raise ValueError("state basis is frozen for different answer options")
            # A frozen provider may be called without repeating options (for
            # example while a retriever adds more nodes); that is an
            # intentional idempotent lookup.
            return self._state.basis.copy()
        if self.mode == "identity" and option_embeddings is None and dimension is None:
            raise ValueError("identity state basis requires dimension or embeddings")
        normalized_options = _normalize_options(options)
        if option_embeddings is None and normalized_options:
            if embed is None:
                raise ValueError("options require an embedding callable")
            option_embeddings = embed(normalized_options)
        if option_embeddings is not None:
            values = np.asarray(option_embeddings, dtype=np.float64)
            if values.ndim != 2:
                raise ValueError("option embeddings must be a two-dimensional matrix")
            # Validate the supplied matrix even for ``mode=identity``.  The
            # identity basis is a semantic fallback, not a way to hide a
            # malformed/non-finite embedding response.  In particular, NaNs
            # would otherwise pass through the identity branch and poison a
            # later cache/provenance record.
            if not np.all(np.isfinite(values)):
                raise ValueError("option embeddings must be finite")
            if normalized_options and len(normalized_options) != len(values):
                raise ValueError("options and option embeddings must have equal length")
            if len(values) == 0:
                # Do not use ``dimension or ...`` here: it silently turns an
                # explicit ``False``/``0`` into an inferred dimension and
                # hides a malformed caller value.
                if dimension is None:
                    dimension = int(values.shape[1])
            else:
                if dimension is not None and dimension != int(values.shape[1]):
                    raise ValueError("state basis dimension and option embeddings disagree")
                dimension = int(values.shape[1])
        if dimension is None:
            raise ValueError("state basis dimension must be positive")
        dimension = _strict_int(dimension, "state basis dimension", positive=True)
        fallback: str | None = None
        if self.mode == "identity" or option_embeddings is None or len(np.asarray(option_embeddings)) == 0:
            basis = np.eye(dimension, dtype=np.float64)
            # Empty option lists are the same semantic fallback as omitted
            # options: there is no contrast direction, so record the stable
            # public value ``identity`` rather than a second near-synonym.
            fallback = "identity"
            used_hash = options_hash or _options_hash(normalized_options)
        else:
            values = np.asarray(option_embeddings, dtype=np.float64)
            centered = values - values.mean(axis=0, keepdims=True)
            # Thin SVD is deterministic up to signs; signs do not change the
            # resulting PSD atoms.  We fix each sign for reproducible JSON.
            _u, singular, vh = np.linalg.svd(centered, full_matrices=False)
            scale = max(1.0, float(singular[0]) if len(singular) else 1.0)
            keep = singular > self.tolerance * scale
            if not np.any(keep):
                basis = np.eye(int(dimension), dtype=np.float64)
                fallback = "identity_zero_contrast"
            else:
                basis = vh[keep].T.copy()
                for column in range(basis.shape[1]):
                    pivot = int(np.argmax(np.abs(basis[:, column])))
                    if basis[pivot, column] < 0.0:
                        basis[:, column] *= -1.0
            used_hash = options_hash or _options_hash(normalized_options if normalized_options else values.tolist())
        self._state = StateBasis(
            basis=np.asarray(basis, dtype=np.float64),
            mode=self.mode,
            fallback=fallback,
            rank=int(basis.shape[1]),
            options_hash=used_hash,
            model_fingerprint=self.model_fingerprint,
        )
        if option_embeddings is not None:
            values = np.asarray(option_embeddings, dtype=np.float64)
            embedding_payload = json.dumps(
                values.tolist(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            self._embedding_hash = hashlib.sha256(embedding_payload.encode("utf-8")).hexdigest()
        else:
            self._embedding_hash = None
        # A query's answer-option basis is part of the frozen evaluation
        # state.  Freeze on the first successful build so repeated calls (for
        # example while A3/A4 expand the same bank) cannot silently rotate the
        # coordinate system.  ``freeze()`` remains as an explicit, backwards
        # compatible spelling for callers that want to document the barrier.
        self._frozen = True
        return self._state.basis.copy()

    def freeze(self) -> np.ndarray:
        if self._state is None:
            raise ValueError("cannot freeze an uninitialized state basis")
        self._frozen = True
        return self._state.basis.copy()

    fit = build
    provide = build
    build_basis = build

    def get_basis(self, *args: Any, **kwargs: Any) -> np.ndarray:
        if self._state is None:
            return self.build(*args, **kwargs)
        return self._state.basis.copy()

    def cache_key(
        self,
        query_hash: str = "",
        ordered_path_ids: Sequence[str] = (),
        options_hash: str | None = None,
    ) -> str:
        payload = {
            "endpoint": self.endpoint,
            "model_fingerprint": self.model_fingerprint,
            "query_hash": query_hash,
            "ordered_path_ids": list(ordered_path_ids),
            "options_hash": options_hash or (self._state.options_hash if self._state else ""),
            "mode": self.mode,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _coerce_basis(basis: np.ndarray | None, dimension: int) -> np.ndarray:
    dimension = _strict_int(dimension, "embedding dimension", positive=True)
    if basis is None:
        return np.eye(dimension, dtype=np.float64)
    value = np.asarray(basis, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != dimension:
        raise ValueError("state basis must have shape (embedding_dimension, rank)")
    if not np.all(np.isfinite(value)):
        raise ValueError("state basis must be finite")
    if value.shape[1] <= 0 or value.shape[1] > dimension:
        raise ValueError("state basis rank must be in [1, embedding_dimension]")
    gram = value.T @ value
    if not np.allclose(gram, np.eye(value.shape[1]), atol=1e-8, rtol=1e-8):
        # Re-orthogonalize rather than silently using a non-orthogonal basis.
        q, _r = np.linalg.qr(value, mode="reduced")
        value = q
    return value


def ancestor_summary(
    ancestor_vectors: Sequence[np.ndarray],
    weights: Sequence[float] | None = None,
) -> np.ndarray | None:
    if not ancestor_vectors:
        return None
    vectors = [np.asarray(vector, dtype=np.float64) for vector in ancestor_vectors]
    if any(vector.ndim != 1 or vector.size == 0 for vector in vectors):
        raise ValueError("ancestor vectors must be one-dimensional")
    if any(not np.all(np.isfinite(vector)) for vector in vectors):
        raise ValueError("ancestor vectors must be finite")
    dimension = vectors[0].shape[0]
    if any(vector.shape[0] != dimension for vector in vectors):
        raise ValueError("ancestor vectors must have equal dimensions")
    if weights is None:
        values = np.ones(len(vectors), dtype=np.float64)
    else:
        values = np.asarray(weights, dtype=np.float64).reshape(-1)
        if len(values) != len(vectors):
            raise ValueError("ancestor weights and vectors must have equal length")
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("ancestor weights must be finite and non-negative")
    combined = np.sum(np.asarray(values)[:, None] * np.vstack(vectors), axis=0)
    if np.linalg.norm(combined) <= 1e-12:
        return np.zeros_like(combined)
    return normalize(combined)


def condition_information(
    q0: np.ndarray,
    ancestor_atoms: Sequence[np.ndarray] | None = None,
) -> np.ndarray:
    """Apply the ancestor ``C_A^{-1/2} Q0 C_A^{-1/2}`` contraction."""
    q_raw = np.asarray(q0, dtype=np.float64)
    if q_raw.ndim != 2 or q_raw.shape[0] != q_raw.shape[1]:
        raise ValueError("information matrix must be square")
    q = symmetrize(q_raw)
    if not np.all(np.isfinite(q)):
        raise ValueError("information matrix must be finite")
    if not is_psd(q):
        raise ValueError("information matrix must be positive semidefinite")
    if not ancestor_atoms:
        return q
    dimension = q.shape[0]
    base = np.eye(dimension, dtype=np.float64)
    for atom in ancestor_atoms:
        item = np.asarray(atom, dtype=np.float64)
        if item.shape != q.shape:
            raise ValueError("ancestor information atoms have incompatible dimensions")
        if not np.all(np.isfinite(item)):
            raise ValueError("ancestor information atoms must be finite")
        item = symmetrize(item)
        if not is_psd(item):
            raise ValueError("ancestor information atoms must be positive semidefinite")
        base += item
    contraction = psd_inverse_sqrt(base)
    return symmetrize(contraction @ q @ contraction)


def build_information_atom(
    candidate_id: str,
    vector: np.ndarray,
    support: float,
    *,
    basis: np.ndarray | None = None,
    ancestor_vectors: Sequence[np.ndarray] = (),
    ancestor_weights: Sequence[float] | None = None,
    ancestor_atoms: Sequence[np.ndarray] = (),
    path_ids: Sequence[str] = (),
    state_information: bool = True,
) -> InformationAtom:
    """Construct one PSD atom under the I=0/I=1 definitions."""
    z = np.asarray(vector, dtype=np.float64)
    if z.ndim != 1:
        raise ValueError("candidate vector must be one-dimensional")
    if not np.all(np.isfinite(z)):
        raise ValueError("candidate vector must be finite")
    norm = np.linalg.norm(z)
    if norm <= 1e-12:
        raise ValueError("candidate vector must be non-zero")
    z = z / norm
    support_value = _strict_float(support, "support", nonnegative=True)
    if support_value > 1.0:
        raise ValueError("support must lie in [0, 1]")
    if not state_information:
        # I=0 is a strict regression of the R1 path-conditioned geometry,
        # rather than an unconditioned rho outer product whenever ancestors
        # are present.  ``path_conditioned_innovation`` implements the
        # thin-SVD (I+B_A B_A^T)^(-1/2) contraction without a dense inverse.
        weighted_ancestors = [np.asarray(item, dtype=np.float64) for item in ancestor_vectors]
        if any(item.ndim != 1 or item.shape != z.shape or not np.all(np.isfinite(item)) for item in weighted_ancestors):
            raise ValueError("ancestor vectors must be finite one-dimensional vectors matching the candidate")
        phi = factor_conditioned_innovation(
            z,
            support_value,
            ancestor_vectors=weighted_ancestors,
            ancestor_weights=ancestor_weights or (),
        )
        matrix = np.outer(phi, phi)
    else:
        b = _coerce_basis(basis, len(z))
        summary = ancestor_summary(ancestor_vectors, ancestor_weights)
        difference = z if summary is None else z - summary
        psi = b.T @ difference
        q0 = support_value**2 * np.outer(psi, psi)
        matrix = condition_information(q0, ancestor_atoms)
    matrix = symmetrize(matrix)
    # Only machine-scale negative eigenvalues may be clipped.  A materially
    # indefinite result indicates a programming/data error and is rejected.
    values, _vectors = np.linalg.eigh(matrix)
    scale = max(1.0, float(np.max(np.abs(values))) if len(values) else 1.0)
    if len(values) and float(np.min(values)) < -1e-10 * scale:
        raise ValueError("information atom is not PSD")
    matrix = matrix - min(0.0, float(np.min(values, initial=0.0))) * np.eye(matrix.shape[0]) if len(values) else matrix
    return InformationAtom(
        candidate_id=str(candidate_id),
        path_ids=tuple(str(value) for value in path_ids),
        matrix=matrix,
        trace=float(np.trace(matrix)),
        support=support_value,
    )


def aggregate_path_atoms(
    candidate_id: str,
    paths: Sequence[PathHypothesis],
    vector: np.ndarray,
    *,
    basis: np.ndarray | None = None,
    ancestor_vectors_by_id: Mapping[str, np.ndarray] | None = None,
    ancestor_atoms_by_id: Mapping[str, np.ndarray] | None = None,
    state_information: bool = True,
) -> InformationAtom:
    """Posterior-convex-combine path-conditioned atoms for one candidate."""
    if not paths:
        return build_information_atom(candidate_id, vector, 0.0, basis=basis, state_information=state_information)
    atoms: list[InformationAtom] = []
    weights = np.asarray([max(0.0, float(path.posterior)) for path in paths], dtype=np.float64)
    if weights.sum() <= 0.0:
        weights[:] = 1.0
    weights /= weights.sum()
    vectors_by_id = ancestor_vectors_by_id or {}
    atoms_by_id = ancestor_atoms_by_id or {}
    for path, probability in zip(paths, weights):
        ancestors = list(path.path_ids[:-1])
        ancestor_vectors = [vectors_by_id[item] for item in ancestors if item in vectors_by_id]
        ancestor_atoms = [atoms_by_id[item] for item in ancestors if item in atoms_by_id]
        # The formal path weight is the posterior mass of this path.  It is
        # attached to each ancestor in the path before normalization; using a
        # constant weight within a path preserves the specified posterior
        # semantics while allowing paths of different lengths to be compared.
        ancestor_weights = [float(path.posterior)] * len(ancestor_vectors)
        atom = build_information_atom(
            candidate_id,
            vector,
            path.support,
            basis=basis,
            ancestor_vectors=ancestor_vectors,
            ancestor_weights=ancestor_weights if ancestor_weights else None,
            ancestor_atoms=ancestor_atoms,
            path_ids=path.path_ids,
            state_information=state_information,
        )
        atoms.append(
            InformationAtom(
                candidate_id=atom.candidate_id,
                path_ids=atom.path_ids,
                matrix=atom.matrix * float(probability),
                trace=atom.trace * float(probability),
                support=atom.support,
            )
        )
    matrix = symmetrize(sum((atom.matrix for atom in atoms), start=np.zeros_like(atoms[0].matrix)))
    return InformationAtom(
        candidate_id=str(candidate_id),
        path_ids=tuple(paths[0].path_ids),
        matrix=matrix,
        trace=float(np.trace(matrix)),
        support=float(sum(weight * atom.support for weight, atom in zip(weights, atoms))),
    )


class InformationObjective:
    """Fixed-candidate log-det objective ``logdet(I + sum Q_j)``."""

    def __init__(
        self,
        dimension: int | None = None,
        atoms: Mapping[str, InformationAtom | np.ndarray] | None = None,
        *,
        tie_tolerance: float = 1e-12,
    ) -> None:
        if dimension is not None:
            dimension = _strict_int(dimension, "information objective dimension", positive=True)
        if dimension is None and atoms:
            first = next(iter(atoms.values()))
            first_matrix = (
                first.matrix
                if isinstance(first, (InformationAtom, SemanticAtom))
                else np.asarray(first)
            )
            first_array = np.asarray(first_matrix)
            if first_array.ndim == 1:
                if first_array.size == 0:
                    raise ValueError("information objective dimension must be positive")
                dimension = int(first_array.size)
            elif first_array.ndim == 2 and first_array.shape[0] == first_array.shape[1] and first_array.shape[0] > 0:
                dimension = int(first_array.shape[0])
            else:
                raise ValueError("information objective atoms must be vectors or square matrices")
        self.dimension = 0 if dimension is None else int(dimension)
        if self.dimension <= 0 and atoms:
            raise ValueError("information objective dimension must be positive")
        self.tie_tolerance = _strict_float(tie_tolerance, "tie_tolerance", nonnegative=True)
        self.atoms: dict[str, np.ndarray] = {}
        seen_ids: set[str] = set()
        for candidate_id, atom in (atoms or {}).items():
            identifier = str(candidate_id)
            if identifier in seen_ids:
                raise ValueError("information atom IDs must remain unique after string normalization")
            seen_ids.add(identifier)
            self.add(candidate_id, atom)

    def add(self, candidate_id: str, atom: InformationAtom | SemanticAtom | np.ndarray) -> np.ndarray:
        identifier = str(candidate_id)
        if not identifier:
            raise ValueError("information atom ID cannot be empty")
        if identifier in self.atoms:
            raise ValueError(f"information atom ID already exists: {identifier}")
        matrix = np.asarray(
            atom.matrix if isinstance(atom, (InformationAtom, SemanticAtom)) else atom,
            dtype=np.float64,
        )
        if matrix.ndim == 1:
            matrix = np.outer(matrix, matrix)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("information atom must be square")
        if self.dimension == 0:
            self.dimension = matrix.shape[0]
        if matrix.shape != (self.dimension, self.dimension):
            raise ValueError("information atom dimension mismatch")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("information atom must be finite")
        values, _vectors = psd_eigendecomposition(matrix)
        del values
        self.atoms[identifier] = symmetrize(matrix).copy()
        return self.atoms[identifier].copy()

    def _matrix(self, selected: Iterable[str | InformationAtom | np.ndarray] = ()) -> np.ndarray:
        base = np.eye(self.dimension, dtype=np.float64)
        for item in selected:
            if isinstance(item, str):
                if item not in self.atoms:
                    raise KeyError(item)
                matrix = self.atoms[item]
            elif isinstance(item, (InformationAtom, SemanticAtom)):
                matrix = item.matrix
            else:
                matrix = item
            matrix = np.asarray(matrix, dtype=np.float64)
            if matrix.ndim == 1:
                matrix = np.outer(matrix, matrix)
            if matrix.shape != base.shape:
                raise ValueError("selected atom dimension mismatch")
            if not np.all(np.isfinite(matrix)):
                raise ValueError("selected information atom must be finite")
            eigenvalues = np.linalg.eigvalsh(symmetrize(matrix))
            scale = max(1.0, float(np.max(np.abs(eigenvalues))) if len(eigenvalues) else 1.0)
            if len(eigenvalues) and float(np.min(eigenvalues)) < -1e-10 * scale:
                raise ValueError("selected information atom must be positive semidefinite")
            base += symmetrize(matrix)
        return symmetrize(base)

    def value(self, selected: Iterable[str | InformationAtom | np.ndarray] = ()) -> float:
        if self.dimension == 0:
            return 0.0
        return stable_logdet(self._matrix(selected))

    def marginal(
        self,
        candidate: str | InformationAtom | SemanticAtom | np.ndarray,
        selected: Iterable[str | InformationAtom | SemanticAtom | np.ndarray] = (),
    ) -> float:
        if isinstance(candidate, str):
            if candidate not in self.atoms:
                raise KeyError(candidate)
            matrix = self.atoms[candidate]
        elif isinstance(candidate, (InformationAtom, SemanticAtom)):
            matrix = candidate.matrix
        else:
            matrix = np.asarray(candidate, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = np.outer(matrix, matrix)
        if self.dimension == 0:
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError("candidate atom must be square")
            if matrix.shape[0] <= 0:
                raise ValueError("candidate atom dimension must be positive")
            self.dimension = int(matrix.shape[0])
        if self.dimension and matrix.shape != (self.dimension, self.dimension):
            raise ValueError("candidate atom dimension mismatch")
        # Keep the objective's PSD contract explicit even when a caller
        # supplies a raw matrix instead of an ``InformationAtom``.  Without
        # this check an indefinite contribution could make a formally
        # positive ``I + sum(Q)`` look like a valid information marginal.
        if not self.is_psd(matrix):
            raise ValueError("candidate information atom must be positive semidefinite")
        return logdet_matrix_marginal(matrix, self._matrix(selected))

    def is_psd(self, matrix: np.ndarray | None = None, tolerance: float = 1e-10) -> bool:
        tolerance = _strict_float(tolerance, "PSD tolerance", nonnegative=True)
        if matrix is None:
            if self.dimension <= 0:
                return True
            value = self._matrix() - np.eye(self.dimension)
        else:
            value = np.asarray(matrix, dtype=np.float64)
            if value.ndim == 1:
                value = np.outer(value, value)
        if value.ndim != 2 or value.shape[0] != value.shape[1] or not np.all(np.isfinite(value)):
            raise ValueError("information atom must be a finite square matrix")
        eigenvalues, _vectors = np.linalg.eigh(symmetrize(value))
        scale = max(1.0, float(np.max(np.abs(eigenvalues))) if len(eigenvalues) else 1.0)
        return bool(np.min(eigenvalues, initial=0.0) >= -tolerance * scale)

    def greedy(self, candidate_ids: Sequence[str], k: int | None = None) -> tuple[list[str], list[float]]:
        # Candidate IDs are a set semantically.  De-duplicate while retaining
        # the caller's deterministic order so a malformed proposal list
        # cannot select the same memory twice.
        remaining: list[str] = []
        seen: set[str] = set()
        for value in candidate_ids:
            identifier = str(value)
            if identifier not in self.atoms:
                raise KeyError(identifier)
            if identifier in seen:
                raise ValueError("candidate_ids must be unique")
            seen.add(identifier)
            remaining.append(identifier)
        selected: list[str] = []
        margins: list[float] = []
        target = len(remaining) if k is None else min(_strict_int(k, "k", nonnegative=True), len(remaining))
        while len(selected) < target:
            scored = [(item, float(self.marginal(item, selected))) for item in remaining]
            # Treat margins within the declared tolerance as an exact tie;
            # this prevents platform-dependent eig/Cholesky noise from
            # changing the reproducible memory-ID tie break.
            best_margin = max(value for _item, value in scored)
            tied = [item for item, value in scored if best_margin - value <= self.tie_tolerance]
            best = min(tied)
            margin = dict(scored)[best]
            selected.append(best)
            margins.append(float(margin))
            remaining.remove(best)
        return selected, margins

    def exhaustive_greedy(self, candidate_ids: Sequence[str], k: int | None = None) -> tuple[list[str], list[float]]:
        """Reference implementation over the complete supplied candidate set.

        It intentionally uses the same deterministic ``(-margin, id)`` tie
        rule as the bounded retriever and is useful for certificate audits.
        """
        return self.greedy(candidate_ids, k)

    def selected_matrix(self, selected: Iterable[str | InformationAtom | SemanticAtom | np.ndarray] = ()) -> np.ndarray:
        return self._matrix(selected)


def information_atom_from_feature(
    candidate_id: str,
    feature: np.ndarray,
    path_ids: Sequence[str] = (),
) -> InformationAtom:
    """Compatibility helper for R1's ``phi phi^T`` representation."""
    vector = np.asarray(feature, dtype=np.float64)
    matrix = np.outer(vector, vector)
    return InformationAtom(
        str(candidate_id),
        tuple(path_ids),
        matrix,
        float(np.trace(matrix)),
        float(np.linalg.norm(vector)),
    )


def exhaustive_greedy(
    atoms: Mapping[str, InformationAtom | np.ndarray],
    k: int | None = None,
    *,
    tie_tolerance: float = 1e-12,
) -> tuple[list[str], list[float]]:
    """Compute the exact fixed-candidate log-det greedy sequence."""
    objective = InformationObjective(atoms=atoms, tie_tolerance=tie_tolerance)
    return objective.greedy(list(atoms), k)


# ---------------------------------------------------------------------------
# Semantic-path v1 quality contract and fixed feature provider
# ---------------------------------------------------------------------------


class QualityProvider(Protocol):
    """Pointwise frozen scorer interface used by semantic selection."""

    def score_all(self, query: Any, records: Any) -> dict[str, QualityRecord]: ...


class RepresentationProvider(Protocol):
    def get(self, query: Any, record: Any) -> np.ndarray: ...


class FrozenFeatureProvider(Protocol):
    def quality_upper(self, memory_id: str) -> float: ...

    def materialize(self, memory_id: str) -> SemanticAtom: ...


def quality_record_from_score(
    memory_id: str,
    raw_score: float,
    *,
    score_space: str = "unit_interval",
    scorer_fingerprint: str = "",
    input_hash: str = "",
) -> QualityRecord:
    """Convert one declared scorer output without pool-dependent scaling."""

    return QualityRecord.from_raw(
        memory_id,
        raw_score,
        score_space,
        scorer_fingerprint=scorer_fingerprint,
        input_hash=input_hash,
    )


def validate_quality_records(
    records: Mapping[str, QualityRecord | float],
    expected_ids: Sequence[str],
    *,
    score_space: str = "unit_interval",
    scorer_fingerprint: str = "",
) -> dict[str, QualityRecord]:
    """Validate complete pointwise quality coverage and stable score space."""

    expected = [str(value) for value in expected_ids]
    if len(expected) != len(set(expected)):
        raise ValueError("expected quality IDs must be unique")
    normalized: dict[str, QualityRecord] = {}
    for raw_id, raw_value in records.items():
        identifier = str(raw_id)
        if identifier in normalized:
            raise ValueError("quality records contain duplicate IDs")
        if isinstance(raw_value, QualityRecord):
            record = raw_value
            if record.memory_id != identifier:
                raise ValueError("quality record key and memory_id disagree")
            normalized[identifier] = record
        else:
            normalized[identifier] = quality_record_from_score(
                identifier,
                float(raw_value),
                score_space=score_space,
                scorer_fingerprint=scorer_fingerprint,
            )
    unknown = sorted(set(normalized) - set(expected))
    missing = sorted(set(expected) - set(normalized))
    if unknown:
        raise ValueError(f"quality provider returned unknown IDs: {unknown}")
    if missing:
        raise ValueError(f"quality provider omitted IDs: {missing}")
    # Re-check the immutable contract even for records supplied by a custom
    # provider.  In particular, do not silently accept a second score space in
    # a table that was declared as unit-interval.
    declared = QualityRecord.from_raw("_contract_check_", 0.0, score_space).score_space
    for identifier, record in normalized.items():
        if record.score_space != declared:
            raise ValueError(
                f"quality record {identifier} uses {record.score_space}, expected {declared}"
            )
        if not np.isfinite(record.value) or record.value < 0.0 or record.value > 1.0:
            raise ValueError(f"quality record {identifier} is outside [0, 1]")
        if scorer_fingerprint and record.scorer_fingerprint != str(scorer_fingerprint):
            raise ValueError(
                f"quality record {identifier} uses scorer {record.scorer_fingerprint!r}, "
                f"expected {str(scorer_fingerprint)!r}"
            )
    return {identifier: normalized[identifier] for identifier in expected}


def _representation_value(
    provider: Any,
    query: Any,
    record: Any,
    memory_id: str,
) -> np.ndarray:
    if provider is None:
        value = getattr(record, "vector", None)
        if value is None and isinstance(record, Mapping):
            value = record.get("vector", record.get("embedding"))
        if value is None:
            raise ValueError(f"no representation available for {memory_id}")
    elif isinstance(provider, Mapping):
        if memory_id not in provider:
            raise KeyError(memory_id)
        value = provider[memory_id]
    elif callable(provider) and not hasattr(provider, "get"):
        value = provider(query, record)
    else:
        value = provider.get(query, record)
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError(f"representation for {memory_id} must be a finite non-empty vector")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError(f"representation for {memory_id} must be non-zero")
    return vector / norm


def expected_ancestor_scatter(
    memory_id: str,
    measure: FrozenGraphMeasure | None,
    quality: Mapping[str, QualityRecord | float],
    representations: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Compute ``D_j = sum_a w_aj r_a v_a v_a^T`` from frozen path weights."""

    candidate = str(memory_id)
    if candidate not in representations:
        raise KeyError(candidate)
    vector = np.asarray(representations[candidate], dtype=np.float64).reshape(-1)
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("candidate representation must be finite and non-empty")
    if np.linalg.norm(vector) <= 1e-12:
        raise ValueError("candidate representation must be non-zero")
    dimension = vector.size
    if measure is None:
        return np.zeros((dimension, dimension), dtype=np.float64)
    positions = {identifier: index for index, identifier in enumerate(measure.graph.memory_ids)}
    if candidate not in positions:
        raise KeyError(candidate)
    column = positions[candidate]
    scatter = np.zeros((dimension, dimension), dtype=np.float64)
    for ancestor, weight in zip(measure.graph.memory_ids, measure.ancestor_occupancy[:, column]):
        coefficient = float(weight)
        if coefficient <= 0.0:
            continue
        if ancestor not in representations:
            raise ValueError(f"missing representation for reachable ancestor {ancestor}")
        if ancestor not in quality:
            raise ValueError(f"missing quality for reachable ancestor {ancestor}")
        raw_quality = quality[ancestor]
        r = float(raw_quality.value if isinstance(raw_quality, QualityRecord) else raw_quality)
        if not np.isfinite(r) or r < 0.0 or r > 1.0:
            raise ValueError("ancestor quality must lie in [0, 1]")
        ancestor_vector = np.asarray(representations[ancestor], dtype=np.float64).reshape(-1)
        if ancestor_vector.shape != (dimension,):
            raise ValueError("representations must have equal dimensions")
        if not np.all(np.isfinite(ancestor_vector)) or np.linalg.norm(ancestor_vector) <= 1e-12:
            raise ValueError("ancestor representations must be finite and non-zero")
        ancestor_vector = ancestor_vector / np.linalg.norm(ancestor_vector)
        scatter += coefficient * r * np.outer(ancestor_vector, ancestor_vector)
    return symmetrize(scatter)


def _inverse_scatter_vector(scatter: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Apply ``(I + D)^(-1/2)`` using a thin SVD of ``D = B B^T``."""

    d = np.asarray(scatter, dtype=np.float64)
    v = np.asarray(vector, dtype=np.float64).reshape(-1)
    if d.shape != (v.size, v.size):
        raise ValueError("scatter and vector dimensions do not match")
    if not np.all(np.isfinite(d)) or not np.all(np.isfinite(v)):
        raise ValueError("scatter and vector must be finite")
    # A symmetric PSD scatter has an SVD ``U diag(s) V^T`` with ``U`` and
    # ``V`` equal up to signs.  Using the thin SVD directly keeps this helper
    # faithful to the reference implementation and avoids constructing a
    # dense inverse-square-root matrix.  Symmetrization removes only the
    # round-off antisymmetric component; materially indefinite input fails.
    symmetric = symmetrize(d)
    left, singular, right = np.linalg.svd(symmetric, full_matrices=False)
    minimum = float(np.min(np.linalg.eigvalsh(symmetric))) if symmetric.size else 0.0
    scale = max(1.0, float(np.max(np.abs(singular))) if len(singular) else 1.0)
    if minimum < -1e-10 * scale:
        raise ValueError("ancestor scatter must be positive semidefinite")
    positive = singular > 1e-14 * scale
    if not np.any(positive):
        return v.copy()
    # For a PSD symmetric matrix, the left singular vectors span the same
    # eigenspaces.  Align the right-singular signs before using the basis; the
    # sign has no effect on the resulting projector but makes the expression
    # stable for repeated singular values.
    basis = left[:, positive]
    eigen = singular[positive]
    projection = basis.T @ v
    return v + basis @ ((1.0 / np.sqrt(1.0 + eigen) - 1.0) * projection)


def semantic_feature(
    quality: float | QualityRecord,
    representation: np.ndarray,
    ancestor_scatter: np.ndarray | None = None,
    *,
    ancestor_vectors: Sequence[np.ndarray] = (),
    ancestor_weights: Sequence[float] = (),
) -> np.ndarray:
    """Return ``phi_j = sqrt(r_j) H_j v_j`` without re-normalizing its norm."""

    r = float(quality.value if isinstance(quality, QualityRecord) else quality)
    if not np.isfinite(r) or r < 0.0 or r > 1.0:
        raise ValueError("semantic quality must lie in [0, 1]")
    vector = np.asarray(representation, dtype=np.float64).reshape(-1)
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("semantic representation must be finite and non-empty")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("semantic representation must be non-zero")
    vector = vector / norm
    if ancestor_scatter is not None and ancestor_vectors:
        raise ValueError("provide ancestor factors or ancestor_scatter, not both")
    if ancestor_vectors:
        return factor_conditioned_innovation(
            vector,
            r,
            ancestor_vectors=ancestor_vectors,
            ancestor_weights=ancestor_weights,
        )
    conditioned = vector if ancestor_scatter is None else _inverse_scatter_vector(ancestor_scatter, vector)
    return np.sqrt(r) * conditioned


def build_semantic_atom(
    memory_id: str,
    graph_hash: str,
    quality: QualityRecord | float,
    representation: np.ndarray,
    *,
    ancestor_scatter: np.ndarray | None = None,
    ancestor_vectors: Sequence[np.ndarray] = (),
    ancestor_weights: Sequence[float] = (),
    path_ids: Sequence[str] = (),
    representation_hash: str = "",
) -> SemanticAtom:
    record = (
        quality
        if isinstance(quality, QualityRecord)
        else QualityRecord.from_raw(str(memory_id), float(quality), "unit_interval")
    )
    feature = semantic_feature(
        record,
        representation,
        ancestor_scatter,
        ancestor_vectors=ancestor_vectors,
        ancestor_weights=ancestor_weights,
    )
    return SemanticAtom(
        memory_id=str(memory_id),
        graph_hash=str(graph_hash),
        quality=record.value,
        feature=feature,
        path_ids=tuple(str(value) for value in path_ids),
        representation_hash=representation_hash,
    )


class SemanticFeatureProvider:
    """Lazily materialized but mathematically frozen semantic atoms."""

    def __init__(
        self,
        graph: FrozenProposalGraph,
        measure: FrozenGraphMeasure | None,
        quality: Mapping[str, QualityRecord | float],
        records: Mapping[str, Any] | Sequence[Any] | None = None,
        representation_provider: Any = None,
        *,
        query: Any = None,
        path_mode: str = "posterior_expected_scatter",
        representation_mode: str = "cached_memory",
        scorer_fingerprint: str = "",
        representation_fingerprint: str = "",
        shuffle_seed: int = 42,
        legacy_parent_id: Mapping[str, str | None] | None = None,
    ) -> None:
        self.graph = graph
        self.measure = measure
        self.query = query
        path_aliases = {
            "posterior": "posterior_expected_scatter",
            "expected_scatter": "posterior_expected_scatter",
            "map": "single_path",
            "map_path": "single_path",
            "shuffled": "shuffle",
            "shuffle_path": "shuffle",
            "flat": "none",
            "no_path": "none",
        }
        self.path_mode = path_aliases.get(str(path_mode).strip().lower(), str(path_mode).strip().lower())
        if self.path_mode not in {"posterior_expected_scatter", "single_path", "shuffle", "none", "legacy"}:
            raise ValueError(f"unsupported semantic path mode: {path_mode}")
        representation_aliases = {"memory": "cached_memory", "cached": "cached_memory", "query": "query_conditioned"}
        self.representation_mode = representation_aliases.get(
            str(representation_mode).strip().lower(), str(representation_mode).strip().lower()
        )
        if self.representation_mode not in {"cached_memory", "query_conditioned"}:
            raise ValueError(f"unsupported representation mode: {representation_mode}")
        self.scorer_fingerprint = str(scorer_fingerprint)
        self.representation_fingerprint = str(representation_fingerprint)
        self.shuffle_seed = _strict_int(shuffle_seed, "shuffle_seed", nonnegative=True)
        self.legacy_parent_id = (
            {str(child): (None if parent is None else str(parent)) for child, parent in legacy_parent_id.items()}
            if legacy_parent_id is not None
            else None
        )
        self._graph_already_shuffled = bool(
            self.path_mode == "shuffle" and isinstance(graph.proposal_config, Mapping)
            and "shuffle_seed" in graph.proposal_config
        )
        if self.legacy_parent_id is not None:
            unknown = (
                set(self.legacy_parent_id)
                | {p for p in self.legacy_parent_id.values() if p is not None}
            ) - set(graph.memory_ids)
            if unknown:
                raise ValueError(f"legacy_parent_id references IDs outside the frozen pool: {sorted(unknown)}")
        if isinstance(quality, Mapping):
            quality_input = quality
        else:
            quality_items = list(quality)
            quality_input = {
                str(item.memory_id) if isinstance(item, QualityRecord) else str(identifier): item
                for identifier, item in zip(self.graph.memory_ids, quality_items)
            }
        self.quality = validate_quality_records(
            quality_input,
            graph.memory_ids,
            score_space=(
                next(iter(quality_input.values())).score_space
                if quality_input and isinstance(next(iter(quality_input.values())), QualityRecord)
                else "unit_interval"
            ),
            scorer_fingerprint=self.scorer_fingerprint,
        )
        if records is None:
            self.records: dict[str, Any] = {identifier: None for identifier in graph.memory_ids}
        elif isinstance(records, Mapping):
            self.records = {str(key): value for key, value in records.items()}
        else:
            self.records = {str(item.memory_id): item for item in records}
        self.representation_provider = representation_provider
        self._representations: dict[str, np.ndarray] = {}
        self._atoms: dict[str, SemanticAtom] = {}
        self.materialization_count = 0
        self.ancestor_materialization_count = 0
        self.representation_hash = hashlib.sha256(
            json.dumps(
                {
                    "mode": self.representation_mode,
                    "fingerprint": self.scorer_fingerprint,
                    "representation_fingerprint": self.representation_fingerprint,
                    "query": query,
                    "graph_hash": graph.graph_hash,
                    "shuffle_seed": self.shuffle_seed if self.path_mode == "shuffle" else None,
                },
                sort_keys=True,
                ensure_ascii=False,
                default=str,
            ).encode()
        ).hexdigest()

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return self.graph.memory_ids

    def quality_value(self, memory_id: str) -> float:
        try:
            return float(self.quality[str(memory_id)].value)
        except KeyError as exc:
            raise KeyError(str(memory_id)) from exc

    def quality_upper(self, memory_id: str) -> float:
        # For a fixed rank-one atom, Delta <= log(1 + r).  This is independent
        # of path mass and remains valid for a deep high-quality leaf.
        # Inflate the floating-point value by a tiny protocol margin.  Without
        # this, an atom whose mathematically bounded trace is equal to ``r``
        # can round one ulp above ``log1p(r)`` and be incorrectly certified
        # before its exact feature is materialized.
        bound = float(np.log1p(self.quality_value(memory_id)))
        return float(bound + 1e-12 * max(1.0, abs(bound)))

    def _get_representation(self, memory_id: str) -> np.ndarray:
        identifier = str(memory_id)
        if identifier in self._representations:
            return self._representations[identifier].copy()
        record = self.records.get(identifier)
        value = _representation_value(self.representation_provider, self.query, record, identifier)
        self._representations[identifier] = value.copy()
        return value

    def _representation_digest(self, memory_id: str, vector: np.ndarray) -> str:
        payload = {
            "provider": self.representation_hash,
            "memory_id": str(memory_id),
            "vector_sha256": hashlib.sha256(np.asarray(vector, dtype=np.float64).tobytes()).hexdigest(),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def _ancestor_weights(self, candidate: str) -> tuple[tuple[str, float], ...]:
        """Return the fixed ancestry coefficients for one candidate.

        ``single_path`` follows a deterministic local-parent chain.  The
        default uses the full posterior occupancy DP.  ``shuffle`` migrates
        the frozen structure through one seeded within-layer node bijection.
        """
        if self.measure is None or self.path_mode in {"none", "flat", "no_path", "legacy"}:
            return ()
        position = {identifier: index for index, identifier in enumerate(self.graph.memory_ids)}
        if candidate not in position:
            raise KeyError(candidate)
        column = position[candidate]
        weights = [
            (identifier, float(weight))
            for identifier, weight in zip(self.graph.memory_ids, self.measure.ancestor_occupancy[:, column])
            if float(weight) > 0.0
        ]
        if self.path_mode in {"single_path", "map_path", "map"}:
            chain: list[str] = []
            current = candidate
            seen: set[str] = set()
            parents = self.legacy_parent_id or {}
            while parents.get(current) is not None and current not in seen:
                seen.add(current)
                parent = str(parents[current])
                chain.append(parent)
                current = parent
            if current in seen:
                raise ValueError("legacy_parent_id contains a cycle")
            return tuple((identifier, 1.0) for identifier in reversed(chain))
        if self.path_mode == "shuffle" and not self._graph_already_shuffled:
            permutation = self._shuffle_permutation()
            inverse = {target: source for source, target in permutation.items()}
            original_candidate = inverse[candidate]
            original_column = position[original_candidate]
            migrated = []
            for original_ancestor, weight in zip(
                self.graph.memory_ids,
                self.measure.ancestor_occupancy[:, original_column],
            ):
                coefficient = float(weight)
                if coefficient > 0.0:
                    migrated.append((permutation[original_ancestor], coefficient))
            return tuple(sorted(migrated))
        return tuple(weights)

    def _shuffle_permutation(self) -> dict[str, str]:
        """Return the fixed within-layer identity migration for this graph."""
        cached = getattr(self, "_cached_shuffle_permutation", None)
        if cached is not None:
            return dict(cached)
        layers = self.graph.layers or tuple((identifier,) for identifier in self.graph.memory_ids)
        rng = np.random.default_rng(self.shuffle_seed)
        permutation: dict[str, str] = {}
        changed = 0
        for layer in layers:
            identifiers = sorted(str(value) for value in layer)
            targets = list(identifiers)
            rng.shuffle(targets)
            permutation.update(zip(identifiers, targets))
            changed += sum(left != right for left, right in zip(identifiers, targets))
        self._cached_shuffle_permutation = dict(permutation)
        self.shuffle_effective_nodes = int(changed)
        return permutation

    def materialize(self, memory_id: str) -> SemanticAtom:
        identifier = str(memory_id)
        if identifier in self._atoms:
            return self._atoms[identifier]
        representation = self._get_representation(identifier)
        self._representations[identifier] = representation.copy()
        if self.path_mode in {"none", "flat", "no_path", "legacy"}:
            ancestor_vectors: list[np.ndarray] = []
            ancestor_weights: list[float] = []
            path_ids: tuple[str, ...] = ()
        else:
            ancestry = self._ancestor_weights(identifier)
            represented_before = set(self._representations)
            representations = self._all_representations(identifier, ancestry)
            ancestor_vectors = []
            ancestor_weights = []
            for ancestor, weight in ancestry:
                if ancestor not in representations:
                    continue
                quality = self.quality[ancestor]
                value = float(quality.value if isinstance(quality, QualityRecord) else quality)
                coefficient = float(weight) * value
                if coefficient > 0.0:
                    ancestor_vectors.append(representations[ancestor])
                    ancestor_weights.append(coefficient)
            path_ids = tuple(ancestor for ancestor, _weight in ancestry)
            # Charge only genuinely new ancestor representations.  Reusing a
            # vector for several candidate scatters is a cache hit, not a new
            # embedding operation.
            self.ancestor_materialization_count += sum(
                1
                for ancestor, _weight in ancestry
                if ancestor not in represented_before and ancestor in self._representations
            )
        atom = build_semantic_atom(
            identifier,
            self.graph.graph_hash,
            self.quality[identifier],
            representation,
            ancestor_vectors=ancestor_vectors,
            ancestor_weights=ancestor_weights,
            path_ids=path_ids,
            representation_hash=self._representation_digest(identifier, representation),
        )
        self._atoms[identifier] = atom
        self.materialization_count += 1
        return atom

    def _all_representations(
        self,
        candidate: str,
        ancestry: Sequence[tuple[str, float]] | None = None,
    ) -> dict[str, np.ndarray]:
        # Ancestor vectors are needed to evaluate D_j; materializing them is a
        # charged operation but they remain frozen once obtained.
        result = {candidate: self._get_representation(candidate)}
        if self.measure is not None:
            if ancestry is None:
                ancestry = self._ancestor_weights(candidate)
            for ancestor, weight in ancestry:
                if weight > 0.0:
                    result[ancestor] = self._get_representation(ancestor)
        return result

    def _ancestor_ids(self, candidate: str) -> tuple[str, ...]:
        return tuple(identifier for identifier, _weight in self._ancestor_weights(candidate))

    def atom(self, memory_id: str) -> SemanticAtom:
        return self.materialize(memory_id)

    def all_atoms(self) -> dict[str, SemanticAtom]:
        return {identifier: self.materialize(identifier) for identifier in self.candidate_ids}


@dataclass(frozen=True)
class SelectionResult:
    selected_ids: tuple[str, ...]
    margins: tuple[float, ...]
    diagnostics: Mapping[str, Any]

    def __iter__(self):
        yield list(self.selected_ids)
        yield list(self.margins)
        yield dict(self.diagnostics)


def _tie_better(candidate_id: str, margin: float, best_id: str | None, best_margin: float | None) -> bool:
    if best_id is None or best_margin is None:
        return True
    if margin > best_margin:
        return True
    return margin == best_margin and str(candidate_id) < str(best_id)


def _rank1_margin(feature: np.ndarray, selected_features: Sequence[np.ndarray]) -> float:
    """Exact rank-one log-det marginal using the selected small Gram matrix."""
    candidate = np.asarray(feature, dtype=np.float64).reshape(-1)
    if candidate.size == 0 or not np.all(np.isfinite(candidate)):
        raise ValueError("semantic feature must be finite and non-empty")
    norm_sq = float(candidate @ candidate)
    if not selected_features:
        residual = norm_sq
    else:
        Phi = np.column_stack([np.asarray(value, dtype=np.float64).reshape(-1) for value in selected_features])
        if Phi.shape[0] != candidate.size or not np.all(np.isfinite(Phi)):
            raise ValueError("selected semantic features have incompatible dimensions")
        gram = np.eye(Phi.shape[1], dtype=np.float64) + Phi.T @ Phi
        b = Phi.T @ candidate
        residual = norm_sq - float(b @ np.linalg.solve(gram, b))
    scale = max(1.0, norm_sq)
    if residual < -1e-10 * scale:
        raise ValueError("rank-one marginal residual is materially negative")
    if residual < 0.0:
        residual = 0.0
    return float(np.log1p(residual))


def _rank1_greedy(
    features: Mapping[str, np.ndarray],
    candidate_ids: Sequence[str],
    target: int,
) -> tuple[list[str], list[float]]:
    selected: list[str] = []
    margins: list[float] = []
    remaining = [str(value) for value in candidate_ids]
    if len(remaining) != len(set(remaining)):
        raise ValueError("fixed-pool candidate IDs must be unique")
    while len(selected) < min(target, len(remaining) + len(selected)):
        selected_features = [features[identifier] for identifier in selected]
        rows = [
            (identifier, _rank1_margin(features[identifier], selected_features))
            for identifier in remaining
        ]
        best_id, best_margin = min(rows, key=lambda item: (-item[1], item[0]))
        selected.append(best_id)
        margins.append(float(best_margin))
        remaining.remove(best_id)
    return selected, margins


class SemanticPathLogDetSelector:
    """Fixed-pool greedy selector with optional lazy quality certificates."""

    def __init__(
        self,
        *,
        certificate_mode: str = "off",
        tie_tolerance: float = 1e-12,
        max_materializations: int | None = None,
    ):
        self.certificate_mode = str(certificate_mode)
        self.tie_tolerance = _strict_float(tie_tolerance, "tie_tolerance", nonnegative=True)
        if max_materializations is not None:
            max_materializations = _strict_int(
                max_materializations, "max_materializations", nonnegative=True
            )
        self.max_materializations = max_materializations

    def select(
        self,
        features: SemanticFeatureProvider,
        k: int,
        *,
        certificate_mode: str | None = None,
    ) -> SelectionResult:
        target = min(_strict_int(k, "k", nonnegative=True), len(features.candidate_ids))
        mode = (self.certificate_mode if certificate_mode is None else str(certificate_mode)).strip().lower()
        if mode in {"off", "eager", "none"}:
            atoms = features.all_atoms()
            selected, margins = _rank1_greedy(
                {key: atom.feature for key, atom in atoms.items()},
                features.candidate_ids,
                target,
            )
            return SelectionResult(tuple(selected), tuple(float(value) for value in margins), {
                "certificate_mode": "off",
                "certified": [False] * len(selected),
                "materialized_features": features.materialization_count,
                "residual_certification_gap": [0.0] * len(selected),
            })
        if mode not in {"lazy", "certificate", "certificate_or_budget", "on"}:
            raise ValueError(f"unsupported certificate mode: {mode}")
        return self._lazy_select(features, target)

    def _lazy_select(self, features: SemanticFeatureProvider, target: int) -> SelectionResult:
        selected: list[str] = []
        margins: list[float] = []
        certified_steps: list[bool] = []
        gaps: list[float] = []
        materialized: dict[str, SemanticAtom] = {}
        candidate_ids = list(features.candidate_ids)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("fixed-pool candidate IDs must be unique")
        starting_materializations = int(features.materialization_count)
        stopped_reason: str | None = None

        def exact_margin(identifier: str) -> float:
            atom = materialized[identifier]
            if hasattr(atom, "feature"):
                selected_features = [materialized[key].feature for key in selected]
                return _rank1_margin(atom.feature, selected_features)
            objective = InformationObjective(
                atoms={key: value.matrix for key, value in materialized.items()},
                tie_tolerance=0.0,
            )
            return float(objective.marginal(identifier, selected))

        def upper_bound(identifier: str) -> float:
            value = float(features.quality_upper(identifier))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"quality upper bound for {identifier} must be finite and non-negative")
            return value

        for _step in range(target):
            # Materialize candidates until the current exact winner cannot be
            # beaten (including a possible ID tie) by any upper bound.
            while True:
                exact_best_id: str | None = None
                exact_best_margin: float | None = None
                available_materialized = [key for key in materialized if key not in selected]
                for identifier in available_materialized:
                    margin = exact_margin(identifier)
                    if _tie_better(identifier, margin, exact_best_id, exact_best_margin):
                        exact_best_id, exact_best_margin = identifier, margin
                unseen = [
                    identifier
                    for identifier in candidate_ids
                    if identifier not in materialized and identifier not in selected
                ]
                upper_rows = [(identifier, upper_bound(identifier)) for identifier in unseen]
                upper_best_id, upper_best = (
                    min(upper_rows, key=lambda item: (-item[1], item[0]))
                    if upper_rows
                    else (None, -np.inf)
                )
                needs_materialization = exact_best_id is None
                if exact_best_id is not None and upper_best_id is not None:
                    # A strict comparison is intentional: equality can change
                    # the deterministic ID tie break and must be evaluated.
                    needs_materialization = upper_best > float(exact_best_margin) or (
                        upper_best == float(exact_best_margin) and upper_best_id < exact_best_id
                    )
                if not needs_materialization:
                    break
                used_materializations = int(features.materialization_count) - starting_materializations
                if self.max_materializations is not None and used_materializations >= self.max_materializations:
                    break
                if upper_best_id is None:
                    break
                materialized[upper_best_id] = features.materialize(upper_best_id)
            exact_rows = [
                (identifier, exact_margin(identifier))
                for identifier in materialized
                if identifier not in selected
            ]
            if exact_rows:
                best_id, best_margin = min(exact_rows, key=lambda item: (-item[1], item[0]))
            else:
                # Budget exhaustion before any candidate was materialized is
                # an explicit incomplete selection.  Do not invent an exact
                # zero marginal or place an unmaterialized ID into S: doing so
                # would make the next Woodbury solve reference a missing atom.
                stopped_reason = "materialization_budget"
                break
            unseen = [
                identifier
                for identifier in candidate_ids
                if identifier not in materialized and identifier not in selected
            ]
            upper_rows = [(identifier, upper_bound(identifier)) for identifier in unseen]
            upper_best_id, upper = (
                min(upper_rows, key=lambda item: (-item[1], item[0]))
                if upper_rows
                else (None, 0.0)
            )
            # Certification uses the complete ordering key (-margin, id).
            # No numerical tolerance is allowed to turn a positive gap into a
            # tie.  If the best remaining upper exactly equals the exact
            # margin, the exact item is safe only when its ID wins that tie.
            certified = not unseen or upper < best_margin or (
                upper == best_margin and upper_best_id is not None and best_id < upper_best_id
            )
            gap = max(0.0, upper - best_margin)
            selected.append(best_id)
            margins.append(float(best_margin))
            certified_steps.append(bool(certified))
            gaps.append(float(gap if not certified else 0.0))
        return SelectionResult(tuple(selected), tuple(margins), {
            "certificate_mode": "lazy",
            "certified": certified_steps,
            "residual_certification_gap": gaps,
            "materialized_features": features.materialization_count,
            "selector_materializations": int(features.materialization_count) - starting_materializations,
            "ancestor_materializations": features.ancestor_materialization_count,
            "certificate_domain": "frozen_pool",
            "tie_rule": "(-margin, memory_id)",
            "tie_tolerance_used_for_commit": 0.0,
            "complete": len(selected) == target,
            "stop_reason": stopped_reason,
        })


class PureRerankSelector:
    """Effect-only comparison: select by the same frozen ``r_j`` values."""

    def select(self, quality: Mapping[str, QualityRecord | float], k: int) -> SelectionResult:
        rows = []
        seen: set[str] = set()
        for identifier, record in quality.items():
            identifier = str(identifier)
            if identifier in seen:
                raise ValueError("pure rerank IDs must remain unique after string normalization")
            seen.add(identifier)
            value = float(record.value if isinstance(record, QualityRecord) else record)
            if not np.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError("pure rerank quality must lie in [0, 1]")
            rows.append((identifier, value))
        rows.sort(key=lambda item: (-item[1], item[0]))
        target = _strict_int(k, "k", nonnegative=True)
        selected = tuple(identifier for identifier, _value in rows[:target])
        return SelectionResult(selected, tuple(value for _identifier, value in rows[: len(selected)]), {
            "selector": "pure_rerank",
            "certified": [False] * len(selected),
        })


class FrozenListwiseSelector:
    """One-shot listwise selector constrained to a frozen candidate whitelist."""

    def __init__(self, callback: Callable[..., Sequence[str]], *, selector_fingerprint: str = ""):
        self.callback = callback
        self.selector_fingerprint = str(selector_fingerprint)

    def select(self, query: Any, records: Mapping[str, Any], k: int) -> SelectionResult:
        target = _strict_int(k, "k", nonnegative=True)
        normalized_records: dict[str, Any] = {}
        for raw_id, value in records.items():
            identifier = str(raw_id)
            if identifier in normalized_records:
                raise ValueError("listwise records must have unique IDs after string normalization")
            normalized_records[identifier] = value
        records = normalized_records
        # Inspect the callable rather than catching ``TypeError`` from its
        # body.  A provider bug must be reported as a provider failure, not
        # retried with a different request shape (which could produce a
        # different candidate ordering or hide the original error).
        try:
            signature = inspect.signature(self.callback)
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
        if has_varargs or len(positional) >= 3:
            proposed = self.callback(query, records, target)
        elif len(positional) >= 2:
            proposed = self.callback(query, records)
        elif len(positional) == 1:
            proposed = self.callback(records)
        else:
            proposed = self.callback()
        if isinstance(proposed, Mapping):
            for key in ("selected_ids", "ids", "memory_ids", "selection"):
                if key in proposed:
                    proposed = proposed[key]
                    break
        if not isinstance(proposed, Sequence) or isinstance(proposed, (str, bytes)):
            raise ValueError("listwise selector must return a sequence of IDs")
        allowed = set(records)
        selected: list[str] = []
        for value in proposed:
            identifier = str(value)
            if identifier not in allowed:
                raise ValueError(f"listwise selector returned ID outside frozen pool: {identifier}")
            if identifier not in selected:
                selected.append(identifier)
            if len(selected) >= target:
                break
        if len(selected) < min(target, len(allowed)):
            raise ValueError("listwise selector returned fewer than k valid IDs")
        return SelectionResult(tuple(selected), (), {
            "selector": "frozen_listwise",
            "selector_fingerprint": self.selector_fingerprint,
            "candidate_domain": "frozen_pool",
        })


def lazy_greedy_fixed_pool(
    atoms: Mapping[str, SemanticAtom | InformationAtom | np.ndarray],
    qualities: Mapping[str, float | QualityRecord] | None = None,
    k: int = 1,
    *,
    max_materializations: int | None = None,
) -> SelectionResult:
    """Run the actual fixed-pool lazy selector without a model service.

    ``atoms`` represents a frozen pool whose values may be hidden behind the
    small adapter below.  The eager reference is retained only for an audit
    comparison; selection itself is performed by
    :class:`SemanticPathLogDetSelector`, so materialization counts, strict
    tie handling, and residual gaps exercise the same implementation used in
    production.
    """

    class _FixedPool:
        def __init__(self) -> None:
            self._atoms = {str(identifier): atom for identifier, atom in atoms.items()}
            if len(self._atoms) != len(atoms):
                raise ValueError("fixed-pool atom IDs must be unique after string normalization")
            self._qualities = {}
            for raw_identifier, value in (qualities or {}).items():
                identifier = str(raw_identifier)
                if identifier in self._qualities:
                    raise ValueError("fixed-pool quality IDs must be unique after string normalization")
                self._qualities[identifier] = value
            unknown = set(self._qualities) - set(self._atoms)
            if unknown:
                raise ValueError(f"qualities contain unknown atom IDs: {sorted(unknown)}")
            self._upper: dict[str, float] = {}
            self.materialization_count = 0
            self.ancestor_materialization_count = 0
            for identifier, atom in self._atoms.items():
                matrix = self._matrix(atom)
                if not InformationObjective(dimension=matrix.shape[0]).is_psd(matrix):
                    raise ValueError(f"fixed-pool atom {identifier} is not PSD")
                trace_bound = float(np.trace(matrix))
                if not np.isfinite(trace_bound) or trace_bound < -1e-10:
                    raise ValueError(f"fixed-pool atom {identifier} has an invalid trace")
                # For arbitrary-rank PSD Q, trace(Q) is the simple valid
                # bound on logdet(I + Q) requested by the compatibility
                # interface.  log1p(trace(Q)) can be smaller than the true
                # high-rank marginal and is therefore not a certificate.
                upper = max(0.0, trace_bound)
                if identifier in self._qualities:
                    raw_quality = self._qualities[identifier]
                    quality_value = float(
                        raw_quality.value if isinstance(raw_quality, QualityRecord) else raw_quality
                    )
                    if not np.isfinite(quality_value) or not 0.0 <= quality_value <= 1.0:
                        raise ValueError("fixed-pool qualities must lie in [0, 1]")
                    if trace_bound > quality_value + 1e-10:
                        raise ValueError(
                            f"quality upper bound for {identifier} is smaller than its atom trace"
                        )
                    eigenvalues = np.linalg.eigvalsh(matrix)
                    numerical_rank = int(
                        np.count_nonzero(
                            eigenvalues > 1e-12 * max(1.0, float(np.max(eigenvalues, initial=0.0)))
                        )
                    )
                    if numerical_rank <= 1:
                        # The semantic provider's rank-one atom satisfies
                        # ||phi||^2 <= r, hence Delta <= log(1+r).
                        upper = float(np.log1p(max(quality_value, trace_bound)))
                self._upper[identifier] = upper

        @staticmethod
        def _matrix(atom: SemanticAtom | InformationAtom | np.ndarray) -> np.ndarray:
            if isinstance(atom, (SemanticAtom, InformationAtom)):
                matrix = np.asarray(atom.matrix, dtype=np.float64)
            else:
                value = np.asarray(atom, dtype=np.float64)
                matrix = np.outer(value, value) if value.ndim == 1 else value
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.all(np.isfinite(matrix)):
                raise ValueError("fixed-pool atoms must be finite square matrices or vectors")
            return (matrix + matrix.T) * 0.5

        @property
        def candidate_ids(self) -> tuple[str, ...]:
            return tuple(self._atoms)

        def quality_upper(self, memory_id: str) -> float:
            try:
                return self._upper[str(memory_id)]
            except KeyError as exc:
                raise KeyError(str(memory_id)) from exc

        def materialize(self, memory_id: str):
            identifier = str(memory_id)
            if identifier not in self._atoms:
                raise KeyError(identifier)
            self.materialization_count += 1
            atom = self._atoms[identifier]
            if isinstance(atom, (SemanticAtom, InformationAtom)):
                return atom
            matrix = self._matrix(atom)
            # The selector only requires a ``matrix`` attribute.  Returning a
            # tiny immutable proxy avoids inventing a semantic quality/path
            # record for a raw mathematical test atom.
            class _MatrixAtom:
                def __init__(self, value: np.ndarray) -> None:
                    self.matrix = value.copy()
                    self.matrix.setflags(write=False)

            return _MatrixAtom(matrix)

    pool = _FixedPool()
    target = min(_strict_int(k, "k", nonnegative=True), len(pool.candidate_ids))
    selector = SemanticPathLogDetSelector(
        certificate_mode="lazy",
        tie_tolerance=0.0,
        max_materializations=max_materializations,
    )
    result = selector.select(pool, target)
    matrices = {identifier: pool._matrix(atom) for identifier, atom in pool._atoms.items()}
    eager = InformationObjective(atoms=matrices, tie_tolerance=0.0)
    eager_ids, eager_margins = eager.greedy(list(pool.candidate_ids), target)
    diagnostics = dict(result.diagnostics)
    diagnostics.update(
        {
            "eager_selected_ids": list(eager_ids),
            "eager_margins": [float(value) for value in eager_margins],
            "lazy_equals_eager": list(result.selected_ids) == eager_ids
            and np.allclose(result.margins, eager_margins, atol=1e-12, rtol=1e-12),
            "quality_upper_bounds": dict(pool._upper),
            "max_materializations": max_materializations,
        }
    )
    return SelectionResult(result.selected_ids, result.margins, diagnostics)


# Short aliases used in audit notebooks.
build_atom = build_information_atom
aggregate_atoms = aggregate_path_atoms
state_basis = StateBasisProvider
information_objective = InformationObjective


__all__ = [
    "StateBasis",
    "StateBasisProvider",
    "InformationObjective",
    "ancestor_summary",
    "condition_information",
    "build_information_atom",
    "aggregate_path_atoms",
    "information_atom_from_feature",
    "build_atom",
    "aggregate_atoms",
    "state_basis",
    "information_objective",
    "exhaustive_greedy",
    "QualityProvider",
    "RepresentationProvider",
    "FrozenFeatureProvider",
    "quality_record_from_score",
    "validate_quality_records",
    "expected_ancestor_scatter",
    "semantic_feature",
    "build_semantic_atom",
    "SemanticFeatureProvider",
    "SelectionResult",
    "SemanticPathLogDetSelector",
    "PureRerankSelector",
    "FrozenListwiseSelector",
    "lazy_greedy_fixed_pool",
    "QualityRecord",
    "SemanticAtom",
]
