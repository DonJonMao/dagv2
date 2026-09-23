"""Real-member branch measures and posterior path enumeration (M operator)."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .types import FrozenProposalGraph, PathHypothesis


@dataclass(frozen=True)
class BranchMeasure:
    """Normalized member measure and its propagated candidate support."""

    member_ids: tuple[str, ...]
    member_mass: dict[str, float]
    pi: dict[str, float]
    support: dict[str, float] = field(default_factory=dict)
    domain_ids: tuple[str, ...] = ()
    support_upper: float | None = None
    bound_ingredients: dict[str, Any] = field(default_factory=dict)
    zero_mass_uniformized: bool = False

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    @property
    def mass(self) -> dict[str, float]:
        return self.member_mass

    @property
    def sB(self) -> dict[str, float]:
        return self.support

    @property
    def support_vector(self) -> dict[str, float]:
        return self.support

    def keys(self):
        return (
            "member_ids",
            "member_mass",
            "pi",
            "support",
            "domain_ids",
            "support_upper",
            "bound_ingredients",
            "zero_mass_uniformized",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "member_ids": list(self.member_ids),
            "member_mass": dict(self.member_mass),
            "pi": dict(self.pi),
            "support": dict(self.support),
            "domain_ids": list(self.domain_ids),
            "support_upper": self.support_upper,
            "bound_ingredients": self.bound_ingredients,
            "zero_mass_uniformized": self.zero_mass_uniformized,
        }


def _ordered_values(
    member_ids: Sequence[str],
    values: Mapping[str, float] | Sequence[float] | np.ndarray | None,
) -> np.ndarray:
    if values is None:
        return np.ones(len(member_ids), dtype=np.float64)
    if isinstance(values, Mapping):
        normalized: dict[str, float] = {}
        for key, value in values.items():
            identifier = str(key)
            if identifier in normalized:
                raise ValueError("member-mass keys must remain unique after string normalization")
            normalized[identifier] = float(value)
        unknown = sorted(set(normalized) - set(member_ids))
        if unknown:
            # A misspelled member ID otherwise becomes an unobservable zero
            # mass.  Sparse maps are still supported (omitted members mean
            # zero), but keys outside the declared branch are malformed.
            raise ValueError(f"member-mass contains unknown member IDs: {unknown}")
        return np.asarray([normalized.get(identifier, 0.0) for identifier in member_ids], dtype=np.float64)
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(array) != len(member_ids):
        raise ValueError("member mass and member_ids must have equal length")
    return array


def _normalized_ids(values: Sequence[Any], *, name: str) -> list[str]:
    result = [str(value) for value in values]
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must be unique")
    return result


def _canonical_layout(value: str | None) -> str | None:
    """Normalize the public spellings for dense transition layouts.

    Keeping this in one place matters when callers provide both the historical
    ``layout`` alias and the newer ``transition_layout`` keyword: ``full`` and
    ``full_bank`` should compare as the same declaration rather than producing
    a spurious disagreement.
    """
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "full": "full_bank",
        "bank": "full_bank",
        "fullbank": "full_bank",
        "branch": "branch_local",
        "local": "branch_local",
        "branchlocal": "branch_local",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"full_bank", "branch_local"}:
        raise ValueError("transition layout must be full_bank or branch_local")
    return normalized


def _mapping_weights(
    values: Mapping[Any, float],
    member_ids: Sequence[str],
) -> np.ndarray:
    normalized: dict[str, float] = {}
    for key, value in values.items():
        identifier = str(key)
        if identifier in normalized:
            raise ValueError("pi keys must remain unique after string normalization")
        normalized[identifier] = float(value)
    unknown = sorted(set(normalized) - set(member_ids))
    if unknown:
        raise ValueError(f"pi contains unknown member IDs: {unknown}")
    return np.asarray([normalized.get(identifier, 0.0) for identifier in member_ids], dtype=np.float64)


def _validate_transition_mapping(
    transition: Mapping[Any, Any],
) -> dict[str, dict[str, float]]:
    """Normalize and validate a sparse ID-addressed transition.

    Missing entries remain valid zero-mass edges.  Present entries, however,
    are part of a probability measure and must never be silently clipped or
    dropped merely because they are negative/non-finite.
    """
    normalized: dict[str, dict[str, float]] = {}
    for raw_row, raw_values in transition.items():
        row_id = str(raw_row)
        if row_id in normalized:
            raise ValueError("transition row IDs must remain unique after string normalization")
        if not isinstance(raw_values, Mapping):
            raise ValueError("transition mapping rows must be mappings")
        row: dict[str, float] = {}
        for raw_column, raw_value in raw_values.items():
            column_id = str(raw_column)
            if column_id in row:
                raise ValueError("transition column IDs must remain unique after string normalization")
            value = float(raw_value)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError("transition must be finite and non-negative")
            row[column_id] = value
        normalized[row_id] = row
    return normalized


def _dense_member_rows(
    transition: np.ndarray,
    member_ids: Sequence[str],
    candidate_ids: Sequence[str] | None,
    layout: str | None = None,
) -> tuple[np.ndarray, list[str] | None]:
    """Resolve a dense full-bank or branch-local transition layout.

    ``candidate_ids`` labels columns.  For the ordinary square full-bank
    matrix it labels rows as well, allowing an arbitrary branch member subset
    to select the correct rows.  A rectangular matrix whose row count already
    equals ``member_ids`` is treated as branch-local and retains that order.
    Ambiguous malformed layouts are rejected instead of guessed.
    """
    matrix = np.asarray(transition, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("transition must be a two-dimensional matrix")
    if np.any(~np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError("transition must be finite and non-negative")

    members = list(member_ids)
    normalized_layout = _canonical_layout(layout) or ""
    if candidate_ids is None:
        if matrix.shape[0] != len(members):
            raise ValueError("transition rows must match member_ids when candidate_ids are absent")
        return matrix, None

    candidates = _normalized_ids(candidate_ids, name="candidate_ids")
    if len(candidates) != matrix.shape[1]:
        raise ValueError("candidate_ids and transition columns must have equal length")

    # A square matrix with one label per row/column is normally the full-bank
    # form.  Select rows by ID so a caller-provided branch order cannot change
    # the result.  ``layout=branch_local`` is an explicit escape hatch for a
    # square branch-local block whose column labels are unrelated to member
    # IDs; without it we infer that form whenever the row count matches the
    # member count but the bank labels do not contain every member.
    full_shape = matrix.shape[0] == matrix.shape[1] == len(candidates)
    all_members_labeled = all(identifier in set(candidates) for identifier in members)
    use_full_bank = normalized_layout == "full_bank" or (
        not normalized_layout and full_shape and all_members_labeled
    )
    if normalized_layout == "full_bank" and not full_shape:
        raise ValueError("full_bank transition must be square with one label per row and column")
    if use_full_bank:
        position = {identifier: index for index, identifier in enumerate(candidates)}
        missing = [identifier for identifier in members if identifier not in position]
        if missing:
            raise ValueError(f"member id {missing[0]} is missing from transition bank")
        rows = np.asarray([position[identifier] for identifier in members], dtype=np.int64)
        return matrix[rows, :], candidates

    # The only other addressable form is a branch-local rectangular matrix:
    # its rows already follow ``member_ids`` and its columns follow
    # ``candidate_ids``.
    if normalized_layout == "branch_local" and matrix.shape[0] != len(members):
        raise ValueError("branch_local transition rows must match member_ids")
    if matrix.shape[0] != len(members):
        raise ValueError("transition rows cannot be mapped to member_ids")
    return matrix, candidates


def branch_measure(
    member_ids: Sequence[str],
    member_mass: Mapping[str, float] | Sequence[float] | np.ndarray | None = None,
    transition: np.ndarray | Mapping[str, Mapping[str, float]] | None = None,
    *,
    candidate_ids: Sequence[str] | None = None,
    ids: Sequence[str] | None = None,
    transition_layout: str | None = None,
    layout: str | None = None,
    domain_ids: Sequence[str] | None = None,
    bound_ingredients: Mapping[str, Any] | None = None,
) -> BranchMeasure:
    """Create ``mu_B`` and optionally propagate it through ``P_q``.

    A zero total first-hop mass is the only case that is uniformized.  Negative
    masses are rejected rather than silently clipped, making malformed
    upstream scores visible in protocol audits.
    """
    members = tuple(str(identifier) for identifier in member_ids)
    if len(set(members)) != len(members):
        raise ValueError("branch member_ids must be unique")
    if (
        candidate_ids is not None
        and ids is not None
        and _normalized_ids(candidate_ids, name="candidate_ids") != _normalized_ids(ids, name="ids")
    ):
        raise ValueError("candidate_ids and ids disagree")
    if candidate_ids is None:
        candidate_ids = ids
    if (
        transition_layout is not None
        and layout is not None
        and _canonical_layout(transition_layout) != _canonical_layout(layout)
    ):
        raise ValueError("transition_layout and layout disagree")
    mass, pi, uniformized = normalize_member_mass(members, member_mass)
    support: dict[str, float] = {}
    if transition is not None:
        support = propagate_mass(
            pi,
            transition,
            member_ids=members,
            candidate_ids=candidate_ids,
            transition_layout=transition_layout,
            layout=layout,
        )
        if isinstance(support, np.ndarray):
            support = {str(index): float(value) for index, value in enumerate(support)}
    normalized_domain = tuple(str(value) for value in (domain_ids or ()))
    if len(set(normalized_domain)) != len(normalized_domain):
        raise ValueError("branch domain_ids must be unique")
    return BranchMeasure(
        member_ids=members,
        member_mass=mass,
        pi=pi,
        support=support,
        domain_ids=normalized_domain,
        bound_ingredients=dict(bound_ingredients or {}),
        zero_mass_uniformized=uniformized,
    )


def normalize_member_mass(
    member_ids: Sequence[str],
    member_mass: Mapping[str, float] | Sequence[float] | np.ndarray | None = None,
) -> tuple[dict[str, float], dict[str, float], bool]:
    """Return ``(raw_mass, pi, zero_mass_uniformized)`` deterministically."""
    ids = tuple(str(value) for value in member_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("branch member_ids must be unique")
    raw = _ordered_values(ids, member_mass)
    if np.any(~np.isfinite(raw)) or np.any(raw < 0.0):
        raise ValueError("member mass must be finite and non-negative")
    total = float(raw.sum())
    uniformized = total <= 0.0
    probs = np.full(len(ids), 1.0 / len(ids)) if uniformized and ids else (raw / total if total > 0 else np.empty(0))
    return (
        {identifier: float(value) for identifier, value in zip(ids, raw)},
        {identifier: float(value) for identifier, value in zip(ids, probs)},
        uniformized,
    )


def _transition_value(transition: Any, row: Any, col: Any, row_index: int, col_index: int) -> float:
    if isinstance(transition, Mapping):
        row_values = transition.get(row, transition.get(str(row), {}))
        if isinstance(row_values, Mapping):
            return float(row_values.get(col, row_values.get(str(col), 0.0)))
        return 0.0
    matrix = np.asarray(transition, dtype=np.float64)
    return float(matrix[row_index, col_index])


def propagate_mass(
    pi: BranchMeasure | Mapping[str, float] | Sequence[float] | np.ndarray,
    transition: np.ndarray | Mapping[str, Mapping[str, float]],
    *,
    member_ids: Sequence[str] | None = None,
    candidate_ids: Sequence[str] | None = None,
    ids: Sequence[str] | None = None,
    transition_layout: str | None = None,
    layout: str | None = None,
) -> dict[str, float] | np.ndarray:
    """Compute ``s_B(j)=sum_i pi_i P(i,j)`` without dropping mass."""
    if (
        candidate_ids is not None
        and ids is not None
        and _normalized_ids(candidate_ids, name="candidate_ids") != _normalized_ids(ids, name="ids")
    ):
        raise ValueError("candidate_ids and ids disagree")
    if candidate_ids is None:
        candidate_ids = ids
    if transition_layout is not None and layout is not None:
        left_layout = _canonical_layout(transition_layout)
        right_layout = _canonical_layout(layout)
        if left_layout != right_layout:
            raise ValueError("transition_layout and layout disagree")
    resolved_layout = transition_layout if transition_layout is not None else layout
    if isinstance(pi, BranchMeasure):
        member_ids = tuple(pi.member_ids)
        probabilities: Mapping[str, float] | Sequence[float] = pi.pi
    elif isinstance(pi, Mapping):
        probabilities = pi
        if member_ids is None:
            member_ids = tuple(str(key) for key in pi)
    else:
        probabilities = np.asarray(pi, dtype=np.float64)
        if member_ids is None:
            member_ids = tuple(str(index) for index in range(len(probabilities)))
    members = tuple(_normalized_ids(member_ids or (), name="member_ids"))
    if isinstance(probabilities, Mapping):
        weights = _mapping_weights(probabilities, members)
    else:
        weights = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if len(weights) != len(members):
        raise ValueError("pi and member_ids must have equal length")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("pi must be finite and non-negative")

    if isinstance(transition, Mapping):
        normalized_transition = _validate_transition_mapping(transition)
        unknown_rows = sorted(set(normalized_transition) - set(members))
        if unknown_rows:
            raise ValueError(f"transition contains unknown member IDs: {unknown_rows}")
        if candidate_ids is None:
            candidates = sorted(
                {
                    str(candidate)
                    for row in normalized_transition.values()
                    for candidate in row
                }
            )
        else:
            candidates = _normalized_ids(candidate_ids, name="candidate_ids")
            declared = {
                str(candidate)
                for row in normalized_transition.values()
                for candidate in row
            }
            unknown = sorted(declared - set(candidates))
            if unknown:
                raise ValueError(f"transition contains unknown candidate IDs: {unknown}")
        output = {
            candidate: float(
                sum(
                    weights[row_index] * normalized_transition.get(member, {}).get(candidate, 0.0)
                    for row_index, member in enumerate(members)
                )
            )
            for candidate in candidates
        }
        return output

    selected_matrix, candidates = _dense_member_rows(transition, members, candidate_ids, resolved_layout)
    result = weights @ selected_matrix
    if candidates is None:
        return result
    return {candidate: float(value) for candidate, value in zip(candidates, result)}


def parent_posterior(
    candidate: str | int,
    member_ids: Sequence[str],
    pi: BranchMeasure | Mapping[str, float] | Sequence[float] | np.ndarray,
    transition: np.ndarray | Mapping[str, Mapping[str, float]],
    *,
    candidate_ids: Sequence[str] | None = None,
    ids: Sequence[str] | None = None,
    transition_layout: str | None = None,
    layout: str | None = None,
) -> dict[str, float]:
    """Return every positive-mass ``p(i|j,B,q)`` parent posterior."""
    if (
        candidate_ids is not None
        and ids is not None
        and _normalized_ids(candidate_ids, name="candidate_ids") != _normalized_ids(ids, name="ids")
    ):
        raise ValueError("candidate_ids and ids disagree")
    if candidate_ids is None:
        candidate_ids = ids
    if transition_layout is not None and layout is not None:
        left_layout = _canonical_layout(transition_layout)
        right_layout = _canonical_layout(layout)
        if left_layout != right_layout:
            raise ValueError("transition_layout and layout disagree")
    resolved_layout = transition_layout if transition_layout is not None else layout
    members = tuple(_normalized_ids(member_ids, name="member_ids"))
    if isinstance(pi, BranchMeasure):
        weights = np.asarray([pi.pi.get(identifier, 0.0) for identifier in members], dtype=np.float64)
    elif isinstance(pi, Mapping):
        weights = _mapping_weights(pi, members)
    else:
        weights = np.asarray(pi, dtype=np.float64).reshape(-1)
    if len(weights) != len(members):
        raise ValueError("pi and member_ids must have equal length")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("pi must be finite and non-negative")

    normalized_transition = (
        _validate_transition_mapping(transition) if isinstance(transition, Mapping) else None
    )
    if normalized_transition is not None:
        unknown_rows = sorted(set(normalized_transition) - set(members))
        if unknown_rows:
            raise ValueError(f"transition contains unknown member IDs: {unknown_rows}")
    candidate_labels = (
        _normalized_ids(candidate_ids, name="candidate_ids")
        if candidate_ids is not None
        else None
    )
    if normalized_transition is not None and candidate_labels is not None:
        declared = {
            str(column_id)
            for row in normalized_transition.values()
            for column_id in row
        }
        unknown = sorted(declared - set(candidate_labels))
        if unknown:
            raise ValueError(f"transition contains unknown candidate IDs: {unknown}")
    candidate_key = str(candidate)
    column = -1
    if candidate_labels is not None:
        # Explicit labels are authoritative, including when those labels are
        # integers.  Treating candidate ``20`` as positional column 20 before
        # checking ``candidate_ids=[10, 20, 30]`` silently erased a perfectly
        # valid posterior.  Positional addressing remains a compatibility
        # fallback only when no label matches.
        try:
            column = candidate_labels.index(candidate_key)
        except ValueError:
            if candidate_key.isdigit():
                column = int(candidate_key)
                if column < 0 or column >= len(candidate_labels):
                    return {}
                candidate_key = candidate_labels[column]
            else:
                return {}
    elif normalized_transition is None:
        # A square dense matrix can use ``member_ids`` as its implicit column
        # universe.  Resolve by ID first so non-zero-based integer IDs work;
        # otherwise retain the historical numeric positional spelling.
        matrix_columns = np.asarray(transition).shape[1]
        if matrix_columns == len(members) and candidate_key in members:
            column = members.index(candidate_key)
        elif candidate_key.isdigit():
            column = int(candidate_key)
        else:
            return {}

    if normalized_transition is not None:
        values = np.asarray(
            [
                weights[row_index]
                * normalized_transition.get(member, {}).get(candidate_key, 0.0)
                for row_index, member in enumerate(members)
            ],
            dtype=np.float64,
        )
    else:
        selected_matrix, resolved_labels = _dense_member_rows(
            np.asarray(transition, dtype=np.float64),
            members,
            candidate_labels,
            resolved_layout,
        )
        if resolved_labels is not None:
            try:
                column = resolved_labels.index(candidate_key)
            except ValueError:
                return {}
        if column < 0 or column >= selected_matrix.shape[1]:
            return {}
        values = weights * selected_matrix[:, column]

    values = np.where(values > 0.0, values, 0.0)
    total = float(values.sum())
    if total <= 0.0:
        return {}
    return {
        member: float(value / total)
        for member, value in zip(members, values)
        if value > 0.0
    }


def _coerce_parent_map(value: Any) -> dict[str, float]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        result: dict[str, float] = {}
        for key, probability in value.items():
            identifier = str(key)
            if identifier in result:
                raise ValueError("parent IDs must remain unique after string normalization")
            weight = float(probability)
            if not np.isfinite(weight) or weight < 0.0:
                raise ValueError("parent posterior must be finite and non-negative")
            if weight > 0.0:
                result[identifier] = weight
        return result
    raise ValueError("parent posterior must be a mapping")


_UNSET_PARENT_POSTERIORS = object()


def enumerate_path_hypotheses(
    candidate_id: str,
    branch_id: str = "",
    parent_posteriors: Mapping[str, float] | None | object = _UNSET_PARENT_POSTERIORS,
    *,
    node_parent_posteriors: Mapping[str, Mapping[str, float]] | None = None,
    root_posterior: Mapping[str, float] | None = None,
    supports: Mapping[str, float] | None = None,
    max_depth: int = 8,
    support: float | None = None,
    transition: np.ndarray | Mapping[str, Mapping[str, float]] | None = None,
    ids: Sequence[str] | None = None,
    root_masses: Mapping[str, float] | None = None,
    **kwargs: Any,
) -> tuple[PathHypothesis, ...]:
    """Enumerate all positive posterior paths ending at ``candidate_id``.

    ``node_parent_posteriors`` maps each real node to all of its positive
    parent probabilities.  A missing entry means that node is a root.  The
    function is intentionally pure and deterministic; callers may pass a MAP
    parent separately for display without affecting this enumeration.
    """
    # Accept both natural positional spellings used by early notebooks:
    # ``(candidate, branch_id, parents)`` (the original implementation) and
    # ``(candidate, parents, branch_id)`` (the mathematical notation).  A
    # mapping is unambiguously a parent map, while a string is a branch id.
    parent_argument_supplied = parent_posteriors is not _UNSET_PARENT_POSTERIORS
    parent_none_explicit = parent_argument_supplied and parent_posteriors is None
    if parent_posteriors is _UNSET_PARENT_POSTERIORS:
        parent_posteriors = None
    if isinstance(branch_id, Mapping):
        if parent_posteriors is None:
            parent_posteriors, branch_id = branch_id, ""
            parent_argument_supplied = True
        elif isinstance(parent_posteriors, str):
            parent_posteriors, branch_id = branch_id, parent_posteriors
            parent_argument_supplied = True
    # Accept common spelling variants used by early experiment notebooks.
    if parent_posteriors is None:
        if "parent_posterior" in kwargs:
            parent_posteriors = kwargs.pop("parent_posterior")
            parent_argument_supplied = True
            parent_none_explicit = False
        elif "parents" in kwargs:
            parent_posteriors = kwargs.pop("parents")
            parent_argument_supplied = True
            parent_none_explicit = False
        else:
            parent_posteriors = {}

    # Early callers used ``parent_posteriors`` for both the terminal
    # parent->probability map and the newer node->(parent->probability) map.
    # Detect the latter structurally so nested maps do not reach
    # ``float(dict)`` below.  The explicit ``node_parent_posteriors`` keyword
    # remains authoritative when both spellings are supplied.
    nested_parent_map: Mapping[Any, Any] = {}
    flat_parent_map_supplied = bool(
        parent_argument_supplied
        and not parent_none_explicit
        and isinstance(parent_posteriors, Mapping)
    )
    if isinstance(parent_posteriors, Mapping) and parent_posteriors:
        nested_values = list(parent_posteriors.values())
        # ``None`` is the compact spelling for an empty/root parent map.  A
        # map containing only ``None`` values is therefore still a nested
        # node map; treating it as a flat probability map would eventually
        # attempt ``float(None)`` and obscure the useful structural error.
        if any(isinstance(value, Mapping) or value is None for value in nested_values):
            if not all(isinstance(value, Mapping) or value is None for value in nested_values):
                raise ValueError("parent_posteriors must be either a flat map or a nested node map")
            nested_parent_map = parent_posteriors
            parent_posteriors = None
            flat_parent_map_supplied = False
    supplied_node_map = node_parent_posteriors
    if supplied_node_map is None:
        supplied_node_map = kwargs.pop("parent_posteriors_by_node", None)
    supplied_node_map = supplied_node_map or {}
    if not isinstance(supplied_node_map, Mapping):
        raise ValueError("node parent posteriors must be a mapping")
    node_map: dict[str, dict[str, float]] = {}
    for node, values in nested_parent_map.items():
        normalized_node = str(node)
        if normalized_node in node_map:
            raise ValueError("node IDs must remain unique after string normalization")
        node_map[normalized_node] = _coerce_parent_map(values)
    explicit_nodes: dict[str, dict[str, float]] = {}
    for node, values in supplied_node_map.items():
        normalized_node = str(node)
        if normalized_node in explicit_nodes:
            raise ValueError("node IDs must remain unique after string normalization")
        explicit_nodes[normalized_node] = _coerce_parent_map(values)
    # The explicit keyword is an intentional override of the legacy nested
    # spelling, while aliases within either individual mapping are rejected.
    node_map.update(explicit_nodes)
    # Keep provenance about whether the terminal ancestry was explicitly
    # supplied.  An explicit empty/all-zero terminal map means "no supported
    # ancestry"; it must not be converted into a synthetic singleton root.
    # A genuinely omitted map (the normal one-node/root spelling) remains a
    # valid implicit root.
    terminal = str(candidate_id)
    terminal_parent_supplied = flat_parent_map_supplied or any(
        str(node) == terminal for node in (*nested_parent_map.keys(), *supplied_node_map.keys())
    )
    raw_roots = root_masses if root_masses is not None else root_posterior
    roots_declared = raw_roots is not None
    if roots_declared and not isinstance(raw_roots, Mapping):
        raise ValueError("root posterior must be a mapping")
    roots: dict[str, float] = {}
    for key, value in (raw_roots or {}).items():
        identifier = str(key)
        if identifier in roots:
            raise ValueError("root IDs must remain unique after string normalization")
        root_weight = float(value)
        if not np.isfinite(root_weight) or root_weight < 0.0:
            raise ValueError("root posterior must be finite and non-negative")
        if root_weight > 0.0:
            roots[identifier] = root_weight
    if roots_declared and raw_roots:
        root_total = float(sum(roots.values()))
        if root_total > 0.0:
            roots = {key: value / root_total for key, value in roots.items()}
        else:
            # Match the explicit all-zero mass convention used for branches:
            # a declared root set with no mass receives a deterministic
            # uniform posterior rather than silently becoming an implicit
            # unrestricted root set.
            declared_ids = [str(key) for key in raw_roots]
            roots = {key: 1.0 / len(declared_ids) for key in declared_ids}
    # ``roots_declared`` may later be set by transition-graph inference.  Keep
    # this separate so an inferred root cannot override an explicit terminal
    # zero-mass ancestry declaration.
    roots_authoritative = bool(raw_roots is not None)
    normalized_supports: dict[str, float] = {}
    for key, value in (supports or {}).items():
        identifier = str(key)
        support_value = float(value)
        if not np.isfinite(support_value) or support_value < 0.0:
            raise ValueError("path supports must be finite and non-negative")
        if identifier in normalized_supports:
            raise ValueError("support IDs must remain unique after string normalization")
        normalized_supports[identifier] = support_value
    supports = normalized_supports
    try:
        numeric_depth = int(max_depth)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_depth must be a positive integer") from exc
    # Do not silently turn 2.5 (or a boolean) into a different search limit.
    # Integer-like NumPy scalars and numeric strings remain accepted for API
    # compatibility, but a fractional value is malformed provenance.
    if isinstance(max_depth, bool):
        raise ValueError("max_depth must be a positive integer")
    if str(max_depth).strip() != str(numeric_depth):
        try:
            if float(max_depth) != float(numeric_depth):
                raise ValueError("max_depth must be a positive integer")
        except (TypeError, ValueError) as exc:
            raise ValueError("max_depth must be a positive integer") from exc
    max_depth = numeric_depth
    if max_depth <= 0:
        raise ValueError("max_depth must be a positive integer")
    direct = _coerce_parent_map(parent_posteriors)
    if not direct:
        direct = node_map.get(terminal, {})

    normalized_transition: Mapping[str, Mapping[str, float]] | None = None
    if transition is not None:
        if isinstance(transition, Mapping):
            normalized_transition = _validate_transition_mapping(transition)
            labels = [
                str(value)
                for value in (
                    ids
                    if ids is not None
                    else sorted(
                        {
                            *normalized_transition.keys(),
                            *(
                                key
                                for row in normalized_transition.values()
                                for key in row
                            ),
                        }
                    )
                )
            ]
            matrix_value = None
        else:
            matrix_value = np.asarray(transition, dtype=np.float64)
            if matrix_value.ndim != 2 or matrix_value.shape[0] != matrix_value.shape[1]:
                raise ValueError("path transition must be a square matrix")
            if np.any(~np.isfinite(matrix_value)) or np.any(matrix_value < 0.0):
                raise ValueError("path transition must be finite and non-negative")
            labels = [str(value) for value in (ids or range(matrix_value.shape[0]))]
            if len(labels) != matrix_value.shape[0]:
                raise ValueError("path transition ids and matrix must have equal length")
        if len(set(labels)) != len(labels):
            raise ValueError("path transition ids must be unique")
    else:
        labels = sorted({terminal, *node_map, *(parent for values in node_map.values() for parent in values)})
        matrix_value = None
    positions = {label: index for index, label in enumerate(labels)}

    def transition_edge(parent: str, child: str) -> float:
        if transition is None:
            return 0.0
        if normalized_transition is not None:
            value = float(normalized_transition.get(parent, {}).get(child, 0.0))
        else:
            if parent not in positions or child not in positions:
                return 0.0
            value = float(matrix_value[positions[parent], positions[child]])
        return value if np.isfinite(value) and value > 0.0 else 0.0

    # Convenience form: when callers provide only a transition graph, infer
    # parent candidates from *all* positive incoming edges.  Earlier code
    # imposed lexicographic ID order, which silently dropped valid paths and
    # made the result depend on arbitrary labels.  Explicit roots (when
    # supplied) are authoritative; otherwise we infer roots as nodes with no
    # positive incoming non-self edge.  A graph whose positive edges form
    # only cycles has no well-defined root path and therefore yields no
    # hypothesis instead of manufacturing one.
    if transition is not None and not node_map:
        incoming: dict[str, dict[str, float]] = {label: {} for label in labels}
        for parent in labels:
            for child in labels:
                if parent == child:
                    continue
                edge = transition_edge(parent, child)
                if edge > 0.0:
                    incoming.setdefault(child, {})[parent] = edge
        roots_authoritative = roots_declared
        if not roots_declared:
            inferred_roots = [label for label in labels if not incoming.get(label)]
            if inferred_roots:
                roots = {label: 1.0 / len(inferred_roots) for label in inferred_roots}
                roots_declared = True
                # Inferred roots are useful for a transition-only graph, but
                # they are not an authority capable of overriding an
                # explicitly supplied terminal zero-mass map.
                if terminal_parent_supplied:
                    roots.pop(terminal, None)
        else:
            roots_authoritative = True
        root_set = set(roots)
        generated_map = {
            child: ({} if child in root_set else dict(incoming.get(child, {})))
            for child in labels
        }
        # A caller-supplied terminal parent map is more specific than the
        # graph-derived incoming set (for example it may encode a branch
        # posterior), but the generated entries for its ancestors are still
        # needed to reach an explicit root.
        if direct or terminal_parent_supplied:
            generated_map[terminal] = direct
        node_map = generated_map
        direct = node_map.get(terminal, {})

    paths: list[tuple[tuple[str, ...], float, float]] = []

    def walk(
        node: str,
        suffix: tuple[str, ...],
        probability: float,
        path_support: float,
        depth: int,
    ) -> None:
        # An explicit root distribution is authoritative even if a malformed
        # caller also supplied parent entries for that root.
        if roots_declared and node in roots and not (
            node == terminal and terminal_parent_supplied and not roots_authoritative
        ):
            paths.append((suffix, probability * roots[node], path_support))
            return
        parents = direct if node == terminal and direct else node_map.get(node, {})
        if not parents:
            # With an explicit root distribution, a path is valid only when
            # it reaches one of those roots.  Without one, every parentless
            # node is a root with neutral mass one.
            if node == terminal and terminal_parent_supplied and not roots_authoritative:
                return
            root_factor = roots.get(node, 0.0 if roots_declared else 1.0)
            if root_factor > 0.0:
                paths.append((suffix, probability * root_factor, path_support))
            return
        if depth + 1 >= max_depth:
            # The depth cap is a search limit, not permission to turn a node
            # which still has parents into an artificial root.  Parentless
            # nodes were handled above, so a capped branch contributes no
            # complete path.
            return
        advanced = False
        for parent, conditional_probability in sorted(parents.items()):
            if (
                not np.isfinite(float(conditional_probability))
                or conditional_probability <= 0.0
                or parent in suffix
            ):
                continue
            formal_edge = transition_edge(parent, node)
            edge_mass = formal_edge if transition is not None else float(conditional_probability)
            if edge_mass <= 0.0:
                continue
            advanced = True
            walk(
                parent,
                (parent,) + suffix,
                probability * edge_mass,
                path_support * edge_mass,
                depth + 1,
            )
        if not advanced:
            # All declared parents were cyclic/zero/invalid.  Only an
            # explicitly declared root can terminate such a branch; in the
            # implicit-root form, a node with a non-empty parent map is not a
            # root merely because traversal could not advance.
            root_factor = roots.get(node, 0.0)
            if root_factor > 0.0:
                paths.append((suffix, probability * root_factor, path_support))

    walk(terminal, (terminal,), 1.0, 1.0, 0)
    if not paths:
        # A genuinely parentless terminal is a one-node path.  If declared
        # ancestry was truncated or cyclic, return no hypotheses rather than
        # manufacturing a root and a normalized posterior for it.
        if (
            not terminal_parent_supplied
            and not direct
            and not node_map.get(terminal)
            and (not roots_declared or terminal in roots)
        ):
            root_factor = roots.get(terminal, 1.0)
            if root_factor > 0.0:
                paths = [((terminal,), root_factor, 1.0)]
        if not paths:
            return ()
    total = sum(max(0.0, probability) for _path, probability, _support in paths)
    # Conditional path posteriors must always form a probability distribution
    # for a discovered candidate.  If every unnormalised path mass is zero,
    # use the explicit all-zero uniformization rule (rather than returning a
    # vector of zeros, which is neither a posterior nor auditable).
    uniformized = total <= 0.0
    if uniformized:
        total = float(len(paths))
    result = []
    terminal_parent_map = {
        str(key): float(value)
        for key, value in sorted(direct.items())
        if np.isfinite(float(value)) and float(value) > 0.0
    }
    parent_total = float(sum(terminal_parent_map.values()))
    if parent_total > 0.0:
        terminal_parent_map = {
            key: value / parent_total for key, value in terminal_parent_map.items()
        }
    for path, probability, formal_support in paths:
        posterior = (1.0 / total) if uniformized else max(0.0, probability) / total
        path_support = float(support if support is not None else supports.get(terminal, formal_support))
        result.append(
            PathHypothesis(
                path_ids=tuple(path),
                parent_posterior=terminal_parent_map,
                branch_id=str(branch_id),
                posterior=float(posterior),
                support=path_support,
            )
        )
    result.sort(key=lambda item: (-item.posterior, item.path_ids, item.branch_id))
    return tuple(result)


def merge_path_hypotheses(*groups: Iterable[PathHypothesis]) -> tuple[PathHypothesis, ...]:
    """Merge duplicate paths while preserving posterior mass."""
    merged: dict[tuple[str, ...], PathHypothesis] = {}
    for group in groups:
        for item in group:
            previous = merged.get(item.path_ids)
            if previous is None:
                merged[item.path_ids] = item
            else:
                # Parent maps are conditional distributions.  A plain dict
                # update can leave a merged map summing to >1 (and made the
                # serialized path fail an otherwise valid audit).  Combine
                # them in proportion to the two path masses instead.
                left_weight = max(0.0, float(previous.posterior))
                right_weight = max(0.0, float(item.posterior))
                weight_total = left_weight + right_weight
                if weight_total <= 0.0:
                    weight_total = 1.0
                parent_keys = set(previous.parent_posterior) | set(item.parent_posterior)
                parent_map = {
                    key: (
                        left_weight * float(previous.parent_posterior.get(key, 0.0))
                        + right_weight * float(item.parent_posterior.get(key, 0.0))
                    )
                    / weight_total
                    for key in parent_keys
                }
                parent_norm = sum(value for value in parent_map.values() if value > 0.0)
                if parent_norm > 0.0:
                    parent_map = {key: value / parent_norm for key, value in parent_map.items() if value > 0.0}
                elif parent_keys:
                    # A duplicate path can occasionally arrive with two
                    # explicitly zero posterior records (for example after a
                    # conservative temporal filter).  Keep the merged path
                    # auditable by applying the same deterministic
                    # all-zero uniformization rule used elsewhere, instead of
                    # serializing a non-empty map whose values sum to zero.
                    uniform = 1.0 / len(parent_keys)
                    parent_map = {key: uniform for key in sorted(parent_keys)}
                merged[item.path_ids] = PathHypothesis(
                    path_ids=item.path_ids,
                    parent_posterior=parent_map,
                    branch_id=previous.branch_id if previous.branch_id == item.branch_id else ",".join(
                        sorted({previous.branch_id, item.branch_id} - {""})
                    ),
                    posterior=previous.posterior + item.posterior,
                    support=max(previous.support, item.support),
                )
    values = list(merged.values())
    total = sum(item.posterior for item in values)
    if total > 0.0:
        values = [
            PathHypothesis(item.path_ids, item.parent_posterior, item.branch_id, item.posterior / total, item.support)
            for item in values
        ]
    elif values:
        uniform = 1.0 / len(values)
        values = [
            PathHypothesis(item.path_ids, item.parent_posterior, item.branch_id, uniform, item.support)
            for item in values
        ]
    return tuple(sorted(values, key=lambda item: (-item.posterior, item.path_ids, item.branch_id)))


def path_posterior(
    path_ids: Sequence[str],
    transition: np.ndarray | Mapping[str, Mapping[str, float]],
    *,
    root_mass: float = 1.0,
    ids: Sequence[str] | None = None,
) -> float:
    """Evaluate the unnormalized Markov path mass ``pi_root Π P(u,v)``."""
    path = tuple(str(value) for value in path_ids)
    if not path:
        return 0.0
    if len(set(path)) != len(path):
        raise ValueError("path_ids must be unique (cycles are not valid hypotheses)")
    mass = float(root_mass)
    if not np.isfinite(mass) or mass < 0.0:
        raise ValueError("root_mass must be finite and non-negative")
    if len(path) == 1:
        return mass
    if isinstance(transition, Mapping):
        normalized_transition = _validate_transition_mapping(transition)
        for left, right in zip(path[:-1], path[1:]):
            mass *= normalized_transition.get(left, {}).get(right, 0.0)
    else:
        matrix = np.asarray(transition, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("transition must be a square matrix")
        if np.any(~np.isfinite(matrix)) or np.any(matrix < 0.0):
            raise ValueError("transition must be finite and non-negative")
        labels = [str(value) for value in (ids if ids is not None else range(matrix.shape[0]))]
        if len(labels) != matrix.shape[0] or len(set(labels)) != len(labels):
            raise ValueError("transition ids and matrix must have equal, unique length")
        positions = {label: index for index, label in enumerate(labels)}
        for left, right in zip(path[:-1], path[1:]):
            if left not in positions or right not in positions:
                return 0.0
            mass *= float(matrix[positions[left], positions[right]])
    return float(mass)


def normalize_path_hypotheses(paths: Iterable[PathHypothesis]) -> tuple[PathHypothesis, ...]:
    """Normalize a finite collection of path hypotheses without changing IDs.

    Zero/negative posterior values are discarded.  If all supplied values are
    zero, the surviving paths receive a deterministic uniform posterior; this
    mirrors the only uniformization rule used for branch mass and makes the
    normalization explicit in provenance.
    """
    values = [item for item in paths if isinstance(item, PathHypothesis)]
    if not values:
        return ()
    positive = [item for item in values if np.isfinite(item.posterior) and item.posterior > 0.0]
    if not positive:
        weight = 1.0 / len(values)
        return tuple(
            PathHypothesis(item.path_ids, item.parent_posterior, item.branch_id, weight, item.support)
            for item in sorted(values, key=lambda item: (item.path_ids, item.branch_id))
        )
    total = sum(float(item.posterior) for item in positive)
    return tuple(
        PathHypothesis(
            item.path_ids,
            item.parent_posterior,
            item.branch_id,
            float(item.posterior) / total,
            item.support,
        )
        for item in sorted(positive, key=lambda item: (-item.posterior, item.path_ids, item.branch_id))
    )


# ---------------------------------------------------------------------------
# Semantic-path v1: sparse frozen-DAG propagation
# ---------------------------------------------------------------------------


def angular_navigation_affinity(left: np.ndarray, right: np.ndarray) -> float:
    """Return ``1 - acos(cosine) / pi`` for one observed proposal edge.

    This is a navigation affinity only.  It is intentionally not clipped to
    zero and is never evaluated for unobserved pairs by the semantic graph
    builder.  Zero vectors are malformed embeddings and fail loudly.
    """

    lhs = np.asarray(left, dtype=np.float64).reshape(-1)
    rhs = np.asarray(right, dtype=np.float64).reshape(-1)
    if lhs.ndim != 1 or rhs.ndim != 1 or lhs.shape != rhs.shape:
        raise ValueError("angular affinity vectors must be one-dimensional with equal dimensions")
    if not np.all(np.isfinite(lhs)) or not np.all(np.isfinite(rhs)):
        raise ValueError("angular affinity vectors must be finite")
    lhs_norm = float(np.linalg.norm(lhs))
    rhs_norm = float(np.linalg.norm(rhs))
    if lhs_norm <= 1e-12 or rhs_norm <= 1e-12:
        raise ValueError("angular affinity requires non-zero vectors")
    cosine = float(np.dot(lhs, rhs) / (lhs_norm * rhs_norm))
    return float(1.0 - np.arccos(np.clip(cosine, -1.0, 1.0)) / np.pi)


def observed_transition(
    graph: FrozenProposalGraph,
    *,
    edge_weights: Mapping[tuple[str, str], float] | None = None,
) -> dict[str, dict[str, float]]:
    """Build sparse row-normalized ``P`` on the graph's observed edges.

    No ``N x N`` transition matrix is constructed.  A row with no positive
    observed weight is represented by an empty mapping and therefore goes to
    the explicit termination state in propagation; it is never connected
    uniformly to the rest of the memory bank.
    """

    ids = set(graph.memory_ids)
    declared = graph.edge_weight_map
    if edge_weights is not None:
        overrides = {(str(left), str(right)): float(value) for (left, right), value in edge_weights.items()}
        unknown = set(overrides) - set(graph.edges)
        if unknown:
            raise ValueError(f"observed edge weights reference undeclared edges: {sorted(unknown)}")
        declared.update(overrides)
    rows: dict[str, dict[str, float]] = {identifier: {} for identifier in graph.memory_ids}
    for left, right in graph.edges:
        value = float(declared.get((left, right), 0.0))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("observed edge weights must be finite and non-negative")
        if left not in ids or right not in ids or left == right:
            raise ValueError("observed transition edge references an invalid node")
        if value > 0.0:
            rows[left][right] = value
    for left, values in rows.items():
        total = float(sum(values.values()))
        if total > 0.0:
            rows[left] = {right: value / total for right, value in sorted(values.items())}
        else:
            rows[left] = {}
    return rows


def _effective_root_ids(
    graph: FrozenProposalGraph,
    transition: Mapping[str, Mapping[str, float]],
) -> tuple[str, ...]:
    """Return the source IDs of the finite forward path measure.

    ``FrozenProposalGraph.layers`` describes the synchronous discovery
    protocol.  If it is present, its first layer is the only declared root
    distribution; a node in a later layer with no positive incoming edge is
    therefore an unreachable candidate rather than an implicit root.  This
    distinction is important when a proposal edge has zero angular mass.  For
    hand-built graphs without layers we retain the usual structural-root
    convention (all nodes with no positive incoming edge).
    """

    ids = tuple(str(value) for value in graph.memory_ids)
    incoming: dict[str, bool] = {identifier: False for identifier in ids}
    for left, values in transition.items():
        if left not in incoming or not isinstance(values, Mapping):
            continue
        for right, weight in values.items():
            if right in incoming and float(weight) > 0.0:
                incoming[right] = True
    if graph.layers:
        first = tuple(identifier for identifier in graph.layers[0] if identifier in incoming)
        if first:
            return first
    return tuple(identifier for identifier in ids if not incoming[identifier])


@dataclass(frozen=True)
class FrozenGraphMeasure:
    """Global path quantities for a frozen proposal DAG.

    Mappings are read-only views and the ancestor matrix is a read-only copy;
    this prevents selection code from changing the posterior after freeze.
    """

    graph: FrozenProposalGraph
    transition: Mapping[str, Mapping[str, float]]
    access_quality: Mapping[str, float]
    parent_posterior: Mapping[str, Mapping[str, float]]
    ancestor_occupancy: np.ndarray
    root_prior: Mapping[str, float]
    termination_quality: float
    path_status: Mapping[str, str]
    zero_weight_rows: tuple[str, ...] = ()
    all_zero_root_prior: bool = False

    def __post_init__(self) -> None:
        ids = tuple(str(value) for value in self.graph.memory_ids)
        id_set = set(ids)
        if len(ids) != len(id_set):
            raise ValueError("graph memory IDs must be unique")

        # A direct ``FrozenGraphMeasure(...)`` construction is a public audit
        # boundary, not an unchecked bag of arrays.  Normalize sparse rows,
        # then verify that every positive transition is an observed graph edge
        # and that each non-terminal row is stochastic.
        raw_transition = self.transition
        if not isinstance(raw_transition, Mapping):
            raise ValueError("transition must be an ID-addressed mapping")
        declared_edges = set(self.graph.edges)
        transition_rows: dict[str, dict[str, float]] = {identifier: {} for identifier in ids}
        seen_transition_rows: set[str] = set()
        for raw_left, raw_values in raw_transition.items():
            left = str(raw_left)
            if left not in id_set:
                raise ValueError("transition contains an unknown row ID")
            if left in seen_transition_rows:
                raise ValueError("transition row IDs must remain unique after string normalization")
            seen_transition_rows.add(left)
            if not isinstance(raw_values, Mapping):
                raise ValueError("transition rows must be mappings")
            seen_transition_columns: set[str] = set()
            for raw_right, raw_value in raw_values.items():
                right = str(raw_right)
                if right not in id_set or left == right:
                    raise ValueError("transition contains an invalid edge")
                if right in seen_transition_columns:
                    raise ValueError("transition column IDs must remain unique after string normalization")
                seen_transition_columns.add(right)
                if (left, right) not in declared_edges:
                    raise ValueError("transition contains an edge absent from the frozen graph")
                value = float(raw_value)
                if not np.isfinite(value) or value < 0.0:
                    raise ValueError("transition values must be finite and non-negative")
                if value > 0.0:
                    transition_rows[left][right] = value
        for left, values in transition_rows.items():
            total = float(sum(values.values()))
            if total > 0.0 and not np.isclose(total, 1.0, atol=1e-8, rtol=1e-8):
                raise ValueError(f"transition row {left} must sum to one")
            transition_rows[left] = {
                right: float(value) for right, value in sorted(values.items()) if value > 0.0
            }

        raw_parents = self.parent_posterior
        if not isinstance(raw_parents, Mapping):
            raise ValueError("parent_posterior must be an ID-addressed mapping")
        parents: dict[str, dict[str, float]] = {identifier: {} for identifier in ids}
        positive_incoming: dict[str, set[str]] = {identifier: set() for identifier in ids}
        for left, values in transition_rows.items():
            for right, value in values.items():
                if value > 0.0:
                    positive_incoming[right].add(left)
        seen_parent_children: set[str] = set()
        for raw_child, raw_values in raw_parents.items():
            child = str(raw_child)
            if child not in id_set:
                raise ValueError("parent_posterior contains an unknown child ID")
            if child in seen_parent_children:
                raise ValueError("parent_posterior child IDs must remain unique after string normalization")
            seen_parent_children.add(child)
            if not isinstance(raw_values, Mapping):
                raise ValueError("parent posterior rows must be mappings")
            row: dict[str, float] = {}
            seen_parent_ids: set[str] = set()
            for raw_parent, raw_probability in raw_values.items():
                parent = str(raw_parent)
                if parent not in id_set or parent == child:
                    raise ValueError("parent posterior contains an invalid parent ID")
                if parent in seen_parent_ids:
                    raise ValueError("parent posterior IDs must remain unique after string normalization")
                seen_parent_ids.add(parent)
                probability = float(raw_probability)
                if not np.isfinite(probability) or probability < 0.0:
                    raise ValueError("parent posterior values must be finite and non-negative")
                if probability > 0.0:
                    if parent not in positive_incoming[child]:
                        raise ValueError("parent posterior references a zero-mass transition")
                    row[parent] = probability
            total = float(sum(row.values()))
            if total > 0.0 and not np.isclose(total, 1.0, atol=1e-8, rtol=1e-8):
                raise ValueError(f"parent posterior for {child} must sum to one")
            parents[child] = {key: float(value) for key, value in sorted(row.items())}

        raw_access = self.access_quality
        if not isinstance(raw_access, Mapping):
            raise ValueError("access_quality must be an ID-addressed mapping")
        access: dict[str, float] = {}
        for raw_key, raw_value in raw_access.items():
            key = str(raw_key)
            if key in access:
                raise ValueError("access_quality IDs must remain unique after string normalization")
            access[key] = float(raw_value)
        if set(access) != id_set:
            raise ValueError("access_quality must cover exactly the frozen graph IDs")
        if any(not np.isfinite(value) or value < 0.0 for value in access.values()):
            raise ValueError("access_quality values must be finite and non-negative")

        raw_root = self.root_prior
        if not isinstance(raw_root, Mapping):
            raise ValueError("root_prior must be an ID-addressed mapping")
        root: dict[str, float] = {}
        for raw_key, raw_value in raw_root.items():
            key = str(raw_key)
            if key in root:
                raise ValueError("root_prior IDs must remain unique after string normalization")
            root[key] = float(raw_value)
        if set(root) - id_set:
            raise ValueError("root_prior contains an unknown ID")
        if any(not np.isfinite(value) or value < 0.0 for value in root.values()):
            raise ValueError("root_prior values must be finite and non-negative")
        # Determine roots with the same explicit-layer rule used by runtime
        # propagation.  A later-layer node with no positive edge is
        # unreachable, not a fresh root eligible for all-zero uniformization.
        roots = set(_effective_root_ids(self.graph, transition_rows))
        if any(value > 1e-12 and key not in roots for key, value in root.items()):
            raise ValueError("root_prior assigns mass to a non-root node")
        root = {identifier: float(root.get(identifier, 0.0)) for identifier in ids if identifier in roots}
        root_total = float(sum(root.values()))
        if roots and root_total > 0.0 and not np.isclose(root_total, 1.0, atol=1e-8, rtol=1e-8):
            raise ValueError("root_prior must sum to one")
        if roots and root_total <= 0.0 and not bool(self.all_zero_root_prior):
            raise ValueError("all-zero root_prior must be explicitly marked")

        raw_status = self.path_status
        if not isinstance(raw_status, Mapping):
            raise ValueError("path_status must be an ID-addressed mapping")
        status: dict[str, str] = {}
        for raw_key, raw_value in raw_status.items():
            key = str(raw_key)
            if key in status:
                raise ValueError("path_status IDs must remain unique after string normalization")
            status[key] = str(raw_value)
        if set(status) != id_set:
            raise ValueError("path_status must cover exactly the frozen graph IDs")

        # Check the defining forward recurrences as well as local ranges.  A
        # manually assembled measure that merely has stochastic rows but a
        # mismatched ``h``/``gamma`` would otherwise look certified while its
        # ancestor matrix described a different graph.
        for child in ids:
            expected_h = float(root.get(child, 0.0)) + sum(
                access[parent] * transition_rows[parent].get(child, 0.0) for parent in ids
            )
            if not np.isclose(access[child], expected_h, atol=1e-8, rtol=1e-8):
                raise ValueError(f"access_quality for {child} does not satisfy graph propagation")
            incoming = {
                parent: access[parent] * transition_rows[parent].get(child, 0.0)
                for parent in ids
                if transition_rows[parent].get(child, 0.0) > 0.0 and access[parent] > 0.0
            }
            incoming_total = float(sum(incoming.values()))
            supplied = parents[child]
            if incoming_total > 0.0:
                expected_gamma = {
                    parent: value / incoming_total for parent, value in incoming.items()
                }
                if set(supplied) != set(expected_gamma) or any(
                    not np.isclose(float(supplied.get(parent, 0.0)), probability, atol=1e-8, rtol=1e-8)
                    for parent, probability in expected_gamma.items()
                ):
                    raise ValueError(f"parent_posterior for {child} does not match access propagation")
            elif supplied:
                raise ValueError(f"parent_posterior for unreachable node {child} must be empty")

        transition = {
            left: MappingProxyType(dict(values)) for left, values in transition_rows.items()
        }
        parents = {child: MappingProxyType(dict(values)) for child, values in parents.items()}
        object.__setattr__(self, "transition", MappingProxyType(transition))
        object.__setattr__(self, "parent_posterior", MappingProxyType(parents))
        object.__setattr__(self, "access_quality", MappingProxyType(access))
        object.__setattr__(self, "root_prior", MappingProxyType(root))
        object.__setattr__(self, "path_status", MappingProxyType(status))
        matrix = np.array(self.ancestor_occupancy, dtype=np.float64, copy=True)
        if matrix.ndim != 2 or matrix.shape != (len(ids), len(ids)):
            raise ValueError("ancestor_occupancy must have shape (N, N)")
        if not np.all(np.isfinite(matrix)) or np.any(matrix < -1e-10) or np.any(matrix > 1.0 + 1e-8):
            raise ValueError("ancestor_occupancy must be finite and lie in [0, 1]")
        matrix.setflags(write=False)
        object.__setattr__(self, "ancestor_occupancy", matrix)
        zero_rows = tuple(identifier for identifier in ids if not transition_rows[identifier])
        supplied_zero_rows = tuple(str(value) for value in self.zero_weight_rows)
        if len(supplied_zero_rows) != len(set(supplied_zero_rows)):
            raise ValueError("zero_weight_rows must be unique")
        if supplied_zero_rows and set(supplied_zero_rows) != set(zero_rows):
            raise ValueError("zero_weight_rows does not match transition terminal rows")
        object.__setattr__(self, "zero_weight_rows", zero_rows)
        terminal = float(self.termination_quality)
        if not np.isfinite(terminal) or terminal < -1e-10 or terminal > 1.0 + 1e-8:
            raise ValueError("termination_quality must lie in [0, 1]")
        expected_terminal = float(sum(access[identifier] for identifier in zero_rows))
        if not np.isclose(terminal, expected_terminal, atol=1e-8, rtol=1e-8):
            raise ValueError("termination_quality does not match terminal access mass")
        object.__setattr__(self, "termination_quality", min(1.0, max(0.0, terminal)))
        if not isinstance(self.all_zero_root_prior, bool):
            raise ValueError("all_zero_root_prior must be boolean")

        # Verify the dynamic-program recurrence for ``w``.  This is O(N^2)
        # and occurs only at freeze/construction time, never per selector
        # marginal.
        expected_ancestor = np.zeros_like(matrix)
        order = _graph_topological_order(self.graph)
        positions = {identifier: index for index, identifier in enumerate(ids)}
        for child in order:
            column = positions[child]
            for parent, probability in parents[child].items():
                expected_ancestor[:, column] += float(probability) * expected_ancestor[:, positions[parent]]
                expected_ancestor[positions[parent], column] += float(probability)
        if not np.allclose(matrix, expected_ancestor, atol=1e-8, rtol=1e-8):
            raise ValueError("ancestor_occupancy does not satisfy the parent-posterior recurrence")

    @property
    def h(self) -> Mapping[str, float]:
        return self.access_quality

    @property
    def gamma(self) -> Mapping[str, Mapping[str, float]]:
        return self.parent_posterior

    @property
    def w(self) -> np.ndarray:
        return self.ancestor_occupancy

    def public_dict(self) -> dict[str, Any]:
        return {
            "graph_hash": self.graph.graph_hash,
            "transition": {key: dict(values) for key, values in self.transition.items()},
            "access_quality": dict(self.access_quality),
            "parent_posterior": {key: dict(values) for key, values in self.parent_posterior.items()},
            "ancestor_occupancy": self.ancestor_occupancy.tolist(),
            "root_prior": dict(self.root_prior),
            "termination_quality": float(self.termination_quality),
            "path_status": dict(self.path_status),
            "zero_weight_rows": list(self.zero_weight_rows),
            "all_zero_root_prior": bool(self.all_zero_root_prior),
        }


def _graph_topological_order(graph: FrozenProposalGraph) -> tuple[str, ...]:
    if graph.layers:
        return tuple(identifier for layer in graph.layers for identifier in layer)
    children = graph.children_map
    indegree = {identifier: 0 for identifier in graph.memory_ids}
    for _left, right in graph.edges:
        indegree[right] += 1
    queue = sorted(identifier for identifier, degree in indegree.items() if degree == 0)
    result: list[str] = []
    while queue:
        current = queue.pop(0)
        result.append(current)
        for child in sorted(children.get(current, ())):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
                queue.sort()
    if len(result) != len(graph.memory_ids):
        raise ValueError("proposal graph must be acyclic")
    return tuple(result)


def propagate_frozen_graph(graph: FrozenProposalGraph) -> FrozenGraphMeasure:
    """Compute ``P``, global visit quality ``h``, ``gamma`` and ``w``.

    ``h`` is an access probability for a random forward path.  Its sum over
    all nodes is generally greater than one (expected number of visited
    nodes), while the explicit terminal-state mass is one.
    """

    order = _graph_topological_order(graph)
    positions = {identifier: index for index, identifier in enumerate(graph.memory_ids)}
    transition = observed_transition(graph)
    # Structural zero-weight exposure records are retained for audit but do
    # not constitute navigation parents.  Root detection therefore uses the
    # positive sparse transition, otherwise a zero edge could hide a genuine
    # root and make terminal mass disappear.
    parents: dict[str, tuple[str, ...]] = {
        identifier: tuple(
            sorted(parent for parent, values in transition.items() if identifier in values and values[identifier] > 0.0)
        )
        for identifier in graph.memory_ids
    }
    # The first synchronous discovery layer is the declared source set.  Do
    # not promote an isolated later-layer proposal to a root merely because a
    # relation weight happened to be zero; such a candidate must remain
    # explicitly ``path_unavailable``.  Unlayered hand-built DAGs use all
    # structural roots, which is the natural interpretation in that API.
    roots = tuple(identifier for identifier in _effective_root_ids(graph, transition))
    declared_root = {identifier: float(graph.root_mass[index]) for index, identifier in enumerate(graph.memory_ids)}
    nonroot_mass = {
        identifier: value
        for identifier, value in declared_root.items()
        if value > 1e-12 and identifier not in roots
    }
    if nonroot_mass:
        raise ValueError(
            "frozen graph root_mass assigns positive mass to non-root IDs: "
            + ", ".join(sorted(nonroot_mass))
        )
    root_values = {identifier: max(0.0, declared_root.get(identifier, 0.0)) for identifier in roots}
    root_total = float(sum(root_values.values()))
    all_zero = root_total <= 0.0
    if all_zero and roots:
        root_values = {identifier: 1.0 / len(roots) for identifier in roots}
    elif root_total > 0.0:
        root_values = {identifier: value / root_total for identifier, value in root_values.items()}
    h = {identifier: 0.0 for identifier in graph.memory_ids}
    gamma: dict[str, dict[str, float]] = {identifier: {} for identifier in graph.memory_ids}
    for identifier in order:
        value = float(root_values.get(identifier, 0.0))
        for parent in parents.get(identifier, ()):
            value += h[parent] * transition.get(parent, {}).get(identifier, 0.0)
        h[identifier] = value
        if value > 0.0:
            incoming = {
                parent: h[parent] * transition.get(parent, {}).get(identifier, 0.0)
                for parent in parents.get(identifier, ())
                if h[parent] > 0.0 and transition.get(parent, {}).get(identifier, 0.0) > 0.0
            }
            total = float(sum(incoming.values()))
            if total > 0.0:
                gamma[identifier] = {parent: edge_mass / total for parent, edge_mass in sorted(incoming.items())}
    ancestor = np.zeros((len(graph.memory_ids), len(graph.memory_ids)), dtype=np.float64)
    for identifier in order:
        col = positions[identifier]
        for parent, probability in gamma.get(identifier, {}).items():
            ancestor[:, col] += probability * ancestor[:, positions[parent]]
            ancestor[positions[parent], col] += probability
    zero_rows = tuple(identifier for identifier in graph.memory_ids if not transition.get(identifier))
    terminal = float(sum(h[identifier] for identifier in zero_rows))
    # A malformed graph with no roots has no path mass; represent that state
    # explicitly rather than fabricating a uniform parent posterior.
    statuses = {
        identifier: ("available" if h[identifier] > 0.0 else "path_unavailable")
        for identifier in graph.memory_ids
    }
    return FrozenGraphMeasure(
        graph=graph,
        transition=transition,
        access_quality=h,
        parent_posterior=gamma,
        ancestor_occupancy=ancestor,
        root_prior=root_values,
        termination_quality=terminal,
        path_status=statuses,
        zero_weight_rows=zero_rows,
        all_zero_root_prior=all_zero,
    )


def shuffle_frozen_graph(graph: FrozenProposalGraph, seed: int = 0) -> FrozenProposalGraph:
    """Apply a seeded within-layer identity permutation to a frozen graph.

    Text, vectors and pointwise quality remain attached to their original
    memory IDs.  Only source-structure positions move: edges, edge weights,
    root mass and proposal provenance are migrated through the same bijection.
    The returned graph therefore has the same layered topology and weight
    multiset while changing which memory occupies each source position.
    """
    if isinstance(seed, (bool, np.bool_)):
        raise ValueError("shuffle seed must be a non-negative integer")
    try:
        numeric_seed = int(seed)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("shuffle seed must be a non-negative integer") from exc
    if numeric_seed < 0 or float(seed) != float(numeric_seed):
        raise ValueError("shuffle seed must be a non-negative integer")
    if graph.layers:
        layers = graph.layers
    else:
        # Hand-built audit graphs may omit explicit layers.  Derive a stable
        # topological layering instead of assigning singleton layers by ID
        # order (which can invert a valid edge such as ``b -> a``).
        depth: dict[str, int] = {}
        for identifier in _graph_topological_order(graph):
            parents = graph.parent_map.get(identifier, ())
            depth[identifier] = max((depth[parent] + 1 for parent in parents), default=0)
        by_depth: dict[int, list[str]] = {}
        for identifier in graph.memory_ids:
            by_depth.setdefault(depth.get(identifier, 0), []).append(identifier)
        layers = tuple(tuple(sorted(values)) for _level, values in sorted(by_depth.items()))
    rng = np.random.default_rng(numeric_seed)
    sigma: dict[str, str] = {}
    for layer in layers:
        ids = sorted(str(value) for value in layer)
        targets = list(ids)
        rng.shuffle(targets)
        sigma.update(zip(ids, targets))
    edges = tuple((sigma[left], sigma[right]) for left, right in graph.edges)
    edge_weights = tuple((sigma[left], sigma[right], float(weight)) for left, right, weight in graph.edge_weights)
    root_mass = {
        sigma[identifier]: float(weight)
        for identifier, weight in zip(graph.memory_ids, graph.root_mass)
        if float(weight) > 0.0
    }
    parent_sources = tuple(
        (sigma[child], tuple(sigma[parent] for parent in parents))
        for child, parents in graph.parent_sources
    )
    shuffled_layers = tuple(
        tuple(sigma[identifier] for identifier in layer)
        for layer in layers
    )
    proposal_records = tuple(
        (sigma[parent], sigma[candidate], layer, rank, score)
        for parent, candidate, layer, rank, score in graph.proposal_records
    )
    proposal_config = dict(graph.proposal_config)
    raw_anchor_exposures = proposal_config.get("raw_anchor_exposures")
    if isinstance(raw_anchor_exposures, (tuple, list)):
        proposal_config["raw_anchor_exposures"] = [
            [sigma.get(str(identifier), str(identifier)), float(score)]
            for identifier, score in raw_anchor_exposures
        ]
    raw_proposal_records = proposal_config.get("raw_proposal_records")
    if isinstance(raw_proposal_records, (tuple, list)):
        proposal_config["raw_proposal_records"] = [
            [
                sigma.get(str(parent), str(parent)),
                sigma.get(str(candidate), str(candidate)),
                layer,
                rank,
                score,
            ]
            for parent, candidate, layer, rank, score in raw_proposal_records
        ]
    return FrozenProposalGraph(
        memory_ids=graph.memory_ids,
        edges=edges,
        edge_weights=edge_weights,
        root_mass=root_mass,
        layers=shuffled_layers,
        parent_sources=parent_sources,
        proposal_records=proposal_records,
        cutoff=graph.cutoff,
        proposal_config={
            **proposal_config,
            "shuffle_seed": numeric_seed,
            # Persist the intervention itself so a frozen shuffle has a
            # distinct cache identity and can be audited without re-drawing
            # the RNG.
            "shuffle_permutation": dict(sorted(sigma.items())),
            "shuffle_effective_nodes": sum(left != right for left, right in sigma.items()),
        },
        domain_scope=graph.domain_scope,
        schema_version=graph.schema_version,
    )


def ancestor_occupancy_dp(
    graph: FrozenProposalGraph,
    parent_posterior: Mapping[str, Mapping[str, float]] | None = None,
) -> np.ndarray:
    """Standalone ``w_aj`` dynamic program used by mathematical audits."""

    measure = propagate_frozen_graph(graph) if parent_posterior is None else None
    gamma = parent_posterior or (measure.parent_posterior if measure is not None else {})
    order = _graph_topological_order(graph)
    positions = {identifier: index for index, identifier in enumerate(graph.memory_ids)}
    result = np.zeros((len(graph.memory_ids), len(graph.memory_ids)), dtype=np.float64)
    for child in order:
        col = positions[child]
        for parent, probability in gamma.get(child, {}).items():
            if parent not in positions:
                continue
            result[:, col] += float(probability) * result[:, positions[parent]]
            result[positions[parent], col] += float(probability)
    result.setflags(write=False)
    return result


def count_path_hypotheses(
    measure: FrozenGraphMeasure,
    candidate_id: str,
    *,
    cap: int | None = None,
) -> int:
    """Count positive posterior paths without materializing the paths.

    The count is a small dynamic program over the frozen DAG.  ``cap`` is an
    optional saturation value useful to decide whether display provenance can
    be enumerated safely; a saturated count is still sufficient to report
    that the path set is larger than the requested display budget.
    """

    candidate = str(candidate_id)
    if candidate not in measure.graph.memory_ids:
        raise KeyError(candidate)
    limit = None if cap is None else int(cap)
    if limit is not None and limit < 0:
        raise ValueError("path count cap must be non-negative")
    order = _graph_topological_order(measure.graph)
    counts: dict[str, int] = {}
    for node in order:
        parents = measure.parent_posterior.get(node, {})
        if not parents:
            counts[node] = 1 if float(measure.access_quality.get(node, 0.0)) > 0.0 else 0
            continue
        total = sum(counts.get(str(parent), 0) for parent, probability in parents.items() if float(probability) > 0.0)
        counts[node] = min(total, limit) if limit is not None else total
    return int(counts.get(candidate, 0))


def representative_path_hypothesis(
    measure: FrozenGraphMeasure,
    candidate_id: str,
    *,
    branch_id: str = "",
) -> tuple[PathHypothesis, ...]:
    """Return one deterministic local-parent path for display/audit only.

    The formal ancestor representation is ``measure.ancestor_occupancy``;
    this helper intentionally returns only a compact representative path and
    never substitutes it for the full parent posterior or claims to solve a
    global MAP/Viterbi problem.
    """

    candidate = str(candidate_id)
    if candidate not in measure.graph.memory_ids:
        raise KeyError(candidate)
    if float(measure.access_quality.get(candidate, 0.0)) <= 0.0:
        return ()
    chain = [candidate]
    current = candidate
    seen: set[str] = set()
    while current in measure.parent_posterior and measure.parent_posterior[current]:
        if current in seen:
            return ()
        seen.add(current)
        parents = measure.parent_posterior[current]
        parent = min(
            parents,
            key=lambda identifier: (-float(parents[identifier]), str(identifier)),
        )
        parent = str(parent)
        if parent in chain:
            return ()
        chain.append(parent)
        current = parent
    path = tuple(reversed(chain))
    path_mass = 1.0
    for parent, child in zip(path[:-1], path[1:]):
        path_mass *= float(measure.parent_posterior.get(child, {}).get(parent, 0.0))
    access = float(measure.access_quality.get(candidate, 0.0))
    posterior = path_mass if access > 0.0 else 0.0
    terminal_parents = dict(measure.parent_posterior.get(candidate, {}))
    return (
        PathHypothesis(
            path_ids=path,
            parent_posterior=terminal_parents,
            branch_id=str(branch_id or "representative_local_parent"),
            posterior=float(min(1.0, max(0.0, posterior))),
            support=access,
        ),
    )


# Compatibility spelling retained for notebooks written against the previous
# implementation.  It is intentionally an alias, not a claim that the local
# parent chain is a global MAP path.
map_path_hypothesis = representative_path_hypothesis


def display_path_hypotheses(
    measure: FrozenGraphMeasure,
    candidate_id: str,
    *,
    max_paths: int = 128,
    branch_id: str = "",
) -> tuple[tuple[PathHypothesis, ...], bool]:
    """Return bounded path provenance and whether it was truncated.

    Small DAGs retain every positive path, which is convenient for audits and
    diamond fixtures.  Once the dynamic count exceeds ``max_paths`` the
    production representation falls back to one representative/local-parent
    path; the exact ``gamma`` and ``w`` values remain available on ``measure``.
    """

    if isinstance(max_paths, bool) or int(max_paths) <= 0 or int(max_paths) != float(max_paths):
        raise ValueError("max_paths must be a positive integer")
    limit = int(max_paths)
    count = count_path_hypotheses(measure, candidate_id, cap=limit + 1)
    if count <= limit:
        paths = enumerate_path_hypotheses(
            str(candidate_id),
            branch_id=branch_id,
            node_parent_posteriors=measure.parent_posterior,
            root_posterior=measure.root_prior,
            supports=measure.access_quality,
            max_depth=max(1, len(measure.graph.memory_ids) + 1),
            transition=measure.transition,
            ids=measure.graph.memory_ids,
        )
        return paths, False
    return representative_path_hypothesis(measure, candidate_id, branch_id=branch_id), True


def cluster_global_propagation(
    cluster_mass: float,
    member_probabilities: Mapping[str, float],
    transition: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    """Propagate a globally weighted cluster without resetting mass to one."""

    mass = float(cluster_mass)
    if not np.isfinite(mass) or mass < 0.0:
        raise ValueError("cluster mass must be finite and non-negative")
    result: dict[str, float] = {}
    for member, probability in member_probabilities.items():
        p = float(probability)
        if not np.isfinite(p) or p < 0.0:
            raise ValueError("cluster member probabilities must be finite and non-negative")
        for candidate, edge in transition.get(str(member), {}).items():
            result[str(candidate)] = result.get(str(candidate), 0.0) + mass * p * float(edge)
    return result


# Public aliases with the terminology used by the semantic design.
build_sparse_transition = observed_transition
compute_global_propagation = propagate_frozen_graph
compute_ancestor_occupancy = ancestor_occupancy_dp
angular_affinity = angular_navigation_affinity


# Descriptive aliases used by the design notes and external audit notebooks.
member_measure = branch_measure
propagate_branch_mass = propagate_mass
compute_parent_posterior = parent_posterior
enumerate_paths = enumerate_path_hypotheses


__all__ = [
    "BranchMeasure",
    "branch_measure",
    "normalize_member_mass",
    "propagate_mass",
    "parent_posterior",
    "enumerate_path_hypotheses",
    "merge_path_hypotheses",
    "normalize_path_hypotheses",
    "path_posterior",
    "member_measure",
    "propagate_branch_mass",
    "compute_parent_posterior",
    "enumerate_paths",
    "angular_navigation_affinity",
    "angular_affinity",
    "observed_transition",
    "build_sparse_transition",
    "FrozenGraphMeasure",
    "propagate_frozen_graph",
    "shuffle_frozen_graph",
    "compute_global_propagation",
    "ancestor_occupancy_dp",
    "compute_ancestor_occupancy",
    "count_path_hypotheses",
    "representative_path_hypothesis",
    "map_path_hypothesis",
    "display_path_hypotheses",
    "cluster_global_propagation",
]
