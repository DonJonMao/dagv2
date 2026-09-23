from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass, field, replace
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

from .diagnostic_identity import DeploymentIdentity, validate_provider_params

# ``dataclasses.replace`` passes every field explicitly to this custom
# constructor.  A private sentinel lets us distinguish that case from a
# caller omitting a semantic field, so profile defaults are applied only when
# they were genuinely omitted (and an explicit ``quality_mode=direct_cosine``
# is not unexpectedly rewritten to ``frozen_reranker``).
_UNSET = object()


def _strict_int(
    value: Any,
    name: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
    maximum: int | None = None,
) -> int:
    """Normalize an integer configuration field without truncation."""
    if isinstance(value, bool):
        raise ValueError(f"retrieval.{name} must be an integer")
    if isinstance(value, Integral):
        result = int(value)
    elif isinstance(value, Real):
        numeric = float(value)
        if not math.isfinite(numeric) or numeric != float(int(numeric)):
            raise ValueError(f"retrieval.{name} must be an integer")
        result = int(numeric)
    elif isinstance(value, str):
        text = value.strip()
        try:
            numeric = float(text)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"retrieval.{name} must be an integer") from exc
        if not text or not math.isfinite(numeric) or numeric != float(int(numeric)):
            raise ValueError(f"retrieval.{name} must be an integer")
        result = int(numeric)
    else:
        raise ValueError(f"retrieval.{name} must be an integer")
    if positive and result <= 0:
        raise ValueError(f"retrieval.{name} must be positive")
    if nonnegative and result < 0:
        raise ValueError(f"retrieval.{name} must be non-negative")
    if maximum is not None and result > maximum:
        raise ValueError(f"retrieval.{name} is too large (maximum {maximum})")
    return result


def _strict_float(value: Any, name: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    if isinstance(value, bool):
        raise ValueError(f"retrieval.{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"retrieval.{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"retrieval.{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"retrieval.{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"retrieval.{name} must be <= {maximum}")
    return result


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def _strict_string(value: Any, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    result = value.strip()
    if not allow_empty and not result:
        raise ValueError(f"{name} cannot be empty")
    return value


def _token(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _alias(value: Any, aliases: Mapping[str, str]) -> str:
    normalized = _token(value)
    return aliases.get(normalized, normalized)


@dataclass(frozen=True, init=False)
class RetrievalConfig:
    """One runtime configuration for every BridgeTree module combination.

    ``first_hop_width`` remains an accepted constructor keyword for source
    compatibility, but ``initial_width`` is the canonical serialized name.
    """

    initial_width: int = 12
    branch_width: int = 8
    context_size: int = 5
    search_budget: int = 64
    max_ann_calls: int | None = None
    max_candidate_exposure: int | None = None
    max_depth: int = 2
    cluster_mode: str = "fixed"
    cluster_count: int = 4
    max_clusters: int = 8
    min_cluster_size: int = 1
    search_order: str = "best_first"
    feature_mode: str = "rho"
    selection_mode: str = "rho_logdet"
    stop_mode: str = "budget"
    diagnostic_level: str = "light"
    root_anchor_weight: float = 0.0
    mmr_lambda: float = 0.7
    index_backend: str = "exact"
    faiss_exclusion_margin: int = 32
    tie_tolerance: float = 1e-12
    # The four formal operators are opt-in so the default remains R1.
    temporal_measure: bool = False
    measure_propagation: bool = False
    state_information: bool = False
    information_certificate: bool = False
    state_basis_mode: str = "option_contrast"
    certificate_domain: str = "exact_partition"
    # Named semantic profiles are additive to the historical TMIC switches.
    # ``legacy_core`` is the direct-constructor compatibility default; the
    # shipped default YAML selects ``semantic_path_v1`` explicitly.
    profile: str = "legacy_core"
    proposal_mode: str = "legacy_first_arrival"
    relation_mode: str = "cosine"
    quality_mode: str = "direct_cosine"
    path_mode: str = "legacy"
    certificate_mode: str = "off"
    context_unit: str = "memory"
    quality_score_space: str = "unit_interval"
    scorer_fingerprint: str = ""
    proposal_width: int | None = None
    certificate_epsilon: float = 1e-12
    context_strict: bool = True
    score_contract: str = "unit_interval"
    representation_mode: str = "cached_memory"
    # Complete path hypotheses are display provenance only.  The production
    # ancestry calculation uses the O(N²) occupancy DP; this cap bounds the
    # optional small-DAG path enumeration retained for human-readable audit
    # records.
    max_path_hypotheses: int = 128

    def __init__(
        self,
        initial_width: int = 12,
        branch_width: int = 8,
        context_size: int = 5,
        search_budget: int = 64,
        max_ann_calls: int | None = None,
        max_candidate_exposure: int | None = None,
        max_depth: int = 2,
        cluster_mode: str = "fixed",
        cluster_count: int = 4,
        max_clusters: int = 8,
        min_cluster_size: int = 1,
        search_order: str = "best_first",
        feature_mode: str | object = _UNSET,
        selection_mode: str | object = _UNSET,
        stop_mode: str = "budget",
        diagnostic_level: str = "light",
        root_anchor_weight: float = 0.0,
        mmr_lambda: float = 0.7,
        index_backend: str = "exact",
        faiss_exclusion_margin: int = 32,
        tie_tolerance: float = 1e-12,
        temporal_measure: bool = False,
        measure_propagation: bool = False,
        state_information: bool = False,
        information_certificate: bool = False,
        state_basis_mode: str = "option_contrast",
        certificate_domain: str = "exact_partition",
        first_hop_width: int | None = None,
        profile: str = "legacy_core",
        proposal_mode: str | object = _UNSET,
        relation_mode: str | object = _UNSET,
        quality_mode: str | object = _UNSET,
        path_mode: str | object = _UNSET,
        certificate_mode: str | object = _UNSET,
        context_unit: str | object = _UNSET,
        quality_score_space: str | object = _UNSET,
        scorer_fingerprint: str = "",
        proposal_width: int | None = None,
        certificate_epsilon: float = 1e-12,
        context_strict: bool = True,
        score_contract: str | object = _UNSET,
        representation_mode: str | object = _UNSET,
        max_path_hypotheses: int = 128,
        path_hypothesis_limit: int | None = None,
    ):
        if first_hop_width is not None:
            if initial_width != 12 and initial_width != first_hop_width:
                raise ValueError("initial_width and legacy first_hop_width disagree")
            initial_width = first_hop_width
        # Canonicalize resource fields at construction time.  This keeps
        # ``dataclasses.replace``/YAML/CLI paths from carrying fractional or
        # boolean values into a later budget object, where they would be
        # silently truncated by ``int(...)``.
        initial_width = _strict_int(initial_width, "initial_width", positive=True)
        branch_width = _strict_int(branch_width, "branch_width", positive=True)
        context_size = _strict_int(context_size, "context_size", positive=True)
        search_budget = _strict_int(search_budget, "search_budget", positive=True)
        max_depth = _strict_int(max_depth, "max_depth", positive=True)
        cluster_count = _strict_int(cluster_count, "cluster_count", positive=True)
        max_clusters = _strict_int(max_clusters, "max_clusters", positive=True)
        min_cluster_size = _strict_int(min_cluster_size, "min_cluster_size", positive=True)
        faiss_exclusion_margin = _strict_int(
            faiss_exclusion_margin, "faiss_exclusion_margin", positive=True
        )
        if max_ann_calls is not None:
            max_ann_calls = _strict_int(max_ann_calls, "max_ann_calls", positive=True)
        if max_candidate_exposure is not None:
            max_candidate_exposure = _strict_int(
                max_candidate_exposure, "max_candidate_exposure", positive=True
            )
        if proposal_width is not None:
            proposal_width = _strict_int(proposal_width, "proposal_width", positive=True)
        max_path_hypotheses = _strict_int(
            max_path_hypotheses,
            "max_path_hypotheses",
            positive=True,
            maximum=1_000_000,
        )
        if path_hypothesis_limit is not None:
            path_hypothesis_limit = _strict_int(
                path_hypothesis_limit,
                "path_hypothesis_limit",
                positive=True,
                maximum=1_000_000,
            )
        # Resolve aliases before applying profile defaults.  Keeping this
        # normalization at the dataclass boundary makes YAML, CLI overrides,
        # ``replace`` and direct API calls hash to the same protocol values.
        profile = _alias(
            profile,
            {
                "semantic_path": "semantic_path_v1",
                "semantic_path1": "semantic_path_v1",
                "semantic": "semantic",
                "legacy": "legacy",
                "legacy_core": "legacy_core",
                "legacy_path": "legacy_path",
            },
        )
        feature_was_set = feature_mode is not _UNSET
        selection_was_set = selection_mode is not _UNSET
        proposal_was_set = proposal_mode is not _UNSET
        relation_was_set = relation_mode is not _UNSET
        quality_was_set = quality_mode is not _UNSET
        path_was_set = path_mode is not _UNSET
        certificate_was_set = certificate_mode is not _UNSET
        context_was_set = context_unit is not _UNSET
        score_space_was_set = quality_score_space is not _UNSET
        contract_was_set = score_contract is not _UNSET
        representation_was_set = representation_mode is not _UNSET
        feature_mode = "rho" if not feature_was_set else _alias(
            feature_mode,
            {
                "memory": "cached_memory",
                "cached": "cached_memory",
                "cached_memory_embedding": "cached_memory",
                "query": "query_conditioned",
                "query_conditioned_memory": "query_conditioned",
            },
        )
        selection_mode = "rho_logdet" if not selection_was_set else _alias(
            selection_mode,
            {
                "semantic_path": "semantic_path_logdet",
                "semantic_path_v1": "semantic_path_logdet",
                "logdet": "rho_logdet",
                "topk": "rho_topk",
            },
        )
        proposal_mode = "legacy_first_arrival" if not proposal_was_set else _alias(
            proposal_mode,
            {
                "member_vector": "real_member_vector",
                "real_member": "real_member_vector",
                "member_query_anchor": "real_member_query_anchor",
                "query_anchor": "real_member_query_anchor",
                "round_robin_members": "round_robin",
            },
        )
        relation_mode = "cosine" if not relation_was_set else _alias(
            relation_mode,
            {"angle": "angular", "angular_affinity": "angular", "dot": "cosine"},
        )
        quality_mode = "direct_cosine" if not quality_was_set else _alias(
            quality_mode,
            {
                "direct": "direct_cosine",
                "cosine": "direct_cosine",
                "reranker": "frozen_reranker",
                "frozen": "frozen_reranker",
                "pointwise_reranker": "frozen_reranker",
            },
        )
        path_mode = "legacy" if not path_was_set else _alias(
            path_mode,
            {
                "expected_scatter": "posterior_expected_scatter",
                "posterior": "posterior_expected_scatter",
                "r1_path": "single_path",
                "no_path": "none",
                "flat": "none",
            },
        )
        certificate_mode = "off" if not certificate_was_set else _alias(
            certificate_mode,
            {"none": "off", "on": "lazy", "certificate": "lazy", "certified": "lazy"},
        )
        context_unit = "memory" if not context_was_set else _alias(
            context_unit, {"tokens": "token", "cardinality": "memory", "count": "memory"}
        )
        quality_score_space = "unit_interval" if not score_space_was_set else _alias(
            quality_score_space,
            {
                "probability": "unit_interval",
                "probabilities": "unit_interval",
                "unit": "unit_interval",
                "unitinterval": "unit_interval",
                "sigmoid": "unit_interval",
                "logit": "logit_difference",
                "logit_diff": "logit_difference",
                "logitdifference": "logit_difference",
                "raw_logit_difference": "logit_difference",
            },
        )
        score_contract = "unit_interval" if not contract_was_set else _alias(
            score_contract,
            {
                "probability": "unit_interval",
                "probabilities": "unit_interval",
                "unit": "unit_interval",
                "logit": "logit_difference",
                "logit_diff": "logit_difference",
            },
        )
        representation_mode = "cached_memory" if not representation_was_set else _alias(
            representation_mode,
            {"memory": "cached_memory", "cached": "cached_memory", "query": "query_conditioned"},
        )
        if path_hypothesis_limit is not None:
            if max_path_hypotheses != 128 and max_path_hypotheses != path_hypothesis_limit:
                raise ValueError("max_path_hypotheses and path_hypothesis_limit disagree")
            max_path_hypotheses = path_hypothesis_limit

        # Named profiles have sensible defaults when fields were omitted.
        # Explicit values always win, including values passed by
        # ``dataclasses.replace``.
        if profile == "legacy_path" and not feature_was_set and not selection_was_set:
            feature_mode = "path_conditioned"
            selection_mode = "path_logdet"
        if profile in {"semantic", "semantic_path_v1"}:
            if not proposal_was_set:
                proposal_mode = "real_member_query_anchor"
            if not relation_was_set:
                relation_mode = "angular"
            if not quality_was_set:
                quality_mode = "frozen_reranker"
            if not path_was_set:
                path_mode = "posterior_expected_scatter"
            if not selection_was_set:
                selection_mode = "semantic_path_logdet"
            if certificate_domain == "exact_partition":
                certificate_domain = "frozen_pool"
            if not contract_was_set:
                score_contract = quality_score_space
            if not feature_was_set:
                feature_mode = "cached_memory"
            if not representation_was_set:
                representation_mode = (
                    "query_conditioned" if feature_mode == "query_conditioned" else "cached_memory"
                )
        values = locals()
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, values[name])

    @property
    def first_hop_width(self) -> int:
        """Deprecated read-only alias for older integrations."""
        return self.initial_width

    @property
    def path_hypothesis_limit(self) -> int:
        """Compatibility alias for the bounded display-provenance cap."""
        return self.max_path_hypotheses

    @property
    def T(self) -> bool:
        return bool(self.temporal_measure)

    @property
    def M(self) -> bool:
        return bool(self.measure_propagation)

    @property
    def I(self) -> bool:  # noqa: E743 - public TMIC switch alias
        return bool(self.state_information)

    @property
    def C(self) -> bool:
        return bool(self.information_certificate)

    @property
    def module_switches(self) -> Dict[str, bool]:
        return {
            "T": self.T,
            "M": self.M,
            "I": self.I,
            "C": self.C,
        }

    def validate(self) -> None:
        for name in ("temporal_measure", "measure_propagation", "state_information", "information_certificate"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"retrieval.{name} must be boolean")
        positive = {
            "initial_width": self.initial_width,
            "branch_width": self.branch_width,
            "context_size": self.context_size,
            "search_budget": self.search_budget,
            "max_depth": self.max_depth,
            "cluster_count": self.cluster_count,
            "max_clusters": self.max_clusters,
            "min_cluster_size": self.min_cluster_size,
            "faiss_exclusion_margin": self.faiss_exclusion_margin,
        }
        for name, value in positive.items():
            _strict_int(value, name, positive=True)
        if self.max_ann_calls is not None:
            _strict_int(self.max_ann_calls, "max_ann_calls", positive=True)
        if self.max_candidate_exposure is not None:
            _strict_int(self.max_candidate_exposure, "max_candidate_exposure", positive=True)
        if self.initial_width > self.search_budget:
            raise ValueError("retrieval.initial_width must be <= retrieval.search_budget")
        if self.context_size > self.search_budget:
            raise ValueError("retrieval.context_size must be <= retrieval.search_budget")
        if self.max_candidate_exposure is not None and self.context_size > self.max_candidate_exposure:
            raise ValueError("retrieval.context_size must be <= retrieval.max_candidate_exposure")
        if self.max_candidate_exposure is not None and self.initial_width > self.max_candidate_exposure:
            raise ValueError("retrieval.initial_width must be <= retrieval.max_candidate_exposure")
        if self.proposal_width is not None:
            _strict_int(self.proposal_width, "proposal_width", positive=True)
        path_limit = _strict_int(
            self.max_path_hypotheses,
            "max_path_hypotheses",
            positive=True,
            maximum=1_000_000,
        )
        if path_limit != self.max_path_hypotheses:
            raise ValueError("retrieval.max_path_hypotheses must be canonicalized as an integer")
        if not isinstance(self.context_strict, bool):
            raise ValueError("retrieval.context_strict must be boolean")
        _strict_float(self.certificate_epsilon, "certificate_epsilon", minimum=0.0)
        choices = {
            "cluster_mode": (self.cluster_mode, {"none", "fixed", "effective_rank"}),
            "search_order": (self.search_order, {"best_first", "bfs"}),
            "feature_mode": (
                self.feature_mode,
                {"rho", "path_conditioned", "cached_memory", "query_conditioned", "memory"},
            ),
            "selection_mode": (
                self.selection_mode,
                {
                    "rho_topk",
                    "mmr",
                    "rho_logdet",
                    "path_logdet",
                    "semantic_path_logdet",
                    "pure_rerank",
                    "frozen_listwise",
                },
            ),
            "stop_mode": (self.stop_mode, {"budget", "certificate_or_budget"}),
            "diagnostic_level": (self.diagnostic_level, {"off", "light", "full"}),
            "index_backend": (self.index_backend, {"exact", "faiss"}),
            "state_basis_mode": (self.state_basis_mode, {"option_contrast", "identity"}),
            "certificate_domain": (self.certificate_domain, {"exact_partition", "frozen_pool"}),
            "profile": (self.profile, {"legacy_core", "legacy_path", "semantic_path_v1", "semantic", "legacy"}),
            "proposal_mode": (
                self.proposal_mode,
                {
                    "legacy_first_arrival",
                    "dense",
                    "centroid",
                    "real_member_vector",
                    "real_member_query_anchor",
                    "offline_q_plus_anchor",
                    "round_robin",
                },
            ),
            "relation_mode": (self.relation_mode, {"cosine", "angular", "legacy_temporal_overlap"}),
            "quality_mode": (self.quality_mode, {"direct_cosine", "rho", "frozen_reranker", "mapping", "constant"}),
            "path_mode": (self.path_mode, {"legacy", "none", "single_path", "posterior_expected_scatter", "shuffle"}),
            "certificate_mode": (self.certificate_mode, {"off", "lazy", "certificate_or_budget"}),
            "context_unit": (self.context_unit, {"memory", "token"}),
            "representation_mode": (self.representation_mode, {"cached_memory", "query_conditioned"}),
        }
        for name, (value, allowed) in choices.items():
            if value not in allowed:
                raise ValueError(f"retrieval.{name} must be one of {sorted(allowed)}")
        if self.selection_mode == "path_logdet" and self.feature_mode != "path_conditioned":
            raise ValueError("retrieval.path_logdet requires feature_mode=path_conditioned")
        if self.selection_mode == "rho_logdet" and self.feature_mode != "rho":
            raise ValueError("retrieval.rho_logdet requires feature_mode=rho")
        if self.stop_mode == "certificate_or_budget" and self.selection_mode not in {
            "rho_logdet",
            "path_logdet",
            "semantic_path_logdet",
        }:
            raise ValueError("retrieval.certificate_or_budget requires a logdet selection mode")
        if self.information_certificate and self.selection_mode not in {
            "rho_logdet",
            "path_logdet",
            "semantic_path_logdet",
        }:
            raise ValueError("retrieval.information_certificate requires a logdet selection mode")
        if (
            self.information_certificate
            and self.profile not in {"semantic", "semantic_path_v1"}
            and self.certificate_domain != "exact_partition"
        ):
            raise ValueError("retrieval.information_certificate requires certificate_domain=exact_partition")
        if self.profile == "legacy_core":
            # Legacy direct configurations intentionally retain their old
            # switch semantics.  A caller may still set the named fields for
            # diagnostics, but a semantic route is selected only by the
            # explicit semantic profile.
            pass
        elif self.profile in {"semantic", "semantic_path_v1"}:
            if self.certificate_mode == "certificate_or_budget" and self.certificate_domain != "frozen_pool":
                raise ValueError("semantic certificate stopping requires certificate_domain=frozen_pool")
            if self.relation_mode == "legacy_temporal_overlap":
                raise ValueError("semantic profiles cannot use temporal-overlap transition gates")
            if self.proposal_mode in {"centroid", "legacy_first_arrival"}:
                raise ValueError(
                    "semantic profiles require real-member or dense proposal modes; centroid is legacy-only"
                )
            if self.context_unit != "memory":
                raise ValueError(
                    "semantic_path_v1 uses cardinality (memory) selection; token budgets need a separate selector"
                )
        elif self.profile in {"legacy_path", "legacy"}:
            pass
        canonical_score_spaces = {
            "unit_interval": "unit_interval",
            "probability": "unit_interval",
            "probabilities": "unit_interval",
            "unit": "unit_interval",
            "sigmoid": "unit_interval",
            "logit_difference": "logit_difference",
            "logit": "logit_difference",
            "logit_diff": "logit_difference",
            "raw_logit_difference": "logit_difference",
        }
        quality_space_token = str(self.quality_score_space).strip().lower().replace("-", "_").replace(" ", "_")
        if quality_space_token not in canonical_score_spaces:
            raise ValueError("retrieval.quality_score_space must declare unit_interval or logit_difference")
        canonical_quality_space = canonical_score_spaces[quality_space_token]
        contract = str(self.score_contract).strip().lower().replace("-", "_")
        if contract not in {"unit_interval", "probability", "logit_difference", "logit", "logit_diff"}:
            raise ValueError("retrieval.score_contract must declare unit_interval or logit_difference")
        canonical_contract = {
            "probability": "unit_interval",
            "unit_interval": "unit_interval",
            "logit": "logit_difference",
            "logit_diff": "logit_difference",
            "logit_difference": "logit_difference",
        }[contract]
        if self.profile in {"semantic", "semantic_path_v1"} and canonical_contract != canonical_quality_space:
            raise ValueError("retrieval.score_contract must match quality_score_space for semantic profiles")
        if self.profile in {"semantic", "semantic_path_v1"}:
            if self.feature_mode == "query_conditioned" and self.representation_mode != "query_conditioned":
                raise ValueError("query-conditioned feature_mode requires representation_mode=query_conditioned")
            if self.representation_mode == "query_conditioned" and self.feature_mode != "query_conditioned":
                raise ValueError("representation_mode=query_conditioned requires feature_mode=query_conditioned")
        # The old root-anchor interpolation is a compatibility option.  It is
        # deliberately disallowed in a TMIC run because it is an additional
        # path/recency-like scoring term outside the specified operators.
        if (
            self.root_anchor_weight != 0.0
            and (
                self.temporal_measure
                or self.measure_propagation
                or self.state_information
                or self.information_certificate
            )
        ):
            raise ValueError("root_anchor_weight is legacy-only and must be zero for TMIC")
        _strict_float(self.root_anchor_weight, "root_anchor_weight", minimum=0.0, maximum=1.0)
        _strict_float(self.mmr_lambda, "mmr_lambda", minimum=0.0, maximum=1.0)
        _strict_float(self.tie_tolerance, "tie_tolerance", minimum=0.0)


@dataclass(frozen=True)
class EndpointConfig:
    endpoint: str
    model: str = ""
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        endpoint = _strict_string(self.endpoint, "endpoint")
        model = _strict_string(self.model, "model", allow_empty=True)
        timeout = _strict_float(self.timeout_seconds, "timeout_seconds", minimum=1e-9)
        object.__setattr__(self, "endpoint", endpoint)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "timeout_seconds", timeout)
        object.__setattr__(self, "deployment_identity", DeploymentIdentity.from_value(getattr(self, "deployment_identity", {})))


@dataclass(frozen=True)
class EmbeddingConfig(EndpointConfig):
    backend: str = "remote"
    batch_size: int = 32
    local_model_path: str = ""
    query_instruction: str = (
        "Instruct: Retrieve past personal interactions that help answer the current request\nQuery: "
    )
    # Appended in each concrete dataclass to preserve its Python 3.9 legacy
    # positional constructor (a new base field would shift all child fields).
    deployment_identity: DeploymentIdentity = field(default_factory=DeploymentIdentity)

    def __post_init__(self) -> None:
        super().__post_init__()
        backend = _strict_string(self.backend, "embedding backend").strip().lower()
        if backend not in {"remote", "local"}:
            raise ValueError("embedding backend must be remote or local")
        batch_size = _strict_int(self.batch_size, "embedding batch_size", positive=True)
        local_model_path = _strict_string(self.local_model_path, "embedding local_model_path", allow_empty=True)
        query_instruction = _strict_string(
            self.query_instruction, "embedding query_instruction", allow_empty=True
        )
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "batch_size", batch_size)
        object.__setattr__(self, "local_model_path", local_model_path)
        object.__setattr__(self, "query_instruction", query_instruction)


@dataclass(frozen=True)
class RerankerConfig(EndpointConfig):
    cache_dir: str = "outputs/rerank_cache"
    # Explicit contract for semantic quality adaptation.  ``unit_interval``
    # is the safe default for existing services; raw logits must be declared
    # as ``logit_difference`` before sigmoid conversion.
    score_space: str = "unit_interval"
    score_contract: str = "pointwise"
    task_instruction: str = ""
    deployment_identity: DeploymentIdentity = field(default_factory=DeploymentIdentity)
    max_batch_documents: int | None = None
    required_max_model_len: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("max_batch_documents", "required_max_model_len"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _strict_int(value, "reranker " + name, positive=True))
        cache_dir = _strict_string(self.cache_dir, "reranker cache_dir")
        score_space = _strict_string(self.score_space, "reranker score_space").strip().lower().replace("-", "_")
        score_contract = _strict_string(self.score_contract, "reranker score_contract").strip().lower()
        task_instruction = _strict_string(self.task_instruction, "reranker task_instruction", allow_empty=True)
        if score_space in {"probability", "probabilities", "unit", "sigmoid"}:
            score_space = "unit_interval"
        elif score_space in {"logit", "logit_diff", "raw_logit_difference"}:
            score_space = "logit_difference"
        if score_space not in {"unit_interval", "logit_difference"}:
            raise ValueError("reranker score_space must be unit_interval or logit_difference")
        if score_contract not in {"pointwise", "listwise"}:
            raise ValueError("reranker score_contract must be pointwise or listwise")
        object.__setattr__(self, "cache_dir", cache_dir)
        object.__setattr__(self, "score_space", score_space)
        object.__setattr__(self, "score_contract", score_contract)
        object.__setattr__(self, "task_instruction", task_instruction)


@dataclass(frozen=True)
class GeneratorConfig(EndpointConfig):
    # A private local config can supply credentials directly. A non-empty
    # environment value remains an optional deployment override.
    api_key: str = ""
    api_key_env: str = "BRIDGETREE_CHAT_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 512
    context_token_budget: int = 8192
    deployment_identity: DeploymentIdentity = field(default_factory=DeploymentIdentity)
    provider_request_params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        api_key = _strict_string(self.api_key, "generator api_key", allow_empty=True)
        api_key_env = _strict_string(self.api_key_env, "generator api_key_env")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", api_key_env):
            raise ValueError("generator api_key_env must be a valid environment variable name")
        temperature = _strict_float(self.temperature, "generator temperature", minimum=0.0)
        max_tokens = _strict_int(self.max_tokens, "generator max_tokens", positive=True)
        context_token_budget = _strict_int(
            self.context_token_budget, "generator context_token_budget", positive=True
        )
        object.__setattr__(self, "api_key", api_key)
        object.__setattr__(self, "api_key_env", api_key_env)
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "max_tokens", max_tokens)
        object.__setattr__(self, "context_token_budget", context_token_budget)
        object.__setattr__(self, "provider_request_params", validate_provider_params(self.provider_request_params))

    def resolved_api_key(self) -> str:
        return os.environ.get(self.api_key_env) or self.api_key


@dataclass(frozen=True)
class ModelsConfig:
    embedding: EmbeddingConfig
    reranker: RerankerConfig
    generator: GeneratorConfig


@dataclass(frozen=True)
class BridgeRerankConfig:
    """Effect-first candidate discovery followed by task-aware reranking."""

    dense_pool_width: int = 20
    anchor_width: int = 12
    expand_branch_count: int = 2
    branch_overfetch_width: int = 8
    branch_keep_width: int = 4
    probe_mode: str = "query_anchor"
    path_filter: bool = True
    use_answer_options: bool = True
    include_time_metadata: bool = True
    bridge_query_instruction: str = (
        "Instruct: Retrieve a distinct past personal interaction that complements an anchor "
        "for answering the current request\nQuery: "
    )
    final_rerank_instruction: str = (
        "Assess whether this recorded interaction provides evidence useful for answering the question about this "
        "user. Respect the time or period asked in the question. Earlier statements may be necessary for historical "
        "states or reasons for change. Distinguish user statements from assistant suggestions. Judge usefulness of "
        "the recorded evidence, not agreement with a guessed answer."
    )
    path_filter_instruction: str = (
        "Rank candidate interactions by whether they add distinct, independently usable personal evidence "
        "beyond the anchor for answering the current request. Avoid anchor paraphrases and merely topical "
        "passages."
    )

    def validate(self, retrieval: RetrievalConfig) -> None:
        positive = {
            "dense_pool_width": self.dense_pool_width,
            "anchor_width": self.anchor_width,
            "expand_branch_count": self.expand_branch_count,
            "branch_overfetch_width": self.branch_overfetch_width,
            "branch_keep_width": self.branch_keep_width,
        }
        for name, value in positive.items():
            _strict_int(value, name, positive=True)
        for name in ("path_filter", "use_answer_options", "include_time_metadata"):
            _strict_bool(getattr(self, name), f"bridge_rerank.{name}")
        for name in (
            "probe_mode",
            "bridge_query_instruction",
            "final_rerank_instruction",
            "path_filter_instruction",
        ):
            _strict_string(getattr(self, name), f"bridge_rerank.{name}", allow_empty=name == "probe_mode")
        if retrieval.context_size > self.dense_pool_width:
            raise ValueError("retrieval.context_size must be <= bridge_rerank.dense_pool_width")
        if self.anchor_width > self.dense_pool_width:
            raise ValueError("bridge_rerank.anchor_width must be <= bridge_rerank.dense_pool_width")
        if self.branch_keep_width > self.branch_overfetch_width:
            raise ValueError("bridge_rerank.branch_keep_width must be <= bridge_rerank.branch_overfetch_width")
        if self.expand_branch_count > retrieval.cluster_count:
            raise ValueError("bridge_rerank.expand_branch_count must be <= retrieval.cluster_count")
        if self.probe_mode not in {"centroid", "query_anchor"}:
            raise ValueError("bridge_rerank.probe_mode must be centroid or query_anchor")
        for name in ("bridge_query_instruction", "final_rerank_instruction", "path_filter_instruction"):
            if not getattr(self, name).strip():
                raise ValueError(f"bridge_rerank.{name} cannot be empty")


@dataclass(frozen=True)
class ChainConfig:
    """Resource limits for the historical joint-evidence Chain runner.

    The conditional-activation runner has a separate configuration surface in
    :mod:`bridgetree.dependency_config`.  Keeping these fields on
    :class:`AppConfig` is solely a compatibility contract for the legacy
    ``chain-*`` methods and does not route the new entry point back through the
    old search implementation.
    """

    horizon: int = 2
    initial_width: int = 12
    proposal_width: int = 4
    max_unique_memories: int = 64
    max_proposal_calls: int = 36
    max_joint_contexts: int = 512
    max_claim_calls: int = 32
    max_verify_calls: int = 512

    def __post_init__(self) -> None:
        horizon = _strict_int(self.horizon, "chain.horizon", nonnegative=True)
        if horizon > 2:
            raise ValueError("chain.horizon must be 0, 1, or 2")
        object.__setattr__(self, "horizon", horizon)
        for name in (
            "initial_width",
            "proposal_width",
            "max_unique_memories",
            "max_joint_contexts",
        ):
            object.__setattr__(
                self,
                name,
                _strict_int(getattr(self, name), f"chain.{name}", positive=True),
            )
        for name in ("max_proposal_calls", "max_claim_calls", "max_verify_calls"):
            object.__setattr__(
                self,
                name,
                _strict_int(getattr(self, name), f"chain.{name}", nonnegative=True),
            )
        if self.initial_width > self.max_unique_memories:
            raise ValueError("chain.initial_width must be <= chain.max_unique_memories")


@dataclass(frozen=True)
class DataConfig:
    raw_dir: str = "data/raw/personamem-v1"
    processed_dir: str = "data/processed/personamem-v1"
    split: str = "32k"
    include_system_persona: bool = True
    memory_granularity: str = "user_assistant_pair"


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "cpu"
    cache_dir: str = "outputs/cache"
    output_dir: str = "outputs/runs"


@dataclass(frozen=True)
class AppConfig:
    seed: int
    retrieval: RetrievalConfig
    models: ModelsConfig
    bridge_rerank: BridgeRerankConfig = field(default_factory=BridgeRerankConfig)
    data: DataConfig = field(default_factory=DataConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    # Appended after the historical fields to preserve positional constructor
    # compatibility for external callers.
    chain: ChainConfig = field(default_factory=ChainConfig)

    def validate(self) -> None:
        seed = _strict_int(self.seed, "seed", nonnegative=True)
        if seed != self.seed:
            raise ValueError("seed must be canonicalized as an integer")
        self.retrieval.validate()
        self.bridge_rerank.validate(self.retrieval)
        self.chain.__post_init__()
        # Model dataclasses validate their own transport and numeric fields at
        # construction time; repeat the checks here for objects assembled by
        # deserializers or ``dataclasses.replace``.
        self.models.embedding.__post_init__()
        self.models.reranker.__post_init__()
        self.models.generator.__post_init__()
        score_space = (
            str(getattr(self.models.reranker, "score_space", "unit_interval"))
            .strip()
            .lower()
            .replace("-", "_")
        )
        if score_space not in {"unit_interval", "probability", "logit_difference", "logit", "logit_diff"}:
            raise ValueError("models.reranker.score_space must declare unit_interval or logit_difference")
        if getattr(self.models.reranker, "score_contract", "pointwise") not in {"pointwise", "listwise"}:
            raise ValueError("models.reranker.score_contract must be pointwise or listwise")
        if self.data.split not in {"32k", "128k", "1M"}:
            raise ValueError("data.split must be one of 32k, 128k, 1M")
        if self.data.memory_granularity not in {"user_only", "user_assistant_pair"}:
            raise ValueError("data.memory_granularity must be user_only or user_assistant_pair")
        _strict_int(self.models.generator.max_tokens, "generator max_tokens", positive=True)
        _strict_int(self.models.generator.context_token_budget, "generator context_token_budget", positive=True)

    def resolved_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def config_hash(self) -> str:
        payload = json.dumps(self.resolved_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return loaded


def _retrieval_from_mapping(raw: Mapping[str, Any]) -> RetrievalConfig:
    values = dict(raw)
    if "first_hop_width" in values:
        if "initial_width" in values and values["initial_width"] != values["first_hop_width"]:
            raise ValueError("retrieval.initial_width and first_hop_width disagree")
        values["initial_width"] = values.pop("first_hop_width")
    return RetrievalConfig(**values)


def _read_config_tree(path: Path, seen: tuple[Path, ...] = ()) -> Dict[str, Any]:
    """Resolve ``base_config`` recursively relative to the declaring file."""

    resolved = path.expanduser().resolve()
    if resolved in seen:
        chain = " -> ".join(str(item) for item in (*seen, resolved))
        raise ValueError(f"cyclic base_config chain: {chain}")
    raw = _read_yaml(resolved)
    base_value = raw.pop("base_config", None)
    if base_value is None:
        return raw
    if not isinstance(base_value, str) or not base_value.strip():
        raise ValueError("base_config must be a non-empty path string")
    base_path = Path(base_value).expanduser()
    if not base_path.is_absolute():
        base_path = resolved.parent / base_path
    return _deep_merge(_read_config_tree(base_path, (*seen, resolved)), raw)


def load_config(path: str | Path, override_path: str | Path | None = None) -> AppConfig:
    config_path = Path(path)
    raw = _read_config_tree(config_path)
    if override_path is not None:
        raw = _deep_merge(raw, _read_config_tree(Path(override_path)))

    retrieval_raw = raw.get("retrieval", {})
    if not isinstance(retrieval_raw, Mapping):
        raise ValueError("configuration retrieval section must be a mapping")
    retrieval = _retrieval_from_mapping(retrieval_raw)
    models_raw = raw.get("models", {})
    if not isinstance(models_raw, Mapping):
        raise ValueError("configuration models section must be a mapping")
    embedding_raw = models_raw.get("embedding", {})
    reranker_raw = models_raw.get("reranker", {})
    generator_raw = models_raw.get("generator", {})
    for name, section in (("embedding", embedding_raw), ("reranker", reranker_raw), ("generator", generator_raw)):
        if not isinstance(section, Mapping):
            raise ValueError(f"configuration models.{name} section must be a mapping")
    credentials_path = config_path.resolve().parent / "credentials.local.yaml"
    if not generator_raw.get("api_key") and credentials_path.is_file():
        private_generator = _read_yaml(credentials_path).get("generator", {})
        if not isinstance(private_generator, Mapping):
            raise ValueError("private credentials generator section must be a mapping")
        # Never attach an existing service's credential to an unrelated URL.
        if private_generator.get("endpoint") == generator_raw.get("endpoint"):
            generator_raw = {**generator_raw, "api_key": private_generator.get("api_key", "")}
    bridge_raw = raw.get("bridge_rerank", {})
    chain_raw = raw.get("chain", {})
    data_raw = raw.get("data", {})
    runtime_raw = raw.get("runtime", {})
    for name, section in (
        ("bridge_rerank", bridge_raw),
        ("chain", chain_raw),
        ("data", data_raw),
        ("runtime", runtime_raw),
    ):
        if not isinstance(section, Mapping):
            raise ValueError(f"configuration {name} section must be a mapping")
    embedding = EmbeddingConfig(**embedding_raw)
    reranker = RerankerConfig(**reranker_raw)
    generator = GeneratorConfig(**generator_raw)
    # The historical background worker stored its reporting cadence beside
    # AppConfig's runtime paths.  Those values never affected retrieval and
    # the current dependency loader canonicalizes them into ``execution``.
    # Ignore them here so legacy overlays remain loadable without adding
    # irrelevant fields to AppConfig (and therefore to dependency identities).
    legacy_runtime_fields = {
        "log_every_questions",
        "evaluate_every_questions",
        "heartbeat_seconds",
    }
    unknown_runtime_fields = (
        set(runtime_raw) - set(RuntimeConfig.__dataclass_fields__) - legacy_runtime_fields
    )
    if unknown_runtime_fields:
        raise ValueError(
            "unknown runtime field(s): " + ", ".join(sorted(unknown_runtime_fields))
        )
    runtime_values = {
        key: value
        for key, value in runtime_raw.items()
        if key in RuntimeConfig.__dataclass_fields__
    }
    config = AppConfig(
        seed=_strict_int(raw.get("seed", 42), "seed", nonnegative=True),
        retrieval=retrieval,
        models=ModelsConfig(embedding=embedding, reranker=reranker, generator=generator),
        bridge_rerank=BridgeRerankConfig(**bridge_raw),
        chain=ChainConfig(**chain_raw),
        data=DataConfig(**data_raw),
        runtime=RuntimeConfig(**runtime_values),
    )
    config.validate()
    return config


def apply_runtime_overrides(config: AppConfig, overrides: Mapping[str, Any]) -> AppConfig:
    """Apply explicit CLI/script values after YAML resolution."""
    retrieval_fields = set(RetrievalConfig.__dataclass_fields__)
    retrieval_values = {key: value for key, value in overrides.items() if key in retrieval_fields and value is not None}
    current = config
    if retrieval_values:
        current = replace(current, retrieval=replace(current.retrieval, **retrieval_values))
    bridge_fields = set(BridgeRerankConfig.__dataclass_fields__)
    bridge_values = {key: value for key, value in overrides.items() if key in bridge_fields and value is not None}
    if bridge_values:
        current = replace(current, bridge_rerank=replace(current.bridge_rerank, **bridge_values))
    if overrides.get("seed") is not None:
        current = replace(current, seed=_strict_int(overrides["seed"], "seed", nonnegative=True))
    data_values = {
        key: overrides[key]
        for key in ("memory_granularity", "include_system_persona")
        if overrides.get(key) is not None
    }
    if data_values:
        current = replace(current, data=replace(current.data, **data_values))
    current.validate()
    return current
