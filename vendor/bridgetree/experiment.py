from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from copy import copy
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .baselines import BaselineResult, cluster_prf, dense_retrieval, rfmem, rfmem_recollection
from .budget import CostTracker, SearchBudget
from .clients import (
    Embedder,
    GenerationCache,
    GeneratorClient,
    RerankerClient,
    StateEmbeddingCache,
    build_context_plan,
    build_generation_messages,
    generation_prompt_hash,
)
from .config import AppConfig, RetrievalConfig
from .guided_retriever import RerankerGuidedBridgeRetriever, cached_rerank_all
from .index import ExactInnerProductIndex, build_index
from .information import StateBasisProvider
from .math_utils import nonnegative_cosine
from .measure import propagate_frozen_graph
from .metrics import (
    answer_accuracy,
    answer_parse_failed,
    bridge_recall_at_k,
    direct_ranks,
    gain_damage_net,
    path_objective_advantage,
    persona_macro_accuracy,
    recall_at_k,
)
from .personamem import (
    PERSONAMEM_REVISION,
    PersonaMemExample,
    iter_examples,
    messages_to_memories,
    parse_options,
)
from .ranking import RerankCache, build_personamem_rank_query, format_memory_document, stable_union
from .retriever import BridgeTreeRetriever
from .semantic import (
    SEMANTIC_QUERY_CONDITIONED_INSTRUCTION,
    build_query_conditioned_representation_text,
    discover_frozen_graph,
    legacy_trace_from_result,
    reranker_quality_records,
    rho_squared_quality_records,
    semantic_retrieve,
)
from .temporal import TransitionCache
from .types import FrozenProposalGraph, Memory, QualityRecord, RetrievalResult


def _public_service_identity(value: Any) -> dict[str, Any]:
    """Return non-sensitive model provenance suitable for run artifacts."""

    if hasattr(value, "__dataclass_fields__"):
        raw = asdict(value)
    elif isinstance(value, Mapping):
        raw = dict(value)
    else:
        raw = {"model": str(value)}
    result: dict[str, Any] = {}
    for key, item in raw.items():
        normalized = str(key).lower()
        if normalized in {"api_key", "api_key_env", "authorization", "token", "password", "secret"}:
            continue
        if normalized == "endpoint":
            # Keep a stable identity for cache/audit comparisons without
            # persisting an internal host or path.
            result["endpoint_sha256"] = hashlib.sha256(str(item).encode("utf-8")).hexdigest()
        else:
            result[str(key)] = item
    return result


def _public_app_config(config: AppConfig) -> dict[str, Any]:
    """Redact credentials and transport endpoints recursively."""

    def redact(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                str(key): redact(item)
                for key, item in value.items()
                if str(key).lower()
                not in {"api_key", "api_key_env", "authorization", "token", "password", "secret"}
                and str(key).lower() != "endpoint"
            }
        if isinstance(value, (list, tuple)):
            return [redact(item) for item in value]
        return value

    result = redact(config.resolved_dict())
    # Endpoint hashes are useful provenance and avoid a raw URL in the public
    # artifact.  Walk the service sections explicitly because the generic
    # redaction above omits the endpoint key.
    models = result.get("models", {}) if isinstance(result, Mapping) else {}
    if isinstance(models, Mapping):
        for section in ("embedding", "reranker", "generator"):
            original = getattr(config.models, section)
            public = models.get(section)
            if isinstance(public, Mapping):
                public["endpoint_sha256"] = hashlib.sha256(
                    str(getattr(original, "endpoint", "")).encode("utf-8")
                ).hexdigest()
    return result


def _read_examples(app_config: AppConfig) -> list[PersonaMemExample]:
    """Load the configured PersonaMem source for matrix runs."""
    root = Path(app_config.data.raw_dir)
    question_path = root / f"questions_{app_config.data.split}.csv"
    context_path = root / f"shared_contexts_{app_config.data.split}.jsonl"
    if not question_path.is_file() or not context_path.is_file():
        raise FileNotFoundError(
            "PersonaMem raw data is missing; run `bridgetree download-personamem` first"
        )
    return list(iter_examples(question_path, context_path))


def _protocol_select_examples(
    config: AppConfig,
    *,
    phase: str,
    examples: Sequence[PersonaMemExample] | None,
    protocol_manifest: str | Path | Mapping[str, Any] | None,
    synthetic: bool = False,
) -> tuple[str, list[PersonaMemExample]]:
    """Canonicalize a phase and enforce persisted role selection before models."""

    from .protocol import canonical_phase, protocol_examples, protocol_gate

    normalized_phase = canonical_phase(phase)
    rows = list(examples) if examples is not None else _read_examples(config)
    if protocol_manifest is None:
        # In-memory fixtures are allowed only through an explicitly marked
        # synthetic path.  Real PersonaMem runs may never interpret the whole
        # raw bank as development data.
        if not synthetic:
            raise ValueError(
                f"phase {phase} requires a persisted protocol manifest for PersonaMem; "
                "use synthetic=True only for an explicit synthetic smoke"
            )
    else:
        protocol_gate(
            normalized_phase,
            manifest=protocol_manifest if isinstance(protocol_manifest, Mapping) else None,
            manifest_path=protocol_manifest if isinstance(protocol_manifest, (str, Path)) else None,
            config_hash=config.config_hash(),
            action="run",
        )
        rows = list(protocol_examples(protocol_manifest, rows, normalized_phase))
    return normalized_phase, rows


def _visible_memory_records(example: PersonaMemExample, memories: Sequence[Memory]) -> list[Memory]:
    """Apply the question visibility envelope before any model/cache call."""

    from .semantic import _visible_records

    # ``_visible_records`` also validates alignment with vectors.  A tiny
    # placeholder column lets runners reuse the same cutoff semantics while
    # keeping future memory text out of the document-embedding request.
    placeholder = np.empty((len(memories), 1), dtype=np.float32)
    visible, _ = _visible_records(
        memories,
        placeholder,
        getattr(example, "query_time", None),
        query_metadata=getattr(example, "metadata", None) or None,
    )
    return visible


def _runtime_provenance(raw_dir: str | Path, split: str) -> dict[str, Any]:
    """Capture source/code/worktree identities without persisting secrets."""

    # Reuse the protocol implementation so standalone runs and confirmatory
    # manifests use exactly the same source/package hashing rules.
    from .protocol import _source_package_snapshot, _source_snapshot, _worktree_snapshot

    source_sha256, source_data_hash = _source_snapshot(raw_dir, split)
    worktree_clean, uncommitted_diff_hash = _worktree_snapshot()
    return {
        "source_sha256": source_sha256,
        "source_data_package_hash": source_data_hash,
        "source_package_hash": _source_package_snapshot(),
        "worktree_clean": worktree_clean,
        "uncommitted_diff_hash": uncommitted_diff_hash,
    }

METHODS = (
    "bridgetree",
    "semantic_path",
    "pure_rerank",
    "frozen_listwise",
    "dense",
    "dense_rerank",
    "dense_rerank_20",
    "dense_rerank_28",
    "bridgetree_union_rerank",
    "bridgetree_guided_rerank",
    "bridgetree_guided_pathfilter",
    "full_pool_rerank",
    "rfmem_familiarity",
    "rfmem_recollection",
    "rfmem",
    "cluster_prf",
    "ablation_no_cluster",
    "ablation_bfs",
    "ablation_fixed_depth",
    "ablation_topk",
    "ablation_rho_dpp",
    "ablation_direct_path",
)

SEMANTIC_METHODS = {"semantic_path", "pure_rerank", "frozen_listwise"}

# Fixed-candidate semantic ablation matrix.  Every row is evaluated on the
# same frozen proposal graph and original memory pool.  L0/L1 deliberately
# use a separate, explicit legacy ``rho²`` quality source; the S rows share
# one frozen pointwise scorer table.  Keeping the source in the row metadata
# prevents an audit from pretending that all seven rows use one quality
# contract.
SEMANTIC_ABLATIONS: Dict[str, Dict[str, Any]] = {
    "L0": {
        "label": "Legacy-Rho2-Flat",
        "quality_source": "rho2",
        "path_mode": "none",
        "feature_mode": "cached_memory",
        "representation_mode": "cached_memory",
        "selection_mode": "semantic_path_logdet",
        "quality_mode": "rho",
        "quality_score_space": "unit_interval",
        "score_contract": "unit_interval",
        "scorer_fingerprint": "legacy-rho2",
        "representation_role": "memory",
    },
    "L1": {
        "label": "Legacy-Rho2-SinglePath",
        "quality_source": "rho2",
        "path_mode": "single_path",
        "feature_mode": "cached_memory",
        "representation_mode": "cached_memory",
        "selection_mode": "semantic_path_logdet",
        "quality_mode": "rho",
        "quality_score_space": "unit_interval",
        "score_contract": "unit_interval",
        "scorer_fingerprint": "legacy-rho2",
        "representation_role": "memory",
    },
    "S0": {
        "label": "Pure-Rerank",
        "quality_source": "pointwise",
        "path_mode": "none",
        "feature_mode": "cached_memory",
        "representation_role": "not_used",
        "selection_mode": "pure_rerank",
    },
    "S1": {
        "label": "Flat-LogDet",
        "quality_source": "pointwise",
        "path_mode": "none",
        "feature_mode": "cached_memory",
        "representation_role": "memory",
        "selection_mode": "semantic_path_logdet",
    },
    "S2": {
        "label": "Semantic-Path",
        "quality_source": "pointwise",
        "path_mode": "posterior_expected_scatter",
        "feature_mode": "cached_memory",
        "representation_role": "memory",
        "selection_mode": "semantic_path_logdet",
    },
    "S3": {
        "label": "Semantic-Path-QueryConditioned",
        "quality_source": "pointwise",
        "path_mode": "posterior_expected_scatter",
        "feature_mode": "query_conditioned",
        "representation_mode": "query_conditioned",
        "representation_role": "query_conditioned",
        "selection_mode": "semantic_path_logdet",
    },
    "S2-shuffle": {
        "label": "Semantic-Path-Shuffled",
        "quality_source": "pointwise",
        "path_mode": "shuffle",
        "feature_mode": "cached_memory",
        "representation_role": "memory",
        "selection_mode": "semantic_path_logdet",
    },
}

SEMANTIC_MATRIX_ARCHITECTURES = tuple(SEMANTIC_ABLATIONS)


def semantic_matrix_configs(base: Any) -> Dict[str, RetrievalConfig]:
    """Return the train-free L0/L1/S0/S1/S2/S3/S2-shuffle configurations."""
    retrieval = getattr(base, "retrieval", base)
    resolved: Dict[str, RetrievalConfig] = {}
    for label, changes in SEMANTIC_ABLATIONS.items():
        # Row labels and quality-source metadata are recorded in the matrix
        # manifest but are not RetrievalConfig constructor fields.
        values = {
            key: value
            for key, value in changes.items()
            if key in RetrievalConfig.__dataclass_fields__
        }
        defaults = {
            "profile": "semantic_path_v1",
            "proposal_mode": "real_member_query_anchor",
            "relation_mode": "angular",
            "quality_mode": "frozen_reranker",
        }
        defaults.update(values)
        candidate = replace(retrieval, **defaults)
        # S0's pure rerank is still a semantic quality comparison and keeps
        # the same frozen profile/contract; it simply does not use path
        # geometry.
        candidate.validate()
        resolved[label] = candidate
    return resolved

# The primary module matrix is intentionally explicit and small.  Auxiliary
# knobs (clustering/search order) stay shared across rows so a matrix run
# changes only the four formal operators.
TMIC_ABLATIONS: Dict[str, Dict[str, Any]] = {
    "A0": {
        "label": "R1-Path-Exhaustive",
        "temporal_measure": False,
        "measure_propagation": False,
        "state_information": False,
        "information_certificate": False,
        "feature_mode": "path_conditioned",
        "selection_mode": "path_logdet",
        "stop_mode": "budget",
    },
    "A1": {
        "label": "R1+T",
        "temporal_measure": True,
        "measure_propagation": False,
        "state_information": False,
        "information_certificate": False,
        "feature_mode": "path_conditioned",
        "selection_mode": "path_logdet",
        "stop_mode": "budget",
    },
    "A2": {
        "label": "R1+T+M",
        "temporal_measure": True,
        "measure_propagation": True,
        "state_information": False,
        "information_certificate": False,
        "feature_mode": "path_conditioned",
        "selection_mode": "path_logdet",
        "stop_mode": "budget",
    },
    "A3": {
        "label": "R1+T+M+I",
        "temporal_measure": True,
        "measure_propagation": True,
        "state_information": True,
        "information_certificate": False,
        "feature_mode": "path_conditioned",
        "selection_mode": "path_logdet",
        "stop_mode": "budget",
    },
    "A4": {
        "label": "Full-TMIC",
        "temporal_measure": True,
        "measure_propagation": True,
        "state_information": True,
        "information_certificate": True,
        "feature_mode": "path_conditioned",
        "selection_mode": "path_logdet",
        "stop_mode": "certificate_or_budget",
    },
}


def tmic_matrix_configs(base: Any) -> Dict[str, Any]:
    """Resolve the fixed A0--A4 configurations from one base config.

    ``base`` may be either a ``RetrievalConfig`` or an ``AppConfig``.  No
    search/tuning values are learned here; this helper is deliberately pure so
    confirmatory runs can hash the resulting matrix before execution.
    """
    retrieval = getattr(base, "retrieval", base)
    # TMIC A0--A4 is the historical operator matrix, not the semantic-path
    # S0--S3 matrix.  The shipped application YAML now selects
    # ``semantic_path_v1`` for the main runtime, so simply calling
    # ``dataclasses.replace`` on it would accidentally route every A row
    # through the semantic profile (and, in particular, make A4's exact
    # partition certificate unavailable).  Start from an explicit legacy
    # baseline and let the four TMIC switches below be the only architectural
    # changes.
    retrieval = replace(
        retrieval,
        profile="legacy_core",
        proposal_mode="legacy_first_arrival",
        relation_mode="cosine",
        quality_mode="direct_cosine",
        path_mode="legacy",
        certificate_domain="exact_partition",
        certificate_mode="off",
        representation_mode="cached_memory",
    )
    resolved: Dict[str, Any] = {}
    for name, changes in TMIC_ABLATIONS.items():
        values = {key: value for key, value in changes.items() if key != "label"}
        candidate = replace(retrieval, **values)
        candidate.validate()
        resolved[name] = candidate
    return resolved

RERANK_METHODS = {
    "dense_rerank",
    "dense_rerank_20",
    "dense_rerank_28",
    "bridgetree_union_rerank",
    "bridgetree_guided_rerank",
    "bridgetree_guided_pathfilter",
    "full_pool_rerank",
}

GUIDED_METHODS = {"bridgetree_guided_rerank", "bridgetree_guided_pathfilter"}
BRIDGE_RERANK_METHODS = GUIDED_METHODS | {"bridgetree_union_rerank"}
EFFECT_FIRST_METHODS = RERANK_METHODS - {"dense_rerank"}


def _state_option_embeddings(
    example: PersonaMemExample,
    embedding_cache: EmbeddingCache | None,
    state_cache: StateEmbeddingCache | None,
    *,
    query_text: str | None = None,
    endpoint: str = "",
    model: str = "",
) -> tuple[np.ndarray | None, bool]:
    """Encode answer options once for I=true without touching labels.

    The option strings only define a frozen coordinate basis; they are never
    compared with the gold answer.  A cache hit still returns a logical state
    embedding matrix, while the boolean lets callers account for physical
    work separately.
    """
    if embedding_cache is None:
        return None, False
    options = parse_options(example.all_options)
    if not options:
        return None, False
    options_hash = hashlib.sha256(
        json.dumps(options, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    # Include both the stable question identifier and the actual query text:
    # a hand-built development fixture may reuse an ID while changing the
    # wording, and the frozen state basis must never be silently shared in
    # that case.
    query_hash = hashlib.sha256(
        json.dumps(
            {"question_id": str(example.question_id), "query": query_text if query_text is not None else example.query},
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    key = (
        state_cache.key_for(
            endpoint=endpoint,
            model=model,
            embedding_fingerprint=embedding_cache.fingerprint,
            query_hash=query_hash,
            ordered_path_ids=(),
            options_hash=options_hash,
        )
        if state_cache is not None
        else ""
    )
    if state_cache is None:
        return embedding_cache.encode_documents(options), False
    values, hit = state_cache.get_or_encode(key, lambda: embedding_cache.encode_documents(options))
    return values, hit


ABLATION_OPTIONS = {
    "bridgetree": {},
    "ablation_no_cluster": {"cluster_mode": "none"},
    "ablation_bfs": {"search_order": "bfs"},
    "ablation_fixed_depth": {"max_depth": 3, "stop_mode": "budget"},
    "ablation_topk": {"selection_mode": "rho_topk", "stop_mode": "budget"},
    "ablation_rho_dpp": {
        "feature_mode": "rho",
        "selection_mode": "rho_logdet",
        "stop_mode": "budget",
    },
    "ablation_direct_path": {
        "feature_mode": "rho",
        "selection_mode": "rho_topk",
        "stop_mode": "budget",
    },
}


class EmbeddingCache:
    def __init__(
        self,
        root: str | Path,
        embedder: Embedder,
        model_name: str,
        embedding_identity: Mapping[str, Any] | None = None,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.model_name = model_name
        self.embedding_identity = dict(embedding_identity or {"model": model_name})

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.embedding_identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cached(self, texts: Sequence[str], purpose: str, encode, instruction: str | None = None) -> np.ndarray:
        payload = json.dumps(
            {
                "embedding": self.embedding_identity,
                "instruction": instruction,
                "purpose": purpose,
                "texts": list(texts),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        key = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        path = self.root / f"{key}.npy"
        if path.exists():
            value = np.load(path, allow_pickle=False)
            if len(value) == len(texts):
                return np.asarray(value, dtype=np.float32)
        value = np.asarray(encode(), dtype=np.float32)
        np.save(path, value, allow_pickle=False)
        return value

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._cached(texts, "document", lambda: self.embedder.encode(texts))

    def encode_query(
        self,
        text: str,
        instruction: str | None = None,
        purpose: str = "query",
    ) -> np.ndarray:
        if instruction is not None:
            return self.encode_queries([text], instruction=instruction, purpose=purpose)[0]
        method = getattr(self.embedder, "encode_query", None)
        encode = (lambda: np.asarray([method(text)])) if callable(method) else (lambda: self.embedder.encode([text]))
        return self._cached([text], purpose, encode, instruction=None)[0]

    def encode_queries(self, texts: Sequence[str], instruction: str, purpose: str) -> np.ndarray:
        method = getattr(self.embedder, "encode_queries", None)
        encode = (
            (lambda: method(texts, instruction=instruction))
            if callable(method)
            else (lambda: self.embedder.encode([instruction + text for text in texts]))
        )
        return self._cached(texts, purpose, encode, instruction=instruction)


class IndexCache:
    """Reuse an index for an identical context cut and embedding fingerprint."""

    def __init__(self):
        self._indexes: Dict[str, ExactInnerProductIndex] = {}

    def get(
        self,
        context_key: str,
        embedding_fingerprint: str,
        backend: str,
        ids: Sequence[str],
        vectors: np.ndarray,
        exclusion_margin: int,
    ) -> tuple[ExactInnerProductIndex, float, bool]:
        vector_fingerprint = hashlib.sha256(np.asarray(vectors, dtype=np.float32).tobytes()).hexdigest()
        payload = json.dumps(
            {
                "context": context_key,
                "embedding": embedding_fingerprint,
                "backend": backend,
                "vectors": vector_fingerprint,
                "exclusion_margin": exclusion_margin,
            },
            sort_keys=True,
        )
        key = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        if key in self._indexes:
            cached = self._indexes[key]
            if cached.ids == list(ids):
                return cached, 0.0, True
            aliased = copy(cached)
            aliased.ids = list(ids)
            aliased.position = {memory_id: index for index, memory_id in enumerate(ids)}
            return aliased, 0.0, True
        started = time.perf_counter()
        index = build_index(backend, ids, vectors, exclusion_margin=exclusion_margin)
        build_ms = (time.perf_counter() - started) * 1000.0
        self._indexes[key] = index
        return index, build_ms, False


def _selected_memories(ids: Iterable[str], memory_by_id: Mapping[str, Memory], chronological: bool) -> List[Memory]:
    selected = [memory_by_id[memory_id] for memory_id in ids]
    if chronological:
        selected.sort(key=lambda memory: (memory.timestamp, memory.memory_id))
    return selected


def _context_hash(query: str, memories: Sequence[Memory], answer_options: str = "") -> str:
    """Hash the exact ordered prompt context sent to a generator.

    Using the shared message builder keeps this provenance hash in lockstep
    with :class:`GeneratorClient` and :class:`GenerationCache`; provenance
    changes in source/time headers or answer options therefore cannot be
    mistaken for the same context.
    """
    payload = {
        "prompt_hash": generation_prompt_hash(),
        "messages": build_generation_messages(query, memories, answer_options),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _exact_context_plan(
    config: AppConfig,
    example: PersonaMemExample,
    selected_ids: Sequence[str],
    selected: Sequence[Memory],
):
    """Freeze the reader request for every retrieval method.

    Retrieval is cardinality-based: the selector's complete ID sequence is
    preserved and the reader receives those same memories in chronological
    order.  A token budget is a protocol constraint, not a post-selection
    filter.  ``build_context_plan`` therefore raises on an over-budget
    selection under the strict configuration, leaving the evaluation row as
    an explicit failure instead of changing the selected set.
    """

    return build_context_plan(
        example.query,
        selected,
        example.all_options,
        token_budget=config.models.generator.context_token_budget,
        strict=config.retrieval.context_strict,
        selected_ids=selected_ids,
        generator_config=config.models.generator,
    )


def _refresh_rerank_selection_diagnostics(
    diagnostics: Dict[str, Any],
    selected_ids: Sequence[str],
) -> None:
    """Make selection diagnostics describe the token-budget-retained context."""
    if "selected_source_by_id" not in diagnostics:
        return
    bridge_ids = set(diagnostics.get("bridge_kept_ids", ()))
    source_by_id = diagnostics.get("selected_source_by_id", {})
    diagnostics["selected_source_by_id"] = {
        memory_id: source_by_id.get(memory_id, "bridge" if memory_id in bridge_ids else "dense")
        for memory_id in selected_ids
    }
    selected_bridge_count = sum(memory_id in bridge_ids for memory_id in selected_ids)
    diagnostics["selected_bridge_count"] = selected_bridge_count
    diagnostics["selected_bridge_rate"] = selected_bridge_count / len(selected_ids) if selected_ids else 0.0
    dense_top_ids = diagnostics.get("dense_rerank_top_ids")
    if dense_top_ids is not None:
        dense_top = set(dense_top_ids)
        diagnostics["dense_rerank_top5_retention"] = (
            sum(memory_id in dense_top for memory_id in selected_ids) / len(dense_top) if dense_top else 0.0
        )


def retrieve_method(
    method: str,
    config: AppConfig,
    example: PersonaMemExample,
    memories: Sequence[Memory],
    query_vector: np.ndarray,
    memory_vectors: np.ndarray,
    reranker: RerankerClient | None = None,
    index: ExactInnerProductIndex | None = None,
    budget: SearchBudget | None = None,
    cost_tracker: CostTracker | None = None,
    index_build_ms: float = 0.0,
    embedding_cache: EmbeddingCache | None = None,
    rerank_cache: RerankCache | None = None,
    transition_cache: TransitionCache | None = None,
    option_embeddings: np.ndarray | None = None,
    state_embedding_cache: StateEmbeddingCache | None = None,
    quality_provider: Any | None = None,
    quality_records: Mapping[str, Any] | None = None,
    representation_provider: Any | None = None,
    listwise_selector: Any | None = None,
    proposal_query_provider: Any | None = None,
    bridge_query_instruction: str | None = None,
) -> Tuple[List[str], List[Memory], Dict[str, Any], RetrievalResult | None]:
    if method not in METHODS:
        raise ValueError(f"unknown method {method}; choose from {METHODS}")
    # Non-tree baselines do not pass through ``BridgeTreeRetriever`` and
    # therefore need the same cutoff gate here.  Keep the ANN index only when
    # its ID domain still exactly matches the visible bank.
    query_cutoff = getattr(example, "query_time", None)
    query_metadata = getattr(example, "metadata", None) or None
    if query_cutoff is not None or query_metadata is not None:
        from .semantic import _visible_records

        visible_memories, visible_vectors = _visible_records(
            memories,
            memory_vectors,
            query_cutoff,
            query_metadata=query_metadata,
        )
        visible_ids = [str(memory.memory_id) for memory in visible_memories]
        index_ids = list(getattr(index, "ids", ())) if index is not None else []
        if index is not None and index_ids != visible_ids:
            index = None
            index_build_ms = 0.0
        memories = visible_memories
        memory_vectors = visible_vectors
    ids = [memory.memory_id for memory in memories]
    memory_by_id = {memory.memory_id: memory for memory in memories}
    k = config.retrieval.context_size
    current_budget = budget or SearchBudget.from_config(config.retrieval)
    semantic_requested = (
        method in SEMANTIC_METHODS
        or getattr(config.retrieval, "profile", "legacy_core") in {"semantic", "semantic_path_v1"}
        and method == "bridgetree"
    )
    if semantic_requested:
        # The semantic proposal query is one shared contract across the
        # matrix, the compatibility wrapper, and BridgeTreeRetriever.  When
        # the caller did not override it, reuse the already-cached embedder
        # rather than silently constructing a q+anchor approximation.
        if proposal_query_provider is None and embedding_cache is not None:
            proposal_query_provider = embedding_cache
        if bridge_query_instruction is None:
            bridge_query_instruction = getattr(config.bridge_rerank, "bridge_query_instruction", "")
        provider_dimension_mismatch = False
        if proposal_query_provider is embedding_cache and embedding_cache is not None:
            try:
                probe = np.asarray(embedding_cache.encode_query(example.query)).reshape(-1)
                provider_dimension_mismatch = probe.shape != np.asarray(query_vector).reshape(-1).shape
            except Exception:
                provider_dimension_mismatch = False
        if method == "frozen_listwise" and listwise_selector is None:
            # Keep the method routable in offline/legacy smoke runs while
            # making the absence explicit.  Real listwise experiments must
            # inject a provider through the public semantic API.
            def _offline_listwise_selector(_query: Any, records: Mapping[str, Any], _k: int = 0) -> list[str]:
                return sorted(str(identifier) for identifier in records)[: int(_k)]

            listwise_selector = _offline_listwise_selector
        semantic_config = config.retrieval
        if provider_dimension_mismatch:
            # Compatibility callers may inject precomputed vectors from a
            # different-dimensional local embedder.  Mark the explicit
            # deterministic adapter instead of pretending those vectors are a
            # real query-anchor service response.
            semantic_config = replace(semantic_config, proposal_mode="offline_q_plus_anchor")
        if method == "semantic_path":
            semantic_config = replace(
                semantic_config,
                profile="semantic_path_v1",
                proposal_mode="real_member_query_anchor",
                relation_mode="angular",
                path_mode="posterior_expected_scatter",
                selection_mode="semantic_path_logdet",
                quality_mode="frozen_reranker" if reranker is not None else semantic_config.quality_mode,
            )
        elif method == "pure_rerank":
            semantic_config = replace(
                semantic_config,
                profile="semantic_path_v1",
                proposal_mode="real_member_query_anchor",
                relation_mode="angular",
                selection_mode="pure_rerank",
                quality_mode="frozen_reranker" if reranker is not None else semantic_config.quality_mode,
            )
        elif method == "frozen_listwise":
            semantic_config = replace(
                semantic_config,
                profile="semantic_path_v1",
                proposal_mode="real_member_query_anchor",
                relation_mode="angular",
                selection_mode="frozen_listwise",
                quality_mode="frozen_reranker" if reranker is not None else semantic_config.quality_mode,
            )
        if provider_dimension_mismatch:
            semantic_config = replace(semantic_config, proposal_mode="offline_q_plus_anchor")
        # ``retrieve_method`` is also the compatibility entry point used by
        # the historical offline matrix tests.  Those callers may request a
        # named semantic method without injecting a remote reranker.  Keep
        # the strict ``semantic_retrieve`` contract (which fails loudly for a
        # declared frozen_reranker with no provider), but make this wrapper's
        # explicit no-service path use the deterministic direct cosine
        # quality adapter.  A supplied quality provider/record table remains
        # authoritative and is never silently replaced.
        if (
            reranker is None
            and quality_provider is None
            and quality_records is None
            and semantic_config.quality_mode == "frozen_reranker"
        ):
            semantic_config = replace(
                semantic_config,
                # ``direct_cosine`` is the semantic constructor's implicit
                # default and is upgraded to ``frozen_reranker`` whenever a
                # semantic profile is instantiated.  Use the explicit
                # ``constant`` compatibility mode here so that ``replace``
                # cannot upgrade it again; ``semantic_retrieve`` still uses
                # its deterministic cosine fallback when no provider is
                # supplied.
                quality_mode="constant",
                score_contract="unit_interval",
            )
        semantic_config.validate()
        tracker = cost_tracker or CostTracker(current_budget)
        result = semantic_retrieve(
            example.query,
            query_vector,
            memories,
            memory_vectors,
            semantic_config,
            index=index,
            budget=current_budget,
            cost_tracker=tracker,
            index_build_ms=index_build_ms,
            quality_provider=quality_provider,
            quality_records=quality_records,
            reranker=reranker,
            representation_provider=representation_provider,
            answer_options=example.all_options,
            query_cutoff=getattr(example, "query_time", None),
            query_metadata=getattr(example, "metadata", None),
            listwise_selector=listwise_selector,
            proposal_query_provider=proposal_query_provider,
            proposal_query_instruction=bridge_query_instruction or "",
            context_token_budget=config.models.generator.context_token_budget,
            generator_config=config.models.generator,
        )
        diagnostics = result.to_dict(include_text=False)
        diagnostics["_cost_tracker"] = tracker
        diagnostics["semantic_method"] = method
        return result.selected_in_greedy_order, result.selected, diagnostics, result
    if method in RERANK_METHODS:
        if not config.models.reranker.endpoint:
            raise ValueError(f"{method} requires a configured reranker endpoint")
        if method in BRIDGE_RERANK_METHODS and config.retrieval.stop_mode == "certificate_or_budget":
            raise ValueError(f"{method} cannot use certificate_or_budget")
        if method in {"dense_rerank", "dense_rerank_20"}:
            required_nodes = config.bridge_rerank.dense_pool_width
        elif method == "dense_rerank_28":
            required_nodes = config.bridge_rerank.dense_pool_width + (
                config.bridge_rerank.expand_branch_count * config.bridge_rerank.branch_keep_width
            )
        elif method == "bridgetree_union_rerank":
            required_nodes = config.bridge_rerank.dense_pool_width + max(
                0,
                config.retrieval.search_budget - config.retrieval.initial_width,
            )
        elif method in GUIDED_METHODS:
            required_nodes = config.bridge_rerank.dense_pool_width + (
                config.bridge_rerank.expand_branch_count * config.bridge_rerank.branch_overfetch_width
            )
        else:
            required_nodes = len(memories)
        current_budget = SearchBudget(
            max_unique_nodes=max(1, min(len(memories), required_nodes)),
            max_ann_calls=current_budget.max_ann_calls,
            max_candidate_exposure=current_budget.max_candidate_exposure,
        )
        if cost_tracker is not None and cost_tracker.cost_unique_count == 0 and cost_tracker.ann_calls_core == 0:
            cost_tracker.budget = current_budget
    tracker = cost_tracker or CostTracker(current_budget)
    tracker.index_build_ms += index_build_ms
    if index is None:
        index_started = time.perf_counter()
        index = build_index(
            config.retrieval.index_backend,
            ids,
            memory_vectors,
            exclusion_margin=config.retrieval.faiss_exclusion_margin,
        )
        tracker.index_build_ms += (time.perf_counter() - index_started) * 1000.0

    if config.retrieval.state_information:
        embedding_fingerprint = getattr(embedding_cache, "fingerprint", "")
        state_basis_provider = StateBasisProvider(
            config.retrieval.state_basis_mode,
            model_fingerprint=embedding_fingerprint,
            endpoint=config.models.embedding.endpoint,
        )
        if option_embeddings is None:
            option_embeddings, state_cache_hit = _state_option_embeddings(
                example,
                embedding_cache,
                state_embedding_cache,
                endpoint=config.models.embedding.endpoint,
                model=config.models.embedding.model,
            )
            if option_embeddings is not None:
                tracker.record_state_embedding(len(parse_options(example.all_options)), 0.0)
                if state_cache_hit:
                    tracker.record_cache_hit()
    else:
        state_basis_provider = None

    if method in ABLATION_OPTIONS:
        retrieval_config = replace(config.retrieval, **ABLATION_OPTIONS[method])
        retrieval_config.validate()
        bridge_result = BridgeTreeRetriever(retrieval_config).retrieve(
            example.query,
            query_vector,
            memories,
            memory_vectors,
            index=index,
            budget=current_budget,
            cost_tracker=tracker,
            answer_options=example.all_options,
            query_cutoff=getattr(example, "query_time", None),
            query_metadata=getattr(example, "metadata", None),
            transition_cache=transition_cache,
            option_embeddings=option_embeddings,
            state_basis_provider=state_basis_provider,
            proposal_query_provider=proposal_query_provider,
            bridge_query_instruction=bridge_query_instruction,
        )
        diagnostics = bridge_result.to_dict(include_text=False)
        diagnostics["path_objective_advantage"] = path_objective_advantage(bridge_result, k)
        diagnostics["_cost_tracker"] = tracker
        return (
            bridge_result.selected_in_greedy_order,
            bridge_result.selected,
            diagnostics,
            bridge_result,
        )

    if method == "dense":
        baseline = dense_retrieval(
            ids,
            memory_vectors,
            query_vector,
            k,
            index=index,
            budget=current_budget,
            cost_tracker=tracker,
        )
    elif method in {"dense_rerank", "dense_rerank_20", "dense_rerank_28"}:
        if reranker is None:
            raise ValueError(f"{method} requires a reranker client")
        initial_k = config.bridge_rerank.dense_pool_width
        if method == "dense_rerank_28":
            initial_k += config.bridge_rerank.expand_branch_count * config.bridge_rerank.branch_keep_width
        initial_k = min(len(memories), initial_k)
        initial = dense_retrieval(
            ids,
            memory_vectors,
            query_vector,
            initial_k,
            index=index,
            budget=current_budget,
            cost_tracker=tracker,
        )
        max_timestamp = max((memory.timestamp for memory in memories), default=0.0)
        rank_query = build_personamem_rank_query(
            example,
            instruction=config.bridge_rerank.final_rerank_instruction,
            use_answer_options=config.bridge_rerank.use_answer_options,
        )
        documents = [
            format_memory_document(
                memory_by_id[item],
                max_timestamp,
                include_time_metadata=config.bridge_rerank.include_time_metadata,
            )
            for item in initial.selected_ids
        ]
        ranking, cache_hit = cached_rerank_all(
            reranker,
            rerank_cache,
            rank_query,
            documents,
            tracker,
            records=[memory_by_id[memory_id] for memory_id in initial.selected_ids],
            cutoff=getattr(example, "query_time", None),
            query_metadata=getattr(example, "metadata", None),
            answer_options=example.all_options,
            include_time_metadata=config.bridge_rerank.include_time_metadata,
            score_contract=getattr(reranker, "score_contract", None),
            task_instruction=config.bridge_rerank.final_rerank_instruction,
        )
        items = ranking[:k]
        baseline = BaselineResult([initial.selected_ids[item.index] for item in items], tracker)
        scores = {initial.selected_ids[item.index]: item.score for item in ranking}
        baseline.diagnostics.update(
            {
                "dense_pool_ids": initial.selected_ids,
                "anchor_ids": [],
                "bridge_raw_ids": [],
                "bridge_kept_ids": [],
                "candidate_union_ids": initial.selected_ids,
                "selected_source_by_id": {memory_id: "dense" for memory_id in baseline.selected_ids},
                "selected_bridge_count": 0,
                "selected_bridge_rate": 0.0,
                "dense_rerank_top5_retention": 1.0,
                "bridge_candidate_novelty": 0.0,
                "dense_rerank_top_ids": [
                    initial.selected_ids[item.index] for item in ranking[:k]
                ],
                "final_rerank_scores": scores,
                "rerank_cache_hits": int(cache_hit),
            }
        )
    elif method == "bridgetree_union_rerank":
        if reranker is None:
            raise ValueError("bridgetree_union_rerank requires a reranker client")
        if config.retrieval.stop_mode == "certificate_or_budget":
            raise ValueError("bridgetree_union_rerank cannot use certificate_or_budget")
        dense_hits = tracker.search_core(
            index,
            query_vector,
            min(config.bridge_rerank.dense_pool_width, len(memories)),
        )
        dense_ids = [memory_id for memory_id, _score in dense_hits]
        bridge_result = BridgeTreeRetriever(config.retrieval).retrieve(
            example.query,
            query_vector,
            memories,
            memory_vectors,
            index=index,
            budget=tracker.budget,
            cost_tracker=tracker,
            initial_hits=dense_hits,
            excluded_candidate_ids=dense_ids,
            answer_options=example.all_options,
            query_cutoff=getattr(example, "query_time", None),
            query_metadata=getattr(example, "metadata", None),
            transition_cache=transition_cache,
            option_embeddings=option_embeddings,
            state_basis_provider=state_basis_provider,
            initial_hits_accounted=True,
        )
        discovered_ids = list(bridge_result.nodes)
        bridge_ids = [memory_id for memory_id in discovered_ids if memory_id not in set(dense_ids)]
        candidate_ids = stable_union(dense_ids, discovered_ids)
        max_timestamp = max((memory.timestamp for memory in memories), default=0.0)
        rank_query = build_personamem_rank_query(
            example,
            instruction=config.bridge_rerank.final_rerank_instruction,
            use_answer_options=config.bridge_rerank.use_answer_options,
        )
        documents = [
            format_memory_document(
                memory_by_id[memory_id],
                max_timestamp,
                include_time_metadata=config.bridge_rerank.include_time_metadata,
            )
            for memory_id in candidate_ids
        ]
        ranking, cache_hit = cached_rerank_all(
            reranker,
            rerank_cache,
            rank_query,
            documents,
            tracker,
            records=[memory_by_id[memory_id] for memory_id in candidate_ids],
            cutoff=getattr(example, "query_time", None),
            query_metadata=getattr(example, "metadata", None),
            answer_options=example.all_options,
            include_time_metadata=config.bridge_rerank.include_time_metadata,
            score_contract=getattr(reranker, "score_contract", None),
            task_instruction=config.bridge_rerank.final_rerank_instruction,
        )
        final_scores = {candidate_ids[item.index]: item.score for item in ranking}
        selected_ids = [candidate_ids[item.index] for item in ranking[:k]]
        selected = _selected_memories(selected_ids, memory_by_id, chronological=True)
        dense_ranked = [candidate_ids[item.index] for item in ranking if candidate_ids[item.index] in set(dense_ids)]
        dense_top5 = set(dense_ranked[:k])
        bridge_set = set(bridge_ids)
        selected_bridge_count = sum(memory_id in bridge_set for memory_id in selected_ids)
        parent_by_bridge = {
            memory_id: node.parent_id
            for memory_id, node in bridge_result.nodes.items()
            if memory_id in bridge_set and node.parent_id is not None
        }
        diagnostics = {
            "dense_pool_ids": dense_ids,
            "anchor_ids": [],
            "bridge_raw_ids": bridge_ids,
            "bridge_kept_ids": bridge_ids,
            "candidate_union_ids": candidate_ids,
            "parent_by_bridge_id": parent_by_bridge,
            "selected_source_by_id": {
                memory_id: "bridge" if memory_id in bridge_set else "dense" for memory_id in selected_ids
            },
            "selected_bridge_count": selected_bridge_count,
            "selected_bridge_rate": selected_bridge_count / len(selected_ids) if selected_ids else 0.0,
            "dense_rerank_top5_retention": (
                sum(memory_id in dense_top5 for memory_id in selected_ids) / len(dense_top5) if dense_top5 else 0.0
            ),
            "bridge_candidate_novelty": len(bridge_set) / len(bridge_ids) if bridge_ids else 0.0,
            "dense_rerank_top_ids": list(dense_top5),
            "final_rerank_scores": final_scores,
            "rerank_cache_hits": int(cache_hit),
            "legacy_internal_selected_ids": list(bridge_result.selected_in_greedy_order),
            "tree_diagnostic": bridge_result.diagnostic_summary(),
            "cost": tracker.snapshot().to_dict(),
            "_cost_tracker": tracker,
        }
        return selected_ids, selected, diagnostics, None
    elif method in GUIDED_METHODS:
        if reranker is None or embedding_cache is None:
            raise ValueError(f"{method} requires reranker and embedding caches")
        selected_ids, selected, pool = RerankerGuidedBridgeRetriever(config).retrieve(
            example,
            memories,
            query_vector,
            memory_vectors,
            embed_query_batch=embedding_cache.encode_queries,
            reranker=reranker,
            rerank_cache=rerank_cache,
            index=index,
            cost_tracker=tracker,
            mode=method,
        )
        return (
            selected_ids,
            selected,
            {**pool.diagnostics, "cost": tracker.snapshot().to_dict(), "_cost_tracker": tracker},
            None,
        )
    elif method == "full_pool_rerank":
        if reranker is None:
            raise ValueError("full_pool_rerank requires a reranker client")
        # Full-pool reranking is an explicitly diagnostic upper-bound method:
        # every document is exposed to the reranker even when the retrieval
        # budget used by efficiency baselines is smaller.  Record that actual
        # exposure instead of leaving the cost at zero; the opt-out is narrow
        # and visible in provenance so it cannot be mistaken for a matched
        # budget run.
        tracker.record_candidate_exposure(len(ids), enforce_budget=False)
        tracker.mark_visited(ids)
        max_timestamp = max((memory.timestamp for memory in memories), default=0.0)
        rank_query = build_personamem_rank_query(
            example,
            instruction=config.bridge_rerank.final_rerank_instruction,
            use_answer_options=config.bridge_rerank.use_answer_options,
        )
        documents = [
            format_memory_document(
                memory,
                max_timestamp,
                include_time_metadata=config.bridge_rerank.include_time_metadata,
            )
            for memory in memories
        ]
        ranking, cache_hit = cached_rerank_all(
            reranker,
            rerank_cache,
            rank_query,
            documents,
            tracker,
            records=list(memories),
            cutoff=getattr(example, "query_time", None),
            query_metadata=getattr(example, "metadata", None),
            answer_options=example.all_options,
            include_time_metadata=config.bridge_rerank.include_time_metadata,
            score_contract=getattr(reranker, "score_contract", None),
            task_instruction=config.bridge_rerank.final_rerank_instruction,
        )
        selected_ids = [ids[item.index] for item in ranking[:k]]
        selected = _selected_memories(selected_ids, memory_by_id, chronological=True)
        tracker.set_stop_reason("frontier_empty")
        diagnostics = {
            "dense_pool_ids": [],
            "anchor_ids": [],
            "bridge_raw_ids": [],
            "bridge_kept_ids": [],
            "candidate_union_ids": ids,
            "candidate_exposure_scope": "full_pool",
            "candidate_exposure_budget_exempt": True,
            "selected_source_by_id": {memory_id: "full_pool" for memory_id in selected_ids},
            "selected_bridge_count": 0,
            "selected_bridge_rate": 0.0,
            "dense_rerank_top5_retention": 0.0,
            "bridge_candidate_novelty": 0.0,
            "final_rerank_scores": {ids[item.index]: item.score for item in ranking},
            "rerank_cache_hits": int(cache_hit),
            "cost": tracker.snapshot().to_dict(),
            "_cost_tracker": tracker,
        }
        return selected_ids, selected, diagnostics, None
    elif method == "rfmem_familiarity":
        raw = dense_retrieval(
            ids,
            memory_vectors,
            query_vector,
            k,
            index=index,
            budget=current_budget,
            cost_tracker=tracker,
        )
        query_norm = query_vector / np.linalg.norm(query_vector)
        score_by_id = {memory_id: float(np.dot(index.vector(memory_id), query_norm)) for memory_id in raw.selected_ids}
        baseline = BaselineResult(
            [memory_id for memory_id in raw.selected_ids if score_by_id[memory_id] >= 0.3],
            tracker,
        )
    elif method == "rfmem_recollection":
        baseline = rfmem_recollection(
            ids,
            memory_vectors,
            query_vector,
            k,
            index=index,
            budget=current_budget,
            cost_tracker=tracker,
        )
    elif method == "rfmem":
        baseline = rfmem(
            ids,
            memory_vectors,
            query_vector,
            k,
            index=index,
            budget=current_budget,
            cost_tracker=tracker,
        )
    elif method == "cluster_prf":
        baseline = cluster_prf(
            ids,
            memory_vectors,
            query_vector,
            k,
            config.retrieval.initial_width,
            index=index,
            budget=current_budget,
            cost_tracker=tracker,
        )
    else:  # pragma: no cover - METHODS and branches stay synchronized
        raise AssertionError(method)

    selected = _selected_memories(baseline.selected_ids, memory_by_id, chronological=True)
    return (
        baseline.selected_ids,
        selected,
        {
            "ann_calls": baseline.ann_calls,
            "cost": baseline.cost.to_dict(),
            "_cost_tracker": tracker,
            **baseline.diagnostics,
        },
        None,
    )


def load_bridge_gold(path: str | Path | None) -> Dict[str, List[str]]:
    if path is None:
        return {}
    result: Dict[str, List[str]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            result[str(item["question_id"])] = [str(value) for value in item["gold_memory_ids"]]
    return result


def run_personamem_experiment(
    config: AppConfig,
    method: str,
    embedder: Embedder,
    limit: int | None = None,
    generate: bool = False,
    bridge_gold_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    run_label: str | None = None,
    transition_cache: TransitionCache | None = None,
    generation_cache: GenerationCache | None = None,
) -> Dict[str, Any]:
    """Run one method under a fully resolved, recorded evaluation protocol."""
    semantic_main_method = method == "bridgetree" and getattr(config.retrieval, "profile", "legacy_core") in {
        "semantic",
        "semantic_path_v1",
    }
    if (method in RERANK_METHODS or semantic_main_method) and not config.models.reranker.endpoint:
        raise ValueError(f"{method} requires a configured reranker endpoint")
    if method in BRIDGE_RERANK_METHODS and config.retrieval.stop_mode == "certificate_or_budget":
        raise ValueError(f"{method} cannot use certificate_or_budget")
    raw_root = Path(config.data.raw_dir)
    split = config.data.split
    question_path = raw_root / f"questions_{split}.csv"
    context_path = raw_root / f"shared_contexts_{split}.jsonl"
    if not question_path.exists() or not context_path.exists():
        raise FileNotFoundError("PersonaMem raw data is missing; run `bridgetree download-personamem` first")
    runtime_provenance = _runtime_provenance(raw_root, split)

    label = re.sub(r"[^a-zA-Z0-9_.-]+", "_", run_label or method).strip("._") or method
    run_root = Path(output_dir or config.runtime.output_dir) / f"{label}_{time.time_ns()}"
    run_root.mkdir(parents=True, exist_ok=False)
    cache = EmbeddingCache(
        config.runtime.cache_dir,
        embedder,
        config.models.embedding.model,
        asdict(config.models.embedding),
    )
    index_cache = IndexCache()
    state_embedding_cache = StateEmbeddingCache(Path(config.runtime.cache_dir) / "state")
    if transition_cache is None:
        transition_cache = TransitionCache(Path(config.runtime.cache_dir) / "transition")
    if generation_cache is None:
        generation_cache = GenerationCache(Path(config.runtime.cache_dir) / "generation")
    generator = GeneratorClient(config.models.generator) if generate else None
    shared_generation_cache = generation_cache
    semantic_main = method == "bridgetree" and getattr(config.retrieval, "profile", "legacy_core") in {
        "semantic",
        "semantic_path_v1",
    }
    reranker = RerankerClient(config.models.reranker) if (method in RERANK_METHODS or semantic_main) else None
    rerank_cache = (
        RerankCache(
            getattr(config.models.reranker, "cache_dir", "outputs/rerank_cache"),
            endpoint=config.models.reranker.endpoint,
            model=config.models.reranker.model,
            score_space=getattr(config.models.reranker, "score_space", "unit_interval"),
            task_instruction=getattr(config.models.reranker, "task_instruction", ""),
            score_contract=getattr(config.models.reranker, "score_contract", "pointwise"),
            model_fingerprint=getattr(config.models.reranker, "model_fingerprint", "")
            or getattr(config.models.reranker, "model", ""),
        )
        if method in RERANK_METHODS or semantic_main
        else None
    )
    bridge_gold = load_bridge_gold(bridge_gold_path)

    # The persisted configuration is a public reproducibility record.  Keep
    # credentials and raw service URLs out of it while retaining endpoint
    # hashes for identity/audit comparisons.
    resolved = {
        "app": _public_app_config(config),
        "execution": {
            "method": method,
            "limit": limit,
            "generate": generate,
            "bridge_gold_path": str(bridge_gold_path) if bridge_gold_path is not None else None,
            "run_label": label,
        },
    }
    resolved_payload = json.dumps(resolved, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    config_hash = hashlib.sha256(resolved_payload.encode("utf-8")).hexdigest()
    with (run_root / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "config_hash": config_hash,
                "app_config_hash": config.config_hash(),
                "config": resolved,
            },
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    repository_root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest: Dict[str, Any] = {
        "status": "running",
        "method": method,
        "run_label": label,
        "config_hash": config_hash,
        "git_commit": commit or None,
        "data_revision": PERSONAMEM_REVISION,
        "data_split": split,
        "embedding_model": _public_service_identity(config.models.embedding),
        "generator_model": _public_service_identity(config.models.generator),
        "prompt_hash": generation_prompt_hash(),
        "seed": config.seed,
        "generate": generate,
        "limit": limit,
        "retrieval_switches": {
            name: bool(getattr(config.retrieval, name))
            for name in (
                "temporal_measure",
                "measure_propagation",
                "state_information",
                "information_certificate",
            )
        },
        "cache_schema": {"embedding": 1, "transition": 1, "state": 1, "generation": 1},
        **runtime_provenance,
    }
    with (run_root / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    output_path = run_root / "predictions.jsonl"
    failure_path = run_root / "failures.jsonl"
    total = 0
    attempted = 0
    failures = 0
    accuracy_sum = 0.0
    parse_failure_sum = 0.0
    recall_sum = 0.0
    bridge_recall_sum = 0.0
    annotated = 0
    bridge_annotated = 0
    bridge_results: List[RetrievalResult] = []
    latencies: List[float] = []
    cost_records: List[Dict[str, Any]] = []
    outcome_records: List[Dict[str, Any]] = []
    with (
        output_path.open("w", encoding="utf-8") as output,
        failure_path.open("w", encoding="utf-8") as failure_output,
    ):
        for example in iter_examples(question_path, context_path):
            if limit is not None and attempted >= limit:
                break
            attempted += 1
            try:
                memories = messages_to_memories(
                    example.messages,
                    source_prefix=example.question_id,
                    include_system_persona=config.data.include_system_persona,
                    memory_granularity=config.data.memory_granularity,
                )
                memories = _visible_memory_records(example, memories)
                if not memories:
                    raise ValueError("no memories after the configured PersonaMem segmentation")
                query_vector = cache.encode_query(example.query)
                memory_vectors = cache.encode_documents([memory.text for memory in memories])
                context_key = (
                    f"{example.shared_context_id}:{example.end_index}:"
                    f"{config.data.memory_granularity}:{config.data.include_system_persona}"
                )
                index, index_build_ms, index_cache_hit = index_cache.get(
                    context_key,
                    cache.fingerprint,
                    config.retrieval.index_backend,
                    [memory.memory_id for memory in memories],
                    memory_vectors,
                    config.retrieval.faiss_exclusion_margin,
                )
                tracker = CostTracker(SearchBudget.from_config(config.retrieval))
                started = time.perf_counter()
                selected_ids, selected, diagnostics, bridge_result = retrieve_method(
                    method,
                    config,
                    example,
                    memories,
                    query_vector,
                    memory_vectors,
                    reranker=reranker,
                    index=index,
                    budget=tracker.budget,
                    cost_tracker=tracker,
                    index_build_ms=index_build_ms,
                    embedding_cache=cache,
                    rerank_cache=rerank_cache,
                    transition_cache=transition_cache,
                    state_embedding_cache=state_embedding_cache,
                )
                tracker = diagnostics.pop("_cost_tracker")
                semantic_result = (
                    bridge_result is not None
                    and bridge_result.first_arrival_semantics == "semantic_path_v1"
                )
                if semantic_result and (
                    bridge_result.context_plan is None or not bridge_result.context_plan.within_budget
                ):
                    # The semantic executor normally freezes its plan before
                    # returning. Keep that invariant explicit here, then
                    # rebuild the same canonical plan below so legacy and
                    # semantic methods share one evaluation contract.
                    raise ValueError("semantic retrieval did not produce an exact ContextPlan")
                # Freeze the exact request after every selector, including
                # dense/rerank and the historical BridgeTree path. This is a
                # hard cardinality boundary: an over-budget selection is a
                # failed protocol row, never a reason to drop scored IDs.
                context_plan = _exact_context_plan(config, example, selected_ids, selected)
                if not context_plan.within_budget:
                    raise ValueError(
                        "selected context exceeds the configured generator token budget"
                    )
                memory_by_id = {str(memory.memory_id): memory for memory in memories}
                selected_ids = list(context_plan.selected_ids)
                selected = [memory_by_id[memory_id] for memory_id in context_plan.chronological_ids]
                context_hash = context_plan.context_hash
                if bridge_result is not None:
                    bridge_result.context_plan = context_plan
                    bridge_result.context_hash = context_hash
                    bridge_result.selected_context = tuple(context_plan.chronological_ids)
                    bridge_result.diagnostics["context_plan"] = context_plan.public_dict()
                _refresh_rerank_selection_diagnostics(diagnostics, selected_ids)
                diagnostics["context_hash"] = context_hash
                diagnostics["context_plan"] = context_plan.public_dict()
                tracker.final_context_count = len(context_plan.chronological_ids)
                tracker.final_context_tokens = context_plan.token_count
                response = ""
                generation_hit = False
                if generator:
                    generation_started = time.perf_counter()
                    if shared_generation_cache is not None:
                        memory_by_id = {str(memory.memory_id): memory for memory in memories}
                        greedy_memories = [memory_by_id[memory_id] for memory_id in selected_ids]
                        response, generation_hit = shared_generation_cache.answer(
                            generator,
                            example.query,
                            greedy_memories,
                            example.all_options,
                            selected_ids=selected_ids,
                        )
                    else:
                        response = generator.answer_plan(context_plan)
                    if generation_hit:
                        tracker.record_cache_hit()
                    tracker.generation_ms = (
                        (time.perf_counter() - generation_started) * 1000.0 if not generation_hit else 0.0
                    )
                latency = time.perf_counter() - started
                latencies.append(latency)
                accuracy = answer_accuracy(response, example.correct_answer) if generator else None
                parse_failure = answer_parse_failed(response) if generator else None
                if accuracy is not None:
                    accuracy_sum += accuracy
                if parse_failure is not None:
                    parse_failure_sum += parse_failure
                gold_ids = bridge_gold.get(example.question_id, [])
                recall = None
                bridge_recall = None
                if gold_ids:
                    annotated += 1
                    ranks = direct_ranks(query_vector, [memory.memory_id for memory in memories], memory_vectors)
                    recall = recall_at_k(selected_ids, gold_ids, config.retrieval.context_size)
                    bridge_recall = bridge_recall_at_k(
                        selected_ids,
                        gold_ids,
                        ranks,
                        config.retrieval.context_size,
                    )
                    recall_sum += recall
                    if bridge_recall is not None:
                        bridge_recall_sum += bridge_recall
                        bridge_annotated += 1
                if bridge_result is not None:
                    bridge_results.append(bridge_result)
                    bridge_result.context_hash = context_hash
                    if isinstance(bridge_result.diagnostics, dict):
                        bridge_result.diagnostics["context_hash"] = context_hash
                    diagnostics = bridge_result.to_dict(include_text=False)
                    diagnostics["diagnostic"] = bridge_result.diagnostic_summary(list(gold_ids) if gold_ids else None)
                    diagnostics["path_objective_advantage"] = path_objective_advantage(
                        bridge_result, config.retrieval.context_size
                    )
                else:
                    diagnostics["cost"] = tracker.snapshot().to_dict()
                diagnostics["index_cache_hit"] = index_cache_hit
                cost = tracker.snapshot().to_dict()
                cost_records.append(cost)
                outcome = {
                    "answer_accuracy": accuracy,
                    "parse_failure_rate": parse_failure,
                    "recall_at_k": recall,
                    "bridge_recall_at_k": bridge_recall,
                }
                outcome_records.append(
                    {
                        **outcome,
                        "question_type": example.question_type,
                        "topic": example.topic,
                        "memory_scale": (
                            "small" if len(memories) < 16 else "medium" if len(memories) < 64 else "large"
                        ),
                    }
                )
                record = {
                    "persona_id": example.persona_id,
                    "question_id": example.question_id,
                    "question_type": example.question_type,
                    "topic": example.topic,
                    "query": example.query,
                    "correct_answer": example.correct_answer,
                    "selected_memory_ids": selected_ids,
                    "context_plan": context_plan.public_dict(),
                    "response": response,
                    "outcome": outcome,
                    "cost": cost,
                    "context_hash": context_hash,
                    **{
                        name: cost[name]
                        for name in (
                            "rerank_calls",
                            "rerank_documents",
                            "rerank_ms",
                            "bridge_embedding_calls",
                            "bridge_embedding_queries",
                            "bridge_embedding_ms",
                        )
                    },
                    "diagnostic": diagnostics,
                    **{
                        key: diagnostics.get(key)
                        for key in (
                            "dense_pool_ids",
                            "anchor_ids",
                            "bridge_raw_ids",
                            "bridge_kept_ids",
                            "candidate_union_ids",
                            "selected_source_by_id",
                            "selected_bridge_count",
                            "selected_bridge_rate",
                            "dense_rerank_top5_retention",
                            "bridge_candidate_novelty",
                            "dense_rerank_top_ids",
                            "parent_by_bridge_id",
                            "branch_by_bridge_id",
                            "raw_bridge_ids_by_branch",
                            "dense_rerank_scores",
                            "bridge_ann_scores",
                            "path_filter_scores",
                            "final_rerank_scores",
                            "rerank_cache_hits",
                        )
                        if key in diagnostics
                    },
                    # Compatibility fields for existing result readers.
                    "accuracy": accuracy,
                    "recall_at_k": recall,
                    "bridge_recall_at_k": bridge_recall,
                    "latency_seconds": latency,
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                total += 1
            except Exception as exc:  # a failed query is explicit run evidence
                failures += 1
                failure_output.write(
                    json.dumps(
                        {
                            "question_id": example.question_id,
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    numeric_cost_names = [
        "ann_calls_core",
        "ann_calls_diagnostic",
        "candidates_returned",
        "candidates_returned_diagnostic",
        "unique_visited_nodes",
        "index_build_ms",
        "retrieval_core_ms",
        "diagnostic_ms",
        "rerank_calls",
        "rerank_documents",
        "rerank_ms",
        "bridge_embedding_calls",
        "bridge_embedding_queries",
        "bridge_embedding_ms",
        "generation_ms",
        "final_context_count",
        "final_context_tokens",
        "duplicate_proposals",
        "proposal_count",
        "new_unique_candidates_per_ann",
        "proposal_ann_calls",
        "candidate_exposure",
        "state_embedding_calls",
        "state_embedding_queries",
        "state_embedding_ms",
        "transition_exact_ops",
        "bound_ops",
        "cache_hits",
    ]
    mean_cost = {
        name: sum(float(record[name]) for record in cost_records) / len(cost_records) if cost_records else 0.0
        for name in numeric_cost_names
    }
    stop_reasons = {
        reason: sum(record["stop_reason"] == reason for record in cost_records)
        for reason in sorted({str(record["stop_reason"]) for record in cost_records})
    }

    def stratify(field: str) -> Dict[str, Any]:
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for record in outcome_records:
            groups.setdefault(str(record[field]), []).append(record)
        result: Dict[str, Any] = {}
        for name, records in sorted(groups.items()):
            result[name] = {"queries": len(records)}
            for metric in ("answer_accuracy", "parse_failure_rate", "recall_at_k", "bridge_recall_at_k"):
                values = [float(record[metric]) for record in records if record[metric] is not None]
                result[name][metric] = sum(values) / len(values) if values else None
        return result

    summary: Dict[str, Any] = {
        "method": method,
        "split": split,
        "queries": total,
        "attempted_queries": attempted,
        "successful_queries": total,
        "failed_queries": failures,
        "failure_rate": failures / attempted if attempted else 0.0,
        "generated": generate,
        "answer_accuracy": accuracy_sum / total if generate and total else None,
        "parse_failure_rate": parse_failure_sum / total if generate and total else None,
        "annotated_queries": annotated,
        "recall_at_k": recall_sum / annotated if annotated else None,
        "bridge_annotated_queries": bridge_annotated,
        "bridge_recall_at_k": bridge_recall_sum / bridge_annotated if bridge_annotated else None,
        "cost": {"mean": mean_cost, "stop_reason_counts": stop_reasons},
        "stratified": {
            "question_type": stratify("question_type"),
            "topic": stratify("topic"),
            "memory_scale": stratify("memory_scale"),
        },
        "mean_retrieval_and_generation_latency_seconds": sum(latencies) / len(latencies) if latencies else 0.0,
        "model_calls_per_query": {"retrieval_llm": 0, "generator": int(generate)},
    }
    if bridge_results:
        from .metrics import summarize_bridge_results

        summary.update(summarize_bridge_results(bridge_results))
    with (run_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    manifest.update({"status": "completed", "queries": total, "failed_queries": failures})
    with (run_root / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return {"run_dir": str(run_root), "summary": summary}


def run_tmic_matrix(
    config: AppConfig,
    embedder: Embedder,
    *,
    phase: str = "development",
    limit: int | None = None,
    generate: bool = False,
    examples: Sequence[PersonaMemExample] | None = None,
    protocol_manifest: str | Path | Mapping[str, Any] | None = None,
    output_dir: str | Path | None = None,
    include_auxiliary: bool = False,
    synthetic: bool = False,
) -> Dict[str, Any]:
    """Run the fixed A0--A4 TMIC matrix with shared inputs and caches.

    This is a deliberately lightweight scheduler around the single retriever;
    it is usable with deterministic local embedders in CI and with the normal
    remote clients in a real experiment.  It never tunes a parameter and
    records a complete per-question provenance record for every row.
    """
    from .clients import GenerationCache
    config.validate()
    if limit is not None and limit <= 0:
        raise ValueError("TMIC limit must be positive")
    phase, all_examples = _protocol_select_examples(
        config,
        phase=phase,
        examples=examples,
        protocol_manifest=protocol_manifest,
        synthetic=synthetic,
    )
    if limit is not None:
        all_examples = all_examples[:limit]
    if not all_examples:
        raise ValueError("TMIC phase has no examples")

    labels = list(TMIC_ABLATIONS)
    if include_auxiliary:
        labels.extend(["R1+M", "R1+I"])
    retrieval_configs = tmic_matrix_configs(config.retrieval)
    if include_auxiliary:
        retrieval_configs["R1+M"] = replace(
            retrieval_configs["A0"],
            temporal_measure=False,
            measure_propagation=True,
            state_information=False,
            information_certificate=False,
            feature_mode="path_conditioned",
            selection_mode="path_logdet",
            stop_mode="budget",
        )
        retrieval_configs["R1+I"] = replace(
            retrieval_configs["A0"],
            temporal_measure=False,
            measure_propagation=False,
            state_information=True,
            information_certificate=False,
            feature_mode="path_conditioned",
            selection_mode="path_logdet",
            stop_mode="budget",
        )
        retrieval_configs["R1+M"].validate()
        retrieval_configs["R1+I"].validate()

    root = Path(output_dir or config.runtime.output_dir) / f"tmic_{phase}_{time.time_ns()}"
    root.mkdir(parents=True, exist_ok=False)
    embedding_cache = EmbeddingCache(
        config.runtime.cache_dir,
        embedder,
        config.models.embedding.model,
        asdict(config.models.embedding),
    )
    index_cache = IndexCache()
    transition_cache = TransitionCache(Path(config.runtime.cache_dir) / "transition")
    state_embedding_cache = StateEmbeddingCache(Path(config.runtime.cache_dir) / "state")
    generation_cache = GenerationCache(Path(config.runtime.cache_dir) / "generation")
    generator = GeneratorClient(config.models.generator) if generate else None
    predictions_by_label: Dict[str, list[Dict[str, Any]]] = {label: [] for label in labels}
    cost_by_label: Dict[str, list[Dict[str, Any]]] = {label: [] for label in labels}
    outcome_by_label: Dict[str, list[float]] = {label: [] for label in labels}
    context_hashes: Dict[str, Dict[str, str]] = {label: {} for label in labels}
    failures: list[Dict[str, Any]] = []

    # Option embeddings are computed once per question and then shared by the
    # I=true rows.  They define a frozen coordinate basis only; no gold answer
    # is ever consulted.  A cache hit still incurs a logical state lookup in
    # each architecture so cost records remain auditable.
    for example in all_examples:
        shared_option_embeddings: np.ndarray | None = None
        shared_state_provider: StateBasisProvider | None = None
        state_cache_hit = False
        state_elapsed_ms = 0.0
        state_physical_recorded = False
        option_count = len(parse_options(example.all_options))
        try:
            if any(retrieval_configs[label].state_information for label in labels):
                state_started = time.perf_counter()
                shared_option_embeddings, state_cache_hit = _state_option_embeddings(
                    example,
                    embedding_cache,
                    state_embedding_cache,
                    endpoint=config.models.embedding.endpoint,
                    model=config.models.embedding.model,
                    query_text=example.query,
                )
                state_elapsed_ms = (time.perf_counter() - state_started) * 1000.0
            if any(retrieval_configs[label].state_information for label in labels):
                # Share one frozen coordinate-system provider between A3 and
                # A4 for this question.  Its fingerprint identifies the
                # actual embedding cache/model rather than the retriever-local
                # fallback used by bare API callers.  It is also created when
                # a question has no options, so the identity-basis fallback is
                # explicitly recorded in provenance.
                shared_state_provider = StateBasisProvider(
                    config.retrieval.state_basis_mode,
                    model_fingerprint=embedding_cache.fingerprint,
                    endpoint=config.models.embedding.endpoint,
                )
        except Exception as exc:
            # Basis construction is a per-question failure.  Preserve one
            # explicit failure for every architecture rather than silently
            # dropping the whole question from the matrix.
            for label in labels:
                failures.append(
                    {
                        "question_id": example.question_id,
                        "architecture": label,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
            continue

        try:
            memories = messages_to_memories(
                example.messages,
                source_prefix=example.question_id,
                include_system_persona=config.data.include_system_persona,
                memory_granularity=config.data.memory_granularity,
            )
            memories = _visible_memory_records(example, memories)
            if not memories:
                raise ValueError("no memories after segmentation")
            query_vector = embedding_cache.encode_query(example.query)
            memory_vectors = embedding_cache.encode_documents([memory.text for memory in memories])
            context_key = (
                f"{example.shared_context_id}:{example.end_index}:"
                f"{config.data.memory_granularity}:{config.data.include_system_persona}"
            )
            index, index_build_ms, index_hit = index_cache.get(
                context_key,
                embedding_cache.fingerprint,
                config.retrieval.index_backend,
                [memory.memory_id for memory in memories],
                memory_vectors,
                config.retrieval.faiss_exclusion_margin,
            )
        except Exception as exc:
            for label in labels:
                failures.append(
                    {
                        "question_id": example.question_id,
                        "architecture": label,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
            continue

        for label in labels:
            try:
                retrieval_config = retrieval_configs[label]
                tracker = CostTracker(SearchBudget.from_config(retrieval_config))
                if retrieval_config.state_information and shared_option_embeddings is not None:
                    tracker.record_state_embedding(
                        option_count,
                        state_elapsed_ms
                        if not state_cache_hit and not state_physical_recorded
                        else 0.0,
                    )
                    if state_cache_hit or state_physical_recorded:
                        tracker.record_cache_hit()
                    state_physical_recorded = True
                result = BridgeTreeRetriever(retrieval_config).retrieve(
                    example.query,
                    query_vector,
                    memories,
                    memory_vectors,
                    index=index,
                    budget=tracker.budget,
                    cost_tracker=tracker,
                    index_build_ms=index_build_ms,
                    answer_options=example.all_options,
                    option_embeddings=shared_option_embeddings if retrieval_config.state_information else None,
                    state_basis_provider=shared_state_provider if retrieval_config.state_information else None,
                    query_cutoff=getattr(example, "query_time", None),
                    query_metadata=getattr(example, "metadata", None),
                    transition_cache=transition_cache,
                    state_embedding_cache=state_embedding_cache,
                    context_token_budget=config.models.generator.context_token_budget,
                    generator_config=config.models.generator,
                    # A0 is intentionally routed through the unified path so
                    # its T=0 transition/path/atom provenance is present even
                    # though ordinary legacy CLI calls remain unchanged.
                    force_tmic=True,
                )
                # ContextPlan is the sole source of the final context.  A
                # missing plan means the selected set could not satisfy the
                # declared request/budget contract; record an explicit row
                # failure rather than silently dropping memories here.
                if result.context_plan is None or not result.context_plan.within_budget:
                    plan_error = result.diagnostics.get("context_plan_error", {})
                    detail = (
                        f": {plan_error.get('message')}"
                        if isinstance(plan_error, Mapping) and plan_error.get("message")
                        else ""
                    )
                    raise ValueError(
                        "TMIC retrieval did not produce a valid ContextPlan" + detail
                    )
                context_plan = result.context_plan
                memory_by_id = {str(memory.memory_id): memory for memory in memories}
                selected_ids = list(context_plan.chronological_ids)
                selected = [memory_by_id[memory_id] for memory_id in selected_ids]
                context_hash = context_plan.context_hash
                response = ""
                generation_hit = False
                generation_elapsed_ms = 0.0
                if generator is not None:
                    generation_started = time.perf_counter()
                    response, generation_hit = generation_cache.answer_plan(generator, context_plan)
                    generation_elapsed_ms = (time.perf_counter() - generation_started) * 1000.0
                    if generation_hit:
                        tracker.record_cache_hit()
                        generation_elapsed_ms = 0.0
                    tracker.generation_ms = generation_elapsed_ms
                accuracy = answer_accuracy(response, example.correct_answer) if generator is not None else None
                if accuracy is not None:
                    outcome_by_label[label].append(float(accuracy))
                tracker.final_context_count = len(selected)
                # ContextPlan is authoritative for the actual request; the
                # memory-only helper would undercount system/query/options
                # tokens and make matrix rows claim a different context cost.
                tracker.final_context_tokens = context_plan.token_count
                cost = tracker.snapshot().to_dict()
                cost_by_label[label].append(cost)
                context_hashes[label][str(example.question_id)] = context_hash
                record = {
                    "persona_id": example.persona_id,
                    "question_id": example.question_id,
                    "architecture": label,
                    "selected_memory_ids": selected_ids,
                    "selected_in_greedy_order": list(result.selected_in_greedy_order),
                    "response": response,
                    "outcome": {"answer_accuracy": accuracy} if accuracy is not None else {},
                    "cost": cost,
                    "context_hash": context_hash,
                    "index_cache_hit": index_hit,
                    "generation_cache_hit": generation_hit,
                    "provenance": result.to_dict(include_text=False),
                }
                predictions_by_label[label].append(record)
            except Exception as exc:
                failures.append(
                    {
                        "question_id": example.question_id,
                        "architecture": label,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )

    summaries: Dict[str, Any] = {}
    for label in labels:
        costs = cost_by_label[label]
        numeric_names = sorted(
            {
                key
                for cost in costs
                for key, value in cost.items()
                if isinstance(value, (int, float))
            }
        )
        mean_cost = {
            key: sum(float(cost.get(key, 0.0)) for cost in costs) / len(costs) if costs else 0.0
            for key in numeric_names
        }
        summaries[label] = {
            "architecture": label,
            "queries": len(predictions_by_label[label]),
            "attempted_queries": len(all_examples),
            "failed_queries": len(all_examples) - len(predictions_by_label[label]),
            "failure_rate": (
                (len(all_examples) - len(predictions_by_label[label])) / len(all_examples)
                if all_examples
                else 0.0
            ),
            "answer_accuracy": (
                sum(outcome_by_label[label]) / len(outcome_by_label[label])
                if outcome_by_label[label]
                else None
            ),
            "mean_cost": mean_cost,
            "context_hashes": context_hashes[label],
        }
        with (root / f"predictions_{label}.jsonl").open("w", encoding="utf-8") as handle:
            for record in predictions_by_label[label]:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Record source and repository identity alongside the resolved matrix.  A
    # later audit can therefore distinguish a reproducible failure from a
    # silently changed dataset/configuration.
    repository_root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository_root, check=False, capture_output=True, text=True
    ).stdout.strip()
    runtime_provenance = _runtime_provenance(config.data.raw_dir, config.data.split)
    question_ids = [str(example.question_id) for example in all_examples]
    persona_ids = sorted({str(example.persona_id) for example in all_examples})
    resolved_matrix = {
        label: {
            **asdict(retrieval_configs[label]),
            "label": TMIC_ABLATIONS.get(label, {}).get("label", label),
        }
        for label in labels
    }
    matrix_payload = {
        "phase": phase,
        "base_config": _public_app_config(config),
        "retrieval_matrix": resolved_matrix,
        "architectures": labels,
    }
    matrix_config_hash = hashlib.sha256(
        json.dumps(matrix_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    (root / "resolved_config.json").write_text(
        json.dumps(
            {
                "config_hash": matrix_config_hash,
                "app_config_hash": config.config_hash(),
                "config": {
                    **matrix_payload,
                    "base_config": _public_app_config(config),
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        # A local ``phase=development`` matrix may intentionally run without
        # a persisted split manifest.  Do not label that exploratory run as a
        # confirmatory_v1 result; the protocol identity is present only when
        # its authoritative manifest was actually gated above.
        "protocol": "confirmatory_v1" if protocol_manifest is not None else None,
        "protocol_version": 1 if protocol_manifest is not None else None,
        "phase": phase,
        "architectures": labels,
        "tmic_ablation_definitions": {label: TMIC_ABLATIONS.get(label, {}) for label in labels},
        "resolved_retrieval_configs": resolved_matrix,
        "queries": len(all_examples),
        "question_ids": question_ids,
        "question_id_sha256": hashlib.sha256("\n".join(question_ids).encode("utf-8")).hexdigest(),
        "personas": persona_ids,
        "persona_count": len(persona_ids),
        "persona_id_sha256": hashlib.sha256("\n".join(persona_ids).encode("utf-8")).hexdigest(),
        "seed": config.seed,
        "config_hash": config.config_hash(),
        "matrix_config_hash": matrix_config_hash,
        "protocol_manifest": str(protocol_manifest) if protocol_manifest is not None else None,
        "resolved_config": _public_app_config(config),
        "data_revision": PERSONAMEM_REVISION,
        "data_split": config.data.split,
        **runtime_provenance,
        "git_commit": commit or None,
        "budget": asdict(SearchBudget.from_config(config.retrieval)),
        "embedding_fingerprint": embedding_cache.fingerprint,
        "embedding_model": _public_service_identity(config.models.embedding),
        "generator_model": _public_service_identity(config.models.generator),
        "prompt_hash": generation_prompt_hash(),
        "generation": generate,
        "failures": failures,
        "cache_schema": {"transition": 1, "generation": 1, "state": 1},
    }
    (root / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "failures.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in failures), encoding="utf-8"
    )
    summary = {
        "phase": phase,
        "run_dir": str(root),
        "architectures": summaries,
        "failures": failures,
        "shared_inputs": True,
        "a3_a4_context_hash_equal_count": sum(
            summaries.get("A3", {}).get("context_hashes", {}).get(q)
            == summaries.get("A4", {}).get("context_hashes", {}).get(q)
            for q in set(summaries.get("A3", {}).get("context_hashes", {}))
            & set(summaries.get("A4", {}).get("context_hashes", {}))
        ),
    }
    (root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    # A compact combined stream is convenient for audit tools.
    with (root / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for label in labels:
            for record in predictions_by_label[label]:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"run_dir": str(root), "summary": summary, "manifest": manifest}


def _legacy_trace_graph(
    result: RetrievalResult,
    memories: Sequence[Memory],
    *,
    rho: Mapping[str, float],
    parent_id: Mapping[str, str | None],
) -> FrozenProposalGraph:
    """Freeze the actual legacy executor nodes for L0/L1 comparisons."""
    nodes = getattr(result, "nodes", None)
    if not isinstance(nodes, Mapping) or not nodes:
        raise ValueError("legacy executor returned no node trace")
    ids = tuple(str(identifier) for identifier in nodes)
    if set(ids) != set(rho) or set(ids) != set(parent_id):
        raise ValueError("legacy trace does not cover the frozen legacy pool")
    edges = tuple(
        (str(parent), identifier)
        for identifier in ids
        for parent in [parent_id.get(identifier)]
        if parent is not None
    )
    if len(set(edges)) != len(edges):
        edges = tuple(dict.fromkeys(edges))
    roots = [identifier for identifier in ids if parent_id.get(identifier) is None]
    root_mass = {identifier: (1.0 / len(roots) if identifier in roots and roots else 0.0) for identifier in ids}
    depths = {identifier: int(getattr(nodes[identifier], "depth", 1) or 1) for identifier in ids}
    layers_map: dict[int, list[str]] = {}
    for identifier, depth in depths.items():
        layers_map.setdefault(max(1, depth), []).append(identifier)
    layers = tuple(tuple(sorted(values)) for _depth, values in sorted(layers_map.items()))
    edge_weights = tuple((parent, child, float(rho.get(child, 0.0))) for parent, child in edges)
    parent_sources = tuple(
        (identifier, (str(parent_id[identifier]),))
        for identifier in ids
        if parent_id.get(identifier) is not None
    )
    proposal_records = tuple(
        (parent, child, depths[child], rank, float(rho.get(child, 0.0)))
        for rank, (parent, child) in enumerate(edges)
    )
    return FrozenProposalGraph(
        memory_ids=ids,
        edges=edges,
        edge_weights=edge_weights,
        root_mass=root_mass,
        layers=layers,
        parent_sources=parent_sources,
        proposal_records=proposal_records,
        domain_scope="legacy_executor_trace",
        proposal_config={
            "legacy_trace": {"rho": dict(rho), "parent_id": dict(parent_id)},
            "source": "legacy_executor",
        },
    )


def run_semantic_matrix(
    config: AppConfig,
    embedder: Embedder,
    *,
    phase: str = "development",
    limit: int | None = None,
    generate: bool = False,
    examples: Sequence[PersonaMemExample] | None = None,
    protocol_manifest: str | Path | Mapping[str, Any] | None = None,
    output_dir: str | Path | None = None,
    reranker: Any | None = None,
    quality_provider: Any | None = None,
    rows: Sequence[str] | str | None = None,
    baseline: str | None = "dense_rerank",
    synthetic: bool = False,
) -> Dict[str, Any]:
    """Run the fixed L0/L1/S0--S2-shuffle matrix on shared frozen pools.

    Candidate discovery and pointwise quality are performed once per question;
    each architecture receives the exact same frozen graph and original memory
    pool.  L0/L1 use an explicitly separate rho² table, while the S rows share
    one pointwise quality table.  This makes both path-vs-flat and legacy-vs-
    semantic comparisons interpretable and keeps the command usable with
    deterministic local clients in CI.
    """
    from .information import validate_quality_records
    config.validate()
    if limit is not None and limit <= 0:
        raise ValueError("semantic matrix limit must be positive")
    phase, all_examples = _protocol_select_examples(
        config,
        phase=phase,
        examples=examples,
        protocol_manifest=protocol_manifest,
        synthetic=synthetic,
    )
    if limit is not None:
        all_examples = all_examples[:limit]
    if not all_examples:
        raise ValueError("semantic matrix phase has no examples")

    if rows is None:
        labels = ["S0", "S1", "S2"]
    elif isinstance(rows, str):
        labels = [item.strip() for item in rows.split(",") if item.strip()]
    else:
        labels = [str(item) for item in rows]
    if not labels:
        raise ValueError("semantic matrix rows must not be empty")
    unknown_rows = sorted(set(labels) - set(SEMANTIC_ABLATIONS))
    if unknown_rows:
        raise ValueError(f"unknown semantic matrix rows: {unknown_rows}")
    if len(set(labels)) != len(labels):
        raise ValueError("semantic matrix rows must be unique")
    baseline_name = None if baseline in (None, "", "none", "off") else str(baseline)
    if baseline_name not in {None, "dense_rerank"}:
        raise ValueError("baseline must be dense_rerank or omitted")
    retrieval_configs = semantic_matrix_configs(config)
    root = Path(output_dir or config.runtime.output_dir) / f"semantic_{phase}_{time.time_ns()}"
    root.mkdir(parents=True, exist_ok=False)
    embedding_cache = EmbeddingCache(
        config.runtime.cache_dir,
        embedder,
        config.models.embedding.model,
        asdict(config.models.embedding),
    )
    generation_cache = GenerationCache(Path(config.runtime.cache_dir) / "generation")
    generator = GeneratorClient(config.models.generator) if generate else None
    predictions_by_label: Dict[str, list[Dict[str, Any]]] = {label: [] for label in labels}
    costs_by_label: Dict[str, list[Dict[str, Any]]] = {label: [] for label in labels}
    accuracy_by_label: Dict[str, list[float]] = {label: [] for label in labels}
    failures: list[Dict[str, Any]] = []
    dense_predictions: list[Dict[str, Any]] = []
    dense_failures: list[Dict[str, Any]] = []
    legacy_executor_records: list[Dict[str, Any]] = []
    legacy_executor_failures: list[Dict[str, Any]] = []
    shared_costs_by_question: dict[str, dict[str, Any]] = {}
    quality_source_by_label = {
        label: str(SEMANTIC_ABLATIONS[label].get("quality_source", "pointwise"))
        for label in labels
    }
    pointwise_runtime_source = (
        "reranker"
        if reranker is not None
        else "provider"
        if quality_provider is not None
        else "offline-direct-cosine"
    )

    def failed_record(example: PersonaMemExample, label: str, exc: Exception) -> Dict[str, Any]:
        """Persist one planned row even when retrieval or generation fails."""
        return {
            "persona_id": str(example.persona_id),
            "question_id": str(example.question_id),
            "architecture": label,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "selected_memory_ids": [],
            "chronological_ids": [],
            "selected_in_greedy_order": [],
            "response": None,
            "outcome": {"answer_accuracy": 0.0 if generate else None},
            "cost": {},
            "pool_id": None,
            "pool_origin": "legacy_trace" if label in {"L0", "L1"} else "current_multi_parent",
        }

    for example in all_examples:
        try:
            memories = messages_to_memories(
                example.messages,
                source_prefix=example.question_id,
                include_system_persona=config.data.include_system_persona,
                memory_granularity=config.data.memory_granularity,
            )
            memories = _visible_memory_records(example, memories)
            if not memories:
                raise ValueError("no memories after segmentation")
            query_vector = embedding_cache.encode_query(example.query)
            memory_vectors = embedding_cache.encode_documents([memory.text for memory in memories])
            index, index_build_ms, index_hit = IndexCache().get(
                f"semantic:{example.shared_context_id}:{example.end_index}:{config.data.memory_granularity}",
                embedding_cache.fingerprint,
                config.retrieval.index_backend,
                [memory.memory_id for memory in memories],
                memory_vectors,
                config.retrieval.faiss_exclusion_margin,
            )
            discovery_budget = SearchBudget.from_config(config.retrieval)
            discovery_tracker = CostTracker(discovery_budget)
            graph, graph_diagnostics, _tracker, index = discover_frozen_graph(
                memories,
                memory_vectors,
                query_vector,
                initial_width=config.retrieval.initial_width,
                branch_width=config.retrieval.branch_width,
                proposal_width=config.retrieval.proposal_width,
                max_depth=config.retrieval.max_depth,
                budget=discovery_budget,
                tracker=discovery_tracker,
                index=index,
                relation_mode="angular",
                proposal_mode="real_member_query_anchor",
                cutoff=getattr(example, "query_time", None),
                query_metadata=getattr(example, "metadata", None),
                query_text=example.query,
                proposal_query_provider=embedding_cache,
                proposal_query_instruction=config.bridge_rerank.bridge_query_instruction,
            )
            shared_discovery_cost = discovery_tracker.snapshot()
            shared_costs_by_question[str(example.question_id)] = {
                "discovery": shared_discovery_cost.to_dict(),
            }
            records = {memory.memory_id: memory for memory in memories if memory.memory_id in graph.memory_ids}
            legacy_graph: FrozenProposalGraph | None = None
            legacy_trace_config: Mapping[str, Any] | None = None
            legacy_discovery_cost: Dict[str, Any] | None = None
            if "L0" in labels or "L1" in labels:
                legacy_config = replace(
                    config.retrieval,
                    profile="legacy_core",
                    proposal_mode="legacy_first_arrival",
                    feature_mode="rho",
                    selection_mode="rho_logdet",
                    quality_mode="rho",
                    path_mode="none",
                    temporal_measure=False,
                    measure_propagation=False,
                    state_information=False,
                    information_certificate=False,
                    stop_mode="budget",
                )
                legacy_config.validate()
                legacy_tracker = CostTracker(SearchBudget.from_config(legacy_config))
                legacy_result = BridgeTreeRetriever(legacy_config).retrieve(
                    example.query,
                    query_vector,
                    memories,
                    memory_vectors,
                    index=index,
                    budget=legacy_tracker.budget,
                    cost_tracker=legacy_tracker,
                    query_cutoff=getattr(example, "query_time", None),
                    query_metadata=getattr(example, "metadata", None),
                    answer_options=example.all_options,
                    context_token_budget=None,
                )
                legacy_rho, legacy_parent_id = legacy_trace_from_result(legacy_result)
                legacy_graph = _legacy_trace_graph(
                    legacy_result,
                    memories,
                    rho=legacy_rho,
                    parent_id=legacy_parent_id,
                )
                legacy_trace_config = {
                    "rho": legacy_rho,
                    "parent_id": legacy_parent_id,
                    "source": "legacy_executor",
                }
                legacy_discovery_cost = legacy_tracker.snapshot().to_dict()
                legacy_executor_records.append(
                    {
                        "persona_id": str(example.persona_id),
                        "question_id": str(example.question_id),
                        "method": "legacy_core",
                        "status": "success",
                        "selected_memory_ids": list(legacy_result.selected_context),
                        "discovered_memory_ids": list(legacy_result.nodes),
                        "pool_id": legacy_graph.graph_hash,
                        "cost": legacy_discovery_cost,
                    }
                )
                legacy_path_config = replace(
                    legacy_config,
                    profile="legacy_path",
                    feature_mode="path_conditioned",
                    selection_mode="path_logdet",
                    path_mode="legacy",
                )
                legacy_path_config.validate()
                legacy_path_tracker = CostTracker(SearchBudget.from_config(legacy_path_config))
                legacy_path_result = BridgeTreeRetriever(legacy_path_config).retrieve(
                    example.query,
                    query_vector,
                    memories,
                    memory_vectors,
                    index=index,
                    budget=legacy_path_tracker.budget,
                    cost_tracker=legacy_path_tracker,
                    query_cutoff=getattr(example, "query_time", None),
                    query_metadata=getattr(example, "metadata", None),
                    answer_options=example.all_options,
                    context_token_budget=None,
                )
                legacy_executor_records.append(
                    {
                        "persona_id": str(example.persona_id),
                        "question_id": str(example.question_id),
                        "method": "legacy_path",
                        "status": "success",
                        "selected_memory_ids": list(legacy_path_result.selected_context),
                        "discovered_memory_ids": list(legacy_path_result.nodes),
                        "pool_id": hashlib.sha256(
                            json.dumps(list(legacy_path_result.nodes), sort_keys=True).encode()
                        ).hexdigest(),
                        "cost": legacy_path_tracker.snapshot().to_dict(),
                    }
                )
            # L0/L1 are the explicit legacy controls.  Their quality is the
            # frozen graph access probability squared, not the reranker table;
            # this gives the exact ``sqrt(rho²)=rho`` scale while preserving a
            # separate provenance identity from the S rows.
            graph_measure = propagate_frozen_graph(graph)
            rho_quality_started = time.perf_counter()
            explicit_legacy_rho = (
                legacy_trace_config.get("rho")
                if isinstance(legacy_trace_config, Mapping)
                else None
            )
            rho_qualities = (
                rho_squared_quality_records(
                    propagate_frozen_graph(legacy_graph) if legacy_graph is not None else graph_measure,
                    legacy_rho=explicit_legacy_rho,
                    scorer_fingerprint=retrieval_configs["L0"].scorer_fingerprint or "legacy-rho2",
                )
                if explicit_legacy_rho is not None
                else {}
            )
            rho_quality_ms = (time.perf_counter() - rho_quality_started) * 1000.0

            quality_started = time.perf_counter()
            if reranker is not None:
                pointwise_qualities = reranker_quality_records(
                    reranker,
                    example.query,
                    records,
                    answer_options=example.all_options,
                    instruction=config.bridge_rerank.final_rerank_instruction,
                    include_time_metadata=config.bridge_rerank.include_time_metadata,
                    use_answer_options=config.bridge_rerank.use_answer_options,
                    score_space=config.retrieval.quality_score_space,
                    scorer_fingerprint=config.retrieval.scorer_fingerprint,
                    query_cutoff=getattr(example, "query_time", None),
                    query_metadata=getattr(example, "metadata", None),
                )
                pointwise_source = "reranker"
            elif quality_provider is not None:
                provider_method = getattr(quality_provider, "score_all", None)
                if callable(provider_method):
                    provider_output = provider_method(example.query, records)
                elif callable(quality_provider):
                    provider_output = quality_provider(example.query, records)
                else:
                    provider_output = quality_provider
                pointwise_qualities = validate_quality_records(
                    provider_output,
                    graph.memory_ids,
                    score_space=config.retrieval.quality_score_space,
                    scorer_fingerprint=config.retrieval.scorer_fingerprint,
                )
                pointwise_source = "provider"
            else:
                # Explicit offline adapter for matrix smoke tests.  The run
                # manifest marks this source so it cannot be confused with a
                # frozen reranker experiment.
                offline_space = str(config.retrieval.quality_score_space)
                pointwise_qualities = {
                    identifier: QualityRecord.from_raw(
                        identifier,
                        (
                            nonnegative_cosine(query_vector, index.vector(identifier))
                            if offline_space == "unit_interval"
                            else np.log(
                                np.clip(
                                    nonnegative_cosine(query_vector, index.vector(identifier)),
                                    1e-6,
                                    1.0 - 1e-6,
                                )
                                / (
                                    1.0
                                    - np.clip(
                                        nonnegative_cosine(query_vector, index.vector(identifier)),
                                        1e-6,
                                        1.0 - 1e-6,
                                    )
                                )
                            )
                        ),
                        offline_space,
                        scorer_fingerprint="offline-direct-cosine",
                    )
                    for identifier in graph.memory_ids
                }
                pointwise_source = "offline-direct-cosine"
            shared_quality_ms = (time.perf_counter() - quality_started) * 1000.0
            shared_quality_documents = len(records) if reranker is not None else 0
            shared_costs_by_question[str(example.question_id)]["quality"] = {
                "rerank_calls": 1 if reranker is not None else 0,
                "rerank_documents": shared_quality_documents,
                "elapsed_ms": shared_quality_ms,
                "source": pointwise_source,
                "rho2": {
                    "source": "rho2",
                    "trace_available": explicit_legacy_rho is not None,
                    "elapsed_ms": rho_quality_ms,
                    "documents": len(rho_qualities),
                    "legacy_executor": legacy_discovery_cost,
                },
            }
            quality_tables = {"rho2": rho_qualities, "pointwise": pointwise_qualities}
            quality_runtime_sources = {"rho2": "rho2", "pointwise": pointwise_source}

            if baseline_name == "dense_rerank" and reranker is not None:
                try:
                    dense_tracker = CostTracker(SearchBudget.from_config(config.retrieval))
                    dense_ids, dense_selected, dense_diagnostics, _ = retrieve_method(
                        "dense_rerank",
                        config,
                        example,
                        memories,
                        query_vector,
                        memory_vectors,
                        reranker=reranker,
                        index=index,
                        budget=dense_tracker.budget,
                        cost_tracker=dense_tracker,
                        index_build_ms=index_build_ms,
                        embedding_cache=embedding_cache,
                    )
                    dense_tracker = dense_diagnostics.pop("_cost_tracker", dense_tracker)
                    dense_plan = _exact_context_plan(config, example, dense_ids, dense_selected)
                    dense_response = ""
                    dense_cache_hit = False
                    if generator is not None:
                        dense_response, dense_cache_hit = generation_cache.answer(
                            generator,
                            example.query,
                            dense_selected,
                            example.all_options,
                            selected_ids=dense_ids,
                        )
                        if dense_cache_hit:
                            dense_tracker.record_cache_hit()
                    dense_accuracy = (
                        answer_accuracy(dense_response, example.correct_answer) if generator is not None else None
                    )
                    dense_predictions.append(
                        {
                            "persona_id": str(example.persona_id),
                            "question_id": str(example.question_id),
                            "architecture": "DenseRerank",
                            "status": "success",
                            "pool_origin": "dense_independent",
                            "pool_id": hashlib.sha256(
                                json.dumps(
                                    list(dense_diagnostics.get("candidate_union_ids", dense_ids)),
                                    sort_keys=True,
                                ).encode()
                            ).hexdigest(),
                            "selected_memory_ids": list(dense_plan.chronological_ids),
                            "selected_in_greedy_order": list(dense_ids),
                            "response": dense_response,
                            "outcome": {"answer_accuracy": dense_accuracy},
                            "context_plan": dense_plan.public_dict(),
                            "cost": dense_tracker.snapshot().to_dict(),
                            "diagnostic": dense_diagnostics,
                            "generation_cache_hit": dense_cache_hit,
                        }
                    )
                except Exception as exc:
                    dense_failures.append(
                        {
                            "question_id": str(example.question_id),
                            "architecture": "DenseRerank",
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        }
                    )
                    dense_predictions.append(
                        {
                            "persona_id": str(example.persona_id),
                            "question_id": str(example.question_id),
                            "architecture": "DenseRerank",
                            "status": "failed",
                            "pool_origin": "dense_independent",
                            "pool_id": None,
                            "selected_memory_ids": [],
                            "selected_in_greedy_order": [],
                            "response": None,
                            "outcome": {"answer_accuracy": 0.0 if generate else None},
                            "cost": {},
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )

            # Materialize query-conditioned representations once and share
            # them between the only row that needs them and any repeated run.
            query_representations: dict[str, np.ndarray] | None = None
            query_representation_error: dict[str, str] | None = None
            shared_state_embedding_ms = 0.0
            if "S3" in labels and retrieval_configs["S3"].feature_mode == "query_conditioned":
                state_texts = [
                    build_query_conditioned_representation_text(
                        example.query,
                        example.all_options,
                        records[identifier],
                        query_cutoff=getattr(example, "query_time", None),
                        query_metadata=getattr(example, "metadata", None),
                        embedding_model=embedding_cache.model_name,
                        embedding_fingerprint=embedding_cache.fingerprint,
                        instruction=SEMANTIC_QUERY_CONDITIONED_INSTRUCTION,
                    )
                    for identifier in graph.memory_ids
                ]
                try:
                    state_started = time.perf_counter()
                    state_vectors = embedding_cache.encode_queries(
                        state_texts,
                        instruction=SEMANTIC_QUERY_CONDITIONED_INSTRUCTION,
                        purpose="semantic_query_conditioned",
                    )
                    query_representations = {
                        identifier: state_vectors[position]
                        for position, identifier in enumerate(graph.memory_ids)
                    }
                    shared_state_embedding_ms = (time.perf_counter() - state_started) * 1000.0
                    shared_costs_by_question[str(example.question_id)]["state_embedding"] = {
                        "calls": 1,
                        "queries": len(graph.memory_ids),
                        "elapsed_ms": shared_state_embedding_ms,
                    }
                except Exception as exc:
                    # Query-conditioned representations are a distinct
                    # architecture, not an optional hint.  Falling back to
                    # cached memory vectors would silently relabel S3 as S2
                    # and invalidate the matrix comparison.  Keep the other
                    # rows runnable, but emit an explicit S3 failure below.
                    query_representation_error = {
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                    shared_costs_by_question[str(example.question_id)]["state_embedding"] = {
                        "calls": 0,
                        "queries": 0,
                        "elapsed_ms": 0.0,
                        "status": "failed",
                        **query_representation_error,
                    }
            for label in labels:
                try:
                    retrieval_config = retrieval_configs[label]
                    quality_source_key = str(SEMANTIC_ABLATIONS[label].get("quality_source", "pointwise"))
                    qualities = quality_tables["rho2" if quality_source_key == "rho2" else "pointwise"]
                    runtime_quality_source = quality_runtime_sources[
                        "rho2" if quality_source_key == "rho2" else "pointwise"
                    ]
                    label_graph = legacy_graph if quality_source_key == "rho2" and legacy_graph is not None else graph
                    if label == "S3" and query_representation_error is not None:
                        raise RuntimeError(
                            "query-conditioned representation service failed: "
                            + query_representation_error["message"]
                        )
                    if quality_source_key == "rho2" and explicit_legacy_rho is None:
                        raise RuntimeError(
                            "original R1 row requires legacy executor trace; "
                            "current multi-parent pool is retained for S rows"
                        )
                    tracker = CostTracker(SearchBudget.from_config(retrieval_config))
                    tracker.index_build_ms = float(index_build_ms)
                    tracker.inherit_shared_cost(shared_discovery_cost)
                    if label == "S3" and query_representations is not None:
                        tracker.shared_state_embedding_calls = 1
                        tracker.shared_state_embedding_queries = len(label_graph.memory_ids)
                        tracker.shared_state_embedding_ms = shared_state_embedding_ms
                    if reranker is not None and quality_source_key != "rho2":
                        tracker.shared_rerank_calls = 1
                        tracker.shared_rerank_documents = len(records)
                        tracker.shared_rerank_ms = shared_quality_ms
                    tracker.mark_visited(label_graph.memory_ids)
                    representation_provider = (
                        query_representations
                        if label == "S3" and query_representations is not None
                        else {identifier: index.vector(identifier) for identifier in label_graph.memory_ids}
                    )
                    result = semantic_retrieve(
                        example.query,
                        query_vector,
                        memories,
                        memory_vectors,
                        retrieval_config,
                        index=index,
                        budget=tracker.budget,
                        cost_tracker=tracker,
                        quality_records=qualities,
                        representation_provider=representation_provider,
                        answer_options=example.all_options,
                        query_cutoff=getattr(example, "query_time", None),
                        query_metadata=getattr(example, "metadata", None),
                        context_token_budget=config.models.generator.context_token_budget,
                        generator_config=config.models.generator,
                        frozen_graph=label_graph,
                        representation_fingerprint=embedding_cache.fingerprint,
                        legacy_rho=(
                            legacy_trace_config.get("rho")
                            if isinstance(legacy_trace_config, Mapping)
                            else None
                        ),
                        legacy_parent_id=(
                            legacy_trace_config.get("parent_id")
                            if isinstance(legacy_trace_config, Mapping)
                            else None
                        ),
                    )
                    if result.context_plan is None or not result.context_plan.within_budget:
                        raise ValueError("semantic retrieval did not produce an exact ContextPlan")
                    response = ""
                    generation_hit = False
                    if generator is not None:
                        memory_by_id = {str(memory.memory_id): memory for memory in memories}
                        greedy_memories = [
                            memory_by_id[memory_id] for memory_id in result.selected_in_greedy_order
                        ]
                        response, generation_hit = generation_cache.answer(
                            generator,
                            example.query,
                            greedy_memories,
                            example.all_options,
                            selected_ids=result.selected_in_greedy_order,
                        )
                        if generation_hit:
                            tracker.record_cache_hit()
                    accuracy = answer_accuracy(response, example.correct_answer) if generator is not None else None
                    if accuracy is not None:
                        accuracy_by_label[label].append(float(accuracy))
                    tracker.final_context_count = len(result.selected_context)
                    tracker.final_context_tokens = result.context_plan.token_count if result.context_plan else 0
                    cost = tracker.snapshot().to_dict()
                    costs_by_label[label].append(cost)
                    record = {
                        "persona_id": example.persona_id,
                        "question_id": example.question_id,
                        "architecture": label,
                        "pool_id": label_graph.graph_hash,
                        "pool_origin": (
                            "legacy_trace" if quality_source_key == "rho2" and legacy_graph is not None
                            else "current_multi_parent"
                        ),
                        "selected_memory_ids": list(result.selected_context),
                        "chronological_ids": list(result.selected_context),
                        "selected_in_greedy_order": list(result.selected_in_greedy_order),
                        "response": response,
                        "outcome": {"answer_accuracy": accuracy} if accuracy is not None else {},
                        "cost": cost,
                        "context_hash": result.context_hash,
                        "context_plan": result.context_plan.public_dict(),
                        "generation_cache_hit": generation_hit,
                        "shared_graph_hash": label_graph.graph_hash,
                        "shared_quality": {identifier: value.public_dict() for identifier, value in qualities.items()},
                        "quality_source": runtime_quality_source,
                        "quality_source_declared": quality_source_key,
                        "quality_digest": hashlib.sha256(
                            json.dumps(
                                {identifier: value.public_dict() for identifier, value in qualities.items()},
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            ).encode("utf-8")
                        ).hexdigest(),
                        "provenance": result.to_dict(include_text=False),
                        "index_cache_hit": index_hit,
                        "shared_discovery_cost": shared_discovery_cost.to_dict(),
                        "shared_quality_cost": (
                            {
                                "rerank_calls": 0,
                                "rerank_documents": 0,
                                "elapsed_ms": rho_quality_ms,
                                "source": "rho2",
                                "logical_documents": len(qualities),
                            }
                            if quality_source_key == "rho2"
                            else {
                                "rerank_calls": 1 if reranker is not None else 0,
                                "rerank_documents": shared_quality_documents,
                                "elapsed_ms": shared_quality_ms,
                                "source": runtime_quality_source,
                            }
                        ),
                        "shared_state_embedding_cost": {
                            "calls": 1 if label == "S3" and query_representations is not None else 0,
                            "queries": len(graph.memory_ids)
                            if label == "S3" and query_representations is not None
                            else 0,
                            "elapsed_ms": shared_state_embedding_ms
                            if label == "S3" and query_representations is not None
                            else 0.0,
                        },
                        "representation_status": (
                            "query_conditioned"
                            if label == "S3"
                            else str(SEMANTIC_ABLATIONS[label].get("representation_role", "cached_memory"))
                        ),
                        "status": "success",
                    }
                    predictions_by_label[label].append(record)
                except Exception as exc:
                    predictions_by_label[label].append(failed_record(example, label, exc))
                    failures.append(
                        {
                            "question_id": example.question_id,
                            "architecture": label,
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        }
                    )
        except Exception as exc:
            for label in labels:
                predictions_by_label[label].append(failed_record(example, label, exc))
                failures.append(
                    {
                        "question_id": example.question_id,
                        "architecture": label,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
            if baseline_name == "dense_rerank" and reranker is not None:
                dense_failures.append(
                    {
                        "question_id": str(example.question_id),
                        "architecture": "DenseRerank",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                dense_predictions.append(
                    {
                        "persona_id": str(example.persona_id),
                        "question_id": str(example.question_id),
                        "architecture": "DenseRerank",
                        "status": "failed",
                        "pool_origin": "dense_independent",
                        "pool_id": None,
                        "selected_memory_ids": [],
                        "selected_in_greedy_order": [],
                        "response": None,
                        "outcome": {"answer_accuracy": 0.0 if generate else None},
                        "cost": {},
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )

    summaries: Dict[str, Any] = {}
    baseline_predictions = predictions_by_label.get("S0", [])
    for label in labels:
        costs = costs_by_label[label]
        planned_records = predictions_by_label[label]
        successful_records = [record for record in planned_records if record.get("status") == "success"]
        failed_count = len(planned_records) - len(successful_records)
        successful_answers = [
            float(record["outcome"]["answer_accuracy"])
            for record in successful_records
            if isinstance(record.get("outcome"), Mapping)
            and record["outcome"].get("answer_accuracy") is not None
        ]
        end_to_end_answers = (
            [
                float(record.get("outcome", {}).get("answer_accuracy", 0.0) or 0.0)
                for record in planned_records
            ]
            if generate
            else []
        )
        metric_records = []
        if generate:
            for record in planned_records:
                metric_record = dict(record)
                metric_record["outcome"] = {
                    "answer_accuracy": float(
                        record.get("outcome", {}).get("answer_accuracy", 0.0) or 0.0
                    )
                }
                metric_records.append(metric_record)
        numeric_names = sorted(
            {
                key for cost in costs for key, value in cost.items() if isinstance(value, (int, float))
            }
        )
        summaries[label] = {
            "architecture": label,
            "queries": len(planned_records),
            "attempted_queries": len(all_examples),
            "successful_queries": len(successful_records),
            "failed_queries": failed_count,
            "failure_rate": failed_count / len(all_examples),
            "status": "incomplete" if failed_count else "completed",
            "has_failures": bool(failed_count),
            "answer_accuracy": (
                sum(end_to_end_answers) / len(end_to_end_answers)
                if end_to_end_answers
                else None
            ),
            "end_to_end_micro_accuracy": (
                sum(end_to_end_answers) / len(end_to_end_answers) if end_to_end_answers else None
            ),
            "end_to_end_macro_accuracy": persona_macro_accuracy(metric_records) if generate else None,
            "successful_response_accuracy": (
                sum(successful_answers) / len(successful_answers) if successful_answers else None
            ),
            "persona_macro_accuracy": persona_macro_accuracy(metric_records) if generate else None,
            "question_micro_accuracy": (
                sum(end_to_end_answers) / len(end_to_end_answers) if end_to_end_answers else None
            ),
            "paired_vs_S0": (
                None
                if label == "S0"
                else gain_damage_net(baseline_predictions, planned_records)
            ),
            # Dense+Rerank is a separate candidate-discovery run and is not
            # silently synthesized from this shared semantic pool.
            "paired_vs_DenseRerank": None,
            "mean_cost": {
                key: sum(float(cost.get(key, 0.0)) for cost in costs) / len(costs) if costs else 0.0
                for key in numeric_names
            },
        }
        (root / f"predictions_{label}.jsonl").write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in predictions_by_label[label]),
            encoding="utf-8",
        )

    repository_root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository_root, check=False, capture_output=True, text=True
    ).stdout.strip()
    runtime_provenance = _runtime_provenance(config.data.raw_dir, config.data.split)
    manifest = {
        "protocol": "confirmatory_v1" if protocol_manifest is not None else None,
        "phase": phase,
        "architectures": labels,
        "architecture_order": labels,
        "semantic_ablation_definitions": SEMANTIC_ABLATIONS,
        "resolved_retrieval_configs": {label: asdict(value) for label, value in retrieval_configs.items()},
        "queries": len(all_examples),
        "question_ids": [str(example.question_id) for example in all_examples],
        "question_id_sha256": hashlib.sha256(
            "\n".join(str(example.question_id) for example in all_examples).encode("utf-8")
        ).hexdigest(),
        "seed": config.seed,
        "config_hash": config.config_hash(),
        "data_revision": PERSONAMEM_REVISION,
        "data_split": config.data.split,
        **runtime_provenance,
        "git_commit": commit or None,
        "embedding_fingerprint": embedding_cache.fingerprint,
        "generation": generate,
        # L0/L1 intentionally use rho² while all S rows share the pointwise
        # source.  Keep both the declared row family and the runtime provider
        # identity so an audit can distinguish a true reranker run from the
        # offline deterministic adapter.
        "quality_source": "mixed_by_architecture",
        "quality_sources": {
            label: "rho2" if quality_source_by_label[label] == "rho2" else pointwise_runtime_source
            for label in labels
        },
        "quality_source_families": quality_source_by_label,
        "quality_groups": {
            "rho2": [label for label in labels if quality_source_by_label[label] == "rho2"],
            "pointwise": [label for label in labels if quality_source_by_label[label] != "rho2"],
        },
        "query_conditioned_representation": {
            "schema": "semantic_query_conditioned_v1",
            "instruction": SEMANTIC_QUERY_CONDITIONED_INSTRUCTION,
            "embedding_model": embedding_cache.model_name,
            "embedding_fingerprint": embedding_cache.fingerprint,
        },
        # S0/S1/S2/S3/shuffle share the current multi-parent frozen pool.  L0/L1
        # are a valid same-pool group only when a real legacy trace is supplied;
        # otherwise they remain explicitly marked as a quality-only control.
        "shared_candidate_pool": True,
        "candidate_pool_groups": {
            "current_multi_parent": [label for label in labels if label not in {"L0", "L1"}],
            "legacy_trace": [label for label in labels if label in {"L0", "L1"}],
        },
        "rows_requested": labels,
        "baseline_requested": baseline_name,
        "baseline_failures": dense_failures,
        "legacy_executor_methods": ["legacy_core", "legacy_path"] if legacy_executor_records else [],
        "legacy_executor_failures": legacy_executor_failures,
        "shared_costs_by_question": shared_costs_by_question,
        "failures": failures,
    }
    (root / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "failures.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in failures), encoding="utf-8"
    )
    dense_end_to_end = [
        float(item.get("outcome", {}).get("answer_accuracy", 0.0) or 0.0)
        for item in dense_predictions
    ]
    dense_successful_answers = [
        float(item["outcome"]["answer_accuracy"])
        for item in dense_predictions
        if item.get("status") == "success"
        and item.get("outcome", {}).get("answer_accuracy") is not None
    ]
    summary = {
        "phase": phase,
        "run_dir": str(root),
        "architectures": summaries,
        "failures": failures,
        "shared_candidate_pool": True,
        "metrics": {
            label: {
                "persona_macro_accuracy": value.get("persona_macro_accuracy"),
                "question_micro_accuracy": value.get("question_micro_accuracy"),
                "Gain": None if value.get("paired_vs_S0") is None else value["paired_vs_S0"].get("Gain"),
                "Damage": None if value.get("paired_vs_S0") is None else value["paired_vs_S0"].get("Damage"),
                "Net": None if value.get("paired_vs_S0") is None else value["paired_vs_S0"].get("Net"),
            }
            for label, value in summaries.items()
        },
        "dense_rerank_comparison": (
            {
                "available": True,
                "status": "incomplete" if dense_failures else "completed",
                "queries": len(dense_predictions),
                "failed_queries": len(dense_failures),
                "answer_accuracy": (
                    sum(dense_end_to_end) / len(dense_end_to_end)
                    if generate and dense_predictions
                    else None
                ),
                "successful_response_accuracy": (
                    sum(dense_successful_answers) / len(dense_successful_answers)
                    if generate and dense_predictions
                    else None
                ),
                "paired_vs_S0": gain_damage_net(baseline_predictions, dense_predictions) if dense_predictions else None,
            }
            if baseline_name == "dense_rerank" and reranker is not None
            else {
                "available": False,
                "status": "not_run",
                "reason": "Dense+Rerank requires an explicit reranker service; no result is synthesized.",
            }
        ),
    }
    (root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (root / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for label in labels:
            for record in predictions_by_label[label]:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    if baseline_name == "dense_rerank":
        (root / "predictions_DenseRerank.jsonl").write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in dense_predictions),
            encoding="utf-8",
        )
    if legacy_executor_records:
        (root / "legacy_executor_predictions.jsonl").write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in legacy_executor_records),
            encoding="utf-8",
        )
    return {"run_dir": str(root), "summary": summary, "manifest": manifest}
