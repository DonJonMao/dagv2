from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import time
from dataclasses import asdict, dataclass, field, replace
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import yaml

from .budget import CostTracker, SearchBudget
from .clients import (
    Embedder,
    GenerationCache,
    GeneratorClient,
    RerankerClient,
    StateEmbeddingCache,
    generation_prompt_hash,
)
from .config import AppConfig, RetrievalConfig
from .experiment import (
    ABLATION_OPTIONS,
    BRIDGE_RERANK_METHODS,
    METHODS,
    RERANK_METHODS,
    EmbeddingCache,
    IndexCache,
    _exact_context_plan,
    _public_app_config,
    _refresh_rerank_selection_diagnostics,
    _runtime_provenance,
    _visible_memory_records,
    load_bridge_gold,
    retrieve_method,
)
from .metrics import (
    answer_accuracy,
    answer_parse_failed,
    bridge_recall_at_k,
    direct_ranks,
    paired_bootstrap_interval,
    recall_at_k,
)
from .module_metrics import (
    MODULE_NAMES,
    aggregate_module_metrics,
    flatten_module_metrics,
    metric_value,
    module_metric_delta,
)
from .module_metrics import collect_module_metrics as collect_metrics
from .personamem import (
    PERSONAMEM_REVISION,
    PERSONAMEM_SOURCE_SHA256,
    PersonaMemExample,
    file_sha256,
    iter_examples,
    messages_to_memories,
)
from .ranking import RerankCache
from .temporal import TransitionCache

DEFAULT_DIAGNOSTIC_METHODS = (
    "bridgetree",
    "ablation_no_cluster",
    "ablation_bfs",
    "ablation_fixed_depth",
    "ablation_topk",
    "ablation_rho_dpp",
    "ablation_direct_path",
)

DEFAULT_MAIN_TABLE_METHODS = (
    "dense",
    "dense_rerank",
    "rfmem_familiarity",
    "rfmem_recollection",
    "rfmem",
    "cluster_prf",
    "bridgetree",
)


@dataclass(frozen=True)
class SplitProtocol:
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15


@dataclass(frozen=True)
class TrainingSchedule:
    max_validation_queries: int | None = None
    max_test_queries: int | None = None


@dataclass(frozen=True)
class SearchSpace:
    initial_width: tuple[int, ...] = (8, 12)
    branch_width: tuple[int, ...] = (4, 8)
    search_budget: tuple[int, ...] = (64,)


FORMAL_32K_SEED = 42
FORMAL_32K_SPLIT = SplitProtocol()
FORMAL_32K_SEARCH_SPACE = SearchSpace(
    initial_width=(8, 12),
    branch_width=(4, 8),
    search_budget=(20, 28, 36, 44),
)
FORMAL_32K_PARTITION_QUERIES = {"train": 432, "validation": 84, "test": 73}
EFFECT_FIRST_VALIDATION_METHODS = (
    "dense_rerank_20",
    "dense_rerank_28",
    "bridgetree_union_rerank",
    "bridgetree_guided_rerank",
    "bridgetree_guided_pathfilter",
    "full_pool_rerank",
)


@dataclass(frozen=True)
class TrainingExperimentConfig:
    seed: int = 42
    split: SplitProtocol = field(default_factory=SplitProtocol)
    schedule: TrainingSchedule = field(default_factory=TrainingSchedule)
    search_space: SearchSpace = field(default_factory=SearchSpace)
    diagnostic_methods: tuple[str, ...] = DEFAULT_DIAGNOSTIC_METHODS
    main_table_methods: tuple[str, ...] = DEFAULT_MAIN_TABLE_METHODS
    objective_metric: str = "auto"
    objective_mode: str = "max"
    bridge_gold_path: str | None = None
    validation_generate: bool = False
    final_generate: bool = False
    fail_on_evaluation_error: bool = True
    keep_example_metrics: bool = True
    output_dir: str = "outputs/training"
    # ``phase`` is optional for backwards compatibility.  A confirmatory
    # protocol explicitly rejects tuning/training before any examples are
    # traversed.
    phase: str = "development"
    # A persisted path is the normal CLI spelling, while accepting an
    # in-memory mapping keeps programmatic/notebook callers from having to
    # round-trip a manifest through a temporary file.  The protocol helpers
    # validate either representation identically.
    protocol_manifest: str | Path | Mapping[str, Any] | None = None

    def validate(self) -> None:
        if self.phase not in {
            "development",
            "development-seen",
            "confirmatory",
            "confirmatory-test",
            "full",
            "full-benchmark",
        }:
            raise ValueError("unknown training protocol phase")
        if self.phase in {"confirmatory", "confirmatory-test"}:
            raise ValueError("confirmatory-test forbids tune/train")
        ratios = (self.split.train_ratio, self.split.validation_ratio, self.split.test_ratio)
        if any(value <= 0.0 for value in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
            raise ValueError("train/validation/test ratios must be positive and sum to 1")
        for name in ("max_validation_queries", "max_test_queries"):
            value = getattr(self.schedule, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when set")
        if (
            not self.search_space.initial_width
            or not self.search_space.branch_width
            or not self.search_space.search_budget
        ):
            raise ValueError("training search space cannot be empty")
        if any(value <= 0 for value in self.search_space.initial_width + self.search_space.branch_width):
            raise ValueError("search widths must be positive")
        if any(value <= 0 for value in self.search_space.search_budget):
            raise ValueError("search budgets must be positive")
        if min(self.search_space.search_budget) < max(self.search_space.initial_width):
            raise ValueError("every search budget must be >= every first-hop width")
        if not self.diagnostic_methods or self.diagnostic_methods[0] != "bridgetree":
            raise ValueError("diagnostic_methods must start with bridgetree")
        unsupported = set(self.diagnostic_methods) - set(ABLATION_OPTIONS)
        if unsupported:
            raise ValueError(f"training diagnostics require tree-producing methods; unsupported: {sorted(unsupported)}")
        unsupported_main = set(self.main_table_methods) - set(METHODS)
        if unsupported_main:
            raise ValueError(f"unsupported main-table methods: {sorted(unsupported_main)}")
        if not self.main_table_methods:
            raise ValueError("main_table_methods cannot be empty")
        if "bridgetree" not in self.main_table_methods:
            raise ValueError("main_table_methods must include bridgetree")
        if self.objective_mode not in {"max", "min"}:
            raise ValueError("objective_mode must be max or min")
        external_metrics = {
            "auto",
            "outcome.answer_accuracy",
            "outcome.recall_at_k",
            "outcome.bridge_recall_at_k",
        }
        if self.objective_metric not in external_metrics:
            raise ValueError("tuning objective must be auto or an external outcome metric")
        if self.objective_metric == "outcome.answer_accuracy" and not self.validation_generate:
            raise ValueError("answer-accuracy tuning requires validation_generate=true")
        if self.validation_generate and not self.final_generate:
            raise ValueError("validation_generate=true requires final_generate=true")
        if self.objective_metric in {"outcome.recall_at_k", "outcome.bridge_recall_at_k"} and not self.bridge_gold_path:
            raise ValueError("recall tuning requires bridge_gold_path")
        has_external_objective = (
            self.objective_metric != "auto" or self.validation_generate or self.bridge_gold_path is not None
        )
        if has_external_objective and not self.fail_on_evaluation_error:
            raise ValueError("external-outcome tuning requires fail_on_evaluation_error=true")


@dataclass(frozen=True)
class ExampleSplits:
    train: tuple[PersonaMemExample, ...]
    validation: tuple[PersonaMemExample, ...]
    test: tuple[PersonaMemExample, ...]
    train_personas: tuple[str, ...]
    validation_personas: tuple[str, ...]
    test_personas: tuple[str, ...]


def _tuple_ints(value: Any, default: tuple[int, ...]) -> tuple[int, ...]:
    values = default if value is None else tuple(int(item) for item in value)
    return tuple(sorted(set(values)))


def load_training_config(path: str | Path) -> TrainingExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("training configuration root must be a mapping")
    split_raw = raw.get("split", {})
    schedule_raw = raw.get("schedule", {})
    search_raw = raw.get("search_space", {})
    initial_width_raw = search_raw.get("initial_width", search_raw.get("first_hop_width"))
    config = TrainingExperimentConfig(
        seed=int(raw.get("seed", 42)),
        split=SplitProtocol(**split_raw),
        schedule=TrainingSchedule(**schedule_raw),
        search_space=SearchSpace(
            initial_width=_tuple_ints(initial_width_raw, (8, 12)),
            branch_width=_tuple_ints(search_raw.get("branch_width"), (4, 8)),
            search_budget=_tuple_ints(search_raw.get("search_budget"), (64,)),
        ),
        diagnostic_methods=tuple(raw.get("diagnostic_methods", DEFAULT_DIAGNOSTIC_METHODS)),
        main_table_methods=tuple(raw.get("main_table_methods", DEFAULT_MAIN_TABLE_METHODS)),
        objective_metric=str(raw.get("objective_metric", "auto")),
        objective_mode=str(raw.get("objective_mode", "max")),
        bridge_gold_path=raw.get("bridge_gold_path"),
        validation_generate=bool(raw.get("validation_generate", False)),
        final_generate=bool(raw.get("final_generate", False)),
        fail_on_evaluation_error=bool(raw.get("fail_on_evaluation_error", True)),
        keep_example_metrics=bool(raw.get("keep_example_metrics", True)),
        output_dir=str(raw.get("output_dir", "outputs/training")),
        phase=str(raw.get("phase", "development")),
        protocol_manifest=raw.get("protocol_manifest"),
    )
    config.validate()
    return config


def _stable_key(seed: int, namespace: str, value: str) -> str:
    return hashlib.sha256(f"{seed}:{namespace}:{value}".encode("utf-8")).hexdigest()


def split_examples_by_persona(
    examples: Sequence[PersonaMemExample], protocol: SplitProtocol, seed: int
) -> ExampleSplits:
    personas = sorted(
        {example.persona_id for example in examples}, key=lambda value: _stable_key(seed, "persona", value)
    )
    if len(personas) < 3:
        raise ValueError("persona-disjoint train/validation/test splitting requires at least 3 personas")
    train_count = max(1, int(round(len(personas) * protocol.train_ratio)))
    validation_count = max(1, int(round(len(personas) * protocol.validation_ratio)))
    if train_count + validation_count >= len(personas):
        train_count = max(1, len(personas) - validation_count - 1)
    test_count = len(personas) - train_count - validation_count
    if test_count < 1:
        validation_count = max(1, validation_count - (1 - test_count))
        test_count = len(personas) - train_count - validation_count
    train_personas = tuple(sorted(personas[:train_count]))
    validation_personas = tuple(sorted(personas[train_count : train_count + validation_count]))
    test_personas = tuple(sorted(personas[train_count + validation_count :]))
    persona_sets = (set(train_personas), set(validation_personas), set(test_personas))

    def select(persona_set: set[str], phase: str) -> tuple[PersonaMemExample, ...]:
        selected = [example for example in examples if example.persona_id in persona_set]
        return tuple(sorted(selected, key=lambda item: _stable_key(seed, phase, item.question_id)))

    return ExampleSplits(
        train=select(persona_sets[0], "train"),
        validation=select(persona_sets[1], "validation"),
        test=select(persona_sets[2], "test"),
        train_personas=train_personas,
        validation_personas=validation_personas,
        test_personas=test_personas,
    )


def _limited(examples: Sequence[PersonaMemExample], limit: int | None) -> tuple[PersonaMemExample, ...]:
    return tuple(examples[:limit]) if limit is not None else tuple(examples)


def _question_id_sha256(question_ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(question_ids).encode("utf-8")).hexdigest()


def build_retrieval_trials(base: RetrievalConfig, search_space: SearchSpace) -> list[RetrievalConfig]:
    trials = []
    for first_hop, branch_width, budget in product(
        search_space.initial_width,
        search_space.branch_width,
        search_space.search_budget,
    ):
        candidate = replace(
            base,
            initial_width=first_hop,
            branch_width=branch_width,
            search_budget=budget,
        )
        candidate.validate()
        trials.append(candidate)
    return trials


class TrainingMetricsWriter:
    def __init__(self, root: Path, keep_examples: bool):
        self.root = root
        self.keep_examples = keep_examples
        self.started_at = time.time()
        self.failure_count = 0
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / "modules").mkdir()
        self.event_id = 0
        with (self.root / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(
                ["event_id", "phase", "trial", "step", "method", "queries", "module", "metric", "value"]
            )

    def write_json(self, name: str, value: Any) -> None:
        destination = self.root / name
        temporary = destination.with_name(destination.name + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(destination)

    def write_progress(self, value: Mapping[str, Any], *, status: str = "running") -> None:
        progress = {**value, "status": status, "updated_at": time.time()}
        self.write_json("progress.json", progress)

    def write_run_status(self, status: str, **values: Any) -> None:
        self.write_json(
            "run_status.json",
            {
                "status": status,
                "started_at": self.started_at,
                "updated_at": time.time(),
                "failure_count": self.failure_count,
                **values,
            },
        )

    def write_example(self, context: Mapping[str, Any], metrics: Mapping[str, Any]) -> None:
        if not self.keep_examples:
            return
        with (self.root / "example_metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({**context, "modules": metrics}, ensure_ascii=False) + "\n")

    def write_event(
        self,
        phase: str,
        trial: int,
        step: int,
        method: str,
        summary: Mapping[str, Any],
        retrieval_config: RetrievalConfig,
    ) -> Dict[str, Any]:
        self.event_id += 1
        retrieval = asdict(retrieval_config)
        retrieval_payload = json.dumps(retrieval, sort_keys=True, separators=(",", ":"))
        event = {
            "event_id": self.event_id,
            "timestamp": time.time(),
            "phase": phase,
            "trial": trial,
            "step": step,
            "method": method,
            "retrieval": retrieval,
            "retrieval_config_hash": hashlib.sha256(retrieval_payload.encode("utf-8")).hexdigest(),
            "metrics": summary,
        }
        with (self.root / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        with (self.root / "metrics.csv").open("a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            for path, value in flatten_module_metrics(summary.get("modules", {})).items():
                module, metric = path.split(".", 1)
                writer.writerow(
                    [self.event_id, phase, trial, step, method, summary.get("queries", 0), module, metric, value]
                )
        for module in MODULE_NAMES:
            record = {
                "event_id": self.event_id,
                "phase": phase,
                "trial": trial,
                "step": step,
                "method": method,
                "queries": summary.get("queries", 0),
                "metrics": summary.get("modules", {}).get(module, {}),
            }
            with (self.root / "modules" / f"{module}.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        outcome_preview = summary.get("modules", {}).get("outcome", {})
        cost_preview = summary.get("modules", {}).get("cost", {}).get("ann_calls_core")
        print(
            f"[BridgeTree tune] phase={phase} trial={trial} step={step} method={method} "
            f"queries={summary.get('queries', 0)} outcome={outcome_preview} ann_calls_core={cost_preview}"
        )
        return event

    def write_comparison(self, context: Mapping[str, Any], value: Mapping[str, Any]) -> None:
        with (self.root / "module_effects.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({**context, "full_minus_ablation": value}, ensure_ascii=False) + "\n")

    def write_failure(self, context: Mapping[str, Any], exc: Exception) -> None:
        self.failure_count += 1
        failure = {
            **context,
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        with (self.root / "failures.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
        self.write_progress({**context, "failure_count": self.failure_count, "last_failure": failure}, status="failed")
        self.write_run_status("failed", last_failure=failure)


class TrainingEvaluator:
    def __init__(
        self,
        app_config: AppConfig,
        embedder: Embedder,
        writer: TrainingMetricsWriter,
        bridge_gold: Mapping[str, Sequence[str]],
    ):
        self.app_config = app_config
        self.cache = EmbeddingCache(
            app_config.runtime.cache_dir,
            embedder,
            app_config.models.embedding.model,
            asdict(app_config.models.embedding),
        )
        self.writer = writer
        self.bridge_gold = bridge_gold
        self.generator = GeneratorClient(app_config.models.generator)
        self.reranker = RerankerClient(app_config.models.reranker)
        self.rerank_cache = RerankCache(
            getattr(
                app_config.models.reranker,
                "cache_dir",
                str(Path(app_config.runtime.cache_dir) / "rerank"),
            ),
            endpoint=app_config.models.reranker.endpoint,
            model=app_config.models.reranker.model,
            score_space=getattr(app_config.models.reranker, "score_space", "unit_interval"),
            task_instruction=getattr(app_config.models.reranker, "task_instruction", ""),
            score_contract=getattr(app_config.models.reranker, "score_contract", "pointwise"),
            model_fingerprint=getattr(app_config.models.reranker, "model_fingerprint", "")
            or getattr(app_config.models.reranker, "model", ""),
        )
        self.index_cache = IndexCache()
        self.state_embedding_cache = StateEmbeddingCache(Path(app_config.runtime.cache_dir) / "state")
        self.transition_cache = TransitionCache(Path(app_config.runtime.cache_dir) / "transition")
        self.generation_cache = GenerationCache(Path(app_config.runtime.cache_dir) / "generation")
        self.example_artifacts: List[Dict[str, Any]] = []

    def evaluate_one(
        self,
        example: PersonaMemExample,
        method: str,
        generate: bool,
        context: Mapping[str, Any],
    ) -> Dict[str, Dict[str, float]]:
        total_started = time.perf_counter()
        segment_started = time.perf_counter()
        memories = messages_to_memories(
            example.messages,
            source_prefix=example.question_id,
            include_system_persona=self.app_config.data.include_system_persona,
            memory_granularity=self.app_config.data.memory_granularity,
        )
        memories = _visible_memory_records(example, memories)
        segmentation_seconds = time.perf_counter() - segment_started
        if not memories:
            raise ValueError(f"example has no retrievable memories: {example.question_id}")

        query_started = time.perf_counter()
        query_vector = self.cache.encode_query(example.query)
        query_embedding_seconds = time.perf_counter() - query_started
        document_started = time.perf_counter()
        memory_vectors = self.cache.encode_documents([memory.text for memory in memories])
        document_embedding_seconds = time.perf_counter() - document_started
        context_key = (
            f"{example.shared_context_id}:{example.end_index}:"
            f"{self.app_config.data.memory_granularity}:{self.app_config.data.include_system_persona}"
        )
        index, index_build_ms, _cache_hit = self.index_cache.get(
            context_key,
            self.cache.fingerprint,
            self.app_config.retrieval.index_backend,
            [memory.memory_id for memory in memories],
            memory_vectors,
            self.app_config.retrieval.faiss_exclusion_margin,
        )
        tracker = CostTracker(SearchBudget.from_config(self.app_config.retrieval))
        retrieval_started = time.perf_counter()
        selected_ids, selected, diagnostics, bridge_result = retrieve_method(
            method,
            self.app_config,
            example,
            memories,
            query_vector,
            memory_vectors,
            reranker=self.reranker if method in RERANK_METHODS else None,
            index=index,
            budget=tracker.budget,
            cost_tracker=tracker,
            index_build_ms=index_build_ms,
            embedding_cache=self.cache,
            rerank_cache=self.rerank_cache if method in RERANK_METHODS else None,
            transition_cache=self.transition_cache,
            state_embedding_cache=self.state_embedding_cache,
        )
        retrieval_seconds = time.perf_counter() - retrieval_started
        tracker = diagnostics.pop("_cost_tracker")
        # Freeze one exact reader request for every method.  The selector's
        # complete cardinality result is retained; an over-budget request is
        # an explicit evaluation failure rather than a silent post-hoc filter.
        context_plan = _exact_context_plan(self.app_config, example, selected_ids, selected)
        if not context_plan.within_budget:
            raise ValueError("selected context exceeds the configured generator token budget")
        memory_by_id = {str(memory.memory_id): memory for memory in memories}
        selected_ids = list(context_plan.selected_ids)
        selected = [memory_by_id[memory_id] for memory_id in context_plan.chronological_ids]
        if bridge_result is not None:
            bridge_result.context_plan = context_plan
            bridge_result.context_hash = context_plan.context_hash
            bridge_result.selected_context = tuple(context_plan.chronological_ids)
            bridge_result.diagnostics["context_plan"] = context_plan.public_dict()
        diagnostics["context_plan"] = context_plan.public_dict()
        diagnostics["context_hash"] = context_plan.context_hash
        _refresh_rerank_selection_diagnostics(diagnostics, selected_ids)
        tracker.final_context_count = len(context_plan.chronological_ids)
        tracker.final_context_tokens = context_plan.token_count

        response = ""
        generation_seconds = 0.0
        accuracy = None
        parse_failure = None
        if generate:
            generation_started = time.perf_counter()
            # ``GenerationCache.answer`` is the compatibility wrapper around
            # the same frozen plan.  Supply the selector order explicitly so
            # its rebuilt plan has the identical selected-ID identity; the
            # adapter itself receives the plan's chronological reader order.
            greedy_memories = [memory_by_id[memory_id] for memory_id in selected_ids]
            response, generation_hit = self.generation_cache.answer(
                self.generator,
                example.query,
                greedy_memories,
                example.all_options,
                selected_ids=selected_ids,
            )
            generation_seconds = time.perf_counter() - generation_started
            tracker.generation_ms = 0.0 if generation_hit else generation_seconds * 1000.0
            if generation_hit:
                tracker.record_cache_hit()
            accuracy = answer_accuracy(response, example.correct_answer)
            parse_failure = answer_parse_failed(response)

        recall = None
        bridge_recall = None
        gold_ids = self.bridge_gold.get(example.question_id, ())
        if gold_ids:
            ranks = direct_ranks(query_vector, [memory.memory_id for memory in memories], memory_vectors)
            recall = recall_at_k(selected_ids, gold_ids, self.app_config.retrieval.context_size)
            bridge_recall = bridge_recall_at_k(
                selected_ids,
                gold_ids,
                ranks,
                self.app_config.retrieval.context_size,
            )
        timings = {
            "segmentation_seconds": segmentation_seconds,
            "query_embedding_seconds": query_embedding_seconds,
            "document_embedding_seconds": document_embedding_seconds,
            "retrieval_seconds": retrieval_seconds,
            "generation_seconds": generation_seconds,
            "total_seconds": time.perf_counter() - total_started,
        }
        if bridge_result is not None:
            metrics = collect_metrics(
                bridge_result,
                query_vector,
                memory_vectors,
                self.app_config.retrieval,
                timings=timings,
                answer_accuracy_value=accuracy,
                recall_value=recall,
                bridge_recall_value=bridge_recall,
                parse_failure_value=parse_failure,
                gold_ids=gold_ids if gold_ids else None,
            )
        else:
            metrics = {name: {} for name in MODULE_NAMES}
            metrics["encoding"] = {
                "memory_count": float(len(memories)),
                "embedding_dimension": float(memory_vectors.shape[1]),
            }
            metrics["selection"] = {"selected_count": float(len(selected_ids))}
            if "selected_bridge_count" in diagnostics:
                metrics["selection"].update(
                    {
                        "selected_bridge_count": float(diagnostics["selected_bridge_count"]),
                        "selected_bridge_rate": float(diagnostics["selected_bridge_rate"]),
                        "dense_top5_retention": float(diagnostics["dense_rerank_top5_retention"]),
                        "mean_final_rerank_score": (
                            sum(
                                float(diagnostics["final_rerank_scores"][memory_id])
                                for memory_id in selected_ids
                                if memory_id in diagnostics.get("final_rerank_scores", {})
                            )
                            / max(
                                1,
                                sum(
                                    memory_id in diagnostics.get("final_rerank_scores", {})
                                    for memory_id in selected_ids
                                ),
                            )
                            if diagnostics.get("final_rerank_scores") and selected_ids
                            else 0.0
                        ),
                    }
                )
            if "candidate" in metrics and "candidate_union_ids" in diagnostics:
                dense_ids = diagnostics.get("dense_pool_ids", [])
                bridge_raw = diagnostics.get("bridge_raw_ids", [])
                bridge_kept = diagnostics.get("bridge_kept_ids", [])
                union_ids = diagnostics.get("candidate_union_ids", [])
                metrics["candidate"] = {
                    "dense_pool_count": float(len(dense_ids)),
                    "anchor_count": float(len(diagnostics.get("anchor_ids", []))),
                    "bridge_raw_count": float(len(bridge_raw)),
                    "bridge_kept_count": float(len(bridge_kept)),
                    "union_count": float(len(union_ids)),
                    "bridge_novelty_rate": float(diagnostics.get("bridge_candidate_novelty", 0.0)),
                }
            metrics["search"] = {
                "ann_calls": float(tracker.ann_calls_core),
                "visited_nodes": float(tracker.cost_unique_count),
            }
            metrics["cost"] = {
                key: float(value)
                for key, value in tracker.snapshot().to_dict().items()
                if key != "stop_reason" and isinstance(value, (int, float))
            }
            metrics["outcome"] = {}
            if accuracy is not None:
                metrics["outcome"]["answer_accuracy"] = accuracy
            if recall is not None:
                metrics["outcome"]["recall_at_k"] = recall
            if bridge_recall is not None:
                metrics["outcome"]["bridge_recall_at_k"] = bridge_recall
            if parse_failure is not None:
                metrics["outcome"]["parse_failure_rate"] = parse_failure
            metrics["timing"] = timings
        self.writer.write_example(
            {
                **context,
                "method": method,
                "persona_id": example.persona_id,
                "question_id": example.question_id,
                "selected_memory_ids": selected_ids,
                "context_plan": context_plan.public_dict(),
                "response": response,
                "retrieval_diagnostics": diagnostics,
            },
            metrics,
        )
        cost_snapshot = tracker.snapshot().to_dict()
        self.example_artifacts.append(
            {
                **context,
                "method": method,
                "persona_id": example.persona_id,
                "question_id": example.question_id,
                "selected_memory_ids": list(selected_ids),
                "context_plan": context_plan.public_dict(),
                "candidate_union_ids": list(diagnostics.get("candidate_union_ids", selected_ids)),
                "outcome": dict(metrics.get("outcome", {})),
                "cost": cost_snapshot,
                **{
                    name: cost_snapshot[name]
                    for name in (
                        "rerank_calls",
                        "rerank_documents",
                        "rerank_ms",
                        "bridge_embedding_calls",
                        "bridge_embedding_queries",
                        "bridge_embedding_ms",
                    )
                },
                "response": response,
                "retrieval_diagnostics": diagnostics,
                **{
                    key: diagnostics[key]
                    for key in (
                        "dense_pool_ids",
                        "anchor_ids",
                        "bridge_raw_ids",
                        "bridge_kept_ids",
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
            }
        )
        return metrics

    def evaluate_set(
        self,
        examples: Sequence[PersonaMemExample],
        method: str,
        generate: bool,
        context: Mapping[str, Any],
        *,
        fail_on_error: bool,
    ) -> Dict[str, Any]:
        records = []
        artifact_start = len(self.example_artifacts)
        attempted_question_ids = []
        successful_question_ids = []
        for evaluation_index, example in enumerate(examples, start=1):
            attempted_question_ids.append(example.question_id)
            try:
                records.append(self.evaluate_one(example, method, generate, context))
                successful_question_ids.append(example.question_id)
            except Exception as exc:
                self.writer.write_failure(
                    {
                        **context,
                        "method": method,
                        "question_id": example.question_id,
                        "evaluation_index": len(attempted_question_ids),
                        "evaluation_queries": len(examples),
                        "successful_queries": len(successful_question_ids),
                        "failed_queries": len(attempted_question_ids) - len(successful_question_ids),
                        "last_question_id": example.question_id,
                        "successful_queries_before_failure": len(successful_question_ids),
                    },
                    exc,
                )
                if fail_on_error:
                    raise RuntimeError(
                        f"{method} failed on evaluation query {example.question_id} "
                        f"({len(attempted_question_ids)}/{len(examples)} attempted); "
                        "formal tuning requires zero failed queries"
                    ) from exc
            if evaluation_index == 1 or evaluation_index % 10 == 0 or evaluation_index == len(examples):
                progress = {
                    **context,
                    "method": method,
                    "evaluation_index": evaluation_index,
                    "evaluation_queries": len(examples),
                    "successful_queries": len(successful_question_ids),
                    "failed_queries": evaluation_index - len(successful_question_ids),
                    "last_question_id": example.question_id,
                }
                self.writer.write_progress(progress)
                print(
                    f"[BridgeTree tune] progress phase={context.get('phase')} "
                    f"trial={context.get('trial')} method={method} "
                    f"queries={evaluation_index}/{len(examples)} failures={progress['failed_queries']}",
                    flush=True,
                )
        summary = aggregate_module_metrics(records)
        attempted_queries = len(attempted_question_ids)
        successful_queries = len(successful_question_ids)
        failed_queries = attempted_queries - successful_queries
        summary.update(
            {
                "attempted_queries": attempted_queries,
                "successful_queries": successful_queries,
                "failed_queries": failed_queries,
                "failure_rate": failed_queries / attempted_queries if attempted_queries else 0.0,
                "attempted_question_id_sha256": _question_id_sha256(attempted_question_ids),
                "successful_question_id_sha256": _question_id_sha256(successful_question_ids),
            }
        )
        outcome_names = sorted({name for record in records for name in record.get("outcome", {})})
        summary["outcome_values"] = {
            name: [float(record["outcome"][name]) for record in records if name in record.get("outcome", {})]
            for name in outcome_names
        }
        summary["outcome_by_question"] = {
            name: {
                question_id: float(record["outcome"][name])
                for question_id, record in zip(successful_question_ids, records)
                if name in record.get("outcome", {})
            }
            for name in outcome_names
        }
        artifacts = self.example_artifacts[artifact_start:]
        summary["selected_ids_by_question"] = {
            str(record["question_id"]): list(record["selected_memory_ids"]) for record in artifacts
        }
        summary["candidate_union_ids_by_question"] = {
            str(record["question_id"]): list(record["candidate_union_ids"]) for record in artifacts
        }
        return summary


def _read_examples(app_config: AppConfig) -> list[PersonaMemExample]:
    raw_root = Path(app_config.data.raw_dir)
    question_path = raw_root / f"questions_{app_config.data.split}.csv"
    context_path = raw_root / f"shared_contexts_{app_config.data.split}.jsonl"
    if not question_path.exists() or not context_path.exists():
        raise FileNotFoundError("PersonaMem raw data is missing; run `bridgetree download-personamem` first")
    return list(iter_examples(question_path, context_path))


def _split_manifest(
    splits: ExampleSplits,
    seed: int,
    *,
    protocol_role: str | None = None,
    internal_split: bool = False,
) -> Dict[str, Any]:
    def describe(examples: Sequence[PersonaMemExample], personas: Sequence[str]) -> Dict[str, Any]:
        ids = [example.question_id for example in examples]
        return {
            "queries": len(ids),
            "personas": list(personas),
            "question_id_sha256": _question_id_sha256(ids),
        }

    manifest = {
        "seed": seed,
        "unit": "persona",
        "train": describe(splits.train, splits.train_personas),
        "validation": describe(splits.validation, splits.validation_personas),
        "test": describe(splits.test, splits.test_personas),
    }
    if protocol_role is not None:
        # ``development-seen`` is a persisted outer role (old validation +
        # test personas).  Tuning may still use a persona-disjoint *internal*
        # split inside that role; naming it explicitly prevents readers from
        # mistaking the derived split for a redefinition of the frozen
        # confirmatory partition.
        manifest["protocol_role"] = str(protocol_role)
        manifest["internal_split"] = bool(internal_split)
        manifest["outer_role_question_count"] = sum(
            section["queries"] for section in (
                manifest["train"],
                manifest["validation"],
                manifest["test"],
            )
        )
        manifest["outer_role_personas"] = sorted(
            {
                *manifest["train"]["personas"],
                *manifest["validation"]["personas"],
                *manifest["test"]["personas"],
            }
        )
    return manifest


def _resolved_objective(config: TrainingExperimentConfig) -> str | None:
    if config.objective_metric != "auto":
        return config.objective_metric
    if config.validation_generate:
        return "outcome.answer_accuracy"
    if config.bridge_gold_path:
        return "outcome.recall_at_k"
    return None


def _optional_metric(summary: Mapping[str, Any], path: str | None) -> float | None:
    if path is None:
        return None
    try:
        return metric_value(summary, path)
    except KeyError:
        return None


def _cost_tuple(summary: Mapping[str, Any]) -> tuple[float, float, float]:
    return (
        metric_value(summary, "cost.ann_calls_core"),
        metric_value(summary, "cost.candidates_returned"),
        metric_value(summary, "cost.retrieval_core_ms"),
    )


def _is_better(
    score: float,
    cost: tuple[float, ...],
    best_score: float | None,
    best_cost: tuple[float, ...] | None,
    mode: str,
    values: Sequence[float] = (),
    best_values: Sequence[float] = (),
) -> bool:
    """Select by the validation point estimate; use cost only for an exact tie.

    Per-query values remain accepted for compatibility with older callers, but
    bootstrap uncertainty is report-only and must not alter the incumbent.
    """
    del values, best_values
    if best_score is None:
        return True
    if (mode == "max" and score > best_score + 1e-12) or (mode == "min" and score < best_score - 1e-12):
        return True
    return abs(score - best_score) <= 1e-12 and (best_cost is None or cost < best_cost)


def _cost_pareto_frontier(trials: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    points = []
    for trial in trials:
        summary = trial["validation"]["bridgetree"]
        points.append((trial, _cost_tuple(summary)))
    frontier = []
    for trial, cost in points:
        dominated = any(
            all(other_value <= value for other_value, value in zip(other_cost, cost))
            and any(other_value < value for other_value, value in zip(other_cost, cost))
            for other_trial, other_cost in points
            if other_trial["trial"] != trial["trial"]
        )
        if not dominated:
            frontier.append(
                {
                    "trial": trial["trial"],
                    "retrieval": trial["retrieval"],
                    "cost": {
                        "ann_calls_core": cost[0],
                        "candidates_returned": cost[1],
                        "retrieval_core_ms": cost[2],
                    },
                }
            )
    return frontier


def preflight_tuning(
    app_config: AppConfig,
    training_config: TrainingExperimentConfig,
    *,
    embedder: Embedder | None = None,
    check_services: bool = False,
    require_full_32k: bool = False,
    phase: str | None = None,
    protocol_manifest: str | Path | Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Validate data, protocol, search space, and optionally every required service."""
    app_config.validate()
    training_config.validate()
    effective_phase = phase or training_config.phase
    # Persisted protocol roles use canonical names.  Keep the historical
    # no-manifest ``development`` mode intact: a number of callers use it for
    # ordinary local tuning and it predates the confirmatory protocol.  Once a
    # manifest is supplied, however, the user-facing alias is resolved to the
    # persisted role name so that the resulting artifacts cannot be confused
    # with an unscoped split.
    from .protocol import canonical_phase

    # Use an explicit ``is not None`` check: an in-memory manifest is a valid
    # value even though a malformed/empty mapping is false-y and should be
    # reported by the protocol audit rather than silently replaced by the
    # config-level value.
    effective_manifest = protocol_manifest if protocol_manifest is not None else training_config.protocol_manifest
    if effective_manifest is None and effective_phase == "development":
        canonical_effective_phase = "development"
    else:
        canonical_effective_phase = canonical_phase(effective_phase)
    if canonical_effective_phase == "confirmatory-test":
        raise PermissionError("confirmatory-test forbids tune/train")
    if effective_manifest is not None:
        from .protocol import protocol_gate
        protocol_gate_kwargs = {
            "manifest": effective_manifest if isinstance(effective_manifest, Mapping) else None,
            "manifest_path": effective_manifest if not isinstance(effective_manifest, Mapping) else None,
            "config_hash": app_config.config_hash(),
            "action": "tune",
        }
        protocol_gate(canonical_effective_phase, **protocol_gate_kwargs)
    elif canonical_effective_phase in {"development-seen", "confirmatory-test", "full-benchmark"}:
        raise ValueError(f"phase {canonical_effective_phase} requires a persisted protocol manifest")
    objective_metric = _resolved_objective(training_config)
    if objective_metric is None:
        raise ValueError("tuning preflight requires an external validation objective")

    if require_full_32k:
        if app_config.data.split != "32k":
            raise ValueError("full 32K tuning requires data.split=32k")
        if not app_config.data.include_system_persona:
            raise ValueError("full 32K tuning requires include_system_persona=true")
        if app_config.data.memory_granularity != "user_assistant_pair":
            raise ValueError("full 32K tuning requires memory_granularity=user_assistant_pair")
        if app_config.seed != FORMAL_32K_SEED or training_config.seed != FORMAL_32K_SEED:
            raise ValueError(f"full 32K tuning requires app and tuning seed={FORMAL_32K_SEED}")
        if training_config.split != FORMAL_32K_SPLIT:
            raise ValueError("full 32K tuning requires the pinned 70/15/15 persona split")
        if training_config.search_space != FORMAL_32K_SEARCH_SPACE:
            raise ValueError("full 32K tuning requires the pinned 2x2x4 search space")
        if training_config.schedule.max_validation_queries is not None:
            raise ValueError("full 32K tuning requires max_validation_queries=null")
        if training_config.schedule.max_test_queries is not None:
            raise ValueError("full 32K tuning requires max_test_queries=null")
        if objective_metric != "outcome.answer_accuracy":
            raise ValueError("full 32K tuning requires outcome.answer_accuracy")
        if not training_config.validation_generate or not training_config.final_generate:
            raise ValueError("full 32K answer tuning requires validation_generate=true and final_generate=true")
        if not training_config.fail_on_evaluation_error:
            raise ValueError("full 32K tuning requires fail_on_evaluation_error=true")
        if not training_config.keep_example_metrics:
            raise ValueError("full 32K tuning requires keep_example_metrics=true for completion auditing")
        if training_config.diagnostic_methods != ("bridgetree",):
            raise ValueError("full 32K tuning must isolate diagnostics to bridgetree during search")
        if training_config.main_table_methods != DEFAULT_MAIN_TABLE_METHODS:
            raise ValueError("full 32K tuning requires all seven main-table methods in the pinned order")
        formal_retrieval = asdict(RetrievalConfig())
        actual_retrieval = asdict(app_config.retrieval)
        for tuned_name in ("initial_width", "branch_width", "search_budget"):
            formal_retrieval.pop(tuned_name)
            actual_retrieval.pop(tuned_name)
        # Named semantic-profile fields were added after the pinned legacy
        # tuning protocol.  They are frozen protocol choices, not search axes;
        # compare them separately below rather than rejecting the shipped
        # semantic_path_v1 default as an accidental tuning change.  The
        # feature/selector pair is included here because a named profile may
        # intentionally choose ``cached_memory`` + ``semantic_path_logdet``.
        semantic_fields = {
            "profile",
            "feature_mode",
            "proposal_mode",
            "relation_mode",
            "quality_mode",
            "path_mode",
            "selection_mode",
            "certificate_mode",
            "context_unit",
            "quality_score_space",
            "scorer_fingerprint",
            "proposal_width",
            "certificate_epsilon",
            "context_strict",
            "certificate_domain",
        }
        for semantic_name in semantic_fields:
            formal_retrieval.pop(semantic_name, None)
            actual_retrieval.pop(semantic_name, None)
        if actual_retrieval != formal_retrieval:
            raise ValueError("full 32K tuning requires the pinned retrieval protocol outside the search axes")

    raw_root = Path(app_config.data.raw_dir)
    split = app_config.data.split
    question_path = raw_root / f"questions_{split}.csv"
    context_path = raw_root / f"shared_contexts_{split}.jsonl"
    for path in (question_path, context_path):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"PersonaMem source file is missing or empty: {path}")

    manifest_path = Path(app_config.data.processed_dir) / split / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"prepared PersonaMem manifest is missing: {manifest_path}; run `bridgetree prepare-personamem`"
        )
    with manifest_path.open("r", encoding="utf-8") as handle:
        data_manifest = json.load(handle)
    if data_manifest.get("revision") != PERSONAMEM_REVISION or data_manifest.get("split") != split:
        raise ValueError("prepared PersonaMem manifest revision/split does not match the pinned protocol")
    source_hashes = data_manifest.get("source_sha256", {})
    actual_source_hashes = {
        question_path.name: file_sha256(question_path),
        context_path.name: file_sha256(context_path),
    }
    if require_full_32k and actual_source_hashes != PERSONAMEM_SOURCE_SHA256["32k"]:
        raise ValueError("PersonaMem 32K source does not match the pinned official checksums")
    for filename, actual_hash in actual_source_hashes.items():
        if source_hashes.get(filename) != actual_hash:
            raise ValueError(f"PersonaMem source checksum mismatch: {filename}")

    source_examples = _read_examples(app_config)
    if len(source_examples) != int(data_manifest.get("questions", -1)):
        raise ValueError("PersonaMem parsed question count does not match the prepared manifest")
    all_examples = source_examples
    protocol_report: Dict[str, Any] | None = None
    if effective_manifest is not None:
        from .protocol import audit_protocol, protocol_examples

        manifest_value = (
            effective_manifest
            if isinstance(effective_manifest, Mapping)
            else effective_manifest
        )
        # ``protocol_examples`` performs the persisted-ID completeness check;
        # retain an audit snapshot in the preflight output for reproducibility.
        all_examples = list(protocol_examples(manifest_value, source_examples, canonical_effective_phase))
        protocol_report = audit_protocol(
            manifest_value,
            examples=source_examples,
            raw_dir=app_config.data.raw_dir,
            split=app_config.data.split,
        )
    splits = split_examples_by_persona(all_examples, training_config.split, training_config.seed)
    validation_examples = _limited(splits.validation, training_config.schedule.max_validation_queries)
    test_examples = _limited(splits.test, training_config.schedule.max_test_queries)
    if not splits.train or not validation_examples or not test_examples:
        raise ValueError("persona-disjoint train, validation, and test partitions must all be non-empty")
    trials = build_retrieval_trials(app_config.retrieval, training_config.search_space)
    if require_full_32k:
        observed_counts = {
            "train": len(splits.train),
            "validation": len(validation_examples),
            "test": len(test_examples),
        }
        if len(all_examples) != sum(FORMAL_32K_PARTITION_QUERIES.values()):
            raise ValueError("full 32K tuning requires exactly 589 parsed questions")
        if observed_counts != FORMAL_32K_PARTITION_QUERIES:
            raise ValueError(
                f"full 32K tuning partition counts changed: expected {FORMAL_32K_PARTITION_QUERIES}, "
                f"observed {observed_counts}"
            )

    service_report: Dict[str, Any] = {"checked": False}
    if check_services:
        if embedder is None:
            raise ValueError("check_services=true requires an embedder")
        example = validation_examples[0]
        memories = messages_to_memories(
            example.messages,
            source_prefix=example.question_id,
            include_system_persona=app_config.data.include_system_persona,
            memory_granularity=app_config.data.memory_granularity,
        )
        memories = _visible_memory_records(example, memories)
        if not memories:
            raise ValueError("preflight example has no retrievable memories")
        query_vector = np.asarray(embedder.encode_query(example.query), dtype=np.float64)
        document_vectors = np.asarray(embedder.encode([memories[0].text]), dtype=np.float64)
        if query_vector.ndim != 1 or document_vectors.ndim != 2 or len(document_vectors) != 1:
            raise ValueError("embedding service returned invalid query/document shapes")
        if document_vectors.shape[1] != query_vector.shape[0]:
            raise ValueError("embedding service returned mismatched query/document dimensions")
        if not np.all(np.isfinite(query_vector)) or not np.all(np.isfinite(document_vectors)):
            raise ValueError("embedding service returned non-finite values")
        if np.linalg.norm(query_vector) <= 0.0 or np.linalg.norm(document_vectors[0]) <= 0.0:
            raise ValueError("embedding service returned a zero vector")

        response = GeneratorClient(app_config.models.generator).answer(
            example.query,
            memories[:1],
            example.all_options,
        )
        if not response:
            raise ValueError("generator service returned an empty response")
        if answer_parse_failed(response):
            raise ValueError("generator preflight response does not contain a parseable answer option")

        reranker_checked = False
        if "dense_rerank" in training_config.main_table_methods:
            items = RerankerClient(app_config.models.reranker).rerank(
                example.query,
                [memory.text for memory in memories[:2]],
                1,
            )
            if not items:
                raise ValueError("reranker service returned no results")
            reranker_checked = True
        service_report = {
            "checked": True,
            "embedding_dimension": int(query_vector.shape[0]),
            "generator_answer_parseable": True,
            "reranker_checked": reranker_checked,
        }

    split_report = _split_manifest(
        splits,
        training_config.seed,
        protocol_role=canonical_effective_phase if effective_manifest is not None else None,
        internal_split=effective_manifest is not None,
    )
    validation_count = len(validation_examples)
    test_count = len(test_examples)
    estimated_generator_calls = (
        validation_count * len(trials) if training_config.validation_generate else 0
    ) + (test_count * len(training_config.main_table_methods) if training_config.final_generate else 0)
    return {
        "status": "ready",
        "data": {
            "dataset_revision": PERSONAMEM_REVISION,
            "split": split,
            "questions": len(all_examples),
            "source_sha256": actual_source_hashes,
            "manifest": str(manifest_path),
        },
        "partitions": {
            **split_report,
            "evaluated_validation_queries": validation_count,
            "evaluated_test_queries": test_count,
        },
        "tuning": {
            "objective_metric": objective_metric,
            "trial_count": len(trials),
            "initial_width": list(training_config.search_space.initial_width),
            "branch_width": list(training_config.search_space.branch_width),
            "search_budget": list(training_config.search_space.search_budget),
            "diagnostic_methods": list(training_config.diagnostic_methods),
            "main_table_methods": list(training_config.main_table_methods),
            "fail_on_evaluation_error": training_config.fail_on_evaluation_error,
            "estimated_generator_calls": estimated_generator_calls,
            "phase": canonical_effective_phase,
            "protocol_manifest": (
                dict(effective_manifest)
                if isinstance(effective_manifest, Mapping)
                else (str(effective_manifest) if effective_manifest is not None else None)
            ),
        },
        "services": service_report,
        "protocol": protocol_report,
    }


def run_training_experiment(
    app_config: AppConfig,
    training_config: TrainingExperimentConfig,
    embedder: Embedder,
    examples: Sequence[PersonaMemExample] | None = None,
    output_dir: str | Path | None = None,
) -> Dict[str, Any]:
    """Tune retrieval configuration once per candidate on external validation outcomes.

    Internal diagnostics are never eligible selection objectives. If neither
    generated validation outcomes nor independent gold are available, trials
    remain unranked, no test data is read, and no best config is written.
    """
    training_config.validate()
    from .protocol import canonical_phase

    # Preserve the legacy local-tuning behavior for the default phase when no
    # persisted protocol manifest is supplied.  Canonical role names are used
    # only for manifest-scoped runs.
    if training_config.protocol_manifest is None and training_config.phase == "development":
        canonical_training_phase = "development"
    else:
        canonical_training_phase = canonical_phase(training_config.phase)
    all_examples = list(examples) if examples is not None else _read_examples(app_config)
    protocol_report: Dict[str, Any] | None = None
    if training_config.protocol_manifest is not None:
        from .protocol import audit_protocol, protocol_examples, protocol_gate

        protocol_gate_kwargs = {
            "manifest": (
                training_config.protocol_manifest
                if isinstance(training_config.protocol_manifest, Mapping)
                else None
            ),
            "manifest_path": (
                training_config.protocol_manifest
                if not isinstance(training_config.protocol_manifest, Mapping)
                else None
            ),
            "config_hash": app_config.config_hash(),
            "action": "tune",
        }
        protocol_gate(canonical_training_phase, **protocol_gate_kwargs)
        all_examples = list(
            protocol_examples(training_config.protocol_manifest, all_examples, canonical_training_phase)
        )
        protocol_report = audit_protocol(
            training_config.protocol_manifest,
            examples=examples if examples is not None else None,
            raw_dir=app_config.data.raw_dir,
            split=app_config.data.split,
        )
    elif canonical_training_phase in {"development-seen", "confirmatory-test", "full-benchmark"}:
        raise ValueError(f"phase {canonical_training_phase} requires a persisted protocol manifest")
    splits = split_examples_by_persona(all_examples, training_config.split, training_config.seed)
    validation_examples = _limited(splits.validation, training_config.schedule.max_validation_queries)
    test_examples = _limited(splits.test, training_config.schedule.max_test_queries)
    if not splits.train or not validation_examples or not test_examples:
        raise ValueError("persona-disjoint train, validation, and test partitions must all be non-empty")

    root = Path(output_dir or training_config.output_dir) / f"tune_{time.time_ns()}"
    writer = TrainingMetricsWriter(root, training_config.keep_example_metrics)
    writer.write_run_status("running", run_dir=str(root), phase="initializing")
    writer.write_json("training_config.json", asdict(training_config))
    # Persist only the public configuration.  Credentials and raw transport
    # endpoints are runtime secrets, not reproducibility evidence; the shared
    # helper removes them and records endpoint hashes for identity checks.
    combined_config = {
        "app": _public_app_config(app_config),
        "tuning": asdict(training_config),
        "execution": {"output_dir": str(output_dir or training_config.output_dir)},
    }
    combined_payload = json.dumps(combined_config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    writer.write_json(
        "resolved_config.json",
        {
            "config_hash": hashlib.sha256(combined_payload.encode("utf-8")).hexdigest(),
            "config": combined_config,
        },
    )
    repository_root = Path(__file__).resolve().parents[2]
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    raw_root = Path(app_config.data.raw_dir)
    source_paths = (
        raw_root / f"questions_{app_config.data.split}.csv",
        raw_root / f"shared_contexts_{app_config.data.split}.jsonl",
    )
    source_sha256 = {path.name: file_sha256(path) for path in source_paths if path.is_file()}
    runtime_provenance = _runtime_provenance(app_config.data.raw_dir, app_config.data.split)
    writer.write_json(
        "run_manifest.json",
        {
            "command": "tune",
            "optimization_kind": "training_free_configuration_tuning",
            "git_commit": git_commit or None,
            "data_revision": PERSONAMEM_REVISION,
            "data_split": app_config.data.split,
            "source_sha256": source_sha256,
            "source_data_package_hash": runtime_provenance.get("source_data_package_hash"),
            "source_package_hash": runtime_provenance.get("source_package_hash"),
            "worktree_clean": runtime_provenance.get("worktree_clean"),
            "uncommitted_diff_hash": runtime_provenance.get("uncommitted_diff_hash"),
            "dataset_queries": len(all_examples),
            "train_queries_reserved": len(splits.train),
            "validation_queries": len(validation_examples),
            "test_queries": len(test_examples),
            "trial_count": len(build_retrieval_trials(app_config.retrieval, training_config.search_space)),
            "fail_on_evaluation_error": training_config.fail_on_evaluation_error,
            "embedding_model": app_config.models.embedding.model,
            "generator_model": app_config.models.generator.model,
            "prompt_hash": generation_prompt_hash(),
            "seed": training_config.seed,
            "phase": canonical_training_phase,
            "protocol_manifest": training_config.protocol_manifest,
            "protocol": protocol_report,
        },
    )
    (root / "failures.jsonl").touch()
    split_manifest = _split_manifest(
        splits,
        training_config.seed,
        protocol_role=canonical_training_phase if training_config.protocol_manifest is not None else None,
        internal_split=training_config.protocol_manifest is not None,
    )
    writer.write_json("split_manifest.json", split_manifest)
    bridge_gold = load_bridge_gold(training_config.bridge_gold_path)
    trials = build_retrieval_trials(app_config.retrieval, training_config.search_space)
    objective_metric = _resolved_objective(training_config)
    expected_validation_hash = _question_id_sha256([example.question_id for example in validation_examples])

    trial_summaries = []
    best_trial = None
    best_score = None
    best_cost: tuple[float, ...] | None = None
    best_app_config = None
    best_validation_metrics = None
    best_objective_by_question: Dict[str, float] = {}

    for trial_index, retrieval_config in enumerate(trials, start=1):
        current_app = replace(app_config, retrieval=retrieval_config)
        evaluator = TrainingEvaluator(current_app, embedder, writer, bridge_gold)
        validation_by_method: Dict[str, Any] = {}
        for method in training_config.diagnostic_methods:
            validation_summary = evaluator.evaluate_set(
                validation_examples,
                method,
                training_config.validation_generate,
                {"phase": "validation", "trial": trial_index, "step": 0},
                fail_on_error=training_config.fail_on_evaluation_error,
            )
            if training_config.fail_on_evaluation_error and (
                validation_summary["successful_queries"] != len(validation_examples)
                or validation_summary["successful_question_id_sha256"] != expected_validation_hash
            ):
                raise RuntimeError(f"validation method {method} did not evaluate the complete common question set")
            validation_by_method[method] = validation_summary
            writer.write_event(
                "validation",
                trial_index,
                0,
                method,
                validation_summary,
                retrieval_config,
            )
        full_validation = validation_by_method["bridgetree"]
        for method, validation_summary in validation_by_method.items():
            if method == "bridgetree":
                continue
            writer.write_comparison(
                {"phase": "validation", "trial": trial_index, "step": 0, "ablation": method},
                module_metric_delta(full_validation, validation_summary),
            )
        objective = _optional_metric(validation_by_method["bridgetree"], objective_metric)
        objective_name = objective_metric.split(".", 1)[1] if objective_metric else ""
        objective_by_question = (
            validation_by_method["bridgetree"].get("outcome_by_question", {}).get(objective_name, {})
        )
        paired_questions = sorted(set(objective_by_question) & set(best_objective_by_question))
        objective_values = [objective_by_question[question_id] for question_id in paired_questions]
        best_objective_values = [best_objective_by_question[question_id] for question_id in paired_questions]
        cost = _cost_tuple(validation_by_method["bridgetree"])
        trial_summary = {
            "trial": trial_index,
            "retrieval": asdict(retrieval_config),
            "objective": objective,
            "objective_metric": objective_metric,
            "tie_break_cost": {
                "ann_calls_core": cost[0],
                "candidates_returned": cost[1],
                "retrieval_core_ms": cost[2],
            },
            "validation": validation_by_method,
        }
        trial_summaries.append(trial_summary)
        if objective is not None and _is_better(
            objective,
            cost,
            best_score,
            best_cost,
            training_config.objective_mode,
            objective_values,
            best_objective_values,
        ):
            best_trial = trial_index
            best_score = objective
            best_cost = cost
            best_app_config = current_app
            best_validation_metrics = validation_by_method
            best_objective_by_question = objective_by_question
        writer.write_json("trials.json", trial_summaries)

    pareto_frontier = _cost_pareto_frontier(trial_summaries)
    writer.write_json(
        "pareto_frontier.json",
        {
            "quality_axis": objective_metric,
            "note": (
                "Cost-only frontier; configurations are intentionally unranked because no external validation "
                "outcome is available."
                if objective_metric is None
                else "External validation outcome is used for selection; costs are unweighted tie-breakers."
            ),
            "trials": pareto_frontier,
        },
    )
    if best_app_config is None or best_trial is None:
        final_summary = {
            "optimization_kind": "training_free_configuration_tuning",
            "run_dir": str(root),
            "selection_status": "unselected_no_observed_external_validation_outcome",
            "best_trial": None,
            "best_retrieval_config": None,
            "objective_metric": objective_metric,
            "test_metrics": {},
            "trial_count": len(trials),
            "evaluated_validation_queries": len(validation_examples),
            "evaluated_test_queries": 0,
            "pareto_frontier": pareto_frontier,
        }
        writer.write_json("final_summary.json", final_summary)
        writer.write_progress(
            {
                "phase": "completed",
                "run_dir": str(root),
                "selection_status": final_summary["selection_status"],
                "best_trial": None,
                "trial_count": len(trials),
            },
            status="completed",
        )
        writer.write_run_status(
            "completed",
            run_dir=str(root),
            selection_status=final_summary["selection_status"],
            best_trial=None,
        )
        return final_summary
    final_evaluator = TrainingEvaluator(best_app_config, embedder, writer, bridge_gold)
    final_test_by_method: Dict[str, Any] = {}
    expected_test_hash = _question_id_sha256([example.question_id for example in test_examples])
    for method in training_config.main_table_methods:
        test_summary = final_evaluator.evaluate_set(
            test_examples,
            method,
            training_config.final_generate,
            {"phase": "test", "trial": best_trial, "step": 0},
            fail_on_error=training_config.fail_on_evaluation_error,
        )
        if training_config.fail_on_evaluation_error and (
            test_summary["successful_queries"] != len(test_examples)
            or test_summary["successful_question_id_sha256"] != expected_test_hash
        ):
            raise RuntimeError(f"test method {method} did not evaluate the complete common question set")
        final_test_by_method[method] = test_summary
        writer.write_event(
            "test",
            best_trial,
            0,
            method,
            test_summary,
            best_app_config.retrieval,
        )

    full_test = final_test_by_method["bridgetree"]
    effects = {
        method: module_metric_delta(full_test, summary)
        for method, summary in final_test_by_method.items()
        if method != "bridgetree"
    }
    final_summary = {
        "optimization_kind": "training_free_configuration_tuning",
        "run_dir": str(root),
        "best_trial": best_trial,
        "best_retrieval_config": asdict(best_app_config.retrieval),
        "objective_metric": objective_metric,
        "objective_mode": training_config.objective_mode,
        "best_validation_objective": best_score,
        "validation_metrics": best_validation_metrics,
        "test_metrics": final_test_by_method,
        "test_module_effects_full_minus_ablation": effects,
        "trial_count": len(trials),
        "evaluated_validation_queries": len(validation_examples),
        "evaluated_test_queries": len(test_examples),
        "common_test_question_id_sha256": expected_test_hash,
        "pareto_frontier": pareto_frontier,
        "selection_status": "selected_on_external_validation_outcome",
    }
    writer.write_json("final_summary.json", final_summary)
    writer.write_json("best_config.json", {"retrieval": asdict(best_app_config.retrieval)})
    writer.write_progress(
        {
            "phase": "completed",
            "run_dir": str(root),
            "selection_status": final_summary["selection_status"],
            "best_trial": best_trial,
            "trial_count": len(trials),
        },
        status="completed",
    )
    writer.write_run_status(
        "completed",
        run_dir=str(root),
        selection_status=final_summary["selection_status"],
        best_trial=best_trial,
    )
    return final_summary


def run_effect_first_validation(
    app_config: AppConfig,
    embedder: Embedder,
    *,
    examples: Sequence[PersonaMemExample] | None = None,
    methods: Sequence[str] = EFFECT_FIRST_VALIDATION_METHODS,
    output_dir: str | Path = "outputs/effect-first-validation",
    limit: int | None = None,
    generate: bool = True,
) -> Dict[str, Any]:
    """Evaluate the predefined effect-first matrix on validation personas only.

    Accuracy point estimates select the reported method. Paired bootstrap is
    descriptive evidence and never changes the selection order.
    """
    unsupported = set(methods) - set(EFFECT_FIRST_VALIDATION_METHODS)
    if unsupported:
        raise ValueError(f"unsupported effect-first methods: {sorted(unsupported)}")
    if not methods:
        raise ValueError("effect-first validation requires at least one method")
    if limit is not None and limit <= 0:
        raise ValueError("effect-first validation limit must be positive")
    if not app_config.models.reranker.endpoint:
        raise ValueError("effect-first validation requires a configured reranker endpoint")
    if (
        app_config.retrieval.stop_mode == "certificate_or_budget"
        and set(methods) & BRIDGE_RERANK_METHODS
    ):
        raise ValueError("effect-first BridgeTree rerank methods cannot use certificate_or_budget")
    app_config.validate()
    all_examples = list(examples) if examples is not None else _read_examples(app_config)
    split = split_examples_by_persona(all_examples, SplitProtocol(), app_config.seed)
    validation_examples = list(split.validation[:limit] if limit is not None else split.validation)
    if not validation_examples:
        raise ValueError("effect-first validation partition is empty")

    root = Path(output_dir) / f"effect_validation_{time.time_ns()}"
    writer = TrainingMetricsWriter(root, keep_examples=True)
    writer.write_run_status("running", run_dir=str(root), phase="validation")
    writer.write_json(
        "resolved_config.json",
        {
            "app": _public_app_config(app_config),
            "execution": {
                "methods": list(methods),
                "partition": "persona_disjoint_validation",
                "validation_queries": len(validation_examples),
                "limit": limit,
                "generate": generate,
            },
        },
    )
    writer.write_json(
        "run_manifest.json",
        {
            "command": "effect-first-validation",
            "status": "running",
            "seed": app_config.seed,
            "data_revision": PERSONAMEM_REVISION,
            "data_split": app_config.data.split,
            "methods": list(methods),
            "validation_queries": len(validation_examples),
            "validation_question_id_sha256": _question_id_sha256(
                [example.question_id for example in validation_examples]
            ),
            "test_queries_read": 0,
            "evaluated_test_queries": 0,
        },
    )
    evaluator = TrainingEvaluator(app_config, embedder, writer, {})
    method_results: Dict[str, Any] = {}
    expected_validation_hash = _question_id_sha256([example.question_id for example in validation_examples])
    current_method = "initializing"
    try:
        for method_index, method in enumerate(methods, start=1):
            current_method = method
            summary = evaluator.evaluate_set(
                validation_examples,
                method,
                generate,
                {"phase": "validation", "trial": 1, "step": method_index, "run_label": method},
                fail_on_error=True,
            )
            if (
                summary["successful_queries"] != len(validation_examples)
                or summary["successful_question_id_sha256"] != expected_validation_hash
            ):
                raise RuntimeError(f"effect-first method {method} did not evaluate the complete validation set")
            method_results[method] = summary
            writer.write_event("validation", 1, method_index, method, summary, app_config.retrieval)
    except Exception as exc:
        if writer.failure_count == 0:
            writer.write_failure(
                {"phase": "validation", "method": current_method, "run_label": current_method},
                exc,
            )
        manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
        writer.write_json(
            "run_manifest.json",
            {**manifest, "status": "failed", "failure_count": writer.failure_count},
        )
        raise

    reference_name = "dense_rerank_28" if "dense_rerank_28" in method_results else methods[0]
    reference_outcomes = method_results[reference_name].get("outcome_by_question", {}).get(
        "answer_accuracy", {}
    )
    paired: Dict[str, Any] = {}
    corrections: Dict[str, Any] = {}
    for method, summary in method_results.items():
        outcomes = summary.get("outcome_by_question", {}).get("answer_accuracy", {})
        common = sorted(set(outcomes) & set(reference_outcomes))
        paired[method] = (
            paired_bootstrap_interval(
                [outcomes[question_id] for question_id in common],
                [reference_outcomes[question_id] for question_id in common],
                seed=app_config.seed,
            )
            if common
            else None
        )
        if method.startswith("bridgetree") and common:
            bridge_correct = sum(outcomes[question_id] > reference_outcomes[question_id] for question_id in common)
            dense_correct = sum(outcomes[question_id] < reference_outcomes[question_id] for question_id in common)
            corrections[method] = {
                "reference": reference_name,
                "paired_queries": len(common),
                "bridge_correct_dense_wrong": bridge_correct,
                "dense_correct_bridge_wrong": dense_correct,
                "bridge_net_correction": bridge_correct - dense_correct,
            }

    full_pool_selected = method_results.get("full_pool_rerank", {}).get("selected_ids_by_question", {})
    full_pool_recall: Dict[str, float | None] = {}
    for method, summary in method_results.items():
        candidate_by_question = summary.get("candidate_union_ids_by_question", {})
        common = sorted(set(candidate_by_question) & set(full_pool_selected))
        recalls = []
        for question_id in common:
            gold = set(full_pool_selected[question_id])
            if gold:
                recalls.append(len(gold & set(candidate_by_question[question_id])) / len(gold))
        full_pool_recall[method] = sum(recalls) / len(recalls) if recalls else None

    selectable = [method for method in methods if method != "full_pool_rerank"]
    ranked_methods = []
    for method in selectable:
        summary = method_results[method]
        try:
            accuracy = metric_value(summary, "outcome.answer_accuracy")
        except KeyError:
            continue
        cost = (
            metric_value(summary, "cost.rerank_documents"),
            metric_value(summary, "cost.ann_calls_core"),
            metric_value(summary, "cost.retrieval_core_ms"),
        )
        ranked_methods.append((method, accuracy, cost))
    ranked_methods.sort(key=lambda item: (-item[1], item[2], item[0]))
    selected_method = ranked_methods[0][0] if ranked_methods else None
    selected_accuracy = ranked_methods[0][1] if ranked_methods else None

    final = {
        "run_dir": str(root),
        "status": "completed",
        "partition": "persona_disjoint_validation",
        "test_queries_read": 0,
        "evaluated_test_queries": 0,
        "validation_queries": len(validation_examples),
        "methods": list(methods),
        "selection_rule": "validation accuracy point estimate; exact ties broken by cost",
        "selected_method": selected_method,
        "selected_validation_accuracy": selected_accuracy,
        "method_results": method_results,
        "paired_vs_dense_rerank_28": paired,
        "bridge_net_correction": corrections,
        "full_pool_top5_recall": full_pool_recall,
    }
    with (root / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for record in evaluator.example_artifacts:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (root / "effect_results.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = (
            "method",
            "validation_queries",
            "correct_answers",
            "answer_accuracy",
            "difference_vs_dense_rerank_28",
            "ci_low",
            "ci_high",
            "bridge_correct_dense_wrong",
            "dense_correct_bridge_wrong",
            "bridge_net_correction",
            "full_pool_top5_recall",
            "rerank_calls",
            "rerank_documents",
            "ann_calls_core",
        )
        table = csv.DictWriter(handle, fieldnames=fieldnames)
        table.writeheader()
        for method in methods:
            summary = method_results[method]
            accuracy = _optional_metric(summary, "outcome.answer_accuracy")
            interval = paired.get(method) or {}
            correction = corrections.get(method, {})
            table.writerow(
                {
                    "method": method,
                    "validation_queries": len(validation_examples),
                    "correct_answers": round(accuracy * len(validation_examples)) if accuracy is not None else "",
                    "answer_accuracy": accuracy if accuracy is not None else "",
                    "difference_vs_dense_rerank_28": interval.get("mean_difference", ""),
                    "ci_low": interval.get("ci_low", ""),
                    "ci_high": interval.get("ci_high", ""),
                    "bridge_correct_dense_wrong": correction.get("bridge_correct_dense_wrong", ""),
                    "dense_correct_bridge_wrong": correction.get("dense_correct_bridge_wrong", ""),
                    "bridge_net_correction": correction.get("bridge_net_correction", ""),
                    "full_pool_top5_recall": full_pool_recall.get(method, ""),
                    "rerank_calls": _optional_metric(summary, "cost.rerank_calls") or 0.0,
                    "rerank_documents": _optional_metric(summary, "cost.rerank_documents") or 0.0,
                    "ann_calls_core": _optional_metric(summary, "cost.ann_calls_core") or 0.0,
                }
            )
    writer.write_json("paired_results.json", {"paired": paired, "bridge_net_correction": corrections})
    writer.write_json("effect_summary.json", final)
    writer.write_progress(
        {
            "phase": "completed",
            "run_dir": str(root),
            "selected_method": selected_method,
            "validation_queries": len(validation_examples),
        },
        status="completed",
    )
    writer.write_run_status(
        "completed",
        run_dir=str(root),
        phase="completed",
        selected_method=selected_method,
    )
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    writer.write_json(
        "run_manifest.json",
        {**manifest, "status": "completed", "failure_count": writer.failure_count},
    )
    return final


# Public tuning names; legacy training names remain import-compatible.
TuningExperimentConfig = TrainingExperimentConfig
load_tuning_config = load_training_config
run_tuning_experiment = run_training_experiment
