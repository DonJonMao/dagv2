from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

from .config import RetrievalConfig
from .math_utils import logdet_value
from .metrics import path_objective_advantage
from .types import RetrievalResult

MODULE_NAMES = (
    "encoding",
    "candidate",
    "coarse_retrieval",
    "clustering",
    "path",
    "innovation",
    "selection",
    "search",
    "cost",
    "outcome",
    "timing",
)


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _max(values: Sequence[float]) -> float:
    return float(np.max(values)) if values else 0.0


def collect_module_metrics(
    result: RetrievalResult,
    query_vector: np.ndarray,
    memory_vectors: np.ndarray,
    config: RetrievalConfig,
    timings: Mapping[str, float] | None = None,
    answer_accuracy_value: float | None = None,
    recall_value: float | None = None,
    bridge_recall_value: float | None = None,
    parse_failure_value: float | None = None,
    gold_ids: Sequence[str] | None = None,
) -> Dict[str, Dict[str, float]]:
    """Collect independent, numeric diagnostics for every BridgeTree module."""
    query = np.asarray(query_vector, dtype=np.float64)
    memories = np.asarray(memory_vectors, dtype=np.float64)
    query_norm = float(np.linalg.norm(query))
    memory_norms = np.linalg.norm(memories, axis=1) if len(memories) else np.asarray([], dtype=np.float64)
    nodes = list(result.nodes.values())
    first_hop_nodes = [node for node in nodes if node.depth == 1]
    first_hop_scores = sorted((node.direct_score for node in first_hop_nodes), reverse=True)
    deep_nodes = [node for node in nodes if node.depth > 1]
    bridge_nodes = [node for node in deep_nodes if node.bridge_lift > 0.0]
    reaches = [node.reachability for node in nodes]
    bridge_lifts = [node.bridge_lift for node in deep_nodes]

    total_branch_members = sum(len(branch.member_ids) for branch in result.all_branches)
    branch_count = len(result.all_branches)
    direction_reduction = 1.0 - branch_count / total_branch_members if total_branch_members else 0.0

    retention = []
    for node in nodes:
        denominator = node.reachability**2
        if denominator > 1e-12:
            retention.append(float(np.dot(node.innovation, node.innovation) / denominator))
    selected_nodes = [result.nodes[memory_id] for memory_id in result.selected_in_greedy_order]
    selected_features = [node.innovation for node in selected_nodes]
    selection_margins = [step.discovered_best_margin for step in result.selection_steps]
    parent_child_cosines = [
        float(np.dot(node.vector, result.nodes[node.parent_id].vector)) for node in nodes if node.parent_id is not None
    ]
    selected_id_set = set(result.selected_in_greedy_order)
    selected_deep = [result.nodes[memory_id] for memory_id in selected_id_set if result.nodes[memory_id].depth > 1]
    navigation_parents = set()
    for memory_id in selected_id_set:
        parent_id = result.nodes[memory_id].parent_id
        while parent_id is not None:
            navigation_parents.add(parent_id)
            parent_id = result.nodes[parent_id].parent_id
    navigation_only = navigation_parents - selected_id_set
    depth_diagnostics: Dict[str, float] = {}
    independent_gold = set(gold_ids or ())
    for depth in sorted({node.depth for node in nodes}):
        depth_nodes = [node for node in nodes if node.depth == depth]
        root_cosines = [node.direct_score for node in depth_nodes]
        depth_diagnostics[f"depth_{depth}_mean_root_cosine"] = _mean(root_cosines)
        depth_diagnostics[f"depth_{depth}_min_root_cosine"] = float(min(root_cosines))
        if gold_ids is not None:
            depth_diagnostics[f"depth_{depth}_gold_rate"] = sum(
                node.memory.memory_id in independent_gold for node in depth_nodes
            ) / len(depth_nodes)

    outcome: Dict[str, float] = {}
    if answer_accuracy_value is not None:
        outcome["answer_accuracy"] = float(answer_accuracy_value)
    if recall_value is not None:
        outcome["recall_at_k"] = float(recall_value)
    if bridge_recall_value is not None:
        outcome["bridge_recall_at_k"] = float(bridge_recall_value)
    if parse_failure_value is not None:
        outcome["parse_failure_rate"] = float(parse_failure_value)

    timing_values = {name: float(value) for name, value in (timings or {}).items()}
    return {
        "encoding": {
            "memory_count": float(len(memories)),
            "embedding_dimension": float(memories.shape[1] if memories.ndim == 2 and len(memories) else len(query)),
            "query_norm": query_norm,
            "mean_memory_norm": _mean([float(value) for value in memory_norms]),
            "nonfinite_value_count": float(
                np.size(memories) - np.isfinite(memories).sum() + np.size(query) - np.isfinite(query).sum()
            ),
        },
        "candidate": {},
        "coarse_retrieval": {
            "first_hop_count": float(len(first_hop_nodes)),
            "max_direct_similarity": _max(first_hop_scores),
            "mean_first_hop_similarity": _mean(first_hop_scores),
            "min_first_hop_similarity": float(min(first_hop_scores)) if first_hop_scores else 0.0,
        },
        "clustering": {
            "probe_count": float(branch_count),
            "actual_cluster_count": float(len(result.cluster_member_counts)),
            "mean_members_per_probe": total_branch_members / branch_count if branch_count else 0.0,
            "direction_reduction_rate": direction_reduction,
            "mean_radius_radians": _mean(result.cluster_radii),
            "max_radius_radians": _max(result.cluster_radii),
            "ranking_stability_rate": _mean(result.cluster_stabilities),
            "clustering_ms": result.clustering_ms,
        },
        "path": {
            "tree_node_count": float(len(nodes)),
            "real_edge_count": float(sum(parent is not None for parent, _child in result.edges)),
            "max_depth": float(max((node.depth for node in nodes), default=0)),
            "deep_node_rate": len(deep_nodes) / len(nodes) if nodes else 0.0,
            "bridge_node_rate": len(bridge_nodes) / len(deep_nodes) if deep_nodes else 0.0,
            "mean_reachability": _mean(reaches),
            "mean_bridge_lift": _mean(bridge_lifts),
            "max_bridge_lift": _max(bridge_lifts),
            "mean_parent_child_cosine": _mean(parent_child_cosines),
            "min_parent_child_cosine": float(min(parent_child_cosines)) if parent_child_cosines else 0.0,
            "selected_deep_node_rate": len(selected_deep) / len(selected_nodes) if selected_nodes else 0.0,
            "navigation_only_parent_rate": (
                len(navigation_only) / len(navigation_parents) if navigation_parents else 0.0
            ),
            **depth_diagnostics,
        },
        "innovation": {
            "mean_retention_ratio": _mean(retention),
            "min_retention_ratio": float(min(retention)) if retention else 0.0,
            "path_objective_advantage": path_objective_advantage(result, config.context_size),
        },
        "selection": {
            "selected_count": float(len(selected_nodes)),
            "logdet_value": logdet_value(selected_features),
            "mean_greedy_margin": _mean(selection_margins),
            "min_greedy_margin": float(min(selection_margins)) if selection_margins else 0.0,
        },
        "search": {
            "ann_calls": float(result.cost.ann_calls_core),
            "proposal_ann_calls": float(result.cost.proposal_ann_calls),
            "candidate_exposure": float(result.cost.candidate_exposure),
            "visited_nodes": float(result.cost.unique_visited_nodes),
            "visited_budget_ratio": result.visited_nodes / config.search_budget,
            "certified_query": float(result.certified),
            "budget_frozen": float(result.budget_frozen),
            "posterior_error": float(result.posterior_error),
            "duplicate_proposal_rate": (
                result.cost.duplicate_proposals / result.cost.proposal_count if result.cost.proposal_count else 0.0
            ),
            "new_unique_candidates_per_ann": result.cost.new_unique_candidates_per_ann,
            "transition_exact_ops": float(result.cost.transition_exact_ops),
            "bound_ops": float(result.cost.bound_ops),
            "cache_hits": float(result.cost.cache_hits),
        },
        "cost": {
            "ann_calls_core": float(result.cost.ann_calls_core),
            "ann_calls_diagnostic": float(result.cost.ann_calls_diagnostic),
            "candidates_returned": float(result.cost.candidates_returned),
            "candidates_returned_diagnostic": float(result.cost.candidates_returned_diagnostic),
            "candidate_exposure": float(result.cost.candidate_exposure),
            "unique_visited_nodes": float(result.cost.unique_visited_nodes),
            "index_build_ms": result.cost.index_build_ms,
            "retrieval_core_ms": result.cost.retrieval_core_ms,
            "diagnostic_ms": result.cost.diagnostic_ms,
            "proposal_ann_calls": float(result.cost.proposal_ann_calls),
            "proposal_count": float(result.cost.proposal_count),
            "duplicate_proposals": float(result.cost.duplicate_proposals),
            "new_unique_candidates_per_ann": float(result.cost.new_unique_candidates_per_ann),
            "rerank_calls": float(result.cost.rerank_calls),
            "rerank_documents": float(result.cost.rerank_documents),
            "rerank_ms": result.cost.rerank_ms,
            "bridge_embedding_calls": float(result.cost.bridge_embedding_calls),
            "bridge_embedding_queries": float(result.cost.bridge_embedding_queries),
            "bridge_embedding_ms": result.cost.bridge_embedding_ms,
            "state_embedding_calls": float(result.cost.state_embedding_calls),
            "state_embedding_queries": float(result.cost.state_embedding_queries),
            "state_embedding_ms": result.cost.state_embedding_ms,
            "transition_exact_ops": float(result.cost.transition_exact_ops),
            "bound_ops": float(result.cost.bound_ops),
            "cache_hits": float(result.cost.cache_hits),
            "generation_ms": result.cost.generation_ms,
            "final_context_count": float(result.cost.final_context_count),
            "final_context_tokens": float(result.cost.final_context_tokens),
        },
        "outcome": outcome,
        "timing": timing_values,
    }


def flatten_module_metrics(metrics: Mapping[str, Mapping[str, float]]) -> Dict[str, float]:
    return {f"{module}.{name}": float(value) for module, values in metrics.items() for name, value in values.items()}


def aggregate_module_metrics(records: Iterable[Mapping[str, Mapping[str, float]]]) -> Dict[str, Any]:
    sums: Dict[str, float] = defaultdict(float)
    observations: Dict[str, int] = defaultdict(int)
    queries = 0
    for record in records:
        queries += 1
        for name, value in flatten_module_metrics(record).items():
            if np.isfinite(value):
                sums[name] += float(value)
                observations[name] += 1
    modules: Dict[str, Dict[str, float]] = {name: {} for name in MODULE_NAMES}
    for path in sorted(sums):
        module, metric = path.split(".", 1)
        modules.setdefault(module, {})[metric] = sums[path] / observations[path]
    return {
        "queries": queries,
        "modules": modules,
        "observations": dict(sorted(observations.items())),
    }


def metric_value(summary: Mapping[str, Any], path: str) -> float:
    module, metric = path.split(".", 1)
    try:
        return float(summary["modules"][module][metric])
    except (KeyError, TypeError, ValueError) as exc:
        raise KeyError(f"metric is unavailable: {path}") from exc


def module_metric_delta(full: Mapping[str, Any], ablation: Mapping[str, Any]) -> Dict[str, Dict[str, float]]:
    full_flat = flatten_module_metrics(full.get("modules", {}))
    ablation_flat = flatten_module_metrics(ablation.get("modules", {}))
    delta: Dict[str, Dict[str, float]] = {name: {} for name in MODULE_NAMES}
    for path in sorted(set(full_flat) & set(ablation_flat)):
        module, metric = path.split(".", 1)
        delta.setdefault(module, {})[metric] = full_flat[path] - ablation_flat[path]
    return delta
