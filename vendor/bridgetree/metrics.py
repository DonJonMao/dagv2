from __future__ import annotations

import re
from collections import defaultdict
from itertools import combinations
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from .index import ExactInnerProductIndex
from .math_utils import logdet_marginal, logdet_value
from .types import Branch, RetrievalResult, TreeNode

_OPTION_PATTERNS = (
    re.compile(r"\(([a-z])\)", re.IGNORECASE),
    re.compile(r"\boption\s+([a-z])\b", re.IGNORECASE),
    re.compile(r"选\s*([a-z])", re.IGNORECASE),
    re.compile(r"\banswer\s*(?:is|:)\s*([a-z])\b", re.IGNORECASE),
    re.compile(r"^\s*([a-z])(?:\s*$|[.):]\s*)", re.IGNORECASE),
)


def extract_option_label(text: str) -> str:
    for pattern in _OPTION_PATTERNS:
        match = pattern.search(text or "")
        if match:
            return f"({match.group(1).lower()})"
    return ""


def answer_accuracy(response: str, gold: str) -> float:
    return float(extract_option_label(response) == extract_option_label(gold) and bool(extract_option_label(gold)))


def answer_parse_failed(response: str) -> float:
    return float(not bool(extract_option_label(response)))


def recall_at_k(selected_ids: Sequence[str], gold_ids: Iterable[str], k: int) -> float:
    gold = set(gold_ids)
    if not gold:
        return 0.0
    return len(set(selected_ids[:k]) & gold) / len(gold)


def direct_ranks(query_vector: np.ndarray, ids: Sequence[str], vectors: np.ndarray) -> Dict[str, int]:
    index = ExactInnerProductIndex(ids, vectors)
    return {memory_id: rank for rank, (memory_id, _score) in enumerate(index.search(query_vector, len(ids)), start=1)}


def bridge_recall_at_k(
    selected_ids: Sequence[str],
    gold_ids: Iterable[str],
    ranks: Mapping[str, int],
    k: int,
) -> float | None:
    # Gold is externally annotated. Direct rank only partitions that gold set;
    # bridge lift is never used to define relevance.
    bridge_gold = {memory_id for memory_id in gold_ids if ranks.get(memory_id, 0) > k}
    return recall_at_k(selected_ids, bridge_gold, k) if bridge_gold else None


def greedy_ids(nodes: Mapping[str, TreeNode], feature_kind: str, k: int) -> List[str]:
    selected: List[str] = []
    selected_features: List[np.ndarray] = []
    while len(selected) < min(k, len(nodes)):
        available = [memory_id for memory_id in nodes if memory_id not in selected]
        features = {
            memory_id: (
                nodes[memory_id].innovation
                if feature_kind == "path_conditioned"
                else nodes[memory_id].reachability * nodes[memory_id].vector
            )
            for memory_id in available
        }
        best = min(
            available,
            key=lambda memory_id: (-logdet_marginal(features[memory_id], selected_features), memory_id),
        )
        selected.append(best)
        selected_features.append(features[best])
    return selected


def path_objective_advantage(result: RetrievalResult, k: int) -> float:
    """Internal fixed-candidate path objective advantage; never a tuning target."""
    path_ids = greedy_ids(result.nodes, "path_conditioned", k)
    rho_ids = greedy_ids(result.nodes, "rho_weighted", k)
    path_value = logdet_value([result.nodes[memory_id].innovation for memory_id in path_ids])
    rho_set_on_path_geometry = logdet_value([result.nodes[memory_id].innovation for memory_id in rho_ids])
    return float(path_value - rho_set_on_path_geometry)


def paired_bootstrap_interval(
    left: Sequence[float],
    right: Sequence[float],
    *,
    seed: int = 42,
    resamples: int = 2000,
    confidence: float = 0.95,
) -> Dict[str, float]:
    if len(left) != len(right) or not left:
        raise ValueError("paired bootstrap requires equally sized non-empty samples")
    if resamples <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid bootstrap settings")
    differences = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sample_positions = rng.integers(0, len(differences), size=(resamples, len(differences)))
    estimates = differences[sample_positions].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean_difference": float(differences.mean()),
        "ci_low": float(np.quantile(estimates, alpha)),
        "ci_high": float(np.quantile(estimates, 1.0 - alpha)),
        "confidence": confidence,
        "resamples": float(resamples),
    }


def branch_ranking_stability(
    branch: Branch,
    index: ExactInnerProductIndex,
    candidate_ids: Sequence[str] | None = None,
) -> float:
    """Fraction of radius-qualified candidate pairs whose order is preserved.

    This directly operationalizes Lemma 3.2: only pairs whose member-distance
    gap exceeds 2*r_c enter the denominator.
    """
    candidates = list(candidate_ids or index.ids)
    if len(candidates) < 2:
        return 1.0
    candidate_vectors = {memory_id: index.vector(memory_id) for memory_id in candidates}
    probe_distances = {
        memory_id: float(np.arccos(np.clip(np.dot(branch.probe, vector), -1.0, 1.0)))
        for memory_id, vector in candidate_vectors.items()
    }
    stable = 0
    eligible = 0
    for member_id in branch.member_ids:
        member = index.vector(member_id)
        member_distances = {
            memory_id: float(np.arccos(np.clip(np.dot(member, vector), -1.0, 1.0)))
            for memory_id, vector in candidate_vectors.items()
        }
        for left, right in combinations(candidates, 2):
            member_gap = member_distances[left] - member_distances[right]
            if abs(member_gap) <= 2.0 * branch.radius_radians:
                continue
            eligible += 1
            probe_gap = probe_distances[left] - probe_distances[right]
            stable += int(member_gap * probe_gap >= 0.0)
    return stable / eligible if eligible else 1.0


def summarize_bridge_results(results: Sequence[RetrievalResult]) -> Dict[str, float]:
    if not results:
        return {
            "queries": 0,
            "certified_stop_rate": 0.0,
            "budget_truncation_rate": 0.0,
            "mean_posterior_error": 0.0,
            "mean_visited_nodes": 0.0,
            "mean_ann_calls_core": 0.0,
            "mean_ann_calls_diagnostic": 0.0,
            "mean_candidates_returned": 0.0,
        }
    count = len(results)
    return {
        "queries": count,
        "certified_stop_rate": sum(result.certified for result in results) / count,
        "budget_truncation_rate": sum(result.budget_frozen for result in results) / count,
        "mean_posterior_error": sum(result.posterior_error for result in results) / count,
        "mean_visited_nodes": sum(result.visited_nodes for result in results) / count,
        "mean_ann_calls_core": sum(result.cost.ann_calls_core for result in results) / count,
        "mean_ann_calls_diagnostic": sum(result.cost.ann_calls_diagnostic for result in results) / count,
        "mean_candidates_returned": sum(result.cost.candidates_returned for result in results) / count,
        "mean_cluster_radius_radians": (
            sum(sum(result.cluster_radii) for result in results)
            / max(1, sum(len(result.cluster_radii) for result in results))
        ),
        "mean_cluster_ranking_stability": (
            sum(sum(result.cluster_stabilities) for result in results)
            / max(1, sum(len(result.cluster_stabilities) for result in results))
        ),
    }


def _record_value(record: Mapping[str, Any], metric: str = "answer_accuracy") -> float | None:
    """Read a metric from either prediction or flattened summary records."""
    value: Any = record.get(metric)
    if value is None and isinstance(record.get("outcome"), Mapping):
        value = record["outcome"].get(metric)
    if value is None and isinstance(record.get("metrics"), Mapping):
        outcome = record["metrics"].get("outcome", {})
        if isinstance(outcome, Mapping):
            value = outcome.get(metric)
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def question_micro_accuracy(records: Sequence[Mapping[str, Any]], metric: str = "answer_accuracy") -> float | None:
    values = [value for record in records if (value := _record_value(record, metric)) is not None]
    return sum(values) / len(values) if values else None


def persona_macro_accuracy(
    records: Sequence[Mapping[str, Any]],
    metric: str = "answer_accuracy",
) -> float | None:
    """Macro-average question accuracy over personas (primary protocol unit)."""
    by_persona: Dict[str, list[float]] = defaultdict(list)
    for record in records:
        value = _record_value(record, metric)
        persona = record.get("persona_id")
        if value is not None and persona is not None:
            by_persona[str(persona)].append(value)
    per_persona = [sum(values) / len(values) for values in by_persona.values() if values]
    return sum(per_persona) / len(per_persona) if per_persona else None


def gain_damage_net(
    baseline: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    treatment: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    metric: str = "answer_accuracy",
) -> Dict[str, float]:
    """Count paired treatment corrections without inventing relevance labels."""
    def keyed(value: Sequence[Mapping[str, Any]] | Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
        if isinstance(value, Mapping):
            # Mapping may already be question_id -> scalar/record.
            output = {}
            for key, item in value.items():
                output[str(key)] = item if isinstance(item, Mapping) else {metric: item}
            return output
        return {
            str(item.get("question_id", index)): item
            for index, item in enumerate(value)
            if isinstance(item, Mapping)
        }
    left, right = keyed(baseline), keyed(treatment)
    common = sorted(set(left) & set(right))
    gain = damage = 0
    paired = 0
    for key in common:
        before, after = _record_value(left[key], metric), _record_value(right[key], metric)
        if before is None or after is None:
            continue
        paired += 1
        gain += int(before < 1.0 and after >= 1.0)
        damage += int(before >= 1.0 and after < 1.0)
    result = {
        "gain": float(gain),
        "damage": float(damage),
        "net": float(gain - damage),
        "paired_questions": float(paired),
    }
    result.update({"Gain": result["gain"], "Damage": result["damage"], "Net": result["net"]})
    return result


def exact_sign_flip_test(
    treatment_by_persona: Mapping[str, float] | Sequence[float],
    baseline_by_persona: Mapping[str, float] | Sequence[float],
    *,
    alternative: str = "two-sided",
) -> Dict[str, Any]:
    """Exact paired sign-flip test at the persona unit (2^14 is tractable)."""
    if isinstance(treatment_by_persona, Mapping) and isinstance(baseline_by_persona, Mapping):
        keys = sorted(set(treatment_by_persona) & set(baseline_by_persona), key=str)
        differences = np.asarray(
            [float(treatment_by_persona[key]) - float(baseline_by_persona[key]) for key in keys], dtype=np.float64
        )
    else:
        left = np.asarray(treatment_by_persona, dtype=np.float64).reshape(-1)
        right = np.asarray(baseline_by_persona, dtype=np.float64).reshape(-1)
        if len(left) != len(right):
            raise ValueError("sign-flip samples must have equal length")
        keys = [str(index) for index in range(len(left))]
        differences = left - right
    n = len(differences)
    if n == 0 or n > 20:
        raise ValueError("exact sign-flip requires 1..20 paired personas")
    alternative = alternative.lower().replace("_", "-")
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be two-sided, greater, or less")
    observed = float(np.mean(differences))
    zero_mask = np.isclose(differences, 0.0, atol=1e-15, rtol=0.0)
    null_values = np.empty(1 << n, dtype=np.float64)
    for mask in range(1 << n):
        signs = np.asarray([1.0 if mask & (1 << index) else -1.0 for index in range(n)])
        null_values[mask] = float(np.mean(signs * differences))
    if alternative == "greater":
        extreme = np.count_nonzero(null_values >= observed - 1e-15)
    elif alternative == "less":
        extreme = np.count_nonzero(null_values <= observed + 1e-15)
    else:
        extreme = np.count_nonzero(np.abs(null_values) >= abs(observed) - 1e-15)
    return {
        "n_personas": n,
        "personas": keys,
        "differences": differences.tolist(),
        # Zero/tie differences do not affect any sign-flipped statistic, but
        # retaining their count makes the effective paired sample explicit
        # instead of silently treating ties as gains or damages.
        "zero_differences": int(np.count_nonzero(zero_mask)),
        "nonzero_personas": int(n - np.count_nonzero(zero_mask)),
        "observed_mean_difference": observed,
        "p_value": float(extreme / (1 << n)),
        "enumerated_sign_flips": int(1 << n),
        "alternative": alternative,
    }


def persona_cluster_bootstrap_interval(
    differences_by_persona: Mapping[str, float],
    clusters: Mapping[str, str] | None = None,
    *,
    seed: int = 42,
    resamples: int = 2000,
    confidence: float = 0.95,
) -> Dict[str, float]:
    """Cluster bootstrap interval, resampling persona clusters as units."""
    if not differences_by_persona:
        raise ValueError("cluster bootstrap requires non-empty differences")
    if resamples <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid bootstrap settings")
    cluster_map = {str(persona): str((clusters or {}).get(persona, persona)) for persona in differences_by_persona}
    by_cluster: Dict[str, list[float]] = defaultdict(list)
    for persona, value in differences_by_persona.items():
        by_cluster[cluster_map[str(persona)]].append(float(value))
    names = sorted(by_cluster)
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=np.float64)
    # Clusters, rather than individual personas, are the resampling units.
    # Each sampled cluster contributes its own persona mean, so a large
    # cluster cannot receive extra weight merely because it contains more
    # personas.  This is the standard cluster-bootstrap estimand and matches
    # the persona-macro analysis used by the protocol.
    cluster_means = {name: float(np.mean(values)) for name, values in by_cluster.items()}
    for index in range(resamples):
        sampled = rng.choice(names, size=len(names), replace=True)
        estimates[index] = float(np.mean([cluster_means[str(name)] for name in sampled]))
    alpha = (1.0 - confidence) / 2.0
    return {
        # The estimand is an equally weighted mean over clusters.  Reporting
        # a persona-weighted point estimate here would make it inconsistent
        # with the resampled interval whenever clusters have different sizes.
        "mean_difference": float(np.mean(list(cluster_means.values()))),
        "ci_low": float(np.quantile(estimates, alpha)),
        "ci_high": float(np.quantile(estimates, 1.0 - alpha)),
        "confidence": float(confidence),
        "resamples": int(resamples),
        "clusters": int(len(names)),
    }


def stratified_outcome_metrics(
    records: Sequence[Mapping[str, Any]],
    fields: Sequence[str] = ("question_type", "topic"),
    metric: str = "answer_accuracy",
) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for field in fields:
        groups: Dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for record in records:
            groups[str(record.get(field, "unknown"))].append(record)
        result[field] = {
            name: {
                "queries": len(values),
                "question_micro": question_micro_accuracy(values, metric),
                "persona_macro": persona_macro_accuracy(values, metric),
            }
            for name, values in sorted(groups.items())
        }
    return result


# Compatibility spellings used in analysis notebooks.
macro_accuracy_by_persona = persona_macro_accuracy
micro_accuracy_by_question = question_micro_accuracy
sign_flip_test = exact_sign_flip_test
cluster_bootstrap_interval = persona_cluster_bootstrap_interval
