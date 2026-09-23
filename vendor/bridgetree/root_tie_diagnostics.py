"""Zero-priority root tie sensitivity, never a coverage scheduler.

The helpers do no retrieval/scoring, mutate no IDs and make no service calls.
Trace costs are first-charged logical costs, not counterfactual independent
costs per target. Missing observations remain ``None`` in comparisons.
"""
from __future__ import annotations

import hashlib
import json
from numbers import Integral
from typing import Any, Iterable, Mapping, Sequence

ROOT_TIE_VERSION = "zero_priority_root_tie_v1"


def validate_root_tie_settings(mode: str, seed: int | None) -> tuple[str, int | None]:
    if not isinstance(mode, str) or mode not in {"legacy_lexical", "seeded_hash"}:
        raise ValueError("root_tie_break must be legacy_lexical or seeded_hash")
    if mode == "legacy_lexical":
        if seed is not None:
            raise ValueError("root_tie_seed must be null for legacy_lexical")
    elif isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("root_tie_seed must be an explicit non-negative integer for seeded_hash")
    return mode, None if seed is None else int(seed)


def _hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def root_tie_order(
    target_ids: Iterable[str], *, mode: str = "legacy_lexical", seed: int | None = None,
) -> tuple[str, ...]:
    """Order a copy of the root IDs; the initial candidate sequence is untouched.

    Full memory IDs already include question identity in the dependency
    pipeline. No process hash, RNG state, insertion order or method ID enters
    the key, so common targets have the same relative order across methods.
    """
    mode, seed = validate_root_tie_settings(mode, seed)
    if isinstance(target_ids, (str, bytes)):
        raise ValueError("root target IDs must be an iterable of non-empty strings")
    values = tuple(target_ids)
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("root target IDs must be non-empty strings")
    ids = tuple(sorted(set(values)))
    if mode == "legacy_lexical":
        return ids
    return tuple(sorted(ids, key=lambda value: (_hash([ROOT_TIE_VERSION, seed, value]), value)))


def _ids(values: Iterable[str]) -> list[str]:
    return sorted(set(values))


def summarize_root_tie_run(
    *, mode: str, seed: int | None, initial_ids: Sequence[str],
    state_observations: Sequence[Mapping[str, Any]],
    initial_ann_calls: int, final_ann_calls: int,
    initial_scored_sets: int, final_scored_sets: int,
    bundle_ids: Iterable[Sequence[str]], proposed_ids: Iterable[str],
    search_complete: bool, stop_reason: str | None,
    selected_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Summarize measured counters without re-running or completing a search."""
    order = root_tie_order(initial_ids, mode=mode, seed=seed)
    roots = set(initial_ids)
    states = []
    for raw in state_observations:
        row = dict(raw)
        # An open state means an exception interrupted execution. The live
        # cumulative counters are still observations, not estimated work.
        ann_after = row.get("ann_calls_after", final_ann_calls)
        sets_after = row.get("scored_sets_after", final_scored_sets)
        row["ann_calls_delta"] = ann_after - row["ann_calls_before"]
        row["logical_unique_sets_charged_delta"] = sets_after - row["scored_sets_before"]
        states.append(row)
    per_target = {
        target: {
            "popped_state_count": 0, "popped_root_count": 0,
            "max_premise_depth": None, "ann_calls_delta": 0,
            "logical_unique_sets_charged_delta": 0, "incomplete_state_count": 0,
        }
        for target in sorted(roots)
    }
    for row in states:
        item = per_target[row["target_id"]]
        item["popped_state_count"] += 1
        item["popped_root_count"] += int(row["is_zero_priority_root"])
        item["max_premise_depth"] = max(item["max_premise_depth"] or 0, row["premise_depth"])
        item["ann_calls_delta"] += row["ann_calls_delta"]
        item["logical_unique_sets_charged_delta"] += row["logical_unique_sets_charged_delta"]
        item["incomplete_state_count"] += int(not row["completed"])
    bundles = sorted({tuple(_ids(bundle)) for bundle in bundle_ids}, key=lambda x: (len(x), x))
    archived = {identifier for bundle in bundles for identifier in bundle}
    external = set(proposed_ids) - roots
    grouped = {identifier for bundle in bundles if len(bundle) > 1 for identifier in bundle}
    selected = None if selected_ids is None else _ids(selected_ids)
    external_selected = None if selected is None else _ids(external.intersection(selected))
    popped_roots = [row["target_id"] for row in states if row["is_zero_priority_root"]]
    return {
        "schema_version": 1, "diagnostic_kind": ROOT_TIE_VERSION,
        "root_tie_break": mode, "root_tie_seed": seed,
        "initial_target_ids": sorted(roots), "initial_candidate_order": list(initial_ids),
        "root_tie_order": list(order), "root_tie_order_hash": _hash(order),
        "root_pop_order": popped_roots,
        "unvisited_root_ids": sorted(roots - set(popped_roots)),
        "states": states, "per_target": per_target,
        "search_complete": search_complete, "search_stop_reason": stop_reason,
        "search_complete_scope": "returned_archive_not_exhaustive_coverage",
        "global_certificate": False,
        "ann_calls_delta": final_ann_calls - initial_ann_calls,
        "logical_unique_sets_charged_delta": final_scored_sets - initial_scored_sets,
        "cost_attribution": "first_charge_during_search; excludes initial_pool_and_selection",
        "archive_bundle_ids": [list(bundle) for bundle in bundles],
        "archive_bundle_ids_hash": _hash(bundles),
        "archived_memory_ids": sorted(archived),
        "final_selected_ids": selected,
        "final_selected_ids_hash": None if selected is None else _hash(selected),
        "external_candidate_ids": sorted(external),
        "external_archived_ids": sorted(external & archived),
        "external_in_multimemory_bundle_ids": sorted(external & grouped),
        "external_selected_ids": external_selected,
        "external_to_selected_rate": (
            None if not external or external_selected is None else len(external_selected) / len(external)
        ),
        "interpretation": "tie-only sensitivity, not root coverage guarantees or effectiveness evidence",
    }


def _comparison(left: Any, right: Any, *, bundles: bool = False) -> dict[str, Any]:
    if left is None or right is None:
        return {"available": False, "equal": None, "jaccard": None}
    a = {tuple(sorted(value)) for value in left} if bundles else set(left)
    b = {tuple(sorted(value)) for value in right} if bundles else set(right)
    union = a | b
    return {"available": True, "equal": a == b, "jaccard": len(a & b) / len(union) if union else 1.0}


def compare_root_tie_runs(baseline: Mapping[str, Any], variant: Mapping[str, Any]) -> dict[str, Any]:
    """Compare actual traces; absent final selections/costs are not fabricated.

    Matching target IDs alone does not establish matching score, model or
    generation protocols. The caller must validate those before interpreting
    this descriptive comparison as a controlled sensitivity experiment.
    """
    result: dict[str, Any] = {
        "schema_version": 1,
        "same_initial_targets": _comparison(baseline.get("initial_target_ids"), variant.get("initial_target_ids")),
        "archive": _comparison(baseline.get("archive_bundle_ids"), variant.get("archive_bundle_ids"), bundles=True),
        "final_selection": _comparison(baseline.get("final_selected_ids"), variant.get("final_selected_ids")),
        "external_candidates": _comparison(baseline.get("external_candidate_ids"), variant.get("external_candidate_ids")),
        "external_selected": _comparison(baseline.get("external_selected_ids"), variant.get("external_selected_ids")),
        "root_pop_order_equal": None,
        "other_protocols_verified": False,
    }
    if baseline.get("root_pop_order") is not None and variant.get("root_pop_order") is not None:
        result["root_pop_order_equal"] = baseline["root_pop_order"] == variant["root_pop_order"]
    for name in ("ann_calls_delta", "logical_unique_sets_charged_delta"):
        a, b = baseline.get(name), variant.get(name)
        result[name] = {"baseline": a, "variant": b, "difference": None if a is None or b is None else b - a}
    return result


__all__ = [
    "ROOT_TIE_VERSION", "validate_root_tie_settings", "root_tie_order",
    "summarize_root_tie_run", "compare_root_tie_runs",
]
