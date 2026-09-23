"""Pure offline subset, reachability, and production-selector diagnostics.

This module has no model client, no transport fallback, and no cache writes.
The 28-set experiment is a post-hoc diagnostic, not a benchmark estimator.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .dependency_scoring import SetBudgetExceeded
from .dependency_search import DynamicBundleSelector
from .diagnostic_import import (
    DiagnosticImportError, HistoricalScoreConflict, canonical_ids,
    import_historical_scores, load_diagnostic_archive,
)


DEFAULT_DIAGNOSTIC_CASES = (
    {"case_id": "painting", "question_id": "17273334-b524-4398-baae-bfb459f149e0",
     "task_id": "3b1003e23012d57db5e0a5bf1cf473689beef60580562ffe71d84c0dc7cd581b",
     "memory_indices": (11, 39, 81)},
    {"case_id": "music", "question_id": "fd81b480-5723-43fe-9fb1-d3efa0df8e49",
     "task_id": "dd67463ac5377f24adb34606b97ec13df4e20e72130624ae43c9c4d7d299dec2",
     "memory_indices": (5, 6, 8, 27)},
    {"case_id": "book_club", "question_id": "d4562a36-f9aa-4dcb-9033-434e5a1a4535",
     "task_id": "55de70ff2877a08d427cfbccbeb317a07625f037c72a98e72e250340f3faccfa",
     "memory_indices": (37, 73)},
)


class MissingDiagnosticScores(ValueError):
    def __init__(self, ids: Sequence[Sequence[str]]):
        self.ids = tuple(canonical_ids(value) for value in ids)
        super().__init__(f"unobserved diagnostic scores: {self.ids}")


class MissingDiagnosticFeasibility(ValueError):
    def __init__(self, ids: Sequence[str]):
        self.ids = canonical_ids(ids)
        super().__init__(f"unobserved diagnostic feasibility: {self.ids}")


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DiagnosticImportError(f"{name} must be a nonnegative integer")
    return value


def case_universe(case: Mapping[str, Any]) -> tuple[str, ...]:
    if "universe_ids" in case:
        values = case["universe_ids"]
    elif "memory_ids" in case:
        values = case["memory_ids"]
    else:
        question = case.get("question_id")
        if not isinstance(question, str) or not question:
            raise DiagnosticImportError("case requires question_id")
        indices = case.get("memory_indices", ())
        values = [f"{question}:m{_nonnegative_int(index, 'memory index'):05d}" for index in indices]
    result = canonical_ids(values)
    if len(result) != len(values):
        raise DiagnosticImportError("case universe contains duplicate IDs")
    if not result or len(result) > 10:
        raise DiagnosticImportError("diagnostic universe must contain 1 to 10 IDs")
    question = case.get("question_id")
    if question and any(":m" in value and value.split(":m", 1)[0] != question for value in result):
        raise DiagnosticImportError("case universe includes another question")
    return result


def enumerate_case_subsets(case: Mapping[str, Any]) -> list[dict[str, Any]]:
    universe = case_universe(case)
    return [{"case_id": case.get("case_id"), "question_id": case.get("question_id"),
             "subset_index": index, "ids": list(ids), "cardinality": len(ids)}
            for index, ids in enumerate(
                ids for size in range(len(universe) + 1)
                for ids in itertools.combinations(universe, size))]


def _score_index(imported: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> dict[tuple[str, ...], float]:
    records = imported.get("records", []) if isinstance(imported, Mapping) else imported
    result = {}
    for row in records:
        ids, value = canonical_ids(row["ids"]), row.get("score")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise DiagnosticImportError("score table must contain finite explicit scores")
        if ids in result and result[ids] != float(value):
            raise HistoricalScoreConflict(f"score view contains conflicting observations for {ids}")
        result[ids] = float(value)
    return result


def _bundle_sequence(bundles: Sequence[Any]) -> tuple[tuple[str, ...], ...]:
    if isinstance(bundles, (str, bytes)) or not isinstance(bundles, Sequence):
        raise DiagnosticImportError("archive bundles must be a sequence")
    result = {canonical_ids(row["memory_ids"] if isinstance(row, Mapping) else row) for row in bundles}
    result.discard(())
    return tuple(sorted(result, key=lambda ids: (len(ids), ids)))


def _comparison_rows_match(actual: Sequence[Mapping[str, Any]],
                           historical: Sequence[Mapping[str, Any]], *, atol: float = 1e-12) -> bool:
    """Compare the historical round's whole feasible domain, not just winner."""
    if len(actual) != len(historical):
        return False
    for current_round, old_round in zip(actual, historical):
        new = {canonical_ids(row["bundle_ids"]): row for row in current_round["comparisons"]}
        old = {canonical_ids(row["bundle_ids"]): row for row in old_round["comparisons"]}
        if new.keys() != old.keys():
            return False
        for key in new:
            for name in ("current_ids", "union_ids", "feasible", "accepted"):
                if new[key].get(name) != old[key].get(name):
                    return False
            for name in ("base_score", "combined_score", "marginal"):
                left, right = new[key].get(name), old[key].get(name)
                if left is None or right is None:
                    if left != right:
                        return False
                elif abs(left - right) > atol:
                    return False
    return True


def _feasibility_map(records: Sequence[Mapping[str, Any]]) -> dict[tuple[str, ...], bool]:
    result = {}
    for row in records:
        ids, feasible = canonical_ids(row["ids"]), row["feasible"]
        if not isinstance(feasible, bool):
            raise DiagnosticImportError("feasibility records must contain booleans")
        if ids in result and result[ids] != feasible:
            raise DiagnosticImportError("conflicting feasibility observations")
        result[ids] = feasible
    return result


def _feasibility_value(check: Any, ids: tuple[str, ...]) -> bool | None:
    if check is None:
        return None
    value = check(ids) if callable(check) else check.get(ids)
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, tuple) and value and isinstance(value[0], bool):
        return value[0]
    if isinstance(value, Mapping):
        for key in ("feasible", "ok", "allowed"):
            if isinstance(value.get(key), bool):
                return value[key]
    raise DiagnosticImportError("feasibility callback must return bool, None, or an explicit decision")


class RecordedSetScorer:
    """A complete-round, logical-unique-budget scorer with no online escape.

    Historical search identities can seed the charged set to reproduce the
    selection stage's cumulative accounting. Literal score duplicates do not
    create additional logical charges.
    """
    def __init__(self, scores: Mapping[tuple[str, ...], float], *,
                 initial_seen: Sequence[Sequence[str]] = (), feasibility: Any = None):
        self.scores = dict(scores)
        self.seen = {canonical_ids(ids) for ids in initial_seen}
        self.feasibility = feasibility
        self.budget_limit: int | None = None
        self.events: list[dict[str, Any]] = []

    @property
    def scored_sets(self) -> int:
        return len(self.seen)

    def set_budget(self, limit: int | None) -> None:
        if limit is not None:
            limit = _nonnegative_int(limit, "set budget")
            if limit < self.scored_sets:
                raise DiagnosticImportError("set budget below already charged sets")
        self.budget_limit = limit

    def feasible(self, ids: Sequence[str]) -> bool:
        key = canonical_ids(ids)
        value = _feasibility_value(self.feasibility, key)
        if value is None:
            raise MissingDiagnosticFeasibility(key)
        return value

    def preflight(self, sets: Sequence[Sequence[str]]) -> None:
        keys = {canonical_ids(ids) for ids in sets}
        new = keys - self.seen
        if self.budget_limit is not None and len(new) > self.budget_limit - self.scored_sets:
            raise SetBudgetExceeded(required=len(new), remaining=self.budget_limit - self.scored_sets,
                                    limit=self.budget_limit)
        missing = sorted(keys - self.scores.keys())
        if missing:
            raise MissingDiagnosticScores(missing)

    def score_sets(self, sets: Sequence[Sequence[str]], *, reason: str = "selection") -> list[float]:
        self.preflight(sets)
        keys = [canonical_ids(ids) for ids in sets]
        new = set(keys) - self.seen
        self.seen.update(keys)
        self.events.append({"reason": reason, "requested_sets": len(keys),
                            "new_unique_sets": len(new), "charged_total": self.scored_sets})
        return [self.scores[key] for key in keys]

    def score_set(self, ids: Sequence[str], *, reason: str = "selection") -> float:
        return self.score_sets([ids], reason=reason)[0]


def analyze_reachability(
    case: Mapping[str, Any], bundles: Sequence[Any],
    scores: Mapping[str, Any] | Sequence[Mapping[str, Any]], *,
    feasibility: Mapping[tuple[str, ...], bool] | Callable | None = None,
    start_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Analyze exact union construction and strictly positive paths on U.

    Actual bundles B outside U are excluded, NEVER replaced by B intersect U.
    A missing score or unknown feasibility is an unknown edge. Reachability
    is false only when even the optimistic graph cannot reach the target.
    """
    universe = case_universe(case)
    universe_set = set(universe)
    start = canonical_ids(start_ids)
    if not set(start) <= universe_set:
        raise DiagnosticImportError("reachability start is outside diagnostic universe")
    original = _bundle_sequence(bundles)
    eligible = tuple(bundle for bundle in original if set(bundle) <= universe_set)
    values = _score_index(scores)
    if feasibility is None and isinstance(scores, Mapping):
        feasibility = _feasibility_map(scores.get("historical_feasibility", []))
    subsets = [tuple(row["ids"]) for row in enumerate_case_subsets(case)]
    edges: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    unknown_score_edges = unknown_feasibility_edges = 0
    for state in subsets:
        outgoing = []
        for bundle in eligible:
            union = canonical_ids((*state, *bundle))
            if union == state:
                continue
            feasible = _feasibility_value(feasibility, union)
            marginal = values[union] - values[state] if union in values and state in values else None
            outgoing.append({"from_ids": list(state), "to_ids": list(union),
                             "bundle_ids": list(bundle), "feasible": feasible, "marginal": marginal})
            unknown_score_edges += marginal is None
            unknown_feasibility_edges += feasible is None
        edges[state] = outgoing

    def visit(mode: str) -> dict[tuple[str, ...], tuple[tuple[str, ...], dict[str, Any]] | None]:
        parents = {start: None}
        queue = deque([start])
        while queue:
            state = queue.popleft()
            for edge in edges[state]:
                f, delta = edge["feasible"], edge["marginal"]
                allowed = (mode == "structural" or
                           mode == "feasible_confirmed" and f is True or
                           mode == "feasible_possible" and f is not False or
                           mode == "positive_confirmed" and f is True and delta is not None and delta > 0 or
                           mode == "positive_possible" and f is not False and (delta is None or delta > 0))
                target = tuple(edge["to_ids"])
                if allowed and target not in parents:
                    parents[target] = (state, edge)
                    queue.append(target)
        return parents

    graphs = {mode: visit(mode) for mode in ("structural", "feasible_confirmed", "feasible_possible",
                                           "positive_confirmed", "positive_possible")}

    def path(target, parents):
        if target not in parents:
            return None
        result = []
        while parents[target] is not None:
            target, edge = parents[target]
            result.append(edge)
        return list(reversed(result))

    def status(target, confirmed, possible):
        return True if target in graphs[confirmed] else None if target in graphs[possible] else False

    results = []
    for subset in subsets:
        results.append({"ids": list(subset), "score": values.get(subset),
                        "constructible": subset in graphs["structural"],
                        "construction_path": path(subset, graphs["structural"]),
                        "feasible_constructible": status(subset, "feasible_confirmed", "feasible_possible"),
                        "positive_path_reachable": status(subset, "positive_confirmed", "positive_possible"),
                        "positive_path": path(subset, graphs["positive_confirmed"]),
                        "optimistic_positive_path": path(subset, graphs["positive_possible"])})
    return {"schema_version": 1, "scope": "restricted_universe", "start_ids": list(start),
            "universe_ids": list(universe), "original_bundle_count": len(original),
            "eligible_bundle_ids": [list(ids) for ids in eligible],
            "excluded_bundle_count": len(original) - len(eligible),
            "unknown_score_edge_count": unknown_score_edges,
            "unknown_feasibility_edge_count": unknown_feasibility_edges,
            "score_origin": scores.get("origin", "explicit_score_view") if isinstance(scores, Mapping) else "explicit_score_view",
            "subsets": results, "edges": [edge for state in subsets for edge in edges[state]],
            "limitations": ["Constructible ignores scores and input capacity; feasible_constructible checks capacity.",
                            "Positive path existence does not imply greedy selection or correct generation.",
                            "null reachability means unresolved score or feasibility observations."]}


def replay_selector(
    artifact: Mapping[str, Any], imported: Mapping[str, Any], *,
    scope: str = "historical_full_archive", case: Mapping[str, Any] | None = None,
    max_selection_sets: int | None = 512,
    feasibility: Mapping[tuple[str, ...], bool] | Callable | None = None,
    initial_seen: Sequence[Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Run the real DynamicBundleSelector over literal offline observations.

    Full-archive replay verifies seed identities against archived accounting.
    Restricted simulation keeps only original bundles wholly inside U. A
    missing score/feasibility returns incomplete; it never calls a service.
    """
    if scope not in {"historical_full_archive", "restricted_universe"}:
        raise DiagnosticImportError("unsupported selector diagnostic scope")
    if max_selection_sets is not None:
        _nonnegative_int(max_selection_sets, "max_selection_sets")
    search, historical = artifact["search"], artifact["selection"]
    bundles = _bundle_sequence(search["bundles"])
    original_count = len(bundles)
    if scope == "restricted_universe":
        if case is None:
            raise DiagnosticImportError("restricted simulation requires a case")
        universe = set(case_universe(case))
        bundles = tuple(bundle for bundle in bundles if set(bundle) <= universe)
    seen = imported.get("search_seen_ids", []) if initial_seen is None else initial_seen
    seen_keys = {canonical_ids(ids) for ids in seen}
    expected_initial = historical.get("initial_scored_sets")
    if expected_initial is not None:
        _nonnegative_int(expected_initial, "historical initial_scored_sets")
    base = {"schema_version": 1, "scope": scope, "network_calls": 0,
            "max_selection_sets": max_selection_sets, "initial_seen_count": len(seen_keys),
            "budget_initialization": "historical_search_literals" if initial_seen is None else "explicit_seen_ids",
            "archived_initial_scored_sets": expected_initial,
            "original_bundle_count": original_count, "active_bundle_count": len(bundles),
            "score_origin": imported.get("origin", "explicit_score_view")}
    if (expected_initial is not None and expected_initial != len(seen_keys)
            and (scope == "historical_full_archive" or initial_seen is None)):
        return {**base, "status": "incomplete", "reason": "initial_charged_identity_coverage",
                "result": None, "historical_match": None,
                "limitations": ["Literal search scores do not enumerate the archived charged-set identities."]}
    if feasibility is None:
        feasibility = _feasibility_map(imported.get("historical_feasibility", []))
        base["feasibility_source"] = "historical_comparisons"
    else:
        base["feasibility_source"] = "explicit_exact_input_check"
    scorer = RecordedSetScorer(_score_index(imported), initial_seen=list(seen_keys), feasibility=feasibility)
    selector = DynamicBundleSelector(scorer, max_selection_sets=max_selection_sets)
    try:
        result = selector.select(bundles).public_dict()
    except (MissingDiagnosticScores, MissingDiagnosticFeasibility) as exc:
        missing_scores = isinstance(exc, MissingDiagnosticScores)
        reason = "incomplete_score_coverage" if missing_scores else "unknown_input_feasibility"
        return {**base, "status": "incomplete", "reason": reason,
                "missing_ids": [list(ids) for ids in exc.ids] if missing_scores else [list(exc.ids)],
                "result": selector.partial_public_dict(stop_reason=reason, detail=str(exc)),
                "score_events": scorer.events, "historical_match": None,
                "limitations": ["The incomplete round commits no winner; no online fallback is permitted."]}
    reason = result["stop"]["reason"]
    complete = reason not in {"score_budget_exhausted", "input_capacity"}
    comparisons = None
    if scope == "historical_full_archive":
        comparisons = {
            "selected_ids": canonical_ids(result["selected_ids"]) == canonical_ids(historical["selected_ids"]),
            "stop_reason": reason == historical["stop"]["reason"],
            "initial_scored_sets": result["initial_scored_sets"] == historical.get("initial_scored_sets"),
            "final_scored_sets": result["final_scored_sets"] == historical.get("final_scored_sets"),
            "round_decisions": [(r["current_ids"], r["accepted_bundle_ids"], r["selected_ids_after"])
                                for r in result["rounds"]] ==
                               [(r["current_ids"], r["accepted_bundle_ids"], r["selected_ids_after"])
                                for r in historical["rounds"]],
            "complete_comparison_domain": _comparison_rows_match(result["rounds"], historical["rounds"]),
        }
    return {**base, "status": "complete" if complete else "resource_stopped", "reason": reason,
            "result": result, "score_events": scorer.events,
            "historical_checks": comparisons,
            "historical_match": all(comparisons.values()) if comparisons is not None else None,
            "limitations": (["Restricted bundles and trajectories are a counterfactual, not full-archive replay."]
                            if scope == "restricted_universe" else [])}


def audit_diagnostic_cases(
    path: str, *, expected_sha256: str | None = None,
    cases: Sequence[Mapping[str, Any]] = DEFAULT_DIAGNOSTIC_CASES,
    max_selection_sets: int | None = 512,
) -> dict[str, Any]:
    """One read-only JSON-serializable audit suitable for CLI/report callers."""
    loaded = load_diagnostic_archive(path, expected_sha256=expected_sha256)
    by_task = {}
    for entry in loaded["artifacts"]:
        task_id = entry["artifact"].get("task_id")
        if task_id in by_task:
            raise DiagnosticImportError(f"duplicate task artifact: {task_id}")
        by_task[task_id] = entry
    results = []
    for case in cases:
        if case.get("task_id") not in by_task:
            raise DiagnosticImportError(f"missing requested task artifact: {case.get('task_id')}")
        entry = by_task[case["task_id"]]
        artifact = entry["artifact"]
        imported = import_historical_scores(artifact, provenance=entry["provenance"])
        if imported["question_id"] != case.get("question_id"):
            raise DiagnosticImportError("case question does not match historical scores")
        values = _score_index(imported)
        subsets = enumerate_case_subsets(case)
        missing = [row["ids"] for row in subsets if tuple(row["ids"]) not in values]
        results.append({"case_id": case["case_id"], "question_id": case["question_id"],
                        "task_id": case["task_id"], "universe_ids": list(case_universe(case)),
                        "subset_count": len(subsets), "historical_score_count": len(subsets) - len(missing),
                        "missing_ids": missing, "score_import": imported,
                        "subsets": [{**row, "score": values.get(tuple(row["ids"])),
                                     "origin": "historical_literal" if tuple(row["ids"]) in values else "missing"}
                                    for row in subsets],
                        "reachability": analyze_reachability(case, artifact["search"]["bundles"], imported),
                        "historical_replay": replay_selector(artifact, imported, max_selection_sets=max_selection_sets),
                        "restricted_simulation": replay_selector(
                            artifact, imported, scope="restricted_universe", case=case,
                            max_selection_sets=max_selection_sets)})
    return {"schema_version": 1, "purpose": "posthoc_diagnostic", "eligible_for_benchmark": False,
            "source_sha256": loaded["source_sha256"], "source_kind": loaded["source_kind"],
            "network_calls": 0, "cases": results,
            "subset_count": sum(row["subset_count"] for row in results),
            "historical_score_count": sum(row["historical_score_count"] for row in results),
            "missing_score_count": sum(len(row["missing_ids"]) for row in results),
            "limitations": ["Historical observations are not a current model cache.",
                            "Four new scores would create a hybrid-time table, not a complete historical function.",
                            "The 28 subsets do not cover all possible full-archive counterfactual trajectories."]}


def main(argv: Sequence[str] | None = None) -> int:
    """Standalone PR2 acceptance entry: read-only audit JSON on stdout."""
    parser = argparse.ArgumentParser(description="Audit the frozen 28 dependency subsets without model calls")
    parser.add_argument("path", help="Read-only candidate_pool archive or directory")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--max-selection-sets", type=int, default=512)
    args = parser.parse_args(argv)
    try:
        result = audit_diagnostic_cases(args.path, expected_sha256=args.expected_sha256,
                                        max_selection_sets=args.max_selection_sets)
    except (DiagnosticImportError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
