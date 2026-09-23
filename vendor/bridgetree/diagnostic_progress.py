"""Small, read-only diagnostic snapshots for a five-second worker heartbeat.

Atomic results are authoritative; a durable terminal task event bridges the
crash window before their rename. Ledgers are tailed incrementally and large
artifacts are reduced once per file version, never copied into progress.json.
No gold labels, model clients, archive loading, or stage/total cost addition.
"""
from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
import json
import math
from pathlib import Path
import threading
import time

from .diagnostic_identity import request_hash
from .diagnostic_root_runner import root_trial_plan
from .diagnostic_runner import atomic_json, load_manifest

_PHASES = ("score", "generation", "root")
_LOCK = threading.RLock()
_FILES = OrderedDict()
_LEDGERS = OrderedDict()
_META = ("manifest_id", "item_id", "phase", "trial_id", "subset_id", "question_id",
         "condition_id", "repeat_index", "status", "task_attempt", "outcome_unknown",
         "error_type", "http_status", "retryable", "event", "at_epoch", "generation_cache_hit")
_HTTP = ("event", "event_id", "run_identity", "phase", "task_id", "task_attempt",
         "request_id", "transport_attempt", "status_code", "cause_type", "elapsed_ms",
         "server_reported_input_tokens", "server_reported_output_tokens", "server_reported_total_tokens")


def _number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def _ratio(a, b):
    return a / b if a is not None and b is not None and b > 0 else None


def _ids(value):
    return set(value) if isinstance(value, list) and all(isinstance(x, str) for x in value) else None


def _count(value):
    values = _ids(value)
    return len(values) if values is not None else None


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _stats(values):
    observed = [v for v in values if _number(v) is not None]
    return {"observed": len(observed), "missing": len(values) - len(observed),
            "sum": sum(observed) if observed else None,
            "mean": sum(observed) / len(observed) if observed else None,
            "min": min(observed) if observed else None, "max": max(observed) if observed else None}


def _root_metrics(row):
    partial = row.get("status") != "success"
    data = row.get("partial_artifacts", {}) if partial else row
    if not isinstance(data, dict):
        data = {}
    trace, search, selection = (_mapping(data.get(k)) for k in ("root_trace", "search", "selection"))
    retrieval = _mapping(data.get("retrieval"))
    initial = _ids(trace.get("initial_target_ids", search.get("initial_target_ids")))
    popped = _ids(trace.get("root_pop_order"))
    per_target = trace.get("per_target")
    target_counts = ([_number(v.get("popped_state_count")) for v in per_target.values()]
                     if isinstance(per_target, dict) and all(isinstance(v, dict) for v in per_target.values()) else None)
    state_count = (sum(target_counts) if target_counts is not None and all(v is not None for v in target_counts) else None)
    acts = search.get("activations")
    signals = ([_number(a.get("signal")) for a in acts]
               if isinstance(acts, list) and all(isinstance(a, dict) for a in acts) else None)
    positive = (sum(v > 0 for v in signals) if signals is not None and all(v is not None for v in signals) else None)
    selection_complete = not partial and bool(selection) and selection.get("partial") is not True
    selected = selection.get("selected_ids") if selection_complete else None
    final_ids = _ids(selected)
    external_ids = _ids(trace.get("external_candidate_ids"))
    external_selected = _count(trace.get("external_selected_ids")) if selection_complete else None
    if external_selected is None and external_ids is not None and final_ids is not None:
        external_selected = len(external_ids & final_ids)
    plan = _mapping(data.get("context_plan"))
    payload = plan.get("request")
    ann_total = _number(search.get("final_ann_calls"))
    if ann_total is None:
        ann_total = _number(retrieval.get("ann_calls"))
    return {"artifact_available": any(k in data for k in ("root_trace", "search", "selection", "retrieval")), "partial": partial,
            "ann_calls_total": ann_total, "search_ann_calls_delta": _number(search.get("ann_calls")),
            "search_logical_sets_delta": _number(search.get("scored_sets")),
            "selection_logical_sets_delta": _number(selection.get("scored_sets")),
            "external_candidate_count": len(external_ids) if external_ids is not None else None,
            "external_selected_count": external_selected,
            "initial_root_count": len(initial) if initial is not None else None,
            "visited_root_count": len(initial & popped) if initial is not None and popped is not None else None,
            "root_coverage": _ratio(len(initial & popped), len(initial)) if initial is not None and popped is not None else None,
            "popped_state_count": state_count,
            "max_target_state_share": _ratio(max(target_counts, default=0), state_count) if target_counts is not None and state_count is not None else None,
            "positive_signal_count": positive, "interaction_measurement_count": len(acts) if isinstance(acts, list) else None,
            "final_memory_count": len(final_ids) if final_ids is not None else None,
            "partial_selected_memory_count": _count(selection.get("selected_ids")) if partial else None,
            "selection_complete": selection_complete,
            "selection_stop": _mapping(selection.get("stop")).get("reason"),
            "search_stop": search.get("stop_reason", trace.get("search_stop_reason")),
            "archive_hash": trace.get("archive_bundle_ids_hash"),
            "selection_hash": trace.get("final_selected_ids_hash") if selection_complete else None,
            "context_hash": plan.get("context_hash") if selection_complete else None,
            "wire_payload_hash": request_hash(payload) if selection_complete and isinstance(payload, dict) else None}


def _compact_outcome(row):
    out = {k: row[k] for k in _META if k in row}
    if row.get("phase") == "root":
        out["root_metrics"] = _root_metrics(row)
    events = row.get("scorer_events", _mapping(row.get("partial_artifacts")).get("scorer_events"))
    if isinstance(events, list):
        out["scorer_snapshot"] = {"event_count": len(events), "sources": dict(Counter(
            e.get("source", "unknown") for e in events if isinstance(e, dict) and e.get("event") == "set_score"))}
    return out


def _signature(path):
    stat = path.stat()
    return stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _json(path, transform=lambda value: value):
    try:
        signature = _signature(path)
    except FileNotFoundError:
        return None, None
    key = str(path)
    if key in _FILES and _FILES[key][0] == signature:
        _FILES.move_to_end(key)
        return _FILES[key][1:]
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("not an object")
        value, error = transform(raw), None
    except (ValueError, TypeError, KeyError, OSError):
        value, error = None, "invalid_json_or_schema"
    _FILES[key] = (signature, value, error)
    while len(_FILES) > 2048:
        _FILES.popitem(last=False)
    return value, error


def _ledger(path, compact):
    """Tail complete lines only; never silently skip corruption in the middle."""
    key = str(path)
    try:
        signature = _signature(path)
    except FileNotFoundError:
        _LEDGERS.pop(key, None)
        return [], []
    old = _LEDGERS.get(key)
    rewritten = False
    if old is not None and old["offset"] and old.get("signature") != signature:
        with path.open("rb") as stream:
            stream.seek(max(0, old["offset"] - 128))
            rewritten = stream.read(min(128, old["offset"])) != old.get("boundary")
    if old is None or old["inode"] != signature[0] or signature[1] < old["offset"] or (
            signature[1] == old["size"] and signature[2:] != old["signature"][2:]) or rewritten:
        old = {"inode": signature[0], "offset": 0, "line": 0, "rows": [], "warnings": []}
        if rewritten:
            old["warnings"].append({"code": "ledger_rewritten_cache_reset", "file": path.name})
        _LEDGERS[key] = old
    _LEDGERS.move_to_end(key)
    while len(_LEDGERS) > 16:
        _LEDGERS.popitem(last=False)
    tail_warning = []
    with path.open("rb") as stream:
        stream.seek(old["offset"])
        while True:
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                tail_warning = [{"code": "incomplete_trailing_line_ignored", "file": path.name, "line": old["line"] + 1}]
                break
            old["line"] += 1
            old["offset"] = stream.tell()
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("not an object")
                old["rows"].append(compact(row))
            except (ValueError, TypeError, KeyError, UnicodeDecodeError):
                old["warnings"].append({"code": "corrupt_ledger_line", "file": path.name, "line": old["line"]})
        stream.seek(max(0, old["offset"] - 128))
        old["boundary"] = stream.read(min(128, old["offset"]))
    old.update(size=signature[1], signature=signature)
    return old["rows"], old["warnings"] + tail_warning


def _classification(row):
    if row is None:
        return "pending"
    if row.get("error_type") == "InterruptedAttempt" or row.get("status") in {"unknown", "interrupted"}:
        return "unknown"
    if row.get("status") == "success":
        return "success"
    if row.get("status") in {"error", "budget_exhausted"}:
        return "failed"
    return "pending" if row.get("status") in {None, "pending", "started", "running"} else "unknown"


def _terminal(row, maximum):
    if row.get("event") == "task_attempt_completed":
        return row.get("status") == "success"
    attempt = row.get("task_attempt")
    return (row.get("event") == "task_attempt_failed" and row.get("status") in {"error", "budget_exhausted"}
            and (row.get("retryable") is False or (isinstance(attempt, int) and not isinstance(attempt, bool) and attempt >= maximum)))


def _evaluation(root, manifest, resolved, warnings):
    value, error = _json(root / "evaluation.json", lambda v: {k: v.get(k) for k in ("manifest_id", "trials")})
    if error:
        warnings.append({"code": error, "file": "evaluation.json"})
    if value is None:
        return {"available": False, "scope": "posthoc_within_condition", "eligible_for_benchmark": False}
    if value.get("manifest_id") != manifest["manifest_id"] or not isinstance(value.get("trials"), list):
        warnings.append({"code": "evaluation_identity_or_schema_mismatch", "file": "evaluation.json"})
        return {"available": False, "scope": "posthoc_within_condition", "eligible_for_benchmark": False}
    planned = {t["trial_id"]: t for t in manifest["trials"]}
    groups, seen = defaultdict(list), set()
    for row in value["trials"]:
        tid = row.get("trial_id") if isinstance(row, dict) else None
        trial = planned.get(tid)
        if (trial is None or tid in seen or any(row.get(k) != trial[k] for k in ("condition_id", "question_id", "repeat_index"))):
            warnings.append({"code": "unknown_or_duplicate_evaluation_trial", "file": "evaluation.json"})
            continue
        seen.add(tid)
        if _classification(resolved.get(tid)) == "success" and row.get("status") == "success" and isinstance(row.get("correct"), bool):
            groups[trial["condition_id"]].append(row["correct"])
    conditions = []
    for context in manifest["contexts"]:
        cid = context["condition_id"]
        n = sum(t["condition_id"] == cid for t in manifest["trials"])
        observed = groups[cid]
        conditions.append({"condition_id": cid, "question_id": context["question_id"], "planned_repeats": n,
                           "evaluated_successful_repeats": len(observed), "correct": sum(observed) if observed else None,
                           "incorrect": len(observed) - sum(observed) if observed else None,
                           "successful_accuracy": _ratio(sum(observed), len(observed))})
    return {"available": True, "scope": "posthoc_within_condition", "eligible_for_benchmark": False,
            "independent_question_count": len({c["question_id"] for c in manifest["cases"]}),
            "technical_repeats_per_condition": manifest["repeats"], "conditions": conditions,
            "note": "Technical repeats are not independent benchmark questions; unknown/pending outputs are not incorrect answers."}


def build_progress(run_dir, *, phase=None, state=None) -> dict:
    """Read a compact snapshot. Four task buckets partition each frozen plan.

    ``started`` and ``outcome_unknown`` overlap those buckets. HTTP failures
    remain execution failures even if server completion is unknown; an
    InterruptedAttempt is an unknown item, not an incorrect answer.
    ``phase``/``state`` are worker labels, never evidence of task completion.
    """
    if any(x is not None and (not isinstance(x, str) or len(x) > 128) for x in (phase, state)):
        raise ValueError("phase and state must be short strings or None")
    with _LOCK:
        return _build(Path(run_dir).resolve(), phase, state)


def _build(root, phase, state):
    manifest, error = _json(root / "manifest.json", lambda _: load_manifest(root))
    if error or manifest is None:
        raise ValueError("missing or invalid frozen diagnostic manifest")
    mid = manifest["manifest_id"]
    plans = {"score": {r["subset_id"]: r for r in manifest["score_inputs"]},
             "generation": {r["trial_id"]: r for r in manifest["trials"]},
             "root": {r["item_id"]: r for r in root_trial_plan(manifest)}}
    attempts, warnings = _ledger(root / "attempts.jsonl", _compact_outcome)
    requests, request_warnings = _ledger(root / "requests.jsonl", lambda r: {k: r[k] for k in _HTTP if k in r})
    warnings = list(warnings) + request_warnings
    valid_attempts, valid_requests = defaultdict(list), defaultdict(list)
    rejected = Counter()
    for rows, target, identity, item_key, ledger in ((attempts, valid_attempts, "manifest_id", "item_id", "attempts.jsonl"),
                                                   (requests, valid_requests, "run_identity", "task_id", "requests.jsonl")):
        for row in rows:
            p = row.get("phase")
            if row.get(identity) != mid or p not in plans or row.get(item_key) not in plans[p]:
                rejected[ledger] += 1
                continue
            if ledger == "attempts.jsonl":
                expected = plans[p][row[item_key]]
                index = row.get("task_attempt")
                if (not isinstance(index, int) or isinstance(index, bool) or index < 1 or
                        any(k in expected and row.get(k) != expected[k] for k in ("trial_id", "subset_id", "question_id", "condition_id", "repeat_index"))):
                    rejected[ledger] += 1
                    continue
            target[p].append(row)
    for name, count in rejected.items():
        warnings.append({"code": "unknown_or_foreign_ledger_records_ignored", "file": name, "count": count})
    phases, resolved_all, root_rows = {}, {}, []
    for p in _PHASES:
        grouped = defaultdict(list)
        for row in valid_attempts[p]:
            grouped[row["item_id"]].append(row)
        resolved, sources, started = {}, Counter(), set()
        extra = [x for x in (root / p).glob("*.json") if x.stem not in plans[p]]
        if extra:
            warnings.append({"code": "unplanned_outcome_files_ignored", "file": p, "count": len(extra)})
        for item, metadata in plans[p].items():
            row, failure = _json(root / p / (item + ".json"), _compact_outcome)
            if failure:
                warnings.append({"code": failure, "file": p + "/" + item + ".json"})
            if row is not None and (row.get("manifest_id") != mid or row.get("phase") != p or row.get("item_id") != item
                                    or any(k in metadata and row.get(k) != metadata[k] for k in ("trial_id", "subset_id", "question_id", "condition_id", "repeat_index"))):
                warnings.append({"code": "outcome_identity_mismatch", "file": p + "/" + item + ".json"})
                row = None
            events = grouped[item]
            if any(e.get("event") in {"task_attempt_started", "task_attempt_completed", "task_attempt_failed"} for e in events) or row is not None:
                started.add(item)
            source = "atomic_outcome" if row is not None else "absent"
            if row is None or _classification(row) == "pending":
                terminals = [e for e in events if _terminal(e, manifest["task_max_attempts"])]
                if terminals:
                    row, source = terminals[-1], "durable_terminal_event"
            resolved[item] = row
            sources[source] += 1
            if p == "root" and row is not None:
                root_rows.append({"item_id": item, **{k: metadata[k] for k in ("question_id", "root_tie_break", "root_tie_seed")},
                                  "status": row.get("status"), "classification": _classification(row),
                                  **row.get("root_metrics", {})})
        resolved_all[p] = resolved
        buckets = Counter(_classification(row) for row in resolved.values())
        physical, seen = [], set()
        for row in valid_requests[p]:
            if row.get("event") not in {"http_attempt_started", "http_attempt_completed", "http_attempt_failed"}:
                continue
            key = (row.get("task_id"), row.get("task_attempt"), row.get("request_id"), row.get("transport_attempt"), row["event"])
            if row.get("request_id") is None or row.get("transport_attempt") is None:
                rejected[p + "_http_identity"] += 1
                continue
            if key not in seen:
                seen.add(key)
                physical.append(row)
        reservations = [r for r in physical if r["event"] == "http_attempt_started"]
        completed = [r for r in physical if r["event"] == "http_attempt_completed"]
        failures = [r for r in physical if r["event"] == "http_attempt_failed"]
        cap = manifest["budgets"].get(p + "_transport_attempts")
        snapshots = [row for row in resolved.values() if row and "scorer_snapshot" in row]
        if p == "score":
            # A scorer is shared by all subsets of a question and each result
            # contains its cumulative events. Never sum these snapshots.
            by_question = {}
            for row in snapshots:
                qid = row["question_id"]
                if qid not in by_question or row["scorer_snapshot"]["event_count"] > by_question[qid]["event_count"]:
                    by_question[qid] = row["scorer_snapshot"]
            scoring_cache = {"scope": "largest_observed_cumulative_snapshot_per_question_not_all_attempt_cost", "snapshots": by_question}
        else:
            counts = Counter()
            for row in snapshots:
                counts.update(row["scorer_snapshot"]["sources"])
            scoring_cache = {"scope": "latest_observed_artifact_per_trial_not_all_attempt_cost", "observed_trials": len(snapshots),
                             "sources": dict(counts) if snapshots else None}
        cache_bools = [row["generation_cache_hit"] for row in resolved.values() if row and isinstance(row.get("generation_cache_hit"), bool)]
        phases[p] = {"planned": len(plans[p]), **{k: buckets[k] for k in ("success", "failed", "pending", "unknown")},
                     "started": len(started), "in_progress_or_retry_pending": sum(i in started and _classification(r) == "pending" for i, r in resolved.items()),
                     "outcome_unknown": sum(row.get("outcome_unknown") is True for row in resolved.values() if row),
                     "outcome_sources": dict(sources), "task_attempt_starts": len({(r["item_id"], r.get("task_attempt")) for r in valid_attempts[p] if r.get("event") == "task_attempt_started"}),
                     "physical_attempts": len(reservations), "physical_completed": len(completed), "physical_failed": len(failures),
                     "transport_budget": {"used_reservations": len(reservations), "cap": cap, "remaining": max(0, cap - len(reservations)) if isinstance(cap, int) else None},
                     "http_errors": dict(Counter(str(r["status_code"]) if r.get("status_code") is not None else "unknown_http_status" for r in failures)),
                     "http_error_types": dict(Counter(r.get("cause_type") or "unknown" for r in failures)),
                     "physical_elapsed_ms": _stats([r.get("elapsed_ms") for r in completed + failures]),
                     "elapsed_scope": "observed_http_attempts_excluding_backoff_and_task_overhead",
                     "server_reported_tokens": {k: _stats([r.get("server_reported_" + k + "_tokens") for r in completed]) for k in ("input", "output", "total")},
                     "cache": {"generation_hits": sum(cache_bools) if cache_bools else None, "generation_observed_trials": len(cache_bools), "scoring": scoring_cache}}
    for p in _PHASES:
        if rejected[p + "_http_identity"]:
            warnings.append({"code": "physical_event_identity_missing", "file": "requests.jsonl", "phase": p, "count": rejected[p + "_http_identity"]})
    comparisons = []
    for qid in {r["question_id"] for r in root_rows}:
        rows = [r for r in root_rows if r["question_id"] == qid]
        baseline = next((r for r in rows if r["root_tie_seed"] is None), None)
        for row in rows:
            if row["root_tie_seed"] is None:
                continue
            values = {}
            for k in ("archive_hash", "selection_hash", "context_hash", "wire_payload_hash"):
                a, b = baseline.get(k) if baseline else None, row.get(k)
                values[k + "_equal"] = a == b if a is not None and b is not None else None
            comparisons.append({"question_id": qid, "variant_item_id": row["item_id"], **values})
    metrics = ("ann_calls_total", "external_candidate_count", "external_selected_count", "root_coverage",
               "max_target_state_share", "positive_signal_count", "final_memory_count")
    result = {"schema_version": 1, "manifest_id": mid, "updated_at_epoch": time.time(), "phase": phase, "state": state,
              "eligible_for_benchmark": False, "phases": phases,
              "counting_note": "success/failed/pending/unknown partition planned items; started/outcome_unknown overlap. HTTP errors are execution failures, not incorrect answers.",
              "cost_note": "Physical start reservations count retries/split children and may include crash-before-send. No stage delta is added to a total. Server token observations are incomplete, not billed-token totals.",
              "module_diagnostics": {"scope": "observed_root_trial_artifacts_only", "observed_trials": len(root_rows),
                  "completed_trials": sum(r["classification"] == "success" for r in root_rows),
                  "partial_trials": sum(r.get("partial") is True for r in root_rows),
                  "metrics": {k: _stats([r.get(k) for r in root_rows]) for k in metrics},
                  "selection_stops": dict(Counter(r.get("selection_stop") or "unavailable" for r in root_rows)),
                  "hash_comparisons": comparisons, "per_root_trial": root_rows,
                  "note": "Positive score signals are not reader benefit; missing final selections in partial failures are not empty selections."},
              "evaluation": _evaluation(root, manifest, resolved_all["generation"], warnings), "warnings": warnings}
    if len(warnings) > 100:
        result["warnings"] = warnings[:100] + [{"code": "additional_warnings_omitted", "count": len(warnings) - 100}]
    return result


def write_progress(run_dir, *, phase=None, state=None) -> dict:
    result = build_progress(run_dir, phase=phase, state=state)
    atomic_json(Path(run_dir) / "progress.json", result)
    return result


__all__ = ["build_progress", "write_progress"]
