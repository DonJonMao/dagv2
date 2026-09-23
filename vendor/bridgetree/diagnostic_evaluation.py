"""Offline-only evaluation. This is the sole diagnostic gold-label consumer."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from .diagnostic_config import file_digest
from .diagnostic_runner import atomic_json, load_manifest, read_jsonl
from .metrics import extract_option_label


def evaluate_diagnostics(root: str | Path, gold_source: str | Path) -> dict:
    root = Path(root)
    manifest = load_manifest(root)
    gold = {}
    with Path(gold_source).open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            qid = row["question_id"]
            if qid in gold:
                raise ValueError("duplicate gold question ID")
            gold[qid] = extract_option_label(row["correct_answer"])
    qids = {c["question_id"] for c in manifest["cases"]}
    if not qids <= gold.keys() or any(not gold[qid] for qid in qids):
        raise ValueError("missing or unparsable diagnostic gold")
    trials, grouped = [], {}
    for trial in manifest["trials"]:
        path = root / "generation" / (trial["trial_id"] + ".json")
        record = json.loads(path.read_text()) if path.exists() else {"status": "pending"}
        if path.exists() and (record.get("manifest_id") != manifest["manifest_id"]
                              or record.get("trial_id") != trial["trial_id"]):
            raise ValueError("generation result identity mismatch")
        successful = record["status"] == "success"
        label = extract_option_label(record.get("response", "")) if successful else None
        row = {**trial, "status": record["status"], "predicted_label": label,
               "parse_failed": not bool(label) if successful else None,
               "correct": label == gold[trial["question_id"]] if successful else None}
        trials.append(row)
        grouped.setdefault(trial["condition_id"], []).append(row)
    conditions = []
    for context in manifest["contexts"]:
        rows = grouped[context["condition_id"]]
        successful = sum(r["status"] == "success" for r in rows)
        correct = sum(r["correct"] is True for r in rows)
        conditions.append({"condition_id": context["condition_id"], "question_id": context["question_id"],
                           "kind": context["kind"], "memory_ids": context["context_plan"]["selected_ids"],
                           "planned": len(rows), "successful": successful,
                           "pending": sum(r["status"] == "pending" for r in rows),
                           "failed": sum(r["status"] not in {"pending", "success"} for r in rows),
                           "correct": correct, "parse_failures": sum(r["parse_failed"] is True for r in rows),
                           "execution_success_rate": successful / len(rows),
                           "fixed_trial_denominator_accuracy": correct / len(rows),
                           "successful_accuracy": correct / successful if successful else None,
                           "labels": dict(Counter(r["predicted_label"] for r in rows if r["predicted_label"]))})
    comparisons = []
    for case in manifest["cases"]:
        original = set(case["historical_selected_ids"])
        references = [c for c in manifest["contexts"] if c["question_id"] == case["question_id"]
                      and c["kind"] == "subset" and set(c["context_plan"]["selected_ids"]) == original]
        if len(references) != 1:
            comparisons.append({"question_id": case["question_id"], "available": False,
                                "reason": "historical selection not uniquely represented among frozen subsets"})
            continue
        reference = references[0]
        base_by_repeat = {r["repeat_index"]: r for r in grouped[reference["condition_id"]]}
        for condition in conditions:
            if condition["question_id"] != case["question_id"]:
                continue
            pairs = [(base_by_repeat[r["repeat_index"]], r) for r in grouped[condition["condition_id"]]]
            common = [(a, b) for a, b in pairs if a["status"] == b["status"] == "success"]
            comparisons.append({"question_id": case["question_id"], "available": True,
                "reference_condition_id": reference["condition_id"], "condition_id": condition["condition_id"],
                "planned_blocks": len(pairs), "common_success_blocks": len(common),
                "rescues": sum(a["correct"] is False and b["correct"] is True for a, b in common),
                "harms": sum(a["correct"] is True and b["correct"] is False for a, b in common),
                "fixed_denominator_correct_difference": sum(int(b["correct"] is True) - int(a["correct"] is True)
                                                             for a, b in pairs) / len(pairs),
                "pairing": "predeclared_repeat_blocks; not proof of independent service randomness"})
    result = {"manifest_id": manifest["manifest_id"], "gold_source_sha256": file_digest(gold_source),
              "evaluation_only_after_generation": True, "eligible_for_benchmark": False,
              "independent_question_count": len(qids), "technical_repeats_per_condition": manifest["repeats"],
              "interpretation": "Post-hoc cases, not independent repeated benchmark questions; failed trials stay in denominator.",
              "conditions": conditions, "paired_to_historical_selection": comparisons, "trials": trials}
    atomic_json(root / "evaluation.json", result)
    return result


def unified_diagnostic_report(root: str | Path) -> dict:
    root = Path(root)
    manifest = load_manifest(root)
    sections = {}
    for name in ("offline_analysis", "fresh_analysis", "score_summary", "generation_summary", "root_summary", "evaluation",
                 "score_gate", "generation_gate", "root_gate"):
        path = root / (name + ".json")
        sections[name] = json.loads(path.read_text()) if path.exists() else None
    attempts = read_jsonl(root / "attempts.jsonl")
    requests = read_jsonl(root / "requests.jsonl")
    result = {"manifest_id": manifest["manifest_id"], "source_snapshot_hash": manifest["source_snapshot_hash"],
              "scope": "PR1–PR4 diagnostics only; legacy algorithms unchanged; PR5 not implemented",
              "counts": manifest["counts"], "budgets": manifest["budgets"],
              "task_attempt_starts": sum(e.get("event") == "task_attempt_started" for e in attempts),
              "task_attempt_failures": sum(e.get("event") == "task_attempt_failed" for e in attempts),
              "physical_attempt_reservations": sum(e.get("event") == "http_attempt_started" for e in requests),
              "physical_failure_events": sum(e.get("event") == "http_attempt_failed" for e in requests),
              "cost_note": "All attempt events included; stage deltas never added to task totals. Reservations may include crash-before-send.",
              "sections": sections,
              "mechanism_directions_not_implemented": ["separate relevance, evidence contribution and sufficiency",
                  "conditional investigation of revisable selection", "conditional investigation of cross-target scheduling"],
              "unresolved": ["reader benefit of rejected historical evidence", "task utility interpretation of R",
                             "joint archive/path/greedy limitations", "deployment and service repeatability"]}
    atomic_json(root / "diagnostic_report.json", result)
    return result
