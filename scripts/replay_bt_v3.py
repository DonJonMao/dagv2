#!/usr/bin/env python3
"""Replay exported zero-mapping BT histories through the DAG v3 raw input view.

No model, retrieval, or answer generation is performed. Old BT requirements are
represented as unknown DAG nodes; this is a source-visibility harness, not a
claim that the two methods have equivalent plans or execution state. Reports
contain identifiers, hashes, and counts, never source text or provider settings.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dagbt.engine import Engine
from dagbt.model_runtime import BTTokenAccounting
from dagbt.resources import ProvenanceDocument
from dagbt.support import select_support
from dagbt.transport import digest


class NoCalls:
    def get(self, *args, **kwargs):
        raise AssertionError("Offline source replay must never call a service")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def original_query(evidence):
    queries = {json.loads(row["messages"][1]["content"])["query"]
               for row in evidence["requests"] if row.get("messages")}
    if len(queries) != 1:
        raise ValueError("Exported requests must contain one consistent query")
    return queries.pop()


def source_documents(visible):
    result = {}
    for memory in visible["visible_memories"]:
        identifier = memory["memory_id"]
        if identifier in result:
            raise ValueError("Duplicate source memory ID")
        if "source_segments" not in memory["metadata"]:
            raise ValueError("Replay requires authoritative source segment metadata")
        result[identifier] = ProvenanceDocument(identifier, "", memory["text"],
                                               deepcopy(memory["metadata"]))
    return result


def replay_case(evidence, visible, baseline_ids, context_plan, question_id):
    """Use current production payload assembly, freezing the mapping-free source."""
    from dagbt.final_selection import FinalSelector

    if evidence["mappings"]:
        raise ValueError("This minimal replay only accepts exports with zero mappings")
    docs = source_documents(visible)
    candidates = evidence["candidate_ids"]
    if len(set(candidates)) != len(candidates) or not set(candidates) <= set(docs):
        raise ValueError("Candidate IDs must name unique visible source memories")
    if len(set(baseline_ids)) != len(baseline_ids) or not set(baseline_ids) <= set(candidates):
        raise ValueError("Dense baseline IDs must be unique discovered candidates")
    if context_plan["selected_ids"] != baseline_ids or context_plan["token_count"] > context_plan["budget"]:
        raise ValueError("Recorded dense reader context does not validate the baseline")
    if any(docs[i].passage not in context_plan["serialized_context"] for i in baseline_ids):
        raise ValueError("Recorded dense reader context omits a complete baseline source")
    config = {"_test_transport": True, "model_profile": "bridgetree",
              "llm_base_url": "https://invalid.example/v1", "llm_model": "offline-replay",
              "embedding_model": "offline-replay", "fusion": {
                  "selection_review": True, "raw_memory_review": True,
                  "allow_unassessed_coverage": True}}
    query = original_query(evidence)
    engine = Engine({"id": question_id, "question": query},
                    (docs, list(docs), None, None, BTTokenAccounting()), NoCalls(), config, "fusion",
                    reader_question=query)
    requirements = evidence["requirements"]
    engine.steps = [{"question": row["description"], "output_slot": row["id"],
                     "answer_type": "personal_history", "inputs": []} for row in requirements]
    engine.nodes = [engine.unknown(step) for step in engine.steps]
    engine.requirements = [{"id": row["id"], "description": row["description"],
                            "necessary": row["necessary"], "terminal_node_ids": [row["id"]],
                            "terminal_mode": "all", "time_scope": row.get("time_scope")}
                           for row in requirements]
    engine.requirements_hash = digest(engine.requirements)
    engine.candidates = list(candidates)
    engine.baseline_ids = list(baseline_ids)
    graph = engine.compile()
    proposal = select_support(graph, engine.feasible)
    view = FinalSelector(engine, graph, proposal).prepare_view()
    raw = {row["doc_id"]: row for row in view.data["raw_memory_candidates"]}
    exact = all(row["passage"] == docs[i].passage and row["metadata"] == docs[i].metadata
                for i, row in raw.items())
    if not exact or not set(raw) <= set(baseline_ids):
        raise AssertionError("Raw view changed a source or exposed a non-baseline source")
    if view.data["evidence"] or proposal["covered_requirement_ids"]:
        raise AssertionError("Raw visibility must not fabricate evidence or structural coverage")
    engine.s["raw_memory_review"] = False
    off_view = FinalSelector(engine, graph, proposal).prepare_view()
    if off_view.data.get("raw_memory_candidates"):
        raise AssertionError("Raw review ablation must omit the raw channel")
    if engine.reasoner.requests or engine.ledger.used or engine.reasoner.sequence:
        raise AssertionError("Visibility-only replay performed a metered operation")
    audit = view.audit
    return {
        "question_id": question_id,
        "source_query_sha256": hashlib.sha256(engine.q["question"].encode()).hexdigest(),
        "source_requirements_sha256": digest(requirements),
        "source_mapping_count": 0, "source_candidate_count": len(candidates),
        "source_selected_ids": list(evidence["selected_ids"]),
        "baseline_ids": list(baseline_ids), "raw_review_ids": list(raw),
        "omitted_baseline_ids": [i for i in baseline_ids if i not in raw],
        "all_baseline_visible": set(raw) == set(baseline_ids),
        "complete_source_text_and_metadata_exact": exact,
        "input_tokens_estimate": audit["input_tokens_after"],
        "input_token_limit": audit["input_token_limit"],
        "within_input_budget": audit["input_tokens_after"] <= audit["input_token_limit"],
        "input_view_audit": {key: value for key, value in audit.items()
                             if key in {"input_truncated",
                                        "raw_review_visible_doc_ids", "omitted_raw_review_doc_ids"}},
        "candidate_ids_match_visible_raw": set(view.data["candidate_doc_ids"]) == set(raw),
        "ablation_same_baseline_ids": engine.baseline_ids == list(baseline_ids),
        "ablation_raw_count": len(off_view.data.get("raw_memory_candidates", [])),
        "structural_covered_requirement_ids": proposal["covered_requirement_ids"],
        "new_model_calls": 0, "new_answers": 0,
    }


def replay(run_dir, output_dir, question_ids=()):
    run, output = Path(run_dir).resolve(), Path(output_dir).resolve()
    if output == run or run in output.parents:
        raise ValueError("Output must be outside the read-only source run")
    destination = output / "replay_report.json"
    if destination.exists():
        raise ValueError("Use a fresh replay output directory")
    indexed, paths = {}, {}
    for path in sorted((run / "outcomes").glob("*.json")):
        value = read_json(path)
        task = value["task"]
        key = (str(task["persona_id"]), task["question_id"], task["method_id"])
        if key in indexed or path.stem != task["task_id"]:
            raise ValueError("Duplicate or mismatched authoritative outcome identity")
        indexed[key], paths[key] = value, path
    contexts = {}
    context_path = run / "modules/context.jsonl"
    for line_number, line in enumerate(context_path.open(encoding="utf-8"), 1):
        value = json.loads(line)
        if value.get("method_id") == "dense" and value.get("context_plan"):
            contexts[value["task_id"]] = (value["context_plan"], line_number)
    wanted = set(question_ids)
    cases, normal_count = [], 0
    for key, outcome in sorted(indexed.items()):
        persona, question, method = key
        normal = outcome.get("diagnostics", {}).get("evidence_bridge_summary", {}).get(
            "reliability", {}).get("reliability_status") == "normal"
        if method != "evidence_bridge" or outcome["status"] != "success" or not normal:
            continue
        normal_count += 1
        if wanted and question not in wanted:
            continue
        task_id = outcome["task"]["task_id"]
        pool_path = run / "candidate_pool" / (task_id + ".json")
        pool = read_json(pool_path)
        if pool["task_id"] != task_id:
            raise ValueError("Candidate pool task identity mismatch")
        evidence = pool["evidence_selection"]
        if evidence["mappings"]:
            if wanted:
                raise ValueError("Requested question has mappings; this replay requires zero mappings")
            continue
        dense = indexed[(persona, question, "dense")]
        if dense["status"] != "success":
            raise ValueError("Replay requires a successful paired dense task")
        dense_task_id = dense["task"]["task_id"]
        visible_paths = list((run / "visible_memories").glob(question + "-*.json"))
        if len(visible_paths) != 1:
            raise ValueError("Expected exactly one visible-memory artifact")
        visible = read_json(visible_paths[0])
        if (str(visible["persona_id"]), visible["question_id"]) != (persona, question):
            raise ValueError("Visible-memory artifact identity mismatch")
        plan, line_number = contexts[dense_task_id]
        case = replay_case(evidence, visible, dense["selected_ids"], plan, question)
        case.update(task_id=task_id, dense_task_id=dense_task_id, persona_id=persona,
                    candidate_pool_path=str(pool_path.relative_to(run)),
                    candidate_pool_sha256=file_hash(pool_path),
                    outcome_sha256=file_hash(paths[key]),
                    visible_memories_path=str(visible_paths[0].relative_to(run)),
                    visible_memories_sha256=file_hash(visible_paths[0]),
                    dense_context_module_line=line_number)
        cases.append(case)
    if not cases or wanted - {case["question_id"] for case in cases}:
        raise ValueError("No matching normal, paired, zero-mapping export case")
    report = {
        "schema_version": 1, "mode": "bt_export_zero_mapping_dag_v3_raw_view_only",
        "source_run": str(run), "source_context_sha256": file_hash(context_path),
        "implementation_sha256": {name: file_hash(ROOT / name) for name in (
            "scripts/replay_bt_v3.py", "dagbt/final_selection.py", "dagbt/evidence_views.py",
            "dagbt/prompts.py", "dagbt/config.py")},
        "interpretation": (
            "Original BT requirements become unknown independent DAG nodes. Only source visibility "
            "and provenance are replayed; no original DAG plan, model selection, answer, semantic "
            "recovery, or accuracy gain is claimed. Nonzero-mapping cases are outside this harness. "
            "Dense IDs are the exported BT baseline, not a newly retrieved DAG baseline."),
        "summary": {"normal_source_tasks": normal_count, "zero_mapping_cases_replayed": len(cases),
                    "all_baseline_visible_cases": sum(case["all_baseline_visible"] for case in cases),
                    "complete_text_and_metadata_exact": all(
                        case["complete_source_text_and_metadata_exact"] for case in cases),
                    "all_payloads_within_input_budget": all(case["within_input_budget"] for case in cases),
                    "raw_visible_count": sum(len(case["raw_review_ids"]) for case in cases),
                    "baseline_count": sum(len(case["baseline_ids"]) for case in cases)},
        "new_model_calls": 0, "new_answers": 0, "cases": cases,
    }
    output.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--question-id", action="append", default=[])
    args = parser.parse_args()
    report = replay(args.run_dir, args.output_dir, args.question_id)
    print(json.dumps({"summary": report["summary"], "new_model_calls": 0, "new_answers": 0}, indent=2))


if __name__ == "__main__":
    main()
