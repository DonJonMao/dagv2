"""Synthetic export scaffolds validate the real offline raw-review adapter.

No real personal histories are checked into these fixtures.
"""
from copy import deepcopy
import json

import pytest

from scripts.replay_bt_v3 import replay, replay_case, source_documents


def exported_case():
    memories = []
    for index in range(2):
        text = f"Header\nPrivate fixture marker {index}: I enjoy reading.\nAssistant: suggestion."
        user_end = text.index("\nAssistant:")
        memories.append({"memory_id": f"q:m{index}", "text": text,
                         "metadata": {"source_segments": [
                             {"start": 7, "end": user_end, "role": "user", "source_message_indices": [index * 2]},
                             {"start": user_end + 1, "end": len(text), "role": "assistant",
                              "source_message_indices": [index * 2 + 1]}]}})
    visible = {"question_id": "q", "persona_id": "p", "visible_memories": memories}
    baseline = [memory["memory_id"] for memory in memories]
    evidence = {"mappings": [], "candidate_ids": baseline, "selected_ids": [],
                "requirements": [{"id": "history", "description": "Earlier reading preferences",
                                   "necessary": True, "time_scope": "unknown"}],
                "requests": [{"messages": [{"role": "system", "content": "Frozen prior prompt"},
                                            {"role": "user", "content": json.dumps({"query": "Which book suits me?"})}]}]}
    context = {"selected_ids": baseline, "token_count": 100, "budget": 8192,
               "serialized_context": "\n".join(memory["text"] for memory in memories)}
    return evidence, visible, baseline, context


def test_real_view_restores_mapping_free_histories_without_coverage_or_calls():
    result = replay_case(*exported_case(), "q")
    assert result["all_baseline_visible"]
    assert result["complete_source_text_and_metadata_exact"]
    assert result["candidate_ids_match_visible_raw"]
    assert result["ablation_same_baseline_ids"] and result["ablation_raw_count"] == 0
    assert result["within_input_budget"]
    assert result["structural_covered_requirement_ids"] == []
    assert result["new_model_calls"] == result["new_answers"] == 0


def test_source_roles_use_metadata_and_leave_headers_unknown():
    _, visible, _, _ = exported_case()
    docs = source_documents(visible)
    segments = docs["q:m0"].metadata["source_segments"]
    assert segments[0]["role"] == "unknown" and segments[0]["start"] == 0
    assert [row["role"] for row in segments if row["provenance"] == "authoritative"] == ["user", "assistant"]
    del visible["visible_memories"][0]["metadata"]["source_segments"]
    with pytest.raises(ValueError, match="authoritative"):
        source_documents(visible)


def test_replay_rejects_nonzero_mappings_and_changed_dense_context():
    evidence, visible, baseline, context = exported_case()
    changed = deepcopy(evidence)
    changed["mappings"] = [{"id": "cannot-be-cast-to-DAG-proof"}]
    with pytest.raises(ValueError, match="zero mappings"):
        replay_case(changed, visible, baseline, context, "q")
    context["serialized_context"] = "Incomplete source"
    with pytest.raises(ValueError, match="complete baseline source"):
        replay_case(evidence, visible, baseline, context, "q")


def write_export(root):
    evidence, visible, baseline, context = exported_case()
    records = {
        "outcomes/eb.json": {"task": {"persona_id": "p", "question_id": "q", "method_id": "evidence_bridge", "task_id": "eb"},
                             "status": "success", "selected_ids": [], "diagnostics": {
                                 "evidence_bridge_summary": {"reliability": {"reliability_status": "normal"}}}},
        "outcomes/dense.json": {"task": {"persona_id": "p", "question_id": "q", "method_id": "dense", "task_id": "dense"},
                                "status": "success", "selected_ids": baseline},
        "candidate_pool/eb.json": {"task_id": "eb", "evidence_selection": evidence},
        "visible_memories/q-hash.json": visible,
        "modules/context.jsonl": {"method_id": "dense", "task_id": "dense", "context_plan": context},
    }
    for name, value in records.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n")


def test_report_excludes_source_text_preserves_export_and_requires_fresh_output(tmp_path):
    run, output = tmp_path / "source", tmp_path / "report"
    write_export(run)
    before = {path: path.read_bytes() for path in run.rglob("*") if path.is_file()}
    result = replay(run, output)
    assert result["summary"]["zero_mapping_cases_replayed"] == 1
    report = (output / "replay_report.json").read_text()
    assert "Private fixture marker" not in report and "Which book suits me" not in report
    assert all(path.read_bytes() == content for path, content in before.items())
    with pytest.raises(ValueError, match="fresh"):
        replay(run, output)
    with pytest.raises(ValueError, match="outside"):
        replay(run, run / "report")


def test_explicit_case_selection_does_not_silently_skip_missing_questions(tmp_path):
    run = tmp_path / "source"
    write_export(run)
    with pytest.raises(ValueError, match="No matching"):
        replay(run, tmp_path / "report", ["unavailable"])
