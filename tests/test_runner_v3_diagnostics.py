"""Evidence/coverage diagnostics and operations are offline and outcome-based."""
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from dagbt import runner as r


def outcome(unit, *, evidence=None, complete=None, ok=True, cost=None):
    semantic = {}
    if evidence is not None:
        mapped = ["mapped"] if evidence in {"mapped_only", "mixed"} else []
        raw = ["raw"] if evidence in {"raw_only", "mixed"} else []
        semantic = {"evidence_state": evidence, "selected_doc_ids": mapped + raw,
                    "selected_mapped_doc_ids": mapped, "selected_raw_only_doc_ids": raw}
    if complete is not None:
        semantic.update(coverage_validation_complete=complete,
                        unassessed_requirement_ids=[] if complete else ["answer"])
    return {"unit_id": unit, "answer": {"status": "ok" if ok else "execution_failed", "prediction": "Y"},
            "budgets": {str(k): {"selected_doc_ids": []} for k in (5, 10, 20)},
            "diagnostics": {"reliability": {"cohort": "normal"}, "semantic_evidence": semantic},
            "runner": {"cost": cost} if cost is not None else {}}


def scored(row, output):
    return {"valid": row["answer"]["status"] == "ok", "em": 1.,
            "modules": r.module_metrics(row, {"gold_groups": [["a"]]}, "hotpotqa", output),
            "latest_attempt_cost": row.get("runner", {}).get("cost")}


def test_normal_reliability_does_not_imply_nonempty_evidence_or_complete_coverage(tmp_path):
    rows = [scored(outcome("empty", evidence="empty_context", complete=True), tmp_path),
            scored(outcome("raw", evidence="raw_only", complete=False), tmp_path),
            scored(outcome("old"), tmp_path)]
    assert r.summarize_reliability(rows, "em")["completion_cohorts"]["normal"]["tasks"] == 3
    summary = r.summarize_semantic_evidence(rows, "em")
    assert summary["all_task_evidence_state_counts"] == {
        "empty_context": 1, "mapped_only": 0, "raw_only": 1, "mixed": 0, "unknown": 1}
    assert summary["all_task_coverage_state_counts"] == {"complete": 1, "unassessed": 1, "unknown": 1}
    assert {tuple(row[k] for k in ("reliability", "evidence_state", "coverage_state"))
            for row in summary["successful_reliability_evidence_coverage_counts"]} == {
        ("normal", "empty_context", "complete"), ("normal", "raw_only", "unassessed"),
        ("normal", "unknown", "unknown")}


@pytest.mark.parametrize("unassessed,complete", [([], False), (["answer"], True), (["answer"], None)])
def test_unassessed_signal_overrides_stale_complete_state(unassessed, complete):
    result = r.normalized_semantic_evidence({"semantic_evidence": {
        "coverage_state": "complete", "coverage_validation_complete": complete,
        "unassessed_requirement_ids": unassessed}})
    assert result["coverage_state"] == "unassessed"


def test_selected_partition_overrides_state_and_inconsistency_is_unknown():
    result = r.normalized_semantic_evidence({"semantic_evidence": {
        "evidence_state": "mapped_only", "selected_doc_ids": ["a", "b"],
        "selected_mapped_doc_ids": ["a"], "selected_raw_only_doc_ids": ["b"]}})
    assert result["evidence_state"] == "mixed"
    result["selected_raw_only_doc_ids"] = ["a"]
    invalid = r.normalized_semantic_evidence({"semantic_evidence": result})
    assert invalid["evidence_state"] == "unknown" and "state_integrity" in invalid


@pytest.mark.parametrize("phase", [None, "selection_pending"])
def test_no_legal_selection_header_is_unknown_even_with_empty_snapshot_lists(phase):
    semantic = {"evidence_state": "unknown", "selected_doc_ids": [],
                "selected_mapped_doc_ids": [], "selected_raw_only_doc_ids": [],
                "coverage_validation_complete": None, "unassessed_requirement_ids": []}
    if phase is not None:
        semantic["phase"] = phase
    result = r.normalized_semantic_evidence({"semantic_evidence": semantic})
    assert result["evidence_state"] == result["coverage_state"] == "unknown"


def test_authoritative_reasoning_counter_includes_failed_call_after_selection_snapshot(tmp_path):
    row = outcome("q", ok=False)
    row["diagnostics"].pop("semantic_evidence")
    row["attempt_directories"] = ["latest"]
    r.save(tmp_path / "latest/fusion_partial.json", {"diagnostics": {
        "semantic_evidence": {"evidence_state": "unknown", "phase": "selection_pending",
            "selected_doc_ids": [], "selected_mapped_doc_ids": [], "selected_raw_only_doc_ids": [],
            "evidence_logical_calls": 3}, "reasoning_call_count": 4}})
    metrics = scored(row, tmp_path)["modules"]
    assert metrics["semantic_evidence"]["evidence_logical_calls"] == 4
    assert metrics["semantic_evidence"]["evidence_state"] == "unknown"


@pytest.mark.parametrize("ok", [True, False])
def test_only_failed_current_row_recovers_latest_semantic_snapshot(tmp_path, ok):
    row = outcome("q", ok=ok)
    row["diagnostics"].pop("semantic_evidence")
    row["attempt_directories"] = ["old", "latest"]
    r.save(tmp_path / "old/fusion_partial.json", outcome("q", evidence="mapped_only", complete=True))
    r.save(tmp_path / "latest/fusion_partial.json", outcome("q", evidence="raw_only", complete=False))
    semantic = scored(row, tmp_path)["modules"]["semantic_evidence"]
    assert semantic["evidence_state"] == ("unknown" if ok else "raw_only")
    assert semantic["coverage_state"] == ("unknown" if ok else "unassessed")
    assert "semantic_evidence" not in row["diagnostics"]


def test_semantic_call_counter_uses_current_diagnostic_not_event_copies(tmp_path):
    row = outcome("q", evidence="mixed", complete=False)
    row["diagnostics"]["semantic_evidence"].update(evidence_logical_calls=3, budgeted_llm_attempts=5,
        reader_logical_calls=1, baseline_retention_ratio=.5)
    row["diagnostics"]["events"] = [{"event": "snapshot", "evidence_logical_calls": 99}] * 5
    summary = r.summarize_semantic_evidence([scored(row, tmp_path)])
    observed = summary["current_task_diagnostic_metrics"]
    assert observed["evidence_logical_calls"]["sum"] == 3
    assert observed["budgeted_llm_attempts"]["sum"] == 5
    assert observed["reader_logical_calls"]["sum"] == 1
    assert observed["baseline_retention_ratio"]["mean"] == .5


def test_early_failure_reasoning_count_survives_latest_partial_recovery(tmp_path):
    row = outcome("q", ok=False)
    row["diagnostics"].pop("semantic_evidence")
    row["attempt_directories"] = ["latest"]
    r.save(tmp_path / "latest/fusion_partial.json", {"diagnostics": {
        "semantic_evidence": {}, "reasoning_call_count": 4}})
    semantic = scored(row, tmp_path)["modules"]["semantic_evidence"]
    assert semantic["evidence_state"] == semantic["coverage_state"] == "unknown"
    assert semantic["evidence_logical_calls"] == 4


def record_request(directory, ref, kind, *, attempts=1, tokens=10, seeded=False):
    physical = [{"kind": kind, "http_status": 503 if i < attempts - 1 else 200,
                 "started_unix": i * 10, "finished_unix": i * 10 + 2} for i in range(attempts)]
    response = {} if tokens is None else {"usage": {"total_tokens": tokens}}
    r.save(directory / "requests" / f"{ref}.json", {"stage": ref, "attempts": physical, "response": response})
    r.append_jsonl(directory / "call_events.jsonl", {
        "event": "call_started", "kind": kind, "request_ref": ref, "cache_hit": seeded})


def test_cost_layers_distinguish_logical_retries_seeded_replay_and_reader(tmp_path):
    directory = tmp_path / "hotpotqa/fusion/attempts/q/attempt-002"
    record_request(directory, "reasoning", "llm", attempts=3, tokens=30)
    r.append_jsonl(directory / "call_events.jsonl", {
        "event": "call_started", "kind": "llm", "request_ref": "reasoning", "cache_hit": True})
    record_request(directory, "seed", "llm", attempts=2, tokens=900, seeded=True)
    r.save(directory / "seeded_requests.json", ["seed"])
    record_request(directory, "reader", "reader", tokens=15)
    record_request(directory, "embed", "embedding_http", tokens=4)
    record_request(directory, "rank", "rerank_http", tokens=None)
    cost = r.request_cost(directory)
    assert cost["reasoning_logical_calls"] == 3 and cost["physical_llm_attempts"] == 3
    assert cost["reader_logical_calls"] == cost["physical_reader_attempts"] == 1
    assert cost["logical_calls"] == 6 and cost["http_attempts"] == 6 and cost["http_retries"] == 2
    assert cost["cache_hits"] == 2 and cost["unique_requests"] == 5
    assert cost["total_tokens"] == 49 and cost["token_usage_complete"] is False
    assert cost["by_kind"]["llm"]["observed_http_seconds"] == 6
    assert cost["by_kind"]["rerank_http"]["missing_usage_responses"] == 1
    assert cost["by_kind"]["reader"]["total_tokens"] == 15
    assert r.total_cost(tmp_path, "hotpotqa", "fusion")["by_kind"] == cost["by_kind"]
    stats = r.latest_cost_statistics([{"latest_attempt_cost": cost}, {}])
    assert stats["metrics"]["by_kind.llm.http_attempts"] == {
        "observed_tasks": 1, "missing_tasks": 1, "sum": 3, "mean": 3}


def test_old_untyped_cost_is_unknown_and_missing_usage_is_not_complete(tmp_path):
    r.save(tmp_path / "requests/old.json", {"attempts": [{"http_status": 200}], "response": {}})
    cost = r.request_cost(tmp_path)
    assert cost["by_kind"]["unknown"]["http_attempts"] == 1
    assert cost["by_kind"]["unknown"]["missing_usage_responses"] == 1
    assert cost["token_usage_complete"] is False


def test_read_only_diagnostics_deduplicates_outcomes_and_never_opens_labels(tmp_path, monkeypatch, capsys):
    r.save(tmp_path / "manifest.json", {"datasets": ["hotpotqa"], "arms": ["fusion"],
                                       "question_ids": {"hotpotqa": ["success", "failed", "pending"]}})
    success = outcome("success", evidence="mixed", complete=False, cost={"total_tokens": 10})
    failure = outcome("failed", ok=False)
    failure["diagnostics"].pop("semantic_evidence")
    failure["attempt_directories"] = ["hotpotqa/fusion/attempts/failed/attempt-001"]
    for row in (success, failure):
        r.save(r.result_path(tmp_path, "hotpotqa", "fusion", row["unit_id"]), row)
    latest = tmp_path / failure["attempt_directories"][-1]
    r.save(latest / "fusion_partial.json", outcome("failed", evidence="raw_only", complete=False))
    r.save(latest / "result.json", outcome("failed", evidence="mapped_only", complete=True))
    r.save(tmp_path / "hotpotqa/fusion/failures/old.json", success)
    r.save(r.result_path(tmp_path, "hotpotqa", "fusion", "outside_manifest"), success)
    r.append_jsonl(latest / "fusion_events.jsonl", success)
    (tmp_path / "evaluation_only.json").write_text("invalid labels must remain unread")
    monkeypatch.setattr(r, "score_all", lambda *a, **kw: pytest.fail("diagnostics called scoring"))
    monkeypatch.setattr(r, "load_questions", lambda *a, **kw: pytest.fail("diagnostics opened data"))
    actual_load = r.load
    def guarded_load(path):
        assert "evaluation_only" not in str(path)
        return actual_load(path)
    monkeypatch.setattr(r, "load", guarded_load)
    before = {str(p): (p.stat().st_mtime_ns, p.read_bytes()) for p in tmp_path.rglob("*") if p.is_file()}
    assert r.main(["diagnostics", "--output", str(tmp_path)]) == 0
    report = json.loads(capsys.readouterr().out)
    current = report["datasets"]["hotpotqa"]["fusion"]
    assert current["expected_tasks"] == 3 and current["current_tasks"] == 2
    assert current["pending_or_invalid_tasks"] == 1
    semantic = current["semantic_evidence"]
    assert semantic["completion_cohorts"]["mixed"]["tasks"] == 1
    assert semantic["failed_evidence_state_counts"]["raw_only"] == 1
    assert semantic["all_task_coverage_state_counts"]["unassessed"] == 2
    assert report["labels_read"] is False and report["answer_accuracy_available"] is False
    assert r.diagnostics(tmp_path) == report
    after = {str(p): (p.stat().st_mtime_ns, p.read_bytes()) for p in tmp_path.rglob("*") if p.is_file()}
    assert before == after


def test_missing_output_diagnostics_does_not_create_run_directory(tmp_path):
    output = tmp_path / "not-created"
    assert r.diagnostics(output)["state"] == "no_manifest"
    assert not output.exists()


@pytest.mark.parametrize("action,runner_command", [("start", "launch"), ("resume", "launch"),
    ("status", "status"), ("stop", "stop"), ("diagnostics", "diagnostics"), ("preflight", "preflight")])
def test_v3_wrapper_reuses_coordinator_with_new_default_output(tmp_path, action, runner_command):
    recorder = tmp_path / "fake python"
    recorder.write_text("#!/usr/bin/env python3\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n")
    recorder.chmod(0o755)
    result = subprocess.run([str(r.ROOT / "scripts/run_v3.sh"), action],
                            env={**os.environ, "DAGBT_PYTHON": str(recorder)}, capture_output=True, text=True, check=True)
    args = json.loads(result.stdout)
    assert args[:3] == ["-m", "dagbt.runner", runner_command]
    if action != "preflight":
        assert args[args.index("--output") + 1] == str(r.ROOT / "outputs/paired_local_terminal_v1")
    if action in ('start','resume'):
        assert args[args.index('--arms')+1:args.index('--arms')+3] == ['original','dagbt_local_terminal_v1']
    assert "--retry-failed" not in args
    if action != "preflight":
        override = subprocess.run([str(r.ROOT / "scripts/run_v3.sh"), action,
                                   "--output", str(tmp_path / "custom output")],
                                  env={**os.environ, "DAGBT_PYTHON": str(recorder)}, capture_output=True, text=True, check=True)
        assert json.loads(override.stdout)[-2:] == ["--output", str(tmp_path / "custom output")]


@pytest.mark.parametrize("dataset,answer_metric", [("hotpotqa", "em"), ("personamem", "accuracy")])
def test_score_all_exposes_independent_semantic_and_coverage_cohorts(tmp_path, dataset, answer_metric):
    questions = [{"id": str(i), "question": "Q?", "options": ["(a) Y", "(b) N"], "persona_id": "p"}
                 for i in range(3)]
    labels = [{"id": q["id"], "answers": ["Y"], "gold_groups": [["a"]], "correct_answer": "(a)"}
              for q in questions]
    for q, evidence, complete in zip(questions, ["empty_context", "raw_only", "mixed"], [True, False, True]):
        row = outcome(q["id"], evidence=evidence, complete=complete, ok=q["id"] != "2")
        row["answer"]["prediction"] = "(a)" if dataset == "personamem" else "Y"
        r.save(r.result_path(tmp_path, dataset, "fusion", q["id"]), row)
    def summarize():
        return r.score_all(tmp_path, {dataset: questions}, ["fusion"], label_loader=lambda _: labels)
    summary = summarize()[dataset]["arms"]["fusion"]
    assert summary["reliability"]["completion_cohorts"]["normal"]["tasks"] == 2
    semantic = summary["semantic_evidence"]
    assert semantic["answer_metric"] == answer_metric
    assert semantic["completion_cohorts"]["empty_context"][answer_metric] == 1
    assert semantic["coverage_completion_cohorts"]["unassessed"]["tasks"] == 1
    assert semantic["completion_cohorts"]["mixed"]["tasks"] == 0
    assert semantic["failed_evidence_state_counts"]["mixed"] == 1
    assert summarize()[dataset]["arms"]["fusion"]["semantic_evidence"] == semantic


@pytest.mark.parametrize("arm", ["original", "fusion"])
def test_personamem_worker_keeps_options_only_in_fusion_reader(tmp_path, monkeypatch, arm):
    from dagbt import engine, personamem, resources
    question = {"id": "q", "question": "Pick a drink\nOptions:\n(a) Tea\n(b) Coffee",
                "user_question": "Pick a drink", "scope_id": "p:1", "shared_context_id": "p",
                "end_index": 1, "options": ["(a) Tea", "(b) Coffee"]}
    visible = ({}, ["memory"], None, None, None)
    pipeline = SimpleNamespace(e=SimpleNamespace(CONFIG={}), prepare=lambda *a: (None, visible, {}))
    actual_import = r.importlib.import_module
    monkeypatch.setattr(r.importlib, "import_module", lambda name, *a: pipeline if name == "pipeline_dagv2"
                        else actual_import(name, *a))
    monkeypatch.setattr(r, "configure_personamem_reader", lambda: None)
    monkeypatch.setattr(r, "load_questions", lambda _: [question])
    monkeypatch.setattr(r, "AuditedCalls", lambda *a: object())
    monkeypatch.setattr(personamem, "load_scopes", lambda _: {"p:1": ["memory"]})
    monkeypatch.setattr(resources, "scope_resources", lambda *a: visible)
    captured = []
    def run(q, *a, **kw):
        captured.append((q, kw))
        result = outcome("q")
        result["answer"]["prediction"] = "(a)"
        return result
    monkeypatch.setattr(r, "original_question", run)
    monkeypatch.setattr(engine, "run_question", run)
    tasks = iter([{"question": question, "output": str(tmp_path)}, None])
    messages = []
    connection = SimpleNamespace(recv=lambda: next(tasks), send=messages.append)
    r.native_worker(connection, "personamem", arm, {})
    assert [message["type"] for message in messages] == ["ready", "result"]
    q, kw = captured[0]
    assert set(q) == {"id", "question"}
    if arm == "fusion":
        assert q["question"] == question["user_question"] and "Options:" not in q["question"]
        assert kw["reader_question"] == question["question"]
    else:
        assert q["question"] == question["question"] and "reader_question" not in kw
