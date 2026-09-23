"""Offline tests exercise actual spawned processes and the paired coordinator."""
import json
import os
import time
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from dagbt import runner as r


def valid_row(unit, prediction="Y"):
    return {"unit_id": unit, "answer": {"status": "ok", "prediction": prediction}, "ranking": {},
            "budgets": {str(k): {"selected_doc_ids": ["doc"]} for k in (5, 10, 20)},
            "diagnostics": {"worker_pid": os.getpid()}}


def fake_worker(connection, dataset, arm, config):
    connection.send({"type": "ready", "hashes": {}})
    while True:
        try:
            task = connection.recv()
        except EOFError:
            return
        if task is None:
            return
        unit = task["question"]["id"]
        if unit == "crash" and arm == "original":
            os._exit(17)
        if unit == "timeout" and arm == "original":
            time.sleep(30)
        if unit == "retry" and arm == "original" and "attempt-001" in task["output"]:
            connection.send({"type": "task_error", "error": "fixture transient failure"})
        else:
            connection.send({"type": "result", "row": valid_row(unit)})


def fake_factory(dataset, arm, config):
    return r.Worker(dataset, arm, config, target=fake_worker)


CONFIG = {"experiment": {"worker_startup_timeout_seconds": 10, "question_timeout_seconds": 1, "max_question_attempts": 3}}


def questions(*ids):
    return [{"id": unit, "question": "Question " + unit} for unit in ids]


def test_crash_isolation_same_question_pairs_and_worker_reuse(tmp_path):
    qs = questions("crash", "next")
    r.generate_dataset(tmp_path, "hotpotqa", qs, ["original", "fusion"], CONFIG, worker_factory=fake_factory)
    rows = {(a, q["id"]): r.load(r.result_path(tmp_path, "hotpotqa", a, q["id"])) for a in ("original", "fusion") for q in qs}
    assert rows["original", "crash"]["answer"]["status"] == "execution_failed"
    assert all(rows[k]["answer"]["status"] == "ok" for k in rows if k != ("original", "crash"))
    assert rows["fusion", "crash"]["diagnostics"]["worker_pid"] == rows["fusion", "next"]["diagnostics"]["worker_pid"]
    assert r.all_terminal(tmp_path, {"hotpotqa": qs}, ["original", "fusion"])
    assert len(list((tmp_path / "hotpotqa/original/failures").glob("*.json"))) == 1


def test_timeout_does_not_abort_other_arm_or_next_question(tmp_path):
    cfg = {"experiment": {**CONFIG["experiment"], "question_timeout_seconds": .15}}
    qs = questions("timeout", "next")
    started = time.monotonic()
    r.generate_dataset(tmp_path, "hotpotqa", qs, ["original", "fusion"], cfg, worker_factory=fake_factory)
    assert time.monotonic() - started < 15
    assert r.load(r.result_path(tmp_path, "hotpotqa", "original", "timeout"))["answer"]["status"] == "timeout"
    assert r.load(r.result_path(tmp_path, "hotpotqa", "original", "next"))["answer"]["status"] == "ok"


def test_resume_requires_explicit_retry_and_preserves_attempts(tmp_path):
    qs = questions("retry")
    args = (tmp_path, "hotpotqa", qs, ["original", "fusion"], CONFIG)
    r.generate_dataset(*args, worker_factory=fake_factory)
    original = r.result_path(tmp_path, "hotpotqa", "original", "retry")
    fusion = r.result_path(tmp_path, "hotpotqa", "fusion", "retry")
    fusion_bytes = fusion.read_bytes()
    assert r.load(original)["answer"]["status"] == "execution_failed"
    r.generate_dataset(*args, worker_factory=fake_factory)
    assert r.load(original)["runner"]["attempt"] == 1
    r.generate_dataset(*args, retry_failed=True, worker_factory=fake_factory)
    assert r.load(original)["answer"]["status"] == "ok"
    assert r.load(original)["runner"]["attempt"] == 2
    assert fusion.read_bytes() == fusion_bytes
    assert len(list((tmp_path / "hotpotqa/original/failures").glob("*.json"))) == 1


def test_interrupted_attempt_cap_still_gets_terminal_failure(tmp_path):
    qs = questions("cap")
    for arm in ("original", "fusion"):
        for i in range(1, 4):
            (tmp_path / "hotpotqa" / arm / "attempts" / r.digest("cap") / f"attempt-{i:03d}").mkdir(parents=True)
    counts = r.generate_dataset(tmp_path, "hotpotqa", qs, ["original", "fusion"], CONFIG, worker_factory=fake_factory)
    assert counts == {"attempt_limit_reached": 2}
    assert r.all_terminal(tmp_path, {"hotpotqa": qs}, ["original", "fusion"])


def test_labels_guard_requires_every_dataset_and_arm_terminal(tmp_path):
    scopes = {"hotpotqa": questions("one"), "musique": questions("two")}
    called = []
    r.save(r.result_path(tmp_path, "hotpotqa", "original", "one"), valid_row("one"))
    with pytest.raises(ValueError, match="labels remain unread"):
        r.score_all(tmp_path, scopes, ["original", "fusion"], label_loader=lambda d: called.append(d))
    assert called == []


def test_scoring_failures_common_success_rescue_harm_and_musique_groups(tmp_path):
    qs = questions("a", "b", "c")
    for arm in ("original", "fusion"):
        for q in qs:
            prediction = "Y" if (arm, q["id"]) in (("original", "a"), ("fusion", "b"), ("fusion", "c")) else "wrong"
            row = valid_row(q["id"], prediction)
            if arm == "original" and q["id"] == "c":
                row = r.failure_row(q, TimeoutError("fixture"))
            r.save(r.result_path(tmp_path, "musique", arm, q["id"]), row)
    labels = [{"id": q["id"], "answers": ["Y"], "gold_groups": [["musique:doc", "musique:alternate"], ["musique:missing"]]} for q in qs]
    summaries = r.score_all(tmp_path, {"musique": qs}, ["original", "fusion"], label_loader=lambda _: labels)
    s = summaries["musique"]
    assert s["common_success_n"] == 2
    assert s["arms"]["original"]["failure_rate"] == pytest.approx(1 / 3)
    assert s["paired"]["fusion"]["rescue_em_common_success"] == 1
    assert s["paired"]["fusion"]["harm_em_common_success"] == 1
    assert s["paired"]["fusion"]["all_task_delta_percentage_points"]["em"] == pytest.approx(100 / 3)
    assert s["arms"]["fusion"]["all_task_metrics_percent"]["r@20"] == 50
    assert s["arms"]["fusion"]["all_task_metrics_percent"]["all@20"] == 0
    assert len((tmp_path / "musique/comparisons.jsonl").read_text().splitlines()) == 3


def test_factorial_interaction_uses_four_arm_success_not_original_success(tmp_path):
    arms = ["original", "fusion", "bt_flat", "dense_dependency", "dense_flat"]
    qs = questions("a", "b")
    # a: interaction +1, all four fusion-family arms valid, original fails.
    # b: interaction -2 under all-task zero-for-failed policy, fusion fails.
    good = {("fusion", "a"), ("bt_flat", "b"), ("dense_dependency", "b")}
    for arm in arms:
        for q in qs:
            row = valid_row(q["id"], "Y" if (arm, q["id"]) in good else "wrong")
            if (arm, q["id"]) in {("original", "a"), ("fusion", "b")}:
                row = r.failure_row(q, TimeoutError("fixture"))
            r.save(r.result_path(tmp_path, "hotpotqa", arm, q["id"]), row)
    labels = [{"id": q["id"], "answers": ["Y"], "gold_groups": [["doc"]]} for q in qs]
    result = r.score_all(tmp_path, {"hotpotqa": qs}, arms, label_loader=lambda _: labels)["hotpotqa"]
    interaction = result["factorial_interaction"]
    assert result["common_success_n"] == 0
    assert interaction["shared_four_success_n"] == 1
    assert interaction["shared_four_success_unit_ids"] == ["a"]
    for metric in ("f1", "em"):
        assert interaction["all_task_interaction_percentage_points"][metric] == -50
        assert interaction["shared_four_success_interaction_percentage_points"][metric] == 100
        assert interaction["all_task_interaction_ci95"]["intervals_percentage_points"][metric] == {"lower": -200, "upper": 100}
    assert interaction["shared_four_success_interaction_ci95"]["estimable"] is False
    assert interaction["shared_four_success_interaction_ci95"]["reason"] == "fewer_than_two_paired_questions"
    assert r.factorial_interaction({"fusion": {}, "bt_flat": {}}, []) is None


def test_interaction_bootstrap_preserves_same_question_pairing_and_is_deterministic():
    arms = ["fusion", "bt_flat", "dense_dependency", "dense_flat"]
    # Each arm varies across questions, but paired contrasts are all exactly zero.
    # Independently resampling arms would create spurious variation in the CI.
    scores = {arm: {str(i): {"valid": True, **{m: float(i % 2) for m in r.METRICS}} for i in range(8)} for arm in arms}
    units = [str(i) for i in range(8)]
    first = r.factorial_interaction(scores, units)
    second = r.factorial_interaction(scores, units)
    assert first == second
    for population in ("all_task_interaction_ci95", "shared_four_success_interaction_ci95"):
        report = first[population]
        assert report["estimable"] and report["replicates"] == 1000
        assert report["seed"] == 20260918 and report["confidence_level"] == .95
        assert all(interval == {"lower": 0, "upper": 0} for interval in report["intervals_percentage_points"].values())


def test_bundled_tokenizer_preflight_really_renders_chat_template():
    report = r.probe_local_tokenizer()
    assert report["chat_template_render"] == "ok"
    assert report["probe_prompt_tokens"] > 0
    assert "tokenizer.json" in report["files_sha256"]


@pytest.mark.parametrize("missing", ["yaml", "jinja2"])
def test_preflight_reports_missing_optional_runtime_dependency(monkeypatch, missing):
    original = r.importlib.util.find_spec
    monkeypatch.setattr(r, "verify_originals", lambda: {})
    monkeypatch.setattr(r.importlib.util, "find_spec", lambda name: None if name == missing else original(name))
    with pytest.raises(RuntimeError, match=missing):
        r.preflight({}, [], endpoints=False)


def test_cost_counts_retries_and_excludes_seeded_api_cost(tmp_path):
    seed = {"stage": "planner", "attempts": [{"http_status": 200}], "response": {"usage": {"prompt_tokens": 9, "completion_tokens": 1, "total_tokens": 10}}}
    r.save(tmp_path / "requests/seed.json", seed)
    r.save(tmp_path / "seeded_requests.json", ["seed"])
    r.save(tmp_path / "requests/new.json", {"stage": "rerank", "attempts": [{"http_status": 503}, {"http_status": 200}], "response": {"results": []}})
    r.append_jsonl(tmp_path / "call_events.jsonl", {"event": "call_started", "cache_hit": True})
    r.append_jsonl(tmp_path / "call_events.jsonl", {"event": "call_started", "cache_hit": False})
    with (tmp_path / "call_events.jsonl").open("a") as handle:
        handle.write('{"truncated":')
    cost = r.request_cost(tmp_path)
    assert cost["logical_calls"] == 2 and cost["cache_hits"] == 1
    assert cost["http_attempts"] == 2 and cost["http_retries"] == 1
    assert cost["missing_usage_responses"] == 1
    assert cost["token_usage_complete"] is False
    assert cost["incomplete_event_lines"] == 1
    assert cost["total_tokens"] == 0  # Unknown, explicitly accompanied by completeness=False.


def test_seed_cache_copies_success_only(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    r.save(old / "requests/good.json", {"response": {"x": 1}})
    r.save(old / "requests/bad.json", {"attempts": [{"http_status": 500}]})
    (old / "requests/interrupted.json").write_text('{"response":')
    r.seed_cache([old], new)
    assert r.load(new / "seeded_requests.json") == ["good"]
    assert not (new / "requests/bad.json").exists()
    assert not (new / "requests/interrupted.json").exists()
    assert r.load(new / "cache_seed_skipped.json")["invalid_records_preserved"] == [str(old / "requests/interrupted.json")]
    assert r.request_cost(old)["malformed_request_records"] == 1


def test_os_lock_ignores_stale_pid_and_prevents_duplicate_writer(tmp_path):
    r.save(tmp_path / "pid.json", {"pid": 99999999})
    assert not r.lock_is_held(tmp_path)
    with r.writer_lock(tmp_path):
        assert r.lock_is_held(tmp_path)
        with pytest.raises(RuntimeError, match="live writer"):
            with r.writer_lock(tmp_path):
                pass
    assert r.stop(tmp_path)["stopped"] is False


def test_credentials_rejected_even_inside_reranker():
    with pytest.raises(ValueError, match="forbidden"):
        r.validate_config({"_test_transport": True})
    with pytest.raises(ValueError, match="credentials"):
        r.validate_config({"reranker": {"api_key": "private"}})
    with pytest.raises(ValueError, match="without credentials"):
        r.validate_config({"reranker": {"url": "http://user:secret@example.test/rerank"}})


def test_module_metrics_separate_gold_discovery_and_model_claims(tmp_path):
    row = valid_row("module")
    row["diagnostics"] = {"candidate_doc_ids": ["doc", "second"],
        "support_graph": {"nodes": [{"status": "unknown"}, {"status": "ambiguous"}]},
        "events": [{"event": "protocol_error", "operation": "map", "error": "quote offset mismatch"}],
        "ledger": {"used": {"ann": 7, "set_score": 22}},
        "reader_feasibility": {"prompt_token_count": 100, "token_count": 1132}}
    row["budgets"]["20"].update(complete_required=True, necessary_covered=1)
    label = {"gold_groups": [["musique:doc"], ["musique:second"]]}
    metrics = r.module_metrics(row, label, "musique", tmp_path)
    assert metrics["gold_candidate_title_group_recall"] == 1
    assert metrics["gold_discovery_minus_selection_recall_at20"] == .5
    assert metrics["structural_complete_required_at20"] is True
    assert metrics["model_unknown_nodes"] == metrics["model_ambiguous_nodes"] == 1
    assert metrics["quote_or_span_protocol_error_events"] == 1
    assert metrics["retrieval_query_count"] == 7 and metrics["set_score_count"] == 22
    assert metrics["reader_prompt_tokens_local"] == 100


def test_module_metrics_distinguish_metered_zero_from_missing_ledger(tmp_path):
    row = valid_row("zero_cost")
    label = {"gold_groups": [["doc"]]}
    row["diagnostics"] = {"ledger": {"limits": {"ann": 36, "set_score": 512}, "used": {}}}
    observed = r.module_metrics(row, label, "hotpotqa", tmp_path)
    assert observed["retrieval_query_count"] == observed["set_score_count"] == 0
    row["diagnostics"] = {}
    missing = r.module_metrics(row, label, "hotpotqa", tmp_path)
    assert missing["retrieval_query_count"] is None and missing["set_score_count"] is None


def test_runner_archives_full_pointwise_preflight_costs(tmp_path):
    from test_bridge_http import server
    with server() as (endpoint, requests):
        config = {"reranker": {"url": endpoint + "/rerank", "model": "fixture-pointwise"},
                  "max_identical_attempts": 1, "request_timeout_seconds": 2}
        report = r.probe_reranker(config, tmp_path)
        assert report["protocol"]["consistent"]
        assert len(requests) == report["cost"]["http_attempts"] == report["cost"]["logical_calls"] == 5
        assert report["cost"]["missing_usage_responses"] == 5
        assert report["cost"]["token_usage_complete"] is False
        assert Path(report["trace_directory"], "report.json").is_file()


def test_original_wrapper_matches_native_work_path(monkeypatch, tmp_path):
    import sys
    sys.path.insert(0, str(r.ROOT / "dagv2"))
    pipeline = __import__("pipeline_dagv2")
    e = pipeline.e
    seen = []
    calls = object()
    monkeypatch.setattr(e, "Calls", lambda *args: calls)
    def plan(question, transport):
        assert transport is calls
        seen.append(("plan", question))
        return {"steps": []}
    def archive(question, plan, ids, vectors, transport):
        assert transport is calls
        seen.append(("archive", question))
        return ["doc"], [{"fixture": True}]
    def solve(row, docs, tokenizer, index, transport):
        assert transport is calls
        seen.append(("solve", row))
        return valid_row(row["unit_id"])
    monkeypatch.setattr(e, "make_plan", plan)
    monkeypatch.setattr(e, "archive", archive)
    monkeypatch.setattr(e.native, "solve", solve)
    # The baseline wrapper needs an output attribute only to preserve the input trace.
    calls = SimpleNamespace(output=tmp_path / "wrapper")
    q = {"id": "parity", "question": "test"}
    resources = ({}, ["doc"], None, object(), object())
    native = e.work(q, tmp_path / "native", *resources)
    first = list(seen)
    seen.clear()
    wrapped = r.original_question(q, resources, calls, e)
    native.pop("seconds")
    diagnostic = wrapped.pop("diagnostics")
    assert diagnostic["candidate_doc_ids"] == ["doc"]
    assert diagnostic["worker_pid"] == native.pop("diagnostics")["worker_pid"]
    assert native == wrapped
    assert first == seen


def test_manifest_rejects_changed_scope_before_worker_start(monkeypatch, tmp_path):
    config = tmp_path / "config.json"
    r.save(config, {"fusion": {}})
    output = tmp_path / "output"
    r.save(output / "manifest.json", {"schema": "incompatible"})
    monkeypatch.setattr(r, "frozen_sources", lambda: {})
    monkeypatch.setattr(r, "load_questions", lambda *args: questions("one"))
    args = SimpleNamespace(config=str(config), output=str(output), datasets=["hotpotqa"], arms=["original", "dense_dependency"],
                           limit=1, retry_failed=False, offline_preflight=True)
    with pytest.raises(ValueError, match="source, data, arms or scope changed"):
        r.run_experiment(args)


def cli_fixture(tmp_path, *, delay=0, fail=False):
    config = r.load(r.ROOT / "configs/paired.example.json")
    config["experiment"].update(fixture_delay=delay, fixture_fail_original=fail)
    config_path = tmp_path / "config.json"
    r.save(config_path, config)
    output = tmp_path / "run"
    script = Path(__file__).parent / "fixtures/dagbt.runner.fixture.py"
    completed = subprocess.run([sys.executable, str(script), "launch", "--config", str(config_path), "--output", str(output),
                                "--datasets", "hotpotqa", "--limit", "1", "--offline-preflight"], cwd=r.ROOT,
                               capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    return output, json.loads(completed.stdout)


def await_progress(output, states, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        path = output / "progress.json"
        if path.exists() and r.load(path)["state"] in states:
            return r.load(path)
        time.sleep(.05)
    raise AssertionError((output / "launcher.log").read_text())


def test_detached_cli_finishes_same_question_pairs_with_failure(tmp_path):
    output, launched = cli_fixture(tmp_path, fail=True)
    try:
        progress = await_progress(output, {"complete_with_failures"})
        assert progress["failed_arm_tasks"] == 1
        rows = [json.loads(s) for s in (output / "hotpotqa/comparisons.jsonl").read_text().splitlines()]
        assert len(rows) == 1 and set(rows[0]["arms"]) == {"original", "fusion"}
        assert rows[0]["arms"]["original"]["answer_status"] == "execution_failed"
        assert rows[0]["arms"]["fusion"]["answer_status"] == "ok"
        assert (output / "manifest.json").is_file()
    finally:
        if r.lock_is_held(output):
            r.stop(output)


def test_detached_cli_stop_reaps_active_workers(tmp_path):
    output, launched = cli_fixture(tmp_path, delay=30)
    try:
        await_progress(output, {"generating"})
        assert r.status(output)["live_writer"]
        assert r.stop(output)["stopped"]
        await_progress(output, {"stopped"})
        deadline = time.monotonic() + 10
        while r.lock_is_held(output) and time.monotonic() < deadline:
            time.sleep(.05)
        assert not r.lock_is_held(output)
    finally:
        if r.lock_is_held(output):
            r.stop(output)


@pytest.mark.parametrize("arms,limit", [(["original", "not_a_method"], 1), (["original", "fusion"], 0)])
def test_launch_rejects_invalid_options_before_service_probe(monkeypatch, tmp_path, arms, limit):
    config = tmp_path / "config.json"
    r.save(config, r.load(r.ROOT / "configs/paired.example.json"))
    def unexpected_preflight(*args, **kwargs):
        raise AssertionError("Service preflight must not run for invalid experiment options")
    monkeypatch.setattr(r, "preflight", unexpected_preflight)
    args = SimpleNamespace(config=str(config), output=str(tmp_path / "run"), arms=arms, limit=limit,
                           datasets=["hotpotqa"], offline_preflight=False, retry_failed=False)
    with pytest.raises(ValueError):
        r.launch(args)
    assert not (tmp_path / "run" / "launch.json").exists()


def test_standalone_preflight_rejects_invalid_fusion_budget_before_http(monkeypatch):
    config = r.load(r.ROOT / "configs/paired.example.json")
    config["fusion"]["ann_calls"] = 0
    def unexpected_probe(*args, **kwargs):
        raise AssertionError("No service request is allowed for invalid budgets")
    monkeypatch.setattr(r, "probe_endpoint", unexpected_probe)
    with pytest.raises(ValueError, match="ann_calls"):
        r.preflight(config, ["hotpotqa"], endpoints=True)


def test_proxy_free_control_can_launch_without_reranker_but_full_bridge_cannot():
    config = r.load(r.ROOT / "configs/paired.example.json")
    config.pop("reranker")
    args = SimpleNamespace(arms=["original", "fusion_proxy_free"], limit=1)
    r.validate_experiment_options(config, args)
    args.arms = ["original", "fusion"]
    with pytest.raises(ValueError, match="pointwise reranker"):
        r.validate_experiment_options(config, args)


def test_stop_foreground_relative_output_path(tmp_path):
    config = r.load(r.ROOT / "configs/paired.example.json")
    config["experiment"]["fixture_delay"] = 30
    config_path = tmp_path / "config.json"
    r.save(config_path, config)
    output = tmp_path / "relative_run"
    script = Path(__file__).parent / "fixtures/dagbt.runner.fixture.py"
    with (tmp_path / "foreground.log").open("w") as log:
        process = subprocess.Popen([sys.executable, str(script), "run", "--config", str(config_path),
                                    "--output", os.path.relpath(output, r.ROOT), "--datasets", "hotpotqa",
                                    "--limit", "1", "--offline-preflight"], cwd=r.ROOT, stdout=log, stderr=log)
    try:
        await_progress(output, {"generating"})
        assert r.stop(output)["stopped"]
        assert process.wait(timeout=10) == 130
        assert r.load(output / "progress.json")["state"] == "stopped"
        assert not r.lock_is_held(output)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
