"""Current-result cohorts and the fusion startup protocol gate; no remote calls."""
import io
import json
import time
import urllib.error
from types import SimpleNamespace

import pytest

from dagbt import runner as r
from test_engine import setup, FakeCalls, step, answer


def row(unit, cohort=None, *, ok=True, prediction="Y", cost=None):
    result = {"unit_id": unit, "answer": {"status": "ok" if ok else "execution_failed", "prediction": prediction},
              "budgets": {str(k): {"selected_doc_ids": []} for k in (5, 10, 20)}, "diagnostics": {}}
    if cohort is not None:
        result["diagnostics"]["reliability"] = {"version": "dagbt_fusion_reliability_v2", "cohort": cohort,
            "mapping_complete": "partially_mapped" not in cohort, "input_truncated": "truncated" in cohort}
    if cost is not None:
        result["runner"] = {"cost": cost}
    return result


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("existing_diagnostics", [False, True])
def test_module_metrics_recovers_failure_reliability_from_latest_partial_only(tmp_path, nested, existing_diagnostics):
    result = row("failed", ok=False)
    result["diagnostics"] = {"token_accounting": {"token_count_is_estimate": True}} if existing_diagnostics else {}
    result["attempt_directories"] = ["old", "latest"]
    r.save(tmp_path / "old/fusion_partial.json", {"reliability": {"cohort": "normal"}})
    reliability = {"version": "dagbt_fusion_reliability_v2", "cohort": "truncated_and_partially_mapped",
                   "mapping_complete": False, "input_truncated": True}
    snapshot = {"diagnostics": {"reliability": reliability}} if nested else {"reliability": reliability}
    r.save(tmp_path / "latest/fusion_partial.json", snapshot)
    actual = r.module_metrics(result, {"gold_groups": [["a"]]}, "hotpotqa", tmp_path)
    assert actual["reliability"] == reliability
    if existing_diagnostics:
        assert actual["local_token_accounting"]["token_count_is_estimate"] is True
    assert "reliability" not in result["diagnostics"]


def test_real_engine_failure_snapshot_reliability_reaches_runner_metrics(setup, tmp_path):
    from dagbt.engine import run_question
    bridge, resources, config = setup
    bridge.routes = {"answer": ["a"]}
    calls = FakeCalls([step("answer")], lambda data: answer(data, "fixture", ["a"]))
    calls.output = tmp_path / "latest"
    get = calls.get
    def fail_reader(stage, url, payload):
        if stage[0] == "reader":
            raise RuntimeError("synthetic reader failure after evidence work")
        return get(stage, url, payload)
    calls.get = fail_reader
    question = {"id": "failed", "question": "Q?"}
    with pytest.raises(RuntimeError, match="synthetic reader failure"):
        run_question(question, resources, calls, config)
    snapshot = r.load(calls.output / "fusion_partial.json")
    assert "reliability" not in snapshot
    expected = snapshot["diagnostics"]["reliability"]
    assert expected["version"] == "dagbt_fusion_reliability_v2"
    result = r.failure_row(question, RuntimeError("synthetic reader failure"))
    result["attempt_directories"] = ["latest"]
    actual = r.module_metrics(result, {"gold_groups": [["a"]]}, "hotpotqa", tmp_path)
    assert actual["reliability"] == expected
    assert actual["candidate_count"] == 1


@pytest.mark.parametrize("cohort", [None, "made_up"])
def test_old_or_unrecognized_reliability_is_unknown(tmp_path, cohort):
    actual = r.module_metrics(row("q", cohort), {"gold_groups": [["a"]]}, "hotpotqa", tmp_path)
    assert actual["reliability"]["cohort"] == "unknown"


@pytest.mark.parametrize("diagnostics", [{}, {"token_accounting": {"token_count_is_estimate": True}}])
def test_successful_result_does_not_adopt_partial_snapshot(tmp_path, diagnostics):
    result = row("q")
    result["diagnostics"] = diagnostics
    result["attempt_directories"] = ["old"]
    r.save(tmp_path / "old/fusion_partial.json", {"reliability": {"cohort": "truncated"}})
    actual = r.module_metrics(result, {"gold_groups": [["a"]]}, "hotpotqa", tmp_path)
    assert actual["reliability"]["cohort"] == "unknown"


@pytest.mark.parametrize("dataset,answer_metric", [("hotpotqa", "em"), ("personamem", "accuracy")])
def test_scoring_cohorts_use_current_rows_and_keep_retry_cost_separate(tmp_path, dataset, answer_metric):
    questions = [{"id": str(i), "question": "Choose", "options": ["Yes", "No"], "persona_id": "p"}
                 for i in range(7)]
    cohorts = ["normal", "truncated", "partially_mapped", "truncated_and_partially_mapped", None, "normal", "normal"]
    labels = [{"id": q["id"], "answers": ["Y"], "gold_groups": [["a"]], "correct_answer": "(a)"} for q in questions]
    for arm in ("original", "fusion"):
        for i, q in enumerate(questions):
            prediction = "(a)" if dataset == "personamem" else "Y"
            if i == 1:
                prediction = "(b)" if dataset == "personamem" else "wrong"
            result = row(q["id"], cohorts[i] if arm == "fusion" else None,
                         ok=i != 5, prediction=prediction,
                         cost=None if i == 6 else {"http_attempts": i + 1, "total_tokens": 10 * (i + 1),
                                                   "token_usage_complete": True})
            r.save(r.result_path(tmp_path, dataset, arm, q["id"]), result)
        old = tmp_path / dataset / arm / "attempts" / r.digest("0") / "attempt-001"
        r.save(old / "requests/old.json", {"stage": "planner", "attempts": [{"http_status": 200}],
                                           "response": {"usage": {"total_tokens": 900}}})
        # Historical attempt rows and failure journals cannot count as current tasks.
        r.save(old / "result.json", row("0", "truncated", ok=False))
        r.save(tmp_path / dataset / arm / "failures/old.json", row("0", "truncated", ok=False))
    summary = r.score_all(tmp_path, {dataset: questions}, ["original", "fusion"], label_loader=lambda _: labels)[dataset]
    reliability = summary["arms"]["fusion"]["reliability"]
    groups = reliability["completion_cohorts"]
    assert {name: value["tasks"] for name, value in groups.items()} == {
        "normal": 2, "truncated": 1, "partially_mapped": 1, "truncated_and_partially_mapped": 1, "unknown": 1}
    assert groups["normal"]["correct"] == 2 and groups["normal"][answer_metric] == 1
    assert groups["truncated"]["correct"] == 0 and groups["truncated"][answer_metric] == 0
    assert groups["normal"]["cost_latest_attempt"]["missing_tasks"] == 1
    assert groups["normal"]["cost_latest_attempt"]["metrics"]["total_tokens"]["sum"] == 10
    assert "token_usage_complete" not in groups["normal"]["cost_latest_attempt"]["metrics"]
    assert reliability["failed_tasks"] == 1
    assert reliability["failure_cost_latest_attempt"]["metrics"]["total_tokens"]["sum"] == 60
    assert summary["arms"]["fusion"]["cost_all_attempts"]["total_tokens"] == 900
    assert summary["arms"]["original"]["reliability"]["completion_cohorts"]["unknown"]["tasks"] == 6
    # Rebuilding the summary never adds the current rows or retry history twice.
    repeated = r.score_all(tmp_path, {dataset: questions}, ["original", "fusion"], label_loader=lambda _: labels)[dataset]
    assert repeated["arms"]["fusion"]["reliability"] == reliability


def test_empty_cohort_preserves_unobserved_accuracy_and_cost():
    result = r.summarize_reliability([], "em")
    assert result["completion_cohorts"]["normal"] == {
        "tasks": 0, "correct": 0, "em": None,
        "cost_latest_attempt": {"observed_tasks": 0, "missing_tasks": 0, "metrics": {}}}


def planner_config(protocol="plain"):
    config = r.load(r.ROOT / "configs/paired.example.json")
    config["llm_base_url"] = "http://protocol.invalid/v1"
    config["fusion"]["response_format"] = protocol
    return config


def model_reply(content=None, *, finish_reason="stop", refusal=None):
    if content is None:
        content = json.dumps({"steps": [{"question": "Which city hosts the museum?", "output_slot": "city",
                                          "answer_type": "city", "inputs": []}]})
    return {"id": "synthetic-probe-response", "choices": [{"message": {"content": content, "refusal": refusal},
            "finish_reason": finish_reason}], "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}


def fake_http(monkeypatch, response):
    from dagbt import transport
    requests = []
    def urlopen(request, timeout):
        requests.append(json.loads(request.data))
        return io.BytesIO(json.dumps(response).encode())
    monkeypatch.setattr(transport.urllib.request, "urlopen", urlopen)
    return requests


@pytest.mark.parametrize("protocol", ["plain", "json_object", "json_schema"])
def test_protocol_probe_uses_actual_reasoner_schema_and_raw_metadata(tmp_path, monkeypatch, protocol):
    response = model_reply()
    requests = fake_http(monkeypatch, response)
    report = r.probe_fusion_planner(planner_config(protocol), tmp_path)
    assert report["status"] == "passed" and report["logical_calls"] == 1
    assert report["repair_calls"] == 0 and report["protocol_fallback"] is False
    assert report["experiment_task"] is False and report["configured_protocol"] == protocol
    assert report["validated_plan"]["steps"][0]["output_slot"] == "city"
    assert len(requests) == 1
    wire = requests[0]
    assert wire["model"] == "deepseek-v4-flash"
    assert wire["messages"][0]["content"].startswith("Decompose a multi-hop question")
    assert json.loads(wire["messages"][1]["content"]) == report["synthetic_query"]
    assert not {"structured_outputs", "chat_template_kwargs", "_dagbt_reasoning"}.intersection(wire)
    if protocol == "plain":
        assert "response_format" not in wire
    else:
        assert wire["response_format"]["type"] == protocol
    if protocol == "json_schema":
        assert wire["response_format"]["json_schema"]["schema"]["required"] == ["steps"]
    assert report["cost"]["logical_calls"] == report["cost"]["http_attempts"] == 1
    assert report["cost"]["total_tokens"] == 30
    record = report["reasoning_requests"][0]
    assert record["response_metadata"] == {"finish_reason": "stop", "refusal": None,
        "usage": response["usage"], "response_id": response["id"], "protocol": protocol}
    assert record["validation_status"] == "valid"
    trace = tmp_path / "preflight_calls"
    reports = list(trace.glob("planner-*/report.json"))
    assert len(reports) == 1 and r.load(reports[0])["status"] == "passed"
    persisted = list(trace.glob("planner-*/requests/*.json"))
    assert len(persisted) == 1 and r.load(persisted[0])["response"] == response
    assert not list(tmp_path.glob("*/attempts/*"))


@pytest.mark.parametrize("response,category", [
    (model_reply("{broken"), "json"),
    (model_reply('{"steps": []}'), "schema"),
    (model_reply(finish_reason="length"), "output_truncated"),
    (model_reply(refusal="synthetic refusal"), "refusal"),
])
def test_probe_has_no_model_repair_or_protocol_fallback(tmp_path, monkeypatch, response, category):
    from dagbt.reasoning import ProtocolError
    requests = fake_http(monkeypatch, response)
    with pytest.raises(ProtocolError):
        r.probe_fusion_planner(planner_config("json_schema"), tmp_path)
    assert len(requests) == 1
    assert requests[0]["response_format"]["type"] == "json_schema"
    report = r.load(next((tmp_path / "preflight_calls").glob("planner-*/report.json")))
    assert report["status"] == "failed" and report["logical_calls"] == 1
    assert report["reasoning_requests"][0]["validation_status"] == "invalid"
    assert report["repair_calls"] == 0 and report["protocol_fallback"] is False
    assert report["cost"]["http_attempts"] == 1


def stub_nonprotocol_preflight(monkeypatch):
    monkeypatch.setattr(r, "verify_originals", lambda **kwargs: {})
    monkeypatch.setattr(r, "frozen_sources", lambda: {})
    monkeypatch.setattr(r, "probe_endpoint", lambda *a, **kw: {"models_endpoint": "ok"})
    monkeypatch.setattr(r, "probe_reranker", lambda *a, **kw: {"status": "passed"})


def test_online_preflight_requires_planner_but_offline_and_original_skip(monkeypatch, tmp_path):
    stub_nonprotocol_preflight(monkeypatch)
    called = []
    def probe(config, output):
        called.append(output)
        return {"status": "passed"}
    monkeypatch.setattr(r, "probe_fusion_planner", probe)
    config = planner_config()
    online = r.preflight(config, [], output=tmp_path, arms=["original", "fusion"])
    assert online["endpoints"]["fusion_planner_protocol"]["status"] == "passed"
    assert online["arms"] == ["original", "fusion"] and called == [tmp_path]
    offline = r.preflight(config, [], endpoints=False, output=tmp_path)
    assert offline["endpoints"] == {"status": "not_checked_offline_preflight"}
    original = r.preflight(config, [], output=tmp_path, arms=["original"])
    assert "fusion_planner_protocol" not in original["endpoints"] and len(called) == 1


def test_failed_actual_protocol_probe_prevents_background_launch(monkeypatch, tmp_path):
    from dagbt.reasoning import ProtocolError
    stub_nonprotocol_preflight(monkeypatch)
    requests = fake_http(monkeypatch, model_reply('{"steps": []}'))
    config_path = tmp_path / "config.json"
    r.save(config_path, planner_config())
    launched = []
    monkeypatch.setattr(r.subprocess, "Popen", lambda *a, **kw: launched.append(a))
    args = SimpleNamespace(config=str(config_path), output=str(tmp_path / "run"), arms=["original", "fusion"],
        limit=1, datasets=[], offline_preflight=False, retry_failed=False)
    with pytest.raises(ProtocolError):
        r.launch(args)
    assert len(requests) == 1 and not launched
    assert not (tmp_path / "run/launch.json").exists()
    report = r.load(next((tmp_path / "run/preflight_calls").glob("planner-*/report.json")))
    assert report["status"] == "failed"


@pytest.mark.parametrize("http_status,expected_attempts", [(400, 1), (503, 3)])
def test_probe_http_retries_are_bounded_and_never_change_protocol(monkeypatch, tmp_path, http_status, expected_attempts):
    from dagbt import transport
    requests = []
    def reject(request, timeout):
        requests.append(json.loads(request.data))
        raise urllib.error.HTTPError(request.full_url, http_status, "synthetic rejection", {}, None)
    monkeypatch.setattr(transport.urllib.request, "urlopen", reject)
    monkeypatch.setattr(transport.time, "sleep", lambda _: None)
    config = planner_config("json_object")
    config["max_identical_attempts"] = 3
    with pytest.raises(transport.ServiceError):
        r.probe_fusion_planner(config, tmp_path)
    assert len(requests) == expected_attempts
    assert all(request == requests[0] for request in requests)
    assert requests[0]["response_format"]["type"] == "json_object"
    report = r.load(next((tmp_path / "preflight_calls").glob("planner-*/report.json")))
    assert report["logical_calls"] == 1 and report["cost"]["http_attempts"] == expected_attempts
    assert report["cost"]["token_usage_complete"] is False
    assert report["protocol_fallback"] is False and report["status"] == "failed"


@pytest.mark.parametrize("planner_status,expected_recheck", [(None, True), ("failed", True), ("passed", False)])
def test_launch_preflight_reuse_requires_successful_protocol_gate(monkeypatch, tmp_path, planner_status, expected_recheck):
    stub_nonprotocol_preflight(monkeypatch)
    config = planner_config()
    config_path = tmp_path / "config.json"
    r.save(config_path, config)
    output = tmp_path / "run"
    prior = {"config_digest": r.digest(config), "source_hashes": {}, "datasets": {},
             "arms": ["original", "fusion"], "created_unix": time.time(), "endpoints": {"llm": {}}}
    if planner_status is not None:
        prior["endpoints"]["fusion_planner_protocol"] = {"status": planner_status}
    r.save(output / "launch_preflight.json", prior)
    rechecked = []
    def preflight(*args, **kwargs):
        rechecked.append(True)
        return {"endpoints": {"fusion_planner_protocol": {"status": "passed"}}}
    monkeypatch.setattr(r, "preflight", preflight)
    args = SimpleNamespace(config=str(config_path), output=str(output), arms=["original", "fusion"],
        limit=1, datasets=[], offline_preflight=False, retry_failed=False)
    r.run_experiment(args)
    assert bool(rechecked) is expected_recheck
    assert r.load(output / "progress.json")["state"] == "complete"
