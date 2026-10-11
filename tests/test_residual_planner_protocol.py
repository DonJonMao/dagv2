"""Typed planner contracts and bounded startup repair; every HTTP response is synthetic."""
from copy import deepcopy
import io
import json
from types import SimpleNamespace
import urllib.error

import pytest

from dagbt import residual_plan, runner
from dagbt.engine import legacy_modules
from dagbt.methods import JOINT, LOCAL, RESIDUAL, UNSCORED
from dagbt.reasoning import OutputTruncated, ProtocolError, RefusalError
from dagbt.residual import UNITS
from test_runner_reliability_v2 import model_reply, planner_config, stub_nonprotocol_preflight


def typed_plan(declaration=None):
    declaration = deepcopy(declaration or {"type": "entity", "demand": "fact", "open": True})
    return {"steps": [{"question": "Which city hosts the museum?", "output_slot": "fact",
                       "answer_type": declaration["type"], "inputs": [], "execution": "retrieval"}],
            "final_node_id": "fact",
            "output_contract": {"mode": "value", "result_type": declaration["type"],
                                "entity_scope": "fictional museum", "time_scope": "unspecified",
                                "explanation": "none"},
            "program": {"variables": {"fact": declaration}, "expression": {"op": "var", "id": "fact"}}}


def reply(plan, identifier):
    response = model_reply(json.dumps(plan))
    response["id"] = identifier
    return response


def scripted_http(monkeypatch, responses, *, repeat_last=False):
    from dagbt import transport
    sent = []

    def urlopen(request, timeout):
        position = len(sent)
        sent.append(json.loads(request.data))
        if repeat_last:
            position = min(position, len(responses) - 1)
        assert position < len(responses), "Unexpected extra model request"
        response = responses[position]
        if isinstance(response, Exception):
            raise response
        return io.BytesIO(json.dumps(response).encode())

    monkeypatch.setattr(transport.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(transport.time, "sleep", lambda _: None)
    return sent


def persisted_report(output):
    return runner.load(next((output / "preflight_calls").glob("planner-*/report.json")))


def wire_schema(payload):
    response_format = payload.get("response_format", {})
    if response_format.get("type") == "json_schema":
        return response_format["json_schema"]["schema"]
    marker = "Return one complete JSON object matching this schema:\n"
    return json.loads(payload["messages"][0]["content"].rsplit(marker, 1)[1])


VALID_DECLARATIONS = [
    {"type": "entity", "demand": "fact", "open": True},
    {"type": "entity", "demand": "fact", "domain": ["River City", "Lake City"], "open": False},
    {"type": "bool", "demand": "fact", "domain": [False, True]},
    {"type": "number", "demand": "fact", "domain": [0, 1, 2], "unit": "s"},
    {"type": "number", "demand": "fact", "domain": {
        "lower": 0, "upper": 60, "lower_closed": True, "upper_closed": False, "unit": "s"}},
    {"type": "number", "demand": "fact", "domain": {"lower": None, "upper": None}},
    {"type": "number", "demand": "fact", "unit": None},
]


@pytest.mark.parametrize("method", [JOINT, RESIDUAL])
@pytest.mark.parametrize("declaration", VALID_DECLARATIONS)
def test_typed_probe_preserves_valid_omitted_finite_and_interval_domains(tmp_path, monkeypatch, method, declaration):
    plan = typed_plan(declaration)
    sent = scripted_http(monkeypatch, [reply(plan, "valid-typed-plan")])
    report = runner.probe_fusion_planner(planner_config(), tmp_path, method)
    assert report["status"] == "passed" and report["initial_response_valid"] is True
    assert report["validated_plan"] == plan
    assert report["logical_calls"] == len(sent) == 1
    assert report["repair_calls"] == report["repair_reservations"] == 0


@pytest.mark.parametrize("domain", [None, "unknown", False, 7])
def test_invalid_domain_type_keeps_strict_error_and_variable_path(domain):
    _, repair, _ = legacy_modules()
    plan = typed_plan({"type": "entity", "demand": "fact", "domain": domain})
    before = deepcopy(plan)
    with pytest.raises(ProtocolError) as raised:
        residual_plan.validate_plan(plan, repair.validate_plan)
    assert "Domain must be finite values or an interval" in str(raised.value)
    assert "program.variables['fact'].domain" in str(raised.value)
    assert plan == before


@pytest.mark.parametrize("method", [JOINT, RESIDUAL])
@pytest.mark.parametrize("protocol", ["plain", "json_object", "json_schema"])
def test_typed_schema_is_explicit_and_identical_in_each_wire_protocol(tmp_path, monkeypatch, method, protocol):
    e, _, _ = legacy_modules()
    original = deepcopy(e.PLAN_SCHEMA)
    expected = residual_plan.planner_contract(e.PLAN_SCHEMA)
    assert e.PLAN_SCHEMA == original
    sent = scripted_http(monkeypatch, [reply(typed_plan(), "typed-schema")])
    report = runner.probe_fusion_planner(planner_config(protocol), tmp_path, method)
    assert wire_schema(sent[0]) == expected
    assert report["schema_sha256"] == runner.digest(expected)
    program = expected["properties"]["program"]
    assert {"variables", "expression"} <= set(program["required"])
    declaration = program["properties"]["variables"]["additionalProperties"]
    assert {"type", "demand"} <= set(declaration["required"])
    assert "domain" not in declaration["required"]
    fields = declaration["properties"]
    assert fields["open"]["type"] == "boolean"
    assert set(fields["unit"]["enum"]) == set(UNITS) | {None}
    branches = fields["domain"].get("anyOf", fields["domain"].get("oneOf", []))
    assert {branch["type"] for branch in branches} == {"array", "object"}
    interval = next(branch for branch in branches if branch["type"] == "object")
    assert {"lower", "upper"} <= set(interval["required"])
    assert interval["additionalProperties"] is False
    for endpoint in ("lower", "upper"):
        assert set(interval["properties"][endpoint]["type"]) == {"number", "null"}
    for closure in ("lower_closed", "upper_closed"):
        assert interval["properties"][closure]["type"] == "boolean"
    system = sent[0]["messages"][0]["content"].lower()
    assert "domain" in system and "omit" in system and "null" in system
    assert report["planner_policy"] == "runtime_bounded_repair"


@pytest.mark.parametrize("method", [JOINT, RESIDUAL])
@pytest.mark.parametrize("protocol", ["plain", "json_object", "json_schema"])
def test_one_valid_repair_passes_probe_and_keeps_initial_raw_failure(tmp_path, monkeypatch, method, protocol):
    invalid = typed_plan({"type": "entity", "demand": "fact", "domain": None})
    responses = [reply(invalid, "invalid-initial"), reply(typed_plan(), "valid-repair")]
    sent = scripted_http(monkeypatch, responses)
    config = planner_config(protocol)
    report = runner.probe_fusion_planner(config, tmp_path, method)
    assert report["status"] == "passed" and report["initial_response_valid"] is False
    assert report["validated_plan"] == typed_plan()
    assert report["logical_calls"] == len(sent) == 2
    assert report["repair_calls"] == report["repair_reservations"] == 1
    limit = min(config["fusion"]["max_repairs_per_request"], config["fusion"]["json_repairs"])
    assert report["logical_call_limit"] == 1 + limit
    assert report["ledger"]["limits"] == {
        "llm": (1 + limit) * config["max_identical_attempts"],
        "http_attempts": (1 + limit) * config["max_identical_attempts"], "json_repairs": limit}
    assert report["protocol_fallback"] is False
    assert [record["validation_status"] for record in report["reasoning_requests"]] == ["invalid", "valid"]
    feedback = json.loads(sent[1]["messages"][1]["content"])["validation_feedback"]
    assert "program.variables['fact'].domain" in feedback
    assert wire_schema(sent[0]) == wire_schema(sent[1])
    assert sent[0].get("response_format") == sent[1].get("response_format")
    records = [runner.load(path) for path in (tmp_path / "preflight_calls").glob("planner-*/requests/*.json")]
    assert {record["response"]["id"] for record in records} == {"invalid-initial", "valid-repair"}
    assert next(record["response"] for record in records if record["response"]["id"] == "invalid-initial") == responses[0]
    assert persisted_report(tmp_path)["initial_response_valid"] is False


@pytest.mark.parametrize("method", [JOINT, RESIDUAL])
@pytest.mark.parametrize("local_repairs,global_repairs", [(2, 6), (2, 1), (0, 6), (2, 0)])
def test_exhausted_domain_repair_stops_launch_with_bounded_audit(tmp_path, monkeypatch, method, local_repairs, global_repairs):
    stub_nonprotocol_preflight(monkeypatch)
    invalid = typed_plan({"type": "entity", "demand": "fact", "domain": "unknown"})
    sent = scripted_http(monkeypatch, [reply(invalid, "invalid-domain")], repeat_last=True)
    config = planner_config("json_object")
    config["fusion"].update(max_repairs_per_request=local_repairs, json_repairs=global_repairs)
    config_path = tmp_path / "config.json"
    runner.save(config_path, config)
    launched = []
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: launched.append(args))
    output = tmp_path / "run"
    args = SimpleNamespace(config=str(config_path), output=str(output), arms=["original", method],
                           limit=1, datasets=[], offline_preflight=False, retry_failed=False)
    with pytest.raises(ProtocolError, match="Domain must be finite values or an interval"):
        runner.launch(args)
    assert not launched and not (output / "launch.json").exists()
    report = persisted_report(output)
    limit = min(local_repairs, global_repairs)
    assert report["status"] == "failed" and report["initial_response_valid"] is False
    assert report["planner_policy"] == "runtime_bounded_repair" and report["protocol_fallback"] is False
    assert report["logical_calls"] == report["logical_call_limit"] == 1 + limit
    assert report["repair_calls"] == report["repair_reservations"] == limit
    assert report["ledger"]["used"].get("json_repairs", 0) == limit
    # Identical repair feedback can replay a cached invalid HTTP-200 response.
    assert 1 <= len(sent) <= 1 + limit
    assert report["cost"]["http_attempts"] == len(sent)
    assert "program.variables['fact'].domain" in report["error"]
    assert all(record["validation_status"] == "invalid" for record in report["reasoning_requests"])


@pytest.mark.parametrize("method", [JOINT, RESIDUAL])
@pytest.mark.parametrize("mode,error", [("refusal", RefusalError), ("length", OutputTruncated)])
def test_typed_probe_does_not_repair_refusal_or_truncation(tmp_path, monkeypatch, method, mode, error):
    response = reply(typed_plan(), "provider-terminal-outcome")
    if mode == "refusal":
        response["choices"][0]["message"]["refusal"] = "synthetic refusal"
    else:
        response["choices"][0]["finish_reason"] = "length"
    sent = scripted_http(monkeypatch, [response])
    with pytest.raises(error):
        runner.probe_fusion_planner(planner_config(), tmp_path, method)
    report = persisted_report(tmp_path)
    assert report["status"] == "failed" and report["logical_calls"] == len(sent) == 1
    assert report["repair_calls"] == report["repair_reservations"] == 0
    assert report["initial_response_valid"] is False
    raw = runner.load(next((tmp_path / "preflight_calls").glob("planner-*/requests/*.json")))
    assert raw["response"] == response


@pytest.mark.parametrize("method", ["fusion", LOCAL, UNSCORED])
def test_legacy_and_local_probes_still_fail_without_planner_repairs(tmp_path, monkeypatch, method):
    sent = scripted_http(monkeypatch, [model_reply('{"steps": []}')])
    with pytest.raises(ProtocolError):
        runner.probe_fusion_planner(planner_config(), tmp_path, method)
    report = persisted_report(tmp_path)
    assert report["status"] == "failed" and report["logical_calls"] == len(sent) == 1
    assert report["logical_call_limit"] == 1 and report["repair_calls"] == 0


def test_typed_planner_http_retries_and_json_repair_share_finite_budget(tmp_path, monkeypatch):
    unavailable = urllib.error.HTTPError("http://protocol.invalid/v1/chat/completions", 503,
                                        "synthetic unavailable", {}, None)
    invalid = typed_plan({"type": "entity", "demand": "fact", "domain": False})
    sent = scripted_http(monkeypatch, [unavailable, reply(invalid, "invalid-first"),
                                      unavailable, reply(typed_plan(), "valid-last")])
    config = planner_config()
    config["max_identical_attempts"] = 2
    config["fusion"].update(max_repairs_per_request=1, json_repairs=1)
    report = runner.probe_fusion_planner(config, tmp_path, RESIDUAL)
    assert report["status"] == "passed" and report["logical_calls"] == 2
    assert report["repair_calls"] == report["repair_reservations"] == 1
    assert len(sent) == report["cost"]["http_attempts"] == 4
    assert report["ledger"]["used"]["llm"] == report["ledger"]["limits"]["llm"] == 4
    assert report["ledger"]["used"]["http_attempts"] == report["ledger"]["limits"]["http_attempts"] == 4
    assert sent[0] == sent[1] and sent[2] == sent[3]


def test_repaired_typed_preflight_allows_background_launch(tmp_path, monkeypatch):
    stub_nonprotocol_preflight(monkeypatch)
    invalid = typed_plan({"type": "entity", "demand": "fact", "domain": "unknown"})
    sent = scripted_http(monkeypatch, [reply(invalid, "invalid"), reply(typed_plan(), "repaired")])
    config_path = tmp_path / "config.json"
    runner.save(config_path, planner_config())
    launched = []
    def popen(command, **kwargs):
        launched.append(command)
        return SimpleNamespace(pid=123456)
    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    output = tmp_path / "run"
    args = SimpleNamespace(config=str(config_path), output=str(output), arms=["original", RESIDUAL],
                           limit=1, datasets=[], offline_preflight=False, retry_failed=False)
    result = runner.launch(args)
    assert result["launched_pid"] == 123456 and len(launched) == 1 and len(sent) == 2
    probe = runner.load(output / "launch_preflight.json")["endpoints"]["fusion_planner_protocol"]
    assert probe["status"] == "passed" and probe["initial_response_valid"] is False
    assert probe["repair_calls"] == 1 and probe["version"] == RESIDUAL
    assert (output / "launch.json").is_file()


def test_typed_http_failure_has_no_claim_about_initial_response_validity(tmp_path, monkeypatch):
    from dagbt.transport import ServiceError
    unavailable = urllib.error.HTTPError("http://protocol.invalid/v1/chat/completions", 503,
                                        "synthetic unavailable", {}, None)
    sent = scripted_http(monkeypatch, [unavailable], repeat_last=True)
    config = planner_config()
    config["max_identical_attempts"] = 2
    with pytest.raises(ServiceError):
        runner.probe_fusion_planner(config, tmp_path, RESIDUAL)
    report = persisted_report(tmp_path)
    assert report["status"] == "failed" and report["initial_response_valid"] is None
    assert report["logical_calls"] == 1 and len(sent) == report["cost"]["http_attempts"] == 2
    assert report["repair_calls"] == report["repair_reservations"] == 0
