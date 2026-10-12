"""Local fact scopes survive reading, derivation, rule revision and resume.

All sources and model responses are synthetic. The production reader and state
validators run unchanged; optional JSON Schema checks need only jsonschema.
"""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from dagbt.fact_state import FactState
from dagbt.joint_reading import JointReader
from dagbt.reasoning import ProtocolError
from dagbt.residual import ProgramError, validate_program
from dagbt.residual_control import ResidualControl
from dagbt.residual_plan import validate_bindings
from dagbt.transport import digest
from test_personamem_joint_protocol import fixture, observation, read, response, visible


PAST = {"entity_scope": "user", "time_scope": "past"}
FUTURE = {"entity_scope": "user", "time_scope": "future"}


def scoped_fixture(respond=None, *, explicit=True, **settings):
    reader, engine, original, calls = fixture(respond, **settings)
    program = deepcopy(original.program)
    contract = {**original.contract, **FUTURE}
    if explicit:
        program["variables"]["history"]["scope"] = deepcopy(PAST)
        program["variables"]["suggestion"]["scope"] = deepcopy(FUTURE)
    state = FactState(program, contract, engine.steps, "suggestion")
    reader.state = state
    return reader, engine, state, calls


def bind_history(reader, engine, state):
    row = reader._observation(observation(**PAST), "history", visible(reader), engine.steps[0])
    state.integrate([row])
    return next(iter(state.routes))


def controller_for(state):
    engine = SimpleNamespace(
        fact_state=state, steps=state.steps, final_node_id=state.final_node_id,
        s={"max_enumeration_states": 10000, "max_alternatives": 2},
        nodes=[{"id": step["output_slot"]} for step in state.steps],
    )
    engine.node_map = lambda: {node["id"]: node for node in engine.nodes}
    engine.compile = lambda: None
    return ResidualControl(engine)


def test_historical_support_keeps_sources_without_repair_under_future_contract():
    raw = response(observation(sources=["s1", "s2"], value=["piano", "painting"], **PAST))
    reader, engine, state, calls = scoped_fixture(lambda index, data: deepcopy(raw))
    reader.read(engine.steps[0], "Read the historical activities", ["history"],
                ["piano", "painting"], "historical-read")
    assert len(calls.requests) == 1
    assert engine.ledger.used.get("json_repairs", 0) == 0
    assert state.valid()[0] == {"history": ["piano", "painting"]}
    assert state.valid()[1]["history"] == sorted(engine.mapper.sources[a]["source_id"] for a in ("s1", "s2"))
    assert len(state.routes) == 1 and not state.pending and not reader.failed_obligations
    assert calls.requests[0]["data"]["requested_scopes"] == {"history": PAST}
    assert calls.requests[0]["data"]["required_scope"] == PAST
    assert state.contract["time_scope"] == "future"
    assert raw == response(observation(sources=["s1", "s2"], value=["piano", "painting"], **PAST))


def test_omitted_scope_inherits_contract_without_inference_from_observation():
    reader, engine, state, _ = scoped_fixture(explicit=False)
    assert state.scope("history") == FUTURE
    with pytest.raises(ProtocolError, match="declared variable history"):
        reader._observation(observation(**PAST), "history", visible(reader), engine.steps[0])
    assert not state.routes
    row = reader._observation(observation(**FUTURE), "history", visible(reader), engine.steps[0])
    state.integrate([row])
    copied_scope = state.scope("history")
    copied_scope["time_scope"] = "mutated"
    assert state.scope("history") == FUTURE and state.contract["time_scope"] == "future"


@pytest.mark.parametrize("audit", [False, True])
def test_terminal_and_parent_audit_keep_distinct_scopes(audit):
    reader, engine, state, calls = scoped_fixture()
    bind_history(reader, engine, state)
    payload = reader._payload(engine.steps[1], "Use the historical fact", ["history", "suggestion"], ["s1"], audit=audit)
    assert payload["requested_scopes"] == {"history": PAST, "suggestion": FUTURE}
    assert "required_scope" not in payload
    assert payload["supported_parents"]["history"]["scope"] == PAST
    spans = {span["id"]: span for span in payload["source_spans"]}
    parent = reader._observation(observation(**PAST), "history", spans, engine.steps[1])
    terminal = reader._observation(observation("proposal", value="Try piano", **FUTURE,
        parent_versions={"history": state.versions["history"]}), "suggestion", spans, engine.steps[1])
    state.integrate([parent, terminal])
    assert state.valid()[0]["suggestion"] == "Try piano"
    calls.respond = lambda index, data: {"slots": [
        {"variable": "history", "observations": [observation(**PAST)], "unresolved": []},
        {"variable": "suggestion", "observations": [observation("proposal", value="Try piano",
            **FUTURE, parent_versions={"history": state.versions["history"]})], "unresolved": []},
    ], **({"audit_verdict": "supported"} if audit else {})}
    reader.read(engine.steps[1], "Use the historical fact", ["history", "suggestion"],
                ["piano"], "mixed-scope-read", audit=audit)
    assert len(calls.requests) == 1 and not state.pending
    assert "required_scope" not in calls.requests[0]["data"]
    if audit:
        assert reader.batches[-1]["audit_complete"] is True
    with pytest.raises(ProtocolError, match="declared variable suggestion"):
        reader._observation(observation("wrong-terminal", value="Try piano", **PAST),
                            "suggestion", spans, engine.steps[1])


@pytest.mark.parametrize("stance", ["support", "contradiction"])
@pytest.mark.parametrize("changes", [{"entity_scope": "someone else"}, {"time_scope": "future"}])
def test_wrong_scope_cannot_bind_or_retract_and_batch_failure_is_atomic(stance, changes):
    reader, engine, state, _ = scoped_fixture()
    route_id = bind_history(reader, engine, state)
    wrong = observation("wrong", **{**PAST, **changes}, stance=stance,
                        retract_ids=[route_id] if stance == "contradiction" else [])
    if stance == "contradiction":
        wrong["value"] = None
    before = deepcopy(state.public_dict())
    with pytest.raises(ProtocolError, match="declared variable"):
        reader._observation(wrong, "history", visible(reader), engine.steps[0])
    assert state.public_dict() == before
    valid_row = {**observation("extra", sources=["another-source"], **PAST), "variable": "history"}
    with pytest.raises(ProgramError, match="declared variable scope"):
        state.integrate([valid_row, {**wrong, "variable": "history"}])
    assert state.public_dict() == before


@pytest.mark.parametrize("stance", ["support", "contradiction"])
def test_json_schema_applies_the_scope_of_each_requested_slot(stance):
    jsonschema = pytest.importorskip("jsonschema")
    reader, engine, state, _ = scoped_fixture()
    history_id = bind_history(reader, engine, state)
    terminal = reader._observation(observation("proposal", value="Try piano", **FUTURE,
        parent_versions={"history": state.versions["history"]}), "suggestion", visible(reader), engine.steps[1])
    state.integrate([terminal])
    suggestion_id = state.valid()[2]["suggestion"][0]
    data = reader._payload(engine.steps[1], "Use past facts", ["history", "suggestion"], ["s1"])
    schema = JointReader._schema(data)
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    slots = []
    for variable, scope, route, value in (
        ("history", PAST, history_id, ["piano"]),
        ("suggestion", FUTURE, suggestion_id, "Try piano"),
    ):
        row = observation(variable, value=value, stance=stance, **scope,
                          retract_ids=[route] if stance == "contradiction" else [])
        if stance == "contradiction":
            row["value"] = None
        slots.append({"variable": variable, "observations": [row], "unresolved": []})
    answer = {"slots": slots}
    assert validator.is_valid(answer)
    for index, wrong_scope in ((0, FUTURE), (1, PAST)):
        wrong = deepcopy(answer)
        wrong["slots"][index]["observations"][0].update(wrong_scope)
        assert not validator.is_valid(wrong)
    wrong = deepcopy(answer)
    wrong["slots"][0]["observations"][0]["entity_scope"] = "another user"
    assert not validator.is_valid(wrong)


@pytest.mark.parametrize("scope", [None, "past", {}, {"entity_scope": "user"},
    {"entity_scope": "", "time_scope": "past"}, {"entity_scope": "user", "time_scope": "  "},
    {"entity_scope": "user", "time_scope": 1}, {**PAST, "extra": "value"}])
def test_malformed_declaration_scope_is_rejected_before_reading(scope):
    _, engine, state, calls = scoped_fixture()
    program = deepcopy(state.program)
    program["variables"]["history"]["scope"] = scope
    with pytest.raises(ProgramError, match="Variable scope requires"):
        validate_program(program)
    with pytest.raises(ProgramError, match="Variable scope requires"):
        FactState(program, state.contract, engine.steps, "suggestion")
    assert not calls.requests


@pytest.mark.parametrize("scope", [PAST, {**FUTURE, "entity_scope": "other user"}])
def test_terminal_scope_cannot_override_the_output_contract(scope):
    _, engine, state, calls = scoped_fixture()
    program = deepcopy(state.program)
    program["variables"]["suggestion"]["scope"] = scope
    with pytest.raises(ProgramError, match="Terminal variable scope"):
        validate_bindings(program, engine.steps, "suggestion", contract=state.contract)
    assert not calls.requests


def test_deterministic_binding_and_node_metadata_use_intermediate_scope():
    reader, engine, original, _ = scoped_fixture()
    program = deepcopy(original.program)
    program["variables"]["had_activity"] = {"type": "bool", "demand": "summary", "scope": PAST}
    program["task_expressions"] = {"had_activity": {"op": "eq", "args": [
        {"op": "var", "id": "history"}, {"op": "const", "value": ["piano"]}]}}
    steps = [engine.steps[0], {"output_slot": "summary", "inputs": ["history"], "execution": "compose"},
             {**engine.steps[1], "inputs": ["summary"]}]
    state = FactState(program, original.contract, steps, "suggestion")
    reader.state = state
    bind_history(reader, engine, state)
    controller = controller_for(state)
    controller.deterministic_bindings()
    assert state.valid()[0]["had_activity"] is True
    route = state.routes[state.valid()[2]["had_activity"][0]]
    assert {key: route[key] for key in PAST} == PAST
    expected_contract = {"mode": "value", "result_type": "bool", **PAST, "explanation": "none"}
    assert route["binding_derivation"]["contract_identity"] == digest(expected_contract)
    assert route["source_ids"] == state.valid()[1]["history"]
    controller.sync_nodes()
    node = controller.e.node_map()["summary"]
    assert node["variable_scopes"] == {"had_activity": PAST}
    assert node["alternatives"][0]["applicable_scope"] == PAST


@pytest.mark.parametrize("mixed", [False, True])
def test_task_projection_only_claims_a_uniform_scope_and_mixed_bindings_survive(mixed):
    second_scope = FUTURE if mixed else PAST
    program = {"variables": {
        "x": {"type": "bool", "demand": "check", "scope": PAST},
        "y": {"type": "bool", "demand": "check", "scope": second_scope}},
        "expression": {"op": "record", "fields": {"check": {"op": "var", "id": "x"}}}}
    steps = [{"output_slot": "check", "inputs": [], "execution": "retrieval"},
             {"output_slot": "decision", "inputs": ["check"], "execution": "compose"}]
    contract = {"mode": "value", "result_type": "record", **FUTURE, "explanation": "none"}
    state = FactState(program, contract, steps, "decision")
    state.integrate([{**observation("x", ["x-source"], False, **PAST), "variable": "x"}])
    controller = controller_for(state)
    projection = controller.task_projection("check")
    if mixed:
        assert projection is None and not controller.local_complete("check")
    else:
        assert projection["scope"] == {"task_projection": "check", **PAST}
        assert projection["output"] is False and controller.local_complete("check")
    state.integrate([{**observation("y", ["y-source"], True, **second_scope), "variable": "y"}])
    assert controller.local_complete("check")
    controller.sync_nodes()
    node = controller.e.node_map()["check"]
    assert node["variable_scopes"] == {"x": PAST, "y": second_scope}
    assert node["alternatives"][0]["applicable_scope"] == (
        {"variables": {"x": PAST, "y": FUTURE}} if mixed else PAST)


def rule_state():
    steps = [{"output_slot": "rules", "inputs": [], "execution": "retrieval"},
             {"output_slot": "fact", "inputs": ["rules"], "execution": "retrieval"},
             {"output_slot": "decision", "inputs": ["rules", "fact"], "execution": "compose"}]
    program = {"variables": {
        "rule": {"type": "rule", "demand": "rules", "scope": {**PAST, "time_scope": "policy"}},
        "result": {"type": "bool", "demand": "decision", "scope": FUTURE}},
        "expression": {"op": "var", "id": "result"}, "rules_complete": False,
        "rule_binding": {"variable": "rule", "condition_demands": ["fact"]}}
    return FactState(program, {"mode": "value", "result_type": "bool", **FUTURE, "explanation": "none"}, steps, "decision")


def integrate_rule(state, scope, source):
    rule = {"variables": {"quantity": {"type": "number", "demand": "fact", "scope": scope}},
            "expression": {"op": "ge", "args": [{"op": "var", "id": "quantity"}, {"op": "const", "value": 0}]},
            "constraints": [], "rules_complete": True}
    row = {**observation(source, [source], rule, **state.scope("rule")), "variable": "rule"}
    state.integrate([row])


def test_sourced_rule_cannot_revise_predeclared_scope():
    state = rule_state()
    # Use an allowed condition task so only the immutable declaration check fails.
    state.initial_program["variables"]["condition"] = {"type": "bool", "demand": "fact", "scope": PAST}
    state.program = deepcopy(state.initial_program)
    rule = {"variables": {"condition": {"type": "bool", "demand": "fact", "scope": FUTURE}},
            "expression": {"op": "var", "id": "condition"}, "constraints": [], "rules_complete": True}
    before = deepcopy(state.program)
    with pytest.raises(ProgramError, match="predeclared variable"):
        state.integrate([{**observation(value=rule, **state.scope("rule")), "variable": "rule"}])
    assert state.program == before and not state.instantiations


@pytest.mark.parametrize("replacement", [False, True])
def test_rule_revision_retains_historical_routes_without_reusing_wrong_scope(replacement):
    state = rule_state()
    integrate_rule(state, PAST, "rule-1")
    state.integrate([{**observation("old", ["old-source"], 4, **PAST), "variable": "quantity"}])
    old_route = state.valid()[2]["quantity"][0]
    state.revoke_source("rule-1", "Superseded rule")
    if replacement:
        integrate_rule(state, FUTURE, "rule-2")
        assert "quantity" not in state.valid()[0]
        state.integrate([{**observation("new", ["new-source"], 4, **FUTURE), "variable": "quantity"}])
    snapshot = deepcopy(state.public_dict())
    restored = rule_state()
    restored.restore(snapshot)
    assert restored.public_dict() == snapshot and old_route in restored.routes
    if replacement:
        assert restored.valid()[1]["quantity"] == ["new-source"]
        controller = controller_for(restored)
        controller.sync_nodes()
        alternatives = controller.e.node_map()["fact"]["alternatives"]
        assert alternatives and all(a["source_span_ids"] == ["new-source"] for a in alternatives)
        assert all(a["applicable_scope"] == FUTURE for a in alternatives)
    else:
        assert "quantity" not in restored.program["variables"]
        assert "quantity" not in restored.valid()[0]


@pytest.mark.parametrize("tamper", ["declaration", "route", "terminal"])
def test_snapshot_rejects_changed_scope_before_adopting_any_state(tamper):
    reader, engine, state, _ = scoped_fixture()
    bind_history(reader, engine, state)
    snapshot = deepcopy(state.public_dict())
    if tamper == "declaration":
        snapshot["program"]["variables"]["history"]["scope"] = FUTURE
    elif tamper == "terminal":
        snapshot["program"]["variables"]["suggestion"]["scope"] = PAST
    else:
        snapshot["routes"][0]["time_scope"] = "future"
    _, _, fresh, _ = scoped_fixture()
    before = deepcopy(fresh.public_dict())
    with pytest.raises(ProgramError):
        fresh.restore(snapshot)
    assert fresh.public_dict() == before


def test_dynamic_snapshot_declaration_must_match_recorded_rule_instantiation():
    state = rule_state()
    integrate_rule(state, PAST, "rule-1")
    snapshot = deepcopy(state.public_dict())
    snapshot["program"]["variables"]["quantity"]["scope"] = FUTURE
    fresh = rule_state()
    with pytest.raises(ProgramError, match="recorded rule instantiation"):
        fresh.restore(snapshot)
    assert fresh.public_dict() == rule_state().public_dict()


def test_later_in_scope_support_does_not_erase_failed_obligation():
    reader, engine, state, calls = scoped_fixture(
        lambda index, data: response(observation(**(FUTURE if index == 1 else PAST))),
        max_repairs_per_request=0)
    read(reader)
    assert "history" in state.pending and reader.failed_obligations
    before = deepcopy(reader.failed_obligations)
    reader.read(engine.steps[0], "Read historical fact", ["history"], ["piano"], "new-query")
    assert state.valid()[0]["history"] == ["piano"]
    assert "history" in state.pending and reader.failed_obligations == before
    assert len(calls.requests) == 2 and engine.ledger.used.get("json_repairs", 0) == 0
