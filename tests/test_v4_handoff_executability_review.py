"""Independent execution-boundary review with synthetic three-task DAGs.

Real Engine, FactState, Reader, Controller, source bank and wire preflight are
used. Only model responses are scripted; no labels or external services exist.
"""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from dagbt.bridge import BridgeSession
from dagbt.engine import Engine
from dagbt.joint_reading import JointReader
from dagbt.methods import RESIDUAL_V2
from dagbt.reasoning import InputOverflow
from dagbt.residual_control import ResidualControl
from test_engine import Tokenizer
from test_personamem_joint_protocol import expand_wire_data


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Handoff executability review must remain offline")
    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    monkeypatch.setattr("urllib.request.build_opener", forbidden)


def step(name, parents=(), *, long=False):
    return {"output_slot": name, "question": "Read synthetic " + name + (
        " future-context" * 4500 if long else ""), "inputs": list(parents),
        "execution": "compose" if parents else "retrieval", "answer_type": "text"}


def plan(*, parent_conflict=False):
    steps = ([step("root"), step("a", ["root"]), step("b"), step("terminal", ["a", "b"], long=True)]
             if parent_conflict else [step("a"), step("b"), step("terminal", ["a", "b"], long=True)])
    return {"steps": steps, "final_node_id": "terminal",
        "output_contract": {"mode": "semantic", "result_type": "text", "explanation": "semantic",
                            "entity_scope": "synthetic user", "time_scope": "history"},
        "program": {"variables": {s["output_slot"]: {"type": "text", "demand": s["output_slot"]}
                                  for s in steps}, "expression": {"op": "var", "id": "terminal"}}}


class SyntheticServices:
    def __init__(self, value):
        self.plan = deepcopy(value)
        self.requests = []
        self.respond = None

    def get(self, stage, url, payload):
        assert url.endswith("/chat/completions"), "No retrieval service is required by these regressions"
        data = expand_wire_data(json.loads(payload["messages"][1]["content"])) if stage[0] != "planner" else None
        self.requests.append({"stage": tuple(stage), "data": data, "payload": deepcopy(payload)})
        value = self.plan if stage[0] == "planner" else self.respond(stage, data)
        return {"response_ref": "synthetic-handoff-" + str(len(self.requests)), "response": {
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}]}}


def setup(*, parent_conflict=False, initialize=True):
    value = plan(parent_conflict=parent_conflict)
    services = SyntheticServices(value)
    texts = {"a": "The synthetic user explicitly prefers quiet activities.",
             "b": "The synthetic user explicitly records a second relevant condition.",
             "counter": "The earlier synthetic preference is withdrawn; the current preference is collaborative activities.",
             "root": "The synthetic parent binding is explicitly recorded."}
    docs = {key: SimpleNamespace(doc_id=key, passage=text, metadata={"source_segments": [{
        "role": "user", "start": 0, "end": len(text), "source_message_indices": [i]}]})
            for i, (key, text) in enumerate(texts.items())}
    config = {"_test_transport": True, "llm_base_url": "http://offline.invalid/v1", "llm_model": "script",
              "embedding_base_url": "http://offline.invalid/v1", "embedding_model": "script",
              "fusion": {"reasoning_output_tokens": 512, "max_repairs_per_request": 0,
                         "initial_width": 1, "proposal_width": 1}}
    engine = Engine({"id": "synthetic-handoff", "question": "Respond using both historical conditions."},
                    (docs, list(docs), np.array([[1., 0., 0.]] * len(docs), np.float32), None, Tokenizer()),
                    services, config, RESIDUAL_V2)
    if not initialize:
        return engine, None, services
    engine.plan()
    engine.fixed_pool = None
    engine.bridge = BridgeSession(engine.q["question"], engine.docs, engine.ids, engine.vectors,
                                  engine.tokenizer, engine.calls, engine.config, engine.ledger)
    control = ResidualControl(engine)
    engine.residual_control = control
    engine.add_candidates(list(docs))
    return engine, control, services


def task(control, name):
    return next(value for value in control.e.steps if value["output_slot"] == name)


def alias(control, doc):
    return next(name for name, source in control.e.mapper.sources.items() if source["doc_id"] == doc)


def wire_row(control, variable, identifier, *, doc=None, value=None, **extra):
    return {"id": identifier, "stance": "support", "value": value or "Synthetic " + variable,
            "source_ids": [alias(control, doc or variable)], **control.state.scope(variable),
            "parent_versions": {}, "retract_ids": [], "reason": "Explicit synthetic original.", **extra}


def normalized(control, variable, identifier, **kwargs):
    return control.reader._observation(wire_row(control, variable, identifier, **kwargs), variable,
                                      control.e.mapper.sources, task(control, variable))


def mark_pending(control, variable="b"):
    source = control.e.mapper.sources[alias(control, variable)]["source_id"]
    control.reader.evidence_obligations["synthetic-gap-" + variable] = {
        "id": "synthetic-gap-" + variable, "variable": variable, **control.state.scope(variable),
        "source_ids": [source], "descriptions": ["A specific sourced premise remains unresolved."],
        "origins": ["prior-owner"]}
    control.state.integrate([], control.reader.pending_variables())


def test_unexecutable_future_capacity_does_not_reject_independent_valid_support():
    engine, control, services = setup()
    mark_pending(control)
    reader = control.reader
    row = normalized(control, "a", "a-proof")
    initial = reader._payload(task(control, "a"), control.grounded(task(control, "a")), ["a"], [alias(control, "a")])
    assert reader._fits("joint_read", initial), "The current paid read must fit the real wire"
    before = deepcopy(control.state.public_dict())
    signature = reader.material_signature()
    budgets = deepcopy(engine.ledger.public_dict())
    assert control.grounded(task(control, "terminal")) is None
    assert reader._handoff_failures([row], {("a", "a-proof")}, pending={"b"}) == {}
    assert control.state.public_dict() == before and reader.material_signature() == signature
    assert engine.ledger.public_dict() == budgets and len(services.requests) == 1
    advisories = [event for event in engine.events if event["event"] == "joint_support_handoff_deferred"]
    assert len(advisories) == 1
    assert advisories[0]["next_node_id"] == "terminal"
    assert advisories[0]["missing_parent_demands"] == ["b"]
    assert advisories[0]["missing_parent_variables"] == ["b"]
    assert advisories[0]["executable"] is False and advisories[0]["capacity_fits"] is False
    assert advisories[0]["source_ids_preserved"] is True


def read_local(control, variable, identity, *, documents=None, progress=False):
    method = control.read_progress if progress else control.reader.read
    selected = task(control, variable)
    return method(selected, control.grounded(selected), [variable], documents or [variable], identity,
                  discovery_doc_ids=documents or [variable])


def simple_response(control, data):
    return {"slots": [{"variable": variable, "observations": [wire_row(control, variable, variable + "-proof")],
                       "unresolved": []} for variable in data["requested_slots"]]}


def test_actual_reader_admits_a_preserves_pending_b_and_does_not_publish():
    engine, control, services = setup()
    mark_pending(control)
    pending_before = deepcopy(control.reader.evidence_obligations)
    original_sources = deepcopy(engine.mapper.sources)
    services.respond = lambda stage, data: simple_response(control, data)
    assert read_local(control, "a", "independent-a")
    assert control.state.valid()[0]["a"] == "Synthetic a"
    assert control.state.pending == {"b"}
    assert control.reader.evidence_obligations == pending_before
    assert not control.reader.failed_batch_reviews
    assert engine.mapper.sources == original_sources
    assert len(services.requests) == 2 and engine.ledger.used["json_repairs"] == 0
    assert control.candidate() is None and not engine.audit_complete
    assert control.grounded(task(control, "terminal")) is None
    row = control.publish()
    assert row["answer"]["prediction"] is None and row["answer"]["status"] != "ok"


@pytest.mark.parametrize("first,second", [("a", "b"), ("b", "a")])
def test_second_independent_support_makes_handoff_executable_and_strict(first, second):
    engine, control, services = setup()
    services.respond = lambda stage, data: simple_response(control, data)
    assert read_local(control, first, "first-" + first)
    routes_before = deepcopy(control.state.routes)
    assert control.grounded(task(control, "terminal")) is None
    deferred_before = len([event for event in engine.events if event["event"] == "joint_support_handoff_deferred"])
    read_local(control, second, "second-" + second)
    assert len(services.requests) == 3, "One planner and exactly one current read per independent input"
    assert control.state.routes == routes_before
    assert first in control.state.valid()[0] and second not in control.state.valid()[0]
    assert second in control.state.pending and control.reader.failed_batch_reviews
    rejected = [event for event in engine.events if event["event"] == "joint_support_capacity_rejected"]
    assert rejected and rejected[-1]["next_node_id"] == "terminal"
    assert rejected[-1]["support_draft_ids"] == [second + "-proof"]
    assert len([event for event in engine.events if event["event"] == "joint_support_handoff_deferred"]) == deferred_before
    assert engine.ledger.used["json_repairs"] == 0 and not engine.audit_complete


def test_later_real_terminal_overflow_preserves_saved_proof_and_fails_before_a_call():
    engine, control, services = setup()
    a = normalized(control, "a", "a-proof")
    assert control.reader._handoff_failures([a], {("a", "a-proof")}) == {}
    control.state.integrate([a])
    # Independent source validation may arrive from another completed local
    # batch. It never grants the now-grounded terminal extra wire capacity.
    b = normalized(control, "b", "b-proof")
    control.state.integrate([b])
    before = deepcopy(control.state.routes)
    selected = task(control, "terminal")
    query = control.grounded(selected)
    assert query is not None
    with pytest.raises(InputOverflow):
        control.reader.read(selected, query, ["terminal"], [], "real-terminal", semantic=True,
                            source_scope="mandatory_closure")
    assert control.state.routes == before
    assert len(services.requests) == 1 and engine.ledger.used["json_repairs"] == 0
    assert not engine.audit_complete and control.reader.semantic_inputs is None
    assert control.publish()["answer"]["prediction"] is None


def test_same_batch_later_counter_is_integrated_before_any_future_capacity_advisory():
    engine, control, services = setup()
    mark_pending(control)
    prior = normalized(control, "a", "old-a", value="quiet activities")
    control.state.integrate([prior], pending={"b"})
    old_id = control.state.valid()[2]["a"][0]
    before_obligations = deepcopy(control.reader.evidence_obligations)
    def respond(stage, data):
        positive = wire_row(control, "a", "new-a", doc="counter", value="collaborative activities")
        counter = wire_row(control, "a", "counter-after-support", doc="counter",
                           stance="contradiction", retract_ids=[old_id])
        counter["value"] = None
        return {"slots": [{"variable": "a", "observations": [positive, counter], "unresolved": []}]}
    services.respond = respond
    read_local(control, "a", "counter-and-new-value", documents=["counter"])
    assert len(services.requests) == 2
    assert control.state.routes[old_id]["revoked"]
    assert control.state.valid()[0]["a"] == "collaborative activities"
    assert control.state.pending == {"b"} and control.reader.evidence_obligations == before_obligations
    assert not control.reader.failed_batch_reviews and control.candidate() is None
    assert any(event["event"] == "joint_support_handoff_deferred" for event in engine.events)
    assert not any(event["event"] == "terminal_published" for event in engine.events)


def test_self_conflict_recheck_stays_strict_when_parent_counter_makes_query_ungrounded():
    engine, control, services = setup(parent_conflict=True)
    root = normalized(control, "root", "root-proof")
    control.state.integrate([root])
    parent_version = control.state.versions["root"]
    old_a = normalized(control, "a", "old-a", parent_versions={"root": parent_version})
    control.state.integrate([old_a])
    new_a = normalized(control, "a", "new-a", value="changed synthetic a",
                       parent_versions={"root": parent_version})
    root_id = control.state.valid()[2]["root"][0]
    wire = wire_row(control, "root", "root-counter", doc="counter", stance="contradiction",
                    retract_ids=[root_id])
    wire["value"] = None
    counter = control.reader._observation(wire, "root", engine.mapper.sources, task(control, "root"))
    proposed = control.reader._prospective_state([new_a, counter])
    assert "root" not in proposed.valid()[0] and "a" not in proposed.valid()[0]
    probe = ResidualControl(SimpleNamespace(**{**vars(engine), "fact_state": proposed}))
    assert probe.grounded(task(control, "a")) is None
    checked = []
    real_fits = control.reader._fits
    def fits(operation, data):
        checked.append((operation, deepcopy(data)))
        return False if data["task"]["output_slot"] == "a" else real_fits(operation, data)
    control.reader._fits = fits
    before = deepcopy(control.state.public_dict())
    failures = control.reader._handoff_failures([new_a, counter], {("a", "new-a")})
    assert ("a", "new-a") in failures and " at a " in failures["a", "new-a"]
    assert any(data["task"]["output_slot"] == "a" for _, data in checked)
    assert any(event["event"] == "joint_support_capacity_rejected" and event["next_node_id"] == "a"
               for event in engine.events)
    assert not any(event["event"] == "joint_support_handoff_deferred" and event["next_node_id"] == "a"
                   for event in engine.events)
    assert control.state.public_dict() == before and len(services.requests) == 1


def test_checkpoint_keeps_admitted_sources_pending_versions_and_replay_guards(monkeypatch):
    engine, control, services = setup()
    mark_pending(control)
    services.respond = lambda stage, data: simple_response(control, data)
    assert read_local(control, "a", "checkpoint-a", progress=True)
    control.forced_demands.add("b")
    control.exhausted.add("older-exhausted-query")
    before = control.reader.material_signature()
    snapshot = json.loads(json.dumps(control.snapshot()))
    fresh, _, fresh_services = setup(initialize=False)
    # Exercise real Engine.resume/Controller.restore and stop at its run
    # boundary: this test reviews checkpoint restoration, not new retrieval.
    monkeypatch.setattr(ResidualControl, "run", lambda self: self)
    restored = fresh.resume(snapshot)
    assert restored.state.public_dict() == control.state.public_dict()
    assert restored.state.versions == control.state.versions
    assert restored.state.pending == {"b"}
    assert restored.reader.evidence_obligations == control.reader.evidence_obligations
    assert restored.reader.material_signature() == before
    assert restored.read_attempts == control.read_attempts and restored.read_effects == control.read_effects
    assert restored.forced_demands == {"b"} and restored.exhausted == {"older-exhausted-query"}
    for route in restored.state.routes.values():
        for source in route["source_ids"]:
            assert fresh.spans[source] == engine.spans[source]
    assert not fresh_services.requests, "Restore performs no planner/model/tokenizer/retrieval call"
    current = task(restored, "terminal")
    assert restored.grounded(current) is None


def test_advisory_events_alone_do_not_change_material_state_or_free_retry_allowance():
    engine, control, services = setup()
    mark_pending(control)
    row = normalized(control, "a", "a-proof")
    signature = control.reader.material_signature()
    ledger = deepcopy(engine.ledger.public_dict())
    markers = deepcopy(control.reader.obligation_snapshot())
    control.forced_demands.add("b")
    control.exhausted.add("saved-query")
    for _ in range(3):
        assert control.reader._handoff_failures([row], {("a", "a-proof")}, pending={"b"}) == {}
    assert control.reader.material_signature() == signature
    assert control.reader.obligation_snapshot() == markers
    assert engine.ledger.public_dict() == ledger and len(services.requests) == 1
    assert control.forced_demands == {"b"} and control.exhausted == {"saved-query"}
    assert not control.read_attempts and not control.read_effects
    services.respond = lambda stage, data: simple_response(control, data)
    assert read_local(control, "a", "bounded-progress", progress=True)
    stable = control.reader.material_signature()
    paid = len(services.requests)
    assert not read_local(control, "a", "bounded-progress", progress=True)
    assert not read_local(control, "a", "bounded-progress", progress=True)
    assert len(services.requests) == paid and control.reader.material_signature() == stable


def test_general_advisory_also_admits_a_support_with_real_bound_parent():
    engine, control, services = setup(parent_conflict=True)
    root = normalized(control, "root", "root-proof")
    control.state.integrate([root])
    value = normalized(control, "a", "parented-a", parent_versions={"root": control.state.versions["root"]})
    before = deepcopy(control.state.public_dict())
    assert control.reader._handoff_failures([value], {("a", "parented-a")}) == {}
    assert control.state.public_dict() == before
    deferred = [event for event in engine.events if event["event"] == "joint_support_handoff_deferred"]
    assert deferred and deferred[-1]["missing_parent_demands"] == ["b"]
    assert value["parent_versions"] and len(services.requests) == 1


def test_general_advisory_preserves_new_alternative_without_owner_source_exemption():
    engine, control, services = setup()
    old = normalized(control, "a", "old-proof", value="same synthetic value")
    control.state.integrate([old])
    novel = normalized(control, "a", "new-source-proof", doc="counter", value="same synthetic value")
    before = deepcopy(control.state.routes)
    assert not control.reader.pending_variables(), "No foreign owner provenance is needed for the general rule"
    assert control.reader._handoff_failures([novel], {("a", "new-source-proof")}) == {}
    control.state.integrate([novel])
    assert len(control.state.routes) == len(before) + 1
    assert all(control.state.routes[key] == value for key, value in before.items())
    assert len(services.requests) == 1 and control.candidate() is None
