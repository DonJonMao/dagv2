"""Future handoffs cannot invalidate a locally validated independent observation."""
from copy import deepcopy

from dagbt.fact_state import FactState
from dagbt.joint_reading import JointReader
from test_personamem_joint_protocol import fixture, observation, response, visible


def ungrounded_fixture(**settings):
    _, engine, _, calls = fixture(**settings)
    engine.steps.insert(1, {
        "output_slot": "context", "question": "Read other historical context",
        "inputs": [], "execution": "retrieval", "answer_type": "fact",
    })
    engine.steps[-1]["inputs"] = ["history", "context"]
    engine.steps[-1]["question"] = "Suggest from {history} and {context}"
    program = {
        "variables": {
            "history": {"type": "set", "demand": "history"},
            "context": {"type": "set", "demand": "context"},
            "suggestion": {"type": "text", "demand": "suggestion"},
        },
        "expression": {"op": "var", "id": "suggestion"},
    }
    contract = {"mode": "semantic", "result_type": "text", "entity_scope": "user",
                "time_scope": "history", "explanation": "semantic"}
    state = FactState(program, contract, engine.steps, "suggestion")
    reader = JointReader(engine, state)
    checked = []

    def fits(operation, data):
        checked.append((operation, deepcopy(data)))
        return data["task"]["output_slot"] != "suggestion"

    reader._fits = fits
    return reader, engine, state, calls, checked


def test_future_ungrounded_handoff_with_new_exclusive_sources_is_only_advisory():
    reader, engine, state, calls, checked = ungrounded_fixture()
    draft = reader._observation(observation("broad", ["s1", "s2"]), "history",
                                visible(reader, ["s1", "s2"]), engine.steps[0])
    before = deepcopy(state.public_dict())
    assert reader._handoff_failures([draft], {("history", "broad")}) == {}
    assert state.public_dict() == before and not calls.requests
    operation, data = checked[-1]
    assert operation == "semantic_compose"
    assert {s["id"] for s in data["source_spans"]} == {"s1", "s2"}
    event = next(e for e in engine.events if e["event"] == "joint_support_handoff_deferred")
    assert event["reason"] == "ungrounded_future_target"
    assert event["missing_parent_demands"] == ["context"]
    assert event["missing_parent_variables"] == ["context"]
    assert event["source_spans"] == 2 and event["source_ids_preserved"]
    assert event["capacity_fits"] is False and event["executable"] is False
    assert event["parent_versions"] == {"history": 1}
    assert not any(e["event"] == "joint_support_capacity_rejected" for e in engine.events)


def test_future_overflow_does_not_spend_local_repairs_or_create_failed_evidence():
    reader, engine, state, calls, _ = ungrounded_fixture(max_repairs_per_request=2)
    calls.respond = lambda index, data: response(observation("broad", ["s1", "s2"]))
    reader.read(engine.steps[0], "Read creative history", ["history"],
                ["piano", "painting"], "history-query")
    assert len(calls.requests) == 1
    assert state.valid()[0] == {"history": ["piano"]}
    assert set(state.valid()[1]["history"]) == {
        engine.mapper.sources[a]["source_id"] for a in ("s1", "s2")}
    assert not state.pending and not reader.failed_batch_reviews
    assert not reader.evidence_obligations and reader.batches[-1]["complete"]
    assert reader.semantic_inputs is None


def test_same_source_closure_is_strict_once_other_parent_is_grounded():
    reader, engine, state, _, checked = ungrounded_fixture()
    state.integrate([reader._observation(observation("context", ["s2"]), "context",
                     visible(reader, ["s2"]), engine.steps[1])])
    draft = reader._observation(observation("broad", ["s1", "s2"]), "history",
                                visible(reader, ["s1", "s2"]), engine.steps[0])
    rejected = reader._handoff_failures([draft], {("history", "broad")})
    assert rejected["history", "broad"].startswith("citation_capacity:")
    assert checked[-1][0] == "semantic_compose"
    assert not any(e["event"] == "joint_support_handoff_deferred" for e in engine.events)
    assert any(e["event"] == "joint_support_capacity_rejected" for e in engine.events)


def test_ungrounded_future_cannot_admit_an_invalid_local_source():
    reader, engine, state, calls, _ = ungrounded_fixture(max_repairs_per_request=0)
    calls.respond = lambda index, data: response(observation("invalid", ["unoffered"]))
    reader.read(engine.steps[0], "Read creative history", ["history"], ["piano"], "history-query")
    assert not state.routes and state.pending == {"history"}
    assert reader.failed_batch_reviews and not reader.batches[-1]["complete"]
    assert not any(e["event"] == "joint_support_handoff_deferred" for e in engine.events)
