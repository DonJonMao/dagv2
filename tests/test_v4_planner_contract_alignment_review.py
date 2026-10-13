"""Independent planner contract and repair adversaries; synthetic data only.

These tests use actual Engine/Reasoner admission and budget code. The offline
response source never performs HTTP, tokenization-service or dataset requests.
"""
from copy import deepcopy
import json

from jsonschema import Draft202012Validator
import numpy as np
import pytest

from dagbt import local_terminal, residual_plan
from dagbt.engine import Engine, legacy_modules
from dagbt.methods import RESIDUAL_V2
from dagbt.personal_task import validate_personal_plan
from dagbt.budget import BudgetExceeded
from dagbt.reasoning import InputOverflow, ProtocolError
from test_engine import Tokenizer


QUESTION = "I would like an activity that fits my interests."
OPTIONS = ["Synthetic response one", "Synthetic response two",
           "Synthetic response three", "Synthetic response four"]


def step(name, question, inputs=(), execution="retrieval"):
    return {"question": question, "output_slot": name, "answer_type": "text",
            "inputs": list(inputs), "execution": execution}


def personal_plan():
    return {
        "steps": [step("history", "Read relevant historical interests"),
                  step("reply", QUESTION, ["history"], "compose")],
        "final_node_id": "reply",
        "output_contract": {"mode": "semantic", "result_type": "text",
                            "explanation": "semantic", "entity_scope": "user",
                            "time_scope": "reply now"},
        "program": {"variables": {
            "history": {"type": "text", "demand": "history",
                        "evidence_semantics": "personal_history_summary",
                        "scope": {"entity_scope": "user", "time_scope": "history"}},
            "reply": {"type": "text", "demand": "reply"}},
            "expression": {"op": "var", "id": "reply"}},
    }


def single_plan(execution="retrieval", *, symbolic=True):
    value = personal_plan()
    value["steps"] = [step("reply", QUESTION, execution=execution)]
    del value["program"]["variables"]["history"]
    if symbolic:
        value["output_contract"].update(mode="value", explanation="none")
    return value


def validate(value, personal=True):
    _, repair, _ = legacy_modules()
    result = residual_plan.validate_plan(value, repair.validate_plan,
                                         personal_history=personal)
    return validate_personal_plan(result, QUESTION) if personal else result


def schema(personal=True):
    legacy, _, _ = legacy_modules()
    return residual_plan.planner_contract(legacy.PLAN_SCHEMA,
                                          personal_history=personal)


class PlannerScript:
    def __init__(self, *responses):
        self.responses = deepcopy(responses)
        self.requests = []

    def get(self, stage, url, payload):
        assert stage[0] == "planner", "No retrieval is allowed before admission"
        assert url.endswith("/chat/completions")
        self.requests.append({"stage": list(stage), "payload": deepcopy(payload)})
        index = len(self.requests) - 1
        assert index < len(self.responses), "Unexpected additional planner call"
        value = self.responses[index]
        raw = value if isinstance(value, str) else json.dumps(value)
        return {"response_ref": "synthetic-planner-" + str(index + 1), "response": {
            "choices": [{"finish_reason": "stop", "message": {"content": raw}}]}}


def engine_for(*responses, personal=True, fusion=None):
    calls = PlannerScript(*responses)
    question = {"id": "synthetic-contract-review", "question": QUESTION}
    config = {"_test_transport": True, "llm_base_url": "http://offline.invalid/v1",
              "llm_model": "synthetic", "fusion": dict(fusion or {})}
    engine = Engine(question, ({}, [], np.empty((0, 3), np.float32), None, Tokenizer()),
                    calls, config, RESIDUAL_V2, reader_question=QUESTION,
                    output_options=OPTIONS,
                    task_kind="personal_response_selection" if personal else None)
    return engine, calls


def data(request):
    return json.loads(request["payload"]["messages"][1]["content"])


@pytest.mark.parametrize("field,value", [("mode", "value"),
                                          ("result_type", "bool"),
                                          ("explanation", "none")])
def test_personal_schema_rejects_nonpersonal_output_contract(field, value):
    plan = personal_plan()
    plan["output_contract"][field] = value
    assert not Draft202012Validator(schema()).is_valid(plan)
    with pytest.raises(ProtocolError):
        validate(plan)


@pytest.mark.parametrize("execution,symbolic", [("compose", False),
                                               ("retrieval", True),
                                               ("retrieval", False)])
def test_personal_schema_rejects_one_node_case14_shapes(execution, symbolic):
    plan = single_plan(execution, symbolic=symbolic)
    assert not Draft202012Validator(schema()).is_valid(plan)
    with pytest.raises(ProtocolError):
        validate(plan)


def test_local_schema_rejects_parentless_compose_but_allows_one_retrieval():
    legacy, _, _ = legacy_modules()
    contract = local_terminal.planner_contract(legacy.PLAN_SCHEMA)
    one = single_plan("compose", symbolic=False)
    local = {key: one[key] for key in ("steps", "final_node_id")}
    assert not Draft202012Validator(contract).is_valid(local)
    local["steps"][0]["execution"] = "retrieval"
    assert Draft202012Validator(contract).is_valid(local)


def test_personal_repair_carries_latest_real_draft_and_both_errors():
    first = single_plan("compose", symbolic=False)
    second = single_plan("retrieval", symbolic=True)
    good = personal_plan()
    engine, calls = engine_for(first, second, good)
    original = deepcopy(engine.q)
    supplied = deepcopy(calls.responses)
    engine.plan()
    assert engine.q == original and calls.responses == supplied
    assert len(calls.requests) == 3
    assert data(calls.requests[0]) == QUESTION
    repair = data(calls.requests[2])
    assert repair["original_input"] == QUESTION
    review = repair["planner_repair"]
    assert review["previous_draft"] == engine.reasoner.requests[1]["raw_output"]
    assert review["previous_call_index"] == engine.reasoner.requests[1]["call_index"]
    assert [entry["error"] for entry in review["validation_errors"]] == [
        row["validation_error"] for row in engine.reasoner.requests[:2]]
    assert [entry["failure_category"] for entry in review["validation_errors"]] == [
        row["failure_category"] for row in engine.reasoner.requests[:2]]
    assert [entry["repair_index"] for entry in review["validation_errors"]] == [1, 2]
    assert "Compose requires parents" in review["validation_errors"][0]["error"]
    assert "semantic/text" in review["validation_errors"][1]["error"]
    assert engine.ledger.used["json_repairs"] == 2
    assert engine.ledger.used["llm"] == 3 and engine.ledger.used.get("ann", 0) == 0
    assert OPTIONS[0] not in str(calls.requests)


@pytest.mark.parametrize("protocol", ["plain", "json_object", "json_schema"])
def test_actual_personal_wire_uses_aligned_schema_in_every_protocol(protocol):
    engine, calls = engine_for(personal_plan(), fusion={"response_format": protocol})
    engine.plan()
    payload = calls.requests[0]["payload"]
    if protocol == "json_schema":
        actual = payload["response_format"]["json_schema"]["schema"]
    else:
        marker = "Return one complete JSON object matching this schema:\n"
        actual = json.loads(payload["messages"][0]["content"].rsplit(marker, 1)[1])
        assert ("response_format" in payload) == (protocol == "json_object")
    assert actual == schema()
    assert Draft202012Validator(actual).is_valid(personal_plan())
    assert not Draft202012Validator(actual).is_valid(single_plan("compose", symbolic=False))
    assert len(calls.requests) == 1 and engine.ledger.used["json_repairs"] == 0


def test_contract_builders_preserve_legacy_and_generic_inputs():
    legacy, _, _ = legacy_modules()
    original = deepcopy(legacy.PLAN_SCHEMA)
    generic = schema(False)
    personal = schema(True)
    assert legacy.PLAN_SCHEMA == original and schema(False) == generic
    personal["properties"]["steps"]["minItems"] = 999
    personal["properties"]["program"]["properties"].clear()
    assert legacy.PLAN_SCHEMA == original and schema(False) == generic
    assert Draft202012Validator(schema()).is_valid(personal_plan())


def test_generic_one_node_retrieval_stays_legal_through_actual_engine():
    value = single_plan()
    saved = deepcopy(value)
    assert Draft202012Validator(schema(False)).is_valid(value)
    assert validate(value, personal=False) == saved
    engine, calls = engine_for(value, personal=False)
    engine.plan()
    assert value == saved and len(calls.requests) == 1
    assert engine.final_node_id == "reply" and len(engine.steps) == 1


@pytest.mark.parametrize("kind", ["number", "record", "set", "bool", "entity"])
def test_personal_intermediate_types_remain_legal_without_text_markers(kind):
    value = personal_plan()
    history = value["program"]["variables"]["history"]
    history["type"] = kind
    del history["evidence_semantics"]
    value["steps"][0]["answer_type"] = kind
    if kind == "number":
        history.update(domain={"lower": None, "upper": None}, unit="s")
    assert Draft202012Validator(schema()).is_valid(value)
    assert validate(value) == value
    engine, calls = engine_for(value)
    engine.plan()
    assert len(calls.requests) == 1 and engine.fact_state.program["variables"]["history"]["type"] == kind


def test_personal_numeric_record_and_deterministic_intermediate_are_not_a_fixed_template():
    value = personal_plan()
    scope = {"entity_scope": "user", "time_scope": "history"}
    value["steps"] = [step("duration", "Read the recorded duration"),
                      step("attributes", "Read the recorded attributes"),
                      step("combined", "Combine supplied fields", ["duration", "attributes"], "compose"),
                      step("reply", QUESTION, ["combined"], "compose")]
    value["program"]["variables"] = {
        "n": {"type": "number", "demand": "duration", "scope": scope},
        "r": {"type": "record", "demand": "attributes", "scope": scope},
        "joined": {"type": "record", "demand": "combined", "scope": scope},
        "terminal_value": {"type": "text", "demand": "reply"}}
    value["program"]["expression"] = {"op": "var", "id": "terminal_value"}
    value["program"]["task_expressions"] = {"joined": {"op": "record", "fields": {
        "duration": {"op": "var", "id": "n"}, "attributes": {"op": "var", "id": "r"}}}}
    original = deepcopy(value)
    assert Draft202012Validator(schema()).is_valid(value)
    assert validate(value) == original
    engine, calls = engine_for(value)
    engine.plan()
    assert len(calls.requests) == 1 and len(engine.steps) == 4
    assert engine.fact_state.program["task_expressions"] == original["program"]["task_expressions"]
    assert value == original


@pytest.mark.parametrize("history_count", [1, 3, 5])
def test_personal_history_count_and_unordered_storage_are_not_fixed(history_count):
    value = personal_plan()
    histories = ["memory_" + str(i) for i in range(history_count)]
    declaration = value["program"]["variables"].pop("history")
    value["steps"] = [step(name, "Read an independent historical fact") for name in histories]
    value["steps"].append(step("reply", QUESTION, histories, "compose"))
    for name in histories:
        value["program"]["variables"][name] = {**deepcopy(declaration), "demand": name}
    value["steps"].reverse()
    assert Draft202012Validator(schema()).is_valid(value)
    engine, calls = engine_for(value)
    engine.plan()
    assert len(calls.requests) == 1 and len(engine.steps) == history_count + 1
    assert engine.steps[-1]["output_slot"] == "reply"


def test_personal_schema_rejects_nonvariable_terminal_expression_and_rule_binding():
    value = personal_plan()
    value["program"]["expression"] = {"op": "const", "value": "invented answer"}
    assert not Draft202012Validator(schema()).is_valid(value)
    with pytest.raises(ProtocolError):
        validate(value)
    value = personal_plan()
    value["program"]["rule_binding"] = {"variable": "history", "condition_demands": ["reply"]}
    assert not Draft202012Validator(schema()).is_valid(value)
    with pytest.raises(ProtocolError):
        validate(value)


@pytest.mark.parametrize("local_repairs,global_repairs,spent,expected_calls", [
    (2, 6, 0, 3), (0, 6, 0, 1), (2, 1, 0, 2), (2, 6, 5, 2), (2, 6, 6, 1)])
def test_personal_bad_plans_keep_local_two_and_shared_repair_limits(
        local_repairs, global_repairs, spent, expected_calls):
    invalid = single_plan("compose", symbolic=False)
    engine, calls = engine_for(*([invalid] * expected_calls), fusion={
        "max_repairs_per_request": local_repairs, "json_repairs": global_repairs})
    engine.ledger.reserve("json_repairs", "earlier-phase", spent)
    limits = deepcopy(engine.ledger.limits)
    with pytest.raises(ProtocolError, match="Compose requires parents"):
        engine.plan()
    assert len(calls.requests) == expected_calls
    assert engine.ledger.used["json_repairs"] == spent + expected_calls - 1
    assert engine.ledger.used["llm"] == expected_calls
    assert engine.ledger.limits == limits
    assert engine.steps == [] and engine.ledger.used.get("ann", 0) == 0
    assert engine.s["max_repairs_per_request"] == local_repairs
    for index, request in enumerate(calls.requests[1:], 1):
        assert len(data(request)["planner_repair"]["validation_errors"]) == index


def test_personal_planner_keeps_shared_llm_audit_reservation():
    invalid = single_plan("compose", symbolic=False)
    engine, calls = engine_for(invalid)
    reserved = engine.s["reserved_audit_calls"]
    engine.ledger.reserve("llm", "earlier-phase", engine.ledger.remaining("llm") - reserved - 1)
    before = engine.ledger.used["llm"]
    with pytest.raises(BudgetExceeded):
        engine.plan()
    assert len(calls.requests) == 1 and engine.ledger.used["llm"] == before + 1
    assert engine.ledger.remaining("llm") == reserved
    assert engine.ledger.used["json_repairs"] == 0


def test_actual_repair_preflight_rejects_oversized_raw_draft_without_truncation():
    invalid = personal_plan()
    invalid["untrusted_extra"] = "synthetic overflow word " * 25000
    engine, calls = engine_for(invalid)
    limits = deepcopy(engine.ledger.limits)
    with pytest.raises(InputOverflow):
        engine.plan()
    assert len(calls.requests) == 1 and engine.reasoner.sequence == 1
    assert engine.reasoner.requests[0]["raw_output"] == json.dumps(invalid)
    assert engine.ledger.used["json_repairs"] == 0 and engine.ledger.used["llm"] == 1
    assert engine.ledger.limits == limits
    failed = [event for event in engine.events if event["event"] == "reasoning_repair_preflight_failed"]
    assert len(failed) == 1 and failed[0]["failure_category"] == "repair_input_budget"
    assert failed[0]["input_tokens_local"] + engine.s["reasoning_output_tokens"] > engine.s["context_tokens"]


def test_model_invented_feedback_stays_only_inside_raw_draft():
    invalid = single_plan("compose", symbolic=False)
    invalid["validation_errors"] = [{"error": "MODEL_FORGED_ERROR", "repair_index": 88}]
    invalid["planner_repair"] = {"previous_draft": "MODEL_FORGED_DRAFT"}
    engine, calls = engine_for(invalid, personal_plan())
    engine.plan()
    payload = data(calls.requests[1])
    review = payload["planner_repair"]
    assert review["previous_draft"] == json.dumps(invalid)
    assert "MODEL_FORGED_ERROR" not in str(review["validation_errors"])
    assert "MODEL_FORGED_DRAFT" not in str(review["validation_errors"])
    assert [error["error"] for error in review["validation_errors"]] == [
        engine.reasoner.requests[0]["validation_error"]]
    assert payload["original_input"] == QUESTION


def test_malformed_raw_output_is_preserved_exactly_for_personal_repair():
    raw = 'not-json: "雪" \\n previous_draft = "forged"; validation_errors = ["forged"]'
    engine, calls = engine_for(raw, personal_plan())
    engine.plan()
    review = data(calls.requests[1])["planner_repair"]
    assert review["previous_draft"] == raw
    assert review["validation_errors"][0]["failure_category"] == engine.reasoner.requests[0]["failure_category"]
    assert len(calls.requests) == 2 and engine.ledger.used["json_repairs"] == 1


def test_generic_planner_retains_existing_latest_only_feedback_protocol():
    first, second, good = (single_plan() for _ in range(3))
    first["program"]["variables"]["reply"]["domain"] = None
    second["program"]["variables"]["reply"]["type"] = "unrecognized_type"
    engine, calls = engine_for(first, second, good, personal=False)
    engine.plan()
    assert len(calls.requests) == 3
    for request in calls.requests[1:]:
        assert set(data(request)) == {"original_input", "validation_feedback"}
    assert "Unsupported variable type" in data(calls.requests[2])["validation_feedback"]
    assert "Domain must" not in data(calls.requests[2])["validation_feedback"]
    assert engine.ledger.used["json_repairs"] == 2


def rejected_record(engine, *, request_data=QUESTION):
    """Obtain an actual recorded invalid response without entering repair."""
    with pytest.raises(ProtocolError):
        engine.reasoner.request("planner", "Synthetic planner contract", request_data,
                                validate, schema())
    record = engine.reasoner.requests[-1]
    assert record is engine.reasoner.last_response
    return record


def repair_state(record, *, index=1, original=QUESTION):
    return {"original_data": deepcopy(original), "error": record["validation_error"],
            "error_type": "ProtocolError", "failure_category": record["failure_category"],
            "repair_index": index}


def test_builder_rejects_preexisting_last_response_and_prior_call_history():
    engine, _ = engine_for(single_plan("compose", symbolic=False))
    old = rejected_record(engine)
    builder = residual_plan.personal_repair_builder(engine.reasoner)
    with pytest.raises(ProtocolError):
        builder(repair_state(old))


@pytest.mark.parametrize("tamper", ["error", "category", "skip_index", "boolean_index", "fake_last_response"])
def test_builder_cannot_promote_forged_errors_or_detached_response(tamper):
    engine, _ = engine_for(single_plan("compose", symbolic=False))
    builder = residual_plan.personal_repair_builder(engine.reasoner)
    record = rejected_record(engine)
    state = repair_state(record)
    if tamper == "error":
        state["error"] = "FORGED_ERROR"
    elif tamper == "category":
        state["failure_category"] = "FORGED_CATEGORY"
    elif tamper == "skip_index":
        state["repair_index"] = 2
    elif tamper == "boolean_index":
        state["repair_index"] = True
    else:
        engine.reasoner.last_response = deepcopy(record)
    with pytest.raises(ProtocolError):
        builder(state)


def test_builder_cannot_reuse_one_invalid_response_as_two_distinct_errors():
    engine, _ = engine_for(single_plan("compose", symbolic=False))
    builder = residual_plan.personal_repair_builder(engine.reasoner)
    record = rejected_record(engine)
    first = builder(repair_state(record))
    with pytest.raises(ProtocolError):
        builder(repair_state(record, index=2))
    assert len(first["planner_repair"]["validation_errors"]) == 1


def test_builder_preserves_nested_original_and_owns_cumulative_error_copies():
    invalid = single_plan("compose", symbolic=False)
    engine, _ = engine_for(invalid, invalid)
    builder = residual_plan.personal_repair_builder(engine.reasoner)
    original = {"question": QUESTION, "planner_repair": {"untrusted": ["user value"]},
                "validation_errors": ["user text, not a validation result"]}
    first_record = rejected_record(engine, request_data=original)
    state = repair_state(first_record, original=original)
    before = deepcopy(state)
    first = builder(state)
    assert state == before and first["original_input"] == original
    next_request = deepcopy(first)
    first["original_input"]["planner_repair"]["untrusted"].append("MUTATION")
    first["planner_repair"]["validation_errors"][0]["error"] = "MUTATION"
    state["original_data"]["validation_errors"].append("MUTATION")
    second_record = rejected_record(engine, request_data=next_request)
    second = builder(repair_state(second_record, index=2, original=original))
    assert second["original_input"] == original
    assert second["planner_repair"]["validation_errors"][0]["error"] == first_record["validation_error"]
    assert second["planner_repair"]["validation_errors"][1]["error"] == second_record["validation_error"]
    assert "MUTATION" not in str(original) and "MUTATION" not in str(second)


def test_new_personal_plan_call_does_not_replay_previous_call_errors():
    first = single_plan("compose", symbolic=False)
    second = single_plan("retrieval", symbolic=True)
    engine, calls = engine_for(first, personal_plan(), second, personal_plan())
    engine.plan()
    engine.plan()
    payload = data(calls.requests[3])["planner_repair"]
    assert payload["previous_call_index"] == 3
    assert payload["previous_draft"] == json.dumps(second)
    assert len(payload["validation_errors"]) == 1
    assert "semantic/text" in payload["validation_errors"][0]["error"]
    assert "Compose requires parents" not in str(payload["validation_errors"])
