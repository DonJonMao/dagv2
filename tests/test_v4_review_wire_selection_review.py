"""Independent offline tests for measured review-wire selection.

The counter is deliberately unrelated to byte length: its declared measurements
are the oracle. Real Reasoner payload construction, preflight, and the reader's
exposure boundary run locally. No tokenizer/model/network service is used.
"""
from copy import deepcopy
import json
import socket

import pytest

from dagbt.reasoning import InputOverflow
from dagbt.history_wire import INSTRUCTION as HISTORY_INSTRUCTION
from dagbt.review_wire import INSTRUCTION as REVIEW_INSTRUCTION
from test_personamem_joint_protocol import fixture, observation


OPERATIONS = (
    "joint_read", "joint_read_repair", "semantic_compose",
    "semantic_compose_repair", "audit_joint", "audit_joint_repair",
)
MODES = ("plain", "json_object", "json_schema")
SCHEMA = {"type": "object", "properties": {"visible": {"type": "string"}},
          "required": ["visible"], "additionalProperties": False}


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Independent selection review cannot call services")
    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    monkeypatch.setattr("urllib.request.build_opener", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr("dagbt.reasoning.estimate_evidence_tokens", lambda value: 0)


def pair():
    # The shorter serialized candidate can be more expensive to the real model
    # tokenizer. This pair intentionally has no byte-length measurement oracle.
    return (({"candidate": "baseline", "text": "unchanged source"}, "plain system"),
            ({"candidate": "review", "text": "unchanged source"}, "system + decoding rule"))


class MeasuredCalls:
    """Scripted *complete payload* counts and a single metered fake transport."""
    def __init__(self, engine, baseline=1000, review=900, *, response=None):
        self.engine = engine
        self.values = {"baseline": baseline, "review": review}
        self.counted = []
        self.sent = []
        self.inspect_count = None
        self.send_error = None
        self.response = {"slots": []} if response is None else response

    @staticmethod
    def kind(payload):
        data = json.loads(payload["messages"][1]["content"])
        return data.get("candidate", "review" if "shared_review_texts" in data else "baseline")

    def count_reasoning_tokens(self, stage, url, payload):
        self.counted.append({"stage": stage, "url": url, "payload": deepcopy(payload)})
        if self.inspect_count:
            self.inspect_count()
        value = self.values[self.kind(payload)]
        if isinstance(value, BaseException):
            raise value
        return deepcopy(value) if isinstance(value, dict) else {
            "count": value, "token_count_is_estimate": False, "counter": "independent-fixture"}

    def get(self, stage, url, payload, **options):
        self.sent.append({"stage": stage, "url": url, "payload": deepcopy(payload),
                          "options": deepcopy(options)})
        self.engine.ledger.reserve("llm", "/".join(stage))
        if self.send_error:
            raise self.send_error
        return {"response": {"choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps(self.response)}}]}}


def setup(*, baseline=1000, review=900, response=None, **settings):
    reader, engine, state, _ = fixture(**settings)
    measured = MeasuredCalls(engine, baseline, review, response=response)
    engine.reasoner.calls = measured
    reader.current_identity = "independent-query"
    data = reader._payload(engine.steps[0], "Read the given sources", ["history"], ["s1", "s2"])
    return reader, engine, state, measured, data


@pytest.mark.parametrize("baseline,review,expected", [
    (1000, 999, "review"), (1000, 1000, "baseline"), (1000, 1001, "baseline"),
    (1000, 5000, "baseline"),
])
@pytest.mark.parametrize("operation", OPERATIONS)
def test_strict_actual_token_improvement_is_the_only_selection_reason(baseline, review, expected, operation):
    reader, engine, state, counts, _ = setup(baseline=baseline, review=review)
    candidates = pair()
    unchanged = deepcopy(candidates)
    facts = deepcopy(state.public_dict())
    obligations = deepcopy(reader.obligation_snapshot())
    wire, system, accounting = reader._select_wire(operation, candidates, SCHEMA, output_token_cap=73)
    selected = 1 if expected == "review" else 0
    assert (wire, system) == candidates[selected]
    assert accounting["count"] == (review if selected else baseline)
    assert [counts.kind(r["payload"]) for r in counts.counted] == ["baseline", "review"]
    assert all(r["payload"]["max_tokens"] == 73 for r in counts.counted)
    assert candidates == unchanged
    assert state.public_dict() == facts and reader.obligation_snapshot() == obligations
    assert engine.ledger.used == {} and engine.reasoner.sequence == 0 and counts.sent == []


@pytest.mark.parametrize("flag", [True, None, "false", 0])
def test_nonactual_baseline_never_authorizes_comparison(flag):
    reader, _, _, counts, _ = setup(baseline={"count": 1000, "token_count_is_estimate": flag},
                                   review=AssertionError("must not count speculative candidate"))
    wire, system, result = reader._select_wire("joint_read", pair(), SCHEMA)
    assert (wire, system) == pair()[0]
    assert result["token_count_is_estimate"] == flag
    assert len(counts.counted) == 1


@pytest.mark.parametrize("accounting", [
    {"count": 1, "token_count_is_estimate": True}, {"count": 1},
    {"count": 1, "token_count_is_estimate": None},
])
def test_nonactual_candidate_cannot_win_even_when_its_number_is_smaller(accounting):
    reader, _, _, counts, _ = setup(review=accounting)
    wire, system, result = reader._select_wire("joint_read", pair(), SCHEMA)
    assert (wire, system) == pair()[0] and result["count"] == 1000
    assert len(counts.counted) == 2


def test_missing_actual_baseline_flag_and_single_candidate_keep_normal_accounting():
    reader, _, _, counts, _ = setup(baseline={"count": 13})
    assert reader._select_wire("joint_read", pair(), SCHEMA) == (*pair()[0], {"count": 13})
    counts.counted.clear()
    counts.values["baseline"] = 41
    assert reader._select_wire("joint_read", pair()[:1], SCHEMA)[2]["count"] == 41
    assert len(counts.counted) == 1


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("operation", OPERATIONS)
def test_selection_counts_full_instruction_schema_cap_then_sends_identical_payload(mode, operation):
    reader, engine, _, counts, data = setup(response_format=mode)
    candidates = pair()
    response = reader._request_with_exposure(operation, data, "independent-query", *candidates[0],
        wire_candidates=candidates, schema=SCHEMA, output_token_cap=73, reserve=0, extra_reserve=2)
    assert response == {"slots": []}
    # Both candidates plus the final Reasoner preflight count are retained.
    assert [counts.kind(r["payload"]) for r in counts.counted] == ["baseline", "review", "review"]
    assert counts.counted[1]["payload"] == counts.counted[2]["payload"] == counts.sent[0]["payload"]
    payload = counts.sent[0]["payload"]
    assert payload["max_tokens"] == 73
    assert json.loads(payload["messages"][1]["content"]) == candidates[1][0]
    if mode == "json_schema":
        assert payload["messages"][0]["content"] == candidates[1][1]
        assert payload["response_format"]["json_schema"]["schema"] == SCHEMA
    else:
        prompt = payload["messages"][0]["content"]
        prefix, encoded = prompt.split("\n\nReturn one complete JSON object matching this schema:\n")
        assert prefix == candidates[1][1] and json.loads(encoded) == SCHEMA
        assert payload.get("response_format") == ({"type": "json_object"} if mode == "json_object" else None)
    assert counts.sent[0]["options"] == {"reserve": 0, "extra_reserve": 2}
    assert engine.reasoner.requests[0]["input_tokens_local"] == 900
    assert engine.reasoner.requests[0]["output_token_reserve"] == 73
    assert engine.ledger.used["llm"] == 1
    record = reader.source_exposures[0]
    assert record["request_status"] == "returned" and record["logical_call_index"] == 1
    assert {s["source_alias"] for s in record["sources"]} == {"s1", "s2"}


@pytest.mark.parametrize("side", ["baseline", "review"])
@pytest.mark.parametrize("error_type", [ConnectionError, ValueError])
def test_counter_failures_propagate_without_send_but_after_preserving_all_exposure(side, error_type):
    reader, engine, state, counts, data = setup()
    error = error_type("independent count failure")
    counts.values[side] = error
    before = deepcopy(state.public_dict())
    def exposure_already_recorded():
        assert len(reader.source_exposures) == 1
        assert reader.unread_obligations == {
            ("independent-query", "s1"): {"history"}, ("independent-query", "s2"): {"history"}}
    counts.inspect_count = exposure_already_recorded
    with pytest.raises(error_type) as caught:
        reader._request_with_exposure("joint_read", data, "independent-query", *pair()[0],
                                      wire_candidates=pair(), schema=SCHEMA)
    assert caught.value is error
    assert counts.sent == [] and engine.reasoner.sequence == 0 and engine.ledger.used == {}
    record = reader.source_exposures[0]
    assert record["request_status"] == "not_sent" and record["error_type"] == error_type.__name__
    assert record["logical_call_index"] is None and state.public_dict() == before
    assert len(reader.unread_obligations) == 2


def test_transport_failure_retains_selected_count_and_original_exposure():
    reader, engine, _, counts, data = setup()
    error = ConnectionError("synthetic transport failure")
    counts.send_error = error
    with pytest.raises(ConnectionError) as caught:
        reader._request_with_exposure("joint_read", data, "independent-query", *pair()[0],
                                      wire_candidates=pair(), schema=SCHEMA)
    assert caught.value is error
    assert counts.sent[0]["payload"] == counts.counted[1]["payload"] == counts.counted[2]["payload"]
    assert engine.ledger.used["llm"] == 1 and engine.reasoner.sequence == 1
    assert reader.source_exposures[0]["request_status"] == "failed"
    assert reader.source_exposures[0]["logical_call_index"] == 1
    assert len(reader.unread_obligations) == 2


@pytest.mark.parametrize("difference,expected", [(0, True), (1, False), (-1, True)])
def test_baseline_fitting_boundary_cannot_regress_to_more_expensive_review(difference, expected, monkeypatch):
    reader, _, _, _, data = setup(baseline=1000 + difference, review=1500,
                                 context_tokens=1000 + 512 + 256 + 8)
    monkeypatch.setattr(reader, "_wire_candidates", lambda data: pair())
    assert reader._estimate("joint_read", data) == 1000 + difference
    assert reader._fits("joint_read", data) is expected


def test_fitting_baseline_is_actually_sent_when_review_would_overflow():
    reader, engine, _, counts, data = setup(baseline=1000, review=1001,
        context_tokens=1000 + 73 + 256 + 8)
    reader._request_with_exposure("joint_read", data, "independent-query", *pair()[0],
        wire_candidates=pair(), schema=SCHEMA, output_token_cap=73)
    assert counts.kind(counts.sent[0]["payload"]) == "baseline"
    assert engine.reasoner.requests[0]["input_tokens_local"] == 1000


def test_final_preflight_is_not_bypassed_when_both_representations_overflow():
    reader, engine, _, counts, data = setup(baseline=1001, review=1002,
        context_tokens=1000 + 73 + 256 + 8)
    with pytest.raises(InputOverflow):
        reader._request_with_exposure("joint_read", data, "independent-query", *pair()[0],
            wire_candidates=pair(), schema=SCHEMA, output_token_cap=73)
    assert counts.sent == [] and engine.ledger.used == {}
    assert reader.source_exposures[0]["request_status"] == "not_sent"
    assert len(reader.unread_obligations) == 2


def review_data(data):
    data = deepcopy(data)
    value = "Unicode 原文 e\u0301 and quoted literal text. " * 300
    draft = observation(value=value, reason=value)
    data["repair_scope"] = {"failed": [{"variable": "history", "id": draft["id"],
        "feedback": "preserve this draft", "observation": draft,
        "slot": {"variable": "history", "observations": [deepcopy(draft)], "unresolved": []}}]}
    return data


def test_pure_candidates_do_not_count_and_review_instruction_is_candidate_specific(monkeypatch):
    reader, _, _, counts, data = setup()
    data = review_data(data)
    original = deepcopy(data)
    counts.values = {k: AssertionError("pure preparation must not count") for k in counts.values}
    candidates = reader._wire_candidates(data)
    assert len(candidates) == 2
    baseline, proposed = candidates
    assert "shared_review_texts" not in baseline[0] and "shared_review_texts" in proposed[0]
    assert REVIEW_INSTRUCTION not in baseline[1] and REVIEW_INSTRUCTION in proposed[1]
    assert reader._prepare_wire(data) == proposed
    assert data == original and counts.counted == []


@pytest.mark.parametrize("review_count,expected", [(900, "review"), (1000, "baseline"), (1100, "baseline")])
def test_real_codec_candidates_use_measured_selection_not_byte_proposal(review_count, expected):
    reader, _, _, counts, data = setup(review=review_count)
    data = review_data(data)
    candidates = reader._wire_candidates(data)
    assert len(candidates) == 2
    assert reader._estimate("joint_read_repair", data, output_token_cap=91) == min(1000, review_count)
    counts.counted.clear()
    reader._request_with_exposure("joint_read_repair", data, "independent-query", *candidates[0],
        wire_candidates=candidates, schema=reader._schema(data), output_token_cap=91)
    selected = counts.sent[0]["payload"]
    assert counts.kind(selected) == expected
    assert selected == counts.counted[0 if expected == "baseline" else 1]["payload"]
    assert (REVIEW_INSTRUCTION in selected["messages"][0]["content"]) is (expected == "review")


@pytest.mark.parametrize("review_count", [900, 1100])
def test_existing_history_representation_survives_either_review_choice(review_count):
    reader, _, _, counts, data = setup(review=review_count)
    data = review_data(data)
    history = "Previously grounded parent history with original scope. " * 200
    data["supported_parents"] = {"parent": {"type": "text", "value": history,
        "version": 1, "source_ids": ["s1"], "demand": "parent"}}
    data["grounded_subquestion"] = "Established parent: " + history + "\nRetained: " + history
    candidates = reader._wire_candidates(data)
    assert len(candidates) == 2
    for wire, system in candidates:
        assert list(wire["shared_history_texts"].values()) == [history]
        assert wire["supported_parents"]["parent"]["value_ref"] in wire["shared_history_texts"]
        assert HISTORY_INSTRUCTION in system
    reader._request_with_exposure("joint_read_repair", data, "independent-query", *candidates[0],
        wire_candidates=candidates, schema=reader._schema(data))
    payload = counts.sent[0]["payload"]
    selected = json.loads(payload["messages"][1]["content"])
    assert list(selected["shared_history_texts"].values()) == [history]
    assert HISTORY_INSTRUCTION in payload["messages"][0]["content"]
    assert (REVIEW_INSTRUCTION in payload["messages"][0]["content"]) is (review_count < 1000)


def test_invalid_review_reference_candidate_falls_back_without_decoding_authority():
    reader, _, _, counts, data = setup()
    data = review_data(data)
    data["repair_scope"]["failed"][0]["observation"]["review_value_ref"] = "caller-supplied"
    before = deepcopy(data)
    candidates = reader._wire_candidates(data)
    assert len(candidates) == 1 and REVIEW_INSTRUCTION not in candidates[0][1]
    assert "shared_review_texts" not in candidates[0][0]
    selected = reader._select_wire("joint_read_repair", candidates, reader._schema(data))
    assert selected[:2] == candidates[0] and len(counts.counted) == 1
    assert data == before


def install_two_candidate_route_spy(reader, monkeypatch):
    """Keep source payload complete; replace only representation generation."""
    original_candidates = reader._wire_candidates
    wrapper = reader._request_with_exposure
    forwarded = []
    def candidates(data):
        baseline, system = original_candidates(data)[0]
        return (({**baseline, "candidate": "baseline"}, system),
                ({**deepcopy(baseline), "candidate": "review"}, system + "\nsynthetic alternative"))
    def exposed(*args, **kwargs):
        assert "wire_candidates" in kwargs and len(kwargs["wire_candidates"]) == 2
        forwarded.append((args[0], deepcopy(kwargs["wire_candidates"])))
        return wrapper(*args, **kwargs)
    monkeypatch.setattr(reader, "_wire_candidates", candidates)
    monkeypatch.setattr(reader, "_request_with_exposure", exposed)
    return forwarded


def test_main_read_production_callsite_forwards_candidates(monkeypatch):
    response = {"slots": [{"variable": "history", "observations": [observation()], "unresolved": []}]}
    reader, engine, _, counts, _ = setup(response=response)
    forwarded = install_two_candidate_route_spy(reader, monkeypatch)
    reader.read(engine.steps[0], "Read past creative activities", ["history"], ["piano"], "independent-query")
    assert forwarded and forwarded[0][0] == "joint_read"
    assert len(counts.sent) == 1 and counts.kind(counts.sent[0]["payload"]) == "review"
    assert engine.ledger.used["llm"] == 1


def test_terminal_repair_production_callsite_forwards_candidates(monkeypatch):
    reader, engine, _, counts, data = setup(response={"slots": [], "semantic_answer": "synthetic", "final_prediction": "a"})
    forwarded = install_two_candidate_route_spy(reader, monkeypatch)
    checks = iter([("synthetic", "format repair required"), ("synthetic", None)])
    monkeypatch.setattr(reader, "_terminal_check", lambda *args: next(checks))
    monkeypatch.setattr(reader, "_repair_fits", lambda *args: True)
    result = reader._repair_terminal("semantic_compose", data, [], None, None,
                                    engine.steps[1], "independent-query", 0)
    assert result == ("a", "synthetic", None)
    assert len(forwarded) == 1 and forwarded[0][0] == "semantic_compose_repair"
    assert len(counts.sent) == 1 and counts.kind(counts.sent[0]["payload"]) == "review"
    assert engine.ledger.used["json_repairs"] == 1 and engine.ledger.used["llm"] == 1
