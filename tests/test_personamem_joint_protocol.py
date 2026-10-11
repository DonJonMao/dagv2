"""Offline regressions for PersonaMem source namespaces and scoped repairs.

The reader, fact state, source validation, request serialization, and token
accounting are real. Only model responses are supplied by a scripted transport.
"""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from dagbt.budget import Ledger
from dagbt.evidence_spans import build_source_spans
from dagbt.fact_state import FactState
from dagbt.joint_reading import JointReader
from dagbt.reasoning import ProtocolError, Reasoner
from dagbt.transport import digest


def observation(identifier="history-1", sources=("s1",), value=None, **changes):
    return {
        "id": identifier,
        "stance": "support",
        "value": ["piano"] if value is None else value,
        "source_ids": list(sources),
        "entity_scope": "user",
        "time_scope": "history",
        "parent_versions": {},
        "retract_ids": [],
        "reason": "The user describes the activity in the original source.",
        **changes,
    }


def response(*rows):
    return {"slots": [{"variable": "history", "observations": list(rows), "unresolved": []}]}


class ScriptedCalls:
    """No network; preserve request identity and the production response shape."""

    def __init__(self, ledger, respond):
        self.ledger, self.respond = ledger, respond
        self.requests = []

    def get(self, stage, url, payload, **kwargs):
        data = json.loads(payload["messages"][1]["content"])
        self.requests.append({"stage": stage, "payload": deepcopy(payload), "data": data})
        self.ledger.reserve("llm", "/".join(stage))
        raw = self.respond(len(self.requests), data)
        return {
            "response_ref": digest(payload),
            "response": {"choices": [{"finish_reason": "stop", "message": {
                "content": json.dumps(raw),
            }}]},
        }


def fixture(respond=None, **settings):
    steps = [
        {"output_slot": "history", "question": "Read past creative activities",
         "inputs": [], "execution": "retrieval", "answer_type": "fact"},
        {"output_slot": "suggestion", "question": "Suggest from {history}",
         "inputs": ["history"], "execution": "compose", "answer_type": "fact"},
    ]
    program = {
        "variables": {"history": {"type": "set", "demand": "history"},
                      "suggestion": {"type": "text", "demand": "suggestion"}},
        "expression": {"op": "var", "id": "suggestion"},
    }
    contract = {"mode": "semantic", "result_type": "text", "entity_scope": "user",
                "time_scope": "history", "explanation": "semantic"}
    state = FactState(program, contract, steps, "suggestion")
    documents = {}
    sources, source_aliases = {}, {}
    for identifier, text, role in (
        ("piano", "I played the piano last year.", "user"),
        ("painting", "I enjoyed painting last spring.", "user"),
        ("advice", "You might enjoy guitar lessons.", "assistant"),
    ):
        documents[identifier] = SimpleNamespace(doc_id=identifier, passage=text, metadata={
            "source_segments": [{"role": role, "start": 0, "end": len(text),
                                 "source_message_indices": [len(documents)]}],
        })
        for span in build_source_spans(identifier, documents[identifier], 400):
            alias = "s" + str(len(sources) + 1)
            source_aliases[span["id"]] = alias
            sources[alias] = {**span, "id": alias, "source_id": span["id"]}
    events = []
    s = {"algorithm_version": "dagbt_residual_memory_v1", "max_quote_chars": 400,
         "reasoning_output_tokens": 512, "input_margin": 256, "context_tokens": 16384,
         "response_format": "plain", "max_repairs_per_request": 2,
         "reserved_audit_calls": 1, **settings}
    ledger = Ledger({"llm": 24, "json_repairs": 6})
    calls = ScriptedCalls(ledger, respond or (lambda index, data: response(observation())))
    config = {"llm_base_url": "http://offline.invalid/v1", "llm_model": "scripted", "fusion": s}
    reasoner = Reasoner(calls, None, config, s, ledger, events.append)
    engine = SimpleNamespace(
        q={"question": "Suggest a creative weekend activity."}, steps=steps,
        docs=documents, mapper=SimpleNamespace(sources=sources, source_aliases=source_aliases),
        s=s, ledger=ledger, reasoner=reasoner, event=events.append, events=events,
        errors=[], spans={}, reader_question="Suggest a creative weekend activity.", output_options=None,
    )
    return JointReader(engine, state), engine, state, calls


def read(reader):
    return reader.read(reader.e.steps[0], "Read past creative activities", ["history"],
                       ["piano"], "history-query")


def visible(reader, aliases=("s1",)):
    data = reader._payload(reader.e.steps[0], "Read past creative activities", ["history"], aliases)
    return {source["id"]: source for source in data["source_spans"]}


def test_wire_uses_one_source_namespace_and_preserves_internal_provenance():
    reader, engine, state, _ = fixture()
    source_id = engine.mapper.sources["s1"]["source_id"]
    other_source = engine.mapper.sources["s2"]["source_id"]
    state.integrate([{**observation(sources=[source_id]), "variable": "history"}])
    old = deepcopy(next(iter(state.routes.values())))
    old["id"] = "old-history-route"
    old["revoked"] = [{"source_ids": [other_source], "reason": "Historical correction"}]
    state.routes[old["id"]] = old
    certificate = {"source_ids": [source_id], "evidence": {"source_ids": [other_source]}}
    engine.residual_control = SimpleNamespace(
        rules_completion_needed=lambda: False,
        task_projection=lambda task: certificate if task == "history" else None,
    )
    before = deepcopy(state.public_dict())
    data = reader._payload(engine.steps[1], "Suggest from piano", ["history", "suggestion"], ["s2"])
    assert all("source_id" not in source for source in data["source_spans"])
    assert {source["id"] for source in data["source_spans"]} == {"s1", "s2"}
    assert data["supported_parents"]["history"]["source_ids"] == ["s1"]
    assert all(row["source_ids"] == ["s1"] for row in data["known_observations"])
    old_wire = next(row for row in data["known_observations"] if row["id"] == old["id"])
    assert old_wire["revoked"][0]["source_ids"] == ["s2"]
    assert data["supported_task_outputs"]["history"] == {
        "source_ids": ["s1"], "evidence": {"source_ids": ["s2"]},
    }
    assert state.public_dict() == before
    assert certificate["source_ids"] == [source_id]
    assert certificate["evidence"]["source_ids"] == [other_source]


def test_typed_values_and_outputs_with_source_ids_fields_are_not_provenance():
    reader, engine, state, _ = fixture()
    stable = engine.mapper.sources["s1"]["source_id"]
    state.program["variables"]["history"]["type"] = "record"
    value = {"source_ids": ["application-record-id"], "nested": {"source_ids": [stable]}}
    state.integrate([{**observation(sources=[stable], value=value), "variable": "history"}])
    certificate = {"source_ids": [stable], "output": deepcopy(value)}
    engine.residual_control = SimpleNamespace(
        rules_completion_needed=lambda: False,
        task_projection=lambda task: certificate if task == "history" else None,
    )
    data = reader._payload(engine.steps[1], "Suggest from historical record",
                           ["history", "suggestion"], ["s1"])
    assert data["known_observations"][0]["value"] == value
    assert data["supported_parents"]["history"]["value"] == value
    assert data["supported_task_outputs"]["history"]["output"] == value
    assert data["supported_task_outputs"]["history"]["source_ids"] == ["s1"]
    assert state.valid()[0]["history"] == value


def expand_document_table(data):
    """Test-side inverse: source metadata sharing must be exactly reversible."""
    restored = deepcopy(data)
    documents = restored.pop("source_documents")
    for source in restored["source_spans"]:
        metadata = documents[source["doc_id"]]
        source["doc_id"] = metadata["doc_id"]
        source["time_metadata"] = deepcopy(metadata["time_metadata"])
    return restored


def test_document_wire_table_roundtrips_all_source_fields_without_mutation():
    reader, engine, _, _ = fixture()
    data = reader._payload(engine.steps[0], "Read past creative activities", ["history"], ["s1", "s2", "s3"])
    data["source_spans"][0]["text"] = '  原文 "source_ids": ["d1"]\nnext line  '
    data["source_spans"][0]["source_role"] = "unknown"
    data["source_spans"][0]["source_message_indices"] = []
    for source in data["source_spans"]:
        source["doc_id"] = "personamem:memory:" + digest(source["doc_id"])
        source["time_metadata"] = {"validity": "unknown", "event_start": None, "event_end": None,
                                   "observed_start": 30, "observed_end": 32, "time_source": "message_index"}
    copies = [{**deepcopy(source), "id": "s" + str(index + 4), "start": 31, "end": 50,
               "text": "More unchanged text."} for index, source in enumerate(data["source_spans"])]
    different_time = {**deepcopy(copies[0]), "id": "s7", "time_metadata": {"validity": "known", "event_start": 42}}
    data["source_spans"].extend([*copies, different_time])
    before = deepcopy(data)
    wire = reader._wire_data(data)
    assert data == before and wire is not data
    assert expand_document_table(wire) == before
    assert all(set(document) == {"doc_id", "time_metadata"} for document in wire["source_documents"].values())
    assert all("time_metadata" not in source for source in wire["source_spans"])
    assert all(source["doc_id"].startswith("d") and source["doc_id"][1:].isdigit()
               for source in wire["source_spans"])
    assert wire["source_spans"][0]["doc_id"] == wire["source_spans"][3]["doc_id"]
    assert wire["source_spans"][0]["doc_id"] != wire["source_spans"][6]["doc_id"]
    assert len(wire["source_documents"]) == 4
    for original, encoded in zip(before["source_spans"], wire["source_spans"]):
        assert {key: value for key, value in encoded.items() if key != "doc_id"} == {
            key: value for key, value in original.items() if key not in {"doc_id", "time_metadata"}
        }
    first_document = wire["source_spans"][0]["doc_id"]
    wire["source_documents"][first_document]["time_metadata"]["event_start"] = 99
    wire["source_spans"][0]["source_message_indices"].append(99)
    wire["variables"]["history"]["type"] = "changed"
    assert data == before


def test_small_source_payload_keeps_original_representation_as_an_independent_copy():
    reader, engine, _, _ = fixture()
    data = reader._payload(engine.steps[0], "Read past creative activities", ["history"], ["s1"])
    wire = reader._wire_data(data)
    assert "source_documents" not in wire
    assert wire == data and wire is not data
    assert wire["source_spans"] is not data["source_spans"]
    wire["source_spans"][0]["time_metadata"]["event_start"] = 17
    assert data["source_spans"][0]["time_metadata"] == {}


def test_read_and_repair_estimate_the_same_document_table_sent_to_the_model(monkeypatch):
    outputs = [response(observation("repair-me", ["missing"])), response(observation("repair-me", ["s1"]))]
    reader, engine, state, calls = fixture(lambda index, data: deepcopy(outputs[index - 1]))
    document_id = "personamem:memory:" + "9" * 64
    passage = "I played the piano last year. " * 30
    time_metadata = {"validity": "unknown", "event_start": None, "event_end": None,
                     "observed_start": 30, "observed_end": 32, "time_source": "message_index"}
    document = SimpleNamespace(doc_id=document_id, passage=passage, metadata={
        "source_segments": [{"role": "user", "start": 0, "end": len(passage), "source_message_indices": [30]}],
        "time": time_metadata,
    })
    engine.docs = {document_id: document}
    engine.mapper.sources = {}
    engine.mapper.source_aliases = {}
    for index, span in enumerate(build_source_spans(document_id, document, 400), 1):
        alias = "s" + str(index)
        engine.mapper.sources[alias] = {**span, "id": alias, "source_id": span["id"]}
        engine.mapper.source_aliases[span["id"]] = alias
    canonical_sources = deepcopy(engine.mapper.sources)
    estimates = []
    estimate = engine.reasoner.estimate

    def inspect_estimate(operation, system, data, schema=None):
        assert data["source_documents"]
        assert all(source["doc_id"] in data["source_documents"] and "time_metadata" not in source
                   for source in data["source_spans"])
        count = estimate(operation, system, data, schema)
        estimates.append((digest(data), count))
        return count

    monkeypatch.setattr(engine.reasoner, "estimate", inspect_estimate)
    assert reader.read(engine.steps[0], "Read past creative activities", ["history"], [document_id], "history-query")
    assert len(calls.requests) == 2
    for sent, recorded in zip(calls.requests, engine.reasoner.requests):
        data = sent["data"]
        assert data["source_documents"] == {"d1": {"doc_id": document_id, "time_metadata": time_metadata}}
        assert data["source_spans"][0]["doc_id"] == "d1"
        assert data["source_spans"][0]["id"] == "s1"
        assert expand_document_table(data)["source_spans"][0]["text"] == canonical_sources["s1"]["text"]
        assert (digest(data), recorded["input_tokens_local"]) in estimates
        assert recorded["input_tokens_local"] == estimate(
            sent["stage"][0], reader._system(data), data, reader._schema(data))
    assert calls.requests[1]["data"]["repair_scope"]["failed"][0]["observation"]["source_ids"] == ["missing"]
    assert engine.mapper.sources == canonical_sources
    assert state.valid()[1]["history"] == [canonical_sources["s1"]["source_id"]]
    system = calls.requests[0]["payload"]["messages"][0]["content"]
    assert "source_documents" in system and "source_ids" in system
    with pytest.raises(ProtocolError):
        reader._observation(observation(sources=["d1"]), "history", visible(reader), engine.steps[0])


@pytest.mark.parametrize("namespace", ["alias", "stable"])
def test_visible_exact_source_identity_is_grounded_to_unchanged_original(namespace):
    reader, engine, _, _ = fixture()
    stable = engine.mapper.sources["s1"]["source_id"]
    row = observation(sources=["s1" if namespace == "alias" else stable])
    validated = reader._observation(row, "history", visible(reader), engine.steps[0])
    assert validated["source_ids"] == [stable]
    assert engine.spans[stable]["source_role"] == "user"
    assert engine.spans[stable]["exact_quote"] == engine.docs["piano"].passage
    assert row["source_ids"] == ["s1" if namespace == "alias" else stable]


@pytest.mark.parametrize("case", ["guessed_numeric", "invisible_stable", "canonical_duplicate", "ambiguous"])
def test_source_compatibility_never_guesses_or_expands_visible_scope(case):
    reader, engine, _, _ = fixture()
    stable = engine.mapper.sources["s1"]["source_id"]
    shown = visible(reader)
    if case == "guessed_numeric":
        # The tempting guessed target really is visible: src_44 must still fail.
        engine.mapper.sources["s44"] = {**deepcopy(engine.mapper.sources["s1"]), "id": "s44"}
        shown = {"s44": {**deepcopy(shown["s1"]), "id": "s44"}}
        identifiers = ["src_44"]
    elif case == "invisible_stable":
        identifiers = [engine.mapper.sources["s2"]["source_id"]]
    elif case == "canonical_duplicate":
        identifiers = ["s1", stable]
    else:
        engine.mapper.sources["s99"] = {**deepcopy(engine.mapper.sources["s1"]), "id": "s99"}
        shown["s99"] = {**deepcopy(shown["s1"]), "id": "s99"}
        identifiers = [stable]
    with pytest.raises(ProtocolError):
        reader._observation(observation(sources=identifiers), "history", shown, engine.steps[0])
    assert not engine.spans


@pytest.mark.parametrize("namespace", ["alias", "stable"])
def test_compatibility_cannot_bypass_authoritative_role_validation(namespace):
    reader, engine, _, _ = fixture()
    shown = visible(reader)
    stable = engine.mapper.sources["s1"]["source_id"]
    engine.mapper.sources["s1"]["source_role"] = "assistant"
    with pytest.raises(ProtocolError, match="authoritative role"):
        reader._observation(observation(sources=["s1" if namespace == "alias" else stable]),
                            "history", shown, engine.steps[0])
    assert not engine.spans


def test_exact_assistant_source_keeps_its_authoritative_role():
    reader, engine, _, _ = fixture()
    stable = engine.mapper.sources["s3"]["source_id"]
    row = observation(sources=[stable], stance="partial", value=None)
    row["value"] = None
    reader._observation(row, "history", visible(reader, ["s3"]), engine.steps[0])
    assert engine.spans[stable]["source_role"] == "assistant"
    assert "assistant" in reader._system({})


def test_repair_preserves_failed_row_context_and_stops_unchanged_semantic_retry():
    bad = observation("failed-counter", ["unavailable-source"])
    reader, engine, state, calls = fixture(lambda index, data: response(deepcopy(bad)))
    assert read(reader)
    assert len(calls.requests) == 2  # The identical failed response cannot buy another sample.
    repair = calls.requests[1]["data"]["repair_scope"]
    assert repair["attempt"] == 1
    assert repair["previous_response_identity"]
    assert repair["failed"][0]["observation"] == bad
    assert repair["failed"][0]["id"] == bad["id"]
    assert engine.ledger.used["json_repairs"] == 1
    assert state.pending == {"history"}
    assert any(event["event"] == "joint_repair_unchanged" for event in engine.events)


def test_changed_failed_observation_can_repair_again_and_validated_row_is_frozen():
    good = observation("accepted", ["s1"])
    bad1 = observation("repair-me", ["unavailable-first"])
    bad2 = observation("repair-me", ["unavailable-second"])
    changed_good = observation("accepted", ["s1"], ["invented new activity"])
    outputs = [response(good, bad1), response(changed_good, bad2),
               response(changed_good, observation("repair-me", ["s1"]))]
    reader, engine, state, calls = fixture(lambda index, data: deepcopy(outputs[index - 1]))
    assert read(reader)
    assert len(calls.requests) == 3
    repairs = [request["data"]["repair_scope"] for request in calls.requests[1:]]
    assert [repair["attempt"] for repair in repairs] == [1, 2]
    assert repairs[0]["previous_response_identity"] != repairs[1]["previous_response_identity"]
    assert [repair["failed"][0]["observation"] for repair in repairs] == [bad1, bad2]
    assert all(row["id"] != "accepted" for repair in repairs for row in repair["failed"])
    assert engine.ledger.used["json_repairs"] == 2
    assert state.valid()[0]["history"] == ["piano"]
    assert not state.pending and reader.batches[-1]["complete"]
    assert all(route["value"] == ["piano"] for route in state.routes.values())


def test_missing_failed_observation_is_explicit_null_in_repair_context():
    outputs = [{"slots": []}, response(observation())]
    reader, _, state, calls = fixture(lambda index, data: deepcopy(outputs[index - 1]))
    assert read(reader)
    failure = calls.requests[1]["data"]["repair_scope"]["failed"][0]
    assert failure["id"] == "*" and failure["observation"] is None
    assert state.valid()[0]["history"] == ["piano"]


@pytest.mark.parametrize("repair_verdict", ["supported", None])
def test_audit_repair_keeps_original_instruction_and_requires_explicit_verdict(repair_verdict):
    first = {**response(observation(sources=["unavailable-source"])), "audit_verdict": "unknown"}
    repaired = response(observation())
    if repair_verdict is not None:
        repaired["audit_verdict"] = repair_verdict
    outputs = [first, repaired]
    reader, engine, _, calls = fixture(lambda index, data: deepcopy(outputs[index - 1]),
                                      max_repairs_per_request=1)
    assert reader.read(engine.steps[0], "Read past creative activities", ["history"],
                       ["piano"], "audit-history", audit=True)
    assert [request["stage"][0] for request in calls.requests] == ["audit_joint", "audit_joint_repair"]
    original_data, repair_data = [request["data"] for request in calls.requests]
    assert original_data["audit_instruction"]
    assert repair_data["audit_instruction"] == original_data["audit_instruction"]
    assert repair_data["source_spans"] == original_data["source_spans"]
    assert reader.batches[-1]["audit_complete"] is (repair_verdict == "supported")
    if repair_verdict is None:
        assert any("Audit must return an explicit verdict" in failure["error"]
                   for failure in reader.batches[-1]["failed_slots"])


@pytest.mark.parametrize("intermediate_failure", ["missing_slot", "malformed_slots", "missing_audit_verdict"])
def test_specific_failed_counter_survives_missing_slot_and_global_protocol_repairs(intermediate_failure):
    reader, engine, state, calls = fixture()
    stable = engine.mapper.sources["s1"]["source_id"]
    state.integrate([{**observation(sources=[stable]), "variable": "history"}])
    route_id = next(iter(state.routes))
    counter = observation("counter-required", ["s2"], stance="contradiction",
                          entity_scope="different-user", retract_ids=[route_id])
    counter["value"] = None
    first = {**response(counter), "audit_verdict": "supported"}
    middle = {
        "missing_slot": {"slots": [], "audit_verdict": "supported"},
        "malformed_slots": {"slots": "not-a-list", "audit_verdict": "supported"},
        "missing_audit_verdict": response(),
    }[intermediate_failure]
    outputs = [first, middle, {**response(), "audit_verdict": "supported"}]
    calls.respond = lambda index, data: deepcopy(outputs[index - 1])
    assert reader.read(engine.steps[0], "Read past creative activities", ["history"],
                       ["piano", "painting"], "counter-audit", audit=True)
    assert len(calls.requests) == 3
    final_repair = calls.requests[-1]["data"]["repair_scope"]["failed"]
    preserved = [item for item in final_repair if item["id"] == "counter-required"]
    assert preserved, "A missing/malformed repair envelope must not erase a concrete counter obligation"
    assert preserved[0]["observation"] == counter
    assert state.pending == {"history"} and reader.failed_obligations
    assert not reader.batches[-1]["complete"] and not reader.batches[-1]["audit_complete"]
    assert any(item["id"] == "counter-required" for item in reader.batches[-1]["failed_slots"])


@pytest.mark.parametrize("order", ["support_first", "counter_first"])
def test_duplicate_observation_id_cannot_hide_counter_as_an_already_frozen_row(order):
    reader, engine, state, calls = fixture(max_repairs_per_request=0)
    stable = engine.mapper.sources["s1"]["source_id"]
    state.integrate([{**observation(sources=[stable]), "variable": "history"}])
    route_id = next(iter(state.routes))
    support = observation("duplicate-id", ["s1"])
    counter = observation("duplicate-id", ["s2"], stance="contradiction", retract_ids=[route_id])
    counter["value"] = None
    rows = [support, counter] if order == "support_first" else [counter, support]
    calls.respond = lambda index, data: {**response(*deepcopy(rows)), "audit_verdict": "supported"}
    assert reader.read(engine.steps[0], "Read past creative activities", ["history"],
                       ["piano", "painting"], "duplicate-audit", audit=True)
    failures = reader.batches[-1]["failed_slots"]
    assert any(item["id"] == "duplicate-id" and "duplicate" in item["error"].lower()
               for item in failures), "Both occurrences must be checked before freezing an ID"
    assert state.pending == {"history"} and reader.failed_obligations
    assert not reader.batches[-1]["complete"] and not reader.batches[-1]["audit_complete"]


@pytest.mark.parametrize("order", ["support_first", "counter_first"])
def test_duplicate_conflict_repair_retains_both_drafts_and_cannot_drop_counter(order):
    reader, engine, state, calls = fixture(max_repairs_per_request=1)
    stable = engine.mapper.sources["s1"]["source_id"]
    state.integrate([{**observation(sources=[stable]), "variable": "history"}])
    route_id = next(iter(state.routes))
    support = observation("conflicting-id", ["s1"])
    counter = observation("conflicting-id", ["s2"], stance="contradiction", retract_ids=[route_id])
    counter["value"] = None
    group = [support, counter] if order == "support_first" else [counter, support]
    outputs = [{**response(*group), "audit_verdict": "supported"},
               {**response(support), "audit_verdict": "supported"}]
    calls.respond = lambda index, data: deepcopy(outputs[index - 1])
    assert reader.read(engine.steps[0], "Read past creative activities", ["history"],
                       ["piano", "painting"], "duplicate-repair-audit", audit=True)
    assert len(calls.requests) == 2
    failed = calls.requests[1]["data"]["repair_scope"]["failed"]
    conflict = next(item for item in failed if item["id"] == "conflicting-id")
    assert conflict["observations"] == group
    assert state.pending == {"history"} and reader.failed_obligations
    assert not reader.batches[-1]["complete"] and not reader.batches[-1]["audit_complete"]
    assert any(item["id"] == "conflicting-id" for item in reader.batches[-1]["failed_slots"])


def test_undeclared_top_level_field_cannot_hide_identifiable_counter_from_repair():
    reader, engine, state, calls = fixture(max_repairs_per_request=1)
    stable = engine.mapper.sources["s1"]["source_id"]
    state.integrate([{**observation(sources=[stable]), "variable": "history"}])
    route_id = next(iter(state.routes))
    counter = observation("counter-in-invalid-envelope", ["s2"], stance="contradiction",
                          entity_scope="different-user", retract_ids=[route_id])
    counter["value"] = None
    outputs = [{**response(counter), "audit_verdict": "supported", "undeclared_trace": "extra field"},
               {**response(), "audit_verdict": "supported"}]
    calls.respond = lambda index, data: deepcopy(outputs[index - 1])
    assert reader.read(engine.steps[0], "Read past creative activities", ["history"],
                       ["piano", "painting"], "invalid-envelope-audit", audit=True)
    assert len(calls.requests) == 2
    failed = calls.requests[1]["data"]["repair_scope"]["failed"]
    concrete = [item for item in failed if item["id"] == "counter-in-invalid-envelope"]
    assert concrete and concrete[0]["observation"] == counter
    assert any(item["id"] == "*" and "undeclared" in item["feedback"].lower() for item in failed)
    assert state.pending == {"history"} and reader.failed_obligations
    assert not reader.batches[-1]["complete"] and not reader.batches[-1]["audit_complete"]
    assert any(item["id"] == "counter-in-invalid-envelope" for item in reader.batches[-1]["failed_slots"])


def test_repair_must_not_reuse_stable_source_visible_only_to_the_initial_request():
    reader, engine, state, calls = fixture(max_repairs_per_request=1)
    declaration = {"type": "set", "demand": "history"}
    state.program["variables"]["extra"] = declaration
    state.initial_program["variables"]["extra"] = deepcopy(declaration)
    state.versions["extra"] = 0
    history_source = engine.mapper.sources["s1"]["source_id"]
    extra_source = engine.mapper.sources["s2"]["source_id"]
    state.integrate([
        {**observation(sources=[history_source]), "variable": "history"},
        {**observation(sources=[extra_source], value=["painting"]), "variable": "extra"},
    ])
    outputs = [
        {"slots": [
            {"variable": "history", "observations": [observation("history-ok", ["s1"])], "unresolved": []},
            {"variable": "extra", "observations": [observation("extra-bad", ["missing"], ["painting"])], "unresolved": []},
        ]},
        {"slots": [{"variable": "extra", "observations": [
            observation("extra-bad", [history_source], ["painting"]),
        ], "unresolved": []}]},
    ]
    calls.respond = lambda index, data: deepcopy(outputs[index - 1])
    assert reader.read(engine.steps[0], "Read past creative activities", ["history", "extra"],
                       ["advice"], "narrow-repair")
    initial_data, repair_data = [request["data"] for request in calls.requests]
    assert {source["id"] for source in initial_data["source_spans"]} == {"s1", "s2", "s3"}
    assert {source["id"] for source in repair_data["source_spans"]} == {"s2", "s3"}
    assert "extra" in state.pending
    assert any(item["id"] == "extra-bad" and "invisible" in item["error"].lower()
               for item in reader.batches[-1]["failed_slots"])
    assert not reader.batches[-1]["complete"]


@pytest.mark.parametrize("repair_slot", ["history", "suggestion"])
def test_parent_slot_repair_schema_and_instructions_follow_requested_slots(repair_slot):
    reader, engine, state, calls = fixture(max_repairs_per_request=1)
    stable = engine.mapper.sources["s1"]["source_id"]
    state.integrate([{**observation(sources=[stable]), "variable": "history"}])
    suggestion = observation("suggestion-ok", ["s1"], "Play piano",
                             parent_versions={"history": state.versions["history"]})
    outputs = [
        {"slots": [{"variable": "suggestion", "observations": [suggestion], "unresolved": []}]},
        {"slots": [{"variable": repair_slot, "observations": [], "unresolved": []}]},
    ]
    calls.respond = lambda index, data: deepcopy(outputs[index - 1])
    assert reader.read(engine.steps[1], "Suggest from piano", ["suggestion"], ["piano"], "parent-slot-repair")
    assert len(calls.requests) == 2
    initial_data, repair_data = [request["data"] for request in calls.requests]
    assert set(initial_data["requested_slots"]) == {"history", "suggestion"}
    assert repair_data["requested_slots"] == ["history"]
    assert repair_data["task"]["output_slot"] == "suggestion"
    for request in calls.requests:
        schema = json.loads(request["payload"]["messages"][0]["content"].rsplit(
            "Return one complete JSON object matching this schema:\n", 1)[1])
        assert schema["properties"]["slots"]["items"]["properties"]["variable"]["enum"] == request["data"]["requested_slots"]
    repair_system = calls.requests[1]["payload"]["messages"][0]["content"]
    assert "requested_slots" in repair_system and "background" in repair_system.lower()
    if repair_slot == "history":
        assert reader.batches[-1]["complete"] and not state.pending
        assert state.valid()[0]["suggestion"] == "Play piano"
    else:
        assert not reader.batches[-1]["complete"] and "history" in state.pending
        assert any("unrequested" in item["error"].lower() for item in reader.batches[-1]["failed_slots"])


@pytest.mark.parametrize("options", [None, ["Piano", "Painting"], ["Piano", "Painting", "Guitar", "Pottery"]])
def test_four_option_terminal_schema_constrains_only_prediction_labels(options):
    reader, engine, _, calls = fixture()
    engine.output_options = options
    data = reader._payload(engine.steps[0], "Read past creative activities", ["history"], ["s1"], semantic=True)
    system = reader._system(data)
    wire = engine.reasoner._payload("semantic_compose", system, reader._wire_data(data), reader._schema(data))
    schema = json.loads(wire["messages"][0]["content"].rsplit(
        "Return one complete JSON object matching this schema:\n", 1)[1])
    prediction = schema["properties"]["final_prediction"]
    assert prediction["type"] == ["string", "null"]
    if options is not None and len(options) == 4:
        assert prediction["enum"] == ["a", "b", "c", "d", None]
    else:
        assert "enum" not in prediction
    assert "enum" not in schema["properties"]["semantic_answer"]
    assert "supported semantic value" in system
    assert "Only final_prediction holds the option label" in system
    assert not calls.requests


def test_repair_input_overflow_is_diagnosed_before_consuming_repair_budget():
    bad = observation("bad-" + "x" * 3000, ["unavailable-source"])
    reader, engine, state, calls = fixture(lambda index, data: response(deepcopy(bad)))
    reader.current_identity = "history-query"
    payload = reader._payload(engine.steps[0], "Read past creative activities", ["history"], ["s1"])
    base = engine.reasoner.estimate("joint_read", reader._system(payload), reader._wire_data(payload), reader._schema(payload))
    engine.s["context_tokens"] = base + engine.s["reasoning_output_tokens"] + engine.s["input_margin"] + 8 + 500
    assert read(reader)
    assert len(calls.requests) == 1 and engine.ledger.used["json_repairs"] == 0
    assert state.pending == {"history"} and reader.failed_obligations
    event = next(event for event in engine.events if event["event"] == "joint_repair_input_unavailable")
    assert event["operation"] == "joint_read_repair"
    assert event["node_id"] == "history" and event["identity"] == "history-query"
    assert event["repair_attempt"] == 1
    assert event["input_tokens"] + event["output_token_reserve"] + event["input_margin"] + 8 > event["context_tokens"]


def test_supported_set_is_a_complete_value_and_disagreeing_subsets_do_not_union():
    reader, engine, state, _ = fixture()
    shown = visible(reader, ["s1", "s2"])
    rows = [reader._observation(observation("one", ["s1"], ["piano"]), "history", shown, engine.steps[0]),
            reader._observation(observation("two", ["s2"], ["painting"]), "history", shown, engine.steps[0])]
    state.integrate(rows)
    assert "history" not in state.valid()[0]
    assert {tuple(route["value"]) for route in state.routes.values()} == {("piano",), ("painting",)}
    assert len(state.routes) == 2


def test_complete_set_supported_by_joint_sources_binds_without_per_source_union():
    reader, engine, state, _ = fixture()
    row = reader._observation(observation("whole", ["s1", "s2"], ["piano", "painting"]),
                              "history", visible(reader, ["s1", "s2"]), engine.steps[0])
    state.integrate([row])
    assert set(state.valid()[0]["history"]) == {"piano", "painting"}
    assert len(state.routes) == 1
    assert set(state.valid()[1]["history"]) == {
        engine.mapper.sources[alias]["source_id"] for alias in ["s1", "s2"]
    }
