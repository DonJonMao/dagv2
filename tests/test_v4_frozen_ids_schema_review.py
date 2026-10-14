"""Independent offline contract review of current-batch frozen ID schemas.

Real Reader validation, repair construction, schema serialization and state
retention are exercised with synthetic observations and local replies only.
"""
from copy import deepcopy
import json
import socket

import pytest
from jsonschema import Draft202012Validator

from test_personamem_joint_protocol import fixture, observation, visible


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Frozen-ID review must not contact services")
    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    monkeypatch.setattr("urllib.request.build_opener", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr("dagbt.reasoning.estimate_evidence_tokens", lambda value: 0)


def response_schema(payload):
    return json.loads(payload["messages"][0]["content"].rsplit(
        "Return one complete JSON object matching this schema:\n", 1)[1])


def reply(*rows, unresolved=(), **fields):
    return {"slots": [{"variable": "history", "observations": list(rows),
                       "unresolved": list(unresolved), **fields}]}


def unset_row(identifier="unsettled", stance="unknown", **fields):
    row = observation(identifier, stance=stance, **fields)
    row["value"] = None
    return row


@pytest.mark.parametrize("stance", ["unknown", "partial"])
def test_current_batch_unsettled_row_is_schema_frozen_without_reference_permission(stance):
    reader, engine, state, calls = fixture(max_repairs_per_request=1)
    counted = []
    def count(stage, url, payload):
        counted.append((len(reader.source_exposures), deepcopy(payload)))
        return {"count": 100, "token_count_is_estimate": False}
    calls.count_reasoning_tokens = count
    stable = engine.mapper.sources["s1"]["source_id"]
    reader.evidence_obligations["original-issue"] = {
        "id": "original-issue", "variable": "history", "entity_scope": "user",
        "time_scope": "history", "source_ids": [stable],
        "descriptions": ["A specific historical relation is not yet established."],
        "origins": ["older-query"],
    }
    draft = observation("frozen-unsettled", stance=stance)
    draft["value"] = None
    replay = {"slots": [{"variable": "history", "observations": [draft],
                          "unresolved": [], "resolved_obligation_ids": ["original-issue"]}]}
    seen = []
    def respond(index, data):
        if index == 2:
            schema = response_schema(calls.requests[-1]["payload"])
            # This is the same saved shape the runtime already forbids. It
            # must be rejected even though unsettled rows cannot be referenced.
            assert not Draft202012Validator(schema).is_valid(replay), (
                "repair schema still permits ordinary restatement of a current-batch " + stance)
            assert data["repair_scope"]["frozen_observation_ids"] == {"history": [draft["id"]]}
            assert all("frozen_observation_reviews" not in entry for entry in data["repair_scope"]["failed"])
            forbidden_reference = {"slots": [{"variable": "history", "observations": [],
                "unresolved": ["Still unknown"], "reviewed_observation_ids": [draft["id"]]}]}
            assert not Draft202012Validator(schema).is_valid(forbidden_reference)
            seen.append(deepcopy(data))
        # Deliberately ignore the schema: the unchanged runtime must still
        # reject the repair and retain the original responsibility.
        return deepcopy(replay)
    calls.respond = respond
    reader.read(engine.steps[0], "Read the scoped historical relation", ["history"],
                ["piano"], "current-batch-query")
    assert len(seen) == 1 and len(calls.requests) == 2
    repair_payload = calls.requests[1]["payload"]
    assert any(exposures == 1 and payload == repair_payload for exposures, payload in counted)
    assert any(exposures == 2 and payload == repair_payload for exposures, payload in counted)
    assert engine.ledger.used["llm"] == 2 and engine.ledger.used["json_repairs"] == 1
    assert "original-issue" in reader.evidence_obligations
    assert "history" in reader.pending_variables() and not state.valid()[0]
    failures = [entry for record in reader.failed_batch_reviews.values() for entry in record["failed"]]
    assert any("Already validated observations remain frozen" in entry["feedback"] for entry in failures)
    assert any(record["validated_observations"][0]["stance"] == stance
               for record in reader.failed_batch_reviews.values())


@pytest.mark.parametrize("stance", ["support", "contradiction", "irrelevant", "unknown", "partial"])
def test_forbidden_set_does_not_grant_reference_permission_for_any_stance(stance):
    reader, engine, _, _ = fixture()
    data = reader._payload(engine.steps[0], "Scoped review", ["history"], ["s1"])
    data["repair_scope"] = {"failed": [{"variable": "history", "id": "*"}],
                            "frozen_observation_ids": {"history": ["reserved"]}}
    schema = reader._schema(data)
    row = observation("reserved") if stance == "support" else unset_row("reserved", stance)
    # Contradiction has its own independent target requirement. Inspect the
    # forbidden-ID predicate directly as well as the full ordinary shape.
    branch = schema["properties"]["slots"]["items"]["oneOf"][0]
    rule = branch["properties"]["observations"]["items"]["properties"]["id"]
    assert not Draft202012Validator(rule).is_valid("reserved")
    assert not Draft202012Validator(schema).is_valid(reply(row))
    assert not Draft202012Validator(schema).is_valid(reply(reviewed_observation_ids=["reserved"]))
    assert Draft202012Validator(schema).is_valid(reply(unresolved=["Still unresolved"]))


@pytest.mark.parametrize("same_id_in_other_variable", [True, False])
def test_bare_id_is_never_a_cross_variable_freeze(same_id_in_other_variable):
    reader, engine, _, _ = fixture()
    data = reader._payload(engine.steps[0], "Scoped review", ["history"], ["s1"])
    data["variables"]["other"] = deepcopy(data["variables"]["history"])
    data["requested_slots"] = ["history", "other"]
    data["requested_scopes"]["other"] = deepcopy(data["requested_scopes"]["history"])
    data["repair_scope"] = {"failed": [], "frozen_observation_ids": {"history": ["same-id"]}}
    history_rows = [] if same_id_in_other_variable else [observation("same-id")]
    other_rows = [observation("same-id")] if same_id_in_other_variable else []
    response = {"slots": [{"variable": "history", "observations": history_rows, "unresolved": []},
                           {"variable": "other", "observations": other_rows, "unresolved": []}]}
    assert Draft202012Validator(reader._schema(data)).is_valid(response) is same_id_in_other_variable


def test_real_named_repair_lists_all_frozen_ids_in_order_even_when_none_is_referenceable():
    reader, engine, _, calls = fixture(max_repairs_per_request=1)
    calls.count_reasoning_tokens = lambda *args: {"count": 100, "token_count_is_estimate": False}
    unknown = unset_row("z-unknown")
    source_free = unset_row("a-source-free", "irrelevant", sources=[])
    invalid = observation("needs-repair", sources=["unavailable-source"])
    def respond(index, data):
        if index == 1:
            return reply(unknown, source_free, invalid)
        scope = data["repair_scope"]
        assert scope["frozen_observation_ids"] == {"history": ["a-source-free", "z-unknown"]}
        assert all("frozen_observation_reviews" not in entry for entry in scope["failed"])
        validator = Draft202012Validator(response_schema(calls.requests[-1]["payload"]))
        repaired = reply(unset_row("needs-repair"), unresolved=["The evidence remains incomplete."])
        assert validator.is_valid(repaired)
        for frozen in (unknown, source_free):
            restated = deepcopy(repaired)
            restated["slots"][0]["observations"].append(frozen)
            assert not validator.is_valid(restated)
        return repaired
    calls.respond = respond
    reader.read(engine.steps[0], "Scoped read", ["history"], ["piano"], "sorted-query")
    assert len(calls.requests) == 2 and "history" in reader.pending_variables()


@pytest.mark.parametrize("location", ["top", "slot", "observation"])
def test_controller_frozen_metadata_is_not_a_new_model_output_permission(location):
    reader, engine, _, _ = fixture()
    data = reader._payload(engine.steps[0], "Scoped review", ["history"], ["s1"])
    data["repair_scope"] = {"failed": [], "frozen_observation_ids": {"history": ["old"]}}
    response = reply(observation("fresh"))
    target = response if location == "top" else response["slots"][0] if location == "slot" else response["slots"][0]["observations"][0]
    target["frozen_observation_ids"] = {"history": []}
    assert not Draft202012Validator(reader._schema(data)).is_valid(response)


@pytest.mark.parametrize("with_metadata", [False, True])
def test_settled_reference_authority_stays_separate_with_legacy_offer_compatibility(with_metadata):
    reader, engine, _, _ = fixture()
    data = reader._payload(engine.steps[0], "Scoped review", ["history"], ["s1"])
    settled = observation("settled")
    data["repair_scope"] = {"failed": [{"variable": "history", "id": "*",
        "frozen_observation_reviews": {"slot_observation_ids": [], "observations": [settled]}}]}
    if with_metadata:
        data["repair_scope"]["frozen_observation_ids"] = {"history": ["unknown-only"]}
    validator = Draft202012Validator(reader._schema(data))
    assert validator.is_valid(reply(reviewed_observation_ids=["settled"]))
    assert not validator.is_valid(reply(settled))
    assert not validator.is_valid(reply(reviewed_observation_ids=["unknown-only"]))
    assert validator.is_valid(reply(observation("fresh")))
    if with_metadata:
        assert not validator.is_valid(reply(unset_row("unknown-only")))


def test_second_batch_can_reuse_a_known_route_id_without_inheriting_prior_freeze():
    reader, engine, state, calls = fixture(max_repairs_per_request=1)
    calls.count_reasoning_tokens = lambda *args: {"count": 100, "token_count_is_estimate": False}
    calls.respond = lambda index, data: reply(observation("original-draft"))
    reader.read(engine.steps[0], "First read", ["history"], ["piano"], "first-query")
    route_id = next(iter(state.routes))
    def second(index, data):
        assert index == 2 and any(row["id"] == route_id for row in data["known_observations"])
        assert not data.get("repair_scope", {}).get("frozen_observation_ids")
        response = reply(observation(route_id))
        assert Draft202012Validator(response_schema(calls.requests[-1]["payload"])).is_valid(response)
        return response
    calls.respond = second
    reader.read(engine.steps[0], "Second read", ["history"], ["piano"], "second-query")
    assert len(calls.requests) == 2 and engine.ledger.used["json_repairs"] == 0
    assert state.valid()[0]["history"] == ["piano"] and not state.pending


def test_durable_saved_draft_id_is_editable_until_the_new_batch_validates_it():
    reader, engine, state, calls = fixture(max_repairs_per_request=1)
    calls.count_reasoning_tokens = lambda *args: {"count": 100, "token_count_is_estimate": False}
    stable = engine.mapper.sources["s1"]["source_id"]
    old = observation("durable-draft", [stable])
    record = {"id": "retained", "variable": "history", "task_id": "history", "phase": "history",
        "query_identity": "old-query", "batch_identity": "old-batch", "source_ids": [stable],
        "failed": [{"variable": "history", "id": old["id"], "feedback": "A retained failed draft",
                    "observation": deepcopy(old)}], "validated_observations": [deepcopy(old)],
        "whole_value_reviews": [], "seen_source_ids": [], "seen_parent_signatures": []}
    reader.failed_batch_reviews["retained"] = deepcopy(record)
    reader.failed_obligations["old-query", "old-batch"] = {"history"}
    def recover(index, data):
        assert index == 1 and data["failed_batch_reviews"][0]["review_allowed"]
        assert not data.get("repair_scope", {}).get("frozen_observation_ids")
        response = reply(observation("durable-draft", ["s1", "s2"]), resolved_failed_batch_ids=["retained"])
        assert Draft202012Validator(response_schema(calls.requests[-1]["payload"])).is_valid(response)
        return response
    calls.respond = recover
    reader.read(engine.steps[0], "Recover with new originals", ["history"], ["painting"], "new-query")
    assert len(calls.requests) == 1 and not reader.failed_batch_reviews and not reader.failed_obligations
    assert engine.ledger.used["json_repairs"] == engine.ledger.used["llm"] == 1
    assert state.valid()[0]["history"] == ["piano"] and not state.pending


def test_real_whole_value_review_unfreezes_only_support_drafts_and_keeps_counter_frozen():
    reader, engine, state, calls = fixture(max_repairs_per_request=1)
    engine.personal_response_selection = True
    calls.count_reasoning_tokens = lambda *args: {"count": 100, "token_count_is_estimate": False}
    state.integrate([reader._observation(observation("prior"), "history", visible(reader), engine.steps[0])])
    old = next(iter(state.routes))
    counter = unset_row("counter", "contradiction", sources=["s2"], retract_ids=[old])
    left = observation("left", ["s2"], value=["painting"])
    right = observation("right", ["s3"], value=["pottery"])
    def review(index, data):
        if index == 1:
            return reply(left, right, counter)
        assert data["repair_scope"]["frozen_observation_ids"] == {"history": ["counter"]}
        assert {row["id"] for row in data["repair_scope"]["whole_value_reviews"][0]["draft_observations"]} == {"left", "right"}
        repaired = reply(observation("left", ["s2", "s3"], value=["painting", "pottery"]),
                         observation("right", ["s2", "s3"], value=["painting", "pottery"]))
        validator = Draft202012Validator(response_schema(calls.requests[-1]["payload"]))
        assert validator.is_valid(repaired)
        restated = deepcopy(repaired)
        restated["slots"][0]["observations"].append(counter)
        assert not validator.is_valid(restated)
        return repaired
    calls.respond = review
    reader.read(engine.steps[0], "Review all scoped history", ["history"], ["painting", "advice"], "whole-query")
    assert len(calls.requests) == 2 and state.routes[old]["revoked"]
    assert state.valid()[0]["history"] == ["painting", "pottery"] and not state.pending


@pytest.mark.parametrize("guard", ["summary", "handoff"])
def test_guard_rejected_support_is_removed_from_frozen_before_next_schema(guard, monkeypatch):
    reader, engine, state, calls = fixture(max_repairs_per_request=1)
    calls.count_reasoning_tokens = lambda *args: {"count": 100, "token_count_is_estimate": False}
    passes = []
    def reject_once(rows, *args, **kwargs):
        passes.append(1)
        return {("history", "recheck"): "scoped review needed"} if len(passes) == 1 else {}
    monkeypatch.setattr(reader, "_summary_revision_failures" if guard == "summary" else "_handoff_failures", reject_once)
    def respond(index, data):
        result = reply(observation("recheck"))
        if index == 2:
            assert "recheck" not in data["repair_scope"].get("frozen_observation_ids", {}).get("history", [])
            assert Draft202012Validator(response_schema(calls.requests[-1]["payload"])).is_valid(result)
        return result
    calls.respond = respond
    reader.read(engine.steps[0], "Recheck rejected support", ["history"], ["piano"], "guard-query")
    assert len(calls.requests) == 2 and not state.pending
    assert state.valid()[0]["history"] == ["piano"]


def test_new_frozen_metadata_and_full_schema_are_counted_before_any_repair_charge():
    reader, engine, state, calls = fixture(max_repairs_per_request=1)
    stable = engine.mapper.sources["s1"]["source_id"]
    reader.evidence_obligations["old-issue"] = {"id": "old-issue", "variable": "history",
        "entity_scope": "user", "time_scope": "history", "source_ids": [stable],
        "descriptions": ["A pending relation"], "origins": ["old"]}
    measured_repairs = []
    def count(stage, url, payload):
        data = json.loads(payload["messages"][1]["content"])
        if data.get("repair_scope", {}).get("frozen_observation_ids"):
            assert data["repair_scope"]["frozen_observation_ids"] == {"history": ["retained-unknown"]}
            schema = response_schema(payload)
            assert not Draft202012Validator(schema).is_valid(reply(unset_row("retained-unknown")))
            measured_repairs.append(deepcopy(payload))
            count_value = engine.s["context_tokens"] - payload["max_tokens"] - engine.s["input_margin"] - 7
            return {"count": count_value, "token_count_is_estimate": False}
        return {"count": 100, "token_count_is_estimate": False}
    calls.count_reasoning_tokens = count
    calls.respond = lambda index, data: reply(unset_row("retained-unknown"), resolved_obligation_ids=["old-issue"])
    reader.read(engine.steps[0], "Scoped read", ["history"], ["piano"], "capacity-query")
    assert measured_repairs and len(calls.requests) == 1
    assert engine.ledger.used["llm"] == 1 and engine.ledger.used["json_repairs"] == 0
    assert "old-issue" in reader.evidence_obligations and "history" in reader.pending_variables()
    assert not state.valid()[0]
