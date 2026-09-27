"""Actual Reasoner parsing with scripted outputs exercises bounded mapping recovery."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from dagbt.budget import Ledger
from dagbt.config import DEFAULTS
from dagbt.evidence_mapping import EvidenceMapper
from dagbt.reasoning import InputOverflow, Reasoner


def assessment(unit, *, node=None, spans=None, claim="The source supplies this fact", **changes):
    return {"span_ids": spans or [unit["source_span_id"]], "node_id": node or unit["node_ids"][0],
            "kind": "explicit", "stance": "support", "claim": claim,
            "entity_scope": "current question", "event_time": None, "time_span_ids": [],
            "reason": "Source wording", **changes}


def row(unit, items=None, reason=""):
    return {"unit_id": unit["unit_id"], "assessments": [assessment(unit)] if items is None else items,
            "irrelevance_reason": reason}


def correct(data, _):
    return {"units": [row(unit) for unit in data["units"]]}


class ScriptedCalls:
    def __init__(self, engine, script):
        self.engine, self.script, self.requests = engine, script, []

    def get(self, stage, url, payload, **options):
        self.engine.ledger.reserve("llm", "/".join(stage))
        data = json.loads(payload["messages"][1]["content"])
        self.requests.append({"stage": stage, "payload": deepcopy(payload), "data": data, "options": options})
        result = self.script(data, len(self.requests))
        if isinstance(result, tuple):
            content, finish, refusal = result
        else:
            content, finish, refusal = json.dumps(result), "stop", None
        return {"response": {"id": "scripted", "choices": [{"finish_reason": finish,
            "message": {"content": content, "refusal": refusal}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}}, "response_ref": str(len(self.requests))}


def engine(script=correct, documents=None, **settings):
    docs = documents or {"d1": SimpleNamespace(passage="A fact about one place."),
                         "d2": SimpleNamespace(passage="A fact about another place.")}
    e = SimpleNamespace(q={"id": "q", "question": "What do the sources establish?"},
        docs=docs, s={**deepcopy(DEFAULTS), **settings}, spans={}, candidates=[], chunks=[], mapped_chunks=set(),
        steps=[{"output_slot": "n1", "question": "What fact?", "inputs": []}],
        errors=[], events=[])
    e.event = lambda value: e.events.append(deepcopy(value))
    e.ledger = Ledger({"llm": e.s["llm_calls"], "json_repairs": e.s["json_repairs"]}, e.event)
    e.calls = ScriptedCalls(e, script)
    e.reasoner = Reasoner(e.calls, None, {"model_profile": "bridgetree", "llm_base_url": "http://unused/v1",
        "llm_model": "test"}, e.s, e.ledger, e.event)
    e.mapper = EvidenceMapper(e)
    return e


def test_mapping_exposes_exact_source_ids_and_retains_valid_units_and_assessments_during_repair():
    def script(data, call):
        units = data["units"]
        if call == 1:
            return {"units": [row(units[0]), row(units[1], [assessment(units[1], claim="Retained valid fact"),
                                   assessment(units[1], node="phantom")])]}
        assert [u["unit_id"] for u in units] == ["u2"]
        assert data["repair_scope"]["retained_evidence_ids"]
        return {"units": [row(units[0], [assessment(units[0], claim="Repaired second fact")])]}
    e = engine(script)
    e.mapper.add_candidates(list(e.docs))
    e.mapper.map_pending()
    assert len(e.calls.requests) == 2 and e.ledger.used["json_repairs"] == 1
    assert len(e.spans) == 3 and e.mapped_chunks == {0, 1}
    assert {s["claim"] for s in e.spans.values()} == {
        "The source supplies this fact", "Retained valid fact", "Repaired second fact"}
    assert all(s["node_ids"] == ["n1"] for s in e.spans.values())
    assert all("span_ids" not in s and "source_ids" in s for s in e.spans.values())
    assert e.mapper.diagnostics()["mapping_incomplete"] is False
    assert all(len(req["payload"]["messages"]) == 2 for req in e.calls.requests)
    assert e.errors[0]["failure_category"] == "evidence_relation"


@pytest.mark.parametrize("failure", ["length", "syntax"])
def test_unparseable_or_truncated_batch_splits_and_each_retry_is_metered(failure):
    def script(data, call):
        if call == 1:
            return ('{"units":[', "length" if failure == "length" else "stop", None)
        return correct(data, call)
    docs = {"d" + str(i): SimpleNamespace(passage="Distinct factual source " + str(i)) for i in range(4)}
    e = engine(script, docs)
    e.mapper.add_candidates(list(docs))
    e.mapper.map_pending()
    assert [len(req["data"]["units"]) for req in e.calls.requests] == [4, 2, 2]
    assert e.ledger.used["json_repairs"] == 2
    assert not e.mapper.diagnostics()["mapping_incomplete"]
    assert all(value["attempts_by_node"] == {"n1": 2} for value in e.mapper.diagnostics()["unit_statuses"].values())
    assert any(event.get("split_for_retry") for event in e.events)


def test_persistent_semantic_errors_stop_at_per_unit_limit_across_future_calls():
    def wrong(data, _):
        return {"units": [row(u, [assessment(u, node="unseen")]) for u in data["units"]]}
    e = engine(wrong)
    e.mapper.add_candidates(list(e.docs))
    for _ in range(5):
        e.mapper.map_pending()
    assert len(e.calls.requests) == 3
    assert e.ledger.used["json_repairs"] == 2
    assert e.mapper.diagnostics()["unavailable_unit_ids"] == ["u1", "u2"]
    assert e.mapper.diagnostics()["mapping_incomplete"]
    assert not e.spans and not e.mapped_chunks


def test_global_repair_limit_prevents_new_round_reset():
    e = engine(lambda data, _: {"units": []}, json_repairs=1)
    e.mapper.add_candidates(list(e.docs))
    e.mapper.map_pending()
    e.mapper.map_pending()
    assert len(e.calls.requests) == 2 and e.ledger.used["json_repairs"] == 1
    assert all("global_repair_budget" in state["unavailable_nodes"].values()
               for state in e.mapper.diagnostics()["unit_statuses"].values())


def test_explicit_irrelevance_completes_units_but_empty_unexplained_mapping_does_not():
    good = engine(lambda data, _: {"units": [row(u, [], "No relevant content") for u in data["units"]]})
    good.mapper.add_candidates(list(good.docs))
    good.mapper.map_pending()
    assert good.mapped_chunks == {0, 1} and not good.spans
    bad = engine(lambda data, _: {"units": [row(u, []) for u in data["units"]]})
    bad.mapper.add_candidates(list(bad.docs))
    bad.mapper.map_pending()
    assert not bad.mapped_chunks and bad.mapper.diagnostics()["mapping_incomplete"]


def test_multifragment_assessment_remains_one_fact_with_no_fabricated_contiguous_quote():
    def script(data, _):
        first, second = data["units"]
        return {"units": [row(first, [assessment(first, spans=[first["source_span_id"], second["source_span_id"]])]),
                          row(second, [], "No separate assessment beyond the joint source") ]}
    e = engine(script, {"d": SimpleNamespace(passage="First claim. Second claim.")}, max_quote_chars=13)
    e.mapper.add_candidates(["d"])
    e.mapper.map_pending()
    assert len(e.spans) == 1
    fact = next(iter(e.spans.values()))
    assert len(fact["fragments"]) == 2
    assert not {"start", "end", "exact_quote"} & set(fact)
    assert "".join(f["exact_quote"] for f in fact["fragments"]) == e.docs["d"].passage
    assert all(f["source_role"] == "document" for f in fact["fragments"])
    assert fact["node_ids"] == ["n1"]


@pytest.mark.parametrize("bad_field", ["foreign_doc", "phantom_source", "time_without_source", "unknown_time_with_source"])
def test_source_and_time_references_are_strictly_local(bad_field):
    def script(data, _):
        first, second = data["units"]
        item = assessment(first)
        if bad_field == "foreign_doc":
            item["span_ids"].append(second["source_span_id"])
        elif bad_field == "phantom_source":
            item["span_ids"].append("s9999")
        elif bad_field == "time_without_source":
            item["event_time"] = "last summer"
        else:
            item["time_span_ids"] = [first["source_span_id"]]
        return {"units": [row(first, [item]), row(second, [], "Unrelated")]}
    e = engine(script, max_repairs_per_request=0)
    e.mapper.add_candidates(list(e.docs))
    e.mapper.map_pending()
    assert not e.spans and e.mapped_chunks == {1}
    assert e.mapper.diagnostics()["unavailable_unit_ids"] == ["u1"]


def test_refinement_is_remapped_separately_and_never_transfers_old_semantics():
    def script(data, call):
        if call == 1:
            return correct(data, call)
        assert [step["output_slot"] for step in data["nodes"]] == ["n2"]
        return {"units": [row(u, [], "Not relevant to the new subquestion") for u in data["units"]]}
    e = engine(script)
    e.mapper.add_candidates(list(e.docs))
    e.mapper.map_pending()
    saved = deepcopy(e.spans)
    e.steps.append({"output_slot": "n2", "question": "A genuinely new need?", "inputs": ["n1"]})
    e.mapper.map_pending()
    assert e.spans == saved and all(f["node_ids"] == ["n1"] for f in e.spans.values())
    assert e.mapped_chunks == {0, 1}
    assert e.ledger.used["json_repairs"] == 0
    assert all(state["mapped_node_ids"] == ["n1", "n2"] for state in e.mapper.diagnostics()["unit_statuses"].values())


@pytest.mark.parametrize("reword_reason", [False, True])
def test_repeating_retained_fact_does_not_discharge_failed_assessment_slot(reword_reason):
    def script(data, call):
        unit = data["units"][0]
        valid = assessment(unit, claim="Already validated")
        if call == 1:
            return {"units": [row(unit, [valid, assessment(unit, spans=["s999"], claim="Missing correction")])]}
        assert data["repair_scope"]["failed_assessment_slots"]["u1"]
        if reword_reason:
            valid["reason"] = "Only diagnostic wording changed " + str(call)
        return {"units": [row(unit, [valid])]}
    e = engine(script, {"d": SimpleNamespace(passage="One exact source.")})
    e.mapper.add_candidates(["d"])
    e.mapper.map_pending()
    assert len(e.calls.requests) == 3
    assert {f["claim"] for f in e.spans.values()} == {"Already validated"}
    assert not e.mapped_chunks
    assert e.mapper.diagnostics()["unit_statuses"]["u1"]["failed_assessment_slots"]
    assert e.mapper.diagnostics()["unavailable_unit_ids"] == ["u1"]
    json.dumps(e.mapper.diagnostics(), allow_nan=False)
    json.dumps(e.events, allow_nan=False)


def test_source_unit_size_is_never_above_400_even_with_a_larger_legacy_cap():
    e = engine(documents={"d": SimpleNamespace(passage="x" * 901)}, max_quote_chars=1000)
    e.mapper.add_candidates(["d"])
    assert [len(source["text"]) for source in e.mapper.sources.values()] == [400, 400, 101]


def test_new_replacement_must_match_failed_slots_known_node():
    def script(data, call):
        unit = data["units"][0]
        if call == 1:
            return {"units": [row(unit, [assessment(unit, node="n1"),
                assessment(unit, node="n2", spans=["s999"])])]}
        # Another novel n1 fact must not repair the failed n2 assessment.
        return {"units": [row(unit, [assessment(unit, node="n1", claim="New n1 fact " + str(call))])]}
    e = engine(script, {"d": SimpleNamespace(passage="One exact source.")})
    e.steps.append({"output_slot": "n2", "question": "Second node?", "inputs": []})
    e.mapper.add_candidates(["d"])
    e.mapper.map_pending()
    assert len(e.calls.requests) == 3 and not e.mapped_chunks
    assert all(f["node_ids"] == ["n1"] for f in e.spans.values())


def test_changed_but_still_invalid_repair_does_not_create_an_extra_obligation():
    slot_ids = []
    def script(data, call):
        unit = data["units"][0]
        if call == 1:
            return {"units": [row(unit, [assessment(unit, claim="Retained old fact"),
                assessment(unit, spans=["s999"], claim="First invalid form")])]}
        slots = data["repair_scope"]["failed_assessment_slots"]["u1"]
        assert len(slots) == 1
        slot_ids.append(slots[0]["slot_id"])
        if call == 2:
            return {"units": [row(unit, [assessment(unit, spans=["s888"], claim="Changed but still invalid")])]}
        return {"units": [row(unit, [assessment(unit, claim="New valid replacement")])]}
    e = engine(script, {"d": SimpleNamespace(passage="One exact source.")})
    e.mapper.add_candidates(["d"])
    e.mapper.map_pending()
    assert len(e.calls.requests) == 3 and e.ledger.used["json_repairs"] == 2
    assert len(set(slot_ids)) == 1
    assert e.mapped_chunks == {0}
    assert {f["claim"] for f in e.spans.values()} == {"Retained old fact", "New valid replacement"}
    assert not e.mapper.diagnostics()["unit_statuses"]["u1"]["failed_assessment_slots"]


def test_pending_failure_targets_stay_frozen_when_refinement_is_added():
    def script(data, call):
        unit = data["units"][0]
        if call == 1:
            return {"units": [row(unit, [assessment(unit), assessment(unit, spans=["s999"])])]}
        if call == 2:
            assert unit["node_ids"] == ["n1"]
            return {"units": [row(unit, [assessment(unit, claim="Corrected failed old-node assessment")])]}
        assert unit["node_ids"] == ["n2"]
        return {"units": [row(unit, [], "No evidence for the refinement") ]}
    e = engine(script, {"d": SimpleNamespace(passage="One exact source.")})
    e.mapper.add_candidates(["d"])
    # Fairness allows one initial call and defers its repair.
    e.mapper.map_pending(remaining_nodes=11)
    assert len(e.calls.requests) == 1
    e.steps.append({"output_slot": "n2", "question": "Refinement?", "inputs": []})
    e.mapper.map_pending()
    assert len(e.calls.requests) == 3 and e.mapped_chunks == {0}
    assert e.ledger.used["json_repairs"] == 1
    assert e.mapper.diagnostics()["unit_statuses"]["u1"]["node_results"]["n2"]["status"] == "no_assessment_emitted"


def test_fixed_fields_overflow_is_typed_and_makes_no_model_request():
    e = engine(map_batch_tokens=50)
    e.mapper.add_candidates(list(e.docs))
    with pytest.raises(InputOverflow, match="schema and fixed mapping fields"):
        e.mapper.map_pending()
    assert not e.calls.requests and not e.ledger.used["json_repairs"]


def test_mapping_request_preserves_future_resolve_audit_and_flat_selection_calls():
    e = engine(selection="flat")
    e.mapper.add_candidates(list(e.docs))
    e.mapper.map_pending(remaining_nodes=2)
    assert e.calls.requests[0]["options"] == {"reserve": 0, "extra_reserve": 4}
    exhausted = engine(llm_calls=4, selection="flat")
    exhausted.mapper.add_candidates(list(exhausted.docs))
    exhausted.mapper.map_pending(remaining_nodes=2)
    assert not exhausted.calls.requests and exhausted.mapper.diagnostics()["mapping_incomplete"]


def test_refusal_stops_without_retry_and_preplan_diagnostics_are_available():
    e = engine(lambda *_: ("", "stop", "Provider refusal"))
    e.steps = []
    assert e.mapper.diagnostics()["mapping_incomplete"] is False
    e.steps = [{"output_slot": "n1", "question": "What?", "inputs": []}]
    e.mapper.add_candidates(list(e.docs))
    e.mapper.map_pending()
    e.mapper.map_pending()
    assert len(e.calls.requests) == 1 and e.ledger.used["json_repairs"] == 0
    assert e.mapper.diagnostics()["unavailable_unit_ids"] == ["u1", "u2"]
