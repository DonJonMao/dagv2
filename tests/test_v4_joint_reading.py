"""Production reader regressions with scripted responses, no model or labels."""
from copy import deepcopy
import json

import pytest

from dagbt.joint_reading import JointReader
from dagbt.evidence_mapping import EvidenceMapper
from dagbt.transport import digest
from test_personamem_joint_protocol import (
    fixture, observation, response, expand_wire_data, visible,
)


def prepare_terminal(*, personal=False, kind="text", max_repairs=2):
    reader, engine, state, calls = fixture(max_repairs_per_request=max_repairs)
    engine.personal_response_selection = personal
    engine.output_options = ["Try a piano duet this weekend.", "Go painting.", "Try rugby.", "No advice."]
    state.program["variables"]["suggestion"]["type"] = kind
    state.initial_program["variables"]["suggestion"]["type"] = kind
    state.contract["result_type"] = kind
    validated = reader._observation(observation(), "history", visible(reader), engine.steps[0])
    state.integrate([validated])
    return reader, engine, state, calls


def terminal_response(data, value, *, prediction="a", semantic_answer=None):
    slots=[]
    for variable in data["variables"]:
        rows=[]
        if variable == "suggestion":
            rows=[observation("suggestion", ["s1"], value=value,
                parent_versions={"history": data["supported_parents"]["history"]["version"]})]
        slots.append({"variable":variable,"observations":rows,"unresolved":[]})
    return {"slots":slots,"final_prediction":prediction,
            "semantic_answer":value if semantic_answer is None else semantic_answer}


def compose(reader):
    return reader.read(reader.e.steps[1], "Choose a response using piano history", ["suggestion"],
                       ["piano"], "terminal", semantic=True, terminal_output=True)


def test_grounded_parent_requirements_are_explicit_in_schema_and_scoped_repair():
    reader,engine,state,calls=prepare_terminal(personal=True)
    selected=engine.output_options[0]
    expected={"history":state.versions["history"]}

    def respond(index,data):
        assert data["required_parent_versions"]["suggestion"]==expected
        schema=reader._schema(data)
        branch=next(branch for branch in schema["properties"]["slots"]["items"]["oneOf"]
                    if branch["properties"]["variable"]["const"]=="suggestion")
        parents=branch["properties"]["observations"]["items"]["then"]["properties"]["parent_versions"]
        assert parents=={"properties":{"history":{"const":expected["history"]}},"required":["history"]}
        result=terminal_response(data,selected)
        if index==1:
            next(slot for slot in result["slots"] if slot["variable"]=="suggestion")["observations"][0]["parent_versions"]={}
        else:
            assert data["repair_scope"]["failed"][0]["required_parent_versions"]==expected
            assert data["repair_scope"]["failed"][0]["observation"]["parent_versions"]=={}
        return result

    calls.respond=respond
    compose(reader)
    assert len(calls.requests)==2 and reader.semantic_inputs["validation_complete"]
    assert state.valid()[0]["suggestion"]==selected


def test_controller_can_defer_repeated_gap_review_without_forgetting_its_sources():
    reader,engine,state,calls=fixture(max_repairs_per_request=0)
    partial=observation("clue",["s1"],stance="partial")
    partial["value"]=None
    calls.respond=lambda index,data:response(partial)
    assert reader.read(engine.steps[0],"Read activity",["history"],["piano"],"first")
    before=deepcopy(reader.obligation_snapshot())
    assert not reader.read(engine.steps[0],"Read activity",["history"],["piano"],"first",
                           review_existing_obligations=False)
    assert len(calls.requests)==1 and reader.obligation_snapshot()==before
    assert state.pending=={"history"}

    def review(index,data):
        result=response(observation("supported",["s1"]))
        result["slots"][0]["resolved_obligation_ids"]=[item["id"] for item in data["evidence_obligations"]]
        return result

    calls.respond=review
    assert reader.read(engine.steps[0],"Read activity",["history"],[],"first")
    assert len(calls.requests)==2 and not state.pending


def test_existing_proof_replay_skips_new_capacity_admission_and_ignores_citation_order():
    reader,engine,state,_=fixture()
    original=reader._observation(observation("initial",["s1","s2"]),"history",visible(reader,["s1","s2"]),engine.steps[0])
    state.integrate([original])
    before=deepcopy(state.routes);version=state.versions["history"]
    replay={**original,"id":"replayed-draft","source_ids":list(reversed(original["source_ids"])),
            "reason":"Same evidence, differently described."}

    def impossible(operation,data):
        raise AssertionError("An already accepted proof is not a new capacity admission")

    reader._fits=impossible
    assert reader._handoff_failures([replay],{("history","replayed-draft")})=={}
    state.integrate([replay])
    assert state.routes==before and state.versions["history"]==version


def test_legacy_ordered_route_is_reused_without_resurrection_after_a_counter():
    reader,engine,state,_=fixture()
    original=reader._observation(observation("legacy",["s1","s2"]),"history",visible(reader,["s1","s2"]),engine.steps[0])
    original["source_ids"]=sorted(original["source_ids"],reverse=True)
    identity=digest({k:original[k] for k in
        ("variable","value","source_ids","entity_scope","time_scope","parent_versions")})
    legacy_id="fact_"+identity[:24]
    state.routes[legacy_id]={**deepcopy(original),"id":legacy_id,"revoked":[],"semantic_status":"model_judgment"}
    state._refresh_versions()
    replay={**original,"id":"new-draft","source_ids":list(reversed(original["source_ids"]))}
    state.integrate([replay])
    assert list(state.routes)==[legacy_id]
    draft=observation("counter",["s3"],stance="contradiction",retract_ids=[legacy_id])
    draft["value"]=None
    counter=reader._observation(draft,"history",visible(reader,["s3"]),engine.steps[0])
    state.integrate([counter,replay])
    assert list(state.routes)==[legacy_id] and state.routes[legacy_id]["revoked"]
    assert "history" not in state.valid()[0]
    reader._fits=lambda operation,data:False
    assert ("history","new-draft") in reader._handoff_failures([replay],{("history","new-draft")})


def test_replaying_old_proof_does_not_exempt_new_sources_or_drop_a_counter():
    reader,engine,state,calls=fixture(max_repairs_per_request=0)
    original=reader._observation(observation("initial",["s1"]),"history",visible(reader),engine.steps[0])
    state.integrate([original]);route=next(iter(state.routes))
    replay=observation("replay",["s1"])
    novel=observation("new-alternative",["s1","s2"])
    counter=observation("counter",["s3"],stance="contradiction",retract_ids=[route])
    counter["value"]=None
    reader._fits=lambda operation,data: data["task"]["output_slot"]=="history"
    calls.respond=lambda index,data:response(replay,novel,counter)
    reader.read(engine.steps[0],"Check originals",["history"],list(engine.docs),"check")
    failures=reader.batches[-1]["failed_slots"]
    assert any(row["id"]=="new-alternative" and row["error"].startswith("citation_capacity:") for row in failures)
    assert not any(row["id"]=="replay" for row in failures)
    assert state.routes[route]["revoked"] and "history" not in state.valid()[0]


def test_terminal_consistency_is_repaired_before_audit_without_rewriting_facts():
    reader, engine, state, calls = prepare_terminal(personal=True)
    selected=engine.output_options[0]

    def respond(index, data):
        if index==1:
            return terminal_response(data, selected, semantic_answer="The user plays piano.")
        assert data["requested_slots"]==[]
        assert data["terminal_repair"]["expected_value"]==selected
        assert {s["id"] for s in data["source_spans"]}=={"s1"}
        return {"slots":[],"semantic_answer":selected,"final_prediction":"a"}

    calls.respond=respond
    assert compose(reader)
    assert len(calls.requests)==2
    assert calls.requests[1]["stage"][0]=="semantic_compose_repair"
    assert reader.semantic_inputs["validation_complete"]
    assert reader.semantic_result==state.valid()[0]["suggestion"]==selected
    assert reader.semantic_inputs["binding_dependencies"]["history"]=={
        "version":state.versions["history"],"identity":digest(["piano"])}
    assert len([r for r in state.routes.values() if r["variable"]=="suggestion"])==1


def test_personal_terminal_fact_summary_is_not_accepted_as_selected_response():
    reader, engine, state, calls=prepare_terminal(personal=True, max_repairs=0)
    calls.respond=lambda index,data: terminal_response(data,"The user plays piano.")
    assert compose(reader)
    assert "suggestion" not in state.valid()[0]
    assert not reader.semantic_inputs["validation_complete"]
    assert any("exact selected public option" in failure["error"]
               for failure in reader.batches[-1]["failed_slots"])


def test_terminal_typed_record_is_not_confused_with_display_json_string():
    reader, _, state, calls=prepare_terminal(kind="record")
    value={"activities":["piano"],"source_ids":["application-record-field"]}

    def respond(index,data):
        if index==1:
            return terminal_response(data,value,semantic_answer=json.dumps(value))
        assert data["terminal_repair"]["expected_value"]==value
        return {"slots":[],"semantic_answer":deepcopy(value),"final_prediction":"a"}

    calls.respond=respond
    compose(reader)
    assert reader.semantic_result==value==state.valid()[0]["suggestion"]
    assert reader.semantic_inputs["validation_complete"]


def test_wrong_option_letter_gets_bounded_consistency_repair():
    reader, engine, _, calls=prepare_terminal(personal=True)
    selected=engine.output_options[0]
    calls.respond=lambda index,data: (terminal_response(data,selected,prediction="b") if index==1
        else {"slots":[],"semantic_answer":selected,"final_prediction":"a"})
    compose(reader)
    assert reader.semantic_prediction=="a" and len(calls.requests)==2


def test_repair_budget_is_shared_with_slot_repairs_and_never_forces_abstention_to_label():
    reader, engine, state, calls=prepare_terminal(personal=True,max_repairs=1)
    selected=engine.output_options[0]

    def respond(index,data):
        output=terminal_response(data,selected,semantic_answer="wrong field")
        if index==1:
            next(slot for slot in output["slots"] if slot["variable"]=="suggestion")["observations"][0]["source_ids"]=["missing"]
        return output

    calls.respond=respond
    compose(reader)
    assert len(calls.requests)==2
    assert not reader.semantic_inputs["validation_complete"]
    assert "suggestion" in state.pending
    assert any(failure["id"]=="terminal_output" for failure in reader.batches[-1]["failed_slots"])

    other,engine,_,calls=prepare_terminal(personal=True)
    calls.respond=lambda index,data: terminal_response(data,engine.output_options[0],prediction=None)
    compose(other)
    assert other.semantic_prediction is None and len(calls.requests)==1


def test_personal_audit_sees_original_task_options_and_candidate_but_never_changes_label():
    reader,engine,state,calls=prepare_terminal(personal=True)
    calls.respond=lambda index,data: terminal_response(data,engine.output_options[0])
    compose(reader)
    selected=reader.semantic_prediction

    def audit_response(index,data):
        assert data["original_question"]==engine.q["question"]
        assert data["public_output_question"]==engine.reader_question
        assert data["public_options"]==engine.output_options
        assert data["terminal_candidate"]=={"value":engine.output_options[0],"final_prediction":"a"}
        assert "ALL public" in data["personal_task_instruction"]
        return {"slots":[{"variable":v,"observations":[],"unresolved":[]} for v in data["variables"]],
                "audit_verdict":"conflict","final_prediction":"b"}

    calls.respond=audit_response
    reader.read(engine.steps[1],engine.q["question"],["history","suggestion"],["piano"],"audit",audit=True)
    assert reader.batches[-1]["audit_verdict"]=="conflict"
    assert not reader.batches[-1]["audit_complete"]
    assert reader.semantic_prediction==selected
    reader.current_terminal_output=False
    upstream=reader._payload(engine.steps[0],"Read history",["history"],["s1"])
    assert "public_options" not in upstream and "terminal_candidate" not in upstream


@pytest.mark.parametrize("failed",[False,True])
def test_repeated_empty_semantic_read_is_not_material_progress(failed):
    reader,engine,state,calls=fixture(max_repairs_per_request=0)

    def no_evidence(index,data):
        rows=[observation(value="unsupported",sources=[])] if failed else []
        return {"slots":[{"variable":"suggestion","observations":rows,
                          "unresolved":["No original personal evidence is available."]}]}

    calls.respond=no_evidence
    assert reader.read(engine.steps[1],"Missing film blog",["suggestion"],[],"no-evidence",semantic=True)
    signature=reader.material_signature()
    assert not reader.read(engine.steps[1],"Missing film blog",["suggestion"],[],"no-evidence",semantic=True)
    assert reader.material_signature()==signature


def test_exact_duplicate_draft_is_idempotent_and_conflicting_duplicate_can_be_repaired_losslessly():
    reader,engine,state,calls=fixture()
    calls.respond=lambda index,data: response(observation(),observation())
    assert reader.read(engine.steps[0],"Read",["history"],["piano"],"first")
    assert len(calls.requests)==1 and len(state.routes)==1
    route=next(iter(state.routes))
    support=observation("collision")
    counter=observation("collision",["s2"],stance="contradiction",retract_ids=[route])
    counter["value"]=None

    def duplicates(index,data):
        if "repair_scope" not in data:
            return {**response(support,counter),"audit_verdict":"unknown"}
        conflict=next(item for item in data["repair_scope"]["failed"] if item["id"]=="collision")
        assert conflict["observations"]==[support,counter]
        replacements=conflict["replacement_observations"]
        assert len(replacements)==2 and len({r["id"] for r in replacements})==2
        return {**response(*deepcopy(replacements)),"audit_verdict":"conflict"}

    calls.respond=duplicates
    assert reader.read(engine.steps[0],"Check disagreement",["history"],["piano","painting"],"duplicate",audit=True)
    assert reader.batches[-1]["complete"] and not state.pending
    assert state.routes[route]["revoked"]
    assert not reader.batches[-1]["audit_complete"]


def test_observation_table_preserves_all_routes_typed_values_and_counter_sources():
    reader,engine,_,_=fixture()
    data=reader._payload(engine.steps[0],"Check all routes",["history"],["s1","s2"])
    rows=[]
    for index in range(80):
        rows.append({"id":"fact_"+str(index),"variable":"history",
            "value":{"activities":["piano","painting"]*12,"source_ids":["literal-app-field"]},
            "source_ids":["s1"],"entity_scope":"user","time_scope":"history",
            "parent_versions":{},"stance":"support","semantic_status":"model_judgment",
            "reason":"The same source-grounded explanation. "*5,
            "revoked":[{"source_ids":["s2"],"reason":"A relevant scoped counter."}] if index%2 else []})
    data["known_observations"]=rows
    before=deepcopy(data)
    wire=JointReader._wire_data(data)
    assert "known_observation_columns" in wire
    restored=expand_wire_data(wire)
    columns=restored.pop("known_observation_columns")
    shared=restored.pop("known_observation_values")
    restored["known_observations"]=[dict(zip(columns,[shared[v["shared"]]
        if isinstance(v,dict) and set(v)=={"shared"} else v for v in row])) for row in restored["known_observations"]]
    assert restored==before and data==before
    assert len(json.dumps(wire))<len(json.dumps(data))
    assert "known_observation_columns" in reader._system(wire)


def test_duplicate_id_repair_cannot_turn_a_valid_counter_into_irrelevant_noise():
    reader,engine,state,calls=fixture(max_repairs_per_request=1)
    state.integrate([reader._observation(observation(),"history",visible(reader),engine.steps[0])])
    route=next(iter(state.routes))
    support=observation("same-id")
    counter=observation("same-id",["s2"],stance="contradiction",retract_ids=[route])
    counter["value"]=None

    def respond(index,data):
        if index==1:
            return {**response(support,counter),"audit_verdict":"supported"}
        rows=deepcopy(data["repair_scope"]["failed"][0]["replacement_observations"])
        for row in rows:
            if row["stance"]=="contradiction":
                row.update(stance="irrelevant",retract_ids=[],reason="Ignore disagreement.")
        return {**response(*rows),"audit_verdict":"supported"}

    calls.respond=respond
    reader.read(engine.steps[0],"Check",["history"],["piano","painting"],"repair",audit=True)
    assert state.pending=={"history"} and not reader.batches[-1]["audit_complete"]
    assert any("discard its counterevidence" in row["error"] for row in reader.batches[-1]["failed_slots"])


def test_literal_reference_shaped_typed_value_never_becomes_a_wire_reference():
    reader,engine,_,_=fixture()
    data=reader._payload(engine.steps[0],"Check routes",["history"],["s1"])
    data["known_observations"]=[{"id":str(index),"value":{"shared":index},"reason":"repeated "*100}
                                for index in range(8)]
    wire=reader._wire_data(data)
    assert "known_observation_columns" not in wire


def capacity_fixture():
    """A real token boundary: broad proof fits reading but not its next join."""
    from types import SimpleNamespace
    reader,engine,state,calls=fixture(max_repairs_per_request=2)
    reader.current_identity="read"
    engine.docs={}
    for index in range(8):
        text=("I played piano with friends and enjoyed making music. "*7).strip()
        name="memory-"+str(index)
        engine.docs[name]=SimpleNamespace(doc_id=name,passage=text,metadata={
            "source_segments":[{"role":"user","start":0,"end":len(text),"source_message_indices":[index]}]})
    engine.candidates=[];engine.chunks=[];engine.mapper=EvidenceMapper(engine)
    engine.mapper.add_candidates(list(engine.docs))
    aliases=list(engine.mapper.sources)
    shown={alias:engine.mapper.sources[alias] for alias in aliases}
    broad=reader._observation(observation("proof",aliases),"history",shown,engine.steps[0])
    narrow=reader._observation(observation("proof",[aliases[0]]),"history",shown,engine.steps[0])
    captured=[];fits=reader._fits
    reader._fits=lambda operation,data:captured.append((operation,deepcopy(data))) or True
    reader._handoff_failures([broad],{("history","proof")})
    broad_operation,broad_data=captured[-1]
    reader._handoff_failures([narrow],{("history","proof")})
    narrow_operation,narrow_data=captured[-1]
    reader._fits=fits
    original=reader._payload(engine.steps[0],"Read past creative activities",["history"],aliases)

    def needed(operation,data):
        repair={**data,"repair_scope":{"failed":[{"variable":v,"id":"*","feedback":"x"*240,"observation":None}
            for v in data["variables"]],"attempt":1,"previous_response_identity":"0"*64}}
        return max(reader._estimate(operation,data),reader._estimate(operation+"_repair",repair))

    floor=max(needed("joint_read",original),needed(narrow_operation,narrow_data))
    ceiling=needed(broad_operation,broad_data)
    assert floor<ceiling
    engine.s["context_tokens"]=floor+(ceiling-floor)//2+engine.s["reasoning_output_tokens"]+engine.s["input_margin"]+8
    assert reader._fits("joint_read",original) and reader._fits(narrow_operation,narrow_data)
    assert not reader._fits(broad_operation,broad_data)
    return reader,engine,state,calls,aliases,broad


def test_actual_handoff_capacity_requires_explicit_sufficient_citation_repair():
    reader,engine,state,calls,aliases,_=capacity_fixture()

    def respond(index,data):
        if index==1:
            return response(observation("proof",aliases))
        assert data["repair_scope"]["failed"][0]["feedback"].startswith("citation_capacity:")
        assert data["repair_scope"]["failed"][0]["observation"]["source_ids"]==aliases
        assert {s["id"] for s in data["source_spans"]}==set(aliases)
        return response(observation("proof",[aliases[0]]))

    calls.respond=respond
    reader.read(engine.steps[0],"Read past creative activities",["history"],list(engine.docs),"read")
    assert len(calls.requests)==2 and not state.pending
    assert state.valid()[1]["history"]==[engine.mapper.sources[aliases[0]]["source_id"]]
    assert len(engine.mapper.sources)==len(aliases)
    assert reader.batches[-1]["complete"]


def test_infeasible_citation_proof_stays_pending_without_slicing_or_overwriting_prior_fact():
    reader,engine,state,calls,aliases,_=capacity_fixture()
    validated=reader._observation(observation("existing",[aliases[0]]),"history",
                                  {a:engine.mapper.sources[a] for a in aliases},engine.steps[0])
    state.integrate([validated]);prior=deepcopy(state.routes)
    # Reading with the already-known parent needs a little more space. The
    # actual candidate join remains oversized after that envelope is admitted.
    reader._fits=lambda operation,data: (True if data["task"]["output_slot"]=="history"
                                         else JointReader._fits(reader,operation,data))
    engine.s["max_repairs_per_request"]=1
    calls.respond=lambda index,data: response(observation("proof",aliases))
    reader.read(engine.steps[0],"Read past creative activities",["history"],list(engine.docs),"read")
    assert state.routes==prior and state.valid()[0]["history"]==["piano"]
    assert state.pending=={"history"} and not reader.batches[-1]["complete"]
    assert len(engine.mapper.sources)==len(aliases)
    assert calls.requests[1]["data"]["repair_scope"]["failed"][0]["observation"]["source_ids"]==aliases


def test_rejected_large_support_cannot_discard_a_valid_counter_in_same_batch():
    reader,engine,state,calls,aliases,_=capacity_fixture()
    state.integrate([reader._observation(observation("existing",[aliases[0]]),"history",
        {a:engine.mapper.sources[a] for a in aliases},engine.steps[0])])
    route=next(iter(state.routes))
    counter=observation("counter",[aliases[0]],stance="contradiction",retract_ids=[route])
    counter["value"]=None
    reader._fits=lambda operation,data: (True if data["task"]["output_slot"]=="history"
                                         else JointReader._fits(reader,operation,data))
    engine.s["max_repairs_per_request"]=0
    calls.respond=lambda index,data: response(observation("proof",aliases),counter)
    reader.read(engine.steps[0],"Read past creative activities",["history"],list(engine.docs),"read")
    assert state.routes[route]["revoked"] and "history" not in state.valid()[0]
    assert state.pending=={"history"}


def test_conflicting_values_cannot_bypass_capacity_admission_by_becoming_unbound():
    reader,engine,state,_,aliases,_=capacity_fixture()
    shown={a:engine.mapper.sources[a] for a in aliases}
    state.integrate([reader._observation(observation("old",[aliases[0]]),"history",shown,engine.steps[0])])
    broad=reader._observation(observation("new",aliases,value=["painting"]),"history",shown,engine.steps[0])
    checked=[]
    reader._fits=lambda operation,data: checked.append(data) or len(data["source_spans"])<len(aliases)
    rejected=reader._handoff_failures([broad],{("history","new")})
    assert ("history","new") in rejected
    assert any(data["task"]["output_slot"]=="history" for data in checked)
    assert state.valid()[0]["history"]==["piano"]


def test_prefix_batching_counts_logarithmically_and_keeps_every_remaining_group():
    reader,engine,_,_=fixture()
    pending=[[str(index)] for index in range(100)]
    original=deepcopy(pending);tested=[]
    reader._payload=lambda step,query,variables,aliases,semantic,audit: {"aliases":aliases}
    reader._fits=lambda operation,data: tested.append(len(data["aliases"])) or len(data["aliases"])<=37
    aliases,consumed=reader._fitting_prefix("joint_read",engine.steps[0],"query",[],pending,False,False)
    assert consumed==37 and aliases==[str(index) for index in range(37)]
    assert len(tested)<=9 and pending==original
    assert pending[consumed:]==[[str(index)] for index in range(37,100)]


def test_nonmonotone_wire_packing_never_declares_capacity_failure_from_first_prefix_only():
    reader,engine,_,_=fixture()
    pending=[[str(index)] for index in range(10)]
    tested=[]
    reader._payload=lambda step,query,variables,aliases,semantic,audit: {"aliases":aliases}
    # Only seven groups fit; full and every normal binary probe fail. A
    # fallback is necessary because table packing isn't token-monotone.
    reader._fits=lambda operation,data: tested.append(len(data["aliases"])) or len(data["aliases"])==7
    aliases,consumed=reader._fitting_prefix("joint_read",engine.steps[0],"query",[],pending,False,False)
    assert consumed==7 and len(aliases)==7 and 7 in tested
