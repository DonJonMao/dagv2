"""Actual wire cap recounting and non-generating structural allowance policy."""
import json

from test_v4_reconcile_review import personal_reader
from test_v4_summary_revision_review import summary_fixture, OLD


def history(reader, engine):
    reader.current_identity='cap-policy'
    return reader._payload(engine.steps[0],'Review history',['history'],['s1'])


def test_structure_floor_counts_the_complete_assistant_envelope_without_a_request_or_answer(monkeypatch):
    reader,engine,state,calls,original=summary_fixture(reasoning_output_tokens=4096)
    data=history(reader,engine)
    before=state.public_dict()
    used=dict(engine.ledger.used)
    seen=[]
    def count(operation,payload):
        seen.append((operation,payload))
        assert payload['max_tokens']==1234
        assert len(payload['messages'])==1
        message=payload['messages'][0]
        assert message['role']=='assistant'
        slots=json.loads(message['content'])['slots']
        assert len(slots)==1 and slots[0]['variable']=='history' and slots[0]['unresolved']==[]
        observation=slots[0]['observations'][0]
        assert observation['value']==OLD and observation['source_ids']==['s1']
        assert observation['summary_revision']['supersedes_route_ids']==[original]
        assert observation['summary_revision']['previous_version']==state.versions['history']
        return {'count':57,'token_count_is_estimate':False}
    monkeypatch.setattr(engine.reasoner,'_count_payload',count)
    assert reader._fresh_structure_floor('joint_read',data,output_token_cap=1234)==57
    assert seen[0][0]=='joint_read_structure'
    assert not calls.requests and not engine.reasoner.requests
    assert engine.ledger.used==used and state.public_dict()==before


def test_cap_recounts_actual_wire_and_shrinks_to_a_consistent_bound(monkeypatch):
    reader,engine,_,_=personal_reader(context_tokens=16384,reasoning_output_tokens=4096,
                                     max_repairs_per_request=1)
    data=history(reader,engine)
    monkeypatch.setattr(reader,'_fits',lambda operation,data:True)
    seen=[]
    def estimate(operation,data,*,output_token_cap=None):
        if 'repair_scope' in data:
            assert output_token_cap is None  # The repair output stays 4096.
            return 7929
        seen.append(output_token_cap)
        # The actual whole-wire counter can react to the new cap field.
        return {None:12024,4095:12026,4094:12027,4093:12027}[output_token_cap]
    monkeypatch.setattr(reader,'_estimate',estimate)
    floors=[]
    def floor(operation,data,*,output_token_cap=None):
        floors.append(output_token_cap)
        return 80
    monkeypatch.setattr(reader,'_fresh_structure_floor',floor)
    chosen=reader._adaptive_fresh_cap('joint_read',data)
    assert seen==[None,4095,4094,4093] and floors==[4093]
    assert chosen['output_token_cap']==4093
    assert chosen['initial_total_tokens']==16384
    assert chosen['forecast_total_tokens']==16382
    assert chosen['repair_output_reserve']==4096


def test_no_positive_capacity_cannot_be_forced_to_a_one_token_draft(monkeypatch):
    reader,engine,_,_=personal_reader(context_tokens=16384,reasoning_output_tokens=4096,
                                     max_repairs_per_request=1)
    data=history(reader,engine)
    monkeypatch.setattr(reader,'_fits',lambda operation,data:True)
    monkeypatch.setattr(reader,'_estimate',lambda operation,data:12024 if 'repair_scope' in data else 7000)
    def forbidden_floor(*args,**kwargs):
        raise AssertionError('No capacity should be coerced into a minimal draft')
    monkeypatch.setattr(reader,'_fresh_structure_floor',forbidden_floor)
    assert reader._adaptive_fresh_cap('joint_read',data) is None


def test_larger_admitted_history_window_avoids_an_extra_call_without_losing_read_sources(monkeypatch):
    """Exercise the actual read loop; only token counts and replies are doubles."""
    from test_v4_adaptive_initial_cap_review import (
        ExactAdmissionCounts, adaptive_fixture, read_all,
    )
    observations=[]
    for adaptive in (False, True):
        reader,engine,state,calls,_=adaptive_fixture(
            ExactAdmissionCounts({1:7000,2:7500,3:9500}))
        originals=dict(engine.mapper.sources)
        if not adaptive:
            monkeypatch.setattr(reader,'_adaptive_fresh_cap',lambda operation,data:None)
        read_all(reader,engine)
        assert state.valid()[0]['history']==OLD and not state.pending
        assert not reader.unread_obligations and not reader.failed_batch_reviews
        assert engine.mapper.sources==originals
        assert {alias for identity,alias in reader.read_states}==set(originals)
        assert all(batch['complete'] for batch in reader.batches)
        observations.append((len(calls.requests),engine.ledger.used['llm'],
            [request['payload']['max_tokens'] for request in calls.requests]))
    assert observations==[(2,2,[4096,4096]),(1,1,[2524])]
