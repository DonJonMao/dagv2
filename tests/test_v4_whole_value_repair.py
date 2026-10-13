"""Scope, provenance and budget boundaries of local whole-value reconciliation."""
from copy import deepcopy

import pytest

from test_personamem_joint_protocol import fixture, observation, response, visible
from test_v4_reconcile_review import fragments, personal_reader, history_payload


def run_history(reader):
    return reader.read(reader.e.steps[0], 'Review past creative activities', ['history'],
                       ['piano', 'painting', 'advice'], 'history-review')


@pytest.mark.parametrize('kind,old,new,whole', [
    ('set', ['piano'], ['painting'], ['piano', 'painting']),
    ('text', 'Played piano.', 'Enjoyed painting.', 'Played piano and enjoyed painting.'),
    ('record', {'activity': 'piano'}, {'activity': 'painting'}, {'activities': ['piano', 'painting']}),
])
def test_existing_complete_value_requires_explicit_sourced_retraction_to_accept_new_whole(kind,old,new,whole):
    reader,engine,state,calls=personal_reader(max_repairs_per_request=1)
    for program in (state.program,state.initial_program):
        program['variables']['history']['type']=kind
    state.integrate([reader._observation(observation('published',['s1'],value=old),
                                        'history',visible(reader),engine.steps[0])])
    old_route=next(iter(state.routes))
    def answer(index,data):
        if index==1:
            return response(observation('draft',['s2'],value=new))
        group=data['repair_scope']['whole_value_reviews'][0]
        assert group['known_observations'][0]['id']==old_route
        assert group['known_observations'][0]['value']==old
        assert group['draft_observations'][0]['value']==new
        corrected=observation('draft',['s1','s2'],value=whole)
        superseded=observation('superseded',['s1','s2'],stance='contradiction',
                               retract_ids=[old_route],
                               reason='The earlier complete account omitted this separately supported activity.')
        superseded['value']=None
        return response(corrected,superseded)
    calls.respond=answer
    run_history(reader)
    assert state.routes[old_route]['revoked']
    assert state.valid()[0]['history']==whole and not state.pending
    assert engine.ledger.used['llm']==2 and engine.ledger.used['json_repairs']==1


def test_existing_ambiguous_routes_can_be_reviewed_without_a_new_support_draft():
    reader,engine,state,calls=personal_reader(max_repairs_per_request=1)
    state.integrate([reader._observation(row,'history',visible(reader,['s1','s2']),engine.steps[0])
                     for row in fragments()])
    old_routes=list(state.routes)
    def answer(index,data):
        if index==1:
            return response()
        group=data['repair_scope']['whole_value_reviews'][0]
        assert group['draft_observations']==[]
        assert {row['id'] for row in group['known_observations']}==set(old_routes)
        counter=observation('superseded',['s1','s2'],stance='contradiction',
                            retract_ids=old_routes,reason='Both old accounts were incomplete scoped wholes.')
        counter['value']=None
        return response(counter,observation('complete',['s1','s2'],value=['piano','painting']))
    calls.respond=answer
    run_history(reader)
    assert state.valid()[0]['history']==['piano','painting'] and not state.pending
    assert all(state.routes[rid]['revoked'] for rid in old_routes)


@pytest.mark.parametrize('invalid_repair',['omitted_draft','renamed_drafts','source_free_partial'])
def test_repair_cannot_hide_an_original_draft_or_its_evidence(invalid_repair):
    reader,engine,state,calls=personal_reader(max_repairs_per_request=1)
    def answer(index,data):
        if index==1:
            return response(*fragments())
        rows=[observation(row['id'],['s1','s2'],value=['piano','painting']) for row in fragments()]
        if invalid_repair=='omitted_draft':
            rows.pop()
        elif invalid_repair=='renamed_drafts':
            for row in rows:row['id']='new-'+row['id']
        else:
            for row in rows:row.update(stance='partial',value=None,source_ids=[])
        return response(*rows)
    calls.respond=answer
    run_history(reader)
    assert state.pending=={'history'} and not reader.batches[-1]['complete']
    data=history_payload(reader,engine)
    assert {s['id'] for s in data['source_spans']}=={'s1','s2','s3'}


def test_initially_acknowledged_conflict_never_triggers_a_forced_agreement_call():
    reader,engine,state,calls=personal_reader()
    def answer(index,data):
        result=response(*fragments())
        result['slots'][0]['unresolved']=['These memories do not establish a single whole account.']
        return result
    calls.respond=answer
    run_history(reader)
    assert len(calls.requests)==1 and engine.ledger.used['json_repairs']==0
    assert 'history' not in state.valid()[0] and state.pending=={'history'}
    assert {s['id'] for s in history_payload(reader,engine)['source_spans']}=={'s1','s2','s3'}


@pytest.mark.parametrize('budget',['json','llm','wire'])
def test_reconciliation_respects_existing_budget_and_wire_limits_without_discarding_sources(budget):
    reader,engine,state,calls=personal_reader(max_repairs_per_request=2)
    calls.respond=lambda index,data:response(*fragments())
    if budget=='json':engine.ledger.limits['json_repairs']=0
    if budget=='llm':engine.ledger.limits['llm']=2  # One request and one reserved audit.
    if budget=='wire':reader._repair_fits=lambda *args:False
    run_history(reader)
    assert len(calls.requests)==1 and engine.ledger.used['llm']==1
    assert state.pending=={'history'} and state.valid()[0]=={}
    assert {s['id'] for s in history_payload(reader,engine)['source_spans']}=={'s1','s2','s3'}


def test_preexisting_negative_evidence_obligation_survives_an_otherwise_successful_whole_review():
    reader,engine,state,calls=personal_reader(max_repairs_per_request=1)
    sid=engine.mapper.sources['s2']['source_id']
    prior={'id':'old-negative-issue','variable':'history','entity_scope':'user','time_scope':'history',
           'source_ids':[sid],'descriptions':['Check the original later negative qualification.'],
           'origins':['older-query']}
    reader.evidence_obligations[prior['id']]=deepcopy(prior)
    state.pending={'history'}
    calls.respond=lambda index,data:response(*(fragments() if index==1 else [
        observation(row['id'],['s1','s2'],value=['piano','painting']) for row in fragments()]))
    run_history(reader)
    assert state.valid()[0]['history']==['piano','painting']
    assert state.pending=={'history'}
    assert reader.evidence_obligations[prior['id']]==prior


def test_generic_conflicting_values_keep_the_existing_non_repair_behavior():
    reader,engine,state,calls=fixture()
    calls.respond=lambda index,data:response(*fragments())
    run_history(reader)
    assert len(calls.requests)==1 and engine.ledger.used['json_repairs']==0
    assert 'history' not in state.valid()[0]
    assert len(state.routes)==2
    assert 'whole_value_reviews' not in reader.batches[-1]
