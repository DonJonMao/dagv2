"""Recovery eligibility is scoped to actual evidence and each durable record."""
from copy import deepcopy

from dagbt.fact_state import FactState
from dagbt.joint_reading import JointReader
from test_personamem_joint_protocol import observation, response, visible
from test_v4_reconcile_review import personal_reader
from test_v4_failed_batch_review import failed_whole, whole_review_response


def test_two_failed_records_for_one_variable_release_only_the_explicitly_offered_record():
    reader,engine,state,calls=failed_whole(max_repairs=1)
    first=next(iter(reader.failed_batch_reviews.values()))
    second=deepcopy(first)
    second.update(id='independent-older-failure',query_identity='independent-old-query')
    reader.failed_batch_reviews[second['id']]=second
    marker=(second['query_identity'],second['batch_identity'])
    reader.failed_obligations[marker]={'history'}
    def review(index,data):
        offered=[record for record in data['failed_batch_reviews'] if record['review_allowed']]
        assert [record['id'] for record in offered]==[first['id']]
        return whole_review_response(data)
    calls.respond=review
    reader.read(engine.steps[0],'Review historical activities',['history'],['advice'],'new-proof')
    assert first['id'] not in reader.failed_batch_reviews
    assert reader.failed_batch_reviews[second['id']]==second
    assert reader.failed_obligations=={marker:{'history'}}
    assert state.pending=={'history'}


def test_identical_original_documents_under_a_new_query_do_not_restart_the_failure():
    reader,engine,state,calls=failed_whole(max_repairs=1)
    before=len(calls.requests)
    reader.read(engine.steps[0],'A differently worded query',['history'],['piano','painting'],'renamed')
    assert len(calls.requests)==before and state.pending=={'history'}


def test_unrelated_program_revision_does_not_unlock_a_failed_local_read():
    reader,engine,state,calls=failed_whole(max_repairs=1)
    before=len(calls.requests)
    state.revision+=1
    assert not reader.failed_review_ready(engine.steps[0],['history'])
    reader.read(engine.steps[0],'Review activities',['history'],[],'revision-only')
    assert len(calls.requests)==before and state.pending=={'history'}


def test_changed_actual_parent_binding_allows_a_paid_review_without_renaming_original_sources():
    _,engine,old_state,calls=personal_reader(max_repairs_per_request=1)
    steps=deepcopy(engine.steps)
    steps[0]['inputs']=['context']
    steps.insert(0,{'output_slot':'context','question':'Establish the historical context',
                    'inputs':[],'execution':'retrieval','answer_type':'fact'})
    program=deepcopy(old_state.program)
    program['variables']={'context':{'type':'text','demand':'context'},**program['variables']}
    state=FactState(program,old_state.contract,steps,old_state.final_node_id)
    engine.steps=steps
    engine.fact_state=state
    reader=JointReader(engine,state)
    source_id=engine.mapper.sources['s1']['source_id']
    state.integrate([{**observation('parent',[source_id],value='Initial local context'),'variable':'context'}])
    original_route=state.valid()[2]['context'][0]
    original_version=state.versions['context']
    def bad(index,data):
        result=response(observation('history-draft',['s1'],entity_scope='wrong',
                                    parent_versions={'context':original_version}))
        result['slots'].append({'variable':'context','observations':[],'unresolved':[]})
        return result
    calls.respond=bad
    reader.read(steps[1],'Read scoped history',['history'],['piano'],'initial-parent')
    assert reader.failed_obligations and state.pending=={'history'}
    counter={**observation('parent-change',[source_id],stance='contradiction',
                          retract_ids=[original_route]),'variable':'context','value':None}
    state.integrate([counter,{**observation('updated-parent',[source_id],value='Updated local context'),
                              'variable':'context'}],state.pending)
    assert state.versions['context']!=original_version
    before_calls=len(calls.requests)
    def corrected(index,data):
        result=response(observation('history-draft',['s1'],
                                    parent_versions={'context':state.versions['context']}))
        result['slots'][0]['resolved_failed_batch_ids']=[record['id'] for record in data['failed_batch_reviews']
                                                        if record['review_allowed']]
        result['slots'].append({'variable':'context','observations':[],'unresolved':[]})
        return result
    calls.respond=corrected
    reader.read(steps[1],'Read scoped history',['history'],[],'changed-actual-parent')
    assert len(calls.requests)==before_calls+1
    assert state.valid()[0]['history']==['piano'] and not state.pending
    assert not reader.failed_batch_reviews and engine.ledger.used['json_repairs']==2


def test_invalid_source_alias_can_be_corrected_to_an_actual_original_source():
    reader,engine,state,calls=personal_reader(max_repairs_per_request=1)
    state.integrate([reader._observation(observation('prior',['s1']),'history',visible(reader),engine.steps[0])])
    route=next(iter(state.routes))
    counter=observation('counter',['invented-source'],stance='contradiction',retract_ids=[route])
    counter['value']=None
    calls.respond=lambda index,data:response(counter)
    reader.read(engine.steps[0],'Review a contrary clue',['history'],['painting'],'bad-alias')
    assert state.pending=={'history'} and not state.routes[route]['revoked']
    def repair(index,data):
        result=response({**counter,'source_ids':['s2']})
        result['slots'][0]['resolved_failed_batch_ids']=[record['id'] for record in data['failed_batch_reviews']
                                                        if record['review_allowed']]
        return result
    calls.respond=repair
    reader.read(engine.steps[0],'Review counter from new text',['history'],['advice'],'fixed-alias')
    assert state.routes[route]['revoked'] and not state.pending


def test_paid_durable_review_remains_authorized_for_its_one_remaining_local_repair():
    reader,engine,state,calls=failed_whole(max_repairs=2)
    previous_calls=len(calls.requests)
    previous_repairs=engine.ledger.used['json_repairs']
    def review(index,data):
        assert any(record['review_allowed'] for record in data['failed_batch_reviews'])
        result=whole_review_response(data)
        if index==previous_calls+1:
            result['slots'][0]['observations'][0]['entity_scope']='mistyped-scope'
        else:
            assert 'repair_scope' in data
        return result
    calls.respond=review
    reader.read(engine.steps[0],'Review histories with new original text',['history'],['advice'],'deferred-repair')
    assert len(calls.requests)==previous_calls+2
    assert engine.ledger.used['json_repairs']==previous_repairs+2
    assert not state.pending and not reader.failed_batch_reviews
    assert state.valid()[0]['history']==['piano','painting']
