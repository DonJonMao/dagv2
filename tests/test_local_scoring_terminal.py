"""Actual payloads and execution contracts for the versioned local method."""
from copy import deepcopy
from itertools import combinations
import json

import pytest

from dagbt import local_terminal
from dagbt.config import resolve
from dagbt.engine import run_question, Engine, ProtocolError
from vendor.bridgetree.dependency_scoring import SetBudgetExceeded
from test_bridge import session, Calls
from test_engine import setup, FakeCalls, answer, unknown, step

VERSION = local_terminal.VERSION


class LocalCalls(Calls):
    def get(self, stage, url, payload):
        if url.endswith('/rerank'):
            self.requests.append((stage, url, deepcopy(payload)))
            base = {'Q':.1, 'q1':.2, 'q2':.3}.get(payload['query'], .4)
            return {'response':{'results':[
                {'index':i,'relevance_score':base+.01*t.count('[Passage ')}
                for i,t in enumerate(payload['documents'])]}}
        return super().get(stage,url,payload)
    post_rerank = get


def local_session(**kw):
    return session(calls=LocalCalls(), algorithm_version=VERSION, **kw)


def test_payload_queries_and_cache_namespaces_are_local():
    s = local_session()
    first = s._score_backend('q1', {'parent_bindings':[]})
    second = s._score_backend('q2', {'parent_bindings':[]})
    assert first.score_sets([['d0']]) == pytest.approx([.21])
    assert second.score_sets([['d0']]) == pytest.approx([.31])
    assert s.ledger.used['set_score'] == 2
    assert s._score_backend('q1', {'parent_bindings':[]}) is first
    assert first.score_sets([['d0']]) == pytest.approx([.21])
    assert s.ledger.used['set_score'] == 2
    assert first.namespace_hash != second.namespace_hash
    assert [p['query'] for _,u,p in s.calls.requests if u.endswith('/rerank')] == ['q1','q2']
    assert first.reranker is second.reranker


@pytest.mark.parametrize('binding', [
    {'answer':'another university','version':1,'applicable_scope':'2012'},
    {'answer':'school','version':2,'applicable_scope':'2012'},
    {'answer':'school','version':1,'applicable_scope':'2013'},
])
def test_parent_semantics_or_version_change_invalidates_cache(binding):
    s = local_session()
    original = {'parent_bindings':[{'answer':'school','version':1,'applicable_scope':'2012'}]}
    a = s._score_backend('q1',original)
    b = s._score_backend('q1',{'parent_bindings':[binding]})
    a.score_sets([['d0']]); b.score_sets([['d0']])
    assert a.namespace_hash != b.namespace_hash
    assert s.ledger.used['set_score'] == 2
    assert a.query == b.query == 'q1'


def test_corpus_content_and_model_identity_invalidate_registry():
    from vendor.bridgetree.types import Memory
    s = local_session()
    a = s._score_backend('q1')
    m = s.records['d0']
    s.records['d0'] = Memory(m.memory_id, 'changed text', m.timestamp, m.source_id, m.metadata)
    b = s._score_backend('q1')
    assert a.namespace_hash != b.namespace_hash
    # A deployment must not silently mutate a shared client under existing scorers.
    assert a.records['d0'].text != b.records['d0'].text
    s.config['reranker']['model'] = 'new-pointwise-model'
    c = s._score_backend('q1')
    assert c.namespace_hash != b.namespace_hash
    a.score_sets([['d0']]); c.score_sets([['d0']])
    assert [p['model'] for _,u,p in s.calls.requests if u.endswith('/rerank')] == ['test-pointwise','new-pointwise-model']
    assert len({stage for stage,_,_ in s.calls.requests}) == len(s.calls.requests)


def test_512_is_global_not_per_scorer_and_preflight_is_atomic():
    s = local_session(sets=512)
    sets = [list(c) for n in range(7) for c in combinations(s.ids,n)]
    for i in range(8):
        sc = s._score_backend(f'task-{i}')
        sc.score_sets(sets)
        sc.score_sets(sets)
    assert s.ledger.used['set_score'] == 512
    next_sc = s._score_backend('task-9')
    before = len(s.calls.requests)
    with pytest.raises(SetBudgetExceeded):
        next_sc.score_sets([[],['d0'],['d1'],['d0','d1']])
    assert len(s.calls.requests) == before
    assert s.ledger.used['set_score'] == 512
    total = s.public_dict()['scorer_cost_total']
    assert total['scored_sets'] == 512
    assert total['memory_cache_hits'] == 512
    assert s.public_dict()['scorer_cost_total'] == total
    stages = [r[0] for r in s.calls.requests]
    assert len(set(stages)) == len(stages)


def test_real_search_records_context_and_unique_requests_across_nodes():
    s = local_session(sets=128)
    for node,query in [('a','q1'),('b','q2'),('a','q1')]:
        s.discover(query,node,remaining_nodes=2)
    assert len(s.scorers) == 2
    for trace in s.traces:
        assert trace['scoring_query'] == trace['query']
        for measured in trace.get('measured_sets_cumulative',[]):
            assert measured['node_id'] == trace['node_id']
            assert measured['scoring_context_id'] == trace['scoring_context_id']
            assert measured['scoring_query'] == trace['query']
    assert sum(c.scored_sets for c in s.scorers.values()) == s.ledger.used['set_score']
    stages = [r[0] for r in s.calls.requests]
    assert len(stages) == len(set(stages))


class TerminalCalls(FakeCalls):
    def __init__(self, steps, final, resolver, **kw):
        super().__init__(steps,resolver,**kw)
        self.final = final

    def get(self, stage, url, payload):
        if stage[0] == 'planner':
            self.counts['planner'] += 1
            self.requests.append({'operation':'planner','payload':deepcopy(payload)})
            return {'response_ref':'planner','response':{'choices':[{'finish_reason':'stop',
                'message':{'content':json.dumps({'steps':self.steps,'final_node_id':self.final})}}]}}
        return super().get(stage,url,payload)


def local_step(nid, question='Resolve requested relation?', inputs=(), execution='retrieval'):
    return {**step(nid,question,inputs), 'execution':execution}


def test_chain_publishes_existing_terminal_without_baseline_selector_or_reader(setup):
    bridge, resources, config = setup
    bridge.routes = {'director':['a'], 'university':['b'], 'city':['c']}
    steps = [local_step('director','Who directed the 2012 film?'),
             local_step('university','Which university did {director} graduate from?',['director']),
             local_step('city','Which city hosts {university}?',['university'])]
    def resolver(data):
        nid = data['node_id']
        return answer(data, {'director':'Lin Zhou','university':'North University','city':'River City'}[nid],
                      [{'director':'a','university':'b','city':'c'}[nid]],
                      {'director':[],'university':['director'],'city':['university']}[nid])
    calls = TerminalCalls(steps,'city',resolver)
    row = run_question({'id':'q','question':'Which city hosts the university of the film director?'},resources,calls,config,VERSION)
    assert row['answer']['status'] == 'ok'
    assert row['answer']['prediction'] == 'River City'
    assert row['answer']['answer_source'] == 'dag_terminal'
    assert row['answer']['dependency_versions'] == {'city':1,'university':1,'director':1}
    assert set(row['answer']['sources']['doc_ids']) == {'a','b','c'}
    assert [c['node_id'] for c in bridge.calls] == ['director','university','city']
    assert calls.counts['resolve'] == 3
    assert calls.counts['planner'] == calls.counts['audit'] == 1
    assert not calls.counts['reader'] and not calls.counts['select']
    assert row['diagnostics']['ledger']['used'].get('reader',0) == 0
    assert 'budgets' in row and row['budgets'] == {}
    assert bridge.calls[1]['scoring_identity']['parent_bindings'][0]['answer'] == 'Lin Zhou'


def test_single_real_node_equal_to_Q_is_allowed(setup):
    bridge, resources, config = setup
    bridge.routes = {'fact':['c']}
    calls = TerminalCalls([local_step('fact','Q')],'fact',lambda d:answer(d,'result',['c']))
    row = run_question({'id':'q','question':'Q'},resources,calls,config,VERSION)
    assert row['answer']['status'] == 'ok'
    assert [(c['node_id'],c['query']) for c in bridge.calls] == [('fact','Q')]


def test_AND_compose_waits_for_both_parents_without_Q_retrieval(setup):
    bridge, resources, config = setup
    bridge.routes = {'p1':['a'],'p2':['b']}
    # Storage order and ID order deliberately do not identify the terminal.
    steps = [local_step('aa_final','Compare both facts',['p1','p2'],'compose'),
             local_step('p2','Fact two'),local_step('p1','Fact one')]
    def resolver(d):
        if d['node_id'] == 'aa_final':
            assert {n['id'] for n in d['supported_parents']} == {'p1','p2'}
            return answer(d,'combined',[],['p1','p2'])
        return answer(d,d['node_id'],['a' if d['node_id']=='p1' else 'b'])
    calls = TerminalCalls(steps,'aa_final',resolver)
    row = run_question({'id':'q','question':'Compare both facts'},resources,calls,config,VERSION)
    assert row['answer']['status'] == 'ok'
    assert {c['node_id'] for c in bridge.calls} == {'p1','p2'}
    assert row['answer']['final_node_id'] == 'aa_final'
    assert calls.counts['resolve'] == 3


def test_compose_does_not_execute_with_missing_parent(setup):
    bridge, resources, config = setup
    bridge.routes = {'p1':['a'],'p2':['b']}
    steps = [local_step('p1'),local_step('p2'),local_step('final',inputs=['p1','p2'],execution='compose')]
    calls = TerminalCalls(steps,'final',lambda d:unknown() if d['node_id']=='p2' else answer(d,'p1',['a']))
    row = run_question({'id':'q','question':'Q'},resources,calls,config,VERSION)
    assert row['answer']['status'] == 'unknown'
    assert row['answer']['prediction'] is None
    assert calls.counts['resolve'] == 2


def test_options_only_enter_terminal_generation_and_format_is_strict(setup):
    bridge, resources, config = setup
    bridge.routes = {'parent':['a'],'final':['c']}
    steps = [local_step('parent'),local_step('final',inputs=['parent'])]
    def resolver(d):
        return answer(d,'city',['c'],['parent'],final_prediction='b') if d['node_id']=='final' else answer(d,'person',['a'])
    calls = TerminalCalls(steps,'final',resolver)
    public = 'Private option sentinel: (a) hill (b) city (c) sea'
    row = run_question({'id':'q','question':'Where?'},resources,calls,config,VERSION,
                       reader_question=public,output_options=['hill','city','sea'])
    assert row['answer']['status'] == 'ok' and row['answer']['prediction'] == 'b'
    for request in calls.requests:
        content = request['payload']['messages'][1]['content']
        data = json.loads(content)
        is_terminal = request['operation']=='resolve' and data.get('node_id')=='final'
        assert ('Private option sentinel' in content) == is_terminal
    assert all('Private option sentinel' not in c['query'] for c in bridge.calls)


def test_bad_format_never_falls_back_to_semantic_answer_or_Reader(setup):
    bridge, resources, config = setup
    bridge.routes = {'final':['c']}
    config['fusion']['max_repairs_per_request'] = 0
    calls = TerminalCalls([local_step('final')],'final',lambda d:answer(d,'city',['c'],final_prediction='b or c'))
    row = run_question({'id':'q','question':'Where?'},resources,calls,config,VERSION,
                       reader_question='(a) hill (b) city (c) sea',output_options=['hill','city','sea'])
    assert row['answer']['status'] == 'invalid_output_format' and row['answer']['prediction'] is None
    assert calls.counts['reader'] == calls.counts['select'] == 0


def test_config_new_contract_and_legacy_reader_contract():
    new = resolve({},VERSION)
    assert new['reader_calls'] == 0
    assert [new[k] for k in ('ann_calls','set_score_calls','llm_calls')] == [36,512,24]
    assert 'final_selection_calls' not in new and 'reserved_selection_repairs' not in new
    with pytest.raises(ValueError,match='reader_calls'):
        resolve({'fusion':{'reader_calls':0}},'fusion')


def test_quartet_uses_one_fixed_query_context_model_and_original_arithmetic():
    s = local_session()
    sc = s._score_backend('q1', {'parent_bindings':[{'version':3,'applicable_scope':'2012'}]})
    def backend(stage,url,payload):
        s.calls.requests.append((stage,url,deepcopy(payload)))
        values = {sc.serialize_set([]):.10,sc.serialize_set(['d0']):.25,
                  sc.serialize_set(['d1','d2']):.35,sc.serialize_set(['d0','d1','d2']):.90}
        return {'response':{'results':[{'index':i,'relevance_score':values[t]} for i,t in enumerate(payload['documents'])]}}
    s.calls.post_rerank = backend
    quartet = sc.score_activation('d0',[],['d1','d2'])
    assert (quartet.P,quartet.Pe,quartet.PG,quartet.PGe) == pytest.approx((.1,.25,.35,.9))
    assert quartet.PGe-quartet.PG == pytest.approx(.9-.35)  # M; context_marginal is the group's own marginal
    assert quartet.activation == pytest.approx((.9-.35)-(.25-.1))
    requests = [p for _,_,p in s.calls.requests]
    assert all(p['query']=='q1' and p['model']=='test-pointwise' for p in requests)
    events = [e for e in s.ledger.events if e['event']=='bridge_rerank_request']
    assert {e['scoring_context_id'] for e in events} == {sc.namespace_hash}
    assert {e['scoring_query'] for e in events} == {'q1'}
    before = s.ledger.used['set_score']
    sc.score_activation('d0',[],['d1','d2'])
    assert s.ledger.used['set_score'] == before == 4


def test_transport_cache_is_context_and_algorithm_scoped(tmp_path):
    from dagbt.transport import Transport,request_identity,save,digest
    from dagbt.budget import Ledger
    payload = {'query':'same query','documents':['same passage'],'top_n':1,'return_documents':False}
    config = {'reranker':{'url':'http://offline/rerank'},'fusion':{'algorithm_version':VERSION}}
    ids = [request_identity('q',config['reranker']['url'],payload,config,c) for c in ['one','two']]
    assert digest(ids[0]) != digest(ids[1])
    assert digest(ids[0]) != digest(request_identity('q',config['reranker']['url'],payload,{},'one'))
    for identity,score in zip(ids,[.2,.8]):
        save(tmp_path/'requests'/(digest(identity)+'.json'),{**identity,'attempts':[],
            'response':{'results':[{'index':0,'relevance_score':score}]}})
    calls = Transport('q',tmp_path,config,Ledger({}),None)
    for c,score in [('one',.2),('two',.8)]:
        response = calls.post_rerank(('rerank',c),config['reranker']['url'],payload,scoring_context_id=c)
        assert response['response']['results'][0]['relevance_score'] == score


def test_audit_invalidates_actual_parent_and_blocks_stale_terminal(setup):
    bridge,resources,config = setup
    bridge.routes = {'parent':['a'],'final':['c']}
    steps = [local_step('parent'),local_step('final',inputs=['parent'])]
    def resolver(d):
        return answer(d,'city',['c'],['parent']) if d['node_id']=='final' else answer(d,'person',['a'])
    def audit(d):
        parent = next(n for n in d['nodes'] if n['id']=='parent')
        source = next(s for s in d['evidence'] if s['doc_id']=='a')
        return {'conflicts':[{'alternative_ids':[parent['alternatives'][0]['id']],
            'span_ids':[source['id']],'reason':'entity binding contradicted','entity_scope':'fixture','event_time':None}],
            'unresolved_guards':[]}
    calls = TerminalCalls(steps,'final',resolver,audit=audit)
    row = run_question({'id':'q','question':'Q'},resources,calls,config,VERSION)
    assert row['answer']['status'] != 'ok' and row['answer']['prediction'] is None
    assert row['answer']['sources'] is None
    assert calls.counts['reader'] == calls.counts['select'] == 0


def test_independent_alternative_survives_parent_conflict(setup):
    bridge,resources,config = setup
    bridge.routes = {'parent':['a'],'final':['b','c']}
    steps = [local_step('parent'),local_step('final',inputs=['parent'])]
    def resolver(d):
        if d['node_id']=='final':
            return answer(d,'city',[],alternatives=[(['b'],['parent']),(['c'],[])])
        return answer(d,'person',['a'])
    def audit(d):
        parent = next(n for n in d['nodes'] if n['id']=='parent')
        return {'conflicts':[{'alternative_ids':[parent['alternatives'][0]['id']],
            'span_ids':[next(s['id'] for s in d['evidence'] if s['doc_id']=='a')],
            'reason':'one route contradicted','entity_scope':'fixture','event_time':None}], 'unresolved_guards':[]}
    calls = TerminalCalls(steps,'final',resolver,audit=audit)
    row = run_question({'id':'q','question':'Q'},resources,calls,config,VERSION)
    assert row['answer']['status'] == 'ok'
    assert row['answer']['sources']['doc_ids'] == ['c']
    assert row['answer']['dependency_versions'] == {'final':1}
    assert calls.counts['resolve'] == 3  # parent audit repair; terminal still runs once


def test_changed_output_options_cannot_publish_previous_prediction(setup):
    bridge,resources,config = setup
    bridge.routes = {'final':['c']}
    calls = TerminalCalls([local_step('final')],'final',lambda d:answer(d,'city',['c'],final_prediction='b'))
    engine = Engine({'id':'q','question':'Q'},resources,calls,config,VERSION,
                    reader_question='(a) sea (b) city',output_options=['sea','city'])
    assert engine.run()['answer']['status'] == 'ok'
    engine.output_identity = 'changed_public_options'
    row = engine.publish_terminal()
    assert row['answer']['status'] == 'invalid_output_format'
    assert row['answer']['prediction'] is None
    assert calls.counts['resolve'] == 1


def test_audit_failure_is_not_success_even_with_supported_terminal(setup):
    bridge,resources,config = setup
    bridge.routes = {'final':['c']}
    config['fusion']['max_repairs_per_request'] = 0
    calls = TerminalCalls([local_step('final')],'final',lambda d:answer(d,'city',['c']),audit=lambda d:{'conflicts':'bad'})
    row = run_question({'id':'q','question':'Q'},resources,calls,config,VERSION)
    assert row['answer']['status'] == 'audit_incomplete'
    assert row['answer']['prediction'] is None


def test_exhaustion_keeps_terminal_incomplete_without_extra_reader_budget(setup):
    bridge,resources,config = setup
    bridge.routes = {'final':['c']}
    config['fusion']['llm_calls'] = 2  # planner then required audit; no answer allowance
    calls = TerminalCalls([local_step('final')],'final',lambda d:answer(d,'city',['c']))
    row = run_question({'id':'q','question':'Q'},resources,calls,config,VERSION)
    assert row['answer']['status'] == 'budget_exhausted'
    assert row['answer']['prediction'] is None
    assert row['diagnostics']['ledger']['used']['llm'] <= 2
    assert calls.counts['reader'] == calls.counts['select'] == 0


@pytest.mark.parametrize('status', ['unknown','partial','ambiguous'])
def test_non_success_terminal_does_not_publish_nonempty_hypothesis(setup,status):
    bridge,resources,config=setup
    bridge.routes={'final':['c']}
    value=unknown()
    value.update(status=status,answer=None if status=='unknown' else 'hypothesis')
    calls=TerminalCalls([local_step('final')],'final',lambda d:value)
    row=run_question({'id':'q','question':'Q'},resources,calls,config,VERSION)
    assert row['answer']['status']==status
    assert row['answer']['prediction'] is None


@pytest.mark.parametrize('bad', ['missing_final','compose_no_parents','cycle','seven_nodes'])
def test_local_plan_contract_rejects_invalid_metadata(setup,bad):
    _,resources,config=setup
    e=Engine({'id':'q','question':'Q'},resources,TerminalCalls([],None,lambda d:unknown()),config,VERSION)
    plan={'steps':[local_step('final')],'final_node_id':'final'}
    if bad=='missing_final':plan['final_node_id']='absent'
    elif bad=='compose_no_parents':plan['steps'][0]['execution']='compose'
    elif bad=='cycle':
        plan['steps']=[local_step('p',inputs=['p']),local_step('final',inputs=['p'])]
    else:plan['steps']=[local_step('n'+str(i)) for i in range(7)];plan['final_node_id']='n6'
    with pytest.raises(ValueError):
        local_terminal.validate_plan(plan,e.repair.validate_plan)


def test_missing_mapping_stays_unassessed_without_raw_review_rescue(setup):
    bridge,resources,config=setup
    bridge.routes={'final':['c']}
    config['fusion']['max_repairs_per_request']=0
    calls=TerminalCalls([local_step('final')],'final',lambda d:unknown(),mapper=lambda d:{'units':[]})
    row=run_question({'id':'q','question':'Q'},resources,calls,config,VERSION)
    assert row['answer']['status']=='mapping_incomplete'
    assert row['answer']['prediction'] is None
    assert row['diagnostics']['reliability']['mapping_incomplete']
    assert row['diagnostics']['candidate_doc_ids']==['c']
    assert calls.counts['select']==calls.counts['reader']==0


def test_mapping_reserves_real_terminal_and_audit_calls(setup):
    bridge,resources,config=setup
    bridge.routes={'final':['c']}
    config['fusion']['llm_calls']=4
    calls=TerminalCalls([local_step('final')],'final',lambda d:answer(d,'city',['c']))
    row=run_question({'id':'q','question':'Q'},resources,calls,config,VERSION)
    assert row['answer']['status']=='ok'
    assert calls.counts=={'planner':1,'map':1,'resolve':1,'audit':1}
    assert row['diagnostics']['ledger']['used']['llm']==4


def test_generation_rejects_injected_gold_fields_before_any_calls(setup):
    _,resources,config=setup
    calls=TerminalCalls([local_step('final')],'final',lambda d:unknown())
    with pytest.raises(ProtocolError,match='only id/question'):
        run_question({'id':'q','question':'Q','correct_answer':'a'},resources,calls,config,VERSION)
    assert calls.requests==[]
