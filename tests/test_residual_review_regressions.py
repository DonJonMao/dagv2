"""Behavioral regressions for the four boundaries found in the a895695 review.

Services/capacity are explicit substitutes; Engine, FactState, JointReader,
the pausable BT adapter and the finite-world evaluator are real implementations.
"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import demo_residual_memory as demo
from dagbt.bridge import BridgeSession
from dagbt.engine import Engine
from dagbt.methods import RESIDUAL
from dagbt.reasoning import InputOverflow
from dagbt.residual import ProgramError,certify_output,possible_outputs,validate_value
from dagbt.residual_control import ResidualControl
from dagbt.transport import digest


def number_plan(domain=None):
    declaration={'type':'number','demand':'x'}
    if domain is not None:
        declaration.update(unit='s',domain=domain)
    return {'steps':[demo.step('x','Find x quantity')],'final_node_id':'x',
        'output_contract':demo.contract('value','number','numbers'),
        'program':{'variables':{'x':declaration},'expression':demo.var('x')}}


def engine(plan,passages,services=None,vectors=None,**settings):
    services=services or demo.SourcesOnly(plan,scenario='comparison')
    services.legacy=False;services.scored=False
    docs={d:SimpleNamespace(doc_id=d,passage=t) for d,t in passages.items()}
    vectors=np.array(vectors if vectors is not None else [[1,0,0]]*len(docs),np.float32)
    config={'_test_transport':True,'llm_base_url':'http://synthetic/v1','llm_model':'script',
        'embedding_base_url':'http://synthetic/v1','embedding_model':'script',
        'fusion':{'initial_width':1,'proposal_width':1,'max_repairs_per_request':0,**settings}}
    return Engine({'id':'review','question':'Review regression question'},
        (docs,list(docs),vectors,None,demo.Tokenizer()),services,config,RESIDUAL),services


def prepare(e):
    e.plan();e.fixed_pool=None
    e.bridge=BridgeSession(e.q['question'],e.docs,e.ids,e.vectors,e.tokenizer,e.calls,e.config,e.ledger)
    c=ResidualControl(e);e.residual_control=c
    return c


def local_read(c,task,docs):
    e=c.e;step=next(s for s in e.steps if s['output_slot']==task)
    query=c.grounded(step)
    parent_vars=[v for p in step['inputs'] for v in c.demand_variables(p)]
    identity={'algorithm':RESIDUAL,'query':query,'node_id':task,'program_revision':c.state.revision,
        'program_identity':digest(c.state.program),'initial_dag_identity':digest(e.steps),
        'bindings':{v:c.state.versions[v] for v in parent_vars},'contract_identity':digest(c.state.contract)}
    key=digest(identity)
    e.add_candidates(docs);c.local_pools[task]=list(docs)
    return step,query,key


def one_new_window(reader):
    # Permit one new source at a time while reproducing prior premise spans.
    reader._fits=lambda operation,data:sum((reader.current_identity,s['id']) not in reader.read_states
                                          for s in data['source_spans'])<=1


def test_single_and_multiple_successful_windows_have_same_answer_and_sources():
    results=[]
    for split in (False,True):
        e,_=engine(number_plan(),{'clue':'No value here.','x':'x = 7.'})
        c=prepare(e);step,query,key=local_read(c,'x',e.ids)
        if split:one_new_window(c.reader)
        assert c.reader.read(step,query,['x'],e.ids,key)
        assert not c.state.pending
        assert not c.reader.unread_obligations and not c.reader.failed_obligations
        assert all(b['complete'] and not b['failed_slots'] for b in c.reader.batches)
        cert=c.candidate();assert cert is not None and cert['output']==7
        results.append((cert['output'],cert['source_ids']))
        assert len(c.reader.batches)==(2 if split else 1)
    assert results[0]==results[1]


def test_failed_window_is_not_cleared_by_later_successful_window():
    class FailedFirst(demo.SourcesOnly):
        def get(self,stage,url,payload):
            response=super().get(stage,url,payload)
            if stage[0]=='joint_read':
                data=json.loads(payload['messages'][1]['content'])
                if not any(s['doc_id']=='x' for s in data['source_spans']):
                    value=json.loads(response['response']['choices'][0]['message']['content'])
                    value['slots'][0]['observations'][0]['source_ids']=['invisible']
                    response['response']['choices'][0]['message']['content']=json.dumps(value)
            return response
    plan=number_plan();e,_=engine(plan,{'clue':'No value here.','x':'x = 7.'},FailedFirst(plan,scenario='comparison'))
    c=prepare(e);step,query,key=local_read(c,'x',e.ids);one_new_window(c.reader)
    c.reader.read(step,query,['x'],e.ids,key)
    assert c.reader.failed_obligations and c.reader.unread_obligations
    c.reader.read(step,query,['x'],e.ids,key)
    assert c.state.valid()[0]['x']==7
    assert not c.reader.unread_obligations and c.reader.failed_obligations
    assert c.state.pending=={'x'} and c.candidate() is None


@pytest.mark.parametrize('failed',[False,True])
def test_checkpoint_preserves_obligation_kind_and_clears_only_completed_windows(failed):
    plan=number_plan();e,services=engine(plan,{'clue':'No value here.','x':'x = 7.'})
    c=prepare(e);step,query,key=local_read(c,'x',e.ids)
    one_new_window(c.reader);fits=c.reader._fits
    c.reader._fits=lambda operation,data:fits(operation,data) and not c.reader.batches
    with pytest.raises(InputOverflow):c.reader.read(step,query,['x'],e.ids,key)
    assert c.state.pending=={'x'} and c.reader.unread_obligations
    if failed:
        c.reader.failed_obligations[key,'failed-counter']={'x'}
    snapshot=json.loads(json.dumps(c.snapshot()))
    fresh,new_services=engine(plan,{'clue':'No value here.','x':'x = 7.'})
    result=fresh.resume(snapshot)
    assert not any(r['stage'][0]=='planner' for r in new_services.requests)
    assert not fresh.residual_control.reader.unread_obligations
    if failed:
        assert fresh.fact_state.pending=={'x'} and result['answer']['prediction'] is None
    else:
        assert not fresh.fact_state.pending and result['answer']['semantic_answer']==7
        assert result['answer']['status']=='ok'


@pytest.mark.parametrize('value',[180,{'number':3,'unit':'min'},
    {'lower':61,'upper':180,'unit':'s'}, {'number':60,'unit':'s'}])
def test_numeric_domain_conflict_never_certifies(value):
    p=number_plan({'lower':0,'upper':60,'upper_closed':False})['program']
    p['expression']=demo.op('le',demo.var('x'),demo.const({'number':1,'unit':'min'}))
    with pytest.raises(ProgramError,match='legal domain'):validate_value(value,p['variables']['x'])
    outcome=possible_outputs(p,{'x':value})
    assert outcome['status']=='inconsistent' and not outcome['nonempty_proven']
    assert certify_output(p,{'x':value},demo.contract('value','bool')) is None


def test_numeric_overlap_is_intersected_before_reduction_with_units_and_open_bounds():
    p=number_plan({'lower':0,'upper':60,'upper_closed':False})['program']
    p['expression']=demo.op('lt',demo.var('x'),demo.const({'number':1,'unit':'min'}))
    value={'lower':0.5,'upper':2,'unit':'min'}
    normalized=validate_value(value,p['variables']['x'])
    assert normalized=={'lower':30.,'upper':60.,'lower_closed':True,'upper_closed':False,'unit':'s'}
    assert certify_output(p,{'x':value},demo.contract('value','bool'))['output'] is True
    with pytest.raises(ProgramError,match='incompatible'):
        validate_value({'number':30,'unit':'m'},p['variables']['x'])


def test_acceptance_threshold_is_not_a_legal_domain_and_invalid_batches_are_atomic():
    p=number_plan()['program'];p['variables']['x']['unit']='s'
    p['expression']=demo.op('le',demo.var('x'),demo.const({'number':60,'unit':'s'}))
    assert certify_output(p,{'x':180},demo.contract('value','bool'))['output'] is False
    e,_=engine(number_plan({'lower':0,'upper':60}),{'x':'x = 180.'})
    c=prepare(e)
    def row(value):
        return {'variable':'x','stance':'support','value':value,'source_ids':['s'],
            'entity_scope':'numbers','time_scope':'本次演示','parent_versions':{},'retract_ids':[],'reason':'test'}
    with pytest.raises(ProgramError):c.state.integrate([row(7),row(180)])
    assert not c.state.routes
    result=c.run()
    assert result['answer']['prediction'] is None and 'x' not in c.state.valid()[0]


def test_unit_declared_on_numeric_domain_is_respected_by_bare_observations():
    p={'variables':{'x':{'type':'number','domain':{'lower':0,'upper':60,'unit':'s'}}},
       'expression':demo.op('le',demo.var('x'),demo.const({'number':1,'unit':'min'}))}
    assert validate_value(7,p['variables']['x'])=={'number':7,'unit':'s'}
    assert certify_output(p,{'x':7},demo.contract('value','bool'))['output'] is True
    assert possible_outputs(p,{'x':180})['status']=='inconsistent'


@pytest.mark.parametrize('mode',['second_root','all_empty','budget','service_failure'])
def test_empty_first_root_does_not_close_a_live_frontier(monkeypatch,mode):
    class Anchors(demo.SourcesOnly):
        def get(self,stage,url,payload):
            if url.endswith('/embeddings') and 'Anchor passages:\n[r2]' in payload['input'][0]:
                self.requests.append({'stage':list(stage),'url':url,'payload':deepcopy(payload)})
                return {'response':{'data':[{'index':0,'embedding':[0.,1.,0.]}]}}
            return super().get(stage,url,payload)
    original=BridgeSession._step_retriever
    def retriever(*args,**kwargs):
        r=original(*args,**kwargs);propose=r.propose
        def probe(target,*a,**kw):
            batch=propose(target,*a,**kw)
            if mode=='service_failure':raise ConnectionError('review synthetic bridge failure')
            if target=='r1' or mode=='all_empty':
                batch=replace(batch,hits=())
                r._batches[-1]=batch
            return batch
        r.propose=probe
        return r
    monkeypatch.setattr(BridgeSession,'_step_retriever',retriever)
    plan=number_plan();e,_=engine(plan,{'r1':'First clue.','r2':'Second clue.','x':'x = 7.'},
        Anchors(plan,scenario='comparison'),[[1,0,0],[.9,0,0],[0,1,0]],
        initial_width=2,ann_calls=2 if mode=='budget' else 8,reserved_gap_ann_calls=0)
    result=e.run();traces=e.bridge.traces
    if mode=='second_root':
        assert result['answer']['status']=='ok' and result['answer']['semantic_answer']==7
        assert traces[1]['search_status']=='empty_probe' and traces[1]['frontier_available']
        assert traces[2]['retrieval']['proposal_batches'][-1]['target_id']=='r2'
        assert result['diagnostics']['cost']['ann_calls']==3
    else:
        assert result['answer']['prediction'] is None
        assert traces[-1]['search_status']=={'all_empty':'frontier_exhausted','budget':'budget_exhausted',
            'service_failure':'service_failed'}[mode]
        assert result['diagnostics']['cost']['ann_calls']<=3


def completeness_plan(mode):
    return {'steps':[demo.step('rules','Find applicable rules'),demo.step('C','Check {rules}',['rules']),
        demo.step('decision','Apply all required conditions',['rules','C'],'compose')],
        'final_node_id':'decision','output_contract':demo.contract(mode,'set' if mode=='all_failures' else 'record'),
        'program':{'variables':{'rule':{'type':'rule','demand':'rules'},'result':{'type':'record','demand':'decision'}},
            'expression':demo.var('result'),'rules_complete':False,
            'rule_binding':{'variable':'rule','condition_demands':['C']}}}


class CompletenessServices(demo.SourcesOnly):
    def __init__(self,plan,flag):
        super().__init__(plan,scenario='completeness');self.flag=flag

    def get(self,stage,url,payload):
        if stage[0]=='planner':return super().get(stage,url,payload)
        self.requests.append({'stage':list(stage),'url':url,'payload':deepcopy(payload)})
        if url.endswith('/embeddings'):
            axis=1 if 'complete rule collection' in payload['input'][0] else 2
            return {'response':{'data':[{'index':0,'embedding':[float(i==axis) for i in range(3)]}]}}
        data=json.loads(payload['messages'][1]['content']);docs={s['doc_id']:s for s in data['source_spans']}
        slots=[]
        for v in data['variables']:
            rows=[]
            def observation(stance,value,doc,retract=()):
                return {'id':v+'-'+stance,'stance':stance,'value':value,'source_ids':[docs[doc]['id']],
                    'entity_scope':data['required_scope']['entity_scope'],'time_scope':data['required_scope']['time_scope'],
                    'parent_versions':{p:x['version'] for p,x in data['supported_parents'].items()} if v=='flag' else {},
                    'retract_ids':list(retract),'reason':'Explicit scripted original evidence'}
            if not stage[0].startswith('audit_joint'):
                if v=='rule' and ('complete' in docs or 'incomplete' in docs):
                    complete='complete' in docs
                    value={'variables':{'flag':{'type':'bool','demand':'C'}},
                        'expression':{'op':'failures' if self.plan['output_contract']['mode']=='all_failures' else 'one_reason',
                            'fields':{'C':demo.var('flag')}},'constraints':[],'rules_complete':complete}
                    old=[r['id'] for r in data['known_observations'] if r['variable']==v and not r['revoked'] and r['value']!=value]
                    if old:rows.append(observation('contradiction',None,'complete',old))
                    rows.append(observation('support',value,'complete' if complete else 'incomplete'))
                elif v=='flag' and 'condition' in docs:
                    rows.append(observation('support',self.flag,'condition'))
            slots.append({'variable':v,'observations':rows,'unresolved':[]})
        value={'slots':slots}
        if stage[0].startswith('audit_joint'):value['audit_verdict']='supported'
        return {'response_ref':'review-script-'+str(len(self.requests)),
            'response':{'choices':[{'finish_reason':'stop','message':{'content':json.dumps(value)}}]}}


@pytest.mark.parametrize('mode,flag',[('one_reason',True),('all_failures',True),('one_reason',False)])
def test_incomplete_rules_remain_a_demand_except_for_sufficient_negative_witness(mode,flag):
    plan=completeness_plan(mode);service=CompletenessServices(plan,flag)
    e,_=engine(plan,{'incomplete':'Known condition: flag; completeness unknown.',
        'complete':'Authoritative complete rules: flag only; no other conditions.',
        'condition':'Flag is '+str(flag)},service,np.eye(3,dtype=np.float32))
    c=prepare(e)
    for task,docs,variables in [('rules',['incomplete'],['rule']),('C',['condition'],['flag'])]:
        step,query,key=local_read(c,task,docs);c.reader.read(step,query,variables,docs,key)
    assert not c.state.program['rules_complete'] and c.state.valid()[0]['flag'] is flag
    outcome,active=c.update();assert outcome['status']=='determined'
    before=len(service.requests)
    if flag:
        assert c.candidate() is None and 'rules' in active
    else:
        assert c.candidate() is not None and 'rules' not in active
    result=c.run()
    assert result['answer']['status']=='ok'
    probes=[r for r in service.requests[before:] if r['url'].endswith('/embeddings')]
    if flag:
        assert probes and 'complete rule collection' in probes[0]['payload']['input'][0]
        assert c.state.program['rules_complete'] and c.state.revision>1
        assert result['answer']['semantic_answer']==([] if mode=='all_failures' else {'decision':True,'reason':None})
    else:
        assert not probes and result['answer']['semantic_answer']=={'decision':False,'reason':'C'}
