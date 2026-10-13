"""Offline behavioral regressions for v4 control/publication boundaries.

The real planner validator, reader, fact state, controller and bridge run against
scripted service responses. No evaluation labels or external services are used.
"""
from copy import deepcopy
import json

import numpy as np
import pytest

from test_residual_review_regressions import demo, engine, prepare, number_plan
from dagbt.engine import Engine
from dagbt.residual_control import ResidualControl


def advice_plan(kind='text'):
    return {'steps':[demo.step('preference','Find x preference quantity'),
        demo.step('advice','Recommend from {preference}',['preference'],'compose')],
        'final_node_id':'advice','output_contract':demo.contract('semantic',kind,'visitor'),
        'program':{'variables':{'x':{'type':'number','demand':'preference'},
            'advice':{'type':kind,'demand':'advice'}},'expression':demo.var('advice')}}


def decoded_request(payload):
    data=json.loads(payload['messages'][1]['content'])
    if 'source_span_columns' in data:
        columns=data.pop('source_span_columns')
        data['source_spans']=[dict(zip(columns,row)) for row in data['source_spans']]
    documents=data.pop('source_documents',{})
    for source in data['source_spans']:
        if documents:
            source.update(deepcopy(documents[source['doc_id']]))
    return data


class ControlServices(demo.SourcesOnly):
    def __init__(self,plan,mode='normal',value=None):
        super().__init__(plan,scenario='advice')
        self.mode=mode;self.value=value
        self.compositions=0;self.audits=0

    def get(self,stage,url,payload):
        result=super().get(stage,url,payload)
        if stage[0]=='planner' or url.endswith('/embeddings'):
            return result
        data=decoded_request(payload)
        message=result['response']['choices'][0]['message']
        value=json.loads(message['content'])
        if stage[0]=='semantic_compose':
            self.compositions+=1
            if self.mode=='unresolved':
                for slot in value['slots']:
                    slot['observations']=[]
                    slot['unresolved']=['Visible history contains no supported response.']
                    slot.pop('resolved_obligation_ids',None)
                value['semantic_answer']=None;value['final_prediction']=None
            else:
                terminal_value=(self.value if self.value is not None else
                    'Choose quiet activity '+str(self.compositions)
                    if self.mode in ('correct','always_conflict','supported_but_changed') else
                    'Choose a quiet visit based on the sourced preference.')
                for slot in value['slots']:
                    if slot['variable']=='advice':
                        for row in slot['observations']:
                            row['value']=deepcopy(terminal_value)
                value['semantic_answer']=deepcopy(terminal_value)
                value['final_prediction']='a' if data.get('public_options') else deepcopy(terminal_value)
        if stage[0]=='audit_joint':
            self.audits+=1
            if self.mode=='unknown':
                value['audit_verdict']='unknown'
            elif self.mode=='always_conflict' or (self.mode in ('correct','supported_but_changed') and self.audits==1):
                route=next(r for r in data['known_observations']
                    if r['variable']=='advice' and not r['revoked'])
                for slot in value['slots']:
                    if slot['variable']=='advice':
                        scope=data['requested_scopes']['advice']
                        slot['observations']=[{'id':'counter-'+str(self.audits),
                            'stance':'contradiction','value':None,'source_ids':route['source_ids'],
                            **scope,'parent_versions':{},'retract_ids':[route['id']],
                            'reason':'The cited memory does not support that recommendation.'}]
                value['audit_verdict']='supported' if self.mode=='supported_but_changed' else 'conflict'
        message['content']=json.dumps(value)
        return result


def run_advice(mode='normal',value=None,kind='text',options=None):
    plan=advice_plan(kind);services=ControlServices(plan,mode,value)
    row,requests,e=demo.run(services,{'x':'Preference 0: quiet places.'},
        np.array([[1,0,0]],np.float32),settings={'max_repairs_per_request':0},options=options)
    return row,e,services


@pytest.mark.parametrize('kind,value',[('record',{'recommendation':'A quiet visit','constraints':['quiet']}),
                                      ('set',['quiet','museum'])])
def test_publication_keeps_typed_terminal_value_instead_of_display_string(kind,value):
    row,e,_=run_advice(value=value,kind=kind,options=['Quiet visit','Busy visit'])
    assert row['answer']['status']=='ok'
    assert row['answer']['semantic_answer']==value
    assert isinstance(e.node_map()['advice']['answer'],str)
    assert json.loads(e.node_map()['advice']['answer'])==value
    assert row['answer']['prediction']=='a'


@pytest.mark.parametrize('mode',['correct','supported_but_changed'])
def test_audit_counter_is_retained_and_corrected_terminal_requires_fresh_audit(mode):
    row,e,services=run_advice(mode)
    assert row['answer']['status']=='ok'
    assert row['answer']['semantic_answer']=='Choose quiet activity 2'
    assert services.compositions==services.audits==2
    assert e.residual_control.audit_corrections==1
    routes=[r for r in e.fact_state.routes.values() if r['variable']=='advice']
    assert len(routes)==2 and sum(bool(r['revoked']) for r in routes)==1
    assert any(event['event']=='audit_correction_scheduled' for event in e.events)
    assert e.audit_complete and not e.fact_state.pending


def test_repeated_audit_conflicts_stop_at_bound_without_dropping_counterevidence():
    row,e,services=run_advice('always_conflict')
    assert row['answer']['status']=='audit_incomplete'
    assert row['answer']['prediction'] is None and not e.audit_complete
    assert services.audits==services.compositions==3
    assert e.residual_control.audit_corrections==2
    routes=[r for r in e.fact_state.routes.values() if r['variable']=='advice']
    assert len(routes)==3 and all(r['revoked'] for r in routes)
    assert row['diagnostics']['ledger']['remaining']['llm']>0


def test_unchanged_rejected_answer_does_not_repeat_same_audit():
    row,e,services=run_advice('unknown')
    assert row['answer']['status']=='audit_incomplete' and row['answer']['prediction'] is None
    assert services.audits==1 and services.compositions==2
    assert e.residual_control.audit_corrections==1
    assert e.residual_control.audit_replayed


def test_unresolved_semantic_response_cannot_drain_logical_call_budget_by_replay():
    row,e,services=run_advice('unresolved')
    assert row['answer']['prediction'] is None
    assert services.compositions<=3
    assert row['diagnostics']['ledger']['remaining']['llm']>=18
    assert not e.audit_complete
    assert e.fact_state.pending and e.residual_control.reader.evidence_obligations
    assert any(event['event']=='local_read_cycle_detected' for event in e.events)
    # The real parent reader repeatedly confirms the exact same fact; its
    # source/version must not be mistaken for newly acquired evidence.
    assert e.fact_state.versions['x']==1
    assert len(e.fact_state.routes)==1


def test_repeated_terminal_state_still_allows_new_bridge_text_and_recomposition():
    class NewSourceServices(ControlServices):
        def get(self,stage,url,payload):
            result=super().get(stage,url,payload)
            if url.endswith('/embeddings'):
                if 'Anchor passages:' in payload['input'][0]:
                    result['response']['data'][0]['embedding']=[0.,1.,0.]
                return result
            if stage[0]=='planner':return result
            data=decoded_request(payload)
            message=result['response']['choices'][0]['message']
            value=json.loads(message['content'])
            sources={source['doc_id']:source for source in data['source_spans']}
            if stage[0]=='semantic_compose' and 'extra' not in sources:
                for slot in value['slots']:
                    slot['observations']=[]
                    slot['unresolved']=['Visible history contains no supported response.']
                    slot.pop('resolved_obligation_ids',None)
                value['semantic_answer']=None;value['final_prediction']=None
            elif stage[0]=='joint_read' and 'extra' in sources:
                slot=next(slot for slot in value['slots'] if slot['variable']=='x')
                support=next(row for row in slot['observations'] if row['stance']=='support')
                slot['observations'].append({**deepcopy(support),'id':'x-extra',
                    'source_ids':[sources['extra']['id']],
                    'reason':'Another original statement confirms the same preference.'})
            message['content']=json.dumps(value)
            return result
    services=NewSourceServices(advice_plan())
    row,_,e=demo.run(services,{'x':'Preference 0: quiet places.',
        'extra':'Preference 0: a quiet museum is suitable.'},
        np.array([[1,0,0],[0,1,0]],np.float32),settings={'max_repairs_per_request':0})
    assert row['answer']['status']=='ok'
    assert services.compositions==3 and services.audits==1
    assert any(event['event']=='local_read_cycle_detected' for event in e.events)
    assert row['diagnostics']['cost']['ann_calls']==2
    assert 'extra' in e.candidates and not e.fact_state.pending
    assert 'extra' in row['answer']['terminal_input_doc_ids']


def test_certified_local_elimination_pauses_but_retains_unresolved_obligation():
    row,_,e=demo.run_demo('all_failures')
    c=e.residual_control
    assert row['answer']['status']=='ok' and row['answer']['semantic_answer']==['D','A']
    assert row['diagnostics']['cost']['ann_calls']==8
    assert 'delete_compliant' in e.fact_state.pending
    obligations=deepcopy(c.reader.evidence_obligations)
    assert any(item['variable']=='delete_compliant' for item in obligations.values())
    assert c.task_projection('D')['output'] is False
    assert 'D' not in c.update()[1]
    decisive=next(s['id'] for s in e.spans.values() if s['doc_id']=='m4')
    e.fact_state.revoke_source(decisive,'The decisive witness was corrected.')
    assert c.task_projection('D') is None
    assert 'D' in c.update()[1]
    assert c.reader.evidence_obligations==obligations
    assert c.candidate() is None


def test_local_projection_cannot_hide_pending_actual_grounding_parent():
    plan={'steps':[demo.step('parent','Find x'),demo.step('D','Check {parent}',['parent']),
        demo.step('final','Return conditions',['D'],'compose')],
        'final_node_id':'final','output_contract':demo.contract('value','record'),
        'program':{'variables':{'x':{'type':'number','demand':'parent'},
            'flag':{'type':'bool','demand':'D'}},
            'expression':{'op':'record','fields':{'D':demo.var('flag')}}}}
    e,_=engine(plan,{'x':'x = 0.'});c=prepare(e)
    e.add_candidates(['x'])
    source=c.reader._source(next(iter(e.mapper.sources)))
    e.spans[source['id']]=source
    def row(variable,value,parents):
        return {'variable':variable,'stance':'support','value':value,
            'source_ids':[source['id']],**c.state.scope(variable),
            'parent_versions':parents,'retract_ids':[],'reason':'Original scoped fact'}
    c.state.integrate([row('x',0,{})])
    c.state.integrate([row('flag',False,{'x':c.state.versions['x']})])
    assert c.task_projection('D')['output'] is False
    c.state.integrate([],{'x'})
    assert c.task_projection('D') is None


def test_previously_audited_semantic_answer_cannot_publish_after_source_revocation():
    row,e,_=run_advice()
    assert row['answer']['status']=='ok'
    source=next(iter(e.fact_state.routes.values()))['source_ids'][0]
    e.fact_state.revoke_source(source,'The source was corrected after this audit.')
    republished=e.residual_control.publish()
    assert republished['answer']['status']=='audit_incomplete'
    assert republished['answer']['prediction'] is None and not e.audit_complete


def test_determined_but_uncertifiable_deadend_has_explicit_diagnostic(monkeypatch):
    e,_=engine(number_plan(),{'x':'x = 7.'})
    c=prepare(e)
    # Defensive runtime path: imitate an evaluator declining the otherwise
    # determined value. It must neither claim success nor call another model.
    monkeypatch.setattr(c,'candidate',lambda:None)
    row=c.run()
    assert row['answer']['status']=='uncertifiable_output'
    assert row['answer']['prediction'] is None
    assert e.fact_state.valid()[0]['x']==7
    diagnostic=next(r for r in e.events if r['event']=='determined_output_not_certifiable')
    assert diagnostic['expression_op']=='var' and diagnostic['pending_variables']==[]


def test_audit_correction_checkpoint_preserves_bound_and_retained_counters(monkeypatch):
    row,e,_=run_advice('always_conflict')
    c=e.residual_control
    checkpoint=json.loads(json.dumps(c.snapshot()))
    fresh=Engine(e.q,(e.docs,e.ids,e.vectors,e.index,e.tokenizer),
        ControlServices(advice_plan(),'always_conflict'),e.config,e.method)
    # Inspect the actual resume boundary before it schedules another action.
    monkeypatch.setattr(ResidualControl,'run',lambda control:control.publish())
    fresh.resume(checkpoint)
    restored=fresh.residual_control
    assert restored.audit_corrections==c.MAX_AUDIT_CORRECTIONS
    assert restored.audit_feedback==c.audit_feedback
    assert restored.state.routes==c.state.routes
    assert restored.read_attempts==c.read_attempts
    assert restored.read_effects==c.read_effects
    assert restored.compose_exhausted==c.compose_exhausted
    assert not restored.recover_after_audit('retry_after_resume')
    assert restored.publish()['answer']['prediction'] is None
