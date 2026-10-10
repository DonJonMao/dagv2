"""Executable behavior checks through Engine, the real BT adapter and sources."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import demo_residual_memory as demo
from dagbt.config import resolve
from dagbt.methods import RESIDUAL,JOINT,UNSCORED,LOCAL
from dagbt.transport import call_reservation,request_identity
from dagbt.residual_plan import validate_plan
from dagbt.fact_state import FactState
from dagbt.residual import certify_output,ProgramError
from dagbt import runner
from dagbt.reasoning import ProtocolError
from test_bridge import session,requirement


def zero_session(ann=8):
 return session(ann=ann,sets=0,proxy_mode='none',algorithm_version=RESIDUAL)

def test_step_dense_returns_before_any_root_expansion():
 s=zero_session();s.config.pop('reranker')
 found=s.step('Find country','country',requirement(),identity={'version':1})
 assert s.ledger.used['ann']==1 and len(s.calls.requests)==1
 assert [b['stage'] for b in found['trace']['retrieval']['proposal_batches']]==['initial_dense']
 s.step('Find country','country',requirement(),identity={'version':1})
 assert s.ledger.used['ann']==2 and len(s.calls.requests)==2
 assert s.scorer is None and s.backend is None and not s.scorers

def test_json_snapshot_restore_retains_frontier_consumption_and_budget():
 a=zero_session(8);a.step('Find country','country',requirement(),identity={'version':1})
 a.step('Find country','country',requirement(),identity={'version':1})
 a.pause_node('country');snapshot=json.loads(json.dumps(a.snapshot()))
 b=zero_session(8);b.restore(snapshot)
 assert b.ledger.used['ann']==2 and not b.calls.requests
 ra=a.step('Find country','country',requirement(),identity={'version':1})
 rb=b.step('Find country','country',requirement(),identity={'version':1})
 assert len(b.calls.requests)==1 and b.ledger.used['ann']==3
 # The search schedule resumes identically (fixture embeddings depend on request
 # count, so compare navigation/request, not the intentionally different hits).
 ba=ra['trace']['retrieval']['proposal_batches'][-1];bb=rb['trace']['retrieval']['proposal_batches'][-1]
 assert (ba['stage'],ba['target_id'],ba['premise_ids'])==(bb['stage'],bb['target_id'],bb['premise_ids'])
 assert bb['stage']=='conditional'
 with pytest.raises(ValueError,match='budget'):
  zero_session(9).restore(snapshot)

def test_new_parent_binding_identity_creates_new_dense_without_resetting_cost():
 s=zero_session();s.step('Find country','country',identity={'version':1})
 s.step('Find country','country',identity={'version':2})
 assert len(s.sessions)==2 and s.ledger.used['ann']==2
 assert all(x['retriever']._dense_batch is not None for x in s.sessions.values())

@pytest.mark.parametrize('method',[UNSCORED,JOINT,RESIDUAL])
def test_zero_score_capabilities_and_terminal_reservation(method):
 settings=resolve({},method)
 assert settings['set_score_calls']==settings['reader_calls']==0 and settings['proxy_mode']=='none'
 assert call_reservation(settings,'joint_read')==1 and call_reservation(settings,'audit_joint')==0
 assert request_identity('q','url',{}, {'fusion':settings})['algorithm_version']==method
 with pytest.raises(ValueError):resolve({'fusion':{'ann_calls':37}},method)
 with pytest.raises(ValueError):resolve({'fusion':{'set_score_calls':0}},LOCAL)

def test_full_dag_exists_before_first_embedding_and_rule_instantiation():
 row,requests,e=demo.run_demo()
 assert requests[0]['stage'][0]=='planner'
 initialized=next(x for x in e.events if x['event']=='typed_dag_initialized')
 assert [s['output_slot'] for s in initialized['dag']]==['rules','D','A','O','decision']
 assert all('rules' in s['inputs'] for s in initialized['dag'][1:4])
 assert e.requirements_hash==row['diagnostics']['requirements_hash']
 assert row['diagnostics']['fact_state']['program_revision']==1
 assert row['answer']['completion_kind']=='certified_symbolic'
 assert row['diagnostics']['cost']['planner_calls']==1

def test_clue_does_not_bind_false_and_short_circuit_really_stops_A_O():
 row,requests,e=demo.run_demo()
 assert row['answer']['status']=='ok'
 assert row['answer']['semantic_answer']=={'decision':False,'reason':'D'}
 probes=[r for r in requests if r['url'].endswith('/embeddings')]
 assert len(probes)==4
 assert not any('检查新增' in r['payload']['input'][0] or '检查脱网' in r['payload']['input'][0] for r in probes)
 batches=e.fact_state.history
 assert any(any(r['variable']=='return_age' and r['stance']=='partial' for r in h['rows']) for h in batches)
 assert e.fact_state.valid()[0]['return_age']==180
 assert 'add_age' not in e.fact_state.valid()[0] and 'offline' not in e.fact_state.valid()[0]
 assert row['answer']['terminal_input_doc_ids']==[] and row['diagnostics']['terminal_input'] is None
 assert 'm4' in row['answer']['sources']['doc_ids'] and 'm6' not in row['answer']['sources']['doc_ids']
 assert row['budgets']=={} and e.mapper.completed=={}
 assert not any(r['stage'][0] in ('map','resolve','select','reader') or 'rerank' in r['url'] for r in requests)

def test_all_failures_cannot_stop_on_one_witness():
 row,requests,e=demo.run_demo('all_failures')
 assert row['answer']['status']=='ok' and row['answer']['semantic_answer']==['D','A']
 assert e.fact_state.valid()[0]['offline'] is True
 assert row['diagnostics']['cost']['ann_calls']==8
 assert set(row['answer']['sources']['doc_ids'])=={'m1','m2','m4','m8','m9'}

def test_fixed_plan_joint_control_has_no_short_circuit_visitation():
 residual,_,_=demo.run_demo();joint,_,_=demo.run_demo(method=JOINT)
 assert joint['answer']['status']=='ok' and joint['answer']['semantic_answer']==residual['answer']['semantic_answer']
 assert joint['diagnostics']['cost']['ann_calls']==8>residual['diagnostics']['cost']['ann_calls']
 assert joint['diagnostics']['fact_state']['facts']['add_age']==900

def test_revoke_decisive_source_reactivates_paused_demands_and_invalidates_certificate():
 row,_,e=demo.run_demo();c=e.residual_control;old=c.certificate
 sid=next(s['id'] for s in e.spans.values() if s['doc_id']=='m4')
 e.fact_state.revoke_source(sid,'DEL-08 corrected to V7; scope no longer applies')
 assert c.candidate() is None
 _,active=c.update()
 assert {'D','A','O'}<=set(active)
 assert c.verify_derivation(e.node_map()['decision'],{'derivation':old}) is False
 assert any(x['event']=='residual_state' and 'D' in x['reactivated_demands'] for x in e.events)

def test_corrected_V7_does_not_bind_V8_deletion_failure():
 row,_,e=demo.run_demo(correction=True)
 assert 'return_age' not in e.fact_state.valid()[0]
 assert row['answer']['semantic_answer']['reason']!='D'

@pytest.mark.parametrize('fn',[demo.run_chain,demo.run_compare,lambda:demo.run_compare(True),demo.run_advice])
def test_other_task_shapes_execute_and_runner_accepts(fn):
 row,requests,e=fn()
 assert row['answer']['status']=='ok'
 assert runner.validate_result(row,row['unit_id']) is row
 assert row['diagnostics']['cost']['set_score_calls']==row['diagnostics']['cost']['rerank_http_attempts']==0
 assert row['answer']['completion_kind']==('semantic_assessment' if fn==demo.run_advice else 'certified_symbolic')

def test_semantic_terminal_has_real_response_and_option_isolation():
 options=['Quiet visit','Busy visit']
 row,requests,e=demo.run_advice(options)
 assert row['answer']['prediction']=='A'
 assert row['answer']['certificate'] is None and row['answer']['terminal_input_doc_ids']
 assert row['diagnostics']['terminal_input']['response_ref'].startswith('synthetic-')
 for r in requests:
  if r['stage'][0]!='semantic_compose':assert 'public_options' not in str(r['payload'])

def test_last_read_and_reserved_audit_can_exhaust_llm_then_free_publish():
 row,_,_=demo.run_demo(llm_calls=7)
 assert row['answer']['status']=='ok' and row['diagnostics']['ledger']['remaining']['llm']==0
 row,_,_=demo.run_demo(llm_calls=6)
 assert row['answer']['prediction'] is None

def test_invalid_visible_alias_never_certifies_or_fakes_mapping_complete():
 services=demo.SourcesOnly(demo.rule_plan('one_reason'),bad_rows=True)
 axes={'m1':0,'m5':0,'m2':1,'m3':2,'m4':3,'m6':4,'m8':5,'m7':6,'m9':7}
 row,requests,e=demo.run(services,demo.PASSAGES,np.array([[float(i==axes[d]) for i in range(8)] for d in demo.PASSAGES],np.float32))
 assert row['answer']['prediction'] is None and e.fact_state.pending
 assert row['diagnostics']['cost']['mapping_calls']==0 and not e.mapper.completed
 assert row['diagnostics']['cost']['repair_calls']>0

def state():
 p={'variables':{'x':{'type':'bool','demand':'x'},'y':{'type':'bool','demand':'y'}},'expression':demo.op('and',demo.var('x'),demo.var('y'))}
 return FactState(p,demo.contract('value','bool','example'),[demo.step('x','x'),demo.step('y','y'),demo.step('out','out',['x','y'],'compose')],'out')
def observation(v,value,source,parents=None):
 return {'variable':v,'value':value,'stance':'support','source_ids':[source],'entity_scope':'example',
  'time_scope':'本次演示','parent_versions':parents or {},'retract_ids':[],'reason':'source'}

def test_same_batch_true_and_counter_value_is_ambiguous_before_certification():
 s=state();s.integrate([observation('x',False,'s1'),observation('x',True,'s2')])
 assert 'x' not in s.valid()[0]
 assert certify_output(s.program,s.valid()[0],s.contract) is None

def test_independent_OR_support_survives_source_revocation():
 s=state();s.integrate([observation('x',False,'s1'),observation('x',False,'s2')]);old=s.versions['x']
 s.revoke_source('s1','bad scope')
 assert s.valid()[0]['x'] is False and s.valid()[1]['x']==['s2']
 assert s.versions['x']>old

def test_parent_version_invalidates_actual_route_but_preserves_independent_one():
 s=state();s.integrate([observation('x',True,'s1')]);s.steps[1]['inputs']=['x']
 s.integrate([observation('y',False,'s2',{'x':s.versions['x']}),observation('y',False,'s3')])
 s.revoke_source('s1','withdrawn')
 assert s.valid()[0]['y'] is False and s.valid()[1]['y']==['s3']

def test_contract_result_type_mismatch_cannot_certify():
 s=state();s.integrate([observation('x',False,'s1')]);k={**s.contract,'result_type':'number'}
 assert certify_output(s.program,s.valid()[0],k) is None

def test_planner_rejects_unplanned_demands_and_missing_rule_dependents():
 from dagbt.engine import legacy_modules
 _,repair,_=legacy_modules()
 p=demo.rule_plan('one_reason');p['program']['variables']['result']['demand']='hidden'
 with pytest.raises(ProtocolError):validate_plan(p,repair.validate_plan)
 p=demo.rule_plan('one_reason');p['steps'][1]['inputs']=[];p['steps'][1]['question']='check later'
 with pytest.raises(ProtocolError):validate_plan(p,repair.validate_plan)

def test_paid_task_equal_to_original_Q_is_legal_without_extra_baseline():
 row,requests,e=demo.run_chain()
 assert all(d['trace']['node_id']!='__baseline__' for d in e.discoveries)
 assert row['diagnostics']['cost']['planner_calls']==1

def test_source_mutation_invalidates_certificate_on_publication_check():
 row,_,e=demo.run_demo()
 e.docs['m4'].passage=e.docs['m4'].passage.replace('V8','V7')
 with pytest.raises(ValueError):e.residual_control.candidate()

def test_completed_checkpoint_resumes_without_any_planner_dense_read_or_audit():
 from dagbt.engine import Engine
 row,requests,e=demo.run_demo()
 checkpoint=json.loads(json.dumps(e.residual_control.snapshot()))
 client=demo.SourcesOnly(demo.rule_plan('one_reason'))
 fresh=Engine(e.q,(e.docs,e.ids,e.vectors,e.index,e.tokenizer),client,e.config,RESIDUAL)
 resumed=fresh.resume(checkpoint)
 assert client.requests==[] and resumed['answer']==row['answer']
 assert fresh.ledger.used==e.ledger.used
 changed=deepcopy(checkpoint);changed['fact_state']['contract']['entity_scope']='another project'
 with pytest.raises(ValueError):
  Engine(e.q,(e.docs,e.ids,e.vectors,e.index,e.tokenizer),client,e.config,RESIDUAL).resume(changed)

def test_real_engine_single_node_q_equal_Q_uses_one_dense_and_no_baseline():
 plan={'steps':[demo.step('x','Find x quantity')],'final_node_id':'x',
  'output_contract':demo.contract('value','number','numbers'),
  'program':{'variables':{'x':{'type':'number','demand':'x'}},'expression':demo.var('x')}}
 services=demo.SourcesOnly(plan,scenario='comparison')
 from dagbt.engine import Engine
 from types import SimpleNamespace
 docs={'x':SimpleNamespace(doc_id='x',passage='x = 0.')}
 e=Engine({'id':'sameQ','question':'Find x quantity'},(docs,['x'],np.array([[1,0,0]],np.float32),None,demo.Tokenizer()),
  services,{'_test_transport':True,'llm_base_url':'http://synthetic/v1','embedding_base_url':'http://synthetic/v1',
  'embedding_model':'script','llm_model':'script','fusion':{'initial_width':1}},RESIDUAL)
 r=e.run()
 assert r['answer']['status']=='ok' and r['answer']['semantic_answer']==0 and r['answer']['prediction']=='0'
 assert [f['trace']['node_id'] for f in e.discoveries]==['x']
 assert 'Current retrieval task:\nFind x quantity' in next(r for r in services.requests if r['url'].endswith('/embeddings'))['payload']['input'][0]

def test_scoped_repair_preserves_valid_rows_and_missing_counter_slot_stays_pending():
 class Broken(demo.SourcesOnly):
  def get(self,stage,url,payload):
   r=super().get(stage,url,payload)
   if stage[0]=='joint_read' and 'return_age' in json.loads(payload['messages'][1]['content'])['variables']:
    value=json.loads(r['response']['choices'][0]['message']['content'])
    for slot in value['slots']:
     if slot['variable']=='returned' and slot['observations'][0]['stance']=='support':
      slot['observations'][0]['source_ids']=['invisible-counter-scope']
    r['response']['choices'][0]['message']['content']=json.dumps(value)
   elif stage[0]=='joint_read_repair':
    r['response']['choices'][0]['message']['content']=json.dumps({'slots':[]})
   return r
 axes={'m1':0,'m5':0,'m2':1,'m3':2,'m4':3,'m6':4,'m8':5,'m7':6,'m9':7}
 row,requests,e=demo.run(Broken(demo.rule_plan('one_reason')),demo.PASSAGES,
  np.array([[float(i==axes[d]) for i in range(8)] for d in demo.PASSAGES],np.float32))
 assert e.fact_state.valid()[0]['return_age']==180
 assert 'returned' in e.fact_state.pending
 assert row['answer']['semantic_answer']!={'decision':False,'reason':'D'}
 # An independently supported A witness remains a complete legal answer; the
 # failed D slot is retained and cannot support a D-based certificate.
 repairs=[json.loads(r['payload']['messages'][1]['content']) for r in requests if r['stage'][0]=='joint_read_repair']
 assert repairs and all('return_age' not in d['variables'] for d in repairs)
 assert row['diagnostics']['cost']['llm_attempts']<=24

def test_audit_retracts_decisive_fact_then_recomputes_instead_of_publishing_old_certificate():
 class Correction(demo.SourcesOnly):
  def get(self,stage,url,payload):
   r=super().get(stage,url,payload)
   if stage[0]=='audit_joint':
    data=json.loads(payload['messages'][1]['content']);value=json.loads(r['response']['choices'][0]['message']['content'])
    for slot in value['slots']:
     if slot['variable']=='returned':
      old=[x for x in data['known_observations'] if x['variable']=='returned' and not x['revoked']]
      if old:
       slot['observations']=[{'id':'audit-correction','stance':'contradiction','value':None,
        'source_ids':[next(s['id'] for s in data['source_spans'] if s['doc_id']=='m4')],
        'entity_scope':data['required_scope']['entity_scope'],'time_scope':data['required_scope']['time_scope'],
        'parent_versions':{},'retract_ids':[x['id'] for x in old],
        'reason':'Offline audit substitute corrects applicability of this observation'}]
    r['response']['choices'][0]['message']['content']=json.dumps(value)
   return r
 axes={'m1':0,'m5':0,'m2':1,'m3':2,'m4':3,'m6':4,'m8':5,'m7':6,'m9':7}
 row,requests,e=demo.run(Correction(demo.rule_plan('one_reason')),demo.PASSAGES,
  np.array([[float(i==axes[d]) for i in range(8)] for d in demo.PASSAGES],np.float32))
 assert any(x['event']=='certificate_invalidated_after_audit' for x in e.events)
 assert row['answer']['semantic_answer']!={'decision':False,'reason':'D'} or row['answer']['prediction'] is None

def test_wire_capacity_is_measured_before_call_and_no_silent_truncation():
 row,_,e=demo.run_demo()
 c=e.residual_control;reader=c.reader
 e.s['context_tokens']=200
 before=len(e.reasoner.requests)
 step=next(s for s in e.steps if s['output_slot']=='D')
 from dagbt.reasoning import InputOverflow
 with pytest.raises(InputOverflow):reader.read(step,'deletion',['return_age','returned'],['m4'],'new-capacity-state')
 assert len(e.reasoner.requests)==before
 assert reader.batches[-1]['status']=='capacity_unavailable'
 assert 'return_age' in e.fact_state.pending

def test_bridge_same_document_shared_across_two_pools_keeps_discovery_routes():
 s=zero_session();a=s.step('First task','first',identity={'v':1});b=s.step('Second task','second',identity={'v':1})
 assert len(s.candidate_ids)==len(set(s.candidate_ids))
 assert len(s.sessions)==2
 assert a['trace']['query_identity']!=b['trace']['query_identity']
 assert all(e['dependency_claim'] is False for t in (a,b) for e in t['trace']['retrieval']['source_graph'])

def test_four_controls_and_fixed_pool_residual_isolation():
 rows={m:demo.run_demo(method=m)[0] for m in (LOCAL,UNSCORED,JOINT,RESIDUAL)}
 assert all(r['answer']['status']=='ok' for r in rows.values())
 assert rows[LOCAL]['diagnostics']['cost']['set_score_calls']>0
 assert rows[UNSCORED]['diagnostics']['cost']['set_score_calls']==0
 assert rows[UNSCORED]['diagnostics']['cost']['node_resolve_calls']==5
 assert rows[JOINT]['diagnostics']['cost']['mapping_calls']==0
 joint,_,_=demo.run_demo(method=JOINT,fixed_pool=True)
 full,_,_=demo.run_demo(method=RESIDUAL,fixed_pool=True)
 assert joint['diagnostics']['cost']['ann_calls']==full['diagnostics']['cost']['ann_calls']==0
 assert joint['answer']['semantic_answer']==full['answer']['semantic_answer']
 assert joint['diagnostics']['cost']['generation_calls']>full['diagnostics']['cost']['generation_calls']

def test_source_role_change_is_rejected_even_when_quote_text_is_unchanged():
 row,_,e=demo.run_demo()
 e.docs['m4'].metadata={'source_segments':[{'start':0,'end':len(e.docs['m4'].passage),
  'role':'assistant','source_message_indices':[1],'provenance':'authoritative'}]}
 with pytest.raises(ValueError,match='role changed'):e.residual_control.candidate()

def test_mid_observation_checkpoint_resumes_same_frontier_without_dense_replay():
 from dagbt.engine import Engine
 from dagbt.bridge import BridgeSession
 from dagbt.residual_control import ResidualControl
 from dagbt.transport import digest
 _,_,e=demo.run_demo()
 client=demo.SourcesOnly(demo.rule_plan('one_reason'))
 fresh=Engine(e.q,(e.docs,e.ids,e.vectors,e.index,e.tokenizer),client,e.config,RESIDUAL)
 fresh.plan();fresh.fixed_pool=None
 fresh.bridge=BridgeSession(fresh.q['question'],fresh.docs,fresh.ids,fresh.vectors,fresh.tokenizer,fresh.calls,fresh.config,fresh.ledger)
 c=ResidualControl(fresh);fresh.residual_control=c
 st=fresh.steps[0];query=st['question']
 identity={'algorithm':RESIDUAL,'query':query,'node_id':'rules','program_revision':0,
  'program_identity':digest(fresh.fact_state.program),'initial_dag_identity':digest(fresh.steps),'bindings':{},
  'contract_identity':digest(fresh.fact_state.contract)}
 found=fresh.bridge.step(query,'rules',[{'id':'rules','description':query,'necessary':True}],[],identity=identity)
 fresh.add_candidates(found['local_candidate_ids']);c.local_pools['rules']=found['local_candidate_ids']
 c.reader.read(st,query,['rule'],found['local_candidate_ids'],digest(identity))
 snapshot=json.loads(json.dumps(c.snapshot()));new_client=demo.SourcesOnly(demo.rule_plan('one_reason'))
 resumed=Engine(e.q,(e.docs,e.ids,e.vectors,e.index,e.tokenizer),new_client,e.config,RESIDUAL)
 row=resumed.resume(snapshot)
 assert row['answer']['status']=='ok'
 assert not any(r['stage'][0]=='planner' for r in new_client.requests)
 assert new_client.requests[0]['url'].endswith('/embeddings')
 assert 'Anchor passages:' in new_client.requests[0]['payload']['input'][0]
 assert row['diagnostics']['cost']['ann_calls']==4

def test_production_transport_zero_score_path_is_spied_over_real_HTTP(tmp_path):
 import threading
 from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
 from types import SimpleNamespace
 from dagbt.engine import Engine
 plan={'steps':[demo.step('x','Find x quantity')],'final_node_id':'x',
  'output_contract':demo.contract('value','number','numbers'),
  'program':{'variables':{'x':{'type':'number','demand':'x'}},'expression':demo.var('x')}}
 service=demo.SourcesOnly(plan,scenario='comparison');seen=[]
 class Handler(BaseHTTPRequestHandler):
  def log_message(self,*args):pass
  def do_POST(self):
   payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
   seen.append(self.path)
   if self.path.endswith('/embeddings'):stage=('dagbt','x','bridge','1','embedding')
   else:
    data=json.loads(payload['messages'][1]['content'])
    stage=('planner' if isinstance(data,str) else 'audit_joint' if 'audit_instruction' in data else 'joint_read','1')
   response=service.get(stage,'http://localhost'+self.path,payload)['response']
   encoded=json.dumps(response).encode();self.send_response(200);self.send_header('Content-Type','application/json')
   self.send_header('Content-Length',str(len(encoded)));self.end_headers();self.wfile.write(encoded)
 server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
 try:
  base=f'http://127.0.0.1:{server.server_port}/v1'
  docs={'x':SimpleNamespace(doc_id='x',passage='x = 0.')}
  e=Engine({'id':'http-0','question':'Find x quantity'},(docs,['x'],np.array([[1,0,0]],np.float32),None,demo.Tokenizer()),
   SimpleNamespace(output=tmp_path),{'llm_base_url':base,'embedding_base_url':base,'llm_model':'script','embedding_model':'script',
    'fusion':{'initial_width':1},'reranker':{'url':'http://127.0.0.1:1/unavailable-rerank'}},RESIDUAL)
  row=e.run()
  assert row['answer']['status']=='ok' and row['answer']['prediction']=='0'
  assert seen.count('/v1/embeddings')==1 and seen.count('/v1/chat/completions')==3
  assert row['diagnostics']['ledger']['used']['llm']==3
  assert not any('rerank' in p for p in seen)
 finally:server.shutdown();server.server_close();thread.join()

def test_intermediate_deterministic_node_short_circuits_then_grounds_real_child():
 class Services(demo.SourcesOnly):
  def get(self,stage,url,payload):
   if stage[0]=='joint_read':
    data=json.loads(payload['messages'][1]['content']);slots=[]
    for v in data['variables']:
     source=next(iter(data['source_spans']))
     slots.append({'variable':v,'unresolved':[],'observations':[{'id':v+'-obs','stance':'support',
      'value':False if v=='a' else 'Result entity','source_ids':[source['id']],
      'entity_scope':data['required_scope']['entity_scope'],'time_scope':data['required_scope']['time_scope'],
      'parent_versions':{p:d['version'] for p,d in data['supported_parents'].items() if d['demand'] in next(st['inputs'] for st in self.plan['steps'] if st['output_slot']==data['variables'][v]['demand'])},'retract_ids':[],'reason':'Visible original source'}]})
    self.requests.append({'stage':list(stage),'url':url,'payload':payload})
    return {'response_ref':'synthetic','response':{'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'slots':slots})}}]}}
   return super().get(stage,url,payload)
 plan={'steps':[demo.step('a','Check A'),demo.step('b','Check B'),demo.step('decision','Combine A B',['a','b'],'compose'),
  demo.step('out','Find the entity for decision {decision}',['decision'])], 'final_node_id':'out',
  'output_contract':demo.contract('value','entity','example'),
  'program':{'variables':{'a':{'type':'bool','demand':'a'},'b':{'type':'bool','demand':'b'},
    'decision':{'type':'bool','demand':'decision'},'out':{'type':'entity','demand':'out'}},
    'task_expressions':{'decision':demo.op('and',demo.var('a'),demo.var('b'))},'expression':demo.var('out')}}
 row,requests,e=demo.run(Services(plan,scenario='intermediate'),{'x':'A false and applicable entity Result entity.'},np.array([[1,0,0]],np.float32))
 assert row['answer']['status']=='ok' and e.fact_state.valid()[0]['decision'] is False
 assert 'b' not in e.fact_state.valid()[0]
 assert not any('Check B' in r['payload']['input'][0] for r in requests if r['url'].endswith('/embeddings'))
 assert row['ranking']['nodes'][2]['alternatives'][0]['support_kind']=='symbolic_bindings'
 assert row['answer']['certificate']['binding_dependencies']['a']['version']>0

def test_symbolic_option_terminal_is_only_paid_format_phase_and_certificate_scope_stays_semantic():
 plan={'steps':[demo.step('x','Find x quantity')],'final_node_id':'x',
  'output_contract':demo.contract('value','number','numbers'),
  'program':{'variables':{'x':{'type':'number','demand':'x'}},'expression':demo.var('x')}}
 row,requests,e=demo.run(demo.SourcesOnly(plan,scenario='comparison'),{'x':'x = 0.'},np.array([[1,0,0]],np.float32),options=['Zero quantity','Other quantity'])
 assert row['answer']['status']=='ok' and row['answer']['prediction']=='A'
 assert row['answer']['completion_kind']=='certified_symbolic' and row['answer']['certificate']['output']==0
 terminal=[r for r in requests if 'public_options' in str(r['payload'])]
 assert len(terminal)==1 and terminal[0]['stage'][0]=='joint_read'
 assert e.residual_control.format_input['scope']=='joint_terminal_output'
 assert row['diagnostics']['cost']['semantic_compose_calls']==0

def test_non_singleton_output_envelope_stays_incomplete_without_answer_fallback():
 from dagbt.residual import possible_outputs
 row,_,e=demo.run_non_singleton()
 envelope=possible_outputs(e.fact_state.program,e.fact_state.valid()[0])
 assert envelope['status']=='indeterminate' and set(envelope['outputs'])=={False,True}
 assert row['answer']['prediction'] is None and row['answer']['certificate'] is None
 assert row['answer']['completion_kind']=='incomplete'

def test_initial_planner_protocol_repair_is_paid_and_reported_once_as_repair():
 class Repaired(demo.SourcesOnly):
  broken=False
  def get(self,stage,url,payload):
   r=super().get(stage,url,payload)
   if stage[0]=='planner' and not self.broken:
    self.broken=True;r['response']['choices'][0]['message']['content']='{invalid JSON'
   return r
 axes={'m1':0,'m5':0,'m2':1,'m3':2,'m4':3,'m6':4,'m8':5,'m7':6,'m9':7}
 row,_,e=demo.run(Repaired(demo.rule_plan('one_reason')),demo.PASSAGES,
  np.array([[float(i==axes[d]) for i in range(8)] for d in demo.PASSAGES],np.float32),settings={'initial_width':2})
 cost=row['diagnostics']['cost']
 assert row['answer']['status']=='ok'
 assert cost['planner_calls']==1 and cost['repair_calls']==cost['json_repair_reservations']==1
 assert cost['generation_calls']==cost['planner_calls']+cost['repair_calls']+cost['joint_read_calls']+cost['audit_calls']+cost['semantic_compose_calls']
 assert cost['llm_attempts']==cost['generation_calls']

def test_local_batch_can_withdraw_grounding_rule_before_any_output_certificate():
 class ParentCounter(demo.SourcesOnly):
  def get(self,stage,url,payload):
   r=super().get(stage,url,payload)
   if stage[0]=='joint_read':
    data=json.loads(payload['messages'][1]['content'])
    if data['task']['output_slot']=='D' and any(s['doc_id']=='m3' for s in data['source_spans']):
     value=json.loads(r['response']['choices'][0]['message']['content'])
     for slot in value['slots']:
      if slot['variable']=='rule':
       routes=[x for x in data['known_observations'] if x['variable']=='rule' and not x['revoked']]
       if routes:
        slot['observations']=[{'id':'local-parent-counter','stance':'contradiction','value':None,
         'source_ids':[next(s['id'] for s in data['source_spans'] if s['doc_id']=='m3')],
         'entity_scope':data['required_scope']['entity_scope'],'time_scope':data['required_scope']['time_scope'],
         'parent_versions':{},'retract_ids':[x['id'] for x in routes],
         'reason':'Offline substitute supplies a scoped correction to the grounding rule'}]
     r['response']['choices'][0]['message']['content']=json.dumps(value)
   return r
 axes={'m1':0,'m5':0,'m2':1,'m3':2,'m4':3,'m6':4,'m8':5,'m7':6,'m9':7}
 row,_,e=demo.run(ParentCounter(demo.rule_plan('one_reason')),demo.PASSAGES,
  np.array([[float(i==axes[d]) for i in range(8)] for d in demo.PASSAGES],np.float32),settings={'initial_width':2})
 assert row['answer']['prediction'] is None
 assert any(r['variable']=='rule' and r['revoked'] for r in e.fact_state.routes.values())
 assert e.fact_state.revision>=2
 assert any(x['event']=='residual_state' and 'rules' in x['reactivated_demands'] for x in e.events)

def test_semantic_constant_without_real_terminal_response_cannot_publish():
 plan={'steps':[demo.step('answer','Give open advice')],'final_node_id':'answer',
  'output_contract':demo.contract('semantic','text','advice'),
  'program':{'variables':{},'expression':demo.const('Fabricated advice')}}
 row,requests,e=demo.run(demo.SourcesOnly(plan,scenario='empty_semantic'),{'x':'No advice evidence.'},np.array([[1,0,0]],np.float32))
 assert row['answer']['prediction'] is None and row['answer']['completion_kind']=='incomplete'
 assert not any(r['stage'][0]=='audit_joint' for r in requests)

def test_deterministic_compose_uses_only_its_first_terminal_generation_for_answer_and_choice():
 row,requests,e=demo.run_compare(options=['x is smaller','x is not smaller'])
 assert row['answer']['status']=='ok' and row['answer']['semantic_answer'] is True and row['answer']['prediction']=='A'
 terminal=[r for r in requests if r['stage'][0]=='semantic_compose']
 assert len(terminal)==1 and e.residual_control.format_input['scope']=='semantic_terminal_output'
 assert row['diagnostics']['cost']['independent_option_matcher_calls']==0
 for r in requests:
  if r['stage'][0]!='semantic_compose':assert 'public_options' not in str(r['payload'])

def test_single_successful_deletion_observation_does_not_prove_universal_compliance():
 from dagbt.residual import possible_outputs
 row,_,e=demo.run_demo()
 state=e.fact_state
 values=state.valid()[0]
 # Replace the supplied point observation only in this pure counterfactual:
 # no return at 180 seconds does not establish all returns at every t>=60.
 values['returned']=False
 assert 'delete_compliant' not in values
 program={**state.program,'expression':state.program['expression']['fields']['D']}
 result=possible_outputs(program,values)
 assert result['status']=='indeterminate' and set(result['outputs'])=={False,True}
 values['returned']=True;values['delete_compliant']=True
 assert possible_outputs(program,values)['status']=='inconsistent'
 assert row['answer']['semantic_answer']=={'decision':False,'reason':'D'}
 assert e.node_map()['D']['alternatives'][0]['support_kind']=='scoped_task_projection'
