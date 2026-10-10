#!/usr/bin/env python3
"""Invented source-driven offline services, actual Engine/BT/residual control.

The substitute reads only visible raw spans; this tests mechanics, not LLM quality.
"""
from pathlib import Path
import argparse
from copy import deepcopy
import json
import re
import sys
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dagbt.engine import Engine
from dagbt.methods import RESIDUAL, JOINT, LOCAL, UNSCORED, capabilities
from demo_local_terminal import Tokenizer

PASSAGES={
 'm1':'本次演示采用岚桥 V8，适用 R17-R2，不再沿用旧版 R17。',
 'm2':'R17-R2要求新增600秒内可检索、删除后60秒起不得再被返回、脱网可用；三项同时满足即可启用，无其他门槛。',
 'm3':'岚桥 V8 删除测试存在待查异常，具体复现见 DEL-08。',
 'm4':'DEL-08针对 V8：同一被删文档在删除后180秒的检索请求中仍被返回。',
 'm5':'旧版 R17 要求新增2秒内可见，旧方案未达标。',
 'm6':'V8新增测试记录见 SYNC-21。', 'm7':'V8脱网运行记录见 OFF-09。',
 'm8':'SYNC-21针对 V8：新增后900秒首次可检索。',
 'm9':'OFF-09针对 V8：脱网可用测试通过。',
}

def var(v):return {'op':'var','id':v}
def const(v):return {'op':'const','value':v}
def op(name,*args):return {'op':name,'args':list(args)}
def step(n,q,parents=(),kind='retrieval'):
 return {'output_slot':n,'question':q,'inputs':list(parents),'execution':kind,'answer_type':'fact'}
def contract(mode='one_reason',kind='record',scope='岚桥 V8 / R17-R2'):
 return {'mode':mode,'result_type':kind,'entity_scope':scope,'time_scope':'本次演示',
         'explanation':'one' if mode=='one_reason' else 'all' if mode=='all_failures' else 'semantic' if mode=='semantic' else 'none'}
def rule_plan(mode):
 return {'steps':[step('rules','本次演示岚桥版本与当前适用规则是什么？'),
     step('D','根据{rules}检查删除后的返回观测。',['rules']),
     step('A','根据{rules}检查新增可检索时间。',['rules']),
     step('O','根据{rules}检查脱网可用性。',['rules']),
     step('decision','综合规则与符合性输出完整所需结果。',['rules','D','A','O'],'compose')],
   'final_node_id':'decision','output_contract':contract(mode,'set' if mode=='all_failures' else 'record'),
   'program':{'variables':{'rule':{'type':'rule','demand':'rules'},'result':{'type':'record','demand':'decision'}},
     'expression':var('result'),'rules_complete':False,
     'rule_binding':{'variable':'rule','condition_demands':['D','A','O']}}}

class SourcesOnly:
 def __init__(self,plan,*,scenario='rule',bad_rows=False,audit_revoke=False):
  self.plan=deepcopy(plan);self.requests=[];self.scenario=scenario;self.bad_rows=bad_rows;self.audit_revoke=audit_revoke
 def get(self,stage,url,payload):
  self.requests.append({'stage':list(stage),'url':url,'payload':deepcopy(payload)})
  if url.endswith('/embeddings'):
   query=payload['input'][0]
   task=query.split('Current retrieval task:\n',1)[-1].split('\n\n',1)[0]
   anchor=query.split('Anchor passages:\n',1)[-1] if 'Anchor passages:\n' in query else ''
   if self.scenario=='rule':
    axis= (3 if anchor and 'DEL-08' in anchor else 2) if '检查删除' in task else (5 if anchor else 4) if '检查新增' in task else (7 if anchor else 6) if '检查脱网' in task else 1 if anchor else 0
    vector=[float(i==axis) for i in range(8)]
   else:
    axis=1 if ('university' in task and 'director' not in task) or 'university did' in task else 2 if 'city contains' in task else 0
    vector=[float(i==axis) for i in range(3)]
   return {'response':{'data':[{'index':0,'embedding':vector}]}}
  if 'rerank' in url:
   if not getattr(self,'scored',False):raise AssertionError('Zero-score path called reranker')
   return {'response':{'results':[{'index':i,'relevance_score':min(.95,.1+.12*t.count('[Passage '))}
    for i,t in enumerate(payload['documents'])]}}
  data=json.loads(payload['messages'][1]['content']);operation=stage[0]
  if operation=='planner':value=self.plan if not getattr(self,'legacy',False) else {k:self.plan[k] for k in ('steps','final_node_id')}
  elif operation=='semantic_compose' and 'certificate' in data:
   value={'semantic_answer':data['semantic_result'],'final_prediction':'A'}
  elif operation.startswith(('joint_read','semantic_compose','audit_joint')):
   slots=[];docs={s['doc_id']:s for s in data['source_spans']};text='\n'.join(s['text'] for s in data['source_spans'])
   for v,decl in data['variables'].items():
    val=None;used=[]
    if self.scenario=='rule':
     if v=='rule' and 'm1' in docs and 'm2' in docs:
      a=int(re.search(r'新增(\d+)秒',text).group(1));d=int(re.search(r'删除后(\d+)秒起',text).group(1))
      violation=op('and',op('ge',var('return_age'),const({'number':d,'unit':'s'})),var('returned'))
      fields={'D':op('if',violation,
                  const(False),var('delete_compliant')),
       'A':op('le',var('add_age'),const({'number':a,'unit':'s'})),'O':var('offline')}
      val={'variables':{'return_age':{'type':'number','unit':'s','demand':'D'},'returned':{'type':'bool','demand':'D'},
       'delete_compliant':{'type':'bool','demand':'D'},
       'add_age':{'type':'number','unit':'s','demand':'A'},'offline':{'type':'bool','demand':'O'}},
       'expression':{'op':'failures' if self.plan['output_contract']['mode']=='all_failures' else 'one_reason','fields':fields},
       'constraints':[op('not',op('and',var('delete_compliant'),violation))],
       'rules_complete':'无其他门槛' in text};used=['m1','m2']
     if v in ('return_age','returned') and 'm4' in docs and '针对 V8' in docs['m4']['text']:
      val=int(re.search(r'删除后(\d+)秒',docs['m4']['text']).group(1)) if v=='return_age' else '仍被返回' in docs['m4']['text'];used=['m4']
     if v=='add_age' and 'm8' in docs:
      val=int(re.search(r'新增后(\d+)秒',docs['m8']['text']).group(1));used=['m8']
     if v=='offline' and 'm9' in docs:
      val='测试通过' in docs['m9']['text'];used=['m9']
    else:
     patterns={'director':('film',r'directed by ([^.]+)'), 'university':('school',r'degrees of ([^.]+)'),
      'city':('city',r'located in ([^.]+)')}
     if v in patterns and patterns[v][0] in docs:
      doc,pattern=patterns[v];val=re.search(pattern,docs[doc]['text']).group(1);used=[doc]
     if decl['type']=='number':
      doc=v
      if doc in docs:
       val=int(re.search(r'(\d+)',docs[doc]['text']).group(1));used=[doc]
     if v=='advice' and docs:
      val='Choose a quiet visit based on the sourced preference.';used=list(docs)
    rows=[]
    if not operation.startswith('audit_joint'):
     if val is not None:
      rows=[{'id':v+'-observation','stance':'support','value':val,'source_ids':[docs[d]['id'] for d in used],
       'entity_scope':data['required_scope']['entity_scope'],'time_scope':data['required_scope']['time_scope'],
       'parent_versions':{p:x['version'] for p,x in data['supported_parents'].items() if x['demand'] in next(st['inputs'] for st in self.plan['steps'] if st['output_slot']==decl['demand'])},'retract_ids':[],
       'reason':'Scripted interpretation of supplied original sources'}]
      if self.bad_rows:rows[0]['source_ids']=['invented']
     elif docs:
      rows=[{'id':v+'-gap','stance':'partial','value':None,'source_ids':[docs.get({'rule':'m1','return_age':'m3','returned':'m3','add_age':'m6','offline':'m7'}.get(v),next(iter(docs.values())))['id']],
       'entity_scope':data['required_scope']['entity_scope'],'time_scope':data['required_scope']['time_scope'],
       'parent_versions':{},'retract_ids':[],'reason':'Only a clue; no value established'}]
    slots.append({'variable':v,'observations':rows,'unresolved':[] if val is not None or operation.startswith('audit_joint') else ['Still need an applicable observation']})
   value={'slots':slots}
   if operation.startswith('semantic_compose') or data.get('public_options'):
    terminal_values=[r['value'] for slot in slots if data['variables'][slot['variable']]['demand']==data['task']['output_slot']
                     for r in slot['observations'] if r['stance']=='support']
    value['semantic_answer']=terminal_values[0] if len(terminal_values)==1 else None
    value['final_prediction']=('A' if data.get('public_options') else value['semantic_answer']) if terminal_values else None
   if operation.startswith('audit_joint'):value['audit_verdict']='supported'
  elif operation.startswith('map'):
   relation={'m1':'rules','m2':'rules','m3':'D','m4':'D','m6':'A','m8':'A','m7':'O','m9':'O'}
   rows=[]
   for unit in data['units']:
    nid=relation.get(unit['doc_id'])
    assessments=[]
    if nid in unit['node_ids']:
     assessments=[{'span_ids':[unit['source_span_id']],'node_id':nid,'kind':'explicit',
      'stance':'partial' if unit['doc_id'] in ('m3','m6','m7') else 'support','claim':'Applicable exact original premise',
      'entity_scope':self.plan['output_contract']['entity_scope'],'event_time':None,'time_span_ids':[],
      'reason':'Scripted interpretation of raw source'}]
    rows.append({'unit_id':unit['unit_id'],'assessments':assessments,'irrelevance_reason':'' if assessments else 'Different scoped task'})
   value={'units':rows}
  elif operation=='resolve':
   nid=data['node_id'];evidence={x['doc_id']:x for x in data['evidence'] if nid in x['node_ids']}
   required={'rules':['m1','m2'],'D':['m4'],'A':['m8'],'O':['m9'],'decision':[]}[nid]
   value={'status':'unknown','answer':None,'alternatives':[], 'unresolved_inputs':[],'unresolved_guards':[],'refinements':[]}
   parents={x['id']:x for x in data['supported_parents']}
   if set(required)<=set(evidence) and (nid!='decision' or {'rules','D','A','O'}<=set(parents)):
    text='\n'.join(f['exact_quote'] for d in required for f in evidence[d]['fragments'])
    if nid=='rules':answer='R17-R2: '+text
    elif nid=='D':answer=json.dumps(not ('仍被返回' in text and int(re.search(r'删除后(\d+)秒',text).group(1))>=int(re.search(r'删除后(\d+)秒起',parents['rules']['answer']).group(1))))
    elif nid=='A':answer=json.dumps(int(re.search(r'新增后(\d+)秒',text).group(1))<=int(re.search(r'新增(\d+)秒',parents['rules']['answer']).group(1)))
    elif nid=='O':answer=json.dumps('测试通过' in text)
    else:
     failed=[v for v in ('D','A','O') if json.loads(parents[v]['answer']) is False]
     answer=json.dumps({'decision':not bool(failed),'reason':failed[0] if failed else None})
    value.update(status='supported',answer=answer,alternatives=[{'source_span_ids':[evidence[d]['id'] for d in required],
     'guard_span_ids':[],'used_parent_ids':data['planned_parent_ids'],'applicable_scope':self.plan['output_contract']['entity_scope'],
     'semantic_status':'supported'}])
    if nid=='decision':value['final_prediction']=answer
  elif operation=='audit':value={'conflicts':[],'unresolved_guards':[]}
  else:raise AssertionError('Unexpected paid operation: '+operation)
  return {'response_ref':'synthetic-'+str(len(self.requests)),
   'response':{'choices':[{'finish_reason':'stop','message':{'content':json.dumps(value,ensure_ascii=False)}}]}}

def run_demo(mode='one_reason',method=RESIDUAL,correction=False,**settings):
 passages=deepcopy(PASSAGES)
 if correction:passages['m4']=passages['m4'].replace('针对 V8','针对 V7')
 axes={'m1':0,'m5':0,'m2':1,'m3':2,'m4':3,'m6':4,'m8':5,'m7':6,'m9':7}
 vectors=np.array([[float(i==axes[d])*(.99 if d=='m5' else 1) for i in range(8)] for d in passages],np.float32)
 services=SourcesOnly(rule_plan(mode))
 return run(services,passages,vectors,method,{'initial_width':2,**settings})

def run(services,passages,vectors,method=RESIDUAL,settings=None,options=None):
 docs={d:SimpleNamespace(doc_id=d,passage=t,title=d,text=t) for d,t in passages.items()}
 services.legacy=not capabilities(method).joint_reading
 services.scored=method==LOCAL
 config={'_test_transport':True,'llm_base_url':'http://synthetic/v1','llm_model':'source-script',
  'embedding_base_url':'http://synthetic/v1','embedding_model':'source-script',
  'fusion':{'initial_width':1,'proposal_width':1,'max_feedback_rounds':0,**(settings or {})}}
 if services.scored:config['reranker']={'url':'http://synthetic/rerank','model':'source-script'}
 if config['fusion'].pop('fixed_pool',False):config['fixed_candidate_pools']={'synthetic-'+services.scenario:list(docs)}
 question=('根据本次演示验收要求，岚桥同步方案现在可以启用吗？请列出全部不满足的条件。'
           if services.plan['output_contract']['mode']=='all_failures' else
           '根据本次演示验收要求，岚桥同步方案现在可以启用吗？不能的话，给一个足以否决的条件即可。')
 engine=Engine({'id':'synthetic-'+services.scenario,'question':question if services.scenario=='rule' else 'Synthetic source question'},
  (docs,list(docs),vectors,None,Tokenizer()),services,config,method,
  reader_question='Public output' if options else None,output_options=options)
 row=engine.run()
 return row,services.requests,engine

def run_chain():
 plan={'steps':[step('director','Who directed Ash Wind?'),step('university','Which university did {director} attend?',['director']),
  step('city','Which city contains {university}?',['university'])],'final_node_id':'city',
  'output_contract':contract('value','entity','Ash Wind'),
  'program':{'variables':{v:{'type':'entity','demand':v,'open':True} for v in ('director','university','city')},'expression':var('city')}}
 passages={'film':'Ash Wind was directed by Lin Zhou.','school':'Lin Zhou earns degrees of North University.','city':'North University is located in River City.'}
 return run(SourcesOnly(plan,scenario='chain'),passages,np.eye(3,dtype=np.float32))

def run_compare(record=False, options=None):
 expression={'op':'record','fields':{'x':var('x'),'y':var('y')}} if record else op('lt',var('x'),var('y'))
 plan={'steps':[step('x','Find x quantity'),step('y','Find y quantity'),step('final','Compare x and y',['x','y'],'compose')],
  'final_node_id':'final','output_contract':contract('value','record' if record else 'bool','quantities'),
  'program':{'variables':{v:{'type':'number','demand':v} for v in ('x','y')},'expression':expression}}
 # Distinct queries both fixed dense rank; full visible source here proves multi-field reuse.
 passages={'x':'x = 0.','y':'y = 2.'}
 return run(SourcesOnly(plan,scenario='comparison'),passages,np.array([[1,0,0],[.9,.1,0]],np.float32),settings={'initial_width':2},options=options)

def run_advice(options=None):
 plan={'steps':[step('preference','Find x preference quantity'),step('advice','Recommend from {preference}',['preference'],'compose')],
  'final_node_id':'advice','output_contract':contract('semantic','text','visitor'),
  'program':{'variables':{'x':{'type':'number','demand':'preference'},'advice':{'type':'text','demand':'advice'}},'expression':var('advice')}}
 return run(SourcesOnly(plan,scenario='advice'),{'x':'Preference 0: quiet places.'},np.array([[1,0,0]],np.float32),options=options)

def run_non_singleton():
 plan={'steps':[step('uncertain','Find the undecided boolean condition')],'final_node_id':'uncertain',
  'output_contract':contract('value','bool','undecided condition'),
  'program':{'variables':{'uncertain':{'type':'bool','demand':'uncertain'}},'expression':var('uncertain')}}
 return run(SourcesOnly(plan,scenario='non_singleton'),{'clue':'The condition may be true or false; it is not decided.'},np.array([[1,0,0]],np.float32))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args()
 rows={name:fn()[0] for name,fn in [('one_reason',run_demo),('all_failures',lambda:run_demo('all_failures')),
  ('V7_correction',lambda:run_demo(correction=True)),('entity_chain',run_chain),('comparison',run_compare),
  ('record_output',lambda:run_compare(True)),('non_singleton_output',run_non_singleton),('open_advice',run_advice)]}
 public={n:{'answer':r['answer'],'cost':r['diagnostics']['cost'],
  'trajectory':[e for e in r['diagnostics']['events'] if e['event'] in ('residual_state','typed_dag_initialized','joint_read_completed')]} for n,r in rows.items()}
 if args.output:
  args.output.mkdir(parents=True,exist_ok=True);args.output.chmod(0o700)
  (args.output/'synthetic.json').write_text(json.dumps(public,ensure_ascii=False,indent=2))
 for name,row in rows.items():print(name,row['answer']['status'],row['answer']['prediction'],row['diagnostics']['cost'])
