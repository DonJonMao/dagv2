#!/usr/bin/env python3
"""Fixed invented corpus/model/plan/budgets; report every control and failure.

Online baseline vs joint changes more than one mechanism; the separate fixed-pool
JOINT/RESIDUAL pair isolates output-driven task elimination with the same raw range.
"""
import argparse
import json
from pathlib import Path
from demo_residual_memory import run_demo,rule_plan,PASSAGES
from dagbt.methods import LOCAL,UNSCORED,JOINT,RESIDUAL
from dagbt.transport import digest

def run_ablation():
 rows=[]
 for pool,methods in ((False,(LOCAL,UNSCORED,JOINT,RESIDUAL)),(True,(JOINT,RESIDUAL))):
  for method in methods:
   result,requests,e=run_demo(method=method,fixed_pool=pool)
   rows.append({'method':method,'retrieval_control':'fixed_full_pool' if pool else 'fixed_embedding_responder',
    'status':result['answer']['status'],'prediction':result['answer']['prediction'],
    'completion_kind':result['answer'].get('completion_kind','legacy_model_terminal'),
    'cost':result['diagnostics']['cost'],'unfinished':result['answer']['status']!='ok',
    'visited_tasks':[s['id'] for s in result['ranking']['nodes'] if s['status']=='supported'],
    'raw_range_identity':digest(PASSAGES),'dag_identity':digest(rule_plan('one_reason')['steps']),
    'resource_limits':{'ann':36,'llm':24,'reader':0,'set_score':512 if method==LOCAL else 0},
    'interpretation':'Scripted semantic service; mechanics only, no natural-language quality claim'})
 return {'rows':rows,'unfinished_count':sum(r['unfinished'] for r in rows),
  'controls':'scored f4 / matched unscored f4 / joint unscored / full residual',
  'limitations':'Online bridge policy differs between the old eager search and pausable joint path. Do not attribute all online savings to residual evaluation; compare the fixed-pool pair for that intervention. All source returns and plans are fixed, no gold labels supplied.'}
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path);args=parser.parse_args()
 report=run_ablation()
 if args.output:
  args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(report,ensure_ascii=False,indent=2))
