#!/usr/bin/env python3
"""Fully offline, source-driven example with the actual engine and frozen BT.

The scripted model and embedding/scoring services demonstrate mechanics only.
All entities/passages are invented; no evaluation labels or real user data.
"""
from pathlib import Path
import argparse
import json
import re
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dagbt.engine import Engine
from dagbt.local_terminal import VERSION


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return '\n'.join(m['content'] for m in messages)

    def encode(self, text, **kwargs):
        return text.split()


PASSAGES = {
    'film': 'Ash Wind (2012) was directed by Lin Zhou.',
    'interview': 'Lin Zhou: I graduated in class K7 of the Screen Arts academy.',
    'class': 'Archive: class K7 belonged to the School of Imaging at the Screen Arts academy.',
    'school': 'The School of Imaging at the Screen Arts academy awards degrees of North University.',
    'city': 'North University is located in River City.',
}


class SourcesOnlyServices:
    def __init__(self):
        self.requests = []

    def get(self, stage, url, payload):
        self.requests.append({'stage':list(stage),'payload':payload})
        if url.endswith('/embeddings'):
            task = payload['input'][0].split('Current retrieval task:\n',1)[-1].split('\n\n',1)[0]
            if 'Which university' in task:
                anchors = payload['input'][0].split('Anchor passages:\n',1)
                index = 2 if len(anchors)>1 and 'graduated in class K7' in anchors[1] else 1
            elif 'Which city' in task:
                index = 3
            else:
                index = 0
            return {'response':{'data':[{'index':0,'embedding':[float(i==index) for i in range(4)]}]}}
        if url.endswith('/rerank'):
            query = payload['query']
            def score(text):
                if 'Which university' in query:
                    # Complementarity requires all THREE raw source premises.
                    return .95 if all(PASSAGES[d] in text for d in ('interview','class','school')) else .2 if PASSAGES['interview'] in text else .1
                relevant = 'city' if 'Which city' in query else 'film'
                return .9 if PASSAGES[relevant] in text else .1
            return {'response':{'results':[{'index':i,'relevance_score':score(t)} for i,t in enumerate(payload['documents'])]}}
        data = json.loads(payload['messages'][1]['content'])
        op = stage[0]
        if op == 'planner':
            value = {'final_node_id':'city', 'steps':[
                {'output_slot':'director','question':'Who directed Ash Wind (2012)?','inputs':[], 'answer_type':'person','execution':'retrieval'},
                {'output_slot':'university','question':'Which university did {director}, director of Ash Wind (2012), graduate from?',
                 'inputs':['director'],'answer_type':'university','execution':'retrieval'},
                {'output_slot':'city','question':'Which city contains {university}?','inputs':['university'],
                 'answer_type':'city','execution':'retrieval'}]}
        elif op in ('map','map_repair'):
            rows=[]
            for unit in data['units']:
                nid = 'director' if unit['doc_id']=='film' else 'city' if unit['doc_id']=='city' else 'university'
                assessment = {'span_ids':[unit['source_span_id']],'node_id':nid,'kind':'explicit',
                    'stance':'support' if nid!='university' else 'partial','claim':'Exact factual source premise',
                    'entity_scope':'Ash Wind (2012) / Lin Zhou','event_time':None,'time_span_ids':[],
                    'reason':'Scripted source-grounded assessment'}
                rows.append({'unit_id':unit['unit_id'],
                    'assessments':[assessment] if nid in unit['node_ids'] else [],
                    'irrelevance_reason':'' if nid in unit['node_ids'] else 'Different task'})
            value={'units':rows}
        elif op == 'resolve':
            nid=data['node_id']
            evidence={e['doc_id']:e for e in data['evidence'] if nid in e['node_ids']}
            required={'director':['film'],'university':['interview','class','school'],'city':['city']}[nid]
            value={'status':'unknown','answer':None,'alternatives':[],
                   'unresolved_inputs':[],'unresolved_guards':[],'refinements':[]}
            if set(required)<=set(evidence):
                # Extract each conclusion from its supplied quoted source, rather
                # than a gold answer table. The university inference requires the
                # interview->K7->School of Imaging->degree university links.
                source = '\n'.join(f['exact_quote'] for d in required for f in evidence[d]['fragments'])
                patterns={'director':r'directed by ([^.]+)', 'university':r'degrees of ([^.]+)',
                          'city':r'located in ([^.]+)'}
                conclusion=re.search(patterns[nid],source).group(1)
                parents={'director':[],'university':['director'],'city':['university']}[nid]
                value.update(status='supported',answer=conclusion,alternatives=[{
                    'source_span_ids':[evidence[d]['id'] for d in required], 'guard_span_ids':[],
                    'used_parent_ids':parents,'applicable_scope':'Ash Wind (2012) / Lin Zhou',
                    'semantic_status':'supported'}])
                if nid=='city': value['final_prediction']=conclusion
        elif op=='audit':
            value={'conflicts':[],'unresolved_guards':[]}
        else:
            raise AssertionError('Unexpected generation: '+op)
        return {'response_ref':'synthetic-'+str(len(self.requests)),
            'response':{'choices':[{'finish_reason':'stop','message':{'content':json.dumps(value)}}]}}


def run_demo():
    docs={d:SimpleNamespace(doc_id=d,title=d,text=t,passage=d+'\n'+t) for d,t in PASSAGES.items()}
    vectors=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,.99,.01],[0,0,0,1]],dtype=np.float32)
    services=SourcesOnlyServices()
    config={'_test_transport':True,'llm_base_url':'http://synthetic/v1','llm_model':'source-script',
        'embedding_base_url':'http://synthetic/v1','embedding_model':'source-script',
        'reranker':{'url':'http://synthetic/rerank','model':'source-script','batch_size':4},
        'fusion':{'initial_width':1,'proposal_width':2,'pair_rescue_width':2,
                  'max_feedback_rounds':0,'refinement':False}}
    engine=Engine({'id':'synthetic-director-university-city',
        'question':'Which city contains the university from which the director of Ash Wind (2012) graduated?'},
        (docs,list(docs),vectors,SimpleNamespace(),Tokenizer()),services,config,VERSION)
    row=engine.run()
    return row,services.requests


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    row,requests=run_demo()
    if args.output:
        args.output.mkdir(parents=True,exist_ok=False)
        (args.output/'result.json').write_text(json.dumps(row,indent=2)+'\n')
        (args.output/'requests.json').write_text(json.dumps(requests,indent=2)+'\n')
    print(json.dumps({'answer':row['answer'],'cost':row['diagnostics']['cost']},indent=2))
    return 0 if row['answer']['status']=='ok' else 1


if __name__=='__main__':
    raise SystemExit(main())
