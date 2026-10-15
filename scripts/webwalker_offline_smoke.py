#!/usr/bin/env python3
"""Offline installation verification with real page and model HTTP fixtures.
These are invented sources and deterministic responses, not a benchmark score.
No external website/model requests are made.
"""
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import tempfile
from threading import Thread
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dagbt.methods import JOINT, RESIDUAL_V2 as RESIDUAL
from dagbt.model_runtime import BTTokenAccounting
from dagbt.webwalker_engine import WebEngine

def step(n,q,parents=(),kind='retrieval'):
    return {'output_slot':n,'question':q,'inputs':list(parents),'execution':kind,'answer_type':'fact'}
def var(v): return {'op':'var','id':v}
def op(name,*args): return {'op':name,'args':list(args)}
def contract(mode,kind,scope):
    return {'mode':mode,'result_type':kind,'entity_scope':scope,'time_scope':'fixture','explanation':'none'}
def website_fixture():

    class Handler(BaseHTTPRequestHandler):
        hits = []

        def do_GET(self):
            self.hits.append(self.path)
            pages = {'/': '<title>Package project</title><p>Primary package: Star E3.</p><a href="/offline">Offline report</a><a href="/backup">Backup package</a>', '/offline': '<title>Offline report</title><p>Star E3 can run offline: true.</p>', '/backup': '<title>Backup</title><p>Backup package: Moon B2.</p>', '/bad': None, '/empty': '<script>var a=1</script>'}
            if self.path not in pages or pages[self.path] is None:
                self.send_error(404)
                return
            raw = pages[self.path].encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield ('http://127.0.0.1:' + str(server.server_port), Handler.hits)
    server.shutdown()
    server.server_close()
    thread.join()

def plan():
    return {'steps': [step('primary', 'Find primary package'), step('offline', 'Can {primary} run offline?', ['primary']), step('backup', 'Find backup package after checking {offline}', ['offline']), step('selection', 'Select primary if offline else backup', ['primary', 'offline', 'backup'], 'compose')], 'final_node_id': 'selection', 'output_contract': contract('value', 'entity', 'package'), 'program': {'variables': {'p': {'type': 'entity', 'demand': 'primary'}, 's': {'type': 'bool', 'demand': 'offline'}, 'b': {'type': 'entity', 'demand': 'backup'}}, 'expression': op('if', var('s'), var('p'), var('b'))}}

class Model:
    """Strict source-only fixture; validates exact wire source packing."""

    def __init__(self):
        self.requests = []

    def get(self, stage, url, payload):
        self.requests.append((stage[0], deepcopy(payload)))
        assert not url.endswith('/embeddings') and 'rerank' not in url
        operation = stage[0]
        data = json.loads(payload['messages'][1]['content'])
        if operation == 'planner':
            value = plan()
        elif operation == 'web_navigate':
            target = '/backup' if 'backup' in data['subproblem'].lower() else '/offline'
            value = {'link_id': next((l['id'] for l in data['links'] if l['url'].endswith(target)), None)}
        else:
            if 'source_span_columns' in data:
                data['source_spans'] = [dict(zip(data['source_span_columns'], r)) for r in data['source_spans']]
            for s in data['source_spans']:
                if data.get('source_documents'):
                    s.update(data['source_documents'][s['doc_id']])
            slots = []
            for variable in data['variables']:
                (marker, answer) = {'p': ('Primary package: Star E3', 'Star E3'), 's': ('can run offline: true', True), 'b': ('Backup package: Moon B2', 'Moon B2')}[variable]
                sources = [s['id'] for s in data['source_spans'] if marker in s['text']]
                rows = []
                if sources and (not operation.startswith('audit_joint')):
                    scope = data.get('requested_scopes', {}).get(variable, data['required_scope'])
                    rows = [{'id': variable + '-source', 'stance': 'support', 'value': answer, 'source_ids': sources, **scope, 'parent_versions': {p: v['version'] for (p, v) in data['supported_parents'].items() if v['demand'] in next((s['inputs'] for s in plan()['steps'] if s['output_slot'] == data['variables'][variable]['demand']))}, 'retract_ids': [], 'reason': 'Explicit page statement'}]
                slot = {'variable': variable, 'observations': rows, 'unresolved': []}
                if operation.startswith('joint_read') and data['task']['execution'] == 'retrieval':
                    slot['reading_status'] = ('evidence_assessment' if rows else 'no_relevant_evidence')
                slots.append(slot)
            value = {'slots': slots}
            if operation.startswith('audit_joint'):
                value['audit_verdict'] = 'supported'
        return {'response': {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(value)}}]}, 'response_ref': 'fixture-' + str(len(self.requests))}

def main():
    fixture=website_fixture()
    root,hits=next(fixture)
    model=Model()
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            count=max(1,len(json.dumps(payload['messages'],ensure_ascii=False))//4)
            if self.path=='/tokenize':
                value={'count':count,'max_model_len':16384}
            else:
                data=json.loads(payload['messages'][1]['content'])
                operation=('planner' if isinstance(data,str) else 'web_navigate' if 'links' in data
                           else 'audit_joint' if 'audit_instruction' in data else 'joint_read')
                value=model.get((operation,'http'), 'http://fixture/chat/completions',payload)['response']
                value['usage']={'prompt_tokens':count,'completion_tokens':50,'total_tokens':count+50}
            raw=json.dumps(value).encode()
            self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(raw)
        def log_message(self,*args): pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=Thread(target=server.serve_forever,daemon=True);thread.start()
    endpoint='http://127.0.0.1:'+str(server.server_port)
    summaries=[]
    try:
        with tempfile.TemporaryDirectory(prefix='dagbt-webwalker-smoke-') as directory:
            directory=Path(directory)
            for method in (RESIDUAL,JOINT):
                config={'model_profile':'bridgetree','llm_base_url':endpoint+'/v1','llm_model':'offline-fixture',
                        'request_timeout_seconds':10,'max_identical_attempts':1,
                        'tokenization':{'mode':'vllm_chat','endpoint':endpoint+'/tokenize','timeout_seconds':5},
                        'webwalker':{'max_actions':6,'http_attempts':1},
                        'fusion':{'set_score_calls':0,'reader_calls':0,'initial_width':1,'proposal_width':1}}
                engine=WebEngine({'id':'offline-http','question':'Use primary if offline, otherwise backup.'},
                                 ({},[],np.empty((0,0),dtype=np.float32),None,BTTokenAccounting()),
                                 SimpleNamespace(output=str(directory/method)),config,method,
                                 root_url=root,page_cache=directory/'pages',allow_local=True)
                row=engine.run()
                assert row['answer']['status']=='ok',row['answer']
                assert row['answer']['prediction']=='Star E3'
                expected=2 if method==RESIDUAL else 3
                assert row['diagnostics']['web']['actions']==expected,row['diagnostics']['web']
                assert row['diagnostics']['cost']['ann_calls']==0
                assert engine.calls.token_counter is not None
                summaries.append({'arm':method,'status':'ok','web_actions':expected,
                                  'completion_kind':row['answer']['completion_kind']})
    finally:
        server.shutdown();server.server_close();thread.join()
        try: next(fixture)
        except StopIteration: pass
    print(json.dumps({'offline_http_smoke':summaries,'external_network_used':False},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
