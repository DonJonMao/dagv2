"""Full packaged-data protocol smoke; ONLY remote model services are scripted.

This is not an NLP accuracy experiment. The fixed answer and semantic judgments
are deliberately arbitrary fixtures. It executes native preparation/tokenizer,
both real worker algorithms, current BT, persistence, scoring and paired output.
The HTTP service never opens or receives evaluation labels.
"""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from dagbt import runner


FIXTURE_ANSWER = "SCRIPTED_HTTP_PROTOCOL_FIXTURE_NOT_A_MODEL_PREDICTION"


def scripted_source_mapping(data):
    """Exercise the actual visible span-ID protocol, with scripted semantics."""
    sources = {source['id']: source for source in data['source_spans']}
    units = []
    for unit in data['units']:
        source = sources[unit['source_span_id']]
        assert source['doc_id'] == unit['doc_id']
        assessments = [{
            'span_ids': [source['id']], 'node_id': node['output_slot'],
            'kind': 'explicit', 'stance': 'support',
            'claim': 'Scripted source-ID fixture: ' + source['text'][:80],
            'entity_scope': 'scripted protocol fixture only', 'event_time': None,
            'time_span_ids': [], 'reason': 'Fixture protocol; no semantic accuracy claim',
        } for node in data['nodes'] if node['output_slot'] in unit['node_ids']] if source['text'].strip() else []
        units.append({'unit_id': unit['unit_id'], 'assessments': assessments,
                      'irrelevance_reason': '' if assessments else 'Empty fixture source'})
    return {'units': units}


@contextmanager
def scripted_models(question, vector):
    requests, errors = [], []
    usage = {'prompt_tokens': 11, 'completion_tokens': 7, 'total_tokens': 18}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, body, status=200):
            raw = json.dumps(body).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers(); self.wfile.write(raw)

        def do_GET(self):
            self.respond({'data': [{'id': 'qwen3.8-27b'}, {'id': 'nvidia/NV-Embed-v2'},
                                  {'id': 'fixture-pointwise'}]})

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append({'path': self.path, 'payload': payload})
            try:
                if self.path.endswith('/embeddings'):
                    self.respond({'data': [{'index': i, 'embedding': vector}
                                           for i, _ in enumerate(payload['input'])], 'usage': usage})
                    return
                if self.path.endswith('/rerank'):
                    documents = payload['documents']
                    self.respond({'results': [
                        {'index': i, 'relevance_score': min(.9, .1 + .06 * documents[i].count('[Passage '))}
                        for i in reversed(range(len(documents)))], 'usage': usage})
                    return
                if self.path.endswith('/completions') and not self.path.endswith('/chat/completions'):
                    schema = payload['structured_outputs']['json']
                    count = schema['properties']['sources']['minItems']
                    value = {'answer': FIXTURE_ANSWER, 'sources': [True] + [False] * (count - 1)}
                    self.respond({'choices': [{'text': json.dumps(value), 'finish_reason': 'stop'}], 'usage': usage})
                    return
                messages = payload['messages']
                system = messages[0]['content']
                if system.startswith('Decompose a multi-hop question'):
                    value = {'steps': [{'question': question, 'output_slot': 'answer',
                                        'answer_type': 'answer', 'inputs': []}]}
                elif system.startswith('Map raw corpus passages'):
                    data = json.loads(messages[1]['content'])
                    value = scripted_source_mapping(data)
                elif system.startswith('Resolve ONE executable subquestion'):
                    data = json.loads(messages[1]['content'])
                    assert data['evidence'], 'Production mapper supplied no exact evidence to resolver'
                    value = {'status': 'supported', 'answer': FIXTURE_ANSWER,
                             'alternatives': [{'source_span_ids': [data['evidence'][0]['id']],
                                               'guard_span_ids': [], 'used_parent_ids': [],
                                               'applicable_scope': 'scripted protocol fixture only',
                                               'semantic_status': 'supported'}],
                             'unresolved_inputs': [], 'unresolved_guards': [], 'refinements': []}
                elif system.startswith('Audit the compiled evidence support alternatives'):
                    value = {'conflicts': [], 'unresolved_guards': []}
                elif system.startswith('You are a long-document QA reader'):
                    content = 'Answer: ' + FIXTURE_ANSWER
                    self.respond({'choices': [{'message': {'content': content}, 'finish_reason': 'stop'}],
                                  'usage': usage})
                    return
                else:
                    raise AssertionError('Unexpected production prompt: ' + system[:100])
                self.respond({'choices': [{'message': {'content': json.dumps(value)}, 'finish_reason': 'stop'}],
                              'usage': usage})
            except Exception as exc:
                errors.append(repr(exc))
                self.respond({'fixture_error': repr(exc)}, status=400)

    httpd = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True); thread.start()
    try:
        yield f'http://127.0.0.1:{httpd.server_port}', requests, errors
    finally:
        httpd.shutdown(); httpd.server_close(); thread.join(timeout=3)


@pytest.mark.parametrize('fusion_arm', ['fusion','fusion_proxy_free'])
def test_actual_packaged_hotpot_question_both_algorithms_with_scripted_http(tmp_path, fusion_arm):
    q = runner.load_questions('hotpotqa', limit=1)[0]
    # Only public corpus vectors enter the scripted service; no labels are read here.
    vectors = np.load(runner.ROOT/'data/hotpotqa/index/passage_vectors.npy', mmap_mode='r', allow_pickle=False)
    vector = np.asarray(vectors[0], dtype=np.float32).tolist()
    assert len(vector) == 4096
    output = tmp_path/'scripted_model_free_system_test'
    with scripted_models(q['question'], vector) as (endpoint, requests, errors):
        config = runner.load(runner.ROOT/'configs/paired.legacy.json')
        config.update(llm_base_url=endpoint+'/v1', embedding_base_url=endpoint+'/v1',
                      request_timeout_seconds=10, max_identical_attempts=1)
        config['reranker'].update(url=endpoint+'/rerank', model='fixture-pointwise')
        if fusion_arm == 'fusion_proxy_free':
            config.pop('reranker')  # This arm must not require even a preflight reranker service.
        config['fusion'].update(ann_calls=12, set_score_calls=64)
        config['experiment'].update(question_timeout_seconds=180, worker_startup_timeout_seconds=120,
                                    fixture_note='Scripted HTTP, real algorithms, not model accuracy')
        path = tmp_path/'scripted_config.json'; runner.save(path, config)
        args = SimpleNamespace(config=str(path), output=str(output), datasets=['hotpotqa'],
                               arms=['original',fusion_arm], limit=1, retry_failed=False, offline_preflight=False)
        runner.run_experiment(args)
        assert not errors, errors
        rows = {arm: runner.load(runner.result_path(output, 'hotpotqa', arm, q['id']))
                for arm in ['original',fusion_arm]}
        assert {arm: row['answer']['status'] for arm,row in rows.items()} == {'original':'ok',fusion_arm:'ok'}, rows
        assert all(row['unit_id'] == q['id'] and row['answer']['prediction'] == FIXTURE_ANSWER for row in rows.values())
        assert runner.load(output/'progress.json')['state'] == 'complete'
        comparison = json.loads((output/'hotpotqa/comparisons.jsonl').read_text().strip())
        assert comparison['unit_id'] == q['id'] and set(comparison['arms']) == {'original',fusion_arm}
        # Confirm this used the source search and the actual map/resolve/closure/reader path.
        fusion = rows[fusion_arm]
        expected_version = ('evidence_bridge_v1' if fusion_arm == 'fusion'
                            else 'day2_proxy_free_requirements_ann_v1')
        assert fusion['ranking']['trace'][0]['trace']['search_archive']['method_version'] == expected_version
        diag = fusion['diagnostics']
        assert diag['spans'] and diag['support_graph']['nodes'][0]['status'] == 'supported'
        if fusion_arm == 'fusion':
            assert diag['ledger']['used']['set_score'] > 0
        else:
            assert diag['ledger']['used'].get('set_score',0) == 0
            assert diag['ledger']['used'].get('rerank_http',0) == 0
            trace=fusion['ranking']['trace'][0]['trace']
            assert any(b['stage']=='conditional' and b['premise_ids']
                       for b in trace['retrieval']['proposal_batches'])
        assert diag['ledger']['used']['ann'] <= 12
        assert diag['ledger']['used'].get('set_score',0) <= 64
        assert any(e['event'] == 'reader_input' and e['raw_only'] for e in diag['events'])
        assert rows['original']['ranking']['nodes'][0]['resolved']
        assert all(row['runner']['cost']['http_attempts'] > 0 for row in rows.values())
        # Same packaged question goes to both planners; evaluation fields never go to HTTP.
        planners = [r for r in requests if r['payload'].get('messages', [{}])[0].get('content','').startswith('Decompose a multi-hop question')]
        assert len(planners) == 2
        assert all(q['question'] in r['payload']['messages'][1]['content'] for r in planners)
        wire = json.dumps(requests)
        assert 'gold_groups' not in wire and 'evaluation_only.json' not in wire
        assert any(r['path'] == '/v1/completions' for r in requests)
        assert any(r['path'] == '/rerank' for r in requests) == (fusion_arm == 'fusion')
        report = {'kind':'scripted_http_protocol_test_not_accuracy', 'question_id':q['id'],
                  'arms':['original',fusion_arm], 'actual_native_workers':True,
                  'actual_packaged_corpus_documents':9811, 'embedding_dimensions':4096,
                  'fixture_answer':FIXTURE_ANSWER, 'http_requests':len(requests),
                  'generation_gold_fields_sent':False}
        runner.save(output/'SCRIPTED_TEST_ONLY.json', report)
