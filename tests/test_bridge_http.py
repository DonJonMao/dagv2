"""Loopback HTTP integration: production BridgeSession and transport, no external models."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

import pytest

from dagbt.bridge import probe_reranker_protocol
from dagbt.budget import Ledger
from dagbt.transport import Transport
from vendor.bridgetree.dependency_scoring import SetResponseError
from test_bridge import session, requirement


@contextmanager
def server(*, listwise=False, fail_once=False):
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append({'path': self.path, 'payload': payload})
            if fail_once and len(requests) == 1:
                self.send_response(503); self.end_headers(); return
            if self.path.endswith('/embeddings'):
                body = {'data': [{'index': 0, 'embedding': [0.,1.,0.,0.]}],
                        'usage': {'prompt_tokens': len(payload['input'][0].split())}}
            elif self.path.endswith('/chat/completions'):
                body = {'choices': [{'message': {'content': '{}'}, 'finish_reason': 'stop'}],
                        'usage': {'prompt_tokens': 10, 'completion_tokens': 2}}
            else:
                documents = payload['documents']
                assert payload['model'] == 'fixture-pointwise'
                if listwise:
                    scores = [1 / len(documents)] * len(documents)
                else:
                    # A fixed pointwise service response depending only on this
                    # query-document pair; sorting tests indexed reconstruction.
                    scores = [min(.9, .08 + .07 * text.count('[Passage ') +
                                  .05 * ('France' in text)) for text in documents]
                body = {'results': [{'index': i, 'relevance_score': scores[i]}
                                    for i in reversed(range(len(documents)))]}
            encoded = json.dumps(body).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(encoded)))
            self.end_headers(); self.wfile.write(encoded)
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True); thread.start()
    try:
        yield f'http://127.0.0.1:{httpd.server_port}', requests
    finally:
        httpd.shutdown(); httpd.server_close(); thread.join(timeout=3)


def configure(s, endpoint, tmp_path):
    s.config.update({'embedding_base_url': endpoint+'/v1', 'embedding_model': 'fixture-embedding',
                     'llm_base_url': endpoint+'/v1', 'llm_model': 'unused-fixture-reader',
                     'max_identical_attempts': 2, 'request_timeout_seconds': 3,
                     'reranker': {'url': endpoint+'/rerank', 'model': 'fixture-pointwise'}})
    s.embedding_url = endpoint+'/v1/embeddings'
    s.calls = Transport('fixture-question', tmp_path, s.config, s.ledger, None)


def test_local_scoring_real_http_payload_and_cache_contexts(tmp_path):
    from dagbt.bridge import LOCAL_TERMINAL_VERSION
    with server() as (endpoint, requests):
        s = session(ann=12,sets=64,algorithm_version=LOCAL_TERMINAL_VERSION)
        configure(s,endpoint,tmp_path)
        s.discover('Find the director','director',remaining_nodes=2)
        s.discover('Find the university of the resolved director','university',remaining_nodes=1,
                   scoring_identity={'parent_bindings':[{'answer':'Person','version':1,'applicable_scope':'2012'}]})
        reranks=[r['payload'] for r in requests if r['path'].endswith('/rerank')]
        assert {p['query'] for p in reranks} == {'Find the director','Find the university of the resolved director'}
        records=[json.loads(p.read_text()) for p in (tmp_path/'requests').glob('*.json')]
        scored=[r for r in records if r['url'].endswith('/rerank')]
        assert len({r['scoring_context_id'] for r in scored}) == 2
        assert all(r['algorithm_version']==LOCAL_TERMINAL_VERSION for r in scored)
        assert sum(sc.scored_sets for sc in s.scorers.values()) == s.ledger.used['set_score'] <= 64


def test_real_http_latest_search_provenance_and_physical_accounting(tmp_path):
    with server() as (endpoint, requests):
        s = session(ann=10, sets=32)
        configure(s, endpoint, tmp_path)
        result = s.discover('Where was the author born?', 'birth', requirement(), ['d0'])
        assert result['trace']['search_archive']['method_version'] == 'evidence_bridge_v1'
        assert result['candidate_ids']
        embeddings = [r for r in requests if r['path'].endswith('/embeddings')]
        reranks = [r for r in requests if r['path'].endswith('/rerank')]
        assert embeddings and reranks
        assert all(r['payload']['model'] == 'fixture-embedding' for r in embeddings)
        assert all(r['payload']['model'] == 'fixture-pointwise' for r in reranks)
        assert all(r['payload']['query'] == s.original_query for r in reranks)
        assert s.ledger.used['ann'] <= 8
        assert s.ledger.used['set_score'] == s.scorer.scored_sets <= 32
        assert s.ledger.used['http_attempts'] == len(requests)
        # Transport counts cache replay separately; it is never a physical HTTP request.
        assert s.ledger.used['embedding_http'] + s.ledger.used['rerank_http'] == (
            len(requests) + s.ledger.used['cache_hits'])
        saved = [json.loads(p.read_text()) for p in (tmp_path/'requests').glob('*.json')]
        assert len(saved) == len(requests)
        assert all('response' in x and x['attempts'] for x in saved)
        assert all(e['dependency_claim'] is False for e in result['trace']['retrieval']['source_graph'])


def test_real_http_five_request_pointwise_probe_and_cache_replay(tmp_path):
    with server() as (endpoint, requests):
        s = session(); configure(s, endpoint, tmp_path)
        report = probe_reranker_protocol(s.calls, s.config, s.ledger)
        assert report['consistent'] is True
        assert report['reranker_adapter_requests'] == 5
        assert len(requests) == s.ledger.used['http_attempts'] == 5
        assert s.ledger.used['ann'] == s.ledger.used['set_score'] == s.ledger.used['llm'] == 0
        probe_reranker_protocol(s.calls, s.config, s.ledger)
        assert len(requests) == 5
        assert s.ledger.used['cache_hits'] == 5
        assert s.ledger.used['rerank_http'] == 10


def test_real_http_listwise_service_rejected_before_experiment(tmp_path):
    with server(listwise=True) as (endpoint, requests):
        s = session(); configure(s, endpoint, tmp_path)
        with pytest.raises(SetResponseError, match='batch composition/order'):
            probe_reranker_protocol(s.calls, s.config, s.ledger)
        assert len(requests) == 5
        assert any(e['event'] == 'reranker_protocol_failed' for e in s.ledger.events)
        assert s.ledger.used['set_score'] == 0


def test_real_http_retry_attempts_are_distinct_from_logical_ann(tmp_path, monkeypatch):
    monkeypatch.setattr('dagbt.transport.time.sleep', lambda _: None)
    with server(fail_once=True) as (endpoint, requests):
        s = session(); configure(s, endpoint, tmp_path)
        s.discover('Find author', 'author', requirement(), mode='dense')
        assert len(requests) == 2
        assert s.ledger.used['ann'] == 1
        assert s.ledger.used['embedding_http'] == s.ledger.used['http_attempts'] == 2
        record = json.loads(next((tmp_path/'requests').glob('*.json')).read_text())
        assert [a['http_status'] for a in record['attempts']] == [503, 200]


def test_cached_llm_retry_replay_preserves_original_reasoning_quota(tmp_path, monkeypatch):
    monkeypatch.setattr('dagbt.transport.time.sleep', lambda _: None)
    with server(fail_once=True) as (endpoint, requests):
        s = session(); configure(s, endpoint, tmp_path)
        payload = {'messages': [{'role': 'user', 'content': 'Fixed protocol request'}], 'max_tokens': 10}
        s.calls.get(('resolve','1'), endpoint+'/v1/chat/completions', payload)
        assert s.ledger.used['llm'] == 2
        assert len(requests) == 2
        resumed = Ledger({'ann':36, 'set_score':128, 'llm':24, 'reader':1})
        calls = Transport('fixture-question', tmp_path, s.config, resumed, None)
        calls.get(('resolve','1'), endpoint+'/v1/chat/completions', payload)
        assert len(requests) == 2
        assert resumed.used['http_attempts'] == 0
        assert resumed.used['llm'] == 2
        assert resumed.used['cache_hits'] == 1
