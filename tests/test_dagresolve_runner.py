"""Loopback HTTP exercises real planning, archive, solve, persistence and costs.

Only model responses are scripted. No external inference endpoint, evaluation
label file, packaged corpus allocation, or production algorithm is replaced.
"""
from contextlib import contextmanager
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from dagresolve import runner as r
from dagresolve.runtime import validate_config
from test_dagresolve_flow import (
    QUESTION, USAGE, completion_value, frozen_runtime, make_corpus, plan,
    query_vector, real_tokenizer,
)


@contextmanager
def loopback_models(*, retry_once=False):
    requests, errors = [], []
    failed = False

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, body, status=200):
            raw = json.dumps(body).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_POST(self):
            nonlocal failed
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append({'path': self.path, 'payload': payload,
                             'authorization': self.headers.get('Authorization')})
            try:
                assert self.headers.get('Authorization') is None, 'No inherited API key may reach fixture'
                if self.path == '/v1/embeddings':
                    assert len(payload['input']) == 1
                    query = payload['input'][0].split('\nQuery: ', 1)[1]
                    self.respond({'data': [{'index': 0, 'embedding': query_vector(query)}], 'usage': USAGE})
                    return
                if self.path == '/v1/completions':
                    if retry_once and not failed:
                        failed = True
                        self.respond({'error': 'Scripted transient unavailability'}, 503)
                        return
                    value = completion_value(payload)
                    self.respond({'choices': [{'text': json.dumps(value), 'finish_reason': 'stop'}], 'usage': USAGE})
                    return
                assert self.path == '/v1/chat/completions'
                messages = payload['messages']
                if messages[0]['content'].startswith('Decompose a multi-hop question'):
                    # A real question-only planner call precedes native archive.
                    assert messages[1] == {'role': 'user', 'content': QUESTION}
                    content = json.dumps(plan())
                else:
                    assert messages[0]['content'].startswith('You are a long-document QA reader')
                    assert 'Lin Zhou' in messages[-1]['content']
                    content = 'Answer: Suzhou'
                self.respond({'choices': [{'message': {'content': content}, 'finish_reason': 'stop'}], 'usage': USAGE})
            except Exception as exc:
                errors.append(repr(exc))
                self.respond({'fixture_error': repr(exc)}, 400)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield 'http://127.0.0.1:' + str(server.server_port), requests, errors
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def local_runtime(monkeypatch, frozen_runtime, real_tokenizer, endpoint):
    monkeypatch.delenv('DAG_LLM_API_KEY', raising=False)
    monkeypatch.delenv('DAG_EMBED_API_KEY', raising=False)
    config = validate_config({'llm_base_url': endpoint + '/v1',
        'embedding_base_url': endpoint + '/v1', 'request_timeout_seconds': 5,
        'max_identical_attempts': 2, 'dataset': 'hotpotqa'})
    monkeypatch.setattr(frozen_runtime.e, 'CONFIG', {**frozen_runtime.e.CONFIG, **config})
    monkeypatch.setattr(frozen_runtime.flow, 'CHAT_URL', endpoint + '/v1/chat/completions')
    monkeypatch.setattr(frozen_runtime.flow, 'COMPLETION_URL', endpoint + '/v1/completions')
    monkeypatch.setattr(frozen_runtime.flow.dense, 'EMBED_URL', endpoint + '/v1/embeddings')
    documents, ids, vectors, index = make_corpus(frozen_runtime)
    return SimpleNamespace(**vars(frozen_runtime),
        config=deepcopy(frozen_runtime.e.CONFIG), requested_config=config,
        questions=[{'id': 'scripted-binding-case', 'question': QUESTION}],
        resources=(documents, ids, vectors, index, real_tokenizer),
        hashes={'scripted_public_corpus': r.digest([ids, [documents[d].passage for d in ids]])})


def protect_labels(monkeypatch):
    """Use fictional labels only after the complete-run guard passes."""
    real_load, real_hash = r.load, r.file_hash
    state = {'evaluating': False, 'loads': [], 'integrity': []}

    def load(path):
        if Path(path).name == 'evaluation_only.json':
            assert state['evaluating'], 'Generation attempted to open evaluation labels'
            state['loads'].append(str(path))
            return [{'id': 'scripted-binding-case', 'answers': ['Suzhou'],
                     'gold_groups': [['lin', 'binding'], ['lin_birth']]}]
        return real_load(path)

    def file_hash(path):
        if Path(path).name == 'evaluation_only.json':
            assert state['evaluating']
            return r.digest('fictional evaluation fixture only')
        return real_hash(path)

    def verify_originals(*, include_labels=False):
        assert state['evaluating'] and include_labels
        state['integrity'].append(include_labels)
        return {'fixture_label_isolation': True}

    monkeypatch.setattr(r, 'load', load)
    monkeypatch.setattr(r, 'file_hash', file_hash)
    monkeypatch.setattr(r, 'verify_originals', verify_originals)
    return state


def test_loopback_real_runner_planner_archive_binding_repair_reader_cache_and_retry(
        tmp_path, monkeypatch, frozen_runtime, real_tokenizer):
    label_state = protect_labels(monkeypatch)
    with loopback_models(retry_once=True) as (endpoint, requests, errors):
        runtime = local_runtime(monkeypatch, frozen_runtime, real_tokenizer, endpoint)
        output = tmp_path / 'run'
        assert r.run(runtime, output, limit=1)['completed'] == 1
        assert errors == []
        row = r.load(r.row_path(output, runtime.questions[0]['id']))
        initial_path = next((output / 'inputs').glob('*.json'))
        initial_bytes = initial_path.read_bytes()
        initial = r.load(initial_path)
        assert initial['plan'] == plan()
        assert len(initial['candidate_doc_ids']) == 23
        assert len(initial['archive_trace']) == 3
        assert initial['archive_trace'][0]['query'] == QUESTION
        assert initial['archive_trace'][2]['query'] == 'In which city was the unknown entity born?'
        assert row['answer']['status'] == 'ok' and row['answer']['prediction'] == 'Suzhou'
        assert row['ranking']['nodes'][0]['answer'] == 'Lin Zhou'
        assert 'binding' not in row['ranking']['trace'][0]['panel_doc_ids']
        assert row['ranking']['nodes'][0]['source_doc_ids'] == ['lin', 'binding']
        assert row['ranking']['trace'][1]['query'] == 'In which city was Lin Zhou born?'
        manifest = r.load(output / 'manifest.json')
        assert manifest['code_hashes'] == r.code_hashes()  # Actual code provenance, not a mock digest.
        assert manifest['protocol']['generation_labels_read'] is False
        assert r.load(output / 'progress.json')['labels_read'] is False
        assert label_state['loads'] == label_state['integrity'] == []
        cost = row['runner_cost']
        assert cost['http_attempts'] == len(requests)
        assert cost['http_retries'] == 1
        assert cost['incomplete_or_failed_http_attempts'] == 1
        assert cost['cache_hits'] >= 1  # Native archive director query is reused by the node.
        assert cost['token_usage_complete'] is False  # Server work for the 503 is unknown.
        records = [r.load(p) for p in r.call_directory(output, row['unit_id']).joinpath('requests').glob('*.json')]
        retried = [record for record in records if len(record['attempts']) == 2]
        assert len(retried) == 1
        assert [attempt['http_status'] for attempt in retried[0]['attempts']] == [503, 200]
        assert cost['total_tokens'] == sum(record['response']['usage']['total_tokens'] for record in records)
        assert sum(request['path'] == '/v1/chat/completions' for request in requests) == 2
        embeddings = [q['payload']['input'][0].split('\nQuery: ', 1)[1]
                      for q in requests if q['path'] == '/v1/embeddings']
        bindings = [q for q in embeddings if q.startswith(('Chen Hai ', 'Lin Zhou '))]
        assert len(bindings) == 2
        assert all(all(term in q for term in ('director', '1998', 'Homeward')) for q in bindings)
        assert embeddings.count('In which city was Lin Zhou born?') == 1
        assert 'In which city was Chen Hai born?' not in embeddings
        assert all(request['authorization'] is None for request in requests)
        assert not any(term in json.dumps(requests) for term in ('gold_groups', 'evaluation_only.json'))
        reader = next(request['payload'] for request in requests if 'messages' in request['payload']
            and request['payload']['messages'][0]['content'].startswith('You are a long-document QA reader'))
        selected = row['ranking']['reader']['panel_doc_ids']
        origin = runtime.v6.reader_input(initial, selected, runtime.resources[0], real_tokenizer, row['ranking']['nodes'])
        assert reader['messages'] == origin['messages']

        # Force actual work to replay its persisted exact calls, rather than
        # relying only on the coordinator's saved-row short circuit.
        physical_before = len(requests)
        replay = r.work(runtime.questions[0], runtime, output)
        assert replay['answer']['prediction'] == 'Suzhou'
        assert len(requests) == physical_before and errors == []
        assert initial_path.read_bytes() == initial_bytes
        replay_cost = replay['runner_cost']
        assert replay_cost['http_attempts'] == cost['http_attempts']
        assert replay_cost['http_retries'] == 1
        assert replay_cost['cache_hits'] > cost['cache_hits']
        assert replay_cost['logical_calls'] > cost['logical_calls']
        assert r.run(runtime, output, limit=1)['completed'] == 1
        assert len(requests) == physical_before
        label_state['evaluating'] = True
        summary = r.evaluate(output, dataset='hotpotqa', config=runtime.requested_config)
        assert summary['n'] == 1 and summary['metrics_percent']['em'] == 100
        assert len(label_state['loads']) == len(label_state['integrity']) == 1
        assert summary['cost']['http_attempts'] == physical_before
        assert errors == []


def test_incomplete_or_invalid_generation_is_rejected_before_labels_are_opened(
        tmp_path, monkeypatch, frozen_runtime, real_tokenizer):
    state = protect_labels(monkeypatch)
    with loopback_models() as (endpoint, requests, errors):
        runtime = local_runtime(monkeypatch, frozen_runtime, real_tokenizer, endpoint)
        output = tmp_path / 'guarded'
        r.run(runtime, output, limit=1)
        progress = r.load(output / 'progress.json')
        r.save(output / 'progress.json', {**progress, 'state': 'paused'})
        with pytest.raises(ValueError, match='labels remain unopened'):
            r.evaluate(output)
        r.save(output / 'progress.json', progress)
        path = r.row_path(output, runtime.questions[0]['id'])
        row = r.load(path)
        r.save(path, {**row, 'answer': {'status': 'ok', 'prediction': 12}})
        with pytest.raises(ValueError, match='Invalid saved result'):
            r.evaluate(output)
        assert state['loads'] == state['integrity'] == []
        assert errors == [] and requests


def test_resume_refuses_changed_scope_or_orphaned_run_before_any_new_http(
        tmp_path, monkeypatch, frozen_runtime, real_tokenizer):
    with loopback_models() as (endpoint, requests, errors):
        runtime = local_runtime(monkeypatch, frozen_runtime, real_tokenizer, endpoint)
        output = tmp_path / 'run'
        r.run(runtime, output, limit=1)
        before = len(requests)
        runtime.requested_config = {**runtime.requested_config, 'request_timeout_seconds': 6}
        with pytest.raises(ValueError, match='changed'):
            r.run(runtime, output, limit=1)
        assert len(requests) == before
        orphan = tmp_path / 'orphan'
        (orphan / 'inputs').mkdir(parents=True)
        with pytest.raises(ValueError, match='without a manifest'):
            r.run(runtime, orphan, limit=1)
        assert len(requests) == before and errors == []


@pytest.mark.parametrize('usage', [
    {}, {'prompt_tokens': 1, 'completion_tokens': -2, 'total_tokens': 4},
    {'prompt_tokens': True, 'completion_tokens': 2, 'total_tokens': 3},
    {'prompt_tokens': 1, 'completion_tokens': 'unreported', 'total_tokens': 4}])
def test_unknown_or_malformed_usage_keeps_physical_work_and_marks_cost_incomplete(tmp_path, usage):
    r.save(tmp_path / 'requests' / 'observed.json', {'stage': ['resolve', 'clarify', 'judge'],
        'url': 'http://127.0.0.1/unused/completions', 'attempts': [{'http_status': 503}, {'http_status': 200}],
        'response': {'usage': usage}})
    cost = r._cost(tmp_path)
    assert cost['http_attempts'] == 2 and cost['http_retries'] == 1
    assert cost['invalid_or_missing_usage_responses'] == 1
    assert not cost['token_usage_complete']
    assert all(type(cost[key]) is int and cost[key] >= 0
               for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'))
