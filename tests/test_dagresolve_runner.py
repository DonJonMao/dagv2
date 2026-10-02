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


def protect_labels(monkeypatch, labels=None):
    """Use fictional labels only after the complete-run guard passes."""
    real_load, real_hash = r.load, r.file_hash
    state = {'evaluating': False, 'loads': [], 'integrity': []}

    def load(path):
        if Path(path).name == 'evaluation_only.json':
            assert state['evaluating'], 'Generation attempted to open evaluation labels'
            state['loads'].append(str(path))
            return labels if labels is not None else [
                {'id': 'scripted-binding-case', 'answers': ['Suzhou'],
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


def evaluate_saved_fixture(tmp_path, monkeypatch, frozen_runtime, rows, labels, dataset='hotpotqa'):
    """Score completed synthetic results without model calls or real labels."""
    state = protect_labels(monkeypatch, labels)
    state['evaluating'] = True
    # Use native answer metrics already imported by frozen_runtime. Dataset
    # loading is unnecessary for these self-contained, fictional label groups.
    monkeypatch.setattr(r, 'import_originals', lambda _: frozen_runtime)
    r.save(tmp_path / 'manifest.json', {'method': r.METHOD, 'dataset': dataset,
        'unit_ids': [row['unit_id'] for row in rows], 'code_hashes': r.code_hashes(),
        'scope': 'explicit_subset'})
    r.save(tmp_path / 'progress.json', {'state': 'complete', 'completed': len(rows)})
    for row in rows:
        r.save(r.row_path(tmp_path, row['unit_id']), row)
    summary = r.evaluate(tmp_path)
    assert len(state['loads']) == len(state['integrity']) == 1
    return summary, r.load(tmp_path / 'scores.json')


def metric_row(unit, selected, *, panel=None, status='ok', removed=None):
    row = {'unit_id': unit, 'answer': {'status': status, 'prediction': 'Suzhou'},
           'budgets': {str(k): {'selected_doc_ids': list(selected)} for k in (5, 10, 20)},
           'ranking': {}}
    if panel is not None:
        row['ranking']['reader'] = {'panel_doc_ids': panel}
        if removed is not None:
            row['ranking']['reader']['context_removed_doc_ids'] = removed
    return row


def test_selector_recall_does_not_claim_reader_saw_trimmed_support(
        tmp_path, monkeypatch, frozen_runtime):
    row = metric_row('trimmed', ['lin', 'lin_birth'], panel=['lin'], removed=['lin_birth'])
    labels = [{'id': 'trimmed', 'answers': ['Suzhou'],
               'gold_groups': [['lin', 'binding'], ['lin_birth']]}]
    summary, scores = evaluate_saved_fixture(tmp_path, monkeypatch, frozen_runtime, [row], labels)
    values = summary['metrics_percent']
    assert values['r@20'] == values['all@20'] == 100
    assert values['reader_support_recall'] == 50
    assert values['reader_all_support'] == 0
    assert values['reader_context_trim_rate'] == 100
    assert scores[0]['reader_evidence_available']
    assert summary['reader_evidence_scope']['context_trimmed_rows'] == 1
    assert 'selector output before Reader capacity trimming' in summary['support_metric']


def test_reader_metrics_keep_answer_failures_separate_and_missing_panels_in_denominator(
        tmp_path, monkeypatch, frozen_runtime):
    rows = [metric_row('ok', ['lin'], panel=['lin']),
            metric_row('failed', ['lin', 'filler'], panel=['lin'], status='reader_failed', removed=['filler']),
            metric_row('missing', ['lin']),
            metric_row('malformed', ['lin'], panel=[{'lin': True}])]
    labels = [{'id': row['unit_id'], 'answers': ['Suzhou'], 'gold_groups': [['lin']]}
              for row in rows]
    summary, scores = evaluate_saved_fixture(tmp_path, monkeypatch, frozen_runtime, rows, labels)
    assert summary['n'] == 4 and summary['invalid_answers'] == 1
    assert summary['metrics_percent']['r@20'] == 100
    assert summary['metrics_percent']['reader_support_recall'] == 50
    assert summary['metrics_percent']['reader_all_support'] == 50
    assert summary['metrics_percent']['reader_context_trim_rate'] == 25
    assert summary['reader_evidence_scope']['denominator'] == 4
    assert summary['reader_evidence_scope']['panel_observed_rows'] == 2
    assert summary['reader_evidence_scope']['unavailable_rows'] == 2
    assert summary['reader_evidence_scope']['observed_panel_trim_rate_denominator'] == 2
    assert summary['reader_evidence_scope']['observed_panel_trim_rate_percent'] == 50
    assert scores[1]['reader_support_recall'] == scores[1]['reader_all_support'] == 1
    assert not scores[1]['valid']
    assert all(item['reader_support_recall'] == item['reader_all_support'] == 0 for item in scores[2:])


def test_no_reader_panels_scores_zero_without_inventing_a_conditional_trim_rate(
        tmp_path, monkeypatch, frozen_runtime):
    rows = [metric_row('missing', ['lin'], status='planner_failed')]
    labels = [{'id': 'missing', 'answers': ['Suzhou'], 'gold_groups': [['lin']]}]
    summary, _ = evaluate_saved_fixture(tmp_path, monkeypatch, frozen_runtime, rows, labels)
    assert summary['metrics_percent']['reader_support_recall'] == 0
    assert summary['metrics_percent']['reader_all_support'] == 0
    assert summary['metrics_percent']['reader_context_trim_rate'] == 0
    assert summary['reader_evidence_scope']['unavailable_rows'] == 1
    assert summary['reader_evidence_scope']['observed_panel_trim_rate_denominator'] == 0
    assert summary['reader_evidence_scope']['observed_panel_trim_rate_percent'] is None


def test_reader_support_groups_and_trim_comparison_use_musique_normalized_ids(
        tmp_path, monkeypatch, frozen_runtime):
    rows = [metric_row('mixed', ['one', 'musique:two'], panel=['musique:one', 'two']),
            metric_row('trimmed', ['one', 'musique:two'], panel=['musique:one'])]
    labels = [{'id': row['unit_id'], 'answers': ['Suzhou'],
               'gold_groups': [['musique:one', 'musique:alternate'], ['musique:two']]}
              for row in rows]
    summary, scores = evaluate_saved_fixture(
        tmp_path, monkeypatch, frozen_runtime, rows, labels, dataset='musique')
    assert summary['metrics_percent']['r@20'] == summary['metrics_percent']['all@20'] == 100
    assert summary['metrics_percent']['reader_support_recall'] == 75
    assert summary['metrics_percent']['reader_all_support'] == 50
    assert summary['metrics_percent']['reader_context_trim_rate'] == 50
    assert scores[0]['reader_context_trim_rate'] == 0
    assert scores[1]['reader_support_recall'] == .5
