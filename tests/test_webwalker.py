"""Web traversal and actual residual control; no oracle URLs given to generation."""
from copy import deepcopy
import fcntl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
from threading import Thread
from types import SimpleNamespace

import numpy as np
import pytest

from dagbt.budget import BudgetExceeded
from dagbt.methods import JOINT, RESIDUAL_V2 as RESIDUAL
from dagbt.webwalker_browser import PageParser, PageStore, WebBridge, normalize_url
from dagbt.webwalker_data import load_questions, prepare
from dagbt.webwalker_engine import WebEngine
from dagbt.webwalker_runner import check_personamem_finished

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import demo_residual_memory as demo


@pytest.fixture
def website():
    class Handler(BaseHTTPRequestHandler):
        hits = []
        def do_GET(self):
            self.hits.append(self.path)
            pages = {'/': '<title>Package project</title><p>Primary package: Star E3.</p><a href="/offline">Offline report</a><a href="/backup">Backup package</a>',
                     '/offline': '<title>Offline report</title><p>Star E3 can run offline: true.</p>',
                     '/backup': '<title>Backup</title><p>Backup package: Moon B2.</p>',
                     '/bad': None, '/empty': '<script>var a=1</script>',
                     '/probe': '<title>Probe root</title><a href="/bad">First clue</a><a href="/offline">Second clue</a>',
                     '/js': '<title>JS page</title><script>document.addEventListener("DOMContentLoaded",()=>{document.body.innerHTML="<p>Generated fact: Star E3.</p><a href=\\"/offline\\">Read details</a>";});</script>',
                     '/challenge': '<title>One moment</title>Please wait while your request is being verified...'}
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
    yield 'http://127.0.0.1:' + str(server.server_port), Handler.hits
    server.shutdown()
    server.server_close()
    thread.join()


def plan():
    return {'steps': [demo.step('primary', 'Find primary package'),
                      demo.step('offline', 'Can {primary} run offline?', ['primary']),
                      demo.step('backup', 'Find backup package after checking {offline}', ['offline']),
                      demo.step('selection', 'Select primary if offline else backup', ['primary', 'offline', 'backup'], 'compose')],
            'final_node_id': 'selection', 'output_contract': demo.contract('value', 'entity', 'package'),
            'program': {'variables': {'p': {'type': 'entity', 'demand': 'primary'},
                                      's': {'type': 'bool', 'demand': 'offline'},
                                      'b': {'type': 'entity', 'demand': 'backup'}},
                        'expression': demo.op('if', demo.var('s'), demo.var('p'), demo.var('b'))}}


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
                marker, answer = {'p': ('Primary package: Star E3', 'Star E3'),
                                  's': ('can run offline: true', True),
                                  'b': ('Backup package: Moon B2', 'Moon B2')}[variable]
                sources = [s['id'] for s in data['source_spans'] if marker in s['text']]
                rows = []
                if sources and not operation.startswith('audit_joint'):
                    scope = data.get('requested_scopes', {}).get(variable, data['required_scope'])
                    rows = [{'id': variable + '-source', 'stance': 'support', 'value': answer,
                             'source_ids': sources, **scope, 'parent_versions': {
                                 p: v['version'] for p, v in data['supported_parents'].items()
                                 if v['demand'] in next(s['inputs'] for s in plan()['steps']
                                                       if s['output_slot'] == data['variables'][variable]['demand'])},
                             'retract_ids': [], 'reason': 'Explicit page statement'}]
                slots.append({'variable': variable, 'observations': rows,
                              'unresolved': []})
            value = {'slots': slots}
            if operation.startswith('audit_joint'):
                value['audit_verdict'] = 'supported'
        return {'response': {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(value)}}]},
                'response_ref': 'fixture-' + str(len(self.requests))}


def execute(root, directory, method):
    model = Model()
    config = {'_test_transport': True, 'llm_base_url': 'http://fixture/v1', 'llm_model': 'fixture',
              'webwalker': {'max_actions': 6, 'http_attempts': 1},
              'fusion': {'initial_width': 1, 'proposal_width': 1, 'max_feedback_rounds': 0}}
    e = WebEngine({'id': 'fixture', 'question': 'Use primary if offline, otherwise backup.'},
                  ({}, [], np.empty((0, 0), dtype=np.float32), None, demo.Tokenizer()), model, config, method,
                  root_url=root, page_cache=directory, allow_local=True)
    row = e.run()
    return row, model, e


def test_real_http_full_dag_then_local_navigation_then_symbolic_elimination(website, tmp_path):
    root, hits = website
    row, model, engine = execute(root, tmp_path, RESIDUAL)
    assert row['answer']['status'] == 'ok'
    assert row['answer']['prediction'] == 'Star E3'
    assert row['answer']['completion_kind'] == 'certified_symbolic'
    assert hits == ['/', '/offline']
    assert model.requests[0][0] == 'planner'
    assert len(engine.steps) == 4
    assert row['diagnostics']['web']['actions'] == 2
    assert row['diagnostics']['cost']['ann_calls'] == 0
    assert engine.navigation_provenance
    assert not any(op in ('reader', 'select') for op, _ in model.requests)
    for _, payload in model.requests:
        assert 'golden_path' not in str(payload) and 'source_website' not in str(payload)


def test_full_dag_control_visits_backup_with_same_output(website, tmp_path):
    root, hits = website
    row, _, _ = execute(root, tmp_path, JOINT)
    assert row['answer']['status'] == 'ok'
    assert row['answer']['prediction'] == 'Star E3'
    assert '/backup' in hits
    assert row['diagnostics']['web']['actions'] == 3


def test_page_cache_frozen_raw_content_and_hash_checked(website, tmp_path):
    root, hits = website
    store = PageStore(tmp_path, {'http_attempts': 1}, allow_local=True)
    first = store.get(root)
    assert store.get(root) == first and hits == ['/']
    path = next(tmp_path.glob('*.json'))
    value = json.loads(path.read_text())
    value['document']['text'] = 'Changed page'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='hash'):
        store.get(root)


def test_html_links_normalized_and_scripts_excluded():
    parser = PageParser('https://example.org/dir/')
    parser.feed('<script>bad</script><h1>Facts</h1><a href="../report#part">Read report</a><a href="javascript:x">bad</a>')
    result = parser.result()
    assert result['links'] == [{'url': 'https://example.org/report', 'text': 'Read report'}]
    assert 'bad' not in result['text'].split('Facts')[0]
    with pytest.raises(ValueError):
        normalize_url('file:///etc/passwd')


def test_public_browser_rejects_local_addresses(website, tmp_path):
    root, _ = website
    with pytest.raises(ValueError, match='non-public'):
        PageStore(tmp_path, {}).get(root)


def test_failed_navigation_not_false_fact_or_unbounded_retry(website, tmp_path):
    root, hits = website
    store = PageStore(tmp_path, {'http_attempts': 3}, allow_local=True)
    with pytest.raises(ConnectionError):
        store.get(root + '/bad')
    assert hits == ['/bad']
    assert not list(tmp_path.glob('*.json'))


def test_access_challenge_is_not_registered_as_source(website, tmp_path):
    root, _ = website
    with pytest.raises(ConnectionError):
        PageStore(tmp_path, {'http_attempts': 1}, allow_local=True).get(root + '/challenge')
    assert not list(tmp_path.glob('*.json'))


def test_actual_chromium_renders_js_and_freezes_dom_snapshot(website, tmp_path):
    root = Path(__file__).resolve().parents[1]
    if not (root / '_webwalker_browser_vendor/playwright').is_dir():
        pytest.skip('Run browser installer for Chromium integration')
    url, _ = website
    store = PageStore(tmp_path, {'http_attempts': 1, 'browser_backend': 'auto', 'browser_wait_ms': 50}, allow_local=True)
    page = store.get(url + '/js')
    assert page['capture_kind'] == 'browser_dom'
    assert 'Generated fact: Star E3.' in page['document']['text']
    assert page['document']['links'] == [{'url': url + '/offline', 'text': 'Read details'}]
    assert store.get(url + '/js') == page
    assert any(e['event'] == 'web_browser_network' for e in store.events)


def test_one_failed_link_does_not_close_the_remaining_frontier(website, tmp_path):
    from dagbt.budget import Ledger
    root, hits = website
    class Selector:
        def json(self, operation, system, data, validate, schema):
            return validate({'link_id': data['links'][0]['id']})
    e = SimpleNamespace(docs={}, ledger=Ledger({}), navigation_provenance={}, reasoner=Selector(), event=lambda v: None)
    bridge = WebBridge(e, root + '/probe', PageStore(tmp_path, {'http_attempts': 1}, allow_local=True), {'max_actions': 3})
    assert bridge.step('Find offline report', 'x')['local_candidate_ids']
    failed = bridge.step('Find offline report', 'x')
    assert failed['trace']['search_status'] == 'empty_probe'
    assert not failed['local_candidate_ids']
    assert bridge.step('Find offline report', 'x')['local_candidate_ids']
    assert hits == ['/probe', '/bad', '/offline']
    assert e.ledger.used['web_actions'] == 3


def test_main_split_pinned_complete_and_generation_label_free(tmp_path):
    root = Path(__file__).resolve().parents[1]
    manifest = prepare(root / 'data/webwalkerqa/main.official.jsonl', tmp_path)
    questions, actual = load_questions(tmp_path)
    assert actual == manifest and len(questions) == 680
    assert all(set(q) == {'id', 'question', 'root_url'} for q in questions)
    assert 'golden_path' not in (tmp_path / 'questions.jsonl').read_text()
    with (tmp_path / 'questions.jsonl').open('a') as f:
        f.write('{}\n')
    with pytest.raises(ValueError, match='changed'):
        load_questions(tmp_path)


def test_personamem_gate_checks_live_os_lock_and_terminal_state(tmp_path):
    (tmp_path / 'progress.json').write_text('{"state":"complete_with_failures"}')
    assert check_personamem_finished(tmp_path)['state'] == 'complete_with_failures'
    with (tmp_path / 'writer.lock').open('a+') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='active writer'):
            check_personamem_finished(tmp_path)
    (tmp_path / 'progress.json').write_text('{"state":"generating"}')
    with pytest.raises(RuntimeError, match='not complete'):
        check_personamem_finished(tmp_path)
