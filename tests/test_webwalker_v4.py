"""Web-specific coverage of the v4 core and immutable experimental execution."""
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
from threading import Thread

import numpy as np
import pytest

from dagbt.methods import JOINT, RESIDUAL_V2
from dagbt.transport import save
from dagbt.webwalker_engine import WebEngine
from dagbt import webwalker_runner as w
import test_webwalker as browser_test
from test_webwalker import website
from test_webwalker_runner import args


def decoded_request(payload):
    data = json.loads(payload['messages'][1]['content'])
    if 'source_span_columns' in data:
        columns = data.pop('source_span_columns')
        data['source_spans'] = [dict(zip(columns, row)) for row in data['source_spans']]
    documents = data.pop('source_documents', {})
    for source in data['source_spans']:
        if documents:
            source.update(deepcopy(documents[source['doc_id']]))
    return data


def failed_task(output, q, arm, config, timeout):
    row = {'unit_id': q['id'], 'method': arm, 'answer': {'status': 'failed', 'prediction': None}}
    save(w.task_path(output, q['id'], arm) / 'result.json', row)
    return row


def test_v4_methods_and_executed_core_are_frozen():
    assert w.ARMS == (JOINT, RESIDUAL_V2)
    hashes = w.generation_sources()
    assert {'dagbt/joint_reading.py', 'dagbt/residual_plan.py', 'dagbt/residual_control.py',
            'dagbt/reasoning.py', 'dagbt/review_wire.py', 'dagbt/transport.py',
            'dagbt/webwalker_browser.py', 'scripts/run_webwalker.sh'} <= set(hashes)
    assert w.CORE_RELEASE == 'e6677081cdfb33c01db3aaa274a497168f314f78'


def test_core_upgrade_cannot_resume_existing_output(monkeypatch, tmp_path):
    monkeypatch.setattr(w, 'run_one', failed_task)
    a = args(tmp_path, 1)
    w.run(a)
    sources = w.generation_sources()
    sources['dagbt/joint_reading.py'] = 'changed-core'
    monkeypatch.setattr(w, 'generation_sources', lambda: sources)
    with pytest.raises(ValueError, match='manifest differs'):
        w.run(a)


@pytest.mark.parametrize('change', ['tampered', 'missing'])
def test_result_integrity_checked_before_opening_references(monkeypatch, tmp_path, change):
    monkeypatch.setattr(w, 'run_one', failed_task)
    a = args(tmp_path, 1)
    w.run(a)
    path = w.task_path(a.output, 'wwqa-0000', RESIDUAL_V2) / 'result.json'
    if change == 'tampered':
        row = json.loads(path.read_text()); row['answer']['prediction'] = 'changed'; save(path, row)
    else:
        path.unlink()
    monkeypatch.setattr(w, 'load_references', lambda *a: pytest.fail('Labels opened before integrity check'))
    with pytest.raises(ValueError, match='changed or is missing'):
        w.score(a)
    with pytest.raises(ValueError, match='changed or is missing'):
        w.run(a)


def test_complete_is_published_after_rows_costs_and_integrity_index(monkeypatch, tmp_path):
    monkeypatch.setattr(w, 'run_one', failed_task)
    original = w.save
    def check(path, row):
        if Path(path).name == 'progress.json' and row.get('state') == 'complete':
            assert all((Path(path).parent / name).is_file() for name in
                       ('results.jsonl', 'generation_summary.json', 'generation_records.json'))
        return original(path, row)
    monkeypatch.setattr(w, 'save', check)
    w.run(args(tmp_path, 1))


def test_interruption_preserves_failed_attempt_and_reaps_own_worker(monkeypatch, tmp_path):
    q = {'id': 'wwqa-0000', 'question': 'Question', 'root_url': 'https://example.org'}
    class Child:
        pid = 234567
        stopped = False
        def poll(self): return 0 if self.stopped else None
        def wait(self, timeout=None):
            if not self.stopped: raise KeyboardInterrupt()
            return 0
    child = Child()
    monkeypatch.setattr(w.subprocess, 'Popen', lambda *a, **k: child)
    def terminate(pid, sig):
        assert pid == child.pid
        child.stopped = True
    monkeypatch.setattr(w.os, 'killpg', terminate)
    with pytest.raises(KeyboardInterrupt):
        w.run_one(tmp_path, q, RESIDUAL_V2, {}, 10)
    assert child.stopped
    row = json.loads((w.task_path(tmp_path, q['id'], RESIDUAL_V2) / 'result.json').read_text())
    assert row['answer']['status'] == 'failed'
    monkeypatch.setattr(w.subprocess, 'Popen', lambda *a, **k: pytest.fail('Free interrupted-task retry'))
    assert w.run_one(tmp_path, q, RESIDUAL_V2, {}, 10) == row


def test_incomplete_previous_attempt_does_not_get_new_budget(monkeypatch, tmp_path):
    q = {'id': 'wwqa-0000', 'question': 'Question', 'root_url': 'https://example.org'}
    save(w.task_path(tmp_path, q['id'], RESIDUAL_V2) / 'attempt_started.json', {'started_unix': 1})
    monkeypatch.setattr(w.subprocess, 'Popen', lambda *a, **k: pytest.fail('Incomplete attempt retried'))
    row = w.run_one(tmp_path, q, RESIDUAL_V2, {}, 10)
    assert row['failure']['error_type'] == 'IncompletePreviousAttempt'
    assert row['answer']['status'] == 'failed'


def test_zero_false_predictions_are_answers_and_only_service_hosts_bypass_proxy(monkeypatch):
    assert w.has_prediction(0) and w.has_prediction(False)
    assert not w.has_prediction(None) and not w.has_prediction('  ')
    monkeypatch.setenv('HTTP_PROXY', 'http://proxy.example:8080')
    monkeypatch.setenv('NO_PROXY', 'existing.example')
    monkeypatch.setenv('no_proxy', 'existing.example')
    w.configure_model_network({'llm_base_url': 'http://model.example:8006/v1',
                               'tokenization': {'endpoint': 'http://count.example/tokenize'}})
    import os
    assert os.environ['HTTP_PROXY'] == 'http://proxy.example:8080'
    assert os.environ['NO_PROXY'] == 'existing.example,model.example,count.example'
    assert 'ehaweb.org' not in os.environ['NO_PROXY']


def test_stale_pid_without_os_lock_is_never_signalled(monkeypatch, tmp_path):
    save(tmp_path / 'pid.json', {'pid': 23456})
    monkeypatch.setattr(w.os, 'kill', lambda *a: pytest.fail('Stale owner signalled'))
    assert not w.stop(tmp_path)['stopped']


def test_web_discovery_can_pause_unoffered_windows_without_clearing_real_obligations(monkeypatch, tmp_path):
    p = browser_test.plan()
    p['steps'] = p['steps'][:1]
    p['final_node_id'] = 'primary'
    p['program'] = {'variables': {'p': {'type': 'entity', 'demand': 'primary'}},
                    'expression': {'op': 'var', 'id': 'p'}}
    monkeypatch.setattr(browser_test, 'plan', lambda: deepcopy(p))
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            raw = ('<title>Project</title><p>Primary package: Star E3.</p><p>' +
                   'Decorative navigation context without further package facts. ' * 1500 + '</p>').encode()
            self.send_response(200); self.send_header('Content-Type', 'text/html'); self.end_headers(); self.wfile.write(raw)
        def log_message(self, *args): pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        config = {'_test_transport': True, 'llm_base_url': 'http://fixture/v1', 'llm_model': 'fixture',
                  'webwalker': {'max_actions': 6, 'http_attempts': 1},
                  'fusion': {'max_quote_chars': 120, 'max_repairs_per_request': 0}}
        model = browser_test.Model()
        e = WebEngine({'id': 'frontier', 'question': 'What is the primary package?'},
                      ({}, [], np.empty((0, 0), dtype=np.float32), None, browser_test.demo.Tokenizer()),
                      model, config, RESIDUAL_V2, root_url='http://127.0.0.1:' + str(server.server_port),
                      page_cache=tmp_path, allow_local=True)
        row = e.run(); reader = e.residual_control.reader
        assert row['answer']['status'] == 'ok' and row['answer']['prediction'] == 'Star E3'
        assert len(e.mapper.sources) > 5 and reader.discovery_frontier
        assert not e.fact_state.pending and not reader.unread_obligations
        assert reader.source_exposures
        assert all(item['status'] == 'unoffered' for item in reader.discovery_frontier.values())
        assert e.audit_complete and row['diagnostics']['web']['actions'] == 1
    finally:
        server.shutdown(); server.server_close(); thread.join()


def test_judge_cache_is_bound_to_actual_inputs_and_zero_is_judged(monkeypatch, tmp_path):
    def answered(output, q, arm, config, timeout):
        row = {'unit_id': q['id'], 'method': arm, 'answer': {'status': 'ok', 'prediction': 0}}
        save(w.task_path(output, q['id'], arm) / 'result.json', row)
        return row
    monkeypatch.setattr(w, 'run_one', answered)
    a = args(tmp_path, 1); w.run(a)
    seen = []
    monkeypatch.setattr(w, 'Transport', lambda *a, **k: object())
    class Judge:
        def __init__(self, *a, **k): pass
        def json(self, operation, system, data, validate, schema, **kw):
            seen.append(deepcopy(data)); return validate({'correct': True, 'reason': 'Fixture equivalence'})
    monkeypatch.setattr(w, 'Reasoner', Judge)
    report = w.score(a)
    assert len(seen) == 2 and all(data['prediction'] == 0 for data in seen)
    assert all(r['accuracy'] == 1 for r in report['arms'].values())
    w.score(a); assert len(seen) == 2
    verdict = next(Path(a.output).glob('judge/*/*/*/*/verdict.json'))
    value = json.loads(verdict.read_text()); value['input_identity'] = 'wrong'; save(verdict, value)
    with pytest.raises(ValueError, match='Cached judge verdict identity'):
        w.score(a)


def test_web_semantic_terminal_and_audit_are_actual_model_calls(website, monkeypatch, tmp_path):
    p = browser_test.plan()
    p['steps'] = [p['steps'][0], browser_test.demo.step('answer', 'Respond from {primary}', ['primary'], 'compose')]
    p['final_node_id'] = 'answer'
    p['output_contract'] = browser_test.demo.contract('semantic', 'text', 'package')
    p['program'] = {'variables': {'p': {'type': 'entity', 'demand': 'primary'},
                                'response': {'type': 'text', 'demand': 'answer'}},
                    'expression': {'op': 'var', 'id': 'response'}}
    answer = 'The project uses the Star E3 package.'
    class Services(browser_test.Model):
        def get(self, stage, url, payload):
            data = decoded_request(payload) if stage[0] != 'planner' else None
            if stage[0] == 'planner':
                value = p
            elif stage[0] == 'web_navigate':
                return super().get(stage, url, payload)
            else:
                slots = []
                for variable, declaration in data['variables'].items():
                    scope = data['requested_scopes'][variable]
                    sources = [s['id'] for s in data['source_spans'] if 'Primary package: Star E3' in s['text']]
                    parents = next(s['inputs'] for s in p['steps'] if s['output_slot'] == declaration['demand'])
                    observation = {'id': variable + '-proof', 'stance': 'support',
                                   'value': 'Star E3' if variable == 'p' else answer,
                                   'source_ids': sources, **scope,
                                   'parent_versions': {v: b['version'] for v, b in data['supported_parents'].items()
                                                       if b['demand'] in parents},
                                   'retract_ids': [], 'reason': 'The source names the project package.'}
                    rows = [] if stage[0] == 'audit_joint' else [observation]
                    slots.append({'variable': variable, 'observations': rows, 'unresolved': []})
                value = {'slots': slots}
                if stage[0] == 'semantic_compose':
                    value.update(semantic_answer=answer, final_prediction=answer)
                if stage[0] == 'audit_joint':
                    value['audit_verdict'] = 'supported'
            self.requests.append((stage[0], deepcopy(payload)))
            return {'response': {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(value)}}]}}
    root, hits = website
    services = Services()
    config = {'_test_transport': True, 'llm_base_url': 'http://fixture/v1', 'llm_model': 'fixture',
              'webwalker': {'max_actions': 6, 'http_attempts': 1}}
    e = WebEngine({'id': 'semantic', 'question': 'State which package the project uses.'},
                  ({}, [], np.empty((0, 0), dtype=np.float32), None, browser_test.demo.Tokenizer()),
                  services, config, RESIDUAL_V2, root_url=root, page_cache=tmp_path, allow_local=True)
    row = e.run()
    assert row['answer']['status'] == 'ok'
    assert row['answer']['prediction'] == answer
    assert row['answer']['completion_kind'] == 'semantic_assessment'
    assert not e.personal and e.output_options is None
    assert [op for op, _ in services.requests].count('semantic_compose') == 1
    assert [op for op, _ in services.requests].count('audit_joint') == 1
    assert hits == ['/']


def test_web_attempted_unresolved_evidence_remains_blocking(website, monkeypatch, tmp_path):
    p = browser_test.plan(); p['steps'] = p['steps'][:1]; p['final_node_id'] = 'primary'
    p['program'] = {'variables': {'p': {'type': 'entity', 'demand': 'primary'}},
                    'expression': {'op': 'var', 'id': 'p'}}
    monkeypatch.setattr(browser_test, 'plan', lambda: deepcopy(p))
    class Unresolved(browser_test.Model):
        def get(self, stage, url, payload):
            result = super().get(stage, url, payload)
            if stage[0] == 'joint_read':
                value = json.loads(result['response']['choices'][0]['message']['content'])
                for slot in value['slots']: slot['unresolved'] = ['Unresolved ambiguity in the package identity.']
                result['response']['choices'][0]['message']['content'] = json.dumps(value)
            return result
    root, _ = website
    config = {'_test_transport': True, 'llm_base_url': 'http://fixture/v1', 'llm_model': 'fixture',
              'webwalker': {'max_actions': 6, 'http_attempts': 1},
              'fusion': {'max_repairs_per_request': 0}}
    e = WebEngine({'id': 'unresolved', 'question': 'What package does the project use?'},
                  ({}, [], np.empty((0, 0), dtype=np.float32), None, browser_test.demo.Tokenizer()),
                  Unresolved(), config, RESIDUAL_V2, root_url=root, page_cache=tmp_path, allow_local=True)
    row = e.run()
    assert row['answer']['status'] != 'ok' and row['answer']['prediction'] is None
    assert 'p' in e.fact_state.pending and e.residual_control.reader.evidence_obligations


@pytest.mark.parametrize('stop_signal', ['SIGTERM', 'SIGINT'])
def test_stop_between_os_spawn_and_handle_registration_reaps_real_child(monkeypatch, tmp_path, stop_signal):
    import os, signal, sys
    number = getattr(signal, stop_signal)
    real_popen = subprocess.Popen
    spawned = []
    def spawn_then_stop(*a, **kw):
        child = real_popen([sys.executable, '-c', 'import time;time.sleep(30)'],
                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           start_new_session=True)
        spawned.append(child)
        os.kill(os.getpid(), number)
        return child
    def handler(signum, frame): raise KeyboardInterrupt('synthetic registration stop')
    previous = signal.signal(number, handler)
    monkeypatch.setattr(w.subprocess, 'Popen', spawn_then_stop)
    q = {'id': 'wwqa-0000', 'question': 'Question', 'root_url': 'https://example.org'}
    try:
        with pytest.raises(KeyboardInterrupt, match='synthetic registration stop'):
            w.run_one(tmp_path, q, RESIDUAL_V2, {}, 10)
        assert spawned and spawned[0].poll() is not None
        assert signal.getsignal(number) is handler
        with pytest.raises(ChildProcessError): os.waitpid(spawned[0].pid, os.WNOHANG)
        row = json.loads((w.task_path(tmp_path, q['id'], RESIDUAL_V2) / 'result.json').read_text())
        assert row['answer']['status'] == 'failed'
    finally:
        signal.signal(number, previous)
        for child in spawned:
            if child.poll() is None: child.kill()
            child.wait()


def test_worker_start_failure_is_a_terminal_failed_task(monkeypatch, tmp_path):
    monkeypatch.setattr(w.subprocess, 'Popen', lambda *a, **k: (_ for _ in ()).throw(OSError('spawn failure')))
    q = {'id': 'wwqa-0000', 'question': 'Question', 'root_url': 'https://example.org'}
    row = w.run_one(tmp_path, q, RESIDUAL_V2, {}, 10)
    assert row['answer']['status'] == 'failed' and row['failure']['error_type'] == 'OSError'


def test_live_old_worker_prevents_duplicate_question_attempt(monkeypatch, tmp_path):
    q = {'id': 'wwqa-0000', 'question': 'Question', 'root_url': 'https://example.org'}
    directory = w.task_path(tmp_path, q['id'], RESIDUAL_V2)
    save(directory / 'attempt_started.json', {'started_unix': 1})
    save(directory / 'worker_pid.json', {'pid': 345678})
    from types import SimpleNamespace
    monkeypatch.setattr(w.subprocess, 'run', lambda *a, **k: SimpleNamespace(
        stdout='python -m dagbt.webwalker_runner _worker --input ' + str(directory.resolve() / 'input.json')))
    monkeypatch.setattr(w.subprocess, 'Popen', lambda *a, **k: pytest.fail('Duplicate old worker'))
    with pytest.raises(RuntimeError, match='still live'):
        w.run_one(tmp_path, q, RESIDUAL_V2, {}, 10)
    assert not (directory / 'result.json').exists()


def test_stop_real_web_runner_preserves_interrupted_row_and_leaves_no_worker(tmp_path):
    import os, sys, time
    class ModelHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            if self.path == '/tokenize':
                raw = b'{"count":100}'
                self.send_response(200); self.send_header('Content-Type', 'application/json')
                self.end_headers(); self.wfile.write(raw)
            else:
                time.sleep(20)
        def log_message(self, *args): pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), ModelHandler)
    thread = Thread(target=server.serve_forever, daemon=True); thread.start()
    endpoint = 'http://127.0.0.1:' + str(server.server_port)
    config = w.effective_config(w.ROOT / 'configs/webwalker-v4.json')
    config['llm_base_url'] = endpoint + '/v1'
    config['tokenization']['endpoint'] = endpoint + '/tokenize'
    config.pop('credentials_file', None)
    config['request_timeout_seconds'] = 20; config['max_identical_attempts'] = 1
    config_file = tmp_path / 'config.json'; save(config_file, config)
    output = tmp_path / 'run'
    with (tmp_path / 'process.log').open('w') as log:
        process = subprocess.Popen([sys.executable, '-m', 'dagbt.webwalker_runner', 'run',
                                    '--config', str(config_file), '--data', str(w.ROOT / 'data/webwalkerqa'),
                                    '--output', str(output), '--limit', '1', '--arms', RESIDUAL_V2],
                                   cwd=w.ROOT, env={**os.environ, 'DAG_LLM_API_KEY': 'fixture'},
                                   stdout=log, stderr=log, start_new_session=True)
    try:
        pid_path = w.task_path(output, 'wwqa-0000', RESIDUAL_V2) / 'worker_pid.json'
        deadline = time.monotonic() + 10
        while not pid_path.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(.05)
        assert pid_path.exists(), (tmp_path / 'process.log').read_text()
        worker_pid = json.loads(pid_path.read_text())['pid']
        assert w.stop(output)['stopped']
        assert process.wait(timeout=10) != 0
        assert not w.status(output)['live_writer']
        row_path = pid_path.with_name('result.json')
        assert json.loads(row_path.read_text())['answer']['status'] == 'failed'
        assert json.loads((output / 'progress.json').read_text())['state'] == 'interrupted'
        ps = subprocess.run(['ps', '-p', str(worker_pid), '-o', 'command='], capture_output=True, text=True)
        assert ps.returncode != 0
        assert not w.stop(output)['stopped']
    finally:
        if process.poll() is None: process.terminate()
        process.wait(timeout=10)
        server.shutdown(); server.server_close(); thread.join()


def test_one_website_failure_is_reported_without_vetoing_experiment(monkeypatch, tmp_path):
    a = args(tmp_path, 1); a.offline = False
    monkeypatch.setattr(w, 'Transport', lambda *a, **k: object())
    class Ready:
        def __init__(self, *a, **k): pass
        def json(self, operation, system, data, validate, schema, **kw):
            return validate({'ready': True})
    monkeypatch.setattr(w, 'Reasoner', Ready)
    from dagbt.webwalker_browser import PageStore
    def inaccessible(*a): raise ConnectionError('Synthetic website temporarily unavailable')
    monkeypatch.setattr(PageStore, 'get', inaccessible)
    report = w.preflight(a)
    assert report['model'] == {'ready': True}
    assert report['website']['ready'] is False
    assert report['website']['error_type'] == 'ConnectionError'
    assert report['website_probe_is_generation_gate'] is False
