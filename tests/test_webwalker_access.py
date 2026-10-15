"""Root access failures are paired exclusions; method failures remain scored."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
import json
import urllib.error

import pytest

from dagbt import webwalker_runner as w
from dagbt.methods import JOINT, RESIDUAL_V2 as RESIDUAL
from dagbt.transport import save
from dagbt.webwalker_browser import PageAccessError, PageStore
from test_webwalker import execute
from test_webwalker_runner import args


def test_cleanly_stopped_personamem_gate_still_requires_released_os_lock(tmp_path):
    import fcntl
    save(tmp_path / 'progress.json', {'state': 'stopped'})
    assert w.check_personamem_finished(tmp_path)['state'] == 'stopped'
    with (tmp_path / 'writer.lock').open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='active writer'):
            w.check_personamem_finished(tmp_path)


@pytest.mark.parametrize('state', ['generating', 'failed', 'interrupted'])
def test_unlocked_but_unfinished_personamem_is_not_a_clean_stop(tmp_path, state):
    save(tmp_path / 'progress.json', {'state': state})
    with pytest.raises(RuntimeError, match='not complete'):
        w.check_personamem_finished(tmp_path)


@pytest.fixture
def access_site():
    class Handler(BaseHTTPRequestHandler):
        hits = []
        def do_GET(self):
            self.hits.append(self.path)
            if self.path in ('/403', '/429', '/503', '/404'):
                self.send_error(int(self.path[1:]))
                return
            if self.path == '/challenge':
                body = '<title>Verify</title>Verify you are human<a href="/help">Help</a>'
            elif self.path == '/empty':
                body = ''
            else:
                body = '<title>Source</title><p>Primary package: Star E3.</p>'
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(body.encode())
        def log_message(self, *a):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield 'http://127.0.0.1:' + str(server.server_port), Handler.hits
    server.shutdown(); server.server_close(); thread.join()


@pytest.mark.parametrize('status,attempts', [(403, 1), (429, 2), (503, 2), (404, 1)])
def test_refusal_has_typed_evidence_and_bounded_retries(access_site, tmp_path, status, attempts):
    url, hits = access_site
    store = PageStore(tmp_path, {'http_attempts': 2}, allow_local=True)
    with pytest.raises(PageAccessError) as exc:
        store.get(url + '/' + str(status))
    assert exc.value.details['http_status'] == status
    assert len(hits) == attempts
    assert not list(tmp_path.glob('*.json'))


@pytest.mark.parametrize('path', ['/403', '/429', '/503'])
def test_http_refusal_falls_back_to_browser_once_and_can_recover(access_site, tmp_path, monkeypatch, path):
    url, hits = access_site
    store = PageStore(tmp_path, {'http_attempts': 1, 'browser_backend': 'auto'}, allow_local=True)
    calls = []
    def browser(target):
        calls.append(target)
        return target, b'<title>Recovered</title><p>Real source content.</p>'
    monkeypatch.setattr(store, 'render_html', browser)
    page = store.get(url + path)
    assert page['capture_kind'] == 'browser_dom'
    assert 'Real source' in page['document']['text']
    assert len(calls) == len(hits) == 1
    assert store.get(url + path) == page
    assert len(calls) == 1


def test_browser_refusal_preserves_final_http_status(access_site, tmp_path, monkeypatch):
    url, _ = access_site
    store = PageStore(tmp_path, {'http_attempts': 1, 'browser_backend': 'auto'}, allow_local=True)
    def browser(target):
        raise PageAccessError(target, 'http_status', 429)
    monkeypatch.setattr(store, 'render_html', browser)
    with pytest.raises(PageAccessError) as exc:
        store.get(url + '/403')
    assert exc.value.details['http_status'] == 429


def test_missing_browser_is_execution_failure_not_environment_exclusion(access_site, tmp_path, monkeypatch):
    url, _ = access_site
    store = PageStore(tmp_path, {'http_attempts': 1, 'browser_backend': 'auto'}, allow_local=True)
    def browser(target):
        raise ValueError('Browser executable missing')
    monkeypatch.setattr(store, 'render_html', browser)
    with pytest.raises(ConnectionError) as exc:
        store.get(url + '/403')
    assert not isinstance(exc.value, PageAccessError)


def test_200_challenge_with_help_link_is_never_registered_as_evidence(access_site, tmp_path):
    url, _ = access_site
    with pytest.raises(PageAccessError) as exc:
        PageStore(tmp_path, {'http_attempts': 1}, allow_local=True).get(url + '/challenge')
    assert exc.value.details['reason'] == 'access_challenge'
    assert not list(tmp_path.glob('*.json'))


def test_dns_error_is_typed_but_private_address_is_not(monkeypatch, tmp_path):
    import socket
    import dagbt.webwalker_browser as b
    def dns(*a, **kw):
        raise socket.gaierror('fixture')
    monkeypatch.setattr(b.socket, 'getaddrinfo', dns)
    with pytest.raises(PageAccessError, match='dns'):
        PageStore(tmp_path, {}).get('https://missing.example/')


def test_corrupt_cache_cannot_be_reclassified_as_environment(access_site, tmp_path):
    url, _ = access_site
    store = PageStore(tmp_path, {'http_attempts': 1}, allow_local=True)
    store.get(url)
    path = next(tmp_path.glob('*.json'))
    value = json.loads(path.read_text()); value['document_hash'] = 'bad'; save(path, value)
    with pytest.raises(ValueError, match='hash'):
        store.get(url)


def test_real_engine_plans_before_denied_root_then_stops_question(access_site, tmp_path):
    url, hits = access_site
    row, model, engine = execute(url + '/403', tmp_path, RESIDUAL)
    assert row['answer']['status'] == 'skipped_environment'
    assert row['environment_exclusion'] == {'scope': 'root_page', 'url': url + '/403',
                                            'reason': 'http_status', 'http_status': 403}
    assert model.requests[0][0] == 'planner'
    assert len(model.requests) == 1
    assert hits == ['/403']
    assert len(engine.steps) == 4
    assert row['diagnostics']['ledger']['used']['llm'] == 1
    assert row['diagnostics']['web']['actions'] == 1


def denied_row(qid, arm):
    return {'unit_id': qid, 'method': arm, 'answer': {'status': 'skipped_environment', 'prediction': None},
            'environment_exclusion': {'scope': 'root_page', 'url': 'https://ehaweb.org/',
                                      'reason': 'http_status', 'http_status': 403}}


def test_denied_question_skips_both_arms_continues_and_scores_common_subset(monkeypatch, tmp_path):
    called = []
    def one(output, q, arm, config, timeout):
        called.append((q['id'], arm))
        row = denied_row(q['id'], arm) if q['id'] == 'wwqa-0000' else {
            'unit_id': q['id'], 'method': arm, 'answer': {'status': 'failed', 'prediction': None}}
        save(w.task_path(output, q['id'], arm) / 'result.json', row)
        return row
    monkeypatch.setattr(w, 'run_one', one)
    a = args(tmp_path, 2)
    result = w.run(a)
    assert result['progress']['state'] == 'complete'
    assert result['progress']['environment_skipped_tasks'] == 2
    assert result['progress']['failed_tasks'] == 2
    assert called == [('wwqa-0000', JOINT), ('wwqa-0001', RESIDUAL), ('wwqa-0001', JOINT)]
    monkeypatch.setattr(w, 'Transport', lambda *a: pytest.fail('No answered question to judge'))
    report = w.score(a)
    for summary in report['arms'].values():
        assert summary['planned_questions'] == 2 and summary['questions'] == 1
        assert summary['environment_excluded'] == 1 and summary['coverage'] == 0.5
        assert summary['accuracy'] == 0 and summary['judge_failures'] == 0
    full = json.loads((Path(a.output) / 'evaluation.json').read_text())
    assert full['rows'][0]['judge_status'] == 'excluded_environment'
    assert (Path(a.output) / 'access_exclusions.json').is_file()
    # A restart reuses both terminal skip records, without reopening any website.
    def reuse(output, q, arm, config, timeout):
        return json.loads((w.task_path(output, q['id'], arm) / 'result.json').read_text())
    monkeypatch.setattr(w, 'run_one', reuse)
    assert w.run(a)['progress']['environment_skipped_tasks'] == 2


def test_all_excluded_has_no_accuracy_and_no_judge_failures(monkeypatch, tmp_path):
    def one(output, q, arm, config, timeout):
        row = denied_row(q['id'], arm)
        save(w.task_path(output, q['id'], arm) / 'result.json', row)
        return row
    monkeypatch.setattr(w, 'run_one', one)
    a = args(tmp_path, 1); w.run(a)
    report = w.score(a)
    assert all(s['accuracy'] is None and s['questions'] == 0 and s['coverage'] == 0
               and s['judge_failures'] == 0 for s in report['arms'].values())


def test_exclusion_on_one_arm_excludes_same_question_on_other_arm(monkeypatch, tmp_path):
    def one(output, q, arm, config, timeout):
        row = denied_row(q['id'], arm) if arm == RESIDUAL else {
            'unit_id': q['id'], 'method': arm, 'answer': {'status': 'ok', 'prediction': 'some prediction'}}
        save(w.task_path(output, q['id'], arm) / 'result.json', row)
        return row
    monkeypatch.setattr(w, 'run_one', one)
    a = args(tmp_path, 1); w.run(a)
    monkeypatch.setattr(w, 'Transport', lambda *a: pytest.fail('Paired exclusion used judge'))
    report = w.score(a)
    assert all(s['environment_excluded'] == 1 and s['questions'] == 0 for s in report['arms'].values())


def test_arbitrary_method_failure_cannot_claim_environment_skip():
    row = denied_row('q', JOINT)
    row['environment_exclusion']['reason'] = 'LLMProtocolError'
    with pytest.raises(ValueError, match='typed root-page'):
        w.environment_exclusions([row])


def test_real_chromium_denied_response_is_typed(access_site, tmp_path):
    root = Path(__file__).resolve().parents[1]
    if not (root / '_webwalker_browser_vendor/playwright').is_dir():
        pytest.skip('Browser installer required')
    url, hits = access_site
    store = PageStore(tmp_path, {'http_attempts': 1, 'browser_backend': 'auto'}, allow_local=True)
    with pytest.raises(PageAccessError) as exc:
        store.get(url + '/403')
    assert exc.value.details['http_status'] == 403
    assert hits.count('/403') == 2  # one HTTP request plus one real browser navigation
    assert not list(tmp_path.glob('*.json'))
