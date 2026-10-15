from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path

import pytest

from dagbt.methods import JOINT, RESIDUAL_V2 as RESIDUAL
from dagbt.transport import save
from dagbt import webwalker_runner as w

ROOT = Path(__file__).resolve().parents[1]


def args(tmp_path, limit=2):
    return Namespace(config=str(ROOT / 'configs/webwalker-v4.json'),
                     data=str(ROOT / 'data/webwalkerqa'), output=str(tmp_path / 'out'),
                     arms=[JOINT, RESIDUAL], limit=limit, after_personamem=None)


def test_generation_never_loads_references_and_keeps_failures_in_denominator(monkeypatch, tmp_path):
    seen = []
    def no_labels(*a):
        raise AssertionError('Generation opened evaluation references')
    monkeypatch.setattr(w, 'load_references', no_labels)
    def one(output, q, arm, config, timeout):
        assert set(q) == {'id', 'question', 'root_url'}
        seen.append((q['id'], arm))
        row = {'unit_id': q['id'], 'method': arm, 'answer': {'status': 'failed', 'prediction': None}, 'seconds': 1}
        save(w.task_path(output, q['id'], arm) / 'result.json', row)
        return row
    monkeypatch.setattr(w, 'run_one', one)
    a = args(tmp_path)
    result = w.run(a)
    assert result['progress']['total'] == result['progress']['completed'] == 4
    assert result['progress']['failed_tasks'] == 4
    assert seen == [('wwqa-0000', JOINT), ('wwqa-0000', RESIDUAL), ('wwqa-0001', RESIDUAL), ('wwqa-0001', JOINT)]
    summary = json.loads((Path(a.output) / 'generation_summary.json').read_text())
    assert all(s['tasks'] == 2 and s['missing_cost_ledgers'] == 2 for s in summary.values())
    changed = deepcopy(a)
    changed.limit = 3
    with pytest.raises(ValueError, match='manifest differs'):
        w.run(changed)


def test_existing_terminal_result_is_not_retried(monkeypatch, tmp_path):
    q = {'id': 'wwqa-0000', 'question': 'Question', 'root_url': 'https://example.org'}
    row = {'unit_id': q['id'], 'method': RESIDUAL, 'answer': {'status': 'failed', 'prediction': None}}
    save(w.task_path(tmp_path, q['id'], RESIDUAL) / 'result.json', row)
    monkeypatch.setattr(w.subprocess, 'Popen', lambda *a, **k: pytest.fail('Failed task was retried'))
    assert w.run_one(tmp_path, q, RESIDUAL, {}, 1) == row


def test_summary_counts_physical_llm_usage_from_durable_journal_after_timeout(tmp_path):
    row = {'unit_id': 'wwqa-0000', 'method': RESIDUAL,
           'answer': {'status': 'failed', 'prediction': None}}
    directory = w.task_path(tmp_path, row['unit_id'], RESIDUAL)
    directory.mkdir(parents=True)
    journal = [
        {'event': 'http_response', 'kind': 'llm', 'usage': {'prompt_tokens': 100, 'completion_tokens': 12}, 'seconds': 2},
        {'event': 'request_cache_hit', 'kind': 'llm'},
        {'event': 'http_response', 'kind': 'embedding_http', 'usage': {'prompt_tokens': 99}, 'seconds': 1},
        {'event': 'http_response', 'kind': 'llm', 'usage': None, 'seconds': 3},
        {'event': 'web_http'}, {'event': 'web_browser_network'},
    ]
    (directory / 'fusion_events.jsonl').write_text(''.join(json.dumps(event) + '\n' for event in journal))
    (directory / 'call_events.jsonl').write_text(json.dumps({'event': 'call_started', 'cache_hit': True}) + '\n')
    (directory / 'tokenization_events.jsonl').write_text(json.dumps({'http_attempts': 2, 'seconds': 0.5}) + '\n')
    report = w.summarize([row], [RESIDUAL], tmp_path)[RESIDUAL]
    assert report['missing_cost_ledgers'] == 1 and report['execution_ok'] == 0
    assert report['all_task_journal_costs'] == {
        'tokenize_http_attempts': 2, 'tokenize_seconds': 0.5,
        'llm_prompt_tokens': 100, 'llm_completion_tokens': 12, 'llm_response_seconds': 5,
        'llm_successful_responses': 2, 'llm_responses_missing_usage': 1,
        'web_http_attempts': 1, 'browser_network_requests': 1}


def test_scoring_requires_complete_generation_before_answer_loader(monkeypatch, tmp_path):
    a = args(tmp_path)
    save(Path(a.output) / 'progress.json', {'state': 'running'})
    monkeypatch.setattr(w, 'load_references', lambda *a: pytest.fail('References opened before gate'))
    with pytest.raises(RuntimeError, match='terminal'):
        w.score(a)


def test_unanswered_tasks_score_zero_without_model_calls(monkeypatch, tmp_path):
    a = args(tmp_path, 1)
    def one(output, q, arm, config, timeout):
        row = {'unit_id': q['id'], 'method': arm, 'answer': {'status': 'failed', 'prediction': None}}
        save(w.task_path(output, q['id'], arm) / 'result.json', row)
        return row
    monkeypatch.setattr(w, 'run_one', one)
    w.run(a)
    monkeypatch.setattr(w, 'Transport', lambda *a: pytest.fail('Unanswered task used judge'))
    report = w.score(a)
    assert all(s['questions'] == 1 and s['accuracy'] == 0 for s in report['arms'].values())


def test_duplicate_arms_rejected(tmp_path):
    a = args(tmp_path)
    a.arms = [RESIDUAL, RESIDUAL]
    with pytest.raises(ValueError, match='unique'):
        w.run(a)


def test_question_deadline_reaps_only_its_worker_and_publishes_failure(monkeypatch, tmp_path):
    import subprocess
    q = {'id': 'wwqa-0000', 'question': 'Question', 'root_url': 'https://example.org'}
    class Child:
        pid = 123456
        stopped = False
        def poll(self):
            return 0 if self.stopped else None
        def wait(self, timeout=None):
            if self.stopped:
                return 0
            raise subprocess.TimeoutExpired('fixture-worker', timeout)
    child = Child()
    def terminate(pid, sig):
        assert pid == child.pid
        child.stopped = True
    monkeypatch.setattr(w.subprocess, 'Popen', lambda *a, **k: child)
    monkeypatch.setattr(w.os, 'killpg', terminate)
    result = w.run_one(tmp_path, q, RESIDUAL, {}, 0.01)
    assert result['answer']['status'] == 'failed'
    assert result['failure']['error_type'] == 'QuestionTimeout'
    assert child.stopped
    assert (w.task_path(tmp_path, q['id'], RESIDUAL) / 'result.json').exists()
