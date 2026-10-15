"""The portable diagnostics collector reads traces without execution or labels."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_collector_reports_plans_obligations_and_costs_without_touching_files(tmp_path):
    task = tmp_path / 'tasks/q/arm'; task.mkdir(parents=True)
    row = {'unit_id': 'q', 'method': 'arm', 'answer': {'status': 'unresolved_or_exhausted'},
           'diagnostics': {'ledger': {'used': {'llm': 3}, 'events': [{'secret_unwanted_noise': True}]},
             'joint_reading': {'batches': [{'operation': 'joint_read', 'source_aliases': ['s1'],
               'visible_source_aliases': ['s1', 's2'], 'input_tokens': 120}],
               'web_paid_recoveries': [{'policy': 'single_paid_durable_recovery_v1',
                                         'additional_local_repairs_allowed': 0}],
               'evidence_obligations': [{'id': 'o1', 'variable': 'x', 'source_ids': ['s1', 's2'],
                                          'descriptions': ['Documented conflict remains.']}],
               'failed_batch_reviews': [{'id': 'f1', 'variable': 'x', 'source_ids': ['s1'],
                                          'failed': [{'id': 'draft', 'feedback': 'wrong alias'}]}]}}}
    (task / 'result.json').write_text(json.dumps(row))
    wire = {'source_spans': [{'id': 's1', 'text': 'original'}], 'evidence_obligations': []}
    events = [{'event': 'reasoning_response', 'operation': 'planner', 'raw_output': '{bad', 'call_index': 1},
              {'event': 'reasoning_response', 'operation': 'joint_read', 'raw_output': '{"slots":[]}',
               'input_tokens_local': 120, 'messages': [{'role': 'system', 'content': 'Read.'},
                                                        {'role': 'user', 'content': json.dumps(wire)}]},
              {'event': 'joint_repair_input_unavailable', 'input_tokens': 13000, 'context_tokens': 16384},
              {'event': 'web_paid_recovery_local_repair_blocked', 'reason': 'single_paid_response_already_attempted'}]
    (task / 'fusion_events.jsonl').write_text('\n'.join(json.dumps(e) for e in events))
    # These are intentionally not valid JSON and must never be opened.
    (tmp_path / 'credentials.local.json').write_text('PRIVATE_SENTINEL')
    (tmp_path / 'evaluation_only.json').write_text('ANSWER_SENTINEL')
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob('*') if p.is_file()}
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/webwalker_diagnose.py'),
                             '--output', str(tmp_path), '--question', 'q'],
                            cwd=tmp_path, capture_output=True, text=True, check=True)
    item, = json.loads(result.stdout)
    assert item['ledger']['used'] == {'llm': 3} and 'events' not in item['ledger']
    assert item['planner_responses'][0]['raw_output'] == '{bad'
    assert item['evidence_obligations'][0]['source_count'] == 2
    assert item['reading_batches'][0]['visible_source_count'] == 2
    assert item['failed_batch_reviews'][0]['feedback'][0]['feedback'] == 'wrong alias'
    assert item['reading_requests'][0]['source_spans_count'] == 1
    assert item['key_events'][0]['input_tokens'] == 13000
    assert item['key_events'][1]['reason'] == 'single_paid_response_already_attempted'
    assert item['web_paid_recoveries'][0]['additional_local_repairs_allowed'] == 0
    assert 'PRIVATE_SENTINEL' not in result.stdout and 'ANSWER_SENTINEL' not in result.stdout
    assert before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob('*') if p.is_file()}
