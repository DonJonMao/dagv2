#!/usr/bin/env python3
"""Read-only WebWalker task diagnostics; no config/credentials/reference loading."""
import argparse
import hashlib
import json
from pathlib import Path


def short(value, limit=1200):
    if not isinstance(value, str):
        return value
    return value if len(value) <= limit else value[:limit] + ' [diagnostic preview truncated]'


def issue(item):
    return {**{key: short(item[key]) for key in ('id', 'variable', 'task_id', 'phase', 'descriptions') if key in item},
            'source_count': len(item.get('source_ids', [])),
            'feedback': [{'id': entry.get('id'), 'feedback': short(entry.get('feedback'))}
                         for entry in item.get('failed', [])],
            'invalid_response': short(item.get('invalid_response'), 4000)}


def batch(item):
    return {**{key: item[key] for key in ('operation', 'status', 'complete', 'input_tokens', 'initial_output_token_cap') if key in item},
            'batch_source_count': len(item.get('source_aliases', [])),
            'visible_source_count': len(item.get('visible_source_aliases', [])),
            'failed_slots': item.get('failed_slots', [])}


def diagnose(directory):
    directory = Path(directory)
    row = json.loads((directory / 'result.json').read_text())
    diag = row.get('diagnostics', {})
    joint = diag.get('joint_reading', {})
    report = {'question_id': row.get('unit_id'), 'method': row.get('method'),
              'answer_status': row.get('answer', {}).get('status'), 'failure': row.get('failure'),
              'errors': diag.get('errors', []),
              'ledger': {key: diag.get('ledger', {}).get(key) for key in ('limits', 'used', 'remaining')},
              'web_visits': diag.get('web', {}).get('visits', []),
              'reading_batches': [batch(item) for item in joint.get('batches', [])],
              'evidence_obligations': [issue(item) for item in joint.get('evidence_obligations', [])],
              'failed_batch_reviews': [issue(item) for item in joint.get('failed_batch_reviews', [])],
              'web_search_gaps': joint.get('web_search_gaps', []),
              'web_capacity_admissions': joint.get('web_capacity_admissions', []),
              'unread_obligation_count': len(joint.get('unread_obligations', [])),
              'failed_obligation_count': len(joint.get('failed_obligations', [])),
              'planner_responses': [], 'reading_requests': [], 'key_events': []}
    path = directory / 'fusion_events.jsonl'
    keep = {'protocol_error', 'joint_repair_input_unavailable', 'reasoning_repair_exhausted',
            'reasoning_repair_preflight_failed', 'residual_action_incomplete'}
    for line in path.read_text().splitlines() if path.exists() else []:
        event = json.loads(line)
        if event.get('event') in keep:
            report['key_events'].append({key: short(value) for key, value in event.items()
                                         if key not in ('messages', 'raw_output', 'response_metadata', 'usage')})
        if event.get('event') != 'reasoning_response':
            continue
        operation = event.get('operation', '')
        if operation == 'planner':
            raw = event.get('raw_output')
            report['planner_responses'].append({'call_index': event.get('call_index'),
                'raw_output': short(raw, 12000), 'raw_characters': len(raw) if isinstance(raw, str) else None,
                'raw_sha256': hashlib.sha256(raw.encode()).hexdigest() if isinstance(raw, str) else None})
        elif operation.startswith(('joint_read', 'semantic_compose', 'audit_joint')):
            request = {'operation': operation, 'call_index': event.get('call_index'),
                       'input_tokens': event.get('input_tokens_local'),
                       'raw_output': short(event.get('raw_output'), 4000)}
            try:
                data = json.loads(event['messages'][1]['content'])
                for key in ('source_spans', 'known_observations', 'evidence_obligations',
                            'failed_batch_reviews', 'repair_scope'):
                    value = data.get(key)
                    request[key + '_json_chars'] = len(json.dumps(value, ensure_ascii=False)) if value is not None else 0
                    if isinstance(value, list):
                        request[key + '_count'] = len(value)
            except (KeyError, IndexError, TypeError, ValueError):
                request['wire_summary_unavailable'] = True
            report['reading_requests'].append(request)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='outputs/webwalkerqa_v4_access_v2_smoke')
    parser.add_argument('--question', required=True)
    args = parser.parse_args()
    root = Path(args.output) / 'tasks' / args.question
    paths = sorted(root.glob('*/result.json'))
    if not paths:
        parser.error('No terminal results found for this question under ' + str(root))
    print(json.dumps([diagnose(path.parent) for path in paths], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
