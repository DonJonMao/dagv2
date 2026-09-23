"""Audited, label-free repair of redundant trailing output-slot annotations."""
import copy
import fcntl
import json
import re
import shutil
import time
from pathlib import Path
import experiment as e

ORIGINAL_VALIDATE = e.validate_plan
ROOT = e.ROOT / 'planner_annotation_repair'
SOURCE = e.ROOT / 'full1000'


def normalize(plan):
    cleaned = copy.deepcopy(plan)
    changes = []
    if isinstance(cleaned, dict) and isinstance(cleaned.get('steps'), list):
        for i, step in enumerate(cleaned['steps']):
            if not isinstance(step, dict):
                continue
            slot, question = step.get('output_slot'), step.get('question')
            if not isinstance(slot, str) or not isinstance(question, str):
                continue
            fixed = re.sub(r'\s*\{slot\s*:\s*' + re.escape(slot) + r'\s*\}\s*$', '', question).rstrip()
            if fixed != question:
                step['question'] = fixed
                changes.append(dict(node_index=i, before=question, after=fixed))
    return cleaned, changes


def validate_plan(plan):
    cleaned, _ = normalize(plan)
    return ORIGINAL_VALIDATE(cleaned)


def main():
    ROOT.mkdir(exist_ok=True)
    with (ROOT / 'writer.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        questions, docs, ids, vectors, index, tokenizer = e.prepare()
        questions = {q['id']: q for q in questions}
        e.validate_plan = validate_plan
        audit = ROOT / 'audit.json'
        records = e.native.load(audit) if audit.exists() else {}
        while True:
            for path in (SOURCE / 'rows').glob('*.json'):
                old = e.native.load(path)
                unit = old['unit_id']
                if old['answer']['status'] != 'planner_failed' or unit in records:
                    continue
                # Exact identity of the original question-only planner request.
                payload = dict(messages=[{'role': 'system', 'content': e.PLAN_SYSTEM},
                    {'role': 'user', 'content': questions[unit]['question']}], max_tokens=2048,
                    chat_template_kwargs={'enable_thinking': False},
                    structured_outputs={'json': e.PLAN_SCHEMA}, **e.SAMPLING, model=e.CONFIG['llm_model'])
                key = e.native.digest(dict(unit_id=unit, url=e.CONFIG['llm_base_url'] + '/chat/completions', payload=payload))
                request_path = SOURCE / 'requests' / (key + '.json')
                request = e.native.load(request_path)
                raw = json.loads(request['response']['choices'][0]['message']['content'])
                cleaned, changes = normalize(raw)
                try:
                    ORIGINAL_VALIDATE(cleaned)
                except e.PlanError:
                    records[unit] = dict(status='not_repairable', original_error=old['answer']['error'])
                    e.native.save(audit, records)
                    continue
                if not changes:
                    records[unit] = dict(status='no_matching_annotation')
                    e.native.save(audit, records)
                    continue
                e.native.save(ROOT / 'original_failures' / path.name, old)
                e.native.save(ROOT / 'changes' / path.name, dict(unit_id=unit, changes=changes,
                    original_request=str(request_path), rule='remove only trailing annotation naming current output_slot'))
                dest = ROOT / 'requests' / request_path.name
                dest.parent.mkdir(exist_ok=True)
                if not dest.exists():
                    shutil.copy2(request_path, dest)
                print('repairing', unit, flush=True)
                result = e.work(questions[unit], ROOT, docs, ids, vectors, index, tokenizer)
                records[unit] = dict(status=result['answer']['status'], changes=changes,
                    original_status=old['answer']['status'], finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
                e.native.save(audit, records)
                print('repaired', unit, result['answer']['status'], flush=True)
            progress = e.native.load(SOURCE / 'progress.json')
            e.native.save(ROOT / 'progress.json', dict(state='watching_original_run',
                original_completed=progress['completed'], repairs=records))
            if progress['state'] == 'complete':
                merged = e.ROOT / 'full1000_annotation_fixed'
                (merged / 'rows').mkdir(parents=True, exist_ok=True)
                manifest = e.native.load(SOURCE / 'manifest.json')
                manifest['repair'] = dict(source_run=str(SOURCE), audit=str(audit),
                    code_sha256=e.native.file_hash(Path(__file__)),
                    rule='remove only trailing {slot: CURRENT_OUTPUT_SLOT}; all remaining validation unchanged')
                e.native.save(merged / 'manifest.json', manifest)
                for path in (SOURCE / 'rows').glob('*.json'):
                    replacement = ROOT / 'rows' / path.name
                    shutil.copy2(replacement if replacement.exists() else path, merged / 'rows' / path.name)
                e.evaluate(merged)
                e.native.save(ROOT / 'progress.json', dict(state='complete', merged_output=str(merged), repairs=records))
                return
            time.sleep(30)


if __name__ == '__main__':
    main()
