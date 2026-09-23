"""Current-environment MuSiQue extension; frozen source modules are imported unchanged."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import concurrent.futures
import fcntl
import json
import re
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent            # dagv2/
PACKAGE_ROOT = ROOT.parent                        # package root (this delivery)
PACKAGE = PACKAGE_ROOT / 'package'                # frozen DAG v1 core
DATA = PACKAGE_ROOT / 'data'
SOURCE = DATA  # legacy alias: per-dataset EvLink++-derived assets live under data/<dataset>/
TOKENIZER = PACKAGE / 'tokenizer'
sys.path.insert(0, str(PACKAGE))
import run as native
import core
import frozen_flow as flow
from reader import Document

CONFIG = dict(method='dag_v2_nothink_chain',
    llm_base_url='http://127.0.0.1:8020/v1', embedding_base_url='http://127.0.0.1:8019/v1',
    llm_model='qwen3.8-27b', embedding_model='nvidia/NV-Embed-v2',
    request_timeout_seconds=600, max_identical_attempts=3, tokenizer=str(TOKENIZER),
    references=str(ROOT / 'references.jsonl'))
_CONFIG_OVERRIDE = os.environ.get('DAGV2_CONFIG')
if _CONFIG_OVERRIDE:
    # Optional JSON override (see config.example.json); keys mirror CONFIG.
    CONFIG.update(json.loads(Path(_CONFIG_OVERRIDE).read_text()))
SAMPLING = dict(temperature=0, top_p=1, top_k=-1, presence_penalty=0,
                repetition_penalty=1, seed=20260918)
PLAN_SYSTEM = '''Decompose a multi-hop question into an executable dependency DAG of 1 to 6 retrieval tasks.
Use ONLY the question. Do not answer it, guess intermediate entities, or use outside knowledge.
Each step has question, output_slot, answer_type, inputs. Output slots are unique simple identifiers.
List steps in dependency order. inputs names only earlier output slots; refer to them in question as {slot}.
Independent branches have empty inputs. Include a final step that resolves the requested relation or comparison.
Keep entity names and constraints from the question. Do not include explanations or extra fields.'''
PLAN_SCHEMA = {'type': 'object', 'properties': {'steps': {'type': 'array', 'minItems': 1,
    'maxItems': 6, 'items': {'type': 'object', 'properties': {
        'question': {'type': 'string'}, 'output_slot': {'type': 'string'},
        'answer_type': {'type': 'string'},
        'inputs': {'type': 'array', 'items': {'type': 'string'}}},
        'required': ['question', 'output_slot', 'answer_type', 'inputs'], 'additionalProperties': False}}},
    'required': ['steps'], 'additionalProperties': False}
EMBED_LOCK = threading.Lock()


class Calls(native.Calls):
    def get(self, stage, url, payload):
        if not url.endswith('/embeddings'):
            payload = {**payload, **SAMPLING}
        return super().get(stage, url, payload)


class PlanError(ValueError):
    pass


def validate_plan(plan):
    if not isinstance(plan, dict) or set(plan) != {'steps'}:
        raise PlanError('plan_object')
    steps = plan['steps']
    if not isinstance(steps, list) or not 1 <= len(steps) <= 6:
        raise PlanError('plan_size')
    seen = set()
    ordered_slots = []
    for s in steps:
        if not isinstance(s, dict) or set(s) != {'question', 'output_slot', 'answer_type', 'inputs'}:
            raise PlanError('step_fields')
        slot = s['output_slot']
        if not isinstance(slot, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', slot) or slot in seen:
            raise PlanError('output_slot')
        if not all(isinstance(s[k], str) and s[k].strip() for k in ('question', 'answer_type')):
            raise PlanError('step_text')
        deps = s['inputs']
        if not isinstance(deps, list) or any(not isinstance(d, str) for d in deps):
            raise PlanError('input_type')
        if len(set(deps)) != len(deps) or not set(deps) <= seen:
            raise PlanError('input_dependency')
        referenced = set(re.findall(r'\{([^{}]+)\}', s['question']))
        if not referenced <= seen:
            raise PlanError('unknown_or_forward_placeholder')
        # Preserve declared dependencies; add referenced predecessors deterministically.
        # Native ground() explicitly supports inputs absent from the question text.
        s['inputs'] = [d for d in ordered_slots if d in set(deps) | referenced]
        seen.add(slot)
        ordered_slots.append(slot)
    core.ordered_steps(plan)
    return plan


def prepare():
    manifest = native.load(SOURCE / 'inputs_manifest.json')
    for name in ('questions.json', 'corpus.json'):
        if native.file_hash(SOURCE / name) != manifest['input_hashes'][name]:
            raise ValueError('Source checksum changed: ' + name)
    questions = native.load(SOURCE / 'questions.json')
    canonical = native.rows(DATA / 'musique' / 'questions.jsonl')
    if questions != canonical or len(questions) != 1000:
        raise ValueError('Must use exactly the current 1000 questions in order')
    corpus = native.load(SOURCE / 'corpus.json')
    ids = [d['id'] for d in corpus]
    index_manifest = native.load(SOURCE / 'index/manifest.json')
    if ids != index_manifest['document_ids'] or index_manifest['embedding_model'] != CONFIG['embedding_model']:
        raise ValueError('Vector provenance mismatch')
    path = ROOT / 'document_vectors.npy'
    if not path.exists():
        with np.load(SOURCE / 'index/vectors.npz', allow_pickle=False) as z:
            vectors = z['document_vectors'].astype(np.float32)
        if vectors.shape != (11656, 4096) or not np.isfinite(vectors).all():
            raise ValueError('Invalid source vectors')
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError('Zero source vectors')
        vectors /= norms
        tmp = ROOT / 'document_vectors.tmp.npy'
        np.save(tmp, vectors)
        tmp.replace(path)
    vectors = np.load(path, mmap_mode='r', allow_pickle=False)
    if vectors.shape != (11656, 4096) or vectors.dtype != np.float32 or not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=2e-5):
        raise ValueError('Invalid normalized vectors')
    docs = {d['id']: Document(d['id'], d['title'], d['text']) for d in corpus}
    index = SimpleNamespace(vectors=dict(zip(ids, vectors)), lock=EMBED_LOCK)
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    native.configure(CONFIG)
    return questions, docs, ids, vectors, index, tokenizer


def make_plan(question, calls):
    response = calls.get('planner', CONFIG['llm_base_url'] + '/chat/completions', {
        'messages': [{'role': 'system', 'content': PLAN_SYSTEM}, {'role': 'user', 'content': question}],
        'max_tokens': 2048, 'chat_template_kwargs': {'enable_thinking': False},
        'structured_outputs': {'json': PLAN_SCHEMA}})['response']
    c = response['choices'][0]
    if c['finish_reason'] != 'stop':
        raise PlanError('planner_finish=' + c['finish_reason'])
    try:
        return validate_plan(json.loads(c['message']['content']))
    except (json.JSONDecodeError, TypeError) as exc:
        raise PlanError('planner_json') from exc


def archive(question, plan, ids, vectors, calls):
    # Fixed, label-free extension: original question first, then DAG steps in order.
    queries = list(dict.fromkeys([question] + [re.sub(r'\{[^{}]+\}', 'the unknown entity', s['question']) for s in plan['steps']]))
    pool, seen, trace = [], set(), []
    for i, query in enumerate(queries):
        with EMBED_LOCK:
            response = calls.get(('archive', str(i)), CONFIG['embedding_base_url'] + '/embeddings',
                {'input': ['Instruct: ' + flow.INSTRUCTION + '\nQuery: ' + query]})['response']
        v = np.asarray(response['data'][0]['embedding'], dtype=np.float32)
        if v.shape != (4096,) or not np.isfinite(v).all() or np.linalg.norm(v) == 0:
            raise native.ServicePause('Invalid query embedding')
        scores = vectors @ (v / np.linalg.norm(v))
        order = np.argsort(-scores, kind='stable')[:50]
        hits = [ids[j] for j in order]
        trace.append(dict(query=query, doc_ids=hits, scores=[float(scores[j]) for j in order]))
        for d in hits:
            if d not in seen:
                seen.add(d)
                pool.append(d)
    return pool, trace


def failure(unit, status, error):
    return dict(unit_id=unit, ranking=dict(status=status, error=str(error), nodes=[], trace=[]),
        budgets={str(k): dict(selected_doc_ids=[], retained_node_proofs=0, available_node_proofs=0) for k in (5, 10, 20)},
        answer=dict(status=status, prediction='', error=str(error)))


def work(q, output, docs, ids, vectors, index, tokenizer):
    path = output / 'rows' / (native.digest(q['id']) + '.json')
    if path.exists():
        return native.load(path)
    started = time.time()
    calls = Calls(q['id'], output, CONFIG)
    input_path = output / 'inputs' / (native.digest(q['id']) + '.json')
    try:
        if input_path.exists():
            row = native.load(input_path)
        else:
            plan = make_plan(q['question'], calls)
            pool, trace = archive(q['question'], plan, ids, vectors, calls)
            row = dict(unit_id=q['id'], question=q['question'], plan=plan, candidate_doc_ids=pool, archive_trace=trace)
            native.save(input_path, row)
        result = native.solve(row, docs, tokenizer, index, calls)
    except PlanError as exc:
        result = failure(q['id'], 'planner_failed', exc)
    except AssertionError as exc:
        if not exc.args or not isinstance(exc.args[0], tuple) or exc.args[0][0] != 'context_budget':
            raise
        result = failure(q['id'], 'context_overflow', exc)
    result['seconds'] = time.time() - started
    native.save(path, result)
    return result


def evaluate(output):
    manifest = native.load(output / 'manifest.json')
    paths = list((output / 'rows').glob('*.json'))
    if {native.load(p)['unit_id'] for p in paths} != set(manifest['unit_ids']):
        raise ValueError('Generation incomplete: labels remain unread')
    # This is the first access to evaluation labels, after all generation is saved.
    labels = native.load(SOURCE / 'evaluation_labels.json')
    expected = native.load(SOURCE / 'inputs_manifest.json')['input_hashes']['evaluation_labels.json']
    if native.file_hash(SOURCE / 'evaluation_labels.json') != expected:
        raise ValueError('Evaluation labels checksum mismatch')
    dest = Path(CONFIG['references'])
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(''.join(json.dumps(dict(unit_id=x['id'], answers=x['answers'], gold_doc_ids=x['gold_doc_ids'])) + '\n' for x in labels))
    native.evaluate(output)
    summary = native.load(output / 'summary.json')
    from collections import Counter
    summary['answer_status_counts'] = dict(Counter(native.load(p)['answer']['status'] for p in paths))
    summary['condition'] = CONFIG['method']
    summary['dataset'] = 'current local MuSiQue 1000-question subset, not official full dev'
    native.save(output / 'summary.json', summary)
    lines = ['# DAG v1 / MuSiQue / Qwen3.8 + NV-Embed-v2', '',
        f"Questions: {summary['n']}; invalid answers: {summary['invalid_answers']}", '',
        '| Metric | Percent |', '|---|---:|']
    lines += [f'| {k} | {v:.2f} |' for k, v in summary['metrics_percent'].items()]
    lines += ['', 'Current-environment extension: new question-only DAG planner and dense candidate archives; original core solver and Reader unchanged. See PROTOCOL.md.']
    (output / 'RESULTS.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError('workers must be 1..4')
    output = ROOT / ('smoke3_v2' if args.smoke else 'full1000')
    output.mkdir(exist_ok=True)
    with (output / 'writer.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        questions, docs, ids, vectors, index, tokenizer = prepare()
        if args.smoke:
            questions = [next(q for q in questions if q['id'].startswith(str(h) + 'hop')) for h in (2, 3, 4)]
        manifest = dict(method=CONFIG['method'], config=CONFIG, unit_ids=[q['id'] for q in questions],
            scope='explicit_subset' if args.smoke else 'full', sampling=SAMPLING,
            planner_system=PLAN_SYSTEM, planner_schema=PLAN_SCHEMA,
            dependency_normalization='stable union of declared inputs and referenced predecessor placeholders; reject unknown/forward refs',
            archive_rule='stable union top50 original question then each ungrounded DAG step',
            workers=args.workers, context_limit=16384, node_thinking_tokens=2048, node_final_tokens=512,
            reader_thinking_tokens=2048, reader_final_tokens=128,
            hashes={str(p): native.file_hash(p) for p in [Path(__file__), SOURCE / 'questions.json', SOURCE / 'corpus.json',
                ROOT / 'document_vectors.npy', TOKENIZER / 'tokenizer_config.json', TOKENIZER / 'tokenizer.json',
                *[PACKAGE / f for f in ('core.py', 'frozen_flow.py', 'reader.py', 'run.py', 'metrics.py')]]})
        mpath = output / 'manifest.json'
        if mpath.exists() and native.load(mpath) != manifest:
            raise ValueError('Experiment changed; use a new output directory')
        native.save(mpath, manifest)
        completed = sum((output / 'rows' / (native.digest(q['id']) + '.json')).exists() for q in questions)
        started = time.time()
        def status(state, **extra):
            record = dict(state=state, completed=completed, total=len(questions), elapsed_seconds=time.time()-started,
                updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), **extra)
            native.save(output / 'progress.json', record)
            print(json.dumps(record), flush=True)
        status('running')
        pending = [q for q in questions if not (output / 'rows' / (native.digest(q['id']) + '.json')).exists()]
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)
        futures = {executor.submit(work, q, output, docs, ids, vectors, index, tokenizer): q['id'] for q in pending}
        try:
            for f in concurrent.futures.as_completed(futures):
                result = f.result()
                completed += 1
                status('running', last_id=result['unit_id'], last_status=result['answer']['status'])
        except Exception as exc:
            for f in futures:
                f.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            completed = sum((output / 'rows' / (native.digest(q['id']) + '.json')).exists() for q in questions)
            status('paused', error_type=type(exc).__name__, error=str(exc))
            raise
        else:
            executor.shutdown(wait=True)
        if args.smoke:
            status('smoke_complete')
        else:
            evaluate(output)
            status('complete')


if __name__ == '__main__':
    main()
