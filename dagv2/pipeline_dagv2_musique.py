"""DAG v2 pipeline for MuSiQue: no-thinking, corpus-wide node retrieval,
reader node-chain injection, musique doc-id normalization. Package layout:
dagv2/ sits beside data/ and package/."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import argparse
import concurrent.futures
import fcntl
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT = Path(__file__).resolve().parent            # dagv2/
PACKAGE_ROOT = ROOT.parent                        # package root (this delivery)
sys.path.insert(0, str(ROOT))
import experiment_v6  # noqa: F401  patches controller/reader/chat_request/solve; puts package/ on sys.path
import experiment as e
import repair_planner as repair
import metrics
DATA = PACKAGE_ROOT / 'data'
SOURCE = DATA  # per-dataset EvLink++-derived assets live under data/<dataset>/
EVLINK_MANIFEST = DATA / 'inputs_manifest.json'


def provenance_hash(inputs, path):
    """Look up the file hash in inputs_manifest by trailing path components.

    The manifest was recorded with absolute source-server paths; this delivery is
    relocatable, so keys are matched by (parent dir name, file name). Exactly one
    consistent hash must match, otherwise preparation fails loudly.
    """
    path = Path(path)
    matches = {h for k, h in inputs.items()
               if Path(k).parent.name == path.parent.name and Path(k).name == path.name}
    if len(matches) != 1:
        raise ValueError(f'provenance entry missing or ambiguous for {path}: {sorted(matches)}')
    return matches.pop()


def prepare(name, target):
    source = SOURCE / name
    qs = e.native.rows(DATA / name / 'questions.jsonl')
    corpus = e.native.load(source / 'corpus.json')
    canonical_docs = e.native.rows(DATA / name / 'corpus.jsonl')
    assert corpus == [dict(doc_id=d['docid'], title=d['title'], text=d['text']) for d in canonical_docs]
    assert qs == [dict(id=q['id'], question=q['question']) for q in e.native.load(source / 'questions.json')]
    assert len(qs) == 1000 and len({q['id'] for q in qs}) == 1000  # musique subset also 1000
    provenance = e.native.load(EVLINK_MANIFEST)[name]
    for f in ('questions.jsonl', 'corpus.jsonl'):
        path = DATA / name / f
        assert e.native.file_hash(path) == provenance_hash(provenance['inputs'], path), f'hash mismatch: {path}'
    im = e.native.load(source / 'index/manifest.json')
    assert im['embedding_model'] == e.CONFIG['embedding_model'] and im['documents'] == len(corpus)
    vectors = np.load(source / 'index/passage_vectors.npy', allow_pickle=False).astype(np.float32)
    assert vectors.shape == (len(corpus), 4096) and np.isfinite(vectors).all()
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    assert (norms > 0).all()
    vectors /= norms
    ids = [d['doc_id'] for d in corpus]
    assert len(set(ids)) == len(ids)
    docs = {d['doc_id']: e.Document(d['doc_id'], d['title'], d['text']) for d in corpus}
    index = SimpleNamespace(vectors=dict(zip(ids, vectors)), lock=e.EMBED_LOCK)
    tokenizer = e.AutoTokenizer.from_pretrained(str(e.TOKENIZER), local_files_only=True)
    e.CONFIG = {**e.CONFIG, 'dataset': name, 'references': str(source / 'evaluation_only.json'),
                'planner_annotation_fix': 'strip only trailing annotation matching current output_slot'}
    e.validate_plan = repair.validate_plan
    e.native.configure(e.CONFIG)
    assets = [DATA / name / 'questions.jsonl', DATA / name / 'corpus.jsonl', source / 'corpus.json',
              source / 'index/passage_vectors.npy', source / 'index/manifest.json',
              Path(__file__), ROOT / 'experiment.py', ROOT / 'repair_planner.py',
              ROOT / 'experiment_v6.py', ROOT / 'frozen_flow_v6.py',
              e.TOKENIZER / 'tokenizer_config.json', e.TOKENIZER / 'tokenizer.json',
              *[e.PACKAGE / f for f in ('run.py', 'core.py', 'frozen_flow.py', 'reader.py', 'metrics.py')]]
    hashes = {str(p): e.native.file_hash(p) for p in assets}
    return qs, (docs, ids, vectors, index, tokenizer), hashes


def evaluate(name, output):
    manifest = e.native.load(output / 'manifest.json')
    predictions = [e.native.load(p) for p in (output / 'rows').glob('*.json')]
    assert len(predictions) == 1000 and {p['unit_id'] for p in predictions} == set(manifest['unit_ids'])
    # Gold is only loaded after all answers are generated; use established title-support groups.
    label_path = SOURCE / name / 'evaluation_only.json'
    labels = {x['id']: x for x in e.native.load(label_path)}
    assert set(labels) == set(manifest['unit_ids'])
    scored = []
    for p in predictions:
        label = labels[p['unit_id']]
        valid = p['answer']['status'] == 'ok'
        s = dict(unit_id=p['unit_id'], valid=valid,
                 f1=metrics.token_f1(p['answer']['prediction'], label['answers']) if valid else 0.,
                 em=metrics.exact_match(p['answer']['prediction'], label['answers']) if valid else 0.)
        groups = label['gold_groups']
        assert groups and all(groups)
        for k in ('5', '10', '20'):
            ids = {d if str(d).startswith('musique:') else 'musique:' + str(d) for d in p['budgets'][k]['selected_doc_ids']}
            hits = [bool(ids.intersection(g)) for g in groups]
            s['r@' + k] = sum(hits) / len(hits)
            s['all@' + k] = float(all(hits))
        scored.append(s)
    report = dict(dataset=name, n=len(scored), scope='current local 1000-question subset',
        invalid_answers=sum(not x['valid'] for x in scored),
        answer_status_counts=dict(Counter(p['answer']['status'] for p in predictions)),
        metrics_percent={k:100*sum(s[k] for s in scored)/len(scored)
                         for k in ('f1','em','r@5','r@10','r@20','all@5','all@10','all@20')},
        labels_sha256=e.native.file_hash(label_path),
        support_metric='macro recall of gold title groups; any matching document satisfies a group')
    e.native.save(output / 'scores.json', scored)
    e.native.save(output / 'summary.json', report)
    lines = [f'# DAG v2 (no thinking, corpus-wide node retrieval, reader node-chain injection) / {name} / Qwen3.8 + NV-Embed-v2', '',
             '| Metric | Percent |', '|---|---:|']
    lines += [f'| {k} | {v:.2f} |' for k,v in report['metrics_percent'].items()]
    (output / 'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report), flush=True)


def execute(name, target, qs, resources, hashes, smoke):
    output = target / ('smoke2' if smoke else 'full1000')
    output.mkdir(exist_ok=True)
    with (output / 'writer.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        selected = qs[:2] if smoke else qs
        manifest = dict(dataset=name, config=e.CONFIG, unit_ids=[q['id'] for q in selected],
            scope='smoke2' if smoke else 'full1000', hashes=hashes, sampling=e.SAMPLING,
            planner_system=e.PLAN_SYSTEM, planner_schema=e.PLAN_SCHEMA,
            archive='stable union top50 original question plus each ungrounded DAG question',
            node_retrieval='corpus-wide top50 union per grounded node query; node panel top20 (frozen_flow_v6)',
            thinking='disabled: nodes single guided-JSON call (512 tokens), reader non-thinking chat (1024 tokens)',
            workers=2, context_limit=16384, node_thinking_tokens=0, node_final_tokens=512,
            reader_thinking_tokens=0, reader_final_tokens=1024,
            reader_passages=20,
            support_metric='gold title groups as existing EvLink++ evaluation')
        path = output / 'manifest.json'
        if path.exists() and e.native.load(path) != manifest:
            raise ValueError('Changed manifest; use separate output directory')
        e.native.save(path, manifest)
        def done(q):return (output / 'rows' / (e.native.digest(q['id'])+'.json')).exists()
        completed = sum(done(q) for q in selected)
        start = time.time()
        def status(state, **kw):
            record = dict(state=state, dataset=name, completed=completed, total=len(selected),
                elapsed_seconds=time.time()-start, updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), **kw)
            e.native.save(output / 'progress.json',record)
            print(json.dumps(record),flush=True)
        status('running')
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        futures = [executor.submit(e.work,q,output,*resources) for q in selected if not done(q)]
        try:
            for f in concurrent.futures.as_completed(futures):
                row = f.result();completed += 1
                status('running',last_id=row['unit_id'],last_status=row['answer']['status'])
        except Exception as exc:
            for f in futures:f.cancel()
            executor.shutdown(wait=True,cancel_futures=True)
            completed = sum(done(q) for q in selected)
            status('paused',error_type=type(exc).__name__,error=str(exc))
            raise
        else:executor.shutdown(wait=True)
        if smoke:
            statuses = [e.native.load(output/'rows'/(e.native.digest(q['id'])+'.json'))['answer']['status'] for q in selected]
            if any(s != 'ok' for s in statuses):
                status('smoke_failed',answer_statuses=statuses)
                raise RuntimeError('Smoke execution failed; inspect before full run')
            status('smoke_complete')
        else:
            evaluate(name,output)
            status('complete')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dataset', nargs='?', default='musique')
    p.add_argument('--prepare-only', action='store_true',
                   help='only validate data paths/hashes and load assets, then exit (no model calls)')
    p.add_argument('--smoke-only', action='store_true',
                   help='run the 2-question smoke pass only, skip full1000')
    a = p.parse_args()
    a.dataset = 'musique'
    target = PACKAGE_ROOT / 'outputs' / a.dataset
    target.mkdir(parents=True, exist_ok=True)
    with (target / 'pipeline.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        qs, resources, hashes = prepare(a.dataset, target)
        if a.prepare_only:
            print(a.dataset, 'prepare ok', len(qs), flush=True)
            return
        execute(a.dataset, target, qs, resources, hashes, True)
        if a.smoke_only:
            return
        dest = target / 'full1000/requests'
        dest.mkdir(parents=True, exist_ok=True)
        for path in (target / 'smoke2/requests').glob('*.json'):
            if 'response' in e.native.load(path) and not (dest / path.name).exists():
                shutil.copy2(path, dest / path.name)
        execute(a.dataset, target, qs, resources, hashes, False)

if __name__=='__main__':main()
