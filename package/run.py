"""Portable adapter for the frozen solve_dag arm. No planning or label access during run."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import fcntl
import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import core
import frozen_flow as flow
from reader import Document

ROOT = Path(__file__).resolve().parent


def load(path):
    return json.loads(Path(path).read_text())


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def asset(config, name):
    p = Path(config[name])
    return p if p.is_absolute() else ROOT / p


def configure(config):
    llm = config['llm_base_url'].rstrip('/')
    embed = config['embedding_base_url'].rstrip('/')
    flow.configure(llm + '/chat/completions', llm + '/completions', embed + '/embeddings')


def data(config):
    inputs = rows(asset(config, 'questions'))
    docs = {k: Document(k, x['title'], x['text']) for k, x in load(asset(config, 'documents')).items()}
    ids = load(asset(config, 'vector_ids'))
    matrix = np.load(asset(config, 'vectors'), mmap_mode='r', allow_pickle=False)
    if matrix.dtype != np.float32 or matrix.ndim != 2 or len(ids) != len(matrix):
        raise ValueError('Invalid vector matrix')
    if len(set(ids)) != len(ids) or set(ids) != set(docs):
        raise ValueError('Document/vector identifiers differ')
    if not np.isfinite(matrix).all() or not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=2e-5):
        raise ValueError('Vectors must be finite unit vectors')
    if len({x['unit_id'] for x in inputs}) != len(inputs):
        raise ValueError('Duplicate question identifiers')
    for x in inputs:
        pool = x['candidate_doc_ids']
        if not pool or len(pool) != len(set(pool)) or not set(pool) <= set(docs):
            raise ValueError('Invalid per-question archive')
        steps = x['plan']['steps']
        if not steps or len(steps) > 6 or len({s['output_slot'] for s in steps}) != len(steps):
            raise ValueError('v1 requires 1-6 uniquely named DAG nodes')
        core.ordered_steps(x['plan'])
    return inputs, docs, SimpleNamespace(vectors=dict(zip(ids, matrix)), lock=threading.Lock())


class ServicePause(RuntimeError):
    pass


class Calls:
    def __init__(self, unit, output, config):
        self.unit, self.output, self.config = unit, Path(output), config

    def get(self, stage, url, payload):
        embedding = url.endswith('/embeddings')
        payload = {**payload, 'model': self.config['embedding_model' if embedding else 'llm_model']}
        identity = {'unit_id': self.unit, 'url': url, 'payload': payload}
        key = digest(identity)
        path = self.output / 'requests' / (key + '.json')
        record = load(path) if path.exists() else {**identity, 'stage': stage, 'attempts': []}
        if any(record[k] != v for k, v in identity.items()):
            raise ValueError('Cache identity mismatch')
        if 'response' in record:
            return {'response': record['response'], 'response_ref': key, 'request': payload}
        api_key = os.environ.get('DAG_EMBED_API_KEY' if embedding else 'DAG_LLM_API_KEY', '')
        while len(record['attempts']) < self.config['max_identical_attempts']:
            attempt = {'started_unix': time.time()}
            record['attempts'].append(attempt)
            save(path, record)
            headers = {'Content-Type': 'application/json'}
            if api_key:
                headers['Authorization'] = 'Bearer ' + api_key
            req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.config['request_timeout_seconds']) as response:
                    body = json.load(response)
                if 'usage' not in body or not body.get('data' if embedding else 'choices'):
                    raise ServicePause('Response missing required usage/data/choices')
            except urllib.error.HTTPError as exc:
                attempt['http_status'] = exc.code
                attempt['retryable'] = exc.code in (408, 429, 500, 502, 503, 504)
                error = exc.read().decode(errors='replace')
                attempt['error'] = error.replace(api_key, '[REDACTED]') if api_key else error
                save(path, record)
                if not attempt['retryable']:
                    raise ServicePause(f'HTTP {exc.code}; see saved attempt') from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError, ServicePause) as exc:
                attempt['error_type'] = type(exc).__name__
                attempt['retryable'] = True
                save(path, record)
            else:
                attempt['http_status'] = 200
                attempt['finished_unix'] = time.time()
                record['response'] = body
                save(path, record)
                return {'response': body, 'response_ref': key, 'request': payload}
            if len(record['attempts']) < self.config['max_identical_attempts']:
                time.sleep(2 * len(record['attempts']))
        raise ServicePause('Persisted identical-request attempt limit reached; preserve pending and history')


def solve(row, documents, tokenizer, index, calls):
    ranking = flow.controller(row, documents, tokenizer, 'solve_dag', calls, index)
    if ranking['status'] == 'ok':
        selections = {str(k): core.select_sources(row['candidate_doc_ids'], ranking['nodes'], k)
                      for k in (5, 10, 20)}
        prepared = flow.reader_input(row, selections['20']['selected_doc_ids'], documents, tokenizer)
        def get(stage, url, payload):
            return calls.get(('solve_dag', 'finalized_thinking', '20', stage), url, payload)
        answer = flow.evaluate(prepared, 'finalized_thinking', get, None)
    else:
        selections = {str(k): {'selected_doc_ids': [], 'retained_node_proofs': 0,
                               'available_node_proofs': 0} for k in (5, 10, 20)}
        answer = {'status': 'node_failed', 'prediction': '', 'error': ranking['error']}
    return {'unit_id': row['unit_id'], 'ranking': ranking, 'budgets': selections, 'answer': answer}


def run(config, output, limit=None):
    from transformers import AutoTokenizer
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'writer.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        configure(config)
        inputs, documents, index = data(config)
        inputs = inputs if limit is None else inputs[:limit]
        if not inputs:
            raise ValueError('Empty run scope')
        manifest = {'method': 'dag_dependency_source_delivery_v1', 'config': config,
                    'unit_ids': [x['unit_id'] for x in inputs], 'inputs_digest': digest(inputs),
                    'asset_checksums': {k: file_hash(asset(config, k)) for k in ('documents', 'vectors', 'vector_ids')},
                    'scope': 'full' if limit is None else 'explicit_subset'}
        existing = output / 'manifest.json'
        if existing.exists() and load(existing) != manifest:
            raise ValueError('Run config/data/scope changed: use a new output directory')
        save(existing, manifest)
        tokenizer = AutoTokenizer.from_pretrained(str(asset(config, 'tokenizer')), local_files_only=True)
        completed = 0
        try:
            for row in inputs:
                result_path = output / 'rows' / (digest(row['unit_id']) + '.json')
                if not result_path.exists():
                    result = solve(row, documents, tokenizer, index, Calls(row['unit_id'], output, config))
                    save(result_path, result)
                completed += 1
                save(output / 'progress.json', {'state': 'running', 'completed': completed, 'total': len(inputs)})
                print(f'{completed}/{len(inputs)}', flush=True)
        except Exception as exc:
            save(output / 'progress.json', {'state': 'paused', 'completed': completed,
                 'total': len(inputs), 'error_type': type(exc).__name__, 'error': str(exc)})
            raise
        save(output / 'progress.json', {'state': 'complete', 'completed': completed, 'total': len(inputs)})


def evaluate(output):
    import metrics
    output = Path(output)
    manifest = load(output / 'manifest.json')
    predictions = {x['unit_id']: x for x in (load(f) for f in (output / 'rows').glob('*.json'))}
    if set(predictions) != set(manifest['unit_ids']):
        raise ValueError('Generation is incomplete; references will not be loaded')
    labels = {x['unit_id']: x for x in rows(asset(manifest['config'], 'references'))}
    scored = []
    for unit in manifest['unit_ids']:
        p, ref = predictions[unit], labels[unit]
        answer = p['answer']
        valid = answer['status'] == 'ok'
        item = {'unit_id': unit, 'f1': metrics.token_f1(answer['prediction'], ref['answers']) if valid else 0.,
                'em': metrics.exact_match(answer['prediction'], ref['answers']) if valid else 0., 'valid': valid}
        gold = set(ref['gold_doc_ids'])
        for k in ('5', '10', '20'):
            ids = set(p['budgets'][k]['selected_doc_ids'])
            item['r@' + k] = len(ids & gold) / len(gold)
            item['all@' + k] = float(gold <= ids)
        scored.append(item)
    report = {'n': len(scored), 'scope': manifest['scope'], 'invalid_answers': sum(not x['valid'] for x in scored),
              'metrics_percent': {k: 100 * sum(x[k] for x in scored) / len(scored)
                                  for k in ('f1', 'em', 'r@5', 'r@10', 'r@20', 'all@5', 'all@10', 'all@20')}}
    save(output / 'scores.json', scored)
    save(output / 'summary.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'run', 'evaluate'))
    parser.add_argument('--config', default=str(ROOT / 'config.example.json'))
    parser.add_argument('--output', default=str(ROOT / 'outputs/run1'))
    parser.add_argument('--limit', type=int, default=None, help='Optional explicit subset; use a distinct output directory')
    args = parser.parse_args()
    config = load(args.config)
    if config['workers'] != 1 or config['max_identical_attempts'] != 3:
        raise ValueError('Portable v1 uses one worker and at most three persisted attempts')
    if args.limit is not None and args.limit < 1:
        raise ValueError('limit must be positive')
    if args.action == 'check':
        inputs, docs, index = data(config)
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(str(asset(config, 'tokenizer')), local_files_only=True)
        print(json.dumps({'questions': len(inputs), 'documents': len(docs),
                          'embedding_dimensions': len(next(iter(index.vectors.values()))),
                          'tokenizer_loaded': True, 'network_calls': 0}))
    elif args.action == 'run':
        run(config, args.output, args.limit)
    else:
        evaluate(args.output)


if __name__ == '__main__':
    main()
