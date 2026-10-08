#!/usr/bin/env python3
"""Export a small read-only answer/log bundle from an existing DAGv2 run.

Compatible with the frozen 20260930 v3 package. No model calls, corpus vectors,
request caches, local credentials, or full diagnostic rows are included. This
extracts the top-level answer from native indent=2 JSON, not the entire row.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tarfile

LOG_TAIL_BYTES = 256 * 1024
MAX_MEMBER_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024
DATASETS = {'hotpotqa', '2wikimultihopqa', 'musique', 'personamem'}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(value):
    return sha(json.dumps(value, sort_keys=True, ensure_ascii=False).encode())


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()


def small_read(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError('Expected a regular file: ' + str(path))
    with path.open('rb') as handle:
        data = handle.read(MAX_MEMBER_BYTES + 1)
    if len(data) > MAX_MEMBER_BYTES:
        raise ValueError('Metadata exceeds export size limit: ' + str(path))
    return data


def read_answer(path):
    """Stop after the complete top-level answer; never parse the diagnostic tail."""
    if path.is_symlink():
        raise ValueError('Result symlink is not allowed')
    prefix = '  "answer": '
    decoder = json.JSONDecoder()
    with path.open(encoding='utf-8') as handle:
        if handle.readline(MAX_MEMBER_BYTES + 1).strip() != '{':
            raise ValueError('Expected native pretty-JSON object root')
        for line in handle:
            if not line.startswith(prefix):
                continue
            text = line[len(prefix):]
            while True:
                if len(text.encode('utf-8')) > MAX_MEMBER_BYTES:
                    raise ValueError('Answer object exceeds export size limit')
                try:
                    value, end = decoder.raw_decode(text)
                except json.JSONDecodeError as exc:
                    line = next(handle, None)
                    if line is None:
                        raise ValueError('Incomplete answer JSON') from exc
                    text += line
                    continue
                if text[end:].strip() not in ('', ','):
                    raise ValueError('Unexpected data after answer object')
                if not isinstance(value, dict) or not isinstance(value.get('status'), str) or not isinstance(value.get('prediction'), str):
                    raise ValueError('Invalid answer status/prediction')
                return value
    raise ValueError('Top-level answer not found in native indent=2 JSON')


def _redact(value):
    if isinstance(value, dict):
        return {k: '[REDACTED]' if any(word in k.lower() for word in ('api_key', 'password', 'secret', 'access_token', 'authorization')) else _redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


def _log_tail(path):
    if path.is_symlink():
        raise ValueError('Log symlink is not allowed')
    with path.open('rb') as handle:
        size = path.stat().st_size
        handle.seek(max(0, size - LOG_TAIL_BYTES))
        data = handle.read(LOG_TAIL_BYTES)
    if size > LOG_TAIL_BYTES:
        data = data.partition(b'\n')[2]
    text = data.decode('utf-8', errors='replace')
    text = re.sub(r'(?i)\bBearer\s+[^\s"\x27]+', 'Bearer [REDACTED]', text)
    return text.encode(), {'source_bytes': size, 'tail_only': size > LOG_TAIL_BYTES}


def export_bundle(root, run, destination):
    root, run, destination = Path(root).resolve(), Path(run).resolve(), Path(destination).absolute()
    resolved_destination = destination.resolve()
    if resolved_destination == run or run in resolved_destination.parents:
        raise ValueError('Export must be outside the source run')
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('Refusing to overwrite export: ' + str(destination))
    partial = destination.with_name(destination.name + '.partial')
    if partial.exists() or partial.is_symlink():
        raise FileExistsError('Existing partial export will not be overwritten')
    if not destination.parent.is_dir():
        raise ValueError('Export parent directory does not exist')

    manifest_bytes = small_read(run / 'manifest.json')
    manifest = json.loads(manifest_bytes)
    arms, datasets = manifest['arms'], manifest['datasets']
    if not arms or len(set(arms)) != len(arms) or not datasets or len(set(datasets)) != len(datasets):
        raise ValueError('Empty or duplicate arms/datasets')
    if not set(datasets) <= DATASETS or any(not isinstance(a, str) or re.fullmatch(r'[a-z][a-z0-9_]*', a) is None for a in arms):
        raise ValueError('Unsafe or unsupported dataset/arm')
    report = {'schema': 'dagv2_compact_results_v1', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'source_manifest_sha256': sha(manifest_bytes), 'model_calls': 0, 'source_run_modified': False,
              'row_validation': 'Top-level answer only; diagnostic tail and full row JSON are not parsed or hashed.',
              'snapshot': 'Each authoritative row read once, with before/after file identity checks; no global transaction.',
              'counts': {}, 'errors': [], 'logs': {}, 'included_datasets': []}
    members = {'run/manifest.json': encode(_redact(manifest))}
    original_bytes = small_read(root / 'original_manifest.json')
    if manifest.get('original_hash_manifest') and sha(original_bytes) != manifest['original_hash_manifest']:
        raise ValueError('Original manifest differs from run identity')
    original = {r['path']: r['sha256'] for r in json.loads(original_bytes)['files']}
    members['original_manifest.json'] = original_bytes
    # Frozen normalization/EM/F1 implementation is small and contains no credentials.
    metric_path = 'package/metrics.py'
    if metric_path in original:
        raw = small_read(root / metric_path)
        if sha(raw) != original[metric_path]:
            raise ValueError('Frozen metrics checksum mismatch')
        members[metric_path] = raw
    records = []
    for dataset in datasets:
        units = manifest['question_ids'][dataset]
        if any(not isinstance(u, str) or not u for u in units) or len(set(units)) != len(units):
            raise ValueError('Invalid or duplicate question IDs')
        report['counts'][dataset] = {}
        dataset_count = 0
        for arm in arms:
            counts = dict(expected=len(units), exported=0, status_ok=0, status_failed=0, missing=0, invalid=0)
            report['counts'][dataset][arm] = counts
            print(f'{dataset}/{arm}: 开始导出 {len(units)} 个任务的答案', flush=True)
            for i, unit in enumerate(units, 1):
                if i == 1 or i % 25 == 0:
                    print(f'  正在读取 {i}/{len(units)}', flush=True)
                relative = f'{dataset}/{arm}/rows/{digest(unit)}.json'
                path = run / relative
                if not path.exists():
                    counts['missing'] += 1
                    continue
                try:
                    if not path.resolve().is_relative_to(run):
                        raise ValueError('Result path escapes source run')
                    before = path.stat()
                    answer = read_answer(path)
                    after = path.stat()
                    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
                        raise ValueError('Result changed during export')
                    minimal = {k: answer[k] for k in ('status', 'prediction', 'error_type', 'error', 'finish_reason') if k in answer}
                    records.append({'dataset': dataset, 'arm': arm, 'unit_id': unit, 'answer': minimal,
                        'source': {'relative_path': relative, 'bytes': before.st_size, 'mtime_ns': before.st_mtime_ns,
                                   'answer_sha256': digest(answer)}})
                    counts['exported'] += 1
                    counts['status_ok' if answer['status'] == 'ok' else 'status_failed'] += 1
                    dataset_count += 1
                except (OSError, ValueError, UnicodeError) as exc:
                    counts['invalid'] += 1
                    report['errors'].append({'dataset': dataset, 'arm': arm, 'unit_id': unit, 'error': str(exc)[:1000]})
            print(f'  已导出 {counts["exported"]}，未落盘 {counts["missing"]}，读取异常 {counts["invalid"]}', flush=True)
        if not dataset_count:
            continue
        report['included_datasets'].append(dataset)
        personal = None
        if dataset == 'personamem':
            relative = 'data/personamem/manifest.json'
            raw = small_read(root / relative)
            expected = manifest.get('dataset_manifests', {}).get(dataset)
            if expected and sha(raw) != expected:
                raise ValueError('PersonaMem manifest differs from run')
            members[relative] = raw
            personal = json.loads(raw)
            # Preserve exact strict choice parser and its canonical label helper.
            for relative in ('dagbt/personamem.py', 'vendor/bridgetree/metrics.py'):
                expected = manifest.get('source_hashes', {}).get(relative)
                if expected:
                    raw = small_read(root / relative)
                    if sha(raw) != expected:
                        raise ValueError('PersonaMem choice parser differs from run')
                    members[relative] = raw
        for filename in ('questions.jsonl', 'evaluation_only.json'):
            relative = f'data/{dataset}/{filename}'
            raw = small_read(root / relative)
            expected = (personal['evaluation_sha256'] if filename == 'evaluation_only.json' else personal['public_sha256'][filename]) if personal else original[relative]
            if sha(raw) != expected:
                raise ValueError('Dataset checksum mismatch: ' + relative)
            members[relative] = raw
    if small_read(run / 'manifest.json') != manifest_bytes:
        raise ValueError('Run manifest changed during export')
    members['answers.jsonl'] = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records).encode()
    for filename in ('progress.json', 'pid.json'):
        path = run / filename
        if path.exists():
            try:
                members['run/' + filename] = encode(_redact(json.loads(small_read(path))))
            except (OSError, ValueError) as exc:
                report['errors'].append({'file': filename, 'error': str(exc)[:1000]})
    for filename in ('events.jsonl', 'launcher.log'):
        path = run / filename
        if path.exists():
            try:
                members['run/' + filename], report['logs'][filename] = _log_tail(path)
            except (OSError, ValueError) as exc:
                report['errors'].append({'file': filename, 'error': str(exc)[:1000]})
    report['finished_utc'] = datetime.now(timezone.utc).isoformat()
    report['export_complete'] = not report['errors']
    report['member_sha256'] = {name: sha(raw) for name, raw in members.items()}
    members['report.json'] = encode(report)
    if sum(map(len, members.values())) > MAX_TOTAL_BYTES:
        raise ValueError('Export exceeds 64 MiB uncompressed limit; no archive written')
    # Own the temporary name exclusively and promote only a complete archive.
    try:
        handle = partial.open('xb')
    except FileExistsError:
        raise FileExistsError('Existing partial export will not be overwritten') from None
    try:
        with handle, tarfile.open(fileobj=handle, mode='w:gz') as archive:
            for name, raw in members.items():
                info = tarfile.TarInfo(name)
                info.size, info.mode = len(raw), 0o600
                archive.addfile(info, io.BytesIO(raw))
        # Hard-link publication refuses a destination created concurrently.
        os.link(partial, destination)
        partial.unlink()
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    print(f'导出完成：{destination}（{destination.stat().st_size / 1024 / 1024:.2f} MiB）', flush=True)
    if report['errors']:
        print('注意：报告中有读取异常，分析时会单列，未伪造缺失答案。', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--run', type=Path, default=Path('outputs/paired_full_reliability_v3'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    run = (root / args.run).resolve()
    destination = args.output or Path.home() / ('dagv2_results_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.tar.gz')
    export_bundle(root, run, destination)


if __name__ == '__main__':
    main()
