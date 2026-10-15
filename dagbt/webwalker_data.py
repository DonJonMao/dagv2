"""Pinned WebWalkerQA main split; generation never reads answers or oracle paths."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

from .transport import save

REVISION = 'c7ab3329a2a4a35ca569fdca369b61a3d1787e5b'
RAW_SHA256 = 'cbe95d9de714ef52fe4a9ddf160dfab462407cb1ba526f1c31b4305ca32c34cb'
SOURCE = f'https://huggingface.co/datasets/callanwu/WebWalkerQA/resolve/{REVISION}/data/main-00000-of-00001.jsonl'
SCHEMA = 'webwalkerqa_main_root_traversal_v1'


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(raw, destination):
    raw, destination = Path(raw), Path(destination)
    if file_hash(raw) != RAW_SHA256:
        raise ValueError('WebWalkerQA raw file differs from the pinned official main split')
    rows = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
    if len(rows) != 680:
        raise ValueError('The official main split must contain all 680 questions')
    questions, references = [], {}
    for i, row in enumerate(rows):
        question, answer, root = (row.get(k) for k in ('question', 'answer', 'root_url'))
        if any(not isinstance(v, str) or not v.strip() for v in (question, answer, root)):
            raise ValueError(f'Invalid WebWalkerQA record {i}')
        parsed = urlsplit(root)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError(f'Invalid root URL at record {i}')
        qid = f'wwqa-{i:04d}'
        questions.append({'id': qid, 'question': question, 'root_url': root})
        references[qid] = {'answer': answer, 'info': row.get('info', {})}
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / 'questions.jsonl'
    path.write_text(''.join(json.dumps(q, ensure_ascii=False) + '\n' for q in questions))
    save(destination / 'evaluation_only.json', references)
    manifest = {'schema': SCHEMA, 'source': SOURCE, 'revision': REVISION,
                'raw_sha256': RAW_SHA256, 'questions': 680,
                'questions_sha256': file_hash(path),
                'evaluation_sha256': file_hash(destination / 'evaluation_only.json'),
                'generation_fields': ['id', 'question', 'root_url'],
                'oracle_fields_for_evaluation_only': ['answer', 'info']}
    save(destination / 'manifest.json', manifest)
    return manifest


def load_questions(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('schema') != SCHEMA or manifest.get('raw_sha256') != RAW_SHA256:
        raise ValueError('Unrecognized WebWalkerQA data manifest')
    path = directory / 'questions.jsonl'
    if file_hash(path) != manifest['questions_sha256']:
        raise ValueError('Generation questions changed after data preparation')
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 680 or len({q.get('id') for q in rows}) != 680:
        raise ValueError('Missing/duplicate WebWalkerQA main questions')
    for row in rows:
        if set(row) != {'id', 'question', 'root_url'}:
            raise ValueError('Oracle metadata is forbidden in generation input')
    return rows, manifest


def load_references(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    path = directory / 'evaluation_only.json'
    if file_hash(path) != manifest['evaluation_sha256']:
        raise ValueError('Evaluation references changed')
    return json.loads(path.read_text())
