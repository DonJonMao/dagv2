"""Model-specific derived corpus indices, without changing original artifacts.

Document embeddings are made from the exact reader passage (title + newline +
text) with no query instruction. Query encoders belong to the separate model
adapter. A same-dimensional embedding from another model is never accepted as
the same vector space. This module never opens evaluation labels.
"""
from __future__ import annotations

from collections.abc import Mapping
import fcntl
import hashlib
import importlib
import json
from pathlib import Path
import re
import time
from types import SimpleNamespace

import numpy as np

from .budget import Ledger
from .transport import Transport, digest, save


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {"hotpotqa", "2wikimultihopqa", "musique"}
SERIALIZATION = "reader.Document.passage:title_newline_text_nonempty_v1;no_document_instruction"
SCHEMA = "dagbt_derived_embeddings_v1"
BATCH_SIZE = 32


def _file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _load(path):
    return json.loads(Path(path).read_text())


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _corpus(root, dataset):
    if dataset not in DATASETS:
        raise ValueError("Unsupported dataset: " + str(dataset))
    path = Path(root) / "data" / dataset / "corpus.jsonl"
    records = _rows(path)
    ids, passages = [], []
    for record in records:
        doc_id, title, text = record.get("docid"), record.get("title"), record.get("text")
        if not isinstance(doc_id, str) or not doc_id or not isinstance(title, str) or not isinstance(text, str):
            raise ValueError("Invalid original corpus document schema")
        passage = title + "\n" + text if title and text else title or text
        if not passage:
            raise ValueError("Empty corpus passages cannot be silently skipped")
        ids.append(doc_id)
        passages.append(passage)
    if not records or len(ids) != len(set(ids)):
        raise ValueError("Corpus must have nonempty unique document IDs")
    return path, records, ids, passages


def _description(config, dataset, root):
    path, records, ids, passages = _corpus(root, dataset)
    model = config.get("embedding_model")
    base = config.get("embedding_base_url")
    if not isinstance(model, str) or not model or not isinstance(base, str) or not base:
        raise ValueError("Explicit embedding_model and embedding_base_url are required")
    endpoint = base.rstrip("/") + "/embeddings"
    identity_fields = {"schema": SCHEMA, "embedding_model": model, "embedding_endpoint": endpoint,
                       "corpus_sha256": _file_hash(path), "serialization": SERIALIZATION,
                       "document_ids_sha256": digest(ids), "passages_sha256": digest(passages)}
    identity = digest(identity_fields)
    base_directory = Path(config.get("derived_index_dir", "data/derived"))
    if not base_directory.is_absolute():
        base_directory = Path(root) / base_directory
    directory = base_directory / identity / dataset
    report = {"status": "needs_build", "identity": identity, "dataset": dataset,
              "index_dir": str(directory), "manifest_path": str(directory / "manifest.json"),
              "vectors_path": str(directory / "passage_vectors.npy"),
              "documents": len(ids), **identity_fields}
    return report, records, ids, passages


def _validate_matrix(path, rows, dimensions):
    matrix = np.load(path, mmap_mode="r", allow_pickle=False)
    if matrix.dtype != np.float32 or matrix.shape != (rows, dimensions):
        raise ValueError("Derived vector shape/dtype does not match its completed manifest")
    for start in range(0, rows, 256):
        block = matrix[start:start + 256]
        if not np.isfinite(block).all() or not np.allclose(np.linalg.norm(block, axis=1), 1, atol=2e-5):
            raise ValueError("Derived vectors must be finite normalized nonzero rows")
    return matrix


def inspect_index(config, dataset, root=ROOT):
    """Check the precise requested model space, never the old NV index path.

    Absent completion manifest means needs_build. A manifest that claims a
    complete but inconsistent index is an error, not permission to fall back.
    """
    report, _, ids, _ = _description(config, dataset, root)
    manifest_path = Path(report["manifest_path"])
    if not manifest_path.exists():
        report["reason"] = "derived_index_not_completed"
        return report
    manifest = _load(manifest_path)
    required = {key: report[key] for key in (
        "identity", "dataset", "schema", "embedding_model", "embedding_endpoint", "corpus_sha256",
        "serialization", "document_ids_sha256", "passages_sha256", "documents")}
    if any(manifest.get(key) != value for key, value in required.items()) or manifest.get("status") != "complete":
        raise ValueError("Derived index manifest identity/model/corpus/serialization mismatch")
    dimensions = manifest.get("dimensions")
    if type(dimensions) is not int or dimensions <= 0 or manifest.get("dtype") != "float32" or manifest.get("normalized") is not True:
        raise ValueError("Derived index lacks valid dimensions/normalization provenance")
    vectors_path = Path(report["vectors_path"])
    if not vectors_path.is_file() or _file_hash(vectors_path) != manifest.get("vectors_sha256"):
        raise ValueError("Completed derived vector artifact is missing or has changed")
    _validate_matrix(vectors_path, len(ids), dimensions)
    report.update(status="ready", dimensions=dimensions, manifest=manifest)
    return report


def _embedding_batch(response, count, expected_dimension=None):
    data = response.get("data") if isinstance(response, Mapping) else None
    if not isinstance(data, list) or len(data) != count:
        raise ValueError("Embedding response must return exactly one indexed vector per document")
    ordered, seen = {}, set()
    dimension = expected_dimension
    for item in data:
        if not isinstance(item, Mapping):
            raise ValueError("Embedding response item must be a mapping")
        index = item.get("index")
        if type(index) is not int or not 0 <= index < count or index in seen:
            raise ValueError("Embedding response has invalid, duplicate, or missing batch indices")
        raw = item.get("embedding")
        if not isinstance(raw, list) or not raw or any(isinstance(x, (bool, str)) for x in raw):
            raise ValueError("Embedding vector must be a nonempty numeric array")
        try:
            vector = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Embedding vector must be numeric") from exc
        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise ValueError("Embedding vector must be one-dimensional and finite")
        if dimension is None:
            dimension = vector.size
        if vector.size != dimension:
            raise ValueError("Embedding response dimension changed within/between batches")
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("Embedding vector has zero or invalid norm")
        normalized = (vector / norm).astype(np.float32)
        if not np.isfinite(normalized).all() or not np.isclose(np.linalg.norm(normalized), 1, atol=2e-5):
            raise ValueError("Embedding vector cannot be represented as normalized float32")
        ordered[index] = normalized
        seen.add(index)
    return np.stack([ordered[index] for index in range(count)]), int(dimension)


def ensure_index(config, dataset, output, root=ROOT):
    """Build/resume the isolated index using cached document HTTP batches.

    Successful HTTP responses are cached under output/index_build/dataset.
    Restart reconstructs the pending matrix from those exact cached responses;
    it never mixes vector spaces or silently skips a failed document/batch.
    The completion manifest is published only after the full vector checksum.
    """
    batch_size = config.get("embedding_batch_size", BATCH_SIZE)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("embedding_batch_size must be a positive integer")
    report = inspect_index(config, dataset, root)
    if report["status"] == "ready":
        return {**report, "built": False, "additional_embedding_requests": 0}
    report, _, ids, passages = _description(config, dataset, root)
    directory = Path(report["index_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    trace = Path(output) / "index_build" / dataset
    trace.mkdir(parents=True, exist_ok=True)
    with (directory / "build.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Derived index build already active: " + str(directory)) from exc
        # Another process may have completed between initial inspection and lock.
        current = inspect_index(config, dataset, root)
        if current["status"] == "ready":
            return {**current, "built": False, "additional_embedding_requests": 0}
        def event(value):
            with (trace / "ledger_events.jsonl").open("a") as handle:
                handle.write(json.dumps(value, ensure_ascii=False) + "\n")
                handle.flush()
        ledger = Ledger({}, sink=event)  # Offline index costs are separate from question budgets.
        calls = Transport("index:" + report["identity"] + ":" + dataset, trace, config, ledger, None)
        started = time.time()
        pending = directory / "passage_vectors.pending.npy"
        matrix, dimensions = None, None
        try:
            for start in range(0, len(passages), batch_size):
                batch = passages[start:start + batch_size]
                response = calls.get(("index_embedding", str(start)), report["embedding_endpoint"],
                                     {"model": report["embedding_model"], "input": batch})["response"]
                values, dimensions = _embedding_batch(response, len(batch), dimensions)
                if matrix is None:
                    matrix = np.lib.format.open_memmap(pending, mode="w+", dtype=np.float32,
                                                      shape=(len(passages), dimensions))
                matrix[start:start + len(batch)] = values
                matrix.flush()
                save(trace / "progress.json", {"status": "building", "identity": report["identity"],
                     "completed_documents": start + len(batch), "documents": len(ids),
                     "dimensions": dimensions, "elapsed_seconds": time.time() - started,
                     "ledger": ledger.public_dict()})
            matrix.flush()
            del matrix
            matrix = None
            _validate_matrix(pending, len(ids), dimensions)
            pending.replace(report["vectors_path"])
            manifest = {key: value for key, value in report.items()
                        if key not in ("status", "index_dir", "manifest_path", "vectors_path")}
            manifest.update(status="complete", dimensions=dimensions, dtype="float32", normalized=True,
                            vectors_sha256=_file_hash(report["vectors_path"]), batch_size=batch_size,
                            document_instruction="", created_unix=time.time(),
                            elapsed_seconds=time.time() - started, index_build_trace_directory=str(trace),
                            cost_scope="separate offline corpus embedding; inspect request cache for retries/usage",
                            ledger=ledger.public_dict())
            save(report["manifest_path"], manifest)
            save(trace / "progress.json", {"status": "complete", "identity": report["identity"],
                 "completed_documents": len(ids), "documents": len(ids), "dimensions": dimensions,
                 "elapsed_seconds": time.time() - started, "ledger": ledger.public_dict()})
        except BaseException as exc:
            if matrix is not None:
                matrix.flush()
            save(trace / "progress.json", {"status": "incomplete", "identity": report["identity"],
                 "error_type": type(exc).__name__, "documents": len(ids), "ledger": ledger.public_dict()})
            raise
    result = inspect_index(config, dataset, root)
    return {**result, "built": True, "additional_embedding_requests": ledger.used["http_attempts"]}


def prepare_resources(config, dataset, pipeline, tokenizer, root=ROOT):
    """Load the new model's index while preserving original solver algorithms."""
    if config.get("model_profile") != "bridgetree":
        raise ValueError("Derived resources require explicit model_profile='bridgetree'")
    report = inspect_index(config, dataset, root)
    if report["status"] != "ready":
        raise ValueError("Requested embedding index must be built before worker startup; no NV fallback")
    corpus_path, records, ids, _ = _corpus(root, dataset)
    question_path = Path(root) / "data" / dataset / "questions.jsonl"
    questions = []
    for item in _rows(question_path):
        if not isinstance(item.get("id"), str) or not item["id"] or not isinstance(item.get("question"), str) or not item["question"].strip():
            raise ValueError("Invalid question ID/text")
        questions.append({"id": item["id"], "question": item["question"]})
    if not questions or len({q["id"] for q in questions}) != len(questions):
        raise ValueError("Questions require nonempty unique IDs")
    e = pipeline.e
    canonical_path = Path(root) / "data" / dataset / "corpus.json"
    if canonical_path.exists():
        canonical = [{"doc_id": d["docid"], "title": d["title"], "text": d["text"]} for d in records]
        if _load(canonical_path) != canonical:
            raise ValueError("Original corpus canonical copies disagree")
    docs = {d["docid"]: e.Document(d["docid"], d["title"], d["text"]) for d in records}
    vectors = _validate_matrix(report["vectors_path"], len(ids), report["dimensions"])
    index = SimpleNamespace(vectors=dict(zip(ids, vectors)), lock=e.EMBED_LOCK)
    e.CONFIG = {**e.CONFIG, **config, "dataset": dataset,
                "references": str(Path(root) / "data" / dataset / "evaluation_only.json"),
                "planner_annotation_fix": "strip only trailing annotation matching current output_slot"}
    repair = getattr(pipeline, "repair", None) or importlib.import_module("repair_planner")
    e.validate_plan = repair.validate_plan
    e.native.configure(e.CONFIG)
    assets = [corpus_path, question_path, Path(report["manifest_path"]), Path(report["vectors_path"]), Path(__file__)]
    if canonical_path.exists():
        assets.append(canonical_path)
    hashes = {str(path): _file_hash(path) for path in assets}
    return questions, (docs, ids, vectors, index, tokenizer), hashes


def archive_for_resources(question, plan, ids, vectors, calls, e):
    """Original archive algorithm with the loaded model's vector dimension.

    Keep placeholder substitution, query order/deduplication, instruction,
    per-query stable top-50, pool union, and trace identical to e.archive.
    Only the original fixed 4096-dimensional validity guard is generalized.
    """
    queries = list(dict.fromkeys([question] + [re.sub(r'\{[^{}]+\}', 'the unknown entity', s['question']) for s in plan['steps']]))
    pool, seen, trace = [], set(), []
    for i, query in enumerate(queries):
        with e.EMBED_LOCK:
            response = calls.get(('archive', str(i)), e.CONFIG['embedding_base_url'] + '/embeddings',
                {'input': ['Instruct: ' + e.flow.INSTRUCTION + '\nQuery: ' + query]})['response']
        v = np.asarray(response['data'][0]['embedding'], dtype=np.float32)
        if v.shape != (vectors.shape[1],) or not np.isfinite(v).all() or np.linalg.norm(v) == 0:
            raise e.native.ServicePause('Invalid query embedding')
        scores = vectors @ (v / np.linalg.norm(v))
        order = np.argsort(-scores, kind='stable')[:50]
        hits = [ids[j] for j in order]
        trace.append(dict(query=query, doc_ids=hits, scores=[float(scores[j]) for j in order]))
        for d in hits:
            if d not in seen:
                seen.add(d)
                pool.append(d)
    return pool, trace
