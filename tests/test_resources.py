"""Derived model-space indices: real loopback HTTP, no model/label access."""
from contextlib import contextmanager
import fcntl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from dagbt.resources import (_embedding_batch, archive_for_resources, ensure_index,
                             inspect_index, prepare_resources)
from dagbt.transport import ServiceError


def corpus(root, count=3):
    directory = root / "data" / "hotpotqa"
    directory.mkdir(parents=True)
    rows = [{"docid": "d" + str(i), "title": "Title " + str(i), "text": "Raw source " + str(i)}
            for i in range(count)]
    (directory / "corpus.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (directory / "corpus.json").write_text(json.dumps([
        {"doc_id": d["docid"], "title": d["title"], "text": d["text"]} for d in rows]))
    (directory / "questions.jsonl").write_text(json.dumps({"id": "q", "question": "A factual question?"}) + "\n")
    (directory / "evaluation_only.json").write_bytes(b"\xff LABELS MUST NEVER BE READ")
    legacy = directory / "index"
    legacy.mkdir()
    (legacy / "manifest.json").write_text('{"embedding_model":"nvidia/NV-Embed-v2"}')
    (legacy / "passage_vectors.npy").write_bytes(b"UNCHANGED ORIGINAL NV ARTIFACT")
    return rows


@contextmanager
def service():
    state = {"requests": [], "fail_second": False}
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            assert self.path == "/v1/embeddings"
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append(payload)
            if state["fail_second"] and len(payload["input"]) == 1:
                self.send_response(503); self.end_headers(); return
            body = {"data": [{"index": i, "embedding": [float(i + 1), 2., 3.]}
                              for i in reversed(range(len(payload["input"])))],
                    "usage": {"prompt_tokens": len(payload["input"]), "total_tokens": len(payload["input"])}}
            encoded = json.dumps(body).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded))); self.end_headers(); self.wfile.write(encoded)
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    config = {"model_profile": "bridgetree", "embedding_model": "qwen3-embedding-8b",
              "embedding_base_url": "http://127.0.0.1:%d/v1" % server.server_port,
              "request_timeout_seconds": 2, "max_identical_attempts": 1}
    try:
        yield state, config
    finally:
        server.shutdown(); server.server_close(); thread.join()


def test_build_raw_passages_reorders_indexed_vectors_and_preserves_nv(tmp_path):
    rows = corpus(tmp_path)
    with service() as (state, config):
        initial = inspect_index(config, "hotpotqa", root=tmp_path)
        assert initial["status"] == "needs_build"
        result = ensure_index(config, "hotpotqa", tmp_path / "run", root=tmp_path)
        assert result["status"] == "ready" and result["built"]
        assert result["dimensions"] == 3
        assert state["requests"][0]["input"] == [d["title"] + "\n" + d["text"] for d in rows]
        assert state["requests"][0]["model"] == config["embedding_model"]
        vectors = np.load(result["vectors_path"])
        expected = np.array([[1., 2., 3.], [2., 2., 3.], [3., 2., 3.]], dtype=np.float32)
        expected /= np.linalg.norm(expected, axis=1, keepdims=True)
        assert vectors.dtype == np.float32 and np.allclose(vectors, expected)
        assert Path(result["index_dir"]).is_relative_to(tmp_path / "data" / "derived")
        old = tmp_path / "data" / "hotpotqa" / "index" / "passage_vectors.npy"
        assert old.read_bytes() == b"UNCHANGED ORIGINAL NV ARTIFACT"
        reused = ensure_index(config, "hotpotqa", tmp_path / "other_run", root=tmp_path)
        assert not reused["built"] and len(state["requests"]) == 1
        assert (tmp_path / "run" / "index_build" / "hotpotqa" / "progress.json").is_file()


def test_resume_replays_successful_batch_cache_without_mixing_spaces(tmp_path):
    corpus(tmp_path, 33)
    output = tmp_path / "run"
    with service() as (state, config):
        state["fail_second"] = True
        with pytest.raises(ServiceError):
            ensure_index(config, "hotpotqa", output, root=tmp_path)
        incomplete = inspect_index(config, "hotpotqa", root=tmp_path)
        assert incomplete["status"] == "needs_build"
        assert not Path(incomplete["manifest_path"]).exists()
        assert [len(p["input"]) for p in state["requests"]] == [32, 1]
        state["fail_second"] = False
        result = ensure_index(config, "hotpotqa", output, root=tmp_path)
        assert result["status"] == "ready" and result["additional_embedding_requests"] == 1
        assert [len(p["input"]) for p in state["requests"]] == [32, 1, 1]
        assert result["manifest"]["ledger"]["used"]["cache_hits"] == 1
        assert np.load(result["vectors_path"]).shape == (33, 3)


def test_identity_changes_with_model_endpoint_corpus_even_if_dimensions_match(tmp_path):
    corpus(tmp_path)
    config = {"embedding_model": "model-a", "embedding_base_url": "http://localhost:1/v1"}
    first = inspect_index(config, "hotpotqa", root=tmp_path)
    model = inspect_index({**config, "embedding_model": "model-b"}, "hotpotqa", root=tmp_path)
    endpoint = inspect_index({**config, "embedding_base_url": "http://localhost:2/v1"}, "hotpotqa", root=tmp_path)
    path = tmp_path / "data" / "hotpotqa" / "corpus.jsonl"
    path.write_text(path.read_text().replace("Raw source 0", "Changed raw source"))
    changed = inspect_index(config, "hotpotqa", root=tmp_path)
    assert len({first["identity"], model["identity"], endpoint["identity"], changed["identity"]}) == 4


@pytest.mark.parametrize("override", ["separate/derived", "absolute"])
def test_explicit_derived_directory_does_not_change_vector_space_identity(tmp_path, override):
    corpus(tmp_path)
    config = {"embedding_model": "model-a", "embedding_base_url": "http://localhost:1/v1"}
    default = inspect_index(config, "hotpotqa", root=tmp_path)
    directory = tmp_path / "absolute" if override == "absolute" else Path(override)
    other = inspect_index({**config, "derived_index_dir": str(directory)}, "hotpotqa", root=tmp_path)
    expected = directory if directory.is_absolute() else tmp_path / directory
    assert Path(other["index_dir"]).is_relative_to(expected)
    assert other["identity"] == default["identity"]


@pytest.mark.parametrize("data", [
    [],
    [{"index": 0, "embedding": [1., 2.]}, {"index": 0, "embedding": [2., 1.]}],
    [{"index": 0.0, "embedding": [1., 2.]}, {"index": 1, "embedding": [2., 1.]}],
    [{"index": 0, "embedding": [0., 0.]}, {"index": 1, "embedding": [2., 1.]}],
    [{"index": 0, "embedding": [float("nan"), 1.]}, {"index": 1, "embedding": [2., 1.]}],
    [{"index": 0, "embedding": [1., 2., 3.]}, {"index": 1, "embedding": [2., 1.]}],
    [{"index": 0, "embedding": ["1", 2.]}, {"index": 1, "embedding": [2., 1.]}],
])
def test_bad_embeddings_are_rejected_not_silently_skipped(data):
    with pytest.raises(ValueError):
        _embedding_batch({"data": data}, 2)


def test_dimension_cannot_change_between_batches():
    with pytest.raises(ValueError, match="dimension changed"):
        _embedding_batch({"data": [{"index": 0, "embedding": [1., 2., 3.]}]}, 1, expected_dimension=4)


def test_complete_corrupt_index_is_error_not_nv_fallback_or_automatic_rebuild(tmp_path):
    corpus(tmp_path)
    with service() as (state, config):
        result = ensure_index(config, "hotpotqa", tmp_path / "run", root=tmp_path)
        Path(result["vectors_path"]).write_bytes(b"corrupt")
        with pytest.raises(ValueError, match="missing or has changed"):
            ensure_index(config, "hotpotqa", tmp_path / "run", root=tmp_path)
        assert len(state["requests"]) == 1


def test_manifest_model_is_checked_even_when_vector_shape_would_match(tmp_path):
    corpus(tmp_path)
    with service() as (_, config):
        result = ensure_index(config, "hotpotqa", tmp_path / "run", root=tmp_path)
        path = Path(result["manifest_path"])
        manifest = json.loads(path.read_text()); manifest["embedding_model"] = "nvidia/NV-Embed-v2"
        path.write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="identity/model/corpus/serialization mismatch"):
            inspect_index(config, "hotpotqa", root=tmp_path)


def test_active_build_lock_does_not_start_second_http_builder(tmp_path):
    corpus(tmp_path)
    with service() as (state, config):
        report = inspect_index(config, "hotpotqa", root=tmp_path)
        directory = Path(report["index_dir"]); directory.mkdir(parents=True)
        with (directory / "build.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with pytest.raises(RuntimeError, match="already active"):
                ensure_index(config, "hotpotqa", tmp_path / "run", root=tmp_path)
        assert state["requests"] == []


def test_prepare_uses_new_vectors_in_original_order_and_never_opens_gold(tmp_path):
    corpus(tmp_path)
    configured = []
    class Document:
        def __init__(self, doc_id, title, text):
            self.doc_id, self.title, self.text = doc_id, title, text
        @property
        def passage(self):
            return self.title + "\n" + self.text
    validator = lambda plan: plan
    pipeline = SimpleNamespace(e=SimpleNamespace(Document=Document, EMBED_LOCK=threading.Lock(),
               CONFIG={}, native=SimpleNamespace(configure=lambda value: configured.append(value))),
               repair=SimpleNamespace(validate_plan=validator))
    tokenizer = object()
    with service() as (_, config):
        with pytest.raises(ValueError, match="must be built"):
            prepare_resources(config, "hotpotqa", pipeline, tokenizer, root=tmp_path)
        built = ensure_index(config, "hotpotqa", tmp_path / "run", root=tmp_path)
        questions, (docs, ids, vectors, index, actual_tokenizer), hashes = prepare_resources(
            config, "hotpotqa", pipeline, tokenizer, root=tmp_path)
        assert questions == [{"id": "q", "question": "A factual question?"}]
        assert ids == ["d0", "d1", "d2"] and list(docs) == ids
        assert vectors.shape == (3, 3) and list(index.vectors) == ids
        assert actual_tokenizer is tokenizer and pipeline.e.validate_plan is validator
        assert configured[-1]["embedding_model"] == config["embedding_model"]
        assert built["manifest_path"] in hashes and built["vectors_path"] in hashes
        assert not any("evaluation_only" in path for path in hashes)


def test_configured_embedding_batch_size_is_used_and_recorded(tmp_path):
    corpus(tmp_path, 5)
    with service() as (state, config):
        result = ensure_index({**config, "embedding_batch_size": 2}, "hotpotqa", tmp_path / "run", root=tmp_path)
        assert [len(p["input"]) for p in state["requests"]] == [2, 2, 1]
        assert result["manifest"]["batch_size"] == 2


@pytest.mark.parametrize("value", [0, -1, True, "32", 1.5])
def test_invalid_embedding_batch_size_fails_before_any_model_request(tmp_path, value):
    corpus(tmp_path)
    with service() as (state, config):
        with pytest.raises(ValueError, match="embedding_batch_size must be a positive integer"):
            ensure_index({**config, "embedding_batch_size": value}, "hotpotqa", tmp_path / "run", root=tmp_path)
        assert state["requests"] == []


class ArchiveCalls:
    def __init__(self, query_vectors):
        self.query_vectors, self.requests = query_vectors, []

    def get(self, stage, url, payload):
        self.requests.append((stage, url, payload))
        return {"response": {"data": [{"embedding": self.query_vectors[int(stage[1])].tolist()}]}}


def test_derived_archive_matches_real_original_queries_pool_trace_and_stable_ties(monkeypatch):
    from dagbt.engine import legacy_modules
    e, _, _ = legacy_modules()
    monkeypatch.setattr(e, "CONFIG", {"embedding_base_url": "http://archive-fixture/v1"})
    # More than 50 documents exercises the original cutoff; tied documents
    # exercise stable sorting. Distinct query directions exercise pool union.
    rng = np.random.default_rng(1234)
    vectors = rng.normal(size=(72, 4096)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors[1] = vectors[0]
    query_vectors = np.stack([vectors[0], vectors[65], vectors[70]])
    ids = ["d" + str(i) for i in range(len(vectors))]
    question = "Who founded the club?"
    plan = {"steps": [{"question": question}, {"question": "Where was {founder} born?"},
                      {"question": "Where was {another_slot} born?"},
                      {"question": "What year did {club} open?"}]}
    original_calls, adapted_calls = ArchiveCalls(query_vectors), ArchiveCalls(query_vectors)
    original = e.archive(question, plan, ids, vectors, original_calls)
    adapted = archive_for_resources(question, plan, ids, vectors, adapted_calls, e)
    assert adapted_calls.requests == original_calls.requests
    assert adapted == original
    pool, trace = adapted
    assert [row["query"] for row in trace] == [question, "Where was the unknown entity born?",
                                               "What year did the unknown entity open?"]
    assert all(len(row["doc_ids"]) == 50 for row in trace)
    assert all(np.isfinite(row["scores"]).all() for row in trace)
    assert trace[0]["doc_ids"][:2] == ["d0", "d1"]
    assert len(pool) == len(set(pool)) > 50


def test_derived_archive_supports_actual_index_dimension(monkeypatch):
    from dagbt.engine import legacy_modules
    e, _, _ = legacy_modules()
    monkeypatch.setattr(e, "CONFIG", {"embedding_base_url": "http://archive-fixture/v1"})
    vectors = np.eye(8, dtype=np.float32)
    ids = ["d" + str(i) for i in range(8)]
    pool, trace = archive_for_resources("Q", {"steps": []}, ids, vectors,
                                        ArchiveCalls([vectors[3]]), e)
    assert pool == ["d3", "d0", "d1", "d2", "d4", "d5", "d6", "d7"]
    assert trace == [{"query": "Q", "doc_ids": pool, "scores": [1., 0., 0., 0., 0., 0., 0., 0.]}]


@pytest.mark.parametrize("query_vector", [np.zeros(8), np.ones(7), np.full(8, np.nan)])
def test_derived_archive_rejects_invalid_query_embeddings(monkeypatch, query_vector):
    from dagbt.engine import legacy_modules
    e, _, _ = legacy_modules()
    monkeypatch.setattr(e, "CONFIG", {"embedding_base_url": "http://archive-fixture/v1"})
    with pytest.raises(e.native.ServicePause, match="Invalid query embedding"):
        archive_for_resources("Q", {"steps": []}, ["d0"], np.ones((1, 8), dtype=np.float32),
                              ArchiveCalls([query_vector]), e)
