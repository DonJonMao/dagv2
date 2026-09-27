"""Full BT model-profile protocol smoke against an entirely loopback service.

This runs real spawned workers and both retrieval algorithms over all 9,811
Hotpot corpus passages. The eight-dimensional embeddings and semantic outputs
are deliberately scripted test values, never model quality evidence. Nothing
in this test contacts the real model endpoints or opens labels before scoring.
"""
from collections import Counter
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
import threading
from types import SimpleNamespace

import numpy as np

from dagbt import runner
from dagbt.resources import inspect_index
from test_system_http import FIXTURE_ANSWER, scripted_source_mapping


@contextmanager
def scripted_bt_models(question):
    requests, errors = [], []
    usage = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
    vector = [1.0] + [0.0] * 7

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, body, status=200):
            raw = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path.endswith("/models"):
                self.respond({"data": [{"id": "deepseek-v4-flash"}, {"id": "qwen3-embedding-8b"}]})
            else:
                self.respond({"status": "ok", "model": "Qwen3-Reranker-8B", "max_model_len": 8192})

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append({"path": self.path, "payload": payload})
            try:
                if self.path.endswith("/embeddings"):
                    assert payload["model"] == "qwen3-embedding-8b"
                    assert isinstance(payload["input"], list) and len(payload["input"]) <= 32
                    # Reverse index order deliberately: the real builder must
                    # reconstruct canonical corpus order from explicit indices.
                    self.respond({"model": "qwen3-embedding-8b", "data": [
                        {"index": i, "embedding": vector} for i in reversed(range(len(payload["input"])))],
                        "usage": usage})
                    return
                if self.path.endswith("/rerank"):
                    assert "model" not in payload, "Empty BT reranker model must be omitted on the wire"
                    documents = payload["documents"]
                    assert 1 <= len(documents) <= 4, "Configured service limit is four passages per HTTP call"
                    self.respond({"model": "Qwen3-Reranker-8B", "results": [
                        {"index": i, "relevance_score": min(.9, .1 + .06 * documents[i].count("[Passage "))}
                        for i in reversed(range(len(documents)))], "usage": usage})
                    return
                assert self.path == "/v1/chat/completions", "BT profile must use chat, not /completions"
                assert payload["model"] == "deepseek-v4-flash"
                assert not {"prompt", "structured_outputs", "chat_template_kwargs", "enable_thinking"}.intersection(payload)
                wire = json.dumps(payload)
                assert all(marker not in wire for marker in ("<|im_start|>", "<|im_end|>", "<|endoftext|>"))
                messages = payload["messages"]
                system = messages[0]["content"]
                if system.startswith("Decompose a multi-hop question"):
                    try:
                        planner_question = json.loads(messages[1]["content"])
                    except json.JSONDecodeError:
                        planner_question = messages[1]["content"]
                    assert isinstance(planner_question, str)
                    value = {"steps": [{"question": planner_question, "output_slot": "answer", "answer_type": "answer", "inputs": []}]}
                elif system.startswith("Solve the current evidence-grounded task"):
                    # Native node semantics/schema are unchanged, only adapted
                    # into chat messages instead of a Qwen completion prompt.
                    match = re.search(r"sources \(exactly (\d+) booleans", messages[1]["content"])
                    assert match, "Native source-panel cardinality must survive chat adaptation"
                    count = int(match.group(1))
                    assert count > 0
                    value = {"answer": FIXTURE_ANSWER, "sources": [True] + [False] * (count - 1)}
                elif system.startswith("Map raw corpus passages"):
                    data = json.loads(messages[1]["content"])
                    value = scripted_source_mapping(data)
                elif system.startswith("Resolve ONE executable subquestion"):
                    data = json.loads(messages[1]["content"])
                    assert data["evidence"]
                    value = {"status": "supported", "answer": FIXTURE_ANSWER,
                             "alternatives": [{"source_span_ids": [data["evidence"][0]["id"]], "guard_span_ids": [],
                                               "used_parent_ids": [], "applicable_scope": "scripted fixture only",
                                               "semantic_status": "supported"}],
                             "unresolved_inputs": [], "unresolved_guards": [], "refinements": []}
                elif system.startswith("Audit the compiled evidence support alternatives"):
                    value = {"conflicts": [], "unresolved_guards": []}
                elif system.startswith("You are a long-document QA reader"):
                    self.respond({"model": "deepseek-v4-flash", "choices": [
                        {"message": {"content": "Answer: " + FIXTURE_ANSWER}, "finish_reason": "stop"}], "usage": usage})
                    return
                else:
                    raise AssertionError("Unexpected production prompt: " + system[:100])
                self.respond({"model": "deepseek-v4-flash", "choices": [
                    {"message": {"content": json.dumps(value)}, "finish_reason": "stop"}], "usage": usage})
            except Exception as exc:
                errors.append(repr(exc))
                self.respond({"fixture_error": repr(exc)}, status=400)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests, errors
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_bt_profile_chat_only_and_model_specific_full_corpus_index(tmp_path, monkeypatch):
    # Never forward an operator's real credential into even this loopback fixture.
    for name in ("DAG_LLM_API_KEY", "DAG_EMBED_API_KEY", "DAG_RERANK_API_KEY", "BRIDGETREE_CHAT_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    q = runner.load_questions("hotpotqa", limit=1)[0]
    corpus = [json.loads(line) for line in (runner.ROOT / "data/hotpotqa/corpus.jsonl").read_text().splitlines()]
    passages = [d["title"] + "\n" + d["text"] if d["title"] and d["text"] else d["title"] or d["text"] for d in corpus]
    assert len(passages) == 9811
    original_index = runner.ROOT / "data/hotpotqa/index/passage_vectors.npy"
    original_index_hash = runner.file_hash(original_index)
    credentials = tmp_path / "empty_credentials.json"
    runner.save(credentials, {})
    output = tmp_path / "scripted_bt_profile_system_test"

    with scripted_bt_models(q["question"]) as (endpoint, requests, errors):
        config = runner.load(runner.ROOT / "configs/paired.example.json")
        assert config["model_profile"] == "bridgetree"
        assert config["llm_model"] == "deepseek-v4-flash"
        assert config["embedding_model"] == "qwen3-embedding-8b"
        assert config["reranker"]["model"] == ""
        config.update(llm_base_url=endpoint + "/v1", embedding_base_url=endpoint + "/v1",
                      credentials_file=str(credentials), derived_index_dir=str(tmp_path / "derived_indices"),
                      request_timeout_seconds=10, max_identical_attempts=1)
        config["reranker"]["url"] = endpoint + "/rerank"
        config["fusion"].update(ann_calls=12, set_score_calls=64)
        config["experiment"].update(question_timeout_seconds=180, worker_startup_timeout_seconds=120,
                                    fixture_note="Entirely loopback scripted BT-profile compatibility test; not accuracy")
        config_path = tmp_path / "profile_config.json"
        runner.save(config_path, config)
        assert inspect_index(config, "hotpotqa")["status"] == "needs_build"
        args = SimpleNamespace(config=str(config_path), output=str(output), datasets=["hotpotqa"],
                               arms=["original", "fusion"], limit=1, retry_failed=False, offline_preflight=False)
        runner.run_experiment(args)

        assert not errors, errors
        rows = {arm: runner.load(runner.result_path(output, "hotpotqa", arm, q["id"])) for arm in ("original", "fusion")}
        assert {arm: row["answer"]["status"] for arm, row in rows.items()} == {"original": "ok", "fusion": "ok"}, rows
        assert all(row["unit_id"] == q["id"] and row["answer"]["prediction"] == FIXTURE_ANSWER for row in rows.values())
        assert runner.load(output / "progress.json")["state"] == "complete"
        paired = json.loads((output / "hotpotqa/comparisons.jsonl").read_text().strip())
        assert paired["unit_id"] == q["id"] and set(paired["arms"]) == {"original", "fusion"}

        index = inspect_index(config, "hotpotqa")
        assert index["status"] == "ready" and index["dimensions"] == 8
        assert index["documents"] == 9811 and index["embedding_model"] == "qwen3-embedding-8b"
        assert index["manifest"]["document_instruction"] == ""
        assert np.load(index["vectors_path"], mmap_mode="r", allow_pickle=False).shape == (9811, 8)
        assert Path(index["vectors_path"]).is_relative_to(tmp_path / "derived_indices")
        assert runner.file_hash(original_index) == original_index_hash
        frozen_index = runner.load(output / "index_artifacts.json")
        assert frozen_index["hotpotqa"]["identity"] == index["identity"]
        resource_hashes = [runner.load(output / "hotpotqa" / arm / "resource_hashes.json") for arm in ("original", "fusion")]
        assert resource_hashes[0] == resource_hashes[1]
        assert resource_hashes[0][index["vectors_path"]] == index["manifest"]["vectors_sha256"]
        assert str(original_index) not in resource_hashes[0]

        # All raw document passages are embedded exactly once, shared by both
        # workers. Queries use separate calls and cannot masquerade as documents.
        known_passages = set(passages)
        embedded_passages = [text for request in requests if request["path"].endswith("/embeddings")
                             for text in request["payload"]["input"] if text in known_passages]
        assert Counter(embedded_passages) == Counter(passages)
        build_cost = runner.request_cost(output / "index_build/hotpotqa")
        assert build_cost["http_attempts"] == math.ceil(len(passages) / 32)
        assert build_cost["token_usage_complete"] and build_cost["total_tokens"] > 0
        assert build_cost["total_tokens"] == 18 * build_cost["http_attempts"]

        for row in rows.values():
            accounting = row["diagnostics"]["token_accounting"]
            assert accounting["token_count_is_estimate"] is True
            assert accounting["token_estimator_id"]
            assert row["runner"]["cost"]["http_attempts"] > 0
            assert row["runner"]["cost"]["token_usage_complete"] is True
            assert row["runner"]["cost"]["total_tokens"] == 18 * row["runner"]["cost"]["http_attempts"]
            assert row["answer"]["response_usage"] == {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
        feasibility = rows["fusion"]["diagnostics"]["reader_feasibility"]
        assert feasibility["token_count_is_estimate"] is True
        assert feasibility["token_estimator_id"]
        assert rows["fusion"]["diagnostics"]["ledger"]["used"]["set_score"] > 0
        assert rows["original"]["ranking"]["nodes"][0]["resolved"]
        chat_requests = [r for r in requests if r["path"] == "/v1/chat/completions"]
        planners = [r for r in chat_requests if r["payload"]["messages"][0]["content"].startswith("Decompose a multi-hop question")]
        assert len(planners) == 3  # Independent protocol gate, original question, fusion question.
        assert sum(q["question"] in r["payload"]["messages"][1]["content"] for r in planners) == 2
        probe = runner.load(output / "preflight.json")["endpoints"]["fusion_planner_protocol"]
        assert probe["status"] == "passed" and probe["logical_calls"] == 1
        assert probe["experiment_task"] is False
        assert not any(r["path"] == "/v1/completions" for r in requests)
        assert all(r["payload"]["model"] == "deepseek-v4-flash" for r in chat_requests)
        wire = json.dumps(requests)
        assert all(marker not in wire for marker in ("structured_outputs", "chat_template_kwargs", "<|im_start|>", "gold_groups", "evaluation_only.json"))
        summary = runner.load(output / "summary.json")["hotpotqa"]
        assert all(summary["arms"][arm]["cost_all_attempts"]["http_attempts"] > 0 for arm in rows)
        runner.save(output / "SCRIPTED_BT_PROFILE_ONLY.json", {
            "kind": "scripted_loopback_profile_protocol_not_accuracy", "actual_native_workers": True,
            "question_id": q["id"], "arms": ["original", "fusion"], "corpus_documents": len(passages),
            "embedding_dimensions": 8, "index_identity": index["identity"], "original_nv_index_unchanged": True,
            "wire_chat_only": True, "reranker_model_omitted": True, "rerank_max_batch_documents": 4,
            "token_accounting_is_estimate": True, "http_requests": len(requests),
            "offline_index_cost": build_cost, "fixture_answer": FIXTURE_ANSWER})
