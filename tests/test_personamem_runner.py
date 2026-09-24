"""PersonaMem integration checks; every model response is a loopback fixture.

The spawned-worker smoke exercises real original/fusion retrieval over the
packaged memory corpus. Its scripted answers do not measure model quality.
"""
from collections import Counter
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import urllib.error
import urllib.request

import numpy as np
import pytest

from dagbt import runner
from dagbt.resources import inspect_index, scope_resources
import test_bt_profile_system as scripted


@contextmanager
def _rank_future_first_models(passages, future_passages):
    """Make a stale native search matrix expose invisible future memories.

    All query vectors rank the later cutoff's exclusive memories first. A
    correct earlier scope can never select them, even while reusing a worker.
    Non-embedding protocols use the established strict scripted BT service.
    """
    known, preferred = set(passages), set(future_passages)
    usage = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
    with scripted.scripted_bt_models("Which answer option matches the user's recorded preferences?") as (upstream, requests, errors):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def respond(self, raw, status=200):
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def forward(self, raw=None):
                request = urllib.request.Request(upstream + self.path, data=raw,
                                                 headers={"Content-Type": "application/json"})
                try:
                    with opener.open(request, timeout=10) as response:
                        self.respond(response.read(), response.status)
                except urllib.error.HTTPError as exc:
                    self.respond(exc.read(), exc.code)

            def do_GET(self):
                self.forward()

            def do_POST(self):
                raw = self.rfile.read(int(self.headers["Content-Length"]))
                if not self.path.endswith("/embeddings"):
                    self.forward(raw)
                    return
                payload = json.loads(raw)
                requests.append({"path": self.path, "payload": payload})
                try:
                    assert payload["model"] == "qwen3-embedding-8b"
                    assert isinstance(payload["input"], list) and 1 <= len(payload["input"]) <= 32
                    data = []
                    for index, text in reversed(list(enumerate(payload["input"]))):
                        future_or_query = text in preferred or text not in known
                        vector = [1.0, 0.0] if future_or_query else [0.0, 1.0]
                        data.append({"index": index, "embedding": vector + [0.0] * 6})
                    self.respond(json.dumps({"model": "qwen3-embedding-8b", "data": data, "usage": usage}).encode())
                except Exception as exc:
                    errors.append(repr(exc))
                    self.respond(json.dumps({"fixture_error": repr(exc)}).encode(), status=400)
        service = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=service.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{service.server_port}", requests, errors
        finally:
            service.shutdown()
            service.server_close()
            thread.join(timeout=3)


def _question(unit, persona="p0"):
    return {"id": unit, "question": "Which choice fits this user?\nOptions:\n(a) Tea\n(b) Coffee",
            "user_question": "Which choice fits this user?", "options": ["(a) Tea", "(b) Coffee"],
            "persona_id": persona, "scope_id": persona + ":1", "shared_context_id": persona,
            "end_index": 1, "question_type": "preference", "topic": "drink"}


def _row(unit, prediction):
    return {"unit_id": unit, "answer": {"status": "ok", "prediction": prediction},
            "ranking": {}, "budgets": {str(k): {"selected_doc_ids": ["memory"]} for k in (5, 10, 20)},
            "diagnostics": {"candidate_doc_ids": ["memory"]}}


def test_scope_resources_isolates_cutoffs_personas_and_native_cached_matrix():
    ids = ["p0:before", "p0:future", "p1:private"]
    vectors = np.eye(3, dtype=np.float32)
    docs = {name: SimpleNamespace(doc_id=name, passage=name) for name in ids}
    tokenizer = object()
    index = SimpleNamespace(vectors=dict(zip(ids, vectors)), lock=threading.Lock(),
                            _v2_all_ids=ids, _v2_matrix=vectors)
    full = (docs, ids, vectors, index, tokenizer)
    later = scope_resources(full, ids[:2])
    # Simulate native controller's persistent cache from the preceding task.
    later[3]._v2_all_ids = later[1]
    later[3]._v2_matrix = later[2]
    earlier = scope_resources(full, [ids[0]])
    other_persona = scope_resources(full, [ids[2]])
    for visible, scoped in ((ids[:2], later), ([ids[0]], earlier), ([ids[2]], other_persona)):
        scoped_docs, scoped_ids, scoped_vectors, scoped_index, scoped_tokenizer = scoped
        assert scoped_ids == visible
        assert set(scoped_docs) == set(scoped_index.vectors) == set(visible)
        np.testing.assert_array_equal(scoped_vectors, vectors[[ids.index(d) for d in visible]])
        assert scoped_tokenizer is tokenizer
        assert scoped_index is not index and scoped_index.lock is index.lock
    assert earlier[3] is not later[3]
    assert not hasattr(earlier[3], "_v2_all_ids")
    assert not hasattr(earlier[3], "_v2_matrix")
    assert set(earlier[0]).isdisjoint({"p0:future", "p1:private"})
    assert full[1] == ids and full[3]._v2_all_ids == ids


@pytest.mark.parametrize("visible", [[], ["known", "known"], ["unknown"]])
def test_scope_resources_rejects_empty_duplicate_or_unknown_visibility(visible):
    resources = ({"known": object()}, ["known"], np.ones((1, 2), dtype=np.float32),
                 SimpleNamespace(lock=threading.Lock()), object())
    with pytest.raises(ValueError, match="scope"):
        scope_resources(resources, visible)


def test_personamem_scores_choices_and_persona_macro_without_fabricated_recall(tmp_path):
    qs = [_question("a"), _question("b"), _question("c", "p1")]
    predictions = {"original": ["(a)", "(b)", "Answer: (b)"],
                   "fusion": ["Option A", "(z)", "(a)"]}
    for arm, values in predictions.items():
        for q, prediction in zip(qs, values):
            runner.save(runner.result_path(tmp_path, "personamem", arm, q["id"]), _row(q["id"], prediction))
    labels = [{"id": q["id"], "correct_answer": "(a)"} for q in qs]
    summary = runner.score_all(tmp_path, {"personamem": qs}, ["original", "fusion"],
                               label_loader=lambda _: labels)["personamem"]
    original, fusion = summary["arms"]["original"], summary["arms"]["fusion"]
    assert original["all_task_metrics_percent"] == {"accuracy": pytest.approx(100 / 3)}
    assert fusion["all_task_metrics_percent"] == {"accuracy": pytest.approx(200 / 3)}
    assert original["persona_macro_accuracy_percent"] == 25
    assert fusion["persona_macro_accuracy_percent"] == 75
    assert original["failure_rate"] == 0
    assert fusion["failure_rate"] == pytest.approx(1 / 3)
    assert fusion["answer_status_counts"]["invalid_choice"] == 1
    assert summary["common_success_n"] == 2
    assert summary["paired"]["fusion"]["all_task_delta_percentage_points"] == {"accuracy": pytest.approx(100 / 3)}
    for arm in predictions:
        scored = runner.load(tmp_path / "personamem" / arm / "scores.json")
        for row in scored:
            assert not {"f1", "em", "r@5", "r@10", "r@20", "all@5", "all@10", "all@20"}.intersection(row)
            assert all(value is None for key, value in row["modules"].items() if key.startswith("gold_"))
            assert row["modules"]["candidate_count"] == 1
    fields = (tmp_path / "personamem/comparisons.csv").read_text().splitlines()[0]
    assert ".accuracy" in fields and "r@" not in fields and ".f1" not in fields


def test_personamem_labels_stay_unread_until_every_dataset_and_arm_terminal(tmp_path):
    qs = [_question("one")]
    scopes = {"personamem": qs, "hotpotqa": [{"id": "other", "question": "Other?"}]}
    for arm in ("original", "fusion"):
        runner.save(runner.result_path(tmp_path, "personamem", arm, "one"), _row("one", "(a)"))
    seen = []
    with pytest.raises(ValueError, match="labels remain unread"):
        runner.score_all(tmp_path, scopes, ["original", "fusion"], label_loader=lambda dataset: seen.append(dataset))
    assert seen == []


def test_personamem_is_enabled_by_default_in_deployment_entrypoint():
    assert "personamem" in runner.DATASETS
    source = (runner.ROOT / "scripts/run_paired.sh").read_text()
    assert "dagbt.runner" in source


def test_personamem_reader_adapter_keeps_full_invalid_responses_and_module_isolation(monkeypatch):
    def generic_messages(**_):
        return [{"role": "system", "content": "You are a long-document QA reader."},
                {"role": "user", "content": "Question and options.\n"
                 "Answer with the shortest exact phrase supported by the context passages.\nOld QA instructions."}]
    def generic_parse(text):
        return str(text).splitlines()[0]
    untouched = SimpleNamespace(reader_messages=generic_messages, parse_answer=generic_parse)
    adapted = SimpleNamespace(reader_messages=generic_messages, parse_answer=generic_parse)
    prior_module = sys.modules.get("reader")
    with monkeypatch.context() as local:
        local.setitem(sys.modules, "reader", adapted)
        runner.configure_personamem_reader()
        installed_messages, installed_parser = adapted.reader_messages, adapted.parse_answer
        runner.configure_personamem_reader()
        assert adapted.reader_messages is installed_messages and adapted.parse_answer is installed_parser
        messages = adapted.reader_messages()
        assert messages[0]["content"].startswith("You are a long-document QA reader")
        assert "personal memory" in messages[-1]["content"]
        assert "shortest exact phrase" not in messages[-1]["content"]
        assert "Old QA instructions" not in messages[-1]["content"]
        assert adapted.parse_answer("Answer: (A)") == "(a)"
        assert adapted.parse_answer("d") == "(d)"
        for invalid in ("(a)\n(c)", "Answer: (a)\nExplanation with another option (c).",
                        "Answer: (a)\nThis is my explanation.", "(z)"):
            assert adapted.parse_answer(invalid) == invalid
        assert untouched.reader_messages is generic_messages
        assert untouched.parse_answer is generic_parse
        assert "shortest exact phrase" in untouched.reader_messages()[-1]["content"]
        assert untouched.parse_answer("(a)\n(c)") == "(a)"
    assert sys.modules.get("reader") is prior_module


def test_personamem_spawned_pair_shares_index_and_cannot_retrieve_future_memories(tmp_path, monkeypatch):
    from dagbt import personamem

    for name in ("DAG_LLM_API_KEY", "DAG_EMBED_API_KEY", "DAG_RERANK_API_KEY", "BRIDGETREE_CHAT_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(scripted, "FIXTURE_ANSWER", "(a)")
    all_questions = runner.load_questions("personamem")
    first = all_questions[0]
    earlier = next(q for q in all_questions if q["shared_context_id"] == first["shared_context_id"]
                   and q["end_index"] < first["end_index"])
    qs = [first, earlier]  # Deliberately shrink visibility in the same live worker.
    original_load_questions = runner.load_questions
    def selected_questions(dataset, limit=None):
        return qs[:limit] if dataset == "personamem" else original_load_questions(dataset, limit)
    monkeypatch.setattr(runner, "load_questions", selected_questions)
    scopes = personamem.load_scopes(root=runner.ROOT)
    assert qs[0]["scope_id"] != qs[1]["scope_id"]
    corpus = [json.loads(line) for line in (runner.ROOT / "data/personamem/corpus.jsonl").read_text().splitlines()]
    all_ids = {record["docid"] for record in corpus}
    passages = [record["title"] + "\n" + record["text"] if record["title"] and record["text"]
                else record["title"] or record["text"] for record in corpus]
    assert set(scopes[qs[1]["scope_id"]]) < all_ids
    future_ids = set(scopes[qs[0]["scope_id"]]) - set(scopes[qs[1]["scope_id"]])
    assert future_ids
    future_passages = [passage for record, passage in zip(corpus, passages) if record["docid"] in future_ids]
    credentials = tmp_path / "empty_credentials.json"
    runner.save(credentials, {})
    output = tmp_path / "scripted_personamem_pair"

    with _rank_future_first_models(passages, future_passages) as (endpoint, requests, errors):
        config = runner.load(runner.ROOT / "configs/paired.example.json")
        config.update(llm_base_url=endpoint + "/v1", embedding_base_url=endpoint + "/v1",
                      credentials_file=str(credentials), derived_index_dir=str(tmp_path / "derived_indices"),
                      request_timeout_seconds=10, max_identical_attempts=1)
        config["reranker"]["url"] = endpoint + "/rerank"
        config["fusion"].update(ann_calls=12, set_score_calls=64)
        config["experiment"].update(question_timeout_seconds=180, worker_startup_timeout_seconds=120,
                                    fixture_note="Scripted loopback PersonaMem scope compatibility test; not accuracy")
        config_path = tmp_path / "profile_config.json"
        runner.save(config_path, config)
        args = SimpleNamespace(config=str(config_path), output=str(output), datasets=["personamem"],
                               arms=["original", "fusion"], limit=2, retry_failed=False, offline_preflight=False)
        runner.run_experiment(args)
        assert not errors, errors
        rows = {(arm, q["id"]): runner.load(runner.result_path(output, "personamem", arm, q["id"]))
                for arm in ("original", "fusion") for q in qs}
        assert all(row["answer"]["status"] == "ok" for row in rows.values()), {
            str(key): row["answer"] for key, row in rows.items()}
        # Ensure this fixture would actually hit the invisible documents if
        # the earlier task reused the preceding task's native matrix cache.
        assert set(rows["original", first["id"]]["diagnostics"]["candidate_doc_ids"]) & future_ids
        for q in qs:
            visible = set(scopes[q["scope_id"]])
            for arm in ("original", "fusion"):
                row = rows[arm, q["id"]]
                assert row["answer"]["prediction"] == "(a)"
                candidates = set(row["diagnostics"]["candidate_doc_ids"])
                assert candidates and candidates <= visible
                for selection in row["budgets"].values():
                    assert set(selection["selected_doc_ids"]) <= visible
                for node in row.get("ranking", {}).get("nodes", []):
                    assert set(node.get("source_doc_ids", [])) <= visible
                assert row["runner"]["cost"]["http_attempts"] > 0
        assert runner.load(output / "progress.json")["state"] == "complete"
        index = inspect_index(config, "personamem")
        assert index["status"] == "ready" and index["dimensions"] == 8
        assert index["documents"] == len(corpus)
        assert Path(index["vectors_path"]).is_relative_to(tmp_path / "derived_indices")
        known_passages = set(passages)
        embedded = [text for request in requests if request["path"].endswith("/embeddings")
                    for text in request["payload"]["input"] if text in known_passages]
        assert Counter(embedded) == Counter(passages)
        assert runner.request_cost(output / "index_build/personamem")["http_attempts"] == math.ceil(len(corpus) / 32)
        resource_hashes = [runner.load(output / "personamem" / arm / "resource_hashes.json")
                           for arm in ("original", "fusion")]
        assert resource_hashes[0] == resource_hashes[1]
        assert str(runner.ROOT / "data/personamem/scopes.json") in resource_hashes[0]
        readers = [r for r in requests if r["path"] == "/v1/chat/completions"
                   and r["payload"]["messages"][0]["content"].startswith("You are a long-document QA reader")]
        assert len(readers) == 4
        assert all("(a)" in json.dumps(r["payload"]["messages"]) for r in readers)
        wire = json.dumps(requests)
        assert all(marker not in wire for marker in ("gold_groups", "correct_answer", "evaluation_only.json"))
        summary = runner.load(output / "summary.json")["personamem"]
        assert all(set(summary["arms"][arm]["all_task_metrics_percent"]) == {"accuracy"}
                   for arm in ("original", "fusion"))
        runner.save(output / "SCRIPTED_PERSONAMEM_ONLY.json", {
            "kind": "scripted_loopback_personamem_compatibility_not_accuracy", "actual_native_workers": True,
            "question_ids": [q["id"] for q in qs], "scope_ids": [q["scope_id"] for q in qs],
            "exclusive_cutoffs": [q["end_index"] for q in qs], "arms": ["original", "fusion"],
            "corpus_documents": len(corpus), "embedding_dimensions": 8,
            "index_identity": index["identity"], "fixture_answer": "(a)",
            "all_retrieved_documents_within_question_scope": True, "http_requests": len(requests)})
