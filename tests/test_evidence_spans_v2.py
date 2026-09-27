"""Exact source spans and retained PersonaMem provenance, without model calls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dagbt.bridge import BridgeSession
from dagbt.budget import Ledger
from dagbt.evidence_spans import (SourceSpanError, build_source_spans,
    document_source_metadata, independent_premise_count, normalize_source_segments)
from dagbt.resources import ProvenanceDocument, personamem_document, scope_resources, prepare_resources


def memory_record():
    user = "I dislike heat. Assistant: this is still a user's quotation."
    assistant = "Try a quiet coastal town."
    text = "User:\n" + user + "\n\nAssistant:\n" + assistant
    start = text.index(assistant)
    return {"docid": "memory-1", "title": "Conversation message indices 2–3", "text": text,
            "shared_context_id": "context-1", "source_message_indices": [2, 3],
            "roles": ["user", "assistant"],
            "time": {"observed_start": 2.0, "observed_end": 3.0, "event_start": None,
                     "event_end": None, "validity": "unknown", "time_source": "message_index"},
            "source_segments": [{"role": "user", "start": len("User:\n"),
                "end": len("User:\n") + len(user), "source_message_indices": [2]},
                {"role": "assistant", "start": start, "end": len(text), "source_message_indices": [3]}]}


def assert_partition(document, spans, limit):
    assert spans and "".join(span["text"] for span in spans) == document.passage
    assert len({span["id"] for span in spans}) == len(spans)
    position = 0
    for span in spans:
        assert span["start"] == position
        assert 0 < span["end"] - span["start"] <= limit
        assert document.passage[span["start"]:span["end"]] == span["text"]
        assert span["source_hash"] == hashlib.sha256(document.passage.encode()).hexdigest()
        position = span["end"]
    assert position == len(document.passage)


def test_full_source_partition_and_ids_are_stable_across_batch_order():
    docs = {"a": SimpleNamespace(passage="Title\n" + "很长的中文记录🙂" * 130 + ". End!\n"),
            "b": SimpleNamespace(passage="User: is a literal marker in a factual article.")}
    first = {key: build_source_spans(key, doc, 73) for key, doc in docs.items()}
    reverse = {key: build_source_spans(key, docs[key], 73) for key in reversed(docs)}
    assert first == reverse
    for key, spans in first.items():
        assert_partition(docs[key], spans, 73)
        assert all(s["source_role"] == "document" for s in spans)
    different_id = build_source_spans("foreign", docs["a"], 73)
    assert not {s["id"] for s in first["a"]} & {s["id"] for s in different_id}
    changed = build_source_spans("a", SimpleNamespace(passage=docs["a"].passage + "Changed"), 73)
    assert not {s["id"] for s in first["a"]} & {s["id"] for s in changed}


def test_forced_sentence_splits_cannot_manufacture_independent_premises():
    long = SimpleNamespace(passage="a" * 1100)
    spans = build_source_spans("d", long, 400)
    assert len(spans) == 3
    assert len({g for s in spans for g in s["premise_group_ids"]}) == 1
    assert independent_premise_count(spans) == 1
    assert independent_premise_count([*spans, spans[0]]) == 1
    two = build_source_spans("two", SimpleNamespace(passage="First claim. Second claim."), 13)
    assert len(two) == 2 and independent_premise_count(two) == 2
    # A bridge assessment sharing both sentence groups conservatively joins them.
    joined = {**two[0], "premise_group_ids": two[0]["premise_group_ids"] + two[1]["premise_group_ids"]}
    assert independent_premise_count([*two, joined]) == 1


def test_personamem_title_is_unknown_and_roles_use_shifted_authoritative_boundaries():
    record = memory_record()
    original = deepcopy(record)
    doc = personamem_document(record)
    spans = build_source_spans(doc.doc_id, doc, 23)
    assert_partition(doc, spans, 23)
    prefix = len(doc.title) + 1
    assert all(s["source_role"] == "unknown" for s in spans if s["start"] < prefix)
    for source in record["source_segments"]:
        selected = [s for s in spans if source["start"] + prefix <= s["start"] < source["end"] + prefix]
        assert "".join(s["text"] for s in selected) == record["text"][source["start"]:source["end"]]
        assert all(s["source_role"] == source["role"] for s in selected)
        assert all(s["source_message_indices"] == source["source_message_indices"] for s in selected)
    assert "Assistant:" in "".join(s["text"] for s in spans if s["source_role"] == "user")
    assert record == original
    # Frozen dataclass stores a copy; external metadata edits cannot reattribute it.
    record["source_segments"][0]["role"] = "assistant"
    assert "user" in {s["role"] for s in doc.metadata["source_segments"]}


@pytest.mark.parametrize("metadata", [
    None, [], {"source_segments": None}, {"source_segments": {}},
    {"source_segments": [{"role": "user", "start": True, "end": 1, "source_message_indices": [0]}]},
    {"source_segments": [{"role": "user", "start": 0, "end": 1000, "source_message_indices": [0]}]},
    {"source_segments": [{"role": "invented", "start": 0, "end": 1, "source_message_indices": [0]}]},
    {"source_segments": [{"role": "user", "start": 0, "end": 1, "source_message_indices": [False]}]},
    {"source_segments": [{"role": "user", "start": 0, "end": 1, "source_message_indices": [2, 1]}]},
    {"source_segments": [{"role": "user", "start": 0, "end": 1}]},
    {"source_segments": [{"role": "user", "start": 0, "end": 3, "source_message_indices": [0]},
                         {"role": "assistant", "start": 2, "end": 4, "source_message_indices": [1]}]},
    {"time": {"event_start": float("nan")}},
])
def test_malformed_provenance_is_explicit_error(metadata):
    with pytest.raises(SourceSpanError):
        build_source_spans("d", SimpleNamespace(passage="text", metadata=metadata))


def test_empty_authoritative_segments_are_unknown_and_not_document_defaults():
    doc = SimpleNamespace(passage="User: untrusted role marker", metadata={"source_segments": []})
    assert {s["source_role"] for s in build_source_spans("d", doc)} == {"unknown"}
    assert normalize_source_segments("", []) == []


@pytest.mark.parametrize("mutate", [
    lambda r: r.pop("source_segments"),
    lambda r: r.update(roles=["assistant", "user"]),
    lambda r: r.update(source_message_indices=[2]),
    lambda r: r.update(time="a made-up date"),
    lambda r: r.update(roles=[{}]),
])
def test_personamem_rejects_inconsistent_source_metadata(mutate):
    record = memory_record()
    mutate(record)
    with pytest.raises(SourceSpanError):
        personamem_document(record)


def test_scoping_and_bridge_preserve_provenance_without_new_model_calls():
    doc = personamem_document(memory_record())
    other = ProvenanceDocument("other", "Other", "Unrelated document")
    docs = {doc.doc_id: doc, other.doc_id: other}
    vectors = np.eye(2, dtype=np.float32)
    resources = (docs, list(docs), vectors, SimpleNamespace(lock=None), None)
    scoped = scope_resources(resources, [doc.doc_id])
    assert scoped[0][doc.doc_id] is doc
    config = {"embedding_base_url": "http://unused/v1", "embedding_model": "unused",
              "fusion": {"proxy_mode": "none"}}
    session = BridgeSession("Question", scoped[0], scoped[1], scoped[2], None,
                            SimpleNamespace(), config, Ledger({"ann": 5, "set_score": 0}))
    memory = session.records[doc.doc_id]
    assert memory.text == doc.passage
    assert memory.metadata == document_source_metadata(doc)
    assert {s["role"] for s in memory.metadata["source_segments"]} == {"unknown", "user", "assistant"}
    assert memory.metadata["time"]["time_source"] == "message_index"
    assert memory.timestamp == 3.0  # Actual observation index, never scope-local row 0 or a date.
    assert set(session.records) == {doc.doc_id}


def test_prepare_resources_keeps_personamem_reader_text_and_provenance(tmp_path, monkeypatch):
    record = memory_record()
    directory = tmp_path / "data/personamem"
    directory.mkdir(parents=True)
    (directory / "corpus.jsonl").write_text(json.dumps(record) + "\n")
    canonical = [{"doc_id": record["docid"], "title": record["title"], "text": record["text"]}]
    (directory / "corpus.json").write_text(json.dumps(canonical))
    (directory / "questions.jsonl").write_text(json.dumps({"id": "q", "question": "Which option?"}) + "\n")
    for name in ("manifest.json", "scopes.json"):
        (directory / name).write_text("{}")
    # A labels read would fail decoding; resources may only name its path.
    (directory / "evaluation_only.json").write_bytes(b"\xff")
    vectors_path = directory / "test-vectors.npy"
    np.save(vectors_path, np.ones((1, 1), dtype=np.float32))
    report = {"status": "ready", "vectors_path": str(vectors_path), "dimensions": 1,
              "manifest_path": str(directory / "manifest.json")}
    monkeypatch.setattr("dagbt.resources.inspect_index", lambda *_: report)
    monkeypatch.setattr("dagbt.personamem.validate_dataset", lambda *_: None)
    e = SimpleNamespace(CONFIG={}, EMBED_LOCK=None, native=SimpleNamespace(configure=lambda _: None))
    pipeline = SimpleNamespace(e=e, repair=SimpleNamespace(validate_plan=lambda x: x))
    questions, resources, hashes = prepare_resources({"model_profile": "bridgetree"}, "personamem", pipeline, None, root=tmp_path)
    doc = resources[0][record["docid"]]
    assert questions == [{"id": "q", "question": "Which option?"}]
    assert doc.title == record["title"] and doc.text == record["text"]
    assert doc.passage == record["title"] + "\n" + record["text"]
    assert build_source_spans(doc.doc_id, doc)[-1]["source_role"] == "assistant"
    assert str(directory / "evaluation_only.json") not in hashes


def test_all_packaged_personamem_memories_have_lossless_role_safe_source_spans():
    path = Path(__file__).resolve().parents[1] / "data/personamem/corpus.jsonl"
    count = 0
    for line in path.read_text().splitlines():
        record = json.loads(line)
        doc = personamem_document(record)
        spans = build_source_spans(doc.doc_id, doc)
        assert_partition(doc, spans, 400)
        for span in spans:
            assert set(span["source_message_indices"]) <= set(record["source_message_indices"])
        count += 1
    assert count == 3187
