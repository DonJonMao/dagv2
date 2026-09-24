"""PersonaMem protocol boundaries: labels, persona isolation, and cutoffs."""
import csv
import json
from pathlib import Path

import pytest

from dagbt import personamem as pm
from vendor.bridgetree.personamem import messages_to_memories


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    contexts = {
        "alice": [
            {"role": "system", "content": "You are talking to Alice."},
            {"role": "user", "content": "I prefer green tea."},
            {"role": "assistant", "content": "Here is the future response."},
            {"role": "user", "content": "I switched to coffee in the future."},
        ],
        "bob": [
            {"role": "system", "content": "You are talking to Bob."},
            {"role": "user", "content": "Bob likes cocoa; private other-person evidence."},
        ],
    }
    source_context = raw / "shared_contexts_32k.jsonl"
    source_context.write_text(json.dumps(contexts) + "\n")
    fields = ["persona_id", "question_id", "question_type", "topic", "user_question_or_message",
              "correct_answer", "all_options", "shared_context_id", "end_index_in_shared_context"]
    source_questions = raw / "questions_32k.csv"
    with source_questions.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for question_id, persona_id, context_id, end_index in (
            ("early", "p_alice", "alice", 2),
            ("later", "p_alice", "alice", 3),
            ("other", "p_bob", "bob", 2),
        ):
            writer.writerow(dict(persona_id=persona_id, question_id=question_id,
                                 question_type="preference", topic="drink",
                                 user_question_or_message="Which drink suits me?",
                                 correct_answer="(a)", all_options=str(["(a) tea", "(b) coffee", "(c) cocoa", "(d) water"]),
                                 shared_context_id=context_id, end_index_in_shared_context=end_index))
    monkeypatch.setattr(pm, "PERSONAMEM_SOURCE_SHA256", {"32k": {
        path.name: pm.file_sha256(path) for path in (source_questions, source_context)
    }})
    pm.prepare_dataset(raw, tmp_path)
    return tmp_path, raw, contexts


def refresh_public_hash(root, filename):
    directory = root / "data" / "personamem"
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["public_sha256"][filename] = pm.file_sha256(directory / filename)
    manifest_path.write_text(json.dumps(manifest))


def test_packaged_dataset_is_entire_pinned_32k():
    manifest = pm.validate_dataset(ROOT, include_labels=True)
    assert {key: manifest[key] for key in ("questions", "personas", "shared_contexts", "scopes", "documents")} == {
        "questions": 589, "personas": 20, "shared_contexts": 37, "scopes": 222, "documents": 3187,
    }
    assert manifest["source_sha256"] == pm.PERSONAMEM_SOURCE_SHA256["32k"]
    assert len(pm.load_questions(ROOT)) == 589
    assert len(pm.load_questions(ROOT, limit=2)) == 2


def test_slice_before_pairing_and_other_persona_exclusion(prepared):
    root, _, contexts = prepared
    questions = {q["id"]: q for q in pm.load_questions(root)}
    docs = {d["docid"]: d for d in pm._jsonl(root / "data/personamem/corpus.jsonl")}
    for question in questions.values():
        visible = [docs[d] for d in pm.scope_doc_ids(question, root)]
        expected = messages_to_memories(contexts[question["shared_context_id"]][:question["end_index"]],
                                        question["shared_context_id"])
        assert [d["text"] for d in visible] == [memory.text for memory in expected]
        assert all(d["title"] == pm.memory_title(d["source_message_indices"]) for d in visible)
        assert all(max(d["source_message_indices"]) < question["end_index"] for d in visible)
        assert all(d["shared_context_id"] == question["shared_context_id"] for d in visible)
    early = "\n".join(docs[d]["text"] for d in pm.scope_doc_ids(questions["early"], root))
    later = "\n".join(docs[d]["text"] for d in pm.scope_doc_ids(questions["later"], root))
    assert "future response" not in early and "future response" in later
    assert "private other-person evidence" not in early
    assert "switched to coffee" not in early + later


def test_public_memory_titles_preserve_observation_order(prepared):
    root, _, _ = prepared
    docs = pm._jsonl(root / "data/personamem/corpus.jsonl")
    paired = next(d for d in docs if d["shared_context_id"] == "alice" and d["source_message_indices"] == [1, 2])
    assert paired["title"] == (
        "Conversation message indices 1–2 (zero-based chronological observation order; not calendar time)"
    )
    assert paired["text"] == "User:\nI prefer green tea.\n\nAssistant:\nHere is the future response."
    assert pm.memory_title([0]) == (
        "Conversation message indices 0–0 (zero-based chronological observation order; not calendar time)"
    )


def test_memory_title_cannot_claim_a_different_source_range(prepared):
    root, _, _ = prepared
    target = root / "data/personamem/corpus.jsonl"
    rows = pm._jsonl(target)
    rows[0]["title"] = pm.memory_title([100, 101])
    target.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    refresh_public_hash(root, target.name)
    with pytest.raises(ValueError, match="source-order title"):
        pm.validate_dataset(root)


def test_public_loaders_do_not_read_or_require_gold(prepared):
    root, _, _ = prepared
    (root / "data/personamem/evaluation_only.json").unlink()
    assert pm.validate_dataset(root)["questions"] == 3
    questions = pm.load_questions(root)
    assert all(set(q) == set(pm.QUESTION_FIELDS) for q in questions)
    assert all("correct_answer" not in q and "answers" not in q for q in questions)
    assert all(len(q["options"]) == 4 for q in questions)
    assert all(q["question"].endswith(pm.ANSWER_INSTRUCTION) for q in questions)
    with pytest.raises(FileNotFoundError):
        pm.validate_dataset(root, include_labels=True)


def test_public_loaders_reject_injected_labels(prepared):
    root, _, _ = prepared
    target = root / "data/personamem/questions.jsonl"
    rows = pm._jsonl(target)
    rows[0]["correct_answer"] = "(a)"
    target.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="label-free whitelist"):
        pm.load_questions(root)


@pytest.mark.parametrize("contamination", ["future", "other_persona"])
def test_scope_semantics_reject_contamination_even_with_updated_artifact_hash(prepared, contamination):
    root, _, _ = prepared
    questions = {q["id"]: q for q in pm.load_questions(root)}
    scopes = pm.load_scopes(root)
    early_scope = scopes[questions["early"]["scope_id"]]
    attacker = questions["later" if contamination == "future" else "other"]
    foreign = scopes[attacker["scope_id"]][-1]
    assert foreign not in early_scope
    early_scope.append(foreign)
    target = root / "data/personamem/scopes.json"
    target.write_text(json.dumps(scopes))
    refresh_public_hash(root, target.name)
    with pytest.raises(ValueError, match="future message|crosses a shared context"):
        pm.validate_dataset(root)


def test_source_checksum_is_checked_before_import(prepared):
    root, raw, _ = prepared
    with (raw / "questions_32k.csv").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="pinned official 32k checksums"):
        pm.prepare_dataset(raw, root)


def test_public_and_evaluation_hashes_are_separate(prepared):
    root, _, _ = prepared
    gold = root / "data/personamem/evaluation_only.json"
    labels = json.loads(gold.read_text())
    labels[0]["correct_answer"] = "(b)"
    gold.write_text(json.dumps(labels))
    pm.validate_dataset(root)
    with pytest.raises(ValueError, match="evaluation labels checksum mismatch"):
        pm.validate_dataset(root, include_labels=True)


def test_public_checksum_change_fails_closed(prepared):
    root, _, _ = prepared
    target = root / "data/personamem/corpus.jsonl"
    target.write_text(target.read_text() + "\n")
    with pytest.raises(ValueError, match="public artifact checksum mismatch"):
        pm.validate_dataset(root)


@pytest.mark.parametrize("text,choice", [
    ("a", "a"), (" (A)\n", "a"), ("Answer: (b)", "b"), ("The answer is C.", "c"),
    ("D.", "d"), ("**(b)**", "b"), ("[C]", "c"), ("Final answer: d", "d"),
    ("(a) or (c)", None), ("A or B", None), ("Answer: A, B", None),
    ("I think (a) because it mentions tea.", None), ("(e)", None),
    ("tea", None), ("", None), (None, None),
])
def test_strict_choice_parser(text, choice):
    assert pm.parse_choice(text, ["tea", "coffee", "cocoa", "water"]) == choice


@pytest.mark.parametrize("limit", [True, -1, 1.0, "2"])
def test_question_limit_requires_integer(limit):
    with pytest.raises(ValueError, match="limit"):
        pm.load_questions(ROOT, limit=limit)


def test_scope_lookup_rejects_question_scope_substitution(prepared):
    root, _, _ = prepared
    questions = pm.load_questions(root)
    substituted = {**questions[0], "scope_id": questions[-1]["scope_id"]}
    with pytest.raises(ValueError, match="scope identity mismatch"):
        pm.scope_doc_ids(substituted, root)
