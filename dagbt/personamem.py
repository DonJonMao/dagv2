"""Pinned PersonaMem 32k data with question-local, cutoff-safe memories.

The corpus is a deduplicated storage pool, never a retrieval scope. Every
question must be served exclusively from ``scopes[question['scope_id']]``.
Gold labels are written separately and are opened only by an explicit
``validate_dataset(..., include_labels=True)`` evaluation operation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

from vendor.bridgetree.metrics import extract_option_label
from vendor.bridgetree.personamem import (
    PERSONAMEM_REPO,
    PERSONAMEM_REVISION,
    PERSONAMEM_SOURCE_SHA256,
    file_sha256,
    load_shared_contexts,
    messages_to_memories,
    parse_options,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "dagbt_personamem_scoped_v1"
PUBLIC_FILES = ("questions.jsonl", "questions.json", "corpus.jsonl", "corpus.json", "scopes.json")
QUESTION_FIELDS = (
    "id", "question_id", "question", "user_question", "options", "persona_id",
    "shared_context_id", "end_index", "scope_id", "question_type", "topic",
)
ANSWER_INSTRUCTION = "Return exactly one option label, (a), (b), (c), or (d), and no explanation."
MEMORY_TITLE_PROTOCOL = "zero_based_message_index_range_v1;observation_order_not_calendar_time"


def _hash(value):
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _directory(root):
    return Path(root) / "data" / "personamem"


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _scope_id(context_id, end_index):
    return "personamem:scope:" + _hash([context_id, end_index])


def memory_title(source_message_indices):
    """Expose source order without interpreting observation indices as dates."""
    if not isinstance(source_message_indices, list) or not source_message_indices or any(
        type(index) is not int or index < 0 for index in source_message_indices
    ):
        raise ValueError("Memory title requires nonnegative source message indices")
    return (
        f"Conversation message indices {min(source_message_indices)}–{max(source_message_indices)} "
        "(zero-based chronological observation order; not calendar time)"
    )


def format_question(user_question, options):
    """Expose the same current request and all four public answers to both arms."""
    if not isinstance(user_question, str) or not user_question.strip():
        raise ValueError("PersonaMem requires a nonempty user question")
    if not isinstance(options, list) or len(options) != 4 or any(
        not isinstance(option, str) or not option.strip() for option in options
    ):
        raise ValueError("PersonaMem 32k requires exactly four nonempty options")
    choices = "\n".join(f"({chr(97 + index)}) {option}" for index, option in enumerate(options))
    return (
        "Use the user's recorded conversation history to select the best answer to the current request.\n\n"
        f"Current request:\n{user_question}\n\nCandidate answers:\n{choices}\n\n{ANSWER_INSTRUCTION}"
    )


def parse_choice(text, options):
    """Return a single lowercase option letter, or None for an invalid answer.

    The upstream label extractor supplies the canonical form, after a strict
    whole-response check. Unlike upstream's first-match metric, explanations,
    multiple labels, and labels outside the supplied option range are invalid.
    Accepted examples: a, (A), A., Answer: (b), The answer is C.
    """
    if not isinstance(text, str) or not isinstance(options, (list, tuple)) or not 1 <= len(options) <= 26:
        return None
    value = text.strip()
    if value.startswith("**") and value.endswith("**"):
        value = value[2:-2].strip()
    match = re.fullmatch(
        r"(?:(?:the\s+)?(?:correct\s+|final\s+)?answer\s*(?:is\s*|:\s*)|option\s+)?"
        r"(?:\(([a-z])\)|\[([a-z])\]|([a-z]))\s*[.):]?",
        value, flags=re.IGNORECASE,
    )
    if not match:
        return None
    letter = next(group for group in match.groups() if group is not None).lower()
    canonical = extract_option_label(f"({letter})")
    return canonical[1] if canonical and 0 <= ord(letter) - 97 < len(options) else None


def _normalize_options(raw):
    options = parse_options(raw)
    if len(options) != 4:
        raise ValueError("Pinned PersonaMem 32k rows require four options")
    normalized = []
    for index, option in enumerate(options):
        match = re.fullmatch(r"\s*\(([a-dA-D])\)\s*(.+)", option, flags=re.DOTALL)
        if not match or match.group(1).lower() != chr(97 + index):
            raise ValueError("PersonaMem options must be ordered (a) through (d)")
        normalized.append(match.group(2).strip())
    return normalized


def load_questions(root=ROOT, limit=None):
    """Read public questions only; never access evaluation labels."""
    if limit is not None and (type(limit) is not int or limit < 0):
        raise ValueError("limit must be a nonnegative integer or None")
    records = _jsonl(_directory(root) / "questions.jsonl")
    result, seen = [], set()
    for record in records:
        if not isinstance(record, dict) or set(record) != set(QUESTION_FIELDS):
            raise ValueError("PersonaMem public question fields differ from the label-free whitelist")
        if any(not isinstance(record[key], str) or not record[key] for key in QUESTION_FIELDS if key not in {"end_index", "options"}):
            raise ValueError("PersonaMem public question fields must be nonempty strings")
        if record["id"] != record["question_id"] or record["id"] in seen:
            raise ValueError("PersonaMem question IDs must be unique and equal to source question_id")
        if type(record["end_index"]) is not int or record["end_index"] < 1:
            raise ValueError("PersonaMem cutoff must be a positive exclusive message index")
        if record["scope_id"] != _scope_id(record["shared_context_id"], record["end_index"]):
            raise ValueError("PersonaMem question scope identity mismatch")
        if record["question"] != format_question(record["user_question"], record["options"]):
            raise ValueError("PersonaMem question must include the exact current request and all public options")
        seen.add(record["id"])
        result.append({key: record[key] for key in QUESTION_FIELDS})
    if not result:
        raise ValueError("PersonaMem questions cannot be empty")
    return result if limit is None else result[:limit]


def load_scopes(root=ROOT):
    scopes = _json(_directory(root) / "scopes.json")
    if not isinstance(scopes, dict) or not scopes:
        raise ValueError("PersonaMem requires nonempty question-local scopes")
    for scope_id, ids in scopes.items():
        if not isinstance(scope_id, str) or not scope_id or not isinstance(ids, list) or not ids:
            raise ValueError("Invalid PersonaMem scope")
        if any(not isinstance(doc_id, str) or not doc_id for doc_id in ids) or len(ids) != len(set(ids)):
            raise ValueError("PersonaMem scope IDs must be nonempty and unique")
    return scopes


def scope_doc_ids(question, root=ROOT):
    expected = _scope_id(question["shared_context_id"], question["end_index"])
    if question.get("scope_id") != expected:
        raise ValueError("PersonaMem question scope identity mismatch")
    return list(load_scopes(root)[expected])


def validate_dataset(root=ROOT, include_labels=False):
    """Validate public hashes and every scope; gold access is explicitly opt-in."""
    directory = _directory(root)
    manifest = _json(directory / "manifest.json")
    if manifest.get("schema") != SCHEMA or manifest.get("dataset") != PERSONAMEM_REPO:
        raise ValueError("PersonaMem dataset manifest schema mismatch")
    if manifest.get("revision") != PERSONAMEM_REVISION or manifest.get("split") != "32k":
        raise ValueError("PersonaMem dataset revision/split mismatch")
    if manifest.get("source_sha256") != PERSONAMEM_SOURCE_SHA256["32k"]:
        raise ValueError("PersonaMem dataset must originate from the pinned official source")
    if manifest.get("protocol", {}).get("memory_title") != MEMORY_TITLE_PROTOCOL:
        raise ValueError("PersonaMem memory title protocol mismatch")
    hashes = manifest.get("public_sha256", {})
    if set(hashes) != set(PUBLIC_FILES):
        raise ValueError("PersonaMem public artifact manifest is incomplete")
    for name in PUBLIC_FILES:
        if file_sha256(directory / name) != hashes[name]:
            raise ValueError("PersonaMem public artifact checksum mismatch: " + name)
    questions = load_questions(root)
    if _json(directory / "questions.json") != questions:
        raise ValueError("PersonaMem canonical question copies disagree")
    records = _jsonl(directory / "corpus.jsonl")
    docs = {}
    for record in records:
        doc_id, context, indices = record.get("docid"), record.get("shared_context_id"), record.get("source_message_indices")
        if not isinstance(doc_id, str) or not doc_id or doc_id in docs or not isinstance(context, str) or not context:
            raise ValueError("Invalid PersonaMem document identity")
        if not isinstance(indices, list) or not indices or any(type(i) is not int or i < 0 for i in indices) or indices != sorted(set(indices)):
            raise ValueError("Invalid PersonaMem document source message indices")
        if record.get("title") != memory_title(indices) or not isinstance(record.get("text"), str) or not record["text"]:
            raise ValueError("PersonaMem memories require the exact role-tagged text and source-order title")
        if doc_id != "personamem:memory:" + _hash([context, indices, record["text"]]):
            raise ValueError("PersonaMem memory identity mismatch")
        docs[doc_id] = record
    if not docs or _json(directory / "corpus.json") != [
        {"doc_id": item["docid"], "title": item["title"], "text": item["text"]} for item in records
    ]:
        raise ValueError("PersonaMem canonical corpus copies disagree")
    scopes = load_scopes(root)
    if set(scopes) != {question["scope_id"] for question in questions}:
        raise ValueError("PersonaMem public questions and scopes disagree")
    covered = set()
    for question in questions:
        ids = scopes[question["scope_id"]]
        for doc_id in ids:
            document = docs.get(doc_id)
            if document is None or document["shared_context_id"] != question["shared_context_id"]:
                raise ValueError("PersonaMem scope crosses a shared context or references missing memory")
            if max(document["source_message_indices"]) >= question["end_index"]:
                raise ValueError("PersonaMem scope exposes a future message")
            covered.add(doc_id)
    if covered != set(docs):
        raise ValueError("PersonaMem corpus has documents outside every public question scope")
    counts = {"questions": len(questions), "personas": len({q["persona_id"] for q in questions}),
              "shared_contexts": len({q["shared_context_id"] for q in questions}),
              "scopes": len(scopes), "documents": len(docs)}
    if any(manifest.get(key) != value for key, value in counts.items()):
        raise ValueError("PersonaMem manifest counts differ from public artifacts")
    if include_labels:
        gold_path = directory / "evaluation_only.json"
        if file_sha256(gold_path) != manifest.get("evaluation_sha256"):
            raise ValueError("PersonaMem evaluation labels checksum mismatch")
        labels = _json(gold_path)
        if not isinstance(labels, list) or len(labels) != len(questions):
            raise ValueError("PersonaMem evaluation labels must cover all questions")
        by_id = {q["id"]: q for q in questions}
        if {row.get("id") for row in labels} != set(by_id):
            raise ValueError("PersonaMem evaluation IDs differ from public question IDs")
        for row in labels:
            if set(row) != {"id", "correct_answer", "persona_id"} or row["persona_id"] != by_id[row["id"]]["persona_id"]:
                raise ValueError("Invalid PersonaMem evaluation record")
            if parse_choice(row.get("correct_answer"), by_id[row["id"]]["options"]) is None:
                raise ValueError("Invalid PersonaMem gold choice")
    return manifest


def prepare_dataset(raw_dir, root=ROOT):
    """Import every official 32k row, slicing before BT memory segmentation.

    Only the importer reads source labels. Raw source CSV is not copied into
    the deployment; its exact revision and checksums remain in the manifest.
    """
    raw_dir = Path(raw_dir)
    source_hashes = {name: file_sha256(raw_dir / name) for name in PERSONAMEM_SOURCE_SHA256["32k"]}
    if source_hashes != PERSONAMEM_SOURCE_SHA256["32k"]:
        raise ValueError("PersonaMem source does not match pinned official 32k checksums")
    contexts = load_shared_contexts(raw_dir / "shared_contexts_32k.jsonl")
    questions, labels, docs, scopes = [], [], {}, {}
    with (raw_dir / "questions_32k.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            context_id, end_index = row["shared_context_id"], int(row["end_index_in_shared_context"])
            if context_id not in contexts or not 0 < end_index <= len(contexts[context_id]):
                raise ValueError("PersonaMem question has missing context or invalid exclusive cutoff")
            scope_id = _scope_id(context_id, end_index)
            options = _normalize_options(row["all_options"])
            question = dict(id=row["question_id"], question_id=row["question_id"],
                            question=format_question(row["user_question_or_message"], options),
                            user_question=row["user_question_or_message"], options=options,
                            persona_id=row["persona_id"], shared_context_id=context_id,
                            end_index=end_index, scope_id=scope_id,
                            question_type=row["question_type"], topic=row["topic"])
            questions.append(question)
            labels.append({"id": row["question_id"], "correct_answer": row["correct_answer"], "persona_id": row["persona_id"]})
            if scope_id in scopes:
                continue
            # Pairing before slicing could include the following assistant
            # message past the cutoff. Segmentation must occur after slicing.
            memories = messages_to_memories(contexts[context_id][:end_index], context_id,
                                           include_system_persona=True, memory_granularity="user_assistant_pair")
            scope = []
            for memory in memories:
                indices = memory.metadata["source_message_indices"]
                doc_id = "personamem:memory:" + _hash([context_id, indices, memory.text])
                document = {"docid": doc_id, "title": memory_title(indices), "text": memory.text,
                            "shared_context_id": context_id, "source_message_indices": indices,
                            "roles": memory.metadata["roles"], "source_segments": memory.metadata["source_segments"],
                            "time": memory.metadata["time"]}
                if doc_id in docs and docs[doc_id] != document:
                    raise ValueError("PersonaMem deduplicated memory identity collision")
                docs[doc_id] = document
                scope.append(doc_id)
            scopes[scope_id] = scope
    output = _directory(root)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=output) as temporary:
        staging = Path(temporary)
        def write_json(name, value):
            (staging / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        def write_jsonl(name, rows):
            with (staging / name).open("w", encoding="utf-8") as target:
                for item in rows:
                    target.write(json.dumps(item, ensure_ascii=False) + "\n")
        corpus = list(docs.values())
        write_jsonl("questions.jsonl", questions)
        write_json("questions.json", questions)
        write_jsonl("corpus.jsonl", corpus)
        write_json("corpus.json", [{"doc_id": d["docid"], "title": d["title"], "text": d["text"]} for d in corpus])
        write_json("scopes.json", scopes)
        write_json("evaluation_only.json", labels)
        manifest = {"schema": SCHEMA, "dataset": PERSONAMEM_REPO, "revision": PERSONAMEM_REVISION,
                    "split": "32k", "questions": len(questions),
                    "personas": len({q["persona_id"] for q in questions}),
                    "shared_contexts": len({q["shared_context_id"] for q in questions}),
                    "scopes": len(scopes), "documents": len(corpus),
                    "source_sha256": source_hashes,
                    "source_urls": {name: f"https://huggingface.co/datasets/{PERSONAMEM_REPO}/resolve/{PERSONAMEM_REVISION}/{name}"
                                    for name in source_hashes},
                    "public_sha256": {name: file_sha256(staging / name) for name in PUBLIC_FILES},
                    "evaluation_sha256": file_sha256(staging / "evaluation_only.json"),
                    "protocol": {"row_selection": "all official 32k rows; no additional sample or train/test claim",
                                 "memory_granularity": "user_assistant_pair", "include_system_persona": True,
                                 "memory_title": MEMORY_TITLE_PROTOCOL,
                                 "cutoff": "messages[:end_index] before memory segmentation",
                                 "retrieval_scope": "only docids listed in the current question scope_id",
                                 "metric": "strict single-choice accuracy; invalid choices count as incorrect"}}
        write_json("manifest.json", manifest)
        for name in (*PUBLIC_FILES, "evaluation_only.json", "manifest.json"):
            os.replace(staging / name, output / name)
    return validate_dataset(root, include_labels=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, help="Pinned PersonaMem-v1 questions_32k.csv and shared_contexts_32k.jsonl directory")
    parser.add_argument("--root", type=Path, default=ROOT)
    arguments = parser.parse_args()
    print(json.dumps(prepare_dataset(arguments.raw_dir, arguments.root), ensure_ascii=False, indent=2))
