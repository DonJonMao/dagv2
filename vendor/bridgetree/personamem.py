from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Sequence

from .temporal import TimeMark
from .types import Memory

PERSONAMEM_REPO = "bowen-upenn/PersonaMem-v1"
PERSONAMEM_REVISION = "fd7c30f071d5c2ee2a211506783be222d7b6002e"
PERSONAMEM_SOURCE_SHA256 = {
    "32k": {
        "questions_32k.csv": "cccd34cf53e0bc4d9536c04cff5ca045156d9a4e227e83327112482840bbc93c",
        "shared_contexts_32k.jsonl": "217247ebfec9e8442fc53570c795ab69f21aad08745f7de78d9beab51b122d4a",
    }
}


@dataclass(frozen=True)
class PersonaMemExample:
    persona_id: str
    question_id: str
    question_type: str
    topic: str
    query: str
    correct_answer: str
    all_options: str
    shared_context_id: str
    end_index: int
    messages: List[Dict[str, str]]
    # Optional explicit query-time annotations used by T.  PersonaMem's
    # shipped rows do not contain these fields, so old positional construction
    # remains valid.
    query_time: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def query_cutoff(self) -> Any:
        return self.query_time

    @property
    def time_metadata(self) -> Dict[str, Any]:
        return dict(self.metadata)


def load_shared_contexts(path: str | Path) -> Dict[str, List[Dict[str, str]]]:
    contexts: Dict[str, List[Dict[str, str]]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"context line {line_number} is not an object")
            for context_id, messages in item.items():
                if context_id in contexts:
                    raise ValueError(f"duplicate shared_context_id: {context_id}")
                contexts[str(context_id)] = list(messages)
    return contexts


def iter_examples(question_path: str | Path, context_path: str | Path) -> Iterator[PersonaMemExample]:
    contexts = load_shared_contexts(context_path)
    with Path(question_path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            context_id = row["shared_context_id"]
            end_index = int(row["end_index_in_shared_context"])
            if context_id not in contexts:
                raise KeyError(f"missing shared context: {context_id}")
            messages = contexts[context_id][:end_index]
            query_metadata: Dict[str, Any] = {}
            for key in ("query_time", "query_date", "cutoff", "time"):
                raw_value = row.get(key)
                if not raw_value:
                    continue
                if key == "time":
                    try:
                        parsed = json.loads(raw_value)
                        query_metadata.update(parsed if isinstance(parsed, dict) else {"value": parsed})
                    except (TypeError, json.JSONDecodeError):
                        query_metadata[key] = raw_value
                else:
                    query_metadata[key] = raw_value
            yield PersonaMemExample(
                persona_id=row["persona_id"],
                question_id=row["question_id"],
                question_type=row["question_type"],
                topic=row["topic"],
                query=row["user_question_or_message"],
                correct_answer=row["correct_answer"],
                all_options=row["all_options"],
                shared_context_id=context_id,
                end_index=end_index,
                messages=messages,
                query_time=(row.get("query_time") or row.get("query_date") or row.get("cutoff") or None),
                metadata=query_metadata,
            )


def messages_to_memories(
    messages: Sequence[Mapping[str, Any]],
    source_prefix: str,
    include_system_persona: bool = True,
    memory_granularity: str = "user_assistant_pair",
) -> List[Memory]:
    """Convert PersonaMem messages into independent, traceable memory turns.

    Consecutive messages with the same role are merged; a user turn and its
    following assistant response form one memory, matching the official
    RF-Mem PersonaMem preprocessing while retaining source indices.
    """
    if memory_granularity not in {"user_only", "user_assistant_pair"}:
        raise ValueError("memory_granularity must be user_only or user_assistant_pair")

    temporal_keys = {
        "time",
        "timestamp",
        "date",
        "datetime",
        "event_time",
        "event_start",
        "event_end",
        "observed",
        "observed_start",
        "observed_end",
        "validity",
        "time_source",
        "start",
        "end",
    }

    def message_time_value(message: Mapping[str, Any]) -> Any:
        """Extract the first explicit temporal value from a message envelope."""
        if "time" in message and message.get("time") is not None:
            return message.get("time")
        metadata = message.get("metadata")
        if isinstance(metadata, Mapping):
            if "time" in metadata and metadata.get("time") is not None:
                return metadata.get("time")
            if any(key in metadata for key in temporal_keys - {"time"}):
                return metadata
        for key in ("timestamp", "datetime", "date", "event_time"):
            if key in message and message.get(key) is not None:
                return message.get(key)
        return None

    def coerce_message_time(value: Any) -> TimeMark | None:
        """Normalize scalar, interval, and nested time spellings.

        ``TimeMark.from_metadata`` intentionally treats a bare scalar as an
        observed fallback (the legacy message-index convention).  Message
        payloads, however, are explicit annotations when a scalar is present,
        so construct those through ``TimeMark.instant`` instead.
        """
        if value is None:
            return None
        if isinstance(value, TimeMark):
            return value
        if isinstance(value, Mapping):
            # Common compact aliases such as ``{"timestamp": ...}`` and
            # nested ``{"metadata": {"time": ...}}`` are explicit dates.
            nested = value.get("time")
            if nested is not None and not isinstance(nested, Mapping):
                return coerce_message_time(nested)
            for key in ("timestamp", "datetime", "date", "event_time"):
                if key in value and value.get(key) is not None:
                    return coerce_message_time(value.get(key))
            try:
                mark = TimeMark.from_metadata(value)
            except (TypeError, ValueError, OverflowError):
                return None
            # An arbitrary metadata mapping is not an explicit time value;
            # leave it out so the normal message-index fallback is retained.
            if mark.unavailable and str(value.get("validity", "")).strip().lower() != "unknown":
                numeric_fields = {
                    key
                    for key in (
                        "observed",
                        "observed_start",
                        "observed_end",
                        "event_start",
                        "event_end",
                        "start",
                        "end",
                    )
                    if value.get(key) is not None
                }
                if not numeric_fields:
                    return None
            return mark
        if isinstance(value, (list, tuple)):
            if len(value) >= 2:
                try:
                    return TimeMark.durative(value[0], value[1], source="explicit")
                except (TypeError, ValueError, OverflowError):
                    return None
            if len(value) == 1:
                return coerce_message_time(value[0])
            return None
        try:
            return TimeMark.instant(value, source="explicit")
        except (TypeError, ValueError, OverflowError):
            return None

    def combined_time_metadata(raw_values: Sequence[Any], indices: Sequence[int]) -> Dict[str, Any]:
        marks = [mark for raw in raw_values if (mark := coerce_message_time(raw)) is not None]
        fallback_start = float(min(indices))
        fallback_end = float(max(indices))
        if not marks:
            return {
                "observed_start": fallback_start,
                "observed_end": fallback_end,
                "event_start": None,
                "event_end": None,
                # Message indices are retained as an observation envelope,
                # but they are not event times.  With no explicit annotation
                # the temporal measure is therefore unknown/neutral rather
                # than an accidental recency or exact-index gate.
                "validity": "unknown",
                "time_source": "message_index",
            }

        # An explicitly unknown annotation remains unknown (and therefore
        # neutral in T) rather than being silently converted to a recency
        # point.  Keep the observation envelope for diagnostics.
        known = [mark for mark in marks if not mark.unavailable]
        if not known:
            return {
                "observed_start": fallback_start,
                "observed_end": fallback_end,
                "event_start": None,
                "event_end": None,
                "validity": "unknown",
                "time_source": "unknown",
            }

        observed_starts = [float(mark.observed_start) for mark in known if mark.observed_start is not None]
        observed_ends = [float(mark.observed_end) for mark in known if mark.observed_end is not None]
        event_starts = [float(mark.event_start) for mark in known if mark.event_start is not None]
        event_ends = [float(mark.event_end) for mark in known if mark.event_end is not None]
        observed_start = min(observed_starts) if observed_starts else fallback_start
        observed_end = max(observed_ends) if observed_ends else fallback_end
        event_start = min(event_starts) if event_starts else None
        event_end = max(event_ends) if event_ends else None
        if event_start is not None or event_end is not None:
            left = event_start if event_start is not None else event_end
            right = event_end if event_end is not None else event_start
            validity = "durative" if left is not None and right is not None and abs(left - right) > 1e-12 else "instant"
            source = "explicit"
        else:
            validity = "instant"
            source = "message_index" if all(mark.time_source == "message_index" for mark in known) else "explicit"
        return {
            "observed_start": observed_start,
            "observed_end": observed_end,
            "event_start": event_start,
            "event_end": event_end,
            "validity": validity,
            "time_source": source,
        }

    normalized: List[Dict[str, Any]] = []
    for index, message in enumerate(messages):
        role = str(message.get("role", "unknown")).strip().lower()
        content = str(message.get("content", "")).strip()
        if not content or (role == "system" and not include_system_persona):
            continue
        if role == "assistant" and memory_granularity == "user_only":
            continue
        message_time = message_time_value(message)
        if memory_granularity == "user_assistant_pair" and normalized and normalized[-1]["role"] == role:
            normalized[-1]["content"] += "\n\n" + content
            normalized[-1]["indices"].append(index)
            if message_time is not None:
                normalized[-1].setdefault("times", []).append(message_time)
        else:
            normalized.append(
                {
                    "role": role,
                    "content": content,
                    "indices": [index],
                    "times": [message_time] if message_time is not None else [],
                }
            )

    memories: List[Memory] = []
    cursor = 0
    memory_index = 0
    while cursor < len(normalized):
        current = normalized[cursor]
        roles = [current["role"]]
        indices = list(current["indices"])
        times = list(current.get("times", []))
        parts = [f"{current['role'].capitalize()}:\n{current['content']}"]
        # Preserve the actual construction boundaries: a role-looking marker
        # inside user content must never be reinterpreted as an assistant turn.
        source_segments = [{
            "role": current["role"],
            "start": len(f"{current['role'].capitalize()}:\n"),
            "end": len(parts[0]),
            "source_message_indices": list(current["indices"]),
        }]
        if (
            memory_granularity == "user_assistant_pair"
            and current["role"] == "user"
            and cursor + 1 < len(normalized)
            and normalized[cursor + 1]["role"] == "assistant"
        ):
            following = normalized[cursor + 1]
            roles.append("assistant")
            indices.extend(following["indices"])
            times.extend(following.get("times", []))
            parts.append(f"Assistant:\n{following['content']}")
            following_start = len(parts[0]) + len("\n\nAssistant:\n")
            source_segments.append({
                "role": "assistant", "start": following_start,
                "end": following_start + len(following["content"]),
                "source_message_indices": list(following["indices"]),
            })
            cursor += 2
        else:
            cursor += 1
        memory_id = f"{source_prefix}:m{memory_index:05d}"
        time_metadata = combined_time_metadata(times, indices)
        memories.append(
            Memory(
                memory_id=memory_id,
                text="\n\n".join(parts),
                timestamp=float(max(indices)),
                source_id=f"{source_prefix}:{min(indices)}-{max(indices)}",
                metadata={
                    "roles": roles,
                    "source_message_indices": indices,
                    "source_segments": source_segments,
                    # Message indices are an ordered observation scale, never
                    # calendar dates.  Explicit event annotations, when a
                    # caller supplies them, may replace this entry upstream.
                    "time": time_metadata,
                },
            )
        )
        memory_index += 1
    return memories


def parse_options(raw: str) -> List[str]:
    try:
        value = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return [raw]
    return [str(item) for item in value] if isinstance(value, list) else [str(value)]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_split(
    raw_dir: str | Path,
    processed_dir: str | Path,
    split: str,
    *,
    verify_pinned_source: bool = False,
) -> Dict[str, Any]:
    raw_root = Path(raw_dir)
    output_root = Path(processed_dir) / split
    output_root.mkdir(parents=True, exist_ok=True)
    questions = raw_root / f"questions_{split}.csv"
    contexts = raw_root / f"shared_contexts_{split}.jsonl"
    if not questions.exists() or not contexts.exists():
        raise FileNotFoundError(f"missing PersonaMem {split} files in {raw_root}")
    source_hashes = {questions.name: file_sha256(questions), contexts.name: file_sha256(contexts)}
    expected_hashes = PERSONAMEM_SOURCE_SHA256.get(split)
    if verify_pinned_source and expected_hashes is None:
        raise ValueError(f"no pinned source checksums are registered for PersonaMem {split}")
    if verify_pinned_source and source_hashes != expected_hashes:
        mismatched = sorted(
            filename
            for filename, expected_hash in expected_hashes.items()
            if source_hashes.get(filename) != expected_hash
        )
        raise ValueError(f"PersonaMem {split} source checksum mismatch: {mismatched}")

    context_map = load_shared_contexts(contexts)
    context_output = output_root / "contexts.jsonl"
    query_output = output_root / "queries.jsonl"
    manifest_path = output_root / "manifest.json"
    staged_paths: list[Path] = []

    def staged_path(target: Path) -> Path:
        descriptor, raw_path = tempfile.mkstemp(
            dir=output_root,
            prefix=f".{target.name}.",
            suffix=".tmp",
        )
        path = Path(raw_path)
        staged_paths.append(path)
        try:
            try:
                output_mode = target.stat().st_mode & 0o777
            except FileNotFoundError:
                output_mode = 0o644
            os.fchmod(descriptor, output_mode)
        finally:
            os.close(descriptor)
        return path

    try:
        context_staged = staged_path(context_output)
        query_staged = staged_path(query_output)
        manifest_staged = staged_path(manifest_path)
        with context_staged.open("w", encoding="utf-8") as handle:
            for context_id in sorted(context_map):
                record = {"shared_context_id": context_id, "messages": context_map[context_id]}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

        query_count = 0
        persona_ids = set()
        with (
            questions.open("r", encoding="utf-8", newline="") as source,
            query_staged.open("w", encoding="utf-8") as target,
        ):
            for row in csv.DictReader(source):
                record = dict(row)
                record["end_index_in_shared_context"] = int(
                    record["end_index_in_shared_context"]
                )
                target.write(json.dumps(record, ensure_ascii=False) + "\n")
                query_count += 1
                persona_ids.add(row["persona_id"])
            target.flush()
            os.fsync(target.fileno())

        manifest = {
            "dataset": PERSONAMEM_REPO,
            "revision": PERSONAMEM_REVISION,
            "split": split,
            "questions": query_count,
            "personas": len(persona_ids),
            "shared_contexts": len(context_map),
            "source_sha256": source_hashes,
            "outputs": {
                context_output.name: file_sha256(context_staged),
                query_output.name: file_sha256(query_staged),
            },
        }
        with manifest_staged.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        # Publish only complete files.  The manifest is the commit marker and
        # therefore moves last, after both data files named by its checksums.
        os.replace(context_staged, context_output)
        os.replace(query_staged, query_output)
        os.replace(manifest_staged, manifest_path)
        directory_fd = os.open(output_root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return manifest
    finally:
        for path in staged_paths:
            path.unlink(missing_ok=True)
