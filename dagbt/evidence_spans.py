"""Exact deterministic source spans, independent of model responses and batches.

Coordinates index the complete reader passage in Python characters. Speaker
roles come only from source metadata; role-looking text is never parsed.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import re


class SourceSpanError(ValueError):
    """The source or its authoritative provenance cannot be represented safely."""


_MISSING = object()
_ROLES = {"document", "system", "user", "assistant", "unknown"}
_BOUNDARY = re.compile(r'[。！？]+[”’"\')）\]]*[ \t]*|[.!?]+[”’"\')）\]]*(?=\s|$)[ \t]*|\r?\n+')


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def document_passage(document):
    if isinstance(document, str):
        passage = document
    elif isinstance(document, Mapping):
        passage = document.get("passage", _MISSING)
        if passage is _MISSING:
            title, text = document.get("title", ""), document.get("text", "")
            if not isinstance(title, str) or not isinstance(text, str):
                raise SourceSpanError("Document title and text must be strings")
            passage = title + "\n" + text if title and text else title or text
    else:
        passage = getattr(document, "passage", None)
    if not isinstance(passage, str) or not passage:
        raise SourceSpanError("Document passage must be a nonempty string")
    return passage


def normalize_source_segments(text, segments=_MISSING, *, default_role="document"):
    """Validate ordered authoritative segments and fill structural gaps unknown.

    An absent segment field denotes an ordinary QA document. An explicitly
    empty segment list denotes unknown provenance, not an attributed speaker.
    """
    if not isinstance(text, str):
        raise SourceSpanError("Source text must be a string")
    if segments is _MISSING:
        if default_role not in _ROLES:
            raise SourceSpanError("Invalid default source role")
        return ([{"role": default_role, "start": 0, "end": len(text),
                  "source_message_indices": [], "provenance": "document"}]
                if text else [])
    if not isinstance(segments, list):
        raise SourceSpanError("source_segments must be a list")
    result, previous_end = [], 0

    def unknown(start, end):
        return {"role": "unknown", "start": start, "end": end,
                "source_message_indices": [], "provenance": "unknown_gap"}

    for segment in segments:
        if not isinstance(segment, Mapping):
            raise SourceSpanError("Source segment must be an object")
        start, end, role = segment.get("start"), segment.get("end"), segment.get("role")
        if type(start) is not int or type(end) is not int or not previous_end <= start <= end <= len(text):
            raise SourceSpanError("Invalid or overlapping source segment offsets")
        if not isinstance(role, str) or role not in _ROLES:
            raise SourceSpanError("Invalid authoritative source role")
        indices = segment.get("source_message_indices")
        if (not isinstance(indices, list) or any(type(i) is not int or i < 0 for i in indices)
                or indices != sorted(set(indices))):
            raise SourceSpanError("Source message indices must be sorted unique nonnegative integers")
        if start > previous_end:
            result.append(unknown(previous_end, start))
        if end > start:
            provenance = segment.get("provenance", "authoritative")
            if not isinstance(provenance, str) or provenance not in {"authoritative", "unknown_gap", "document", "title_metadata"}:
                raise SourceSpanError("Invalid source provenance")
            if provenance in {"unknown_gap", "title_metadata"} and (role != "unknown" or indices):
                raise SourceSpanError("Structural source metadata cannot attribute a speaker")
            result.append({"role": role, "start": start, "end": end,
                           "source_message_indices": list(indices), "provenance": provenance})
        previous_end = end
    if previous_end < len(text):
        result.append(unknown(previous_end, len(text)))
    return result


def document_source_metadata(document):
    """Return a validated copy with source coordinates relative to passage."""
    passage = document_passage(document)
    metadata = (document.get("metadata", _MISSING) if isinstance(document, Mapping)
                else getattr(document, "metadata", _MISSING))
    if metadata is _MISSING:
        metadata = {}
    if not isinstance(metadata, Mapping) or any(not isinstance(k, str) for k in metadata):
        raise SourceSpanError("Document metadata must be an object with string keys")
    metadata = deepcopy(dict(metadata))
    try:
        json.dumps(metadata, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise SourceSpanError("Document metadata must contain finite JSON values") from exc
    if "source_message_indices" in metadata:
        indices = metadata["source_message_indices"]
        if (not isinstance(indices, list) or any(type(i) is not int or i < 0 for i in indices)
                or indices != sorted(set(indices))):
            raise SourceSpanError("Document source message indices must be sorted unique nonnegative integers")
    if "time" in metadata and not isinstance(metadata["time"], Mapping):
        raise SourceSpanError("Document time metadata must be an object")
    metadata["source_segments"] = normalize_source_segments(passage, metadata.get("source_segments", _MISSING))
    metadata["observation_order_is_event_time"] = False
    return metadata


def _sentence_ranges(text, start, end):
    cursor = start
    for match in _BOUNDARY.finditer(text, start, end):
        right = match.end()
        if right > cursor:
            yield cursor, right
            cursor = right
    if cursor < end:
        yield cursor, end


def build_source_spans(doc_id, document, max_chars=400):
    """Partition all source characters without crossing authoritative role boundaries.

    Long sentence pieces retain one premise group. IDs bind the exact source,
    its metadata and coordinates, and do not depend on mapping batch order.
    """
    if not isinstance(doc_id, str) or not doc_id:
        raise SourceSpanError("Source document ID must be a nonempty string")
    own_id = document.get("doc_id", doc_id) if isinstance(document, Mapping) else getattr(document, "doc_id", doc_id)
    if own_id != doc_id:
        raise SourceSpanError("Source document ID disagrees with the supplied document")
    if type(max_chars) is not int or max_chars < 1:
        raise SourceSpanError("max_chars must be a positive integer")
    passage = document_passage(document)
    metadata = document_source_metadata(document)
    source_hash = hashlib.sha256(passage.encode("utf-8")).hexdigest()
    identity = [doc_id, source_hash]
    spans = []
    for segment in metadata["source_segments"]:
        pending_start = pending_end = segment["start"]
        pending_groups = []

        def emit(left, right, groups):
            identifier = "src_" + _digest([*identity, segment, left, right])[:24]
            spans.append({"id": identifier, "doc_id": doc_id, "start": left, "end": right,
                          "text": passage[left:right], "source_role": segment["role"],
                          "source_message_indices": list(segment["source_message_indices"]),
                          "source_hash": source_hash, "premise_group_ids": list(groups),
                          "source_segments": [deepcopy(segment)], "provenance": segment["provenance"],
                          "time_metadata": deepcopy(metadata.get("time", {}))})

        for left, right in _sentence_ranges(passage, segment["start"], segment["end"]):
            group = "prem_" + _digest([*identity, segment, left, right])[:24]
            if right - left > max_chars:
                if pending_groups:
                    emit(pending_start, pending_end, pending_groups)
                    pending_groups = []
                for start in range(left, right, max_chars):
                    emit(start, min(start + max_chars, right), [group])
                pending_start = pending_end = right
            else:
                if pending_groups and right - pending_start > max_chars:
                    emit(pending_start, pending_end, pending_groups)
                    pending_groups = []
                if not pending_groups:
                    pending_start = left
                pending_end = right
                pending_groups.append(group)
        if pending_groups:
            emit(pending_start, pending_end, pending_groups)
    for index, span in enumerate(spans):
        span["previous_span_id"] = spans[index - 1]["id"] if index else None
        span["next_span_id"] = spans[index + 1]["id"] if index + 1 < len(spans) else None
    return spans


def independent_premise_count(spans: Sequence[Mapping]):
    """Count provenance-disjoint source groups, not the number of source chunks.

    Inputs are validated source span records, not model claims. Overlapping
    ranges and shared original-sentence groups are merged transitively. This
    conservative count is not a semantic independence or support guarantee.
    """
    if isinstance(spans, (str, bytes)) or not isinstance(spans, Sequence):
        raise SourceSpanError("Premises must be a source span sequence")
    values = []
    for span in spans:
        if not isinstance(span, Mapping):
            raise SourceSpanError("Premise source span must be an object")
        doc, start, end = span.get("doc_id"), span.get("start"), span.get("end")
        groups = span.get("premise_group_ids")
        if (not isinstance(doc, str) or not doc or type(start) is not int or type(end) is not int
                or not 0 <= start < end or not isinstance(groups, list)
                or not groups or any(not isinstance(g, str) or not g for g in groups)):
            raise SourceSpanError("Premise source span lacks validated coordinates/groups")
        values.append((doc, start, end, set(groups)))
    parents = list(range(len(values)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for index, (doc, left, right, groups) in enumerate(values):
        for other, (old_doc, old_left, old_right, old_groups) in enumerate(values[:index]):
            if doc == old_doc and (groups & old_groups or left < old_right and old_left < right):
                parents[root(index)] = root(other)
    return len({root(index) for index in range(len(values))})
