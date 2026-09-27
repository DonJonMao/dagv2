"""Strict evidence object parser adapted from BridgeTree commit 16809bd.

Source: bridgetree_preference_rag/src/bridgetree/evidence_protocol.py,
2026-09-27. No source text or partial JSON content is silently repaired.
"""
from __future__ import annotations

import json
import math
import re


class EvidenceJSONError(ValueError):
    def __init__(self, message: str, category: str = "json_syntax"):
        super().__init__(message)
        self.category = category


def parse_evidence_object(text: str) -> dict:
    """Accept complete BOM/fence/prose wrappers, never repair JSON content.

    Prose extraction permits one unambiguous complete top-level object only;
    JSON delimiters/quotes outside it, multiple objects, duplicate keys and
    non-finite numbers stay errors. Punctuation alone is not a prose wrapper.
    A malformed leading object cannot be replaced by one of its nested objects.
    """
    if not isinstance(text, str):
        raise EvidenceJSONError("response must be text")
    value = text.strip().lstrip("\ufeff").strip()
    fence = re.fullmatch(r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```", value, re.DOTALL | re.IGNORECASE)
    if fence:
        value = fence.group(1).strip()

    def unique(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise EvidenceJSONError(f"duplicate JSON key: {key}")
            result[key] = item
        return result

    def invalid(constant):
        raise EvidenceJSONError(f"non-finite JSON constant: {constant}")

    def finite_float(number):
        result = float(number)
        if not math.isfinite(result):
            raise EvidenceJSONError("JSON number exceeds finite range")
        return result

    decoder = json.JSONDecoder(object_pairs_hook=unique, parse_constant=invalid, parse_float=finite_float)
    start = value.find("{")
    if start < 0 or any(c in value[:start] for c in '{}[]"'):
        raise EvidenceJSONError("invalid complete JSON object: no unique top-level object")
    try:
        result, end = decoder.raw_decode(value, start)
    except EvidenceJSONError:
        raise
    except RecursionError as exc:
        raise EvidenceJSONError("JSON nesting exceeds parser limit") from exc
    except json.JSONDecodeError as exc:
        raise EvidenceJSONError(f"invalid complete JSON object: {exc}") from exc
    except ValueError as exc:
        raise EvidenceJSONError("JSON numeric value exceeds parser limits") from exc
    if any(c in value[end:] for c in '{}[]"'):
        raise EvidenceJSONError("multiple objects or incomplete trailing JSON", "json_wrapper")
    if not isinstance(result, dict):
        raise EvidenceJSONError("response must be a JSON object")
    # The only permitted extra material is prose. JSON scalars at either side
    # are competing JSON values rather than a prose wrapper.
    for extra in (value[:start].strip(), value[end:].strip()):
        if extra:
            if "```" in extra or not any(character.isalpha() for character in extra):
                raise EvidenceJSONError("incomplete or competing JSON wrapper", "json_wrapper")
            try:
                decoder.raw_decode(extra)
            except json.JSONDecodeError:
                continue
            raise EvidenceJSONError("competing JSON values", "json_wrapper")
    return result

