"""Auditable embedding-only views; source records and evidence spans stay intact.

The packaged Qwen tokenizer plus one token per input matched all 163 successful
embedding requests (3,250 inputs) in the October 11 smoke artifact. This is
empirical compatibility, not a verified hash of the server tokenizer. The 4,096
capacity was reported by that deployment; reserve another 64 tokens for drift.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path
from typing import Mapping, Sequence


NAVIGATION_VIEW_VERSION = "qwen_navigation_prefix_view_v1"
MAX_INPUT_TOKENS = 4096
SAFETY_TOKENS = 64
SERVICE_ADDED_TOKENS = 1
TEXT_TOKEN_LIMIT = MAX_INPUT_TOKENS - SAFETY_TOKENS - SERVICE_ADDED_TOKENS
TOKENIZER_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
TOKENIZER_PATH = Path(__file__).resolve().parents[1] / "package/tokenizer/tokenizer.json"
EXCERPT_MARKER = "\n[Navigation excerpt; remainder omitted.]"


def navigation_limits_apply(config: Mapping) -> bool:
    """The measured capacity belongs to this deployment, not every embedding."""
    return (config.get("model_profile") == "bridgetree"
            and config.get("embedding_model") == "qwen3-embedding-8b"
            and str(config.get("embedding_base_url", "")).rstrip("/")
            == "http://111.19.156.74:8001/v1")


class NavigationQueryLimitError(ValueError):
    """Required query fields cannot fit without changing the retrieval task."""

    def __init__(self, diagnostics: dict):
        self.diagnostics = diagnostics
        super().__init__(
            "Fixed navigation query exceeds embedding capacity: "
            f"{diagnostics['fixed_tokens_local']} local tokens + "
            f"{SERVICE_ADDED_TOKENS} service token + {SAFETY_TOKENS} reserve "
            f"> {MAX_INPUT_TOKENS}"
        )


class NavigationTokenizerError(RuntimeError):
    """The empirically validated tokenizer is missing or has changed."""


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def _tokenizer():
    from tokenizers import Tokenizer

    try:
        raw = TOKENIZER_PATH.read_bytes()
    except OSError as exc:
        raise NavigationTokenizerError("Packaged navigation tokenizer is unavailable") from exc
    if hashlib.sha256(raw).hexdigest() != TOKENIZER_SHA256:
        raise NavigationTokenizerError("Packaged navigation tokenizer fingerprint changed")
    tokenizer = Tokenizer.from_str(raw.decode("utf-8"))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer


def navigation_tokens(text: str) -> int:
    """Count the actual embedding string, without a generator chat template."""
    return len(_tokenizer().encode(text, add_special_tokens=False).ids)


@dataclass(frozen=True)
class NavigationSource:
    source_id: str
    text: str


@dataclass(frozen=True)
class NavigationPassages:
    """Structured sources avoid parsing section-like strings inside evidence."""
    kind: str
    heading: str
    sources: tuple[NavigationSource, ...]


@dataclass(frozen=True)
class NavigationQueryView:
    text: str
    diagnostics: dict


def render_navigation_query(parts: Sequence[str | NavigationPassages]) -> str:
    return "".join(part if isinstance(part, str) else part.heading + "\n\n".join(
        f"[{source.source_id}]\n{source.text}" for source in part.sources) for part in parts)


def bounded_navigation_query(
    parts: Sequence[str | NavigationPassages], *, instruction: str = ""
) -> NavigationQueryView:
    """Keep fixed fields; fill source prefixes in stable rendered order.

    Small queries are byte-for-byte unchanged. For oversized queries, each
    source is admitted whole if possible, otherwise a prefix is selected using
    full-query tokenization. Later sources can use any remaining space. Prefix
    search need not find the longest possible BPE prefix; every accepted view
    is independently checked, so nonmonotonic token counts cannot break the cap.
    Offsets refer to Python Unicode character positions in the untouched source.
    """
    occurrences = [(part, source) for part in parts if isinstance(part, NavigationPassages)
                   for source in part.sources]
    full_lengths = [len(source.text) for _, source in occurrences]

    def render(lengths: Sequence[int]) -> str:
        pieces = []
        index = 0
        for part in parts:
            if isinstance(part, str):
                pieces.append(part)
                continue
            passages = []
            for source in part.sources:
                size = lengths[index]
                index += 1
                if size:
                    text = source.text[:size]
                    if size < len(source.text):
                        text += EXCERPT_MARKER
                    passages.append(f"[{source.source_id}]\n{text}")
            if passages:
                pieces.append(part.heading + "\n\n".join(passages))
        return "".join(pieces)

    original = render(full_lengths)
    original_tokens = navigation_tokens(instruction + original)
    lengths = full_lengths.copy()
    fixed = render([0] * len(occurrences))
    fixed_tokens = navigation_tokens(instruction + fixed)
    diagnostics = {
        "event": "navigation_query_view",
        "view_version": NAVIGATION_VIEW_VERSION,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "token_count_source": "packaged_qwen_tokenizer_empirically_validated",
        "server_tokenizer_identity_verified": False,
        "max_input_tokens": MAX_INPUT_TOKENS,
        "service_added_tokens_observed": SERVICE_ADDED_TOKENS,
        "safety_tokens": SAFETY_TOKENS,
        "text_token_limit": TEXT_TOKEN_LIMIT,
        "original_tokens_local": original_tokens,
        "fixed_tokens_local": fixed_tokens,
        "original_input_sha256": _sha(instruction + original),
        "selection_policy": "rendered_source_order_then_original_prefix",
        "source_offsets_unit": "unicode_characters",
        "source_records_modified": False,
        "evidence_support_claim": False,
    }
    if fixed_tokens > TEXT_TOKEN_LIMIT:
        diagnostics.update(status="fixed_input_exceeds_capacity", bounded=False)
        raise NavigationQueryLimitError(diagnostics)
    if original_tokens > TEXT_TOKEN_LIMIT:
        lengths = [0] * len(occurrences)
        for index, (_, source) in enumerate(occurrences):
            lengths[index] = len(source.text)
            if navigation_tokens(instruction + render(lengths)) <= TEXT_TOKEN_LIMIT:
                continue
            lengths[index] = 0
            low, high, accepted = 1, len(source.text) - 1, 0
            while low <= high:
                middle = (low + high) // 2
                lengths[index] = middle
                if navigation_tokens(instruction + render(lengths)) <= TEXT_TOKEN_LIMIT:
                    accepted = middle
                    low = middle + 1
                else:
                    high = middle - 1
            lengths[index] = accepted
    text = render(lengths)
    final_tokens = navigation_tokens(instruction + text)
    if final_tokens > TEXT_TOKEN_LIMIT:
        raise AssertionError("Navigation view exceeded its final serialized token limit")
    diagnostics.update(
        status="ready", bounded=lengths != full_lengths,
        final_tokens_local=final_tokens,
        final_tokens_with_observed_service_overhead=final_tokens + SERVICE_ADDED_TOKENS,
        final_input_sha256=_sha(instruction + text),
        sources=[{
            "section": part.kind,
            "source_id": source.source_id,
            "source_sha256": _sha(source.text),
            "source_characters": len(source.text),
            "included_char_ranges": [{"start": 0, "end": size}] if size else [],
            "omitted_char_ranges": [{"start": size, "end": len(source.text)}]
                                   if size < len(source.text) else [],
            "included": size > 0,
            "truncated": size < len(source.text),
        } for (part, source), size in zip(occurrences, lengths)],
    )
    return NavigationQueryView(text, diagnostics)
