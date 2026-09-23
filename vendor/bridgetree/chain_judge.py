"""Contracts for the train-free BridgeTree-Chain semantic judge.

The module is deliberately backend agnostic.  A backend must return the two
labels from one prompt and one output position; callers cannot silently turn a
missing label into a probability.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .config import GeneratorConfig


@dataclass(frozen=True)
class PublicQuery:
    dataset_revision: str
    persona_id: str
    question_id: str
    query: str
    public_options: str = ""
    shared_context_id: str = ""
    end_index: int | None = None
    visible_memories: tuple[str, ...] = ()
    time_metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.query:
            raise ValueError("PublicQuery.query cannot be empty")
        object.__setattr__(self, "visible_memories", tuple(str(x) for x in self.visible_memories))
        object.__setattr__(self, "time_metadata", tuple((str(k), str(v)) for k, v in self.time_metadata))

    def identity(self) -> str:
        payload = {"dataset_revision": self.dataset_revision, "persona_id": self.persona_id,
                   "question_id": self.question_id, "query": self.query,
                   "public_options": self.public_options, "shared_context_id": self.shared_context_id,
                   "end_index": self.end_index, "visible_memories": list(self.visible_memories),
                   "time_metadata": list(self.time_metadata)}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True)
class JointScore:
    raw_difference: float
    log_u: float
    u: float
    input_hash: str
    sufficient_label: str = "A"
    insufficient_label: str = "B"

    @classmethod
    def from_logits(cls, sufficient: float, insufficient: float, *, input_hash: str,
                    sufficient_label: str = "A", insufficient_label: str = "B") -> "JointScore":
        if not math.isfinite(sufficient) or not math.isfinite(insufficient):
            raise ValueError("joint logits must be finite")
        d = float(sufficient) - float(insufficient)
        log_u = -math.log1p(math.exp(-d)) if d >= 0 else d - math.log1p(math.exp(d))
        return cls(d, log_u, math.exp(log_u), str(input_hash), sufficient_label, insufficient_label)


@dataclass(frozen=True)
class Claim:
    text: str
    source_ids: tuple[str, ...]
    input_hash: str


@dataclass(frozen=True)
class Verification:
    supported: bool
    reason: str = ""
    input_hash: str = ""


class JointJudge(Protocol):
    def score(self, query: PublicQuery, raw_ids: tuple[str, ...]) -> JointScore: ...
    def claim(self, query: PublicQuery, raw_ids: tuple[str, ...]) -> Claim: ...
    def verify(self, query: PublicQuery, fixed_claim: Claim, raw_ids: tuple[str, ...]) -> Verification: ...


def require_label_pair(response: Mapping[str, Any], *, sufficient_label: str = "A",
                       insufficient_label: str = "B") -> tuple[float, float]:
    """Extract two same-position label scores and fail closed if either is absent."""
    scores = response.get("logprobs", response.get("label_logprobs"))
    if not isinstance(scores, Mapping):
        raise ValueError("joint judge response must contain label logprobs")
    if sufficient_label not in scores or insufficient_label not in scores:
        raise ValueError("joint judge response must contain both configured labels")
    try:
        first, second = float(scores[sufficient_label]), float(scores[insufficient_label])
    except (TypeError, ValueError) as exc:
        raise ValueError("joint label logprobs must be numeric") from exc
    if not math.isfinite(first) or not math.isfinite(second):
        raise ValueError("joint label logprobs must be finite")
    return first, second


def chat_label_pair(response: Mapping[str, Any], *, sufficient_label: str = "A",
                    insufficient_label: str = "B") -> tuple[float, float]:
    """Read two label log-probabilities from one chat output position.

    OpenAI-compatible chat APIs report an emitted token and its alternatives
    under ``choices[0].logprobs.content[0]``.  Looking for the two labels in
    different positions changes the event being compared, so this adapter is
    intentionally stricter than a generic log-probability parser.
    """

    if not isinstance(response, Mapping):
        raise ValueError("chat judge response must be a mapping")
    choices = response.get("choices")
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or not choices:
        raise ValueError("chat judge response must contain a choice")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise ValueError("chat judge choice must be a mapping")
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise ValueError("chat judge choice must contain a message")
    emitted_content = message.get("content")
    if not isinstance(emitted_content, str) or emitted_content.strip() not in {
        sufficient_label,
        insufficient_label,
    }:
        raise ValueError("chat judge must output exactly one configured label")
    logprobs = choice.get("logprobs")
    positions = logprobs.get("content") if isinstance(logprobs, Mapping) else None
    if (
        not isinstance(positions, Sequence)
        or isinstance(positions, (str, bytes))
        or not positions
        or not isinstance(positions[0], Mapping)
    ):
        raise ValueError("chat judge response must contain token logprobs")
    position = positions[0]
    emitted_token = position.get("token")
    if not isinstance(emitted_token, str) or emitted_token.strip() != emitted_content.strip():
        raise ValueError("chat judge decision must be the first scored output token")
    alternatives = position.get("top_logprobs")
    if not isinstance(alternatives, Sequence) or isinstance(alternatives, (str, bytes)):
        raise ValueError("chat judge output position must contain top_logprobs")
    scores: dict[str, float] = {}
    for item in alternatives:
        if not isinstance(item, Mapping):
            raise ValueError("chat judge top_logprobs entries must be mappings")
        token = item.get("token")
        if not isinstance(token, str):
            raise ValueError("chat judge top_logprobs entry is missing token")
        normalized = token.strip()
        if normalized not in {sufficient_label, insufficient_label}:
            continue
        if normalized in scores:
            raise ValueError("chat judge response contains a duplicate configured label")
        try:
            value = float(item.get("logprob"))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("chat judge label logprobs must be numeric") from exc
        if not math.isfinite(value):
            raise ValueError("chat judge label logprobs must be finite")
        scores[normalized] = value
    return require_label_pair(
        {"logprobs": scores},
        sufficient_label=sufficient_label,
        insufficient_label=insufficient_label,
    )


def preflight_joint_backend(backend: Any, *, sufficient_label: str = "A",
                            insufficient_label: str = "B") -> dict[str, Any]:
    """Fail before a run if a backend cannot provide the required label pair.

    Backends may expose ``capabilities()`` or ``probe_label_scores()``.  A
    generic truthy ``supports_logprobs`` flag is intentionally insufficient:
    the response itself must contain both labels, so a configured probe is
    always parsed with :func:`require_label_pair`.
    """
    capability = getattr(backend, "capabilities", None)
    if callable(capability):
        details = capability()
        if isinstance(details, Mapping) and details.get("same_position") is False:
            raise ValueError("joint judge backend does not score labels at one output position")
    probe = getattr(backend, "probe_label_scores", None)
    if not callable(probe):
        raise ValueError("joint judge backend has no label-score probe")
    response = probe(sufficient_label=sufficient_label, insufficient_label=insufficient_label)
    first, second = require_label_pair(response, sufficient_label=sufficient_label,
                                       insufficient_label=insufficient_label)
    return {"sufficient_label": sufficient_label, "insufficient_label": insufficient_label,
            "same_position": True, "sufficient_logprob": first, "insufficient_logprob": second}


def judge_input_hash(query: PublicQuery, raw_ids: Sequence[str], records: Mapping[str, Any]) -> str:
    payload = {"query": query.identity(), "raw_ids": list(raw_ids),
               "records": [records.get(identifier) for identifier in raw_ids]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


JOINT_PROMPT = """Read the recorded evidence jointly to assess the ORIGINAL question.
A premise may make another record useful; evaluate that relation explicitly.
Respect speaker roles, entity identity, negation, and the time requested.
Do not invent personal facts. Same-topic text alone is not sufficient.
General reasoning may connect explicitly supported personal constraints.
Decide whether these records support a specific question-appropriate answer.
Output only A (sufficient) or B (insufficient)."""

CLAIM_PROMPT = """State one concise answer-relevant claim supported jointly by the supplied records.
Use only facts in the records, preserve speaker and time distinctions, and do not guess an answer label.
Return a JSON object with exactly `text` and `source_ids`."""

VERIFY_PROMPT = """Check whether the supplied records support the FIXED claim.
Do not revise the claim and do not use outside personal facts.
Output only A (supported) or B (unsupported)."""

_JUDGE_SCHEMA = "legacy_http_joint_v1"
_TIME_KEYS = (
    "observed_start",
    "observed_end",
    "event_start",
    "event_end",
    "validity",
    "time_source",
)


def _json_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    ).hexdigest()


def _record_public_view(identifier: str, record: Any) -> dict[str, Any]:
    """Serialize only evidence/provenance fields, excluding answer metadata."""

    if isinstance(record, Mapping):
        text = record.get("text", "")
        timestamp = record.get("timestamp")
        source_id = record.get("source_id", identifier)
        metadata = record.get("metadata", {})
    else:
        text = getattr(record, "text", "")
        timestamp = getattr(record, "timestamp", None)
        source_id = getattr(record, "source_id", identifier)
        metadata = getattr(record, "metadata", {})
    if not isinstance(text, str) or not text:
        raise ValueError(f"joint judge record {identifier} has no text")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    raw_roles = metadata.get("roles", ())
    if isinstance(raw_roles, str):
        roles = [raw_roles]
    elif isinstance(raw_roles, Sequence):
        roles = [str(role) for role in raw_roles]
    else:
        roles = []
    raw_time = metadata.get("time")
    if isinstance(raw_time, Mapping):
        time_view = {str(key): raw_time[key] for key in _TIME_KEYS if key in raw_time}
    else:
        time_view = {key: metadata[key] for key in _TIME_KEYS if key in metadata}
    return {
        "id": str(identifier),
        "source_id": str(source_id),
        "text": text,
        "timestamp": timestamp,
        "roles": roles,
        "time": time_view,
    }


class HTTPJointJudge:
    """Production HTTP adapter for the historical fixed-label Chain judge.

    ``transport`` is an explicit dependency-injection seam used by offline
    tests.  When it is omitted the normal HTTP client is used; no fixture or
    heuristic response is ever selected as a fallback.
    """

    def __init__(
        self,
        config: GeneratorConfig,
        records: Mapping[str, Any],
        *,
        transport: Any | None = None,
        backend: str = "http",
        cache_dir: str | Path | None = None,
        repeat_id: str = "",
        sufficient_label: str = "A",
        insufficient_label: str = "B",
    ):
        if not isinstance(config, GeneratorConfig):
            raise ValueError("HTTPJointJudge requires a GeneratorConfig")
        if not isinstance(records, Mapping):
            raise ValueError("HTTPJointJudge records must be a mapping")
        if not isinstance(backend, str) or not backend.strip():
            raise ValueError("HTTPJointJudge backend must be a non-empty string")
        if backend == "fixture" and transport is None:
            raise ValueError("fixture backend requires an explicit fixture transport")
        if not isinstance(sufficient_label, str) or not sufficient_label.strip():
            raise ValueError("sufficient_label must be a non-empty string")
        if not isinstance(insufficient_label, str) or not insufficient_label.strip():
            raise ValueError("insufficient_label must be a non-empty string")
        if sufficient_label == insufficient_label:
            raise ValueError("joint judge labels must be distinct")
        if transport is None:
            from .clients import _post_json

            transport = _post_json
        if not callable(transport):
            raise ValueError("HTTPJointJudge transport must be callable")
        self.config = config
        self.records = records
        self.transport = transport
        self.backend = backend.strip()
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.repeat_id = str(repeat_id)
        self.sufficient_label = sufficient_label
        self.insufficient_label = insufficient_label
        self.cost: dict[str, dict[str, int]] = {
            name: {"requests": 0, "cache_hits": 0}
            for name in ("score", "claim", "verify", "preflight")
        }

    def capabilities(self) -> dict[str, Any]:
        return {
            "same_position": True,
            "labels": [self.sufficient_label, self.insufficient_label],
            "backend": self.backend,
        }

    def _headers(self) -> dict[str, str]:
        key = self.config.resolved_api_key()
        return {"Authorization": f"Bearer {key}"} if key else {}

    def _send(self, operation: str, payload: dict[str, Any]) -> Mapping[str, Any]:
        self.cost[operation]["requests"] += 1
        response = self.transport(
            self.config.endpoint,
            payload,
            self.config.timeout_seconds,
            headers=self._headers(),
        )
        if not isinstance(response, Mapping):
            raise ValueError("joint judge transport response must be a mapping")
        return response

    def _canonical_records(
        self,
        query: PublicQuery,
        raw_ids: Sequence[str],
    ) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
        if isinstance(raw_ids, (str, bytes)) or not isinstance(raw_ids, Sequence):
            raise ValueError("joint judge raw_ids must be a sequence")
        identifiers = [str(value) for value in raw_ids]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("joint judge raw_ids must be unique")
        missing = sorted(identifier for identifier in identifiers if identifier not in self.records)
        if missing:
            raise ValueError(f"joint judge received unknown record IDs: {missing}")
        visible = set(query.visible_memories)
        if visible:
            hidden = sorted(set(identifiers) - visible)
            if hidden:
                raise ValueError(f"joint judge received non-visible record IDs: {hidden}")
        views = [_record_public_view(identifier, self.records[identifier]) for identifier in identifiers]

        def chronology(item: Mapping[str, Any]) -> tuple[float, str]:
            try:
                stamp = float(item.get("timestamp"))
                if not math.isfinite(stamp):
                    stamp = float("inf")
            except (TypeError, ValueError, OverflowError):
                stamp = float("inf")
            return stamp, str(item["id"])

        views.sort(key=chronology)
        canonical = tuple(str(item["id"]) for item in views)
        return canonical, views

    def _query_view(self, query: PublicQuery) -> dict[str, Any]:
        return {
            "dataset_revision": query.dataset_revision,
            "persona_id": query.persona_id,
            "question_id": query.question_id,
            "original_question": query.query,
            "public_options": query.public_options,
            "shared_context_id": query.shared_context_id,
            "end_index": query.end_index,
            "time_metadata": list(query.time_metadata),
        }

    def _identity(
        self,
        operation: str,
        query: PublicQuery,
        canonical_ids: Sequence[str],
        views: Sequence[Mapping[str, Any]],
        *,
        fixed_claim: Claim | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "schema": _JUDGE_SCHEMA,
            "operation": operation,
            "backend": self.backend,
            "repeat_id": self.repeat_id,
            "endpoint_sha256": hashlib.sha256(self.config.endpoint.encode("utf-8")).hexdigest(),
            "model": self.config.model,
            "labels": [self.sufficient_label, self.insufficient_label],
            "query": self._query_view(query),
            "canonical_ids": list(canonical_ids),
            "records": list(views),
        }
        if fixed_claim is not None:
            payload["fixed_claim"] = {
                "text": fixed_claim.text,
                "source_ids": list(fixed_claim.source_ids),
                "input_hash": fixed_claim.input_hash,
            }
        return _json_hash(payload)

    def _cache_path(self, operation: str, input_hash: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{operation}_{input_hash}.json"

    def _read_cache(self, operation: str, input_hash: str) -> Mapping[str, Any] | None:
        path = self._cache_path(operation, input_hash)
        if path is None or not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid joint judge cache entry: {path}") from exc
        if not isinstance(value, Mapping) or value.get("input_hash") != input_hash:
            raise ValueError(f"joint judge cache identity mismatch: {path}")
        self.cost[operation]["cache_hits"] += 1
        return value

    def _write_cache(self, operation: str, input_hash: str, value: Mapping[str, Any]) -> None:
        path = self._cache_path(operation, input_hash)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(dict(value), ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def _label_payload(self, prompt: str, public: Mapping[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(public, ensure_ascii=False, sort_keys=True)},
            ],
            "temperature": 0.0,
            "max_tokens": 1,
            "logprobs": True,
            "top_logprobs": 20,
        }
        if not self.config.model:
            payload.pop("model")
        return payload

    def probe_label_scores(self, **_: Any) -> Mapping[str, Any]:
        public = {
            "schema": _JUDGE_SCHEMA,
            "original_question": "Backend label-score capability check.",
            "public_options": "",
            "records": [],
        }
        response = self._send("preflight", self._label_payload(JOINT_PROMPT, public))
        first, second = chat_label_pair(
            response,
            sufficient_label=self.sufficient_label,
            insufficient_label=self.insufficient_label,
        )
        return {"logprobs": {self.sufficient_label: first, self.insufficient_label: second}}

    def score(self, query: PublicQuery, raw_ids: tuple[str, ...]) -> JointScore:
        canonical, views = self._canonical_records(query, raw_ids)
        input_hash = self._identity("score", query, canonical, views)
        cached = self._read_cache("score", input_hash)
        if cached is not None:
            first, second = require_label_pair(
                cached,
                sufficient_label=self.sufficient_label,
                insufficient_label=self.insufficient_label,
            )
        else:
            public = {**self._query_view(query), "schema": _JUDGE_SCHEMA, "records": views}
            response = self._send("score", self._label_payload(JOINT_PROMPT, public))
            first, second = chat_label_pair(
                response,
                sufficient_label=self.sufficient_label,
                insufficient_label=self.insufficient_label,
            )
            self._write_cache(
                "score",
                input_hash,
                {
                    "input_hash": input_hash,
                    "logprobs": {
                        self.sufficient_label: first,
                        self.insufficient_label: second,
                    },
                },
            )
        return JointScore.from_logits(
            first,
            second,
            input_hash=input_hash,
            sufficient_label=self.sufficient_label,
            insufficient_label=self.insufficient_label,
        )

    def claim(self, query: PublicQuery, raw_ids: tuple[str, ...]) -> Claim:
        canonical, views = self._canonical_records(query, raw_ids)
        if not canonical:
            raise ValueError("cannot create a claim from an empty evidence set")
        input_hash = self._identity("claim", query, canonical, views)
        cached = self._read_cache("claim", input_hash)
        if cached is None:
            public = {**self._query_view(query), "schema": _JUDGE_SCHEMA, "records": views}
            payload: dict[str, Any] = {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": CLAIM_PROMPT},
                    {"role": "user", "content": json.dumps(public, ensure_ascii=False, sort_keys=True)},
                ],
                "temperature": 0.0,
                "max_tokens": min(self.config.max_tokens, 256),
                "response_format": {"type": "json_object"},
            }
            if not self.config.model:
                payload.pop("model")
            response = self._send("claim", payload)
            choices = response.get("choices")
            try:
                content = choices[0]["message"]["content"]
                decoded = json.loads(content)
            except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError("claim response must contain a JSON object") from exc
            if not isinstance(decoded, Mapping):
                raise ValueError("claim response must decode to an object")
            text = decoded.get("text")
            source_ids = decoded.get("source_ids")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("claim response must contain non-empty text")
            if (
                not isinstance(source_ids, Sequence)
                or isinstance(source_ids, (str, bytes))
                or not source_ids
            ):
                raise ValueError("claim response must contain source_ids")
            normalized_sources = tuple(dict.fromkeys(str(value) for value in source_ids))
            if not set(normalized_sources) <= set(canonical):
                raise ValueError("claim response cites evidence outside the supplied set")
            cached = {
                "input_hash": input_hash,
                "text": text.strip(),
                "source_ids": list(normalized_sources),
            }
            self._write_cache("claim", input_hash, cached)
        text = cached.get("text")
        source_ids = cached.get("source_ids")
        if not isinstance(text, str) or not text:
            raise ValueError("cached claim text is invalid")
        if not isinstance(source_ids, Sequence) or isinstance(source_ids, (str, bytes)):
            raise ValueError("cached claim source_ids are invalid")
        normalized_sources = tuple(str(value) for value in source_ids)
        if not normalized_sources or not set(normalized_sources) <= set(canonical):
            raise ValueError("cached claim cites evidence outside the supplied set")
        return Claim(text, normalized_sources, input_hash)

    def verify(
        self,
        query: PublicQuery,
        fixed_claim: Claim,
        raw_ids: tuple[str, ...],
    ) -> Verification:
        if not isinstance(fixed_claim, Claim) or not fixed_claim.text:
            raise ValueError("verify requires a fixed Claim")
        canonical, views = self._canonical_records(query, raw_ids)
        input_hash = self._identity(
            "verify",
            query,
            canonical,
            views,
            fixed_claim=fixed_claim,
        )
        cached = self._read_cache("verify", input_hash)
        if cached is not None:
            first, second = require_label_pair(
                cached,
                sufficient_label=self.sufficient_label,
                insufficient_label=self.insufficient_label,
            )
            supported = cached.get("supported")
            if not isinstance(supported, bool):
                raise ValueError("cached verification decision is invalid")
        else:
            public = {
                **self._query_view(query),
                "schema": _JUDGE_SCHEMA,
                "fixed_claim": fixed_claim.text,
                "claim_source_ids": list(fixed_claim.source_ids),
                "records": views,
            }
            response = self._send("verify", self._label_payload(VERIFY_PROMPT, public))
            first, second = chat_label_pair(
                response,
                sufficient_label=self.sufficient_label,
                insufficient_label=self.insufficient_label,
            )
            choices = response.get("choices")
            emitted = choices[0]["message"]["content"].strip()
            supported = emitted == self.sufficient_label
            self._write_cache(
                "verify",
                input_hash,
                {
                    "input_hash": input_hash,
                    "supported": supported,
                    "logprobs": {
                        self.sufficient_label: first,
                        self.insufficient_label: second,
                    },
                },
            )
        label = self.sufficient_label if supported else self.insufficient_label
        return Verification(bool(supported), f"label:{label}", input_hash)
