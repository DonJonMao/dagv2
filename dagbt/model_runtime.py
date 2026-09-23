"""Explicit protocol adaptation for the current BridgeTree model deployment.

The BridgeTree generator uses chat messages and a tokenizer-independent regex
estimate.  A recoverable message envelope satisfies the frozen DAG v2 API; it
is not a DeepSeek tokenizer or a Qwen chat template.  Wire responses remain
unchanged in the transport cache and are adapted only for legacy consumers.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Any

from vendor.bridgetree.clients import estimate_tokens
from vendor.bridgetree.diagnostic_identity import validate_provider_params


ROOT = Path(__file__).resolve().parents[1]
MESSAGE_ENVELOPE = "DAGBT_MESSAGES_JSON_V1\n"
TOKEN_ESTIMATOR_ID = "regex_word_or_punctuation_v1"


def is_bridgetree(config: Mapping[str, Any]) -> bool:
    return config.get("model_profile") == "bridgetree"


def _messages(value: Any) -> list[dict[str, str]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ValueError("Chat messages must be a nonempty sequence")
    result = []
    for message in value:
        if (not isinstance(message, Mapping) or set(message) != {"role", "content"}
                or message["role"] not in {"system", "user", "assistant"}
                or not isinstance(message["content"], str)):
            raise ValueError("Chat messages require a supported role and text content")
        result.append(dict(message))
    return result


def _recover_messages(prompt: Any) -> list[dict[str, str]]:
    if not isinstance(prompt, str) or not prompt.startswith(MESSAGE_ENVELOPE):
        raise ValueError("BridgeTree completion adaptation requires the DAGBT message envelope")
    try:
        value = json.loads(prompt[len(MESSAGE_ENVELOPE):])
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid DAGBT message envelope; legacy thinking continuations are unsupported") from exc
    if not isinstance(value, Mapping) or set(value) != {"messages"}:
        raise ValueError("Invalid DAGBT message envelope fields")
    return _messages(value["messages"])


class BTTokenAccounting:
    """Length-compatible token estimator, with no invented model token IDs."""

    name_or_path = "estimate:regex_word_or_punctuation_v1"
    token_count_is_estimate = True

    def apply_chat_template(self, messages, *, tokenize=False, **kwargs):
        if kwargs.get("enable_thinking") is True:
            raise ValueError("BridgeTree model profile supports the actual no-thinking DAG v2 flow only")
        rendered = MESSAGE_ENVELOPE + json.dumps(
            {"messages": _messages(messages)}, ensure_ascii=False, separators=(",", ":"))
        return self.encode(rendered) if tokenize else rendered

    def encode(self, text: str, **_kwargs) -> range:
        if text.startswith(MESSAGE_ENVELOPE):
            text = json.dumps(_recover_messages(text), ensure_ascii=False)
        # Existing callers only need len(...). A range deliberately does not
        # imply that the returned values are actual vocabulary token IDs.
        return range(estimate_tokens(text))


def load_tokenizer(config: Mapping[str, Any]):
    if is_bridgetree(config):
        return BTTokenAccounting()
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(ROOT / "package" / "tokenizer"), local_files_only=True)


def token_accounting(config: Mapping[str, Any]) -> dict[str, Any]:
    if is_bridgetree(config):
        return {"token_count_is_estimate": True, "token_estimator_id": TOKEN_ESTIMATOR_ID,
                "tokenizer_id": None,
                "token_accounting_note": "BridgeTree regex estimate; not the deployed DeepSeek tokenizer"}
    return {"token_count_is_estimate": False,
            "tokenizer_path": str(ROOT / "package" / "tokenizer")}


def prepare_request(stage, url: str, payload: Mapping[str, Any], config: Mapping[str, Any]):
    """Return exact wire URL/payload and whether legacy choices.text is needed."""
    wire = deepcopy(dict(payload))
    embed = url.rstrip("/").endswith("/embeddings")
    rerank = (url == config.get("reranker", {}).get("url")
              or url.rstrip("/").endswith(("/rerank", "/reranks")))
    if embed:
        wire["model"] = config["embedding_model"]
        return url, wire, False
    if rerank:
        model = config.get("reranker", {}).get("model", "")
        if model:
            wire["model"] = model
        else:
            wire.pop("model", None)
        return url, wire, False
    if not is_bridgetree(config):
        wire.update(model=config["llm_model"], temperature=0, top_p=1, seed=20260918)
        return url, wire, False

    legacy_text = not url.rstrip("/").endswith("/chat/completions")
    if legacy_text:
        if not url.rstrip("/").endswith("/completions") or "messages" in wire:
            raise ValueError("Unsupported BridgeTree generator endpoint or mixed completion payload")
        messages = _recover_messages(wire.get("prompt"))
        url = url.rstrip("/")[:-len("/completions")] + "/chat/completions"
    else:
        if "prompt" in wire:
            raise ValueError("BridgeTree chat request cannot contain a native prompt")
        messages = _messages(wire.get("messages"))
    if wire.get("chat_template_kwargs", {}).get("enable_thinking") is True:
        raise ValueError("Legacy thinking chat requests are unsupported by this no-thinking adapter")
    structured = wire.get("structured_outputs")
    if structured is not None:
        if not isinstance(structured, Mapping) or set(structured) != {"json"} or not isinstance(structured["json"], Mapping):
            raise ValueError("Only explicit JSON schemas can be adapted to BridgeTree chat")
        instruction = "Output only a JSON value matching this schema, without Markdown fences:\n" + json.dumps(
            structured["json"], ensure_ascii=False, sort_keys=True, allow_nan=False)
        if messages[0]["role"] == "system":
            messages[0]["content"] += "\n\n" + instruction
        else:
            messages.insert(0, {"role": "system", "content": instruction})
    output_tokens = wire.get("max_tokens")
    if isinstance(output_tokens, bool) or not isinstance(output_tokens, int) or output_tokens < 1:
        raise ValueError("BridgeTree generator request requires positive max_tokens")
    provider = validate_provider_params(config.get("llm_provider_request_params", {}))
    # Only the actual BT provider configuration may add generation options;
    # Qwen stops, local templates, sampling and guided-output fields are not
    # propagated from the original algorithm's payload.
    wire = {"model": config["llm_model"], "messages": messages,
            "temperature": 0.0, "max_tokens": output_tokens, **provider}
    return url, wire, legacy_text


def count_request_tokens(stage, url, payload, config, tokenizer=None) -> int:
    """Count/estimate the final wire messages, including schema instructions."""
    _, wire, _ = prepare_request(stage, url, payload, config)
    if is_bridgetree(config):
        return estimate_tokens(json.dumps(wire["messages"], ensure_ascii=False))
    tokenizer = tokenizer or load_tokenizer(config)
    prompt = wire.get("prompt")
    if prompt is None:
        prompt = tokenizer.apply_chat_template(
            wire["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return len(tokenizer.encode(prompt, add_special_tokens=False))


def normalize_response(response: Mapping[str, Any], legacy_text_response: bool = False) -> dict[str, Any]:
    """Copy the response; never mutate the exact cached provider envelope."""
    result = deepcopy(dict(response))
    if legacy_text_response:
        choices = result.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("Adapted chat response requires choices")
        for choice in choices:
            content = choice.get("message", {}).get("content") if isinstance(choice, Mapping) else None
            if not isinstance(content, str):
                raise ValueError("Adapted chat response requires text message.content")
            choice["text"] = content
    return result


def resolve_api_key(config: Mapping[str, Any], kind: str) -> str:
    """Environment first, then an endpoint-bound ignored local credential file."""
    env_names = {"llm": ("DAG_LLM_API_KEY", "BRIDGETREE_CHAT_API_KEY"),
                 "embedding": ("DAG_EMBED_API_KEY",), "reranker": ("DAG_RERANK_API_KEY",)}
    if kind not in env_names:
        raise ValueError("Unknown model service kind")
    for name in env_names[kind]:
        if os.environ.get(name):
            return os.environ[name]
    configured = config.get("credentials_file")
    if not configured:
        return ""
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return ""
    try:
        private = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError("Cannot read local model credentials JSON") from exc
    if not isinstance(private, Mapping):
        raise ValueError("Local model credentials must be an object")
    item = private.get(kind, {})
    if not isinstance(item, Mapping):
        raise ValueError("Local model credential entry must be an object")
    endpoint = (str(config.get("llm_base_url", "")).rstrip("/") + "/chat/completions" if kind == "llm"
                else str(config.get("embedding_base_url", "")).rstrip("/") + "/embeddings" if kind == "embedding"
                else config.get("reranker", {}).get("url"))
    if item.get("endpoint") != endpoint:
        return ""
    secret = item.get("api_key", "")
    if not isinstance(secret, str) or "\n" in secret or "\r" in secret:
        raise ValueError("Local model API key must be a single-line string")
    return secret
