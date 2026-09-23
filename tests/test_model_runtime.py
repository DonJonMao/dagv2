"""Model migration preserves wire/cache identity and the actual BT protocol."""
from copy import deepcopy
import json

import numpy as np
import pytest

from dagbt.bridge import _Embedder, _Reranker
from dagbt.budget import Ledger
from dagbt.model_runtime import (
    BTTokenAccounting, MESSAGE_ENVELOPE, count_request_tokens, is_bridgetree,
    load_tokenizer, normalize_response, prepare_request, resolve_api_key, token_accounting,
)
from dagbt.transport import Transport
from vendor.bridgetree.clients import estimate_tokens
from vendor.bridgetree.dependency_scoring import SetResponseError
from test_bridge import session
from test_bridge_http import server


def profile(endpoint="http://fixture"):
    return {"model_profile": "bridgetree", "llm_model": "deepseek-v4-flash",
            "llm_base_url": endpoint + "/v1", "embedding_model": "qwen3-embedding-8b",
            "embedding_base_url": endpoint + "/v1", "llm_provider_request_params": {},
            "reranker": {"url": endpoint + "/rerank", "model": "", "max_batch_documents": 4},
            "max_identical_attempts": 1, "request_timeout_seconds": 3,
            "fusion": {"reserved_audit_calls": 24}}


def native_payload():
    messages = [{"role": "system", "content": "Solve the current evidence-grounded task."},
                {"role": "user", "content": "Preserve 原始 evidence exactly."}]
    return {"prompt": BTTokenAccounting().apply_chat_template(messages, tokenize=False),
            "model": "old-qwen", "max_tokens": 512, "stop": ["<|im_end|>"],
            "top_p": .8, "seed": 1, "temperature": .7,
            "chat_template_kwargs": {"enable_thinking": False},
            "structured_outputs": {"json": {"type": "object", "properties": {
                "answer": {"type": "string"}, "sources": {"type": "array", "items": {"type": "boolean"}}}}}}


def test_frozen_native_prompt_is_recovered_as_chat_with_schema_and_exact_bt_params():
    config = profile()
    payload = native_payload()
    before = deepcopy(payload)
    url, wire, legacy = prepare_request("node", "http://fixture/v1/completions", payload, config)
    assert url == "http://fixture/v1/chat/completions" and legacy
    assert set(wire) == {"model", "messages", "temperature", "max_tokens"}
    assert wire["model"] == "deepseek-v4-flash" and wire["temperature"] == 0
    assert wire["messages"][0]["content"].startswith("Solve the current evidence-grounded task.")
    assert "Output only a JSON value matching this schema" in wire["messages"][0]["content"]
    assert wire["messages"][1]["content"] == "Preserve 原始 evidence exactly."
    assert MESSAGE_ENVELOPE not in json.dumps(wire)
    assert "<|im_end|>" not in json.dumps(wire)
    assert payload == before
    assert count_request_tokens("node", "http://fixture/v1/completions", payload, config) == estimate_tokens(
        json.dumps(wire["messages"], ensure_ascii=False))


def test_bt_accounting_is_explicit_estimate_and_never_loads_qwen_tokenizer():
    tokenizer = load_tokenizer(profile())
    assert isinstance(tokenizer, BTTokenAccounting)
    messages = [{"role": "user", "content": "A factual question?"}]
    rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
    assert rendered.startswith(MESSAGE_ENVELOPE)
    assert len(tokenizer.encode(rendered)) == estimate_tokens(json.dumps(messages, ensure_ascii=False))
    assert token_accounting(profile())["token_count_is_estimate"] is True
    assert token_accounting(profile())["tokenizer_id"] is None
    assert "estimate" in tokenizer.name_or_path
    assert is_bridgetree(profile()) and not is_bridgetree({})
    with pytest.raises(ValueError, match="no-thinking"):
        tokenizer.apply_chat_template(messages, enable_thinking=True)


@pytest.mark.parametrize("prompt", ["<|im_start|>user\nquestion", MESSAGE_ENVELOPE + "{}",
                                   native_payload()["prompt"] + "\nlegacy thinking continuation"])
def test_no_silent_interpretation_of_qwen_templates_or_thinking_continuations(prompt):
    with pytest.raises(ValueError):
        prepare_request("node", "http://fixture/v1/completions", {"prompt": prompt, "max_tokens": 5}, profile())


def test_only_explicit_provider_params_extend_bt_payload():
    config = profile()
    config["llm_provider_request_params"] = {"reasoning_effort": "low"}
    _, wire, _ = prepare_request("node", "http://fixture/v1/completions", native_payload(), config)
    assert wire["reasoning_effort"] == "low"
    assert "seed" not in wire and "top_p" not in wire
    config["llm_provider_request_params"] = {"headers": {"Authorization": "fixture-secret"}}
    with pytest.raises(ValueError, match="credentials"):
        prepare_request("node", "http://fixture/v1/completions", native_payload(), config)
    config["llm_provider_request_params"] = {"model": "unexpected-model"}
    with pytest.raises(ValueError, match="canonical"):
        prepare_request("node", "http://fixture/v1/completions", native_payload(), config)


def test_normalization_preserves_exact_raw_response_and_usage():
    raw = {"choices": [{"message": {"role": "assistant", "content": '{"answer":"Paris"}'},
                        "finish_reason": "stop"}], "usage": {"prompt_tokens": 9, "completion_tokens": 4}}
    adapted = normalize_response(raw, True)
    assert adapted["choices"][0]["text"] == raw["choices"][0]["message"]["content"]
    assert adapted["usage"] == raw["usage"]
    assert "text" not in raw["choices"][0]
    assert "</think>" not in adapted["choices"][0]["text"]


def test_original_unbounded_transport_caches_wire_chat_and_adapts_only_return(tmp_path):
    with server() as (endpoint, requests):
        config = profile(endpoint)
        ledger = Ledger({})
        calls = Transport("native-question", tmp_path, config, ledger, load_tokenizer(config))
        result = calls.get(("solve_dag", "node", "sources"), endpoint + "/v1/completions", native_payload())
        assert result["response"]["choices"][0]["text"] == "{}"
        assert len(requests) == 1 and requests[0]["path"] == "/v1/chat/completions"
        record = json.loads(next((tmp_path / "requests").glob("*.json")).read_text())
        assert record["url"] == endpoint + "/v1/chat/completions"
        assert record["payload"] == result["request"] == requests[0]["payload"]
        assert "text" not in record["response"]["choices"][0]
        replay = calls.get(("solve_dag", "node", "sources"), endpoint + "/v1/completions", native_payload())
        assert replay == result and len(requests) == 1
        # Equal wire request can also be consumed as native chat without cache
        # contamination by the legacy choices.text response adapter.
        chat = calls.get("chat", endpoint + "/v1/chat/completions", result["request"])
        assert "text" not in chat["response"]["choices"][0]
        assert len(requests) == ledger.used["http_attempts"] == 1
        assert ledger.used["cache_hits"] == 2 and ledger.used["llm"] == 3


def test_bt_wire_budget_boundary_includes_schema_output_and_rechecks_cached_request(tmp_path):
    from dagbt.transport import ResponseError
    with server() as (endpoint, requests):
        config = profile(endpoint)
        payload = native_payload()
        url = endpoint + "/v1/completions"
        input_estimate = count_request_tokens("node", url, payload, config)
        assert input_estimate > len(load_tokenizer(config).encode(payload["prompt"]))
        config["fusion"]["context_tokens"] = input_estimate + payload["max_tokens"] + 8
        ledger = Ledger({})
        calls = Transport("boundary", tmp_path, config, ledger, load_tokenizer(config))
        calls.get("node", url, payload)
        assert len(requests) == 1
        config["fusion"]["context_tokens"] -= 1
        with pytest.raises(ResponseError, match="regex estimate, not the deployed model tokenizer"):
            calls.get("node", url, payload)
        assert len(requests) == 1 and ledger.used["cache_hits"] == 0
        events = [event for event in ledger.events if event["event"] == "model_context_budget"]
        assert [event["within_estimated_budget"] for event in events] == [True, False]
        assert all(event["token_count_is_estimate"] is True for event in events)
        assert events[-1]["input_tokens_local"] == input_estimate


def test_endpoint_bound_credentials_env_precedence_and_no_payload_secret(tmp_path, monkeypatch):
    for name in ("DAG_LLM_API_KEY", "BRIDGETREE_CHAT_API_KEY", "DAG_EMBED_API_KEY", "DAG_RERANK_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    config = profile()
    path = tmp_path / "credentials.json"
    config["credentials_file"] = str(path)
    path.write_text(json.dumps({"llm": {"endpoint": "http://fixture/v1/chat/completions", "api_key": "file-secret"},
                               "embedding": {"endpoint": "http://fixture/v1/embeddings", "api_key": "embedding-secret"},
                               "reranker": {"endpoint": "http://fixture/rerank", "api_key": "rerank-secret"}}))
    assert resolve_api_key(config, "llm") == "file-secret"
    assert resolve_api_key(config, "embedding") == "embedding-secret"
    assert resolve_api_key(config, "reranker") == "rerank-secret"
    monkeypatch.setenv("BRIDGETREE_CHAT_API_KEY", "bt-env-secret")
    assert resolve_api_key(config, "llm") == "bt-env-secret"
    monkeypatch.setenv("DAG_LLM_API_KEY", "dag-env-secret")
    assert resolve_api_key(config, "llm") == "dag-env-secret"
    monkeypatch.delenv("DAG_LLM_API_KEY"); monkeypatch.delenv("BRIDGETREE_CHAT_API_KEY")
    config["llm_base_url"] = "http://different-service/v1"
    assert resolve_api_key(config, "llm") == ""
    _, wire, _ = prepare_request("node", "http://different-service/v1/completions", native_payload(), config)
    assert "secret" not in json.dumps(wire)


def test_reranker_empty_model_omission_and_physical_batches_restore_global_indices():
    class Calls:
        def __init__(self): self.requests = []
        def get(self, stage, url, payload):
            self.requests.append(payload)
            return {"response": {"results": [
                {"index": i, "relevance_score": int(text) / 10}
                for i, text in reversed(list(enumerate(payload["documents"])))]}}
    calls = Calls()
    backend = _Reranker(calls, {"url": "http://fixture/rerank", "model": "", "max_batch_documents": 99}, Ledger({}))
    result = backend.rerank_all("query", [str(i) for i in range(9)])
    assert [len(p["documents"]) for p in calls.requests] == [4, 4, 1]
    assert all("model" not in p and p["top_n"] == len(p["documents"]) for p in calls.requests)
    assert result["results"] == [{"index": i, "relevance_score": i / 10} for i in range(9)]
    _, wire, _ = prepare_request("rerank", "http://fixture/rerank", {"model": "old", **calls.requests[0]}, profile())
    assert "model" not in wire


@pytest.mark.parametrize("bad", [
    {"truncated": True, "results": [{"index": 0, "score": .5}]},
    {"results": [{"index": 1, "score": .5}]},
    {"results": [{"index": 0, "score": float("nan")}]},
    {"results": [{"index": 0, "score": 1.5}]},
])
def test_physical_rerank_batch_retains_source_validation(bad):
    class Calls:
        def get(self, *_): return {"response": bad}
    backend = _Reranker(Calls(), {"url": "http://fixture/rerank"}, Ledger({}))
    with pytest.raises(SetResponseError):
        backend.rerank_all("query", ["doc"])


def test_bridge_query_embedding_matches_bt_l2_normalization_and_index_contract():
    s = session()
    s.calls.get = lambda *_: {"response": {"data": [{"index": 0, "embedding": [3, 4, 0, 0]}]}}
    vector = _Embedder(s, "node").encode_query("question")
    np.testing.assert_allclose(vector, [.6, .8, 0, 0], atol=1e-7)
    s.calls.get = lambda *_: {"response": {"data": [{"index": 1, "embedding": [3, 4, 0, 0]}]}}
    with pytest.raises(ValueError, match="index zero"):
        _Embedder(s, "node").encode_query("question")
