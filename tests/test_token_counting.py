from copy import deepcopy
import io
import json
import urllib.error

import pytest

from dagbt.token_counting import (DeploymentTokenCounter, TokenizationError,
                                  validate_tokenization_config)
import dagbt.token_counting as counting


def configuration(**extra):
    return {"model_profile": "bridgetree", "llm_model": "test-deployment",
            "llm_base_url": "http://fixture:8006/v1",
            "llm_provider_request_params": {},
            "fusion": {"context_tokens": 16384, "response_format": "plain"},
            "tokenization": {"mode": "vllm_chat", "endpoint": "http://fixture:8006/tokenize",
                             "timeout_seconds": 15}, **extra}


def wire(text="Evidence: 汉字.\nReturn JSON matching {\"type\":\"object\"}."):
    return {"model": "test-deployment", "temperature": 0, "max_tokens": 4096,
            "messages": [{"role": "system", "content": text},
                         {"role": "user", "content": "Question?"}]}


class Response(io.BytesIO):
    def getcode(self):
        return 200


class Opener:
    def __init__(self, bodies=None):
        self.bodies = list(bodies or [{"count": 73, "max_model_len": 120000}])
        self.requests = []

    def open(self, request, timeout):
        self.requests.append((request, timeout))
        item = self.bodies.pop(0)
        if isinstance(item, BaseException):
            raise item
        return Response(item if isinstance(item, bytes) else json.dumps(item).encode())


def counter(monkeypatch, bodies=None, config=None):
    opener = Opener(bodies)
    monkeypatch.setattr(counting.urllib.request, "build_opener", lambda *args: opener)
    monkeypatch.setattr(counting, "resolve_api_key", lambda config, kind: "private-test-key")
    events = []
    return DeploymentTokenCounter(config or configuration(), events.append), opener, events


def test_opt_in_and_config_validation_are_offline():
    assert validate_tokenization_config({}) is None
    assert validate_tokenization_config(configuration())["timeout_seconds"] == 15
    with pytest.raises(TokenizationError, match="not_enabled"):
        DeploymentTokenCounter({}, lambda event: None)


@pytest.mark.parametrize("patch", [
    {"tokenization": None},
    {"tokenization": {"mode": "automatic", "endpoint": "http://fixture:8006/tokenize"}},
    {"tokenization": {"mode": "vllm_chat", "endpoint": "http://other:8006/tokenize"}},
    {"tokenization": {"mode": "vllm_chat", "endpoint": "https://fixture:8006/tokenize"}},
    {"tokenization": {"mode": "vllm_chat", "endpoint": "http://fixture:8007/tokenize"}},
    {"tokenization": {"mode": "vllm_chat", "endpoint": "http://fixture:8006/v1/tokenize"}},
    {"tokenization": {"mode": "vllm_chat", "endpoint": "http://u:p@fixture:8006/tokenize"}},
    {"tokenization": {"mode": "vllm_chat", "endpoint": "http://fixture:8006/tokenize?key=secret"}},
    {"tokenization": {"mode": "vllm_chat", "endpoint": "http://fixture:8006/tokenize#x"}},
    {"tokenization": {"mode": "vllm_chat", "endpoint": "http://fixture:8006/tokenize", "extra": 1}},
    {"tokenization": {"mode": "vllm_chat", "endpoint": "http://fixture:8006/tokenize", "timeout_seconds": True}},
    {"tokenization": {"mode": "vllm_chat", "endpoint": "http://fixture:8006/tokenize", "timeout_seconds": 0}},
    {"tokenization": {"mode": "vllm_chat", "endpoint": "http://fixture:8006/tokenize", "timeout_seconds": float("nan")}},
    {"model_profile": "other"},
    {"fusion": {"response_format": "json_schema", "context_tokens": 16384}},
    {"fusion": {"response_format": "plain", "context_tokens": True}},
    {"llm_provider_request_params": {"chat_template_kwargs": {"enable_thinking": False}}},
    {"llm_provider_request_params": {"tools": []}},
])
def test_invalid_or_unverified_configuration_rejected_without_network(patch):
    with pytest.raises(TokenizationError):
        validate_tokenization_config(configuration(**patch))


def test_final_messages_schema_and_default_template_are_counted_without_logging_text(monkeypatch, tmp_path):
    body = {"count": 73, "max_model_len": 120000,
            "tokens": [11001, 11002], "token_strs": ["private-token-text"]}
    count, opener, events = counter(monkeypatch, [body])
    original = wire()
    before = deepcopy(original)
    result = count.count(original)
    request, timeout = opener.requests[0]
    assert json.loads(request.data) == {"model": original["model"], "messages": original["messages"],
                                        "add_generation_prompt": True}
    assert timeout == 15
    assert request.get_header("Authorization") == "Bearer private-test-key"
    assert original == before
    assert result["count"] == 73 and result["max_model_len"] == 120000
    assert result["token_count_is_estimate"] is False
    assert "not independently verified" in result["token_accounting_note"]
    assert count.context_limit == 16384
    event = events[0]
    assert event["kind"] == "tokenization_http" and event["status"] == "ok"
    assert event["http_attempts"] == event["successful_http_requests"] == 1
    assert event["seconds"] >= 0
    logged = json.dumps(events)
    for private in ("private-test-key", "private-token-text", "11001", "Evidence:", "Question?"):
        assert private not in logged
    assert not list(tmp_path.iterdir())


def test_cache_shares_preflight_request_transport_counts_but_not_changed_messages(monkeypatch):
    count, opener, events = counter(monkeypatch, [
        {"count": 73, "max_model_len": 120000}, {"count": 74, "max_model_len": 120000}])
    first = count.count(wire())
    changed_sampling = {**wire(), "temperature": 1, "max_tokens": 100, "top_p": .9, "seed": 4}
    second = count.count(changed_sampling)
    first["count"] = 999
    third = count.count(wire())
    fourth = count.count(wire("Different inline schema"))
    assert second["count"] == third["count"] == 73 and fourth["count"] == 74
    assert second["tokenization_request_sha256"] == third["tokenization_request_sha256"]
    assert second["tokenization_wire_sha256"] != third["tokenization_wire_sha256"]
    assert len(opener.requests) == 2
    assert [event["http_attempts"] for event in events] == [1, 0, 0, 1]
    assert [event["cache_hit"] for event in events] == [False, True, True, False]


def test_lru_is_bounded_and_no_cache_crosses_instances(monkeypatch):
    monkeypatch.setattr(counting, "MAX_CACHE_ENTRIES", 2)
    count, opener, _ = counter(monkeypatch, [{"count": 1, "max_model_len": 16384}] * 6)
    for text in ("one", "two", "one", "three", "two"):
        count.count(wire(text))
    assert len(opener.requests) == 4
    assert len(count._cache) == 2
    independent = DeploymentTokenCounter(configuration(), lambda event: None)
    independent.count(wire("two"))
    assert len(opener.requests) == 5


@pytest.mark.parametrize("field,value", [
    ("response_format", {"type": "json_schema"}), ("response_format", {"type": "json_object"}),
    ("tools", []), ("tool_choice", "none"), ("chat_template", "custom"),
    ("chat_template_kwargs", {}), ("continue_final_message", False),
    ("add_generation_prompt", True), ("structured_outputs", {"json": {}}),
    ("prompt", "text"), ("unknown_sampling_parameter", 1),
])
def test_unverified_wire_fields_fail_closed_and_are_recorded(monkeypatch, field, value):
    count, opener, events = counter(monkeypatch)
    with pytest.raises(TokenizationError, match="unsupported_wire_fields"):
        count.count({**wire(), field: value})
    assert not opener.requests
    assert len(events) == 1 and events[0]["status"] == "error"
    assert events[0]["http_attempts"] == 0


@pytest.mark.parametrize("messages", [
    [], [{"role": "user", "content": [{"type": "text", "text": "x"}]}],
    [{"role": "tool", "content": "x"}], [{"role": "user", "content": "x", "name": "x"}],
])
def test_multimodal_tool_or_extended_messages_are_not_silently_simplified(monkeypatch, messages):
    count, opener, _ = counter(monkeypatch)
    with pytest.raises(TokenizationError):
        count.count({**wire(), "messages": messages})
    assert not opener.requests


def test_model_change_cannot_reuse_count(monkeypatch):
    count, opener, _ = counter(monkeypatch)
    count.count(wire())
    with pytest.raises(TokenizationError, match="model_mismatch"):
        count.count({**wire(), "model": "another-model"})
    assert len(opener.requests) == 1


@pytest.mark.parametrize("body", [
    {"count": True, "max_model_len": 120000}, {"count": -1, "max_model_len": 120000},
    {"count": 1.5, "max_model_len": 120000}, {"count": "7", "max_model_len": 120000},
    {"count": 7, "max_model_len": True}, {"count": 7, "max_model_len": 0},
    {"count": 7}, [7], b"not JSON private-test-key", {"count": 7, "max_model_len": 8192},
])
def test_malformed_or_insufficient_deployment_responses_never_enter_cache(monkeypatch, body):
    count, opener, events = counter(monkeypatch, [body, {"count": 7, "max_model_len": 16384}])
    with pytest.raises(TokenizationError) as error:
        count.count(wire())
    assert "private-test-key" not in str(error.value)
    assert count.count(wire())["count"] == 7
    assert len(opener.requests) == 2 and not count._cache.get("invalid")
    assert events[0]["status"] == "error" and events[0]["http_attempts"] == 1


@pytest.mark.parametrize("error", [
    urllib.error.HTTPError("private-test-key", 503, "private-test-key", {}, io.BytesIO(b"secret body")),
    urllib.error.URLError("private-test-key"), TimeoutError("private-test-key"),
    RuntimeError("private-test-key"), TokenizationError("redirect_not_allowed"),
])
def test_network_failures_have_no_retry_fallback_or_sensitive_exception(monkeypatch, error):
    count, opener, events = counter(monkeypatch, [error])
    with pytest.raises(TokenizationError) as failure:
        count.count(wire())
    assert len(opener.requests) == 1 and not count._cache
    assert len(events) == 1 and events[0]["status"] == "error"
    assert events[0]["http_attempts"] == 1 and events[0]["successful_http_requests"] == 0
    assert "private-test-key" not in str(failure.value) + json.dumps(events)


def test_redirect_and_environment_proxy_are_disabled(monkeypatch):
    captured = []
    monkeypatch.setattr(counting.urllib.request, "build_opener", lambda *args: captured.extend(args))
    DeploymentTokenCounter(configuration(), lambda event: None)
    assert next(x for x in captured if isinstance(x, counting.urllib.request.ProxyHandler)).proxies == {}
    redirect = next(x for x in captured if isinstance(x, counting._NoRedirect))
    with pytest.raises(TokenizationError, match="redirect_not_allowed"):
        redirect.redirect_request(None, None, 302, "private-test-key", {}, "http://other/tokenize")
