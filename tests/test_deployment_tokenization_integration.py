"""Production accounting integration with synthetic HTTP and unchanged budgets."""
from copy import deepcopy
import io
import json
from types import SimpleNamespace
import urllib.error

import pytest

from dagbt import runner, token_counting, transport
from dagbt.budget import Ledger
from dagbt.methods import RESIDUAL
from dagbt.model_runtime import BTTokenAccounting
from dagbt.reasoning import InputOverflow, Reasoner


SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}},
          "required": ["ok"], "additionalProperties": False}
DATA = {"source": "  Preserve 原文 and punctuation.\n", "value": [False, 0, None]}


class JSONResponse(io.BytesIO):
    def __init__(self, value):
        super().__init__(json.dumps(value).encode())

    def getcode(self):
        return 200


@pytest.fixture(autouse=True)
def deny_real_network_and_credentials(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected network call in tokenization integration test")
    monkeypatch.setattr(transport.urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(token_counting.urllib.request, "build_opener", forbidden)
    monkeypatch.setattr(transport, "resolve_api_key", lambda *args: "")
    monkeypatch.setattr(token_counting, "resolve_api_key", lambda *args: "")


class SyntheticHTTP:
    def __init__(self, monkeypatch, count=800, usage="matched", failure=None):
        self.count = count
        self.usage = usage
        self.failure = failure
        self.tokenized = []
        self.generated = []
        monkeypatch.setattr(token_counting.urllib.request, "build_opener",
                            lambda *handlers: SimpleNamespace(open=self.tokenize))
        monkeypatch.setattr(transport.urllib.request, "urlopen", self.generate)

    def tokenize(self, request, timeout):
        assert request.full_url == "http://offline.invalid/tokenize"
        assert timeout == 15
        payload = json.loads(request.data)
        assert set(payload) == {"model", "messages", "add_generation_prompt"}
        assert payload["add_generation_prompt"] is True
        self.tokenized.append(payload)
        if self.failure:
            raise self.failure
        return JSONResponse({"count": self.count, "max_model_len": 120000})

    def generate(self, request, timeout):
        assert request.full_url == "http://offline.invalid/v1/chat/completions"
        payload = json.loads(request.data)
        self.generated.append(payload)
        value = {"choices": [{"finish_reason": "stop", "message": {"content": '{"ok":true}'}}]}
        if self.usage != "missing":
            prompt = self.count if self.usage == "matched" else self.usage
            value["usage"] = {"prompt_tokens": prompt, "completion_tokens": 5,
                              "total_tokens": int(prompt) + 5}
        return JSONResponse(value)


def configuration(*, enabled=True, original=False):
    settings = {"response_format": "plain", "context_tokens": 16384,
                "reasoning_output_tokens": 4096, "input_margin": 256,
                "reserved_audit_calls": 1, "llm_calls": 24, "ann_calls": 36}
    if not original:
        settings["algorithm_version"] = RESIDUAL
    config = {"model_profile": "bridgetree", "llm_base_url": "http://offline.invalid/v1",
              "llm_model": "scripted-deployment", "fusion": settings,
              "max_identical_attempts": 3, "request_timeout_seconds": 15}
    if enabled:
        config["tokenization"] = {"mode": "vllm_chat", "endpoint": "http://offline.invalid/tokenize",
                                  "timeout_seconds": 15}
    return config


def components(directory, config=None):
    config = configuration() if config is None else config
    ledger = Ledger({"llm": 24, "ann": 36, "json_repairs": 6})
    calls = transport.Transport("synthetic-question", directory, config, ledger, BTTokenAccounting())
    reasoner = Reasoner(calls, calls.tokenizer, config, config["fusion"], ledger, lambda event: None)
    return reasoner, calls, ledger


def tokenization_events(directory):
    return [json.loads(line) for line in (directory / "tokenization_events.jsonl").read_text().splitlines()]


def test_estimate_request_and_transport_share_real_counter_and_costs(tmp_path, monkeypatch):
    server = SyntheticHTTP(monkeypatch)
    reasoner, _, ledger = components(tmp_path)
    before = deepcopy(DATA)
    assert reasoner.estimate("joint_read", "Read evidence.", DATA, SCHEMA) == 800
    assert reasoner.estimate("joint_read", "Read evidence.", DATA, SCHEMA) == 800
    assert reasoner.request("joint_read", "Read evidence.", DATA, schema=SCHEMA) == {"ok": True}
    assert DATA == before
    assert len(server.tokenized) == len(server.generated) == 1
    assert server.tokenized[0]["messages"] == server.generated[0]["messages"]
    assert "matching this schema" in server.tokenized[0]["messages"][0]["content"]
    assert json.loads(server.tokenized[0]["messages"][1]["content"]) == DATA
    assert ledger.used["llm"] == ledger.used["http_attempts"] == 1
    assert ledger.used["ann"] == ledger.used["json_repairs"] == 0
    assert reasoner.requests[0]["input_tokens_local"] == 800
    assert reasoner.requests[0]["token_count_is_estimate"] is False
    budget = next(event for event in ledger.events if event["event"] == "model_context_budget")
    assert budget["input_tokens_local"] == 800 and budget["context_tokens"] == 16384
    events = tokenization_events(tmp_path)
    assert len(events) == 4 and sum(event["cache_hit"] for event in events) == 3
    cost = runner.request_cost(tmp_path)
    assert cost["physical_llm_attempts"] == cost["http_attempts"] == 1
    assert cost["prompt_tokens"] == 800 and cost["completion_tokens"] == 5
    assert cost["tokenization"]["logical_calls"] == 4
    assert cost["tokenization"]["cache_hits"] == 3
    assert cost["tokenization"]["http_attempts"] == cost["tokenization"]["successful_http_requests"] == 1
    assert cost["tokenization"]["observed_seconds"] >= 0
    assert "tokenization_http" not in cost["by_kind"]


def test_system_schema_and_source_changes_invalidate_shared_token_count(tmp_path, monkeypatch):
    server = SyntheticHTTP(monkeypatch)
    reasoner, _, ledger = components(tmp_path)
    reasoner.estimate("joint_read", "Instruction one.", DATA, SCHEMA)
    reasoner.estimate("joint_read", "Instruction two.", DATA, SCHEMA)
    new_schema = {**SCHEMA, "description": "A different schema instruction."}
    reasoner.estimate("joint_read", "Instruction one.", DATA, new_schema)
    new_data = {**DATA, "source": DATA["source"] + "Additional original source."}
    reasoner.estimate("joint_read", "Instruction one.", new_data, SCHEMA)
    assert len(server.tokenized) == 4 and not server.generated
    assert len({json.dumps(row["messages"]) for row in server.tokenized}) == 4
    assert ledger.used["llm"] == ledger.used["ann"] == ledger.used["http_attempts"] == 0


@pytest.mark.parametrize("offset", [0, 1])
def test_reasoner_uses_unchanged_16k_boundary_before_generation(tmp_path, monkeypatch, offset):
    count = 16384 - 4096 - 256 - 8 + offset
    server = SyntheticHTTP(monkeypatch, count=count)
    reasoner, _, ledger = components(tmp_path)
    if offset:
        with pytest.raises(InputOverflow):
            reasoner.request("joint_read", "Read.", DATA, schema=SCHEMA)
        assert not server.generated and reasoner.sequence == 0
        assert ledger.used["llm"] == ledger.used["http_attempts"] == 0
    else:
        assert reasoner.request("joint_read", "Read.", DATA, schema=SCHEMA) == {"ok": True}
        event = next(event for event in ledger.events if event["event"] == "model_context_budget")
        assert event["total_tokens_local"] == event["context_tokens"] == 16384
        assert len(server.generated) == ledger.used["llm"] == 1
    assert len(server.tokenized) == 1
    assert reasoner.settings["reasoning_output_tokens"] == 4096


def original_payload():
    messages = [{"role": "system", "content": "Read original evidence."},
                {"role": "user", "content": "Do not alter 原始 text."}]
    return {"prompt": BTTokenAccounting().apply_chat_template(messages), "max_tokens": 512,
            "structured_outputs": {"json": SCHEMA},
            "chat_template_kwargs": {"enable_thinking": False}}


@pytest.mark.parametrize("offset", [0, 1])
def test_original_adapted_chat_uses_same_final_server_gate(tmp_path, monkeypatch, offset):
    server = SyntheticHTTP(monkeypatch, count=16384 - 512 - 8 + offset)
    _, calls, ledger = components(tmp_path, configuration(original=True))
    payload = original_payload()
    before = deepcopy(payload)
    if offset:
        with pytest.raises(transport.ResponseError, match="context budget"):
            calls.get(("solve_dag", "node"), "http://offline.invalid/v1/completions", payload)
        assert not server.generated and ledger.used["llm"] == ledger.used["http_attempts"] == 0
    else:
        result = calls.get(("solve_dag", "node"), "http://offline.invalid/v1/completions", payload)
        assert result["response"]["choices"][0]["text"] == '{"ok":true}'
        assert server.tokenized[0]["messages"] == server.generated[0]["messages"]
    assert payload == before and len(server.tokenized) == 1
    assert "matching this schema" in server.tokenized[0]["messages"][0]["content"]
    assert "prompt" not in server.tokenized[0]
    event = next(event for event in ledger.events if event["event"] == "model_context_budget")
    assert event["context_tokens"] == 16384 and event["total_tokens_local"] == 16384 + offset


@pytest.mark.parametrize("usage", [801, "missing", True])
def test_usage_failure_is_preserved_not_retried_and_replay_cannot_succeed(tmp_path, monkeypatch, usage):
    server = SyntheticHTTP(monkeypatch, usage=usage)
    reasoner, calls, ledger = components(tmp_path)
    payload = reasoner._payload("joint_read", "Read.", DATA, SCHEMA)
    for _ in range(2):
        with pytest.raises(transport.ResponseError, match="does not match"):
            calls.get(("joint_read", "1"), reasoner.url, payload)
    assert len(server.generated) == 1 and len(server.tokenized) == 1
    assert ledger.used["llm"] == ledger.used["http_attempts"] == 1
    record = json.loads(next((tmp_path / "requests").glob("*.json")).read_text())
    assert len(record["attempts"]) == 1 and "response" in record
    assert record["tokenization_validation"]["status"] == "failed"
    assert record["tokenization_validation"]["expected_prompt_tokens"] == 800
    assert record["tokenization_validation"]["observed_prompt_tokens"] == (801 if usage == 801 else None)
    validations = [event for event in ledger.events if event["event"] == "tokenization_usage_validation"]
    assert len(validations) == 2 and all(event["status"] == "failed" for event in validations)


def test_seeded_generation_replay_is_reverified_without_new_generation_cost(tmp_path, monkeypatch):
    server = SyntheticHTTP(monkeypatch)
    old, new = tmp_path / "old", tmp_path / "new"
    reasoner, _, _ = components(old)
    assert reasoner.request("joint_read", "Read.", DATA, schema=SCHEMA) == {"ok": True}
    runner.seed_cache([old], new)
    replay, _, ledger = components(new)
    assert replay.request("joint_read", "Read.", DATA, schema=SCHEMA) == {"ok": True}
    assert len(server.generated) == 1 and len(server.tokenized) == 2
    assert ledger.used["llm"] == ledger.used["cache_hits"] == 1
    assert ledger.used["http_attempts"] == ledger.used["ann"] == 0
    assert any(event["event"] == "tokenization_usage_validation" and event["status"] == "matched"
               for event in ledger.events)
    cost = runner.request_cost(new)
    assert cost["physical_llm_attempts"] == cost["http_attempts"] == cost["prompt_tokens"] == 0
    assert cost["tokenization"]["http_attempts"] == 1


def test_new_counter_cannot_erase_an_earlier_usage_failure(tmp_path, monkeypatch):
    server = SyntheticHTTP(monkeypatch, count=800, usage=801)
    reasoner, calls, _ = components(tmp_path)
    payload = reasoner._payload("joint_read", "Read.", DATA, SCHEMA)
    with pytest.raises(transport.ResponseError, match="does not match"):
        calls.get(("joint_read", "1"), reasoner.url, payload)
    path = next((tmp_path / "requests").glob("*.json"))
    prior = json.loads(path.read_text())["tokenization_validation"]
    server.count = 801
    _, restarted, ledger = components(tmp_path)
    with pytest.raises(transport.ResponseError, match="prior validation failed"):
        restarted.get(("joint_read", "1"), reasoner.url, payload)
    assert json.loads(path.read_text())["tokenization_validation"] == prior
    assert len(server.generated) == 1
    assert ledger.used["llm"] == ledger.used["http_attempts"] == 0


def test_tokenization_outage_fails_closed_without_generation_or_budget_charge(tmp_path, monkeypatch):
    server = SyntheticHTTP(monkeypatch, failure=urllib.error.URLError("synthetic outage"))
    reasoner, _, ledger = components(tmp_path)
    with pytest.raises(token_counting.TokenizationError, match="connection_or_timeout"):
        reasoner.request("joint_read", "Read.", DATA, schema=SCHEMA)
    assert len(server.tokenized) == 1 and not server.generated
    assert ledger.used["llm"] == ledger.used["ann"] == ledger.used["http_attempts"] == 0
    cost = runner.request_cost(tmp_path)
    assert cost["http_attempts"] == cost["logical_calls"] == 0
    assert cost["tokenization"]["http_attempts"] == cost["tokenization"]["failed_counts"] == 1


@pytest.mark.parametrize("profile", ["bridgetree", "native"])
def test_optin_disabled_never_constructs_counter_and_native_chat_still_replays(tmp_path, monkeypatch, profile):
    sent = []
    def generate(request, timeout):
        sent.append(json.loads(request.data))
        return JSONResponse({"choices": [{"message": {"content": '{}'}}]})
    monkeypatch.setattr(transport.urllib.request, "urlopen", generate)
    config = configuration(enabled=False, original=True)
    if profile == "native":
        config.pop("model_profile")
    _, calls, ledger = components(tmp_path, config)
    payload = {"messages": [{"role": "user", "content": "Ordinary chat."}], "max_tokens": 16}
    for _ in range(2):
        result = calls.get(("planner", "1"), "http://offline.invalid/v1/chat/completions", payload)
        assert result["response"]["choices"][0]["message"]["content"] == '{}'
    assert calls.token_counter is None and len(sent) == 1
    assert ledger.used["llm"] == 2 and ledger.used["cache_hits"] == 1
    assert not (tmp_path / "tokenization_events.jsonl").exists()
