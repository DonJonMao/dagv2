"""Per-request output caps through production accounting and cached transport.

All HTTP boundaries and credentials are replaced with synthetic fixtures.
Reader admission policy and its minimum useful output size are tested separately.
"""
from copy import deepcopy
import io
import json
from types import SimpleNamespace
import urllib.error

import pytest

from dagbt import token_counting, transport
from dagbt.budget import BudgetExceeded, Ledger
from dagbt.methods import RESIDUAL
from dagbt.model_runtime import BTTokenAccounting, evidence_wire_tokens, prepare_request
from dagbt.reasoning import InputOverflow, OutputTruncated, Reasoner


DATA = {"sources": [{"id": "s1", "text": " Preserve 原文.\n"}]}
SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}},
          "required": ["ok"], "additionalProperties": False}


class JSONResponse(io.BytesIO):
    def __init__(self, body):
        super().__init__(json.dumps(body).encode())

    def getcode(self):
        return 200


@pytest.fixture(autouse=True)
def offline_boundaries(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected service or credential access")
    monkeypatch.setattr(transport.urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(token_counting.urllib.request, "build_opener", forbidden)
    monkeypatch.setattr(transport, "resolve_api_key", lambda *args: "")
    monkeypatch.setattr(token_counting, "resolve_api_key", lambda *args: "")
    monkeypatch.setattr(transport.time, "sleep", lambda seconds: None)


class SyntheticHTTP:
    def __init__(self, monkeypatch, *, count=800, finish="stop", fail_first=False):
        self.count, self.finish, self.fail_first = count, finish, fail_first
        self.tokenized, self.generated = [], []
        monkeypatch.setattr(token_counting.urllib.request, "build_opener",
                            lambda *handlers: SimpleNamespace(open=self.tokenize))
        monkeypatch.setattr(transport.urllib.request, "urlopen", self.generate)

    def tokenize(self, request, timeout):
        assert request.full_url == "http://offline.invalid/tokenize"
        self.tokenized.append(json.loads(request.data))
        return JSONResponse({"count": self.count, "max_model_len": 16384})

    def generate(self, request, timeout):
        assert request.full_url == "http://offline.invalid/v1/chat/completions"
        self.generated.append(json.loads(request.data))
        if self.fail_first and len(self.generated) == 1:
            raise urllib.error.URLError("synthetic transient failure")
        return JSONResponse({"choices": [{"finish_reason": self.finish,
            "message": {"content": '{"ok":true}'}}],
            "usage": {"prompt_tokens": self.count, "completion_tokens": 5,
                      "total_tokens": self.count + 5}})


def components(directory, *, enabled=True, maximum=4096, generic=False):
    settings = {"response_format": "plain", "context_tokens": 16384,
                "reasoning_output_tokens": maximum, "input_margin": 256,
                "reserved_audit_calls": 1, "max_repairs_per_request": 2}
    if not generic:
        settings["algorithm_version"] = RESIDUAL
    config = {"model_profile": "bridgetree", "llm_base_url": "http://offline.invalid/v1",
              "llm_model": "synthetic-model", "fusion": settings, "max_identical_attempts": 3}
    if enabled:
        config["tokenization"] = {"mode": "vllm_chat", "endpoint": "http://offline.invalid/tokenize"}
    ledger = Ledger({"llm": 24, "json_repairs": 6, "ann": 36})
    calls = transport.Transport("synthetic-cap-question", directory, config, ledger, BTTokenAccounting())
    events = []
    reasoner = Reasoner(calls, calls.tokenizer, config, settings, ledger, events.append)
    return reasoner, calls, ledger, events


@pytest.mark.parametrize("method", ["payload", "estimate", "request", "preflight"])
@pytest.mark.parametrize("cap", [True, False, 0, -1, 1.0, "1024", 4097])
def test_invalid_cap_fails_before_accounting_or_generation(tmp_path, monkeypatch, method, cap):
    server = SyntheticHTTP(monkeypatch)
    reasoner, _, ledger, _ = components(tmp_path)
    with pytest.raises(ValueError, match="output_token_cap"):
        if method == "preflight":
            reasoner._preflight("joint_read", 100, output_token_cap=cap)
        else:
            function = reasoner._payload if method == "payload" else getattr(reasoner, method)
            function("joint_read", "Read.", DATA, output_token_cap=cap)
    assert not server.tokenized and not server.generated
    assert not reasoner.requests and reasoner.sequence == 0
    assert ledger.used["llm"] == ledger.used["http_attempts"] == 0


@pytest.mark.parametrize("offset", [0, 1])
def test_actual_cap_has_same_exact_boundary_in_reasoner_and_transport(tmp_path, monkeypatch, offset):
    cap = 1024
    count = 16384 - cap - 256 - 8 + offset
    server = SyntheticHTTP(monkeypatch, count=count)
    reasoner, _, ledger, events = components(tmp_path)
    settings, limits, data = deepcopy(reasoner.settings), deepcopy(ledger.limits), deepcopy(DATA)
    assert reasoner.estimate("joint_read", "Read.", DATA, SCHEMA, output_token_cap=cap) == count
    if offset:
        with pytest.raises(InputOverflow):
            reasoner.request("joint_read", "Read.", DATA, schema=SCHEMA, output_token_cap=cap)
        assert not server.generated and not reasoner.requests and reasoner.sequence == 0
        assert ledger.used["llm"] == ledger.used["http_attempts"] == 0
    else:
        # The unchanged old/default check cannot admit this same input.
        with pytest.raises(InputOverflow):
            reasoner._preflight("joint_read", count)
        assert reasoner.request("joint_read", "Read.", DATA, schema=SCHEMA,
                                output_token_cap=cap) == {"ok": True}
        assert server.generated[0]["max_tokens"] == cap
        record = reasoner.last_response
        assert record["input_tokens_local"] == count and record["output_token_reserve"] == cap
        context = next(event for event in ledger.events if event["event"] == "model_context_budget")
        assert context["total_tokens_local"] == 16384 and context["output_token_reserve"] == cap
        response = next(event for event in events if event["event"] == "reasoning_response")
        assert response["output_token_reserve"] == cap
    assert reasoner.settings == settings and ledger.limits == limits and DATA == data


def test_cap_separates_generation_cache_but_shares_tokenization_and_replay_charge(tmp_path, monkeypatch):
    server = SyntheticHTTP(monkeypatch)
    reasoner, _, ledger, _ = components(tmp_path)
    for cap in (1024, 2048, 1024):
        assert reasoner.estimate("joint_read", "Read.", DATA, SCHEMA, output_token_cap=cap) == 800
        reasoner.request("joint_read", "Read.", DATA, schema=SCHEMA, output_token_cap=cap)
    assert [wire["max_tokens"] for wire in server.generated] == [1024, 2048]
    assert len(server.tokenized) == 1
    records = reasoner.requests
    assert records[0]["response_ref"] == records[2]["response_ref"] != records[1]["response_ref"]
    assert records[0]["tokenization_request_sha256"] == records[1]["tokenization_request_sha256"]
    assert records[0]["tokenization_wire_sha256"] != records[1]["tokenization_wire_sha256"]
    saved = [json.loads(path.read_text()) for path in (tmp_path / "requests").glob("*.json")]
    assert sorted(record["payload"]["max_tokens"] for record in saved) == [1024, 2048]
    assert ledger.used["llm"] == 3 and ledger.used["http_attempts"] == 2 and ledger.used["cache_hits"] == 1


@pytest.mark.parametrize("maximum", [128, 4096])
@pytest.mark.parametrize("generic", [False, True])
def test_default_cap_and_legacy_json_preflight_are_unchanged(tmp_path, monkeypatch, maximum, generic):
    server = SyntheticHTTP(monkeypatch)
    reasoner, _, _, _ = components(tmp_path, maximum=maximum, generic=generic)
    reasoner._preflight("planner", 16384 - maximum - 256 - 8)
    with pytest.raises(InputOverflow):
        reasoner._preflight("planner", 16384 - maximum - 256 - 7)
    result = reasoner.json("planner", "Plan.", DATA, lambda value: value, SCHEMA)
    assert result == {"ok": True} and server.generated[0]["max_tokens"] == maximum
    assert reasoner.last_response["output_token_reserve"] == maximum


def test_truncated_valid_json_is_rejected_and_cached_replay_still_pays(tmp_path, monkeypatch):
    server = SyntheticHTTP(monkeypatch, finish="length")
    reasoner, _, ledger, events = components(tmp_path)
    validated = []
    for _ in range(2):
        with pytest.raises(OutputTruncated) as error:
            reasoner.request("joint_read", "Read.", DATA, validate=validated.append,
                             schema=SCHEMA, output_token_cap=1024)
        assert error.value.metadata["finish_reason"] == "length"
        assert reasoner.last_response["validation_status"] == "invalid"
        assert reasoner.last_response["failure_category"] == "output_truncated"
    assert not validated and len(server.generated) == 1
    assert ledger.used["llm"] == 2 and ledger.used["json_repairs"] == 0
    assert len([event for event in events if event["event"] == "protocol_error"]) == 2


def test_preflight_uses_constructed_payload_cap(tmp_path, monkeypatch):
    server = SyntheticHTTP(monkeypatch, count=16384 - 768 - 256 - 8)
    reasoner, _, _, _ = components(tmp_path)
    original = reasoner._payload
    def smaller_payload(*args, **kwargs):
        payload = original(*args, **kwargs)
        payload["max_tokens"] = 768
        return payload
    monkeypatch.setattr(reasoner, "_payload", smaller_payload)
    assert reasoner.request("joint_read", "Read.", DATA, schema=SCHEMA,
                            output_token_cap=1024) == {"ok": True}
    assert server.generated[0]["max_tokens"] == reasoner.last_response["output_token_reserve"] == 768


def test_wire_estimator_sees_cap_and_protocol_schema_without_deployment_counter(tmp_path):
    reasoner, _, _, _ = components(tmp_path, enabled=False)
    reasoner.settings["response_format"] = "json_schema"
    for cap in (64, 1024, 4096):
        payload = reasoner._payload("joint_read", "Read.", DATA, SCHEMA, output_token_cap=cap)
        _, wire, _ = prepare_request(("joint_read", "1"), reasoner.url, payload, reasoner.config)
        assert wire["max_tokens"] == cap and wire["response_format"]["type"] == "json_schema"
        assert reasoner.estimate("joint_read", "Read.", DATA, SCHEMA,
                                 output_token_cap=cap) == evidence_wire_tokens(wire)


def test_assistant_only_structural_floor_uses_production_wire_count_without_generation(tmp_path, monkeypatch):
    server = SyntheticHTTP(monkeypatch, count=137)
    reasoner, _, ledger, _ = components(tmp_path)
    payload = reasoner._payload("joint_read", "Read.", DATA, SCHEMA, output_token_cap=1024)
    skeleton = {"slots": [{"variable": "history", "observations": [], "unresolved": []}]}
    payload["messages"] = [{"role": "assistant", "content": json.dumps(skeleton)}]
    counted = reasoner._count_payload("joint_read", payload)
    assert counted["count"] == 137
    assert server.tokenized == [{"model": "synthetic-model", "messages": payload["messages"],
                                 "add_generation_prompt": True}]
    assert not server.generated and not reasoner.requests and reasoner.sequence == 0
    assert ledger.used["llm"] == ledger.used["http_attempts"] == ledger.used["json_repairs"] == 0


def test_cap_does_not_change_call_reservation_and_physical_retry_payload(tmp_path, monkeypatch):
    server = SyntheticHTTP(monkeypatch, fail_first=True)
    reasoner, _, ledger, _ = components(tmp_path)
    ledger.limits["llm"] = 3
    with pytest.raises(BudgetExceeded):
        reasoner.request("joint_read", "Read.", DATA, schema=SCHEMA,
                         output_token_cap=1024, extra_reserve=1)
    assert len(server.generated) == ledger.used["llm"] == 1
    assert server.generated[0]["max_tokens"] == 1024
    assert reasoner.settings["reasoning_output_tokens"] == 4096


def test_physical_retry_preserves_cap_and_exact_wire(tmp_path, monkeypatch):
    server = SyntheticHTTP(monkeypatch, fail_first=True)
    reasoner, _, ledger, _ = components(tmp_path)
    assert reasoner.request("joint_read", "Read.", DATA, schema=SCHEMA,
                            output_token_cap=1024) == {"ok": True}
    assert len(server.generated) == ledger.used["llm"] == ledger.used["http_attempts"] == 2
    assert server.generated[0] == server.generated[1]
    assert server.generated[0]["max_tokens"] == 1024
