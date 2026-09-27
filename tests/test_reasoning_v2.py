"""Fusion v2 reasoning protocols; scripted models and a monkeypatched HTTP boundary."""
from copy import deepcopy
import io
import json
import urllib.error

import pytest

from dagbt.budget import BudgetExceeded, Ledger
from dagbt.evidence_protocol import EvidenceJSONError, parse_evidence_object
from dagbt.model_runtime import (BTTokenAccounting, REASONING_MARKER, count_request_tokens,
                                evidence_wire_tokens, prepare_request, token_accounting)
from dagbt.reasoning import (EmptyResponseError, InputOverflow, OutputTruncated,
                            ProtocolError, Reasoner, RefusalError)
from dagbt.transport import StubMeter, Transport


def settings(**updates):
    return {"reasoning_output_tokens": 128, "context_tokens": 4096,
            "selection": "dependency", "reserved_audit_calls": 1,
            "max_repairs_per_request": 2, "input_margin": 256,
            "response_format": "plain", **updates}


def profile(s):
    return {"model_profile": "bridgetree", "llm_model": "fixture",
            "llm_base_url": "http://fixture/v1", "llm_provider_request_params": {},
            "fusion": s, "max_identical_attempts": 3}


def response(raw="{}", finish="stop", refusal=None):
    choice = {"message": {"content": raw}}
    if finish is not None:
        choice["finish_reason"] = finish
    if refusal is not None:
        choice["message"]["refusal"] = refusal
    return {"response_ref": "scripted", "response": {"choices": [choice],
            "id": "provider-id", "usage": {"prompt_tokens": 7, "completion_tokens": 5}}}


class Scripted:
    def __init__(self, replies):
        self.replies, self.requests = list(replies), []

    def get(self, stage, url, payload):
        self.requests.append((stage, url, deepcopy(payload)))
        return self.replies.pop(0)


def reasoner(replies, **overrides):
    s = settings(**overrides)
    config = profile(s)
    ledger = Ledger({"llm": 24, "json_repairs": 6})
    base = Scripted(replies)
    events = []
    r = Reasoner(StubMeter(base, ledger, config), BTTokenAccounting(), config, s, ledger, events.append)
    return r, base, ledger, events


@pytest.mark.parametrize("raw", ["\ufeff{}", "```json\n{}\n```", "Here is the result: {} Done."])
def test_strict_parser_accepts_only_complete_unambiguous_wrappers(raw):
    assert parse_evidence_object(raw) == {}


@pytest.mark.parametrize("raw", ['{"x":1,"x":2}', '{"x": NaN}', '{"x":1e999}',
                                     '{"x":', '{} {}', '[] {}', 'Here {"x": } {}',
                                     '```json\n{}', '{} true'])
def test_strict_parser_never_invents_truncated_or_competing_json(raw):
    with pytest.raises(EvidenceJSONError):
        parse_evidence_object(raw)


def test_single_request_exposes_metadata_but_does_not_repair():
    r, base, ledger, events = reasoner([response('{"x":')])
    with pytest.raises(ProtocolError) as error:
        r.request("map", "Map the evidence.", {"allowed_ids": ["s1"]})
    assert error.value.category == "json_syntax"
    assert error.value.response_ref == "scripted"
    assert error.value.metadata["usage"]["prompt_tokens"] == 7
    assert r.last_response["raw_output"] == '{"x":'
    assert ledger.used["llm"] == 1 and not ledger.used["json_repairs"]
    assert len(base.requests) == len(r.requests) == 1
    assert any(e["event"] == "reasoning_request_started" for e in events)


@pytest.mark.parametrize("finish,refusal,raw,error", [
    ("length", None, '{"unfinished":', OutputTruncated),
    ("stop", "refused", "{}", RefusalError),
    ("content_filter", None, "", RefusalError),
])
def test_truncation_and_refusal_do_not_repeat_large_payload(finish, refusal, raw, error):
    r, base, ledger, _ = reasoner([response(raw, finish, refusal)])
    with pytest.raises(error):
        r.json("map", "Map.", {"evidence": "source"}, lambda x: x)
    assert len(base.requests) == 1 and not ledger.used["json_repairs"]


def test_missing_finish_metadata_is_unknown_not_invented_stop():
    r, _, _, events = reasoner([response('{"x":1}', None)])
    assert r.request("map", "Map.", {}) == {"x": 1}
    assert r.last_response["response_metadata"]["finish_reason"] is None
    assert [e for e in events if e["event"] == "reasoning_response"][0]["finish_reason"] is None


def test_empty_response_has_explicit_classification():
    r, _, _, _ = reasoner([response(" ")])
    with pytest.raises(EmptyResponseError):
        r.request("map", "Map.", {})


def test_local_repairs_preserve_original_and_never_append_model_output():
    raw = 'BROKEN_PRIVATE_CONTENT_' * 400
    r, base, ledger, _ = reasoner([response(raw), response(raw), response("{}")])
    original = {"evidence": [{"id": "s1", "text": "Exact 原文"}], "allowed_ids": ["s1"]}
    saved = deepcopy(original)
    assert r.json("resolve", "Resolve.", original, lambda x: x) == {}
    assert ledger.used["json_repairs"] == 2 and ledger.used["llm"] == 3
    assert original == saved
    for _, _, payload in base.requests:
        assert len(payload["messages"]) == 2
        assert "BROKEN_PRIVATE_CONTENT" not in json.dumps(payload)
        data = json.loads(payload["messages"][1]["content"])
        assert data["evidence"] == saved["evidence"] and data["allowed_ids"] == ["s1"]
        assert len(data.get("validation_feedback", "")) <= 1200


def test_local_and_global_repair_caps_are_both_enforced():
    r, base, ledger, _ = reasoner([response("not JSON")] * 10)
    for _ in range(3):
        with pytest.raises(ProtocolError):
            r.json("resolve", "Resolve.", {}, lambda x: x)
    assert len(base.requests) == 9 and ledger.used["json_repairs"] == 6
    with pytest.raises(ProtocolError):
        r.json("resolve", "Resolve.", {}, lambda x: x)
    assert len(base.requests) == 10 and ledger.used["json_repairs"] == 6


@pytest.mark.parametrize("mode", ["plain", "json_object", "json_schema"])
def test_response_protocol_and_estimate_are_the_exact_wire_configuration(mode):
    r, base, _, _ = reasoner([response()], response_format=mode)
    schema = {"type": "object", "properties": {"text": {"type": "string"}},
              "required": ["text"], "additionalProperties": False}
    before = r.estimate("map", "Map.", {"text": "汉" * 300}, schema)
    r.request("map", "Map.", {"text": "汉" * 300}, schema=schema)
    stage, url, payload = base.requests[0]
    _, wire, _ = prepare_request(stage, url, payload, r.config)
    assert REASONING_MARKER not in wire and "structured_outputs" not in wire
    assert before == evidence_wire_tokens(wire) == r.last_response["input_tokens_local"]
    assert before >= 300
    if mode == "plain":
        assert "response_format" not in wire
    else:
        assert wire["response_format"]["type"] == mode
    if mode == "json_schema":
        assert wire["response_format"]["json_schema"]["schema"] == schema


def test_reasoning_context_includes_wire_output_reserve_margin_and_rejects_before_call():
    r, base, ledger, _ = reasoner([response()])
    count = r.estimate("map", "Map.", {"text": "中" * 100})
    r.settings["context_tokens"] = count + 128 + 8 + 256 - 1
    with pytest.raises(InputOverflow):
        r.request("map", "Map.", {"text": "中" * 100})
    assert not base.requests and not ledger.used["llm"]


def test_audit_flat_and_extra_resolver_reservations_are_preserved():
    r, base, ledger, _ = reasoner([response(), response()], selection="flat")
    ledger.limits["llm"] = 3
    with pytest.raises(BudgetExceeded):
        r.request("map", "Map.", {}, extra_reserve=1)
    r.request("audit", "Audit.", {})
    r.request("select", "Select.", {})
    assert len(base.requests) == ledger.used["llm"] == 2


def test_transport_retry_cannot_spend_reserved_followup_call(tmp_path, monkeypatch):
    s = settings()
    config = profile(s)
    ledger = Ledger({"llm": 3, "json_repairs": 6})
    attempts = []
    def unavailable(*args, **kwargs):
        attempts.append(1)
        raise urllib.error.URLError("scripted unavailability")
    monkeypatch.setattr("dagbt.transport.urllib.request.urlopen", unavailable)
    monkeypatch.setattr("dagbt.transport.time.sleep", lambda _: None)
    calls = Transport("q", tmp_path, config, ledger, BTTokenAccounting())
    r = Reasoner(calls, calls.tokenizer, config, s, ledger, lambda _: None)
    with pytest.raises(BudgetExceeded):
        r.request("map", "Map.", {}, extra_reserve=1)
    assert ledger.used["llm"] == ledger.used["http_attempts"] == len(attempts) == 1
    assert r.sequence == 1


def test_transport_matches_reasoning_estimate_and_preserves_wire_and_cache(tmp_path, monkeypatch):
    s = settings(response_format="json_schema")
    config = profile(s)
    ledger = Ledger({"llm": 24})
    sent = []
    def serve(req, **kwargs):
        sent.append(json.loads(req.data))
        return io.BytesIO(json.dumps(response()["response"]).encode())
    monkeypatch.setattr("dagbt.transport.urllib.request.urlopen", serve)
    calls = Transport("q", tmp_path, config, ledger, BTTokenAccounting())
    r = Reasoner(calls, calls.tokenizer, config, s, ledger, lambda _: None)
    estimated = r.estimate("map", "Map.", {"text": "汉" * 100})
    r.request("map", "Map.", {"text": "汉" * 100})
    r.request("map", "Map.", {"text": "汉" * 100})
    assert len(sent) == 1 and sent[0]["response_format"]["type"] == "json_schema"
    assert REASONING_MARKER not in sent[0]
    budgets = [e for e in ledger.events if e["event"] == "model_context_budget"]
    assert all(e["input_tokens_local"] == estimated and e["input_margin"] == 256 for e in budgets)
    assert all(e["token_estimator_id"] == "regex_or_utf8_bytes_div3_v2" for e in budgets)
    assert ledger.used["llm"] == 2 and ledger.used["cache_hits"] == 1


def test_reader_original_keep_their_existing_token_estimator():
    s = settings()
    config = profile(s)
    payload = {"messages": [{"role": "user", "content": "汉" * 100}], "max_tokens": 10}
    assert count_request_tokens("planner", "http://fixture/v1/chat/completions", payload, config) < 100
    assert token_accounting(config)["token_estimator_id"] == "regex_word_or_punctuation_v1"


def test_invalid_protocol_is_not_silently_downgraded():
    r, base, _, _ = reasoner([response()], response_format="unsupported")
    with pytest.raises(ValueError, match="response_format"):
        r.request("map", "Map.", {})
    assert not base.requests
