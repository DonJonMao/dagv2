"""Joint wire compaction preserves JSON values and meters what is sent."""
from copy import deepcopy
import json

import pytest

from dagbt.budget import Ledger
from dagbt.methods import JOINT, RESIDUAL, LOCAL
from dagbt.model_runtime import count_evidence_request_tokens
from dagbt.reasoning import InputOverflow, Reasoner


DATA = {"source_spans": [{"id": "s1", "text": '  原文  :  ,  "quoted"\n\tvalue  '},
                          {"id": "s2", "text": "Keep every source."}],
        "known_observations": [{"value": {"x": [None, True, 1, 1.25]}, "source_ids": ["s1"]}],
        "repair_scope": {"failed": [{"id": "counter", "observation": {"stance": "contradiction"}}]}}
SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean", "description": "Keep  this spacing."}},
          "required": ["ok"], "additionalProperties": False}


class Calls:
    def __init__(self, ledger):
        self.ledger, self.sent = ledger, []

    def get(self, stage, url, payload):
        self.ledger.reserve("llm", "/".join(stage))
        self.sent.append(deepcopy(payload))
        return {"response": {"choices": [{"finish_reason": "stop", "message": {"content": '{"ok":true}'}}]}}


def reasoner(method=RESIDUAL, mode="plain"):
    settings = {"algorithm_version": method, "response_format": mode, "reasoning_output_tokens": 512,
                "context_tokens": 16384, "input_margin": 256, "reserved_audit_calls": 1}
    config = {"llm_base_url": "http://offline.invalid/v1", "llm_model": "scripted", "fusion": settings}
    ledger = Ledger({"llm": 24, "json_repairs": 6})
    calls = Calls(ledger)
    return Reasoner(calls, None, config, settings, ledger, lambda event: None), calls


@pytest.mark.parametrize("method", [JOINT, RESIDUAL])
@pytest.mark.parametrize("mode", ["plain", "json_object", "json_schema"])
def test_compact_prompt_roundtrips_sources_values_counters_and_schema(method, mode):
    engine, _ = reasoner(method, mode)
    before_data, before_schema = deepcopy(DATA), deepcopy(SCHEMA)
    for operation in ("joint_read", "semantic_compose", "audit_joint",
                      "joint_read_repair", "semantic_compose_repair", "audit_joint_repair"):
        payload = engine._payload(operation, "Instruction", DATA, SCHEMA)
        user = payload["messages"][1]["content"]
        assert json.loads(user) == before_data
        assert len(user) < len(json.dumps(DATA, ensure_ascii=False))
        if mode == "json_schema":
            assert payload["response_format"]["json_schema"]["schema"] == before_schema
        else:
            schema_text = payload["messages"][0]["content"].split("matching this schema:\n", 1)[1]
            assert json.loads(schema_text) == before_schema
            assert len(schema_text) < len(json.dumps(SCHEMA, ensure_ascii=False, sort_keys=True))
    assert DATA == before_data and SCHEMA == before_schema


@pytest.mark.parametrize("method,operation", [(None, "joint_read"), (LOCAL, "joint_read"),
    ("unknown_method", "audit_joint"), (RESIDUAL, "planner"), (RESIDUAL, "select"),
    (RESIDUAL, "joint_read_unrecognized")])
def test_legacy_and_other_operations_keep_exact_previous_prompt(method, operation):
    engine, _ = reasoner(method)
    payload = engine._payload(operation, "Instruction", DATA, SCHEMA)
    assert payload["messages"] == [
        {"role": "system", "content": "Instruction\n\nReturn one complete JSON object matching this schema:\n"
            + json.dumps(SCHEMA, ensure_ascii=False, sort_keys=True, allow_nan=False)},
        {"role": "user", "content": json.dumps(DATA, ensure_ascii=False, allow_nan=False)},
    ]


def test_exact_compact_wire_capacity_boundary_and_uncharged_overflow():
    engine, calls = reasoner()
    operation = "joint_read"
    estimate = engine.estimate(operation, "Instruction", DATA, SCHEMA)
    engine.settings["context_tokens"] = estimate + 512 + 256 + 8
    assert engine.request(operation, "Instruction", DATA, schema=SCHEMA) == {"ok": True}
    assert len(calls.sent) == 1
    actual = count_evidence_request_tokens((operation, "1"), engine.url, calls.sent[0], engine.config)
    assert actual == estimate == engine.requests[0]["input_tokens_local"]
    assert json.loads(calls.sent[0]["messages"][1]["content"]) == DATA
    engine.settings["context_tokens"] -= 1
    with pytest.raises(InputOverflow):
        engine.request(operation, "Instruction", DATA, schema=SCHEMA)
    assert len(calls.sent) == 1 and engine.ledger.used["llm"] == 1
