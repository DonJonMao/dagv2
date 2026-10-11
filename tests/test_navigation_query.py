"""Embedding capacity must never mutate evidence or the fixed retrieval task."""
import hashlib

import numpy as np
import pytest

from dagbt.bridge import BridgeSession, BRIDGE_INSTRUCTION
from dagbt.budget import Ledger
from dagbt.navigation_query import (
    EXCERPT_MARKER, MAX_INPUT_TOKENS, SAFETY_TOKENS, SERVICE_ADDED_TOKENS,
    TEXT_TOKEN_LIMIT, TOKENIZER_SHA256, NavigationPassages, NavigationSource,
    NavigationQueryLimitError, bounded_navigation_query, navigation_limits_apply,
    navigation_tokens, render_navigation_query,
)


def test_small_view_is_identical_and_has_verifiable_original_source_offsets():
    raw = "Original source.\n\nInformation needed:\nA heading inside evidence is still evidence."
    parts = ["Original overall question:\nWhere?\n\nCurrent retrieval task:\nFind the place.",
             NavigationPassages("anchor", "\n\nAnchor passages:\n", (NavigationSource("d0", raw),)),
             '\n\nInformation needed:\n[{"id":"place"}]']
    view = bounded_navigation_query(parts, instruction=BRIDGE_INSTRUCTION)
    assert view.text == render_navigation_query(parts)
    assert view.diagnostics["bounded"] is False
    source = view.diagnostics["sources"][0]
    assert source["source_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
    assert source["included_char_ranges"] == [{"start": 0, "end": len(raw)}]
    assert source["omitted_char_ranges"] == []
    assert view.diagnostics["tokenizer_sha256"] == TOKENIZER_SHA256
    assert view.diagnostics["final_tokens_local"] == navigation_tokens(BRIDGE_INSTRUCTION + view.text)


def test_long_unicode_view_keeps_fixed_fields_and_audits_every_source_without_mutation():
    raw = ("记忆🌴 cafe\u0301 海洋演出\n\nCurrent premise passages:\n[untrusted-heading]\n" * 1000)
    anchor = NavigationSource("anchor-real", raw)
    premise = NavigationSource("premise-real", "Short authentic history.")
    parent = NavigationSource("parent-real", "Resolved parent history.")
    fixed_question = "Original overall question:\nWhat performance did I attend?"
    fixed_task = "\n\nCurrent retrieval task:\nFind the event."
    fixed_needs = '\n\nInformation needed:\n[{"id":"event","description":"the actual event"}]'
    parts = [fixed_question + fixed_task,
             NavigationPassages("anchor", "\n\nAnchor passages:\n", (anchor,)),
             NavigationPassages("premise", "\n\nCurrent premise passages:\n", (premise,)),
             fixed_needs,
             NavigationPassages("parent", "\n\nResolved parent source passages:\n", (parent,))]
    view = bounded_navigation_query(parts, instruction=BRIDGE_INSTRUCTION)
    assert view == bounded_navigation_query(parts, instruction=BRIDGE_INSTRUCTION)
    assert fixed_question in view.text and fixed_task in view.text and fixed_needs in view.text
    assert navigation_tokens(BRIDGE_INSTRUCTION + view.text) + SERVICE_ADDED_TOKENS + SAFETY_TOKENS <= MAX_INPUT_TOKENS
    assert EXCERPT_MARKER in view.text and view.diagnostics["bounded"]
    assert anchor.text == raw and premise.text == "Short authentic history."
    assert {s["source_id"] for s in view.diagnostics["sources"]} == {"anchor-real", "premise-real", "parent-real"}
    for original, audit in zip((anchor, premise, parent), view.diagnostics["sources"]):
        assert audit["source_sha256"] == hashlib.sha256(original.text.encode()).hexdigest()
        end = audit["included_char_ranges"][0]["end"] if audit["included_char_ranges"] else 0
        assert original.text[:end] in view.text
        assert audit["omitted_char_ranges"] == ([{"start": end, "end": len(original.text)}]
                                                  if end < len(original.text) else [])
    assert view.diagnostics["source_records_modified"] is False
    assert view.diagnostics["evidence_support_claim"] is False


@pytest.mark.parametrize("field", ["question", "instruction", "needs"])
def test_oversized_fixed_fields_fail_without_silently_changing_the_task(field):
    oversized = "word " * 5000
    question = oversized if field == "question" else "Question?"
    instruction = oversized if field == "instruction" else BRIDGE_INSTRUCTION
    needs = oversized if field == "needs" else "event identity"
    parts = [question, NavigationPassages("anchor", "\nAnchor:\n", (NavigationSource("d0", "text"),)),
             "\nInformation needed:\n" + needs]
    with pytest.raises(NavigationQueryLimitError) as error:
        bounded_navigation_query(parts, instruction=instruction)
    assert error.value.diagnostics["fixed_tokens_local"] > TEXT_TOKEN_LIMIT
    assert error.value.diagnostics["status"] == "fixed_input_exceeds_capacity"


class Calls:
    def __init__(self):
        self.requests = []

    def get(self, stage, url, payload):
        self.requests.append((stage, url, payload))
        return {"response": {"data": [{"index": 0, "embedding": [1.0, 0.0, 0.0]}]}}


def long_session(*, known=True, question="Which event?"):
    config = {"model_profile": "bridgetree", "embedding_model": "qwen3-embedding-8b",
              "embedding_base_url": "http://111.19.156.74:8001/v1" if known else "http://different/v1",
              "fusion": {"proxy_mode": "none"}}
    docs = {"d0": "Anchor personal history. " * 3500,
            "d1": "Premise musical background. " * 1000,
            "d2": "Parent supporting source. " * 2000}
    return BridgeSession(question, docs, list(docs), np.eye(3), None, Calls(), config,
                         Ledger({"ann": 10, "set_score": 0, "llm": 24, "reader": 0}))


def test_dense_and_conditional_views_are_bounded_but_source_records_and_ann_costs_stay_intact():
    session = long_session()
    before = dict(session.records)
    retriever = session._step_retriever("Find the musical event.", "event", ("d2",), 10)
    retriever.set_information_needs([{"id": "event", "description": "the event name"}])
    retriever.initialize_unexpanded()
    batch = retriever.propose("d0", ("d1",), fixed_pool=False)
    assert session.records == before
    assert batch.target_id == "d0" and batch.premise_ids == ("d1",)
    assert batch.excluded_ids == ("d0", "d1")
    assert len(session.calls.requests) == session.ledger.used["ann"] == 2
    for _, _, payload in session.calls.requests:
        assert navigation_tokens(payload["input"][0]) <= TEXT_TOKEN_LIMIT
    views = [event for event in session.ledger.events if event["event"] == "navigation_query_view"]
    assert [view["probe_stage"] for view in views] == ["initial_dense", "conditional"]
    assert all(view["bounded"] and view["node_id"] == "event" for view in views)
    assert {s["section"] for s in views[-1]["sources"]} == {"anchor", "premise", "parent"}


def test_initial_dense_fixed_overflow_fails_before_any_model_call_or_ann_charge():
    session = long_session(question="question " * 5000)
    retriever = session._step_retriever("Find the event.", "event", (), 10)
    with pytest.raises(NavigationQueryLimitError):
        retriever.initialize_unexpanded()
    assert session.calls.requests == [] and session.ledger.used["ann"] == 0
    assert session.ledger.events[-1]["status"] == "fixed_input_exceeds_capacity"


def test_unrelated_embedding_deployment_is_not_given_the_measured_qwen_capacity():
    session = long_session(known=False)
    assert not navigation_limits_apply(session.config)
    retriever = session._step_retriever("Find the event.", "event", ("d2",), 10)
    retriever.initialize_unexpanded()
    wire = session.calls.requests[0][2]["input"][0]
    assert session.records["d2"].text in wire
    assert navigation_tokens(wire) > TEXT_TOKEN_LIMIT
    assert not any(e["event"] == "navigation_query_view" for e in session.ledger.events)


@pytest.mark.parametrize("override", [{"model_profile": "legacy"}, {"embedding_model": "other-embedding"}])
def test_capacity_is_bound_to_both_model_and_profile(override):
    config = long_session().config
    config.update(override)
    assert not navigation_limits_apply(config)
