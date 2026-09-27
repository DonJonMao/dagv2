"""Whole-record budget views and citation closure use actual Reasoner estimates."""
from copy import deepcopy

import pytest

from dagbt.evidence_views import build_view
from dagbt.reasoning import InputOverflow, ProtocolError
from test_reasoning_v2 import reasoner


def span(identifier, doc, node="n1", quote="Exact source text.", stance="support", **extra):
    return {"id": identifier, "doc_id": doc, "node_ids": [node], "stance": stance,
            "exact_quote": quote, "start": 0, "end": len(quote), "event_time": None,
            "source_role": "user", "metadata": {"raw_role": "authoritative"}, **extra}


def alternative(identifier, spans, parents=()):
    return {"id": identifier, "source_span_ids": list(spans), "guard_span_ids": [],
            "used_parent_ids": list(parents), "semantic_status": "supported", "eligible": True}


def node(identifier, alternatives):
    return {"id": identifier, "answer": "model claim", "status": "supported", "version": 1,
            "alternatives": alternatives, "partial_span_ids": []}


def build(records, fixed=None, **kwargs):
    r, calls, ledger, events = reasoner([])
    view = build_view(r, "resolve", "Resolve using visible evidence.", fixed or {"original_question": "Q?"},
                      records, r.settings, events.append, **kwargs)
    assert not calls.requests and not ledger.used["llm"]
    assert view.audit["input_tokens_after"] == r.estimate("resolve", "Resolve using visible evidence.", view.data)
    return view, r, events


def test_full_records_metadata_fragments_and_literal_text_are_preserved():
    fragment = {"id": "raw_span_long_1", "doc_id": "doc", "exact_quote": "Literal e1 and a1"}
    record = span("persistent_long_span_id", "doc", quote="persistent_long_span_id e1 a1",
                  fragments=[fragment], premise_group_ids=["original-sentence"])
    before = deepcopy(record)
    view, _, events = build([record])
    shown = view.data["evidence"][0]
    assert shown["id"] == "e1"
    assert {**shown, "id": record["id"]} == before
    assert record == before
    response = {"alternatives": [{"source_span_ids": ["e1"], "guard_span_ids": []}],
                "answer": "literal e1", "reason": "persistent_long_span_id e1 a1"}
    decoded = view.decode(response)
    assert decoded["alternatives"][0]["source_span_ids"] == [record["id"]]
    assert decoded["answer"] == response["answer"] and decoded["reason"] == response["reason"]
    assert view.encode(decoded) == response
    assert events[-1]["policy_version"] == "dag_node_document_round_robin_atomic_support_v2"


def test_round_robin_across_nodes_and_documents_with_contradiction_first():
    records = [span("a1", "d1"), span("a2", "d1"), span("a3", "d2"),
               span("b1", "d3", "n2"), span("b2", "d3", "n2"),
               span("c1", "d1", stance="contradiction")]
    view, _, _ = build(records, node_order=["n1", "n2"])
    assert view.audit["visible_span_ids"] == ["c1", "a1", "b1", "a3", "b2", "a2"]
    assert not view.audit["truncated"]


def test_multinode_record_is_emitted_once():
    records = [span("shared", "d", node_ids=["n1", "n2"]), span("other", "x", "n2")]
    view, _, _ = build(records, node_order=["n1", "n2"])
    assert view.audit["visible_span_ids"] == ["shared", "other"]


def test_large_ledger_is_record_truncated_and_flat_costs_follow_visible_documents():
    records = [span(f"s{i}", f"doc{i}", f"n{i % 3}", quote=("中" * 800)) for i in range(12)]
    costs = {f"doc{i}": i + 100 for i in range(12)}
    view, r, _ = build(records, document_costs=costs)
    assert view.audit["truncated"] and 0 < len(view.visible_span_ids) < len(records)
    assert view.audit["input_tokens_after"] <= 4096 - 128 - 8 - 256
    assert r.estimate("resolve", "Resolve using visible evidence.", view.data) <= view.audit["input_token_limit"]
    assert set(view.data["candidate_doc_ids"]) == view.visible_doc_ids
    assert view.data["document_token_counts"] == {d: costs[d] for d in view.visible_doc_ids}
    assert set(view.audit["available_span_ids"]) == set(view.audit["visible_span_ids"]) | set(view.audit["omitted_span_ids"])
    for shown in view.data["evidence"]:
        assert shown["exact_quote"] == "中" * 800
    assert view.data["evidence_visibility"]["omitted_record_count"] > 0


def test_omitted_persistent_ids_and_aliases_cannot_be_cited():
    records = [span(f"long_persistent_{i}", f"d{i}", quote="中" * 2000) for i in range(4)]
    view, _, _ = build(records)
    omitted = view.audit["omitted_span_ids"][0]
    omitted_alias = "e" + str(next(i for i, r in enumerate(records) if r["id"] == omitted) + 1)
    for reference in (omitted, omitted_alias, "invented", next(iter(view.visible_span_ids))):
        with pytest.raises(ProtocolError, match="unknown or invisible"):
            view.decode({"source_span_ids": [reference]})
    with pytest.raises(ProtocolError):
        view.decode({"source_span_ids": list(view.span_alias_to_id) * 2})


def test_fixed_query_overflow_is_typed_and_happens_without_calls():
    r, calls, _, events = reasoner([])
    with pytest.raises(InputOverflow, match="irreducible fixed fields"):
        build_view(r, "resolve", "Resolve.", {"original_question": "中" * 5000}, [], r.settings, events.append)
    assert not calls.requests and events[-1]["event"] == "evidence_view_irreducible_overflow"


def test_schema_output_and_input_margin_are_included_in_actual_wire_guard():
    r, _, _, _ = reasoner([], response_format="json_schema", input_margin=512)
    schema = {"type": "object", "properties": {"claim": {"type": "string", "description": "中" * 200}},
              "required": ["claim"], "additionalProperties": False}
    view = build_view(r, "resolve", "Resolve.", {"query": "Q"},
                      [span(f"s{i}", f"d{i}", quote="中" * 600) for i in range(12)],
                      r.settings, schema=schema)
    assert view.audit["input_tokens_after"] == r.estimate("resolve", "Resolve.", view.data, schema)
    assert view.audit["input_tokens_after"] + 128 + 8 + 512 <= 4096


def test_resolver_drops_parent_and_descendants_without_visible_support_closure():
    records = [span("root_source", "root_doc", quote="中" * 6000), span("child_source", "child_doc")]
    parents = [node("p1", [alternative("p1a", ["root_source"])]),
               node("p2", [alternative("p2a", ["child_source"], ["p1"])])]
    fixed = {"original_question": "Q", "supported_parents": parents}
    view, _, _ = build(records, fixed)
    assert "child_source" in view.visible_span_ids and "root_source" not in view.visible_span_ids
    assert view.data["supported_parents"] == []
    assert view.data["omitted_supported_parent_ids"] == ["p1", "p2"]
    assert not view.visible_alternative_ids
    assert parents[0]["alternatives"][0]["source_span_ids"] == ["root_source"]


def test_resolver_keeps_complete_parent_route_and_encodes_prior_alternatives():
    records = [span("root", "d1"), span("child", "d2")]
    fixed = {"supported_parents": [node("p1", [alternative("pa", ["root"])])],
             "prior_state": node("target", [alternative("ta", ["child"], ["p1"])])}
    view, _, _ = build(records, fixed)
    assert view.data["supported_parents"][0]["id"] == "p1"
    assert view.data["supported_parents"][0]["alternatives"][0]["id"] == "a1"
    assert view.data["prior_state"]["alternatives"][0]["source_span_ids"] == ["e2"]
    assert view.visible_alternative_ids == {"pa", "ta"}


def test_audit_every_visible_alternative_has_all_own_and_parent_spans():
    records = [span("root", "d1", "p1", quote="中" * 450),
               span("child", "d2", "p2", quote="中" * 450),
               span("guard", "d3", "p2", quote="中" * 450),
               span("counter", "d4", "p2", quote="中" * 450, stance="contradiction")]
    child = alternative("child_alt", ["child"], ["p1"])
    child["guard_span_ids"] = ["guard"]
    nodes = [node("p1", [alternative("root_alt", ["root"])]), node("p2", [child])]
    results = []
    for limit in (1600, 2400, 3600):
        r, _, _, _ = reasoner([])
        view = build_view(r, "audit", "Audit.", {"original_question": "Q"}, records,
                          r.settings, audit_nodes=nodes, input_limit=limit)
        results.append(view)
        for shown_node in view.data["nodes"]:
            for shown_alt in shown_node["alternatives"]:
                decoded = view.decode({"alternative_ids": [shown_alt["id"]],
                                       "source_span_ids": shown_alt["source_span_ids"],
                                       "guard_span_ids": shown_alt["guard_span_ids"]})
                assert set(decoded["source_span_ids"] + decoded["guard_span_ids"]) <= view.visible_span_ids
                if decoded["alternative_ids"] == ["child_alt"]:
                    assert {"root", "child", "guard"} <= view.visible_span_ids
                    assert "root_alt" in view.visible_alternative_ids
        assert view.audit["input_tokens_after"] <= limit
    assert any(view.audit["truncated"] for view in results)
    assert any("child_alt" in view.visible_alternative_ids for view in results)


def test_conflict_resolution_proof_is_not_partially_rewritten_when_omitted():
    records = [span("counter", "d1"), span("proof", "d2", quote="中" * 6000)]
    conflict = {"id": "conflict_1", "source_span_ids": ["counter"],
                "resolution": {"source_span_ids": ["proof"], "addressed_conflict_span_ids": ["counter"]}}
    view, _, _ = build(records, {"known_conflicts": [conflict]})
    assert view.data["known_conflicts"] == []
    assert view.data["omitted_known_conflicts_count"] == 1
    assert conflict["resolution"]["source_span_ids"] == ["proof"]


def test_view_construction_is_deterministic_and_does_not_mutate_records():
    records = [span("shared", "d1", quote="中" * 1800), span("other", "d2", quote="中" * 1800)]
    before = deepcopy(records)
    first, _, _ = build(records)
    second, _, _ = build(records)
    assert first.data == second.data and first.audit == second.audit
    assert records == before


def test_audit_rejects_unknown_raw_support_instead_of_manufacturing_visible_proof():
    with pytest.raises(ProtocolError, match="unknown evidence"):
        build([span("s1", "d")], audit_nodes=[node("n1", [alternative("a1", ["missing"])])])


@pytest.mark.parametrize("bad_fields", [
    {"eligible": False, "semantic_status": "partial"},
    {"eligible": False, "semantic_status": "supported"},
    {"eligible": True, "semantic_status": "partial"},
    {"eligible": True, "semantic_status": "supported", "disputed_by": ["conflict_1"]},
    {"eligible": True, "semantic_status": "supported", "invalidated_by": ["conflict_1"]},
    {"eligible": None, "semantic_status": "supported"},
])
def test_parent_with_only_visible_bad_route_is_removed_despite_supported_node_status(bad_fields):
    records = [span("bad_source", "bad_doc"), span("good_source", "good_doc", quote="中" * 6000),
               span("child_source", "child_doc")]
    bad = {**alternative("bad_alt", ["bad_source"]), **bad_fields}
    good = alternative("good_alt", ["good_source"])
    parents = [node("parent", [bad, good]),
               node("child", [alternative("child_alt", ["child_source"], ["parent"])])]
    view, _, _ = build(records, {"supported_parents": parents})
    assert {"bad_source", "child_source"} <= view.visible_span_ids
    assert "good_source" not in view.visible_span_ids
    assert view.data["supported_parents"] == []
    assert view.data["omitted_supported_parent_ids"] == ["parent", "child"]
    assert not view.visible_alternative_ids
    with pytest.raises(ProtocolError, match="unknown or invisible"):
        view.decode({"alternative_ids": ["a1"]})


def test_audit_can_show_bad_route_without_using_it_as_parent_basis():
    records = [span("bad_source", "bad_doc"), span("good_source", "good_doc", quote="中" * 6000),
               span("child_source", "child_doc")]
    bad = {**alternative("bad_alt", ["bad_source"]), "eligible": False, "semantic_status": "partial"}
    good = alternative("good_alt", ["good_source"])
    nodes = [node("parent", [bad, good]),
             node("child", [alternative("child_alt", ["child_source"], ["parent"])])]
    r, _, _, _ = reasoner([])
    view = build_view(r, "audit", "Audit.", {"original_question": "Q"}, records,
                      r.settings, audit_nodes=nodes)
    assert "bad_alt" in view.visible_alternative_ids
    assert "good_alt" not in view.visible_alternative_ids and "child_alt" not in view.visible_alternative_ids
    assert view.data["nodes"][0]["alternatives"][0]["eligible"] is False
    assert view.data["nodes"][0]["established_in_view"] is False
    assert view.data["nodes"][1]["alternatives"] == []


def test_audit_parent_route_chooses_eligible_alternative_after_ineligible_first():
    records = [span("bad_source", "bad_doc", quote="中" * 6000), span("good_source", "good_doc"),
               span("child_source", "child_doc")]
    bad = {**alternative("bad_alt", ["bad_source"]), "eligible": False, "semantic_status": "partial"}
    good = alternative("good_alt", ["good_source"])
    nodes = [node("parent", [bad, good]),
             node("child", [alternative("child_alt", ["child_source"], ["parent"])])]
    r, _, _, _ = reasoner([])
    view = build_view(r, "audit", "Audit.", {"original_question": "Q"}, records,
                      r.settings, audit_nodes=nodes)
    assert {"good_alt", "child_alt"} <= view.visible_alternative_ids
    assert "bad_alt" not in view.visible_alternative_ids
    assert all(n["established_in_view"] for n in view.data["nodes"])
