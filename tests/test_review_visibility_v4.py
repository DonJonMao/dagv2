"""Review visibility follows one provenance route per parent OR, never all ORs."""
from copy import deepcopy

import pytest

from dagbt import prompts
from dagbt.proof_review import REVIEW_SCHEMA
from dagbt.reasoning import ProtocolError
from dagbt.support import compile_graph, make_span
from test_engine import Document, step
from test_final_selection_v4 import support_selector
from test_support_review_v4 import proof, raw_span, review, update


def alternative(aid, source, parents=(), *, invalidated=False):
    return {"id": aid, "source_span_ids": [source], "guard_span_ids": [],
            "used_parent_ids": list(parents), "used_parent_versions": {parent: 0 for parent in parents},
            "applicable_scope": "fixture", "semantic_status": "supported",
            "invalidated_by": ["historical_retraction"] if invalidated else []}


def node(nid, alternatives, parents=()):
    return {"id": nid, "answer": nid + "_answer", "status": "supported", "version": 0,
            "applicable_scope": "fixture", "planned_parent_ids": list(parents),
            "alternatives": alternatives}


def branch_selector(responses=(), *, invalidated_child=False, joined=False, strict=False):
    docs = {"a": Document("a", "Short supported parent fact."),
            "b": Document("b", "中" * 20_000),
            "c": Document("c", "The child relation.")}
    assignments = [("a", "parent"), ("b", "parent"), ("c", "child")]
    nodes = [node("parent", [alternative("pa", "sa"), alternative("pb", "sb")]),
             node("child", [alternative("child_route", "sc", ["parent"], invalidated=invalidated_child)], ["parent"])]
    terminal = "child"
    if joined:
        docs.update(d=Document("d", "Second branch relation."), e=Document("e", "The combined conclusion."))
        assignments.extend([("d", "other"), ("e", "joined")])
        nodes.extend([node("other", [alternative("other_route", "sd", ["parent"])], ["parent"]),
                      node("joined", [alternative("joined_route", "se", ["child", "other"])], ["child", "other"])])
        terminal = "joined"
    spans = [make_span("s" + doc_id, doc_id, docs[doc_id].passage[:30], docs, start=0,
                       node_ids=[nid], node_id=nid, kind="explicit", stance="support",
                       claim="Scripted source relationship", source_role="document")
             for doc_id, nid in assignments]
    requirements = [{"id": "answer", "necessary": True, "terminal_node_ids": [terminal], "terminal_mode": "all"}]
    graph = compile_graph(nodes, spans, requirements, docs)
    selector, engine, calls, events = support_selector(responses, docs=docs, config={
        "context_tokens": 16_384, "max_repairs_per_request": 0, "allow_unassessed_coverage": not strict})
    selector.graph = deepcopy(graph)
    selector.proposal = {"selected_doc_ids": []}
    engine.graph = deepcopy(graph)
    engine.spans = {span["id"]: deepcopy(span) for span in spans}
    engine.nodes = deepcopy(graph["nodes"])
    engine.requirements = deepcopy(requirements)
    engine.steps = [step(n["id"], inputs=n["planned_parent_ids"]) for n in nodes]
    engine.baseline_ids = [d for d in docs if d != "b"]
    return selector, engine, calls, events


def assert_bounded_full_sources(view, engine):
    count = engine.reasoner.estimate("select", prompts.SUPPORT_REVIEW, view.data, REVIEW_SCHEMA)
    assert count == view.audit["input_tokens_after"]
    assert count <= view.audit["input_token_limit"]
    for raw in view.data["raw_memory_candidates"]:
        assert raw["passage"] == engine.docs[raw["doc_id"]].passage


def test_short_parent_or_keeps_child_visible_when_long_alternative_cannot_fit():
    selector, engine, _, _ = branch_selector()
    before = deepcopy(selector.graph)
    view = selector.prepare_view()
    assert view.visible_doc_ids == {"a", "c"}
    assert view.visible_alternative_ids == {"pa", "child_route"}
    assert view.visible_span_ids == {"sa", "sc"}
    assert view.audit["omitted_alternative_ids"] == ["pb"]
    assert view.audit["omitted_raw_review_doc_ids"] == ["b"]
    assert selector.graph == before
    assert_bounded_full_sources(view, engine)

    # Direct evidence alone cannot display a used parent, and a parent alone
    # cannot display the child's own premise. The other OR is equally valid
    # for visibility when its complete original source is supplied.
    assert selector._data(["c"])[1] == set()
    assert selector._data(["a"])[1] == {"pa"}
    assert selector._data(["b", "c"])[1] == {"pb", "child_route"}


@pytest.mark.parametrize("invalidated_child", [False, True])
def test_fully_visible_child_can_be_withdrawn_without_deleting_hidden_parent_route(invalidated_child):
    def withdraw(data):
        child = next(n for n in data["nodes"] if n["id"] == "child")
        assert [alt["id"] for alt in child["alternatives"]] == ["child_route"]
        assert child["alternatives"][0]["eligible"] is not invalidated_child
        return review(node_updates=[{
            "node_id": "child", "answer": None, "status": "unknown", "applicable_scope": "fixture",
            "retained_alternative_ids": [], "alternatives": [],
            "unresolved_inputs": ["Review withdrew the old relation"], "unresolved_guards": [],
            "reason": "The fully shown parent and child sources do not establish the old relation"}],
            supplemental_doc_ids=["c"])

    selector, engine, calls, _ = branch_selector([withdraw], invalidated_child=invalidated_child)
    old_parent = deepcopy(selector.graph["nodes"][0])
    result = selector.run()
    assert len(calls.requests) == 1 and result["20"]["review_complete"]
    assert selector.graph["nodes"][0] == old_parent
    assert {a["id"] for a in selector.graph["nodes"][0]["alternatives"]} == {"pa", "pb"}
    assert selector.graph["nodes"][1]["alternatives"] == []
    assert selector.graph["nodes"][1]["status"] == "unknown"
    assert not result["20"]["complete_required"] and result["20"]["selected_doc_ids"] == ["c"]
    assert_bounded_full_sources(selector.view, engine)


def test_long_hidden_parent_or_still_cannot_be_deleted_by_review():
    def drop_hidden(data):
        parent = next(n for n in data["nodes"] if n["id"] == "parent")
        assert [a["id"] for a in parent["alternatives"]] == ["pa"]
        return review(node_updates=[{
            "node_id": "parent", "answer": parent["answer"], "status": "supported", "applicable_scope": "fixture",
            "retained_alternative_ids": ["pa"], "alternatives": [], "unresolved_inputs": [],
            "unresolved_guards": [], "reason": "Attempt to drop an omitted OR"}])
    selector, _, calls, _ = branch_selector([drop_hidden], strict=True)
    before = deepcopy(selector.graph)
    with pytest.raises(ProtocolError, match="cannot_drop_invisible_alternative"):
        selector.run()
    assert selector.graph == before and len(calls.requests) == 1


def test_joined_branches_share_one_visible_parent_route_without_unioning_all_ors():
    selector, engine, _, _ = branch_selector(joined=True)
    groups = selector._display_source_groups()
    assert frozenset({"a", "c", "d", "e"}) in groups
    assert frozenset({"b", "c", "d", "e"}) in groups
    assert frozenset({"a", "b", "c", "d", "e"}) not in groups
    view = selector.prepare_view()
    assert view.visible_doc_ids == {"a", "c", "d", "e"}
    assert view.visible_alternative_ids == {"pa", "child_route", "other_route", "joined_route"}
    assert view.audit["omitted_alternative_ids"] == ["pb"]
    assert_bounded_full_sources(view, engine)


def missing_parent_selector(responses=(), *, strict=False):
    selector, engine, calls, events = branch_selector(responses, strict=strict)
    graph = deepcopy(selector.graph)
    graph["nodes"][0].update(alternatives=[], status="unknown", declared_status="unknown", answer=None)
    child_source = {**deepcopy(graph["spans"][0]), "id": "sa_child", "node_id": "child", "node_ids": ["child"]}
    graph["spans"].append(child_source)
    graph["nodes"][1]["alternatives"].append(alternative("child_other", "sa_child", ["parent"]))
    graph = compile_graph(graph["nodes"], graph["spans"], graph["requirements"], engine.docs)
    selector.graph = deepcopy(graph)
    engine.graph, engine.nodes = deepcopy(graph), deepcopy(graph["nodes"])
    engine.spans = {span["id"]: deepcopy(span) for span in graph["spans"]}
    return selector, engine, calls, events


@pytest.mark.parametrize("replace_with_direct", [False, True])
def test_missing_parent_is_explicit_and_broken_full_cap_routes_can_be_removed_or_replaced(replace_with_direct):
    def respond(data):
        parent, child = data["nodes"]
        assert parent["alternatives"] == [] and parent["omitted_alternative_ids"] == []
        assert child["omitted_alternative_ids"] == []
        assert len(child["alternatives"]) == 2
        for old_route in child["alternatives"]:
            assert not old_route["eligible"]
            assert old_route["missing_parent_proof_ids"] == ["parent"]
        if replace_with_direct:
            return review(new_spans=[raw_span(data, "independent_c", "c", "child")],
                          node_updates=[update("child", "child_answer", [proof(["independent_c"])])])
        return review(node_updates=[{
            "node_id": "child", "answer": None, "status": "unknown", "applicable_scope": "fixture",
            "retained_alternative_ids": [], "alternatives": [], "unresolved_inputs": ["Parent has no proof"],
            "unresolved_guards": [], "reason": "Withdraw explicitly broken old routes"}], supplemental_doc_ids=["c"])
    selector, engine, calls, _ = missing_parent_selector([respond])
    old_parent = deepcopy(selector.graph["nodes"][0])
    result = selector.run()
    assert len(calls.requests) == 1 and result["20"]["review_complete"]
    assert selector.graph["nodes"][0] == old_parent
    assert result["20"]["complete_required"] is replace_with_direct
    assert result["20"]["selected_doc_ids"] == ["c"]
    routes = selector.graph["nodes"][1]["alternatives"]
    assert len(routes) == int(replace_with_direct)
    if routes:
        assert routes[0]["used_parent_ids"] == [] and routes[0]["eligible"]
    assert_bounded_full_sources(selector.view, engine)


def test_displaying_missing_parent_never_makes_it_available_for_a_new_proof():
    def bind_missing(data):
        assert data["nodes"][1]["alternatives"][0]["missing_parent_proof_ids"] == ["parent"]
        return review(node_updates=[update("child", "child_answer", [proof(["sc"], parents=["parent"])])])
    selector, _, calls, _ = missing_parent_selector([bind_missing], strict=True)
    before = deepcopy(selector.graph)
    with pytest.raises(ProtocolError, match="parent_unknown_unavailable_nonpreceding_or_invisible"):
        selector.run()
    assert selector.graph == before and len(calls.requests) == 1


def test_budget_hidden_parent_sources_do_not_count_as_an_explicitly_missing_parent():
    def drop_hidden_child(data):
        parent, child = data["nodes"]
        assert parent["alternatives"] == [] and parent["omitted_alternative_ids"] == ["pb"]
        assert child["alternatives"] == [] and child["omitted_alternative_ids"] == ["child_route"]
        return review(node_updates=[update("child", "child_answer", [proof(["direct_c"])])],
                      new_spans=[raw_span(data, "direct_c", "c", "child")])
    selector, engine, calls, _ = branch_selector([drop_hidden_child], strict=True)
    graph = deepcopy(selector.graph)
    graph["nodes"][0]["alternatives"] = graph["nodes"][0]["alternatives"][1:]
    selector.graph = compile_graph(graph["nodes"], graph["spans"], graph["requirements"], engine.docs)
    engine.graph = deepcopy(selector.graph)
    before = deepcopy(selector.graph)
    with pytest.raises(ProtocolError, match="cannot_drop_invisible_alternative"):
        selector.run()
    assert selector.graph == before and len(calls.requests) == 1
