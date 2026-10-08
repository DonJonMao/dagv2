"""Offline protocol tests for atomic review of bounded AND/OR support proofs."""
from copy import deepcopy

import pytest

from dagbt.proof_review import REVIEW_SCHEMA, apply_review
from dagbt.support import (SupportError, compile_graph, invalidate_support, make_span,
                          select_support, with_navigation_closure)


DOCS = {"a": "A establishes the writer.", "b": "B gives the birthplace.",
        "c": "C directly says the requested writer was born in France.",
        "d": "D independently establishes the birthplace.",
        "x": "The earlier claim refers to another person.",
        "g": "The corrected scope is the requested person.",
        "nav": "Navigation clue only."}


def alt(aid, sources=(), parents=(), guards=(), **extra):
    return {"id": aid, "source_span_ids": list(sources), "guard_span_ids": list(guards),
            "used_parent_ids": list(parents), "used_parent_versions": {p: 0 for p in parents},
            "semantic_status": "supported", "applicable_scope": "scope", **extra}


def node(nid, alternatives=(), *, answer="France", status="supported", parents=()):
    return {"id": nid, "answer": answer, "status": status, "version": 0,
            "applicable_scope": "scope", "planned_parent_ids": list(parents),
            "alternatives": list(alternatives), "unresolved_inputs": [], "unresolved_guards": []}


def graph(nodes=None, documents=None):
    documents = documents or DOCS
    nodes = nodes or [node("n", [alt("old", ["sa"])])]
    spans = [make_span("s" + doc_id, doc_id, text if isinstance(text, str) else text["passage"],
                       documents, start=0, node_ids=[n["id"] for n in nodes])
             for doc_id, text in documents.items()]
    return compile_graph(nodes, spans, [{"id": "req", "necessary": True,
        "terminal_node_ids": [nodes[-1]["id"]], "terminal_mode": "all"}], documents)


def review(**changes):
    return {"new_spans": [], "invalidations": [], "resolutions": [], "node_updates": [],
            "supplemental_doc_ids": [], "reason": "Review the available raw proofs", **changes}


def proof(sources=(), parents=(), guards=(), **changes):
    return {"source_span_ids": list(sources), "guard_span_ids": list(guards),
            "used_parent_ids": list(parents), "semantic_status": "supported",
            "applicable_scope": "scope", **changes}


def update(nid="n", alternatives=(), retained=(), **changes):
    return {"node_id": nid, "answer": "France", "status": "supported", "applicable_scope": "scope",
            "retained_alternative_ids": list(retained), "alternatives": list(alternatives),
            "unresolved_inputs": [], "unresolved_guards": [], "reason": "Ground a fresh proof", **changes}


def new_span(alias="direct", doc_id="c", nid="n", **changes):
    return {"id": alias, "doc_id": doc_id, "start": 0, "quote": DOCS[doc_id], "node_id": nid,
            "kind": "explicit", "stance": "support", "claim": "The answer is directly stated",
            "entity_scope": "scope", **changes}


def apply(g, payload, documents=None, **visibility):
    documents = documents or DOCS
    return apply_review(g, payload, documents, **{
        "visible_doc_ids": set(documents),
        "visible_span_ids": {s["id"] for s in g["spans"]},
        "visible_alternative_ids": {a["id"] for n in g["nodes"] for a in n["alternatives"]},
        **visibility})


def select(g):
    return select_support(g, lambda docs: {"feasible": True, "token_count": len(docs)})


def test_new_raw_direct_route_replaces_multihop_without_planned_parent_dependency():
    g = graph([node("writer", [alt("parent", ["sa"])], answer="Name"),
               node("n", [alt("old", ["sb"], ["writer"])], parents=["writer"])])
    payload = review(new_spans=[new_span()], node_updates=[update(alternatives=[proof(["direct"])])])
    revised = apply(g, payload)
    assert select(g)["selected_doc_ids"] == ["a", "b"]
    assert select(revised)["selected_doc_ids"] == ["c"]
    assert revised["nodes"][1]["planned_parent_ids"] == ["writer"]
    assert revised["nodes"][1]["alternatives"][0]["used_parent_ids"] == []
    assert revised["nodes"][1]["version"] == 0
    assert revised["requirements"] == g["requirements"]


def test_updates_sort_topologically_and_new_child_binds_current_parent_versions():
    g = graph([node("left", [alt("left.old", ["sa"])], answer="old"),
               node("right", [alt("right.old", ["sb"])], answer="old"),
               node("n", [alt("old", ["sc"], ["left", "right"])])])
    revised = apply(g, review(node_updates=[
        update(alternatives=[proof(["sc"], ["left", "right"], ["sg"])]),
        update("right", [proof(["sb"])], answer="new right"),
        update("left", [proof(["sa"])], answer="new left")]))
    route = revised["nodes"][-1]["alternatives"][0]
    assert route["used_parent_versions"] == {"left": 1, "right": 1}
    assert revised["proof_review"]["updated_node_ids"] == ["left", "right", "n"]
    assert select(revised)["selected_doc_ids"] == ["a", "b", "c", "g"]


@pytest.mark.parametrize("change", [{"answer": "Germany"}, {"applicable_scope": "new scope"}])
def test_answer_or_scope_change_invalidates_untouched_descendants(change):
    g = graph([node("parent", [alt("parent.old", ["sa"]) ]),
               node("child", [alt("child.old", ["sb"], ["parent"]) ]),
               node("n", [alt("old", ["sc"], ["child"]) ])])
    proposal = proof(["sa"], applicable_scope=change.get("applicable_scope", "scope"))
    revised = apply(g, review(node_updates=[update("parent", [proposal], **change)]))
    assert revised["nodes"][0]["version"] == 1
    assert revised["nodes"][0]["status"] == "supported"
    assert "parent_version_changed:parent" in revised["nodes"][1]["alternatives"][0]["ineligible_reasons"]
    assert "parent_unavailable:child" in revised["nodes"][2]["alternatives"][0]["ineligible_reasons"]
    assert not select(revised)["complete_required"]


def test_shared_document_does_not_join_independent_or_routes_or_transfer_disputes():
    g = graph([node("n", [alt("bad", ["sa", "sb"]), alt("clean", ["sa", "sc"])])])
    g = invalidate_support(g, ["bad"], ["sx"], disputed=True)
    revised = apply(g, review(node_updates=[update(
        alternatives=[proof(["sa", "sd"])], retained=["clean"])]))
    clean, fresh = revised["nodes"][0]["alternatives"]
    assert clean == g["nodes"][0]["alternatives"][1]
    assert fresh["disputed_by"] == ["conflict_1"]
    assert not fresh["eligible"]
    assert select(revised)["node_closures"]["n"] == ["a", "c"]
    assert select(revised)["chosen_alternatives"] == {"n": "clean"}
    assert revised["nodes"][0]["status"] == "supported"


@pytest.mark.parametrize("changes,match", [
    ({"quote": "not an exact quote"}, "quote_not_exact"),
    ({"start": True}, "quote_offsets_invalid"),
    ({"start": -1}, "quote_offsets_invalid"),
    ({"start": 10000}, "quote_offsets_invalid"),
    ({"quote": "x" * 401}, "oversize"),
    ({"quote": " "}, "invalid_or_oversize"),
    ({"node_id": "invented"}, "unknown_node"),
    ({"kind": "inferred"}, "invalid_new_span_assessment"),
])
def test_new_quotes_reject_invalid_grounding(changes, match):
    with pytest.raises(SupportError, match=match):
        apply(graph(), review(new_spans=[new_span(**changes)]))


def test_raw_document_visibility_and_existing_span_visibility_are_separate_boundaries():
    g = graph()
    with pytest.raises(SupportError, match="document_not_visible"):
        apply(g, review(new_spans=[new_span()]), visible_doc_ids=["a"])
    with pytest.raises(SupportError, match="span_not_visible"):
        apply(g, review(node_updates=[update(alternatives=[proof(["sc"])])]), visible_span_ids=["sa"])
    with pytest.raises(SupportError, match="supplemental_document_not_visible"):
        apply(g, review(supplemental_doc_ids=["c"]), visible_doc_ids=["a"])


def test_wrong_node_evidence_cannot_be_reused_as_a_fresh_proof():
    g = graph([node("other", [], answer=None, status="unknown"), node("n", [alt("old", ["sa"])])])
    g["spans"][2]["node_ids"] = ["other"]
    with pytest.raises(SupportError, match="evidence_not_mapped_to_node"):
        apply(g, review(node_updates=[update(alternatives=[proof(["sc"])])]))
    with pytest.raises(SupportError, match="evidence_not_mapped_to_node"):
        apply(g, review(new_spans=[new_span(nid="other")],
                        node_updates=[update(alternatives=[proof(["direct"])])]))


def test_invisible_routes_must_be_explicitly_retained_and_cannot_be_invalidated():
    g = graph([node("n", [alt("visible", ["sa"]), alt("hidden", ["sb"])])])
    with pytest.raises(SupportError, match="cannot_drop_invisible"):
        apply(g, review(node_updates=[update(retained=["visible"])]), visible_alternative_ids=["visible"])
    with pytest.raises(SupportError, match="invalidation_target_not_visible"):
        apply(g, review(invalidations=[{"alternative_ids": ["hidden"], "source_span_ids": ["sx"],
            "reason": "Wrong entity", "disputed": True}]), visible_alternative_ids=["visible"])
    revised = apply(g, review(node_updates=[update(alternatives=[proof(["sc"])], retained=["hidden"])]),
                    visible_alternative_ids=["visible"])
    assert revised["nodes"][0]["alternatives"][0] == g["nodes"][0]["alternatives"][1]


@pytest.mark.parametrize("change", [{"answer": "new answer"}, {"status": "partial"},
                                    {"applicable_scope": "changed"}, {"unresolved_guards": ["missing time"]}])
def test_hidden_route_blocks_indirect_node_mutation(change):
    with pytest.raises(SupportError, match="cannot_mutate_node_with_invisible"):
        apply(graph(), review(node_updates=[update(retained=["old"], **change)]), visible_alternative_ids=[])


def test_retained_route_preserves_invalidated_state_and_does_not_become_supported():
    g = invalidate_support(graph(), ["old"], ["sx"], disputed=False)
    revised = apply(g, review(node_updates=[update(retained=["old"])]))
    assert revised["nodes"][0]["alternatives"] == g["nodes"][0]["alternatives"]
    assert revised["nodes"][0]["status"] == "invalidated"


def test_changed_answer_cannot_transfer_old_retained_route():
    with pytest.raises(SupportError, match="changed_node_requires_fresh"):
        apply(graph(), review(node_updates=[update(retained=["old"], answer="Germany")]))


def test_new_parent_requires_a_visible_complete_eligible_route_not_opaque_supported_metadata():
    g = graph([node("p", [alt("parent", ["sa"]) ]), node("n", [alt("old", ["sb"])])])
    payload = review(node_updates=[update(alternatives=[proof(["sc"], ["p"])])])
    with pytest.raises(SupportError, match="parent_unknown_unavailable_nonpreceding_or_invisible"):
        apply(g, payload, visible_alternative_ids=["old"])
    with pytest.raises(SupportError, match="parent_unknown_unavailable_nonpreceding_or_invisible"):
        apply(g, payload, visible_span_ids=["sb", "sc"])
    assert apply(g, payload)["nodes"][1]["alternatives"][0]["used_parent_versions"] == {"p": 0}


def test_forward_parent_and_unavailable_parent_are_rejected():
    g = graph([node("n", [alt("old", ["sa"]) ]), node("later", [alt("later", ["sb"])])])
    with pytest.raises(SupportError, match="nonpreceding"):
        apply(g, review(node_updates=[update(alternatives=[proof(["sa"], ["later"])])]))
    g = graph([node("p", [], answer=None, status="unknown"), node("n", [alt("old", ["sa"])])])
    with pytest.raises(SupportError, match="unavailable"):
        apply(g, review(node_updates=[update(alternatives=[proof(["sa"], ["p"])])]))


def resolution(**changes):
    return {"conflict_id": "conflict_1", "resolution_span_ids": ["sg"],
            "addressed_conflict_span_ids": ["sx", "sb"], "reason": "All contrary claims concern another person",
            "resolution_kind": "entity_distinction", **changes}


def test_explicit_resolution_addresses_all_counterevidence_and_remains_in_new_route_guards():
    g = invalidate_support(graph(), ["old"], ["sx", "sb"], disputed=True)
    with pytest.raises(SupportError, match="address_all_opposing"):
        apply(g, review(resolutions=[resolution(addressed_conflict_span_ids=["sx"])]))
    revised = apply(g, review(resolutions=[resolution()],
                              node_updates=[update(alternatives=[proof(["sc"])])]))
    route = revised["nodes"][0]["alternatives"][0]
    assert route["guard_span_ids"] == ["sg"]
    assert route["disputed_by"] == [] and route["eligible"]
    assert select(revised)["selected_doc_ids"] == ["c", "g"]
    assert revised["conflicts"][0]["resolution_status"] == "resolved"


def test_resolution_cannot_mutate_hidden_routes_or_use_hidden_opposing_or_guard_quotes():
    g = invalidate_support(graph(), ["old"], ["sx", "sb"], disputed=True)
    with pytest.raises(SupportError, match="modify_invisible"):
        apply(g, review(resolutions=[resolution()]), visible_alternative_ids=[])
    with pytest.raises(SupportError, match="span_not_visible"):
        apply(g, review(resolutions=[resolution()]), visible_span_ids=["sg", "sx"])
    resolved = apply(g, review(resolutions=[resolution()]))
    with pytest.raises(SupportError, match="inherited_resolution_proof_not_visible"):
        apply(resolved, review(node_updates=[update(alternatives=[proof(["sc"])])]), visible_span_ids=["sc"])


def test_noop_and_raw_supplement_cannot_promote_unknown_nodes():
    g = graph([node("n", [], status="unknown", answer=None)])
    revised = apply(g, review(supplemental_doc_ids=["c"]))
    assert revised["nodes"] == g["nodes"] and revised["spans"] == g["spans"]
    assert revised["supplemental_doc_ids"] == ["c"]
    assert not select(revised)["complete_required"]
    assert select(revised)["selected_doc_ids"] == []


def test_frozen_navigation_and_graph_metadata_survive_compilation():
    g = with_navigation_closure(graph(), {"nav": [], "a": ["nav"]}, DOCS)
    g["engine_extra"] = {"keep": [1, 2]}
    revised = apply(g, review(new_spans=[new_span()], node_updates=[update(alternatives=[proof(["direct"])])]))
    assert revised["navigation_closure"] == g["navigation_closure"]
    assert revised["engine_extra"] == g["engine_extra"]
    assert "nav" in revised["document_order"]
    assert revised["requirements_hash"] == g["requirements_hash"]


def test_authoritative_roles_and_coordinates_are_preserved_without_guessing_event_time():
    text = "Assistant: old answer. User: corrected answer."
    split = text.index("User:")
    documents = {"a": {"passage": text, "metadata": {"source_segments": [
        {"start": 0, "end": split, "role": "assistant", "source_message_indices": [1]},
        {"start": split, "end": len(text), "role": "user", "source_message_indices": [2]}],
        "time": {"observation_order": 22}}}}
    g = graph(documents=documents)
    revised = apply(g, review(new_spans=[new_span(doc_id="a", quote=text)],
                              node_updates=[update(alternatives=[proof(["direct"])])]), documents)
    span = revised["spans"][-1]
    assert span["source_role"] == "ambiguous"
    assert span["source_message_indices"] == [1, 2]
    assert span["event_time"] is None
    assert span["assessment_origin"] == "proof_review" and span["node_ids"] == ["n"]
    assert len(span["source_segments"]) == 2 and len(span["premise_group_ids"]) >= 2
    assert span["start"] == 0 and span["end"] == len(text)
    documents = {"a": text}
    revised = apply(graph(documents=documents), review(new_spans=[new_span(doc_id="a", quote=text)]), documents)
    assert revised["spans"][-1]["source_role"] == "document"


def test_canonical_span_and_route_ids_do_not_depend_on_response_local_alias():
    g = graph()
    first = apply(g, review(new_spans=[new_span(alias="first")],
                            node_updates=[update(alternatives=[proof(["first"])])]))
    second = apply(g, review(new_spans=[new_span(alias="second")],
                             node_updates=[update(alternatives=[proof(["second"])])]))
    assert first["spans"] == second["spans"]
    assert first["nodes"] == second["nodes"]
    with pytest.raises(SupportError, match="identity_already_exists"):
        apply(first, review(node_updates=[update(alternatives=[proof([first["spans"][-1]["id"]])])]))


@pytest.mark.parametrize("changes,match", [
    ({"answer": "Germany"}, "fields differ"),
    ({"applicable_scope": "unrelated"}, "scope_mismatch"),
    ({"id": "forged_permanent_id"}, "fields differ"),
    ({"used_parent_versions": {"p": 999}}, "fields differ"),
    ({"invalidated_by": []}, "fields differ"),
])
def test_new_alternative_cannot_forge_route_state_or_mix_conclusions(changes, match):
    with pytest.raises(SupportError, match=match):
        apply(graph(), review(node_updates=[update(alternatives=[proof(["sc"], **changes)])]))


def test_supported_claim_requires_all_declared_gaps_closed():
    with pytest.raises(SupportError, match="unresolved_premises"):
        apply(graph(), review(node_updates=[update(alternatives=[proof(["sc"])], unresolved_guards=["unknown date"])]))


def test_atomic_copy_on_write_for_success_and_late_failure():
    documents = deepcopy(DOCS)
    g = graph(documents=documents)
    payload = review(new_spans=[new_span()], invalidations=[{
        "alternative_ids": ["old"], "source_span_ids": ["sx"], "reason": "Wrong scope", "disputed": True}],
        node_updates=[update(alternatives=[proof(["direct"], applicable_scope="wrong scope")])])
    before = deepcopy((g, payload, documents))
    with pytest.raises(SupportError, match="scope_mismatch"):
        apply(g, payload, documents)
    assert (g, payload, documents) == before
    payload["node_updates"][0]["alternatives"][0]["applicable_scope"] = "scope"
    before = deepcopy((g, payload, documents))
    revised = apply(g, payload, documents)
    assert (g, payload, documents) == before
    assert revised["nodes"][0]["status"] == "ambiguous"
    assert revised["proof_review"]["dropped_alternative_ids"] == ["old"]


def test_schema_matches_required_payload_and_rejects_extra_fields():
    assert set(REVIEW_SCHEMA["required"]) == set(review())
    with pytest.raises(SupportError, match="fields differ"):
        apply(graph(), {**review(), "selected_doc_ids": ["a"]})


def test_invalidation_requires_nonempty_raw_proof():
    with pytest.raises(SupportError, match="invalidation source_span_ids"):
        apply(graph(), review(invalidations=[{"alternative_ids": ["old"], "source_span_ids": [],
            "reason": "Unsupported model accusation", "disputed": False}]))


@pytest.mark.parametrize("new_quote", [False, True])
def test_hard_invalidated_proof_cannot_reappear_under_a_new_id_or_quote_alias(new_quote):
    g = invalidate_support(graph(), ["old"], ["sx"], disputed=False)
    payload = review(new_spans=[new_span(doc_id="a")] if new_quote else [],
                     node_updates=[update(alternatives=[proof(["direct" if new_quote else "sa"])])])
    with pytest.raises(SupportError, match="cannot_recreate_invalidated_proof"):
        apply(g, payload)


def test_hard_invalidated_proof_cannot_reappear_after_dropping_it_in_an_earlier_review():
    g = invalidate_support(graph(), ["old"], ["sx"], disputed=False)
    first = apply(g, review(node_updates=[update(alternatives=[proof(["sc"])])]))
    assert first["proof_review"]["invalidated_proof_signatures"]
    with pytest.raises(SupportError, match="cannot_recreate_invalidated_proof"):
        apply(first, review(node_updates=[update(alternatives=[proof(["sa"])])]))


def test_same_raw_quote_can_be_reassessed_for_a_new_answer_or_scope():
    g = invalidate_support(graph(), ["old"], ["sx"], disputed=False)
    revised = apply(g, review(node_updates=[update(alternatives=[proof(["sa"])], answer="Germany")]))
    assert revised["nodes"][0]["status"] == "supported"
    assert revised["nodes"][0]["version"] == 1


def test_splitting_a_hard_invalidated_quote_does_not_make_a_new_proof():
    g = invalidate_support(graph(), ["old"], ["sx"], disputed=False)
    text = DOCS["a"]
    quotes = [new_span(alias="part1", doc_id="a", quote=text[:10]),
              new_span(alias="part2", doc_id="a", start=10, quote=text[10:])]
    with pytest.raises(SupportError, match="cannot_recreate_invalidated_proof"):
        apply(g, review(new_spans=quotes,
                        node_updates=[update(alternatives=[proof(["part1", "part2"])])]))


def test_schema_is_compatible_with_strict_structured_output():
    def check(value):
        if isinstance(value, dict):
            if value.get("type") == "object":
                assert set(value["required"]) == set(value["properties"])
                assert value["additionalProperties"] is False
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)
    check(REVIEW_SCHEMA)


def hidden_partial_graph(*, semantic_status="partial"):
    item = node("n", [alt("hidden", ["sa", "sb"], semantic_status=semantic_status)], status="partial")
    item["unresolved_inputs"] = ["Old multi-hop route lacks a parent"]
    item["unresolved_guards"] = ["Old route lacks a date condition"]
    return graph([item])


def test_new_complete_direct_route_can_upgrade_node_while_retaining_hidden_partial_route():
    g = hidden_partial_graph()
    payload = review(new_spans=[new_span()], node_updates=[update(
        alternatives=[proof(["direct"])], retained=["hidden"])])
    revised = apply(g, payload, visible_alternative_ids=[], visible_span_ids=[], visible_doc_ids=["c"])
    current = revised["nodes"][0]
    assert current["status"] == "supported" and current["version"] == g["nodes"][0]["version"]
    assert current["unresolved_inputs"] == current["unresolved_guards"] == []
    assert current["alternatives"][0] == g["nodes"][0]["alternatives"][0]
    assert select(revised)["selected_doc_ids"] == ["c"]
    assert select(revised)["complete_required"]


def test_new_parent_route_can_upgrade_hidden_partial_node_only_with_visible_supported_parent():
    parent = node("p", [alt("parent", ["sc"])])
    partial = hidden_partial_graph()["nodes"][0]
    g = graph([parent, partial])
    payload = review(node_updates=[update(alternatives=[proof(parents=["p"])], retained=["hidden"])])
    revised = apply(g, payload, visible_alternative_ids=["parent"], visible_span_ids=["sc"])
    assert revised["nodes"][1]["status"] == "supported"
    assert select(revised)["selected_doc_ids"] == ["c"]
    with pytest.raises(SupportError, match="parent_unknown_unavailable_nonpreceding_or_invisible"):
        apply(g, payload, visible_alternative_ids=[], visible_span_ids=["sc"])


@pytest.mark.parametrize("proposal", [[], [proof(["sc"], semantic_status="partial")]])
def test_hidden_partial_node_cannot_clear_old_gaps_without_a_new_complete_proof(proposal):
    g = hidden_partial_graph()
    before = deepcopy(g)
    with pytest.raises(SupportError, match="invisible_alternative|requires_new_complete_visible_proof"):
        apply(g, review(node_updates=[update(alternatives=proposal, retained=["hidden"])]),
              visible_alternative_ids=[])
    assert g == before


def test_new_route_materializes_old_incompleteness_without_activating_hidden_conditioned_proof():
    g = hidden_partial_graph(semantic_status="supported")
    assert not g["nodes"][0]["alternatives"][0]["eligible"]
    payload = review(new_spans=[new_span()], node_updates=[update(
        alternatives=[proof(["direct"])], retained=["hidden"])])
    before = deepcopy(g)
    revised = apply(g, payload, visible_alternative_ids=[])
    assert g == before
    current = revised["nodes"][0]
    hidden, direct = current["alternatives"]
    assert current["status"] == "supported"
    assert hidden["semantic_status"] == "partial" and not hidden["eligible"]
    assert direct["eligible"]
    for field in ("id", "answer", "applicable_scope", "source_span_ids", "guard_span_ids",
                  "used_parent_ids", "used_parent_versions", "invalidated_by", "disputed_by"):
        assert hidden[field] == g["nodes"][0]["alternatives"][0][field]
    assert select(revised)["selected_doc_ids"] == ["c"]
    materialized = [action for action in revised["proof_review"]["actions"]
                    if action["action"] == "materialize_prior_node_incompleteness"]
    assert len(materialized) == 1
    assert materialized[0]["alternative_id"] == "hidden"
    assert materialized[0]["unresolved_guards"] == g["nodes"][0]["unresolved_guards"]


def test_new_partial_route_cannot_downgrade_a_hidden_valid_proof():
    g = graph()
    with pytest.raises(SupportError, match="cannot_mutate_node_with_invisible_alternative"):
        apply(g, review(node_updates=[update(alternatives=[proof(["sc"], semantic_status="partial")],
                                            retained=["old"], status="partial")]), visible_alternative_ids=[])


def test_unresolved_node_conflict_prevents_hidden_partial_upgrade_by_renaming_a_route():
    g = invalidate_support(hidden_partial_graph(), ["hidden"], ["sx"], disputed=True)
    with pytest.raises(SupportError, match="requires_new_complete_visible_proof"):
        apply(g, review(node_updates=[update(alternatives=[proof(["sc"])], retained=["hidden"])]),
              visible_alternative_ids=[])


@pytest.mark.parametrize("scope", ["", " \t\n"])
def test_new_supported_route_requires_an_explicit_nonempty_scope(scope):
    with pytest.raises(SupportError, match="alternative applicable_scope: expected nonempty string"):
        apply(graph(), review(node_updates=[update(
            alternatives=[proof(["sc"], applicable_scope=scope)], applicable_scope=scope)]))


def test_partial_or_unknown_node_may_keep_scope_unknown_without_fabricating_complete_support():
    partial = apply(graph(), review(node_updates=[update(status="partial", applicable_scope="",
        alternatives=[proof(["sc"], applicable_scope="", semantic_status="partial")])]))
    assert partial["nodes"][0]["status"] == "partial"
    assert partial["nodes"][0]["applicable_scope"] == ""
    unknown = apply(graph(), review(node_updates=[update(status="unknown", answer=None, applicable_scope="")]))
    assert unknown["nodes"][0]["status"] == "unknown"
    assert unknown["nodes"][0]["applicable_scope"] == ""
    assert not select(partial)["complete_required"] and not select(unknown)["complete_required"]
