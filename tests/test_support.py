"""Offline mechanism tests: no model, embedding service, or evaluation labels."""
from copy import deepcopy
from itertools import combinations

import pytest

from dagbt.support import (SupportError, compile_graph, invalidate_support,
                          make_span, resolve_conflict, select_support, update_node_version,
                          with_navigation_closure)


DOCS = {
    "a": {"title": "A", "text": "Old classes require copying."},
    "b": {"title": "B", "text": "New classes permit free topics."},
    "c": {"title": "C", "text": "I returned because that arrangement fits."},
    "d": {"title": "D", "text": "A direct summary supports the same answer."},
    "nav": {"title": "N", "text": "A friend mentioned the new class."},
    "conflict": {"title": "X", "text": "The free-topic class was cancelled."},
}


def spans():
    return [make_span("s_" + doc_id, doc_id, document["text"], DOCS)
            for doc_id, document in DOCS.items()]


def alternative(aid, source_ids=(), parents=(), versions=None, **kwargs):
    return {"id": aid, "source_span_ids": ["s_" + source for source in source_ids],
            "guard_span_ids": [], "used_parent_ids": list(parents),
            "used_parent_versions": dict(versions or {p: 0 for p in parents}),
            "semantic_status": "supported", **kwargs}


def node(nid, alternatives=(), parents=(), answer="same answer", status="supported", **kwargs):
    return {"id": nid, "answer": answer, "version": 0, "status": status,
            "planned_parent_ids": list(parents), "alternatives": list(alternatives), **kwargs}


def requirement(nid, necessary=True, terminal_mode="all", rid=None):
    return {"id": rid or "r_" + str(nid), "necessary": necessary,
            "terminal_node_ids": nid if isinstance(nid, list) else [nid],
            "terminal_mode": terminal_mode}


def graph(nodes, requirements=None):
    return compile_graph(nodes, spans(), requirements or [requirement(nodes[-1]["id"])], DOCS)


def capacity(n, costs=None):
    return lambda ids: {"feasible": sum((costs or {}).get(d, 1) for d in ids) <= n,
                        "token_count": 10 + sum((costs or {}).get(d, 1) for d in ids)}


def chain():
    return graph([
        node("old", [alternative("old.a", ["a"])]),
        node("new", [alternative("new.a", ["b"])]),
        node("answer", [alternative("answer.a", ["c"], ["old", "new"])], ["old", "new"]),
    ])


def test_and_closure_cannot_cover_after_deleting_one_necessary_document():
    result = select_support(chain(), capacity(2))
    assert not result["complete_required"]
    assert result["covered_requirement_ids"] == []
    assert result["status"] == "partial"
    full = select_support(chain(), capacity(3))
    assert full["selected_doc_ids"] == ["a", "b", "c"]
    assert full["node_closures"]["answer"] == ["a", "b", "c"]


def test_or_alternative_can_replace_conjunction_without_union_of_all_proofs():
    source = chain()
    source["nodes"][-1]["alternatives"].append(alternative("answer.direct", ["d"]))
    compiled = compile_graph(source["nodes"], source["spans"], source["requirements"], DOCS)
    result = select_support(compiled, capacity(3))
    assert result["selected_doc_ids"] == ["d"]
    assert result["chosen_alternatives"] == {"answer": "answer.direct"}
    assert result["complete_required"]


def test_navigation_and_unused_planned_parent_do_not_enter_closure():
    compiled = graph([
        node("navigation", [alternative("nav.a", ["nav"])]),
        node("answer", [alternative("answer.a", ["d"])], parents=["navigation"]),
    ])
    result = select_support(compiled, capacity(1))
    assert result["selected_doc_ids"] == ["d"]
    assert result["node_closures"]["answer"] == ["d"]


def test_unknown_parent_is_not_a_supported_binding():
    compiled = graph([
        node("parent", answer=None, status="unknown"),
        node("child", [alternative("child.a", ["b"], ["parent"])], parents=["parent"]),
    ])
    assert compiled["nodes"][1]["status"] == "invalidated"
    assert "parent_unavailable:parent" in compiled["nodes"][1]["alternatives"][0]["ineligible_reasons"]
    assert not select_support(compiled, capacity(10))["complete_required"]


def test_declared_partial_and_ambiguous_never_promoted_by_structural_support():
    for status in ("partial", "ambiguous"):
        compiled = graph([node("n", [alternative("n.a", ["a"])], status=status,
                               unresolved_guards=["unknown applicability"])])
        assert compiled["nodes"][0]["status"] == status
        assert not select_support(compiled, capacity(10))["complete_required"]


def test_shared_source_charged_once_and_callback_includes_template_overhead():
    compiled = graph([
        node("a", [alternative("a.a", ["a"])]),
        node("b", [alternative("b.a", ["a"], ["a"])], parents=["a"]),
    ])
    calls = []
    def feasibility(ids):
        calls.append(ids)
        return {"feasible": 12 + len(ids) <= 13, "token_count": 12 + len(ids)}
    result = select_support(compiled, feasibility)
    assert result["selected_doc_ids"] == ["a"]
    assert result["token_count"] == 13
    assert len(calls) == len({tuple(ids) for ids in calls})


def test_token_guard_rejects_context_even_if_document_count_fits():
    result = select_support(chain(), capacity(5, {"a": 4, "b": 4, "c": 4}))
    assert not result["complete_required"]
    assert result["selected_doc_ids"] == []


def test_no_complete_proof_means_no_reward_for_intermediate_nodes():
    result = select_support(chain(), capacity(2))
    assert result["selected_doc_ids"] == []
    assert result["chosen_alternatives"] == {}


def test_terminal_all_and_any_have_distinct_semantics():
    nodes = [node("a", [alternative("a.a", ["a"])]), node("b", [alternative("b.a", ["b"])])]
    all_result = select_support(graph(nodes, [requirement(["a", "b"])]), capacity(1))
    any_result = select_support(graph(nodes, [requirement(["a", "b"], terminal_mode="any")]), capacity(1))
    assert not all_result["complete_required"]
    assert any_result["complete_required"]


def test_necessary_coverage_precedes_optional_then_true_token_cost():
    nodes = [node("a", [alternative("a.a", ["a"])]), node("b", [alternative("b.a", ["b"])]),
             node("c", [alternative("c.a", ["c"])])]
    requirements = [requirement("a"), requirement("b", False), requirement("c", False)]
    result = select_support(graph(nodes, requirements), capacity(2, {"a": 2}))
    assert result["selected_doc_ids"] == ["a"]
    assert result["necessary_covered"] == 1
    assert result["optional_covered"] == 0


@pytest.mark.parametrize("kind", ["future", "cycle", "unknown"])
def test_unknown_forward_and_cyclic_actual_edges_are_errors(kind):
    parent = "missing" if kind == "unknown" else "later"
    nodes = [node("first", [alternative("first.a", ["a"], [parent])]),
             node("later", [alternative("later.a", ["b"], ["first"] if kind == "cycle" else [])])]
    with pytest.raises(SupportError, match="unknown_forward_or_cycle"):
        graph(nodes)


def test_version_must_pin_every_actually_used_parent():
    nodes = [node("p", [alternative("p.a", ["a"])]),
             node("c", [alternative("c.a", ["b"], ["p"])], parents=["p"])]
    nodes[1]["alternatives"][0]["used_parent_versions"] = {}
    with pytest.raises(SupportError, match="parent_version_bindings_incomplete"):
        graph(nodes)


def test_exact_quote_hash_offsets_and_visibility():
    grounded = make_span("s", "a", DOCS["a"]["text"], DOCS)
    assert grounded["start"] == 2
    assert grounded["event_time"] is None
    for field, value, error in [("start", 0, "quote_not_exact"),
                                ("raw_text_hash", "wrong", "source_hash_changed"),
                                ("doc_id", "absent", "not_visible")]:
        bad = {**grounded, field: value}
        with pytest.raises(SupportError, match=error):
            compile_graph([node("n", [alternative("n.a")], status="unknown", answer=None)],
                          [bad], [requirement("n")], DOCS)


def test_ambiguous_repeated_quote_requires_offset():
    docs = {"d": "same then same"}
    with pytest.raises(SupportError, match="ambiguous_quote_offset"):
        make_span("s", "d", "same", docs)
    assert make_span("s", "d", "same", docs, start=10)["end"] == 14


@pytest.mark.parametrize("field,value,error", [
    ("answer", "different answer", "conclusion_mismatch"),
    ("applicable_scope", "different time", "scope_mismatch"),
])
def test_or_alternatives_must_support_same_conclusion_and_scope(field, value, error):
    alts = [alternative("x.a", ["a"]), alternative("x.b", ["b"], **{field: value})]
    with pytest.raises(SupportError, match=error):
        graph([node("x", alts)])


def test_invalidate_one_alternative_preserves_another_and_same_answer_descendants():
    compiled = graph([
        node("p", [alternative("p.a", ["a"]), alternative("p.b", ["b"])]),
        node("c", [alternative("c.a", ["c"], ["p"])], parents=["p"]),
    ])
    updated = invalidate_support(compiled, ["p.a"], ["s_conflict"])
    assert compiled["nodes"][0]["alternatives"][0]["eligible"]  # copy-on-write
    assert updated["nodes"][0]["version"] == 0
    assert updated["nodes"][1]["status"] == "supported"
    assert updated["last_update"]["requires_semantic_recheck"] == []
    selected = select_support(updated, capacity(2))
    assert selected["selected_doc_ids"] == ["b", "c"]
    assert selected["complete_required"]


def test_invalidate_all_alternatives_locally_blocks_actual_descendants_only():
    compiled = graph([
        node("p", [alternative("p.a", ["a"])]),
        node("c", [alternative("c.a", ["c"], ["p"])], parents=["p"]),
        node("independent", [alternative("i.a", ["d"])], parents=["p"]),
    ], [requirement("c"), requirement("independent")])
    updated = invalidate_support(compiled, ["p.a"], ["s_conflict"])
    assert [n["status"] for n in updated["nodes"]] == ["invalidated", "invalidated", "supported"]
    assert updated["last_update"]["requires_semantic_recheck"] == ["c"]
    assert select_support(updated, capacity(5))["covered_requirement_ids"] == ["r_independent"]


def test_changed_answer_version_does_not_reuse_old_support_or_child_binding():
    compiled = chain()
    updated = update_node_version(compiled, "old", "changed conclusion", 1)
    assert updated["nodes"][0]["answer"] == "changed conclusion"
    assert updated["nodes"][0]["status"] != "supported"
    assert "parent_version_changed:old" in updated["nodes"][-1]["alternatives"][0]["ineligible_reasons"]
    assert not select_support(updated, capacity(10))["complete_required"]


def test_unresolved_conflict_partial_sources_are_kept_together_or_not_at_all():
    compiled = graph([node("answer", [alternative("a.a", ["a"])])])
    disputed = invalidate_support(compiled, ["a.a"], ["s_conflict"], disputed=True)
    assert disputed["nodes"][0]["status"] == "ambiguous"
    fits = select_support(disputed, capacity(2), fill_partial=True)
    assert fits["selected_doc_ids"] == ["a", "conflict"]
    assert fits["covered_requirement_ids"] == []
    assert fits["status"] == "partial"
    too_small = select_support(disputed, capacity(1), fill_partial=True)
    assert too_small["selected_doc_ids"] == []


def test_partial_supplementation_cannot_sneak_one_side_of_conflict():
    disputed = invalidate_support(graph([node("n", [alternative("n.a", ["a"])])]),
                                 ["n.a"], ["s_conflict"], disputed=True)
    result = select_support(disputed, capacity(1),
                            partial_groups=[{"id": "sneak", "doc_ids": ["a"]}], fill_partial=True)
    assert result["selected_doc_ids"] == []


def test_disputed_raw_pair_survives_replaced_historical_support_alternative():
    disputed = invalidate_support(graph([node("n", [alternative("n.a", ["a"])])]),
                                 ["n.a"], ["s_conflict"], disputed=True)
    conflict = disputed["conflicts"][0]
    assert conflict["protected_doc_ids"] == ["a", "conflict"]
    assert conflict["target_node_ids"] == ["n"]
    replacement = graph([node("n", [], answer=None, status="partial")])
    replacement["conflicts"] = deepcopy(disputed["conflicts"])
    result = select_support(replacement, capacity(2), fill_partial=True)
    assert result["selected_doc_ids"] == ["a", "conflict"]
    assert not result["complete_required"]


def test_conflict_resolution_requires_proof_and_acknowledges_counterquote():
    disputed = invalidate_support(graph([node("n", [alternative("n.a", ["a"])])]),
                                 ["n.a"], ["s_conflict"], disputed=True)
    with pytest.raises(SupportError, match="address_all_opposing_quotes"):
        resolve_conflict(disputed, "conflict_1", ["s_b"], [], "Different entity")
    with pytest.raises(SupportError, match="visible_proof"):
        resolve_conflict(disputed, "conflict_1", [], ["s_conflict"], "Different entity")
    resolved = resolve_conflict(disputed, "conflict_1", ["s_b"], ["s_conflict"],
                                "The counterquote names a distinct entity", "entity_distinction")
    assert resolved["conflicts"][0]["resolution_status"] == "resolved"
    assert select_support(resolved, capacity(2))["selected_doc_ids"] == ["a", "b"]
    assert not select_support(resolved, capacity(1))["complete_required"]


def test_recompile_preserves_semantic_status_for_explicit_conflict_resolution():
    disputed = invalidate_support(graph([node("n", [alternative("n.a", ["a"])])]),
                                 ["n.a"], ["s_conflict"], disputed=True)
    rebuilt = compile_graph(disputed["nodes"], disputed["spans"], disputed["requirements"], DOCS)
    rebuilt["conflicts"] = deepcopy(disputed["conflicts"])
    assert rebuilt["nodes"][0]["status"] == "ambiguous"
    assert rebuilt["nodes"][0]["declared_status"] == "supported"
    resolved = resolve_conflict(rebuilt, "conflict_1", ["s_b"], ["s_conflict"],
                                "Different entity is explicit", "entity_distinction")
    assert resolved["nodes"][0]["status"] == "supported"
    assert select_support(resolved, capacity(2))["complete_required"]


def test_recompile_does_not_override_model_declared_ambiguity():
    compiled = graph([node("n", [alternative("n.a", ["a"])], status="ambiguous")])
    rebuilt = compile_graph(compiled["nodes"], compiled["spans"], compiled["requirements"], DOCS)
    assert rebuilt["nodes"][0]["declared_status"] == "ambiguous"
    assert rebuilt["nodes"][0]["status"] == "ambiguous"


def test_partial_evidence_never_increases_coverage_or_displaces_complete_support():
    compiled = graph([node("n", [alternative("n.a", ["a"])])])
    result = select_support(compiled, capacity(1),
                            partial_groups=[{"id": "extra", "doc_ids": ["b"]}], fill_partial=True)
    assert result["selected_doc_ids"] == ["a"]
    assert result["partial_group_ids"] == []


def test_enumeration_bound_is_3_to_8_and_truncation_does_not_claim_optimum():
    nodes = [node("n" + str(i), [alternative("a" + str(i), ["a"]),
                                alternative("b" + str(i), ["b"])]) for i in range(8)]
    compiled = graph(nodes, [requirement(n["id"]) for n in nodes])
    exact = select_support(compiled, capacity(1))
    assert exact["diagnostics"]["assignments_total"] == 3 ** 8
    assert exact["diagnostics"]["assignments_examined"] == 6561
    assert exact["diagnostics"]["finite_graph_optimum"]
    truncated = select_support(compiled, capacity(1), max_states=3)
    assert not truncated["diagnostics"]["exhaustive"]
    assert not truncated["diagnostics"]["finite_graph_optimum"]
    assert truncated["status"] == "partial"


def test_requirements_cannot_change_to_fit_candidates():
    compiled = chain()
    compiled["requirements"][0]["terminal_node_ids"] = ["old"]
    with pytest.raises(SupportError, match="frozen_requirements_changed"):
        select_support(compiled, capacity(2))


def test_context_infeasible_is_not_semantic_unknown_or_success():
    result = select_support(chain(), lambda ids: {"feasible": False, "token_count": 99999})
    assert result["status"] == "context_infeasible"
    assert not result["feasible"]
    assert not result["diagnostics"]["finite_graph_optimum"]


def test_compare_to_independent_document_subset_oracle_on_small_graph():
    # Oracle enumerates document subsets and checks the manually specified proof
    # alternatives. It does not call the implementation's closure traversal.
    nodes = [node("a", [alternative("a.a", ["a"])]),
             node("n", [alternative("n.chain", ["b"], ["a"]),
                         alternative("n.direct", ["d"])], parents=["a"]),
             node("o", [alternative("o.a", ["c"])])]
    compiled = graph(nodes, [requirement("n"), requirement("o", False)])
    costs = {"a": 2, "b": 1, "c": 1, "d": 4}
    for limit in range(1, 8):
        candidates = []
        for count in range(5):
            for chosen in combinations(costs, count):
                chosen = set(chosen)
                cost = sum(costs[x] for x in chosen)
                if cost <= limit:
                    candidates.append((-int({"a", "b"} <= chosen or "d" in chosen),
                                       -int("c" in chosen), 10 + cost, tuple(sorted(chosen))))
        expected = min(candidates)
        result = select_support(compiled, capacity(limit, costs))
        assert (-result["necessary_covered"], -result["optional_covered"], result["token_count"],
                tuple(sorted(result["selected_doc_ids"]))) == expected


def test_navigation_forced_ablation_charges_path_without_changing_support_graph():
    compiled = graph([node("answer", [alternative("answer.a", ["c"])])])
    forced = with_navigation_closure(compiled, {"nav": [], "a": ["nav"], "c": ["a"]}, DOCS)
    assert forced["nodes"] == compiled["nodes"]
    assert forced["requirements_hash"] == compiled["requirements_hash"]
    normal = select_support(compiled, capacity(3))
    result = select_support(forced, capacity(3))
    assert normal["selected_doc_ids"] == ["c"]
    assert result["selected_doc_ids"] == ["a", "c", "nav"]
    assert result["support_selected_doc_ids"] == ["c"]
    assert result["forced_navigation_doc_ids"] == ["a", "nav"]
    assert result["node_closures"]["answer"] == ["c"]
    assert result["token_count"] == normal["token_count"] + 2
    assert not select_support(forced, capacity(2))["complete_required"]
    # Partial raw supplementation also cannot evade navigation closure cost.
    small = select_support(forced, capacity(2), partial_groups=[{"id": "p", "doc_ids": ["c"]}], fill_partial=True)
    assert small["selected_doc_ids"] == []


def test_navigation_cost_can_change_choice_between_semantic_or_alternatives():
    compiled = graph([node("answer", [alternative("answer.a", ["c"]), alternative("answer.b", ["d"])])])
    forced = with_navigation_closure(compiled, {"nav": [], "a": ["nav"], "c": ["a"], "d": []}, DOCS)
    result = select_support(forced, capacity(2))
    assert result["complete_required"] and result["selected_doc_ids"] == ["d"]
    assert result["forced_navigation_doc_ids"] == []


def test_navigation_shared_ancestor_is_counted_once():
    compiled = graph([node("left", [alternative("left.a", ["b"])]),
                      node("right", [alternative("right.a", ["c"])])], [requirement(["left", "right"])])
    forced = with_navigation_closure(compiled, {"nav": [], "b": ["nav"], "c": ["nav"]}, DOCS)
    result = select_support(forced, capacity(3))
    assert result["complete_required"] and result["token_count"] == 13
    assert result["selected_doc_ids"] == ["b", "c", "nav"]


def test_navigation_ablation_rejects_invisible_missing_and_cyclic_sources():
    compiled = graph([node("n", [alternative("n.a", ["c"])])])
    for paths, error in [({"c": ["absent"]}, "not_visible"),
                         ({"c": ["a"]}, "first_discovery_missing"),
                         ({"c": ["a"], "a": ["c"]}, "cycle")]:
        with pytest.raises(SupportError, match=error):
            with_navigation_closure(compiled, paths, DOCS)
