"""Run explicit offline responses through Engine, proof review, closure and Reader.

The language model responses are scripted semantic judgments, not accuracy tests.
Only retrieval and transport are fixtures; proof mutation and final selection use
the production implementation, including actual Reader prompt/token accounting.
"""
from copy import deepcopy
import re

import numpy as np
import pytest

from dagbt.engine import Engine
from test_engine import Document, FakeCalls, Tokenizer, answer, setup, step, unknown


def review(**changes):
    return {"new_spans": [], "invalidations": [], "resolutions": [], "node_updates": [],
            "supplemental_doc_ids": [], "reason": "Scripted support proof review", **changes}


def fact(data, doc_id, node_id):
    return next(item["id"] for item in data["evidence"]
                if item["doc_id"] == doc_id and node_id in item.get("node_ids", [item.get("node_id")]))


def proof(source_ids, *, parents=(), guards=(), scope="fixture"):
    return {"source_span_ids": list(source_ids), "guard_span_ids": list(guards),
            "used_parent_ids": list(parents), "applicable_scope": scope, "semantic_status": "supported"}


def update(node_id, value, alternatives, *, scope="fixture", retained=()):
    return {"node_id": node_id, "answer": value, "status": "supported", "applicable_scope": scope,
            "retained_alternative_ids": list(retained), "alternatives": alternatives,
            "unresolved_inputs": [], "unresolved_guards": [], "reason": "Fresh exact-source proof"}


def raw_span(data, local_id, doc_id, node_id):
    raw = next(item for item in data["raw_memory_candidates"] if item["doc_id"] == doc_id)
    return {"id": local_id, "doc_id": doc_id, "start": 0, "quote": raw["passage"],
            "node_id": node_id, "kind": "explicit", "stance": "support",
            "claim": "Direct answer to the requested relation", "entity_scope": "fixture"}


def no_mapping(data):
    return {"units": [{"unit_id": unit["unit_id"], "assessments": [],
                       "irrelevance_reason": "Scripted mapper missed the relevant direct source"}
                      for unit in data["units"]]}


def run_engine(fixture, steps, resolver, selector, *, docs=None, mapper=None, tokenizer=None):
    bridge, resources, original_config = fixture
    resources = list(resources)
    if docs is not None:
        resources[:3] = [docs, list(docs), np.eye(len(docs), dtype=np.float32)]
    if tokenizer is not None:
        resources[-1] = tokenizer
    docs = resources[0]
    config = deepcopy(original_config)
    config["fusion"].update(selection_review=True, condition_audit=False)
    bridge.routes = {"__baseline__": list(docs), **{s["output_slot"]: list(docs) for s in steps}}
    calls = FakeCalls(steps, resolver, mapper=mapper, selector=selector)
    engine = Engine({"id": "q", "question": "Which requested relation is supported?"},
                    tuple(resources), calls, config, "fusion")
    result = engine.run()
    assert result["answer"]["status"] == "ok"
    assert calls.counts["select"] == calls.counts["reader"] == 1
    assert any(event["event"] == "final_selection" for event in result["diagnostics"]["events"])
    select_request = next(request for request in calls.requests if request["operation"] == "select")
    assert select_request["payload"]["messages"][0]["content"].startswith("Review support proofs")
    reader = next(request for request in calls.requests if request["operation"] == "reader")
    reader_text = reader["payload"]["messages"][1]["content"]
    assert re.findall(r"source_doc_id=([^\s]+)", reader_text) == result["budgets"]["20"]["selected_doc_ids"]
    return result, engine, calls, reader_text


def test_supplement_b_cannot_drop_its_required_parent_a(setup):
    steps = [step("author"), step("nation", "Birthplace of {author}?", ["author"])]
    def resolve(data):
        parent = data["node_id"] == "author"
        return answer(data, "MODEL_ONLY_PARENT" if parent else "MODEL_ONLY_FINAL",
                      ["a"] if parent else ["b"], [] if parent else ["author"])

    result, _, _, reader_text = run_engine(setup, steps, resolve,
                                           lambda data: review(supplemental_doc_ids=["b"]))

    for budget in result["budgets"].values():
        assert budget["complete_required"]
        assert budget["selected_doc_ids"] == ["a", "b"]
        assert budget["node_closures"]["nation"] == ["a", "b"]
    assert "MODEL_ONLY_PARENT" not in reader_text and "MODEL_ONLY_FINAL" not in reader_text


def test_empty_mapper_review_adds_exact_c_as_direct_route_without_parent(setup):
    docs = {"a": Document("a", "Earlier indirect evidence."),
            "b": Document("b", "Another incomplete clue."),
            "c": Document("c", "The requested writer was born in France.")}
    docs["c"].metadata = {"source_segments": [{"start": 0, "end": len(docs["c"].passage),
        "role": "user", "source_message_indices": [7]}], "observation_order": 7}
    steps = [step("author"), step("nation", "Birthplace of {author}?", ["author"])]
    captured = []
    def select(data):
        captured.append(deepcopy(data))
        assert data["evidence"] == []
        assert {node["id"] for node in data["nodes"]} == {"author", "nation"}
        return review(new_spans=[raw_span(data, "direct_c", "c", "nation")],
                      node_updates=[update("nation", "France", [proof(["direct_c"])])])

    result, engine, _, reader_text = run_engine(setup, steps, lambda data: unknown(), select,
                                               docs=docs, mapper=no_mapping)

    assert result["budgets"]["20"]["selected_doc_ids"] == ["c"]
    assert result["budgets"]["20"]["complete_required"]
    graph = result["diagnostics"]["support_graph"]
    nodes = {node["id"]: node for node in graph["nodes"]}
    assert nodes["author"]["status"] != "supported"
    assert nodes["nation"]["status"] == "supported"
    assert nodes["nation"]["alternatives"][0]["used_parent_ids"] == []
    assert nodes["nation"]["planned_parent_ids"] == ["author"]
    assert graph["requirements"] == engine.requirements
    assert len(result["diagnostics"]["spans"]) == 1
    span = result["diagnostics"]["spans"][0]
    assert span["doc_id"] == "c" and span["exact_quote"] == docs["c"].passage
    assert docs["c"].passage[span["start"]:span["end"]] == span["exact_quote"]
    raw = next(item for item in captured[0]["raw_memory_candidates"] if item["doc_id"] == "c")
    assert raw["passage"] == docs["c"].passage
    assert raw["metadata"]["source_segments"][0]["source_message_indices"] == [7]
    assert docs["c"].passage in reader_text and docs["a"].passage not in reader_text


def test_document_shared_by_or_routes_does_not_force_all_other_premises(setup):
    docs = {"a": Document("a", "Shared exact fact."),
            "b": Document("b", "Longer discarded route. " * 7),
            "c": Document("c", "Short complete guard.")}
    result, _, _, reader_text = run_engine(setup, [step("answer")],
        lambda data: answer(data, "fixture_answer", [], alternatives=[(["a", "b"], []), (["a", "c"], [])]),
        lambda data: review(), docs=docs)
    choice = result["budgets"]["20"]
    assert choice["complete_required"] and choice["selected_doc_ids"] == ["a", "c"]
    assert len(choice["chosen_alternatives"]) == 1
    assert docs["b"].passage not in reader_text


def test_review_revokes_one_support_and_removes_its_old_document(setup):
    docs = {"a": Document("a", "Old fact."), "c": Document("c", "Longer independent valid support."),
            "x": Document("x", "The old fact refers to the wrong entity.")}
    old_ids = []
    def select(data):
        alternatives = data["nodes"][0]["alternatives"]
        old_ids.append(alternatives[0]["id"])
        contrary = {**raw_span(data, "disprove_a", "x", "answer"), "stance": "contradiction",
                    "claim": "The old proof binds the wrong entity"}
        return review(new_spans=[contrary], invalidations=[{"alternative_ids": old_ids,
            "source_span_ids": ["disprove_a"], "reason": "Old entity binding is wrong", "disputed": False}])
    result, _, _, reader_text = run_engine(setup, [step("answer")],
        lambda data: answer(data, "fixture_answer", [], alternatives=[(["a"], []), (["c"], [])]),
        select, docs=docs)
    assert result["budgets"]["20"]["complete_required"]
    assert result["budgets"]["20"]["selected_doc_ids"] == ["c"]
    alternatives = result["ranking"]["nodes"][0]["alternatives"]
    revoked = next(alt for alt in alternatives if alt["id"] == old_ids[0])
    assert revoked["invalidated_by"] and not revoked["eligible"]
    assert docs["a"].passage not in reader_text


class CostTokenizer(Tokenizer):
    """A deterministic tokenizer makes a short fixture passage expensive."""
    def encode(self, text, **kwargs):
        count = len(text.split()) + text.count("COST_A") * 9_000 + text.count("COST_B") * 9_000
        return list(range(count))


@pytest.mark.parametrize("has_alternative", [True, False])
def test_reader_budget_reselects_whole_proof_or_marks_raw_supplement_incomplete(setup, has_alternative):
    docs = {"a": Document("a", "COST_A necessary premise."), "b": Document("b", "COST_B final relation."),
            "c": Document("c", "Independent complete source.")}
    alternatives = [(["a", "b"], [])] + ([(["c"], [])] if has_alternative else [])
    result, engine, _, reader_text = run_engine(setup, [step("answer")],
        lambda data: answer(data, "fixture_answer", [], alternatives=alternatives),
        lambda data: review(supplemental_doc_ids=[] if has_alternative else ["b"]),
        docs=docs, tokenizer=CostTokenizer())

    assert not engine.feasible(["a", "b"])["feasible"]
    assert engine.feasible(["b"])["feasible"] and engine.feasible(["c"])["feasible"]
    for choice in result["budgets"].values():
        assert choice["complete_required"] is has_alternative
        assert choice["selected_doc_ids"] == (["c"] if has_alternative else ["b"])
        assert choice["feasible"]
    if has_alternative:
        assert "COST_A" not in reader_text and "COST_B" not in reader_text
    else:
        assert "COST_B" in reader_text and "COST_A" not in reader_text
        assert result["budgets"]["20"]["chosen_alternatives"] == {}


def test_five_document_budget_reselects_complete_or_route_instead_of_slicing_six_premises(setup):
    short_ids = ["p" + str(index) for index in range(6)]
    docs = {doc_id: Document(doc_id, "Necessary premise.") for doc_id in short_ids}
    docs["direct"] = Document("direct", "Independent full support. " * 20)
    result, _, _, reader_text = run_engine(setup, [step("answer")],
        lambda data: answer(data, "fixture_answer", [], alternatives=[(short_ids, []), (["direct"], [])]),
        lambda data: review(), docs=docs)
    assert result["budgets"]["5"]["complete_required"]
    assert result["budgets"]["5"]["selected_doc_ids"] == ["direct"]
    for k in ("10", "20"):
        assert result["budgets"][k]["complete_required"]
        assert result["budgets"][k]["selected_doc_ids"] == short_ids
    assert docs["direct"].passage not in reader_text


def test_converging_branches_preserve_both_parents_and_exact_guard(setup):
    docs = {"a": Document("a", "First branch fact."), "b": Document("b", "Second branch fact."),
            "c": Document("c", "Converging conclusion."), "g": Document("g", "Required time condition."),
            "unused": Document("unused", "Unrelated observation.")}
    steps = [step("left"), step("right"), step("final", "Join {left} with {right}?", ["left", "right"])]
    def resolve(data):
        nid = data["node_id"]
        if nid != "final":
            return answer(data, nid + "_value", ["a" if nid == "left" else "b"])
        value = answer(data, "final_value", ["c"], ["left", "right"])
        value["alternatives"][0]["guard_span_ids"] = [fact(data, "g", "final")]
        return value
    result, _, _, reader_text = run_engine(setup, steps, resolve, lambda data: review(), docs=docs)
    choice = result["budgets"]["20"]
    assert choice["complete_required"] and choice["selected_doc_ids"] == ["a", "b", "c", "g"]
    assert choice["node_closures"]["final"] == ["a", "b", "c", "g"]
    assert set(choice["chosen_alternatives"]) == {"left", "right", "final"}
    assert docs["g"].passage in reader_text and docs["unused"].passage not in reader_text


def test_parent_proof_replacement_preserves_same_claim_version_and_recloses_child(setup):
    steps = [step("parent"), step("child", "Use {parent}?", ["parent"])]
    def resolve(data):
        parent = data["node_id"] == "parent"
        return answer(data, "same_parent" if parent else "same_child", ["a"] if parent else ["b"],
                      [] if parent else ["parent"])
    before = []
    def select(data):
        before.append(deepcopy(data["nodes"]))
        return review(new_spans=[raw_span(data, "replacement_c", "c", "parent")],
                      node_updates=[update("parent", "same_parent", [proof(["replacement_c"])])])
    result, _, _, reader_text = run_engine(setup, steps, resolve, select)
    nodes = {node["id"]: node for node in result["ranking"]["nodes"]}
    assert nodes["parent"]["version"] == before[0][0]["version"]
    assert nodes["child"]["status"] == "supported"
    assert result["budgets"]["20"]["complete_required"]
    assert result["budgets"]["20"]["selected_doc_ids"] == ["b", "c"]
    assert result["budgets"]["20"]["node_closures"]["child"] == ["b", "c"]
    assert setup[1][0]["a"].passage not in reader_text


@pytest.mark.parametrize("change", ["answer", "scope"])
def test_review_parent_change_invalidates_stale_child_and_grandchild_bindings(setup, change):
    steps = [step("parent"), step("child", "Use {parent}?", ["parent"]),
             step("grandchild", "Use {child}?", ["child"])]
    def resolve(data):
        nid = data["node_id"]
        parent = {"parent": [], "child": ["parent"], "grandchild": ["child"]}[nid]
        return answer(data, "old_" + nid, [{"parent": "a", "child": "b", "grandchild": "c"}[nid]], parent)
    before = []
    def select(data):
        before.append(deepcopy(data["nodes"]))
        scope = "different_entity_scope" if change == "scope" else "fixture"
        value = "revised_parent" if change == "answer" else "old_parent"
        return review(node_updates=[update("parent", value,
            [proof([fact(data, "a", "parent")], scope=scope)], scope=scope)])
    result, _, _, _ = run_engine(setup, steps, resolve, select)

    nodes = {node["id"]: node for node in result["ranking"]["nodes"]}
    old = {node["id"]: node for node in before[0]}
    assert nodes["parent"]["status"] == "supported"
    assert nodes["parent"]["version"] == old["parent"]["version"] + 1
    assert nodes["child"]["status"] != "supported" and nodes["grandchild"]["status"] != "supported"
    child_alt = nodes["child"]["alternatives"][0]
    assert not child_alt["eligible"] and "parent_version_changed:parent" in child_alt["ineligible_reasons"]
    assert not nodes["grandchild"]["alternatives"][0]["eligible"]
    assert not result["budgets"]["20"]["complete_required"]
    assert "grandchild" not in result["budgets"]["20"]["chosen_alternatives"]
