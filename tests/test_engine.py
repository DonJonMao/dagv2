"""Execute actual plan/map/resolve/closure/reader wiring with offline transport.

Only BridgeSession is replaced to make retrieval deterministic. Semantic model
outputs are explicit fixtures; these tests make no claim about QA accuracy.
"""
from collections import Counter
import json
from types import SimpleNamespace

import numpy as np
import pytest

from dagbt.engine import Engine, ProtocolError, run_question
from dagbt.budget import BudgetExceeded, Ledger
from dagbt.transport import Transport


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(m["role"] + ": " + m["content"] for m in messages)

    def encode(self, text, **kwargs):
        return list(range(len(text.split())))


class Document:
    def __init__(self, doc_id, text):
        self.doc_id, self.title, self.text = doc_id, doc_id, text

    @property
    def passage(self):
        return self.title + "\n" + self.text


class FakeCalls:
    def __init__(self, steps, resolver, audit=None, mapper=None, selector=None):
        self.steps, self.resolver, self.auditor, self.mapper, self.selector = steps, resolver, audit, mapper, selector
        self.requests = []
        self.counts = Counter()

    def get(self, stage, url, payload):
        operation = stage[0]
        self.counts[operation] += 1
        self.requests.append({"operation": operation, "payload": payload})
        if operation == "reader":
            result = "Answer: fixture_answer"
        else:
            data = json.loads(payload["messages"][1]["content"])
            if operation == "planner":
                value = {"steps": self.steps}
            elif operation == "map":
                if self.mapper:
                    value = self.mapper(data)
                else:
                    value = {"spans": [{"doc_id": c["doc_id"], "start": c["start"], "quote": c["text"],
                               "node_ids": [n["output_slot"] for n in data["nodes"]],
                               "stance": "contradiction" if c["doc_id"] == "x" else "support",
                               "entity_scope": "fixture", "event_time": None, "time_quote": None,
                               "reason": "fixture quotation"} for c in data["chunks"]]}
            elif operation == "resolve":
                value = self.resolver(data)
            elif operation == "audit":
                value = self.auditor(data) if self.auditor else {"conflicts": [], "unresolved_guards": []}
            elif operation == "select":
                value = self.selector(data) if self.selector else {"selected_doc_ids": data["candidate_doc_ids"],
                          "reason": "fixture whole-set judgment", "covered_requirement_ids": ["answer"]}
            else:
                raise AssertionError("Unexpected fixture operation: " + operation)
            result = json.dumps(value)
        return {"response_ref": "fixture_" + str(len(self.requests)),
                "response": {"choices": [{"finish_reason": "stop", "message": {"content": result}}],
                             "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}}}


def step(nid, question="Resolve requested relation?", inputs=()):
    return {"question": question, "output_slot": nid, "answer_type": "entity", "inputs": list(inputs)}


def answer(data, value, sources, parents=(), alternatives=None, **extra):
    mapped = {s["doc_id"]: s["id"] for s in data["evidence"]}
    alternatives = alternatives or [(sources, parents)]
    return {"status": "supported", "answer": value,
            "alternatives": [{"source_span_ids": [mapped[d] for d in docs], "guard_span_ids": [],
                "used_parent_ids": list(parent_ids), "applicable_scope": "fixture", "semantic_status": "supported"}
                for docs, parent_ids in alternatives],
            "unresolved_inputs": [], "unresolved_guards": [], "refinements": [], **extra}


def unknown(**extra):
    return {"status": "unknown", "answer": None, "alternatives": [],
            "unresolved_inputs": ["missing fact"], "unresolved_guards": [], "refinements": [], **extra}


@pytest.fixture
def setup(monkeypatch):
    class Bridge:
        routes = {}
        calls = []

        def __init__(self, *args):
            self.ledger = args[-1]

        def discover(self, query, node_id, **kwargs):
            self.ledger.reserve("ann", node_id)
            self.calls.append({"query": query, "node_id": node_id, **kwargs})
            found = self.routes.get(node_id, [])
            return {"candidate_ids": found, "new_candidate_ids": found,
                    "stop_reason": "fixture", "trace": []}

    import dagbt.bridge
    monkeypatch.setattr(dagbt.bridge, "BridgeSession", Bridge)
    docs = {"a": Document("a", "The book was written by a pen name explained elsewhere."),
            "b": Document("b", "The author's birthplace is inside the requested nation."),
            "c": Document("c", "Direct short fact."),
            "x": Document("x", "An opposing source disputes that claimed relation.")}
    ids = list(docs)
    resources = (docs, ids, np.eye(len(ids), dtype=np.float32), SimpleNamespace(), Tokenizer())
    config = {"_test_transport": True, "llm_base_url": "https://invalid.example/v1",
              "llm_model": "fixture", "embedding_model": "fixture", "fusion": {
                  "max_feedback_rounds": 0, "max_quote_chars": 400, "refinement": False}}
    return Bridge, resources, config


def test_real_engine_and_dependency_binds_parent_and_raw_reader_has_no_chain(setup):
    bridge, resources, config = setup
    bridge.routes = {"author": ["a"], "nation": ["b"]}
    steps = [step("author", "Who wrote the book?"), step("nation", "Where is {author}'s birthplace?", ["author"])]
    def resolver(data):
        return answer(data, "MODEL_ONLY_PARENT" if data["node_id"] == "author" else "MODEL_ONLY_FINAL",
                      ["a"] if data["node_id"] == "author" else ["b"],
                      [] if data["node_id"] == "author" else ["author"])
    calls = FakeCalls(steps, resolver)
    result = run_question({"id": "q", "question": "Where was the book author born?"}, resources, calls, config)
    assert result["budgets"]["20"]["selected_doc_ids"] == ["a", "b"]
    assert result["budgets"]["20"]["complete_required"]
    assert "MODEL_ONLY_PARENT" in bridge.calls[1]["query"]
    assert bridge.calls[1]["premise_doc_ids"] == ["a"]
    reader_request = next(x for x in calls.requests if x["operation"] == "reader")
    text = json.dumps(reader_request["payload"])
    assert "MODEL_ONLY_PARENT" not in text and "MODEL_ONLY_FINAL" not in text
    assert "Model-derived intermediate claims" not in text
    assert all(x["event_time"] is None for x in result["diagnostics"]["spans"])
    assert result["diagnostics"]["ledger"]["used"]["reader"] == 1


def test_real_engine_or_chooses_short_direct_support(setup):
    bridge, resources, config = setup
    bridge.routes = {"answer": ["a", "b", "c"]}
    calls = FakeCalls([step("answer")], lambda data: answer(data, "fixture_answer", [],
                       alternatives=[(["a", "b"], []), (["c"], [])]))
    result = run_question({"id": "q", "question": "Requested relation?"}, resources, calls, config)
    assert result["budgets"]["20"]["selected_doc_ids"] == ["c"]
    assert result["budgets"]["20"]["complete_required"]


def test_missing_parent_is_never_grounded_or_sent_to_child_search(setup):
    bridge, resources, config = setup
    bridge.routes = {"parent": ["a"], "child": ["b"]}
    calls = FakeCalls([step("parent"), step("child", "Relation of {parent}?", ["parent"])],
                      lambda data: unknown())
    result = run_question({"id": "q", "question": "Requested relation?"}, resources, calls, config)
    assert [call["node_id"] for call in bridge.calls] == ["parent"]
    assert not result["budgets"]["20"]["complete_required"]
    assert any(e["event"] == "node_blocked" for e in result["diagnostics"]["events"])


def test_refinement_keeps_original_terminal_and_becomes_real_parent(setup):
    bridge, resources, config = setup
    config["fusion"]["refinement"] = True
    bridge.routes = {"answer": ["a"], "refinement_1": ["b"]}
    resolved = Counter()
    def resolver(data):
        nid = data["node_id"]
        resolved[nid] += 1
        if nid == "answer" and resolved[nid] == 1:
            return unknown(status="partial", refinements=[{"question": "Identify the pen name in this evidence?",
                         "answer_type": "person", "inputs": [],
                         "source_span_ids": [data["evidence"][0]["id"]]}])
        if nid == "refinement_1":
            return answer(data, "MODEL_ONLY_REFINED", ["b"])
        assert "MODEL_ONLY_REFINED" in data["subquestion"]
        return answer(data, "fixture_answer", ["a"], ["refinement_1"])
    calls = FakeCalls([step("answer")], resolver)
    result = run_question({"id": "q", "question": "Requested relation?"}, resources, calls, config)
    assert result["diagnostics"]["requirements"][0]["terminal_node_ids"] == ["answer"]
    assert [n["id"] for n in result["ranking"]["nodes"]] == ["refinement_1", "answer"]
    assert result["budgets"]["20"]["selected_doc_ids"] == ["a", "b"]
    frozen = [e["requirements_hash"] for e in result["diagnostics"]["events"]
              if e["event"] in ("plan_frozen", "evidence_refinement")]
    assert len(set(frozen)) == 1


def test_audit_invalidates_only_named_alternative_and_preserves_other(setup):
    bridge, resources, config = setup
    bridge.routes = {"answer": ["a", "c", "x"]}
    def audit(data):
        target = data["nodes"][0]["alternatives"][0]["id"]
        opposite = next(s["id"] for s in data["evidence"] if s["doc_id"] == "x")
        return {"conflicts": [{"alternative_ids": [target], "span_ids": [opposite],
                "reason": "Specific first support route has a conflicting entity binding", "entity_scope": "fixture",
                "event_time": None}], "unresolved_guards": []}
    calls = FakeCalls([step("answer")], lambda data: answer(data, "fixture_answer", [],
                      alternatives=[(["a"], []), (["c"], [])]), audit)
    result = run_question({"id": "q", "question": "Requested relation?"}, resources, calls, config)
    assert result["ranking"]["nodes"][0]["status"] == "supported"
    alternatives = result["ranking"]["nodes"][0]["alternatives"]
    assert not alternatives[0]["eligible"] and alternatives[1]["eligible"]
    assert result["budgets"]["20"]["chosen_alternatives"]["answer"] == alternatives[1]["id"]
    assert result["budgets"]["20"]["complete_required"]


def test_re_resolving_same_claim_does_not_launder_unresolved_conflict(setup):
    bridge, resources, config = setup
    bridge.routes = {"answer": ["a", "x"]}
    def audit(data):
        target = data["nodes"][0]["alternatives"][0]["id"]
        opposite = next(s["id"] for s in data["evidence"] if s["doc_id"] == "x")
        return {"conflicts": [{"alternative_ids": [target], "span_ids": [opposite],
                "reason": "Same entity relation contradicted", "entity_scope": "fixture", "event_time": None}],
                "unresolved_guards": []}
    calls = FakeCalls([step("answer")], lambda data: answer(data, "fixture_answer", ["a"]), audit)
    result = run_question({"id": "q", "question": "Requested relation?"}, resources, calls, config)
    assert not result["budgets"]["20"]["complete_required"]
    assert result["budgets"]["20"]["selected_doc_ids"] == ["a", "x"]
    assert result["ranking"]["nodes"][0]["status"] != "supported"
    assert result["diagnostics"]["support_graph"]["conflicts"]


def test_unresolved_guard_remains_partial_after_compile(setup):
    bridge, resources, config = setup
    bridge.routes = {"answer": ["a"]}
    calls = FakeCalls([step("answer")],
         lambda data: answer(data, "candidate", ["a"], status="partial", unresolved_guards=["event time unknown"]))
    result = run_question({"id": "q", "question": "Requested relation at specified time?"}, resources, calls, config)
    assert result["ranking"]["nodes"][0]["status"] == "partial"
    assert not result["budgets"]["20"]["complete_required"]


def test_gold_fields_are_rejected_before_any_model_request(setup):
    bridge, resources, config = setup
    calls = FakeCalls([step("answer")], lambda data: unknown())
    with pytest.raises(ProtocolError, match="only id/question"):
        Engine({"id": "q", "question": "Q?", "answers": ["gold"]}, resources, calls, config, "fusion")
    assert calls.requests == []


def test_bad_quote_is_logged_and_never_compiled_as_valid_evidence(setup):
    bridge, resources, config = setup
    bridge.routes = {"answer": ["a"]}
    def mapper(data):
        c = data["chunks"][0]
        return {"spans": [{"doc_id": c["doc_id"], "start": c["start"], "quote": "FABRICATED QUOTE",
                 "node_ids": ["answer"], "stance": "support", "event_time": None, "time_quote": None}]}
    calls = FakeCalls([step("answer")], lambda data: unknown(), mapper=mapper)
    result = run_question({"id": "q", "question": "Requested relation?"}, resources, calls, config)
    assert result["diagnostics"]["spans"] == []
    assert result["diagnostics"]["errors"]
    assert result["diagnostics"]["ledger"]["used"]["json_repairs"] == 2
    assert not result["budgets"]["20"]["complete_required"]


def test_same_pool_flat_and_dependency_use_same_raw_reader_and_evidence(setup):
    bridge, resources, config = setup
    config["fixed_candidate_pools"] = {"q": ["a", "b", "c"]}
    results, transports = {}, {}
    for method in ("fusion", "dense_dependency", "bt_flat", "dense_flat"):
        calls = FakeCalls([step("answer")], lambda data: answer(data, "fixture_answer", [],
                         alternatives=[(["a", "b"], []), (["c"], [])]),
                         selector=lambda data: {"selected_doc_ids": ["a", "b"],
                                  "reason": "fixture flat judgment", "covered_requirement_ids": ["answer"]})
        results[method] = run_question({"id": "q", "question": "Requested relation?"}, resources, calls, config, method)
        transports[method] = calls
    assert bridge.calls == []  # exact fixed-candidate analysis, no hidden search
    assert results["fusion"]["budgets"]["20"]["selected_doc_ids"] == ["c"]
    assert results["bt_flat"]["budgets"]["20"]["selected_doc_ids"] == ["a", "b"]
    evidence_tables = [r["diagnostics"]["spans"] for r in results.values()]
    assert all(table == evidence_tables[0] for table in evidence_tables)
    systems = [next(x for x in calls.requests if x["operation"] == "reader")["payload"]["messages"][0]
               for calls in transports.values()]
    assert all(system == systems[0] for system in systems)


def test_chain_ablation_never_injects_claim_without_selected_raw_closure(setup):
    bridge, resources, config = setup
    bridge.routes = {"parent": ["a"], "child": ["b"]}
    calls = FakeCalls([step("parent"), step("child", "Question about {parent}?", ["parent"])],
         lambda data: answer(data, "INTERNAL_" + data["node_id"],
                      ["a"] if data["node_id"] == "parent" else ["b"],
                      [] if data["node_id"] == "parent" else ["parent"]))
    engine = Engine({"id": "q", "question": "Requested relation?"}, resources, calls, config, "fusion_chain")
    result = engine.run()
    assert result["budgets"]["20"]["selected_doc_ids"] == ["a", "b"]
    full = json.dumps(engine.reader_messages(["a", "b"], engine.graph))
    incomplete = json.dumps(engine.reader_messages(["a"], engine.graph))
    assert "INTERNAL_child" in full
    assert "INTERNAL_child" not in incomplete
    assert "exact_source_spans" in full and "raw_text_hash" in full
    # A conservative chain may omit all intermediate claims when the terminal
    # proof is incomplete; the hard requirement is not injecting the child.


def test_scope_change_increments_semantic_version_and_invalidates_child(setup):
    bridge, resources, config = setup
    count = Counter()
    def resolver(data):
        nid = data["node_id"]
        count[nid] += 1
        value = answer(data, "same_value", ["a"] if nid == "parent" else ["b"], [] if nid == "parent" else ["parent"])
        value["alternatives"][0]["applicable_scope"] = "new_scope" if nid == "parent" and count[nid] > 1 else "old_scope"
        return value
    calls = FakeCalls([step("parent"), step("child", "Question of {parent}?", ["parent"])], resolver)
    engine = Engine({"id": "q", "question": "Requested relation?"}, resources, calls, config, "fusion")
    engine.plan(); engine.add_candidates(["a", "b"]); engine.map_pending(2)
    engine.resolve_node(engine.steps[0], "parent query")
    before = engine.node_map()["parent"]["version"]
    engine.resolve_node(engine.steps[1], "child query")
    engine.resolve_node(engine.steps[0], "new scoped query")
    assert engine.node_map()["parent"]["version"] == before + 1
    assert engine.node_map()["child"]["status"] != "supported"


@pytest.mark.parametrize("second_audit", ["malformed", "empty_without_resolution"])
def test_post_audit_rewrite_needs_explicit_conflict_resolution(setup, second_audit):
    bridge, resources, config = setup
    bridge.routes = {"answer": ["a", "x"]}
    audits = []
    def audit(data):
        audits.append(data)
        if len(audits) > 1:
            return {"conflicts": "invalid" if second_audit == "malformed" else [], "unresolved_guards": []}
        aid = data["nodes"][0]["alternatives"][0]["id"]
        sid = next(s["id"] for s in data["evidence"] if s["doc_id"] == "x")
        return {"conflicts": [{"alternative_ids": [aid], "span_ids": [sid],
                    "reason": "Contradiction remains unresolved", "entity_scope": "fixture", "event_time": None}],
                "unresolved_guards": []}
    calls = FakeCalls([step("answer")], lambda data: answer(data, "same_value", ["a"]), audit)
    result = run_question({"id": "q", "question": "Requested relation?"}, resources, calls, config)
    assert not result["budgets"]["20"]["complete_required"]
    assert result["ranking"]["nodes"][0]["status"] != "supported"
    assert result["budgets"]["20"]["selected_doc_ids"] == ["a", "x"]


def test_explicit_audit_resolution_retains_its_proof_as_raw_guard(setup):
    bridge, resources, config = setup
    bridge.routes = {"answer": ["a", "c", "x"]}
    audits = []
    def audit(data):
        audits.append(data)
        mapped = {s["doc_id"]: s["id"] for s in data["evidence"]}
        if len(audits) == 1:
            aid = data["nodes"][0]["alternatives"][0]["id"]
            return {"conflicts": [{"alternative_ids": [aid], "span_ids": [mapped["x"]],
                       "reason": "Potential entity ambiguity", "entity_scope": "fixture", "event_time": None}],
                    "unresolved_guards": []}
        cid = data["previous_conflicts"][0]["id"]
        return {"conflicts": [], "unresolved_guards": [], "resolutions": [
                    {"conflict_id": cid, "resolution_span_ids": [mapped["c"]],
                     "addressed_conflict_span_ids": [mapped["x"]], "resolution_kind": "entity_distinction",
                     "reason": "The exact c quote distinguishes the two named entities"}]}
    calls = FakeCalls([step("answer")], lambda data: answer(data, "same_value", ["a"]), audit)
    result = run_question({"id": "q", "question": "Requested relation?"}, resources, calls, config)
    assert result["budgets"]["20"]["complete_required"]
    assert result["budgets"]["20"]["selected_doc_ids"] == ["a", "c"]
    graph = result["diagnostics"]["support_graph"]
    assert graph["conflicts"][0]["resolution_status"] == "resolved"
    guard_ids = graph["nodes"][0]["alternatives"][0]["guard_span_ids"]
    assert {s["doc_id"] for s in graph["spans"] if s["id"] in guard_ids} == {"c"}


def test_chunk_overlap_preserves_quote_crossing_old_boundary_with_absolute_offsets(setup):
    bridge, resources, config = setup
    docs, ids, vectors, index, tokenizer = resources
    docs = {"long": Document("long", " ".join("word%04d" % i for i in range(3000)))}
    resources = (docs, list(docs), np.ones((1, 4), dtype=np.float32), index, tokenizer)
    config["fusion"]["map_batch_tokens"] = 1500
    target = {}
    def mapper(data):
        matching = [c for c in data["chunks"] if c["start"] <= target["start"]
                    and target["end"] <= c["start"] + len(c["text"])]
        return {"spans": [] if not matching else [{"doc_id": "long", "start": target["start"],
                    "quote": target["quote"], "node_ids": ["answer"], "stance": "support",
                    "event_time": None, "time_quote": None, "entity_scope": "fixture"}]}
    calls = FakeCalls([step("answer")], lambda data: unknown(), mapper=mapper)
    engine = Engine({"id": "q", "question": "Requested relation?"}, resources, calls, config, "fusion")
    engine.plan(); engine.add_candidates(["long"])
    assert len(engine.chunks) > 1
    boundary = len(engine.chunks[0]["text"])
    raw = docs["long"].passage
    target.update(start=boundary - 40, end=boundary + 40, quote=raw[boundary - 40:boundary + 40])
    assert any(c["start"] <= target["start"] and target["end"] <= c["start"] + len(c["text"])
               for c in engine.chunks[1:])
    engine.map_pending()
    assert len(engine.mapped_chunks) == len(engine.chunks)
    assert len(engine.spans) == 1
    span = next(iter(engine.spans.values()))
    assert span["start"] == target["start"] and span["end"] == target["end"]
    assert span["exact_quote"] == target["quote"]


def test_post_audit_missing_guard_triggers_budgeted_bridge_feedback_and_reaudit(setup):
    bridge, resources, config = setup
    bridge.routes = {"answer": ["a"]}
    guard = "Verify whether the relation applied in the question's specified year"
    audits = []
    def resolver(data):
        result = answer(data, "same_value", ["a"])
        correction = next((s for s in data["evidence"] if s["doc_id"] == "c"), None)
        if correction:
            result["alternatives"][0]["guard_span_ids"] = [correction["id"]]
        return result
    def audit(data):
        audits.append(data)
        if len(audits) == 1:
            # The newly identified condition leads to new evidence on the next
            # actual BridgeSession call, not just a second resolver invocation.
            bridge.routes["answer"] = ["a", "c"]
            return {"conflicts": [], "unresolved_guards": [{"node_id": "answer", "description": guard}]}
        return {"conflicts": [], "unresolved_guards": []}
    calls = FakeCalls([step("answer")], resolver, audit)
    result = run_question({"id": "q", "question": "Requested relation in a specified year?"}, resources, calls, config)
    assert len(bridge.calls) == 2 and len(audits) == 2
    assert not bridge.calls[0]["feedback"] and bridge.calls[1]["feedback"]
    assert guard in [need["description"] for need in bridge.calls[1]["requirements"]]
    assert result["budgets"]["20"]["selected_doc_ids"] == ["a", "c"]
    assert result["budgets"]["20"]["complete_required"]
    assert result["diagnostics"]["ledger"]["used"]["ann"] == 2
    reservations = [e["used"] for e in result["diagnostics"]["ledger"]["events"]
                    if e["event"] == "budget_reserved" and e["kind"] == "ann"]
    assert reservations == [1, 2]
    kinds = [e["event"] for e in result["diagnostics"]["events"]]
    first_audit = kinds.index("condition_audit")
    second_discovery = kinds.index("node_discovery", first_audit)
    second_audit = kinds.index("condition_audit", second_discovery)
    assert first_audit < second_discovery < second_audit
    assert result["diagnostics"]["requirements"][0]["terminal_node_ids"] == ["answer"]


def test_audit_repairs_cannot_spend_flat_final_selection_reservation(setup):
    bridge, resources, config = setup
    bridge.routes = {"answer": ["a"]}
    config["fusion"]["llm_calls"] = 6
    calls = FakeCalls([step("answer")], lambda data: answer(data, "fixture_answer", ["a"]),
                      audit=lambda data: {"conflicts": "malformed", "unresolved_guards": []})
    result = run_question({"id": "q", "question": "Requested relation?"}, resources, calls, config, "bt_flat")
    assert calls.counts["audit"] == 2
    assert calls.counts["select"] == 1 and calls.counts["reader"] == 1
    assert result["diagnostics"]["ledger"]["used"]["llm"] == 6
    assert any(e.get("stage") == "audit" and e["type"] == "BudgetExceeded"
               for e in result["diagnostics"]["errors"])


def test_transport_audit_retry_preserves_last_flat_selection_call(tmp_path):
    ledger = Ledger({"llm": 2})
    transport = Transport("q", tmp_path, {"fusion": {"selection": "flat", "reserved_audit_calls": 1}},
                          ledger, Tokenizer())
    transport._reserve("llm", "audit/1")
    with pytest.raises(BudgetExceeded):
        transport._reserve("llm", "audit/1")  # the next physical HTTP retry
    assert ledger.remaining("llm") == 1
    transport._reserve("llm", "select/final")
    assert ledger.remaining("llm") == 0


def test_no_conditions_ablation_skips_audit_not_source_or_guard_integrity(setup):
    bridge, resources, config = setup
    bridge.routes = {"answer": ["a"]}
    def unexpected_audit(data):
        raise AssertionError("Dedicated audit must be disabled for this ablation")
    calls = FakeCalls([step("answer")], lambda data: answer(data, "candidate", ["a"],
                     status="partial", unresolved_guards=["time condition unknown"]), unexpected_audit)
    result = run_question({"id": "q", "question": "Requested relation at a specified time?"},
                          resources, calls, config, "fusion_no_conditions")
    assert calls.counts["audit"] == 0
    assert any(e["event"] == "condition_audit_disabled" for e in result["diagnostics"]["events"])
    assert result["ranking"]["nodes"][0]["status"] == "partial"
    assert not result["budgets"]["20"]["complete_required"]
    assert result["diagnostics"]["spans"][0]["raw_text_hash"]


def proposal(candidate, sources=(), stage="initial_bridge", probe="probe1"):
    return {"candidate_id": candidate, "source_memory_ids": list(sources), "premise_ids": [],
            "target_id": None, "stage": stage, "probe_id": probe,
            "edge_type": "retrieval_proposal", "dependency_claim": False}


def test_navigation_ablation_retains_actual_path_only_at_final_selection(setup):
    bridge, resources, config = setup
    bridge.routes = {"answer": ["a", "b", "c"]}
    original_discover = bridge.discover
    def discover(self, query, node_id, **kwargs):
        result = original_discover(self, query, node_id, **kwargs)
        result["trace"] = {"node_id": node_id, "parent_source_doc_ids": [], "retrieval": {
            "source_graph": [proposal("a", stage="dense"), proposal("b", ["a"]), proposal("c", ["b"])]}}
        return result
    bridge.discover = discover
    outputs = {}
    for method in ("fusion", "fusion_navigation_closure"):
        calls = FakeCalls([step("answer")], lambda data: answer(data, "fixture_answer", ["c"]))
        outputs[method] = run_question({"id": "q", "question": "Requested relation?"}, resources, calls, config, method)
    normal, forced = outputs["fusion"], outputs["fusion_navigation_closure"]
    assert bridge.calls[0] == bridge.calls[1]  # selection-only intervention
    assert normal["budgets"]["20"]["selected_doc_ids"] == ["c"]
    assert forced["budgets"]["20"]["selected_doc_ids"] == ["a", "b", "c"]
    assert forced["budgets"]["20"]["forced_navigation_doc_ids"] == ["a", "b"]
    assert forced["budgets"]["20"]["token_count"] > normal["budgets"]["20"]["token_count"]
    assert forced["ranking"]["nodes"] == normal["ranking"]["nodes"]
    assert forced["diagnostics"]["navigation_provenance"][-1]["dependency_claim"] is False


def test_navigation_first_exposure_ignores_later_cyclic_rediscovery(setup):
    bridge, resources, config = setup
    bridge.routes = {"answer": ["c", "b"]}
    original_discover = bridge.discover
    def discover(self, query, node_id, **kwargs):
        result = original_discover(self, query, node_id, **kwargs)
        result["trace"] = {"node_id": node_id, "parent_source_doc_ids": [], "retrieval": {
            "source_graph": [proposal("c", stage="dense"), proposal("b", ["c"]), proposal("c", ["b"])]}}
        return result
    bridge.discover = discover
    calls = FakeCalls([step("answer")], lambda data: answer(data, "fixture_answer", ["c"]))
    result = run_question({"id": "q", "question": "Requested relation?"}, resources, calls, config, "fusion_navigation_closure")
    assert result["budgets"]["20"]["selected_doc_ids"] == ["c"]
    assert result["budgets"]["20"]["forced_navigation_doc_ids"] == []
    assert result["diagnostics"]["navigation_provenance"][0]["source_doc_ids"] == []


def test_navigation_includes_actual_parent_passages_injected_into_probe(setup):
    bridge, resources, config = setup
    bridge.routes = {"parent": ["a"], "answer": ["c"]}
    original_discover = bridge.discover
    def discover(self, query, node_id, **kwargs):
        result = original_discover(self, query, node_id, **kwargs)
        result["trace"] = {"node_id": node_id, "parent_source_doc_ids": kwargs["premise_doc_ids"], "retrieval": {
            "source_graph": [proposal(bridge.routes[node_id][0], stage="dense")]}}
        return result
    bridge.discover = discover
    calls = FakeCalls([step("parent"), step("answer", "Relation of {parent}?", ["parent"])],
              lambda data: answer(data, "fixture_answer", ["a"] if data["node_id"] == "parent" else ["c"]))
    result = run_question({"id": "q", "question": "Requested relation?"}, resources, calls, config, "fusion_navigation_closure")
    assert result["budgets"]["20"]["support_selected_doc_ids"] == ["c"]
    assert result["budgets"]["20"]["selected_doc_ids"] == ["a", "c"]
    assert result["budgets"]["20"]["forced_navigation_doc_ids"] == ["a"]


def test_navigation_ablation_rejects_fixed_pool_without_discovery_paths(setup):
    bridge, resources, config = setup
    config["fixed_candidate_pools"] = {"q": ["c"]}
    calls = FakeCalls([step("answer")], lambda data: answer(data, "fixture_answer", ["c"]))
    with pytest.raises(ProtocolError, match="discovery provenance"):
        run_question({"id": "q", "question": "Requested relation?"}, resources, calls, config, "fusion_navigation_closure")
