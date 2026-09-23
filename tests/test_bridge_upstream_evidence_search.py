from __future__ import annotations

import re
from dataclasses import replace

import numpy as np
import pytest

from vendor.bridgetree.clients import RerankItem
from vendor.bridgetree.dependency_retrieval import InitialCandidatePool, ProposalBatch, ProposalHit
from vendor.bridgetree.dependency_scoring import SetBudgetExceeded, SetReranker
from vendor.bridgetree.evidence_config import EvidenceSearchConfig
from vendor.bridgetree.evidence_search import EvidenceBridgeSearcher
from vendor.bridgetree.types import Memory


class Scorer:
    def __init__(self, values=None, default=0.0):
        self.values = {frozenset(ids): score for ids, score in (values or {}).items()}
        self.default = default
        self.seen = set()
        self.completed = {}
        self.limit = None
        self.requests = []

    @property
    def scored_sets(self):
        return len(self.seen)

    def set_budget(self, limit):
        assert limit >= self.scored_sets
        self.limit = limit

    def preflight(self, sets):
        required = len({frozenset(ids) for ids in sets} - self.seen)
        if self.limit is not None and required > self.limit - self.scored_sets:
            raise SetBudgetExceeded(required=required, remaining=self.limit - self.scored_sets, limit=self.limit)

    def score_sets(self, sets, reason="activation"):
        self.preflight(sets)
        self.requests.append(tuple(tuple(ids) for ids in sets))
        values = []
        for ids in sets:
            key = frozenset(ids)
            self.seen.add(key)
            value = self.values.get(key, self.default)
            self.completed[key] = value
            values.append(value)
        return values

    def measured_sets_snapshot(self):
        return tuple({"ids": sorted(ids), "score": score} for ids, score in self.completed.items())


class Retriever:
    def __init__(self, ids, scripts=None, max_ann_calls=12, vectors=None):
        self.visible_ids = tuple(ids)
        self.scripts = scripts or {}
        self.max_ann_calls = max_ann_calls
        self.ann_calls = 0
        self.calls = []
        self.memory_vectors = np.array(vectors if vectors is not None else np.eye(len(ids)), dtype=float)

    def propose(self, target_id, premise_ids=(), fixed_pool=False):
        assert self.ann_calls < self.max_ann_calls, "search exceeded absolute ANN budget"
        key = (target_id, tuple(premise_ids))
        assert key not in self.calls, "paused workspace repeated ANN"
        self.calls.append(key)
        self.ann_calls += 1
        ids = self.scripts.get(key, ())
        return ProposalBatch(
            probe_id=f"probe:{self.ann_calls}",
            stage="conditional",
            probe_text="test",
            hits=tuple(ProposalHit(identifier, 1 - index * 0.01, index) for index, identifier in enumerate(ids)),
            target_id=target_id,
            premise_ids=tuple(premise_ids),
            ann_call_index=self.ann_calls,
        )


def settings(**kwargs):
    return replace(EvidenceSearchConfig(root_selection="discovery"), **kwargs)


def run(ids, *, values=None, scripts=None, config=None, ann=12, sets=128, pair_width=4, roots=None):
    scorer, retriever = Scorer(values), Retriever(ids, scripts, max_ann_calls=ann)
    searcher = EvidenceBridgeSearcher(
        scorer, retriever, settings=config or settings(), max_scored_sets=sets, pair_rescue_width=pair_width
    )
    result = searcher.run(ids if roots is None else roots)
    return result, scorer, retriever, searcher


def test_explicit_root_coverage_survives_large_positive_successors():
    ids = tuple("abcdefghijk")
    values = {(identifier,): 0.1 for identifier in ids}
    values.update({("a", identifier): 0.9 for identifier in ids[1:]})
    result, _, retriever, _ = run(ids, values=values, ann=21, sets=240, config=settings(quantum_measurements=2))
    first = [
        event["target_id"]
        for event in result.scheduler_events
        if event["event"] == "quantum_started" and event["phase"] == "coverage"
    ]
    assert first == list(ids[:6])
    assert retriever.calls[:6] == [(identifier, ()) for identifier in ids[:6]]
    assert any(event["phase"] == "exploration" for event in result.scheduler_events if "phase" in event)
    assert result.final_scored_sets <= 240


def test_pause_resumes_without_ann_and_drains_measurements_after_ann_cap():
    ids = tuple("abcdefgh")
    result, scorer, retriever, _ = run(
        ids, ann=1, sets=100, config=settings(quantum_measurements=2, quantum_new_sets=4)
    )
    assert retriever.calls == [("a", ())]
    measured = {record.group_ids for record in result.activations}
    assert all((identifier,) in measured for identifier in ids[1:])
    assert len(measured) == 13  # seven singletons plus six independent pairs
    quanta = [event for event in result.scheduler_events if event["event"] == "quantum_completed"]
    assert len(quanta) > 1
    assert all(event["scored_sets_after"] - event["scored_sets_before"] <= 4 for event in quanta)
    assert all(event["measurements_completed"] <= 2 for event in quanta)
    assert any(event["phase"] == "pending_only" for event in quanta)
    assert scorer.scored_sets == len(scorer.seen)
    assert sum(record.singleton_tests for record in result.state_records) == 7
    assert sum(record.pair_tests for record in result.state_records) == 6
    assert all(record.proposed_count == 7 for record in result.state_records)


def test_pair_is_measured_after_positive_singleton_before_scanning_whole_pool():
    ids = tuple("abcdef")
    result, _, _, _ = run(
        ids,
        ann=1,
        values={("a", "b"): 0.8},
        scripts={("a", ()): ("b", "c", "d", "e")},
        config=settings(quantum_measurements=3),
    )
    records = [record for record in result.activations if record.target_id == "a" and not record.premise_ids]
    assert records[0].accepted
    assert [record.group_ids for record in records[:3]] == [("b",), ("c",), ("b", "c")]
    assert sum(len(record.group_ids) == 2 for record in records) == 6


def test_positive_activation_with_negative_target_uses_speculation_and_real_pivot():
    values = {(): 0.03, ("e",): 0.0, ("g",): 0.56, ("e", "g"): 0.54}
    result, _, retriever, _ = run(
        ("e", "g", "h"),
        roots=("e",),
        values=values,
        scripts={("e", ()): ("g",), ("g", ()): ("h",)},
        ann=8,
        pair_width=0,
        config=settings(coverage_roots=1, exploration_roots=2),
    )
    measurement = next(
        record for record in result.activations if record.target_id == "e" and record.group_ids == ("g",)
    )
    assert measurement.activation > 0
    assert not measurement.accepted
    decisions = [event for event in result.target_events if event["event"] == "target_decision"]
    assert decisions[0]["decision"] == "speculate"
    pivots = [event for event in result.target_events if event["event"] == "target_pivot" and event["queued"]]
    assert pivots[0]["replacement_ids"] == ("g",)
    assert pivots[0]["removed_ids"] == ("e",)
    assert ("g", ()) in retriever.calls
    assert ("e", "g") in result.bundle_ids and ("g",) in result.bundle_ids


def test_pivot_limits_and_cycle_detection_are_independent_of_archive_admission():
    values = {(): 0.4, ("e",): 0.1, ("g",): 0.5, ("e", "g"): 0.0}
    result, _, _, _ = run(
        ("e", "g"),
        roots=("e",),
        values=values,
        scripts={("e", ()): ("g",), ("g", ()): ("e",)},
        ann=6,
        pair_width=0,
        config=settings(coverage_roots=1, exploration_roots=2, max_pivots=4, max_pivot_depth=2),
    )
    pivots = [event for event in result.target_events if event["event"] == "target_pivot"]
    assert sum(event["queued"] for event in pivots) == 1
    assert any(event["reason"] == "target_cycle" for event in pivots)
    assert ("e", "g") in result.bundle_ids  # poor sets are not silently lost


def test_all_four_measured_sets_archive_even_for_rejected_activation():
    result, _, _, _ = run(
        ("e", "a", "b"),
        roots=("e",),
        values={("e", "a"): 0.5, ("a", "b"): 0.9, ("e", "a", "b"): 0.3},
        scripts={("e", ()): ("a",), ("e", ("a",)): ("b",)},
        ann=3,
        config=settings(coverage_roots=1, exploration_roots=0),
        pair_width=0,
    )
    assert ("a", "b") in result.bundle_ids
    all_scored = {tuple(item["ids"]) for item in result.measured_sets if item["ids"]}
    assert all_scored <= set(result.bundle_ids)
    assert any(record.activation < 0 and record.premise_ids for record in result.activations)


def test_atomic_quartet_budget_never_invents_activation_or_spends_gap_ann_reserve():
    scorer, retriever = Scorer(), Retriever(tuple("abcdef"), max_ann_calls=36)
    retriever.ann_calls = 30
    searcher = EvidenceBridgeSearcher(
        scorer, retriever, settings=settings(), max_scored_sets=3, pair_rescue_width=4, max_ann_calls=34
    )
    result = searcher.run(tuple("abcdef"))
    assert not result.activations
    assert scorer.scored_sets == 0  # four required sets were preflighted atomically
    assert retriever.ann_calls <= 34


def test_diverse_roots_keep_dense_first_and_use_cosine_farthest_with_stable_ties():
    retriever = Retriever(("z", "y", "x", "w"), vectors=((1, 0), (0.9, 0.1), (-1, 0), (0, 1)))
    dense = ProposalBatch("dense", "initial_dense", "q", hits=(ProposalHit("y", 1, 0), ProposalHit("z", 0.9, 1)))
    pool = InitialCandidatePool(("z", "y", "x", "w"), dense)
    searcher = EvidenceBridgeSearcher(
        Scorer(), retriever, settings=settings(root_selection="diverse"), max_scored_sets=32, pair_rescue_width=0
    )
    assert searcher.run(pool).root_order == ("y", "x", "w", "z")


class PhysicalReranker:
    score_space = "unit_interval"
    score_contract = "pointwise"
    model_fingerprint = "evidence-search-test"

    def __init__(self, fail_at=None):
        self.calls = 0
        self.fail_at = fail_at

    def rerank_all(self, query, documents):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("physical test failure")
        return [
            RerankItem(index=index, score=0.1 * len(re.findall('"memory_id":', document)))
            for index, document in enumerate(documents)
        ]


def records():
    return {
        identifier: Memory(memory_id=identifier, text=f"evidence {identifier}", timestamp=index, source_id=identifier)
        for index, identifier in enumerate(("e", "g", "h"))
    }


def test_mid_quartet_physical_failure_preserves_scores_without_activation():
    scorer = SetReranker("q", records(), PhysicalReranker(fail_at=3), batch_size=1)
    searcher = EvidenceBridgeSearcher(
        scorer, Retriever(("e", "g", "h")), settings=settings(), max_scored_sets=64, pair_rescue_width=0
    )
    with pytest.raises(RuntimeError, match="physical test failure"):
        searcher.run(("e", "g", "h"))
    partial = searcher.partial_public_dict()
    assert partial["partial"]
    assert partial["activations"] == []
    assert partial["scored_sets"] == 4  # attempted four distinct sets are charged
    assert len(partial["measured_sets"]) == 2  # only two physically returned
    assert ("e",) in {tuple(bundle["memory_ids"]) for bundle in partial["bundles"]}
    assert not partial["active_measurement"]["complete"]
    assert len(partial["active_measurement"]["available_sets"]) == 2


def test_warm_persistent_scores_keep_same_logical_budget_and_search(tmp_path):
    archives = []
    clients = []
    for _ in range(2):
        client = PhysicalReranker()
        scorer = SetReranker("q", records(), client, cache_dir=tmp_path, batch_size=2)
        retriever = Retriever(("e", "g", "h"), max_ann_calls=3)
        searcher = EvidenceBridgeSearcher(
            scorer, retriever, settings=settings(quantum_new_sets=4), max_scored_sets=7, pair_rescue_width=0
        )
        archives.append(searcher.run(("e", "g", "h")))
        clients.append(client)
    assert clients[0].calls > 0 and clients[1].calls == 0
    assert archives[0].final_scored_sets == archives[1].final_scored_sets
    assert archives[0].bundle_ids == archives[1].bundle_ids
    assert [record.public_dict() for record in archives[0].activations] == [
        record.public_dict() for record in archives[1].activations
    ]


def test_set_reserve_survives_coverage_and_mid_search_is_released_before_ann_depletion():
    ids = tuple(f"m{index:02}" for index in range(15))
    values = {(ids[0], identifier): 0.9 for identifier in ids[1:]}
    result, _, _, _ = run(ids, values=values, ann=21, sets=128)
    allocation = result.scheduler_events[0]
    coverage = [
        event
        for event in result.scheduler_events
        if event["event"] == "quantum_completed" and event["phase"] == "coverage"
    ]
    assert all(event["scored_sets_after"] <= 128 - allocation["reserved_sets"] for event in coverage)
    release = next(event for event in result.scheduler_events if event["event"] == "exploration_released")
    assert release["ann_calls"] < 21
    assert result.final_scored_sets <= 128
    assert sum(cost["scored_sets"] for cost in result.target_costs) == result.final_scored_sets
    assert sum(cost["ann_calls"] for cost in result.target_costs) == result.final_ann_calls


def test_speculative_state_and_path_depth_are_bounded():
    # Every singleton extension reduces the negative e marginal but e never
    # becomes beneficial. Three helper memories make successive probes possible.
    ids = ("e", "a", "b", "c")
    values = {(): 0.4, ("e",): 0.0}
    for size in range(1, 4):
        from itertools import combinations

        for group in combinations(ids[1:], size):
            values[group] = 0.6
            values[("e", *group)] = 0.3 + size * 0.05
    scripts = {("e", ()): ("a",), ("e", ("a",)): ("b",), ("e", ("a", "b")): ("c",)}
    result, _, _, _ = run(
        ids,
        roots=("e",),
        values=values,
        scripts=scripts,
        ann=8,
        config=settings(
            coverage_roots=1, exploration_roots=4, max_pivots=0, max_speculative_depth=1, max_speculative_states=1
        ),
        pair_width=0,
    )
    events = [
        event
        for event in result.target_events
        if event["event"] == "target_decision" and event["decision"] == "speculate"
    ]
    assert sum(event["queued"] for event in events) == 1
    assert any(event["reason"] == "speculative_depth_limit" for event in events)


def test_each_promised_coverage_root_gets_a_complete_measurement_under_small_budget():
    ids = tuple(f"m{index}" for index in range(20))
    result, _, _, _ = run(ids, ann=21, sets=64)
    starts = [
        event["target_id"]
        for event in result.scheduler_events
        if event["event"] == "quantum_started" and event["phase"] == "coverage"
    ]
    completed = [
        event
        for event in result.scheduler_events
        if event["event"] == "quantum_completed" and event["phase"] == "coverage"
    ]
    assert len(starts) == 6
    assert all(event["measurements_completed"] >= 1 for event in completed)
    assert result.final_scored_sets <= 64


def test_fairness_chooses_available_other_targets_and_records_unavoidable_relaxation():
    ids = tuple("abcdefgh")
    values = {(ids[0], identifier): 0.9 for identifier in ids[1:]}
    result, _, _, _ = run(
        ids, values=values, ann=6, sets=100, config=settings(max_consecutive_quanta=1, quantum_measurements=1)
    )
    target = None
    for event in result.scheduler_events:
        if event["event"] == "target_fairness_relaxed":
            target = None
        if event["event"] == "quantum_started":
            assert event["target_id"] != target
            target = event["target_id"]
    single, _, _, _ = run(ids, ann=1, config=settings(max_consecutive_quanta=1, quantum_measurements=1))
    assert any(event["event"] == "target_fairness_relaxed" for event in single.scheduler_events)


def test_zero_new_set_measurements_still_yield_at_measurement_quantum():
    ids = tuple("abcde")
    scorer, retriever = Scorer(), Retriever(ids, max_ann_calls=1)
    import itertools

    # Simulate already-charged task-local scores. Persistent warm caches are
    # separately tested and do not make logical scoring free.
    for size in range(len(ids) + 1):
        for group in itertools.combinations(ids, size):
            scorer.score_sets([group])
    initial = scorer.scored_sets
    searcher = EvidenceBridgeSearcher(
        scorer, retriever, settings=settings(quantum_measurements=2), max_scored_sets=0, pair_rescue_width=4
    )
    result = searcher.run(ids)
    assert result.final_scored_sets == initial
    completed = [event for event in result.scheduler_events if event["event"] == "quantum_completed"]
    assert len(completed) > 1
    assert all(event["measurements_completed"] <= 2 for event in completed)


def test_exception_quantum_is_logged_as_failure_without_mislabeled_success():
    scorer = SetReranker("q", records(), PhysicalReranker(fail_at=1), batch_size=1)
    searcher = EvidenceBridgeSearcher(
        scorer, Retriever(("e", "g", "h")), settings=settings(), max_scored_sets=32, pair_rescue_width=0
    )
    with pytest.raises(RuntimeError):
        searcher.run(("e", "g"))
    final = searcher.partial_public_dict()["scheduler_events"][-1]
    assert final["event"] == "quantum_completed"
    assert final["failed"] and not final["completed"]
    assert final["reason"] == "execution_error"


def test_module_files_preserve_mechanism_fields_and_reasons(tmp_path):
    import json

    from vendor.bridgetree.diagnostic_observability import ModuleEventRecorder, observation_scope

    recorder = ModuleEventRecorder(tmp_path, durable=False)
    with observation_scope(recorder):
        result, _, _, _ = run(
            tuple("abcdef"),
            ann=1,
            config=settings(quantum_measurements=1),
            values={("a",): -0.03, ("b",): 0.55, ("a", "b"): 0.53},
        )
    rows = [json.loads(line) for line in recorder.path.read_text().splitlines()]
    allocation = next(row for row in rows if row["event"] == "search_allocation")
    assert allocation["root_order"] == list(result.root_order)
    assert allocation["coverage_roots"] == 1
    quanta = [row for row in rows if row["event"] == "quantum_completed"]
    assert quanta and all("remaining_measurements" in row and "scored_sets_after" in row for row in quanta)
    decision = next(row for row in rows if row["event"] == "target_decision")
    assert decision["decision"] == "speculate"
    assert decision["target_marginal_after"] < 0 and decision["activation"] > 0
    pivot = next(row for row in rows if row["event"] == "target_pivot")
    assert pivot["old_target_id"] == "a" and pivot["new_target_id"] == "b"
    assert pivot["replacement_ids"] == ["b"]
    assert all(row.get("reason") != "[redacted]" for row in rows)
    assert (tmp_path / "modules" / "scheduler.jsonl").is_file()
    assert (tmp_path / "modules" / "target.jsonl").is_file()
    assert (tmp_path / "modules" / "archive.jsonl").is_file()


def test_varied_budgets_and_scores_preserve_global_and_per_quantum_invariants():
    import itertools
    import random

    # Bounded deterministic scenario exploration: interactions, cache reuse,
    # pivots and fairness can otherwise hide overspending at phase transitions.
    for seed in range(80):
        rng = random.Random(seed)
        ids = tuple("abcdef")
        values = {group: rng.random() for size in range(7) for group in itertools.combinations(ids, size)}
        scripts = {
            (target, group): tuple(identifier for identifier in ids if identifier != target and identifier not in group)
            for target in ids
            for size in range(6)
            for group in itertools.combinations(tuple(identifier for identifier in ids if identifier != target), size)
        }
        set_budget, ann_budget = rng.randrange(100), rng.randrange(1, 15)
        config = settings(
            quantum_new_sets=rng.choice((4, 6, 24)),
            quantum_measurements=rng.choice((1, 3, 8)),
            coverage_roots=rng.randrange(1, 7),
            exploration_roots=rng.randrange(5),
            max_consecutive_quanta=rng.randrange(1, 4),
        )
        result, scorer, retriever, _ = run(
            ids,
            values=values,
            scripts=scripts,
            ann=ann_budget,
            sets=set_budget,
            config=config,
            pair_width=rng.randrange(5),
        )
        assert scorer.scored_sets <= set_budget, seed
        assert retriever.ann_calls <= ann_budget, seed
        assert len(retriever.calls) == len(set(retriever.calls)), seed
        assert {tuple(item["ids"]) for item in result.measured_sets if item["ids"]} <= set(result.bundle_ids), seed
        assert sum(item.singleton_tests + item.pair_tests for item in result.state_records) == len(
            result.activations
        ), seed
        for event in result.scheduler_events:
            if event["event"] == "quantum_completed":
                assert event["ann_calls_after"] - event["ann_calls_before"] <= 1, seed
                assert event["scored_sets_after"] - event["scored_sets_before"] <= config.quantum_new_sets, seed
                assert event["measurements_completed"] <= config.quantum_measurements, seed


def test_scheduler_action_reports_actual_workspace_kind_independent_of_phase():
    archives = []
    # Negative target contribution creates both a pivot and speculative path.
    archives.append(
        run(
            ("e", "g", "h"),
            roots=("e",),
            ann=8,
            pair_width=0,
            values={(): 0.03, ("e",): 0.0, ("g",): 0.56, ("e", "g"): 0.54},
            scripts={("e", ()): ("g",), ("g", ()): ("h",)},
            config=settings(coverage_roots=1, exploration_roots=2),
        )[0]
    )
    # Discovery with no positive signal promotes an external memory as a root.
    archives.append(
        run(
            ("e", "g", "h"),
            roots=("e",),
            ann=8,
            pair_width=0,
            scripts={("e", ()): ("g", "h")},
            config=settings(coverage_roots=1, exploration_roots=2),
        )[0]
    )
    # Positive target retention is a genuinely deeper state; small quanta also
    # resume existing roots after yielding, without reclassifying them as opens.
    archives.append(
        run(
            tuple("abcdef"),
            ann=10,
            pair_width=0,
            values={("a", "b"): 0.8, ("a", "c"): 0.7},
            config=settings(coverage_roots=1, exploration_roots=1, quantum_measurements=1),
        )[0]
    )
    expected = {
        "root": "open_root",
        "promoted": "open_promoted_root",
        "pivot": "open_pivot",
        "speculative": "open_speculation",
        "retain": "deepen_state",
    }
    first_kinds = set()
    resumed = 0
    for archive in archives:
        starts = [event for event in archive.scheduler_events if event["event"] == "quantum_started"]
        completions = [event for event in archive.scheduler_events if event["event"] == "quantum_completed"]
        assert len(starts) == len(completions)
        for start, completion in zip(starts, completions):
            assert start["action"] == completion["action"]
            if start["resumed"]:
                assert start["action"] == "resume_state"
                resumed += 1
            else:
                first_kinds.add(start["lane"])
                assert start["action"] == expected[start["lane"]]
    assert first_kinds == set(expected)
    assert resumed > 0
