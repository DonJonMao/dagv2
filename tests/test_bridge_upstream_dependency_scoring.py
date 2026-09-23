import json
from dataclasses import replace

import pytest

from vendor.bridgetree.clients import RerankItem
from vendor.bridgetree.config import RerankerConfig
from vendor.bridgetree.dependency_scoring import (
    EMPTY_SET_SERIALIZATION,
    SetBudgetExceeded,
    SetInputTooLong,
    SetReranker,
    SetResponseError,
    probe_pointwise_consistency,
)
from vendor.bridgetree.types import Memory


def memory(identifier, text, timestamp, role="user", **metadata):
    return Memory(
        identifier,
        text,
        timestamp,
        f"source:{identifier}",
        {
            "roles": [role],
            "source_message_indices": [int(timestamp)],
            "time": {"observed_start": timestamp, "observed_end": timestamp},
            **metadata,
        },
    )


class FakeReranker:
    def __init__(self, values=None, *, response=None, score_space="unit_interval"):
        self.values = values if values is not None else {}
        self.response = response
        self.score_space = score_space
        self.score_contract = "pointwise"
        self.model_fingerprint = "deterministic-fixture-v1"
        self.config = RerankerConfig(
            endpoint="http://fixture.invalid/rerank",
            model="fixture",
            score_space=score_space,
            score_contract="pointwise",
        )
        self.calls = []

    def rerank_all(self, query, documents):
        self.calls.append((query, list(documents)))
        if self.response is not None:
            return self.response(query, documents)
        # Return score order rather than input order to exercise index restore.
        items = [
            RerankItem(index=index, score=self.values.get(document, 0.5))
            for index, document in enumerate(documents)
        ]
        return sorted(items, key=lambda item: (-item.score, item.index))


def fixture_scorer(fake=None, **kwargs):
    records = {
        "e": memory("e", "Target evidence", 2),
        "p": memory("p", "Premise evidence", 1, role="assistant"),
        "x": memory("x", "Independent evidence", 3),
    }
    fake = fake or FakeReranker()
    scorer = SetReranker("Original question?", records, fake, **kwargs)
    return scorer, fake, records


def test_four_scores_use_real_empty_set_and_shuffled_indices():
    scorer, fake, _ = fixture_scorer(batch_size=8, set_budget=4)
    fake.values.update(
        {
            scorer.serialize_set(()): 0.10,
            scorer.serialize_set(("e",)): 0.25,
            scorer.serialize_set(("p",)): 0.35,
            scorer.serialize_set(("p", "e")): 0.90,
        }
    )

    result = scorer.score_activation("e", (), ("p",))

    assert pytest.approx(0.10) == result.P
    assert result.Pe == pytest.approx(0.25)
    assert pytest.approx(0.35) == result.PG
    assert result.PGe == pytest.approx(0.90)
    assert result.activation == pytest.approx(0.40)
    assert result.context_marginal == pytest.approx(0.65)
    assert fake.calls[0][0] == "Original question?"
    assert EMPTY_SET_SERIALIZATION in fake.calls[0][1]
    assert scorer.scored_sets == 4
    assert scorer.reranker_adapter_requests == 1


def test_serialization_is_set_canonical_and_contains_real_provenance():
    scorer, fake, _ = fixture_scorer(set_budget=3)
    forward = scorer.serialize_set(("e", "p"))
    reverse = scorer.serialize_set(("p", "e", "p"))

    assert forward == reverse
    assert forward.index('"memory_id":"p"') < forward.index('"memory_id":"e"')
    assert '"roles":["assistant"]' in forward
    assert '"source_id":"source:p"' in forward
    assert '"observed_start":1' in forward
    assert "Premise evidence" in forward and "Target evidence" in forward
    assert scorer.score_set(("e", "p")) == scorer.score_set(("p", "e"))
    assert len(fake.calls) == 1
    assert scorer.scored_sets == 1


def test_warm_cache_still_consumes_logical_budget(tmp_path):
    cold, cold_client, records = fixture_scorer(cache_dir=tmp_path, set_budget=1)
    cold.score_set(("e",))
    assert cold_client.calls

    warm_client = FakeReranker()
    warm = SetReranker(
        "Original question?",
        records,
        warm_client,
        cache_dir=tmp_path,
        set_budget=1,
    )
    assert warm.score_set(("e",)) == 0.5
    assert warm.scored_sets == 1
    assert warm.persistent_cache_hits == 1
    assert warm.reranker_adapter_requests == 0
    with pytest.raises(SetBudgetExceeded):
        warm.score_set(("p",))
    assert warm.scored_sets == 1
    assert warm_client.calls == []


def test_cache_identity_includes_visible_raw_metadata(tmp_path):
    first, _, records = fixture_scorer(cache_dir=tmp_path)
    first.score_set(("e",))
    changed = dict(records)
    changed["e"] = replace(records["e"], metadata={**records["e"].metadata, "revision": 2})
    client = FakeReranker()
    second = SetReranker("Original question?", changed, client, cache_dir=tmp_path)

    second.score_set(("e",))
    assert second.namespace_hash != first.namespace_hash
    assert len(client.calls) == 1
    assert second.persistent_cache_hits == 0


def test_complete_request_preflight_commits_no_partial_scores():
    scorer, fake, _ = fixture_scorer(set_budget=1)

    with pytest.raises(SetBudgetExceeded):
        scorer.score_sets([("e",), ("p",)])

    assert scorer.scored_sets == 0
    assert fake.calls == []
    scorer.score_set(("e",))
    with pytest.raises(ValueError, match="already scored"):
        scorer.set_budget(0)


def test_four_term_capacity_preflight_happens_before_calls():
    scorer, fake, _ = fixture_scorer(max_input_tokens=5, set_budget=20)

    with pytest.raises(SetInputTooLong, match="exceeding limit"):
        scorer.score_activation("e", (), ("p",))

    assert scorer.scored_sets == 0
    assert fake.calls == []


@pytest.mark.parametrize(
    "response, match",
    [
        (lambda _q, _d: [{"index": 0, "score": 0.5}], "returned 1"),
        (
            lambda _q, _d: [
                {"index": 0, "score": 0.5},
                {"index": 0, "score": 0.4},
            ],
            "duplicate",
        ),
        (
            lambda _q, _d: [
                {"index": 0, "score": float("nan")},
                {"index": 1, "score": 0.4},
            ],
            "finite",
        ),
        (
            lambda _q, _d: {
                "truncated": True,
                "results": [
                    {"index": 0, "score": 0.5},
                    {"index": 1, "score": 0.4},
                ],
            },
            "truncation",
        ),
        (
            lambda _q, _d: {
                "results": [
                    {"index": 0, "score": 0.5},
                    {"index": 1, "score": 0.4},
                ],
                "provider": {"limits": {"documents-truncated": "yes"}},
            },
            "truncation",
        ),
    ],
)
def test_invalid_batch_responses_fail_explicitly(response, match):
    fake = FakeReranker(response=response)
    scorer, _, _ = fixture_scorer(fake, set_budget=2)

    with pytest.raises(SetResponseError, match=match):
        scorer.score_sets([("e",), ("p",)])


@pytest.mark.parametrize(
    "flag_envelope",
    [
        {"documents_truncated": True},
        {"inputs_truncated": "true"},
        {"meta": {"document_truncated": "yes"}},
        {"response": {"usage": {"inputs-truncated": "1"}}},
        {"audit": [{"was_truncated": 1}]},
        {"provider": {"finish_reason": "length"}},
    ],
)
def test_direct_reranker_rejects_all_shared_positive_truncation_forms(flag_envelope):
    def response(_query, documents):
        return {
            **flag_envelope,
            "results": [
                {"index": index, "score": 0.5}
                for index, _ in enumerate(documents)
            ],
        }

    scorer, _, _ = fixture_scorer(FakeReranker(response=response), set_budget=2)
    with pytest.raises(SetResponseError, match="truncation"):
        scorer.score_sets([("e",), ("p",)])


@pytest.mark.parametrize(
    "non_flag_envelope",
    [
        {"documents_truncated": False},
        {"inputs_truncated": "false"},
        {"meta": {"document_truncated": 0}},
        {"usage": {"input_tokens": 8192, "max_input_tokens": 8192}},
        {"truncation": "longest_first"},
    ],
)
def test_direct_reranker_does_not_guess_truncation(non_flag_envelope):
    def response(_query, documents):
        return {
            **non_flag_envelope,
            "results": [
                {"index": index, "score": 0.5}
                for index, _ in enumerate(documents)
            ],
        }

    scorer, _, _ = fixture_scorer(FakeReranker(response=response), set_budget=2)
    assert scorer.score_sets([("e",), ("p",)]) == [0.5, 0.5]


def test_batching_restores_each_batch_by_index():
    scorer, fake, _ = fixture_scorer(batch_size=2, set_budget=4)
    fake.values.update(
        {
            scorer.serialize_set(()): 0.1,
            scorer.serialize_set(("e",)): 0.2,
            scorer.serialize_set(("p",)): 0.3,
            scorer.serialize_set(("x",)): 0.4,
        }
    )

    assert scorer.score_sets([(), ("e",), ("p",), ("x",)]) == pytest.approx(
        [0.1, 0.2, 0.3, 0.4]
    )
    assert scorer.reranker_adapter_requests == 2


def test_transport_fallback_stats_are_audited_without_changing_logical_budget():
    class TrackedReranker(FakeReranker):
        def __init__(self):
            super().__init__()
            self.counters = {
                "logical_calls": 7,
                "logical_documents": 21,
                "batch_requests": 8,
                "batch_documents": 25,
                "transport_attempts": 10,
                "transport_document_attempts": 31,
                "failed_batch_requests": 1,
                "split_events": 1,
                "split_recovered_calls": 1,
                "failed_calls": 0,
            }

        @property
        def transport_stats(self):
            return dict(self.counters)

        def rerank_all(self, query, documents):
            result = super().rerank_all(query, documents)
            count = len(documents)
            self.counters["logical_calls"] += 1
            self.counters["logical_documents"] += count
            self.counters["batch_requests"] += 3
            self.counters["batch_documents"] += count * 2
            self.counters["transport_attempts"] += 5
            self.counters["transport_document_attempts"] += count * 4
            self.counters["failed_batch_requests"] += 1
            self.counters["split_events"] += 1
            self.counters["split_recovered_calls"] += 1
            return result

    scorer, _, _ = fixture_scorer(
        TrackedReranker(), batch_size=8, set_budget=2
    )

    assert scorer.score_sets([("e",), ("p",)]) == [0.5, 0.5]
    assert scorer.scored_sets == 2
    assert scorer.reranker_adapter_requests == 1
    assert scorer.cost["reranker_transport"] == {
        "logical_calls": 1,
        "logical_documents": 2,
        "batch_requests": 3,
        "batch_documents": 4,
        "transport_attempts": 5,
        "transport_document_attempts": 8,
        "failed_batch_requests": 1,
        "split_events": 1,
        "split_recovered_calls": 1,
        "failed_calls": 0,
    }


def test_pointwise_probe_reports_and_rejects_batch_dependence():
    stable = FakeReranker()
    report = probe_pointwise_consistency(stable, "q", ["a", "b"])
    assert report.consistent
    assert report.adapter_requests == 5

    def normalized(_query, documents):
        return [
            {"index": index, "score": 1.0 / len(documents)}
            for index, _ in enumerate(documents)
        ]

    unstable = FakeReranker(response=normalized)
    with pytest.raises(SetResponseError, match="batch composition"):
        probe_pointwise_consistency(unstable, "q", ["a", "b"])


def test_cache_files_do_not_persist_query_or_endpoint(tmp_path):
    scorer, _, _ = fixture_scorer(cache_dir=tmp_path)
    scorer.score_set(("e",))
    files = list(tmp_path.rglob("*.json"))
    assert len(files) == 1
    persisted = files[0].read_text(encoding="utf-8")
    assert "Original question" not in persisted
    assert "fixture.invalid" not in persisted
    assert json.loads(persisted)["ids"] == ["e"]
