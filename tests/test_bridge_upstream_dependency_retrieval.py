from __future__ import annotations

import numpy as np

from vendor.bridgetree.dependency_retrieval import (
    DEPENDENCY_PROPOSAL_INSTRUCTION,
    DependencyRetriever,
)
from vendor.bridgetree.types import Memory


class RoutedEmbedder:
    def __init__(self, document_vectors):
        self.document_vectors = document_vectors
        self.queries = []
        self.instructions = []

    def encode(self, texts):
        return np.asarray([self.document_vectors[text] for text in texts], dtype=np.float32)

    def encode_query(self, text, instruction=None):
        self.queries.append(text)
        self.instructions.append(instruction)
        if "Fixed target memory" in text:
            return np.asarray([0.0, 1.0], dtype=np.float32)
        if "Bridge memory" in text:
            return np.asarray([0.0, 1.0], dtype=np.float32)
        return np.asarray([1.0, 0.0], dtype=np.float32)


def memory(identifier, text, timestamp):
    return Memory(identifier, text, timestamp, f"source:{identifier}", {"roles": ["user"]})


def test_dense_does_not_pay_for_unused_bridge_expansion_and_provenance_is_not_a_claim():
    memories = [memory("e", "target", 0), memory("p", "premise", 1)]
    embedder = RoutedEmbedder({"target": [1.0, 0.0], "premise": [0.0, 1.0]})
    retriever = DependencyRetriever(
        "question",
        memories,
        embedder,
        initial_width=1,
        initial_expansion_width=1,
        proposal_width=1,
        max_ann_calls=4,
    )

    dense = retriever.retrieve_dense()
    assert dense.ids == ("e",)
    assert retriever.ann_calls == 1
    assert all("Bridge memory" not in query for query in embedder.queries)

    pool = retriever.build_initial_pool()
    assert pool.candidate_ids == ("e", "p")
    assert retriever.ann_calls == 2
    assert any(edge.stage == "initial_bridge" for edge in pool.provenance)
    assert all(edge.edge_type == "retrieval_proposal" for edge in pool.provenance)
    assert all(edge.dependency_claim is False for edge in pool.provenance)
    assert embedder.instructions[-1] == DEPENDENCY_PROPOSAL_INSTRUCTION


def test_empty_proposal_instruction_cannot_bypass_fixed_bridge_instruction():
    memories = [memory("e", "target", 0), memory("p", "premise", 1)]
    embedder = RoutedEmbedder({"target": [1.0, 0.0], "premise": [0.0, 1.0]})
    retriever = DependencyRetriever(
        "question",
        memories,
        embedder,
        initial_width=1,
        initial_expansion_width=0,
        proposal_width=1,
        proposal_instruction="",
    )
    retriever.build_initial_pool()
    retriever.propose("e")
    assert embedder.instructions[-1] == DEPENDENCY_PROPOSAL_INSTRUCTION


def test_dynamic_proposal_reaches_full_visible_bank_but_fixed_pool_cannot():
    memories = [
        memory("e", "target", 0),
        memory("p", "pool premise", 1),
        memory("outside", "outside premise", 2),
    ]
    vectors = {
        "target": [1.0, 0.0],
        "pool premise": [0.99, 0.01],
        "outside premise": [0.0, 1.0],
    }

    dynamic = DependencyRetriever(
        "question",
        memories,
        RoutedEmbedder(vectors),
        initial_width=2,
        initial_expansion_width=0,
        proposal_width=1,
        max_ann_calls=4,
    )
    dynamic_pool = dynamic.build_initial_pool()
    assert dynamic_pool.candidate_ids == ("e", "p")
    dynamic_hit = dynamic.propose("e")
    assert dynamic_hit.ids == ("outside",)
    assert dynamic_hit.domain_scope == "full_visible_bank"

    fixed = DependencyRetriever(
        "question",
        memories,
        RoutedEmbedder(vectors),
        initial_width=2,
        initial_expansion_width=0,
        proposal_width=1,
        max_ann_calls=4,
        fixed_pool=True,
    )
    fixed.build_initial_pool()
    fixed_hit = fixed.propose("e")
    assert fixed_hit.ids == ("p",)
    assert "outside" not in fixed_hit.ids
    assert fixed_hit.domain_scope == "fixed_initial_pool"
    assert set(fixed_hit.excluded_ids) >= {"e", "outside"}


def test_conditional_probe_keeps_target_and_accumulated_premises_in_text_and_exclusions():
    memories = [
        memory("e", "fixed target text", 2),
        memory("p1", "first premise text", 0),
        memory("p2", "second premise text", 1),
        memory("x", "candidate text", 3),
    ]
    vectors = {
        "fixed target text": [1.0, 0.0],
        "first premise text": [0.9, 0.1],
        "second premise text": [0.8, 0.2],
        "candidate text": [0.0, 1.0],
    }
    retriever = DependencyRetriever(
        "original question",
        memories,
        RoutedEmbedder(vectors),
        initial_width=3,
        initial_expansion_width=0,
        proposal_width=1,
        max_ann_calls=4,
    )
    retriever.build_initial_pool()
    batch = retriever.propose("e", ("p2", "p1"))
    assert batch.target_id == "e"
    assert batch.premise_ids == ("p1", "p2")
    assert set(batch.excluded_ids) >= {"e", "p1", "p2"}
    assert "original question" in batch.probe_text
    assert "fixed target text" in batch.probe_text
    # Probe serialization is chronological, not dependent on caller order.
    assert batch.probe_text.index("first premise text") < batch.probe_text.index("second premise text")
