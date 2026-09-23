"""Deterministic proposal retrieval for conditional-activation search.

This module only *proposes* real memories.  A retrieval edge is deliberately
labelled ``retrieval_proposal`` and never represents a dependency claim; the
set scorer in :mod:`bridgetree.dependency_scoring` is the authority for all
activation decisions.

The caller must pass the already visibility-filtered memory bank.  Keeping
that boundary explicit prevents a future record from being embedded, indexed,
or exposed through diagnostics before the question's cutoff is applied.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from numbers import Integral
from typing import Any, Iterable, Mapping, Protocol, Sequence

import numpy as np

from .index import ExactInnerProductIndex
from .diagnostic_observability import observe
from .types import Memory

DEPENDENCY_PROPOSAL_INSTRUCTION = (
    "Instruct: Retrieve a distinct past personal interaction that complements an anchor "
    "for answering the current request\nQuery: "
)


class DependencyEmbedder(Protocol):
    """Small embedding protocol used by :class:`DependencyRetriever`."""

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_query(self, text: str, instruction: str | None = None) -> np.ndarray: ...


class RetrievalBudgetExhausted(RuntimeError):
    """Raised by :meth:`DependencyRetriever.require_proposal` at the ANN cap."""


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _memory_value(record: Memory | Mapping[str, Any], name: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _memory_id(record: Memory | Mapping[str, Any]) -> str:
    value = str(_memory_value(record, "memory_id", ""))
    if not value:
        raise ValueError("every visible memory must have a non-empty memory_id")
    return value


def _memory_text(record: Memory | Mapping[str, Any]) -> str:
    value = _memory_value(record, "text", None)
    if not isinstance(value, str) or not value:
        raise ValueError("every visible memory must have non-empty text")
    return value


def _chronology_key(record: Memory | Mapping[str, Any]) -> tuple[float, str]:
    try:
        timestamp = float(_memory_value(record, "timestamp", float("inf")))
        if not np.isfinite(timestamp):
            timestamp = float("inf")
    except (TypeError, ValueError, OverflowError):
        timestamp = float("inf")
    return timestamp, _memory_id(record)


def _encode_query(embedder: DependencyEmbedder, text: str, instruction: str | None) -> np.ndarray:
    method = getattr(embedder, "encode_query", None)
    if callable(method):
        if instruction is None:
            value = method(text)
        else:
            try:
                value = method(text, instruction=instruction)
            except TypeError as exc:
                # Lightweight deterministic fakes often expose only
                # ``encode_query(text)``.  Fall back only when the optional
                # instruction cannot be supplied, not after arbitrary model
                # errors from a compatible signature.
                if "instruction" not in str(exc):
                    raise
                value = method(instruction + text)
    else:
        prefix = "" if instruction is None else instruction
        matrix = np.asarray(embedder.encode([prefix + text]), dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != 1:
            raise ValueError("embedding provider returned an invalid query matrix")
        value = matrix[0]
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("embedding provider returned an empty or non-finite query vector")
    return vector


@dataclass(frozen=True)
class ProposalHit:
    """One real-memory exposure from one exact proposal probe."""

    memory_id: str
    score: float
    rank: int

    def __post_init__(self) -> None:
        identifier = str(self.memory_id)
        score = float(self.score)
        rank = _nonnegative_int(self.rank, "proposal rank")
        if not identifier:
            raise ValueError("proposal hit memory_id cannot be empty")
        if not np.isfinite(score):
            raise ValueError("proposal hit score must be finite")
        object.__setattr__(self, "memory_id", identifier)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "rank", rank)

    def public_dict(self) -> dict[str, Any]:
        return {"memory_id": self.memory_id, "score": self.score, "rank": self.rank}


@dataclass(frozen=True)
class ProposalEdge:
    """Auditable provenance for a proposal, explicitly not a dependency."""

    probe_id: str
    candidate_id: str
    stage: str
    rank: int
    score: float
    source_memory_ids: tuple[str, ...] = ()
    target_id: str | None = None
    premise_ids: tuple[str, ...] = ()
    edge_type: str = "retrieval_proposal"
    dependency_claim: bool = False

    def __post_init__(self) -> None:
        if self.edge_type != "retrieval_proposal" or self.dependency_claim is not False:
            raise ValueError("retrieval provenance cannot assert a dependency claim")
        if not str(self.probe_id) or not str(self.candidate_id):
            raise ValueError("proposal edge IDs cannot be empty")
        object.__setattr__(self, "probe_id", str(self.probe_id))
        object.__setattr__(self, "candidate_id", str(self.candidate_id))
        object.__setattr__(self, "stage", str(self.stage))
        object.__setattr__(self, "rank", _nonnegative_int(self.rank, "proposal edge rank"))
        score = float(self.score)
        if not np.isfinite(score):
            raise ValueError("proposal edge score must be finite")
        object.__setattr__(self, "score", score)
        sources = tuple(str(value) for value in self.source_memory_ids)
        premises = tuple(sorted({str(value) for value in self.premise_ids}))
        if any(not value for value in sources + premises):
            raise ValueError("proposal edge source IDs cannot be empty")
        object.__setattr__(self, "source_memory_ids", sources)
        object.__setattr__(self, "premise_ids", premises)
        object.__setattr__(self, "target_id", None if self.target_id is None else str(self.target_id))

    def public_dict(self) -> dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "candidate_id": self.candidate_id,
            "stage": self.stage,
            "rank": self.rank,
            "score": self.score,
            "source_memory_ids": list(self.source_memory_ids),
            "target_id": self.target_id,
            "premise_ids": list(self.premise_ids),
            "edge_type": self.edge_type,
            "dependency_claim": self.dependency_claim,
        }


@dataclass(frozen=True)
class ProposalBatch:
    """Complete result of one logical ANN proposal attempt."""

    probe_id: str
    stage: str
    probe_text: str
    hits: tuple[ProposalHit, ...] = ()
    target_id: str | None = None
    premise_ids: tuple[str, ...] = ()
    source_memory_ids: tuple[str, ...] = ()
    domain_scope: str = "full_visible_bank"
    excluded_ids: tuple[str, ...] = ()
    ann_call_index: int | None = None
    stop_reason: str | None = None

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(hit.memory_id for hit in self.hits)

    @property
    def proposed_count(self) -> int:
        return len(self.hits)

    @property
    def budget_exhausted(self) -> bool:
        return self.stop_reason == "ann_budget_exhausted"

    def public_dict(self) -> dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "stage": self.stage,
            "probe_text": self.probe_text,
            "hits": [hit.public_dict() for hit in self.hits],
            "candidate_ids": list(self.ids),
            "proposed_count": self.proposed_count,
            "target_id": self.target_id,
            "premise_ids": list(self.premise_ids),
            "source_memory_ids": list(self.source_memory_ids),
            "domain_scope": self.domain_scope,
            "excluded_ids": list(self.excluded_ids),
            "ann_call_index": self.ann_call_index,
            "stop_reason": self.stop_reason,
        }


@dataclass(frozen=True)
class InitialCandidatePool:
    """Union of the dense roots and every bridge exposure."""

    candidate_ids: tuple[str, ...]
    dense_batch: ProposalBatch
    expansion_batches: tuple[ProposalBatch, ...] = ()
    provenance: tuple[ProposalEdge, ...] = ()
    ann_calls: int = 0
    stop_reason: str | None = None

    def __post_init__(self) -> None:
        ids = tuple(str(value) for value in self.candidate_ids)
        if len(ids) != len(set(ids)) or any(not value for value in ids):
            raise ValueError("initial candidate IDs must be unique and non-empty")
        object.__setattr__(self, "candidate_ids", ids)

    @property
    def ids(self) -> tuple[str, ...]:
        return self.candidate_ids

    @property
    def initial_hits(self) -> tuple[ProposalHit, ...]:
        return self.dense_batch.hits

    def public_dict(self) -> dict[str, Any]:
        return {
            "candidate_ids": list(self.candidate_ids),
            "dense_batch": self.dense_batch.public_dict(),
            "expansion_batches": [batch.public_dict() for batch in self.expansion_batches],
            "provenance": [edge.public_dict() for edge in self.provenance],
            "ann_calls": self.ann_calls,
            "stop_reason": self.stop_reason,
        }


def bridge_probe_text(query: str, seed: Memory | Mapping[str, Any]) -> str:
    """Serialize the original query plus one real seed for embedding only."""

    return (
        "Original question:\n"
        f"{query}\n\n"
        "Bridge memory (real visible history):\n"
        f"[{_memory_id(seed)}]\n{_memory_text(seed)}"
    )


def conditional_probe_text(
    query: str,
    target: Memory | Mapping[str, Any],
    premises: Sequence[Memory | Mapping[str, Any]],
) -> str:
    """Serialize ``q + fixed target + current premises`` for embedding only."""

    ordered = sorted(premises, key=_chronology_key)
    premise_text = "[No accepted premises]" if not ordered else "\n\n".join(
        f"[{_memory_id(record)}]\n{_memory_text(record)}" for record in ordered
    )
    return (
        "Original question:\n"
        f"{query}\n\n"
        "Fixed target memory (real visible history):\n"
        f"[{_memory_id(target)}]\n{_memory_text(target)}\n\n"
        "Already accepted premise memories (real visible history):\n"
        f"{premise_text}"
    )


class DependencyRetriever:
    """One deterministic proposal interface over a complete visible bank.

    ``fixed_pool`` changes only the domain of conditional probes.  Initial
    dense and bridge retrieval always run against the complete visible bank,
    establishing the frozen pool used by the control method.
    """

    def __init__(
        self,
        query: str,
        memories: Sequence[Memory | Mapping[str, Any]],
        embedder: DependencyEmbedder,
        *,
        memory_vectors: np.ndarray | None = None,
        index: ExactInnerProductIndex | None = None,
        initial_width: int = 12,
        initial_expansion_width: int = 4,
        proposal_width: int = 4,
        max_ann_calls: int | None = 36,
        fixed_pool: bool = False,
        query_instruction: str | None = None,
        proposal_instruction: str | None = None,
    ) -> None:
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        if isinstance(memories, (str, bytes)) or not isinstance(memories, Sequence):
            raise ValueError("memories must be the visibility-filtered sequence")
        if not isinstance(fixed_pool, bool):
            raise ValueError("fixed_pool must be boolean")
        self.query = query
        self.memories = tuple(memories)
        self.embedder = embedder
        self.initial_width = _nonnegative_int(initial_width, "initial_width")
        self.initial_expansion_width = _nonnegative_int(
            initial_expansion_width, "initial_expansion_width"
        )
        self.proposal_width = _nonnegative_int(proposal_width, "proposal_width")
        self.max_ann_calls = (
            None if max_ann_calls is None else _nonnegative_int(max_ann_calls, "max_ann_calls")
        )
        self.fixed_pool = fixed_pool
        self.query_instruction = query_instruction
        if proposal_instruction is not None and not isinstance(proposal_instruction, str):
            raise ValueError("proposal_instruction must be a string or None")
        # An empty overlay must not bypass the established bridge-query
        # instruction.  Conditional probes need the same retrieval role as
        # initial bridge probes, while the richer probe body remains q+e+P.
        self.proposal_instruction = (
            DEPENDENCY_PROPOSAL_INSTRUCTION
            if proposal_instruction is None or not proposal_instruction.strip()
            else proposal_instruction
        )

        ids = tuple(_memory_id(record) for record in self.memories)
        if len(ids) != len(set(ids)):
            raise ValueError("visible memory IDs must be unique")
        self.memory_by_id: dict[str, Memory | Mapping[str, Any]] = dict(zip(ids, self.memories))
        self.visible_ids = ids

        if memory_vectors is None:
            if self.memories:
                vectors = np.asarray(
                    embedder.encode([_memory_text(record) for record in self.memories]),
                    dtype=np.float32,
                )
            else:
                vectors = np.empty((0, 0), dtype=np.float32)
        else:
            vectors = np.asarray(memory_vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(self.memories):
            raise ValueError("memory_vectors must contain one row per visible memory")
        if self.memories and (vectors.shape[1] == 0 or not np.all(np.isfinite(vectors))):
            raise ValueError("memory vectors must be non-empty and finite")
        self.memory_vectors = vectors
        if index is not None:
            if tuple(str(value) for value in getattr(index, "ids", ())) != ids:
                raise ValueError("the ANN index must contain exactly the visible bank in the same order")
            self.index = index
        elif self.memories:
            self.index = ExactInnerProductIndex(ids, vectors)
        else:
            self.index = None

        self.ann_calls = 0
        self._attempts = 0
        self._batches: list[ProposalBatch] = []
        self._edges: list[ProposalEdge] = []
        self._dense_batch: ProposalBatch | None = None
        self._initial_pool: InitialCandidatePool | None = None
        self.information_needs: tuple[dict[str, Any], ...] = ()

    def set_information_needs(self, requirements: Sequence[Mapping[str, Any]]) -> None:
        """Attach q-only planning to conditional proposals, not the scorer/query baseline."""
        needs = []
        for requirement in requirements:
            if not isinstance(requirement, Mapping):
                raise ValueError("information requirements must be mappings")
            value = {key: requirement[key] for key in ("id", "description", "time_scope") if key in requirement}
            if not isinstance(value.get("description"), str) or not value["description"].strip():
                raise ValueError("information requirement needs a description")
            needs.append(value)
        self.information_needs = tuple(needs)

    def retrieve_missing(
        self, requirements: Sequence[Mapping[str, Any]], *, exclude: Iterable[str] = (),
        source_memory_ids: Sequence[str] = (), width: int | None = None,
    ) -> ProposalBatch:
        """Spend the ordinary ANN budget on an explicitly missing evidence need."""
        missing = []
        for requirement in requirements:
            if not isinstance(requirement, Mapping) or not isinstance(requirement.get("description"), str):
                raise ValueError("missing requirements need descriptions")
            missing.append({key: requirement[key] for key in ("id", "description", "time_scope") if key in requirement})
        sources = tuple(str(identifier) for identifier in source_memory_ids)
        if set(sources).difference(self.memory_by_id):
            raise ValueError("feedback sources must belong to the visible bank")
        # Source IDs preserve provenance. Do not feed a guessed answer back
        # into retrieval as a fact or expose reader options in this probe.
        text = (f"User query:\n{self.query}\n\nMissing historical information:\n"
                + json.dumps(missing, ensure_ascii=False, sort_keys=True))
        return self._run_probe(
            stage="evidence_gap", probe_text=text,
            width=self.proposal_width if width is None else width,
            exclude=exclude, source_memory_ids=sources, instruction=self.proposal_instruction,
        )

    @property
    def remaining_ann_calls(self) -> int | None:
        if self.max_ann_calls is None:
            return None
        return max(0, self.max_ann_calls - self.ann_calls)

    @property
    def proposal_batches(self) -> tuple[ProposalBatch, ...]:
        return tuple(self._batches)

    @property
    def provenance(self) -> tuple[ProposalEdge, ...]:
        return tuple(self._edges)

    @property
    def initial_candidate_ids(self) -> tuple[str, ...]:
        return () if self._initial_pool is None else self._initial_pool.candidate_ids

    def _next_probe_id(self, stage: str) -> str:
        self._attempts += 1
        return f"{stage}:{self._attempts:05d}"

    def _run_probe(
        self,
        *,
        stage: str,
        probe_text: str,
        width: int,
        exclude: Iterable[str] = (),
        allowed_ids: Iterable[str] | None = None,
        source_memory_ids: Sequence[str] = (),
        target_id: str | None = None,
        premise_ids: Sequence[str] = (),
        instruction: str | None = None,
    ) -> ProposalBatch:
        width = _nonnegative_int(width, "proposal width")
        excluded = {str(value) for value in exclude}
        allowed: set[str] | None = None
        if allowed_ids is not None:
            allowed = {str(value) for value in allowed_ids}
            unknown = allowed.difference(self.memory_by_id)
            if unknown:
                raise ValueError(f"proposal domain contains unknown memory IDs: {sorted(unknown)}")
            excluded.update(set(self.visible_ids).difference(allowed))
        unknown_exclusions = excluded.difference(self.memory_by_id)
        if unknown_exclusions:
            raise ValueError(f"proposal exclusions contain unknown memory IDs: {sorted(unknown_exclusions)}")

        probe_id = self._next_probe_id(stage)
        domain_scope = "fixed_initial_pool" if allowed is not None else "full_visible_bank"
        canonical_premises = tuple(sorted({str(value) for value in premise_ids}))
        sources = tuple(str(value) for value in source_memory_ids)
        available_count = sum(identifier not in excluded for identifier in self.visible_ids)
        if width == 0 or self.index is None or available_count == 0:
            batch = ProposalBatch(
                probe_id,
                stage,
                probe_text,
                (),
                target_id,
                canonical_premises,
                sources,
                domain_scope,
                tuple(sorted(excluded)),
                None,
                "no_candidates",
            )
            self._batches.append(batch)
            observe("proposal", batch, event="proposal_completed")
            return batch
        if self.max_ann_calls is not None and self.ann_calls >= self.max_ann_calls:
            batch = ProposalBatch(
                probe_id,
                stage,
                probe_text,
                (),
                target_id,
                canonical_premises,
                sources,
                domain_scope,
                tuple(sorted(excluded)),
                None,
                "ann_budget_exhausted",
            )
            self._batches.append(batch)
            observe("proposal", batch, event="proposal_completed")
            return batch

        observe("proposal", "proposal_started", probe_id=probe_id, stage=stage,
                target_id=target_id, premise_ids=canonical_premises,
                source_memory_ids=sources, domain_scope=domain_scope,
                width=width, available_count=available_count, ann_calls=self.ann_calls)
        vector = _encode_query(self.embedder, probe_text, instruction)
        raw_hits = self.index.search(vector, width, exclude=excluded)
        self.ann_calls += 1
        hits = tuple(
            ProposalHit(memory_id, score, rank)
            for rank, (memory_id, score) in enumerate(raw_hits)
        )
        batch = ProposalBatch(
            probe_id,
            stage,
            probe_text,
            hits,
            target_id,
            canonical_premises,
            sources,
            domain_scope,
            tuple(sorted(excluded)),
            self.ann_calls,
            None if hits else "no_candidates",
        )
        self._batches.append(batch)
        for hit in hits:
            self._edges.append(
                ProposalEdge(
                    probe_id=probe_id,
                    candidate_id=hit.memory_id,
                    stage=stage,
                    rank=hit.rank,
                    score=hit.score,
                    source_memory_ids=sources,
                    target_id=target_id,
                    premise_ids=canonical_premises,
                )
            )
        observe("proposal", batch, event="proposal_completed")
        return batch

    def retrieve_dense(self) -> ProposalBatch:
        """Run (or return) the one original-query retrieval only.

        Dense baselines call this method and therefore never pay for bridge
        expansion that they do not use.
        """

        if self._dense_batch is None:
            self._dense_batch = self._run_probe(
                stage="initial_dense",
                probe_text=self.query,
                width=min(self.initial_width, len(self.memories)),
                instruction=self.query_instruction,
            )
        return self._dense_batch

    # Useful compatibility spelling for callers that think in stages.
    dense = retrieve_dense

    def build_initial_pool(self, *, expand: bool = True) -> InitialCandidatePool:
        """Return the dense/bridge union in deterministic discovery order."""

        if not isinstance(expand, bool):
            raise ValueError("expand must be boolean")
        if self._initial_pool is not None:
            if not expand:
                # Once expansion has run, returning only the dense subset
                # would misrepresent the retriever's frozen pool.  Dense
                # callers should use ``retrieve_dense`` directly.
                raise ValueError("cannot request an unexpanded pool after expansion")
            return self._initial_pool

        dense = self.retrieve_dense()
        expansion_batches: list[ProposalBatch] = []
        ordered_ids: list[str] = []
        seen: set[str] = set()

        def admit(values: Iterable[str]) -> None:
            for identifier in values:
                if identifier not in seen:
                    seen.add(identifier)
                    ordered_ids.append(identifier)

        admit(dense.ids)
        stop_reason = dense.stop_reason if dense.budget_exhausted else None
        if not expand:
            return InitialCandidatePool(
                candidate_ids=tuple(ordered_ids),
                dense_batch=dense,
                expansion_batches=(),
                provenance=tuple(self._edges),
                ann_calls=self.ann_calls,
                stop_reason=stop_reason,
            )
        if expand and self.initial_expansion_width > 0:
            for seed_id in dense.ids:
                seed = self.memory_by_id[seed_id]
                batch = self._run_probe(
                    stage="initial_bridge",
                    probe_text=bridge_probe_text(self.query, seed),
                    width=min(self.initial_expansion_width, max(0, len(self.memories) - 1)),
                    exclude=(seed_id,),
                    source_memory_ids=(seed_id,),
                    instruction=self.proposal_instruction,
                )
                expansion_batches.append(batch)
                admit(batch.ids)
                if batch.budget_exhausted:
                    stop_reason = "ann_budget_exhausted"
                    break

        self._initial_pool = InitialCandidatePool(
            candidate_ids=tuple(ordered_ids),
            dense_batch=dense,
            expansion_batches=tuple(expansion_batches),
            provenance=tuple(self._edges),
            ann_calls=self.ann_calls,
            stop_reason=stop_reason,
        )
        return self._initial_pool

    # Compatibility spellings used by experiment adapters and tests.
    initial_pool = build_initial_pool
    retrieve_initial = build_initial_pool

    def propose(
        self,
        target_id: str,
        premise_ids: Sequence[str] = (),
        *,
        width: int | None = None,
        fixed_pool: bool | None = None,
    ) -> ProposalBatch:
        """Retrieve conditions for one immutable ``(target, premises)`` state."""

        target = str(target_id)
        premises = tuple(sorted({str(value) for value in premise_ids}))
        if target not in self.memory_by_id:
            raise ValueError(f"unknown target memory ID: {target}")
        if target in premises:
            raise ValueError("the fixed target cannot also be a premise")
        unknown = set(premises).difference(self.memory_by_id)
        if unknown:
            raise ValueError(f"unknown premise memory IDs: {sorted(unknown)}")
        use_fixed_pool = self.fixed_pool if fixed_pool is None else fixed_pool
        if not isinstance(use_fixed_pool, bool):
            raise ValueError("fixed_pool must be boolean")
        if self._initial_pool is None:
            self.build_initial_pool(expand=True)
        allowed = self.initial_candidate_ids if use_fixed_pool else None
        records = [self.memory_by_id[identifier] for identifier in premises]
        text = conditional_probe_text(self.query, self.memory_by_id[target], records)
        if self.information_needs:
            text += "\n\nInformation needed to answer the query:\n" + json.dumps(
                self.information_needs, ensure_ascii=False, sort_keys=True
            )
        return self._run_probe(
            stage="conditional",
            probe_text=text,
            width=self.proposal_width if width is None else width,
            exclude=(target, *premises),
            allowed_ids=allowed,
            source_memory_ids=(target, *premises),
            target_id=target,
            premise_ids=premises,
            instruction=self.proposal_instruction,
        )

    propose_conditions = propose

    def require_proposal(
        self,
        target_id: str,
        premise_ids: Sequence[str] = (),
        **kwargs: Any,
    ) -> ProposalBatch:
        """Strict variant for callers that prefer an exception at the ANN cap."""

        batch = self.propose(target_id, premise_ids, **kwargs)
        if batch.budget_exhausted:
            raise RetrievalBudgetExhausted("conditional proposal ANN budget exhausted")
        return batch

    def public_dict(self) -> dict[str, Any]:
        """Return all proposal attempts and edges accumulated so far."""

        return {
            "visible_memory_ids": list(self.visible_ids),
            "initial_candidate_ids": list(self.initial_candidate_ids),
            "ann_calls": self.ann_calls,
            "max_ann_calls": self.max_ann_calls,
            "remaining_ann_calls": self.remaining_ann_calls,
            "fixed_pool": self.fixed_pool,
            "proposal_batches": [batch.public_dict() for batch in self._batches],
            "source_graph": [edge.public_dict() for edge in self._edges],
        }


def retrieve_initial_candidates(
    query: str,
    memories: Sequence[Memory | Mapping[str, Any]],
    embedder: DependencyEmbedder,
    **kwargs: Any,
) -> tuple[DependencyRetriever, InitialCandidatePool]:
    """Convenience constructor used by small experiments and offline tests."""

    retriever = DependencyRetriever(query, memories, embedder, **kwargs)
    return retriever, retriever.build_initial_pool()


__all__ = [
    "DEPENDENCY_PROPOSAL_INSTRUCTION",
    "DependencyEmbedder",
    "DependencyRetriever",
    "InitialCandidatePool",
    "ProposalBatch",
    "ProposalEdge",
    "ProposalHit",
    "RetrievalBudgetExhausted",
    "bridge_probe_text",
    "conditional_probe_text",
    "retrieve_initial_candidates",
]
