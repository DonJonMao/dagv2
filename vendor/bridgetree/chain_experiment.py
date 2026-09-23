"""Fixed full-run planning and denominator-safe outcome bookkeeping."""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .chain_judge import HTTPJointJudge, PublicQuery, preflight_joint_backend
from .chain_search import ChainSearcher, EvidenceState, Terminal, choose_terminal
from .chain_support import close_support
from .clients import RerankItem, build_context_plan
from .config import AppConfig
from .experiment import retrieve_method
from .index import ExactInnerProductIndex
from .math_utils import normalize_rows
from .metrics import answer_accuracy
from .personamem import PersonaMemExample, messages_to_memories
from .types import ContextPlan, Memory

DEFAULT_CHAIN_METHODS = (
    "dense", "dense_rerank", "rfmem", "semantic_s2", "chain_h1_no_closure",
    "chain_h2_no_closure", "chain_full", "chain_no_join", "chain_dense_pool",
)


@dataclass(frozen=True)
class ChainTask:
    dataset_revision: str
    split: str
    persona_id: str
    question_id: str
    method_id: str
    config_hash: str

    @property
    def key(self) -> tuple[str, ...]:
        return (self.dataset_revision, self.split, self.persona_id,
                self.question_id, self.method_id, self.config_hash)


@dataclass(frozen=True)
class Outcome:
    task: ChainTask
    status: str
    correct: bool | None = None
    prediction: str | None = None
    error: str | None = None


class _TransportEmbedder:
    """Embedding adapter that preserves an explicitly injected HTTP seam."""

    def __init__(self, config: Any, transport: Any):
        self.config = config
        self.transport = transport
        self.fingerprint = hashlib.sha256(
            json.dumps(
                {"endpoint": config.endpoint, "model": config.model},
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise ValueError("embedding texts must be a sequence")
        values = list(texts)
        if any(not isinstance(text, str) for text in values):
            raise ValueError("embedding texts must contain strings")
        if not values:
            return np.empty((0, 0), dtype=np.float64)
        rows: list[list[float]] = []
        dimension: int | None = None
        batch_size = int(self.config.batch_size)
        for start in range(0, len(values), batch_size):
            batch = values[start : start + batch_size]
            payload: dict[str, Any] = {"input": batch}
            if self.config.model:
                payload["model"] = self.config.model
            response = self.transport(
                self.config.endpoint,
                payload,
                self.config.timeout_seconds,
                headers=None,
            )
            data = response.get("data") if isinstance(response, Mapping) else None
            if not isinstance(data, list) or len(data) != len(batch):
                raise ValueError("embedding response must cover the complete batch")
            indexed: dict[int, list[float]] = {}
            for item in data:
                if not isinstance(item, Mapping) or "index" not in item or "embedding" not in item:
                    raise ValueError("embedding response item is incomplete")
                index = item["index"]
                if isinstance(index, bool) or not isinstance(index, int):
                    raise ValueError("embedding response index must be an integer")
                if index < 0 or index >= len(batch) or index in indexed:
                    raise ValueError("embedding response contains an unknown or duplicate index")
                vector = item["embedding"]
                if isinstance(vector, (str, bytes)) or not isinstance(vector, Sequence):
                    raise ValueError("embedding response vector must be a sequence")
                try:
                    numeric = [float(value) for value in vector]
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError("embedding response vector must be numeric") from exc
                if not numeric or not np.all(np.isfinite(numeric)):
                    raise ValueError("embedding response vector must be non-empty and finite")
                if dimension is None:
                    dimension = len(numeric)
                elif len(numeric) != dimension:
                    raise ValueError("embedding response dimensions are inconsistent")
                indexed[index] = numeric
            if set(indexed) != set(range(len(batch))):
                raise ValueError("embedding response indices must cover the complete batch")
            rows.extend(indexed[index] for index in range(len(batch)))
        return normalize_rows(np.asarray(rows, dtype=np.float64))

    def encode_queries(
        self,
        texts: Sequence[str],
        instruction: str | None = None,
        **_: Any,
    ) -> np.ndarray:
        prefix = self.config.query_instruction if instruction is None else instruction
        if not isinstance(prefix, str):
            raise ValueError("embedding query instruction must be a string")
        return self.encode([prefix + text for text in texts])

    def encode_query(self, text: str, instruction: str | None = None, **kwargs: Any) -> np.ndarray:
        return self.encode_queries([text], instruction=instruction, **kwargs)[0]


def _declares_truncation(value: Any) -> bool:
    flags = {
        "truncated",
        "is_truncated",
        "was_truncated",
        "input_truncated",
        "inputs_truncated",
        "document_truncated",
        "documents_truncated",
    }
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in flags and (
                child is True
                or isinstance(child, str)
                and child.strip().lower() in {"true", "yes", "1"}
                or isinstance(child, int)
                and not isinstance(child, bool)
                and child == 1
            ):
                return True
            if isinstance(child, (Mapping, list, tuple)) and _declares_truncation(child):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_declares_truncation(child) for child in value)
    return False


class _TransportReranker:
    """Pointwise reranker compatible with :func:`retrieve_method`."""

    def __init__(self, config: Any, transport: Any):
        self.config = config
        self.transport = transport
        self.score_space = config.score_space
        self.score_contract = config.score_contract
        self.model_fingerprint = str(config.model or config.endpoint)

    def rerank(self, query: str, documents: Sequence[str], top_n: int) -> list[RerankItem]:
        if self.score_contract != "pointwise":
            raise ValueError("legacy Chain requires a pointwise reranker")
        if not isinstance(query, str):
            raise ValueError("rerank query must be a string")
        if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence):
            raise ValueError("rerank documents must be a sequence")
        docs = list(documents)
        if any(not isinstance(document, str) for document in docs):
            raise ValueError("rerank documents must contain strings")
        if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 0:
            raise ValueError("top_n must be a non-negative integer")
        if not docs or top_n == 0:
            return []
        payload: dict[str, Any] = {
            "query": query,
            "documents": docs,
            "top_n": min(top_n, len(docs)),
            "return_documents": False,
        }
        if self.config.model:
            payload["model"] = self.config.model
        response = self.transport(
            self.config.endpoint,
            payload,
            self.config.timeout_seconds,
            headers=None,
        )
        if _declares_truncation(response):
            raise ValueError("rerank backend explicitly reported input truncation")
        if isinstance(response, Mapping):
            raw_results = response.get("results", response.get("data"))
        elif isinstance(response, list):
            raw_results = response
        else:
            raw_results = None
        if not isinstance(raw_results, list):
            raise ValueError("rerank response must contain results or data")
        seen: set[int] = set()
        items: list[RerankItem] = []
        for raw in raw_results:
            if not isinstance(raw, Mapping) or "index" not in raw:
                raise ValueError("rerank response item must contain index and score")
            index = raw["index"]
            if isinstance(index, bool) or not isinstance(index, int):
                raise ValueError("rerank response index must be an integer")
            if index < 0 or index >= len(docs) or index in seen:
                raise ValueError("rerank response contains an unknown or duplicate index")
            score_value = raw.get("relevance_score", raw.get("score"))
            try:
                score = float(score_value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("rerank response score must be numeric") from exc
            if not np.isfinite(score):
                raise ValueError("rerank response score must be finite")
            if self.score_space == "unit_interval" and not 0.0 <= score <= 1.0:
                raise ValueError("unit-interval rerank score is outside [0, 1]")
            seen.add(index)
            items.append(RerankItem(index, score))
        if top_n >= len(docs) and seen != set(range(len(docs))):
            raise ValueError("rerank_all response must cover every document")
        return sorted(items, key=lambda item: (-item.score, item.index))[:top_n]

    def rerank_all(self, query: str, documents: Sequence[str]) -> list[RerankItem]:
        return self.rerank(query, documents, len(documents))


def _diagnostic_value(value: Any) -> Any:
    """Convert legacy retrieval diagnostics to a persistable representation."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "snapshot") and callable(value.snapshot):
        return _diagnostic_value(asdict(value.snapshot()))
    if hasattr(value, "__dataclass_fields__"):
        return _diagnostic_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _diagnostic_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_diagnostic_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class ChainTaskExecutor:
    """Execute one task through the historical Chain comparison matrix.

    This class intentionally remains separate from ``DependencyTaskExecutor``:
    old experiment callers retain their nine method IDs, while ``chain-run``
    continues to use the conditional-activation implementation explicitly.
    """

    _CHAIN_METHODS = frozenset(
        {
            "chain_h1_no_closure",
            "chain_h2_no_closure",
            "chain_full",
            "chain_no_join",
            "chain_dense_pool",
        }
    )

    def __init__(
        self,
        config: AppConfig,
        *,
        transport: Any | None = None,
        backend: str = "http",
    ):
        if not isinstance(config, AppConfig):
            raise ValueError("ChainTaskExecutor requires an AppConfig")
        config.validate()
        if not isinstance(backend, str) or not backend.strip():
            raise ValueError("ChainTaskExecutor backend must be a non-empty string")
        if backend == "fixture" and transport is None:
            raise ValueError("fixture backend requires an explicit fixture transport")
        if transport is None:
            from .clients import _post_json

            transport = _post_json
        if not callable(transport):
            raise ValueError("ChainTaskExecutor transport must be callable")
        self.config = config
        self.transport = transport
        self.backend = backend.strip()
        self.embedder = _TransportEmbedder(config.models.embedding, transport)
        self.reranker = _TransportReranker(config.models.reranker, transport)
        self._joint_preflight_complete = False

    def _memories(self, example: PersonaMemExample) -> list[Memory]:
        if isinstance(example.end_index, bool) or not isinstance(example.end_index, int):
            raise ValueError("PersonaMem end_index must be an integer")
        if example.end_index < 0:
            raise ValueError("PersonaMem end_index must be non-negative")
        visible_messages = list(example.messages[: example.end_index])
        source = example.shared_context_id or f"{example.persona_id}:{example.question_id}"
        return messages_to_memories(
            visible_messages,
            source,
            include_system_persona=self.config.data.include_system_persona,
            memory_granularity=self.config.data.memory_granularity,
        )

    def _public_query(
        self,
        task: ChainTask,
        example: PersonaMemExample,
        memories: Sequence[Memory],
    ) -> PublicQuery:
        temporal: list[tuple[str, str]] = []
        if example.query_time is not None:
            temporal.append(("query_time", str(example.query_time)))
        metadata = example.metadata if isinstance(example.metadata, Mapping) else {}
        for key in ("query_date", "cutoff", "time"):
            if key in metadata and metadata[key] is not None:
                temporal.append(
                    (
                        key,
                        json.dumps(metadata[key], ensure_ascii=False, sort_keys=True, default=str),
                    )
                )
        return PublicQuery(
            dataset_revision=task.dataset_revision,
            persona_id=example.persona_id,
            question_id=example.question_id,
            query=example.query,
            public_options=example.all_options,
            shared_context_id=example.shared_context_id,
            end_index=example.end_index,
            visible_memories=tuple(memory.memory_id for memory in memories),
            time_metadata=tuple(temporal),
        )

    def _standard_retrieve(
        self,
        method: str,
        example: PersonaMemExample,
        memories: Sequence[Memory],
        query_vector: np.ndarray,
        memory_vectors: np.ndarray,
        index: ExactInnerProductIndex | None,
    ) -> tuple[list[str], dict[str, Any]]:
        routed = "semantic_path" if method == "semantic_s2" else method
        selected_ids, _selected, diagnostics, _result = retrieve_method(
            routed,
            self.config,
            example,
            memories,
            query_vector,
            memory_vectors,
            reranker=self.reranker,
            index=index,
            proposal_query_provider=self.embedder,
            bridge_query_instruction=self.config.bridge_rerank.bridge_query_instruction,
        )
        public_diagnostics = _diagnostic_value(diagnostics)
        if not isinstance(public_diagnostics, dict):
            public_diagnostics = {"detail": public_diagnostics}
        public_diagnostics["routed_method"] = routed
        return [str(identifier) for identifier in selected_ids], public_diagnostics

    def _close_candidates(
        self,
        searcher: ChainSearcher,
        judge: HTTPJointJudge,
        query: PublicQuery,
        observed: Sequence[Terminal],
    ) -> tuple[Terminal | None, dict[str, Any]]:
        claim_calls = 0
        verify_calls = 0
        verified: list[Terminal] = []
        records: list[dict[str, Any]] = []
        ranked = sorted(
            observed,
            key=lambda item: (-item.score.log_u, len(item.selected_ids), item.selected_ids),
        )
        for terminal in ranked:
            if claim_calls >= self.config.chain.max_claim_calls:
                break
            remaining = self.config.chain.max_verify_calls - verify_calls
            if remaining <= 0:
                break
            result = close_support(
                query,
                terminal.state.raw_ids,
                judge,
                max_verify_calls=remaining,
            )
            claim_calls += int(result.claim is not None)
            consumed = len(result.deletion_trace) + int(result.initial_verification is not None)
            verify_calls += consumed
            records.append(
                {
                    "original_ids": list(result.original_ids),
                    "retained_ids": list(result.retained_ids),
                    "status": result.status,
                    "verify_calls": consumed,
                }
            )
            if result.initial_verification and result.initial_verification.supported:
                # A completed closure is an additional terminal; it must not
                # overwrite the independently scored supported original.
                verified.append(terminal)
                if result.status == "single_deletion_minimal":
                    try:
                        closed_score = searcher.score(result.retained_ids)
                    except RuntimeError as exc:
                        if str(exc) != "joint budget exhausted":
                            raise
                    else:
                        verified.append(Terminal(terminal.state, closed_score, result))
        return choose_terminal(verified or observed), {
            "claim_calls": claim_calls,
            "verify_calls": verify_calls,
            "closures": records,
            "verified_terminal_count": len(verified),
        }

    def _chain_retrieve(
        self,
        method: str,
        task: ChainTask,
        example: PersonaMemExample,
        memories: Sequence[Memory],
        query_vector: np.ndarray,
        memory_vectors: np.ndarray,
        index: ExactInnerProductIndex,
    ) -> tuple[list[str], dict[str, Any]]:
        chain = self.config.chain
        hits = index.search(query_vector, min(chain.initial_width, len(memories)))
        initial_ids = [identifier for identifier, _score in hits]
        roots = [
            EvidenceState((identifier,), paths=((identifier,),), expandable_ids=(identifier,))
            for identifier in initial_ids
        ]
        query = self._public_query(task, example, memories)
        judge = HTTPJointJudge(
            self.config.models.generator,
            {memory.memory_id: memory for memory in memories},
            transport=self.transport,
            backend=self.backend,
            cache_dir=Path(self.config.runtime.cache_dir) / "legacy_chain_joint",
            repeat_id=task.dataset_revision,
        )
        if not self._joint_preflight_complete:
            preflight_joint_backend(judge)
            self._joint_preflight_complete = True

        proposal_calls = 0
        admitted = set(initial_ids)
        memory_by_id = {memory.memory_id: memory for memory in memories}
        dense_pool = set(initial_ids)

        def proposals(state: EvidenceState) -> Sequence[str]:
            nonlocal proposal_calls
            if proposal_calls >= chain.max_proposal_calls:
                return ()
            if len(admitted) >= chain.max_unique_memories:
                return ()
            if method == "chain_dense_pool":
                return tuple(identifier for identifier in initial_ids if identifier not in state.raw_ids)
            anchors = state.expandable_ids or state.raw_ids[-1:]
            result: list[str] = []
            for anchor_id in anchors:
                if proposal_calls >= chain.max_proposal_calls:
                    break
                if anchor_id not in memory_by_id:
                    continue
                proposal_calls += 1
                probe = (
                    "Original question:\n"
                    + example.query
                    + "\n\nEvidence endpoint:\n"
                    + memory_by_id[anchor_id].text
                )
                vector = self.embedder.encode_query(
                    probe,
                    instruction=self.config.models.embedding.query_instruction,
                )
                available = chain.max_unique_memories - len(admitted)
                request = min(chain.proposal_width, available)
                for identifier, _score in index.search(vector, request, exclude=state.raw_ids):
                    if identifier in admitted and identifier not in dense_pool:
                        # Reusing a candidate in a different state is valid;
                        # the admission cap counts unique records, not probes.
                        pass
                    admitted.add(identifier)
                    result.append(identifier)
                    if len(result) >= chain.proposal_width or len(admitted) >= chain.max_unique_memories:
                        break
            return tuple(dict.fromkeys(result))

        horizon = {
            "chain_h1_no_closure": 1,
            "chain_h2_no_closure": 2,
        }.get(method, chain.horizon)
        join_enabled = method != "chain_no_join"
        searcher = ChainSearcher(
            query,
            judge,
            horizon=horizon,
            max_joint_contexts=chain.max_joint_contexts,
            max_verify_calls=chain.max_verify_calls,
        )
        archive = searcher.search(
            roots,
            proposals=proposals,
            join_pool=roots if join_enabled else (),
            closure=False,
        )
        closure_enabled = method not in {"chain_h1_no_closure", "chain_h2_no_closure"}
        closure_diagnostics: dict[str, Any] = {
            "claim_calls": 0,
            "verify_calls": 0,
            "closures": [],
            "verified_terminal_count": 0,
        }
        if closure_enabled:
            selected, closure_diagnostics = self._close_candidates(
                searcher,
                judge,
                query,
                archive.observed_terminals,
            )
        else:
            selected = choose_terminal(archive.observed_terminals)
        selected_ids = list(selected.selected_ids) if selected is not None else []
        return selected_ids, {
            "routed_method": method,
            "horizon": horizon,
            "closure_enabled": closure_enabled,
            "join_enabled": join_enabled,
            "fixed_dense_pool": method == "chain_dense_pool",
            "initial_ids": initial_ids,
            "proposal_calls": proposal_calls,
            "admitted_ids": sorted(admitted),
            "observed_terminal_count": len(archive.observed_terminals),
            "open_state_count": len(archive.open_states),
            "joint_cost": _diagnostic_value(judge.cost),
            **closure_diagnostics,
        }

    def _answer(self, plan: ContextPlan) -> str:
        if not plan.within_budget:
            raise ValueError("legacy Chain selected context exceeds the generator budget")
        payload = plan.request_dict()
        payload.pop("endpoint", None)
        payload.pop("endpoint_sha256", None)
        key = self.config.models.generator.resolved_api_key()
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        response = self.transport(
            self.config.models.generator.endpoint,
            payload,
            self.config.models.generator.timeout_seconds,
            headers=headers,
        )
        choices = response.get("choices") if isinstance(response, Mapping) else None
        if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or not choices:
            raise ValueError("generator response has no choices")
        try:
            content = choices[0]["message"]["content"]
        except (KeyError, TypeError, IndexError) as exc:
            raise ValueError("generator response choice is invalid") from exc
        if not isinstance(content, str) or not content.strip():
            raise ValueError("generator response content is empty")
        return content.strip()

    def execute(self, task: ChainTask, example: PersonaMemExample) -> dict[str, Any]:
        if not isinstance(task, ChainTask):
            raise ValueError("execute requires a ChainTask")
        if not isinstance(example, PersonaMemExample):
            raise ValueError("execute requires a PersonaMemExample")
        if task.method_id not in DEFAULT_CHAIN_METHODS:
            raise ValueError(f"unsupported Chain method: {task.method_id}")
        if task.config_hash != self.config.config_hash():
            raise ValueError("task config_hash does not match the executor configuration")
        if task.persona_id != example.persona_id or task.question_id != example.question_id:
            raise ValueError("task identity does not match the PersonaMem example")

        memories = self._memories(example)
        if memories:
            memory_vectors = self.embedder.encode([memory.text for memory in memories])
            query_vector = self.embedder.encode_query(example.query)
            index: ExactInnerProductIndex | None = ExactInnerProductIndex(
                [memory.memory_id for memory in memories],
                memory_vectors,
                exclusion_margin=self.config.retrieval.faiss_exclusion_margin,
            )
        else:
            memory_vectors = np.empty((0, 0), dtype=np.float64)
            query_vector = np.empty((0,), dtype=np.float64)
            index = None

        if not memories:
            selected_ids, diagnostics = [], {
                "routed_method": "semantic_path" if task.method_id == "semantic_s2" else task.method_id,
                "empty_bank": True,
            }
        elif task.method_id in self._CHAIN_METHODS:
            if index is None:
                selected_ids, diagnostics = [], {"routed_method": task.method_id, "empty_bank": True}
            else:
                selected_ids, diagnostics = self._chain_retrieve(
                    task.method_id,
                    task,
                    example,
                    memories,
                    query_vector,
                    memory_vectors,
                    index,
                )
        else:
            selected_ids, diagnostics = self._standard_retrieve(
                task.method_id,
                example,
                memories,
                query_vector,
                memory_vectors,
                index,
            )
        memory_by_id = {memory.memory_id: memory for memory in memories}
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError("legacy retrieval returned duplicate memory IDs")
        unknown = sorted(set(selected_ids) - set(memory_by_id))
        if unknown:
            raise ValueError(f"legacy retrieval returned unknown memory IDs: {unknown}")
        selected_memories = [memory_by_id[identifier] for identifier in selected_ids]
        plan = build_context_plan(
            example.query,
            selected_memories,
            example.all_options,
            token_budget=self.config.models.generator.context_token_budget,
            strict=True,
            selected_ids=selected_ids,
            generator_config=self.config.models.generator,
        )
        prediction = self._answer(plan)
        return {
            "task": asdict(task),
            "status": "success",
            "prediction": prediction,
            "correct": bool(answer_accuracy(prediction, example.correct_answer)),
            "backend": self.backend,
            "selected_ids": selected_ids,
            "context_hash": plan.context_hash,
            "context_plan": plan.public_dict(),
            "diagnostics": diagnostics,
        }


def build_full_plan(queries: Iterable[Mapping[str, object]], *, dataset_revision: str,
                    config_hash: str, methods: Sequence[str] = DEFAULT_CHAIN_METHODS) -> list[ChainTask]:
    """Freeze all task keys before execution, preserving input split/persona order."""
    result: list[ChainTask] = []
    for query in queries:
        split = str(query.get("split", "confirmation"))
        persona = str(query["persona_id"])
        question = str(query["question_id"])
        result.extend(ChainTask(dataset_revision, split, persona, question, method, config_hash)
                      for method in methods)
    return result


def write_plan(path: str | Path, tasks: Sequence[ChainTask]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(asdict(task), ensure_ascii=False, sort_keys=True) + "\n")


def summarize_outcomes(tasks: Sequence[ChainTask], outcomes: Sequence[Outcome]) -> dict[str, object]:
    expected = len(tasks)
    by_key = {outcome.task.key: outcome for outcome in outcomes}
    completed = [item for item in by_key.values() if item.status in {"success", "error"}]
    correct = sum(1 for item in completed if item.correct is True)
    failed = sum(1 for item in completed if item.status == "error")
    return {
        "expected_tasks": expected,
        "completed_tasks": len(completed),
        "pending_tasks": max(0, expected - len(completed)),
        "correct": correct,
        "failed": failed,
        "coverage": len(completed) / expected if expected else 1.0,
        "completed_accuracy": correct / len(completed) if completed else None,
        "final_accuracy": correct / expected if expected and len(completed) == expected else None,
    }


def write_metrics(path: str | Path, rows: Sequence[Mapping[str, object]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
