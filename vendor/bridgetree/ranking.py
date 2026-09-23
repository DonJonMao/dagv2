from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Sequence

import numpy as np

from .clients import RerankerClient, RerankItem
from .personamem import PersonaMemExample, parse_options
from .types import Memory

DEFAULT_FINAL_RERANK_INSTRUCTION = (
    "Assess whether this recorded interaction provides evidence useful for answering the question about this user. "
    "Respect the time or period asked in the question. Earlier statements may be necessary for historical states "
    "or reasons for change. Distinguish user statements from assistant suggestions. Judge usefulness of the "
    "recorded evidence, not agreement with a guessed answer."
)
DEFAULT_PATH_FILTER_INSTRUCTION = (
    "Rank candidate interactions by whether they add distinct, independently usable personal evidence beyond "
    "the anchor for answering the current request. Avoid anchor paraphrases and merely topical passages."
)


def _option_text(value: str) -> str:
    return re.sub(r"^\s*[\(\[]?[A-Za-z][\)\]\.:-]\s*", "", value).strip()


def _formatted_options(example: PersonaMemExample) -> str:
    options = parse_options(example.all_options)
    return "\n".join(
        f"{chr(ord('A') + index)}. {_option_text(option)}" if index < 26 else f"{index + 1}. {_option_text(option)}"
        for index, option in enumerate(options)
    )


def build_personamem_rank_query(
    example: PersonaMemExample,
    *,
    instruction: str = DEFAULT_FINAL_RERANK_INSTRUCTION,
    use_answer_options: bool = True,
) -> str:
    """Build the shared task query without ever exposing the gold answer field."""
    sections = [instruction.strip(), f"Current request:\n{example.query}"]
    if use_answer_options:
        sections.append(f"Candidate answers:\n{_formatted_options(example)}")
    return "\n\n".join(sections)


def format_memory_document(
    memory: Memory,
    max_timestamp: float,
    *,
    include_time_metadata: bool = True,
) -> str:
    if not include_time_metadata:
        return memory.text
    roles = memory.metadata.get("roles", ())
    role_text = ", ".join(str(role) for role in roles) if roles else "unknown"
    current = int(memory.timestamp) if float(memory.timestamp).is_integer() else memory.timestamp
    maximum = int(max_timestamp) if float(max_timestamp).is_integer() else max_timestamp
    return (
        f"[Interaction index: {current} / {maximum}]\n"
        f"[Roles: {role_text}]\n"
        f"{memory.text}"
    )


def build_bridge_embedding_text(query: str, anchor: Memory) -> str:
    return (
        f"Current request:\n{query}\n\n"
        f"Already retrieved anchor interaction:\n{anchor.text}\n\n"
        "Retrieve another past interaction that supplies distinct, complementary personal evidence for the request. "
        "Avoid paraphrases of the anchor and passages that are only topically related."
    )


def build_path_filter_query(
    example: PersonaMemExample,
    *,
    instruction: str = DEFAULT_PATH_FILTER_INSTRUCTION,
    use_answer_options: bool = True,
) -> str:
    sections = [instruction.strip(), f"Current request:\n{example.query}"]
    if use_answer_options:
        sections.append(f"Candidate answers:\n{_formatted_options(example)}")
    return "\n\n".join(sections)


def format_path_document(
    anchor: Memory,
    candidate: Memory,
    max_timestamp: float,
    *,
    include_time_metadata: bool = True,
) -> str:
    return (
        "Anchor interaction:\n"
        f"{format_memory_document(anchor, max_timestamp, include_time_metadata=include_time_metadata)}\n\n"
        "Candidate interaction:\n"
        f"{format_memory_document(candidate, max_timestamp, include_time_metadata=include_time_metadata)}"
    )


def stable_union(*groups: Iterable[str]) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for group in groups:
        for value in group:
            if value not in seen:
                seen.add(value)
                result.append(value)
    return result


class RerankCache:
    """Content-addressed, order-sensitive cache of complete reranker rankings."""

    def __init__(
        self,
        root: str | Path,
        *,
        endpoint: str,
        model: str,
        score_space: str = "unit_interval",
        task_instruction: str = "",
        score_contract: str = "pointwise",
        model_fingerprint: str = "",
        validate_scores: bool = False,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.endpoint = endpoint
        self.model = model
        self.score_space = str(score_space)
        self.task_instruction = str(task_instruction)
        self.score_contract = str(score_contract)
        self.model_fingerprint = str(model_fingerprint or model)
        # Legacy guided experiments historically accepted arbitrary relevance
        # scores.  Semantic quality adapters opt into strict range checking;
        # retaining the default keeps the cache source-compatible while the
        # actual RerankerClient still enforces its declared contract.
        self.validate_scores = bool(validate_scores)

    @staticmethod
    def _jsonable(value: Any) -> Any:
        if isinstance(value, Memory):
            return {
                "memory_id": str(value.memory_id),
                "text": str(value.text),
                "timestamp": value.timestamp,
                "source_id": str(value.source_id),
                "metadata": RerankCache._jsonable(value.metadata),
            }
        if isinstance(value, Mapping):
            return {str(key): RerankCache._jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [RerankCache._jsonable(item) for item in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer, np.floating, np.bool_)):
            return value.item()
        return value

    def _provenance_rows(
        self,
        documents: Sequence[Any],
        *,
        records: Mapping[str, Any] | Sequence[Any] | None = None,
        memory_metadata: Mapping[str, Any] | Sequence[Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return order-preserving full provenance for every rerank input.

        ``include_time_metadata=False`` controls only the rendered document;
        it must not remove provenance from the cache identity.  This prevents
        two equal-text memories with different source/time metadata from
        sharing a score.
        """
        docs = list(documents)
        if records is None and all(isinstance(item, Memory) for item in docs):
            records = docs
        record_rows: list[Any] = []
        if records is not None:
            if isinstance(records, Mapping):
                for item in docs:
                    identifier = str(getattr(item, "memory_id", "")) if isinstance(item, Memory) else ""
                    if not identifier and isinstance(item, str):
                        # A document string has no intrinsic ID; mapping order
                        # is the only compatible alignment in this legacy form.
                        identifier = str(len(record_rows))
                    record_rows.append(records.get(identifier, records.get(str(len(record_rows)))))
            else:
                record_rows = list(records)
            if len(record_rows) != len(docs):
                raise ValueError("rerank records and documents must have equal length")
        metadata_rows: list[Any] = []
        if memory_metadata is not None:
            if isinstance(memory_metadata, Mapping):
                for index, item in enumerate(docs):
                    identifier = str(getattr(item, "memory_id", index))
                    metadata_rows.append(memory_metadata.get(identifier, memory_metadata.get(str(index), {})))
            else:
                metadata_rows = list(memory_metadata)
            if len(metadata_rows) != len(docs):
                raise ValueError("memory_metadata and documents must have equal length")
        rows: list[dict[str, Any]] = []
        for index, document in enumerate(docs):
            record = record_rows[index] if record_rows else (document if isinstance(document, Memory) else None)
            if isinstance(record, Memory):
                row = {
                    "memory_id": str(record.memory_id),
                    "text": str(record.text),
                    "timestamp": record.timestamp,
                    "source_id": str(record.source_id),
                    "metadata": record.metadata,
                }
            elif isinstance(record, Mapping):
                row = {str(key): value for key, value in record.items()}
            elif record is not None:
                row = {"record": str(record)}
            else:
                row = {}
            row["document"] = str(document.text if isinstance(document, Memory) else document)
            if metadata_rows:
                row["memory_metadata"] = metadata_rows[index]
            rows.append(self._jsonable(row))
        return rows

    def key_for(
        self,
        query: str,
        documents: Sequence[Any],
        *,
        records: Mapping[str, Any] | Sequence[Any] | None = None,
        memory_metadata: Mapping[str, Any] | Sequence[Any] | None = None,
        cutoff: Any = None,
        query_metadata: Mapping[str, Any] | None = None,
        answer_options: str = "",
        include_time_metadata: bool | None = None,
        score_contract: str | None = None,
        task_instruction: str | None = None,
    ) -> str:
        payload = json.dumps(
            {
                "schema": 2,
                "endpoint": self.endpoint,
                "model": self.model,
                "model_fingerprint": self.model_fingerprint,
                "score_space": self.score_space,
                "score_contract": score_contract or self.score_contract,
                "task_instruction": (
                    self.task_instruction if task_instruction is None else str(task_instruction)
                ),
                "query": query,
                "documents": list(documents),
                "provenance": self._provenance_rows(
                    documents, records=records, memory_metadata=memory_metadata
                ),
                "cutoff": cutoff,
                "query_metadata": query_metadata or {},
                "answer_options": answer_options,
                "include_time_metadata": include_time_metadata,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _path(self, query: str, documents: Sequence[Any], **kwargs: Any) -> Path:
        return self.root / f"{self.key_for(query, documents, **kwargs)}.json"

    @staticmethod
    def _validate_items(
        items: Sequence[RerankItem],
        document_count: int,
        score_space: str,
        *,
        validate_range: bool = False,
    ) -> list[RerankItem]:
        if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
            raise ValueError("rerank results must be a sequence")
        if len(items) != document_count:
            raise ValueError("rerank results must contain one item per document")
        seen: set[int] = set()
        normalized_space = str(score_space).strip().lower().replace("-", "_")
        result: list[RerankItem] = []
        for item in items:
            try:
                index = int(item.index)
                score = float(item.score)
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError("rerank result item is invalid") from exc
            if index < 0 or index >= document_count or index in seen or not np.isfinite(score):
                raise ValueError("rerank results contain an unknown, duplicate, or non-finite item")
            if (
                validate_range
                and normalized_space in {"unit_interval", "probability", "sigmoid", "unit"}
                and not 0.0 <= score <= 1.0
            ):
                raise ValueError("unit-interval rerank score is outside [0, 1]")
            seen.add(index)
            result.append(RerankItem(index=index, score=score))
        if seen != set(range(document_count)):
            raise ValueError("rerank results omit one or more document indices")
        return sorted(result, key=lambda item: (-item.score, item.index))

    def get(self, query: str, documents: Sequence[Any], **kwargs: Any) -> List[RerankItem] | None:
        path = self._path(query, documents, **kwargs)
        if not path.is_file():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            items = [RerankItem(index=int(item["index"]), score=float(item["score"])) for item in raw["items"]]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
        try:
            return self._validate_items(
                items, len(documents), self.score_space, validate_range=self.validate_scores
            )
        except ValueError:
            return None

    def put(self, query: str, documents: Sequence[Any], items: Sequence[RerankItem], **kwargs: Any) -> None:
        ranked = self._validate_items(
            items, len(documents), self.score_space, validate_range=self.validate_scores
        )
        path = self._path(query, documents, **kwargs)
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        payload = {
            "items": [
                {"index": item.index, "score": item.score}
                for item in ranked
            ]
        }
        temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)

    def rerank_all(
        self,
        client: RerankerClient,
        query: str,
        documents: Sequence[Any],
        **kwargs: Any,
    ) -> tuple[List[RerankItem], bool, float]:
        if not documents:
            return [], True, 0.0
        cached = self.get(query, documents, **kwargs)
        if cached is not None:
            return cached, True, 0.0
        started = time.perf_counter()
        method = getattr(client, "rerank_all", None)
        items = method(query, documents) if callable(method) else client.rerank(query, documents, len(documents))
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        ranked = self._validate_items(items, len(documents), self.score_space)
        self.put(query, documents, ranked, **kwargs)
        return ranked, False, elapsed_ms
