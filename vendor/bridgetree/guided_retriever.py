from __future__ import annotations

import time
from typing import Callable, Dict, List, Sequence

import numpy as np

from .budget import CostTracker
from .clients import RerankerClient, RerankItem
from .clustering import cluster_siblings
from .config import AppConfig
from .index import ExactInnerProductIndex
from .personamem import PersonaMemExample
from .ranking import (
    RerankCache,
    build_bridge_embedding_text,
    build_path_filter_query,
    build_personamem_rank_query,
    format_memory_document,
    format_path_document,
    stable_union,
)
from .types import GuidedCandidatePool, Memory

EmbedQueryBatch = Callable[..., np.ndarray]


def cached_rerank_all(
    reranker: RerankerClient,
    rerank_cache: RerankCache | None,
    query: str,
    documents: Sequence[str],
    tracker: CostTracker,
    **cache_kwargs,
) -> tuple[List[RerankItem], bool]:
    if not documents:
        return [], True
    if rerank_cache is not None:
        items, cache_hit, elapsed_ms = rerank_cache.rerank_all(
            reranker, query, documents, **cache_kwargs
        )
    else:
        started = time.perf_counter()
        method = getattr(reranker, "rerank_all", None)
        items = method(query, documents) if callable(method) else reranker.rerank(query, documents, len(documents))
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        cache_hit = False
    tracker.record_rerank(len(documents), elapsed_ms)
    if cache_hit:
        tracker.record_cache_hit()
    return sorted(items, key=lambda item: (-item.score, item.index)), cache_hit


def _score_map(ids: Sequence[str], ranking: Sequence[RerankItem]) -> Dict[str, float]:
    return {ids[item.index]: item.score for item in ranking}


class RerankerGuidedBridgeRetriever:
    """Use BridgeTree only to discover candidates; a task reranker owns final selection."""

    def __init__(self, config: AppConfig):
        config.validate()
        self.config = config

    def retrieve(
        self,
        example: PersonaMemExample,
        memories: Sequence[Memory],
        query_vector: np.ndarray,
        memory_vectors: np.ndarray,
        *,
        embed_query_batch: EmbedQueryBatch,
        reranker: RerankerClient,
        rerank_cache: RerankCache | None,
        index: ExactInnerProductIndex,
        cost_tracker: CostTracker,
        mode: str,
    ) -> tuple[List[str], List[Memory], GuidedCandidatePool]:
        if mode not in {"bridgetree_guided_rerank", "bridgetree_guided_pathfilter"}:
            raise ValueError(f"unsupported guided retrieval mode: {mode}")
        if self.config.retrieval.stop_mode == "certificate_or_budget":
            raise ValueError("reranker-guided selection cannot use certificate_or_budget")
        bridge = self.config.bridge_rerank
        use_path_filter = mode == "bridgetree_guided_pathfilter" and bridge.path_filter
        memory_by_id = {memory.memory_id: memory for memory in memories}
        max_timestamp = max((memory.timestamp for memory in memories), default=0.0)

        dense_hits = cost_tracker.search_core(
            index,
            query_vector,
            min(bridge.dense_pool_width, len(memories)),
        )
        dense_ids = [memory_id for memory_id, _score in dense_hits]
        rank_query = build_personamem_rank_query(
            example,
            instruction=bridge.final_rerank_instruction,
            use_answer_options=bridge.use_answer_options,
        )
        dense_documents = [
            format_memory_document(
                memory_by_id[memory_id],
                max_timestamp,
                include_time_metadata=bridge.include_time_metadata,
            )
            for memory_id in dense_ids
        ]
        dense_ranking, dense_cache_hit = cached_rerank_all(
            reranker,
            rerank_cache,
            rank_query,
            dense_documents,
            cost_tracker,
            records=[memory_by_id[memory_id] for memory_id in dense_ids],
            cutoff=getattr(example, "query_time", None),
            query_metadata=getattr(example, "metadata", None),
            answer_options=example.all_options,
            include_time_metadata=bridge.include_time_metadata,
            score_contract=getattr(reranker, "score_contract", None),
            task_instruction=bridge.final_rerank_instruction,
        )
        dense_scores = _score_map(dense_ids, dense_ranking)
        dense_ranked_ids = [dense_ids[item.index] for item in dense_ranking]
        anchor_ids = dense_ranked_ids[: min(bridge.anchor_width, len(dense_ranked_ids))]

        ordered_anchors = sorted(anchor_ids, key=lambda memory_id: (-dense_scores[memory_id], memory_id))
        anchor_vectors = (
            np.vstack([index.vector(memory_id) for memory_id in ordered_anchors]) if ordered_anchors else []
        )
        clusters = (
            cluster_siblings(
                anchor_vectors,
                list(range(len(ordered_anchors), 0, -1)),
                mode=self.config.retrieval.cluster_mode,
                fixed_count=self.config.retrieval.cluster_count,
                max_clusters=self.config.retrieval.max_clusters,
                min_cluster_size=self.config.retrieval.min_cluster_size,
            )
            if ordered_anchors
            else []
        )
        branches = []
        for cluster_index, cluster in enumerate(clusters):
            member_ids = [ordered_anchors[position] for position in cluster.member_positions]
            anchor_id = min(member_ids, key=lambda memory_id: (-dense_scores[memory_id], memory_id))
            branches.append((anchor_id, cluster_index, cluster, member_ids))
        branches.sort(key=lambda item: (-dense_scores[item[0]], item[0], item[1]))
        branches = branches[: bridge.expand_branch_count]

        bridge_texts = [
            build_bridge_embedding_text(example.query, memory_by_id[anchor_id]) for anchor_id, *_ in branches
        ]
        if bridge.probe_mode == "query_anchor" and bridge_texts:
            embedding_started = time.perf_counter()
            probe_vectors = np.asarray(
                embed_query_batch(
                    bridge_texts,
                    instruction=bridge.bridge_query_instruction,
                    purpose="bridge_query",
                )
            )
            embedding_ms = (time.perf_counter() - embedding_started) * 1000.0
            cost_tracker.record_bridge_embedding(len(bridge_texts), embedding_ms)
        else:
            probe_vectors = np.vstack([item[2].probe for item in branches]) if branches else np.empty((0, 0))

        bridge_raw_ids: List[str] = []
        bridge_kept_ids: List[str] = []
        parent_by_bridge_id: Dict[str, str] = {}
        branch_by_bridge_id: Dict[str, str] = {}
        raw_by_branch: Dict[str, List[str]] = {}
        ann_scores: Dict[str, float] = {}
        path_scores: Dict[str, float] = {}
        path_cache_hits = 0
        proposed = set(dense_ids)
        for branch_position, ((anchor_id, cluster_index, _cluster, _members), probe) in enumerate(
            zip(branches, probe_vectors)
        ):
            branch_id = f"guided_b{cluster_index:04d}_{branch_position:02d}"
            hits = cost_tracker.search_core(
                index,
                probe,
                min(bridge.branch_overfetch_width, len(memories)),
                exclude=proposed,
            )
            raw_ids = [memory_id for memory_id, _score in hits]
            raw_by_branch[branch_id] = raw_ids
            for memory_id, score in hits:
                proposed.add(memory_id)
                bridge_raw_ids.append(memory_id)
                parent_by_bridge_id[memory_id] = anchor_id
                branch_by_bridge_id[memory_id] = branch_id
                ann_scores[memory_id] = score

            if use_path_filter and raw_ids:
                path_query = build_path_filter_query(
                    example,
                    instruction=bridge.path_filter_instruction,
                    use_answer_options=bridge.use_answer_options,
                )
                path_documents = [
                    format_path_document(
                        memory_by_id[anchor_id],
                        memory_by_id[memory_id],
                        max_timestamp,
                        include_time_metadata=bridge.include_time_metadata,
                    )
                    for memory_id in raw_ids
                ]
                path_ranking, cache_hit = cached_rerank_all(
                    reranker,
                    rerank_cache,
                    path_query,
                    path_documents,
                    cost_tracker,
                    records=[
                        {
                            "anchor": memory_by_id[anchor_id],
                            "candidate": memory_by_id[memory_id],
                        }
                        for memory_id in raw_ids
                    ],
                    cutoff=getattr(example, "query_time", None),
                    query_metadata=getattr(example, "metadata", None),
                    answer_options=example.all_options,
                    include_time_metadata=bridge.include_time_metadata,
                    score_contract=getattr(reranker, "score_contract", None),
                    task_instruction=bridge.path_filter_instruction,
                )
                path_cache_hits += int(cache_hit)
                path_scores.update(_score_map(raw_ids, path_ranking))
                kept = [raw_ids[item.index] for item in path_ranking[: bridge.branch_keep_width]]
            else:
                kept = raw_ids[: bridge.branch_keep_width]
            bridge_kept_ids.extend(kept)

        candidate_ids = stable_union(dense_ids, bridge_kept_ids)
        candidate_documents = [
            format_memory_document(
                memory_by_id[memory_id],
                max_timestamp,
                include_time_metadata=bridge.include_time_metadata,
            )
            for memory_id in candidate_ids
        ]
        final_ranking, final_cache_hit = cached_rerank_all(
            reranker,
            rerank_cache,
            rank_query,
            candidate_documents,
            cost_tracker,
            records=[memory_by_id[memory_id] for memory_id in candidate_ids],
            cutoff=getattr(example, "query_time", None),
            query_metadata=getattr(example, "metadata", None),
            answer_options=example.all_options,
            include_time_metadata=bridge.include_time_metadata,
            score_contract=getattr(reranker, "score_contract", None),
            task_instruction=bridge.final_rerank_instruction,
        )
        final_scores = _score_map(candidate_ids, final_ranking)
        selected_ids = [candidate_ids[item.index] for item in final_ranking[: self.config.retrieval.context_size]]
        selected_memories = sorted(
            (memory_by_id[memory_id] for memory_id in selected_ids),
            key=lambda memory: (memory.timestamp, memory.memory_id),
        )
        bridge_id_set = set(bridge_kept_ids)
        selected_bridge_count = sum(memory_id in bridge_id_set for memory_id in selected_ids)
        dense_top5 = set(dense_ranked_ids[: self.config.retrieval.context_size])
        dense_retained = sum(memory_id in dense_top5 for memory_id in selected_ids)
        selected_source = {
            memory_id: "bridge" if memory_id in bridge_id_set else "dense" for memory_id in selected_ids
        }
        cost_tracker.set_stop_reason("frontier_empty")
        pool = GuidedCandidatePool(
            dense_ids=dense_ids,
            anchor_ids=anchor_ids,
            bridge_raw_ids=bridge_raw_ids,
            bridge_kept_ids=bridge_kept_ids,
            candidate_ids=candidate_ids,
            parent_by_bridge_id=parent_by_bridge_id,
            branch_by_bridge_id=branch_by_bridge_id,
            rerank_scores=final_scores,
            diagnostics={
                "dense_pool_ids": dense_ids,
                "anchor_ids": anchor_ids,
                "bridge_raw_ids": bridge_raw_ids,
                "bridge_kept_ids": bridge_kept_ids,
                "candidate_union_ids": candidate_ids,
                "parent_by_bridge_id": parent_by_bridge_id,
                "branch_by_bridge_id": branch_by_bridge_id,
                "raw_bridge_ids_by_branch": raw_by_branch,
                "selected_source_by_id": selected_source,
                "selected_bridge_count": selected_bridge_count,
                "selected_bridge_rate": selected_bridge_count / len(selected_ids) if selected_ids else 0.0,
                "dense_rerank_top5_retention": dense_retained / len(dense_top5) if dense_top5 else 0.0,
                "bridge_candidate_novelty": (
                    len(set(bridge_kept_ids) - set(dense_ids)) / len(set(bridge_kept_ids))
                    if bridge_kept_ids
                    else 0.0
                ),
                "dense_rerank_top_ids": dense_ranked_ids[: self.config.retrieval.context_size],
                "dense_rerank_scores": dense_scores,
                "bridge_ann_scores": ann_scores,
                "path_filter_scores": path_scores,
                "final_rerank_scores": final_scores,
                "rerank_cache_hits": int(dense_cache_hit) + path_cache_hits + int(final_cache_hit),
                "bridge_embedding_calls": cost_tracker.bridge_embedding_calls,
            },
        )
        return selected_ids, selected_memories, pool
