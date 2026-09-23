"""Bounded end-to-end root tie sensitivity; no generation and no new policy."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .clients import RerankerClient, build_context_plan, build_embedder
from .dependency_retrieval import DEPENDENCY_PROPOSAL_INSTRUCTION, DependencyRetriever
from .dependency_search import DependencySearcher, DynamicBundleSelector
from .diagnostic_config import digest
from .diagnostic_runner import (_execute_item, _phase_budget, _phase_summary, _records,
                                atomic_json, load_manifest, make_scorer, plan_summary,
                                run_lock, validate_online)
from .root_tie_diagnostics import compare_root_tie_runs
from .diagnostic_observability import observe
from .request_audit import request_audit_scope


def root_trial_plan(manifest: dict) -> list[dict]:
    result = []
    for case in manifest["cases"]:
        for seed in [None, *manifest["root_seeds"]]:
            mode = "legacy_lexical" if seed is None else "seeded_hash"
            result.append({"item_id": digest([manifest["manifest_id"], "root", case["question_id"], mode, seed]),
                           "question_id": case["question_id"], "root_tie_break": mode, "root_tie_seed": seed})
    return result


def run_root_diagnostics(root: str | Path, config_path: str | Path, *, execute: bool = False,
                         embedder: Any = None, reranker: Any = None) -> dict:
    root = Path(root)
    manifest = load_manifest(root)
    trials = root_trial_plan(manifest)
    if len(trials) > manifest["budgets"]["root_runs"]:
        raise ValueError("root run plan exceeds frozen budget")
    if not execute:
        return {**plan_summary(manifest), "phase": "root", "execute": False, "root_trials": trials,
                "generation_calls": 0,
                "max_ann_calls_total": len(trials) * manifest["dependency"]["max_ann_calls"],
                "unique_set_limits_per_trial": {k: manifest["dependency"][k]
                    for k in ("max_scored_sets", "max_selection_sets")}}
    try:
        config = validate_online(manifest, config_path, ("embedding", "reranker"))
    except Exception as exc:
        atomic_json(root / "root_gate.json", {"status": "blocked_before_network", "error_type": type(exc).__name__,
                    "reason": str(exc), "network_calls": 0, "manifest_id": manifest["manifest_id"]})
        raise
    with run_lock(root):
        atomic_json(root / "root_trial_plan.json", {"manifest_id": manifest["manifest_id"], "trials": trials})
        budget = _phase_budget(root, manifest, "root")
        embedding = embedder if embedder is not None else build_embedder(config.models.embedding, device=config.runtime.device)
        ranker = reranker if reranker is not None else RerankerClient(config.models.reranker)
        cases = {c["question_id"]: c for c in manifest["cases"]}
        # A complete successful bank is shared across variants. Query/proposal
        # calls and logical score budgets remain per trial. No hidden ANN cache.
        vectors = {}
        outcomes = []
        for trial in trials:
            case = cases[trial["question_id"]]
            def action(case=case, trial=trial):
                records = _records(case)
                memories = list(records.values())
                retriever = searcher = selector = archive = scorer = None
                settings = config.dependency
                try:
                    if case["question_id"] not in vectors:
                        with request_audit_scope({"stage": "memory_embedding"}):
                            vectors[case["question_id"]] = embedding.encode([m.text for m in memories])
                    retriever = DependencyRetriever(case["query"], memories, embedding,
                        memory_vectors=vectors[case["question_id"]], initial_width=settings.initial_width,
                        initial_expansion_width=settings.initial_expansion_width, proposal_width=settings.proposal_width,
                        max_ann_calls=settings.max_ann_calls, fixed_pool=False,
                        query_instruction=config.models.embedding.query_instruction,
                        proposal_instruction=DEPENDENCY_PROPOSAL_INSTRUCTION)
                    with request_audit_scope({"stage": "initial_retrieval"}):
                        initial = retriever.build_initial_pool(expand=True)
                    scorer = make_scorer(case, config, ranker, root / "cache" / "root_scores", manifest["manifest_id"])
                    # Original reranker batch width is preserved for the search
                    # experiment, unlike the singleton frozen 28-score probe.
                    scorer.batch_size = settings.reranker_batch_size
                    searcher = DependencySearcher(scorer, retriever, signal="activation",
                        pair_rescue_width=settings.pair_rescue_width, fixed_pool=False,
                        max_scored_sets=settings.max_scored_sets,
                        root_tie_break=trial["root_tie_break"], root_tie_seed=trial["root_tie_seed"])
                    with request_audit_scope({"stage": "dependency_search"}):
                        archive = searcher.run(initial)
                    def feasible(ids):
                        plan = build_context_plan(case["query"], [records[i] for i in ids], case["all_options"],
                                                  generator_config=config.models.generator, strict=False, selected_ids=ids)
                        return {"feasible": plan.within_budget, "reason": plan.budget_status}
                    selector = DynamicBundleSelector(scorer, max_selection_sets=settings.max_selection_sets,
                                                      generation_feasible=feasible)
                    with request_audit_scope({"stage": "selection"}):
                        selected = selector.select(archive)
                    final_plan = build_context_plan(case["query"], [records[i] for i in selected.selected_ids],
                        case["all_options"], generator_config=config.models.generator, selected_ids=selected.selected_ids)
                    observe("context", "root_final_context", selected_ids=list(selected.selected_ids),
                            final_memory_count=len(selected.selected_ids), context_hash=final_plan.context_hash,
                            context_within_budget=final_plan.within_budget,
                            generator_input_tokens_estimate=final_plan.token_count,
                            token_count_is_estimate=final_plan.token_count_is_estimate,
                            context_budget=final_plan.budget, context_budget_status=final_plan.budget_status)
                    return {"search": archive.public_dict(), "selection": selected.public_dict(),
                            "root_trace": searcher.root_tie_diagnostics(selected.selected_ids),
                            "context_plan": final_plan.public_dict(), "generation_calls": 0,
                            "scorer_events": list(scorer.events)}
                except Exception as exc:
                    partial = {"generation_calls": 0}
                    if searcher is not None:
                        partial["root_trace"] = searcher.root_tie_diagnostics()
                        partial["search"] = (archive.public_dict() if archive is not None else
                            searcher.partial_public_dict(stop_reason="execution_error", detail=type(exc).__name__))
                    if selector is not None:
                        partial["selection"] = selector.partial_public_dict(stop_reason="execution_error", detail=type(exc).__name__)
                    if retriever is not None:
                        partial["retrieval"] = retriever.public_dict()
                    if scorer is not None:
                        partial["scorer_events"] = list(scorer.events)
                    exc.diagnostic_partial_artifacts = partial
                    raise
            outcomes.append(_execute_item(root, manifest, "root", trial["item_id"], budget, action, trial))
        comparisons = []
        for qid in cases:
            rows = [o for o in outcomes if o["question_id"] == qid]
            baseline = next(o for o in rows if o["root_tie_seed"] is None)
            def trace(value):
                return value.get("root_trace", value.get("partial_artifacts", {}).get("root_trace", {}))
            for variant in rows:
                if variant["root_tie_seed"] is None:
                    continue
                comparison = compare_root_tie_runs(trace(baseline), trace(variant))
                comparison.update(question_id=qid, seed=variant["root_tie_seed"],
                                  other_protocols_verified=True, baseline_status=baseline["status"], variant_status=variant["status"])
                comparisons.append(comparison)
        result = {**_phase_summary(manifest, "root", outcomes, budget), "comparisons": comparisons,
                  "generation_calls": 0, "all_seeds_reported": True,
                  "timing_note": "Shared warmed caches make wall time order dependent; logical budgets remain matched."}
        atomic_json(root / "root_summary.json", result)
        return result
