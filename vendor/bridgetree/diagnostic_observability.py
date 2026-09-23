"""Opt-in, task-local module observations; never a scoring or policy input.

The disabled path returns before serializing records. Enabled observations
contain identifiers, counts and measured scores only, not model payloads.
Each event is synchronously flushed to the combined and module-specific log;
an audit failure aborts execution rather than silently losing evidence.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable, Mapping
import uuid

from .request_audit import AuditWriteError, current_audit_scope


MODULES = frozenset({"proposal", "scoring", "activation", "state", "selection", "stop", "context", "execution",
                     "planner", "scheduler", "target", "archive", "evidence", "feedback"})
_FIELDS = frozenset({
    "schema_version", "event", "event_id", "at_epoch", "sequence", "module", "run_identity",
    "task_id", "persona_id", "question_id", "method_id", "diagnostic_id", "case_id",
    "item_id", "trial_id", "subset_id", "condition_id", "repeat_index", "root_tie_seed",
    "root_tie_break", "failed", "planned", "success", "type", "request_hash",
    "execution_attempt", "task_attempt", "attempt", "repetition", "phase", "stage", "scope",
    "operation", "status", "cause_type", "status_code", "http_status", "elapsed_ms", "retryable", "partial",
    "probe_id", "target_id", "premise_ids", "source_memory_ids", "candidate_id", "candidate_ids",
    "hits", "memory_id", "score", "rank", "domain_scope", "excluded_ids", "ann_call_index",
    "ann_calls", "max_ann_calls", "width", "available_count", "initial_target_ids", "initial_ids",
    "initial_candidate_ids", "initial_pool_ids", "visible_memory_count", "fixed_pool",
    "edge_type", "dependency_claim", "proposal_sources", "reason", "detail", "stop_reason",
    "ids", "set_ids", "sets", "cache_key", "source", "cache_hit", "batch_index", "batch_size",
    "batch_documents", "logical_call_index", "estimated_input_tokens", "token_count_is_estimate",
    "logical_unique_sets_charged", "objective_semantics", "utility_validation_id", "score_contract", "score_space",
    "group_ids", "group_kind", "P", "Pe", "PG", "PGe", "activation", "target_marginal_before",
    "target_marginal_after", "context_marginal", "signal_kind", "signal", "accepted", "queued",
    "successor", "state", "state_event", "identity", "bundle_ids", "target_ids", "archive_reasons",
    "pop_index", "premise_depth", "priority", "is_zero_priority_root", "root_tie_rank",
    "ann_calls_before", "ann_calls_after", "scored_sets_before", "scored_sets_after", "completed",
    "proposed_count", "singleton_tests", "pair_tests", "positive_successors", "new_successors",
    "initial_ann_calls", "final_ann_calls", "initial_scored_sets", "final_scored_sets",
    "visited_state_count", "archived_bundle_count", "frontier_count", "global_certificate",
    "round", "current_ids", "comparisons", "union_ids", "added_ids", "feasible", "base_score",
    "combined_score", "marginal", "accepted_bundle_ids", "selected_ids_after", "selected_ids",
    "complete", "frozen_bundle_ids", "context_hash", "context_budget", "context_within_budget",
    "generator_input_tokens_estimate", "final_memory_count", "evaluation_performed", "correct",
    "parse_failed", "generator_calls", "context_budget_status", "score_reason",
    "deployment_fingerprint", "request_id", "logical_call_id", "transport_budget_used",
    "transport_budget_max", "model_calls", "error_type", "infrastructure_failure",
    "root_order", "root_selection", "coverage_roots", "exploration_roots", "reserved_sets", "full_score_cap",
    "phase_score_cap", "quantum_score_cap", "ann_cap", "action", "lane", "resumed",
    "measurement_cursor", "measurement_cursor_before", "remaining_measurements", "measurements_completed",
    "consecutive_quanta", "decision", "pivot_depth", "speculative_depth", "old_target_id", "new_target_id",
    "replacement_ids", "removed_ids", "target_path", "pivot_count", "speculative_count", "pending_workspaces",
    "requirement_id", "requirement_ids", "requirements_count", "mapping_count", "mapped_candidate_count",
    "candidate_count", "covered_count", "missing_count", "ambiguous_count", "partial_count", "coverage_status",
    "missing_requirement_ids", "covered_requirement_ids", "quote_start", "quote_end", "role", "evidence_kind",
    "support_ids", "mapping_ids", "chunk_id", "chunk_start", "chunk_end", "input_tokens_estimate",
    "output_tokens_estimate", "llm_calls", "call_index", "response_hash", "prompt_hash", "validation_status",
    "repair_index", "revision", "feedback_round", "new_candidate_ids", "added_ids", "token_count", "budget",
    "reasoning_elapsed_ms", "map_batch", "method_version",
    "record_kind", "event_index", "requirements", "requirements_hash", "query_hash", "id", "necessary",
    "input_token_budget", "max_llm_calls", "evidence_llm_calls", "repairs_used", "revisions_used",
    "character_count", "unit_ids", "memory_ids", "ranges", "complete_raw_coverage", "mappings",
    "evidence_id", "relation", "kind", "quote_verified", "quote_occurrences", "start", "end",
    "source_message_indices", "observed_order", "time_metadata", "unit_id", "coverage", "previous_coverage",
    "observed_start", "observed_end", "event_start", "event_end", "validity", "time_source",
    "coverage_basis", "evidence_ids", "supporting_ids", "round_index", "selected_before",
    "generation_feasibility", "missing_requirements", "returned_ids", "new_ids", "duplicate_ids",
    "costs", "diagnostics", "evidence_json_repairs", "evidence_input_tokens_estimate",
    "evidence_output_tokens_estimate", "evidence_llm_elapsed_ms", "evidence_elapsed_ms",
    "evidence_calls_by_operation", "evidence_candidates", "evidence_mapped_candidates",
    "evidence_plan", "evidence_map", "evidence_select", "evidence_plan_repair", "evidence_map_repair",
    "evidence_select_repair", "token_count_is_estimate",
    "evidence_unmapped_candidates", "evidence_units", "evidence_verified_mappings", "evidence_requirements",
    "evidence_covered_requirements", "evidence_partial_requirements", "evidence_missing_requirements",
    "evidence_ambiguous_requirements", "evidence_validation_failures", "evidence_selection_revisions",
    "evidence_feedback_rounds", "evidence_selected_count", "evidence_coverage_is_model_judgement",
})
_TOKEN = re.compile(r"[\w.:/@+\-]{0,512}\Z", re.ASCII)
_SECRET = re.compile(r"https?://|Bearer\s|sk-[A-Za-z0-9_-]{8,}", re.I)
_REASONS = frozenset({
    "", "search", "activation", "selection", "selection_round", "dense_rerank", "diagnostic_fresh_all",
    "finite_frontier_exhausted", "no_initial_candidates", "score_budget_exhausted", "ann_budget_exhausted",
    "input_capacity", "reranker_input_capacity", "generator_input_capacity", "infeasible",
    "no_candidates", "successors_already_seen", "no_positive_signal", "no_positive_marginal",
    "positive_successors_queued", "positive_successors_queued_with_input_capacity_skips",
    "archive_exhausted", "not_scored_incomplete_round", "ranked_candidates_exhausted",
    "execution_error", "infrastructure_failure", "retry_budget_exhausted", "transport_budget_exhausted",
    "audit_failure", "interrupted",
    "measured_P", "measured_Pe", "measured_PG", "measured_PGe", "completed_score_snapshot",
    "positive_conditional_target", "state_already_seen", "nonpositive_activation", "speculative_depth_limit",
    "speculative_state_limit", "bounded_negative_target_exploration", "pivot_limit", "pivot_depth_limit",
    "target_cycle", "target_replaced", "external_discovery", "mid_search_release", "quantum_measurement_limit",
    "quantum_set_limit", "reserved_score_limit", "workspace_completed", "within_budget",
    "fairness_release", "no_alternative_target", "requirements_covered", "necessary_requirements_covered",
    "unresolved_without_feedback",
    "feedback_round_budget_exhausted", "feedback_no_new_candidates", "planning_error",
})
_INTERPRETATION = "Measured reranker interactions are not causal proof or answer-accuracy improvement."
_CURRENT: ContextVar[Callable[[Mapping[str, Any]], None] | None] = ContextVar(
    "bridgetree_module_observer", default=None,
)


def _sanitize(value: Any, field: str = "") -> Any:
    if isinstance(value, Mapping):
        return {key: _sanitize(item, key) for key, item in value.items() if isinstance(key, str) and key in _FIELDS}
    if isinstance(value, (tuple, list)):
        return [_sanitize(item, field) for item in value]
    if isinstance(value, str):
        # IDs and enum-like reasons are sufficient. Free-form messages can
        # contain excerpts of requests or server errors, so are never logged.
        if field in {"reason", "detail", "stop_reason", "score_reason"}:
            reason = value.split(":", 1)[0]
            return reason if reason in _REASONS else "[redacted]"
        return value if _TOKEN.fullmatch(value) and not _SECRET.search(value) else "[redacted]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return "[redacted]"


class ModuleEventRecorder:
    """Append immediately to ``RUN/modules/events.jsonl`` and ``MODULE.jsonl``.

    The two files are not a transaction: if either write fails, the caller
    receives AuditWriteError and must stop. Event IDs allow reconciling a
    partially written pair after a crash. Sequence numbers are recorder-local;
    event IDs remain unique across resumes. No observer changes model payloads.
    """

    def __init__(self, run_dir: str | Path, *, run_identity: str = "",
                 metadata: Mapping[str, Any] | None = None, durable: bool = True):
        self.directory = Path(run_dir) / "modules"
        self.path = self.directory / "events.jsonl"
        self.run_identity = run_identity
        self.metadata = _sanitize(metadata or {})
        self.durable = durable
        self._lock = threading.Lock()
        self._sequence = 0

    def __call__(self, event: Mapping[str, Any]) -> None:
        try:
            value = _sanitize({**self.metadata, **event})
            module = value.get("module")
            if module not in MODULES:
                raise ValueError("unknown observation module")
            value["run_identity"] = _sanitize(
                self.run_identity or event.get("run_identity") or self.metadata.get("run_identity", ""),
            )
            value["interpretation"] = _INTERPRETATION
            with self._lock:
                self._sequence += 1
                value["sequence"] = self._sequence
                line = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
                self.directory.mkdir(parents=True, exist_ok=True)
                for path in (self.path, self.directory / f"{module}.jsonl"):
                    with path.open("a", encoding="utf-8") as stream:
                        stream.write(line)
                        stream.flush()
                        if self.durable:
                            os.fsync(stream.fileno())
        except (OSError, ValueError, TypeError) as exc:
            raise AuditWriteError("could not persist module observation") from exc


@contextmanager
def observation_scope(recorder: Callable[[Mapping[str, Any]], None] | None):
    """Bind a recorder in this context; None explicitly disables a parent."""
    token = _CURRENT.set(recorder)
    try:
        yield recorder
    finally:
        _CURRENT.reset(token)


def observe(module: str, event_or_record: Any, **fields: Any) -> None:
    """Observe a named event, mapping, or existing public_dict record lazily."""
    recorder = _CURRENT.get()
    if recorder is None:
        return
    try:
        if module not in MODULES:
            raise ValueError("unknown observation module")
        if isinstance(event_or_record, str):
            record = {"event": event_or_record}
        elif isinstance(event_or_record, Mapping):
            record = event_or_record
        else:
            record = event_or_record.public_dict()
        value = _sanitize({**current_audit_scope().metadata, **record, **fields,
                           "schema_version": 1, "module": module,
                           "event_id": uuid.uuid4().hex, "at_epoch": time.time()})
        value.setdefault("event", "observation")
        value.setdefault("run_identity", "")
        value["interpretation"] = _INTERPRETATION
        recorder(value)
    except AuditWriteError:
        raise
    except Exception as exc:
        raise AuditWriteError("module observation failed") from exc
