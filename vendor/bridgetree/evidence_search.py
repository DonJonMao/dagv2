"""Budgeted multi-target evidence search with revisable target identities.

R remains frozen query relevance.  Scheduling, retaining a target, and
archiving measured evidence are separate decisions; none asserts answer
utility.  A workspace owns its proposal and measurement cursor, so yielding
never repeats an ANN query and ANN exhaustion does not discard pending work.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from typing import Any, Iterable, Sequence

import numpy as np

from .dependency_retrieval import InitialCandidatePool, ProposalBatch
from .dependency_search import (
    ActivationRecord,
    DependencyState,
    EvidenceBundle,
    SearchArchive,
    SearchStateRecord,
    SkippedMeasurement,
    _canonical_ids,
    _preflight_and_score,
    _scored_sets,
    _ScoringStop,
    _set_cumulative_budget,
    _stable_unique,
)
from .diagnostic_observability import observe
from .evidence_config import EvidenceSearchConfig


@dataclass(frozen=True)
class EvidenceSearchArchive(SearchArchive):
    scheduler_events: tuple[dict[str, Any], ...] = ()
    target_events: tuple[dict[str, Any], ...] = ()
    measured_sets: tuple[dict[str, Any], ...] = ()
    pending_workspaces: tuple[dict[str, Any], ...] = ()
    target_costs: tuple[dict[str, Any], ...] = ()
    root_order: tuple[str, ...] = ()
    incomplete_measurements: tuple[dict[str, Any], ...] = ()

    def public_dict(self) -> dict[str, Any]:
        return {
            **super().public_dict(),
            "method_version": "evidence_bridge_v1",
            "scheduler_events": list(self.scheduler_events),
            "target_events": list(self.target_events),
            "measured_sets": list(self.measured_sets),
            "pending_workspaces": list(self.pending_workspaces),
            "target_costs": list(self.target_costs),
            "root_order": list(self.root_order),
            "incomplete_measurements": list(self.incomplete_measurements),
        }


@dataclass
class _Workspace:
    state: DependencyState
    lane: str
    priority: float
    sequence: int
    target_path: tuple[str, ...]
    pivot_depth: int = 0
    speculative_depth: int = 0
    parent_identity: tuple[str, tuple[str, ...]] | None = None
    proposal: ProposalBatch | None = None
    candidate_ids: tuple[str, ...] = ()
    groups: tuple[tuple[str, ...], ...] = ()
    cursor: int = 0
    started: bool = False
    done: bool = False
    singleton_tests: int = 0
    pair_tests: int = 0
    positive_successors: int = 0
    new_successors: int = 0
    blocked_seen: int | None = None
    blocked_cap: int | None = None

    def public_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.public_dict(),
            "lane": self.lane,
            "priority": self.priority,
            "target_path": list(self.target_path),
            "pivot_depth": self.pivot_depth,
            "speculative_depth": self.speculative_depth,
            "parent_identity": self.parent_identity,
            "started": self.started,
            "completed": self.done,
            "measurement_cursor": self.cursor,
            "remaining_measurements": len(self.groups) - self.cursor,
            "singleton_tests": self.singleton_tests,
            "pair_tests": self.pair_tests,
            "proposal_id": None if self.proposal is None else self.proposal.probe_id,
            "candidate_ids": list(self.candidate_ids),
            "pending_group_ids": list(self.groups[self.cursor]) if self.cursor < len(self.groups) else [],
            "blocked_at_scored_sets": self.blocked_seen,
            "blocked_cap": self.blocked_cap,
        }


class EvidenceBridgeSearcher:
    def __init__(
        self,
        scorer: Any,
        retriever: Any,
        *,
        settings: EvidenceSearchConfig,
        max_scored_sets: int,
        pair_rescue_width: int,
        max_ann_calls: int | None = None,
    ):
        if not isinstance(settings, EvidenceSearchConfig):
            raise TypeError("settings must be EvidenceSearchConfig")
        for name, value in (("max_scored_sets", max_scored_sets), ("pair_rescue_width", pair_rescue_width)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if max_ann_calls is not None and (
            isinstance(max_ann_calls, bool) or not isinstance(max_ann_calls, int) or max_ann_calls < 0
        ):
            raise ValueError("max_ann_calls must be a nonnegative absolute ANN cap")
        self.scorer, self.retriever, self.settings = scorer, retriever, settings
        self.max_scored_sets, self.pair_rescue_width = max_scored_sets, pair_rescue_width
        limits = [x for x in (max_ann_calls, getattr(retriever, "max_ann_calls", None)) if x is not None]
        self.ann_cap = min(limits) if limits else None
        self._initialized = False

    def _emit(self, module: str, event: str, **values: Any) -> dict[str, Any]:
        value = {"event": event, **values}
        (self.scheduler_events if module == "scheduler" else self.target_events).append(value)
        observe(module, value)
        return value

    def _ann_available(self) -> bool:
        return self.ann_cap is None or self.retriever.ann_calls < self.ann_cap

    def _root_order(self, initial_ids: tuple[str, ...], first_id: str | None) -> tuple[str, ...]:
        if self.settings.root_selection == "discovery" or len(initial_ids) < 2:
            return initial_ids
        vectors = getattr(self.retriever, "memory_vectors", None)
        visible = tuple(getattr(self.retriever, "visible_ids", ()))
        if vectors is None or not visible:
            raise ValueError("diverse root selection requires public memory_vectors and visible_ids")
        matrix = np.asarray(vectors, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != len(visible) or not np.all(np.isfinite(matrix)):
            raise ValueError("invalid visible memory vectors for diverse root selection")
        lookup = {identifier: rank for rank, identifier in enumerate(visible)}
        selected = [first_id if first_id in initial_ids else initial_ids[0]]
        rows = {identifier: matrix[lookup[identifier]] for identifier in initial_ids}
        norms = {identifier: float(np.linalg.norm(row)) for identifier, row in rows.items()}

        def similarity(left: str, right: str) -> float:
            divisor = norms[left] * norms[right]
            return float(np.dot(rows[left], rows[right]) / divisor) if divisor else 0.0

        rank = {identifier: index for index, identifier in enumerate(initial_ids)}
        while len(selected) < len(initial_ids):
            remaining = [identifier for identifier in initial_ids if identifier not in selected]
            # Minimize similarity to the nearest chosen representative. Original
            # retrieval order, then ID, breaks exact ties deterministically.
            chosen = min(
                remaining,
                key=lambda identifier: (
                    max(similarity(identifier, old) for old in selected),
                    rank[identifier],
                    identifier,
                ),
            )
            selected.append(chosen)
        return tuple(selected)

    def _archive(self, ids: Iterable[str], reason: str, target: str | None = None, score: float | None = None) -> None:
        key = _canonical_ids(ids)
        if score is not None:
            existing = self.measured.get(key)
            if existing is not None and existing["score"] != float(score):
                raise ValueError("one frozen set received inconsistent scores")
            if existing is None:
                self.measured[key] = {
                    "ids": list(key),
                    "score": float(score),
                    "objective_semantics": "legacy_query_relevance",
                }
                observe(
                    "archive",
                    "measured_set_archived",
                    ids=key,
                    score=float(score),
                    reason=reason,
                    target_id=target,
                    objective_semantics="legacy_query_relevance",
                )
        if key:
            self.bundle_reasons.setdefault(key, set()).add(reason)
            if target is not None:
                self.bundle_targets.setdefault(key, set()).add(target)

    def _sync_measured(self) -> None:
        snapshot = getattr(self.scorer, "measured_sets_snapshot", None)
        if callable(snapshot):
            for item in snapshot():
                self._archive(item["ids"], "completed_score_snapshot", score=float(item["score"]))

    def _queue(
        self,
        state: DependencyState,
        *,
        lane: str,
        priority: float,
        parent: _Workspace | None = None,
        target_path: tuple[str, ...] | None = None,
        pivot_depth: int | None = None,
        speculative_depth: int | None = None,
    ) -> tuple[_Workspace, bool]:
        if state.identity in self.workspaces:
            return self.workspaces[state.identity], False
        workspace = _Workspace(
            state,
            lane,
            priority,
            len(self.workspaces),
            target_path or (state.target_id,),
            pivot_depth if pivot_depth is not None else (parent.pivot_depth if parent else 0),
            speculative_depth if speculative_depth is not None else (parent.speculative_depth if parent else 0),
            parent.state.identity if parent else None,
        )
        self.workspaces[state.identity] = workspace
        return workspace, True

    def _groups(self, candidates: tuple[str, ...], proposal_ids: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
        source = tuple(identifier for identifier in (proposal_ids or candidates) if identifier in candidates)
        pairs = list(itertools.combinations(source[: self.pair_rescue_width], 2))[: self.settings.max_pairs_per_state]
        singles = [(identifier,) for identifier in candidates]
        # The first pair follows two singletons; later pairs and remaining
        # singletons alternate. Positive singleton values never suppress pairs.
        groups = singles[:2]
        remaining = singles[2:]
        for index in range(max(len(pairs), len(remaining))):
            if index < len(pairs):
                groups.append(pairs[index])
            if index < len(remaining):
                groups.append(remaining[index])
        return tuple(groups)

    def _start(self, workspace: _Workspace) -> None:
        workspace.started = True
        proposal = self.retriever.propose(workspace.state.target_id, workspace.state.premise_ids, fixed_pool=False)
        workspace.proposal = proposal
        self.proposal_batches.append(proposal)
        initial = self.initial_ids if workspace.lane == "root" else ()
        candidates = tuple(
            identifier
            for identifier in _stable_unique((*proposal.ids, *initial))
            if identifier != workspace.state.target_id and identifier not in workspace.state.premise_ids
        )
        workspace.candidate_ids = candidates
        workspace.groups = self._groups(candidates, proposal.ids)
        for identifier in candidates:
            self._archive((identifier,), "discovered_singleton", workspace.state.target_id)
            if identifier not in self.initial_ids and identifier not in self.external_discoveries:
                self.external_discoveries[identifier] = workspace
        self._archive(workspace.state.bundle_ids, "started_state", workspace.state.target_id)
        if not workspace.groups:
            workspace.done = True

    def _measure(self, workspace: _Workspace, group: tuple[str, ...]) -> ActivationRecord:
        state = workspace.state
        quartet = (
            state.premise_ids,
            state.bundle_ids,
            _canonical_ids((*state.premise_ids, *group)),
            _canonical_ids((*state.bundle_ids, *group)),
        )
        self.active_measurement = {
            "state": state.public_dict(),
            "group_ids": list(group),
            "sets": {name: list(ids) for name, ids in zip(("P", "Pe", "PG", "PGe"), quartet)},
            "complete": False,
        }
        try:
            p, pe, pg, pge = _preflight_and_score(self.scorer, quartet, reason="activation")
        finally:
            # This also retains cache or early physical-batch results if a
            # later batch raises, without manufacturing an incomplete A.
            self._sync_measured()
        for name, ids, score in zip(("P", "Pe", "PG", "PGe"), quartet, (p, pe, pg, pge)):
            self._archive(ids, f"measured_{name}", state.target_id, score)
        before, after, group_marginal = pe - p, pge - pg, pge - pe
        activation = after - before
        retained = activation > self.settings.marginal_epsilon and after > self.settings.marginal_epsilon
        self.active_measurement["complete"] = True
        sources = (workspace.proposal.probe_id,) if workspace.proposal is not None else ()
        return ActivationRecord(
            state.target_id,
            state.premise_ids,
            group,
            p,
            pe,
            pg,
            pge,
            activation,
            group_marginal,
            "activation",
            activation,
            retained,
            False,
            sources,
        )

    def _transition(self, workspace: _Workspace, record: ActivationRecord) -> ActivationRecord:
        epsilon = self.settings.marginal_epsilon
        after = record.score_PGe - record.score_PG
        successor = record.successor
        decision, queued, reason = "reject", False, "nonpositive_activation"
        if record.accepted:
            decision = "retain"
            _, queued = self._queue(
                successor,
                lane="retain",
                priority=record.activation,
                parent=workspace,
                target_path=workspace.target_path,
            )
            reason = "positive_conditional_target" if queued else "state_already_seen"
        elif record.activation > epsilon:
            decision = "speculate"
            if workspace.speculative_depth >= self.settings.max_speculative_depth:
                reason = "speculative_depth_limit"
            elif self.speculative_count >= self.settings.max_speculative_states:
                reason = "speculative_state_limit"
            else:
                _, queued = self._queue(
                    successor,
                    lane="speculative",
                    priority=record.activation,
                    parent=workspace,
                    target_path=workspace.target_path,
                    speculative_depth=workspace.speculative_depth + 1,
                )
                if queued:
                    self.speculative_count += 1
                reason = "bounded_negative_target_exploration" if queued else "state_already_seen"
        self._emit(
            "target",
            "target_decision",
            target_id=record.target_id,
            premise_ids=record.premise_ids,
            group_ids=record.group_ids,
            activation=record.activation,
            target_marginal_before=record.score_Pe - record.score_P,
            target_marginal_after=after,
            context_marginal=record.context_marginal,
            decision=decision,
            queued=queued,
            reason=reason,
            pivot_depth=workspace.pivot_depth,
            speculative_depth=workspace.speculative_depth,
        )
        if queued:
            workspace.new_successors += 1
        if record.accepted:
            workspace.positive_successors += 1
        if after < -epsilon:
            pg_ids = _canonical_ids((*record.premise_ids, *record.group_ids))
            pivot_queued = False
            pivot_reason = "pivot_limit"
            new_target: str | None = None
            if workspace.pivot_depth >= self.settings.max_pivot_depth:
                pivot_reason = "pivot_depth_limit"
            elif self.pivot_count < self.settings.max_pivots:
                pivot_reason = "target_cycle"
                for candidate in record.group_ids:
                    if candidate in workspace.target_path:
                        continue
                    new_target = candidate
                    pivot = DependencyState(
                        candidate, tuple(identifier for identifier in pg_ids if identifier != candidate)
                    )
                    _, pivot_queued = self._queue(
                        pivot,
                        lane="pivot",
                        priority=record.score_PG - record.score_PGe,
                        parent=workspace,
                        target_path=(*workspace.target_path, candidate),
                        pivot_depth=workspace.pivot_depth + 1,
                    )
                    pivot_reason = "target_replaced" if pivot_queued else "state_already_seen"
                    if pivot_queued:
                        self.pivot_count += 1
                        workspace.new_successors += 1
                        break
            self._emit(
                "target",
                "target_pivot",
                old_target_id=record.target_id,
                new_target_id=new_target,
                premise_ids=record.premise_ids,
                group_ids=record.group_ids,
                replacement_ids=pg_ids,
                removed_ids=(record.target_id,),
                queued=pivot_queued,
                reason=pivot_reason,
                target_path=workspace.target_path,
                pivot_depth=workspace.pivot_depth + 1,
                pivot_count=self.pivot_count,
            )
        return replace(record, queued=queued)

    def _eligible(self, workspace: _Workspace, cap: int, *, new: bool = True) -> bool:
        if workspace.done or (not workspace.started and (not new or not self._ann_available())):
            return False
        return not (
            workspace.blocked_seen == _scored_sets(self.scorer)
            and workspace.blocked_cap is not None
            and cap <= workspace.blocked_cap
        )

    def _pick(self, work: Sequence[_Workspace]) -> _Workspace | None:
        if not work:
            return None
        alternatives = [item for item in work if item.state.target_id != self.last_target]
        eligible = (
            alternatives if self.consecutive >= self.settings.max_consecutive_quanta and alternatives else list(work)
        )
        return min(eligible, key=lambda item: (-item.priority, item.sequence))

    def _quantum(self, workspace: _Workspace, phase: str, phase_cap: int) -> bool:
        # Phase is a resource-allocation category, while action describes the
        # actual workspace being run. A fairness-triggered fresh root is still
        # an open-root action even when paid from exploitation resources.
        action = (
            "resume_state"
            if workspace.started
            else {
                "root": "open_root",
                "promoted": "open_promoted_root",
                "pivot": "open_pivot",
                "speculative": "open_speculation",
                "retain": "deepen_state",
            }[workspace.lane]
        )
        before_ann, before_sets = self.retriever.ann_calls, _scored_sets(self.scorer)
        before_cursor = workspace.cursor
        before_singletons, before_pairs = workspace.singleton_tests, workspace.pair_tests
        before_positive, before_successors = workspace.positive_successors, workspace.new_successors
        cap = min(self.full_score_cap, phase_cap, before_sets + self.settings.quantum_new_sets)
        _set_cumulative_budget(self.scorer, cap)
        self._emit(
            "scheduler",
            "quantum_started",
            phase=phase,
            action=action,
            target_id=workspace.state.target_id,
            premise_ids=workspace.state.premise_ids,
            lane=workspace.lane,
            ann_calls_before=before_ann,
            scored_sets_before=before_sets,
            phase_score_cap=phase_cap,
            quantum_score_cap=cap,
            ann_cap=self.ann_cap,
            measurement_cursor=before_cursor,
            resumed=workspace.started,
        )
        processed = 0
        reason = "quantum_measurement_limit"
        failed = False
        try:
            if not workspace.started:
                self._start(workspace)
            while workspace.cursor < len(workspace.groups) and processed < self.settings.quantum_measurements:
                group = workspace.groups[workspace.cursor]
                try:
                    record = self._measure(workspace, group)
                except _ScoringStop as exc:
                    if exc.reason == "score_budget_exhausted":
                        reason = (
                            "quantum_set_limit"
                            if cap < min(phase_cap, self.full_score_cap)
                            else (
                                "reserved_score_limit" if phase_cap < self.full_score_cap else "score_budget_exhausted"
                            )
                        )
                        if processed == 0:
                            workspace.blocked_seen, workspace.blocked_cap = _scored_sets(self.scorer), cap
                        break
                    self.skipped.append(SkippedMeasurement(workspace.state, group, exc.reason, exc.detail))
                    observe("activation", self.skipped[-1], event="measurement_skipped")
                    self.capacity_limited = True
                    workspace.cursor += 1
                    processed += 1
                    self.active_measurement = None
                    continue
                self.activations.append(record)
                # The four scores constitute a completed measurement even if
                # persisting or applying the later target decision fails.
                record = self._transition(workspace, record)
                self.activations[-1] = record
                observe("activation", record, event="activation_measured")
                workspace.cursor += 1
                processed += 1
                if len(group) == 1:
                    workspace.singleton_tests += 1
                else:
                    workspace.pair_tests += 1
                self.active_measurement = None
            if workspace.cursor == len(workspace.groups):
                workspace.done = True
                reason = "workspace_completed"
            return bool(processed or self.retriever.ann_calls != before_ann)
        except BaseException:
            reason, failed = "execution_error", True
            raise
        finally:
            _set_cumulative_budget(self.scorer, self.full_score_cap)
            after_ann, after_sets = self.retriever.ann_calls, _scored_sets(self.scorer)
            costs = self.target_costs.setdefault(
                workspace.state.target_id,
                {"target_id": workspace.state.target_id, "ann_calls": 0, "scored_sets": 0, "quanta": 0},
            )
            costs["ann_calls"] += after_ann - before_ann
            costs["scored_sets"] += after_sets - before_sets
            costs["quanta"] += 1
            self.consecutive = self.consecutive + 1 if self.last_target == workspace.state.target_id else 1
            self.last_target = workspace.state.target_id
            self._emit(
                "scheduler",
                "quantum_completed",
                phase=phase,
                action=action,
                target_id=workspace.state.target_id,
                premise_ids=workspace.state.premise_ids,
                ann_calls_before=before_ann,
                ann_calls_after=after_ann,
                scored_sets_before=before_sets,
                scored_sets_after=after_sets,
                measurement_cursor_before=before_cursor,
                measurement_cursor=workspace.cursor,
                remaining_measurements=len(workspace.groups) - workspace.cursor,
                measurements_completed=processed,
                completed=workspace.done,
                reason=reason,
                consecutive_quanta=self.consecutive,
                failed=failed,
            )
            self.state_records.append(
                SearchStateRecord(
                    workspace.state,
                    workspace.priority,
                    "proposal_exhausted" if workspace.done else "expanded",
                    reason,
                    len(workspace.candidate_ids),
                    workspace.singleton_tests - before_singletons,
                    workspace.pair_tests - before_pairs,
                    workspace.positive_successors - before_positive,
                    workspace.new_successors - before_successors,
                    workspace.proposal.ann_call_index if workspace.proposal else None,
                )
            )

    def _exploration_candidate(self, phase_cap: int, avoid_target: str | None = None) -> _Workspace | None:
        pending = [
            item
            for item in self.workspaces.values()
            if item.lane in {"pivot", "speculative"}
            and item.state.target_id != avoid_target
            and not item.started
            and self._eligible(item, phase_cap)
        ]
        # A positive branch cannot monopolize explicit exploration merely by
        # producing several speculative states with the same target.
        if avoid_target is None and self.consecutive >= self.settings.max_consecutive_quanta:
            alternative = self._exploration_candidate(phase_cap, avoid_target=self.last_target)
            if alternative is not None:
                return alternative
        if pending:
            return self._pick(pending)
        if self.pivot_count < self.settings.max_pivots:
            for identifier, parent in self.external_discoveries.items():
                if identifier == avoid_target:
                    continue
                if (
                    (identifier, ()) in self.workspaces
                    or parent.pivot_depth >= self.settings.max_pivot_depth
                    or identifier in parent.target_path
                ):
                    continue
                workspace, added = self._queue(
                    DependencyState(identifier),
                    lane="promoted",
                    priority=0.0,
                    parent=parent,
                    target_path=(*parent.target_path, identifier),
                    pivot_depth=parent.pivot_depth + 1,
                )
                if added:
                    self.pivot_count += 1
                    self._emit(
                        "target",
                        "external_root_promoted",
                        old_target_id=parent.state.target_id,
                        new_target_id=identifier,
                        pivot_depth=workspace.pivot_depth,
                        pivot_count=self.pivot_count,
                        target_path=workspace.target_path,
                        reason="external_discovery",
                        queued=True,
                    )
                    return workspace
        for identifier in self.root_order:
            if identifier == avoid_target:
                continue
            workspace = self.workspaces.get((identifier, ()))
            if workspace is None:
                workspace, _ = self._queue(DependencyState(identifier), lane="root", priority=0.0)
                return workspace
        return None

    def run(self, initial_pool: InitialCandidatePool | Sequence[str] | None = None) -> EvidenceSearchArchive:
        if initial_pool is None:
            initial_pool = self.retriever.build_initial_pool(expand=True)
        if isinstance(initial_pool, InitialCandidatePool):
            initial_ids = initial_pool.candidate_ids
            dense_first = initial_pool.dense_batch.ids[0] if initial_pool.dense_batch.ids else None
        elif isinstance(initial_pool, Sequence) and not isinstance(initial_pool, (str, bytes)):
            initial_ids, dense_first = _stable_unique(initial_pool), None
        else:
            raise ValueError("initial_pool must be a pool or sequence of memory IDs")
        self.initial_ids = _stable_unique(initial_ids)
        unknown = set(self.initial_ids) - set(self.retriever.visible_ids)
        if unknown:
            raise ValueError(f"initial pool contains unknown memory IDs: {sorted(unknown)}")
        self.root_order = self._root_order(self.initial_ids, dense_first)
        self.initial_ann, self.initial_sets = self.retriever.ann_calls, _scored_sets(self.scorer)
        self.full_score_cap = self.initial_sets + self.max_scored_sets
        _set_cumulative_budget(self.scorer, self.full_score_cap)
        self.bundle_reasons: dict[tuple[str, ...], set[str]] = {}
        self.bundle_targets: dict[tuple[str, ...], set[str]] = {}
        self.measured: dict[tuple[str, ...], dict[str, Any]] = {}
        self.workspaces: dict[tuple[str, tuple[str, ...]], _Workspace] = {}
        self.external_discoveries: dict[str, _Workspace] = {}
        self.activations: list[ActivationRecord] = []
        self.state_records: list[SearchStateRecord] = []
        self.skipped: list[SkippedMeasurement] = []
        self.proposal_batches: list[ProposalBatch] = []
        self.scheduler_events: list[dict[str, Any]] = []
        self.target_events: list[dict[str, Any]] = []
        self.target_costs: dict[str, dict[str, Any]] = {}
        self.pivot_count = self.speculative_count = self.consecutive = 0
        self.last_target: str | None = None
        self.active_measurement: dict[str, Any] | None = None
        self.capacity_limited = False
        self._initialized = True
        for identifier in self.initial_ids:
            self._archive((identifier,), "initial_candidate")
        self._sync_measured()
        available_ann = (
            max(0, self.ann_cap - self.initial_ann)
            if self.ann_cap is not None
            else max(len(self.initial_ids), self.settings.coverage_roots + self.settings.exploration_roots + 1)
        )
        coverage = (
            min(self.settings.coverage_roots, len(self.root_order), max(1, available_ann // 2)) if available_ann else 0
        )
        exploration = min(self.settings.exploration_roots, max(0, (available_ann - coverage) // 3))
        reserve_sets = min(
            int(self.max_scored_sets * self.settings.reserve_score_fraction),
            exploration * self.settings.quantum_new_sets,
        )
        # Even the small-budget configuration must leave a complete first
        # quartet for each explicitly promised root. Shared sets may make the
        # actual cost smaller; they never justify promising unavailable quota.
        exploration = min(exploration, reserve_sets // 4)
        reserve_sets = min(reserve_sets, exploration * self.settings.quantum_new_sets)
        if self.initial_sets == 0:
            coverage = min(coverage, (self.max_scored_sets - reserve_sets) // 4)
        prelate_cap = self.full_score_cap - reserve_sets
        self._emit(
            "scheduler",
            "search_allocation",
            initial_target_ids=self.initial_ids,
            root_order=self.root_order,
            root_selection=self.settings.root_selection,
            coverage_roots=coverage,
            exploration_roots=exploration,
            reserved_sets=reserve_sets,
            initial_ann_calls=self.initial_ann,
            ann_cap=self.ann_cap,
            initial_scored_sets=self.initial_sets,
            full_score_cap=self.full_score_cap,
        )
        if self.initial_ids and self.initial_sets == 0 and self.max_scored_sets < 4:
            self._emit(
                "scheduler",
                "evidence_search_stop",
                reason="score_budget_exhausted",
                pending_workspaces=0,
                final_ann_calls=self.retriever.ann_calls,
                final_scored_sets=0,
                pivot_count=0,
                speculative_count=0,
            )
            return self._archive_result("score_budget_exhausted")
        for root_index, identifier in enumerate(self.root_order[:coverage]):
            if not self._ann_available():
                break
            workspace, _ = self._queue(DependencyState(identifier), lane="root", priority=0.0)
            root_cap = max(_scored_sets(self.scorer), prelate_cap - 4 * (coverage - root_index - 1))
            self._quantum(workspace, "coverage", root_cap)
        coverage_ann, coverage_sets = self.retriever.ann_calls, _scored_sets(self.scorer)
        exploration_released = exploration == 0
        exploration_used = 0
        while True:
            if not exploration_released:
                deep_ann = max(1, available_ann - coverage - exploration)
                deep_sets = max(1, prelate_cap - coverage_sets)
                progress = max(
                    (self.retriever.ann_calls - coverage_ann) / deep_ann,
                    (_scored_sets(self.scorer) - coverage_sets) / deep_sets,
                )
                active = [
                    workspace
                    for workspace in self.workspaces.values()
                    if workspace.lane in {"root", "retain", "promoted"} and self._eligible(workspace, prelate_cap)
                ]
                if progress >= self.settings.exploration_fraction or not active:
                    exploration_released = True
                    self._emit(
                        "scheduler",
                        "exploration_released",
                        phase="exploration",
                        ann_calls=self.retriever.ann_calls,
                        scored_sets_after=_scored_sets(self.scorer),
                        reserved_sets=reserve_sets,
                        reason="mid_search_release",
                    )
            cap = self.full_score_cap if exploration_released else prelate_cap
            if exploration_released and exploration_used < exploration and self._ann_available():
                workspace = self._exploration_candidate(cap)
                if workspace is not None:
                    root_cap = max(_scored_sets(self.scorer), cap - 4 * (exploration - exploration_used - 1))
                    self._quantum(workspace, "exploration", root_cap)
                    exploration_used += 1
                    continue
                exploration_used = exploration
            active = [
                workspace
                for workspace in self.workspaces.values()
                if (workspace.started or workspace.lane == "retain") and self._eligible(workspace, cap)
            ]
            workspace = self._pick(active)
            if workspace is None and exploration_released and self._ann_available():
                workspace = self._exploration_candidate(cap)
            if workspace is None:
                if not exploration_released:
                    exploration_released = True
                    continue
                break
            if (
                workspace.state.target_id == self.last_target
                and self.consecutive >= self.settings.max_consecutive_quanta
            ):
                if not exploration_released:
                    exploration_released = True
                    self._emit(
                        "scheduler",
                        "exploration_released",
                        phase="exploration",
                        ann_calls=self.retriever.ann_calls,
                        scored_sets_after=_scored_sets(self.scorer),
                        reserved_sets=reserve_sets,
                        reason="fairness_release",
                    )
                    continue
                alternative = (
                    self._exploration_candidate(cap, avoid_target=self.last_target) if self._ann_available() else None
                )
                if alternative is not None:
                    workspace = alternative
                else:
                    self._emit(
                        "scheduler",
                        "target_fairness_relaxed",
                        target_id=self.last_target,
                        reason="no_alternative_target",
                        ann_calls=self.retriever.ann_calls,
                        consecutive_quanta=self.consecutive,
                    )
            self._quantum(
                workspace,
                "exploitation" if self._ann_available() else "pending_only",
                cap,
            )
        pending = [workspace for workspace in self.workspaces.values() if not workspace.done]
        if not self.initial_ids:
            stop = "no_initial_candidates"
        elif any(workspace.blocked_seen is not None for workspace in pending):
            stop = "score_budget_exhausted"
        elif not self._ann_available():
            stop = "ann_budget_exhausted"
        elif self.capacity_limited:
            stop = "input_capacity"
        else:
            stop = "finite_frontier_exhausted"
        self._emit(
            "scheduler",
            "evidence_search_stop",
            reason=stop,
            pending_workspaces=len(pending),
            final_ann_calls=self.retriever.ann_calls,
            final_scored_sets=_scored_sets(self.scorer),
            pivot_count=self.pivot_count,
            speculative_count=self.speculative_count,
        )
        return self._archive_result(stop)

    search = run

    def _archive_result(self, stop: str) -> EvidenceSearchArchive:
        self._sync_measured()
        incomplete = []
        completed_measurements = {
            (record.target_id, record.premise_ids, record.group_ids) for record in self.activations
        }
        for workspace in self.workspaces.values():
            if workspace.done or workspace.cursor >= len(workspace.groups):
                continue
            group = workspace.groups[workspace.cursor]
            state = workspace.state
            if (state.target_id, state.premise_ids, group) in completed_measurements:
                continue
            quartet = (
                state.premise_ids,
                state.bundle_ids,
                _canonical_ids((*state.premise_ids, *group)),
                _canonical_ids((*state.bundle_ids, *group)),
            )
            incomplete.append(
                {
                    "state": state.public_dict(),
                    "group_ids": list(group),
                    "complete": False,
                    "sets": {name: list(ids) for name, ids in zip(("P", "Pe", "PG", "PGe"), quartet)},
                    "available_sets": [self.measured[ids] for ids in quartet if ids in self.measured],
                    "missing_sets": [list(ids) for ids in quartet if ids not in self.measured],
                }
            )
        return EvidenceSearchArchive(
            initial_target_ids=self.initial_ids,
            bundles=tuple(
                EvidenceBundle(ids, tuple(self.bundle_reasons[ids]), tuple(self.bundle_targets.get(ids, ())))
                for ids in sorted(self.bundle_reasons, key=lambda item: (len(item), item))
            ),
            activations=tuple(self.activations),
            state_records=tuple(self.state_records),
            skipped_measurements=tuple(self.skipped),
            proposal_batches=tuple(self.proposal_batches),
            stop_reason=stop,
            initial_ann_calls=self.initial_ann,
            final_ann_calls=self.retriever.ann_calls,
            initial_scored_sets=self.initial_sets,
            final_scored_sets=_scored_sets(self.scorer),
            signal_kind="activation",
            fixed_pool=False,
            pair_rescue_width=self.pair_rescue_width,
            scheduler_events=tuple(self.scheduler_events),
            target_events=tuple(self.target_events),
            measured_sets=tuple(
                self.measured[key] for key in sorted(self.measured, key=lambda item: (len(item), item))
            ),
            pending_workspaces=tuple(
                workspace.public_dict() for workspace in self.workspaces.values() if not workspace.done
            ),
            target_costs=tuple(self.target_costs[key] for key in sorted(self.target_costs)),
            root_order=self.root_order,
            incomplete_measurements=tuple(incomplete),
        )

    def partial_public_dict(
        self, *, stop_reason: str = "execution_error", detail: str = "search incomplete"
    ) -> dict[str, Any] | None:
        if not self._initialized:
            return None
        value = self._archive_result(stop_reason).public_dict()
        value.update(
            {
                "partial": True,
                "active_measurement": self.active_measurement,
                "detail": detail,
                "resume_semantics": "task_restart_not_frontier_replay",
            }
        )
        if self.active_measurement is not None:
            value["active_measurement"] = dict(self.active_measurement)
            value["active_measurement"]["available_sets"] = [
                self.measured[_canonical_ids(ids)]
                for ids in self.active_measurement["sets"].values()
                if _canonical_ids(ids) in self.measured
            ]
        return value
