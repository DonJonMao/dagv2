"""Finite, auditable Chain search primitives (extend, join, stop)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence

from .chain_judge import JointJudge, JointScore, PublicQuery
from .chain_support import ClosureResult, close_support


@dataclass(frozen=True)
class EvidenceState:
    raw_ids: tuple[str, ...]
    paths: tuple[tuple[str, ...], ...] = ()
    expandable_ids: tuple[str, ...] = ()
    completed_actions: frozenset[str] = frozenset()
    graph_epoch: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.raw_ids, (str, bytes)) or not isinstance(self.raw_ids, Sequence):
            raise ValueError("EvidenceState.raw_ids must be a sequence")
        ids = tuple(str(x) for x in self.raw_ids)
        if not ids or any(not identifier for identifier in ids):
            raise ValueError("EvidenceState.raw_ids must contain non-empty IDs")
        if len(set(ids)) != len(ids):
            raise ValueError("EvidenceState.raw_ids must be unique")
        if self.paths:
            paths = tuple(tuple(str(identifier) for identifier in path) for path in self.paths)
        else:
            # Independent roots remain independent provenance paths.  Treating
            # (a, x) as one implicit path would incorrectly make x a child of
            # a and would lose a as an expandable endpoint.
            paths = tuple((identifier,) for identifier in ids)
        for path in paths:
            if not path:
                raise ValueError("an evidence path cannot be empty")
            if len(path) != len(set(path)):
                raise ValueError("a path cannot repeat a memory")
            if not set(path) <= set(ids):
                raise ValueError("evidence paths must reference raw_ids")
        paths = tuple(dict.fromkeys(paths))
        if self.expandable_ids:
            expandable = tuple(dict.fromkeys(str(value) for value in self.expandable_ids))
        else:
            expandable = tuple(dict.fromkeys(path[-1] for path in paths))
        if not set(expandable) <= set(ids):
            raise ValueError("expandable_ids must reference raw_ids")
        object.__setattr__(self, "raw_ids", ids)
        object.__setattr__(self, "paths", paths)
        object.__setattr__(self, "expandable_ids", expandable)
        object.__setattr__(
            self,
            "completed_actions",
            frozenset(str(value) for value in self.completed_actions),
        )

    @property
    def navigation_key(self) -> tuple[object, ...]:
        """Identity for future navigation, excluding diagnostic graph epochs."""

        return (
            tuple(sorted(self.raw_ids)),
            tuple(sorted(self.paths)),
            tuple(sorted(self.expandable_ids)),
            tuple(sorted(self.completed_actions)),
        )

    def extend(
        self,
        memory_id: str,
        *,
        parent: str | None = None,
        source_path: Sequence[str] | None = None,
        epoch: int | None = None,
    ) -> "EvidenceState":
        identifier = str(memory_id)
        if not identifier:
            raise ValueError("extended memory ID cannot be empty")
        parent_id = None if parent is None else str(parent)
        if parent_id is not None and parent_id not in self.raw_ids:
            raise ValueError("extension parent must belong to the evidence state")
        fixed_source = None if source_path is None else tuple(str(value) for value in source_path)
        if fixed_source is not None and fixed_source not in self.paths:
            return self
        parents = set(self.expandable_ids if parent_id is None else (parent_id,))
        new_paths: list[tuple[str, ...]] = []
        changed = False
        for path in self.paths:
            selected_source = fixed_source is None or path == fixed_source
            can_extend = selected_source and path[-1] in parents
            if can_extend and identifier not in path:
                new_paths.append(path + (identifier,))
                changed = True
            else:
                new_paths.append(path)
        if not changed:
            return self
        normalized_paths = tuple(dict.fromkeys(new_paths))
        if identifier in self.raw_ids:
            raw_ids = self.raw_ids
        else:
            # Keep display/source order aligned with the path on which a new
            # record was introduced, then retain any independent records.
            raw_ids = tuple(
                dict.fromkeys(
                    value
                    for path in normalized_paths
                    for value in path
                )
            )
            raw_ids += tuple(value for value in self.raw_ids if value not in raw_ids)
        action_parent = parent_id if parent_id is not None else ",".join(sorted(parents))
        action_source = "" if fixed_source is None else "/".join(fixed_source)
        return EvidenceState(
            raw_ids,
            normalized_paths,
            (),
            self.completed_actions
            | {f"extend:{action_parent}->{identifier}:source={action_source}"},
            self.graph_epoch if epoch is None else epoch,
        )

    def join(self, other: "EvidenceState", *, epoch: int | None = None) -> "EvidenceState":
        """Return a true provenance union while retaining both endpoints."""

        if not isinstance(other, EvidenceState):
            raise ValueError("EvidenceState.join requires another EvidenceState")
        raw_ids = tuple(dict.fromkeys(self.raw_ids + other.raw_ids))
        paths = tuple(dict.fromkeys(self.paths + other.paths))
        expandable = tuple(dict.fromkeys(self.expandable_ids + other.expandable_ids))
        adds_information = (
            raw_ids != self.raw_ids
            or paths != self.paths
            or expandable != self.expandable_ids
            or not other.completed_actions <= self.completed_actions
        )
        if not adds_information:
            return self
        other_identity = repr(other.navigation_key)
        return EvidenceState(
            raw_ids,
            paths,
            expandable,
            self.completed_actions
            | other.completed_actions
            | {f"join:{other_identity}"},
            max(self.graph_epoch, other.graph_epoch) if epoch is None else epoch,
        )


@dataclass(frozen=True)
class Terminal:
    state: EvidenceState
    score: JointScore
    closure: ClosureResult | None = None

    @property
    def selected_ids(self) -> tuple[str, ...]:
        if self.closure and self.closure.status == "single_deletion_minimal":
            return self.closure.retained_ids
        return self.state.raw_ids

    @property
    def selected_is_minimal(self) -> bool:
        return bool(
            self.closure
            and self.closure.status == "single_deletion_minimal"
            and self.closure.selected_is_minimal
        )


@dataclass
class SearchArchive:
    observed_terminals: list[Terminal] = field(default_factory=list)
    verified_terminals: list[Terminal] = field(default_factory=list)
    open_states: list[EvidenceState] = field(default_factory=list)
    probes: list[dict] = field(default_factory=list)
    probe_complete: bool = True
    stop_reason: str = "frontier_exhausted"


class _JointBudgetExhausted(RuntimeError):
    pass


class ChainSearcher:
    def __init__(
        self,
        query: PublicQuery,
        judge: JointJudge,
        *,
        horizon: int = 2,
        max_joint_contexts: int = 512,
        max_verify_calls: int = 512,
        reader_limit: int | None = None,
        judge_limit: int | None = None,
    ):
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 0 or horizon > 2:
            raise ValueError("ChainSearcher horizon must be 0, 1, or 2")
        for name, value in (
            ("max_joint_contexts", max_joint_contexts),
            ("max_verify_calls", max_verify_calls),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"ChainSearcher {name} must be a non-negative integer")
        for name, value in (("reader_limit", reader_limit), ("judge_limit", judge_limit)):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"ChainSearcher {name} must be a non-negative integer or None")
        if reader_limit is not None and judge_limit is not None and reader_limit > judge_limit:
            raise ValueError("reader_limit cannot exceed judge_limit")
        self.query, self.judge = query, judge
        self.horizon = horizon
        self.max_joint_contexts = max_joint_contexts
        self.max_verify_calls = max_verify_calls
        self.reader_limit = reader_limit
        self.judge_limit = judge_limit
        self._scores: dict[tuple[str, ...], JointScore] = {}
        self._joint_calls = 0

    def score(self, ids: Sequence[str]) -> JointScore:
        # Joint evidence is a set for scoring.  Navigation paths remain on the
        # EvidenceState, so canonicalizing this cache key does not merge
        # distinct historical prerequisites or expandable endpoints.
        canonical = tuple(sorted(dict.fromkeys(str(x) for x in ids)))
        if canonical not in self._scores:
            if self._joint_calls >= self.max_joint_contexts:
                raise _JointBudgetExhausted("joint budget exhausted")
            self._scores[canonical] = self.judge.score(self.query, canonical)
            self._joint_calls += 1
        return self._scores[canonical]

    def search(
        self,
        roots: Sequence[EvidenceState], *,
        proposals: Mapping[str, Sequence[str]] | Callable[[EvidenceState], Sequence[str]] = (),
               join_pool: Sequence[EvidenceState] = (), closure: bool = False) -> SearchArchive:
        if isinstance(roots, (str, bytes)) or not isinstance(roots, Sequence):
            raise ValueError("roots must be a sequence of EvidenceState values")
        if any(not isinstance(state, EvidenceState) for state in roots):
            raise ValueError("roots must contain EvidenceState values")
        if isinstance(join_pool, (str, bytes)) or not isinstance(join_pool, Sequence):
            raise ValueError("join_pool must be a sequence of EvidenceState values")
        if any(not isinstance(state, EvidenceState) for state in join_pool):
            raise ValueError("join_pool must contain EvidenceState values")

        archive = SearchArchive()
        root_states = _dedupe_states(roots)
        open_keys: set[tuple[object, ...]] = set()
        state_terminals: dict[tuple[object, ...], Terminal] = {}
        closure_seen: set[tuple[object, ...]] = set()
        successor_cache: dict[tuple[object, ...], list[EvidenceState]] = {}
        verify_calls = 0

        def reader_feasible(ids: Sequence[str]) -> bool:
            return self.reader_limit is None or len(ids) <= self.reader_limit

        def add_open(state: EvidenceState) -> None:
            key = state.navigation_key
            if key not in open_keys:
                open_keys.add(key)
                archive.open_states.append(state)

        def add_terminal(target: list[Terminal], terminal: Terminal) -> None:
            # Preserve C and W as separate scored alternatives.  If C == W,
            # retaining the closure-bearing record last exposes the proven
            # minimality without losing the original score (the set is equal).
            key = (terminal.state.navigation_key, terminal.selected_ids)
            for position, existing in enumerate(target):
                if (existing.state.navigation_key, existing.selected_ids) == key:
                    if terminal.selected_is_minimal and not existing.selected_is_minimal:
                        target[position] = terminal
                    return
            target.append(terminal)

        def observe(state: EvidenceState) -> Terminal | None:
            nonlocal verify_calls
            add_open(state)
            key = state.navigation_key
            if key in state_terminals:
                return state_terminals[key]
            if self.judge_limit is not None and len(state.raw_ids) > self.judge_limit:
                return None
            original = Terminal(state, self.score(state.raw_ids))
            state_terminals[key] = original
            if reader_feasible(original.selected_ids):
                add_terminal(archive.observed_terminals, original)
            if not closure:
                if reader_feasible(original.selected_ids):
                    add_terminal(archive.verified_terminals, original)
                return original
            if key in closure_seen or verify_calls >= self.max_verify_calls:
                return original
            closure_seen.add(key)
            remaining = self.max_verify_calls - verify_calls
            result = close_support(
                self.query,
                state.raw_ids,
                self.judge,
                max_verify_calls=remaining,
            )
            verify_calls += len(result.deletion_trace) + int(result.initial_verification is not None)
            if not result.initial_verification or not result.initial_verification.supported:
                return original
            if reader_feasible(original.selected_ids):
                add_terminal(archive.verified_terminals, original)
            if result.status == "single_deletion_minimal" and reader_feasible(result.retained_ids):
                # Scoring W is a new joint context.  A depleted score budget
                # must leave the already verified C intact and terminate the
                # probe cleanly rather than converting W to C's score.
                reduced = Terminal(state, self.score(result.retained_ids), result)
                add_terminal(archive.observed_terminals, reduced)
                add_terminal(archive.verified_terminals, reduced)
            return original

        def successors(state: EvidenceState) -> list[EvidenceState]:
            key = state.navigation_key
            if key in successor_cache:
                return successor_cache[key]
            values: list[EvidenceState] = []
            if callable(proposals):
                proposed = proposals(state)
                if isinstance(proposed, (str, bytes)) or not isinstance(proposed, Sequence):
                    raise ValueError("proposal callback must return a sequence")
                for candidate in proposed:
                    child = state.extend(str(candidate), epoch=state.graph_epoch + 1)
                    if child is not state:
                        values.append(child)
            elif isinstance(proposals, Mapping):
                for anchor in state.expandable_ids:
                    proposed = proposals.get(anchor, ())
                    if isinstance(proposed, (str, bytes)) or not isinstance(proposed, Sequence):
                        raise ValueError("proposal mappings must contain sequences")
                    for candidate in proposed:
                        child = state.extend(
                            str(candidate),
                            parent=anchor,
                            epoch=state.graph_epoch + 1,
                        )
                        if child is not state:
                            values.append(child)
            elif proposals:
                raise ValueError("proposals must be a mapping, callback, or empty sequence")
            for other in join_pool:
                joined = state.join(other, epoch=max(state.graph_epoch, other.graph_epoch) + 1)
                if joined is not state:
                    values.append(joined)
            result = _dedupe_states(values)
            successor_cache[key] = result
            return result

        def stop_for_budget(prefix: EvidenceState | None = None) -> SearchArchive:
            archive.probe_complete = False
            archive.stop_reason = "joint_budget_exhausted"
            if prefix is not None:
                archive.probes.append(
                    {
                        "prefix": prefix.raw_ids,
                        "reason": "joint_budget_exhausted",
                        "horizon": self.horizon,
                    }
                )
            return archive

        # Root scoring is a fairness boundary: every root gets access to the
        # joint-score budget before any root's descendants consume it.
        active: list[tuple[EvidenceState, Terminal]] = []
        for state in root_states:
            try:
                terminal = observe(state)
            except _JointBudgetExhausted:
                return stop_for_budget()
            if terminal is not None:
                active.append((state, terminal))
        if self.horizon == 0:
            return archive

        while active:
            prefix, prefix_terminal = active.pop(0)
            local_frontier = [prefix]
            local_seen = {prefix.navigation_key}
            candidates: list[Terminal] = []
            discovered = 0
            for _depth in range(1, self.horizon + 1):
                next_frontier: list[EvidenceState] = []
                for state in local_frontier:
                    for child in successors(state):
                        if child.navigation_key in local_seen:
                            continue
                        local_seen.add(child.navigation_key)
                        discovered += 1
                        try:
                            terminal = observe(child)
                        except _JointBudgetExhausted:
                            return stop_for_budget(prefix)
                        if terminal is not None:
                            candidates.append(terminal)
                            next_frontier.append(child)
                local_frontier = next_frontier
                if not local_frontier:
                    break
            best = choose_terminal(candidates)
            if best is not None and best.score.log_u > prefix_terminal.score.log_u:
                archive.probes.append(
                    {
                        "prefix": prefix.raw_ids,
                        "reason": "rolled_forward",
                        "horizon": self.horizon,
                        "selected": best.state.raw_ids,
                        "discovered": discovered,
                    }
                )
                active.append((best.state, best))
            else:
                archive.probes.append(
                    {
                        "prefix": prefix.raw_ids,
                        "reason": "local_no_improvement" if candidates else "proposal_exhausted",
                        "horizon": self.horizon,
                        "discovered": discovered,
                    }
                )
        return archive


def _dedupe_states(states: Iterable[EvidenceState]) -> list[EvidenceState]:
    seen: set[tuple[object, ...]] = set()
    result = []
    for state in states:
        key = state.navigation_key
        if key not in seen:
            seen.add(key)
            result.append(state)
    return result


def choose_terminal(terminals: Sequence[Terminal]) -> Terminal | None:
    if not terminals:
        return None
    return sorted(terminals, key=lambda item: (-item.score.log_u, len(item.selected_ids), item.selected_ids))[0]
