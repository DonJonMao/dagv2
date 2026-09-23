"""Fixed-claim support closure for BridgeTree-Chain."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from .chain_judge import Claim, JointJudge, PublicQuery, Verification


@dataclass(frozen=True)
class ClosureResult:
    original_ids: tuple[str, ...]
    retained_ids: tuple[str, ...]
    claim: Claim | None
    status: str
    initial_verification: Verification | None
    deletion_trace: tuple[dict, ...] = ()
    closure_applied: bool = False
    closure_completed: bool = False
    selected_is_minimal: bool = False


def close_support(query: PublicQuery, raw_ids: Sequence[str], judge: JointJudge, *,
                  source_order: Callable[[Sequence[str]], Sequence[str]] | None = None,
                  max_verify_calls: int | None = None) -> ClosureResult:
    """Try single deletions while keeping one immutable claim.

    A failed verification is different from an exhausted budget.  The returned
    status therefore remains explicit and callers can keep the original state
    open for navigation.
    """
    original = tuple(dict.fromkeys(str(x) for x in raw_ids))
    if not original:
        return ClosureResult((), (), None, "empty", None)
    if max_verify_calls is not None and max_verify_calls < 0:
        raise ValueError("max_verify_calls must be non-negative")
    claim = judge.claim(query, original)
    if max_verify_calls == 0:
        return ClosureResult(original, original, claim, "closure_budget_exhausted", None,
                             closure_applied=True, closure_completed=False)
    calls = 0
    initial = judge.verify(query, claim, original)
    calls += 1
    if not initial.supported:
        return ClosureResult(original, original, claim, "initial_not_supported", initial,
                             closure_applied=False, closure_completed=True)
    retained = list(original)
    trace: list[dict] = []
    order_fn = source_order or (lambda values: sorted(values))
    while True:
        removed = False
        for identifier in order_fn(retained):
            if max_verify_calls is not None and calls >= max_verify_calls:
                return ClosureResult(original, tuple(retained), claim, "closure_budget_exhausted", initial,
                                     tuple(trace), True, False, False)
            trial = tuple(x for x in retained if x != identifier)
            verification = judge.verify(query, claim, trial)
            calls += 1
            trace.append({"removed": identifier, "trial_ids": list(trial), "supported": verification.supported,
                          "reason": verification.reason})
            if verification.supported:
                retained.remove(identifier)
                removed = True
                break
        if not removed:
            return ClosureResult(original, tuple(retained), claim, "single_deletion_minimal", initial,
                                 tuple(trace), True, True, True)
