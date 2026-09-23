"""One ledger per question; reservations happen before operations, including retries."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, field

class BudgetExceeded(RuntimeError):
    def __init__(self, kind, stage, requested, remaining):
        self.kind, self.stage = kind, stage
        super().__init__(f'{kind} budget at {stage}: requested={requested}, remaining={remaining}')

@dataclass
class Ledger:
    limits: dict
    sink: object = None
    used: Counter = field(default_factory=Counter)
    events: list = field(default_factory=list)

    def remaining(self, kind):
        return max(0, self.limits.get(kind, 10**12) - self.used[kind])

    def reserve(self, kind, stage, amount=1):
        if not isinstance(amount, int) or amount < 0:
            raise ValueError('Budget reservation must be a nonnegative integer')
        if amount > self.remaining(kind):
            self.record({'event':'budget_exhausted','kind':kind,'stage':stage,'requested':amount})
            raise BudgetExceeded(kind, stage, amount, self.remaining(kind))
        self.used[kind] += amount
        self.record({'event':'budget_reserved','kind':kind,'stage':stage,'amount':amount,'used':self.used[kind]})

    def record(self, event):
        event=dict(event); self.events.append(event)
        if self.sink: self.sink(event)

    def public_dict(self):
        return {'limits':dict(self.limits),'used':dict(self.used),
                'remaining':{k:self.remaining(k) for k in self.limits},'events':self.events}
