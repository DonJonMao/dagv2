"""Day2's explicit proxy-free control, using the real BT proposal interface.

This is NOT an activation switch inside EvidenceBridgeSearcher. Requirements
take turns; each alternates a fresh ANN-ranked root and a continuation through
raw passages returned by earlier conditional probes. No set score, fabricated
gain, score-dependent quantum, pair test, speculation gate or scored pivot is
used. Every proposal is the vendored retriever's actual query+target+premises
retrieval, and all hits remain available to the downstream evidence resolver.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy


class ProxyFreeSearch:
    def __init__(self, retriever, requirements, query, record):
        if retriever.max_ann_calls is None:
            raise ValueError('Proxy-free control requires a finite shared ANN allocation')
        self.retriever, self.record = retriever, record
        self.requirements = [dict(r) for r in requirements if r.get('necessary', True)]
        if not self.requirements:
            # The current executable subquestion remains the retrieval task
            # even when the caller supplied only optional/no global needs.
            self.requirements = [{'id':'current_task','description':query,'time_scope':'unknown','necessary':True}]
        self.archive = {
            'method_version':'day2_proxy_free_requirements_ann_v1',
            'search_family':'explicit_proxy_free_control', 'signal_kind':'none',
            'uses_set_proxy':False, 'measured_sets':[], 'activations':[],
            'disabled_modules':['set_score_gain','score_quantum','four_set_tests','pair_tests',
                                'activation_retention','score_gated_speculation','scored_pivot'],
            'requirements':deepcopy(self.requirements), 'scheduler_events':[],
            'states':[], 'proposal_batches':[], 'stop_reason':'not_started',
            'interpretation':'Multiple mechanisms disabled together; not an isolated mathematical A intervention',
        }
        self.lanes = []

    def emit(self, event, **fields):
        value = {'event':event,'module':'proxy_free_scheduler',**fields}
        self.archive['scheduler_events'].append(value)
        self.record(value)

    def initialize(self, initial_pool):
        if self.lanes:
            raise ValueError('Search already initialized')
        self.archive.update(initial_target_ids=list(initial_pool.candidate_ids),
                            initial_ann_calls=self.retriever.ann_calls,
                            initial_scored_sets=0, final_scored_sets=0)
        roots = tuple(initial_pool.candidate_ids)
        self.lanes = [{'requirement': r, 'roots': deque(roots), 'continuations': deque(),
                       'turns': 0, 'seen': set()} for r in self.requirements]
        self.cursor = 0
        self.paused = False
        self.archive['stop_reason'] = 'ready'
        self.emit('proxy_free_started', requirement_ids=[r['id'] for r in self.requirements],
                  initial_target_ids=list(roots), ann_cap=self.retriever.max_ann_calls,
                  proposal_order='necessary_requirement_round_robin_then_local_ANN_rank',
                  root_continuation_schedule='alternate_when_both_available')

    def step(self, active_ids=None):
        """At most one vendor proposal; never initializes/expands the pool."""
        if not self.lanes:
            raise ValueError('Initialize before stepping')
        if self.paused:
            return None
        if self.retriever.remaining_ann_calls == 0:
            self.archive['stop_reason'] = 'ann_budget_exhausted'
            return None
        active = set(active_ids) if active_ids is not None else {r['id'] for r in self.requirements}
        for _ in self.lanes:
            lane = self.lanes[self.cursor]
            self.cursor = (self.cursor + 1) % len(self.lanes)
            if lane['requirement']['id'] not in active:
                continue
            state = self._next(lane)
            if state is None:
                continue
            target, premises, path, origin, rank = state
            # The search belongs to one immutable grounded-query/binding identity.
            lane['seen'].add((target, premises))
            lane['turns'] += 1
            self.retriever.set_information_needs([lane['requirement']])
            self.emit('proxy_free_conditional_started', requirement_id=lane['requirement']['id'],
                      target_id=target, premise_ids=list(premises), navigation_path=list(path),
                      origin=origin, local_ANN_rank=rank, requirement_turn=lane['turns'])
            before = self.retriever.ann_calls
            proposal = self.retriever.propose(target, premises, fixed_pool=False)
            self.archive['proposal_batches'].append(proposal.public_dict())
            self.archive['states'].append({'requirement_id':lane['requirement']['id'],
                'target_id':target, 'premise_ids':list(premises), 'navigation_path':list(path),
                'probe_id':proposal.probe_id, 'candidate_ids':list(proposal.ids)})
            descendants = []
            for hit in proposal.hits:
                if hit.memory_id in path:
                    continue
                new_premises = tuple(dict.fromkeys((*premises, target)))
                if (hit.memory_id, new_premises) not in lane['seen']:
                    descendants.append((hit.memory_id, new_premises, (*path, hit.memory_id),
                                        'conditional_hit', hit.rank))
            for descendant in reversed(descendants):
                lane['continuations'].appendleft(descendant)
            self.emit('proxy_free_conditional_completed', requirement_id=lane['requirement']['id'],
                      probe_id=proposal.probe_id, target_id=target, premise_ids=list(premises),
                      candidate_ids=list(proposal.ids), new_navigation_states=len(descendants),
                      completed_ann_calls=self.retriever.ann_calls-before, evidence_dependency_claim=False)
            self.archive['stop_reason'] = 'ready'
            return proposal
        self.archive['stop_reason'] = 'finite_frontier_exhausted' if active else 'paused_demands'
        return None

    def pause(self):
        self.paused = True
        self.archive['stop_reason'] = 'paused'

    def frontier_available(self):
        """Empty probes do not exhaust other roots or continuation states."""
        return any(any((target,()) not in lane['seen'] for target in lane['roots']) or
                   any((s[0],s[1]) not in lane['seen'] for s in lane['continuations'])
                   for lane in self.lanes)

    def resume(self):
        self.paused = False
        self.archive['stop_reason'] = 'ready'

    def snapshot(self):
        return {'archive':deepcopy(self.archive), 'cursor':self.cursor, 'paused':self.paused,
                'lanes':[{'requirement':deepcopy(l['requirement']), 'roots':list(l['roots']),
                          'continuations':list(l['continuations']), 'turns':l['turns'],
                          'seen':list(l['seen'])} for l in self.lanes]}

    def restore(self, state):
        self.archive = deepcopy(state['archive'])
        self.cursor, self.paused = state['cursor'], state['paused']
        self.lanes = [{'requirement':deepcopy(l['requirement']), 'roots':deque(l['roots']),
                      'continuations':deque((s[0], tuple(s[1]), tuple(s[2]), s[3], s[4])
                                            for s in l['continuations']),
                      'turns':l['turns'], 'seen':{(s[0], tuple(s[1])) for s in l['seen']}}
                     for l in state['lanes']]

    def run(self, initial_pool):
        self.initialize(initial_pool)
        while self.step() is not None:
            pass
        self.archive['final_ann_calls'] = self.retriever.ann_calls
        self.archive['pending_states'] = self._pending()
        self.emit('proxy_free_stopped', reason=self.archive['stop_reason'],
                  final_ann_calls=self.retriever.ann_calls, pending_states=self.archive['pending_states'],
                  scored_sets=0)
        return deepcopy(self.archive)

    def _next(self, lane):
        while lane['roots'] or lane['continuations']:
            take_continuation = bool(lane['continuations']) and (lane['turns'] % 2 == 1 or not lane['roots'])
            if take_continuation:
                state=lane['continuations'].popleft()
            else:
                target=lane['roots'].popleft()
                state=(target,(),(target,),'initial_ANN_rank',None)
            if (state[0],state[1]) not in lane['seen']:
                return state
        return None

    def _pending(self):
        return [{'requirement_id':lane['requirement']['id'],'roots':list(lane['roots']),
                 'continuations':[{'target_id':s[0],'premise_ids':list(s[1]),'navigation_path':list(s[2])}
                                  for s in lane['continuations']]} for lane in self.lanes]

    def partial_public_dict(self, *, detail='', stop_reason='execution_error'):
        return {**deepcopy(self.archive),'stop_reason':stop_reason,'detail':detail,
                'final_ann_calls':self.retriever.ann_calls,'pending_states':self._pending()}
