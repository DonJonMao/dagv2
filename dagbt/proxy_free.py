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

    def run(self, initial_pool):
        self.archive.update(initial_target_ids=list(initial_pool.candidate_ids),
                            initial_ann_calls=self.retriever.ann_calls,
                            initial_scored_sets=0,final_scored_sets=0)
        # Initial dense hits and each expansion batch retain their actual ANN
        # rank order. Across batches, stable discovery order is the tie-break.
        roots = tuple(initial_pool.candidate_ids)
        self.lanes = [{'requirement':r, 'roots':deque(roots), 'continuations':deque(),
                       'turns':0, 'seen':set()} for r in self.requirements]
        self.emit('proxy_free_started', requirement_ids=[r['id'] for r in self.requirements],
                  initial_target_ids=list(roots), ann_cap=self.retriever.max_ann_calls,
                  proposal_order='necessary_requirement_round_robin_then_local_ANN_rank',
                  root_continuation_schedule='alternate_when_both_available')
        while self.retriever.remaining_ann_calls is None or self.retriever.remaining_ann_calls > 0:
            progressed = False
            for lane in self.lanes:
                if self.retriever.remaining_ann_calls == 0:
                    break
                state = self._next(lane)
                if state is None:
                    continue
                target, premises, path, origin, rank = state
                lane['seen'].add((target,premises))
                lane['turns'] += 1
                self.retriever.set_information_needs([lane['requirement']])
                self.emit('proxy_free_conditional_started', requirement_id=lane['requirement']['id'],
                          target_id=target,premise_ids=list(premises),navigation_path=list(path),
                          origin=origin,local_ANN_rank=rank,requirement_turn=lane['turns'])
                before = self.retriever.ann_calls
                proposal = self.retriever.propose(target, premises, fixed_pool=False)
                public = proposal.public_dict()
                self.archive['proposal_batches'].append(public)
                self.archive['states'].append({'requirement_id':lane['requirement']['id'],
                    'target_id':target,'premise_ids':list(premises),'navigation_path':list(path),
                    'probe_id':proposal.probe_id,'candidate_ids':list(proposal.ids)})
                descendants=[]
                for hit in proposal.hits:  # Source response is already in actual ANN rank order.
                    candidate=hit.memory_id
                    if candidate in path:
                        continue
                    new_premises=tuple(dict.fromkeys((*premises,target)))
                    key=(candidate,new_premises)
                    if key not in lane['seen']:
                        descendants.append((candidate,new_premises,(*path,candidate),'conditional_hit',hit.rank))
                # Highest-ranked continuation runs before lower-ranked siblings;
                # alternating fresh roots prevents a single chain taking every turn.
                for descendant in reversed(descendants):
                    lane['continuations'].appendleft(descendant)
                progressed = True
                self.emit('proxy_free_conditional_completed', requirement_id=lane['requirement']['id'],
                          probe_id=proposal.probe_id,target_id=target,premise_ids=list(premises),
                          candidate_ids=list(proposal.ids),new_navigation_states=len(descendants),
                          completed_ann_calls=self.retriever.ann_calls-before,
                          evidence_dependency_claim=False)
            if not progressed:
                break
        self.archive['stop_reason'] = ('ann_budget_exhausted' if self.retriever.remaining_ann_calls == 0
                                       else 'finite_frontier_exhausted')
        self.archive['final_ann_calls']=self.retriever.ann_calls
        self.archive['pending_states']=self._pending()
        self.emit('proxy_free_stopped', reason=self.archive['stop_reason'],
                  final_ann_calls=self.retriever.ann_calls,pending_states=self.archive['pending_states'],
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
