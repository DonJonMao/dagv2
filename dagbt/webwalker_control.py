"""Web-only scheduling over the unchanged initial DAG and reading frontiers.

``run`` follows the common controller verbatim except for step ordering,
completed-turn accounting and the existing-window yield hook. Source, fact,
certificate, recovery, audit and budget semantics remain in the common classes.
"""
from copy import deepcopy

from .budget import BudgetExceeded
from .reasoning import InputOverflow, ProtocolError, RefusalError
from .residual import ProgramError
from .residual_control import ResidualControl
from .transport import digest


class WebResidualControl(ResidualControl):
    def __init__(self, engine):
        super().__init__(engine)
        self._web_turn_clock = 0
        self._web_last_turn = {}
        self._web_yield_ready = set()
        self._web_resume_required = set()

    def _web_step_order(self):
        # A stable view, never an in-place reorder of the immutable planned DAG.
        return sorted(self.e.steps, key=lambda step: self._web_last_turn.get(step['output_slot'], 0))

    def _web_completed_turn(self, node_id):
        self._web_turn_clock += 1
        self._web_last_turn[node_id] = self._web_turn_clock
        self.e.event({'event': 'web_task_turn_completed', 'node_id': node_id,
                      'turn': self._web_turn_clock, 'initial_dag_unchanged': True})

    def read_progress(self, step, query, variables, doc_ids, identity, **options):
        first_read = len(getattr(self.reader, 'web_navigation_reads', []))
        before = self.e.ledger.used.get('llm', 0)
        changed = super().read_progress(step, query, variables, doc_ids, identity, **options)
        empty_reads = getattr(self.reader, 'web_navigation_reads', [])[first_read:]
        complete_empty = bool(empty_reads) and set(variables) <= {row['variable'] for row in empty_reads}
        has_frontier = any(key == identity for key, _ in self.reader.discovery_frontier)
        if complete_empty and has_frontier and step['execution'] == 'retrieval':
            self._web_yield_ready.add(identity)
        elif self.e.ledger.used.get('llm', 0) > before:
            self._web_yield_ready.discard(identity)
        return changed

    def web_read_existing(self, step, query, variables, doc_ids, identity, **options):
        # Alternate a deferred document window with a new observed branch.
        # A newly fetched page cannot consume the deferred window's resume turn.
        related_failed = any(record['variable'] in variables for record in self.reader.failed_batch_reviews.values())
        has_frontier = any(key == identity for key, _ in self.reader.discovery_frontier)
        bridge_has_links = getattr(self.e.bridge, 'has_unvisited_links', lambda *args, **kwargs: False)
        if (identity in self._web_yield_ready and identity not in self._web_resume_required
                and has_frontier and not related_failed
                and self.e.ledger.remaining('web_actions') > 0
                and self.e.ledger.remaining('llm') - self.e.s['reserved_audit_calls'] >= 2
                and bridge_has_links(step['output_slot'], identity=identity)):
            self._web_yield_ready.discard(identity)
            self._web_resume_required.add(identity)
            self.e.event({'event': 'web_document_window_yielded', 'node_id': step['output_slot'],
                          'query_identity': identity, 'reason': 'completed_empty_window_with_live_link_frontier',
                          'unoffered_sources_preserved': True})
            return False
        before = self.e.ledger.used.get('llm', 0)
        changed = self.read_progress(step, query, variables, doc_ids, identity, **options)
        if self.e.ledger.used.get('llm', 0) > before and identity in self._web_resume_required:
            self._web_resume_required.discard(identity)
            self.e.event({'event': 'web_document_window_resumed', 'node_id': step['output_slot'],
                          'query_identity': identity, 'original_frontier_preserved': True})
        return changed

    def run(self):
        if self.e.audit_complete and self.certificate is not None and self.certificate==self.candidate():
            self.stop_reason='certified'
            return self.publish()
        if self.e.audit_complete and self.stop_reason=='semantic_assessed':
            return self.publish()
        self.e.event({'event':'typed_dag_initialized','dag':deepcopy(self.e.steps),
            'final_node_id':self.e.final_node_id,'output_contract':self.state.contract,
            'F':self.state.initial_program,'planning_before_first_retrieval':True})
        while True:
            outcome,active=self.update()
            cert=self.candidate()
            ready=cert is not None and not self.forced_demands and (self.e.capabilities.residual_control or not active)
            semantic=self.state.contract['mode']=='semantic'
            values,_,_=self.state.valid()
            terminal_variables=self.demand_variables(self.e.final_node_id)
            semantic_ready=(semantic and not self.forced_demands and bool(terminal_variables) and self.reader.semantic_inputs is not None
                and self.reader.semantic_inputs['validation_complete'] and
                self.semantic_inputs_current())
            if ready or semantic_ready:
                binding_identity=self.semantic_binding_identity()
                try:
                    if cert and not self.terminal_format(cert):
                        self.stop_reason='invalid_output_format';break
                    if not self.audit(cert):
                        if self.recover_after_audit('audit_rejected'):
                            continue
                        self.stop_reason='audit_incomplete';break
                except (BudgetExceeded,InputOverflow,ProtocolError,ProgramError) as exc:
                    self.error('audit',exc);self.stop_reason='audit_incomplete';break
                # Audit may correct/revoke facts. Recompute; no old certificate.
                if cert is not None:
                    new=self.candidate()
                    if new!=cert:
                        self.e.audit_complete=False
                        self.e.event({'event':'certificate_invalidated_after_audit','old_identity':digest(cert)})
                        if self.recover_after_audit('certificate_changed'):
                            continue
                        self.stop_reason='audit_incomplete';break
                    self.certificate=new;self.stop_reason='certified';break
                values,_,_=self.state.valid()
                if (binding_identity!=self.semantic_binding_identity() or
                        not all(v in values for v in self.semantic_dependencies())):
                    if self.recover_after_audit('semantic_bindings_changed'):
                        continue
                    self.stop_reason='audit_incomplete';break
                self.semantic_audit_identity=self.semantic_publication_identity()
                self.stop_reason='semantic_assessed';break
            if outcome['status']=='inconsistent':
                self.stop_reason='inconsistent';break
            if outcome['status']=='determined' and not semantic and not active and cert is None:
                self.stop_reason='uncertifiable_output'
                self.e.event({'event':'determined_output_not_certifiable',
                    'mode':self.state.contract['mode'],'result_type':self.state.contract['result_type'],
                    'expression_op':self.state.program['expression']['op'],
                    'rules_complete':self.state.program.get('rules_complete',True),
                    'pending_variables':sorted(self.state.pending),
                    'reason':'No certificate and no remaining admissible evidence action; output is withheld.'})
                break
            executed=False
            for step in self._web_step_order():
                nid=step['output_slot']
                if nid not in active:
                    continue
                query=self.grounded(step)
                if query is None:
                    continue
                variables=self.demand_variables(nid)
                # Placeholder rule-dependent tasks are retained in the initial
                # DAG but can execute only after sourced instantiation.
                if not variables:
                    continue
                if any(v in self.state.program.get('task_expressions',{}) for v in variables):
                    continue
                _, sources, _=self.state.valid()
                parent_vars=[v for p in step['inputs'] for v in self.demand_variables(p)]
                compose=step['execution']=='compose'
                parent_docs=[] if compose else list(dict.fromkeys(
                    self.e.spans[s]['doc_id'] for v in parent_vars for s in sources.get(v,[])))
                identity={'algorithm':self.e.s['algorithm_version'],'query':query,'node_id':nid,
                    'program_revision':self.state.revision,'program_identity':digest(self.state.program),
                    'initial_dag_identity':digest(self.e.steps),'bindings':{v:self.state.versions[v] for v in parent_vars},
                    'contract_identity':digest(self.state.contract)}
                if self.audit_feedback is not None:
                    identity['audit_correction']=digest(self.audit_feedback)
                key=digest(identity)
                if key in self.exhausted:
                    continue
                if (step['execution']=='compose' and
                        digest([key,self.reader.material_signature()]) in self.compose_exhausted):
                    continue
                try:
                    before_read=self.reader.material_signature()
                    local_docs=list(dict.fromkeys(self.local_pools.get(nid,[])+parent_docs))
                    if self.web_read_existing(step,query,variables,local_docs,key,
                            semantic=semantic and nid==self.e.final_node_id and step['execution']=='compose',
                            terminal_output=nid==self.e.final_node_id and self.e.output_options is not None,
                            review_existing_obligations=self.review_existing_obligations(step,variables,key),
                            source_scope='mandatory_closure' if compose else 'document_windows',
                            **({} if compose else {'discovery_doc_ids':self.local_pools.get(nid,[])})):
                        self.forced_demands.discard(nid)
                        self.visited.update([nid] if self.local_complete(nid) else []);self._web_completed_turn(nid);executed=True;break
                    self.forced_demands.discard(nid)
                    if step['execution']=='compose':
                        blocked=digest([key,self.reader.material_signature()])
                        newly_blocked=blocked not in self.compose_exhausted
                        self.compose_exhausted.add(blocked)
                        if newly_blocked and before_read!=self.reader.material_signature():
                            # This read recreated an old unresolved state. Close
                            # that composition once, then let its newly pending
                            # parents advance their still-live bridge frontiers.
                            # New text/facts may later admit a different attempt.
                            self._web_completed_turn(nid);executed=True;break
                        continue
                    if self.e.ledger.remaining('llm')<=self.e.s['reserved_audit_calls']:
                        self.exhausted.add(key);continue
                    if self.reader.failed_review_blocked(step,variables):
                        # Extra ANN windows cannot repair a required failed
                        # batch after its repair budget is unavailable.
                        self.exhausted.add(key);continue
                    if self.e.fixed_pool is not None:
                        local=self.e.fixed_pool
                        if set(local)<=set(self.e.candidates) and all((key,a) in self.reader.read_states for a,s in self.e.mapper.sources.items() if s['doc_id'] in local):
                            self.exhausted.add(key);continue
                    else:
                        found=self.e.bridge.step(query,nid,[{'id':nid,'description':query,'necessary':True}],parent_docs,identity=identity)
                        self.e.discoveries.append(deepcopy(found));self.e.record_navigation(found)
                        local=found['local_candidate_ids']
                        if not local:
                            if found['trace'].get('search_status')=='empty_probe':
                                self._web_completed_turn(nid);executed=True;break
                            self.exhausted.add(key);continue
                    self.local_pools[nid]=list(dict.fromkeys(self.local_pools.get(nid,[])+local))
                    self.e.add_candidates(local)
                    self.read_progress(step,query,variables,local,key,semantic=semantic and nid==self.e.final_node_id,
                        terminal_output=nid==self.e.final_node_id and self.e.output_options is not None,
                        review_existing_obligations=False,discovery_doc_ids=local)
                    self.visited.update([nid] if self.local_complete(nid) else []);self._web_completed_turn(nid);executed=True;break
                except RefusalError:
                    raise
                except (BudgetExceeded,InputOverflow,ProtocolError,ProgramError,ConnectionError,ValueError) as exc:
                    self.error(nid,exc);self.exhausted.add(key)
            if not executed:
                self.stop_reason='budget_exhausted' if self.e.ledger.remaining('ann')==0 or self.e.ledger.remaining('llm')<=self.e.s['reserved_audit_calls'] else 'unresolved_or_exhausted'
                break
        # Free final evaluation always happens, including at budget zero.
        self.update()
        return self.publish()

