"""V3 raw-source review with DAG certificates and scoped coverage recovery.

Adapted from BridgeTree c4b04c9's raw review / evidence_recovery contract.
Original raw documents may be selected without a mapping. They never acquire
fabricated support; the actual chosen set gets a fresh DAG closure certificate.
"""
from copy import deepcopy

from . import prompts
from .budget import BudgetExceeded
from .evidence_spans import document_source_metadata
from .evidence_views import build_view
from .reasoning import ProtocolError, InputOverflow, OutputTruncated, RefusalError
from .support import select_support, _protected_groups
from .transport import digest


def _ids(value, name):
    if not isinstance(value, list) or any(not isinstance(x, str) or not x for x in value) or len(set(value)) != len(value):
        raise ProtocolError(name + ' must contain unique nonempty IDs', category='selection_header')
    return value


def _object(properties):
    return {'type': 'object', 'additionalProperties': False, 'required': list(properties), 'properties': properties}


STRING = {'type': 'string'}
STRINGS = {'type': 'array', 'items': STRING}
SELECT_SCHEMA = _object({'selected_doc_ids': STRINGS, 'reason': STRING, 'conflicts': STRINGS,
    'coverage': {'type': 'array', 'items': _object({'requirement_id': STRING,
        'status': {'enum': ['covered', 'partial', 'missing', 'ambiguous']},
        'source_span_ids': STRINGS, 'kind': {'enum': ['explicit', 'inference']}, 'reason': STRING})}})


def independent_assessments(facts):
    """Connected source fragments of different judgments are not independent."""
    parent = list(range(len(facts)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i, fact in enumerate(facts):
        for j, earlier in enumerate(facts[:i]):
            if fact['doc_id'] != earlier['doc_id']:
                continue
            first, second = fact.get('fragments', [fact]), earlier.get('fragments', [earlier])
            if any(set(a.get('premise_group_ids', ())) & set(b.get('premise_group_ids', ()))
                   or a['start'] < b['end'] and b['start'] < a['end'] for a in first for b in second):
                parent[root(i)] = root(j)
    return len({root(i) for i in range(len(facts))})


class FinalSelector:
    def __init__(self, engine, graph, proposal):
        self.e, self.graph, self.proposal = engine, graph, proposal
        self.requirements = [{'id': step['output_slot'], 'description': step['question'],
                              'inputs': step['inputs']} for step in engine.steps]
        self.header = None
        self.valid = {}
        self.valid_input = {}
        self.envelope_errors = []
        self.failures = []
        self.actions = []
        self.view = None

    def certificate(self, selected, max_docs=20):
        allowed = set(selected)
        def feasible(ids):
            result = self.e.feasible(ids, max_docs)
            return {**result, 'feasible': result['feasible'] and set(ids) <= allowed}
        return select_support(self.graph, feasible, max_states=self.e.s['max_enumeration_states'])

    def _node_closure(self, node_id, selected):
        established = set()
        spans = self.e.spans
        for node in self.graph['nodes']:
            if node['status'] == 'supported' and any(alt.get('eligible')
                and set(alt['used_parent_ids']) <= established
                and {spans[s]['doc_id'] for s in alt['source_span_ids'] + alt['guard_span_ids']} <= set(selected)
                for alt in node['alternatives']):
                established.add(node['id'])
        return node_id in established

    def prepare_view(self):
        e = self.e
        costs = {d: len(e.tokenizer.encode(e.docs[d].passage, add_special_tokens=False)) for d in e.candidates}
        raw = [{'doc_id': d, 'passage': e.docs[d].passage, 'metadata': document_source_metadata(e.docs[d])}
               for d in e.baseline_ids] if e.s['raw_memory_review'] else []
        fixed = {'original_question': e.q['question'], 'requirements': self.requirements,
                 'selection_mode': e.s['selection'], 'max_documents': 20,
                 'reader_context_budget': e.s['context_tokens'], 'reader_output_reserve': e.s['reader_output_tokens'],
                 'proposed_doc_ids': [], 'raw_memory_candidates': [],
                 'mapping_incomplete': e.mapper.diagnostics()['mapping_incomplete']}
        limit = e.s['context_tokens'] - e.s['reasoning_output_tokens'] - e.s['input_margin'] - 8 - 384
        def build(records, raw_rows, emit=False):
            current = {**fixed, 'raw_memory_candidates': raw_rows}
            raw_ids = [r['doc_id'] for r in raw_rows]
            # The proposal is advisory. Never expose a selectable ID only via
            # an invisible model-derived graph summary.
            visible_docs = set(raw_ids) | {r['doc_id'] for r in records}
            current['proposed_doc_ids'] = [d for d in self.proposal.get('selected_doc_ids', []) if d in visible_docs]
            current['protected_document_groups'] = [g['doc_ids'] for g in _protected_groups(self.graph)]
            supplied = {r['id'] for r in records}
            nodes = deepcopy(self.graph['nodes'])
            for node in nodes:
                node['alternatives'] = [a for a in node['alternatives'] if set(a['source_span_ids'] + a['guard_span_ids']) <= supplied]
            return build_view(e.reasoner, 'select', prompts.SELECT_V3, current, records, e.s,
                e.event if emit else None, SELECT_SCHEMA,
                audit_nodes=nodes if e.s['selection'] == 'dependency' else None,
                node_order=[r['id'] for r in self.requirements], document_costs=costs,
                raw_doc_ids=raw_ids, input_limit=limit)
        kept = []
        build([], [])  # Fixed-header overflow is a real failure, not an empty review.
        for record in raw:
            try:
                build([], kept + [record])
            except InputOverflow:
                continue
            kept.append(record)
        view = build(list(e.spans.values()), kept, True)
        # build_view may omit ledger records. A proposal naming such a document
        # is removed, and the final exact payload is re-estimated below.
        view.data['proposed_doc_ids'] = [d for d in view.data['proposed_doc_ids'] if d in view.visible_doc_ids]
        view.audit['input_tokens_after'] = e.reasoner.estimate('select', prompts.SELECT_V3, view.data, SELECT_SCHEMA)
        visible_raw = [r['doc_id'] for r in kept]
        omitted_raw = [r['doc_id'] for r in raw if r['doc_id'] not in visible_raw]
        view.audit.update(policy_version='dense_raw_then_atomic_dag_ledger_v3',
                          baseline_doc_ids=list(e.baseline_ids), raw_review_visible_doc_ids=visible_raw,
                          omitted_raw_review_doc_ids=omitted_raw,
                          input_truncated=bool(view.audit['truncated'] or omitted_raw))
        e.input_views.append(deepcopy(view.audit))
        e.event({'event': 'raw_review_prepared', **view.audit})
        self.view = view
        return view

    def _header(self, value):
        if not isinstance(value, dict) or set(value) != {'selected_doc_ids', 'reason', 'conflicts', 'coverage'}:
            raise ProtocolError('Selection requires a complete global header and coverage field', category='selection_header')
        ids = _ids(value['selected_doc_ids'], 'selected_doc_ids')
        if not set(ids) <= self.view.visible_doc_ids:
            raise ProtocolError('Selection cites unknown or undisplayed documents', category='selection_header')
        if not isinstance(value['reason'], str) or not value['reason'].strip():
            raise ProtocolError('Selection reason must be nonempty', category='selection_header')
        _ids(value['conflicts'], 'conflicts')
        if not self.e.feasible(ids)['feasible']:
            raise ProtocolError('Final whole-document reader context is infeasible', category='reader_budget')
        if self.e.s['selection'] == 'dependency':
            for group in _protected_groups(self.graph):
                if set(ids) & set(group['doc_ids']) and not set(group['doc_ids']) <= set(ids):
                    raise ProtocolError('A disputed source group must be retained together', category='selection_header')
            for d in ids:
                if not set(self.graph.get('navigation_closure', {}).get('ancestors', {}).get(d, [])) <= set(ids):
                    raise ProtocolError('Navigation ablation requires its complete source closure', category='selection_header')
        header = {k: deepcopy(value[k]) for k in ('selected_doc_ids', 'reason', 'conflicts')}
        if self.header is not None and header != self.header:
            raise ProtocolError('Coverage repair changed the fixed selection header', category='selection_header')
        return header

    def _coverage(self, row):
        keys = {'requirement_id', 'status', 'source_span_ids', 'kind', 'reason'}
        if not isinstance(row, dict) or set(row) != keys:
            raise ProtocolError('Malformed coverage row', category='coverage')
        rid = row['requirement_id']
        if not isinstance(rid, str) or rid not in {r['id'] for r in self.requirements}:
            raise ProtocolError('Unknown coverage requirement', category='coverage')
        if row['status'] not in {'covered', 'partial', 'missing', 'ambiguous'} or row['kind'] not in {'explicit', 'inference'}:
            raise ProtocolError('Invalid coverage status/kind', category='coverage')
        if not isinstance(row['reason'], str) or not row['reason'].strip():
            raise ProtocolError('Coverage reason missing', category='coverage')
        decoded = self.view.decode(row)
        refs = decoded['source_span_ids']
        facts = [self.e.spans[s] for s in refs]
        if any(rid not in f['node_ids'] or f['doc_id'] not in self.header['selected_doc_ids'] for f in facts):
            raise ProtocolError('Coverage references wrong-node or nonselected evidence', category='coverage')
        if (row['status'] == 'missing') != (not facts):
            raise ProtocolError('Missing coverage requires no evidence; other statuses require evidence', category='coverage')
        supportive = [f for f in facts if f['stance'] == 'support']
        partial = [f for f in facts if f['stance'] == 'partial']
        if row['status'] == 'covered':
            if not supportive and (len(partial) < 2 or independent_assessments(partial) < 2):
                raise ProtocolError('Covered requires support or two independent partial assessments', category='coverage')
            if self.e.s['selection'] == 'dependency' and not self._node_closure(rid, self.header['selected_doc_ids']):
                raise ProtocolError('Covered requires the actual selected DAG support closure', category='coverage')
        kind = row['kind']; why = None
        if kind == 'explicit' and any(f['kind'] == 'implicit' for f in facts):
            kind, why = 'inference', 'cited_implicit_assessment'
        elif kind == 'explicit' and row['status'] == 'covered' and not supportive:
            kind, why = 'inference', 'joint_partial_inference'
        return {**decoded, 'kind': kind, 'declared_kind': row['kind'], 'normalization_reason': why,
                'validation_complete': True}

    def _accept_rows(self, value):
        pending = [r['id'] for r in self.requirements if r['id'] not in self.valid]
        errors = {}
        self.envelope_errors = []
        rows = value['coverage']
        if not isinstance(rows, list):
            self.envelope_errors.append('Coverage must be a list')
            return {rid: 'Coverage must be a list' for rid in pending}
        by_id = {}
        for row in rows:
            rid = row.get('requirement_id') if isinstance(row, dict) else None
            if not isinstance(rid, str):
                self.envelope_errors.append('Unknown or malformed coverage requirement row')
                continue
            if rid in self.valid_input:
                if row != self.valid_input[rid]:
                    self.envelope_errors.append('Previously validated coverage was changed: '+rid)
                continue
            if rid not in pending:
                self.envelope_errors.append('Unknown or malformed coverage requirement row')
                continue
            by_id.setdefault(rid, []).append(row)
        for rid in pending:
            items = by_id.get(rid, [])
            if len(items) != 1:
                errors[rid] = 'Missing or duplicate coverage row'; continue
            try:
                self.valid[rid] = self._coverage(items[0])
                self.valid_input[rid] = deepcopy(items[0])
            except (ValueError, KeyError, TypeError) as exc:
                errors[rid] = str(exc)
        return errors

    def compact_repair(self, errors):
        pending = list(errors)
        selected = self.header['selected_doc_ids']
        allowed = {alias: sid for alias, sid in self.view.span_alias_to_id.items()
                   if self.e.spans[sid]['doc_id'] in selected
                   and any(rid in self.e.spans[sid]['node_ids'] for rid in pending)}
        data = {'original_question': self.e.q['question'],
                'requirements': [r for r in self.requirements if r['id'] in pending],
                'candidate_doc_ids': selected,
                'evidence': [row for row in self.view.data['evidence'] if row['id'] in allowed],
                'allowed_evidence_by_requirement': {rid: [a for a, s in allowed.items() if rid in self.e.spans[s]['node_ids']] for rid in pending},
                'mapping_incomplete': self.e.mapper.diagnostics()['mapping_incomplete'],
                'repair_scope': {'fixed_header': deepcopy(self.header), 'pending_requirement_ids': pending,
                    'errors': {rid: text[:350] for rid, text in errors.items()},
                    'coverage_envelope_errors': self.envelope_errors[:8],
                    'instruction': 'Return the fixed header verbatim and ONLY pending coverage rows. If none pending, return coverage=[] to correct only the malformed envelope. No raw source selection changes.'}}
        # Retain only supported/unsupported status of a full selected route,
        # never a cropped proof masquerading as a complete parent conclusion.
        data['selected_dag_closure_by_requirement'] = {rid: self._node_closure(rid, selected) for rid in pending}
        return data

    def _snapshot(self, phase, unresolved=()):
        e=self.e
        selected=[] if self.header is None else list(self.header['selected_doc_ids'])
        mapped={sp['doc_id'] for sp in e.spans.values()}
        mapped_selected=[d for d in selected if d in mapped]
        raw_selected=[d for d in selected if d not in mapped]
        e.semantic_evidence={'version':'dagbt_semantic_evidence_v3','phase':phase,
            'evidence_state':'unknown' if self.header is None else 'empty_context' if not selected else
                'raw_only' if not mapped_selected else 'mapped_only' if not raw_selected else 'mixed',
            'selected_doc_ids':selected,'selected_mapped_doc_ids':mapped_selected,'selected_raw_only_doc_ids':raw_selected,
            'baseline_doc_ids':list(e.baseline_ids),'candidate_doc_ids':list(e.candidates),
            'raw_review_visible_doc_ids':self.view.audit['raw_review_visible_doc_ids'],
            'omitted_raw_review_doc_ids':self.view.audit['omitted_raw_review_doc_ids'],
            'coverage_validation_complete':None if self.header is None else False,
            'unassessed_requirement_ids':list(unresolved),'coverage_rows':list(self.valid.values()),
            'coverage_envelope_errors':list(self.envelope_errors),
            'coverage_failures':deepcopy(self.failures),'recovery_actions':deepcopy(self.actions),
            'evidence_logical_calls':len(e.reasoner.requests),'budgeted_llm_attempts':e.ledger.used['llm'],
            'budgeted_reader_attempts':e.ledger.used['reader']}
        e.event({'event':'selection_progress',**deepcopy(e.semantic_evidence)})

    def _subselection(self, selected, maximum):
        if maximum >= len(selected):
            return list(selected)
        chosen=set(self.certificate(selected,maximum)['selected_doc_ids'])
        groups=[set(g['doc_ids']) for g in _protected_groups(self.graph)] if self.e.s['selection']=='dependency' else []
        ancestors=self.graph.get('navigation_closure',{}).get('ancestors',{})
        def protect(ids):
            result=set(ids)
            while True:
                before=set(result)
                for group in groups:
                    if result & group:result.update(group)
                for d in list(result):result.update(ancestors.get(d,[]))
                if result==before:return result
        for d in selected:
            trial=protect(chosen|{d})
            ordered=[i for i in selected if i in trial]
            if trial<=set(selected) and self.e.feasible(ordered,maximum)['feasible']:
                chosen=trial
        return [d for d in selected if d in chosen]

    def run(self):
        e = self.e
        self.prepare_view()
        self._snapshot('selection_pending')
        current = self.view.data
        repairs = 0
        unresolved = {}; cause = None
        while True:
            try:
                value = e.reasoner.request('select' if repairs == 0 else 'select_repair', prompts.SELECT_V3,
                                           current, schema=SELECT_SCHEMA, reserve=0)
            except (BudgetExceeded, InputOverflow) as exc:
                if self.header is None or not (unresolved or self.envelope_errors) or 'repair_scope' not in current:
                    raise
                cause = exc
                self.actions.append({'event': 'coverage_repair_failed', 'error_type': type(exc).__name__, 'error': str(exc)})
                e.event(self.actions[-1])
                self._snapshot('coverage_incomplete', unresolved)
                break
            try:
                header = self._header(value)
            except ProtocolError as exc:
                # A legal header is never replaced during annotation repair.
                # Initial malformed IDs/budget may get a bounded whole-set revision.
                if self.header is not None or repairs >= e.s['max_repairs_per_request'] or e.ledger.remaining('json_repairs') == 0:
                    raise
                current = deepcopy(self.view.data)
                current['selection_revision'] = {'error': str(exc)[:300], 'instruction': 'Return a legal whole-document set and all coverage rows.'}
                count = e.reasoner.estimate('select_repair', prompts.SELECT_V3, current, SELECT_SCHEMA)
                e.reasoner._preflight('select_repair', count, reserve=0)
                e.ledger.reserve('json_repairs', 'select_revision'); repairs += 1
                e.event({'event': 'selection_revision', 'error': str(exc), 'input_tokens': count})
                continue
            self.header = header
            self._snapshot('coverage_pending',[r['id'] for r in self.requirements if r['id'] not in self.valid])
            unresolved = self._accept_rows(value)
            if not unresolved and not self.envelope_errors:
                break
            cause = ProtocolError('Coverage rows remain unverified', category='coverage')
            self.failures.append({'errors': deepcopy(unresolved), 'coverage_envelope_errors':list(self.envelope_errors), 'response_ref': e.reasoner.last_response.get('response_ref')})
            self._snapshot('coverage_incomplete',unresolved)
            if repairs >= e.s['max_repairs_per_request'] or e.ledger.remaining('json_repairs') == 0:
                break
            current = self.compact_repair(unresolved)
            count = e.reasoner.estimate('select_repair', prompts.SELECT_V3, current, SELECT_SCHEMA)
            action = {'event': 'coverage_repair_prepared', 'pending_requirement_ids': list(unresolved),
                      'selected_doc_ids': list(header['selected_doc_ids']), 'input_tokens': count,
                      'visible_evidence_ids': [r['id'] for r in current['evidence']], 'raw_records_resent': 0}
            try:
                e.reasoner._preflight('select_repair', count, reserve=0)
            except (InputOverflow, BudgetExceeded) as exc:
                cause = exc
                action.update(status='not_sent', error_type=type(exc).__name__, error=str(exc))
                self.actions.append(action); e.event(action)
                self._snapshot('coverage_incomplete', unresolved)
                break
            action['status'] = 'sent'; self.actions.append(action); e.event(action)
            self._snapshot('coverage_repair_pending', unresolved)
            e.ledger.reserve('json_repairs', 'select_coverage_repair'); repairs += 1
        incomplete=bool(unresolved or self.envelope_errors)
        if incomplete and not e.s['allow_unassessed_coverage']:
            raise cause
        rows = []
        for requirement in self.requirements:
            rid = requirement['id']
            rows.append(self.valid.get(rid, {'requirement_id': rid, 'status': 'unassessed',
                'source_span_ids': [], 'kind': None, 'validation_complete': False,
                'reason': unresolved.get(rid, 'Unverified'), 'failure_type': type(cause).__name__,
                'failure_detail': str(cause)}))
        selected = self.header['selected_doc_ids']
        # Preserve corpus observation order in final context, independent of a
        # model's ranking order. The ranking itself remains in the global header.
        selected = [d for d in e.docs if d in selected]
        selections = {}
        for k in (5, 10, 20):
            chosen = self._subselection(selected,k)
            certificate = self.certificate(chosen, k)
            selections[str(k)] = {**certificate, 'selected_doc_ids': chosen,
                'status': 'reviewed' if not incomplete else 'coverage_unassessed',
                'token_count': e.feasible(chosen, k)['token_count'], 'selection_review': True,
                'coverage': rows if k == 20 else [], 'coverage_validation_complete': not incomplete if k == 20 else None,
                'reason': self.header['reason'], 'verified_support_doc_ids': certificate['selected_doc_ids']}
        mapped = {s['doc_id'] for s in e.spans.values()}
        mapped_selected = [d for d in selected if d in mapped]
        raw_selected = [d for d in selected if d not in mapped]
        fully = {d for d in e.candidates if all(i in e.mapped_chunks for i, c in enumerate(e.chunks) if c['doc_id'] == d)}
        visible = set(self.view.visible_doc_ids)
        dispositions = [{'doc_id': d, 'baseline': d in e.baseline_ids, 'fully_mapped': d in fully,
            'eligible': d in mapped, 'raw_review_visible': d in self.view.audit['raw_review_visible_doc_ids'],
            'selection_visible': d in visible, 'selected': d in selected,
            'destination': 'selected_mapped' if d in mapped_selected else 'selected_raw_only' if d in raw_selected
                else 'visible_not_selected' if d in visible else 'omitted_from_selection_input'} for d in e.candidates]
        e.semantic_evidence = {'version': 'dagbt_semantic_evidence_v3',
            'evidence_state': 'empty_context' if not selected else 'raw_only' if not mapped_selected else 'mapped_only' if not raw_selected else 'mixed',
            'selected_doc_ids': selected, 'selected_mapped_doc_ids': mapped_selected, 'selected_raw_only_doc_ids': raw_selected,
            'baseline_doc_ids': list(e.baseline_ids), 'candidate_doc_ids': list(e.candidates),
            'fully_mapped_doc_ids': sorted(fully), 'eligible_doc_ids': sorted(mapped),
            'raw_review_visible_doc_ids': self.view.audit['raw_review_visible_doc_ids'],
            'omitted_raw_review_doc_ids': self.view.audit['omitted_raw_review_doc_ids'],
            'baseline_retention_ratio': len(set(selected) & set(e.baseline_ids)) / len(e.baseline_ids) if e.baseline_ids else None,
            'coverage_validation_complete': not incomplete, 'unassessed_requirement_ids': list(unresolved),
            'coverage_envelope_unassessed':bool(self.envelope_errors),'coverage_envelope_errors':list(self.envelope_errors),
            'coverage_rows': rows, 'coverage_failures': self.failures, 'recovery_actions': self.actions,
            'candidate_dispositions': dispositions, 'evidence_logical_calls': len(e.reasoner.requests),
            'budgeted_llm_attempts': e.ledger.used['llm'], 'budgeted_reader_attempts': e.ledger.used['reader']}
        e.event({'event': 'semantic_selection_complete', **deepcopy(e.semantic_evidence)})
        return selections
