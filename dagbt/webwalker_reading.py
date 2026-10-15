"""Web-only source reading with explicit search gaps and repair headroom.

The common reader still owns facts, sources, counter capture and certification.
A search gap is a completed *empty-evidence* read, never a sourced unresolved
issue. Legacy ``unresolved`` responses retain their original conservative scope.
"""
from copy import deepcopy

from .joint_reading import JointReader
from .transport import digest


WEB_READING_INSTRUCTION = '''
Web navigation distinguishes finding no evidence from finding unresolved evidence.
For a page/window with NO relevant factual observation, return observations=[] and
unresolved=[]; optionally put untried navigation directions or missing search
targets in search_gaps. search_gaps is allowed ONLY on an otherwise empty-evidence
retrieval slot. It creates no fact, proof, source obligation or resolution claim.
Do not emit a source-cited unknown observation, or slot.unresolved, merely to say
that this page did not contain the answer. The next local bridge can continue.
If the original text establishes a partial fact, counter, conflict, ambiguity,
missing required part of an otherwise proposed answer, or another concrete
unresolved evidence issue, keep the ordinary sourced observations/unresolved
protocol. Never move such an issue to search_gaps. Existing evidence obligations,
failed drafts, frozen observations and source/counter review requirements remain
mandatory; search_gaps cannot discharge or reclassify them. Do not restate frozen
unknown/partial observations. An empty-evidence slot claims no review resolution.
search_gaps is forbidden during terminal composition and audit. A search gap is
neither evidence of a negative fact nor evidence that all directions are exhausted.
'''


class WebJointReader(JointReader):
    def __init__(self, engine, state):
        super().__init__(engine, state)
        self.web_search_gaps = []
        self.web_capacity_admissions = []
        self._web_headroom_candidates = {}
        self._selected_web_headroom = None

    @staticmethod
    def _search_gap_scope(data):
        return (data.get('task', {}).get('execution') == 'retrieval'
                and not any(key in data for key in
                            ('audit_instruction', 'public_output_question', 'terminal_repair')))

    @staticmethod
    def _system(data):
        return JointReader._system(data) + WEB_READING_INSTRUCTION

    @staticmethod
    def _schema(data):
        schema = JointReader._schema(data)
        if not WebJointReader._search_gap_scope(data):
            return schema
        slot = schema['properties']['slots']['items']
        slot['properties']['search_gaps'] = {
            'type': 'array', 'minItems': 1,
            'items': {'type': 'string', 'minLength': 1},
            'description': 'Untried search directions only; permitted with empty observations and unresolved.'}
        empty = {key: {'maxItems': 0} for key in
                 ('observations', 'unresolved', 'resolved_obligation_ids',
                  'resolved_failed_batch_ids', 'reviewed_observation_ids')
                 if key in slot['properties']}
        slot.setdefault('allOf', []).append({
            'if': {'required': ['search_gaps']},
            'then': {'properties': empty}})
        return schema

    @staticmethod
    def _gap_response(response, data):
        """Decode only a structurally valid empty-evidence extension.

        Invalid extensions remain on the original response. The core therefore
        captures its observation/counter drafts before rejecting the envelope,
        instead of an adapter exception bypassing that mandatory capture.
        """
        if (not WebJointReader._search_gap_scope(data) or not isinstance(response, dict)
                or set(response) - {'slots', 'final_prediction', 'semantic_answer', 'audit_verdict'}
                or not isinstance(response.get('slots'), list)):
            return response, []
        if ('final_prediction' in response
                and response['final_prediction'] is not None
                and not isinstance(response['final_prediction'], str)):
            return response, []
        if ('audit_verdict' in response
                and response['audit_verdict'] not in ('supported', 'unknown', 'conflict')):
            return response, []
        requested = set(data['variables'])
        seen, records = set(), []
        required = {'variable', 'observations', 'unresolved'}
        reviews = {'resolved_obligation_ids', 'resolved_failed_batch_ids', 'reviewed_observation_ids'}
        allowed = required | reviews | {'search_gaps'}
        for slot in response['slots']:
            if (not isinstance(slot, dict) or not required <= set(slot) or set(slot) - allowed
                    or not isinstance(slot.get('variable'), str)
                    or slot['variable'] not in requested or slot['variable'] in seen
                    or not isinstance(slot['observations'], list)
                    or not isinstance(slot['unresolved'], list)
                    or any(not isinstance(gap, str) for gap in slot['unresolved'])):
                return response, []
            seen.add(slot['variable'])
            for key in reviews & set(slot):
                values = slot[key]
                if (not isinstance(values, list) or any(not isinstance(value, str) for value in values)
                        or len(set(values)) != len(values)):
                    return response, []
            if 'search_gaps' not in slot:
                continue
            gaps = slot['search_gaps']
            if (not isinstance(gaps, list) or not gaps
                    or any(not isinstance(gap, str) or not gap.strip() for gap in gaps)
                    or slot['observations'] or slot['unresolved']
                    or any(slot.get(key) for key in reviews)):
                return response, []
            records.append({'variable': slot['variable'], 'search_gaps': deepcopy(gaps)})
        if seen != requested or not records:
            return response, []
        decoded = deepcopy(response)
        for slot in decoded['slots']:
            slot.pop('search_gaps', None)
        return decoded, records

    def _web_headroom(self, operation, data, *, fresh_sources=True):
        if (operation != 'joint_read' or not fresh_sources or not self._search_gap_scope(data)
                or data.get('repair_scope') or data.get('failed_batch_reviews')
                or self.e.s['max_repairs_per_request'] <= 0
                or not self.e.ledger.remaining('json_repairs')
                or self.e.ledger.remaining('llm') < self.e.s['reserved_audit_calls'] + 2):
            return None
        forecast = {**data, 'repair_scope': {
            'failed': [{'variable': variable, 'id': '*', 'feedback': 'x' * 240,
                        'observation': None,
                        'slot': {'variable': variable, 'observations': [], 'unresolved': []}}
                       for variable in data['variables']],
            'attempt': 1, 'previous_response_identity': '0' * 64,
            'invalid_response': ''}}
        count = self._estimate(operation + '_repair', forecast)
        output = self.e.s['reasoning_output_tokens']
        total = count + 2 * output + self.e.s['input_margin'] + 8
        return {'repair_forecast_input_tokens': count, 'unknown_draft_reserve': output,
                'repair_output_reserve': output, 'forecast_total_tokens': total,
                'context_tokens': self.e.s['context_tokens'],
                'within_budget': total <= self.e.s['context_tokens'],
                'policy': 'generic_repair_envelope_plus_one_unknown_draft_allowance_v1'}

    def _fresh_window_fits(self, operation, data, *, fresh_sources=True):
        if not super()._fresh_window_fits(operation, data, fresh_sources=fresh_sources):
            return False
        headroom = self._web_headroom(operation, data, fresh_sources=fresh_sources)
        if headroom is not None:
            self._web_headroom_candidates[digest(data)] = headroom
        return headroom is None or headroom['within_budget']

    def _fitting_prefix(self, operation, step, query, variables, pending, semantic, audit):
        self._web_headroom_candidates = {}
        self._selected_web_headroom = None
        try:
            aliases, consumed = super()._fitting_prefix(
                operation, step, query, variables, pending, semantic, audit)
            if consumed:
                data = self._payload(step, query, variables, aliases, semantic, audit=audit)
                identity = digest(data)
                if identity in self._web_headroom_candidates:
                    self._selected_web_headroom = (identity, self._web_headroom_candidates[identity])
            return aliases, consumed
        finally:
            self._web_headroom_candidates = {}

    def _preserve_unresolved_issues(self, response, data, identity):
        """A malformed Web envelope cannot erase an ordinary factual issue.

        Named observation drafts are captured by the common reader even when
        an envelope is invalid. Bare slot.unresolved strings have no draft ID,
        so preserve their conservative original-source scope before that
        rejection. No content-based interpretation, fact or resolution occurs.
        """
        slots = response.get('slots') if isinstance(response, dict) else None
        if not isinstance(slots, list):
            return
        reviews = {}
        for slot in slots:
            if (not isinstance(slot, dict) or not isinstance(slot.get('variable'), str)
                    or slot['variable'] not in data['variables']
                    or not isinstance(slot.get('unresolved'), list)):
                continue
            # Invalid non-string fields remain in the raw draft. Identifiable
            # string issues are retained even in a malformed containing list.
            issues = [issue for issue in slot['unresolved'] if isinstance(issue, str) and issue]
            if issues:
                review = reviews.setdefault(slot['variable'], {'observations': [], 'unresolved': []})
                review['unresolved'].extend(issues)
        if reviews:
            self._update_evidence_obligations(
                reviews, [], data,
                {(variable, '*'): 'Web response retains ordinary unresolved evidence before envelope validation'
                 for variable in reviews}, identity)

    def _request_with_exposure(self, operation, data, identity, wire, system, **options):
        # Reuse admission's measurement, never contact the tokenizer before
        # the common reader has registered attempted exposure or after a
        # successful response. Logging is not another fallible admission step.
        raw = super()._request_with_exposure(operation, data, identity, wire, system, **options)
        # Ordinary unresolved issues have no observation draft IDs. Retain
        # their source scope even if an outer/duplicate-slot error is repaired
        # by an empty response. Valid envelopes produce the same obligation ID
        # in the common update below; this early capture cannot resolve it.
        self._preserve_unresolved_issues(raw, data, identity)
        selection = self._selected_web_headroom
        if selection is not None and selection[0] == digest(data):
            headroom = selection[1]
            admission = {'query_identity': identity, 'task_id': data['task']['output_slot'],
                         'source_count': len(data['source_spans']), **headroom}
            self.web_capacity_admissions.append(admission)
            self.e.event({'event': 'web_reading_capacity_admitted', **deepcopy(admission)})
        decoded, records = self._gap_response(raw, data)
        for record in records:
            diagnostic = {**record, 'query_identity': identity,
                          'task_id': data['task']['output_slot'],
                          'request_identity': digest(data),
                          'logical_call_index': self.e.reasoner.sequence,
                          'source_ids': [self.e.mapper.sources[source['id']]['source_id']
                                         for source in data['source_spans']]}
            self.web_search_gaps.append(diagnostic)
            self.e.event({'event': 'web_search_gap_observed', **deepcopy(diagnostic)})
        # Reasoner.last_response and archived raw_output are never rewritten.
        return decoded

    def public_dict(self):
        # These archives are diagnostic only. In particular, do not include
        # them in obligation_snapshot(): the core hashes that behavioral state
        # when deciding whether a read made progress.
        return {**super().public_dict(),
                'web_reading_protocol': 'explicit_empty_evidence_search_gap_v1',
                'web_search_gaps': deepcopy(self.web_search_gaps),
                'web_capacity_admissions': deepcopy(self.web_capacity_admissions)}
