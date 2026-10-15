"""Web-only source reading with explicit search gaps and repair headroom.

The common reader still owns facts, sources, counter capture and certification.
A search gap is a completed *empty-evidence* read, never a sourced unresolved
issue. Legacy ``unresolved`` responses retain their original conservative scope.
"""
from copy import deepcopy

from .joint_reading import JointReader
from .transport import digest


WEB_READING_INSTRUCTION = '''
REQUIRED WEB RETRIEVAL PROTOCOL: each requested slot must declare reading_status.
Use no_relevant_evidence when this window has no relevant factual observation:
observations=[], unresolved=[], no resolution/review IDs. search_gaps optionally
lists untried directions. Example for requested variable v, when the page only
contains unrelated navigation:
{"slots":[{"variable":"v","reading_status":"no_relevant_evidence","observations":[],"unresolved":[],"search_gaps":["Follow another observed category link."]}]}
No-relevant-evidence is a search result, never a negative fact or proof of absence.
Use evidence_assessment for one or more original-source-cited ordinary observations.
A concrete incomplete statement, ambiguity or conflict must be a named partial/
unknown observation with its exact source_ids and reason, not bare slot.unresolved.
For BOTH statuses unresolved MUST be []. Do not cite a whole window as unknown
merely because it does not contain the answer. The next local bridge can continue.
Only a genuine textual issue needs a sourced partial/unknown observation.
Existing evidence obligations, failed drafts and counters remain mandatory. To
resolve a previously offered issue, use evidence_assessment and explicitly review
its original text with sourced supported/corrective/irrelevant observations and
the offered resolution ID. A genuinely irrelevant original-source review may
cover an earlier absence-only issue; never label a real partial fact, supporting
text or counter irrelevant just to close it. Cover every required original source.
An empty slot never resolves a prior issue. Preserve every required failed draft ID.
Do not restate frozen observations. In a repair, evidence_assessment may instead
use the currently offered reviewed_observation_ids for immutable sourced rows;
those existing references still require the common source and resolution checks.
No-relevant-evidence forbids these review/resolution claims. search_gaps is only
allowed for no_relevant_evidence. Terminal composition and audit use the ordinary
protocol without reading_status or search_gaps.
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
        system = JointReader._system(data)
        return system + WEB_READING_INSTRUCTION if WebJointReader._search_gap_scope(data) else system

    @staticmethod
    def _schema(data):
        schema = JointReader._schema(data)
        if not WebJointReader._search_gap_scope(data):
            return schema
        slot = schema['properties']['slots']['items']
        slot['properties']['reading_status'] = {
            'type': 'string', 'enum': ['no_relevant_evidence', 'evidence_assessment']}
        slot['required'].append('reading_status')
        slot['properties']['unresolved']['maxItems'] = 0
        slot['properties']['observations']['items']['properties']['source_ids']['minItems'] = 1
        slot['properties']['search_gaps'] = {
            'type': 'array', 'minItems': 1,
            'items': {'type': 'string', 'minLength': 1},
            'description': 'Untried search directions only; permitted with empty observations and unresolved.'}
        empty = {key: {'maxItems': 0} for key in
                 ('observations', 'unresolved', 'resolved_obligation_ids',
                  'resolved_failed_batch_ids', 'reviewed_observation_ids')
                 if key in slot['properties']}
        evidence = [{'properties': {'observations': {'minItems': 1}}}]
        if 'reviewed_observation_ids' in slot['properties']:
            evidence.append({'required': ['reviewed_observation_ids'],
                             'properties': {'reviewed_observation_ids': {'minItems': 1}}})
        slot.setdefault('allOf', []).append({
            'if': {'properties': {'reading_status': {'const': 'no_relevant_evidence'}}},
            'then': {'properties': empty},
            'else': {'anyOf': evidence, 'not': {'required': ['search_gaps']}}})
        return schema

    @staticmethod
    def _gap_response(response, data):
        """Decode the required typed retrieval disposition without new facts.

        Invalid extensions remain on the original response. The core therefore
        captures its observation/counter drafts before rejecting the envelope,
        instead of an adapter exception bypassing that mandatory capture.
        """
        if not WebJointReader._search_gap_scope(data):
            return response, []
        rejected = deepcopy(response)
        if isinstance(rejected, dict) and isinstance(rejected.get('slots'), list):
            for slot in rejected['slots']:
                if isinstance(slot, dict) and 'reading_status' not in slot:
                    # The unmodified legacy shape would otherwise pass core
                    # validation. An undeclared adapter marker rejects it only
                    # AFTER the core has captured every named/counter draft.
                    slot['web_reading_protocol_error'] = 'Required reading_status is missing'
        if (not isinstance(response, dict)
                or set(response) - {'slots', 'final_prediction', 'semantic_answer', 'audit_verdict'}
                or not isinstance(response.get('slots'), list)):
            return rejected, []
        if ('final_prediction' in response
                and response['final_prediction'] is not None
                and not isinstance(response['final_prediction'], str)):
            return rejected, []
        if ('audit_verdict' in response
                and response['audit_verdict'] not in ('supported', 'unknown', 'conflict')):
            return rejected, []
        requested = set(data['variables'])
        seen, records = set(), []
        required = {'variable', 'observations', 'unresolved'}
        reviews = {'resolved_obligation_ids', 'resolved_failed_batch_ids', 'reviewed_observation_ids'}
        allowed = required | reviews | {'search_gaps', 'reading_status'}
        for slot in response['slots']:
            if (not isinstance(slot, dict) or not required <= set(slot) or set(slot) - allowed
                    or not isinstance(slot.get('variable'), str)
                    or slot['variable'] not in requested or slot['variable'] in seen
                    or not isinstance(slot['observations'], list)
                    or not isinstance(slot['unresolved'], list)
                    or slot['unresolved']):
                return rejected, []
            seen.add(slot['variable'])
            for key in reviews & set(slot):
                values = slot[key]
                if (not isinstance(values, list) or any(not isinstance(value, str) for value in values)
                        or len(set(values)) != len(values)):
                    return rejected, []
            status = slot.get('reading_status')
            if status == 'no_relevant_evidence':
                gaps = slot.get('search_gaps', [])
                if (not isinstance(gaps, list)
                        or ('search_gaps' in slot and not gaps)
                        or any(not isinstance(gap, str) or not gap.strip() for gap in gaps)
                        or slot['observations'] or any(slot.get(key) for key in reviews)):
                    return rejected, []
                records.append({'variable': slot['variable'], 'reading_status': status,
                                'search_gaps': deepcopy(gaps)})
            elif status == 'evidence_assessment':
                rows = slot['observations']
                references = slot.get('reviewed_observation_ids', [])
                if ('search_gaps' in slot or (not rows and not references)
                        or any(not isinstance(row, dict)
                               or not isinstance(row.get('source_ids'), list) or not row['source_ids']
                               or any(not isinstance(source, str) or not source for source in row['source_ids'])
                               for row in rows)):
                    return rejected, []
            else:
                return rejected, []
        if seen != requested:
            return rejected, []
        decoded = deepcopy(response)
        for slot in decoded['slots']:
            slot.pop('search_gaps', None)
            slot.pop('reading_status')
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
                'web_reading_protocol': 'required_typed_web_reading_v1',
                'web_search_gaps': deepcopy(self.web_search_gaps),
                'web_capacity_admissions': deepcopy(self.web_capacity_admissions)}
