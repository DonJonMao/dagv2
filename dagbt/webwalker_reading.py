"""Web-only source reading with explicit search gaps and repair headroom.

The common reader still owns facts, sources, counter capture and certification.
A search gap is a completed *empty-evidence* read, never a sourced unresolved
issue. Legacy ``unresolved`` responses retain their original conservative scope.
"""
from copy import deepcopy

from .joint_reading import JointReader
from .transport import digest


WEB_ENTAILMENT_INSTRUCTION = '''
WEB SOURCE CHECK: establish the exact requested entity/event/relation from original
evidence. Distinguish publication, revision, effective and described-event dates;
people/approving organizations are not geographic places. Do not transfer claims
between policies/documents/processes without explicit connecting evidence. Audit
must independently check originals: prior bindings/plausible explanations prove
no entailment. Unsupported conclusions stay unknown.
'''


WEB_READING_INSTRUCTION = '''
REQUIRED WEB RETRIEVAL PROTOCOL: every requested slot declares reading_status;
BOTH statuses require unresolved=[]. no_relevant_evidence means NO NEW admissible
observation: observations=[], no review/resolution IDs. Optional search_gaps lists
untried directions, only with this status. Old facts/obligations stay unchanged in
controller state; no need to repeat them. This search result is never a negative
fact or proof of absence. Do not cite an answer-free window as unknown; continue
the local bridge. evidence_assessment supplies NEW original-source-cited ordinary
observations. A genuine incomplete statement/ambiguity/conflict requires a named
partial/unknown row with exact source_ids, requested scope and reason; unknown
uses value=null. Never replace it with bare slot.unresolved or a resolution claim.
Old evidence obligations, failed drafts and counters remain mandatory. To settle
an offered issue, use evidence_assessment, explicitly review its original text
with sourced support/corrective/irrelevant observations, cover EVERY required
original source, and claim the offered resolution ID. Genuinely irrelevant text
may settle an earlier absence-only issue; never relabel real partial facts,
supporting text or counters irrelevant just to close them. Empty slots and
partial/unknown observations cannot settle evidence obligations: omit their IDs
from resolved_obligation_ids. Correcting a failed draft need not settle a source
issue. Preserve every required failed draft ID. Frozen rows, including partial/
unknown, must not be restated. Repairs may instead use currently offered
reviewed_observation_ids for immutable sourced rows, under all common source and
resolution checks. Terminal composition/audit use the ordinary protocol without
reading_status/search_gaps.
'''


WEB_REPAIR_INSTRUCTION = '''
WEB REPAIR FIELD CHECK: resolved_obligation_ids optionally claims settlement;
never copy obligation IDs as a checklist. unresolved=[] is wire format, not
settlement. With partial/unknown rows, omit resolution claims and leave issues
pending; invent no support/irrelevance. Frozen rows cannot be renamed/repeated;
old issues need no repeated row. Preserve/correct required failed drafts,
replacements and protected counters under offered IDs; empty slots cannot discard
them. Failed-batch format recovery and source-issue settlement remain separate.
'''


WEB_EMPTY_REPAIR_INSTRUCTION = '''
This repair has only wildcard slot failures; no mandatory draft/counter/whole-value
or durable review. With NO NEW observation, correct using
reading_status=no_relevant_evidence, observations=[], unresolved=[], no review/
resolution IDs. Withdraw only this call's invalid claim;
the frozen rows and the controller's evidence obligations remain unchanged.
'''


class WebJointReader(JointReader):
    def __init__(self, engine, state):
        super().__init__(engine, state)
        self.web_search_gaps = []
        self.web_capacity_admissions = []
        self._web_headroom_candidates = {}
        self._selected_web_headroom = None
        self.web_paid_recoveries = []
        self._selected_web_paid_recovery = None
        self._active_web_paid_recovery = None

    def _read(self, *args, **kwargs):
        # This mode authorizes one selected batch, never a later read/audit.
        self._selected_web_paid_recovery = None
        self._active_web_paid_recovery = None
        try:
            return super()._read(*args, **kwargs)
        finally:
            self._selected_web_paid_recovery = None
            self._active_web_paid_recovery = None

    @staticmethod
    def _search_gap_scope(data):
        return (data.get('task', {}).get('execution') == 'retrieval'
                and not any(key in data for key in
                            ('audit_instruction', 'public_output_question', 'terminal_repair')))

    @staticmethod
    def _system(data):
        system = JointReader._system(data) + WEB_ENTAILMENT_INSTRUCTION
        if WebJointReader._search_gap_scope(data):
            system += WEB_READING_INSTRUCTION
            if data.get('repair_scope'):
                system += WEB_REPAIR_INSTRUCTION
                if WebJointReader._empty_repair_allowed(data):
                    system += WEB_EMPTY_REPAIR_INSTRUCTION
        return system

    @staticmethod
    def _empty_repair_allowed(data):
        scope = data.get('repair_scope', {})
        failures = scope.get('failed', [])
        return bool(failures and all(entry.get('id') == '*' for entry in failures)
                    and not any(entry.get('observation') is not None or entry.get('observations')
                                for entry in failures)
                    and not scope.get('invalid_response')
                    and not scope.get('whole_value_reviews')
                    and not any(entry.get(key) for entry in failures for key in
                                ('replacement_observations', 'required_counter_observations'))
                    and not any(review.get('review_allowed')
                                for review in data.get('failed_batch_reviews', [])))

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
        if 'resolved_obligation_ids' in slot['properties']:
            # Mirror the existing core guard. Unknown/partial observations can
            # remain genuine issues, but cannot justify an obligation release.
            # A corrected failed draft is a separate kind of resolution.
            slot['allOf'].append({
                'if': {'required': ['resolved_obligation_ids'],
                       'properties': {'resolved_obligation_ids': {'minItems': 1}}},
                'then': {'properties': {
                    'reading_status': {'const': 'evidence_assessment'},
                    'observations': {'items': {'properties': {
                        'stance': {'enum': ['support', 'contradiction', 'irrelevant']}}}}}}})
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
        # An initial request can also be a paid recovery of a durable draft.
        # Its existing originals/drafts are already in data, but any newly
        # exposed text can still produce another failed response. Recovery is
        # therefore not an exemption from fresh-window repair forecasting.
        recovery = any(item.get('review_allowed')
                       for item in data.get('failed_batch_reviews', []))
        if (operation != 'joint_read' or not fresh_sources or not self._search_gap_scope(data)
                or data.get('repair_scope')
                or self.e.s['max_repairs_per_request'] <= bool(recovery)
                or self.e.ledger.remaining('json_repairs') <= bool(recovery)
                or self.e.ledger.remaining('llm') < self.e.s['reserved_audit_calls'] + 2):
            return None
        failed = []
        for variable in data['variables']:
            entry = {'variable': variable, 'id': '*', 'feedback': 'x' * 240,
                     'observation': None,
                     'slot': {'variable': variable, 'observations': [], 'unresolved': []}}
            obligations = [item for item in data.get('evidence_obligations', [])
                           if item['variable'] == variable]
            if obligations:
                # Coverage feedback repeats required/missing source aliases;
                # unlike the future draft, this growth is known before reading.
                entry['obligation_review_sources'] = [
                    {'obligation_id': item['id'],
                     'required_source_ids': list(item['source_ids']),
                     # A failed review may cite every other visible source
                     # while missing all required originals. Count that known
                     # coverage footprint as well as the missing originals.
                     'covered_source_ids': [source['id'] for source in data['source_spans']
                                            if source['id'] not in item['source_ids']],
                     'missing_source_ids': list(item['source_ids'])}
                    for item in obligations]
            # These are measurement-only empty shells. They grant no frozen
            # review authority; the common reader offers actual IDs later.
            entry['frozen_observation_reviews'] = {
                'slot_observation_ids': [], 'observations': []}
            failed.append(entry)
        forecast = {**data, 'repair_scope': {
            'failed': failed,
            'attempt': 1, 'previous_response_identity': '0' * 64,
            'invalid_response': ''}}
        count = self._estimate(operation + '_repair', forecast)
        output = self.e.s['reasoning_output_tokens']
        total = count + 2 * output + self.e.s['input_margin'] + 8
        return {'repair_forecast_input_tokens': count, 'unknown_draft_reserve': output,
                'repair_output_reserve': output, 'forecast_total_tokens': total,
                'context_tokens': self.e.s['context_tokens'],
                'within_budget': total <= self.e.s['context_tokens'],
                'durable_recovery': bool(recovery),
                'policy': 'scoped_repair_envelope_plus_one_unknown_draft_allowance_v2'}

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
        self._selected_web_paid_recovery = None
        self._active_web_paid_recovery = None
        try:
            aliases, consumed = super()._fitting_prefix(
                operation, step, query, variables, pending, semantic, audit)
            if consumed:
                data = self._payload(step, query, variables, aliases, semantic, audit=audit)
                identity = digest(data)
                if identity in self._web_headroom_candidates:
                    self._selected_web_headroom = (identity, self._web_headroom_candidates[identity])
            else:
                aliases, consumed = self._paid_recovery_prefix(
                    operation, step, query, variables, pending, semantic, audit)
            return aliases, consumed
        finally:
            self._web_headroom_candidates = {}

    def _paid_recovery_prefix(self, operation, step, query, variables, pending, semantic, audit):
        """Try one paid old-draft review only after every full forecast fails.

        The full original request must fit. No unknown-response workspace is
        promised: if this single response fails, the common reader retains it
        and all originals, and this batch buys no further local repair.
        """
        if (operation != 'joint_read' or semantic or audit or self.current_terminal_output
                or step.get('execution') != 'retrieval'):
            return [], 0
        common = {'query_identity': self.current_identity, 'task_id': step['output_slot'],
                  'policy': 'single_paid_durable_recovery_v1'}
        reason = 'no_new_source_authorized_failed_review'
        if (self.e.s['max_repairs_per_request'] <= 0
                or self.e.ledger.remaining('json_repairs') <= 0
                or self.e.ledger.remaining('llm') <= self.e.s['reserved_audit_calls']):
            reason = 'durable_recovery_budget_unavailable'
        else:
            # Ascending prefixes deliberately select the smallest new group.
            for size in range(1, len(pending) + 1):
                aliases = list(dict.fromkeys(a for group in pending[:size] for a in group))
                data = self._payload(step, query, variables, aliases, semantic, audit=audit)
                if not self._search_gap_scope(data) or self._failed_phase(data) != 'history':
                    continue
                forecast = self._web_headroom_candidates.get(digest(data))
                if forecast is None or forecast['within_budget']:
                    continue
                offered = []
                new_sources = set()
                candidate_sources = {self.e.mapper.sources[a]['source_id'] for a in aliases}
                for review in data.get('failed_batch_reviews', []):
                    record = self.failed_batch_reviews.get(review['id'])
                    if not review.get('review_allowed') or record is None:
                        continue
                    fresh = candidate_sources - set(record['seen_source_ids'])
                    if fresh:
                        offered.append(review['id'])
                        new_sources.update(fresh)
                # Parent changes or a renamed query alone cannot unlock this.
                if not offered:
                    continue
                if not super()._fits(operation, data):
                    reason = 'complete_recovery_request_exceeds_context'
                    continue
                selection = {**common, 'request_identity': digest(data),
                             'prefix_groups': size, 'source_count': len(data['source_spans']),
                             'failed_batch_ids': offered, 'new_source_ids': sorted(new_sources),
                             'base_input_tokens': self._estimate(operation, data),
                             'output_token_reserve': self.e.s['reasoning_output_tokens'],
                             'context_tokens': self.e.s['context_tokens'],
                             'additional_local_repairs_allowed': 0,
                             'rejected_full_forecast': deepcopy(forecast), 'status': 'selected'}
                self.web_paid_recoveries.append(selection)
                self._selected_web_paid_recovery = selection
                self.e.event({'event': 'web_paid_recovery_selected', **deepcopy(selection)})
                return aliases, size
        self.e.event({'event': 'web_paid_recovery_blocked', **common, 'reason': reason})
        return [], 0

    def _repair_fits(self, operation, data, step, identity, attempt):
        active = self._active_web_paid_recovery
        if (active is not None and operation == 'joint_read_repair'
                and identity == active['query_identity']
                and step['output_slot'] == active['task_id']
                and self._search_gap_scope(data)):
            active['local_repair_blocked'] = True
            self.e.event({'event': 'web_paid_recovery_local_repair_blocked',
                          'query_identity': identity, 'task_id': step['output_slot'],
                          'repair_attempt': attempt, 'policy': active['policy'],
                          'reason': 'single_paid_response_already_attempted'})
            # The common reader calls this before reserving another repair.
            return False
        return super()._repair_fits(operation, data, step, identity, attempt)

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
        selection = self._selected_web_paid_recovery
        if (selection is not None and selection['request_identity'] == digest(data)
                and selection['query_identity'] == identity and operation == 'joint_read_repair'):
            self._active_web_paid_recovery = selection
            self._selected_web_paid_recovery = None
            selection['status'] = 'started'
            self.e.event({'event': 'web_paid_recovery_started', **deepcopy(selection)})
        try:
            raw = super()._request_with_exposure(operation, data, identity, wire, system, **options)
        except BaseException:
            if self._active_web_paid_recovery is not None:
                self._active_web_paid_recovery['status'] = 'request_failed'
            raise
        if self._active_web_paid_recovery is not None:
            self._active_web_paid_recovery['status'] = 'response_received'
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
                'web_reading_protocol': 'required_typed_web_reading_v3',
                'web_search_gaps': deepcopy(self.web_search_gaps),
                'web_capacity_admissions': deepcopy(self.web_capacity_admissions),
                'web_paid_recoveries': deepcopy(self.web_paid_recoveries)}
