"""Independent review regressions: evidence obligations and residual handoffs.

The adversarial responses are synthetic. Assertions are about retained raw
sources, actual pending state and certified elimination, not answer labels.
"""
from copy import deepcopy

import pytest

from dagbt.fact_state import FactState
from dagbt.joint_reading import JointReader
from dagbt.residual import certify_output
from dagbt.residual_control import ResidualControl
from test_personamem_joint_protocol import fixture, observation, response, visible
from test_v4_joint_reading import capacity_fixture


def capacity_downgrade():
    reader, engine, state, calls, aliases, _ = capacity_fixture()
    shown = {alias: engine.mapper.sources[alias] for alias in aliases}
    state.integrate([reader._observation(observation('prior', [aliases[0]]),
                                        'history', shown, engine.steps[0])])
    # Reading the draft is allowed, but holding both contradictory whole-value
    # proofs does not fit the explicit recheck footprint. Downstream operations
    # still use the fixture's real token counter.
    reader._fits = lambda operation, data: (len(data['known_observations']) < 2
        if data['task']['output_slot'] == 'history' else JointReader._fits(reader, operation, data))
    engine.s['max_repairs_per_request'] = 1
    def respond(index, data):
        if index == 1:
            return response(observation('new-proof', aliases, value=['painting']))
        partial = observation('new-proof', [aliases[-1]], stance='partial')
        partial['value'] = None
        result = response(partial)
        result['slots'][0]['unresolved'] = [
            'A potentially conflicting later preference remains unresolved; required proof does not fit.']
        return result
    calls.respond = respond
    reader.read(engine.steps[0], 'Check activity history', ['history'], list(engine.docs), 'new-history')
    assert len(calls.requests) == 2
    return reader, engine, state, calls, aliases


@pytest.mark.parametrize('audit', [False, True])
def test_capacity_to_partial_remains_pending_and_forwards_its_original_text(audit):
    reader, engine, state, _, aliases = capacity_downgrade()
    assert state.valid()[0]['history'] == ['piano']
    assert 'history' in state.pending
    assert 'history' in reader.pending_variables()
    reader.current_identity = 'different-audit' if audit else 'different-terminal'
    payload = reader._payload(engine.steps[1], 'Assess from history', ['suggestion', 'history'], [],
                              semantic=not audit, audit=audit)
    available = {source['id'] for source in payload['source_spans']}
    assert aliases[-1] in available
    assert any(source['text'] == engine.mapper.sources[aliases[-1]]['text']
               for source in payload['source_spans'])


def test_successful_empty_read_cannot_clear_another_querys_evidence_obligation():
    reader, engine, state, calls, aliases = capacity_downgrade()
    before = deepcopy(reader.obligation_snapshot())
    exposures_before = before.pop('source_exposures')
    calls_before = len(calls.requests)
    calls.respond = lambda index, data: response()
    reader.read(engine.steps[0], 'Recheck only established history', ['history'],
                [engine.mapper.sources[aliases[0]]['doc_id']], 'unrelated-check')
    assert state.valid()[0]['history'] == ['piano']
    assert 'history' in state.pending
    assert 'history' in reader.pending_variables()
    after = reader.obligation_snapshot()
    exposures_after = after.pop('source_exposures')
    assert after == before
    assert len(calls.requests) == calls_before + 1
    assert exposures_after[:len(exposures_before)] == exposures_before
    assert len(exposures_after) == len(exposures_before) + 1
    exposure = exposures_after[-1]
    assert exposure['query_identity'] == 'unrelated-check'
    assert exposure['operation'] == 'joint_read' and exposure['request_status'] == 'returned'
    assert exposure['status'] == 'incomplete' and exposure['variables'] == ['history']
    # The earlier slot-level unresolved issue has no per-issue citation map:
    # all originals from that window remain required, not only the partial's citation.
    assert {source['source_alias'] for source in exposure['sources']} == set(aliases)
    assert {source['id'] for source in calls.requests[-1]['data']['source_spans']} == set(aliases)


def test_evidence_obligation_survives_reader_snapshot_restore_and_new_identity():
    reader, engine, state, _, aliases = capacity_downgrade()
    snapshot = deepcopy(reader.obligation_snapshot())
    fresh = JointReader(engine, state)
    fresh.restore_obligations(snapshot)
    fresh.current_identity = 'resumed-next-task'
    assert 'history' in fresh.pending_variables()
    payload = fresh._payload(engine.steps[1], 'Assess history after resume', ['history', 'suggestion'], [], True)
    assert aliases[-1] in {source['id'] for source in payload['source_spans']}


def short_circuit_fixture():
    _, engine, _, _ = fixture()
    steps = [
        {'output_slot': 'necessary', 'question': 'Check a necessary condition', 'inputs': [],
         'execution': 'retrieval', 'answer_type': 'fact'},
        {'output_slot': 'later', 'question': 'Detailed later criterion ' + ('long criterion ' * 3000),
         'inputs': ['necessary'], 'execution': 'retrieval', 'answer_type': 'fact'},
        {'output_slot': 'decision', 'question': 'Do both necessary conditions hold?',
         'inputs': ['necessary', 'later'], 'execution': 'compose', 'answer_type': 'fact'}]
    program = {'variables': {'a': {'type': 'bool', 'demand': 'necessary'},
                             'b': {'type': 'bool', 'demand': 'later'}},
               'expression': {'op': 'and', 'args': [{'op': 'var', 'id': 'a'}, {'op': 'var', 'id': 'b'}]},
               'rules_complete': True}
    contract = {'mode': 'value', 'result_type': 'bool', 'entity_scope': 'user',
                'time_scope': 'history', 'explanation': 'none'}
    state = FactState(program, contract, steps, 'decision')
    engine.steps = steps
    engine.fact_state = state
    engine.final_node_id = 'decision'
    engine.output_options = None
    reader = JointReader(engine, state)
    supported = reader._observation(observation('decisive', ['s1'], value=False),
                                    'a', visible(reader), steps[0])
    return reader, engine, state, supported


def test_certified_negative_witness_is_not_rejected_for_an_eliminated_task():
    reader, _, state, support = short_circuit_fixture()
    proposed = reader._prospective_state([support])
    certificate = certify_output(proposed.program, proposed.valid()[0], proposed.contract)
    assert certificate['output'] is False and certificate['eliminated_variables'] == ['b']
    checked = []
    original = reader._fits
    reader._fits = lambda operation, data: checked.append((operation, deepcopy(data))) or original(operation, data)
    assert reader._handoff_failures([support], {('a', 'decisive')}) == {}
    assert not any(data['task']['output_slot'] == 'later' for _, data in checked)
    # No variable resides in the terminal task: nevertheless the real next
    # mandatory operation is an audit of the decisive original evidence.
    assert any(operation == 'audit_joint' for operation, _ in checked)
    assert state.valid()[0] == {}


def test_symbolic_completion_still_checks_audit_capacity_without_terminal_variables():
    reader, _, state, support = short_circuit_fixture()
    checked = []
    def fits(operation, data):
        checked.append((operation, deepcopy(data)))
        return operation != 'audit_joint'
    reader._fits = fits
    failures = reader._handoff_failures([support], {('a', 'decisive')})
    assert ('a', 'decisive') in failures
    assert 'audit' in failures['a', 'decisive']
    audited = [data for operation, data in checked if operation == 'audit_joint']
    assert audited
    assert 's1' in {source['id'] for source in audited[0]['source_spans']}
    assert state.valid()[0] == {}


def test_sourced_unresolved_parent_prevents_reusing_an_already_valid_terminal():
    from test_v4_joint_reading import prepare_terminal, terminal_response, compose
    reader, engine, state, calls = prepare_terminal(personal=True)
    calls.respond = lambda index, data: terminal_response(data, engine.output_options[0])
    compose(reader)
    assert reader.semantic_inputs['validation_complete']
    assert reader.semantic_prediction == 'a'
    engine.fact_state = state
    engine.final_node_id = state.final_node_id
    engine.s['max_enumeration_states'] = 10000
    controller = ResidualControl(engine)
    engine.residual_control = controller
    controller.reader = reader
    assert controller.semantic_inputs_current()
    def partial_counter(index, data):
        draft = observation('new-counter-clue', ['s2'], stance='partial')
        draft['value'] = None
        result = response(draft)
        result['slots'][0]['unresolved'] = ['Original source may contradict the assumed current hobby.']
        return result
    calls.respond = partial_counter
    reader.read(engine.steps[0], 'Check newer personal memory', ['history'], ['painting'], 'new-evidence')
    assert state.valid()[0]['history'] == ['piano']
    assert 'history' in state.pending
    assert not controller.semantic_inputs_current()
    assert reader.semantic_prediction == 'a'  # No silent replacement of prior output.


def sourced_gap_fixture():
    reader, engine, state, calls = fixture(max_repairs_per_request=0)
    state.integrate([reader._observation(observation('prior', ['s1']), 'history', visible(reader), engine.steps[0])])
    def report_gap(index, data):
        draft = observation('gap', ['s2'], stance='partial')
        draft['value'] = None
        result = response(draft)
        result['slots'][0]['unresolved'] = ['Check this contrary personal-history clue.']
        return result
    calls.respond = report_gap
    reader.read(engine.steps[0], 'Check personal history', ['history'], ['painting'], 'same-query')
    assert 'history' in state.pending
    return reader, engine, state, calls


@pytest.mark.parametrize('attempt', ['empty', 'wrong_source', 'unresolved', 'omitted_id', 'invalid_counter', 'new_partial'])
def test_evidence_resolution_cannot_be_claimed_without_a_complete_sourced_review(attempt):
    reader, engine, state, calls = sourced_gap_fixture()
    def report_resolution(index, data):
        offered = [item['id'] for item in data['evidence_obligations']]
        assert offered
        rows = [] if attempt == 'empty' else [observation('review', ['s1' if attempt == 'wrong_source' else 's2'],
                                                        stance='irrelevant')]
        for row in rows:
            row['value'] = None
        if attempt == 'invalid_counter':
            rows[0]['stance'] = 'contradiction'
            rows[0]['retract_ids'] = ['invented-route']
        if attempt == 'new_partial':
            partial = observation('still-open', ['s2'], stance='partial')
            partial['value'] = None
            rows.append(partial)
        result = response(*rows)
        result['slots'][0]['resolved_obligation_ids'] = [] if attempt == 'omitted_id' else offered
        result['slots'][0]['unresolved'] = ['The clue still conflicts.'] if attempt == 'unresolved' else []
        return result
    calls.respond = report_resolution
    reader.read(engine.steps[0], 'Check personal history', ['history'], [], 'same-query')
    assert state.valid()[0]['history'] == ['piano']
    assert 'history' in state.pending
    assert 'history' in reader.pending_variables()
    reader.current_identity = 'later-audit'
    payload = reader._payload(engine.steps[1], 'Review known history', ['history'], [], audit=True)
    assert 's2' in {source['id'] for source in payload['source_spans']}


def test_explicit_sourced_resolution_can_clear_gap_without_new_unread_documents():
    reader, engine, state, calls = sourced_gap_fixture()
    before = len(calls.requests)
    def report_resolution(index, data):
        row = observation('resolved-context', ['s2'], stance='irrelevant')
        row['value'] = None
        row['reason'] = 'The source concerns an earlier separate activity and does not contradict this scoped fact.'
        # The original unresolved text had no per-issue citation mapping, so
        # explicitly review both its original support and the additional clue.
        result = response(observation('confirmed-old-context', ['s1']), row)
        result['slots'][0]['resolved_obligation_ids'] = [item['id'] for item in data['evidence_obligations']]
        return result
    calls.respond = report_resolution
    assert reader.read(engine.steps[0], 'Check personal history', ['history'], [], 'same-query')
    assert len(calls.requests) == before + 1
    assert not state.pending and not reader.pending_variables()
    assert state.valid()[0]['history'] == ['piano']
    assert not reader.read(engine.steps[0], 'Check personal history', ['history'], [], 'same-query')
    assert len(calls.requests) == before + 1


def test_repeating_an_unchanged_unresolved_review_is_not_material_progress():
    reader, engine, state, calls = sourced_gap_fixture()
    calls.respond = lambda index, data: response()
    reader.read(engine.steps[0], 'Check personal history', ['history'], [], 'same-query')
    signature = reader.material_signature()
    assert not reader.read(engine.steps[0], 'Check personal history', ['history'], [], 'same-query')
    assert reader.material_signature() == signature
    assert 'history' in state.pending


def test_handoff_capacity_uses_the_same_batchs_validated_obligation_resolution():
    reader, engine, state, calls, aliases, _ = capacity_fixture()
    engine.s['max_repairs_per_request'] = 0
    reader._fits = lambda operation, data: (True if data['task']['output_slot'] == 'history'
                                             else JointReader._fits(reader, operation, data))
    def review(index, data):
        if index == 1:
            draft = observation('uncertain-history', aliases, stance='partial')
            draft['value'] = None
            result = response(draft)
            result['slots'][0]['unresolved'] = ['Check whether these memories establish one complete activity value.']
            return result
        sufficient = observation('narrow-proof', [aliases[0]])
        other_sources = observation('other-context-reviewed', aliases[1:], stance='irrelevant')
        other_sources['value'] = None
        other_sources['reason'] = 'These independently reviewed contexts add no necessary premise to this scoped value.'
        result = response(sufficient, other_sources)
        result['slots'][0]['resolved_obligation_ids'] = [item['id'] for item in data['evidence_obligations']]
        return result
    calls.respond = review
    reader.read(engine.steps[0], 'Read past activities', ['history'], list(engine.docs), 'history')
    assert 'history' in state.pending
    reader.read(engine.steps[0], 'Read past activities', ['history'], [], 'history')
    assert state.valid()[0]['history'] == ['piano']
    assert not state.pending and not reader.pending_variables()
    reader.current_identity = 'new-terminal'
    payload = reader._payload(engine.steps[1], 'Use reviewed history', ['suggestion', 'history'], [], True)
    assert {source['id'] for source in payload['source_spans']} == {aliases[0]}
    assert reader._fits('semantic_compose', payload)


def test_unresolved_text_about_bound_fact_retains_visible_context_even_without_observation_citations():
    reader, engine, state, calls = fixture(max_repairs_per_request=0)
    state.integrate([reader._observation(observation('prior', ['s1']), 'history', visible(reader), engine.steps[0])])
    def unresolved(index, data):
        result = response()
        result['slots'][0]['unresolved'] = ['The newer statement may contradict this current-hobby fact.']
        return result
    calls.respond = unresolved
    reader.read(engine.steps[0], 'Read newer original memory', ['history'], ['painting'], 'check-new')
    reader.current_identity = 'next-task'
    payload = reader._payload(engine.steps[1], 'Assess history', ['history', 'suggestion'], [], True)
    assert 'history' in state.pending
    assert 's2' in {source['id'] for source in payload['source_spans']}
    def review_only_old_source(index, data):
        draft = observation('old-only', ['s1'], stance='irrelevant')
        draft['value'] = None
        result = response(draft)
        result['slots'][0]['resolved_obligation_ids'] = [item['id'] for item in data['evidence_obligations']]
        return result
    calls.respond = review_only_old_source
    reader.read(engine.steps[0], 'Read newer original memory', ['history'], [], 'check-new')
    assert 'history' in state.pending
    assert reader.pending_variables() == {'history'}


def test_unbound_missing_support_gap_can_be_solved_by_later_supported_value():
    reader, engine, state, calls = fixture(max_repairs_per_request=0)
    def missing(index, data):
        result = response()
        result['slots'][0]['unresolved'] = ['No complete activity value has been established.']
        return result
    calls.respond = missing
    # This is genuinely missing evidence: no original source text has yet been
    # examined. Visible-but-uncited conflicting text cannot be treated this way.
    reader.read(engine.steps[0], 'Find documented activity', ['history'], [], 'lookup', semantic=True)
    assert 'history' not in state.valid()[0]
    calls.respond = lambda index, data: response(observation('found', ['s1']))
    reader.read(engine.steps[0], 'Find documented activity', ['history'], ['piano'], 'lookup')
    assert state.valid()[0]['history'] == ['piano']
    assert 'history' not in state.pending
    assert 'history' not in reader.pending_variables()


def test_reviewing_separate_subsets_does_not_silently_discharge_a_whole_source_obligation():
    reader, engine, state, calls = fixture(max_repairs_per_request=0)
    state.integrate([reader._observation(observation('prior', ['s1']), 'history', visible(reader), engine.steps[0])])
    def unresolved(index, data):
        result = response()
        result['slots'][0]['unresolved'] = ['The newly read memory may change the established activity.']
        return result
    calls.respond = unresolved
    reader.read(engine.steps[0], 'Reconcile memory', ['history'], ['painting'], 'broad-obligation')
    obligation = next(iter(reader.evidence_obligations.values()))
    assert set(obligation['source_ids']) == {
        engine.mapper.sources[alias]['source_id'] for alias in ('s1', 's2')}
    def review_old(index, data):
        row = observation('old-part-reviewed', ['s1'], stance='irrelevant')
        row['value'] = None
        return response(row)
    calls.respond = review_old
    reader.read(engine.steps[0], 'Reconcile memory', ['history'], [], 'partial-review')
    assert 'history' in state.pending
    def claim_from_new_only(index, data):
        row = observation('new-part-reviewed', ['s2'], stance='irrelevant')
        row['value'] = None
        result = response(row)
        result['slots'][0]['resolved_obligation_ids'] = [item['id'] for item in data['evidence_obligations']]
        return result
    calls.respond = claim_from_new_only
    reader.read(engine.steps[0], 'Reconcile memory', ['history'], [], 'later-partial-review')
    assert 'history' in state.pending
    assert obligation['id'] in reader.evidence_obligations
    assert state.valid()[0]['history'] == ['piano']


def test_settling_an_unrelated_partial_cannot_remove_an_already_integrated_counter():
    reader, engine, state, calls = fixture(max_repairs_per_request=0)
    state.integrate([reader._observation(observation('prior', ['s1']), 'history', visible(reader), engine.steps[0])])
    prior_route = next(iter(state.routes))
    def read_counter_and_partial(index, data):
        counter = observation('real-counter', ['s2'], stance='contradiction', retract_ids=[prior_route])
        counter['value'] = None
        partial = observation('separate-uncertainty', ['s3'], stance='partial')
        partial['value'] = None
        return response(counter, partial)
    calls.respond = read_counter_and_partial
    reader.read(engine.steps[0], 'Inspect relevant evidence', ['history'], ['painting', 'advice'], 'counter-and-gap')
    assert 'history' not in state.valid()[0]
    counter_record = deepcopy(state.routes[prior_route]['revoked'])
    assert counter_record and 'history' in state.pending
    def settle_partial(index, data):
        row = observation('unrelated-advice', ['s3'], stance='irrelevant')
        row['value'] = None
        row['reason'] = 'This separate assistant suggestion makes no historical assertion about the user.'
        result = response(row)
        result['slots'][0]['resolved_obligation_ids'] = [item['id'] for item in data['evidence_obligations']]
        return result
    calls.respond = settle_partial
    reader.read(engine.steps[0], 'Inspect relevant evidence', ['history'], [], 'settle-only-partial')
    assert not reader.evidence_obligations
    assert state.routes[prior_route]['revoked'] == counter_record
    assert 'history' not in state.valid()[0]
    reader.current_identity = 'future-audit'
    payload = reader._payload(engine.steps[1], 'Audit historical claims', ['history'], [], audit=True)
    assert {'s1', 's2'} <= {source['id'] for source in payload['source_spans']}
    retained_route = next(row for row in payload['known_observations'] if row['id'] == prior_route)
    assert retained_route['revoked'][0]['source_ids'] == ['s2']


@pytest.mark.parametrize('stance', ['support', 'partial'])
def test_old_evidence_row_does_not_supply_citations_for_a_separate_unresolved_challenge(stance):
    reader, engine, state, calls = fixture(max_repairs_per_request=0)
    state.integrate([reader._observation(observation('prior', ['s1']), 'history', visible(reader), engine.steps[0])])
    def reaffirm_with_gap(index, data):
        row = observation('reaffirm-old', ['s1'], stance=stance)
        if stance == 'partial':
            row['value'] = None
            row['reason'] = 'The old piano evidence alone is only a partial account.'
        result = response(row)
        result['slots'][0]['unresolved'] = ['The newly read painting statement may contradict the claimed current hobby.']
        return result
    calls.respond = reaffirm_with_gap
    reader.read(engine.steps[0], 'Recheck established history', ['history'], ['painting'], 'read-new-challenge')
    reader.current_identity = 'different-terminal'
    payload = reader._payload(engine.steps[1], 'Select a suitable response', ['history', 'suggestion'], [], True)
    assert 'history' in state.pending
    assert 's2' in {source['id'] for source in payload['source_spans']}
    def review_only_old(index, data):
        row = observation('old-only-reviewed', ['s1'], stance='irrelevant')
        row['value'] = None
        result = response(row)
        result['slots'][0]['resolved_obligation_ids'] = [item['id'] for item in data['evidence_obligations']]
        return result
    calls.respond = review_only_old
    reader.read(engine.steps[0], 'Recheck established history', ['history'], [], 'old-evidence-only')
    assert 'history' in state.pending
    assert reader.evidence_obligations


def test_uncited_challenge_before_any_binding_keeps_its_original_evidence():
    reader, engine, state, calls = fixture(max_repairs_per_request=0)
    def unbound_conflict(index, data):
        result = response()
        result['slots'][0]['unresolved'] = ['This painting memory challenges a simple piano-only account; inspect it before binding the whole value.']
        return result
    calls.respond = unbound_conflict
    reader.read(engine.steps[0], 'Find complete activity history', ['history'], ['painting'], 'unbound-challenge')
    assert 'history' not in state.valid()[0]
    reader.current_identity = 'later-read'
    payload = reader._payload(engine.steps[0], 'Check another memory', ['history'], ['s1'])
    assert 's2' in {source['id'] for source in payload['source_spans']}
    calls.respond = lambda index, data: response(observation('piano-only', ['s1']))
    reader.read(engine.steps[0], 'Find complete activity history', ['history'], ['piano'], 'later-read')
    assert 'history' in state.pending
    assert reader.evidence_obligations
