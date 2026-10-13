"""Independent mandatory-evidence and repair contracts; all transport is offline.

The production reader, fact state, budgets and wire serializer remain active.
Only model replies are scripted. These checks do not access labels or services.
"""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from dagbt.evidence_spans import build_source_spans
from dagbt.fact_state import FactState
from dagbt.joint_reading import JointReader
from dagbt.reasoning import InputOverflow, ProtocolError
from test_personamem_joint_protocol import fixture, observation, response, visible
from test_v4_joint_reading import prepare_terminal
from test_v4_obligation_review_feedback_review import (
    OID, PROOF, claim, irrelevant, obligation_fixture, read_issue,
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Independent contract checks must not contact a service')
    monkeypatch.setattr('urllib.request.urlopen', forbidden)
    monkeypatch.setattr('urllib.request.build_opener', forbidden)


def mandatory(reader, *, identity='bound-read', variables=None, audit=False,
              docs=(), step=None, **kwargs):
    return reader.read(step or reader.e.steps[1], 'Use the established scoped evidence',
                       variables or ['suggestion'], list(docs), identity,
                       audit=audit, source_scope='mandatory_closure', **kwargs)


def empty_slots(data, *, audit=False):
    result = {'slots': [{'variable': variable, 'observations': [], 'unresolved': []}
                        for variable in data['requested_slots']]}
    if audit:
        result['audit_verdict'] = 'supported'
    return result


def known_rows(data):
    columns = data.get('known_observation_columns')
    if not columns:
        return data['known_observations']
    shared = data['known_observation_values']
    return [dict(zip(columns, [shared[value['shared']]
                              if isinstance(value, dict) and set(value) == {'shared'} else value
                              for value in row])) for row in data['known_observations']]


def scoped_fixture(*, unrelated=False):
    reader, engine, state, calls = fixture(max_repairs_per_request=0, max_quote_chars=30)
    if unrelated:
        program = deepcopy(state.initial_program)
        program['variables']['unrelated'] = {'type': 'set', 'demand': 'unrelated'}
        engine.steps.append({'output_slot': 'unrelated', 'question': 'Independent history',
                             'inputs': [], 'execution': 'retrieval', 'answer_type': 'fact'})
        state = FactState(program, state.contract, engine.steps, 'suggestion')
        reader = JointReader(engine, state)
    engine.docs.clear()
    engine.mapper.sources.clear()
    engine.mapper.source_aliases.clear()
    text = 'I played piano last year. I painted last spring. I tried pottery earlier. '
    text += 'I attended a concert later. I went to a theatre before that. '
    doc = SimpleNamespace(doc_id='many-spans', passage=text, metadata={
        'source_segments': [{'role': 'user', 'start': 0, 'end': len(text),
                             'source_message_indices': [0]}]})
    engine.docs[doc.doc_id] = doc
    for span in build_source_spans(doc.doc_id, doc, 30):
        alias = 's' + str(len(engine.mapper.sources) + 1)
        engine.mapper.sources[alias] = {**span, 'id': alias, 'source_id': span['id']}
        engine.mapper.source_aliases[span['id']] = alias
    assert len(engine.mapper.sources) >= 4
    state.integrate([reader._observation(observation('supported-parent'), 'history',
                                        visible(reader), engine.steps[0])])
    return reader, engine, state, calls


def snapshot(reader, engine, state, calls):
    return deepcopy((state.public_dict(), reader.obligation_snapshot(),
                     reader.material_signature(), reader.current_identity,
                     reader.current_terminal_output, reader.clues, reader.read_states,
                     engine.events, engine.spans, engine.ledger.used, len(calls.requests)))


@pytest.mark.parametrize('misuse', ['retrieval', 'nonempty_docs', 'unknown_scope'])
def test_scope_rejection_has_no_request_budget_or_state_side_effect(misuse):
    reader, engine, state, calls = scoped_fixture()
    reader.current_identity = 'existing-identity'
    reader.current_terminal_output = True
    before = snapshot(reader, engine, state, calls)
    with pytest.raises(ProtocolError):
        if misuse == 'retrieval':
            mandatory(reader, step=engine.steps[0], variables=['history'])
        elif misuse == 'nonempty_docs':
            mandatory(reader, docs=['many-spans'])
        else:
            reader.read(engine.steps[1], 'Read', ['suggestion'], [], 'new',
                        source_scope='unrecognized-scope')
    assert snapshot(reader, engine, state, calls) == before


@pytest.mark.parametrize('audit', [False, True])
def test_mandatory_sources_are_one_full_batch_without_containing_doc_or_neighbours(audit):
    reader, engine, state, calls = scoped_fixture()
    settings, limits = deepcopy(engine.s), deepcopy(engine.ledger.limits)
    original = deepcopy(engine.mapper.sources)
    def answer(index, data):
        assert index == 1
        assert {row['id'] for row in data['source_spans']} == {'s1'}
        assert data['source_spans'][0]['text'] == original['s1']['text']
        assert data['supported_parents']['history']['value'] == ['piano']
        return empty_slots(data, audit=audit)
    calls.respond = answer
    mandatory(reader, variables=['history', 'suggestion'] if audit else None, audit=audit)
    assert len(calls.requests) == engine.ledger.used['llm'] == 1
    assert len(reader.batches) == 1
    assert reader.batches[0]['complete']
    assert not reader.unread_obligations
    assert engine.s == settings and engine.ledger.limits == limits
    assert engine.mapper.sources == original


def test_nonterminal_compose_with_only_bound_sources_really_calls_and_solves():
    reader, engine, state, calls = scoped_fixture()
    program = deepcopy(state.initial_program)
    program['variables']['final'] = {'type': 'text', 'demand': 'final'}
    program['expression'] = {'op': 'var', 'id': 'final'}
    engine.steps.append({'output_slot': 'final', 'question': 'Use {suggestion}',
                         'inputs': ['suggestion'], 'execution': 'compose', 'answer_type': 'fact'})
    state = FactState(program, state.contract, engine.steps, 'final')
    reader = JointReader(engine, state)
    state.integrate([reader._observation(observation('supported-parent'), 'history',
                                        visible(reader), engine.steps[0])])
    def answer(index, data):
        value = empty_slots(data)
        next(slot for slot in value['slots'] if slot['variable'] == 'suggestion')['observations'] = [
            observation('intermediate', value='Historical creative activities include piano.',
                        parent_versions={'history': state.versions['history']})]
        return value
    calls.respond = answer
    mandatory(reader)
    assert len(calls.requests) == 1
    assert state.valid()[0]['suggestion'] == 'Historical creative activities include piano.'
    assert reader.semantic_inputs is None


@pytest.mark.parametrize('problem', ['foreign_overlap', 'same_outside_source', 'same_extra_variable'])
def test_unread_admission_rejects_unowned_or_incomplete_work_without_changing_markers(problem):
    reader, engine, state, calls = scoped_fixture(unrelated=True)
    identity, alias, variables = 'bound-read', 's1', {'history'}
    if problem == 'foreign_overlap':
        identity = 'older-owner'
    elif problem == 'same_outside_source':
        alias = 's2'
    else:
        variables.add('unrelated')
    reader.unread_obligations[identity, alias] = variables
    state.integrate([], reader.pending_variables())
    before = snapshot(reader, engine, state, calls)
    with pytest.raises(ProtocolError):
        mandatory(reader)
    assert snapshot(reader, engine, state, calls) == before


def test_owned_unread_keeps_its_exact_variable_set_until_real_processing():
    reader, engine, state, calls = scoped_fixture()
    reader.unread_obligations['bound-read', 's1'] = {'history'}
    state.integrate([], reader.pending_variables())
    def answer(index, data):
        assert reader.unread_obligations['bound-read', 's1'] == {'history'}
        assert {row['id'] for row in data['source_spans']} == {'s1'}
        return empty_slots(data)
    calls.respond = answer
    mandatory(reader)
    assert len(calls.requests) == 1
    assert ('bound-read', 's1') not in reader.unread_obligations
    assert not reader.pending_variables()


def test_foreign_unrelated_unread_survives_bound_read_and_checkpoint():
    reader, engine, state, calls = scoped_fixture(unrelated=True)
    reader.unread_obligations['foreign', 's2'] = {'unrelated'}
    state.integrate([], reader.pending_variables())
    saved = json.loads(json.dumps(reader.obligation_snapshot()))
    restored = JointReader(engine, state)
    restored.restore_obligations(saved)
    calls.respond = lambda index, data: empty_slots(data)
    mandatory(restored, identity='after-checkpoint')
    assert len(calls.requests) == 1
    assert restored.unread_obligations == {('foreign', 's2'): {'unrelated'}}
    assert 'unrelated' in state.pending
    assert {row['id'] for row in calls.requests[0]['data']['source_spans']} == {'s1'}


def test_same_identity_unread_is_not_erased_when_capacity_prevents_any_call():
    reader, engine, state, calls = scoped_fixture()
    reader.unread_obligations['bound-read', 's1'] = {'history'}
    state.integrate([], reader.pending_variables())
    engine.s['context_tokens'] = 1
    old = deepcopy(reader.unread_obligations)
    with pytest.raises(InputOverflow):
        mandatory(reader)
    assert not calls.requests
    assert reader.unread_obligations == old
    assert 'history' in state.pending


@pytest.mark.parametrize('personal', [False, True])
@pytest.mark.parametrize('repair', [False, True])
def test_actual_audit_schema_requires_explicit_three_way_verdict(personal, repair):
    reader, engine, state, _ = prepare_terminal(personal=personal)
    data = reader._payload(engine.steps[1], 'Audit', ['history', 'suggestion'], [], audit=True)
    if repair:
        data['repair_scope'] = {'failed': [{'variable': 'history', 'id': '*',
                                           'feedback': 'Audit must return an explicit verdict'}]}
    schema = reader._schema(data)
    assert {'slots', 'audit_verdict'} <= set(schema['required'])
    assert set(schema['properties']['audit_verdict']['enum']) == {'supported', 'unknown', 'conflict'}
    assert 'default' not in schema['properties']['audit_verdict']


@pytest.mark.parametrize('phase', ['history', 'terminal', 'terminal_only'])
def test_other_phase_schema_never_demands_an_audit_verdict(phase):
    reader, engine, state, _ = prepare_terminal(personal=True)
    reader.current_terminal_output = phase != 'history'
    data = reader._payload(engine.steps[0 if phase == 'history' else 1], 'Read',
                           ['history'] if phase == 'history' else ['suggestion'], [],
                           semantic=phase != 'history')
    if phase == 'terminal_only':
        data.update(variables={}, requested_slots=[], requested_scopes={},
                    terminal_repair={'expected_value': engine.output_options[0]})
    schema = reader._schema(data)
    assert 'audit_instruction' not in data
    assert 'audit_verdict' not in schema['required']
    if phase == 'history':
        assert set(schema['properties']) == {'slots'}


@pytest.mark.parametrize('bad_row', ['identical', 'changed', 'id_only'])
def test_plain_frozen_restatement_cannot_borrow_old_coverage_without_a_ref(bad_row):
    reader, engine, state, calls, aliases = obligation_fixture(repairs=1)
    proof = observation(PROOF, aliases[:3])
    def answer(index, data):
        if index == 1:
            return claim(deepcopy(proof))
        repeated = deepcopy(proof)
        if bad_row == 'changed':
            repeated.update(value=['invented'], source_ids=[aliases[53]])
        elif bad_row == 'id_only':
            repeated = {'id': PROOF}
        return claim(repeated, irrelevant('remaining-review', aliases[3:53]))
    calls.respond = answer
    read_issue(reader, engine)
    assert OID in reader.evidence_obligations
    assert state.pending == {'history'}
    assert not reader.batches[-1]['complete']
    assert state.valid()[0]['history'] == ['piano']
    assert all(route['value'] != ['invented'] for route in state.routes.values())


def test_restatement_failure_can_be_repaired_without_requiring_the_frozen_row_again():
    reader, engine, state, calls, aliases = obligation_fixture(repairs=2)
    proof = observation(PROOF, aliases[:3])
    def answer(index, data):
        if index == 1:
            return claim(deepcopy(proof))
        if index == 2:
            return claim(deepcopy(proof))
        failures = data['repair_scope']['failed']
        assert not any(item['id'] == PROOF for item in failures)
        # Clearing the invalid repeated row is a legal repair. It does not
        # magically finish the obligation or acquire a new frozen reference.
        return response()
    calls.respond = answer
    read_issue(reader, engine)
    assert len(calls.requests) == 3
    assert OID in reader.evidence_obligations and state.pending == {'history'}
    assert not any(item.get('id') == PROOF and 'still unavailable' in item.get('error', '')
                   for item in reader.batches[-1]['failed_slots'])


def test_explicit_local_frozen_reference_remains_valid_with_full_independent_review():
    reader, engine, state, calls, aliases = obligation_fixture(repairs=1)
    calls.respond = lambda index, data: (claim(observation(PROOF, aliases[:3])) if index == 1 else
        claim(irrelevant('remaining-review', aliases[3:53]), refs=[PROOF]))
    read_issue(reader, engine)
    assert len(calls.requests) == 2
    assert not state.pending and OID not in reader.evidence_obligations
    assert reader.batches[-1]['complete']


@pytest.mark.parametrize('also_reference', [False, True])
def test_restatement_gets_only_a_fresh_local_offer_and_can_then_recover(also_reference):
    reader, engine, state, calls, aliases = obligation_fixture(repairs=2)
    proof = observation(PROOF, aliases[:3])
    def answer(index, data):
        if index == 1:
            return claim(deepcopy(proof))
        if index == 2:
            return claim(deepcopy(proof), refs=[PROOF] if also_reference else [])
        entries = data['repair_scope']['failed']
        assert len(entries) == 1 and entries[0]['id'] == '*'
        assert 'obligation_review_sources' not in entries[0], 'The old coverage difference is stale'
        offer = entries[0]['frozen_observation_reviews']
        offered = set(offer['slot_observation_ids']) | {row['id'] for row in offer['observations']}
        assert offered == {PROOF}
        return claim(irrelevant('complete-original-review', aliases[3:53]), refs=[PROOF])
    calls.respond = answer
    read_issue(reader, engine)
    assert len(calls.requests) == 3 and engine.ledger.used['json_repairs'] == 2
    assert not reader.evidence_obligations and not reader.failed_batch_reviews
    assert not state.pending and reader.batches[-1]['complete']


def test_restatement_offer_does_not_survive_an_unrelated_next_error():
    reader, engine, state, calls, aliases = obligation_fixture(repairs=3)
    proof = observation(PROOF, aliases[:3])
    def answer(index, data):
        if index == 1:
            return claim(deepcopy(proof))
        if index == 2:
            return claim(deepcopy(proof))
        if index == 3:
            assert any('frozen_observation_reviews' in item for item in data['repair_scope']['failed'])
            return {'slots': [{'variable': 'history', 'observations': [], 'unresolved': 'malformed'}]}
        assert not any('frozen_observation_reviews' in item or 'obligation_review_sources' in item
                       for item in data['repair_scope']['failed'])
        assert 'reviewed_observation_ids' not in reader._schema(data)['properties']['slots']['items']['properties']
        return claim(irrelevant('remaining-review', aliases[3:53]), refs=[PROOF])
    calls.respond = answer
    read_issue(reader, engine)
    assert len(calls.requests) == 4
    assert OID in reader.evidence_obligations and state.pending == {'history'}
    assert not reader.batches[-1]['complete']


def add_source(engine, number):
    text = f'The user reported a scoped original historical event number {number}.'
    doc = SimpleNamespace(doc_id=f'extra-{number}', passage=text, metadata={
        'source_segments': [{'role': 'user', 'start': 0, 'end': len(text),
                             'source_message_indices': [number]}]})
    engine.docs[doc.doc_id] = doc
    span = build_source_spans(doc.doc_id, doc, 400)[0]
    alias = f's{number}'
    engine.mapper.sources[alias] = {**span, 'id': alias, 'source_id': span['id']}
    engine.mapper.source_aliases[span['id']] = alias


def test_audit_closure_reproduces_alternative_routes_counters_failed_evidence_and_current_clues():
    reader, engine, state, calls = fixture(max_repairs_per_request=0)
    for number in range(4, 8):
        add_source(engine, number)
    state.integrate([reader._observation(observation('original-proof'), 'history',
                                        visible(reader), engine.steps[0])])
    original = next(iter(state.routes))
    calls.respond = lambda index, data: response(observation('failed-draft', ['s2'], entity_scope='bad-scope'))
    reader.read(engine.steps[0], 'Read original evidence', ['history'], ['painting'], 'failed-history')
    assert reader.failed_batch_reviews
    alternative = reader._observation(observation('alternative-proof', ['s5']), 'history',
                                      visible(reader, ['s5']), engine.steps[0])
    counter = observation('real-counter', ['s4'], stance='contradiction', retract_ids=[original])
    counter['value'] = None
    state.integrate([alternative, reader._observation(counter, 'history', visible(reader, ['s4']), engine.steps[0])])
    reader.evidence_obligations['original-issue'] = {
        'id': 'original-issue', 'variable': 'history', **state.scope('history'),
        'source_ids': [engine.mapper.sources['s3']['source_id']],
        'descriptions': ['A real issue remains to be reviewed.'], 'origins': ['previous']}
    reader.clues['closure-review'] = ['s6']
    state.integrate([], reader.pending_variables())
    old_failed, old_issues = deepcopy(reader.failed_batch_reviews), deepcopy(reader.evidence_obligations)
    def audit(index, data):
        assert index == 2
        shown = {row['id']: row for row in data['source_spans']}
        assert set(shown) == {f's{i}' for i in range(1, 7)}
        assert all(row['text'] == engine.mapper.sources[alias]['text'] for alias, row in shown.items())
        assert len(known_rows(data)) >= 2
        assert any(route['revoked'] for route in known_rows(data))
        assert data['failed_batch_reviews'] and not any(item['review_allowed'] for item in data['failed_batch_reviews'])
        assert data['evidence_obligations']
        return empty_slots(data, audit=True)
    calls.respond = audit
    mandatory(reader, identity='closure-review', variables=['history', 'suggestion'], audit=True)
    assert len(calls.requests) == 2
    assert reader.failed_batch_reviews == old_failed and reader.evidence_obligations == old_issues
    assert state.routes[original]['revoked'] and 'history' in state.pending


def test_bound_read_includes_actual_transitive_parent_sources():
    _, engine, initial, calls = fixture(max_repairs_per_request=0)
    program = deepcopy(initial.initial_program)
    program['variables']['ancestor'] = {'type': 'set', 'demand': 'ancestor'}
    engine.steps[0]['inputs'] = ['ancestor']
    engine.steps.insert(0, {'output_slot': 'ancestor', 'question': 'Read earlier scope',
                            'inputs': [], 'execution': 'retrieval', 'answer_type': 'fact'})
    state = FactState(program, initial.contract, engine.steps, 'suggestion')
    reader = JointReader(engine, state)
    ancestor = reader._observation(observation('root-proof', ['s2']), 'ancestor',
                                   visible(reader, ['s2']), engine.steps[0])
    state.integrate([ancestor])
    parent = reader._observation(observation('parent-proof', ['s1'],
                                             parent_versions={'ancestor': state.versions['ancestor']}),
                                 'history', visible(reader, ['s1']), engine.steps[1])
    state.integrate([parent])
    def answer(index, data):
        assert {'history', 'ancestor'} <= set(data['supported_parents'])
        assert {row['id'] for row in data['source_spans']} == {'s1', 's2'}
        return empty_slots(data)
    calls.respond = answer
    mandatory(reader, step=engine.steps[2])
    assert len(calls.requests) == 1


@pytest.mark.parametrize('counter', [False, True])
def test_owned_source_is_processed_into_durable_failure_or_real_counter(counter):
    reader, engine, state, calls = scoped_fixture()
    reader.unread_obligations['bound-read', 's1'] = {'history'}
    state.integrate([], reader.pending_variables())
    route = next(iter(state.routes))
    def answer(index, data):
        result = empty_slots(data)
        row = observation('reviewed-parent', stance='contradiction' if counter else 'support',
                          retract_ids=[route] if counter else [], entity_scope='user' if counter else 'invalid-scope')
        if counter:
            row['value'] = None
        next(slot for slot in result['slots'] if slot['variable'] == 'history')['observations'] = [row]
        return result
    calls.respond = answer
    mandatory(reader)
    assert len(calls.requests) == 1
    assert ('bound-read', 's1') not in reader.unread_obligations
    if counter:
        assert state.routes[route]['revoked']
        assert 'history' not in state.valid()[0]
    else:
        assert reader.failed_obligations and reader.failed_batch_reviews
        assert 'history' in state.pending
        assert engine.mapper.sources['s1']['source_id'] in next(iter(reader.failed_batch_reviews.values()))['source_ids']


@pytest.mark.parametrize('restore', [False, True])
def test_switching_to_mandatory_mode_and_renaming_cannot_buy_failed_compose_recovery(restore):
    reader, engine, state, calls = scoped_fixture()
    def invalid(index, data):
        result = empty_slots(data)
        next(slot for slot in result['slots'] if slot['variable'] == 'suggestion')['observations'] = [
            observation('failed-compose', value='A suggestion', entity_scope='bad-scope',
                        parent_versions={'history': state.versions['history']})]
        return result
    calls.respond = invalid
    # Existing document-window mode creates the initial paid failure.
    reader.read(engine.steps[1], 'Use historical evidence', ['suggestion'], ['many-spans'], 'first-failed')
    assert reader.failed_batch_reviews
    if restore:
        restored = JointReader(engine, state)
        restored.restore_obligations(json.loads(json.dumps(reader.obligation_snapshot())))
        reader = restored
    before_calls, before_budget = len(calls.requests), deepcopy(engine.ledger.used)
    failed = deepcopy(reader.failed_batch_reviews)
    def no_retry(index, data):
        raise AssertionError('Mode and identity changes cannot authorize failed compose recovery')
    calls.respond = no_retry
    mandatory(reader, identity='renamed-after-mode-change')
    assert len(calls.requests) == before_calls and engine.ledger.used == before_budget
    assert reader.failed_batch_reviews == failed and 'suggestion' in state.pending


def test_global_envelope_repair_can_resubmit_its_not_yet_frozen_counter():
    reader, engine, state, calls = fixture(max_repairs_per_request=1)
    engine.personal_response_selection = True
    state.integrate([reader._observation(observation('prior'), 'history', visible(reader), engine.steps[0])])
    route = next(iter(state.routes))
    counter = observation('required-negative', ['s2'], stance='contradiction', retract_ids=[route])
    counter['value'] = None
    def answer(index, data):
        if index == 1:
            return {**response(deepcopy(counter)), 'semantic_answer': None}
        item = next(item for item in data['repair_scope']['failed'] if item['id'] == counter['id'])
        assert item['required_counter_observations'] == [counter]
        assert not state.routes[route]['revoked']
        return response(deepcopy(counter))
    calls.respond = answer
    reader.read(engine.steps[0], 'Check a source correction', ['history'], ['painting'], 'envelope-counter')
    assert len(calls.requests) == 2 and engine.ledger.used['json_repairs'] == 1
    assert state.routes[route]['revoked'] and not state.pending
    assert reader.batches[-1]['complete']


def test_explicit_whole_value_review_unfreezes_only_the_failed_support_drafts():
    from test_v4_reconcile_review import fragments, personal_reader
    reader, engine, state, calls = personal_reader(max_repairs_per_request=1)
    drafts = fragments()
    def answer(index, data):
        if index == 1:
            return response(*deepcopy(drafts))
        group = data['repair_scope']['whole_value_reviews'][0]
        assert {row['id'] for row in group['draft_observations']} == {row['id'] for row in drafts}
        assert not state.routes
        return response(*[observation(row['id'], ['s1', 's2'], value=['piano', 'painting']) for row in drafts])
    calls.respond = answer
    reader.read(engine.steps[0], 'Read compatible original activities', ['history'],
                ['piano', 'painting'], 'whole-review')
    assert len(calls.requests) == 2
    assert state.valid()[0]['history'] == ['piano', 'painting']
    assert not state.pending and reader.batches[-1]['complete']


def test_explicit_duplicate_replacement_ids_can_preserve_the_original_counter():
    reader, engine, state, calls = fixture(max_repairs_per_request=1)
    state.integrate([reader._observation(observation('prior'), 'history', visible(reader), engine.steps[0])])
    route = next(iter(state.routes))
    support = observation('colliding-draft')
    counter = observation('colliding-draft', ['s2'], stance='contradiction', retract_ids=[route])
    counter['value'] = None
    def answer(index, data):
        if index == 1:
            return {**response(deepcopy(support), deepcopy(counter)), 'audit_verdict': 'unknown'}
        item = next(item for item in data['repair_scope']['failed'] if item['id'] == 'colliding-draft')
        replacements = item['replacement_observations']
        assert len({row['id'] for row in replacements}) == 2
        assert any(row['stance'] == 'contradiction' and row['retract_ids'] == [route] for row in replacements)
        return {**response(*deepcopy(replacements)), 'audit_verdict': 'conflict'}
    calls.respond = answer
    reader.read(engine.steps[0], 'Review conflicting drafts', ['history'], ['piano', 'painting'],
                'duplicate-review', audit=True)
    assert len(calls.requests) == 2 and state.routes[route]['revoked']
    assert reader.batches[-1]['complete'] and not reader.batches[-1]['audit_complete']


def test_paid_new_batch_may_reuse_required_failed_draft_ids_after_checkpoint():
    from test_v4_failed_batch_review import failed_whole, restored_reader, whole_review_response
    reader, engine, state, calls = failed_whole(max_repairs=1)
    reader, _ = restored_reader(reader, engine, state)
    before_calls, before_repairs = len(calls.requests), engine.ledger.used['json_repairs']
    def answer(index, data):
        assert any(item['review_allowed'] for item in data['failed_batch_reviews'])
        assert {'s1', 's2', 's3'} <= {row['id'] for row in data['source_spans']}
        return whole_review_response(data)
    calls.respond = answer
    reader.read(engine.steps[0], 'Review original failed drafts with a new source', ['history'],
                ['advice'], 'authorized-new-batch')
    assert len(calls.requests) == before_calls + 1
    assert engine.ledger.used['json_repairs'] == before_repairs + 1
    assert not state.pending and not reader.failed_batch_reviews


@pytest.mark.parametrize('audit', [False, True])
def test_parent_summary_declaration_does_not_authorize_revision_at_terminal_or_audit(audit):
    from test_v4_summary_revision_review import summary_fixture, revised
    reader, engine, state, calls, route = summary_fixture()
    reader.current_terminal_output = not audit
    data = reader._payload(engine.steps[1], 'Review the final response', ['history', 'suggestion'], [],
                           semantic=not audit, audit=audit)
    assert 'summary_revision_contracts' not in data
    properties = reader._schema(data)['properties']['slots']['items']['properties']['observations']['items']['properties']
    assert 'summary_revision' not in properties
    with pytest.raises(ProtocolError, match='never terminal or audit'):
        reader._observation(revised(route, state.versions['history'], sources=['s1']), 'history',
                            {row['id']: row for row in data['source_spans']}, engine.steps[1],
                            require_grounding=not audit, allow_summary_revision=False)
    assert not calls.requests
