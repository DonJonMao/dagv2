"""Independent proof-scope contracts using real state and scripted transport.

Sources and model judgments here are synthetic. No labels or service requests
are used; assertions concern provenance, visibility and authorization.
"""
from copy import deepcopy
import json

import jsonschema
import pytest

from dagbt.fact_state import FactState
from dagbt.joint_reading import JointReader
from dagbt.methods import capabilities
from dagbt.reasoning import ProtocolError
from dagbt.residual_control import ResidualControl
from test_personamem_joint_protocol import fixture, observation, response, visible
from test_v4_joint_reading import terminal_response
from test_v4_reader_review import sourced_gap_fixture
from test_v4_failed_batch_review import failed_whole
from test_v4_summary_revision_review import (
    OLD, EXPANDED, summary_fixture, revised, read_new_history,
    add_explicit_piano_correction,
)


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Independent proof-scope review must remain offline')
    monkeypatch.setattr('urllib.request.urlopen', forbidden)
    monkeypatch.setattr('urllib.request.build_opener', forbidden)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def proof_fixture(*, personal=True):
    reader, engine, state, calls = fixture(max_repairs_per_request=0, context_tokens=100000)
    engine.personal_response_selection = personal
    engine.output_options = ['Try a piano duet.', 'Go painting.', 'Try rugby.', 'No advice.']
    engine.final_node_id = state.final_node_id
    rows = [reader._observation(observation(name, sources), 'history', visible(reader, sources), engine.steps[0])
            for name, sources in [('selected-two-premises', ['s1', 's2']),
                                  ('independent-three-premises', ['s1', 's2', 's3'])]]
    state.integrate(rows)
    chosen = state.valid()[2]['history'][0]
    alternative = next(rid for rid in state.routes if rid != chosen)
    assert len(state.routes[chosen]['source_ids']) == 2
    assert len(state.routes[alternative]['source_ids']) == 3
    return reader, engine, state, calls, chosen, alternative


def terminal_payload(reader, *, audit=False):
    reader.current_terminal_output = not audit
    return reader._payload(reader.e.steps[-1], 'Use the established history',
                           ['suggestion', 'history'], [], semantic=not audit,
                           audit=audit, proof_projection=True)


def source_ids(data):
    return {source['id'] for source in data['source_spans']}


def known_ids(data):
    return {row['id'] for row in data['known_observations']}


def counter(reader, route_id, *, source='s4', name='sourced-counter'):
    draft = observation(name, [source], stance='contradiction', retract_ids=[route_id])
    draft['value'] = None
    return reader._observation(draft, 'history', visible(reader, [source]), reader.e.steps[0])


@pytest.mark.parametrize('audit', [False, True])
@pytest.mark.parametrize('personal', [False, True])
def test_selected_or_route_retains_every_internal_premise_without_deleting_alternatives(audit, personal):
    reader, engine, state, calls, chosen, alternative = proof_fixture(personal=personal)
    before = canonical((state.public_dict(), engine.mapper.sources, engine.ledger.used))
    payload = terminal_payload(reader, audit=audit)
    assert known_ids(payload) == {chosen}
    assert source_ids(payload) == {'s1', 's2'}, 'Both premises inside the selected AND proof remain visible'
    assert set(payload['supported_parents']['history']['source_ids']) == {'s1', 's2'}
    assert alternative in state.routes and not state.routes[alternative]['revoked']
    assert canonical((state.public_dict(), engine.mapper.sources, engine.ledger.used)) == before
    assert not calls.requests


@pytest.mark.parametrize('audit', [False, True])
def test_counter_to_nonselected_route_preserves_target_and_all_counter_originals(audit):
    reader, engine, state, calls, chosen, alternative = proof_fixture()
    add_explicit_piano_correction(engine)
    state.integrate([counter(reader, alternative)])
    assert state.valid()[2]['history'] == [chosen]
    before = canonical(state.public_dict())
    payload = terminal_payload(reader, audit=audit)
    assert {chosen, alternative} <= known_ids(payload)
    assert source_ids(payload) == {'s1', 's2', 's3', 's4'}
    revoked = next(row for row in payload['known_observations'] if row['id'] == alternative)
    assert revoked['revoked'][0]['source_ids'] == ['s4']
    assert canonical(state.public_dict()) == before and not calls.requests


@pytest.mark.parametrize('audit', [False, True])
def test_conflicting_values_cannot_disappear_when_no_selected_binding_exists(audit):
    reader, engine, state, _, chosen, alternative = proof_fixture()
    conflicting = reader._observation(observation('different-whole-value', ['s3'], value=['painting']),
                                      'history', visible(reader, ['s3']), engine.steps[0])
    state.integrate([conflicting])
    assert 'history' not in state.valid()[0]
    before = canonical(state.public_dict())
    payload = terminal_payload(reader, audit=audit)
    assert known_ids(payload) == set(state.routes)
    assert {canonical(row['value']) for row in payload['known_observations']} == {
        canonical(['piano']), canonical(['painting'])}
    assert source_ids(payload) == {'s1', 's2', 's3'}
    assert canonical(state.public_dict()) == before


def test_selected_summary_keeps_cumulative_original_sources_and_archived_ancestry():
    reader, engine, state, calls, original = summary_fixture(context_tokens=100000)
    calls.respond = lambda index, data: response(revised(original, state.versions['history']))
    read_new_history(reader, engine)
    selected = state.valid()[2]['history'][0]
    assert selected != original and state.summary_ancestors(state.routes[selected]) == {original}
    before = canonical(state.public_dict())
    payload = terminal_payload(reader)
    assert source_ids(payload) == {'s1', 's2'}
    selected_row = next(row for row in payload['known_observations'] if row['id'] == selected)
    assert set(selected_row['source_ids']) == {'s1', 's2'}
    assert selected_row['value'] == EXPANDED
    assert state.routes[original]['superseded_by'] == [selected]
    assert canonical(state.public_dict()) == before


@pytest.mark.parametrize('audit', [False, True])
def test_revoked_summary_ancestor_and_inheriting_descendant_remain_visible(audit):
    reader, engine, state, calls, original = summary_fixture(context_tokens=100000)
    calls.respond = lambda index, data: response(revised(original, state.versions['history']))
    read_new_history(reader, engine)
    descendant = state.valid()[2]['history'][0]
    add_explicit_piano_correction(engine)
    state.integrate([counter(reader, original)])
    assert not state.is_active_route(state.routes[descendant])
    assert 'history' not in state.valid()[0]
    before = canonical(state.public_dict())
    payload = terminal_payload(reader, audit=audit)
    assert {original, descendant} <= known_ids(payload)
    assert source_ids(payload) == {'s1', 's2', 's4'}
    ancestor = next(row for row in payload['known_observations'] if row['id'] == original)
    child = next(row for row in payload['known_observations'] if row['id'] == descendant)
    assert ancestor['revoked'][0]['source_ids'] == ['s4']
    assert child['summary_revision']['supersedes_route_ids'] == [original]
    assert canonical(state.public_dict()) == before


def test_archived_or_route_can_be_selected_after_checkpoint_and_real_selected_route_revocation():
    reader, engine, state, _, chosen, alternative = proof_fixture()
    payload = terminal_payload(reader)
    assert alternative not in known_ids(payload)
    checkpoint = json.loads(json.dumps(state.public_dict()))
    restored = FactState(state.initial_program, state.contract, engine.steps, state.final_node_id)
    restored.restore(checkpoint)
    reader = JointReader(engine, restored)
    add_explicit_piano_correction(engine)
    before_version = restored.versions['history']
    restored.integrate([counter(reader, chosen)])
    assert restored.valid()[2]['history'] == [alternative]
    assert restored.versions['history'] != before_version
    assert {chosen, alternative} == set(restored.routes)
    after = terminal_payload(reader)
    assert set(after['supported_parents']['history']['source_ids']) == {'s1', 's2', 's3'}
    assert alternative in known_ids(after)
    assert 's4' in source_ids(after)


def test_history_extraction_and_unprojected_payload_keep_all_route_candidates():
    reader, engine, state, _, chosen, alternative = proof_fixture()
    reader.current_terminal_output = False
    history = reader._payload(engine.steps[0], 'Read history', ['history'], [], proof_projection=True)
    assert history['phase'] == 'history_evidence'
    assert known_ids(history) == {chosen, alternative}
    assert source_ids(history) == {'s1', 's2', 's3'}
    old_boundary = reader._payload(engine.steps[1], 'Use history', ['suggestion', 'history'], [], semantic=True)
    assert known_ids(old_boundary) == {chosen, alternative}


def test_intermediate_history_compose_is_not_mistaken_for_terminal_projection():
    _, engine, initial, calls, _, _ = proof_fixture()
    program = deepcopy(initial.initial_program)
    program['variables']['suggestion'].update(
        evidence_semantics='personal_history_summary', scope={'entity_scope': 'user', 'time_scope': 'history'})
    program['variables']['final'] = {'type': 'text', 'demand': 'final'}
    program['expression'] = {'op': 'var', 'id': 'final'}
    engine.steps.append({'output_slot': 'final', 'question': 'Use {suggestion}', 'inputs': ['suggestion'],
                         'execution': 'compose', 'answer_type': 'fact'})
    engine.final_node_id = 'final'
    state = FactState(program, initial.contract, engine.steps, 'final')
    reader = JointReader(engine, state)
    for index, aliases in enumerate([['s1', 's2'], ['s1', 's2', 's3']]):
        row = reader._observation(observation(f'parent-{index}', aliases), 'history',
                                  visible(reader, aliases), engine.steps[0])
        state.integrate([row])
    payload = reader._payload(engine.steps[1], 'Summarize history', ['suggestion', 'history'], [],
                              proof_projection=True)
    assert payload['phase'] == 'history_evidence'
    assert known_ids(payload) == set(state.routes)
    assert source_ids(payload) == {'s1', 's2', 's3'}
    assert not calls.requests


def test_history_summary_revision_keeps_all_current_targets_and_their_required_sources():
    reader, engine, state, _, original = summary_fixture(context_tokens=100000)
    alternate = reader._observation(observation('other-whole-proof', ['s2'], value=OLD),
                                    'history', visible(reader, ['s2']), engine.steps[0])
    state.integrate([alternate])
    reader.current_terminal_output = False
    data = reader._payload(engine.steps[0], 'Extend history', ['history'], [], proof_projection=True)
    assert known_ids(data) == set(state.routes)
    contract = data['summary_revision_contracts']['history']
    assert set(contract['eligible_route_ids']) == set(state.routes)
    assert set(contract['required_source_ids']) == {'s1', 's2'}
    assert source_ids(data) == {'s1', 's2'}


@pytest.mark.parametrize('restore', [False, True])
def test_projection_preserves_sourced_unresolved_obligations_and_pending(restore):
    reader, engine, state, calls = sourced_gap_fixture()
    if restore:
        fresh = JointReader(engine, state)
        fresh.restore_obligations(json.loads(json.dumps(reader.obligation_snapshot())))
        reader = fresh
    before = canonical((reader.obligation_snapshot(), state.public_dict(), engine.ledger.used))
    count = len(calls.requests)
    data = terminal_payload(reader)
    assert data['evidence_obligations']
    assert {'s1', 's2'} <= source_ids(data)
    assert 'history' in state.pending and 'history' in reader.pending_variables()
    assert canonical((reader.obligation_snapshot(), state.public_dict(), engine.ledger.used)) == before
    assert len(calls.requests) == count


@pytest.mark.parametrize('audit', [False, True])
def test_terminal_projection_preserves_failed_drafts_whole_reviews_and_their_originals(audit):
    reader, engine, state, calls = failed_whole()
    fresh = JointReader(engine, state)
    fresh.restore_obligations(json.loads(json.dumps(reader.obligation_snapshot())))
    before = canonical((fresh.obligation_snapshot(), state.public_dict(), engine.ledger.used))
    count = len(calls.requests)
    data = terminal_payload(fresh, audit=audit)
    assert data['failed_batch_reviews']
    assert {record['id'] for record in data['failed_batch_reviews']} == set(fresh.failed_batch_reviews)
    assert not any(record['review_allowed'] for record in data['failed_batch_reviews'])
    assert {'s1', 's2'} <= source_ids(data)
    assert all(record['failed'] and record['whole_value_reviews'] for record in data['failed_batch_reviews'])
    assert 'history' in state.pending
    assert canonical((fresh.obligation_snapshot(), state.public_dict(), engine.ledger.used)) == before
    assert len(calls.requests) == count


@pytest.mark.parametrize('owner', ['foreign-history', 'terminal-proof'])
def test_unread_work_on_omitted_alternative_source_is_not_erased_or_migrated(owner):
    reader, engine, state, calls, _, _ = proof_fixture()
    reader.unread_obligations[owner, 's3'] = {'history'}
    state.integrate([], reader.pending_variables())
    before = canonical((reader.obligation_snapshot(), state.public_dict(), engine.ledger.used))
    if owner == 'foreign-history':
        with pytest.raises(ProtocolError, match='mandatory_closure'):
            reader.read(engine.steps[1], 'Use history', ['suggestion'], [], 'terminal-proof',
                        semantic=True, terminal_output=True, source_scope='mandatory_closure')
    else:
        # A conservative projector may retain all sources of this pending
        # variable. Admission is legal only with the owned source and variable
        # still explicitly present; it must not erase the old marker to fit.
        try:
            data = reader._mandatory_scope_payload(engine.steps[1], 'Use history', ['suggestion', 'history'],
                [], 'terminal-proof', semantic=True, audit=False, terminal_output=True)
        except ProtocolError:
            pass
        else:
            assert 's3' in source_ids(data) and 'history' in data['requested_slots']
    assert not calls.requests
    assert canonical((reader.obligation_snapshot(), state.public_dict(), engine.ledger.used)) == before


def test_hidden_route_is_rejected_by_schema_and_local_offered_target_validation():
    reader, engine, state, _, chosen, alternative = proof_fixture()
    data = terminal_payload(reader)
    draft = observation('guessed-hidden-counter', ['s1'], stance='contradiction',
                        retract_ids=[alternative])
    draft['value'] = None
    result = {'slots': [{'variable': variable, 'observations': [draft] if variable == 'history' else [],
                         'unresolved': []} for variable in data['requested_slots']]}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(result, reader._schema(data))
    before = canonical(state.public_dict())
    with pytest.raises(ProtocolError):
        reader._observation(draft, 'history', {row['id']: row for row in data['source_spans']},
                            engine.steps[0], offered_route_ids=known_ids(data))
    assert canonical(state.public_dict()) == before
    assert chosen in known_ids(data) and alternative not in known_ids(data)


@pytest.mark.parametrize('repairs', [0, 1])
def test_actual_terminal_response_cannot_guess_hidden_route_id_to_revoke_it(repairs):
    reader, engine, state, calls, _, alternative = proof_fixture()
    engine.s['max_repairs_per_request'] = repairs

    def malicious_reply(index, data):
        assert alternative not in known_ids(data)
        draft = observation('guessed-hidden-counter', ['s1'], stance='contradiction',
                            retract_ids=[alternative])
        draft['value'] = None
        return {'slots': [{'variable': variable, 'observations': [draft] if variable == 'history' else [],
                           'unresolved': []} for variable in data['requested_slots']],
                'semantic_answer': None, 'final_prediction': None}

    calls.respond = malicious_reply
    reader.read(engine.steps[1], 'Use history', ['suggestion'], [], 'malicious-terminal',
                semantic=True, terminal_output=True, source_scope='mandatory_closure')
    assert len(calls.requests) == 1 + repairs
    assert not state.routes[alternative]['revoked']
    assert reader.batches[-1]['failed_slots']
    assert 'history' in state.pending


@pytest.mark.parametrize('kind', ['pending', 'evidence', 'failed'])
def test_unresolved_terminal_target_protects_its_parent_alternatives_even_in_parent_only_repair(kind):
    reader, engine, state, calls, chosen, alternative = proof_fixture()
    initial = terminal_payload(reader)
    assert alternative not in known_ids(initial) and 's3' not in source_ids(initial)
    if kind == 'pending':
        state.integrate([], {'suggestion'})
    else:
        def incomplete(index, data):
            assert alternative not in known_ids(data)
            if kind == 'failed':
                result = terminal_response(data, engine.output_options[0])
                next(slot for slot in result['slots'] if slot['variable'] == 'suggestion')['observations'][0][
                    'entity_scope'] = 'wrong-entity'
                return result
            return {'slots': [{'variable': variable, 'observations': [],
                               'unresolved': ['The old history does not establish a suitable response.']
                               if variable == 'suggestion' else []}
                              for variable in data['requested_slots']],
                    'semantic_answer': None, 'final_prediction': None}
        calls.respond = incomplete
        reader.read(engine.steps[1], 'Resolve the response choice', ['suggestion'], [], 'unsolved-terminal',
                    semantic=True, terminal_output=True, source_scope='mandatory_closure')
        records = reader.evidence_obligations if kind == 'evidence' else reader.failed_batch_reviews
        assert any(record['variable'] == 'suggestion' for record in records.values())
        assert 'suggestion' in state.pending
    before = canonical((state.public_dict(), reader.obligation_snapshot(), engine.ledger.used))
    count = len(calls.requests)
    for requested in [['suggestion', 'history'], ['history']]:
        data = reader._payload(engine.steps[1], 'Reconsider response using new parent context',
                              requested, [], semantic=True, proof_projection=True)
        assert {chosen, alternative} <= known_ids(data)
        assert source_ids(data) >= {'s1', 's2', 's3'}
    assert canonical((state.public_dict(), reader.obligation_snapshot(), engine.ledger.used)) == before
    assert len(calls.requests) == count


def terminal_sibling_proof_fixture():
    _, engine, _, calls = fixture(max_repairs_per_request=0, context_tokens=100000)
    program = {
        'variables': {'history': {'type': 'bool', 'demand': 'history'},
                      'decisive': {'type': 'bool', 'demand': 'suggestion'},
                      'sibling': {'type': 'bool', 'demand': 'suggestion'}},
        'rules_complete': True,
        'expression': {'op': 'and', 'args': [
            {'op': 'var', 'id': 'decisive'}, {'op': 'var', 'id': 'sibling'}]},
    }
    contract = {'mode': 'value', 'result_type': 'bool', 'entity_scope': 'user',
                'time_scope': 'history', 'explanation': 'none'}
    state = FactState(program, contract, engine.steps, 'suggestion')
    engine.fact_state, engine.final_node_id = state, 'suggestion'
    engine.s['max_enumeration_states'] = 10000
    engine.capabilities = capabilities(engine.s['algorithm_version'])
    controller = ResidualControl(engine)
    engine.residual_control = controller
    reader = controller.reader
    for name, aliases in [('selected', ['s1', 's2']), ('alternative', ['s1', 's2', 's3'])]:
        state.integrate([reader._observation(observation(name, aliases, value=True), 'history',
                         visible(reader, aliases), engine.steps[0])])
    chosen = state.valid()[2]['history'][0]
    alternative = next(rid for rid in state.routes if rid != chosen)
    state.integrate([reader._observation(
        observation('decisive-proof', ['s1'], value=False,
                    parent_versions={'history': state.versions['history']}),
        'decisive', visible(reader, ['s1']), engine.steps[1])])
    certificate = controller.candidate()
    assert certificate['output'] is False
    assert set(certificate['binding_dependencies']) == {'decisive', 'history'}
    calls.respond = lambda index, data: {
        'slots': [{'variable': variable, 'observations': [], 'unresolved': []}
                  for variable in data['requested_slots']], 'audit_verdict': 'supported'}
    return controller, reader, engine, state, calls, chosen, alternative, certificate


def test_audit_eliminated_pending_terminal_sibling_does_not_protect_parent_alternatives():
    controller, reader, engine, state, calls, chosen, alternative, certificate = terminal_sibling_proof_fixture()
    reader.evidence_obligations['sibling-gap'] = {
        'id': 'sibling-gap', 'variable': 'sibling', **state.scope('sibling'),
        'source_ids': [engine.mapper.sources['s3']['source_id']],
        'descriptions': ['The independent sibling remains unresolved.'], 'origins': ['sibling-read']}
    state.integrate([], reader.pending_variables())
    assert state.pending == {'sibling'}
    assert controller.candidate() == certificate
    before = canonical((state.routes, state.versions, state.program, reader.obligation_snapshot()))
    assert controller.audit(certificate)
    assert len(calls.requests) == 1
    data = calls.requests[0]['data']
    assert set(data['requested_slots']) == {'decisive', 'history'}
    assert chosen in known_ids(data) and alternative not in known_ids(data)
    assert source_ids(data) == {'s1', 's2'}
    assert not data.get('evidence_obligations')
    assert canonical((state.routes, state.versions, state.program, reader.obligation_snapshot())) == before
    assert state.pending == {'sibling'}
    assert alternative in state.routes and controller.candidate() == certificate


def test_actual_pending_terminal_dependency_keeps_parent_alternatives_and_blocks_certificate():
    controller, reader, engine, state, calls, chosen, alternative, certificate = terminal_sibling_proof_fixture()
    state.integrate([], {'decisive'})
    assert controller.candidate() is None, 'An actual unresolved dependency cannot be published'
    step, variables = controller.audit_scope(certificate)
    before = canonical((state.public_dict(), reader.obligation_snapshot(), engine.ledger.used))
    data = reader._payload(step, 'Inspect the unresolved actual frontier', variables, [],
                           audit=True, proof_projection=True)
    assert {chosen, alternative} <= known_ids(data)
    assert source_ids(data) == {'s1', 's2', 's3'}
    assert canonical((state.public_dict(), reader.obligation_snapshot(), engine.ledger.used)) == before
    assert not calls.requests and controller.candidate() is None


def test_valid_counter_captured_before_envelope_repair_cannot_be_retargeted_to_hidden_route():
    reader, engine, state, calls, chosen, alternative = proof_fixture()
    engine.s['max_repairs_per_request'] = 1

    def reply(index, data):
        assert chosen in known_ids(data) and alternative not in known_ids(data)
        draft = observation('protected-counter', ['s1'], stance='contradiction',
                            retract_ids=[chosen if index == 1 else alternative])
        draft['value'] = None
        result = {'slots': [{'variable': variable, 'observations': [draft] if variable == 'history' else [],
                             'unresolved': []} for variable in data['requested_slots']],
                  'semantic_answer': None, 'final_prediction': None}
        if index == 1:
            result['undeclared_envelope_field'] = True
        return result

    calls.respond = reply
    reader.read(engine.steps[1], 'Audit the scoped old belief', ['suggestion'], [], 'counter-envelope',
                semantic=True, terminal_output=True, source_scope='mandatory_closure')
    assert len(calls.requests) == 2
    assert not state.routes[alternative]['revoked']
    required = [counter for record in reader.failed_batch_reviews.values()
                for failure in record['failed'] for counter in failure.get('required_counter_observations', [])]
    assert any(row['retract_ids'] == [chosen] for row in required)
    assert 'history' in state.pending


def test_prospective_handoff_and_actual_terminal_share_selected_proof_scope(monkeypatch):
    reader, engine, state, calls = fixture(max_repairs_per_request=0, context_tokens=100000)
    engine.personal_response_selection = True
    engine.output_options = ['Try a piano duet.', 'Go painting.', 'Try rugby.', 'No advice.']
    engine.final_node_id = 'suggestion'
    broad = reader._observation(observation('broad-existing', ['s1', 's2', 's3']),
                                'history', visible(reader, ['s1', 's2', 's3']), engine.steps[0])
    state.integrate([broad])
    old = state.valid()[2]['history'][0]
    narrow = reader._observation(observation('new-sufficient', ['s1', 's2']),
                                 'history', visible(reader, ['s1', 's2']), engine.steps[0])
    captured = []
    fits = reader._fits

    def capture(operation, data):
        captured.append((operation, deepcopy(data)))
        return True

    monkeypatch.setattr(reader, '_fits', capture)
    assert reader._handoff_failures([narrow], {('history', 'new-sufficient')}) == {}
    proposed = next(data for operation, data in captured if operation == 'semantic_compose')
    assert old not in known_ids(proposed)
    assert source_ids(proposed) == {'s1', 's2'}
    state.integrate([narrow])
    monkeypatch.setattr(reader, '_fits', fits)
    calls.respond = lambda index, data: terminal_response(data, engine.output_options[0])
    reader.read(engine.steps[1], proposed['grounded_subquestion'], ['suggestion'], [], 'actual-terminal',
                semantic=True, terminal_output=True, source_scope='mandatory_closure')
    assert len(calls.requests) == 1
    actual = calls.requests[0]['data']
    for field in ['supported_parents', 'known_observations', 'source_spans', 'requested_slots',
                  'required_parent_versions']:
        assert canonical(actual[field]) == canonical(proposed[field]), field
    assert old in state.routes


def test_generic_final_retrieval_handoff_matches_unprojected_document_window_execution(monkeypatch):
    _, engine, initial, calls = fixture(max_repairs_per_request=0, context_tokens=100000)
    engine.steps[-1]['execution'] = 'retrieval'
    engine.personal_response_selection = False
    engine.output_options = ['Try a piano duet.', 'Go painting.', 'Try rugby.', 'No advice.']
    engine.final_node_id = 'suggestion'
    state = FactState(initial.initial_program, initial.contract, engine.steps, 'suggestion')
    reader = JointReader(engine, state)
    broad = reader._observation(observation('prior-broad', ['s1', 's2', 's3']),
                                'history', visible(reader, ['s1', 's2', 's3']), engine.steps[0])
    state.integrate([broad])
    old = state.valid()[2]['history'][0]
    narrow = reader._observation(observation('new-narrow', ['s1', 's2']),
                                 'history', visible(reader, ['s1', 's2']), engine.steps[0])
    captured = []
    fits = reader._fits
    monkeypatch.setattr(reader, '_fits', lambda operation, data: captured.append((operation, deepcopy(data))) or True)
    assert reader._handoff_failures([narrow], {('history', 'new-narrow')}) == {}
    proposed = next(data for operation, data in captured if data['task']['output_slot'] == 'suggestion')
    assert old in known_ids(proposed)
    assert source_ids(proposed) == {'s1', 's2', 's3'}
    state.integrate([narrow])
    monkeypatch.setattr(reader, '_fits', fits)
    calls.respond = lambda index, data: terminal_response(data, engine.output_options[0])
    reader.read(engine.steps[-1], proposed['grounded_subquestion'], ['suggestion'], ['piano'],
                'generic-retrieval-terminal', terminal_output=True, source_scope='document_windows')
    assert len(calls.requests) == 1
    actual = calls.requests[0]['data']
    assert known_ids(actual) == known_ids(proposed)
    assert source_ids(actual) == source_ids(proposed)
    assert canonical(actual['supported_parents']) == canonical(proposed['supported_parents'])
