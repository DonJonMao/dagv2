"""Independent summary-version contracts; model answers are scripted offline.

Compatible historical additions do not become factual contradictions. Revisions
must survive the real reader, provenance state, and downstream admission checks.
"""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from dagbt.fact_state import FactState
from dagbt.evidence_spans import build_source_spans
from dagbt.joint_reading import JointReader
from dagbt.reasoning import ProtocolError
from dagbt.residual import ProgramError
from dagbt.residual_plan import validate_plan
from test_personamem_joint_protocol import fixture, observation, response, visible


OLD = 'The user played piano last year.'
EXPANDED = OLD + ' The user enjoyed painting last spring.'


def summary_fixture(*, marked=True, **settings):
    _, engine, old_state, calls = fixture(max_repairs_per_request=0, **settings)
    engine.personal_response_selection = True
    program = deepcopy(old_state.program)
    program['variables']['history'] = {
        'type': 'text', 'demand': 'history',
        'scope': {'entity_scope': 'user', 'time_scope': 'history'},
        **({'evidence_semantics': 'personal_history_summary'} if marked else {}),
    }
    contract = {**old_state.contract, 'time_scope': 'response'}
    state = FactState(program, contract, engine.steps, 'suggestion')
    reader = JointReader(engine, state)
    row = observation('original-summary', ['s1'], value=OLD)
    state.integrate([reader._observation(row, 'history', visible(reader), engine.steps[0])])
    original = next(iter(state.routes))
    return reader, engine, state, calls, original


def revised(original, version, *, sources=('s1', 's2'), value=EXPANDED, **changes):
    return observation('expanded-summary', sources, value=value, summary_revision={
        'supersedes_route_ids': [original], 'previous_version': version,
        'reason': 'Reviewing both user originals preserves the piano history and adds the painting history.',
    }, **changes)


def read_new_history(reader, engine, *, identity='new-history', audit=False):
    reader.read(engine.steps[0], 'Review sourced historical activities', ['history'],
                ['painting'], identity, audit=audit)


def add_explicit_piano_correction(engine):
    text = 'My earlier claim that I played piano last year was a mistake. I did not play piano then.'
    document = SimpleNamespace(doc_id='piano-correction', passage=text, metadata={
        'source_segments': [{'role': 'user', 'start': 0, 'end': len(text),
                             'source_message_indices': [3]}]})
    engine.docs[document.doc_id] = document
    original = build_source_spans(document.doc_id, document, 400)[0]
    engine.mapper.sources['s4'] = {**original, 'id': 's4', 'source_id': original['id']}
    engine.mapper.source_aliases[original['id']] = 's4'


def assert_unchanged_original(state, original, saved):
    assert state.routes[original] == saved
    assert state.valid()[0]['history'] == OLD
    assert state.pending == {'history'}


@pytest.mark.parametrize('forbidden', ['number', 'rule', 'finite_domain', 'terminal', 'deterministic'])
def test_summary_representation_marker_cannot_redefine_other_variable_semantics(forbidden):
    _, engine, state, _, _ = summary_fixture()
    program, steps = deepcopy(state.initial_program), deepcopy(engine.steps)
    declaration = program['variables']['history']
    if forbidden in {'number', 'rule'}:
        declaration['type'] = forbidden
    elif forbidden == 'finite_domain':
        declaration['domain'] = [OLD, EXPANDED]
    elif forbidden == 'terminal':
        program['variables']['suggestion']['evidence_semantics'] = 'personal_history_summary'
    else:
        steps.insert(0, {'output_slot': 'source_history', 'question': 'Read a historical fact',
                         'inputs': [], 'execution': 'retrieval', 'answer_type': 'fact'})
        steps[1]['execution'] = 'compose'
        steps[1]['inputs'] = ['source_history']
        program['variables']['source_history'] = {
            'type': 'text', 'demand': 'source_history', 'scope': deepcopy(declaration['scope'])}
        program['task_expressions'] = {'history': {'op': 'var', 'id': 'source_history'}}
    with pytest.raises(ProgramError):
        FactState(program, state.contract, steps, 'suggestion')


def test_generic_planning_cannot_opt_itself_into_personal_summary_revision():
    _, engine, state, _, _ = summary_fixture()
    plan = {'steps': deepcopy(engine.steps), 'final_node_id': 'suggestion',
            'program': deepcopy(state.initial_program), 'output_contract': deepcopy(state.contract)}
    with pytest.raises(ProtocolError):
        validate_plan(plan, lambda value: value)
    accepted = validate_plan(plan, lambda value: value, personal_history=True)
    assert accepted['program']['variables']['history']['evidence_semantics'] == 'personal_history_summary'


def test_explicit_compatible_revision_retires_binding_without_falsifying_old_facts():
    reader, engine, state, calls, original = summary_fixture()
    engine.output_options = ['A sourced suggestion.']
    version = state.versions['history']
    terminal = observation('dependent-response', ['s1'], value='A sourced suggestion.',
                           time_scope='response', parent_versions={'history': version})
    state.integrate([reader._observation(terminal, 'suggestion', visible(reader), engine.steps[1])])
    assert 'suggestion' in state.valid()[0]
    old = deepcopy(state.routes[original])
    def answer(index, data):
        assert data['phase'] == 'history_evidence' and 'public_options' not in data
        assert {'s1', 's2'} <= {source['id'] for source in data['source_spans']}
        assert state.routes[original] == old, 'The old binding must still exist before validation'
        return response(revised(original, version))
    calls.respond = answer
    read_new_history(reader, engine)
    values, sources, route_ids = state.valid()
    replacement = route_ids['history'][0]
    assert values['history'] == EXPANDED and 'suggestion' not in values
    assert state.versions['history'] > version and not state.pending
    assert state.routes[original]['value'] == OLD and not state.routes[original]['revoked']
    assert replacement in state.routes[original]['superseded_by']
    assert state.routes[replacement]['summary_revision']['supersedes_route_ids'] == [original]
    assert set(sources['history']) == {engine.mapper.sources[s]['source_id'] for s in ('s1', 's2')}
    assert len(calls.requests) == 1 and engine.ledger.used['llm'] == 1
    assert not engine.ledger.used.get('json_repairs', 0)


@pytest.mark.parametrize('mistake', [
    'unmarked_text', 'generic_engine', 'audit', 'old_source_missing',
    'unknown_route', 'stale_version', 'wrong_scope', 'invented_parent',
])
def test_revision_cannot_bypass_eligibility_original_text_or_binding_identity(mistake):
    reader, engine, state, calls, original = summary_fixture(marked=mistake != 'unmarked_text')
    old = deepcopy(state.routes[original])
    draft = revised(original, state.versions['history'])
    if mistake == 'generic_engine':
        engine.personal_response_selection = False
    elif mistake == 'old_source_missing':
        draft['source_ids'] = ['s2']
    elif mistake == 'unknown_route':
        draft['summary_revision']['supersedes_route_ids'] = ['missing-route']
    elif mistake == 'stale_version':
        draft['summary_revision']['previous_version'] -= 1
    elif mistake == 'wrong_scope':
        draft['time_scope'] = 'current'
    elif mistake == 'invented_parent':
        draft['parent_versions'] = {'suggestion': 0}
    def answer(index, data):
        result = response(deepcopy(draft))
        if mistake == 'audit':
            result['audit_verdict'] = 'supported'
        return result
    calls.respond = answer
    read_new_history(reader, engine, audit=mistake == 'audit')
    assert_unchanged_original(state, original, old)
    assert reader.pending_variables() == {'history'}


def test_direct_observation_normalization_cannot_authorize_a_summary_revision():
    reader, engine, state, _, original = summary_fixture()
    draft = revised(original, state.versions['history'])
    with pytest.raises(ProtocolError):
        reader._observation(draft, 'history', visible(reader, ['s1', 's2']), engine.steps[0])
    assert not state.routes[original].get('superseded_by')


def test_failed_downstream_handoff_cannot_retire_old_summary_or_publish_revision(monkeypatch):
    reader, engine, state, calls, original = summary_fixture()
    old = deepcopy(state.routes[original])
    calls.respond = lambda index, data: response(revised(original, state.versions['history']))
    probed = []
    def reject_handoff(rows, selected, prediction, semantic_result, **kwargs):
        assert state.routes[original] == old
        probed.extend(rows)
        return {(row['variable'], row['id']): 'citation_capacity: original evidence cannot fit the mandatory downstream request'
                for row in rows if row['stance'] == 'support'}
    monkeypatch.setattr(reader, '_handoff_failures', reject_handoff)
    read_new_history(reader, engine)
    assert probed and any(row.get('summary_revision') for row in probed)
    assert_unchanged_original(state, original, old)
    assert reader.failed_obligations


def test_superseded_binding_is_not_an_eligible_target_for_a_later_revision():
    reader, engine, state, calls, original = summary_fixture()
    calls.respond = lambda index, data: response(revised(original, state.versions['history']))
    read_new_history(reader, engine)
    active = state.valid()[2]['history'][0]
    before = deepcopy(state.routes)
    # Current variable version does not make an obsolete ancestor a current route.
    calls.respond = lambda index, data: response(revised(original, state.versions['history']))
    reader.read(engine.steps[0], 'Review later historical context', ['history'],
                ['advice'], 'later-original-context')
    assert state.routes == before and state.valid()[2]['history'] == [active]
    assert state.pending == {'history'}


def test_real_counter_to_superseded_ancestor_invalidates_its_inheriting_replacement():
    reader, engine, state, calls, original = summary_fixture()
    calls.respond = lambda index, data: response(revised(original, state.versions['history']))
    read_new_history(reader, engine)
    active = state.valid()[2]['history'][0]
    add_explicit_piano_correction(engine)
    counter = observation('source-grounded-correction', ['s4'], stance='contradiction',
                          value=None, retract_ids=[original],
                          reason='The user explicitly corrects the earlier piano claim for the same year.')
    counter['value'] = None
    normalized = reader._observation(counter, 'history', visible(reader, ['s4']), engine.steps[0])
    state.integrate([normalized])
    assert state.routes[original]['revoked']
    assert state.routes[active]['value'] == EXPANDED
    assert 'history' not in state.valid()[0]
    assert original in state.routes[active]['summary_revision']['supersedes_route_ids']


def test_same_batch_counter_survives_rejection_of_revision_that_inherits_refuted_content():
    reader, engine, state, calls, original = summary_fixture()
    add_explicit_piano_correction(engine)
    version = state.versions['history']
    counter = observation('piano-correction', ['s4'], stance='contradiction',
                          retract_ids=[original],
                          reason='The user explicitly withdraws the earlier piano claim.')
    counter['value'] = None
    calls.respond = lambda index, data: response(
        revised(original, version, sources=('s1', 's2', 's4')), deepcopy(counter))
    reader.read(engine.steps[0], 'Review both new activities and the explicit correction',
                ['history'], ['painting', 'piano-correction'], 'counter-and-revision')
    assert state.routes[original]['revoked']
    assert not state.routes[original].get('superseded_by')
    assert 'history' not in state.valid()[0] and state.pending == {'history'}
    assert reader.pending_variables() == {'history'}
    reader.current_identity = 'later-required-originals'
    data = reader._payload(engine.steps[0], 'Review unresolved history', ['history'], [])
    assert {'s1', 's2', 's4'} <= {source['id'] for source in data['source_spans']}


def test_json_checkpoint_does_not_reactivate_old_binding_or_forget_real_counter_ancestry():
    reader, engine, state, calls, original = summary_fixture()
    calls.respond = lambda index, data: response(revised(original, state.versions['history']))
    read_new_history(reader, engine)
    snapshot = json.loads(json.dumps(state.public_dict()))
    restored = FactState(state.initial_program, state.contract, engine.steps, 'suggestion')
    restored.restore(snapshot)
    assert restored.valid()[0]['history'] == EXPANDED
    assert restored.routes[original]['superseded_by'] and not restored.routes[original]['revoked']
    original_source = engine.mapper.sources['s1']['source_id']
    restored.revoke_source(original_source, 'The original premise has been withdrawn by sourced correction.')
    assert 'history' not in restored.valid()[0]
    assert restored.routes[original]['superseded_by']


def test_failed_recovery_reports_whole_binding_cause_instead_of_claiming_draft_id_missing():
    reader, engine, state, calls, original = summary_fixture()
    # An extended support without explicit revision leaves an old whole live.
    calls.respond = lambda index, data: response(observation('expanded-summary', ['s1', 's2'], value=EXPANDED))
    read_new_history(reader, engine)
    assert reader.failed_obligations and state.pending == {'history'}
    engine.s['max_repairs_per_request'] = 1
    def review(index, data):
        result = response(observation('expanded-summary', ['s1', 's2'], value=EXPANDED))
        result['slots'][0]['resolved_failed_batch_ids'] = [
            r['id'] for r in data.get('failed_batch_reviews', []) if r.get('review_allowed')]
        result['slots'][0]['resolved_obligation_ids'] = [
            r['id'] for r in data.get('evidence_obligations', [])]
        return result
    calls.respond = review
    reader.read(engine.steps[0], 'Review the unchanged failed summary with new original context',
                ['history'], ['advice'], 'new-paid-review', review_existing_obligations=True)
    errors = [row['error'] for row in reader.batches[-1]['failed_slots']]
    assert any(error.startswith('whole_value_disagreement:') for error in errors)
    assert not any('requires every original failed draft ID' in error for error in errors)
    slot_errors = [row['error'] for row in reader.batches[-1]['failed_slots'] if row['id'] == '*']
    assert slot_errors and all(error.startswith('whole_value_disagreement:') for error in slot_errors)
    assert not state.routes[original].get('superseded_by')
    assert state.pending == {'history'}
