"""Independent exact-summary recheck regressions through the real read path.

Only model replies are scripted. An unchanged review must preserve provenance
and version identity while retaining the same validation and paid-review gates.
"""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from dagbt.evidence_spans import build_source_spans
from dagbt.fact_state import FactState
from dagbt.joint_reading import JointReader

from test_personamem_joint_protocol import observation, response, visible
from test_v4_summary_revision_review import (
    OLD, EXPANDED, summary_fixture, revised, read_new_history, add_explicit_piano_correction,
)


def revised_fixture():
    reader, engine, state, calls, original = summary_fixture()
    calls.respond = lambda index, data: response(revised(original, state.versions['history']))
    read_new_history(reader, engine)
    active = state.valid()[2]['history'][0]
    assert active != original and state.summary_ancestors(state.routes[active]) == {original}
    return reader, engine, state, calls, original, active


def unchanged(active, version, **changes):
    draft = revised(active, version, **changes)
    draft['summary_revision']['reason'] = 'Rechecked the same original sources; no new evidence or content.'
    return draft


def recheck(reader, engine, **kwargs):
    return reader.read(engine.steps[0], 'Recheck the current sourced history', ['history'],
                       ['painting'], 'independent-unchanged-review', **kwargs)


def test_exact_recheck_preserves_revised_ancestry_and_dependent_binding_then_counter_invalidates_both():
    reader, engine, state, calls, original, active = revised_fixture()
    engine.output_options = ['A sourced suggestion.']
    terminal = observation('dependent-response', ['s1', 's2'], value=engine.output_options[0],
                           time_scope='response', parent_versions={'history': state.versions['history']})
    state.integrate([reader._observation(terminal, 'suggestion', visible(reader, ['s1', 's2']),
                                        engine.steps[1])])
    before = deepcopy((state.routes, state.versions, state.valid()))
    calls.respond = lambda index, data: response(unchanged(active, state.versions['history']))
    recheck(reader, engine)
    assert (state.routes, state.versions, state.valid()) == before
    assert state.routes[active]['summary_revision']['supersedes_route_ids'] == [original]
    assert not state.pending and not reader.failed_obligations
    assert len(calls.requests) == engine.ledger.used['llm'] == 2
    assert not engine.ledger.used.get('json_repairs', 0)

    add_explicit_piano_correction(engine)
    counter = observation('ancestor-correction', ['s4'], stance='contradiction', retract_ids=[original])
    counter['value'] = None
    state.integrate([reader._observation(counter, 'history', visible(reader, ['s4']), engine.steps[0])])
    assert state.routes[original]['revoked']
    assert not state.is_active_route(state.routes[active])
    assert 'history' not in state.valid()[0] and 'suggestion' not in state.valid()[0]


def test_same_batch_true_counter_is_integrated_when_exact_recheck_is_rejected():
    reader, engine, state, calls, original, active = revised_fixture()
    add_explicit_piano_correction(engine)
    before_ids = set(state.routes)
    counter = observation('ancestor-correction', ['s4'], stance='contradiction', retract_ids=[original])
    counter['value'] = None
    calls.respond = lambda index, data: response(
        unchanged(active, state.versions['history']), deepcopy(counter))
    reader.read(engine.steps[0], 'Review the exact summary and a sourced correction',
                ['history'], ['painting', 'piano-correction'], 'counter-with-noop')
    assert set(state.routes) == before_ids
    assert state.routes[original]['revoked']
    assert not state.is_active_route(state.routes[active])
    assert 'history' not in state.valid()[0] and state.pending == {'history'}
    assert reader.failed_obligations
    assert any(row['error'].startswith('summary_revision_conflict:')
               for row in reader.batches[-1]['failed_slots'])


def test_repeated_exact_recheck_with_permuted_citations_cannot_create_material_progress():
    reader, engine, state, calls, original, active = revised_fixture()
    draft = unchanged(active, state.versions['history'], sources=['s2', 's1'])
    normalized = reader._observation(draft, 'history', visible(reader, ['s1', 's2']),
                                     engine.steps[0], allow_summary_revision=True)
    before = deepcopy((state.routes, state.versions, reader.material_signature()))
    old_history_size = len(state.history)
    state.integrate([normalized])
    state.integrate([normalized])
    assert len(state.history) == old_history_size + 2
    assert (state.routes, state.versions, reader.material_signature()) == before


@pytest.mark.parametrize('change', ['new_proof', 'new_text'])
def test_noop_recognition_does_not_suppress_real_proof_or_exact_text_changes(change):
    reader, engine, state, calls, original, active = revised_fixture()
    version = state.versions['history']
    changes = ({'sources': ['s1', 's2', 's3']} if change == 'new_proof'
               else {'value': EXPANDED + ' These were past creative activities.'})
    calls.respond = lambda index, data: response(unchanged(active, version, **changes))
    reader.read(engine.steps[0], 'Review the updated whole account', ['history'],
                ['advice'], 'real-change-after-revision')
    latest = state.valid()[2]['history'][0]
    assert latest != active and state.versions['history'] > version
    assert state.routes[active]['superseded_by'] == [latest]
    assert state.summary_ancestors(state.routes[latest]) == {original, active}
    assert not state.pending


@pytest.mark.parametrize('mistake', ['stale_version', 'retired_target', 'missing_original', 'wrong_scope'])
def test_exact_value_does_not_bypass_revision_validation(mistake):
    reader, engine, state, calls, original, active = revised_fixture()
    draft = unchanged(active, state.versions['history'])
    if mistake == 'stale_version':
        draft['summary_revision']['previous_version'] -= 1
    elif mistake == 'retired_target':
        draft['summary_revision']['supersedes_route_ids'] = [original]
    elif mistake == 'missing_original':
        draft['source_ids'] = ['s2']
    else:
        draft['time_scope'] = 'current'
    before = deepcopy((state.routes, state.versions))
    calls.respond = lambda index, data: response(deepcopy(draft))
    recheck(reader, engine)
    assert (state.routes, state.versions) == before
    assert state.valid()[0]['history'] == EXPANDED
    assert state.pending == {'history'} and reader.failed_obligations


@pytest.mark.parametrize('acknowledge', [False, True])
def test_only_explicit_paid_durable_recheck_can_discharge_failure_without_fact_version_progress(monkeypatch, acknowledge):
    reader, engine, state, calls, original, active = revised_fixture()
    def blocked_review(index, data):
        result = response(unchanged(active, state.versions['history']))
        result['slots'][0]['unresolved'] = ['Review whether this sourced account remains sufficient.']
        return result
    calls.respond = blocked_review
    before = deepcopy((state.routes, state.versions))
    real_handoff = reader._handoff_failures

    def fail_handoff(rows, *args, **kwargs):
        return {(row['variable'], row['id']): 'citation_capacity: previous mandatory handoff did not fit'
                for row in rows if row['stance'] == 'support'}

    monkeypatch.setattr(reader, '_handoff_failures', fail_handoff)
    recheck(reader, engine)
    assert reader.failed_batch_reviews and reader.evidence_obligations
    assert (state.routes, state.versions) == before
    monkeypatch.setattr(reader, '_handoff_failures', real_handoff)
    engine.s['max_repairs_per_request'] = 1

    def review(index, data):
        result = response(unchanged(active, state.versions['history']))
        if acknowledge:
            result['slots'][0]['resolved_failed_batch_ids'] = [
                row['id'] for row in data.get('failed_batch_reviews', []) if row['review_allowed']]
            result['slots'][0]['resolved_obligation_ids'] = [
                row['id'] for row in data.get('evidence_obligations', [])]
        return result

    calls.respond = review
    reader.read(engine.steps[0], 'Review the original failed draft and all its original evidence',
                ['history'], ['advice'], 'authorized-durable-noop', review_existing_obligations=True)
    assert (state.routes, state.versions) == before
    assert state.summary_ancestors(state.routes[active]) == {original}
    assert engine.ledger.used['json_repairs'] == 1
    assert engine.ledger.used['llm'] == 3
    if acknowledge:
        assert not state.pending and not reader.failed_obligations
        assert not reader.failed_batch_reviews and not reader.evidence_obligations
    else:
        assert state.pending == {'history'} and reader.failed_obligations
        assert reader.failed_batch_reviews and reader.evidence_obligations


def test_exact_recheck_cannot_resolve_an_unrelated_whole_value_conflict():
    reader, engine, state, calls, original, active = revised_fixture()
    conflicting = observation('other-history', ['s2'], value='The user has never enjoyed painting.')
    state.integrate([reader._observation(conflicting, 'history', visible(reader, ['s2']), engine.steps[0])])
    assert 'history' not in state.valid()[0]
    before = deepcopy((state.routes, state.versions))
    calls.respond = lambda index, data: response(unchanged(active, state.versions['history']))
    recheck(reader, engine)
    assert (state.routes, state.versions) == before
    assert state.pending == {'history'} and reader.failed_obligations
    assert any(row['error'].startswith('whole_value_disagreement:')
               for row in reader.batches[-1]['failed_slots'])


@pytest.mark.parametrize('counter_first', [False, True])
def test_ordinary_exact_duplicate_preserves_ancestry_and_cannot_revive_after_ancestor_counter(counter_first):
    reader, engine, state, calls, original, active = revised_fixture()
    add_explicit_piano_correction(engine)
    counter = observation('ancestor-correction', ['s4'], stance='contradiction', retract_ids=[original])
    counter['value'] = None
    normalized = reader._observation(counter, 'history', visible(reader, ['s4']), engine.steps[0])
    if counter_first:
        state.integrate([normalized])
    before = deepcopy((state.routes, state.versions))
    calls.respond = lambda index, data: response(
        observation('plain-replay', ['s2', 's1'], value=EXPANDED))
    reader.read(engine.steps[0], 'Recheck unchanged original evidence', ['history'],
                ['painting'], 'ordinary-exact-replay')
    assert (state.routes, state.versions) == before
    if not counter_first:
        assert state.valid()[2]['history'] == [active]
        state.integrate([normalized])
    assert state.summary_ancestors(state.routes[active]) == {original}
    assert not state.is_active_route(state.routes[active])
    assert 'history' not in state.valid()[0]


def test_plain_replay_and_ancestor_counter_in_same_batch_cannot_create_independent_route():
    reader, engine, state, calls, original, active = revised_fixture()
    add_explicit_piano_correction(engine)
    before_ids = set(state.routes)
    counter = observation('ancestor-correction', ['s4'], stance='contradiction', retract_ids=[original])
    counter['value'] = None
    calls.respond = lambda index, data: response(
        observation('plain-replay', ['s1', 's2'], value=EXPANDED), deepcopy(counter))
    reader.read(engine.steps[0], 'Review the old account and its sourced correction', ['history'],
                ['painting', 'piano-correction'], 'plain-replay-with-counter')
    assert set(state.routes) == before_ids and state.routes[original]['revoked']
    assert not state.is_active_route(state.routes[active])
    assert 'history' not in state.valid()[0]


@pytest.mark.parametrize('revision_first', [False, True])
def test_plain_clone_in_same_batch_as_new_revision_cannot_lose_ancestry_by_row_order(revision_first):
    reader, engine, state, calls, original = summary_fixture()
    revision = revised(original, state.versions['history'])
    plain = observation('plain-clone', ['s2', 's1'], value=EXPANDED)
    rows = [revision, plain] if revision_first else [plain, revision]
    calls.respond = lambda index, data: response(*deepcopy(rows))
    read_new_history(reader, engine)
    assert len(state.routes) == 2
    active = state.valid()[2]['history'][0]
    assert state.summary_ancestors(state.routes[active]) == {original}
    add_explicit_piano_correction(engine)
    counter = observation('ancestor-correction', ['s4'], stance='contradiction', retract_ids=[original])
    counter['value'] = None
    state.integrate([reader._observation(counter, 'history', visible(reader, ['s4']), engine.steps[0])])
    assert 'history' not in state.valid()[0]


def test_genuinely_independent_original_source_remains_a_separate_or_proof():
    reader, engine, state, calls, original, active = revised_fixture()
    text = 'I confirm that I played piano last year and enjoyed painting last spring.'
    document = SimpleNamespace(doc_id='independent-confirmation', passage=text, metadata={
        'source_segments': [{'role': 'user', 'start': 0, 'end': len(text),
                             'source_message_indices': [4]}]})
    engine.docs[document.doc_id] = document
    source = build_source_spans(document.doc_id, document, 400)[0]
    engine.mapper.sources['s5'] = {**source, 'id': 's5', 'source_id': source['id']}
    engine.mapper.source_aliases[source['id']] = 's5'
    calls.respond = lambda index, data: response(observation('independent-proof', ['s5'], value=EXPANDED))
    reader.read(engine.steps[0], 'Review independent confirmation', ['history'],
                ['independent-confirmation'], 'independent-original-proof')
    independent = [row for row in state.routes.values() if row['source_ids'] == [source['id']]]
    assert len(independent) == 1 and not independent[0].get('summary_revision')
    add_explicit_piano_correction(engine)
    counter = observation('ancestor-correction', ['s4'], stance='contradiction', retract_ids=[original])
    counter['value'] = None
    state.integrate([reader._observation(counter, 'history', visible(reader, ['s4']), engine.steps[0])])
    assert not state.is_active_route(state.routes[active])
    assert state.valid()[0]['history'] == EXPANDED
    assert state.valid()[2]['history'] == [independent[0]['id']]


def test_new_actual_parent_version_is_a_different_proof_not_a_replay_of_stale_revision():
    _, engine, old_state, _, _ = summary_fixture()
    program = deepcopy(old_state.initial_program)
    program['variables']['context'] = {
        'type': 'text', 'demand': 'context',
        'scope': {'entity_scope': 'user', 'time_scope': 'history'},
    }
    engine.steps = [{
        'output_slot': 'context', 'question': 'Read the activity context',
        'inputs': [], 'execution': 'retrieval', 'answer_type': 'fact',
    }, *deepcopy(engine.steps)]
    engine.steps[1]['inputs'] = ['context']
    state = FactState(program, old_state.contract, engine.steps, 'suggestion')
    reader = JointReader(engine, state)

    def normalize(row, variable, index):
        return reader._observation(row, variable, visible(reader, ['s1', 's2', 's3']),
                                   engine.steps[index], allow_summary_revision=True)

    state.integrate([normalize(observation('old-context', ['s3'], value='Earlier context.'), 'context', 0)])
    context_route = state.valid()[2]['context'][0]
    parent_version = state.versions['context']
    state.integrate([normalize(observation('old-summary', ['s1'], value=OLD,
        parent_versions={'context': parent_version}), 'history', 1)])
    original = state.valid()[2]['history'][0]
    state.integrate([normalize(revised(original, state.versions['history'], sources=['s1', 's2', 's3'],
        parent_versions={'context': parent_version}), 'history', 1)])
    active = state.valid()[2]['history'][0]
    old_proof = deepcopy(state.routes[active])

    # A sourced change to the actual context invalidates the old dependency.
    counter = observation('context-correction', ['s3'], stance='contradiction', retract_ids=[context_route])
    counter['value'] = None
    new_context = observation('new-context', ['s3'], value='Revised context.')
    state.integrate([normalize(counter, 'context', 0), normalize(new_context, 'context', 0)])
    assert state.versions['context'] != parent_version and 'history' not in state.valid()[0]
    new_proof = observation('new-parent-proof', ['s1', 's2', 's3'], value=EXPANDED,
                            parent_versions={'context': state.versions['context']})
    state.integrate([normalize(new_proof, 'history', 1)])
    current = state.valid()[2]['history'][0]
    assert current != active and state.routes[active] == old_proof
    assert state.routes[current]['parent_versions']['context'] == state.versions['context']
    assert state.valid()[0]['history'] == EXPANDED
