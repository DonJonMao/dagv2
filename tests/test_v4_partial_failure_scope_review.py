"""Independent repeat-admission review: partial outputs versus parent evidence.

Only scripted transport is used. No production or pre-existing test is edited.
"""
from copy import deepcopy
import json

import pytest

from dagbt.fact_state import FactState
from dagbt.joint_reading import JointReader
from test_personamem_joint_protocol import fixture, observation, response, visible
from test_v4_mandatory_contract_review import scoped_fixture, empty_slots


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Repeat-admission review must stay offline')
    monkeypatch.setattr('urllib.request.urlopen', forbidden)
    monkeypatch.setattr('urllib.request.build_opener', forbidden)


def two_output_compose_failure(*, companion_ready=True, max_repairs=0):
    _, engine, initial, calls = scoped_fixture()
    engine.s['max_repairs_per_request'] = max_repairs
    program = deepcopy(initial.initial_program)
    program['variables']['companion'] = {'type': 'text', 'demand': 'suggestion'}
    program['variables']['final'] = {'type': 'text', 'demand': 'final'}
    program['expression'] = {'op': 'var', 'id': 'final'}
    engine.steps.append({'output_slot': 'final', 'question': 'Use {suggestion}',
                         'inputs': ['suggestion'], 'execution': 'compose', 'answer_type': 'fact'})
    state = FactState(program, initial.contract, engine.steps, 'final')
    reader = JointReader(engine, state)
    state.integrate([reader._observation(observation('prior-parent'), 'history',
                                        visible(reader), engine.steps[0])])
    def first_reply(index, data):
        result = empty_slots(data)
        for slot in result['slots']:
            if slot['variable'] == 'suggestion':
                slot['observations'] = [observation('failed-output', value='A sourced summary',
                    entity_scope='wrong-scope', parent_versions={'history': state.versions['history']})]
            elif slot['variable'] == 'companion' and companion_ready and index == 1:
                slot['observations'] = [observation('accepted-output', value='An accepted companion fact',
                    parent_versions={'history': state.versions['history']})]
        return result
    calls.respond = first_reply
    reader.read(engine.steps[1], 'Combine history', ['suggestion', 'companion'],
                ['many-spans'], 'first-paid-attempt')
    assert len(calls.requests) == max_repairs + 1
    if companion_ready:
        assert state.valid()[0]['companion'] == 'An accepted companion fact'
    else:
        assert 'companion' not in state.valid()[0]
    assert 'suggestion' not in state.valid()[0]
    assert {item['variable'] for item in reader.failed_batch_reviews.values()} == {'suggestion'}
    assert state.pending == {'suggestion'}
    return reader, engine, state, calls


def replace_actual_parent(reader, engine, state):
    old_version = state.versions['history']
    route = state.valid()[2]['history'][0]
    counter = observation('actual-parent-counter', ['s2'], stance='contradiction', retract_ids=[route])
    counter['value'] = None
    replacement = observation('replacement-parent', ['s1', 's2'], value=['piano', 'painting'])
    sources = visible(reader, ['s1', 's2'])
    state.integrate([reader._observation(row, 'history', sources, engine.steps[0])
                     for row in [counter, replacement]], state.pending)
    assert state.versions['history'] != old_version
    assert state.valid()[0]['history'] == ['piano', 'painting']


def corrected_output(data, state, *, variable='suggestion', identifier='failed-output'):
    result = empty_slots(data)
    slot = next(slot for slot in result['slots'] if slot['variable'] == variable)
    slot['observations'] = [observation(identifier, value='A sourced summary',
                                       parent_versions={'history': state.versions['history']})]
    slot['resolved_failed_batch_ids'] = [record['id'] for record in data.get('failed_batch_reviews', [])
                                         if record['variable'] == variable and record['review_allowed']]
    return result


@pytest.mark.parametrize('restore', [False, True])
def test_settled_other_output_does_not_buy_a_repeat_of_the_failed_output(restore):
    reader, engine, state, calls = two_output_compose_failure()
    if restore:
        fresh = JointReader(engine, state)
        fresh.restore_obligations(json.loads(json.dumps(reader.obligation_snapshot())))
        reader = fresh
    before = deepcopy((engine.ledger.used, reader.failed_batch_reviews, state.public_dict()))
    before_requests = len(calls.requests)
    def no_repeat(index, data):
        assert any(item['variable'] == 'suggestion' and not item['review_allowed']
                   for item in data['failed_batch_reviews'])
        pytest.fail('A settled companion output must not authorize another attempt on unchanged failed output')
    calls.respond = no_repeat
    reader.read(engine.steps[1], 'Combine history', ['suggestion', 'companion'], [],
                'renamed-and-new-mode', source_scope='mandatory_closure')
    assert len(calls.requests) == before_requests
    assert (engine.ledger.used, reader.failed_batch_reviews, state.public_dict()) == before


def test_explicit_independent_unfailed_output_can_make_its_first_progress():
    reader, engine, state, calls = two_output_compose_failure(companion_ready=False)
    retained = deepcopy(reader.failed_batch_reviews)
    retained_markers = deepcopy(reader.failed_obligations)
    before_budget = deepcopy(engine.ledger.used)
    before_requests = len(calls.requests)
    def companion_only(index, data):
        assert set(data['requested_slots']) == {'companion', 'history'}
        assert not data.get('failed_batch_reviews')
        return corrected_output(data, state, variable='companion', identifier='new-independent-output')
    calls.respond = companion_only
    assert reader.read(engine.steps[1], 'Read the unfinished independent output', ['companion'], [],
                       'new-independent-output', source_scope='mandatory_closure')
    assert len(calls.requests) == before_requests + 1
    assert state.valid()[0]['companion'] == 'A sourced summary'
    assert 'suggestion' not in state.valid()[0]
    assert reader.failed_batch_reviews == retained and reader.failed_obligations == retained_markers
    assert state.pending == {'suggestion'}
    assert engine.ledger.used['llm'] == before_budget['llm'] + 1
    assert engine.ledger.used['json_repairs'] == before_budget['json_repairs']


def test_actual_parent_change_authorizes_paid_nonterminal_compose_recovery():
    reader, engine, state, calls = two_output_compose_failure(companion_ready=False, max_repairs=1)
    old_record = deepcopy(next(iter(reader.failed_batch_reviews.values())))
    replace_actual_parent(reader, engine, state)
    before_budget, before_requests = deepcopy(engine.ledger.used), len(calls.requests)
    def paid_recovery(index, data):
        offered = [record for record in data['failed_batch_reviews'] if record['review_allowed']]
        assert [record['id'] for record in offered] == [old_record['id']]
        originals = {engine.mapper.sources[source['id']]['source_id'] for source in data['source_spans']}
        assert originals <= set(old_record['seen_source_ids']), 'Recovery uses changed parents, not new text'
        assert old_record['source_ids'] == offered[0]['source_ids'] or {
            engine.mapper.source_aliases[source] for source in old_record['source_ids']
        } == set(offered[0]['source_ids'])
        return corrected_output(data, state)
    calls.respond = paid_recovery
    assert reader.read(engine.steps[1], 'Reconsider the changed actual parent', ['suggestion'], [],
                       'actual-parent-changed', source_scope='mandatory_closure')
    assert len(calls.requests) == before_requests + 1
    assert engine.ledger.used['llm'] == before_budget['llm'] + 1
    assert engine.ledger.used['json_repairs'] == before_budget['json_repairs'] + 1
    assert state.valid()[0]['suggestion'] == 'A sourced summary'
    assert not state.pending and not reader.failed_batch_reviews and not reader.failed_obligations


def test_one_eligible_failed_output_cannot_authorize_another_failed_output():
    reader, engine, state, calls = two_output_compose_failure(companion_ready=False, max_repairs=1)
    replace_actual_parent(reader, engine, state)
    def companion_failure(index, data):
        result = corrected_output(data, state, variable='companion', identifier='failed-companion')
        next(slot for slot in result['slots'] if slot['variable'] == 'companion')['observations'][0]['entity_scope'] = 'wrong'
        return result
    calls.respond = companion_failure
    reader.read(engine.steps[1], 'Read the other local output', ['companion'], ['many-spans'],
                'independent-output-fails')
    assert {record['variable'] for record in reader.failed_batch_reviews.values()} == {'suggestion', 'companion'}
    assert state.pending == {'suggestion', 'companion'}
    probe = reader._mandatory_scope_payload(engine.steps[1], 'Both outputs', ['suggestion', 'companion', 'history'],
                                            [], 'mixed-eligibility', semantic=False, audit=False, terminal_output=False)
    assert {record['variable']: record['review_allowed'] for record in probe['failed_batch_reviews']} == {
        'suggestion': True, 'companion': False}
    before_budget, before_requests = deepcopy(engine.ledger.used), len(calls.requests)
    before = deepcopy((reader.obligation_snapshot(), reader.current_identity, state.public_dict()))
    calls.respond = lambda index, data: pytest.fail('Recovery for suggestion cannot authorize companion')
    assert not reader.read(engine.steps[1], 'Both outputs', ['suggestion', 'companion'], [],
                           'mixed-eligibility', source_scope='mandatory_closure')
    assert len(calls.requests) == before_requests and engine.ledger.used == before_budget
    assert (reader.obligation_snapshot(), reader.current_identity, state.public_dict()) == before
    companion_record = deepcopy(next(record for record in reader.failed_batch_reviews.values()
                                    if record['variable'] == 'companion'))
    calls.respond = lambda index, data: corrected_output(data, state)
    assert reader.read(engine.steps[1], 'Only the authorized output', ['suggestion'], [],
                       'authorized-output-only', source_scope='mandatory_closure')
    assert len(calls.requests) == before_requests + 1
    assert engine.ledger.used['json_repairs'] == before_budget['json_repairs'] + 1
    assert reader.failed_batch_reviews == {companion_record['id']: companion_record}
    assert state.pending == {'companion'} and state.valid()[0]['suggestion'] == 'A sourced summary'


def test_multiple_failed_records_for_one_output_keep_the_unoffered_record_pending():
    reader, engine, state, calls = two_output_compose_failure(companion_ready=False, max_repairs=1)
    first = deepcopy(next(iter(reader.failed_batch_reviews.values())))
    replace_actual_parent(reader, engine, state)
    second = deepcopy(first)
    second.update(id='separate-retained-failure', query_identity='separate-old-query',
                  seen_parent_signatures=[reader._failed_parent_signature('suggestion')])
    reader.failed_batch_reviews[second['id']] = second
    marker = (second['query_identity'], second['batch_identity'])
    reader.failed_obligations[marker] = {'suggestion'}
    before_budget, before_requests = deepcopy(engine.ledger.used), len(calls.requests)
    def one_paid_record(index, data):
        assert {record['id']: record['review_allowed'] for record in data['failed_batch_reviews']} == {
            first['id']: True, second['id']: False}
        return corrected_output(data, state)
    calls.respond = one_paid_record
    reader.read(engine.steps[1], 'One offered history recovery', ['suggestion'], [],
                'one-offered-record', source_scope='mandatory_closure')
    assert len(calls.requests) == before_requests + 1
    assert engine.ledger.used['json_repairs'] == before_budget['json_repairs'] + 1
    assert reader.failed_batch_reviews == {second['id']: second}
    assert reader.failed_obligations == {marker: {'suggestion'}} and state.pending == {'suggestion'}
    assert reader.semantic_prediction is None


def test_related_parent_failure_is_retained_while_a_new_task_reads_real_counterevidence():
    reader, engine, state, calls = fixture(max_repairs_per_request=0)
    state.integrate([reader._observation(observation('prior-parent'), 'history',
                                        visible(reader), engine.steps[0])])
    route = next(iter(state.routes))
    calls.respond = lambda index, data: response(observation('failed-parent-draft', ['s2'], entity_scope='wrong'))
    reader.read(engine.steps[0], 'Review history', ['history'], ['painting'], 'old-parent-read')
    assert reader.failed_batch_reviews and state.valid()[0]['history'] == ['piano']
    retained = deepcopy(reader.failed_batch_reviews)
    retained_markers = deepcopy(reader.failed_obligations)
    before_repairs = engine.ledger.used['json_repairs']
    def downstream(index, data):
        assert index == 2
        assert {'s1', 's2'} <= {source['id'] for source in data['source_spans']}
        assert data['failed_batch_reviews'] and not any(item['review_allowed'] for item in data['failed_batch_reviews'])
        result = empty_slots(data)
        counter = observation('new-task-counter', ['s2'], stance='contradiction', retract_ids=[route])
        counter['value'] = None
        next(slot for slot in result['slots'] if slot['variable'] == 'history')['observations'] = [counter]
        return result
    calls.respond = downstream
    reader.read(engine.steps[1], 'Review the actual grounded parent', ['suggestion'], [],
                'first-real-child-task', source_scope='mandatory_closure')
    assert len(calls.requests) == 2
    assert state.routes[route]['revoked'] and 'history' not in state.valid()[0]
    assert reader.failed_batch_reviews == retained and reader.failed_obligations == retained_markers
    assert 'history' in state.pending and engine.ledger.used['json_repairs'] == before_repairs
    assert reader.semantic_prediction is None


def test_related_failed_parent_cannot_be_resolved_by_an_unpaid_child_request():
    reader, engine, state, calls = fixture(max_repairs_per_request=0)
    state.integrate([reader._observation(observation('prior-parent'), 'history',
                                        visible(reader), engine.steps[0])])
    calls.respond = lambda index, data: response(observation('failed-parent-draft', ['s2'], entity_scope='wrong'))
    reader.read(engine.steps[0], 'Review history', ['history'], ['painting'], 'old-parent-read')
    retained = deepcopy(reader.failed_batch_reviews)
    def illegal_recovery(index, data):
        assert not any(item['review_allowed'] for item in data['failed_batch_reviews'])
        result = empty_slots(data)
        parent = next(slot for slot in result['slots'] if slot['variable'] == 'history')
        parent['observations'] = [observation('failed-parent-draft', ['s2'])]
        parent['resolved_failed_batch_ids'] = list(retained)
        return result
    calls.respond = illegal_recovery
    reader.read(engine.steps[1], 'Read without recovery authority', ['suggestion'], [],
                'not-an-authorized-parent-repair', source_scope='mandatory_closure')
    assert len(calls.requests) == 2
    assert reader.failed_batch_reviews.keys() >= retained.keys()
    assert all(reader.failed_batch_reviews[key] == value for key, value in retained.items())
    assert 'history' in state.pending
    assert engine.ledger.used['json_repairs'] == 0
    assert not reader.batches[-1]['complete']
