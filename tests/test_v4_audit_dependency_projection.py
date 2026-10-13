"""Actual audit and handoff must use the certified dependency frontier.

The reader, fact state, certificates and request serializer are real. The
transport is scripted; these checks establish scope, not semantic accuracy.
"""
from copy import deepcopy

import pytest

from dagbt.fact_state import FactState
from dagbt.methods import capabilities
from dagbt.residual import possible_outputs
from dagbt.residual_control import ResidualControl
from test_personamem_joint_protocol import fixture, observation


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Audit dependency tests must stay offline')
    monkeypatch.setattr('urllib.request.urlopen', forbidden)
    monkeypatch.setattr('urllib.request.build_opener', forbidden)


def setup_case(*, ancestor=False, same_task=False):
    _, engine, _, calls = fixture(max_repairs_per_request=0)
    def task(name, parents=(), execution='retrieval'):
        return {'output_slot': name, 'question': 'Check ' + name,
                'inputs': list(parents), 'execution': execution, 'answer_type': 'fact'}
    steps = ([task('origin')] if ancestor else []) + [
        task('necessary', ['origin'] if ancestor else [])]
    if not same_task:
        steps.append(task('later'))
    steps.append(task('decision', ['necessary'] if same_task else ['necessary', 'later'], 'compose'))
    declarations = {'a': {'type': 'bool', 'demand': 'necessary'},
                    'b': {'type': 'bool', 'demand': 'necessary' if same_task else 'later'}}
    if ancestor:
        declarations = {'p': {'type': 'bool', 'demand': 'origin'}, **declarations}
    program = {'variables': declarations, 'rules_complete': True,
               'expression': {'op': 'and', 'args': [
                   {'op': 'var', 'id': 'a'}, {'op': 'var', 'id': 'b'}]}}
    contract = {'mode': 'value', 'result_type': 'bool', 'entity_scope': 'user',
                'time_scope': 'history', 'explanation': 'none'}
    state = FactState(program, contract, steps, 'decision')
    engine.steps, engine.fact_state, engine.final_node_id = steps, state, 'decision'
    engine.s['max_enumeration_states'] = 10000
    engine.capabilities = capabilities(engine.s['algorithm_version'])
    controller = ResidualControl(engine)
    engine.residual_control = controller
    reader = controller.reader
    def support(variable, alias, value, parents=None):
        step = next(s for s in steps if s['output_slot'] == declarations[variable]['demand'])
        return reader._observation(
            observation(variable + '-proof', [alias], value=value, parent_versions=parents or {}),
            variable, {alias: engine.mapper.sources[alias]}, step)
    if ancestor:
        state.integrate([support('p', 's3', True)])
    state.integrate([support('b', 's2', True,
                             {'p': state.versions['p']} if ancestor and same_task else {})])
    reader.evidence_obligations['b-unresolved'] = {
        'id': 'b-unresolved', 'variable': 'b', **state.scope('b'),
        'source_ids': [engine.mapper.sources['s2']['source_id']],
        'descriptions': ['A separate issue still needs review.'], 'origins': ['later-read']}
    state.integrate([], reader.pending_variables())
    decisive = support('a', 's1', False, {'p': state.versions['p']} if ancestor else {})
    calls.respond = lambda index, data: {
        'slots': [{'variable': v, 'observations': [], 'unresolved': []}
                  for v in data['requested_slots']], 'audit_verdict': 'supported'}
    return controller, reader, engine, state, calls, decisive


@pytest.mark.parametrize('ancestor', [False, True])
@pytest.mark.parametrize('same_task', [False, True])
def test_actual_audit_uses_selected_dependencies_and_preserves_eliminated_obligation(ancestor, same_task):
    controller, reader, engine, state, calls, decisive = setup_case(ancestor=ancestor, same_task=same_task)
    state.integrate([decisive], reader.pending_variables())
    certificate = controller.candidate()
    expected_variables = {'a', 'p'} if ancestor else {'a'}
    assert certificate['output'] is False
    assert set(certificate['binding_dependencies']) == expected_variables
    old_routes = deepcopy(state.routes)
    old_issues = deepcopy(reader.evidence_obligations)
    assert controller.audit(certificate)
    assert len(calls.requests) == 1
    data = calls.requests[0]['data']
    assert set(data['requested_slots']) == expected_variables
    assert set(data['supported_parents']) == expected_variables
    assert set(data['task']['inputs']) == ({'necessary', 'origin'} if ancestor else {'necessary'})
    assert {s['id'] for s in data['source_spans']} == ({'s1', 's3'} if ancestor else {'s1'})
    assert not data.get('evidence_obligations')
    assert reader.evidence_obligations == old_issues
    assert state.routes == old_routes and 'b' in state.pending
    assert controller.candidate() == certificate


@pytest.mark.parametrize('ancestor', [False, True])
@pytest.mark.parametrize('same_task', [False, True])
def test_prospective_handoff_and_actual_audit_have_the_same_dependency_scope(ancestor, same_task):
    controller, reader, engine, state, calls, decisive = setup_case(ancestor=ancestor, same_task=same_task)
    checked = []
    original_fits = reader._fits
    def checked_fits(operation, data):
        checked.append((operation, deepcopy(data)))
        return original_fits(operation, data)
    reader._fits = checked_fits
    before = deepcopy((state.public_dict(), reader.obligation_snapshot()))
    assert reader._handoff_failures([decisive], {('a', decisive['id'])}) == {}
    assert (state.public_dict(), reader.obligation_snapshot()) == before
    prospective = [data for operation, data in checked if operation == 'audit_joint']
    assert len(prospective) == 1
    state.integrate([decisive], reader.pending_variables())
    assert controller.audit(controller.candidate())
    actual = calls.requests[-1]['data']
    for field in ('task', 'requested_slots', 'supported_parents', 'source_spans'):
        assert prospective[0][field] == actual[field]
    assert 'b' not in actual['supported_parents']
    assert 'b' in state.pending and 'b-unresolved' in reader.evidence_obligations


@pytest.mark.parametrize('same_task', [False, True])
def test_eliminated_unresolved_branch_reactivates_when_decisive_route_is_revoked(same_task):
    controller, reader, engine, state, calls, decisive = setup_case(same_task=same_task)
    state.integrate([decisive], reader.pending_variables())
    certificate = controller.candidate()
    assert controller.audit(certificate)
    route_id = state.valid()[2]['a'][0]
    counter = observation('later-counter', ['s3'], stance='contradiction', retract_ids=[route_id])
    counter['value'] = None
    counter = reader._observation(counter, 'a', {'s3': engine.mapper.sources['s3']}, engine.steps[0])
    state.integrate([counter], reader.pending_variables())
    assert controller.candidate() is None
    outcome = possible_outputs(state.program, state.valid()[0], max_states=10000)
    active = set(controller.active(outcome))
    assert ({'necessary'} if same_task else {'necessary', 'later'}) <= active
    assert state.routes[route_id]['revoked']
    assert 'b' in state.pending and 'b-unresolved' in reader.evidence_obligations
