"""Independent Persona evidence declarations at actual planning/runtime boundaries.

No dataset or labels are loaded. Only model responses are scripted; admission,
repair accounting, source normalization and fact-state validation are real.
"""
from copy import deepcopy
import json

import numpy as np
import pytest

from dagbt.engine import Engine, legacy_modules
from dagbt.fact_state import FactState
from dagbt.joint_reading import JointReader
from dagbt.methods import RESIDUAL_V2
from dagbt.personal_task import validate_personal_plan
from dagbt.reasoning import ProtocolError
from dagbt.residual import ProgramError
from dagbt.residual_plan import planner_contract, validate_plan
from test_personamem_joint_protocol import fixture, observation, response, visible
from test_residual_runtime import demo


SUMMARY = 'personal_history_summary'
ATOMIC = 'personal_atomic_fact'
QUESTION = 'I am thinking about a creative weekend.'
OPTIONS = ['Explore your painting interest.', 'Try piano.', 'Buy a book.', 'Visit a cafe.']
OLD = 'The user played piano last year.'
EXPANDED = OLD + ' The user enjoyed painting last spring.'


def plan(marker=SUMMARY):
    history = {'type': 'text', 'demand': 'history',
               'scope': {'entity_scope': 'user', 'time_scope': 'recorded history'}}
    if marker is not None:
        history['evidence_semantics'] = marker
    return {
        'steps': [demo.step('history', 'Read relevant personal experiences'),
                  demo.step('response', QUESTION, ['history'], 'compose')],
        'final_node_id': 'response',
        'output_contract': {'mode': 'semantic', 'result_type': 'text', 'explanation': 'semantic',
                            'entity_scope': 'user', 'time_scope': 'reply now'},
        'program': {'variables': {'history': history, 'response': {'type': 'text', 'demand': 'response'}},
                    'expression': {'op': 'var', 'id': 'response'}},
    }


def validate(value, *, personal=True):
    _, repair, _ = legacy_modules()
    result = validate_plan(value, repair.validate_plan, personal_history=personal)
    return validate_personal_plan(result, QUESTION) if personal else result


class PlannerResponses:
    def __init__(self, *plans):
        self.plans = deepcopy(plans)
        self.requests = []

    def get(self, stage, url, payload):
        assert stage[0] == 'planner', 'Admission must finish before any retrieval/reading'
        assert url.endswith('/chat/completions')
        self.requests.append({'stage': list(stage), 'payload': deepcopy(payload)})
        assert len(self.requests) <= len(self.plans), 'Unexpected unbounded planner retry'
        return {'response_ref': 'independent-plan-' + str(len(self.requests)), 'response': {
            'choices': [{'finish_reason': 'stop', 'message': {
                'content': json.dumps(self.plans[len(self.requests) - 1])}}]}}


def engine_for(*plans, repairs=0, personal=True):
    calls = PlannerResponses(*plans)
    engine = Engine({'id': 'semantics-admission', 'question': QUESTION},
                    ({}, [], np.empty((0, 3), np.float32), None, demo.Tokenizer()), calls,
                    {'_test_transport': True, 'llm_base_url': 'http://synthetic.invalid/v1',
                     'llm_model': 'script', 'fusion': {'max_repairs_per_request': repairs}},
                    RESIDUAL_V2, reader_question=QUESTION, output_options=OPTIONS,
                    task_kind='personal_response_selection' if personal else None)
    return engine, calls


def test_actual_planner_rejects_missing_declaration_without_silent_default():
    unmarked = plan(None)
    saved = deepcopy(unmarked)
    engine, calls = engine_for(unmarked)
    with pytest.raises(ProtocolError, match='evidence_semantics'):
        engine.plan()
    assert unmarked == saved and calls.plans[0] == saved
    assert len(calls.requests) == 1 and engine.steps == []
    assert not any(event['event'] == 'plan_frozen' for event in engine.events)
    assert engine.ledger.used.get('ann', 0) == 0


@pytest.mark.parametrize('marker', [SUMMARY, ATOMIC])
def test_missing_semantics_is_repairable_through_real_planner_feedback(marker):
    unmarked, repaired = plan(None), plan(marker)
    engine, calls = engine_for(unmarked, repaired, repairs=1)
    engine.plan()
    assert len(calls.requests) == 2
    assert [request['validation_status'] for request in engine.reasoner.requests] == ['invalid', 'valid']
    feedback = json.loads(calls.requests[1]['payload']['messages'][1]['content'])
    assert feedback['original_input'] == QUESTION and 'evidence_semantics' in feedback['validation_feedback']
    assert OPTIONS[0] not in str(calls.requests)
    assert engine.fact_state.program['variables']['history']['evidence_semantics'] == marker
    assert 'evidence_semantics' not in unmarked['program']['variables']['history']
    assert engine.ledger.used['json_repairs'] == 1 and engine.ledger.used['llm'] == 2
    assert engine.ledger.used.get('ann', 0) == 0


def test_declaring_only_later_aggregate_does_not_authorize_three_unmarked_inputs():
    value = plan(SUMMARY)
    names = ['evidence_a', 'evidence_b', 'evidence_c']
    value['steps'][:0] = [demo.step(name, 'Read a relevant historical fact') for name in names]
    value['steps'][-2].update(execution='compose', inputs=names)
    for name in names:
        value['program']['variables'][name] = {
            'type': 'text', 'demand': name,
            'scope': {'entity_scope': 'user', 'time_scope': 'history'}}
    engine, _ = engine_for(value)
    with pytest.raises(ProtocolError, match='evidence_semantics'):
        engine.plan()
    for name in names:
        value['program']['variables'][name]['evidence_semantics'] = SUMMARY
    repaired, calls = engine_for(value)
    repaired.plan()
    assert len(calls.requests) == 1 and len(repaired.steps) == 5


@pytest.mark.parametrize('marker', [SUMMARY, ATOMIC])
def test_explicit_summary_and_atomic_plans_are_accepted_without_mutation(marker):
    value = plan(marker)
    saved = deepcopy(value)
    assert validate(value) == saved and value == saved
    engine, calls = engine_for(value)
    engine.plan()
    assert len(calls.requests) == 1
    assert engine.fact_state.program['variables']['history']['evidence_semantics'] == marker
    assert 'evidence_semantics' not in engine.fact_state.program['variables']['response']


@pytest.mark.parametrize('marker', [SUMMARY, ATOMIC])
def test_generic_engine_rejects_both_personal_markers(marker):
    engine, _ = engine_for(plan(marker), personal=False)
    with pytest.raises(ProtocolError, match='explicit PersonaMem'):
        engine.plan()
    assert engine.steps == []


def test_generic_unmarked_text_remains_legal():
    engine, calls = engine_for(plan(None), personal=False)
    engine.plan()
    assert len(calls.requests) == 1 and len(engine.steps) == 2


@pytest.mark.parametrize('marker', [SUMMARY, ATOMIC])
@pytest.mark.parametrize('boundary', ['terminal', 'no_scope', 'finite_domain', 'number', 'bool', 'rule', 'set', 'record', 'entity'])
def test_markers_cannot_escape_nonterminal_unrestricted_scoped_text(marker, boundary):
    value = plan(marker)
    history = value['program']['variables']['history']
    if boundary == 'terminal':
        value['program']['variables']['response'].update(
            evidence_semantics=marker, scope={'entity_scope': 'user', 'time_scope': 'reply now'})
    elif boundary == 'no_scope':
        del history['scope']
    elif boundary == 'finite_domain':
        history['domain'] = ['piano', 'painting']
    else:
        history['type'] = boundary
    with pytest.raises(ProtocolError):
        validate(value)


@pytest.mark.parametrize('kind', ['finite_domain', 'number', 'bool', 'rule', 'set', 'record', 'entity'])
def test_other_evidence_types_do_not_require_a_personal_text_marker(kind):
    value = plan(None)
    history = value['program']['variables']['history']
    if kind == 'finite_domain':
        history['domain'] = ['piano', 'painting']
    else:
        history['type'] = kind
    assert validate(value) == value


def deterministic_plan(marker):
    value = plan(marker)
    value['steps'].insert(0, demo.step('source', 'Read an exact sourced fact'))
    value['steps'][1].update(inputs=['source'], execution='compose')
    value['program']['variables']['source'] = {
        'type': 'text', 'demand': 'source', 'evidence_semantics': ATOMIC,
        'scope': {'entity_scope': 'user', 'time_scope': 'history'}}
    value['program']['task_expressions'] = {'history': {'op': 'var', 'id': 'source'}}
    return value


def test_deterministic_text_result_needs_no_marker():
    value = deterministic_plan(None)
    assert validate(value) == value


@pytest.mark.parametrize('marker', [SUMMARY, ATOMIC])
def test_deterministic_binding_rejects_either_personal_marker(marker):
    with pytest.raises(ProtocolError):
        validate(deterministic_plan(marker))


@pytest.mark.parametrize('marker', [SUMMARY, ATOMIC])
def test_explicit_history_scope_may_share_current_time_with_terminal(marker):
    value = plan(marker)
    value['output_contract']['time_scope'] = 'current'
    value['program']['variables']['history']['scope']['time_scope'] = 'current'
    assert validate(value) == value


def test_schema_offers_two_markers_without_forcing_them_on_terminal_or_other_types():
    legacy, _, _ = legacy_modules()
    schema = planner_contract(legacy.PLAN_SCHEMA, personal_history=True)
    declaration = schema['properties']['program']['properties']['variables']['additionalProperties']
    assert set(declaration['properties']['evidence_semantics']['enum']) == {SUMMARY, ATOMIC}
    assert 'evidence_semantics' not in declaration['required']
    generic = planner_contract(legacy.PLAN_SCHEMA)
    generic_declaration = generic['properties']['program']['properties']['variables']['additionalProperties']
    assert 'evidence_semantics' not in generic_declaration['properties']


def atomic_state():
    _, engine, old_state, calls = fixture(max_repairs_per_request=0)
    engine.personal_response_selection = True
    program = deepcopy(old_state.program)
    program['variables']['history'] = {
        'type': 'text', 'demand': 'history', 'evidence_semantics': ATOMIC,
        'scope': {'entity_scope': 'user', 'time_scope': 'history'}}
    state = FactState(program, {**old_state.contract, 'time_scope': 'response'}, engine.steps, 'suggestion')
    reader = JointReader(engine, state)
    initial = reader._observation(observation('atomic-old', ['s1'], value=OLD),
                                  'history', visible(reader), engine.steps[0])
    state.integrate([initial])
    original = next(iter(state.routes))
    revision = observation('atomic-expanded', ['s1', 's2'], value=EXPANDED, summary_revision={
        'supersedes_route_ids': [original], 'previous_version': state.versions['history'],
        'reason': 'These two compatible experiences belong in a broader summary.'})
    return reader, engine, state, calls, initial, original, revision


def test_atomic_reader_cannot_offer_or_accept_summary_revision():
    reader, engine, state, calls, _, original, revision = atomic_state()
    before_routes, before_versions = deepcopy(state.routes), deepcopy(state.versions)
    def respond(index, data):
        assert 'history' not in data.get('summary_revision_contracts', {})
        return response(deepcopy(revision))
    calls.respond = respond
    reader.read(engine.steps[0], 'Review later personal activity', ['history'], ['painting'], 'atomic-query')
    assert len(calls.requests) == 1
    assert state.routes == before_routes and state.versions == before_versions
    assert not state.routes[original].get('superseded_by')
    assert state.valid()[0]['history'] == OLD and reader.pending_variables() == {'history'}


def test_atomic_state_rejects_revision_even_if_caller_bypasses_reader():
    _, engine, state, _, initial, _, revision = atomic_state()
    normalized = {**initial, 'id': revision['id'], 'value': EXPANDED,
                  'source_ids': [engine.mapper.sources[s]['source_id'] for s in ('s1', 's2')],
                  'summary_revision': revision['summary_revision']}
    before = deepcopy(state.public_dict())
    with pytest.raises(ProgramError, match='summary'):
        state.integrate([normalized])
    assert state.public_dict() == before


def test_atomic_marker_preserves_whole_value_conflict_instead_of_merging_history():
    reader, engine, state, _, _, original, _ = atomic_state()
    later = reader._observation(observation('another-whole-value', ['s2'], value=EXPANDED),
                                'history', visible(reader, ['s2']), engine.steps[0])
    state.integrate([later])
    assert 'history' not in state.valid()[0]
    assert len(state.routes) == 2
    assert state.routes[original]['value'] == OLD
    assert all(not route.get('superseded_by') and not route['revoked'] for route in state.routes.values())
    assert state.program['variables']['history']['evidence_semantics'] == ATOMIC
