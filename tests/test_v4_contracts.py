"""Cross-component admission checks for observed certificate dead ends.

All data are invented; no evaluation labels or remote service calls are used.
The tests retain incomplete-rule negative witnesses and sourced rule revision.
"""
from copy import deepcopy

import pytest

from dagbt.fact_state import FactState
from dagbt.reasoning import ProtocolError
from dagbt.residual import (ProgramError, certify_output, expression_type,
                            validate_program_contract, validate_output_contract)
from dagbt.residual_plan import validate_plan


def var(name):
    return {'op': 'var', 'id': name}


def contract(mode='value', kind='text'):
    return {'mode': mode, 'result_type': kind, 'entity_scope': 'fictional user',
            'time_scope': 'documented history', 'explanation': {
                'one_reason': 'one', 'all_failures': 'all', 'semantic': 'semantic'
            }.get(mode, 'none')}


def step(name, parents=(), kind='retrieval'):
    return {'output_slot': name, 'question': 'Read ' + name, 'answer_type': 'fact',
            'inputs': list(parents), 'execution': kind}


def static_plan(mode='value', kind='text', complete=True):
    return {'steps': [step('answer')], 'final_node_id': 'answer',
            'output_contract': contract(mode, kind), 'program': {
                'variables': {'result': {'type': kind, 'demand': 'answer'}},
                'expression': var('result'), 'rules_complete': complete}}


def check_plan(plan):
    # local_terminal still performs all DAG validation. Legacy validation is
    # irrelevant to these fully typed, synthetic plans.
    return validate_plan(plan, lambda value: value)


def dynamic_plan(mode='one_reason', kind='record'):
    return {'steps': [step('rules'), step('condition', ['rules']),
                      step('answer', ['rules', 'condition'], 'compose')],
            'final_node_id': 'answer', 'output_contract': contract(mode, kind),
            'program': {'variables': {'rule': {'type': 'rule', 'demand': 'rules'},
                                      'result': {'type': kind, 'demand': 'answer'}},
                        'expression': var('result'), 'rules_complete': False,
                        'rule_binding': {'variable': 'rule', 'condition_demands': ['condition']}}}


def state_for(plan):
    return FactState(plan['program'], plan['output_contract'], plan['steps'], plan['final_node_id'])


def rule_value(mode='one_reason', complete=True):
    return {'variables': {'flag': {'type': 'bool', 'demand': 'condition'}},
            'expression': {'op': 'failures' if mode == 'all_failures' else 'one_reason',
                           'fields': {'necessary_condition': var('flag')}},
            'constraints': [], 'rules_complete': complete}


def support(state, name, value, source='invented-original-source'):
    return {'variable': name, 'value': deepcopy(value), 'stance': 'support',
            'source_ids': [source], 'parent_versions': {}, 'retract_ids': [],
            'reason': 'Synthetic supported observation', **state.scope(name)}


@pytest.mark.parametrize('mode,kind', [('one_reason', 'text'), ('one_reason', 'entity'),
                                      ('all_failures', 'text'), ('all_failures', 'record')])
def test_explanation_modes_cannot_claim_unrelated_result_types(mode, kind):
    with pytest.raises(ProtocolError, match='Boolean-witness'):
        check_plan(static_plan(mode, kind))


@pytest.mark.parametrize('mode,kind', [('one_reason', 'record'), ('all_failures', 'set')])
def test_static_var_with_special_mode_is_rejected_before_retrieval(mode, kind):
    plan = static_plan(mode, kind)
    with pytest.raises(ProtocolError, match='requires expression root'):
        check_plan(plan)
    with pytest.raises(ProgramError, match='requires expression root'):
        state_for(plan)
    result = {'decision': False, 'reason': 'fictional'} if kind == 'record' else ['fictional']
    assert certify_output(plan['program'], {'result': result}, plan['output_contract']) is None


def test_static_incomplete_ordinary_value_has_no_legal_completion_route():
    with pytest.raises(ProtocolError, match='no rule completion route'):
        check_plan(static_plan('value', 'entity', complete=False))
    # Open synthesis has no symbolic-completeness claim and is admitted to
    # its separate semantic assessment path.
    plan = static_plan('semantic', 'text', complete=False)
    assert check_plan(plan)['program'] == plan['program']
    assert certify_output(plan['program'], {'result': 'Supported synthesis'}, plan['output_contract']) is None


@pytest.mark.parametrize('kind', ['entity', 'text', 'set', 'record', 'number', 'bool'])
def test_ordinary_sourced_values_preserve_their_type(kind):
    assert check_plan(static_plan(kind=kind))['output_contract']['result_type'] == kind
    plan = static_plan(kind=kind)
    plan['output_contract']['result_type'] = 'number' if kind != 'number' else 'text'
    with pytest.raises(ProtocolError, match='result type differs'):
        check_plan(plan)


@pytest.mark.parametrize('mode,kind', [('one_reason', 'record'), ('all_failures', 'set')])
def test_typed_rule_placeholder_and_compatible_instantiation(mode, kind):
    plan = dynamic_plan(mode, kind)
    assert check_plan(plan)['program'] == plan['program']
    state = state_for(plan)
    assert certify_output(state.program, {}, state.contract) is None
    state.integrate([support(state, 'rule', rule_value(mode))])
    state.integrate([support(state, 'flag', False)])
    certificate = certify_output(state.program, state.valid()[0], state.contract)
    assert certificate['output'] == ({'decision': False, 'reason': 'necessary_condition'}
                                    if mode == 'one_reason' else ['necessary_condition'])
    assert state.contract == plan['output_contract']


@pytest.mark.parametrize('mode,kind', [('one_reason', 'record'), ('all_failures', 'set')])
def test_rule_placeholder_does_not_excuse_wrong_immutable_type(mode, kind):
    plan = dynamic_plan(mode, kind)
    plan['program']['variables']['result']['type'] = 'text'
    with pytest.raises(ProtocolError, match='result type differs'):
        check_plan(plan)


@pytest.mark.parametrize('mode,kind', [('one_reason', 'record'), ('all_failures', 'set')])
def test_incomplete_dynamic_rules_can_reach_later_complete_revision(mode, kind):
    state = state_for(dynamic_plan(mode, kind))
    state.integrate([support(state, 'rule', rule_value(mode, complete=False), 'partial-policy')])
    state.integrate([support(state, 'flag', True)])
    assert certify_output(state.program, state.valid()[0], state.contract) is None
    assert not state.program['rules_complete']
    partial = deepcopy(state.public_dict())
    restored = state_for(dynamic_plan(mode, kind))
    restored.restore(partial)
    assert restored.public_dict() == partial
    state.revoke_source('partial-policy', 'Source superseded by complete policy')
    state.integrate([support(state, 'rule', rule_value(mode), 'complete-policy')])
    assert state.program['rules_complete']
    assert certify_output(state.program, state.valid()[0], state.contract) is not None


@pytest.mark.parametrize('root,mode,kind', [('one_reason', 'one_reason', 'record'),
                                           ('and', 'value', 'bool')])
def test_static_negative_witness_survives_without_complete_rule_collection(root, mode, kind):
    p = {'variables': {'a': {'type': 'bool'}, 'b': {'type': 'bool'}},
         'expression': ({'op': root, 'fields': {'a': var('a'), 'b': var('b')}}
                        if root == 'one_reason' else {'op': root, 'args': [var('a'), var('b')]}),
         'rules_complete': False}
    c = contract(mode, kind)
    assert validate_program_contract(p, c) == p
    assert certify_output(p, {'a': False}, c) is not None
    assert certify_output(p, {'a': True, 'b': True}, c) is None
    assert not p['rules_complete']


@pytest.mark.parametrize('tamper', ['mode_root', 'type', 'fields'])
def test_invalid_rule_batch_is_rejected_before_state_mutation(tamper):
    state = state_for(dynamic_plan())
    bad = rule_value()
    if tamper == 'mode_root':
        bad['expression'] = {'op': 'record', 'fields': {'decision': var('flag')}}
    elif tamper == 'type':
        bad['expression'] = var('flag')
    else:
        bad['expression']['fields']['necessary_condition'] = {'op': 'const', 'value': 'not Boolean'}
    before = deepcopy(state.public_dict())
    with pytest.raises(ProgramError):
        state.integrate([support(state, 'rule', bad)])
    assert state.public_dict() == before


@pytest.mark.parametrize('tamper', ['current_mode', 'historical_mode', 'route_type'])
def test_restore_revalidates_cross_component_contract_before_adopting_snapshot(tamper):
    plan = dynamic_plan()
    state = state_for(plan)
    state.integrate([support(state, 'rule', rule_value())])
    snapshot = deepcopy(state.public_dict())
    if tamper == 'current_mode':
        snapshot['program']['expression'] = var('result')
    elif tamper == 'historical_mode':
        snapshot['instantiations'][0]['program']['expression'] = var('result')
    else:
        snapshot['routes'][0]['value']['expression'] = var('flag')
    fresh = state_for(plan)
    before = deepcopy(fresh.public_dict())
    with pytest.raises(ProgramError):
        fresh.restore(snapshot)
    assert fresh.public_dict() == before


@pytest.mark.parametrize('c', [contract('value', 'set') | {'fields': ['x']},
                               contract('one_reason', 'record') | {'fields': ['explanation']}])
def test_impossible_declared_record_fields_are_rejected(c):
    with pytest.raises(ProgramError, match='fields'):
        validate_output_contract(c)


def test_static_record_fields_must_match_contract():
    p = {'variables': {}, 'expression': {'op': 'record', 'fields': {'x': {'op': 'const', 'value': 1}}}}
    with pytest.raises(ProgramError, match='record fields differ'):
        validate_program_contract(p, contract(kind='record') | {'fields': ['y']})


def test_tuple_type_matches_its_evaluated_json_array():
    p = {'variables': {}, 'expression': {'op': 'tuple', 'args': [{'op': 'const', 'value': 2},
                                                               {'op': 'const', 'value': 'fictional'}]}}
    assert expression_type(p['expression'], {}) == 'set'
    assert validate_program_contract(p, contract(kind='set')) == p
    assert certify_output(p, {}, contract(kind='set'))['output'] == [2, 'fictional']
    with pytest.raises(ProgramError, match='result type differs'):
        validate_program_contract(p, contract(kind='record'))


def test_string_literal_accepts_text_without_retyping_declared_entity():
    p = {'variables': {}, 'expression': {'op': 'const', 'value': 'Documented literal'}}
    assert validate_program_contract(p, contract(kind='text')) == p
    assert certify_output(p, {}, contract(kind='text'))['output'] == 'Documented literal'


@pytest.mark.parametrize('root,mode,kind', [('one_reason', 'one_reason', 'record'),
                                           ('and', 'value', 'bool')])
def test_negative_witness_operator_with_no_possible_negative_is_a_static_dead_end(root, mode, kind):
    p = {'variables': {'flag': {'type': 'bool', 'domain': [True]}},
         'expression': ({'op': root, 'fields': {'necessary': var('flag')}}
                        if root == 'one_reason' else {'op': root, 'args': [var('flag')]}),
         'rules_complete': False}
    with pytest.raises(ProgramError, match='no rule completion route'):
        validate_program_contract(p, contract(mode, kind))


def test_semantic_mode_never_receives_symbolic_negative_witness_certificate():
    p = {'variables': {'a': {'type': 'bool'}},
         'expression': {'op': 'and', 'args': [var('a')]}, 'rules_complete': False}
    assert validate_program_contract(p, contract('semantic', 'bool')) == p
    assert certify_output(p, {'a': False}, contract('semantic', 'bool')) is None


def test_constant_record_keys_are_checked_before_certification():
    p = {'variables': {}, 'expression': {'op': 'const', 'value': {'x': 1}}}
    with pytest.raises(ProgramError, match='record fields differ'):
        validate_program_contract(p, contract(kind='record') | {'fields': ['y']})


def test_restore_rejects_fact_value_changed_to_wrong_type():
    plan = dynamic_plan()
    state = state_for(plan)
    state.integrate([support(state, 'rule', rule_value())])
    state.integrate([support(state, 'flag', False)])
    snapshot = deepcopy(state.public_dict())
    next(r for r in snapshot['routes'] if r['variable'] == 'flag')['value'] = 'false'
    fresh = state_for(plan)
    before = deepcopy(fresh.public_dict())
    with pytest.raises(ProgramError, match='declared type'):
        fresh.restore(snapshot)
    assert fresh.public_dict() == before


def test_restore_cannot_smuggle_another_placeholder_into_a_rule_instantiation():
    plan = dynamic_plan()
    state = state_for(plan)
    state.integrate([support(state, 'rule', rule_value())])
    snapshot = deepcopy(state.public_dict())
    snapshot['instantiations'][0]['program']['expression'] = var('result')
    snapshot['instantiations'][0]['program']['rules_complete'] = False
    snapshot['instantiations'][0]['program']['rule_binding'] = deepcopy(plan['program']['rule_binding'])
    fresh = state_for(plan)
    with pytest.raises(ProgramError, match='unbound rule placeholder'):
        fresh.restore(snapshot)


@pytest.mark.parametrize('expression', [{'op': 'var', 'id': []}, {'op': [], 'args': []},
                                       {'op': 'one_reason', 'fields': {'': {'op': 'const', 'value': False}}}])
def test_malformed_expression_metadata_is_a_repairable_program_error(expression):
    p = {'variables': {}, 'expression': expression}
    with pytest.raises(ProgramError):
        validate_program_contract(p, contract(kind='record'))


def test_null_constant_is_not_a_record_result():
    p = {'variables': {}, 'expression': {'op': 'const', 'value': None}}
    with pytest.raises(ProgramError, match='result type differs'):
        validate_program_contract(p, contract(kind='record'))
    # A null field remains a valid part of a genuine record.
    p['expression'] = {'op': 'record', 'fields': {'optional': {'op': 'const', 'value': None}}}
    assert certify_output(p, {}, contract(kind='record'))['output'] == {'optional': None}


def test_interval_constant_has_numeric_type_matching_runtime_evaluation():
    p = {'variables': {}, 'expression': {'op': 'const', 'value': {'lower': 3, 'upper': 3}}}
    assert expression_type(p['expression'], {}) == 'number'
    assert certify_output(p, {}, contract(kind='number'))['output'] == 3
