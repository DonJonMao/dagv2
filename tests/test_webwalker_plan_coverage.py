"""Label-free replay of output loss during bounded Web planner repair."""
from copy import deepcopy
import json

import pytest

from dagbt.engine import legacy_modules
from dagbt.reasoning import ProtocolError
from dagbt.residual import certify_output, possible_outputs
from dagbt.residual_plan import validate_plan
from test_webwalker_reasoning import (invoke, plan, reasoner, sent,
                                      tuple_text_trace)


def ordinary_validate(value):
    _, validator, _ = legacy_modules()
    return validate_plan(value, validator.validate_plan)


def repaired(first, second, *, match):
    r, calls, ledger, _ = reasoner([json.dumps(first), json.dumps(second)],
                                   max_repairs_per_request=1)
    with pytest.raises(ProtocolError, match=match):
        invoke(r)
    assert len(calls.requests) == ledger.used['llm'] == 2
    assert ledger.used['json_repairs'] == 1
    assert sent(calls, 1)['planner_repair']['previous_draft'] == json.dumps(first)
    assert all(stage[0] == 'planner' for stage, _, _ in calls.requests)


def narrowed_trace():
    first, _, _ = tuple_text_trace()
    second = deepcopy(first)
    second['final_node_id'] = 'agreement_location'
    second['program']['expression'] = {'op': 'var', 'id': 'agreement_location'}
    return first, second


def test_reported_repair_cannot_certify_one_of_two_requested_outputs():
    first, second = narrowed_trace()
    # The previously admitted plan passes the frozen common validator. Web
    # admission must reject it before retrieval or an incomplete certificate.
    assert ordinary_validate(second) == second
    repaired(first, second, match='outside the terminal dependency closure')


def test_initial_plan_cannot_start_with_a_disconnected_answering_branch():
    _, second = narrowed_trace()
    r, calls, ledger, _ = reasoner([json.dumps(second)], max_repairs_per_request=0)
    with pytest.raises(ProtocolError, match='outside the terminal dependency closure'):
        invoke(r)
    assert len(calls.requests) == 1 and ledger.used['json_repairs'] == 0


def test_repair_cannot_hide_output_loss_by_deleting_the_task_and_variable():
    first, second = narrowed_trace()
    second['steps'] = [second['steps'][1]]
    del second['program']['variables']['agreement_date']
    assert ordinary_validate(second) == second
    repaired(first, second, match='removed a recognized output task or expression dependency')


def test_adding_omitted_output_as_retrieval_parent_does_not_expose_its_value():
    first, second = narrowed_trace()
    second['steps'][1]['inputs'] = ['agreement_date']
    assert ordinary_validate(second) == second
    repaired(first, second, match='removed a recognized output task or expression dependency')


def test_parallel_outputs_cannot_be_repaired_into_mutually_exclusive_branches():
    first, second, _ = tuple_text_trace()
    second['program']['expression'] = {'op': 'if', 'args': [
        {'op': 'const', 'value': True}, {'op': 'var', 'id': 'date'}, {'op': 'var', 'id': 'location'}]}
    assert ordinary_validate(second) == second
    repaired(first, second, match='removed a recognized output task or expression dependency')


def test_terminal_graph_parents_do_not_replace_missing_F_dependencies():
    value = plan()
    value['output_contract'] = {k: v for k, v in value['output_contract'].items() if k != 'fields'}
    value['output_contract']['result_type'] = 'entity'
    value['program']['expression'] = {'op': 'var', 'id': 'place'}
    assert ordinary_validate(value) == value
    r, _, _, _ = reasoner([json.dumps(value)], max_repairs_per_request=0)
    with pytest.raises(ProtocolError, match='initial F omits planned task dependencies'):
        invoke(r)


def test_deterministic_compose_must_use_all_preserved_output_components():
    first = plan()
    first['output_contract']['result_type'] = 'text'
    first['output_contract'].pop('fields')
    first['program']['expression'] = {'op': 'tuple', 'args': [
        {'op': 'var', 'id': 'time'}, {'op': 'var', 'id': 'place'}]}
    second = deepcopy(first)
    second['program']['variables']['details'] = {'type': 'entity', 'demand': 'details'}
    second['output_contract']['result_type'] = 'entity'
    second['program']['expression'] = {'op': 'var', 'id': 'details'}
    second['program']['task_expressions'] = {'details': {'op': 'var', 'id': 'place'}}
    assert ordinary_validate(second) == second
    repaired(first, second, match='initial F omits planned task dependencies')


def joint_record():
    return {
        'steps': [{'question': 'Find the project sponsor, launch date and duration.',
                   'output_slot': 'project_details', 'answer_type': 'record',
                   'inputs': [], 'execution': 'retrieval'}],
        'final_node_id': 'project_details',
        'output_contract': {'mode': 'value', 'result_type': 'record', 'explanation': 'none',
                            'entity_scope': 'fictional project', 'time_scope': 'documented project',
                            'fields': ['sponsor', 'launch_date', 'duration']},
        'program': {'variables': {
            name: {'type': 'text', 'demand': 'project_details'}
            for name in ('sponsor', 'launch_date', 'duration')},
            'expression': {'op': 'record', 'fields': {
                name: {'op': 'var', 'id': name} for name in ('sponsor', 'launch_date', 'duration')}}},
    }


@pytest.mark.parametrize('delete_variable', [False, True])
def test_same_reading_task_cannot_lose_one_of_its_output_variables(delete_variable):
    first = joint_record()
    first['output_contract']['result_type'] = 'text'
    first['output_contract'].pop('fields')
    second = joint_record()
    second['output_contract']['fields'].remove('launch_date')
    del second['program']['expression']['fields']['launch_date']
    if delete_variable:
        del second['program']['variables']['launch_date']
    assert ordinary_validate(second) == second
    repaired(first, second, match='removed a recognized output task or expression dependency')


def test_record_field_contract_cannot_be_removed_even_with_dependencies_preserved():
    first = joint_record()
    first['program']['expression'] = {'op': 'tuple', 'args': [
        {'op': 'var', 'id': name} for name in ('sponsor', 'launch_date', 'duration')]}
    second = joint_record()
    second['output_contract'].pop('fields')
    assert ordinary_validate(second) == second
    repaired(first, second, match='removed required output record fields')


def test_new_value_record_requires_explicit_fields_without_automatic_rewriting():
    value = joint_record()
    value['output_contract'].pop('fields')
    before = deepcopy(value)
    r, _, _, _ = reasoner([json.dumps(value)], max_repairs_per_request=0)
    with pytest.raises(ProtocolError, match='removed required output record fields|requires explicit'):
        invoke(r)
    assert value == before


def test_generic_compound_record_can_repair_without_inventing_any_answers():
    valid = joint_record()
    first = deepcopy(valid)
    first['program']['expression'] = {'op': 'tuple', 'args': [
        {'op': 'var', 'id': name} for name in ('sponsor', 'launch_date', 'duration')]}
    r, calls, ledger, _ = reasoner([json.dumps(first), json.dumps(valid)])
    result = invoke(r, 'Who sponsored the project, when did it start, and how long did it run?')
    assert result == valid
    assert set(result['output_contract']['fields']) == {'sponsor', 'launch_date', 'duration'}
    assert ledger.used['llm'] == 2 and ledger.used['json_repairs'] == 1
    assert len(calls.requests) == 2
    assert all('value' not in expression for expression in result['program']['expression']['fields'].values())


def test_complete_text_terminal_composition_preserves_independent_outputs():
    first, second, valid = tuple_text_trace()
    valid['program']['expression'] = {'op': 'var', 'id': 'answer'}
    r, calls, ledger, _ = reasoner([json.dumps(value) for value in (first, second, valid)])
    assert invoke(r) == valid
    assert len(calls.requests) == 3
    assert ledger.used['json_repairs'] == 2


def test_legal_conditional_branches_can_still_be_eliminated_by_Ft():
    value = {
        'steps': [
            {'question': 'Determine which documented route applies.', 'output_slot': 'choice',
             'answer_type': 'bool', 'inputs': [], 'execution': 'retrieval'},
            {'question': 'Find the value for the first route.', 'output_slot': 'first',
             'answer_type': 'number', 'inputs': [], 'execution': 'retrieval'},
            {'question': 'Find the value for the second route.', 'output_slot': 'second',
             'answer_type': 'number', 'inputs': [], 'execution': 'retrieval'},
            {'question': 'Return the applicable route value.', 'output_slot': 'result',
             'answer_type': 'number', 'inputs': ['choice', 'first', 'second'], 'execution': 'compose'}],
        'final_node_id': 'result',
        'output_contract': {'mode': 'value', 'result_type': 'number', 'explanation': 'none',
                            'entity_scope': 'fictional routes', 'time_scope': 'specified event'},
        'program': {'variables': {name: {'type': kind, 'demand': name}
                                  for name, kind in [('choice', 'bool'), ('first', 'number'), ('second', 'number')]},
                    'expression': {'op': 'if', 'args': [
                        {'op': 'var', 'id': name} for name in ('choice', 'first', 'second')]}}}
    r, calls, ledger, _ = reasoner([json.dumps(value)])
    assert invoke(r) == value
    facts = {'choice': True, 'first': 7}
    outcome = possible_outputs(value['program'], facts)
    certificate = certify_output(value['program'], facts, value['output_contract'])
    assert outcome['status'] == 'determined'
    assert certificate['output'] == 7
    assert certificate['eliminated_variables'] == ['second']
    assert len(calls.requests) == 1 and ledger.used['json_repairs'] == 0

