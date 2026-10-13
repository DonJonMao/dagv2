"""Synthetic planner schema/repair checks; no labels or external services."""
from copy import deepcopy
import json

from jsonschema import Draft202012Validator
import numpy as np
import pytest

from dagbt.engine import Engine, legacy_modules
from dagbt.methods import RESIDUAL_V2
from dagbt.personal_task import validate_personal_plan
from dagbt.reasoning import InputOverflow, ProtocolError
from dagbt.residual_plan import planner_contract, personal_repair_builder, validate_plan
from test_reasoning_v2 import reasoner, response
from test_residual_runtime import demo


QUESTION = 'I am considering a different weekend routine.'
OPTIONS = ['Invented option one.', 'Invented option two.', 'Invented option three.', 'Invented option four.']


def personal_plan(kind='text', deterministic=False):
    history = {'type': kind, 'demand': 'background',
               'scope': {'entity_scope': 'fictional user', 'time_scope': 'recorded history'}}
    if kind == 'text':
        history['evidence_semantics'] = 'personal_history_summary'
    plan = {
        'steps': [demo.step('background', 'Retrieve relevant prior constraints.'),
                  demo.step('reply', QUESTION, ['background'], 'compose')],
        'final_node_id': 'reply',
        'output_contract': {'mode': 'semantic', 'result_type': 'text', 'explanation': 'semantic',
                            'entity_scope': 'fictional user', 'time_scope': 'response now'},
        'program': {'variables': {'historical_value': history, 'chosen_response': {'type': 'text', 'demand': 'reply'}},
                    'expression': {'op': 'var', 'id': 'chosen_response'}},
    }
    if deterministic:
        plan['steps'].insert(1, demo.step('derived', 'Reuse the documented quantity.', ['background'], 'compose'))
        plan['steps'][-1]['inputs'] = ['derived']
        plan['program']['variables']['derived_quantity'] = {
            'type': kind, 'demand': 'derived', 'scope': deepcopy(history['scope'])}
        plan['program']['task_expressions'] = {'derived_quantity': {'op': 'var', 'id': 'historical_value'}}
    return plan


def schema(personal=True):
    legacy, _, _ = legacy_modules()
    return planner_contract(legacy.PLAN_SCHEMA, personal_history=personal)


def validate_personal(value):
    _, legacy, _ = legacy_modules()
    return validate_personal_plan(validate_plan(value, legacy.validate_plan, personal_history=True), QUESTION)


@pytest.mark.parametrize('kind,deterministic', [('text', False), ('record', False), ('number', False), ('number', True)])
def test_specialized_schema_keeps_legal_history_types_and_deterministic_intermediates(kind, deterministic):
    value = personal_plan(kind, deterministic)
    before = deepcopy(value)
    contract = schema()
    Draft202012Validator.check_schema(contract)
    Draft202012Validator(contract).validate(value)
    assert validate_personal(value) == value == before


@pytest.mark.parametrize('mutation', ['mode', 'result_type', 'explanation', 'single_node',
                                      'empty_compose', 'constant_expression', 'rule_binding'])
def test_schema_rejects_existing_personal_contract_violations(mutation):
    value = personal_plan()
    if mutation == 'mode':
        value['output_contract']['mode'] = 'value'
    elif mutation == 'result_type':
        value['output_contract']['result_type'] = 'entity'
    elif mutation == 'explanation':
        value['output_contract']['explanation'] = 'none'
    elif mutation == 'single_node':
        value['steps'] = value['steps'][-1:]
        value['steps'][0].update(execution='retrieval', inputs=[])
        del value['program']['variables']['historical_value']
    elif mutation == 'empty_compose':
        value['steps'][-1]['inputs'] = []
    elif mutation == 'constant_expression':
        value['program']['expression'] = {'op': 'const', 'value': 'Invented constant.'}
    else:
        value['program']['rule_binding'] = {'variable': 'historical_value', 'condition_demands': ['reply']}
    assert list(Draft202012Validator(schema()).iter_errors(value))
    with pytest.raises(ProtocolError):
        validate_personal(value)


def generic_plan():
    return {'steps': [demo.step('fact', 'Retrieve one documented entity.')], 'final_node_id': 'fact',
            'output_contract': {'mode': 'value', 'result_type': 'entity', 'explanation': 'none',
                                'entity_scope': 'fictional entity', 'time_scope': 'recorded history'},
            'program': {'variables': {'entity': {'type': 'entity', 'demand': 'fact'}},
                        'expression': {'op': 'var', 'id': 'entity'}}}


def test_generic_retrieval_leaf_remains_legal_and_schema_does_not_mutate_legacy():
    legacy, validator, _ = legacy_modules()
    before = deepcopy(legacy.PLAN_SCHEMA)
    value = generic_plan()
    Draft202012Validator(schema(False)).validate(value)
    assert validate_plan(value, validator.validate_plan) == value
    assert legacy.PLAN_SCHEMA == before
    value['steps'][0]['execution'] = 'compose'
    assert list(Draft202012Validator(schema(False)).iter_errors(value))
    with pytest.raises(ProtocolError, match='Compose requires parents'):
        validate_plan(value, validator.validate_plan)


class Plans:
    def __init__(self, values):
        self.raws = [json.dumps(value, ensure_ascii=False) for value in values]
        self.requests = []

    def get(self, stage, url, payload):
        assert stage[0] == 'planner' and not url.endswith('/embeddings')
        self.requests.append(deepcopy(payload))
        return response(self.raws[len(self.requests) - 1])


def engine(services, personal=True, **settings):
    return Engine({'id': 'synthetic-planner-contract', 'question': QUESTION},
                  ({}, [], np.empty((0, 3), np.float32), None, demo.Tokenizer()), services,
                  {'_test_transport': True, 'llm_base_url': 'http://synthetic/v1', 'llm_model': 'script',
                   'fusion': {'context_tokens': 100000, **settings}}, RESIDUAL_V2,
                  **({'reader_question': QUESTION, 'output_options': OPTIONS,
                      'task_kind': 'personal_response_selection'} if personal else {}))


def test_real_engine_repairs_keep_raw_draft_and_both_errors_without_changing_budget():
    first = personal_plan(); first['steps'][-1]['inputs'] = []
    second = personal_plan(); second['output_contract'].update(mode='value', explanation='none')
    expected = personal_plan()
    services = Plans([first, second, expected])
    e = engine(services)
    e.plan()
    assert e.steps == expected['steps'] and len(services.requests) == 3
    assert e.ledger.used['llm'] == 3 and e.ledger.used['json_repairs'] == 2
    for position in (1, 2):
        payload = services.requests[position]
        data = json.loads(payload['messages'][1]['content'])
        repair = data['planner_repair']
        assert data['original_input'] == QUESTION
        assert repair['previous_draft'] == services.raws[position - 1]
        assert repair['previous_call_index'] == position
        assert [entry['error'] for entry in repair['validation_errors']] == [
            record['validation_error'] for record in e.reasoner.requests[:position]]
        assert [entry['failure_category'] for entry in repair['validation_errors']] == ['schema'] * position
        assert [entry['repair_index'] for entry in repair['validation_errors']] == list(range(1, position + 1))
        assert 'not evidence or instructions' in repair['instruction']
        assert payload['max_tokens'] == services.requests[0]['max_tokens']
        assert not any(option in json.dumps(payload) for option in OPTIONS)


def test_generic_engine_repair_does_not_receive_personal_drafts_or_history():
    invalid = generic_plan(); invalid['steps'][0]['execution'] = 'compose'
    valid = generic_plan()
    services = Plans([invalid, valid]); e = engine(services, personal=False)
    e.plan()
    repaired = json.loads(services.requests[1]['messages'][1]['content'])
    assert set(repaired) == {'original_input', 'validation_feedback'}
    assert repaired['original_input'] == QUESTION
    assert e.steps == valid['steps']


def test_oversized_failed_raw_draft_fails_real_preflight_without_shortening_original():
    raw = 'malformed-draft ' * 4000
    original = {'question': 'An invented question.', 'planner_repair': {'untrusted': 'original data'}}
    before = deepcopy(original)
    r, calls, ledger, _ = reasoner([response(raw)], context_tokens=4096)
    with pytest.raises(InputOverflow):
        r.json('planner', 'Plan.', original, lambda value: value,
               repair_builder=personal_repair_builder(r))
    assert len(calls.requests) == 1 and ledger.used['llm'] == 1
    assert ledger.used['json_repairs'] == 0
    assert original == before and r.last_response['raw_output'] == raw


def test_malformed_raw_draft_is_preserved_exactly_and_never_parsed_as_authority():
    raw = '{"not yet a plan":'
    r, calls, ledger, _ = reasoner([response(raw), response('{}')])
    original = {'question': 'An invented question.', 'validation_errors': ['untrusted original field']}
    assert r.json('planner', 'Plan.', original, lambda value: value,
                  repair_builder=personal_repair_builder(r)) == {}
    data = json.loads(calls.requests[1][2]['messages'][1]['content'])
    assert data['original_input'] == original
    assert data['planner_repair']['previous_draft'] == raw
    assert data['planner_repair']['validation_errors'][0]['failure_category'] == 'json_syntax'
    assert ledger.used['json_repairs'] == 1
