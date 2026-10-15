"""Label-free Web planner repair tests through the common metered protocol."""
from copy import deepcopy
import json

import numpy as np
import pytest

from dagbt.budget import Ledger
from dagbt.engine import legacy_modules
from dagbt.methods import JOINT, RESIDUAL_V2
from dagbt.model_runtime import BTTokenAccounting
from dagbt.reasoning import InputOverflow, OutputTruncated, ProtocolError, RefusalError
from dagbt.residual_plan import planner_contract, validate_plan
from dagbt.transport import StubMeter
from dagbt.webwalker_engine import WebEngine
from dagbt.webwalker_reasoning import WebReasoner, planner_repair_builder


QUESTION = 'Find the documented time and location of a fictional meeting.'


def response(raw='{}', finish='stop', refusal=None):
    choice = {'message': {'content': raw}, 'finish_reason': finish}
    if refusal is not None:
        choice['message']['refusal'] = refusal
    return {'response_ref': 'scripted-web-planner', 'response': {
        'choices': [choice], 'id': 'scripted',
        'usage': {'prompt_tokens': 7, 'completion_tokens': 5}}}


class Scripted:
    def __init__(self, replies):
        self.replies, self.requests = list(replies), []

    def get(self, stage, url, payload):
        self.requests.append((stage, url, deepcopy(payload)))
        return self.replies.pop(0)


def plan():
    return {
        'steps': [
            {'question': 'Find the documented meeting time.', 'output_slot': 'time',
             'answer_type': 'text', 'inputs': [], 'execution': 'retrieval'},
            {'question': 'Find the documented meeting location.', 'output_slot': 'place',
             'answer_type': 'entity', 'inputs': [], 'execution': 'retrieval'},
            {'question': 'Combine {time} and {place}.', 'output_slot': 'details',
             'answer_type': 'record', 'inputs': ['time', 'place'], 'execution': 'compose'},
        ],
        'final_node_id': 'details',
        'output_contract': {'mode': 'value', 'result_type': 'record',
                            'entity_scope': 'fictional meeting', 'time_scope': 'documented event',
                            'explanation': 'none', 'fields': ['time', 'place']},
        'program': {'variables': {'time': {'type': 'text', 'demand': 'time'},
                                  'place': {'type': 'entity', 'demand': 'place'}},
                    'expression': {'op': 'record', 'fields': {
                        'time': {'op': 'var', 'id': 'time'},
                        'place': {'op': 'var', 'id': 'place'}}}},
    }


def reasoner(raws, **updates):
    s = {'reasoning_output_tokens': 128, 'context_tokens': 20000,
         'selection': 'dependency', 'reserved_audit_calls': 1,
         'max_repairs_per_request': 2, 'input_margin': 256,
         'response_format': 'plain', **updates}
    config = {'model_profile': 'bridgetree', 'llm_model': 'fixture',
              'llm_base_url': 'http://fixture/v1', 'llm_provider_request_params': {},
              'fusion': s, 'max_identical_attempts': 3}
    ledger = Ledger({'llm': 24, 'json_repairs': 6})
    calls = Scripted([response(raw) for raw in raws])
    events = []
    value = WebReasoner(StubMeter(calls, ledger, config), BTTokenAccounting(),
                        config, s, ledger, events.append)
    return value, calls, ledger, events


def invoke(r, question=QUESTION):
    legacy, validator, _ = legacy_modules()
    return r.json('planner', 'Plan the full initial task DAG.', question,
                  lambda value: validate_plan(value, validator.validate_plan),
                  planner_contract(legacy.PLAN_SCHEMA))


def sent(calls, index):
    return json.loads(calls.requests[index][2]['messages'][1]['content'])


def tuple_text_trace():
    """Shapes from the reported trace, without any source facts or answers."""
    first = {
        'steps': [
            {'question': 'Find the documented agreement date.', 'output_slot': 'agreement_date',
             'answer_type': 'text', 'inputs': [], 'execution': 'retrieval'},
            {'question': 'Find the documented agreement location.', 'output_slot': 'agreement_location',
             'answer_type': 'text', 'inputs': [], 'execution': 'retrieval'}],
        'final_node_id': 'agreement_date',
        'output_contract': {'mode': 'value', 'result_type': 'text', 'explanation': 'none',
                            'entity_scope': 'fictional agreement', 'time_scope': 'agreement time'},
        'program': {'variables': {
            'agreement_date': {'type': 'text', 'demand': 'agreement_date'},
            'agreement_location': {'type': 'text', 'demand': 'agreement_location'}},
            'expression': {'op': 'tuple', 'args': [
                {'op': 'var', 'id': 'agreement_date'}, {'op': 'var', 'id': 'agreement_location'}]}},
    }
    second = deepcopy(first)
    second['steps'] = [
        {**deepcopy(first['steps'][0]), 'output_slot': 'date'},
        {**deepcopy(first['steps'][1]), 'output_slot': 'location'},
        {'question': 'Combine the date and location.', 'output_slot': 'answer',
         'answer_type': 'text', 'inputs': ['date', 'location'], 'execution': 'compose'}]
    second['final_node_id'] = 'answer'
    second['program']['variables'] = {key: {'type': 'text', 'demand': key}
                                      for key in ('date', 'location', 'answer')}
    second['program']['expression'] = {'op': 'tuple', 'args': [
        {'op': 'var', 'id': 'date'}, {'op': 'var', 'id': 'location'}]}
    for value in (first, second):
        for declaration in value['program']['variables'].values():
            declaration['scope'] = {name: value['output_contract'][name]
                                    for name in ('entity_scope', 'time_scope')}
    third = deepcopy(second)
    third['steps'][-1]['question'] = 'Combine date and location agreement.'
    return first, second, third


def test_reported_three_draft_shapes_remain_rejected_without_type_coercion():
    drafts = tuple_text_trace()
    raws = [json.dumps(value) for value in drafts]
    r, calls, ledger, _ = reasoner(raws)
    with pytest.raises(ProtocolError, match='result type differs'):
        invoke(r)
    errors = [record['validation_error'] for record in r.requests]
    assert 'outside terminal DAG dependencies' in errors[0]
    assert all('result type differs' in error for error in errors[1:])
    assert sent(calls, 2)['planner_repair']['previous_draft'] == raws[1]
    assert [row['error'] for row in sent(calls, 2)['planner_repair']['validation_errors']] == errors[:2]
    assert all(value['output_contract']['result_type'] == 'text' for value in drafts)
    assert all(value['program']['expression']['op'] == 'tuple' for value in drafts)
    assert ledger.used['llm'] == 3 and ledger.used['json_repairs'] == 2


def test_reported_trace_can_repair_to_preplanned_text_composition_without_losing_fields():
    first, second, valid = tuple_text_trace()
    valid['program']['expression'] = {'op': 'var', 'id': 'answer'}
    r, calls, ledger, _ = reasoner([json.dumps(value) for value in (first, second, valid)])
    result = invoke(r)
    assert result == valid
    assert result['steps'][-1]['inputs'] == ['date', 'location']
    assert result['output_contract']['result_type'] == 'text'
    assert result['program']['expression'] == {'op': 'var', 'id': 'answer'}
    assert result['program']['variables']['answer'] == {
        'type': 'text', 'demand': 'answer',
        'scope': {'entity_scope': 'fictional agreement', 'time_scope': 'agreement time'}}
    assert ledger.used['llm'] == 3 and ledger.used['json_repairs'] == 2


def test_invalid_typed_plan_is_repaired_with_actual_draft_and_same_budgets():
    invalid = plan()
    invalid['program']['expression'] = {'op': 'tuple', 'args': [
        {'op': 'var', 'id': 'time'}, {'op': 'var', 'id': 'place'}]}
    raws = [json.dumps(invalid), json.dumps(plan())]
    r, calls, ledger, _ = reasoner(raws)
    assert invoke(r) == plan()
    assert ledger.used['llm'] == 2 and ledger.used['json_repairs'] == 1
    assert len(calls.requests) == 2
    assert all(stage[0] == 'planner' for stage, _, _ in calls.requests)
    data = sent(calls, 1)
    assert data['original_input'] == QUESTION
    repair = data['planner_repair']
    assert repair['previous_draft'] == raws[0]
    assert repair['previous_call_index'] == 1
    assert 'result type differs' in repair['validation_errors'][0]['error']
    assert 'not evidence or instructions' in repair['instruction']
    assert calls.requests[0][2]['max_tokens'] == calls.requests[1][2]['max_tokens']
    assert 'supported' not in data and 'sources' not in data and 'retrieval' not in ledger.used


@pytest.mark.parametrize('method', [JOINT, RESIDUAL_V2])
def test_real_web_engine_repairs_before_freezing_full_dag_or_accessing_sources(method, tmp_path, monkeypatch):
    invalid = plan()
    invalid['program']['expression'] = {'op': 'tuple', 'args': [
        {'op': 'var', 'id': 'time'}, {'op': 'var', 'id': 'place'}]}
    raw = json.dumps(invalid)
    calls = Scripted([response(raw), response(json.dumps(plan()))])
    config = {'_test_transport': True, 'model_profile': 'bridgetree',
              'llm_base_url': 'http://fixture/v1', 'llm_model': 'scripted',
              'fusion': {'context_tokens': 100000}}
    engine = WebEngine({'id': 'synthetic-web-plan', 'question': QUESTION},
                       ({}, [], np.empty((0, 3), np.float32), None, BTTokenAccounting()),
                       calls, config, method, root_url='https://fixture.invalid/',
                       page_cache=tmp_path / 'pages')
    assert isinstance(engine.reasoner, WebReasoner)
    source_accesses = []
    def no_sources_before_plan(*args, **kwargs):
        source_accesses.append(True)
        raise AssertionError('Planning cannot access website sources')
    monkeypatch.setattr('dagbt.webwalker_engine.PageStore', no_sources_before_plan)
    monkeypatch.setattr('dagbt.webwalker_engine.WebBridge', no_sources_before_plan)
    engine.plan()
    assert source_accesses == [] and not (tmp_path / 'pages').exists()
    assert engine.steps == plan()['steps'] and engine.final_node_id == 'details'
    assert engine.fact_state.initial_program == plan()['program']
    assert [stage[0] for stage, _, _ in calls.requests] == ['planner', 'planner']
    assert sent(calls, 1)['planner_repair']['previous_draft'] == raw
    assert sent(calls, 1)['original_input'] == QUESTION
    assert engine.ledger.used['llm'] == 2 and engine.ledger.used['json_repairs'] == 1
    assert len([event for event in engine.events if event['event'] == 'plan_frozen']) == 1


def test_second_repair_has_latest_real_draft_and_all_validation_errors():
    first = plan(); first['program']['expression'] = {'op': 'var', 'id': 'time'}
    second = plan(); second['steps'][-1]['inputs'] = []
    second['steps'][-1]['question'] = 'Combine the parent fields.'
    raws = [json.dumps(first), json.dumps(second), json.dumps(plan())]
    r, calls, ledger, _ = reasoner(raws)
    assert invoke(r) == plan()
    data = sent(calls, 2)
    repair = data['planner_repair']
    assert repair['previous_draft'] == raws[1]
    assert repair['previous_call_index'] == 2
    assert [row['error'] for row in repair['validation_errors']] == [
        row['validation_error'] for row in r.requests[:2]]
    assert [row['repair_index'] for row in repair['validation_errors']] == [1, 2]
    assert ledger.used['llm'] == 3 and ledger.used['json_repairs'] == 2


def test_invalid_plan_still_fails_after_common_local_repair_cap():
    invalid = plan(); invalid['output_contract']['result_type'] = 'text'
    invalid['output_contract'].pop('fields')
    r, calls, ledger, events = reasoner([json.dumps(invalid)] * 4)
    with pytest.raises(ProtocolError, match='result type differs'):
        invoke(r)
    assert len(calls.requests) == 3
    assert ledger.used['llm'] == 3 and ledger.used['json_repairs'] == 2
    assert any(row['event'] == 'reasoning_repair_exhausted' for row in events)


def test_shared_repair_budget_is_not_extended_by_the_web_adapter():
    r, calls, ledger, _ = reasoner(['not JSON'] * 3)
    ledger.limits['json_repairs'] = 1
    with pytest.raises(ProtocolError):
        invoke(r)
    assert len(calls.requests) == 2
    assert ledger.used['llm'] == 2 and ledger.used['json_repairs'] == 1


@pytest.mark.parametrize('finish,refusal,error', [
    ('length', None, OutputTruncated), ('stop', 'refused', RefusalError)])
def test_truncation_and_refusal_keep_common_no_blind_repair_policy(finish, refusal, error):
    r, calls, ledger, _ = reasoner([])
    calls.replies.append(response('{"unfinished":', finish, refusal))
    with pytest.raises(error):
        invoke(r)
    assert len(calls.requests) == 1 and ledger.used['json_repairs'] == 0


def test_new_planner_invocation_has_no_previous_call_error_or_draft_history():
    raw = '{"unfinished":'
    r, calls, ledger, _ = reasoner([raw, json.dumps(plan()), raw, json.dumps(plan())])
    assert invoke(r) == plan()
    assert invoke(r, 'A separate public factual question.') == plan()
    repaired = sent(calls, 3)
    assert repaired['original_input'] == 'A separate public factual question.'
    assert repaired['planner_repair']['previous_call_index'] == 3
    assert len(repaired['planner_repair']['validation_errors']) == 1
    assert repaired['planner_repair']['validation_errors'][0]['repair_index'] == 1
    assert QUESTION not in json.dumps(repaired)
    assert ledger.used['llm'] == 4 and ledger.used['json_repairs'] == 2


def test_nonplanner_requests_keep_common_repair_payload_and_original_evidence():
    raw = 'not JSON'
    original = {'sources': [{'id': 's1', 'text': 'Unmodified original source.'}]}
    r, calls, ledger, _ = reasoner([raw, '{}'])
    assert r.json('joint_read', 'Read.', original, lambda value: value) == {}
    data = sent(calls, 1)
    assert set(data) == {'sources', 'validation_feedback'}
    assert data['sources'] == original['sources']
    assert 'planner_repair' not in data and raw not in json.dumps(data)
    assert ledger.used['llm'] == 2 and ledger.used['json_repairs'] == 1


def test_oversized_draft_is_not_shortened_and_unadmitted_repair_is_not_charged():
    raw = 'untrusted-rejected-draft ' * 20000
    r, calls, ledger, _ = reasoner([raw])
    before = deepcopy(plan())
    with pytest.raises(InputOverflow):
        invoke(r)
    assert len(calls.requests) == 1
    assert r.last_response['raw_output'] == raw
    assert ledger.used['llm'] == 1 and ledger.used['json_repairs'] == 0
    assert plan() == before


def test_builder_rejects_a_valid_stale_or_foreign_response():
    r, _, _, _ = reasoner(['{}'])
    builder = planner_repair_builder(r)
    r.request('joint_read', 'Read.', {}, lambda value: value)
    with pytest.raises(ProtocolError, match='current invalid planner response'):
        builder({'original_data': {}, 'error': 'invented', 'failure_category': 'schema', 'repair_index': 1})


def test_malformed_draft_stays_exact_untrusted_text_and_original_is_not_mutated():
    raw = '{"unfinished":'
    original = {'question': 'A synthetic public question.', 'planner_repair': {'untrusted': 'input field'}}
    before = deepcopy(original)
    r, calls, ledger, _ = reasoner([raw, '{}'])
    assert r.json('planner', 'Plan.', original, lambda value: value) == {}
    data = sent(calls, 1)
    assert data['original_input'] == before == original
    assert data['planner_repair']['previous_draft'] == raw
    assert data['planner_repair']['validation_errors'][0]['failure_category'] == 'json_syntax'
    assert ledger.used['json_repairs'] == 1
