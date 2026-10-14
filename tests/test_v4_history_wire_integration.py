"""Shared input text in real reader paths, with offline counters and replies."""
from copy import deepcopy
import json

import pytest

from dagbt.fact_state import FactState
from dagbt.history_wire import INSTRUCTION, unpack
from dagbt.joint_reading import JointReader
from dagbt.reasoning import ProtocolError
from test_personamem_joint_protocol import fixture, observation, visible
from test_v4_joint_reading import terminal_response


HISTORY = 'The user enjoyed playing piano and composing music last year. ' * 30
QUERY = 'Choose a response from the established personal history: ' + HISTORY


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('History wire integration tests cannot contact services')
    monkeypatch.setattr('urllib.request.urlopen', forbidden)
    monkeypatch.setattr('urllib.request.build_opener', forbidden)


def history_terminal():
    _, engine, old_state, calls = fixture(max_repairs_per_request=1)
    engine.personal_response_selection = True
    engine.output_options = ['Try a piano duet.', 'Go painting.', 'Try rugby.', 'No advice.']
    program = deepcopy(old_state.initial_program)
    program['variables']['history'] = {
        'type': 'text', 'demand': 'history',
        'scope': {'entity_scope': 'user', 'time_scope': 'history'},
        'evidence_semantics': 'personal_history_summary',
    }
    state = FactState(program, old_state.contract, engine.steps, 'suggestion')
    reader = JointReader(engine, state)
    row = observation('long-history', ['s1'], value=HISTORY)
    state.integrate([reader._observation(row, 'history', visible(reader), engine.steps[0])])
    reader.current_identity = 'shared-history'
    return reader, engine, state, calls


def payload(reader, engine, operation):
    audit = operation.startswith('audit_joint')
    semantic = operation.startswith('semantic_compose')
    data = reader._payload(engine.steps[1], QUERY, ['history', 'suggestion'], ['s1'],
                           semantic=semantic, audit=audit)
    if operation.endswith('_repair'):
        data['repair_scope'] = {'failed': [{'variable': 'suggestion', 'id': '*',
            'feedback': 'Correct the invalid observation.', 'observation': None}],
            'attempt': 1, 'previous_response_identity': '0' * 64}
    return data


def capture_estimates(monkeypatch, reader):
    reasoner = reader.e.reasoner
    original = reasoner._count_payload
    estimate = reader._estimate
    in_estimate = False
    seen = []
    def count(operation, request):
        if in_estimate:
            seen.append((operation, deepcopy(request)))
        return original(operation, request)
    def inspect_estimate(*args, **kwargs):
        nonlocal in_estimate
        in_estimate = True
        try:
            return estimate(*args, **kwargs)
        finally:
            in_estimate = False
    monkeypatch.setattr(reasoner, '_count_payload', count)
    monkeypatch.setattr(reader, '_estimate', inspect_estimate)
    return seen


@pytest.mark.parametrize('operation', ['joint_read', 'semantic_compose', 'audit_joint',
    'joint_read_repair', 'semantic_compose_repair', 'audit_joint_repair'])
def test_estimate_and_request_share_input_system_and_canonical_schema(operation, monkeypatch):
    reader, engine, state, calls = history_terminal()
    data = payload(reader, engine, operation)
    before, state_before = deepcopy(data), deepcopy(state.public_dict())
    canonical_identity = reader._selection_data_identity(data)
    estimates = capture_estimates(monkeypatch, reader)
    count = reader._estimate(operation, data)
    wire, system = reader._prepare_wire(data)
    assert wire['shared_history_texts'] == {'t0': HISTORY}
    assert system.count(INSTRUCTION) == 1
    assert unpack(wire) == reader._wire_data(data)
    assert len(estimates) == 1
    admitted = deepcopy(estimates[0])
    calls.respond = lambda index, current: {'slots': []}
    engine.reasoner.request(operation, system, wire, schema=reader._schema(data))
    assert len(estimates) == 1
    assert admitted == estimates[-1] == (operation, calls.requests[0]['payload'])
    assert engine.reasoner.requests[-1]['input_tokens_local'] == count
    actual_schema = json.loads(calls.requests[0]['payload']['messages'][0]['content'].split(
        'matching this schema:\n', 1)[1])
    assert actual_schema == reader._schema(before)
    assert 'value_ref' not in json.dumps(actual_schema) and 'text_ref' not in json.dumps(actual_schema)
    assert data == before and state.public_dict() == state_before
    assert reader._selection_data_identity(data) == canonical_identity


def test_actual_read_and_terminal_only_repair_use_the_same_prepared_admission(monkeypatch):
    reader, engine, state, calls = history_terminal()
    estimates = capture_estimates(monkeypatch, reader)
    selected = engine.output_options[0]
    def respond(index, data):
        assert data['shared_history_texts'] == {'t0': HISTORY}
        assert unpack(data)['supported_parents']['history']['value'] == HISTORY
        if index == 1:
            return terminal_response(data, selected, semantic_answer='An inconsistent terminal summary.')
        assert data['terminal_repair']['expected_value'] == selected
        assert data['requested_slots'] == []
        return {'slots': [], 'semantic_answer': selected, 'final_prediction': 'a'}
    calls.respond = respond
    reader.read(engine.steps[1], QUERY, ['suggestion'], [], 'shared-history',
                semantic=True, terminal_output=True, source_scope='mandatory_closure')
    assert len(calls.requests) == 2
    for request in calls.requests:
        assert (request['stage'][0], request['payload']) in estimates
        assert request['payload']['messages'][0]['content'].count(INSTRUCTION) == 1
    repair_schema = json.loads(calls.requests[1]['payload']['messages'][0]['content'].split(
        'matching this schema:\n', 1)[1])
    assert repair_schema['required'] == ['slots', 'semantic_answer', 'final_prediction']
    assert state.valid()[0] == {'history': HISTORY, 'suggestion': selected}
    assert reader.semantic_inputs['validation_complete'] and reader.semantic_prediction == 'a'
    assert engine.ledger.used['llm'] == 2 and engine.ledger.used['json_repairs'] == 1


def test_terminal_repair_schema_overflow_is_rejected_before_repair_charge(monkeypatch):
    reader, engine, state, calls = history_terminal()
    data = payload(reader, engine, 'semantic_compose')
    value = engine.output_options[0]
    row = observation('suggestion', ['s1'], value=value,
                      parent_versions={'history': state.versions['history']})
    rows = [reader._observation(row, 'suggestion', visible(reader), engine.steps[1])]
    counted = []
    def count(operation, request):
        schema = json.loads(request['messages'][0]['content'].split('matching this schema:\n', 1)[1])
        counted.append(schema)
        assert schema['required'] == ['slots', 'semantic_answer', 'final_prediction']
        maximum_input = engine.s['context_tokens'] - request['max_tokens'] - engine.s['input_margin'] - 8
        return {'count': maximum_input + 1}
    monkeypatch.setattr(engine.reasoner, '_count_payload', count)
    result = reader._repair_terminal('semantic_compose', data, rows, 'a', 'wrong',
                                    engine.steps[1], 'shared-history', 0)
    assert result[2] and len(counted) == 1
    assert not calls.requests and not engine.reasoner.requests
    assert engine.ledger.used['json_repairs'] == engine.ledger.used['llm'] == 0
    assert state.valid()[0] == {'history': HISTORY}


@pytest.mark.parametrize('change', [
    {'phase': 'history_evidence'}, {'phase': 'response_audit'},
    {'audit_instruction': None}, {'audit_instruction': 'Review the response.'},
])
def test_terminal_only_repair_rejects_history_and_audit_contracts(change):
    reader, engine, _, _ = history_terminal()
    data = payload(reader, engine, 'semantic_compose')
    data.update(variables={}, requested_slots=[], requested_scopes={},
                terminal_repair={'expected_value': engine.output_options[0]}, **change)
    with pytest.raises(ProtocolError, match='history or audit phase'):
        reader._schema(data)


def test_fresh_forecast_and_actual_repair_share_encoding_without_budget_changes(monkeypatch):
    reader, engine, _, calls = history_terminal()
    data = payload(reader, engine, 'joint_read')
    # This capacity-policy fixture represents a historical retrieval with an
    # established parent; it does not integrate or manufacture any observation.
    data['task'] = {**data['task'], 'execution': 'retrieval'}
    before = deepcopy(data)
    settings = deepcopy(engine.s)
    estimates = capture_estimates(monkeypatch, reader)
    assert reader._fresh_window_fits('joint_read', data)
    forecast = reader._whole_review_forecast(data)
    assert reader._repair_fits('joint_read_repair', forecast, engine.steps[1], 'capacity', 1)
    assert len(estimates) == 4
    for operation, request in estimates:
        wire = json.loads(request['messages'][1]['content'])
        assert wire['shared_history_texts'] == {'t0': HISTORY}
        assert request['messages'][0]['content'].count(INSTRUCTION) == 1
        assert request['max_tokens'] == settings['reasoning_output_tokens']
    assert data == before and engine.s == settings
    assert not calls.requests and engine.ledger.used['llm'] == 0


def test_output_structure_floor_retains_full_values_and_bypasses_input_encoding(monkeypatch):
    reader, engine, _, calls = history_terminal()
    data = reader._payload(engine.steps[0], 'Review full history', ['history'], ['s1'])
    def forbidden(*args, **kwargs):
        raise AssertionError('The assistant output floor must not share input references')
    monkeypatch.setattr(reader, '_prepare_wire', forbidden)
    monkeypatch.setattr(reader, '_wire_candidates', forbidden)
    monkeypatch.setattr(reader, '_select_wire', forbidden)
    captured = []
    def count(operation, request):
        captured.append(request)
        assert len(request['messages']) == 1 and request['messages'][0]['role'] == 'assistant'
        result = json.loads(request['messages'][0]['content'])
        values = [row['value'] for slot in result['slots'] for row in slot['observations']]
        assert HISTORY in values
        assert 'value_ref' not in request['messages'][0]['content']
        return {'count': 401, 'token_count_is_estimate': False}
    monkeypatch.setattr(engine.reasoner, '_count_payload', count)
    assert reader._fresh_structure_floor('joint_read', data, output_token_cap=450) == 401
    assert len(captured) == 1 and captured[0]['max_tokens'] == 450
    assert not calls.requests and not engine.reasoner.requests and engine.ledger.used['llm'] == 0
