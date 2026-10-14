"""Offline complete-wire selection tests; all service responses are explicit fakes."""
from copy import deepcopy
from io import BytesIO
import json
from types import SimpleNamespace

import pytest

from dagbt import review_wire
from dagbt.reasoning import InputOverflow
from dagbt.token_counting import TokenizationError
from dagbt.transport import Transport
from test_personamem_joint_protocol import fixture, response, observation
from test_v4_review_wire import review_fixture, exact


def payload_fixture(**settings):
    reader, engine, state, calls = fixture(**settings)
    data = reader._payload(engine.steps[0], 'Read history', ['history'], ['s1'])
    reviews = review_fixture()
    data['known_observations'] = reviews['known_observations']
    data['failed_batch_reviews'] = reviews['failed_batch_reviews']
    return reader, engine, state, calls, data


def fake_accounting(monkeypatch, reader, baseline=100, reviewed=90,
                    baseline_flag=False, reviewed_flag=False, fail_side=None, error=None):
    calls = []
    def count(operation, payload):
        candidate = 'shared_review_texts' in json.loads(payload['messages'][1]['content'])
        side = 'reviewed' if candidate else 'baseline'
        calls.append((operation, deepcopy(payload), side))
        if side == fail_side:
            raise error
        return {'count': reviewed if candidate else baseline,
                'token_count_is_estimate': reviewed_flag if candidate else baseline_flag}
    monkeypatch.setattr(reader.e.reasoner, '_count_payload', count)
    return calls


def test_pure_candidates_are_independent_and_do_not_contact_counter(monkeypatch):
    reader, engine, state, calls, data = payload_fixture()
    def forbidden(*args, **kwargs):
        raise AssertionError('Pure preparation must not count or generate')
    monkeypatch.setattr(engine.reasoner, '_count_payload', forbidden)
    monkeypatch.setattr(engine.reasoner, 'request', forbidden)
    before = exact((data, state.public_dict(), engine.ledger.used))
    candidates = reader._wire_candidates(data)
    assert len(candidates) == 2
    baseline, reviewed = candidates
    assert review_wire.INSTRUCTION not in baseline[1]
    assert review_wire.INSTRUCTION in reviewed[1]
    assert 'shared_review_texts' not in baseline[0]
    assert 'shared_review_texts' in reviewed[0]
    assert exact(review_wire.unpack(reviewed[0])) == exact(baseline[0])
    assert exact(reader._prepare_wire(data)) == exact(reviewed)
    baseline[0]['known_observations'][0]['extra'].append('baseline-only')
    assert 'baseline-only' not in reviewed[0]['known_observations'][0]['extra']
    reviewed[0]['known_observations'][0]['extra'].append('review-only')
    assert exact((data, state.public_dict(), engine.ledger.used)) == before
    assert not calls.requests


@pytest.mark.parametrize('reviewed_count,select_review', [(90, True), (100, False), (109, False)])
def test_complete_actual_count_overrides_byte_benefit(monkeypatch, reviewed_count, select_review):
    reader, engine, state, calls, data = payload_fixture()
    candidates = reader._wire_candidates(data)
    schema = reader._schema(data)
    before = exact((data, candidates, schema, state.public_dict(), engine.ledger.used))
    counted = fake_accounting(monkeypatch, reader, reviewed=reviewed_count)
    wire, system, accounting = reader._select_wire('joint_read_repair', candidates, schema, output_token_cap=127)
    assert ('shared_review_texts' in wire) is select_review
    assert (review_wire.INSTRUCTION in system) is select_review
    assert accounting['count'] == min(100, reviewed_count)
    assert len(counted) == 2
    expected_schema = json.dumps(schema, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(',', ':'))
    for (operation, payload, _), candidate in zip(counted, candidates):
        assert operation == 'joint_read_repair' and payload['max_tokens'] == 127
        assert payload['messages'][0]['content'] == candidate[1] + '\n\nReturn one complete JSON object matching this schema:\n' + expected_schema
        assert payload['messages'][1]['content'] == exact(candidate[0])
    assert exact((data, candidates, schema, state.public_dict(), engine.ledger.used)) == before
    assert not calls.requests


@pytest.mark.parametrize('baseline_flag,reviewed_flag,expected_counts', [
    (True, False, 1), (None, False, 1), (0, False, 1), ('false', False, 1),
    (False, True, 2), (False, None, 2), (False, 0, 2), (False, 'false', 2),
])
def test_only_explicit_actual_counts_can_select_review(monkeypatch, baseline_flag, reviewed_flag, expected_counts):
    reader, _, _, _, data = payload_fixture()
    candidates = reader._wire_candidates(data)
    counted = fake_accounting(monkeypatch, reader, reviewed=1,
                              baseline_flag=baseline_flag, reviewed_flag=reviewed_flag)
    wire, system, accounting = reader._select_wire('joint_read', candidates, reader._schema(data))
    assert exact((wire, system)) == exact(candidates[0])
    assert accounting['count'] == 100 and len(counted) == expected_counts


def test_single_candidate_has_no_optional_comparison(monkeypatch):
    reader, engine, _, _ = fixture()
    data = reader._payload(engine.steps[0], 'Read history', ['history'], ['s1'])
    candidates = reader._wire_candidates(data)
    assert len(candidates) == 1
    counted = fake_accounting(monkeypatch, reader)
    assert reader._estimate('joint_read', data) == 100
    assert len(counted) == 1


@pytest.mark.parametrize('side', ['baseline', 'reviewed'])
@pytest.mark.parametrize('kind', ['tokenization', 'value', 'connection'])
def test_count_errors_propagate_after_exposure_without_generation_or_state_loss(monkeypatch, side, kind):
    reader, engine, state, calls, data = payload_fixture()
    errors = {'tokenization': TokenizationError('connection_or_timeout'),
              'value': ValueError('explicit offline invalid count'),
              'connection': ConnectionError('explicit offline connection failure')}
    error = errors[kind]
    candidates = reader._wire_candidates(data)
    before = exact((state.public_dict(), engine.ledger.used))
    counted = fake_accounting(monkeypatch, reader, fail_side=side, error=error)
    original_counter = engine.reasoner._count_payload
    def after_exposure(operation, payload):
        assert len(reader.source_exposures) == 1
        assert reader.unread_obligations[('owner', 's1')] == {'history'}
        return original_counter(operation, payload)
    monkeypatch.setattr(engine.reasoner, '_count_payload', after_exposure)
    with pytest.raises(type(error)) as caught:
        reader._request_with_exposure('joint_read', data, 'owner', *candidates[0],
                                      wire_candidates=candidates, schema=reader._schema(data))
    assert caught.value is error
    assert reader.source_exposures[0]['request_status'] == 'not_sent'
    assert reader.source_exposures[0]['error_type'] == type(error).__name__
    assert reader.source_exposures[0]['logical_call_index'] is None
    assert reader.unread_obligations[('owner', 's1')] == {'history'}
    assert engine.reasoner.sequence == 0 and engine.reasoner.last_response is None
    assert exact((state.public_dict(), engine.ledger.used)) == before
    assert len(counted) == (1 if side == 'baseline' else 2) and not calls.requests


@pytest.mark.parametrize('baseline,reviewed,success,selected_review', [
    (724, 733, True, False), (733, 724, True, True), (733, 742, False, False),
])
def test_selection_precedes_existing_capacity_guard_without_changing_reserves(monkeypatch, baseline, reviewed, success, selected_review):
    reader, engine, _, calls, data = payload_fixture(context_tokens=1500)
    candidates = reader._wire_candidates(data)
    fake_accounting(monkeypatch, reader, baseline=baseline, reviewed=reviewed)
    invoke = lambda: reader._request_with_exposure('joint_read', data, 'owner', *candidates[0],
                                                   wire_candidates=candidates, schema=reader._schema(data))
    if success:
        invoke()
        assert len(calls.requests) == 1
        payload = calls.requests[0]['payload']
        assert ('shared_review_texts' in json.loads(payload['messages'][1]['content'])) is selected_review
        assert payload['max_tokens'] == 512
        assert engine.reasoner.requests[0]['input_tokens_local'] == 724
        assert engine.reasoner.requests[0]['input_margin'] == 256
        assert engine.ledger.used['llm'] == 1
    else:
        with pytest.raises(InputOverflow):
            invoke()
        assert not calls.requests and engine.reasoner.sequence == 0
        assert reader.source_exposures[0]['request_status'] == 'not_sent'
        assert not any(engine.ledger.used.values())


class FakeHTTPResponse(BytesIO):
    def getcode(self):
        return 200


def real_transport_with_fake_services(monkeypatch, tmp_path, reader, baseline_count=100, reviewed_count=90):
    config = {**deepcopy(reader.e.reasoner.config), 'model_profile': 'bridgetree',
              'tokenization': {'mode': 'vllm_chat', 'endpoint': 'http://offline.invalid/tokenize'},
              'max_identical_attempts': 1}
    config['fusion'] = deepcopy(reader.e.s)
    reader.e.reasoner.config = config
    monkeypatch.setattr('dagbt.transport.resolve_api_key', lambda *args: '')
    monkeypatch.setattr('dagbt.token_counting.resolve_api_key', lambda *args: '')
    transport = Transport('test-unit', tmp_path, config, reader.e.ledger, None)
    reader.e.reasoner.calls = transport
    token_payloads, sent = [], []
    def count_for(messages):
        return reviewed_count if 'shared_review_texts' in json.loads(messages[1]['content']) else baseline_count
    def tokenize(request, **options):
        assert request.full_url == 'http://offline.invalid/tokenize'
        payload = json.loads(request.data)
        assert set(payload) == {'model', 'messages', 'add_generation_prompt'}
        assert payload['add_generation_prompt'] is True
        token_payloads.append(payload)
        return FakeHTTPResponse(json.dumps({'count': count_for(payload['messages']), 'max_model_len': 16384}).encode())
    def generate(request, **options):
        assert request.full_url == 'http://offline.invalid/v1/chat/completions'
        payload = json.loads(request.data)
        sent.append(payload)
        return FakeHTTPResponse(json.dumps({'choices': [{'finish_reason': 'stop', 'message': {
            'content': json.dumps(response(observation()))}}], 'usage': {
                'prompt_tokens': count_for(payload['messages']), 'completion_tokens': 1}}).encode())
    transport.token_counter._opener = SimpleNamespace(open=tokenize)
    monkeypatch.setattr('urllib.request.urlopen', generate)
    return transport, token_payloads, sent


@pytest.mark.parametrize('operation', ['joint_read', 'joint_read_repair', 'semantic_compose',
                                      'semantic_compose_repair', 'audit_joint', 'audit_joint_repair'])
def test_estimate_and_final_transport_count_exact_same_selected_messages(monkeypatch, tmp_path, operation):
    reader, engine, state, _, data = payload_fixture()
    transport, token_payloads, sent = real_transport_with_fake_services(monkeypatch, tmp_path, reader)
    before = exact((data, state.public_dict()))
    assert reader._estimate(operation, data, output_token_cap=127) == 90
    assert not sent and not any(engine.ledger.used.values())
    candidates = reader._wire_candidates(data)
    reader._request_with_exposure(operation, data, 'owner', *candidates[0],
                                  wire_candidates=candidates, schema=reader._schema(data), output_token_cap=127,
                                  reserve=0 if operation.startswith('audit') else None)
    assert len(sent) == 1 and len(token_payloads) == 2
    assert sent[0]['messages'] == token_payloads[1]['messages']
    assert sent[0]['max_tokens'] == 127 and engine.reasoner.requests[0]['output_token_reserve'] == 127
    assert engine.reasoner.requests[0]['input_tokens_local'] == 90
    assert exact((data, state.public_dict())) == before
    assert engine.ledger.used['llm'] == 1
    assert engine.ledger.used.get('json_repairs', 0) == 0
    saved = json.loads(next((tmp_path / 'requests').glob('*.json')).read_text())
    assert saved['tokenization_validation']['status'] == 'matched'
    events = [json.loads(line) for line in (tmp_path / 'tokenization_events.jsonl').read_text().splitlines()]
    assert len(events) == 6 and sum(not event['cache_hit'] for event in events) == 2
    assert events[-1]['tokenization_request_sha256'] == events[1]['tokenization_request_sha256']


def test_counter_cache_binds_full_messages_and_cap_keeps_its_own_wire_identity(monkeypatch, tmp_path):
    reader, _, _, _, data = payload_fixture()
    _, token_payloads, sent = real_transport_with_fake_services(monkeypatch, tmp_path, reader)
    candidates = reader._wire_candidates(data)
    schema = reader._schema(data)
    _, _, first = reader._select_wire('joint_read', candidates, schema, output_token_cap=127)
    _, _, second = reader._select_wire('joint_read', candidates, schema, output_token_cap=128)
    assert len(token_payloads) == 2 and not sent
    assert first['tokenization_request_sha256'] == second['tokenization_request_sha256']
    assert first['tokenization_wire_sha256'] != second['tokenization_wire_sha256']
    changed_schema = {**deepcopy(schema), 'description': 'A distinct schema instruction.'}
    reader._select_wire('joint_read', candidates, changed_schema, output_token_cap=128)
    assert len(token_payloads) == 4
    reordered = [(dict(reversed(list(wire.items()))), system) for wire, system in candidates]
    reader._select_wire('joint_read', reordered, schema, output_token_cap=128)
    assert len(token_payloads) == 6


def test_canonical_forecast_identity_is_independent_of_selected_representation(monkeypatch):
    reader, engine, state, _, data = payload_fixture()
    original_identity = reader._selection_data_identity(data)
    before = exact((data, state.public_dict(), reader.unread_obligations, reader.source_exposures))
    fake_accounting(monkeypatch, reader, reviewed=90)
    assert reader._estimate('joint_read', data) == 90
    fake_accounting(monkeypatch, reader, reviewed=109)
    assert reader._estimate('joint_read', data) == 100
    assert reader._selection_data_identity(data) == original_identity
    assert exact((data, state.public_dict(), reader.unread_obligations, reader.source_exposures)) == before
    assert not any(engine.ledger.used.values())
