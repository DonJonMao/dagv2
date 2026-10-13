"""Independent adaptive fresh-history admission boundaries, entirely offline.

The reader, source obligations, reasoner, parser and request payloads are real.
Only deployment token counts and model replies are deterministic test doubles.
"""
from copy import deepcopy
import json

import pytest

from dagbt.reasoning import InputOverflow
from dagbt.transport import digest
from test_personamem_joint_protocol import observation as base_observation, response, expand_wire_data
from test_v4_summary_revision_review import OLD, summary_fixture


def observation(*args, **kwargs):
    kwargs.setdefault("value", OLD)
    return base_observation(*args, **kwargs)


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Independent adaptive-cap tests must never contact a service')
    monkeypatch.setattr('urllib.request.urlopen', forbidden)
    monkeypatch.setattr('urllib.request.build_opener', forbidden)


class ExactAdmissionCounts:
    """Expose deliberately non-monotone prefix counts without estimating text."""

    def __init__(self, forecasts=None, *, floor=300, exact_floor=True, base=2000,
                 ordinary_repair=2100, actual_repair=2500):
        self.forecasts = forecasts or {1: 8000, 2: 9000, 3: 9500}
        self.floor, self.exact_floor = floor, exact_floor
        self.base, self.ordinary_repair, self.actual_repair = base, ordinary_repair, actual_repair
        self.records = []

    def __call__(self, stage, url, payload):
        messages = payload['messages']
        is_floor = len(messages) == 1 and messages[0]['role'] == 'assistant'
        if is_floor:
            data = json.loads(messages[0]['content'])
            count, kind = self.floor, 'floor'
        else:
            data = expand_wire_data(json.loads(messages[1]['content']))
            repair = data.get('repair_scope', {})
            reviews = repair.get('whole_value_reviews')
            if reviews and all(not group['draft_observations'] for group in reviews):
                count = self.forecasts[len(data['source_spans'])]
                kind = 'forecast'
            elif repair and all(item.get('observation') is None for item in repair.get('failed', [])) \
                    and 'invalid_response' not in repair and not reviews:
                count, kind = self.ordinary_repair, 'ordinary_repair'
            elif repair:
                count, kind = self.actual_repair, 'actual_repair'
            else:
                count, kind = self.base, 'base'
        self.records.append({'kind': kind, 'data': deepcopy(data), 'payload': deepcopy(payload), 'count': count})
        return {'count': count, 'token_count_is_estimate': not self.exact_floor if is_floor else False,
                'token_estimator_id': 'synthetic_exact_admission',
                'tokenization_wire_sha256': digest(payload)}


def adaptive_fixture(counter=None, **settings):
    reader, engine, state, calls, _ = summary_fixture(
        context_tokens=16384, reasoning_output_tokens=4096, **settings)
    engine.s["max_repairs_per_request"] = 1
    calls.respond = lambda index, data: response(observation())
    counter = counter or ExactAdmissionCounts()
    calls.count_reasoning_tokens = counter
    reader.current_identity = 'adaptive-history'
    return reader, engine, state, calls, counter


def read_all(reader, engine, identity='adaptive-history'):
    return reader.read(engine.steps[0], 'Review all recorded activities', ['history'],
                       list(engine.docs), identity)


def choose(reader, engine):
    groups = [[alias] for alias in engine.mapper.sources]
    before = deepcopy(groups)
    result = reader._fitting_prefix('joint_read', engine.steps[0], 'Review all recorded activities',
                                    ['history'], groups, False, False)
    assert groups == before
    return result


def test_fallback_uses_largest_window_and_actual_cap_reaches_preflight_request_and_record(monkeypatch):
    reader, engine, state, calls, counter = adaptive_fixture()
    settings, limits, originals = deepcopy(engine.s), deepcopy(engine.ledger.limits), deepcopy(engine.mapper.sources)
    preflights = []
    original_preflight = engine.reasoner._preflight
    def capture(operation, count, **kwargs):
        preflights.append((operation, count, kwargs.get('output_token_cap')))
        return original_preflight(operation, count, **kwargs)
    monkeypatch.setattr(engine.reasoner, '_preflight', capture)
    read_all(reader, engine)
    cap = 16384 - 9500 - 4096 - 256 - 8
    assert len(calls.requests) == 1
    assert len(calls.requests[0]['data']['source_spans']) == 3
    assert calls.requests[0]['payload']['max_tokens'] == cap < 4096
    assert preflights == [('joint_read', 2000, cap)]
    assert engine.reasoner.requests[0]['output_token_reserve'] == cap
    assert any(record['kind'] == 'base' and record['payload']['max_tokens'] == cap for record in counter.records)
    assert all(record['payload']['max_tokens'] == 4096 for record in counter.records if record['kind'] == 'forecast')
    assert any(record['kind'] == 'floor' for record in counter.records)
    selected = next(event for event in engine.events if event['event'] == 'joint_fresh_output_cap_selected')
    assert selected['output_token_cap'] == cap
    assert selected['whole_review_forecast_input_tokens'] == 9500
    assert selected['repair_output_reserve'] == 4096
    assert selected['forecast_total_tokens'] == 9500 + cap + 4096 + 264 == 16384
    assert state.valid()[0]['history'] == OLD and not state.pending
    assert engine.s == settings and engine.ledger.limits == limits and engine.mapper.sources == originals


def test_existing_full_output_prefix_wins_over_larger_adaptive_window():
    reader, engine, _, calls, counter = adaptive_fixture(ExactAdmissionCounts({1: 7900, 2: 9000, 3: 9500}))
    aliases, consumed = choose(reader, engine)
    assert aliases == ['s1'] and consumed == 1
    assert not calls.requests and not any(record['kind'] == 'floor' for record in counter.records)
    assert not getattr(reader, '_fresh_initial_selection', None)


def test_cap_equal_to_structure_floor_is_admitted_without_rounding_or_empirical_buffer():
    reader, engine, _, calls, _ = adaptive_fixture(ExactAdmissionCounts(floor=2524))
    aliases, consumed = choose(reader, engine)
    assert aliases == ['s1', 's2', 's3'] and consumed == 3
    assert not calls.requests
    assert reader._fresh_initial_selection['output_token_cap'] == 2524


def test_largest_fallback_window_below_structure_floor_is_skipped_without_generation():
    reader, engine, _, calls, counter = adaptive_fixture(
        ExactAdmissionCounts({1: 10500, 2: 11000, 3: 11725}))
    aliases, consumed = choose(reader, engine)
    assert aliases == ['s1', 's2'] and consumed == 2
    assert not calls.requests and engine.ledger.used['llm'] == engine.ledger.used['json_repairs'] == 0
    assert any(record['kind'] == 'floor' for record in counter.records)


@pytest.mark.parametrize('floor,exact', [(300, True), (300, False), (0, True), (True, True)])
def test_impossible_or_unverified_structure_floor_keeps_every_source_pending_without_paid_calls(floor, exact):
    counter = ExactAdmissionCounts({1: 11725, 2: 11725, 3: 11725}, floor=floor, exact_floor=exact)
    reader, engine, state, calls, counter = adaptive_fixture(counter)
    originals = deepcopy(engine.mapper.sources)
    with pytest.raises(InputOverflow):
        read_all(reader, engine)
    assert not calls.requests and not engine.reasoner.requests and engine.reasoner.sequence == 0
    assert engine.ledger.used['llm'] == engine.ledger.used['json_repairs'] == 0
    assert set(reader.unread_obligations) == {('adaptive-history', alias) for alias in originals}
    assert state.pending == {'history'} and state.valid()[0]['history'] == OLD
    assert not reader.failed_batch_reviews and engine.mapper.sources == originals


def test_adaptive_headroom_does_not_rescue_a_request_failing_ordinary_full_repair_capacity():
    reader, engine, state, calls, _ = adaptive_fixture(ExactAdmissionCounts(ordinary_repair=12500))
    with pytest.raises(InputOverflow):
        read_all(reader, engine)
    assert not calls.requests and state.pending == {'history'}
    assert engine.ledger.used['llm'] == engine.ledger.used['json_repairs'] == 0


@pytest.mark.parametrize('case', [
    'generic', 'terminal', 'audit', 'history_compose', 'existing_local_repair', 'durable_recovery',
    'no_request_repairs', 'no_global_repairs', 'only_one_normal_call',
])
def test_adaptive_cap_is_never_offered_outside_repairable_fresh_personal_history(case):
    reader, engine, _, calls, counter = adaptive_fixture()
    data = reader._payload(engine.steps[0], 'Read history', ['history'], ['s1'])
    if case == 'generic':
        data['personal_response_selection'] = False
    elif case == 'terminal':
        data['phase'] = 'response_selection'
    elif case == 'audit':
        data['phase'] = 'response_audit'
    elif case == 'history_compose':
        data['task']['execution'] = 'compose'
    elif case == 'existing_local_repair':
        data['repair_scope'] = {'failed': [{'variable': 'history', 'id': 'old'}]}
    elif case == 'durable_recovery':
        data['failed_batch_reviews'] = [{'id': 'old-batch', 'variable': 'history'}]
    elif case == 'no_request_repairs':
        engine.s['max_repairs_per_request'] = 0
    elif case == 'no_global_repairs':
        engine.ledger.limits['json_repairs'] = 0
    else:
        engine.ledger.limits['llm'] = engine.s['reserved_audit_calls'] + 1
    before = deepcopy(data)
    assert reader._adaptive_fresh_cap('joint_read', data) is None
    assert data == before and not calls.requests
    assert not any(record['kind'] in {'floor', 'forecast'} for record in counter.records)


def test_window_without_fresh_aliases_does_not_enable_output_cap_or_forecast():
    reader, engine, _, calls, counter = adaptive_fixture()
    aliases, consumed = reader._fitting_prefix(
        'joint_read', engine.steps[0], 'Known history only', ['history'], [[]], False, False)
    assert aliases == [] and consumed == 1 and reader._fresh_initial_selection is None
    assert not calls.requests and not any(record['kind'] in {'floor', 'forecast'} for record in counter.records)


def test_repair_restores_full_4096_and_keeps_the_exact_initial_source_window():
    reader, engine, state, calls, _ = adaptive_fixture()
    def reply(index, data):
        if index == 1:
            return response(observation('repair-this', ['not-visible']))
        assert index == 2 and data['repair_scope']
        assert data['source_spans'] == calls.requests[0]['data']['source_spans']
        assert calls.requests[-1]['payload']['max_tokens'] == 4096
        return response(observation('repair-this', ['s1']))
    calls.respond = reply
    read_all(reader, engine)
    assert [request['payload']['max_tokens'] for request in calls.requests] == [2524, 4096]
    assert [record['output_token_reserve'] for record in engine.reasoner.requests] == [2524, 4096]
    assert engine.ledger.used['llm'] == 2 and engine.ledger.used['json_repairs'] == 1
    assert not state.pending and state.valid()[0]['history'] == OLD


def test_length_with_valid_json_never_publishes_and_same_originals_do_not_buy_new_attempts(monkeypatch):
    reader, engine, state, calls, _ = adaptive_fixture()
    original_get = calls.get
    def truncated(*args, **kwargs):
        result = original_get(*args, **kwargs)
        result['response']['choices'][0]['finish_reason'] = 'length'
        return result
    monkeypatch.setattr(calls, 'get', truncated)
    originals = deepcopy(engine.mapper.sources)
    read_all(reader, engine)
    assert len(calls.requests) == 2
    assert [request['payload']['max_tokens'] for request in calls.requests] == [2524, 4096]
    assert state.valid()[0]['history'] == OLD and state.pending == {'history'}
    assert reader.failed_batch_reviews
    held = {source for review in reader.failed_batch_reviews.values() for source in review['source_ids']}
    assert held == {source['source_id'] for source in originals.values()}
    assert all('truncated' in failure['feedback'].lower()
               for review in reader.failed_batch_reviews.values() for failure in review['failed'])
    assert all(json.loads(record['raw_output']) == response(observation()) for record in engine.reasoner.requests)
    failed, used = deepcopy(reader.failed_batch_reviews), deepcopy(engine.ledger.used)
    assert read_all(reader, engine) is False
    assert read_all(reader, engine, 'same-originals-renamed') is False
    assert len(calls.requests) == 2 and engine.ledger.used == used
    assert reader.failed_batch_reviews == failed and engine.mapper.sources == originals


def test_actual_repair_overflow_cannot_lower_repair_cap_or_remove_originals_to_force_a_call():
    reader, engine, state, calls, _ = adaptive_fixture(ExactAdmissionCounts(actual_repair=12500))
    calls.respond = lambda index, data: response(observation('invalid-source', ['not-visible']))
    originals = deepcopy(engine.mapper.sources)
    read_all(reader, engine)
    assert len(calls.requests) == 1 and calls.requests[0]['payload']['max_tokens'] == 2524
    assert engine.ledger.used['llm'] == 1 and engine.ledger.used['json_repairs'] == 0
    assert state.pending == {'history'} and state.valid()[0]['history'] == OLD
    held = {source for review in reader.failed_batch_reviews.values() for source in review['source_ids']}
    assert held == {source['source_id'] for source in originals.values()}
    denial = next(event for event in engine.events if event['event'] == 'joint_repair_input_unavailable')
    assert denial['input_tokens'] == 12500 and denial['output_token_reserve'] == 4096
    assert read_all(reader, engine) is False and len(calls.requests) == 1
    assert engine.mapper.sources == originals


def test_later_capacity_failure_retains_unread_suffix_and_already_accepted_history():
    reader, engine, state, calls, counter = adaptive_fixture(
        ExactAdmissionCounts({1: 10500, 2: 11000, 3: 11725}))
    originals = deepcopy(engine.mapper.sources)
    def first_window(index, data):
        assert index == 1
        assert {source['id'] for source in data['source_spans']} == {'s1', 's2'}
        counter.floor = 10000  # A later window is unavailable; no current source is removed.
        return response(observation('accepted-first-window', ['s1']))
    calls.respond = first_window
    with pytest.raises(InputOverflow):
        read_all(reader, engine)
    assert len(calls.requests) == 1 and calls.requests[0]['payload']['max_tokens'] == 1024
    assert state.valid()[0]['history'] == OLD and state.pending == {'history'}
    assert set(reader.unread_obligations) == {('adaptive-history', 's3')}
    assert engine.mapper.sources == originals
    assert not reader.failed_batch_reviews


def test_changed_payload_after_selection_rejects_before_generation_and_keeps_original_obligations(monkeypatch):
    reader, engine, _, calls, _ = adaptive_fixture()
    original_payload, original_prefix = reader._payload, reader._fitting_prefix
    selected = False
    def payload(*args, **kwargs):
        data = original_payload(*args, **kwargs)
        if selected:
            data['question'] = 'Changed after capacity selection'
        return data
    def prefix(*args, **kwargs):
        nonlocal selected
        result = original_prefix(*args, **kwargs)
        selected = True
        return result
    monkeypatch.setattr(reader, '_payload', payload)
    monkeypatch.setattr(reader, '_fitting_prefix', prefix)
    with pytest.raises(InputOverflow, match='does not match'):
        read_all(reader, engine)
    assert not calls.requests and not engine.reasoner.requests
    assert engine.ledger.used['llm'] == engine.ledger.used['json_repairs'] == 0
    assert set(reader.unread_obligations) == {('adaptive-history', alias) for alias in engine.mapper.sources}
