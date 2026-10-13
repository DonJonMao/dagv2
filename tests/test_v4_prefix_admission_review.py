"""Independent per-prefix selection boundaries; all counts and replies offline."""
from copy import deepcopy

import pytest

from dagbt.reasoning import InputOverflow
from test_v4_adaptive_initial_cap_review import (
    ExactAdmissionCounts, adaptive_fixture, choose, read_all, assert_unattempted_history_scope,
)


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Prefix admission review must never contact a service')
    monkeypatch.setattr('urllib.request.urlopen', forbidden)
    monkeypatch.setattr('urllib.request.build_opener', forbidden)


def selected_events(engine):
    return [event for event in engine.events if event['event'] == 'joint_fresh_output_cap_selected']


def scripted_search(monkeypatch, *, full=(), adaptive=None):
    """Isolate search ordering; real capacity and request checks are tested below."""
    reader, engine, _, calls, _ = adaptive_fixture()
    adaptive = adaptive or {}
    probes, payloads = [], {}
    reader._fresh_initial_selection = {'stale': True}
    events_before = len(selected_events(engine))

    def uninstalled():
        assert reader._fresh_initial_selection is None
        assert len(selected_events(engine)) == events_before

    def payload(step, query, variables, aliases, semantic, *, audit):
        uninstalled()
        data = {'question': query, 'aliases': list(aliases)}
        payloads[len(aliases)] = deepcopy(data)
        return data

    def full_fits(operation, data, *, fresh_sources):
        uninstalled()
        size = len(data['aliases'])
        probes.append(('full', size))
        return size in full

    def adaptive_fits(operation, data):
        uninstalled()
        size = len(data['aliases'])
        probes.append(('adaptive', size))
        if size not in adaptive:
            return None
        return {'output_token_cap': adaptive[size], 'structure_floor_tokens': 300,
                'whole_review_forecast_input_tokens': 12024 - adaptive[size]}

    monkeypatch.setattr(reader, '_payload', payload)
    monkeypatch.setattr(reader, '_fresh_window_fits', full_fits)
    monkeypatch.setattr(reader, '_adaptive_fresh_cap', adaptive_fits)
    groups = [[f's{i}'] for i in range(1, 9)]
    before, used = deepcopy(groups), deepcopy(engine.ledger.used)
    aliases, consumed = reader._fitting_prefix(
        'joint_read', engine.steps[0], 'Review history', ['history'], groups, False, False)
    assert groups == before and engine.ledger.used == used
    assert not calls.requests and not engine.reasoner.requests
    return reader, engine, aliases, consumed, probes, payloads


@pytest.mark.parametrize('full,adaptive,expected_cap', [
    ((), {4: 1500, 6: 1000}, 1000),
    ((4,), {6: 1000}, 1000),
    ((6,), {4: 1500}, None),
])
def test_mixed_binary_probes_install_only_the_winning_candidate(
        monkeypatch, full, adaptive, expected_cap):
    reader, engine, aliases, consumed, probes, payloads = scripted_search(
        monkeypatch, full=full, adaptive=adaptive)
    assert consumed == 6 and aliases == [f's{i}' for i in range(1, 7)]
    assert [size for kind, size in probes if kind == 'full'] == [8, 4, 6, 7]
    assert probes[-1] == ('adaptive', 7), 'The last probe must fail after the selected one'
    if expected_cap is None:
        assert reader._fresh_initial_selection is None and not selected_events(engine)
        assert ('adaptive', 6) not in probes
    else:
        selection = reader._fresh_initial_selection
        assert selection['output_token_cap'] == expected_cap
        assert selection['data_identity'] == reader._selection_data_identity(payloads[6])
        assert selection['whole_review_forecast_input_tokens'] == 12024 - expected_cap
        assert len(selected_events(engine)) == 1
        assert selected_events(engine)[0]['prefix_groups'] == 6
        assert selected_events(engine)[0]['output_token_cap'] == expected_cap


def test_nonmonotone_no_fit_binary_path_checks_untested_prefixes(monkeypatch):
    reader, engine, aliases, consumed, probes, _ = scripted_search(
        monkeypatch, adaptive={3: 1300, 7: 700})
    assert consumed == 3 and aliases == ['s1', 's2', 's3']
    assert [size for kind, size in probes if kind == 'full'] == [8, 4, 2, 1, 3]
    assert reader._fresh_initial_selection['output_token_cap'] == 1300
    assert len(selected_events(engine)) == 1
    # A checked prefix is promised, not the global maximum of nonmonotone packing.
    assert ('full', 7) not in probes


def test_no_candidate_clears_stale_selection_and_does_not_emit_selection(monkeypatch):
    reader, engine, aliases, consumed, probes, _ = scripted_search(monkeypatch)
    assert aliases == [] and consumed == 0
    assert reader._fresh_initial_selection is None and not selected_events(engine)
    assert sorted(size for kind, size in probes if kind == 'full') == list(range(1, 9))


def test_full_candidate_never_counts_floor_and_resets_a_previous_adaptive_choice():
    counter = ExactAdmissionCounts()
    reader, engine, _, calls, _ = adaptive_fixture(counter)
    assert choose(reader, engine)[1] == 3
    assert reader._fresh_initial_selection['output_token_cap'] == 2524
    event_count = len(selected_events(engine))
    counter.forecasts = {1: 7900, 2: 7900, 3: 7900}
    counter.records.clear()
    read_all(reader, engine)
    assert reader._fresh_initial_selection is None
    assert len(selected_events(engine)) == event_count
    assert not any(record['kind'] == 'floor' for record in counter.records)
    assert len(calls.requests) == 1 and calls.requests[0]['payload']['max_tokens'] == 4096


def test_structure_floor_rejection_falls_back_to_smaller_full_candidate():
    reader, engine, _, calls, counter = adaptive_fixture(
        ExactAdmissionCounts({1: 7900, 2: 11800, 3: 11725}, floor=300))
    aliases, consumed = choose(reader, engine)
    assert aliases == ['s1'] and consumed == 1
    assert any(record['kind'] == 'floor' for record in counter.records)
    assert reader._fresh_initial_selection is None and not selected_events(engine)
    assert not calls.requests and engine.ledger.used['llm'] == engine.ledger.used['json_repairs'] == 0


def test_missing_supported_baseline_cannot_authorize_adaptive_large_prefix():
    reader, engine, state, calls, counter = adaptive_fixture(
        ExactAdmissionCounts({1: 7900, 2: 9000, 3: 9500}))
    for route in state.routes.values():
        route['revoked'] = True
    assert 'history' not in state.valid()[0]
    aliases, consumed = choose(reader, engine)
    assert aliases == ['s1'] and consumed == 1
    assert reader._fresh_initial_selection is None and not selected_events(engine)
    assert not any(record['kind'] == 'floor' for record in counter.records)
    assert not calls.requests


@pytest.mark.parametrize('mode', ['generic', 'semantic', 'audit'])
def test_nonfresh_phase_keeps_full_output_admission(mode):
    reader, engine, _, calls, counter = adaptive_fixture()
    if mode == 'generic':
        engine.personal_response_selection = False
    aliases, consumed = reader._fitting_prefix(
        'joint_read', engine.steps[0], 'Review history', ['history'],
        [['s1'], ['s2'], ['s3']], mode == 'semantic', mode == 'audit')
    assert aliases == ['s1', 's2', 's3'] and consumed == 3
    assert reader._fresh_initial_selection is None and not selected_events(engine)
    assert not any(record['kind'] in {'floor', 'forecast'} for record in counter.records)
    assert not calls.requests


@pytest.mark.parametrize('discovery', [False, True])
def test_rebuilt_payload_order_change_rejects_before_generation(monkeypatch, discovery):
    reader, engine, _, calls, _ = adaptive_fixture()
    original_payload, original_prefix = reader._payload, reader._fitting_prefix
    chosen = False

    def payload(*args, **kwargs):
        data = original_payload(*args, **kwargs)
        if chosen:
            reordered = dict(reversed(list(data.items())))
            assert reordered == data
            assert reader._selection_data_identity(reordered) != reader._selection_data_identity(data)
            return reordered
        return data

    def prefix(*args, **kwargs):
        nonlocal chosen
        result = original_prefix(*args, **kwargs)
        chosen = True
        return result

    monkeypatch.setattr(reader, '_payload', payload)
    monkeypatch.setattr(reader, '_fitting_prefix', prefix)
    with pytest.raises(InputOverflow, match='does not match'):
        read_all(reader, engine, discovery=discovery)
    assert not calls.requests and not engine.reasoner.requests
    assert engine.ledger.used['llm'] == engine.ledger.used['json_repairs'] == 0
    assert_unattempted_history_scope(reader, engine, discovery)
