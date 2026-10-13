"""Independent fixed-budget window admission and original-source preservation.

The large history is synthetic, not a replay that moves formerly invisible
PersonaMem citations into a different reading window.
"""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from dagbt.evidence_mapping import EvidenceMapper
from test_personamem_joint_protocol import observation, response
from test_v4_reconcile_review import personal_reader


def large_personal_history():
    reader, engine, state, calls = personal_reader(
        context_tokens=16384, reasoning_output_tokens=4096, max_repairs_per_request=1)
    engine.docs = {}
    for number in range(64):
        activity = 'piano' if number == 0 else 'painting' if number == 1 else 'listening to music'
        text = ('I enjoyed ' + activity + ' during this recorded period. ') * 6
        name = 'memory-' + str(number)
        engine.docs[name] = SimpleNamespace(doc_id=name, passage=text, metadata={
            'source_segments': [{'role': 'user', 'start': 0, 'end': len(text),
                                 'source_message_indices': [number]}]})
    engine.candidates = []
    engine.chunks = []
    engine.mapper = EvidenceMapper(engine)
    engine.mapper.add_candidates(list(engine.docs))
    reader.current_identity = 'headroom-history'
    return reader, engine, state, calls


def test_smaller_initial_window_admits_real_whole_repair_without_dropping_later_sources():
    reader, engine, state, calls = large_personal_history()
    aliases = list(engine.mapper.sources)
    groups = [[alias] for alias in aliases]
    chosen, consumed = reader._fitting_prefix(
        'joint_read', engine.steps[0], 'Review activities', ['history'], groups, False, False)
    assert 2 <= consumed < len(groups)
    admitted = reader._payload(engine.steps[0], 'Review activities', ['history'], chosen)
    assert reader._fresh_window_fits('joint_read', admitted)
    richer = None
    for size in range(consumed + 1, len(groups) + 1):
        data = reader._payload(engine.steps[0], 'Review activities', ['history'], aliases[:size])
        if reader._fits('joint_read', data) and not reader._fresh_window_fits('joint_read', data):
            richer = data
            break
    assert richer is not None, 'Fresh-window admission must reserve more than the old null-draft envelope'

    # These are two real sources in the admitted prefix; no recorded external
    # draft is rewritten to conceal a citation that fell into unread work.
    cited = aliases[:2]
    assert set(cited) <= set(chosen)
    original_sources = deepcopy(engine.mapper.sources)
    fixed_settings = deepcopy(engine.s)
    engine.ledger.limits['llm'] = 3  # Two normal calls and one reserved audit.
    def answer(index, data):
        shown = {source['id'] for source in data['source_spans']}
        assert set(cited) <= shown
        if index == 1:
            assert len(shown) < len(original_sources)
            return response(observation('piano-draft', [cited[0]], value=['piano']),
                            observation('painting-draft', [cited[1]], value=['painting']))
        assert index == 2 and data['repair_scope']['whole_value_reviews']
        assert shown == {source['id'] for source in calls.requests[0]['data']['source_spans']}
        return response(*[observation(identifier, cited, value=['piano', 'painting'])
                          for identifier in ('piano-draft', 'painting-draft')])
    calls.respond = answer
    reader.read(engine.steps[0], 'Review activities', ['history'], list(engine.docs), 'headroom-history')
    assert len(calls.requests) == 2 and engine.ledger.used['json_repairs'] == 1
    assert engine.s == fixed_settings and engine.mapper.sources == original_sources
    assert state.valid()[0]['history'] == ['piano', 'painting']
    assert reader.batches[0]['complete'] and reader.batches[0]['valid_rows'] == 2
    assert reader.unread_obligations and state.pending == {'history'}
    for request in calls.requests:
        data = request['data']
        operation = request['stage'][0]
        count = reader._estimate(operation, data)
        assert count + 4096 + engine.s['input_margin'] + 8 <= 16384


@pytest.mark.parametrize('case', [
    'no_fresh_sources', 'generic', 'terminal', 'audit', 'history_compose', 'existing_local_repair',
    'durable_recovery', 'no_request_repairs', 'no_global_repairs', 'only_one_normal_call',
])
def test_unknown_draft_headroom_does_not_reject_known_or_nonrepairable_requests(case, monkeypatch):
    reader, engine, _, _ = personal_reader(max_repairs_per_request=1)
    reader.current_identity = 'known-request'
    data = reader._payload(engine.steps[0], 'Review history', ['history'], ['s1'])
    fresh = True
    if case == 'no_fresh_sources':
        fresh = False
    elif case == 'generic':
        data['personal_response_selection'] = False
    elif case == 'terminal':
        data['phase'] = 'response_selection'
    elif case == 'audit':
        data['phase'] = 'response_audit'
    elif case == 'history_compose':
        data['task']['execution'] = 'compose'
    elif case == 'existing_local_repair':
        data['repair_scope'] = {'failed': [{'variable': 'history', 'id': 'original'}]}
    elif case == 'durable_recovery':
        data['failed_batch_reviews'] = [{'id': 'original-batch', 'variable': 'history'}]
    elif case == 'no_request_repairs':
        engine.s['max_repairs_per_request'] = 0
    elif case == 'no_global_repairs':
        engine.ledger.limits['json_repairs'] = 0
    else:
        engine.ledger.limits['llm'] = engine.s['reserved_audit_calls'] + 1
    monkeypatch.setattr(reader, '_fits', lambda operation, payload: True)
    def no_extra_forecast(*args, **kwargs):
        raise AssertionError('A request already admitted by actual wire capacity must not receive this fresh-draft forecast')
    monkeypatch.setattr(reader, '_estimate', no_extra_forecast)
    assert reader._fresh_window_fits('joint_read', data, fresh_sources=fresh)
