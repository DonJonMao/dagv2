"""v3 repair scope, true budget causes and hard global-failure boundaries."""
from copy import deepcopy
import json
import urllib.error

import pytest

from dagbt.budget import BudgetExceeded, Ledger
from dagbt.reasoning import InputOverflow, OutputTruncated, ProtocolError, Reasoner, RefusalError
from dagbt.row_recovery import recover_rows
from dagbt.transport import ServiceError, Transport, call_reservation
from dagbt.model_runtime import BTTokenAccounting
from test_reasoning_v2 import reasoner, response, settings, profile


def body(rows, **extra):
    return response(json.dumps({'answer': 'fixed', 'rows': rows, **extra}))


def validate(value):
    if value.get('answer') != 'fixed':
        raise ProtocolError('Invalid conclusion header')
    for key in ('rows', 'coverage'):
        for row in value.get(key, []):
            if not isinstance(row, dict) or row.get('source') not in {'a', 'b'}:
                raise ProtocolError('Bad evidence citation')
    return deepcopy(value)


def header(value):
    if value != {'answer': 'fixed'}:
        raise ProtocolError('Header is not the frozen legal conclusion')


def near_limit(r, data):
    count = r.estimate('resolve', 'Resolve.', data)
    r.settings['context_tokens'] = count + r.settings['reasoning_output_tokens'] + 8 + r.settings['input_margin']


def test_reasoner_reports_real_repair_overflow_instead_of_original_schema_error():
    r, calls, ledger, events = reasoner([response('{"unfinished":')])
    data = {'raw': '中' * 2000}
    near_limit(r, data)
    with pytest.raises(InputOverflow) as error:
        r.json('resolve', 'Resolve.', data, lambda value: value)
    assert isinstance(error.value.__cause__, ProtocolError)
    assert not ledger.used['json_repairs'] and len(calls.requests) == 1
    failed = [e for e in events if e['event'] == 'reasoning_repair_preflight_failed'][-1]
    assert failed['failure_category'] == 'repair_input_budget'
    assert failed['error_type'] == 'InputOverflow' and 'exceeds' in failed['error']


def test_reasoner_accepts_caller_scoped_payload_without_repeating_full_raw_view():
    r, calls, ledger, _ = reasoner([response('broken'), response('{}')])
    data = {'raw': '中' * 2000, 'query': 'Q', 'allowed_ids': ['e1']}
    near_limit(r, data)
    states = []
    def compact(state):
        states.append(deepcopy(state))
        return {key: state['original_data'][key] for key in ('query', 'allowed_ids')}
    assert r.json('resolve', 'Resolve.', data, lambda value: value, repair_builder=compact) == {}
    second = json.loads(calls.requests[1][2]['messages'][1]['content'])
    assert 'raw' not in second and second['allowed_ids'] == ['e1']
    assert states[0]['failure_category'] == 'json_syntax'
    assert data['raw'] == '中' * 2000 and ledger.used['json_repairs'] == 1


def test_full_row_repair_overflow_retains_valid_rows_and_true_terminal_cause():
    r, calls, ledger, events = reasoner([body([{'source': 'a'}, {'source': 'bad'}])])
    data = {'raw_memories': '中' * 2000, 'allowed_ids': ['a', 'b']}
    near_limit(r, data)
    result, info = recover_rows(r, 'resolve', 'Resolve.', data, validate, ('rows',), validate_header=header)
    assert result['rows'] == [{'source': 'a'}] and not info['complete']
    assert info['error_type'] == 'InputOverflow' and info['failure_category'] == 'input_budget'
    assert info['failure_scope'] == 'row_budget' and 'exceeds' in info['error']
    assert info['failed_rows'][-1]['field'] == 'repair_input'
    assert info['recovery_actions'][-1]['action'] == 'repair_preflight_failed'
    assert events[-1]['error_type'] == 'InputOverflow'
    assert len(calls.requests) == 1 and not ledger.used['json_repairs']


def test_scoped_row_repair_passes_full_view_overflow_case_and_freezes_retained_units():
    r, calls, ledger, _ = reasoner([body([{'source': 'a'}, {'source': 'bad'}]), body([{'source': 'b'}])])
    data = {'raw_memories': '中' * 2000, 'allowed_ids': ['a', 'b'], 'query': 'Q'}
    near_limit(r, data)
    captured = []
    def compact(state):
        captured.append(deepcopy(state))
        return {'query': state['original_data']['query'], 'allowed_ids': ['a', 'b']}
    result, info = recover_rows(r, 'resolve', 'Resolve.', data, validate, ('rows',),
                                validate_header=header, repair_builder=compact)
    assert result['rows'] == [{'source': 'a'}, {'source': 'b'}] and info['complete']
    assert captured[0]['pending_fields'] == ['rows']
    assert captured[0]['pending_row_counts'] == {'rows': 1}
    assert captured[0]['retained_rows'] == {'rows': [{'source': 'a'}]}
    assert captured[0]['fixed_header'] == {'answer': 'fixed'}
    second = json.loads(calls.requests[1][2]['messages'][1]['content'])
    assert 'raw_memories' not in second and 'bad' not in json.dumps(second)
    assert info['recovery_actions'][0]['scoped_builder'] is True and ledger.used['json_repairs'] == 1


@pytest.mark.parametrize('second', [
    body([], coverage='bad type'),
    body([]),
])
def test_failed_secondary_field_cannot_be_completed_by_omission_or_nonlist(second):
    r, _, _, _ = reasoner([body([{'source': 'a'}], coverage='bad type'), second], max_repairs_per_request=1)
    result, info = recover_rows(r, 'resolve', 'Resolve.', {}, validate, ('rows', 'coverage'), validate_header=header)
    assert result['rows'] == [{'source': 'a'}] and result['coverage'] == []
    assert not info['complete'] and info['failed_fields'] == ['coverage']
    assert info['pending_fields'] == ['coverage']


def test_known_failed_secondary_row_cannot_be_silently_omitted():
    r, _, _, _ = reasoner([body([{'source': 'a'}], coverage=[{'source': 'bad'}]), body([])], max_repairs_per_request=1)
    _, info = recover_rows(r, 'resolve', 'Resolve.', {}, validate, ('rows', 'coverage'), validate_header=header)
    assert not info['complete'] and info['pending_row_counts']['coverage'] == 1


@pytest.mark.parametrize('failed,error', [
    (response('not JSON'), ProtocolError),
    (response('', finish='length'), OutputTruncated),
    (response('{}', refusal='service refusal'), RefusalError),
    (response(json.dumps({'answer': 'changed', 'rows': [{'source': 'b'}]})), ProtocolError),
])
def test_global_failures_after_valid_rows_are_never_partial_success(failed, error):
    r, _, _, events = reasoner([body([{'source': 'a'}, {'source': 'bad'}]), failed], max_repairs_per_request=1)
    with pytest.raises(error):
        recover_rows(r, 'resolve', 'Resolve.', {}, validate, ('rows',), validate_header=header)
    final = events[-1]
    assert final['event'] == 'local_rows_complete' and not final['complete']
    assert final['retained_counts']['rows'] == 1
    assert final['failure_scope'] in {'global_response', 'header_or_aggregate'}


def test_service_failure_preserves_partial_diagnostic_but_propagates():
    r, calls, _, events = reasoner([body([{'source': 'a'}, {'source': 'bad'}])])
    base = calls.get
    def fail(*args, **kwargs):
        if calls.requests:
            raise ServiceError('HTTP failure')
        return base(*args, **kwargs)
    calls.get = fail
    with pytest.raises(ServiceError, match='HTTP failure'):
        recover_rows(r, 'resolve', 'Resolve.', {}, validate, ('rows',), validate_header=header)
    assert events[-1]['retained_counts'] == {'rows': 1}
    assert events[-1]['failure_category'] == 'service_or_backend'


def test_budget_after_global_parse_failure_does_not_reuse_earlier_valid_rows():
    r, calls, _, events = reasoner([body([{'source': 'a'}, {'source': 'bad'}]), response('not JSON')])
    base = calls.get
    def fail(*args, **kwargs):
        if len(calls.requests) == 2:
            raise BudgetExceeded('llm', 'physical retry', 1, 0)
        return base(*args, **kwargs)
    calls.get = fail
    with pytest.raises(BudgetExceeded):
        recover_rows(r, 'resolve', 'Resolve.', {}, validate, ('rows',), validate_header=header)
    assert events[-1]['failure_scope'] == 'global_budget'


def test_header_validator_cannot_be_bypassed_by_permissive_row_validator():
    r, _, _, _ = reasoner([response(json.dumps({'answer': 'bad', 'rows': []}))], max_repairs_per_request=0)
    with pytest.raises(ProtocolError, match='frozen legal conclusion'):
        recover_rows(r, 'resolve', 'Resolve.', {}, lambda value: value, ('rows',), validate_header=header)


@pytest.mark.parametrize('api', ['json', 'rows'])
def test_nonselection_repairs_cannot_consume_reserved_final_selection_allowance(api):
    replies = [response('bad')] if api == 'json' else [body([{'source': 'a'}, {'source': 'bad'}])]
    r, calls, ledger, _ = reasoner(replies, reserved_selection_repairs=2)
    ledger.limits['json_repairs'] = 2
    if api == 'json':
        with pytest.raises(ProtocolError):
            r.json('resolve', 'Resolve.', {}, lambda value: value)
    else:
        _, info = recover_rows(r, 'audit', 'Resolve.', {}, validate, ('rows',), validate_header=header)
        assert not info['complete'] and info['reserved_repairs'] == 2
    assert len(calls.requests) == 1 and not ledger.used['json_repairs']


def test_final_selection_can_use_reserved_repair_allowance():
    r, _, ledger, _ = reasoner([response('bad'), response('{}')], reserved_selection_repairs=2)
    ledger.limits['json_repairs'] = 2
    assert r.json('select_final', 'Select.', {}, lambda value: value) == {}
    assert ledger.used['json_repairs'] == 1


def test_final_call_reservation_applies_even_to_dependency_arm_and_audits():
    s = {'selection': 'dependency', 'reserved_audit_calls': 1, 'final_selection_calls': 3}
    assert call_reservation(s, 'map/1') == 4
    assert call_reservation(s, 'resolve/1') == 4
    assert call_reservation(s, 'audit/1') == 3
    assert call_reservation(s, 'select_coverage_repair/1') == 0
    assert call_reservation(s, 'reader/20') == 0
    assert call_reservation(s, 'map/1', reserve=0, extra_reserve=5) == 5


def test_final_reservation_is_respected_inside_each_transport_retry(tmp_path, monkeypatch):
    s = settings(final_selection_calls=3)
    config = profile(s)
    ledger = Ledger({'llm': 5, 'json_repairs': 6})
    attempts = []
    def offline(*args, **kwargs):
        attempts.append(1)
        raise urllib.error.URLError('scripted failure')
    monkeypatch.setattr('dagbt.transport.urllib.request.urlopen', offline)
    monkeypatch.setattr('dagbt.transport.time.sleep', lambda _: None)
    transport = Transport('q', tmp_path, config, ledger, BTTokenAccounting())
    r = Reasoner(transport, transport.tokenizer, config, s, ledger, lambda _: None)
    with pytest.raises(BudgetExceeded):
        r.request('resolve', 'Resolve.', {})
    assert len(attempts) == ledger.used['llm'] == 1 and ledger.remaining('llm') == 4
