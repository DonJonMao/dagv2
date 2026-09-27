import json
from copy import deepcopy

import pytest

from dagbt.budget import Ledger, BudgetExceeded
from dagbt.reasoning import Reasoner, ProtocolError
from dagbt.row_recovery import recover_rows


class Calls:
    def __init__(self, values):
        self.values = iter(values)
        self.inputs = []

    def get(self, stage, url, payload):
        self.inputs.append(json.loads(payload['messages'][1]['content']))
        return {'response_ref': str(len(self.inputs)), 'response': {'choices': [
            {'message': {'content': json.dumps(next(self.values))}, 'finish_reason': 'stop'}]}}


def reasoner(values):
    calls = Calls(values)
    ledger = Ledger({'json_repairs': 6, 'llm': 24})
    obj = Reasoner(calls, None, {'llm_base_url': 'http://invalid/v1', 'model_profile': 'bridgetree', 'llm_model': 'fixture'},
                   {'context_tokens': 16000, 'reasoning_output_tokens': 1000,
                    'max_repairs_per_request': 2}, ledger, lambda x: None)
    return obj, calls


def validate(value):
    if value.get('answer') != 'fixed':
        raise ProtocolError('wrong conclusion')
    for row in value['rows']:
        if not isinstance(row, dict) or row.get('source') not in {'a', 'b'}:
            raise ProtocolError('invalid source')
    return deepcopy(value)


def test_valid_row_survives_failed_neighbor_and_repair_only_returns_replacement():
    obj, calls = reasoner([
        {'answer': 'fixed', 'rows': [{'source': 'a'}, {'source': 'bogus'}]},
        {'answer': 'fixed', 'rows': [{'source': 'b'}]},
    ])
    value, info = recover_rows(obj, 'resolve', 'S', {'evidence': ['a', 'b']}, validate, ('rows',))
    assert value['rows'] == [{'source': 'a'}, {'source': 'b'}]
    assert info['complete'] and info['repairs'] == 1
    assert calls.inputs[1]['local_repair']['accepted_row_counts'] == {'rows': 1}
    assert 'bogus' not in json.dumps(calls.inputs[1])
    assert obj.ledger.used['json_repairs'] == 1


@pytest.mark.parametrize('replacement', [[], [{'source': 'a'}], [{'source': 'bogus'}], ['bad'], [{'source': 'a', 'reason': 'rewritten'}]])
def test_failed_row_cannot_be_silently_omitted_or_replaced_by_already_accepted_row(replacement):
    obj, calls = reasoner([
        {'answer': 'fixed', 'rows': [{'source': 'a'}, {'source': 'bogus'}]},
        *[{'answer': 'fixed', 'rows': replacement} for _ in range(2)],
    ])
    value, info = recover_rows(obj, 'audit', 'S', {}, validate, ('rows',))
    assert value['rows'] == [{'source': 'a'}]
    assert not info['complete'] and info['failed_rows']
    assert len(calls.inputs) == 3


def test_repair_cannot_rewrite_the_accepted_conclusion():
    obj, calls = reasoner([
        {'answer': 'fixed', 'rows': [{'source': 'a'}, {'source': 'bogus'}]},
        {'answer': 'changed', 'rows': [{'source': 'b'}]},
        {'answer': 'fixed', 'rows': [{'source': 'b'}]},
    ])
    value, info = recover_rows(obj, 'resolve', 'S', {}, validate, ('rows',))
    assert value == {'answer': 'fixed', 'rows': [{'source': 'a'}, {'source': 'b'}]}
    assert info['complete'] and info['repairs'] == 2


def test_no_valid_header_fails_instead_of_fabricating_empty_success():
    obj, _ = reasoner([{'answer': 'bad', 'rows': []}] * 3)
    with pytest.raises(ProtocolError):
        recover_rows(obj, 'resolve', 'S', {}, validate, ('rows',))


def test_transport_budget_exhaustion_during_repair_keeps_valid_rows():
    obj, calls = reasoner([{'answer': 'fixed', 'rows': [{'source': 'a'}, {'source': 'bogus'}]}])
    base_get = calls.get
    def get(stage, url, payload):
        if calls.inputs:
            raise BudgetExceeded('llm', 'HTTP retry', 1, 0)
        return base_get(stage, url, payload)
    calls.get = get
    value, info = recover_rows(obj, 'audit', 'S', {}, validate, ('rows',))
    assert value['rows'] == [{'source': 'a'}]
    assert info['error_type'] == 'BudgetExceeded' and not info['complete']
