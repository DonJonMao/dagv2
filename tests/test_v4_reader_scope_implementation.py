"""Offline reader scope and repair-contract regressions; no service endpoints."""
from copy import deepcopy

import pytest

from dagbt.joint_reading import FROZEN_RESTATEMENT_ERROR, JointReader
from dagbt.reasoning import InputOverflow, ProtocolError
from test_personamem_joint_protocol import fixture, observation, response, visible
from test_v4_joint_reading import prepare_terminal, terminal_response
from test_v4_failed_batch_review import failed_whole, whole_review_response


def empty_slots(data):
    return {'slots': [{'variable': variable, 'observations': [], 'unresolved': []}
                      for variable in data['variables']]}


def mandatory(reader, *, identity='closure', audit=False, variables=None, docs=()):
    return reader.read(reader.e.steps[1], 'Assess current sourced history', variables or ['suggestion'],
                       docs, identity, semantic=not audit, audit=audit, terminal_output=not audit,
                       source_scope='mandatory_closure')


@pytest.mark.parametrize('kind', ['doc_ids', 'outside', 'uncovered_variable', 'foreign'])
def test_scope_rejection_preserves_reader_and_source_ownership_before_counting(kind, monkeypatch):
    reader, engine, state, calls = prepare_terminal()
    if kind == 'outside':
        reader.unread_obligations['closure', 's2'] = {'history'}
    elif kind == 'uncovered_variable':
        reader.unread_obligations['closure', 's1'] = {'history', 'not-requested'}
    elif kind == 'foreign':
        reader.unread_obligations['old-query', 's1'] = {'history'}
    reader.current_identity = 'old-current'
    reader.current_terminal_output = False
    before = deepcopy((reader.obligation_snapshot(), state.public_dict(), engine.mapper.sources,
                       reader.current_identity, reader.current_terminal_output, engine.ledger.used))
    monkeypatch.setattr(reader, '_fits', lambda *args: pytest.fail('Guard must precede counting'))
    with pytest.raises(ProtocolError):
        mandatory(reader, docs=['piano'] if kind == 'doc_ids' else ())
    assert before == (reader.obligation_snapshot(), state.public_dict(), engine.mapper.sources,
                      reader.current_identity, reader.current_terminal_output, engine.ledger.used)
    assert calls.requests == [] and engine.events == []


def test_unbound_nonterminal_compose_still_pays_for_one_real_request(monkeypatch):
    reader, engine, state, calls = fixture()
    engine.steps[0]['execution'] = 'compose'
    calls.respond = lambda index, data: empty_slots(data)
    monkeypatch.setattr(reader, '_fitting_prefix', lambda *args: pytest.fail('No prefix search in closure mode'))
    reader.read(engine.steps[0], 'Combine scoped history', ['history'], [], 'middle',
                source_scope='mandatory_closure')
    assert len(calls.requests) == engine.ledger.used['llm'] == 1
    assert calls.requests[0]['stage'][0] == 'joint_read'
    assert calls.requests[0]['data']['source_spans'] == []
    assert reader.batches[-1]['complete'] is True


def test_whole_closure_capacity_failure_keeps_every_owned_marker_without_generation(monkeypatch):
    reader, engine, state, calls = prepare_terminal()
    reader.unread_obligations['closure', 's1'] = {'history'}
    reader.unread_obligations['foreign-independent', 's3'] = {'independent'}
    state.pending = {'history', 'independent'}
    reader._fresh_initial_selection = {'output_token_cap': 1}
    monkeypatch.setattr(reader, '_fits', lambda *args: False)
    monkeypatch.setattr(reader, '_fitting_prefix', lambda *args: pytest.fail('No prefix fallback'))
    with pytest.raises(InputOverflow):
        mandatory(reader)
    assert reader.unread_obligations['closure', 's1'] == {'history'}
    assert reader.unread_obligations['foreign-independent', 's3'] == {'independent'}
    assert reader.batches[-1]['unread_source_aliases'] == [['s1']]
    assert reader._fresh_initial_selection is None
    assert len(calls.requests) == engine.ledger.used['llm'] == engine.ledger.used['json_repairs'] == 0
    assert state.pending == {'history', 'independent'}


def test_complete_closure_settles_only_its_owned_same_query_marker(monkeypatch):
    reader, engine, state, calls = prepare_terminal(personal=True)
    reader.unread_obligations['closure', 's1'] = {'history'}
    reader.unread_obligations['different-independent', 's1'] = {'independent'}
    state.pending = {'history', 'independent'}
    calls.respond = lambda index, data: terminal_response(data, engine.output_options[0])
    monkeypatch.setattr(reader, '_fitting_prefix', lambda *args: pytest.fail('Whole closure is one batch'))
    mandatory(reader)
    assert len(calls.requests) == 1
    assert calls.requests[0]['payload']['max_tokens'] == engine.s['reasoning_output_tokens']
    assert reader.batches[-1]['source_aliases'] == ['s1']
    assert reader.unread_obligations == {('different-independent', 's1'): {'independent'}}
    assert state.pending == {'independent'}


def test_frozen_ordinary_restatement_is_slot_failure_then_explicit_ref_can_recover():
    reader, engine, state, calls = fixture(max_repairs_per_request=2)
    kept = observation('kept')
    bad = observation('fix-me', ['invisible'])

    def answer(index, data):
        if index == 1:
            return response(kept, bad)
        if index == 2:
            # This changed ordinary row must neither bind its value nor gain
            # coverage by silently borrowing the old row with the same ID.
            return response(observation('kept', ['s1'], value=['forged']), observation('fix-me'))
        failure = next(item for item in data['repair_scope']['failed'] if item['id'] == '*')
        assert failure['feedback'] == FROZEN_RESTATEMENT_ERROR
        assert 'obligation_review_sources' not in failure
        assert 'frozen_observation_reviews' in failure
        result = response()
        result['slots'][0]['reviewed_observation_ids'] = ['kept']
        return result

    calls.respond = answer
    reader.read(engine.steps[0], 'Read history', ['history'], ['piano'], 'history')
    assert len(calls.requests) == 3
    assert state.valid()[0]['history'] == ['piano'] and not state.pending
    assert reader.batches[-1]['complete']


def test_named_new_id_becomes_required_failed_draft_on_following_repair():
    reader, engine, state, calls = fixture(max_repairs_per_request=2)

    def answer(index, data):
        if index == 1:
            return response(observation('bad', ['invisible']))
        if index == 2:
            return response(observation('bad'), observation('new-id', ['s1']))
        failure = next(item for item in data['repair_scope']['failed'] if item['id'] == 'new-id')
        assert 'Preserve and correct it' in failure['feedback']
        return response(observation('new-id'))

    calls.respond = answer
    reader.read(engine.steps[0], 'Read history', ['history'], ['piano'], 'history')
    assert len(calls.requests) == 3 and not state.pending
    assert reader.batches[-1]['complete']


def test_paid_failed_batch_local_repair_omits_frozen_draft_and_still_recovers():
    reader, engine, state, calls = failed_whole(max_repairs=2)
    before_calls = len(calls.requests)
    before_repairs = engine.ledger.used['json_repairs']
    review_ids = []

    def answer(index, data):
        offered = [item['id'] for item in data['failed_batch_reviews'] if item['review_allowed']]
        assert len(offered) == 1
        review_ids.append(offered)
        if len(review_ids) == 1:
            return whole_review_response(data, drop_source=True)
        assert review_ids[0] == offered
        required = {entry['id'] for entry in data['repair_scope']['failed'] if entry['id'] != '*'}
        result = whole_review_response(data)
        result['slots'][0]['observations'] = [row for row in result['slots'][0]['observations']
                                               if row['id'] in required]
        return result

    calls.respond = answer
    reader.read(engine.steps[0], 'Review historical activities', ['history'],
                ['advice'], 'paid-recovery-with-scoped-repair', review_existing_obligations=True)
    assert len(calls.requests) == before_calls + 2
    assert engine.ledger.used['json_repairs'] == before_repairs + 2
    assert state.valid()[0]['history'] == ['piano', 'painting']
    assert not state.pending and not reader.failed_obligations


@pytest.mark.parametrize('personal', [False, True])
@pytest.mark.parametrize('repair', [False, True])
def test_only_actual_audit_payload_requires_explicit_three_state_verdict(personal, repair):
    reader, engine, state, _ = fixture()
    engine.personal_response_selection = personal
    data = reader._payload(engine.steps[0], 'Check history', ['history'], ['s1'], audit=True)
    if repair:
        data['repair_scope'] = {'failed': [{'variable': 'history', 'id': '*'}]}
    schema = reader._schema(data)
    assert schema['required'] == ['slots', 'audit_verdict']
    assert schema['properties']['audit_verdict']['enum'] == ['supported', 'unknown', 'conflict']
    ordinary = reader._payload(engine.steps[0], 'Read history', ['history'], ['s1'])
    assert reader._schema(ordinary)['required'] == ['slots']
