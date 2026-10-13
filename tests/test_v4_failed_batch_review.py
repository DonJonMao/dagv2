"""Independent recovery checks for failed drafts, source text and counters.

Scripted model replies run through the production reader and budget ledger.
Capacity cases use the real request serializer/token estimate boundary.
"""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from dagbt.evidence_spans import build_source_spans
from dagbt.joint_reading import JointReader
from test_personamem_joint_protocol import observation, response, visible
from test_v4_joint_reading import capacity_fixture
from test_v4_reconcile_review import fragments, personal_reader


def failed_whole(max_repairs=0):
    reader, engine, state, calls = personal_reader(max_repairs_per_request=max_repairs)
    calls.respond = lambda index, data: response(*fragments())
    reader.read(engine.steps[0], 'Review historical activities', ['history'],
                ['piano', 'painting'], 'original-failed-window')
    assert state.pending == {'history'} and reader.failed_obligations
    return reader, engine, state, calls


def failed_capacity(max_repairs=0):
    reader, engine, state, calls, aliases, _ = capacity_fixture()
    engine.s['max_repairs_per_request'] = max_repairs
    calls.respond = lambda index, data: response(observation('broad-proof', aliases))
    reader.read(engine.steps[0], 'Read past creative activities', ['history'],
                list(engine.docs), 'original-capacity-window')
    assert state.pending == {'history'} and reader.failed_obligations
    assert any(item['error'].startswith('citation_capacity:')
               for item in reader.batches[-1]['failed_slots'])
    return reader, engine, state, calls, aliases


def restored_reader(reader, engine, state):
    # The actual JSON boundary must carry the drafts; sharing a live Python
    # reference to the original reader would hide checkpoint data loss.
    checkpoint = json.loads(json.dumps(reader.obligation_snapshot()))
    fresh = JointReader(engine, state)
    fresh.restore_obligations(checkpoint)
    fresh.current_identity = 'different-identity-after-checkpoint'
    return fresh, checkpoint


def test_failed_whole_checkpoint_preserves_original_draft_identity_and_original_text():
    reader, engine, state, _ = failed_whole()
    fresh, checkpoint = restored_reader(reader, engine, state)
    encoded = json.dumps(checkpoint)
    for draft in fragments():
        assert draft['id'] in encoded, 'A failed variable name is not a recoverable original draft'
    payload = fresh._payload(engine.steps[0], 'Review history', ['history'], [])
    shown = {source['id']: source['text'] for source in payload['source_spans']}
    assert shown == {alias: engine.mapper.sources[alias]['text'] for alias in ('s1', 's2')}
    assert fresh.failed_obligations and fresh.pending_variables() == {'history'}


def test_real_capacity_failure_keeps_rejected_draft_and_all_original_text_after_checkpoint():
    reader, engine, state, _, aliases = failed_capacity()
    fresh, checkpoint = restored_reader(reader, engine, state)
    assert 'broad-proof' in json.dumps(checkpoint)
    payload = fresh._payload(engine.steps[0], 'Read past creative activities', ['history'], [])
    shown = {source['id']: source['text'] for source in payload['source_spans']}
    assert shown == {alias: engine.mapper.sources[alias]['text'] for alias in aliases}
    assert state.valid()[0] == {} and state.pending == {'history'}


def test_invalid_counter_draft_and_its_original_text_survive_new_identity_and_checkpoint():
    reader, engine, state, calls = personal_reader(max_repairs_per_request=0)
    state.integrate([reader._observation(observation('established', ['s1']),
                                        'history', visible(reader), engine.steps[0])])
    old_route = next(iter(state.routes))
    counter = observation('unresolved-counter', ['s2'], stance='contradiction',
                          entity_scope='mistyped-user', retract_ids=[old_route])
    counter['value'] = None
    calls.respond = lambda index, data: response(deepcopy(counter))
    reader.read(engine.steps[0], 'Recheck older history', ['history'],
                ['painting'], 'invalid-counter-window')
    fresh, checkpoint = restored_reader(reader, engine, state)
    assert 'unresolved-counter' in json.dumps(checkpoint)
    payload = fresh._payload(engine.steps[0], 'Review failed counter', ['history'], [])
    assert {'s1', 's2'} <= {source['id'] for source in payload['source_spans']}
    assert state.valid()[0]['history'] == ['piano'] and not state.routes[old_route]['revoked']
    assert fresh.pending_variables() == {'history'}


def whole_review_response(data, *, resolve_failed=True, omit_draft=False, drop_source=False):
    rows = [observation(row['id'], ['s1', 's2'], value=['piano', 'painting'])
            for row in fragments()]
    if omit_draft:
        rows.pop()
    if drop_source:
        rows[-1]['source_ids'] = ['s1']
    result = response(*rows)
    slot = result['slots'][0]
    slot['resolved_obligation_ids'] = [item['id'] for item in data.get('evidence_obligations', [])]
    if resolve_failed:
        slot['resolved_failed_batch_ids'] = [item['id'] for item in data.get('failed_batch_reviews', [])
                                             if item.get('review_allowed')]
    return result


def test_new_original_evidence_allows_budgeted_explicit_whole_draft_recovery():
    reader, engine, state, calls = failed_whole(max_repairs=1)
    assert len(calls.requests) == 2 and engine.ledger.used['json_repairs'] == 1
    def review(index, data):
        records = data['failed_batch_reviews']
        assert len([item for item in records if item.get('review_allowed')]) == 1
        assert {'s1', 's2', 's3'} <= {source['id'] for source in data['source_spans']}
        return whole_review_response(data)
    calls.respond = review
    reader.read(engine.steps[0], 'Review historical activities', ['history'],
                ['advice'], 'new-evidence-review', review_existing_obligations=True)
    assert len(calls.requests) == 3 and engine.ledger.used['json_repairs'] == 2
    assert state.valid()[0]['history'] == ['piano', 'painting']
    assert not state.pending and not reader.pending_variables()
    assert not reader.failed_obligations and not reader.failed_batch_reviews
    assert not reader.evidence_obligations


@pytest.mark.parametrize('restored', [False, True])
def test_query_rename_or_checkpoint_without_new_original_evidence_cannot_restart_failed_repair(restored):
    reader, engine, state, calls = failed_whole(max_repairs=1)
    if restored:
        reader, _ = restored_reader(reader, engine, state)
    before_requests, before_budget = len(calls.requests), deepcopy(engine.ledger.used)
    def forbidden(index, data):
        raise AssertionError('Changing identity or restoring state is not new original evidence')
    calls.respond = forbidden
    reader.read(engine.steps[0], 'Review historical activities', ['history'],
                [], 'renamed-query', review_existing_obligations=True)
    assert len(calls.requests) == before_requests and engine.ledger.used == before_budget
    assert state.pending == {'history'} and reader.failed_obligations


@pytest.mark.parametrize('attempt', ['omitted_failed_id', 'omitted_draft', 'dropped_source'])
def test_failed_whole_cannot_be_cleared_by_unacknowledged_or_incomplete_review(attempt):
    reader, engine, state, calls = failed_whole(max_repairs=1)
    old_markers = deepcopy(reader.failed_obligations)
    calls.respond = lambda index, data: whole_review_response(
        data, resolve_failed=attempt != 'omitted_failed_id',
        omit_draft=attempt == 'omitted_draft', drop_source=attempt == 'dropped_source')
    reader.read(engine.steps[0], 'Review historical activities', ['history'],
                ['advice'], 'incomplete-review', review_existing_obligations=True)
    assert state.pending == {'history'}
    for key, variables in old_markers.items():
        assert variables <= reader.failed_obligations.get(key, set())
    reader.current_identity = 'still-pending-later'
    payload = reader._payload(engine.steps[0], 'Review old issue', ['history'], [])
    assert {'s1', 's2'} <= {source['id'] for source in payload['source_spans']}


@pytest.mark.parametrize('budget', ['request_repairs', 'global_repairs', 'llm_audit_reserve'])
def test_failed_recovery_cannot_bypass_exhausted_repair_or_normal_llm_budget(budget):
    reader, engine, state, calls = failed_whole(max_repairs=0 if budget == 'request_repairs' else 1)
    if budget == 'global_repairs':
        engine.ledger.limits['json_repairs'] = engine.ledger.used['json_repairs']
    elif budget == 'llm_audit_reserve':
        engine.ledger.limits['llm'] = engine.ledger.used['llm'] + engine.s['reserved_audit_calls']
    before = deepcopy(engine.ledger.used)
    def answer(index, data):
        assert not any(item.get('review_allowed') for item in data.get('failed_batch_reviews', []))
        return response()
    calls.respond = answer
    reader.read(engine.steps[0], 'Review historical activities', ['history'],
                ['advice'], 'new-evidence-without-budget', review_existing_obligations=True)
    assert state.pending == {'history'} and reader.failed_obligations
    assert engine.ledger.used['json_repairs'] == before['json_repairs']
    if budget == 'llm_audit_reserve':
        assert engine.ledger.used['llm'] == before['llm']


def counter_failure(*, valid_counter_invalid_envelope=False):
    reader, engine, state, calls = personal_reader(max_repairs_per_request=1)
    state.integrate([reader._observation(observation('prior', ['s1']), 'history',
                                        visible(reader), engine.steps[0])])
    old_route = next(iter(state.routes))
    counter = observation('counter-draft', ['s2'], stance='contradiction',
                          retract_ids=[old_route],
                          entity_scope='user' if valid_counter_invalid_envelope else 'mistyped-user')
    counter['value'] = None
    def answer(index, data):
        result = response(deepcopy(counter))
        if valid_counter_invalid_envelope:
            result['final_prediction'] = 'a'
        return result
    calls.respond = answer
    reader.read(engine.steps[0], 'Read negative historical evidence', ['history'],
                ['painting'], 'counter-failure')
    assert len(calls.requests) == 2 and reader.failed_obligations
    assert not state.routes[old_route]['revoked']
    return reader, engine, state, calls, old_route, counter


def test_later_explicit_correction_of_failed_counter_retracts_the_actual_old_route():
    reader, engine, state, calls, old_route, counter = counter_failure()
    reader, _ = restored_reader(reader, engine, state)
    def repair(index, data):
        offered = [item for item in data['failed_batch_reviews'] if item.get('review_allowed')]
        assert offered and 'counter-draft' in json.dumps(offered)
        assert {'s1', 's2', 's3'} <= {source['id'] for source in data['source_spans']}
        corrected = {**deepcopy(counter), 'entity_scope': 'user'}
        result = response(corrected)
        result['slots'][0]['resolved_failed_batch_ids'] = [item['id'] for item in offered]
        return result
    calls.respond = repair
    reader.read(engine.steps[0], 'Review counter with new original text', ['history'],
                ['advice'], 'new-counter-review', review_existing_obligations=True)
    assert state.routes[old_route]['revoked'] and 'history' not in state.valid()[0]
    assert not state.pending and not reader.pending_variables()
    assert not reader.failed_obligations and not reader.failed_batch_reviews
    assert engine.ledger.used['json_repairs'] == 2 and len(calls.requests) == 3


@pytest.mark.parametrize('discard', [False, True])
def test_valid_counter_inside_bad_envelope_cannot_be_reclassified_on_cross_batch_review(discard):
    reader, engine, state, calls, old_route, counter = counter_failure(valid_counter_invalid_envelope=True)
    def review(index, data):
        repaired = deepcopy(counter)
        if discard:
            repaired.update(stance='irrelevant', retract_ids=[],
                            reason='Claiming this original negative clue is irrelevant without retaining the counter.')
        result = response(repaired)
        result['slots'][0]['resolved_failed_batch_ids'] = [
            item['id'] for item in data.get('failed_batch_reviews', []) if item.get('review_allowed')]
        return result
    calls.respond = review
    reader.read(engine.steps[0], 'Review counter with new original text', ['history'],
                ['advice'], 'cross-batch-counter-review', review_existing_obligations=True)
    if discard:
        assert not state.routes[old_route]['revoked']
        assert state.pending == {'history'} and reader.failed_obligations
        reader.current_identity = 'later-still-pending'
        payload = reader._payload(engine.steps[0], 'Review original counter', ['history'], [])
        assert 's2' in {source['id'] for source in payload['source_spans']}
    else:
        assert state.routes[old_route]['revoked'] and 'history' not in state.valid()[0]
        assert not state.pending and not reader.failed_obligations


def test_history_review_cannot_clear_the_same_variables_failed_audit_batch():
    reader, engine, state, calls = personal_reader(max_repairs_per_request=1)
    # A missing audit verdict is an audit-stage failure even when its factual
    # slots have legal shapes. Re-reading history is not a replacement audit.
    calls.respond = lambda index, data: response(observation('history-proof', ['s1']))
    reader.read(engine.steps[0], 'Audit historical support', ['history'],
                ['piano'], 'failed-audit-window', audit=True)
    assert reader.failed_obligations
    original_markers = deepcopy(reader.failed_obligations)
    def forged_history_release(index, data):
        records = data.get('failed_batch_reviews', [])
        assert not any(item.get('review_allowed') for item in records)
        result = response(observation('history-proof', ['s1']))
        result['slots'][0]['resolved_failed_batch_ids'] = [item['id'] for item in records]
        return result
    calls.respond = forged_history_release
    reader.read(engine.steps[0], 'Read historical support', ['history'],
                ['painting'], 'history-is-not-audit', review_existing_obligations=True)
    for key, variables in original_markers.items():
        assert variables <= reader.failed_obligations.get(key, set())
    assert state.pending == {'history'}


def test_handoff_failure_keeps_original_marker_but_does_not_roll_back_valid_counter(monkeypatch):
    reader, engine, state, calls, old_route, counter = counter_failure()
    original_markers = deepcopy(reader.failed_obligations)
    original_routes = deepcopy(state.routes)
    original_handoff = reader._handoff_failures
    original_fits = reader._fits
    # Isolate atomic state handling from token-length accidents: the actual
    # prospective handoff runs, but its final join has no wire capacity.
    def fits(operation, data):
        return False if data['task']['output_slot'] == 'suggestion' else original_fits(operation, data)
    monkeypatch.setattr(reader, '_fits', fits)
    probes = []
    def inspect_handoff(*args, **kwargs):
        before_routes = deepcopy(state.routes)
        before_markers = deepcopy(reader.failed_obligations)
        result = original_handoff(*args, **kwargs)
        assert state.routes == before_routes == original_routes
        assert reader.failed_obligations == before_markers == original_markers
        probes.append(result)
        return result
    monkeypatch.setattr(reader, '_handoff_failures', inspect_handoff)
    def review(index, data):
        corrected = {**deepcopy(counter), 'entity_scope': 'user'}
        support = observation('new-proof', ['s1', 's2'], value=['painting'])
        result = response(corrected, support)
        result['slots'][0]['resolved_failed_batch_ids'] = [
            item['id'] for item in data['failed_batch_reviews'] if item.get('review_allowed')]
        return result
    calls.respond = review
    reader.read(engine.steps[0], 'Review counter with new source', ['history'],
                ['advice'], 'review-with-infeasible-handoff', review_existing_obligations=True)
    assert probes and any(any('citation_capacity' in message for message in failure.values()) for failure in probes)
    assert state.routes[old_route]['revoked'], 'A valid counter survives rejected positive support'
    assert 'history' not in state.valid()[0] and state.pending == {'history'}
    for key, variables in original_markers.items():
        assert variables <= reader.failed_obligations.get(key, set())


def test_real_capacity_failed_draft_recovers_only_after_explicit_sufficient_proof_review():
    reader, engine, state, calls, aliases, broad = capacity_fixture()
    # Fix the entire experiment budget before the first model call. A large
    # downstream contract makes the original broad proof infeasible while
    # leaving room to show its original text and durable review record locally.
    engine.s.update(context_tokens=16384, max_repairs_per_request=1)
    engine.steps[1]['question'] += ' Preserve original source qualifications.' * 370
    narrow = reader._observation(observation('proof', [aliases[0]]), 'history',
                                  {alias: engine.mapper.sources[alias] for alias in aliases}, engine.steps[0])
    assert reader._handoff_failures([broad], {('history', 'proof')})
    assert not reader._handoff_failures([narrow], {('history', 'proof')})
    calls.respond = lambda index, data: response(observation('proof', aliases))
    reader.read(engine.steps[0], 'Read past creative activities', ['history'],
                list(engine.docs), 'broad-original-window')
    assert len(calls.requests) == 2 and state.pending == {'history'}
    assert state.valid()[0] == {} and reader.failed_obligations
    original_budgets = deepcopy(engine.s)

    text = 'My current musical activity is still playing piano with friends.'
    engine.docs['new-original-memory'] = SimpleNamespace(
        doc_id='new-original-memory', passage=text,
        metadata={'source_segments': [{'role': 'user', 'start': 0, 'end': len(text),
                                       'source_message_indices': [8]}]})
    engine.mapper.add_candidates(['new-original-memory'])
    def review(index, data):
        offered = [item for item in data['failed_batch_reviews'] if item.get('review_allowed')]
        assert len(offered) == 1
        assert set(aliases) <= {source['id'] for source in data['source_spans']}
        old = next(item['observation'] for item in offered[0]['failed'] if item['id'] == 'proof')
        assert set(old['source_ids']) == set(aliases)
        proof = observation('proof', [aliases[0]], reason=(
            'Reviewed every original span: they repeat the same piano activity. '
            'This source alone establishes the full scoped value; the others add no independent premise.'))
        result = response(proof)
        result['slots'][0]['resolved_failed_batch_ids'] = [offered[0]['id']]
        return result
    calls.respond = review
    reader.read(engine.steps[0], 'Read past creative activities', ['history'],
                ['new-original-memory'], 'capacity-review-with-new-text', review_existing_obligations=True)
    assert engine.s == original_budgets
    assert len(calls.requests) == 3 and engine.ledger.used['json_repairs'] == 2
    assert not state.pending and not reader.failed_obligations and not reader.failed_batch_reviews
    assert state.valid()[0]['history'] == ['piano']
    assert state.valid()[1]['history'] == [engine.mapper.sources[aliases[0]]['source_id']]
    assert all(alias in engine.mapper.sources for alias in aliases)


def add_fixture_original(engine, name, text):
    document = SimpleNamespace(doc_id=name, passage=text, metadata={
        'source_segments': [{'role': 'user', 'start': 0, 'end': len(text),
                             'source_message_indices': [len(engine.docs)]}]})
    engine.docs[name] = document
    for source in build_source_spans(name, document, 400):
        alias = 's' + str(len(engine.mapper.sources) + 1)
        engine.mapper.sources[alias] = {**source, 'id': alias, 'source_id': source['id']}
        engine.mapper.source_aliases[source['id']] = alias


def test_counter_validated_during_a_later_failed_review_remains_protected_on_next_review():
    reader, engine, state, calls, old_route, counter = counter_failure()
    corrected = {**deepcopy(counter), 'entity_scope': 'user'}
    def valid_counter_bad_envelope(index, data):
        result = response(deepcopy(corrected))
        result['slots'][0]['resolved_failed_batch_ids'] = [
            item['id'] for item in data['failed_batch_reviews'] if item.get('review_allowed')]
        result['final_prediction'] = 'a'
        return result
    calls.respond = valid_counter_bad_envelope
    reader.read(engine.steps[0], 'Review counter with new original text', ['history'],
                ['advice'], 'second-counter-window', review_existing_obligations=True)
    assert state.pending == {'history'} and not state.routes[old_route]['revoked']

    add_fixture_original(engine, 'later-user-memory', 'I am still checking my earlier musical preferences.')
    def attempt_to_discard_now_valid_counter(index, data):
        ignored = {**deepcopy(corrected), 'stance': 'irrelevant', 'retract_ids': [],
                   'reason': 'Trying to ignore the previously validated original counter.'}
        result = response(ignored)
        result['slots'][0]['resolved_failed_batch_ids'] = [
            item['id'] for item in data['failed_batch_reviews'] if item.get('review_allowed')]
        return result
    calls.respond = attempt_to_discard_now_valid_counter
    reader.read(engine.steps[0], 'Review counter with still newer text', ['history'],
                ['later-user-memory'], 'third-counter-window', review_existing_obligations=True)
    assert state.pending == {'history'} and reader.failed_obligations
    assert not state.routes[old_route]['revoked']
    reader.current_identity = 'counter-must-still-be-reviewed'
    payload = reader._payload(engine.steps[0], 'Review original counter', ['history'], [])
    assert 's2' in {source['id'] for source in payload['source_spans']}


def test_invalid_counter_cannot_be_discharged_using_only_an_unrelated_original_source():
    reader, engine, state, calls, old_route, counter = counter_failure()
    def drop_original_negative_source(index, data):
        ignored = {**deepcopy(counter), 'entity_scope': 'user', 'stance': 'irrelevant',
                   'retract_ids': [], 'source_ids': ['s1'],
                   'reason': 'Calling the old draft irrelevant while citing only an unrelated old positive source.'}
        result = response(ignored)
        result['slots'][0]['resolved_failed_batch_ids'] = [
            item['id'] for item in data['failed_batch_reviews'] if item.get('review_allowed')]
        return result
    calls.respond = drop_original_negative_source
    reader.read(engine.steps[0], 'Review failed counter with new text', ['history'],
                ['advice'], 'unrelated-positive-review', review_existing_obligations=True)
    assert state.pending == {'history'} and reader.failed_obligations
    assert not state.routes[old_route]['revoked']
    reader.current_identity = 'still-need-negative-source-review'
    payload = reader._payload(engine.steps[0], 'Review original negative memory', ['history'], [])
    assert 's2' in {source['id'] for source in payload['source_spans']}


@pytest.mark.parametrize('source_review', ['same_draft', 'separate_review'])
def test_invalid_counter_can_be_reviewed_as_irrelevant_only_with_its_actual_original_source(source_review):
    reader, engine, state, calls, old_route, counter = counter_failure()
    def review_original_source(index, data):
        ignored = {**deepcopy(counter), 'entity_scope': 'user', 'stance': 'irrelevant',
                   'retract_ids': [], 'source_ids': ['s2' if source_review == 'same_draft' else 's1'],
                   'reason': 'Reviewed the draft against the original history; painting does not contradict playing piano.'}
        rows = [ignored]
        if source_review == 'separate_review':
            evidence_review = observation('original-negative-source-review', ['s2'], stance='irrelevant',
                reason='The actual painting memory describes another activity, not a retraction of piano history.')
            evidence_review['value'] = None
            rows.append(evidence_review)
        result = response(*rows)
        result['slots'][0]['resolved_failed_batch_ids'] = [
            item['id'] for item in data['failed_batch_reviews'] if item.get('review_allowed')]
        return result
    calls.respond = review_original_source
    reader.read(engine.steps[0], 'Review failed counter with new original text', ['history'],
                ['advice'], 'explicit-original-source-review', review_existing_obligations=True)
    assert not state.pending and not reader.failed_obligations
    assert state.valid()[0]['history'] == ['piano'] and not state.routes[old_route]['revoked']


def test_paid_durable_review_keeps_its_authorization_during_its_remaining_local_repair():
    reader, engine, state, calls = failed_whole(max_repairs=2)
    before_calls = len(calls.requests)
    before_repairs = engine.ledger.used['json_repairs']
    offered_ids = []
    def review_then_correct(index, data):
        offered = [item['id'] for item in data['failed_batch_reviews'] if item.get('review_allowed')]
        assert len(offered) == 1
        offered_ids.append(offered)
        if len(offered_ids) == 1:
            assert 'repair_scope' not in data
            return whole_review_response(data, drop_source=True)
        assert 'repair_scope' in data
        assert offered_ids[0] == offered, 'Updating seen-source state cannot revoke this already-paid local repair'
        result = whole_review_response(data)
        failed_ids = {row['id'] for row in data['repair_scope']['failed']}
        result['slots'][0]['observations'] = [
            row for row in result['slots'][0]['observations'] if row['id'] in failed_ids]
        return result
    calls.respond = review_then_correct
    reader.read(engine.steps[0], 'Review historical activities', ['history'],
                ['advice'], 'new-evidence-then-local-correction', review_existing_obligations=True)
    assert len(calls.requests) == before_calls + 2
    assert engine.ledger.used['json_repairs'] == before_repairs + 2
    assert state.valid()[0]['history'] == ['piano', 'painting']
    assert not state.pending and not reader.failed_obligations
