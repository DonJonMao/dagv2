"""Current-batch immutability metadata mirrors existing repair guards offline."""
from copy import deepcopy
import json
import socket

import pytest
from jsonschema import Draft202012Validator

from dagbt.joint_reading import FROZEN_IDS_INSTRUCTION, JointReader
from dagbt.transport import digest
from test_personamem_joint_protocol import fixture, observation, response, visible
from test_v4_obligation_review_feedback_review import (
    OID, PROOF, claim, failed_slot, irrelevant, obligation_fixture, read_issue,
)


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Frozen-ID tests must not contact a service')
    monkeypatch.setattr('urllib.request.urlopen', forbidden)
    monkeypatch.setattr('urllib.request.build_opener', forbidden)
    monkeypatch.setattr(socket, 'create_connection', forbidden)


def request_validator(calls):
    system = calls.requests[-1]['payload']['messages'][0]['content']
    return Draft202012Validator(json.loads(system.rsplit(
        'Return one complete JSON object matching this schema:\n', 1)[1]))


def source_issue(reader, engine):
    reader.evidence_obligations['original-issue'] = {
        'id': 'original-issue', 'variable': 'history', 'entity_scope': 'user',
        'time_scope': 'history', 'source_ids': [engine.mapper.sources['s1']['source_id']],
        'descriptions': ['The historical relation needs a sourced review.'], 'origins': ['old-owner']}


def unsettled(identifier, *, stance='unknown', sources=('s1',)):
    row = observation(identifier, sources, stance=stance)
    row['value'] = None
    return row


@pytest.mark.parametrize('legacy', [False, True])
def test_only_explicit_settled_offers_authorize_references_and_keep_legacy_ban(legacy):
    reader, engine, _, _ = fixture()
    data = reader._payload(engine.steps[0], 'Read history', ['history'], ['s1'])
    data['repair_scope'] = {'failed': [{'variable': 'history', 'id': '*',
        'frozen_observation_reviews': {'slot_observation_ids': ['settled'], 'observations': []}}]}
    if not legacy:
        data['repair_scope']['frozen_observation_ids'] = {'history': ['unknown', 'settled']}
    validator = Draft202012Validator(reader._schema(data))
    assert not validator.is_valid(response(observation('settled')))
    reviewed = response()
    reviewed['slots'][0]['reviewed_observation_ids'] = ['settled']
    assert validator.is_valid(reviewed)
    reviewed['slots'][0]['reviewed_observation_ids'] = ['unknown']
    assert not validator.is_valid(reviewed)
    assert validator.is_valid(response(unsettled('unknown'))) is legacy


def test_forbidden_ids_are_scoped_per_variable_even_when_ids_are_shared():
    reader, engine, _, _, _ = obligation_fixture(other=True)
    data = reader._payload(engine.steps[0], 'Read history', ['history', 'other'], [])
    data['repair_scope'] = {'frozen_observation_ids': {'history': ['same-id']}}
    payload = {'slots': [
        {'variable': 'history', 'observations': [], 'unresolved': []},
        {'variable': 'other', 'observations': [unsettled('same-id')], 'unresolved': []}]}
    validator = Draft202012Validator(reader._schema(data))
    assert validator.is_valid(payload)
    payload['slots'][0]['observations'] = [unsettled('same-id')]
    assert not validator.is_valid(payload)


def test_current_batch_ids_are_sorted_and_do_not_import_known_routes():
    reader, engine, state, calls = fixture(max_repairs_per_request=1)
    state.integrate([reader._observation(observation('prior'), 'history', visible(reader), engine.steps[0])])
    old_ids = set(state.routes)
    source_issue(reader, engine)
    calls.count_reasoning_tokens = lambda *args: {'count': 100, 'token_count_is_estimate': False}
    drafts = [unsettled('z-current'), unsettled('a-current', stance='partial')]
    def reply(index, data):
        if index == 2:
            ids = data['repair_scope']['frozen_observation_ids']
            assert ids == {'history': ['a-current', 'z-current']}
            assert not old_ids & set(ids['history'])
            assert FROZEN_IDS_INSTRUCTION in calls.requests[-1]['payload']['messages'][0]['content']
        result = response(*deepcopy(drafts))
        result['slots'][0]['resolved_obligation_ids'] = ['original-issue']
        return result
    calls.respond = reply
    reader.read(engine.steps[0], 'Read history', ['history'], ['piano'], 'current-batch')
    assert len(calls.requests) == 2 and state.pending == {'history'}


def test_real_explicit_settled_review_remains_usable_with_full_frozen_metadata():
    reader, engine, state, calls, aliases = obligation_fixture()
    calls.count_reasoning_tokens = lambda *args: {'count': 100, 'token_count_is_estimate': False}
    def reply(index, data):
        if index == 1:
            return claim(observation(PROOF, aliases[:3]))
        assert data['repair_scope']['frozen_observation_ids'] == {'history': [PROOF]}
        assert failed_slot(data)['frozen_observation_reviews']['slot_observation_ids'] == [PROOF]
        result = claim(irrelevant('other-originals', aliases[3:53]), refs=[PROOF])
        assert request_validator(calls).is_valid(result)
        return result
    calls.respond = reply
    read_issue(reader, engine)
    assert len(calls.requests) == 2 and not state.pending
    assert not reader.evidence_obligations and state.valid()[0]['history'] == ['piano']


def test_whole_value_thaw_keeps_drafts_editable_and_counter_frozen():
    from test_v4_reconcile_review import fragments, personal_reader
    reader, engine, state, calls = personal_reader(max_repairs_per_request=1)
    calls.count_reasoning_tokens = lambda *args: {'count': 100, 'token_count_is_estimate': False}
    state.integrate([reader._observation(observation('prior', ['s3'], value=['guitar']),
                                       'history', visible(reader, ['s3']), engine.steps[0])])
    route = next(iter(state.routes))
    counter = unsettled('retained-counter', stance='contradiction', sources=['s2'])
    counter['retract_ids'] = [route]
    def reply(index, data):
        if index == 1:
            return response(*fragments(), deepcopy(counter))
        assert data['repair_scope']['frozen_observation_ids'] == {'history': ['retained-counter']}
        assert data['repair_scope']['whole_value_reviews']
        edited = response(*[unsettled(x['id'], stance='partial', sources=x['source_ids']) for x in fragments()])
        assert request_validator(calls).is_valid(edited)
        assert not request_validator(calls).is_valid(response(deepcopy(counter)))
        return edited
    calls.respond = reply
    reader.read(engine.steps[0], 'Review both activities', ['history'], ['piano', 'painting'], 'whole-thaw')
    assert len(calls.requests) == 2 and state.routes[route]['revoked']
    assert state.pending == {'history'}


def test_summary_counter_conflict_thaws_revision_without_thawing_counter():
    from test_v4_summary_revision_review import summary_fixture, revised, add_explicit_piano_correction
    reader, engine, state, calls, original = summary_fixture()
    engine.s['max_repairs_per_request'] = 1
    calls.count_reasoning_tokens = lambda *args: {'count': 100, 'token_count_is_estimate': False}
    add_explicit_piano_correction(engine)
    draft = revised(original, state.versions['history'], sources=('s1', 's2', 's4'))
    counter = unsettled('real-counter', stance='contradiction', sources=['s4'])
    counter['retract_ids'] = [original]
    def reply(index, data):
        if index == 1:
            return response(deepcopy(draft), deepcopy(counter))
        assert data['repair_scope']['frozen_observation_ids'] == {'history': ['real-counter']}
        edited = response(unsettled(draft['id'], sources=draft['source_ids']))
        assert request_validator(calls).is_valid(edited)
        return edited
    calls.respond = reply
    reader.read(engine.steps[0], 'Review explicit correction', ['history'],
                ['painting', 'piano-correction'], 'summary-conflict')
    assert len(calls.requests) == 2 and state.routes[original]['revoked']
    assert not state.routes[original].get('superseded_by') and state.pending == {'history'}


def test_summary_source_failure_never_freezes_the_still_editable_draft():
    from test_v4_summary_source_feedback_review import source_fixture, read_revision
    reader, engine, state, calls, aliases, original, draft = source_fixture()
    calls.count_reasoning_tokens = lambda *args: {'count': 100, 'token_count_is_estimate': False}
    def reply(index, data):
        if index == 1:
            return response(deepcopy(draft))
        assert 'frozen_observation_ids' not in data['repair_scope']
        edited = response({**deepcopy(draft), 'source_ids': aliases[:23]})
        assert request_validator(calls).is_valid(edited)
        return edited
    calls.respond = reply
    read_revision(reader, engine)
    assert len(calls.requests) == 2 and not state.pending
    assert state.routes[original]['superseded_by']


def test_real_handoff_rejection_thaws_its_draft_for_same_id_source_repair():
    from test_v4_joint_reading import capacity_fixture
    reader, engine, state, calls, aliases, _ = capacity_fixture()
    def reply(index, data):
        if index == 1:
            return response(observation('proof', aliases))
        assert 'frozen_observation_ids' not in data['repair_scope']
        edited = response(observation('proof', aliases[:1]))
        assert request_validator(calls).is_valid(edited)
        return edited
    calls.respond = reply
    reader.read(engine.steps[0], 'Read past creative activities', ['history'], list(engine.docs), 'read')
    assert len(calls.requests) == 2 and not state.pending
    assert state.valid()[1]['history'] == [engine.mapper.sources[aliases[0]]['source_id']]


def test_new_paid_batch_does_not_import_old_frozen_ids_from_durable_failure():
    reader, engine, state, calls = fixture(max_repairs_per_request=1)
    calls.count_reasoning_tokens = lambda *args: {'count': 100, 'token_count_is_estimate': False}
    source_issue(reader, engine)
    draft = unsettled('reused-id')
    def fail(index, data):
        result = response(deepcopy(draft))
        result['slots'][0]['resolved_obligation_ids'] = ['original-issue']
        return result
    calls.respond = fail
    reader.read(engine.steps[0], 'Read history', ['history'], ['piano'], 'old-batch')
    assert len(calls.requests) == 2 and reader.failed_batch_reviews
    checkpoint = json.loads(json.dumps(reader.obligation_snapshot()))
    reader = JointReader(engine, state)
    reader.restore_obligations(checkpoint)
    def recover(index, data):
        assert index == 3 and data['failed_batch_reviews']
        assert any(x['review_allowed'] for x in data['failed_batch_reviews'])
        assert 'frozen_observation_ids' not in data.get('repair_scope', {})
        result = response(deepcopy(draft))
        assert request_validator(calls).is_valid(result)
        return result
    calls.respond = recover
    reader.read(engine.steps[0], 'Read history with another original', ['history'],
                ['painting'], 'new-batch', review_existing_obligations=True)
    assert len(calls.requests) == 3 and state.pending == {'history'}
    assert reader.evidence_obligations and reader.failed_batch_reviews


def test_frozen_metadata_is_in_both_complete_wire_candidates_and_exact_counts(monkeypatch):
    from test_v4_review_wire_selection import payload_fixture, fake_accounting
    from dagbt.review_wire import unpack
    reader, _, _, _, data = payload_fixture()
    data['repair_scope'] = {'frozen_observation_ids': {'history': ['unsettled-id']}}
    baseline_identity = reader._selection_data_identity(data)
    candidates = reader._wire_candidates(data)
    assert len(candidates) == 2
    for wire, system in candidates:
        assert unpack(wire)['repair_scope']['frozen_observation_ids'] == {'history': ['unsettled-id']}
        assert FROZEN_IDS_INSTRUCTION in system
    counted = fake_accounting(monkeypatch, reader)
    reader._select_wire('joint_read_repair', candidates, reader._schema(data))
    assert len(counted) == 2
    for _, payload, _ in counted:
        assert json.loads(payload['messages'][1]['content'])['repair_scope']['frozen_observation_ids'] == {'history': ['unsettled-id']}
        schema = json.loads(payload['messages'][0]['content'].rsplit(
            'Return one complete JSON object matching this schema:\n', 1)[1])
        assert not Draft202012Validator(schema).is_valid(response(unsettled('unsettled-id')))
    other = deepcopy(data)
    other['repair_scope']['frozen_observation_ids']['history'] = ['newly-frozen-id']
    assert reader._selection_data_identity(other) != baseline_identity
    assert digest(other) != digest(data)
