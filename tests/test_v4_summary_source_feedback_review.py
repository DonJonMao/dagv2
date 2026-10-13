"""Independent exact-source repair checks through the real bounded reader.

All originals and scripted responses are synthetic. No labels, network or live
model are used; provenance, wire serialization, admission and budgets are real.
"""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from dagbt.evidence_spans import build_source_spans
from dagbt.fact_state import FactState, SummaryRevisionSourcesMissing
from dagbt.joint_reading import JointReader
from dagbt.reasoning import ProtocolError
from test_personamem_joint_protocol import fixture, observation, response, visible


OLD = 'The user reported several creative experiences in the recorded history.'
EXPANDED = OLD + ' A later painting experience was also enjoyable.'
DRAFT_ID = 'revised-history'


def source_fixture(*, repairs=1, atomic=False):
    _, engine, old_state, calls = fixture(max_repairs_per_request=repairs)
    engine.personal_response_selection = True
    engine.docs.clear()
    engine.mapper.sources.clear()
    engine.mapper.source_aliases.clear()
    aliases = []
    for index in range(24):
        name = f'historical-original-{index:02d}'
        text = f'I enjoyed creative activity number {index} during that recorded period.'
        doc = SimpleNamespace(doc_id=name, passage=text, metadata={
            'source_segments': [{'role': 'user', 'start': 0, 'end': len(text),
                                 'source_message_indices': [index]}]})
        engine.docs[name] = doc
        original = build_source_spans(name, doc, 400)[0]
        alias = f'source_alias_{index:05d}'
        engine.mapper.sources[alias] = {**original, 'id': alias, 'source_id': original['id']}
        engine.mapper.source_aliases[original['id']] = alias
        aliases.append(alias)
    program = deepcopy(old_state.program)
    program['variables']['history'] = {
        'type': 'text', 'demand': 'history',
        'evidence_semantics': 'personal_atomic_fact' if atomic else 'personal_history_summary',
        'scope': {'entity_scope': 'user', 'time_scope': 'history'}}
    state = FactState(program, {**old_state.contract, 'time_scope': 'response'}, engine.steps, 'suggestion')
    reader = JointReader(engine, state)
    old_aliases = aliases[:22]
    state.integrate([reader._observation(observation('old-history', old_aliases, value=OLD),
                                        'history', visible(reader, old_aliases), engine.steps[0])])
    original = next(iter(state.routes))
    draft = observation(DRAFT_ID, old_aliases[:9] + [aliases[22]], value=EXPANDED,
                        summary_revision={'supersedes_route_ids': [original],
                                          'previous_version': state.versions['history'],
                                          'reason': 'Review all old experiences and the new painting original.'})
    return reader, engine, state, calls, aliases, original, draft


def read_revision(reader, engine, *, review=False, second=False):
    doc = f'historical-original-{23 if second else 22:02d}'
    reader.read(engine.steps[0], 'Review the sourced historical summary', ['history'], [doc],
                'durable-new-evidence' if second else 'initial-source-review',
                review_existing_obligations=review)


def feedback_record(items):
    matches = [item for item in items if item['id'] == DRAFT_ID and item.get('variable', 'history') == 'history']
    assert len(matches) == 1
    return matches[0]


def assert_source_feedback(item, required, missing, data):
    detail = item['summary_revision_sources']
    assert set(detail) == {'required_source_ids', 'missing_source_ids'}
    assert set(detail['required_source_ids']) == set(required)
    assert set(detail['missing_source_ids']) == set(missing)
    assert len(detail['required_source_ids']) == len(required)
    assert len(detail['missing_source_ids']) == len(missing)
    assert set(required) <= {span['id'] for span in data['source_spans']}
    assert len(json.dumps(detail['missing_source_ids'])) > 240


def test_thirteen_missing_sources_are_complete_and_real_paid_repair_must_supply_them():
    reader, engine, state, calls, aliases, original, draft = source_fixture()
    before_routes, before_versions, limits = deepcopy(state.routes), deepcopy(state.versions), deepcopy(engine.ledger.limits)
    def reply(index, data):
        assert state.routes == before_routes and state.versions == before_versions
        if index == 1:
            return response(deepcopy(draft))
        assert index == 2
        item = feedback_record(data['repair_scope']['failed'])
        assert_source_feedback(item, aliases[:22], aliases[9:22], data)
        assert item['observation']['source_ids'] == draft['source_ids'], 'Feedback must not auto-fill the draft'
        wire = calls.requests[-1]['wire_data']
        assert feedback_record(wire['repair_scope']['failed'])['summary_revision_sources'] == item['summary_revision_sources']
        return response({**deepcopy(draft), 'source_ids': aliases[:23]})
    calls.respond = reply
    read_revision(reader, engine)
    assert len(calls.requests) == 2 and engine.ledger.used['llm'] == 2
    assert engine.ledger.used['json_repairs'] == 1 and engine.ledger.limits == limits
    assert state.valid()[0]['history'] == EXPANDED and not state.pending
    assert state.routes[original]['superseded_by'] and not state.routes[original]['revoked']
    assert not reader.failed_obligations and not reader.failed_batch_reviews


def test_unchanged_missing_citations_stay_pending_after_bounded_paid_repair():
    reader, engine, state, calls, aliases, original, draft = source_fixture()
    before_routes, before_versions = deepcopy(state.routes), deepcopy(state.versions)
    def reply(index, data):
        if index == 2:
            assert_source_feedback(feedback_record(data['repair_scope']['failed']),
                                   aliases[:22], aliases[9:22], data)
        return response(deepcopy(draft))
    calls.respond = reply
    read_revision(reader, engine)
    assert len(calls.requests) == 2 and engine.ledger.used['json_repairs'] == 1
    assert state.routes == before_routes and state.versions == before_versions
    assert state.pending == {'history'} and reader.pending_variables() == {'history'}
    assert reader.failed_obligations and reader.failed_batch_reviews
    assert not state.routes[original].get('superseded_by')


def test_durable_feedback_keeps_stable_ids_and_rebinds_every_alias_after_checkpoint():
    reader, engine, state, calls, aliases, original, draft = source_fixture()
    calls.respond = lambda index, data: response(deepcopy(draft))
    read_revision(reader, engine)
    assert len(calls.requests) == 2 and state.pending == {'history'}
    limits, settings = deepcopy(engine.ledger.limits), deepcopy(engine.s)
    checkpoint = json.loads(json.dumps(reader.obligation_snapshot()))
    old_record = next(iter(reader.failed_batch_reviews.values()))
    stable = feedback_record(old_record['failed'])['summary_revision_sources']
    assert set(stable['required_source_ids']) == {engine.mapper.sources[a]['source_id'] for a in aliases[:22]}
    assert set(stable['missing_source_ids']) == {engine.mapper.sources[a]['source_id'] for a in aliases[9:22]}
    rename = {alias: f'current_alias_{index:05d}' for index, alias in enumerate(aliases)}
    engine.mapper.sources = {rename[alias]: {**row, 'id': rename[alias]}
                             for alias, row in engine.mapper.sources.items()}
    engine.mapper.source_aliases = {row['source_id']: alias for alias, row in engine.mapper.sources.items()}
    restored = JointReader(engine, state)
    restored.restore_obligations(checkpoint)
    def reply(index, data):
        assert index == 3
        offered = [record for record in data['failed_batch_reviews'] if record['review_allowed']]
        assert len(offered) == 1
        failed = feedback_record(offered[0]['failed'])
        assert_source_feedback(failed, [rename[a] for a in aliases[:22]], [rename[a] for a in aliases[9:22]], data)
        assert set(failed['observation']['source_ids']) == {rename[a] for a in draft['source_ids']}
        corrected = {**deepcopy(draft), 'source_ids': [rename[a] for a in aliases[:23]]}
        result = response(corrected)
        result['slots'][0]['resolved_failed_batch_ids'] = [offered[0]['id']]
        result['slots'][0]['resolved_obligation_ids'] = [item['id'] for item in data.get('evidence_obligations', [])]
        return result
    calls.respond = reply
    read_revision(restored, engine, review=True, second=True)
    assert len(calls.requests) == 3 and engine.ledger.used['llm'] == 3
    assert engine.ledger.used['json_repairs'] == 2
    assert engine.ledger.limits == limits and engine.s == settings
    assert state.valid()[0]['history'] == EXPANDED and not state.pending
    assert state.routes[original]['superseded_by']
    assert not restored.failed_obligations and not restored.failed_batch_reviews


@pytest.mark.parametrize('mistake', ['stale_version', 'unknown_route', 'wrong_scope', 'invented_parent', 'atomic'])
def test_source_omission_feedback_does_not_mask_ineligible_or_stale_metadata(mistake):
    reader, engine, state, calls, _, original, draft = source_fixture(atomic=mistake == 'atomic')
    if mistake == 'stale_version':
        draft['summary_revision']['previous_version'] -= 1
    elif mistake == 'unknown_route':
        draft['summary_revision']['supersedes_route_ids'] = ['nonexistent-route']
    elif mistake == 'wrong_scope':
        draft['time_scope'] = 'current'
    elif mistake == 'invented_parent':
        draft['parent_versions'] = {'suggestion': 0}
    before_routes, before_versions = deepcopy(state.routes), deepcopy(state.versions)
    def reply(index, data):
        if index == 2:
            item = feedback_record(data['repair_scope']['failed'])
            assert 'summary_revision_sources' not in item
            assert 'retain every original premise' not in item['feedback']
        return response(deepcopy(draft))
    calls.respond = reply
    read_revision(reader, engine)
    assert len(calls.requests) == 2
    assert state.routes == before_routes and state.versions == before_versions
    assert state.pending == {'history'} and reader.failed_obligations
    assert not state.routes[original].get('superseded_by')
    for record in reader.failed_batch_reviews.values():
        for item in record['failed']:
            assert 'summary_revision_sources' not in item


def test_prior_source_feedback_is_removed_when_next_failed_draft_has_stale_version():
    reader, engine, state, calls, aliases, original, draft = source_fixture(repairs=2)
    stale = deepcopy(draft)
    stale['summary_revision']['previous_version'] -= 1
    def reply(index, data):
        if index == 1:
            return response(deepcopy(draft))
        item = feedback_record(data['repair_scope']['failed'])
        if index == 2:
            assert_source_feedback(item, aliases[:22], aliases[9:22], data)
        else:
            assert index == 3 and 'stale' in item['feedback']
            assert 'summary_revision_sources' not in item
        return response(deepcopy(stale))
    calls.respond = reply
    read_revision(reader, engine)
    assert len(calls.requests) == 3 and engine.ledger.used['json_repairs'] == 2
    assert state.pending == {'history'} and not state.routes[original].get('superseded_by')
    for record in reader.failed_batch_reviews.values():
        assert 'summary_revision_sources' not in feedback_record(record['failed'])


def test_no_available_repair_budget_keeps_missing_sources_unresolved():
    reader, engine, state, calls, _, original, draft = source_fixture()
    engine.ledger.limits['json_repairs'] = 0
    calls.respond = lambda index, data: response(deepcopy(draft))
    read_revision(reader, engine)
    assert len(calls.requests) == 1 and engine.ledger.used.get('json_repairs', 0) == 0
    assert state.pending == {'history'} and reader.failed_obligations
    assert not state.routes[original].get('superseded_by')


def test_unavailable_original_is_not_exposed_as_actionable_missing_alias():
    reader, engine, state, _, _, original, draft = source_fixture()
    provided_only = {alias: engine.mapper.sources[alias] for alias in draft['source_ids']}
    with pytest.raises(ProtocolError, match='visible original text') as caught:
        reader._observation(draft, 'history', provided_only, engine.steps[0], allow_summary_revision=True)
    assert not isinstance(caught.value, SummaryRevisionSourcesMissing)
    assert not state.routes[original].get('superseded_by')


def test_durable_feedback_unknown_stable_id_fails_before_any_model_request():
    reader, engine, state, calls, _, original, draft = source_fixture(repairs=0)
    calls.respond = lambda index, data: response(deepcopy(draft))
    read_revision(reader, engine)
    checkpoint = json.loads(json.dumps(reader.obligation_snapshot()))
    failed = feedback_record(checkpoint['failed_batch_reviews'][0]['failed'])
    failed['summary_revision_sources']['missing_source_ids'].append('unavailable-stable-original')
    restored = JointReader(engine, state)
    restored.restore_obligations(checkpoint)
    before = len(calls.requests)
    with pytest.raises(ProtocolError, match='unavailable original source'):
        restored._payload(engine.steps[0], 'Review retained history', ['history'], [])
    assert len(calls.requests) == before
    assert state.pending == {'history'} and restored.failed_obligations
    assert not state.routes[original].get('superseded_by')
