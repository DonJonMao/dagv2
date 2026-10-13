"""Source feedback describes the validated draft, never other proofs or errors."""
from copy import deepcopy

import pytest

from dagbt.fact_state import SummaryRevisionSourcesMissing
from dagbt.joint_reading import JointReader
from dagbt.reasoning import ProtocolError
from test_personamem_joint_protocol import observation, response, visible
from test_v4_summary_revision_review import OLD, summary_fixture, revised, read_new_history
from test_v4_summary_source_feedback_review import (
    source_fixture, read_revision, feedback_record, DRAFT_ID,
)


def test_feedback_uses_selected_target_proof_not_all_eligible_or_routes():
    reader, engine, state, calls, original = summary_fixture()
    engine.s['max_repairs_per_request'] = 1
    state.integrate([reader._observation(observation('alternative-proof', ['s2'], value=OLD),
                                        'history', visible(reader, ['s1', 's2']), engine.steps[0])])
    before = deepcopy((state.routes, state.versions))
    draft = revised(original, state.versions['history'], sources=['s2'], value=OLD)
    def reply(index, data):
        # The offered contract covers all eligible routes; this draft explicitly
        # selects only the original route, so s2 is not inherited from its target.
        assert set(data['summary_revision_contracts']['history']['required_source_ids']) == {'s1', 's2'}
        if index == 1:
            return response(deepcopy(draft))
        item = next(item for item in data['repair_scope']['failed'] if item['id'] == draft['id'])
        assert item['summary_revision_sources'] == {
            'required_source_ids': ['s1'], 'missing_source_ids': ['s1']}
        return response({**deepcopy(draft), 'source_ids': ['s1']})
    calls.respond = reply
    read_new_history(reader, engine)
    assert len(calls.requests) == engine.ledger.used['llm'] == 2
    assert engine.ledger.used['json_repairs'] == 1
    assert (state.routes, state.versions) == before, 'The corrected exact review is a no-op'
    assert not state.pending and not reader.failed_batch_reviews


@pytest.mark.parametrize('next_error', ['stale_version', 'changed_draft_bad_envelope'])
def test_source_feedback_cannot_follow_a_different_error_or_changed_unvalidated_draft(next_error):
    reader, engine, state, calls, aliases, original, draft = source_fixture(repairs=2)
    before = deepcopy((state.routes, state.versions))
    changed = {**deepcopy(draft), 'source_ids': aliases[:23]}
    if next_error == 'stale_version':
        changed['summary_revision']['previous_version'] -= 1
    def reply(index, data):
        if index == 1:
            return response(deepcopy(draft))
        item = feedback_record(data['repair_scope']['failed'])
        if index == 2:
            assert len(item['summary_revision_sources']['missing_source_ids']) == 13
        else:
            assert index == 3
            assert 'summary_revision_sources' not in item
            assert item['observation'] == changed
        result = response(deepcopy(changed))
        if next_error == 'changed_draft_bad_envelope':
            result['semantic_answer'] = None
        return result
    calls.respond = reply
    read_revision(reader, engine)
    assert len(calls.requests) == engine.ledger.used['llm'] == 3
    assert engine.ledger.used['json_repairs'] == 2
    assert (state.routes, state.versions) == before
    assert state.pending == {'history'} and reader.failed_obligations
    for record in reader.failed_batch_reviews.values():
        assert 'summary_revision_sources' not in feedback_record(record['failed'])


def test_correcting_citations_does_not_bypass_downstream_capacity_or_retire_route(monkeypatch):
    reader, engine, state, calls, aliases, original, draft = source_fixture(repairs=2)
    before = deepcopy((state.routes, state.versions))
    def reject_handoff(rows, rejectable, *args, **kwargs):
        return {(row['variable'], row['id']): 'citation_capacity: synthetic exact handoff limit'
                for row in rows if (row['variable'], row['id']) in rejectable}
    monkeypatch.setattr(reader, '_handoff_failures', reject_handoff)
    def reply(index, data):
        if index == 1:
            return response(deepcopy(draft))
        item = feedback_record(data['repair_scope']['failed'])
        if index == 2:
            assert 'summary_revision_sources' in item
        else:
            assert item['feedback'].startswith('citation_capacity:')
            assert 'summary_revision_sources' not in item
        return response({**deepcopy(draft), 'source_ids': aliases[:23]})
    calls.respond = reply
    read_revision(reader, engine)
    assert len(calls.requests) == 3 and engine.ledger.used['json_repairs'] == 2
    assert (state.routes, state.versions) == before
    assert state.pending == {'history'} and reader.failed_obligations


def test_missing_but_invisible_original_is_not_advertised_as_an_actionable_alias():
    reader, engine, state, _, original = summary_fixture()
    draft = revised(original, state.versions['history'], sources=['s2'])
    only_new = {'s2': engine.mapper.sources['s2']}
    with pytest.raises(ProtocolError, match='visible original text') as raised:
        reader._observation(draft, 'history', only_new, engine.steps[0], allow_summary_revision=True)
    assert not isinstance(raised.value, SummaryRevisionSourcesMissing)
    assert len(state.routes) == 1 and not state.routes[original].get('superseded_by')


def test_new_durable_feedback_requires_exact_mapping_while_old_drafts_remain_compatible():
    old = {'id': DRAFT_ID, 'observation': {'source_ids': ['unvalidated-old-draft-id']}}
    assert JointReader._remap_review_sources(old, {}) == old
    newer = {**old, 'summary_revision_sources': {
        'required_source_ids': ['src_original'], 'missing_source_ids': ['src_original']}}
    with pytest.raises(ProtocolError, match='unavailable original source'):
        JointReader._remap_failed_entry_sources(newer, {})
    mapped = JointReader._remap_failed_entry_sources(newer, {'src_original': 'visible_alias'})
    assert mapped['summary_revision_sources'] == {
        'required_source_ids': ['visible_alias'], 'missing_source_ids': ['visible_alias']}
