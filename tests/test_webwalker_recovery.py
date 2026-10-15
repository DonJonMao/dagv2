"""One paid Web durable recovery; no external model or reference answers."""
from copy import deepcopy

import pytest

from test_webwalker_reading import (ExactCounts, gap_response, observation, read,
                                    response, retained_failure_fixture)
from dagbt.transport import digest


def recovery_fixture():
    reader, engine, state, calls = retained_failure_fixture()
    calls.count_reasoning_tokens = ExactCounts(9000)
    return reader, engine, state, calls


def test_one_valid_paid_recovery_preserves_full_original_scope_and_closes_failed_draft():
    reader, engine, state, calls = recovery_fixture()
    before = deepcopy(reader.failed_batch_reviews)
    issues = deepcopy(reader.evidence_obligations)
    old_source = engine.mapper.sources['s1']['source_id']
    def corrected(index, data):
        result = response(observation('old-failed-draft', ['s1', 's2']))
        result['slots'][0]['resolved_failed_batch_ids'] = [
            row['id'] for row in data['failed_batch_reviews'] if row['review_allowed']]
        result['slots'][0]['resolved_obligation_ids'] = [row['id'] for row in data['evidence_obligations']]
        return result
    calls.respond = corrected
    read(reader, 'painting', 'single-paid-valid')
    assert len(calls.requests) == 4 and engine.ledger.used['llm'] == 4
    assert engine.ledger.used['json_repairs'] == 3
    data = calls.requests[-1]['data']
    assert calls.requests[-1]['stage'][0] == 'joint_read_repair'
    assert {source['id'] for source in data['source_spans']} == {'s1', 's2'}
    assert data['failed_batch_reviews'] and data['evidence_obligations']
    assert old_source in next(iter(before.values()))['source_ids']
    assert issues and not reader.failed_batch_reviews and not reader.evidence_obligations
    assert state.valid()[0]['history'] == ['piano'] and not state.pending
    record = reader.public_dict()['web_paid_recoveries'][0]
    assert record['status'] == 'response_received' and record['prefix_groups'] == 1
    assert record['additional_local_repairs_allowed'] == 0
    assert not record['rejected_full_forecast']['within_budget']
    assert reader._active_web_paid_recovery is None and reader._selected_web_paid_recovery is None


def test_invalid_paid_response_retains_old_and_new_drafts_without_another_local_charge():
    reader, engine, state, calls = recovery_fixture()
    before = deepcopy(reader.failed_batch_reviews)
    issues = deepcopy(reader.evidence_obligations)
    calls.respond = lambda i, data: response(observation('new-invalid-draft', ['not-visible']))
    read(reader, 'painting', 'single-paid-invalid')
    assert len(calls.requests) == 4 and engine.ledger.used['json_repairs'] == 3
    assert engine.ledger.used['llm'] == 4 and state.pending == {'history'}
    assert reader.evidence_obligations == issues
    for identifier, old in before.items():
        current = reader.failed_batch_reviews[identifier]
        for field in ('failed', 'source_ids', 'validated_observations'):
            assert all(item in current[field] for item in old[field])
        assert engine.mapper.sources['s2']['source_id'] in current['seen_source_ids']
    assert any(entry['id'] == 'new-invalid-draft'
               for record in reader.failed_batch_reviews.values() for entry in record['failed'])
    assert reader.web_paid_recoveries[0]['local_repair_blocked']
    assert any(event['event'] == 'web_paid_recovery_local_repair_blocked' for event in engine.events)
    assert reader._active_web_paid_recovery is None
    # Renaming the query cannot buy another request for the same originals.
    read(reader, 'painting', 'renamed-query-same-evidence')
    assert len(calls.requests) == 4 and engine.ledger.used['json_repairs'] == 3


def test_recovery_chooses_smallest_prefix_and_selection_alone_spends_no_budget():
    reader, engine, _, calls = recovery_fixture()
    reader.current_identity = 'smallest-prefix'
    budget, exposures = deepcopy(engine.ledger.public_dict()), deepcopy(reader.source_exposures)
    aliases, size = reader._fitting_prefix('joint_read', engine.steps[0], 'query', ['history'],
                                           [['s2'], ['s3']], False, False)
    assert aliases == ['s2'] and size == 1
    assert engine.ledger.public_dict() == budget and reader.source_exposures == exposures
    assert len(calls.requests) == 3


@pytest.mark.parametrize('edge', ['repair_budget', 'llm_budget', 'no_new_source', 'parent_change_only'])
def test_recovery_does_not_bypass_budget_or_requirements_for_genuinely_new_source(edge):
    reader, engine, _, calls = recovery_fixture()
    reader.current_identity = 'renamed-for-no-authority'
    pending = [['s2']]
    if edge == 'repair_budget':
        engine.ledger.reserve('json_repairs', 'other-work', engine.ledger.remaining('json_repairs'))
    elif edge == 'llm_budget':
        engine.ledger.reserve('llm', 'other-work', engine.ledger.remaining('llm') - engine.s['reserved_audit_calls'])
    else:
        pending = [['s1']]
        if edge == 'parent_change_only':
            # Explicitly unlock the core's changed-parent review, with no new
            # source. The Web fallback must independently reject it.
            reader._failed_parent_signature = lambda variable: 'different-parent-signature'
    before = deepcopy(engine.ledger.public_dict())
    data = reader._payload(engine.steps[0], 'query', ['history'], pending[0])
    if edge == 'parent_change_only':
        assert any(review['review_allowed'] for review in data['failed_batch_reviews'])
    reader._web_headroom_candidates = {digest(data): {'within_budget': False}}
    assert reader._paid_recovery_prefix('joint_read', engine.steps[0], 'query', ['history'],
                                        pending, False, False) == ([], 0)
    assert engine.ledger.public_dict() == before and len(calls.requests) == 3
    assert not reader.web_paid_recoveries
    assert engine.events[-1]['event'] == 'web_paid_recovery_blocked'


def test_single_response_mode_does_not_leak_to_later_batches_or_audit():
    reader, engine, _, calls = recovery_fixture()
    calls.respond = lambda i, data: response(observation('another-bad-draft', ['not-visible']))
    read(reader, 'painting', 'first-paid-batch')
    assert reader._active_web_paid_recovery is None
    # Actual later repairs use the ordinary exact fit check, including audit.
    data = reader._payload(engine.steps[0], 'later', ['history'], ['s3'])
    assert reader._repair_fits('joint_read_repair', data, engine.steps[0], 'first-paid-batch', 1)
    audit = reader._payload(engine.steps[0], 'audit', ['history'], ['s1'], audit=True)
    assert reader._repair_fits('audit_joint_repair', audit, engine.steps[0], 'first-paid-batch', 1)
    assert not reader._selected_web_paid_recovery
    # A later new-source batch with full forecast room may spend its normal
    # paid recovery plus one local repair; it does not inherit the prior cap.
    calls.count_reasoning_tokens = ExactCounts(2000)
    read(reader, 'advice', 'later-ordinary-full-headroom')
    assert len(calls.requests) == 6 and engine.ledger.used['json_repairs'] == 5
    assert len(reader.web_paid_recoveries) == 1


def test_single_paid_recovery_cannot_admit_an_oversized_complete_request():
    reader, engine, _, calls = recovery_fixture()
    reader.current_identity = 'oversized-original-closure'
    class Oversized(ExactCounts):
        def __call__(self, stage, url, payload):
            result = super().__call__(stage, url, payload)
            result['count'] = 15000
            return result
    calls.count_reasoning_tokens = Oversized(9000)
    data = reader._payload(engine.steps[0], 'query', ['history'], ['s2'])
    reader._web_headroom_candidates = {digest(data): {'within_budget': False}}
    assert reader._paid_recovery_prefix('joint_read', engine.steps[0], 'query', ['history'],
                                        [['s2']], False, False) == ([], 0)
    assert len(calls.requests) == 3 and not reader.web_paid_recoveries
    assert engine.events[-1]['reason'] == 'complete_recovery_request_exceeds_context'


@pytest.mark.parametrize('operation,semantic,audit', [('semantic_compose', True, False),
                                                   ('audit_joint', False, True)])
def test_single_paid_fallback_is_never_available_for_compose_or_audit(operation, semantic, audit):
    reader, engine, _, calls = recovery_fixture()
    assert reader._paid_recovery_prefix(operation, engine.steps[0], 'query', ['history'],
                                        [['s2']], semantic, audit) == ([], 0)
    assert len(calls.requests) == 3 and not reader.web_paid_recoveries
