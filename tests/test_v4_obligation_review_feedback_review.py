"""Independent scoped-obligation repair checks, with no service or label access.

Only generation replies are scripted. The reader, source mapping, state, wire,
repair admission and budget ledger are the production implementations.
"""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from dagbt.evidence_spans import build_source_spans
from dagbt.fact_state import FactState
from dagbt.joint_reading import JointReader
from dagbt.reasoning import ProtocolError
from test_personamem_joint_protocol import fixture, observation, response, visible


OID = 'retained-scoped-issue'
PROOF = 'current-small-proof'


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Independent obligation review must never contact a service')
    monkeypatch.setattr('urllib.request.urlopen', forbidden)
    monkeypatch.setattr('urllib.request.build_opener', forbidden)


def obligation_fixture(*, repairs=1, other=False):
    reader, engine, state, calls = fixture(max_repairs_per_request=repairs)
    engine.docs.clear()
    engine.mapper.sources.clear()
    engine.mapper.source_aliases.clear()
    aliases = []
    for index in range(54):
        doc_id = f'retained-original-{index:02d}'
        text = f'This is original scoped history record {index}.'
        doc = SimpleNamespace(doc_id=doc_id, passage=text, metadata={
            'source_segments': [{'role': 'user', 'start': 0, 'end': len(text),
                                 'source_message_indices': [index]}]})
        engine.docs[doc_id] = doc
        source = build_source_spans(doc_id, doc, 400)[0]
        alias = f'review_alias_{index:05d}'
        engine.mapper.sources[alias] = {**source, 'id': alias, 'source_id': source['id']}
        engine.mapper.source_aliases[source['id']] = alias
        aliases.append(alias)
    if other:
        program = deepcopy(state.program)
        program['variables']['other'] = {'type': 'set', 'demand': 'history'}
        state = FactState(program, state.contract, engine.steps, 'suggestion')
        reader = JointReader(engine, state)
    baseline = observation('prior-three-source-binding', aliases[:3])
    state.integrate([reader._observation(baseline, 'history', visible(reader, aliases[:3]), engine.steps[0])])
    reader.evidence_obligations[OID] = {
        'id': OID, 'variable': 'history', **state.scope('history'),
        'source_ids': [engine.mapper.sources[a]['source_id'] for a in aliases[:53]],
        'descriptions': ['Review the retained scoped issue against all its original text.'],
        'origins': ['original-history-read']}
    state.integrate([], {'history'})
    return reader, engine, state, calls, aliases


def read_issue(reader, engine, *, identity='explicit-obligation-review', new=False, other=False):
    return reader.read(engine.steps[0], 'Review the retained scoped issue',
                       ['history', 'other'] if other else ['history'],
                       ['retained-original-53'] if new else [], identity,
                       review_existing_obligations=True)


def irrelevant(identifier, sources):
    row = observation(identifier, sources, stance='irrelevant')
    row['value'] = None
    row['reason'] = 'These independently reviewed originals do not change this scoped activity value.'
    return row


def claim(*rows, refs=()):
    result = response(*rows)
    result['slots'][0]['resolved_obligation_ids'] = [OID]
    if refs:
        result['slots'][0]['reviewed_observation_ids'] = list(refs)
    return result


def failed_slot(data):
    matches = [item for item in data['repair_scope']['failed']
               if item['variable'] == 'history' and item['id'] == '*']
    assert len(matches) == 1
    return matches[0]


def assert_coverage(item, aliases, covered):
    records = item['obligation_review_sources']
    assert len(records) == 1
    detail = records[0]
    required = set(aliases[:53])
    assert detail['obligation_id'] == OID
    assert set(detail['required_source_ids']) == required
    assert set(detail['covered_source_ids']) == set(covered)
    assert set(detail['missing_source_ids']) == required - set(covered)
    assert len(detail['required_source_ids']) == 53
    assert len(detail['missing_source_ids']) == len(required - set(covered))
    return detail


def offered_frozen_ids(item):
    offered = item['frozen_observation_reviews']
    assert set(offered) == {'slot_observation_ids', 'observations'}
    referenced = offered['slot_observation_ids']
    slot_rows = {row['id']: row for row in item['slot']['observations']}
    assert set(referenced) <= set(slot_rows)
    return set(referenced) | {row['id'] for row in offered['observations']}


@pytest.mark.parametrize('omit_last', [False, True])
def test_exact_fifty_missing_sources_need_explicit_frozen_review_and_real_remaining_review(omit_last):
    reader, engine, state, calls, aliases = obligation_fixture()
    settings, limits = deepcopy(engine.s), deepcopy(engine.ledger.limits)
    original_sources = deepcopy(engine.mapper.sources)
    def reply(index, data):
        if index == 1:
            return claim(observation(PROOF, aliases[:3]))
        assert index == 2 and OID in reader.evidence_obligations
        item = failed_slot(data)
        detail = assert_coverage(item, aliases, aliases[:3])
        assert len(json.dumps(detail['missing_source_ids'])) > 240
        assert item['slot']['observations'][0]['source_ids'] == aliases[:3]
        assert offered_frozen_ids(item) == {PROOF}
        assert failed_slot(calls.requests[-1]['wire_data'])['obligation_review_sources'] == item['obligation_review_sources']
        return claim(irrelevant('reviewed-independent-context', aliases[3:52 if omit_last else 53]), refs=[PROOF])
    calls.respond = reply
    read_issue(reader, engine)
    assert len(calls.requests) == engine.ledger.used['llm'] == 2
    assert engine.ledger.used['json_repairs'] == 1
    assert engine.s == settings and engine.ledger.limits == limits
    assert engine.mapper.sources == original_sources
    assert state.valid()[0]['history'] == ['piano']
    assert set(state.valid()[1]['history']) == {engine.mapper.sources[a]['source_id'] for a in aliases[:3]}
    assert (OID in reader.evidence_obligations) is omit_last
    assert bool(state.pending) is omit_last
    assert bool(reader.failed_batch_reviews) is omit_last


def test_repair_does_not_implicitly_borrow_three_previously_validated_sources():
    reader, engine, state, calls, aliases = obligation_fixture()
    calls.respond = lambda index, data: (claim(observation(PROOF, aliases[:3])) if index == 1
        else claim(irrelevant('only-missing-fifty', aliases[3:53])))
    read_issue(reader, engine)
    assert OID in reader.evidence_obligations and state.pending == {'history'}
    assert len(calls.requests) == 2 and engine.ledger.used['json_repairs'] == 1


@pytest.mark.parametrize('mistake', ['current_new_row', 'unknown_id', 'known_route', 'nonsettled', 'changed_frozen'])
def test_only_unchanged_offered_local_settled_rows_can_be_explicitly_referenced(mistake):
    reader, engine, state, calls, aliases = obligation_fixture()
    before_routes = deepcopy(state.routes)
    known_route = next(iter(state.routes))
    def reply(index, data):
        if index == 1:
            rows = [observation(PROOF, aliases[:3])]
            if mistake == 'nonsettled':
                row = irrelevant('unsourced-unknown', [])
                row['stance'] = 'unknown'
                rows.append(row)
            return claim(*rows)
        assert index == 2
        offered = offered_frozen_ids(failed_slot(data))
        assert offered == {PROOF}
        rows = [irrelevant('independently-reviewed-fifty', aliases[3:53])]
        ref = {'current_new_row': 'new-in-this-response', 'unknown_id': 'invented',
               'known_route': known_route, 'nonsettled': 'unsourced-unknown',
               'changed_frozen': PROOF}[mistake]
        if mistake == 'current_new_row':
            rows.append(observation(ref, aliases[:3]))
        if mistake == 'changed_frozen':
            rows.append(observation(PROOF, aliases[:3], value=['painting']))
        return claim(*rows, refs=[ref])
    calls.respond = reply
    read_issue(reader, engine)
    assert OID in reader.evidence_obligations and state.pending == {'history'}
    assert state.valid()[0]['history'] == ['piano']
    assert all(state.routes[key] == value for key, value in before_routes.items())
    assert len(calls.requests) == 2 and engine.ledger.used['json_repairs'] == 1


def test_identical_scope_does_not_authorize_a_different_variables_frozen_observation():
    reader, engine, state, calls, aliases = obligation_fixture(other=True)
    def reply(index, data):
        if index == 1:
            result = claim(observation(PROOF, aliases[:3]))
            result['slots'].append({'variable': 'other', 'observations': [
                observation('other-variable-proof', aliases[:3])], 'unresolved': []})
            return result
        assert offered_frozen_ids(failed_slot(data)) == {PROOF}
        return claim(irrelevant('reviewed-fifty', aliases[3:53]), refs=['other-variable-proof'])
    calls.respond = reply
    read_issue(reader, engine, other=True)
    assert OID in reader.evidence_obligations and 'history' in state.pending
    assert state.valid()[0]['other'] == ['piano']
    assert len(calls.requests) == 2


def test_failure_keeps_exact_durable_sources_but_does_not_grant_free_or_cross_batch_refs():
    reader, engine, state, calls, aliases = obligation_fixture()
    calls.respond = lambda index, data: claim(observation(PROOF, aliases[:3]))
    read_issue(reader, engine)
    assert len(calls.requests) == 2 and OID in reader.evidence_obligations
    checkpoint = json.loads(json.dumps(reader.obligation_snapshot()))
    entry = next(item for record in checkpoint['failed_batch_reviews'] for item in record['failed'] if item['id'] == '*')
    stable_required = {engine.mapper.sources[a]['source_id'] for a in aliases[:53]}
    assert set(entry['obligation_review_sources'][0]['required_source_ids']) == stable_required
    assert 'frozen_observation_reviews' not in entry
    restored = JointReader(engine, state)
    restored.restore_obligations(checkpoint)
    used = deepcopy(engine.ledger.used)
    assert read_issue(restored, engine, identity='same-originals-renamed') is False
    assert len(calls.requests) == 2 and engine.ledger.used == used
    def reply(index, data):
        assert index == 3
        assert 'repair_scope' not in data
        assert any(record['review_allowed'] for record in data['failed_batch_reviews'])
        return claim(irrelevant('new-batch-fifty', aliases[3:53]), refs=[PROOF])
    calls.respond = reply
    # Existing global repair budget permits one paid recovery. Its first reply
    # still cannot cite a local frozen ID from the previous batch/checkpoint.
    read_issue(restored, engine, identity='new-original-recovery', new=True)
    assert len(calls.requests) == 3
    assert engine.ledger.used['json_repairs'] == used['json_repairs'] + 1
    assert OID in restored.evidence_obligations and 'history' in state.pending


def test_frozen_reference_cannot_reintroduce_an_original_missing_from_actual_repair_payload(monkeypatch):
    reader, engine, state, calls, aliases = obligation_fixture()
    original_payload = reader._payload
    initial_answered = False
    def payload(*args, **kwargs):
        data = original_payload(*args, **kwargs)
        if initial_answered:
            data['source_spans'] = [span for span in data['source_spans'] if span['id'] != aliases[0]]
        return data
    monkeypatch.setattr(reader, '_payload', payload)
    def reply(index, data):
        nonlocal initial_answered
        if index == 1:
            initial_answered = True
            return claim(observation(PROOF, aliases[:3]))
        assert index == 2 and aliases[0] not in {span['id'] for span in data['source_spans']}
        return claim(irrelevant('reviewed-fifty', aliases[3:53]), refs=[PROOF])
    calls.respond = reply
    try:
        read_issue(reader, engine)
    except ProtocolError:
        pass  # Rejecting the inconsistent repair before generation is also valid.
    assert OID in reader.evidence_obligations and state.pending == {'history'}
    assert state.valid()[0]['history'] == ['piano']
    assert len(calls.requests) <= 2


@pytest.mark.parametrize('tamper', ['wire_observation', 'memory_offer'])
def test_explicit_reference_is_bound_to_both_unchanged_transmitted_row_and_local_candidate(monkeypatch, tamper):
    reader, engine, state, calls, aliases = obligation_fixture()
    original_offer = reader._offer_frozen_reviews
    def offer(current, valid, frozen, feedback):
        offered = original_offer(current, valid, frozen, feedback)
        assert ('history', PROOF) in offered
        if tamper == 'memory_offer':
            offered['history', PROOF]['value'] = ['painting']
        else:
            item = failed_slot(current)
            assert PROOF in item['frozen_observation_reviews']['slot_observation_ids']
            next(row for row in item['slot']['observations'] if row['id'] == PROOF)['value'] = ['painting']
        return offered
    monkeypatch.setattr(reader, '_offer_frozen_reviews', offer)
    calls.respond = lambda index, data: (claim(observation(PROOF, aliases[:3])) if index == 1
        else claim(irrelevant('reviewed-fifty', aliases[3:53]), refs=[PROOF]))
    read_issue(reader, engine)
    assert OID in reader.evidence_obligations and state.pending == {'history'}
    assert state.valid()[0]['history'] == ['piano']
    assert len(calls.requests) == 2 and engine.ledger.used['json_repairs'] == 1


def test_trusted_obligation_feedback_remaps_strictly_without_trusting_nested_model_echo():
    detail = [{'obligation_id': OID, 'required_source_ids': ['stable-one', 'stable-two'],
               'covered_source_ids': ['stable-one'], 'missing_source_ids': ['stable-two']}]
    entry = {'id': '*', 'obligation_review_sources': detail}
    with pytest.raises(ProtocolError):
        JointReader._remap_failed_entry_sources(entry, {'stable-one': 'a1'})
    mapped = JointReader._remap_failed_entry_sources(entry, {'stable-one': 'a1', 'stable-two': 'a2'})
    assert mapped['obligation_review_sources'] == [{
        'obligation_id': OID, 'required_source_ids': ['a1', 'a2'],
        'covered_source_ids': ['a1'], 'missing_source_ids': ['a2']}]
    untrusted = {'id': '*', 'slot': {'obligation_review_sources': deepcopy(detail)}}
    retained = JointReader._remap_failed_entry_sources(untrusted, {})
    assert 'obligation_review_sources' not in retained
    assert retained['slot']['obligation_review_sources'] == detail


def test_obligation_snapshot_change_invalidates_previously_computed_coverage_feedback(monkeypatch):
    reader, engine, state, calls, aliases = obligation_fixture()
    changed = False
    def handoff(*args, **kwargs):
        nonlocal changed
        if not changed:
            reader.evidence_obligations[OID]['descriptions'].append('A newly retained qualification remains required.')
            changed = True
        return {}
    monkeypatch.setattr(reader, '_handoff_failures', handoff)
    def reply(index, data):
        if index == 1:
            return claim(observation(PROOF, aliases[:3]))
        assert index == 2
        assert 'obligation_review_sources' not in failed_slot(data)
        result = claim()
        result['slots'][0]['unresolved'] = ['The changed scoped issue remains unresolved.']
        return result
    calls.respond = reply
    read_issue(reader, engine)
    assert changed and OID in reader.evidence_obligations and state.pending == {'history'}
    assert len(calls.requests) == 2


def test_changed_failure_cannot_reuse_prior_coverage_feedback():
    reader, engine, state, calls, aliases = obligation_fixture(repairs=2)
    def reply(index, data):
        if index == 1:
            return claim(observation(PROOF, aliases[:3]))
        item = failed_slot(data)
        if index == 2:
            assert_coverage(item, aliases, aliases[:3])
        else:
            assert index == 3 and 'obligation_review_sources' not in item
            assert item['slot']['unresolved'] == ['The scoped issue remains unresolved.']
        result = claim(irrelevant('all-context-reviewed', aliases[:53]))
        result['slots'][0]['unresolved'] = ['The scoped issue remains unresolved.']
        return result
    calls.respond = reply
    read_issue(reader, engine)
    assert OID in reader.evidence_obligations and state.pending == {'history'}
    assert len(calls.requests) == 3 and engine.ledger.used['json_repairs'] == 2
    assert all('obligation_review_sources' not in item for record in reader.failed_batch_reviews.values()
               for item in record['failed'])


def test_successful_coverage_does_not_bypass_real_downstream_handoff_rollback(monkeypatch):
    reader, engine, state, calls, aliases = obligation_fixture()
    before = deepcopy(state.routes)
    def reject(rows, rejectable, *args, **kwargs):
        return {(row['variable'], row['id']): 'citation_capacity: synthetic downstream refusal'
                for row in rows if row['id'] == 'new-minimal-proof'
                and (row['variable'], row['id']) in rejectable}
    monkeypatch.setattr(reader, '_handoff_failures', reject)
    def reply(index, data):
        if index == 1:
            return claim(irrelevant(PROOF, aliases[:3]))
        assert_coverage(failed_slot(data), aliases, aliases[:3])
        return claim(observation('new-minimal-proof', aliases[:1]),
                     irrelevant('reviewed-fifty', aliases[3:53]), refs=[PROOF])
    calls.respond = reply
    read_issue(reader, engine)
    assert reader.evidence_obligations[OID]['source_ids'] == [
        engine.mapper.sources[a]['source_id'] for a in aliases[:53]]
    assert state.routes == before and state.pending == {'history'}
    assert len(calls.requests) == 2 and engine.ledger.used['json_repairs'] == 1
    assert any('citation_capacity:' in failure['error'] for failure in reader.batches[-1]['failed_slots'])
