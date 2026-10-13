"""Independent discovery-frontier boundaries; real state, scripted transport.

These tests make no semantic-accuracy claim. All observations are synthetic and
network calls are forbidden; admission can be bounded without token services.
"""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from dagbt.joint_reading import JointReader
from dagbt.fact_state import FactState
from dagbt.evidence_spans import build_source_spans
from dagbt.reasoning import InputOverflow, ProtocolError
from dagbt.residual_control import ResidualControl
from test_personamem_joint_protocol import fixture, observation, response, visible
from test_residual_review_regressions import engine, local_read, number_plan, prepare
from test_v4_control import ControlServices, advice_plan
from test_v4_failed_batch_review import failed_whole, whole_review_response


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Discovery frontier review must remain offline')
    monkeypatch.setattr('urllib.request.urlopen', forbidden)
    monkeypatch.setattr('urllib.request.build_opener', forbidden)


def numeric_discovery():
    e, services = engine(number_plan(), {'x': 'x = 7.', 'unused': 'Unexamined decorative context.'})
    control = prepare(e)
    step, query, identity = local_read(control, 'x', e.ids)
    return e, services, control, step, query, identity


def test_successful_numeric_batch_returns_before_unoffered_candidate_without_claiming_it_read():
    e, _, control, step, query, identity = numeric_discovery()
    reader = control.reader
    reader._fits = lambda operation, data: operation != 'joint_read' or sum(
        (identity, source['id']) not in reader.read_states for source in data['source_spans']) <= 1
    assert reader.read(step, query, ['x'], e.ids, identity, discovery_doc_ids=e.ids)
    assert control.state.valid()[0]['x'] == 7
    assert len(reader.batches) == 1, 'Return after one atomic discovery batch to re-evaluate the actual task'
    remaining = next(alias for alias, source in e.mapper.sources.items() if source['doc_id'] == 'unused')
    assert (identity, remaining) not in reader.read_states
    assert (identity, remaining) in reader.discovery_frontier
    assert 'x' not in control.state.pending
    assert control.candidate()['output'] == 7


def test_unoffered_suffix_capacity_does_not_retroactively_block_a_sufficient_numeric_binding():
    e, _, control, step, query, identity = numeric_discovery()
    reader = control.reader
    originals = deepcopy(e.mapper.sources)
    reader._fits = lambda operation, data: operation != 'joint_read' or (
        not reader.batches and len(data['source_spans']) <= 1)
    assert reader.read(step, query, ['x'], e.ids, identity, discovery_doc_ids=e.ids)
    assert len(reader.batches) == 1
    assert control.state.valid()[0]['x'] == 7 and 'x' not in control.state.pending
    assert control.candidate()['output'] == 7
    assert e.mapper.sources == originals


def history_discovery(reader, engine, *, identity='discovery', docs=None, variables=None):
    docs = list(engine.docs) if docs is None else docs
    return reader.read(engine.steps[0], 'Read the scoped history', variables or ['history'], docs, identity,
                       discovery_doc_ids=docs)


def one_fresh_history_source(reader, identity='discovery'):
    reader._fits = lambda operation, data: operation != 'joint_read' or sum(
        (identity, source['id']) not in reader.read_states for source in data['source_spans']) <= 1


def test_entire_successful_batch_integrates_later_counter_before_yielding():
    reader, e, state, calls = fixture(max_repairs_per_request=0, context_tokens=100000)
    prior = reader._observation(observation('old-piano', ['s1']), 'history', visible(reader), e.steps[0])
    state.integrate([prior])
    old = state.valid()[2]['history'][0]
    reader._fits = lambda operation, data: operation != 'joint_read' or len(data['source_spans']) <= 2

    def reply(index, data):
        assert index == 1 and {source['id'] for source in data['source_spans']} == {'s1', 's2'}
        correction = observation('counter-after-support', ['s2'], stance='contradiction', retract_ids=[old])
        correction['value'] = None
        return response(observation('new-painting', ['s2'], value=['painting']), correction)

    calls.respond = reply
    history_discovery(reader, e)
    assert len(calls.requests) == 1
    assert state.routes[old]['revoked'], 'A later same-response counter cannot be skipped after the support row'
    assert state.valid()[0]['history'] == ['painting']
    assert not state.pending
    assert ('discovery', 's3') in reader.discovery_frontier
    assert ('discovery', 's3') not in reader.read_states


@pytest.mark.parametrize('problem', ['partial', 'unresolved', 'invalid_source'])
def test_unoffered_frontier_never_discharges_observed_hard_history_problem(problem):
    reader, e, state, calls = fixture(max_repairs_per_request=0, context_tokens=100000)
    one_fresh_history_source(reader)
    if problem == 'partial':
        draft = observation('located-required-detail', ['s1'], stance='partial',
                            reason='This original points to another document needed to establish the fact.')
        draft['value'] = None
        calls.respond = lambda index, data: response(draft)
    elif problem == 'unresolved':
        calls.respond = lambda index, data: {'slots': [{'variable': 'history', 'observations': [],
            'unresolved': ['The visible original leaves a specific required condition unresolved.']}]}
    else:
        calls.respond = lambda index, data: response(observation('invalid-proof', ['unavailable']))
    history_discovery(reader, e)
    assert len(calls.requests) == 1
    assert 'history' in state.pending and 'history' in reader.pending_variables()
    assert reader.failed_batch_reviews if problem == 'invalid_source' else reader.evidence_obligations
    assert ('discovery', 's2') in reader.discovery_frontier
    assert ('discovery', 's2') not in reader.read_states


def test_plain_legacy_unread_checkpoint_cannot_be_inferred_to_be_a_pure_frontier():
    reader, e, state, calls = fixture(max_repairs_per_request=0, context_tokens=100000)
    old = {'unread_obligations': [['old-owner', 's3', ['history']]],
           'failed_obligations': [], 'evidence_obligations': [], 'failed_batch_reviews': []}
    reader.restore_obligations(json.loads(json.dumps(old)))
    state.integrate([], reader.pending_variables())
    assert reader.unread_obligations[('old-owner', 's3')] == {'history'}
    assert ('old-owner', 's3') not in reader.discovery_frontier
    one_fresh_history_source(reader)
    history_discovery(reader, e, docs=['piano'])
    assert state.valid()[0]['history'] == ['piano']
    assert 'history' in state.pending
    assert reader.unread_obligations[('old-owner', 's3')] == {'history'}
    assert ('old-owner', 's3') not in reader.discovery_frontier


def test_mandatory_common_premises_remain_one_hard_batch_when_capacity_is_insufficient():
    reader, e, state, calls = fixture(max_repairs_per_request=0, context_tokens=100000)
    state.integrate([reader._observation(observation('joint-proof', ['s1', 's2']), 'history',
                     visible(reader, ['s1', 's2']), e.steps[0])])
    originals = deepcopy(state.routes)
    reader._fits = lambda operation, data: False
    with pytest.raises(InputOverflow):
        reader.read(e.steps[1], 'Use both original premises', ['suggestion'], [], 'mandatory',
                    semantic=True, source_scope='mandatory_closure')
    assert not calls.requests and state.routes == originals
    assert set(reader.unread_obligations) == {('mandatory', 's1'), ('mandatory', 's2')}
    assert not reader.discovery_frontier
    assert {'history', 'suggestion'} <= state.pending


def test_discovery_provenance_cannot_be_attached_to_mandatory_scope():
    reader, e, state, calls = fixture(max_repairs_per_request=0)
    before = deepcopy(state.public_dict())
    with pytest.raises(ProtocolError):
        reader.read(e.steps[1], 'Do not split a mandatory closure', ['suggestion'], [], 'mandatory',
                    semantic=True, source_scope='mandatory_closure', discovery_doc_ids=['piano'])
    assert state.public_dict() == before and not calls.requests
    assert not reader.discovery_frontier and not reader.unread_obligations


def test_new_identity_and_frontier_registration_cannot_buy_same_source_failed_recovery():
    reader, e, state, calls = fixture(max_repairs_per_request=1, context_tokens=100000)
    calls.respond = lambda index, data: response(observation('bad-proof', ['unavailable']))
    history_discovery(reader, e, docs=['piano'])
    assert reader.failed_batch_reviews and 'history' in state.pending
    requests = len(calls.requests)
    used = deepcopy(e.ledger.used)
    failed = deepcopy(reader.failed_batch_reviews)
    for identity in ['renamed', 'renamed-again']:
        assert history_discovery(reader, e, docs=['piano'], identity=identity) is False
    assert len(calls.requests) == requests
    assert e.ledger.used == used and reader.failed_batch_reviews == failed
    assert 'history' in state.pending


def test_actual_symbolic_controller_stops_with_frontier_only_after_real_audit(monkeypatch):
    e, services, control, _, _, identity = numeric_discovery()
    one_fresh_history_source(control.reader, identity)
    monkeypatch.setattr(e.bridge, 'step', lambda *args, **kwargs: pytest.fail('Local candidates suffice; no new ANN'))
    row = control.run()
    assert row['answer']['status'] == 'ok'
    assert row['answer']['completion_kind'] == 'certified_symbolic'
    assert row['answer']['semantic_answer'] == 7
    operations = [request['stage'][0] for request in services.requests]
    assert operations.count('joint_read') == operations.count('audit_joint') == 1
    assert control.certificate and e.audit_complete
    unused = next(alias for alias, source in e.mapper.sources.items() if source['doc_id'] == 'unused')
    assert (identity, unused) in control.reader.discovery_frontier
    assert (identity, unused) not in control.reader.read_states


def test_actual_semantic_controller_requires_terminal_and_audit_with_numeric_history_frontier(monkeypatch):
    plan = advice_plan()
    services = ControlServices(plan)
    e, _ = engine(plan, {'x': 'Preference 0: quiet places.', 'unused': 'Unexamined decorative context.'}, services)
    control = prepare(e)
    _, _, identity = local_read(control, 'preference', e.ids)
    one_fresh_history_source(control.reader, identity)
    monkeypatch.setattr(e.bridge, 'step', lambda *args, **kwargs: pytest.fail('The actual local numeric support suffices'))
    row = control.run()
    assert row['answer']['status'] == 'ok'
    assert row['answer']['completion_kind'] == 'semantic_assessment'
    assert row['answer']['certificate'] is None
    assert services.compositions == services.audits == 1 and e.audit_complete
    assert control.reader.semantic_inputs['validation_complete']
    assert control.semantic_inputs_current()
    unused = next(alias for alias, source in e.mapper.sources.items() if source['doc_id'] == 'unused')
    assert (identity, unused) in control.reader.discovery_frontier
    assert (identity, unused) not in control.reader.read_states


def test_controller_resumes_existing_local_frontier_before_any_new_ann(monkeypatch):
    e, services = engine(number_plan(), {'clue': 'No value here.', 'x': 'x = 7.'})
    control = prepare(e)
    _, _, identity = local_read(control, 'x', e.ids)
    one_fresh_history_source(control.reader, identity)
    monkeypatch.setattr(e.bridge, 'step', lambda *args, **kwargs: pytest.fail('Read the retained local suffix before searching'))
    row = control.run()
    assert row['answer']['status'] == 'ok' and row['answer']['semantic_answer'] == 7
    operations = [request['stage'][0] for request in services.requests]
    assert operations.count('joint_read') == 2 and operations.count('audit_joint') == 1
    assert not control.state.pending
    assert not control.reader.discovery_frontier
    assert not control.reader.evidence_obligations


def test_new_frontier_checkpoint_preserves_unoffered_status_and_can_resume_without_planner(monkeypatch):
    e, _, control, step, query, identity = numeric_discovery()
    one_fresh_history_source(control.reader, identity)
    control.reader.read(step, query, ['x'], e.ids, identity, discovery_doc_ids=e.ids)
    snapshot = json.loads(json.dumps(control.snapshot()))
    old_used = deepcopy(e.ledger.used)
    fresh, services = engine(number_plan(), {'x': 'x = 7.', 'unused': 'Unexamined decorative context.'})
    with monkeypatch.context() as paused:
        paused.setattr(ResidualControl, 'run', lambda current: current.publish())
        fresh.resume(snapshot)
    restored = fresh.residual_control
    assert restored.reader.discovery_frontier == control.reader.discovery_frontier
    assert restored.reader.source_exposures == control.reader.source_exposures
    assert restored.reader.read_states == control.reader.read_states
    assert restored.reader.unread_obligations == control.reader.unread_obligations
    assert not restored.state.pending
    assert fresh.ledger.used == old_used
    request_count = len(services.requests)
    monkeypatch.setattr(fresh.bridge, 'step', lambda *args, **kwargs: pytest.fail('Restored sufficient binding requires no search'))
    row = restored.run()
    assert row['answer']['status'] == 'ok'
    new_operations = [request['stage'][0] for request in services.requests[request_count:]]
    assert new_operations == ['audit_joint']
    assert restored.reader.discovery_frontier
    assert not any(request['stage'][0] == 'planner' for request in services.requests)


def test_revisiting_identical_completed_window_does_not_create_a_read_cycle_or_extra_cost():
    reader, e, state, calls = fixture(max_repairs_per_request=0, context_tokens=100000)
    history_discovery(reader, e, docs=['piano'])
    assert state.valid()[0]['history'] == ['piano'] and not state.pending
    before = deepcopy((reader.material_signature(), reader.obligation_snapshot(), state.public_dict(), e.ledger.used))
    count = len(calls.requests)
    for _ in range(4):
        assert history_discovery(reader, e, docs=['piano']) is False
    assert len(calls.requests) == count
    assert (reader.material_signature(), reader.obligation_snapshot(), state.public_dict(), e.ledger.used) == before


def test_discovery_origin_is_explicit_and_does_not_soften_other_unknown_origin_documents():
    reader, e, state, calls = fixture(max_repairs_per_request=0, context_tokens=100000)
    one_fresh_history_source(reader)
    reader.read(e.steps[0], 'Read one discovered source with other required work', ['history'], list(e.docs),
                'discovery', discovery_doc_ids=['piano'])
    assert len(calls.requests) == 1 and state.valid()[0]['history'] == ['piano']
    assert 'history' in state.pending
    assert {('discovery', 's2'), ('discovery', 's3')} <= set(reader.unread_obligations)
    assert ('discovery', 's2') not in reader.discovery_frontier
    assert ('discovery', 's3') not in reader.discovery_frontier


def test_mixed_variables_keep_only_real_observed_gap_hard_after_one_complete_batch():
    _, e, initial, calls = fixture(max_repairs_per_request=0, context_tokens=100000)
    program = deepcopy(initial.initial_program)
    program['variables']['detail'] = {'type': 'text', 'demand': 'history'}
    state = FactState(program, initial.contract, e.steps, initial.final_node_id)
    reader = JointReader(e, state)
    one_fresh_history_source(reader)
    partial = observation('required-detail', ['s1'], stance='partial', reason='The cited original leaves one requested detail open.')
    partial['value'] = None
    calls.respond = lambda index, data: {'slots': [
        {'variable': 'history', 'observations': [observation('history-proof', ['s1'])], 'unresolved': []},
        {'variable': 'detail', 'observations': [partial], 'unresolved': []}]}
    history_discovery(reader, e, variables=['history', 'detail'])
    assert len(calls.requests) == 1
    assert state.valid()[0]['history'] == ['piano']
    assert state.pending == {'detail'}
    assert {item['variable'] for item in reader.evidence_obligations.values()} == {'detail'}
    assert ('discovery', 's2') in reader.discovery_frontier
    assert ('discovery', 's2') not in reader.read_states


def test_legacy_direct_reader_does_not_infer_discovery_provenance_from_document_presence():
    reader, e, state, calls = fixture(max_repairs_per_request=0, context_tokens=100000)
    one_fresh_history_source(reader, 'legacy')
    reader._fits = lambda operation, data: operation != 'joint_read' or (
        not reader.batches and len(data['source_spans']) <= 1)
    with pytest.raises(InputOverflow):
        reader.read(e.steps[0], 'Legacy source scope is unknown', ['history'], list(e.docs), 'legacy')
    assert len(calls.requests) == 1 and state.valid()[0]['history'] == ['piano']
    assert state.pending == {'history'}
    assert not reader.discovery_frontier
    assert {('legacy', 's2'), ('legacy', 's3')} <= set(reader.unread_obligations)


def append_document(engine, identifier, text):
    document = SimpleNamespace(doc_id=identifier, passage=text)
    engine.docs[identifier] = document
    aliases = []
    for span in build_source_spans(identifier, document, engine.s['max_quote_chars']):
        alias = 's' + str(len(engine.mapper.sources) + 1)
        engine.mapper.source_aliases[span['id']] = alias
        engine.mapper.sources[alias] = {**span, 'id': alias, 'source_id': span['id']}
        aliases.append(alias)
    return aliases


def test_paid_failed_review_and_bounded_repair_record_originals_beyond_primary_discovery_window():
    reader, e, state, calls = failed_whole(max_repairs=2)
    unseen = append_document(e, 'unoffered', 'Unexamined optional background.')[0]
    previous_calls = len(calls.requests)
    previous_exposures = len(reader.source_exposures)
    previous_repairs = e.ledger.used['json_repairs']
    originals = deepcopy(e.mapper.sources)
    reader._fits = lambda operation, data: len(data['source_spans']) <= 3

    def review(index, data):
        assert {source['id'] for source in data['source_spans']} == {'s1', 's2', 's3'}
        result = whole_review_response(data)
        if index == previous_calls + 1:
            result['slots'][0]['observations'][0]['entity_scope'] = 'mistyped-user'
        else:
            assert 'repair_scope' in data
            failed_ids = {row['id'] for row in data['repair_scope']['failed']}
            result['slots'][0]['observations'] = [row for row in result['slots'][0]['observations']
                                                 if row['id'] in failed_ids]
        return result

    calls.respond = review
    history_discovery(reader, e, identity='paid-recovery', docs=['advice', 'unoffered'])
    assert len(calls.requests) == previous_calls + 2
    assert e.ledger.used['json_repairs'] == previous_repairs + 2
    exposures = reader.source_exposures[previous_exposures:]
    assert len(exposures) == 2
    for exposed, request in zip(exposures, calls.requests[previous_calls:]):
        actual = {source['id'] for source in request['data']['source_spans']}
        assert {source['source_alias'] for source in exposed['sources']} == actual
        assert all(reader.read_states.get(('paid-recovery', alias)) == 'read' for alias in actual)
        assert not any(('paid-recovery', alias) in reader.discovery_frontier for alias in actual)
        assert exposed['request_status'] == 'returned' and exposed['status'] == 'batch_assessed'
    assert reader.batches[-1]['source_aliases'] == ['s3'], 'Failed-review originals are outside this primary window'
    assert ('paid-recovery', unseen) in reader.discovery_frontier
    assert ('paid-recovery', unseen) not in reader.read_states
    assert not state.pending and not reader.failed_batch_reviews and not reader.failed_obligations
    assert state.valid()[0]['history'] == ['piano', 'painting']
    assert e.mapper.sources == originals


@pytest.mark.parametrize('reuse', [None, 'already_read_document', 'new_document'])
def test_actual_parent_revision_invalidates_old_descendant_and_retained_frontier_keeps_scope_history(reuse):
    _, e, initial, calls = fixture(max_repairs_per_request=0, context_tokens=100000)
    steps = deepcopy(e.steps)
    steps[0]['inputs'] = ['context']
    steps.insert(0, {'output_slot': 'context', 'question': 'Establish the relevant context',
                     'inputs': [], 'execution': 'retrieval', 'answer_type': 'fact'})
    program = deepcopy(initial.program)
    program['variables'] = {'context': {'type': 'text', 'demand': 'context'}, **program['variables']}
    state = FactState(program, initial.contract, steps, initial.final_node_id)
    e.steps = steps
    reader = JointReader(e, state)
    sid = e.mapper.sources['s1']['source_id']
    state.integrate([{**observation('old-context', [sid], value='Initial context'), 'variable': 'context'}])
    old_context = state.valid()[2]['context'][0]
    old_version = state.versions['context']
    reader._fits = lambda operation, data: len(data['source_spans']) <= 2

    def answer(identifier, sources):
        return {'slots': [
            {'variable': 'history', 'observations': [observation(identifier, sources,
                value=['painting'], parent_versions={'context': state.versions['context']})], 'unresolved': []},
            {'variable': 'context', 'observations': [], 'unresolved': []}]}

    calls.respond = lambda index, data: answer('history-before-change', ['s2'])
    reader.read(steps[1], 'Read with initial context', ['history'], ['painting', 'advice'], 'old-context-query',
                discovery_doc_ids=['painting', 'advice'])
    assert state.valid()[0]['history'] == ['painting'] and not state.pending
    previous_frontier = deepcopy(reader.discovery_frontier['old-context-query', 's3'])
    assert previous_frontier['parent_versions'] == {'context': old_version}
    previous_exposure = deepcopy(reader.source_exposures[-1])
    assert previous_exposure['parent_versions'] == {'context': old_version}
    correction = {**observation('context-corrected', [sid], stance='contradiction',
                    retract_ids=[old_context]), 'variable': 'context', 'value': None}
    state.integrate([correction, {**observation('new-context', [sid], value='Revised context'),
                                  'variable': 'context'}])
    assert state.versions['context'] != old_version
    assert 'history' not in state.valid()[0], 'The old descendant cannot survive its actual parent correction'
    reader._fits = lambda operation, data: True
    if reuse is not None:
        docs = ['painting']
        selected = 's2'
        if reuse == 'new_document':
            docs = ['brand-new']
            selected = append_document(e, 'brand-new', 'The relevant context has changed.')[0]
        calls.respond = lambda index, data: answer('changed-history-same-identity', [selected])
        before = deepcopy((state.public_dict(), reader.obligation_snapshot(), e.ledger.used, calls.requests))
        with pytest.raises(ProtocolError, match='identity'):
            reader.read(steps[1], 'Read with revised context', ['history'], docs, 'old-context-query',
                        discovery_doc_ids=docs)
        assert (state.public_dict(), reader.obligation_snapshot(), e.ledger.used, calls.requests) == before
        return
    calls.respond = lambda index, data: answer('history-after-change', ['s3'])
    before_calls = len(calls.requests)
    reader.read(steps[1], 'Read with revised context', ['history'], ['advice'], 'new-context-query',
                discovery_doc_ids=['advice'])
    assert len(calls.requests) == before_calls + 1
    assert state.valid()[0]['history'] == ['painting'] and not state.pending
    assert reader.discovery_frontier['old-context-query', 's3'] == previous_frontier
    assert reader.source_exposures[-2] == previous_exposure
    assert ('new-context-query', 's3') in reader.read_states
    assert ('old-context-query', 's3') not in reader.read_states
    assert reader.source_exposures[-1]['parent_versions'] == {'context': state.versions['context']}
    assert not any(owner == 'new-context-query' for owner, _ in reader.discovery_frontier)
    checkpoint = json.loads(json.dumps(reader.obligation_snapshot()))
    restored = JointReader(e, state)
    restored.restore_obligations(checkpoint)
    assert restored.discovery_frontier == reader.discovery_frontier
    assert restored.source_exposures == reader.source_exposures
    assert restored.discovery_frontier['old-context-query', 's3']['parent_versions'] != {
        'context': state.versions['context']}, 'A historical frontier records its old scope, not the current parent'


@pytest.mark.parametrize('corruption', ['missing_version', 'future_version', 'changed_source', 'already_hard',
                                        'missing_context_revision', 'different_variable_scope'])
def test_ambiguous_frontier_checkpoint_never_silently_becomes_soft_discovery(corruption):
    reader, e, _, _ = fixture(max_repairs_per_request=0, context_tokens=100000)
    one_fresh_history_source(reader)
    history_discovery(reader, e)
    checkpoint = json.loads(json.dumps(reader.obligation_snapshot()))
    if corruption == 'missing_version':
        checkpoint.pop('frontier_policy_version')
    elif corruption == 'future_version':
        checkpoint['frontier_policy_version'] += 1
    elif corruption == 'changed_source':
        checkpoint['discovery_frontier'][0][2]['source']['source_identity'] = 'different-original'
    elif corruption == 'missing_context_revision':
        checkpoint['discovery_frontier'][0][2].pop('program_revision')
    elif corruption == 'different_variable_scope':
        checkpoint['discovery_frontier'][0][2]['variables'] = ['another-variable']
    else:
        identity, alias, _ = checkpoint['discovery_frontier'][0]
        checkpoint['unread_obligations'].append([identity, alias, ['history']])
    fresh = JointReader(e, reader.state)
    with pytest.raises(ProtocolError):
        fresh.restore_obligations(checkpoint)


def test_cross_query_unoffered_scope_preserves_prior_actual_exposure_of_uncited_source():
    reader, e, state, calls = fixture(max_repairs_per_request=0, context_tokens=100000)
    history_discovery(reader, e, identity='previous-query', docs=['piano', 'painting'])
    assert {source['id'] for source in calls.requests[-1]['data']['source_spans']} == {'s1', 's2'}
    assert state.valid()[1]['history'] == [e.mapper.sources['s1']['source_id']]
    assert reader.read_states['previous-query', 's2'] == 'read'
    history = deepcopy(reader.source_exposures)
    reader._fits = lambda operation, data: len(data['source_spans']) <= 1
    history_discovery(reader, e, identity='new-query')
    fresh_work = reader.discovery_frontier['new-query', 's2']
    assert fresh_work['query_identity'] == 'new-query' and fresh_work['status'] == 'unoffered'
    assert ('new-query', 's2') not in reader.read_states
    assert reader.read_states['previous-query', 's2'] == 'read'
    assert reader.source_exposures[:-1] == history
    assert any(source['source_alias'] == 's2' for source in history[0]['sources'])
    checkpoint = json.loads(json.dumps(reader.obligation_snapshot()))
    restored = JointReader(e, state)
    restored.restore_obligations(checkpoint)
    assert restored.discovery_frontier == reader.discovery_frontier
    assert restored.source_exposures == reader.source_exposures


def test_same_query_candidate_that_becomes_a_required_proof_retains_discovery_origin_when_promoted():
    reader, e, state, calls = fixture(max_repairs_per_request=0, context_tokens=100000)
    one_fresh_history_source(reader)
    history_discovery(reader, e)
    original = deepcopy(reader.discovery_frontier['discovery', 's2'])
    proof = reader._observation(observation('additional-current-proof', ['s2']),
                                'history', visible(reader, ['s2']), e.steps[0])
    state.integrate([proof], reader.pending_variables())
    reader._fits = lambda operation, data: len(data['source_spans']) <= 2
    history_discovery(reader, e)
    assert {source['id'] for source in calls.requests[-1]['data']['source_spans']} == {'s1', 's2'}
    assert ('discovery', 's2') not in reader.discovery_frontier
    assert reader.read_states['discovery', 's2'] == 'read'
    assert {'status': 'required_evidence', 'origin': original} in reader.discovery_promotions
    assert ('discovery', 's3') in reader.discovery_frontier and ('discovery', 's3') not in reader.read_states
    assert not state.pending


def promoted_frontier_checkpoint():
    reader, e, state, calls = fixture(max_repairs_per_request=0, context_tokens=100000)
    append_document(e, 'last', 'Another unexamined candidate.')
    one_fresh_history_source(reader)
    history_discovery(reader, e)
    state.integrate([reader._observation(observation('second-proof', ['s2']), 'history',
                     visible(reader, ['s2']), e.steps[0])], reader.pending_variables())
    reader._fits = lambda operation, data: len(data['source_spans']) <= 2
    history_discovery(reader, e)
    assert list(reader.discovery_frontier) == [('discovery', 's3'), ('discovery', 's4')]
    assert reader.discovery_promotions and reader.source_exposures
    checkpoint = json.loads(json.dumps({**reader.obligation_snapshot(),
        'read_states': [[*key, value] for key, value in reader.read_states.items()], 'clues': reader.clues}))
    return reader, e, state, calls, checkpoint


@pytest.mark.parametrize('changed', [None, 'exposed_text', 'promoted_role', 'frontier_time'])
def test_checkpoint_alias_permutation_rebinds_stable_sources_without_rewriting_history(changed):
    reader, e, state, calls, checkpoint = promoted_frontier_checkpoint()
    rename = {'s1': 's3', 's2': 's4', 's3': 's1', 's4': 's2'}
    old_sources = deepcopy(e.mapper.sources)
    e.mapper.sources = {rename[alias]: {**source, 'id': rename[alias]}
                         for alias, source in reversed(list(old_sources.items()))}
    e.mapper.source_aliases = {source['source_id']: alias for alias, source in e.mapper.sources.items()}
    if changed == 'exposed_text':
        e.mapper.sources[rename['s1']]['text'] = 'This original text was changed.'
    elif changed == 'promoted_role':
        e.mapper.sources[rename['s2']]['source_role'] = 'assistant'
    elif changed == 'frontier_time':
        e.mapper.sources[rename['s3']]['time_metadata'] = {'date': '2026-10-15'}
    restored = JointReader(e, state)
    if changed is not None:
        with pytest.raises(ProtocolError):
            restored.restore_obligations(checkpoint)
        return
    rebound = restored.restore_obligations(checkpoint)
    assert list(restored.discovery_frontier) == [('discovery', rename['s3']), ('discovery', rename['s4'])]
    for (_, old_alias), (_, new_alias) in zip(reader.discovery_frontier, restored.discovery_frontier):
        old = reader.discovery_frontier['discovery', old_alias]
        new = restored.discovery_frontier['discovery', new_alias]
        assert new['query_identity'] == old['query_identity']
        assert new['source']['source_id'] == old['source']['source_id']
        assert new['source']['source_identity'] == old['source']['source_identity']
        assert new['source']['source_alias'] == rename[old_alias]
        assert old_alias in new['source']['source_alias_history']
        assert new['variables'] == old['variables'] and new['scopes'] == old['scopes']
    assert len(restored.source_exposures) == len(reader.source_exposures)
    for old, new in zip(reader.source_exposures, restored.source_exposures):
        assert new['request_identity'] == old['request_identity'], 'Historical wire identity must not be regenerated'
        assert new['query_identity'] == old['query_identity'] and new['status'] == old['status']
        assert [source['source_id'] for source in new['sources']] == [source['source_id'] for source in old['sources']]
        assert [source['source_alias'] for source in new['sources']] == [source['source_alias'] for source in old['sources']]
        assert [source['current_source_alias'] for source in new['sources']] == [rename[source['source_alias']] for source in old['sources']]
        assert not any((new['query_identity'], source['current_source_alias']) in restored.discovery_frontier
                       for source in new['sources'])
    assert restored.discovery_promotions[0]['origin']['source']['source_id'] == old_sources['s2']['source_id']
    assert restored.discovery_promotions[0]['origin']['source']['source_alias'] == 's2'
    assert restored.discovery_promotions[0]['origin']['source']['current_source_alias'] == rename['s2']
    assert rebound['read_states'] == [[identity, rename[alias], value] for identity, alias, value in checkpoint['read_states']]
    assert rebound['clues'] == {identity: [rename[alias] for alias in aliases]
                                for identity, aliases in checkpoint['clues'].items()}
    restored.read_states = {(owner, alias): status for owner, alias, status in rebound['read_states']}
    restored.clues = deepcopy(rebound['clues'])
    restored._fits = lambda operation, data: len(data['source_spans']) <= 3
    calls.respond = lambda index, data: response(observation('same-current-history', [rename['s1']]))
    before_requests = len(calls.requests)
    history_discovery(restored, e, docs=['advice', 'last'])
    assert len(calls.requests) == before_requests + 1
    assert restored.batches[-1]['source_aliases'] == [rename['s3']]
    assert list(restored.discovery_frontier) == [('discovery', rename['s4'])]
    assert restored.read_states['discovery', rename['s3']] == 'read'


def test_actual_controller_restore_with_reordered_mapper_never_rereads_already_exposed_sources(monkeypatch):
    passages = {'x': 'x = 7.', 'alternate': 'The same x equals 7.',
                'later': 'Unexamined later source.', 'last': 'Unexamined last source.'}
    e, _ = engine(number_plan(), passages)
    control = prepare(e)
    step, query, identity = local_read(control, 'x', e.ids)
    reader = control.reader
    reader._fits = lambda operation, data: len(data['source_spans']) <= 1
    reader.read(step, query, ['x'], e.ids, identity, discovery_doc_ids=e.ids)
    alternate_alias = next(alias for alias, source in e.mapper.sources.items() if source['doc_id'] == 'alternate')
    alternate_span = reader._source(alternate_alias)
    alternate = alternate_span['id']
    e.spans[alternate] = alternate_span
    control.state.integrate([{'id': 'independent-equal-proof', 'variable': 'x', 'stance': 'support', 'value': 7,
        'source_ids': [alternate], **control.state.scope('x'), 'parent_versions': {}, 'retract_ids': [],
        'reason': 'The synthetic source independently gives the same quantity.'}])
    reader._fits = lambda operation, data: len(data['source_spans']) <= 2
    reader.read(step, query, ['x'], e.ids, identity, discovery_doc_ids=e.ids)
    assert len(reader.discovery_frontier) == 2 and reader.discovery_promotions
    checkpoint = json.loads(json.dumps(control.snapshot()))
    fresh, services = engine(number_plan(), passages)
    fresh.add_candidates(list(reversed(fresh.ids)))
    old_to_new = {alias: fresh.mapper.source_aliases[source['source_id']] for alias, source in e.mapper.sources.items()}
    assert all(alias != new for alias, new in old_to_new.items())
    with monkeypatch.context() as paused:
        paused.setattr(ResidualControl, 'run', lambda current: current.publish())
        fresh.resume(checkpoint)
    restored = fresh.residual_control
    assert restored.reader.material_signature() == reader.material_signature()
    assert restored.read_attempts == control.read_attempts and restored.read_effects == control.read_effects
    assert restored.compose_exhausted == control.compose_exhausted
    assert restored.reader.read_states == {(owner, old_to_new[alias]): value
                                           for (owner, alias), value in reader.read_states.items()}
    assert list(restored.reader.discovery_frontier) == [(owner, old_to_new[alias])
                                                       for owner, alias in reader.discovery_frontier]
    assert restored.reader.discovery_promotions[0]['origin']['source']['source_id'] == alternate
    before = deepcopy((fresh.ledger.used, restored.reader.obligation_snapshot()))
    assert restored.reader.read(step, query, ['x'], ['x', 'alternate'], identity,
                                discovery_doc_ids=['x', 'alternate']) is False
    assert (fresh.ledger.used, restored.reader.obligation_snapshot()) == before
    assert not services.requests
