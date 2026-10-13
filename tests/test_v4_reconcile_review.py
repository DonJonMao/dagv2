"""Independent whole-value review: explicit reconciliation, no automatic union."""
from copy import deepcopy

from test_personamem_joint_protocol import fixture, observation, response, visible


def personal_reader(**settings):
    reader, engine, state, calls = fixture(**settings)
    engine.personal_response_selection = True
    return reader, engine, state, calls


def fragments():
    return [observation('piano-part', ['s1'], value=['piano']),
            observation('painting-part', ['s2'], value=['painting'])]


def history_payload(reader, engine):
    reader.current_identity = 'later-history-review'
    return reader._payload(engine.steps[0], 'Review historical activities', ['history'], [])


def test_no_repair_budget_never_unions_fragments_and_keeps_both_original_sources_pending():
    reader, engine, state, calls = personal_reader(max_repairs_per_request=0)
    calls.respond = lambda index, data: response(*fragments())
    reader.read(engine.steps[0], 'Review historical activities', ['history'],
                ['piano', 'painting'], 'fragments')
    assert len(calls.requests) == 1
    assert 'history' not in state.valid()[0]
    assert 'history' in state.pending and 'history' in reader.pending_variables()
    payload = history_payload(reader, engine)
    assert {'s1', 's2'} <= {source['id'] for source in payload['source_spans']}


def test_only_explicit_model_review_can_turn_fragments_into_one_jointly_sourced_whole():
    reader, engine, state, calls = personal_reader(max_repairs_per_request=1)
    def answer(index, data):
        if index == 1:
            return response(*fragments())
        assert 'history' not in state.valid()[0]
        reviews = data['repair_scope']['whole_value_reviews']
        group = next(item for item in reviews if item['variable'] == 'history')
        assert {row['id'] for row in group['draft_observations']} == {'piano-part', 'painting-part'}
        assert {'s1', 's2'} <= set(group['required_source_ids'])
        return response(*[
            observation(row['id'], ['s1', 's2'], value=['piano', 'painting'])
            for row in fragments()])
    calls.respond = answer
    reader.read(engine.steps[0], 'Review historical activities', ['history'],
                ['piano', 'painting'], 'fragments')
    assert len(calls.requests) == 2
    assert state.valid()[0]['history'] == ['piano', 'painting']
    assert not state.pending and not reader.pending_variables()
    assert len(state.routes) == 1  # Identical complete proofs are OR alternatives.
    assert set(state.valid()[1]['history']) == {
        engine.mapper.sources[alias]['source_id'] for alias in ('s1', 's2')}


def test_genuine_unresolved_conflict_keeps_original_text_instead_of_forcing_agreement():
    reader, engine, state, calls = personal_reader(max_repairs_per_request=1)
    def answer(index, data):
        result = response(*fragments())
        if index > 1:
            result['slots'][0]['unresolved'] = [
                'The scope of the two claims cannot be reconciled from these original memories.']
        return result
    calls.respond = answer
    reader.read(engine.steps[0], 'Review historical activities', ['history'],
                ['piano', 'painting'], 'conflicting-wholes')
    assert len(calls.requests) == 2
    assert 'history' not in state.valid()[0]
    assert 'history' in state.pending and reader.evidence_obligations
    payload = history_payload(reader, engine)
    assert {'s1', 's2'} <= {source['id'] for source in payload['source_spans']}
    assert any('cannot be reconciled' in description
               for item in payload['evidence_obligations'] for description in item['descriptions'])


def test_whole_value_repair_cannot_discard_an_already_valid_counter():
    reader, engine, state, calls = personal_reader(max_repairs_per_request=1)
    prior = observation('old-account', ['s3'], value=['guitar'])
    state.integrate([reader._observation(prior, 'history', visible(reader, ['s3']), engine.steps[0])])
    old_route = next(iter(state.routes))
    counter = observation('real-counter', ['s2'], stance='contradiction', retract_ids=[old_route])
    counter['value'] = None
    def answer(index, data):
        if index == 1:
            return response(*fragments(), counter)
        unresolved = []
        for row in fragments():
            row = deepcopy(row)
            row.update(stance='partial', value=None,
                       reason='This original memory alone cannot establish the whole scoped activity value.')
            unresolved.append(row)
        return response(*unresolved)
    calls.respond = answer
    reader.read(engine.steps[0], 'Review historical activities', ['history'],
                ['piano', 'painting'], 'reconcile-with-counter')
    assert len(calls.requests) == 2
    assert state.routes[old_route]['revoked']
    assert 'history' not in state.valid()[0]
    assert 'history' in state.pending and reader.pending_variables()
    payload = history_payload(reader, engine)
    assert {'s1', 's2', 's3'} <= {source['id'] for source in payload['source_spans']}
    retained = next(row for row in payload['known_observations'] if row['id'] == old_route)
    assert retained['revoked']


def test_reconciled_new_value_cannot_silently_overwrite_an_existing_complete_route():
    reader, engine, state, calls = personal_reader(max_repairs_per_request=1)
    state.integrate([reader._observation(observation('old-whole', ['s1']), 'history',
                                        visible(reader), engine.steps[0])])
    old_route = next(iter(state.routes))
    old = deepcopy(state.routes[old_route])
    def answer(index, data):
        if index == 1:
            return response(observation('new-whole', ['s2'], value=['painting']))
        group = next(item for item in data['repair_scope']['whole_value_reviews']
                     if item['variable'] == 'history')
        assert any(row['id'] == old_route for row in group['known_observations'])
        # This changes the whole value but never retracts the existing whole.
        return response(observation('new-whole', ['s1', 's2'], value=['piano', 'painting']))
    calls.respond = answer
    reader.read(engine.steps[0], 'Review historical activities', ['history'],
                ['painting'], 'existing-whole-conflict')
    assert len(calls.requests) == 2
    assert state.routes[old_route] == old
    assert 'history' in state.pending and reader.pending_variables()
    payload = history_payload(reader, engine)
    assert {'s1', 's2'} <= {source['id'] for source in payload['source_spans']}


def test_whole_value_repair_cannot_drop_one_drafts_original_source():
    reader, engine, state, calls = personal_reader(max_repairs_per_request=1)
    def answer(index, data):
        if index == 1:
            return response(*fragments())
        # Claim agreement by discarding the painting source, with no sourced
        # review of why that original evidence became irrelevant.
        return response(*[observation(row['id'], ['s1'], value=['piano']) for row in fragments()])
    calls.respond = answer
    reader.read(engine.steps[0], 'Review historical activities', ['history'],
                ['piano', 'painting'], 'discarded-original')
    assert len(calls.requests) == 2
    assert 'history' in state.pending and reader.pending_variables()
    payload = history_payload(reader, engine)
    assert {'s1', 's2'} <= {source['id'] for source in payload['source_spans']}
