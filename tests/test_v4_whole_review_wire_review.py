"""Independent lossless whole-review route references, with no remote calls."""
from copy import deepcopy

import pytest

from dagbt.fact_state import FactState
from dagbt.joint_reading import JointReader
from test_personamem_joint_protocol import fixture, observation, response, visible, expand_wire_data
from test_v4_summary_revision_review import summary_fixture, revised


def expand_all_wire(data):
    """Independent inverse: source tables, known-route tables, then group refs."""
    restored = expand_wire_data(data)
    columns = restored.pop('known_observation_columns', None)
    if columns is not None:
        shared = restored.pop('known_observation_values')
        restored['known_observations'] = [dict(zip(columns, [
            deepcopy(shared[value['shared']])
            if isinstance(value, dict) and set(value) == {'shared'} else deepcopy(value)
            for value in row])) for row in restored['known_observations']]
    routes = {row['id']: row for row in restored.get('known_observations', [])}
    for group in restored.get('repair_scope', {}).get('whole_value_reviews', []):
        if 'known_observation_ids' in group and 'known_observations' not in group:
            group['known_observations'] = [deepcopy(routes[rid]) for rid in group.pop('known_observation_ids')]
    return restored


def wire_fixture(count=3):
    reader, engine, _, _ = fixture()
    data = reader._payload(engine.steps[0], 'Review all original history', ['history'], ['s1', 's2', 's3'])
    rows = []
    for index in range(count):
        rows.append({
            'id': 'historical-route-' + str(index), 'variable': 'history', 'stance': 'support',
            'value': {'summary': 'A complete scoped historical account, including negative experiences. ' * 30,
                      'source_ids': ['literal-value-field'], 'known_observation_ids': ['literal-value-id']},
            'source_ids': ['s1', 's2'], 'entity_scope': 'user', 'time_scope': 'history',
            'parent_versions': {'context': 7}, 'retract_ids': [],
            'reason': 'The original quotations jointly support this scoped account. ' * 20,
            'revoked': [{'source_ids': ['s3'], 'reason': 'A real counter to an inherited premise.'}],
            'summary_revision': {'supersedes_route_ids': ['older-route'], 'previous_version': 6,
                                 'reason': 'Compatible representation update retaining all originals.'},
            'superseded_by': ['later-route'], 'semantic_status': 'model_judgment',
        })
    data['known_observations'] = rows
    data['repair_scope'] = {
        'failed': [{'variable': 'history', 'id': '*', 'feedback': 'whole_value_disagreement',
                    'observation': None, 'slot': {'variable': 'history', 'observations': [], 'unresolved': []}}],
        'whole_value_reviews': [{'variable': 'history', 'scope': {'entity_scope': 'user', 'time_scope': 'history'},
                                'draft_observations': [observation('new-draft', ['s3'], value=['literal'])],
                                'known_observations': deepcopy(list(reversed(rows))),
                                'required_source_ids': ['s1', 's2', 's3']}],
        'attempt': 1, 'previous_response_identity': '0' * 64,
    }
    return reader, engine, data


@pytest.mark.parametrize('count', [3, 16])
def test_exact_routes_roundtrip_with_all_sources_counters_ancestry_and_existing_table_encoding(count):
    reader, _, data = wire_fixture(count)
    before = deepcopy(data)
    wire = reader._wire_data(data)
    group = wire['repair_scope']['whole_value_reviews'][0]
    assert group['known_observation_ids'] == [row['id'] for row in before['repair_scope']['whole_value_reviews'][0]['known_observations']]
    assert 'known_observations' not in group
    assert expand_all_wire(wire) == before and data == before
    assert group['draft_observations'] == before['repair_scope']['whole_value_reviews'][0]['draft_observations']
    assert group['required_source_ids'] == ['s1', 's2', 's3']
    system = reader._system(wire)
    assert 'known_observation_ids' in system and 'known_observations' in system
    if count >= 8:
        assert 'known_observation_columns' in wire
        assert 'known_observation_columns' in system
    else:
        assert 'known_observation_columns' not in wire


@pytest.mark.parametrize('field', ['value', 'source_ids', 'parent_versions', 'revoked', 'summary_revision', 'superseded_by'])
def test_same_route_id_with_any_different_content_keeps_the_entire_original_group(field):
    reader, _, data = wire_fixture()
    group = data['repair_scope']['whole_value_reviews'][0]
    row = group['known_observations'][1]
    if field == 'value':
        row[field]['summary'] += ' A distinct claim.'
    elif field == 'source_ids':
        row[field] = ['s1']
    elif field == 'parent_versions':
        row[field]['context'] = 8
    elif field == 'revoked':
        row[field] = []
    elif field == 'summary_revision':
        row[field]['previous_version'] = 5
    else:
        row[field] = []
    before = deepcopy(data)
    wire = reader._wire_data(data)
    assert 'known_observation_ids' not in wire['repair_scope']['whole_value_reviews'][0]
    assert expand_all_wire(wire) == before and data == before


@pytest.mark.parametrize('problem', ['duplicate_top_id', 'duplicate_group_id', 'unknown_id', 'existing_reference_field'])
def test_ambiguous_or_preexisting_reference_shape_is_not_guessed(problem):
    reader, _, data = wire_fixture()
    group = data['repair_scope']['whole_value_reviews'][0]
    if problem == 'duplicate_top_id':
        data['known_observations'].append(deepcopy(data['known_observations'][0]))
    elif problem == 'duplicate_group_id':
        group['known_observations'].append(deepcopy(group['known_observations'][0]))
    elif problem == 'unknown_id':
        group['known_observations'][0]['id'] = 'not-in-top-level'
    else:
        group['known_observation_ids'] = ['literal-existing-field']
    before = deepcopy(data)
    wire = reader._wire_data(data)
    assert wire['repair_scope']['whole_value_reviews'][0] == before['repair_scope']['whole_value_reviews'][0]
    assert expand_all_wire(wire) == before and data == before


def test_json_type_difference_does_not_match_by_python_boolean_numeric_equality():
    reader, _, data = wire_fixture()
    top = data['known_observations'][0]
    top['parent_versions']['context'] = 1
    group = data['repair_scope']['whole_value_reviews'][0]
    copied = next(row for row in group['known_observations'] if row['id'] == top['id'])
    copied['parent_versions']['context'] = True
    before = deepcopy(data)
    wire = reader._wire_data(data)
    assert 'known_observation_ids' not in wire['repair_scope']['whole_value_reviews'][0]
    assert expand_all_wire(wire) == before


def long_summary_reader(*, repetitions=18):
    _, engine, old_state, calls, _ = summary_fixture()
    engine.s.update(max_repairs_per_request=1, reasoning_output_tokens=4096, context_tokens=16384)
    state = FactState(old_state.initial_program, old_state.contract, engine.steps, 'suggestion')
    reader = JointReader(engine, state)
    value = 'The user played piano last year, enjoying its creative and social aspects. ' * repetitions
    row = observation('old-long-summary', ['s1'], value=value)
    state.integrate([reader._observation(row, 'history', visible(reader), engine.steps[0])])
    return reader, engine, state, calls, next(iter(state.routes)), value


def test_real_whole_repair_request_uses_references_and_preserves_original_review_semantics():
    # This case verifies lossless long-value references and semantic repair.
    # Give that fixture room instead of repeatedly shrinking its value when
    # protocol instructions grow. The helper and the separate capacity case
    # below keep 16k and both 4096 output reserves; this is no 16k fit claim.
    reader, engine, state, calls, original, value = long_summary_reader()
    engine.s['context_tokens'] = 32768
    expanded = value + ' The user also enjoyed painting last spring.'
    def reply(index, data):
        if index == 1:
            return response(observation('expanded-summary', ['s1', 's2'], value=expanded))
        wire = calls.requests[-1]['wire_data']
        assert 'known_observation_ids' in wire['repair_scope']['whole_value_reviews'][0]
        restored = expand_all_wire(wire)
        known = restored['repair_scope']['whole_value_reviews'][0]['known_observations']
        assert len(known) == 1 and known[0]['id'] == original and known[0]['value'] == value
        return response(revised(original, state.versions['history'], value=expanded))
    calls.respond = reply
    reader.read(engine.steps[0], 'Review historical additions', ['history'], ['painting'], 'actual-whole-review')
    assert len(calls.requests) == 2 and engine.ledger.used['json_repairs'] == 1
    assert state.valid()[0]['history'] == expanded and not state.pending
    assert state.routes[original]['superseded_by']


def test_forecast_and_actual_repair_use_same_reference_wire_and_keep_both_output_reserves(monkeypatch):
    reader, engine, _, _, _, _ = long_summary_reader()
    data = reader._payload(engine.steps[0], 'Review historical additions', ['history'], ['s2'])
    before, settings, limits = deepcopy(data), deepcopy(engine.s), deepcopy(engine.ledger.limits)
    captured = []
    threshold = 16384 - 4096 - 4096 - engine.s['input_margin'] - 8
    extra = 0
    def estimate(operation, system, payload, schema):
        captured.append((operation, system, deepcopy(payload)))
        if payload.get('repair_scope', {}).get('whole_value_reviews'):
            assert 'known_observation_ids' in payload['repair_scope']['whole_value_reviews'][0]
            assert 'known_observation_ids' in system
            return threshold + extra
        return 2000
    monkeypatch.setattr(engine.reasoner, 'estimate', estimate)
    assert reader._fresh_window_fits('joint_read', data)
    forecast = deepcopy(captured[-1][2])
    extra = 1
    assert not reader._fresh_window_fits('joint_read', data), 'Both existing 4096 reserves must still count'
    actual = expand_all_wire(forecast)
    actual['personal_task_instruction'] = before['personal_task_instruction']
    assert reader._repair_fits('joint_read_repair', actual, engine.steps[0], 'actual-envelope', 1)
    actual_wire = captured[-1][2]
    assert actual_wire['repair_scope']['whole_value_reviews'] == forecast['repair_scope']['whole_value_reviews']
    assert expand_all_wire(actual_wire) == expand_all_wire(forecast)
    assert data == before and engine.s == settings and engine.ledger.limits == limits
