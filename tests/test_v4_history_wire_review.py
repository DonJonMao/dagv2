"""Independent lossless history-wire review; no labels or service calls.

The oracle is deliberately independent of the production encoder and decoder.
"""
from copy import deepcopy
import json
import random

import pytest

from dagbt import history_wire as codec
from dagbt import review_wire
from dagbt.fact_state import FactState
from dagbt.joint_reading import JointReader
from dagbt.reasoning import ProtocolError
from test_personamem_joint_protocol import fixture, observation, visible, expand_wire_data
from test_v4_review_wire_independent import independent_inverse as review_inverse


def canonical(value):
    """JSON comparison distinguishes Boolean, numeric and string values."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def oracle_decode(encoded):
    """Independent protocol inverse; do not call production normalization/unpack.

    Old shared cells are dereferenced once, then only explicitly defined new
    top-level fields and direct parent/known value fields can hold references.
    Nested application dictionaries always remain application dictionaries.
    """
    result = deepcopy(encoded)
    if 'known_observation_columns' in result:
        columns = result.pop('known_observation_columns')
        pool = result.pop('known_observation_values', [])
        if (type(columns) is not list or len(set(map(str, columns))) != len(columns)
                or any(type(column) is not str for column in columns)
                or type(pool) is not list):
            raise ValueError('invalid old table declaration')
        if type(result.get('known_observations')) is not list:
            raise ValueError('missing or invalid old table records')
        records = []
        for cells in result.get('known_observations', []):
            if type(cells) is not list or len(cells) != len(columns):
                raise ValueError('invalid old table record')
            record = {}
            for position, field in enumerate(columns):
                value = cells[position]
                if type(value) is dict and list(value.keys()) == ['shared']:
                    index = value['shared']
                    if type(index) is not int or index < 0 or index >= len(pool):
                        raise ValueError('invalid old table index')
                    value = pool[index]
                record[field] = deepcopy(value)
            records.append(record)
        result['known_observations'] = records
    elif 'known_observation_values' in result:
        raise ValueError('orphan old shared pool')

    if 'shared_history_texts' not in result:
        if 'grounded_subquestion_parts' in result:
            raise ValueError('orphan new query references')
        return result
    table = result.pop('shared_history_texts')
    if (type(table) is not dict or not table
            or any(type(k) is not str or type(v) is not str for k, v in table.items())):
        raise ValueError('invalid new text dictionary')

    def lookup(reference):
        if type(reference) is not str or reference not in table:
            raise ValueError('unresolved text reference')
        return table[reference]

    for records in [result.get('supported_parents', {}).values(), result.get('known_observations', [])]:
        for record in records:
            if 'value_ref' in record:
                if 'value' in record:
                    raise ValueError('ambiguous value field')
                reference = record.pop('value_ref')
                record['value'] = lookup(reference)
    if 'grounded_subquestion_parts' in result:
        if 'grounded_subquestion' in result:
            raise ValueError('ambiguous query field')
        fragments = result.pop('grounded_subquestion_parts')
        if type(fragments) is not list:
            raise ValueError('invalid query fragment list')
        restored = ''
        for fragment in fragments:
            if type(fragment) is str:
                restored += fragment
            elif type(fragment) is dict and len(fragment) == 1 and 'text_ref' in fragment:
                restored += lookup(fragment['text_ref'])
            else:
                raise ValueError('invalid query fragment')
        result['grounded_subquestion'] = restored
    return result


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Independent protocol review must stay offline')
    monkeypatch.setattr('urllib.request.urlopen', forbidden)
    monkeypatch.setattr('urllib.request.build_opener', forbidden)


def long_text():
    return '历史🙂 mixed e\u0301 words "quoted" \\slash\nnext\tline. ' * 70


def wire_fixture(value=None):
    value = long_text() if value is None else value
    return {
        'grounded_subquestion': 'Use this exact history:\n' + value + '\nThen decide.',
        'supported_parents': {
            'past': {'type': 'text', 'value': value, 'version': 2,
                     'scope': {'entity_scope': 'user', 'time_scope': 'past'}, 'source_ids': ['s1', 's2']},
            'current': {'type': 'text', 'value': value, 'version': 7,
                        'scope': {'entity_scope': 'colleague', 'time_scope': 'current'}, 'source_ids': ['s3']},
        },
        'known_observations': [
            {'id': 'old-route', 'variable': 'past', 'value': value, 'source_ids': ['s1', 's2'],
             'parent_versions': {'context': 4}, 'entity_scope': 'user', 'time_scope': 'past',
             'revoked': [{'source_ids': ['s4'], 'reason': 'A real source-backed counter'}],
             'summary_revision': {'supersedes_route_ids': ['ancestor'], 'previous_version': 1},
             'superseded_by': ['new-route']},
            {'id': 'new-route', 'variable': 'current', 'value': value, 'source_ids': ['s3'],
             'parent_versions': {}, 'entity_scope': 'colleague', 'time_scope': 'current',
             'revoked': [], 'superseded_by': []},
        ],
        'source_spans': [{'id': 's1', 'text': '{"value_ref":"t0"} is original quoted source text'}],
        'evidence_obligations': [{'id': 'pending-issue', 'source_ids': ['s5'], 'variable': 'past'}],
        'failed_batch_reviews': [{'id': 'failed-batch', 'review_allowed': False, 'source_ids': ['s1', 's5'],
                                 'failed': [{'id': 'bad-draft', 'observation': {'value': value, 'source_ids': ['s5']}}]}],
        'repair_scope': {'whole_value_reviews': [{'variable': 'past', 'known_observation_ids': ['old-route'],
                                                 'required_source_ids': ['s1', 's2', 's4']}]},
    }


def verify_roundtrip(original, *, packed_expected=True):
    before = canonical(original)
    result, applied = codec.pack_with_status(original)
    assert canonical(original) == before
    assert applied is packed_expected
    assert canonical(codec.pack(original)) == canonical(result)
    assert canonical(oracle_decode(result)) == canonical(oracle_decode(original))
    assert canonical(codec.unpack(result)) == canonical(oracle_decode(original))
    if packed_expected:
        assert 'shared_history_texts' in result
        assert codec.byte_size(result) + len(codec.INSTRUCTION.encode()) < codec.byte_size(original)
    return result


def test_scope_and_versions_are_preserved_for_equal_history_values():
    original = wire_fixture()
    result = verify_roundtrip(original)
    restored = oracle_decode(result)
    assert len(restored['known_observations']) == 2
    assert restored['supported_parents']['past']['scope'] != restored['supported_parents']['current']['scope']
    for field in ['source_spans', 'evidence_obligations', 'failed_batch_reviews', 'repair_scope']:
        assert canonical(result[field]) == canonical(original[field])
    result['supported_parents']['past']['scope']['time_scope'] = 'mutated-test-copy'
    assert original['supported_parents']['past']['scope']['time_scope'] == 'past'


def test_application_reference_shaped_dictionaries_and_other_typed_values_remain_literal():
    original = wire_fixture()
    literals = [{'value_ref': 't0'}, {'text_ref': 't0'}, {'shared': 0},
                {'nested': [{'value_ref': 't0'}, {'shared': True}]},
                True, 1, 1.0, False, 0, None, ['t0'], {'value': long_text()}]
    for index, value in enumerate(literals):
        original['known_observations'].append({'id': 'literal-' + str(index), 'value': deepcopy(value)})
        original['supported_parents']['literal-' + str(index)] = {'type': 'record', 'value': deepcopy(value)}
    result = verify_roundtrip(original)
    restored = oracle_decode(result)
    assert canonical([row['value'] for row in restored['known_observations'][2:]]) == canonical(literals)


@pytest.mark.parametrize('query_case', ['unicode', 'overlap', 'empty', 'no_match', 'adjacent', 'quotes'])
def test_grounded_query_is_exact_including_overlap_and_literal_reference_shapes(query_case):
    original = wire_fixture()
    short = '甲🙂\\\n"' * 90
    long = short + 'tail'
    original['supported_parents']['short'] = {'type': 'text', 'value': short}
    original['supported_parents']['long'] = {'type': 'text', 'value': long}
    cases = {
        'unicode': '前\n' + long_text() + '\u2028\u200d尾',
        'overlap': long + short + long,
        'empty': '',
        'no_match': 'literal {"text_ref":"t0"}; retain all punctuation',
        'adjacent': long_text() + long_text(),
        'quotes': '"\\\t\r\n' + short + '\n\\"',
    }
    original['grounded_subquestion'] = cases[query_case]
    restored = oracle_decode(verify_roundtrip(original))
    assert restored['grounded_subquestion'] == cases[query_case]


def old_table_fixture():
    original = wire_fixture()
    text = long_text()
    original['known_observation_columns'] = ['id', 'variable', 'value', 'metadata']
    original['known_observation_values'] = [text, {'shared': 0}, {'text_ref': 't0'}, {'value_ref': 't0'}]
    original['known_observations'] = [
        ['old-route', 'past', {'shared': 0}, {'shared': 1}],
        ['new-route', 'current', {'shared': 0}, {'shared': 2}],
        ['literal-row', 'record', {'shared': 3}, {'shared': 1}],
    ]
    return original


def test_old_columnar_table_is_decoded_once_with_nested_reference_shapes_left_literal():
    original = old_table_fixture()
    restored = oracle_decode(verify_roundtrip(original))
    assert restored['known_observations'][0]['metadata'] == {'shared': 0}
    assert restored['known_observations'][1]['metadata'] == {'text_ref': 't0'}
    assert restored['known_observations'][2]['value'] == {'value_ref': 't0'}
    assert restored['repair_scope'] == original['repair_scope']


@pytest.mark.parametrize('bad_index', [True, False, -1, 4, 999, 0.0, '0', None])
def test_invalid_old_table_indices_are_rejected_and_pack_falls_back(bad_index):
    original = old_table_fixture()
    original['known_observations'][0][2] = {'shared': bad_index}
    before = canonical(original)
    result, applied = codec.pack_with_status(original)
    assert applied is False
    assert canonical(result) == before
    assert canonical(codec.pack(original)) == before
    assert canonical(original) == before
    with pytest.raises(ValueError):
        codec.unpack(original)
    with pytest.raises(ValueError):
        oracle_decode(original)


@pytest.mark.parametrize('shape', ['duplicate_column', 'non_string_column', 'short_row',
                                  'long_row', 'non_list_row', 'non_list_pool',
                                  'missing_rows', 'orphan_pool'])
def test_invalid_old_table_shapes_are_not_repaired_by_encoding(shape):
    original = old_table_fixture()
    if shape == 'duplicate_column':
        original['known_observation_columns'][-1] = 'value'
    elif shape == 'non_string_column':
        original['known_observation_columns'][-1] = 7
    elif shape == 'short_row':
        original['known_observations'][0].pop()
    elif shape == 'long_row':
        original['known_observations'][0].append('unlisted-source')
    elif shape == 'non_list_row':
        original['known_observations'][0] = {'id': 'wrong-shape'}
    elif shape == 'non_list_pool':
        original['known_observation_values'] = {'0': 'wrong-shape'}
    elif shape == 'missing_rows':
        del original['known_observations']
    else:
        del original['known_observation_columns']
    before = canonical(original)
    result, applied = codec.pack_with_status(original)
    assert applied is False
    assert canonical(result) == before
    assert canonical(codec.pack(original)) == before
    assert canonical(original) == before
    with pytest.raises(ValueError):
        codec.unpack(original)
    with pytest.raises(ValueError):
        oracle_decode(original)


@pytest.mark.parametrize('collision', ['shared_history_texts', 'grounded_subquestion_parts',
                                        'parent_value_ref', 'known_value_ref'])
def test_reserved_collisions_return_a_separate_unchanged_representation(collision):
    original = wire_fixture()
    if collision == 'parent_value_ref':
        original['supported_parents']['past']['value_ref'] = 'literal-application-value'
    elif collision == 'known_value_ref':
        original['known_observations'][0]['value_ref'] = 'literal-application-value'
    else:
        original[collision] = {'arbitrary': True}
    before = canonical(original)
    result, applied = codec.pack_with_status(original)
    assert applied is False
    assert canonical(result) == before and canonical(original) == before
    assert result is not original
    result['supported_parents']['past']['version'] = 900
    assert original['supported_parents']['past']['version'] == 2


def test_nonbeneficial_encoding_including_instruction_cost_falls_back():
    value = 'x' * 256
    original = {'grounded_subquestion': 'short question',
                'supported_parents': {'p': {'type': 'text', 'value': value}},
                'known_observations': [{'id': 'p', 'value': value}]}
    before = canonical(original)
    result, applied = codec.pack_with_status(original)
    assert applied is False
    assert canonical(result) == before and canonical(original) == before
    assert 'shared_history_texts' not in result


def test_seeded_unicode_roundtrips():
    randomizer = random.Random(20261014)
    alphabet = ['甲', 'é', 'e\u0301', '🙂', '\u200d', '\u2028', ' ', '"', '\\', '\n', '\t', 'x', '9']
    for _ in range(18):
        value = ''.join(randomizer.choice(alphabet) for _ in range(900))
        original = wire_fixture(value)
        original['grounded_subquestion'] = 'pre\n' + value + '\\' + value[100:] + '\npost'
        verify_roundtrip(original)


@pytest.mark.parametrize('malformation', ['unknown_value', 'bool_ref', 'both_value', 'unknown_query',
                                           'query_extra_key', 'query_boolean', 'both_query'])
def test_malformed_new_references_are_rejected_by_both_decoders(malformation):
    encoded = codec.pack(wire_fixture())
    if malformation == 'unknown_value':
        encoded['supported_parents']['past']['value_ref'] = 'missing'
    elif malformation == 'bool_ref':
        encoded['supported_parents']['past']['value_ref'] = True
    elif malformation == 'both_value':
        encoded['supported_parents']['past']['value'] = 'ambiguous'
    elif malformation == 'unknown_query':
        encoded['grounded_subquestion_parts'] = [{'text_ref': 'missing'}]
    elif malformation == 'query_extra_key':
        encoded['grounded_subquestion_parts'] = [{'text_ref': 't0', 'literal': 1}]
    elif malformation == 'query_boolean':
        encoded['grounded_subquestion_parts'] = [True]
    else:
        encoded['grounded_subquestion'] = 'ambiguous'
    with pytest.raises(ValueError):
        codec.unpack(encoded)
    with pytest.raises(ValueError):
        oracle_decode(encoded)


def reader_payload_fixture():
    """Use a real canonical payload, state and reasoner with scripted transport."""
    _, engine, initial, calls = fixture(max_repairs_per_request=0, context_tokens=100000)
    program = deepcopy(initial.initial_program)
    program['variables']['history']['type'] = 'text'
    state = FactState(program, initial.contract, engine.steps, 'suggestion')
    reader = JointReader(engine, state)
    row = observation('established-parent', value=long_text())
    state.integrate([reader._observation(row, 'history', visible(reader), engine.steps[0])])
    data = reader._payload(engine.steps[1], 'Use this history:\n' + long_text(),
                           ['history', 'suggestion'], ['s1'])
    return reader, engine, state, calls, data


@pytest.mark.parametrize('collision', ['shared_history_texts', 'grounded_subquestion_parts',
                                        'parent_value_ref', 'known_value_ref'])
def test_prepare_history_collision_keeps_literals_and_never_adds_history_instruction(collision):
    reader, engine, state, calls, original = reader_payload_fixture()
    if collision == 'parent_value_ref':
        original['supported_parents']['history']['value_ref'] = 'caller-literal'
    elif collision == 'known_value_ref':
        original['known_observations'][0]['value_ref'] = 'caller-literal'
    elif collision == 'shared_history_texts':
        original[collision] = {'t0': 'A caller-owned table must not activate decoding.'}
    else:
        original[collision] = [{'text_ref': 't0'}]
    before = canonical(original)
    state_before = canonical((state.public_dict(), reader._schema(original),
                              engine.mapper.sources, engine.ledger.used))
    old_wire = reader._wire_data(original)
    wire, system = reader._prepare_wire(original)
    # History must fall back; independent review sharing may still apply.
    # Its handwritten inverse must recover every caller-owned history field.
    assert canonical(review_inverse(wire)) == canonical(old_wire)
    assert canonical(original) == before
    assert 'shared_review_texts' in wire
    assert system == reader._system(wire) + review_wire.INSTRUCTION
    assert codec.INSTRUCTION not in system
    assert canonical((state.public_dict(), reader._schema(original),
                      engine.mapper.sources, engine.ledger.used)) == state_before
    assert not calls.requests and not any(engine.ledger.used.values())


def test_successful_prepare_preserves_canonical_state_schema_and_all_review_metadata():
    reader, engine, state, calls, data = reader_payload_fixture()
    route = deepcopy(data['known_observations'][0])
    route['revoked'] = [{'source_ids': ['s2'], 'reason': 'Source-backed counter, still visible.'}]
    data['known_observations'][0] = route
    data['failed_batch_reviews'] = [{
        'id': 'failed-history', 'variable': 'history', 'review_allowed': False,
        'source_ids': ['s1', 's3'],
        'failed': [{'id': 'failed-draft', 'observation': {
            'value': long_text(), 'source_ids': ['s3'], 'value_ref': 'literal-draft-field'}}],
    }]
    data['repair_scope'] = {'whole_value_reviews': [{
        'variable': 'history', 'known_observation_ids': [route['id']],
        'required_source_ids': ['s1', 's2'],
    }]}
    data['evidence_obligations'] = [{
        'id': 'pending-evidence', 'variable': 'history', 'source_ids': ['s3'],
    }]
    before = canonical((data, state.public_dict(), engine.mapper.sources, engine.ledger.used))
    schema = reader._schema(data)
    wire, system = reader._prepare_wire(data)
    assert 'shared_history_texts' in wire
    assert system == reader._system(wire) + codec.INSTRUCTION
    assert canonical(expand_wire_data(oracle_decode(wire))) == canonical(data)
    assert canonical((data, state.public_dict(), engine.mapper.sources, engine.ledger.used)) == before
    assert canonical(reader._schema(data)) == canonical(schema)
    for field in ['failed_batch_reviews', 'repair_scope', 'evidence_obligations']:
        assert canonical(wire[field]) == canonical(data[field])
    assert 'value_ref' not in schema['properties']['slots']['items']['properties']['observations']['items']['properties']
    assert 'text_ref' not in canonical(schema)
    assert not calls.requests


def test_prepare_has_no_stale_applied_status_between_requests():
    reader, _, _, _, original = reader_payload_fixture()
    shared_wire, shared_system = reader._prepare_wire(original)
    assert codec.INSTRUCTION in shared_system
    collision = deepcopy(original)
    collision['shared_history_texts'] = {'t0': 'Caller-owned literal.'}
    fallback_wire, fallback_system = reader._prepare_wire(collision)
    assert canonical(review_inverse(fallback_wire)) == canonical(reader._wire_data(collision))
    assert fallback_wire['shared_history_texts'] == collision['shared_history_texts']
    assert 'shared_review_texts' in fallback_wire
    assert fallback_system == reader._system(fallback_wire) + review_wire.INSTRUCTION
    assert codec.INSTRUCTION not in fallback_system
    wire_again, system_again = reader._prepare_wire(original)
    assert canonical(wire_again) == canonical(shared_wire)
    assert system_again == shared_system


def test_estimation_receives_encoded_wire_with_canonical_schema_and_unchanged_token_cap(monkeypatch):
    reader, engine, state, calls, data = reader_payload_fixture()
    route = data['known_observations'][0]
    # The legacy table is part of the real preparation path, not an isolated
    # decoder fixture; its one-time normalization must not change schema IDs.
    data['known_observations'] = [{**deepcopy(route), 'id': f'route-{i}'} for i in range(9)]
    old_wire = reader._wire_data(data)
    assert 'known_observation_columns' in old_wire
    expected_schema = reader._schema(data)
    before = canonical(data)
    estimates = []

    def inspect(operation, request):
        estimates.append((operation, deepcopy(request)))
        return {'count': 123, 'token_count_is_estimate': True}

    monkeypatch.setattr(engine.reasoner, '_count_payload', inspect)
    assert reader._estimate('semantic_compose', data, output_token_cap=127) == 123
    assert len(estimates) == 1
    operation, request = estimates[0]
    system = request['messages'][0]['content']
    wire = json.loads(request['messages'][1]['content'])
    schema = json.loads(system.split('matching this schema:\n', 1)[1])
    assert operation == 'semantic_compose'
    assert request['max_tokens'] == 127
    assert 'shared_history_texts' in wire and 'known_observation_columns' not in wire
    assert codec.INSTRUCTION in system
    assert canonical(schema) == canonical(expected_schema)
    assert canonical(expand_wire_data(oracle_decode(wire))) == before
    assert canonical(data) == before and not calls.requests
    assert not any(engine.ledger.used.values())


def test_actual_read_sends_the_same_encoded_input_and_schema_that_were_estimated(monkeypatch):
    reader, engine, state, calls, _ = reader_payload_fixture()
    canonical_sources = canonical(engine.mapper.sources)
    estimates = []
    count = engine.reasoner._count_payload
    estimate = reader._estimate
    in_estimate = False

    def inspect(operation, request):
        if in_estimate:
            estimates.append((operation, canonical(request)))
        return count(operation, request)

    def inspect_estimate(*args, **kwargs):
        nonlocal in_estimate
        in_estimate = True
        try:
            return estimate(*args, **kwargs)
        finally:
            in_estimate = False

    def reply(index, expanded):
        decoded = oracle_decode(expanded)
        assert decoded['grounded_subquestion'] == 'Use this history:\n' + long_text()
        assert decoded['supported_parents']['history']['value'] == long_text()
        return {'slots': [{'variable': variable, 'observations': [], 'unresolved': []}
                          for variable in decoded['requested_slots']]}

    monkeypatch.setattr(engine.reasoner, '_count_payload', inspect)
    monkeypatch.setattr(reader, '_estimate', inspect_estimate)
    calls.respond = reply
    reader.read(engine.steps[1], 'Use this history:\n' + long_text(), ['suggestion'],
                ['piano'], 'independent-history-wire-read')
    assert len(calls.requests) == 1
    request = calls.requests[0]
    assert 'shared_history_texts' in request['wire_data']
    assert codec.INSTRUCTION in request['payload']['messages'][0]['content']
    assert (request['stage'][0], canonical(request['payload'])) in estimates
    assert state.valid()[0]['history'] == long_text()
    assert canonical(engine.mapper.sources) == canonical_sources
    assert engine.ledger.used['llm'] == 1 and engine.ledger.used['json_repairs'] == 0


@pytest.mark.parametrize('wire_only_field', ['value_ref', 'text_ref'])
def test_input_reference_fields_are_not_accepted_as_output_observation_shortcuts(wire_only_field):
    reader, _, state, _, _ = reader_payload_fixture()
    row = observation('output-shortcut', value=long_text())
    del row['value']
    row[wire_only_field] = 't0'
    before = canonical(state.public_dict())
    with pytest.raises(ProtocolError):
        reader._observation(row, 'history', visible(reader), reader.e.steps[0])
    assert canonical(state.public_dict()) == before


def test_terminal_only_repair_estimate_requires_the_same_full_terminal_schema(monkeypatch):
    reader, engine, _, _, data = reader_payload_fixture()
    data.update(variables={}, requested_slots=[], requested_scopes={}, phase='response_selection',
                public_options=['piano', 'painting', 'pottery', 'hiking'],
                terminal_repair={'expected_value': 'piano', 'feedback': 'Inconsistent prediction.'})
    captured = []

    def inspect(operation, request):
        captured.append((operation, deepcopy(request)))
        return {'count': 123, 'token_count_is_estimate': True}

    monkeypatch.setattr(engine.reasoner, '_count_payload', inspect)
    reader._estimate('semantic_compose_repair', data)
    assert len(captured) == 1
    operation, request = captured[0]
    assert operation == 'semantic_compose_repair'
    assert request['max_tokens'] == engine.s['reasoning_output_tokens']
    system = request['messages'][0]['content']
    wire = json.loads(request['messages'][1]['content'])
    schema = json.loads(system.split('matching this schema:\n', 1)[1])
    assert schema['required'] == ['slots', 'semantic_answer', 'final_prediction']
    assert schema['properties']['slots']['minItems'] == schema['properties']['slots']['maxItems'] == 0
    assert schema['properties']['final_prediction']['enum'] == ['a', 'b', 'c', 'd', None]
    assert canonical(schema) == canonical(reader._schema(data))
    assert 'Terminal-only repair: return slots=[]' in system
    assert codec.INSTRUCTION in system
    assert oracle_decode(wire)['terminal_repair'] == data['terminal_repair']


@pytest.mark.parametrize('conflict', ['history_evidence', 'response_audit', 'audit_instruction'])
def test_terminal_only_repair_rejects_conflicting_phase_without_budget_or_request(conflict):
    reader, engine, state, calls, data = reader_payload_fixture()
    data.update(variables={}, requested_slots=[], requested_scopes={},
                terminal_repair={'expected_value': 'piano'})
    if conflict == 'audit_instruction':
        data['audit_instruction'] = ''
    else:
        data['phase'] = conflict
    before = canonical((data, state.public_dict(), engine.ledger.used, engine.events))
    with pytest.raises(ProtocolError, match='Terminal-only repair'):
        reader._estimate('semantic_compose_repair', data)
    assert canonical((data, state.public_dict(), engine.ledger.used, engine.events)) == before
    assert not calls.requests
