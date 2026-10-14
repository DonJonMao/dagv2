"""Offline review-wire regression boundaries; no tokenizer or model calls."""
from copy import deepcopy
import json

import pytest

from dagbt import history_wire, review_wire as codec
from dagbt.reasoning import ProtocolError
from test_personamem_joint_protocol import fixture, observation, visible


def exact(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':'), sort_keys=False)


def text_value():
    return '  原文🙂 e\u0301 é "quote" \\slash\nnext\tline {"review_value_ref":"r0"}  ' * 45


def review_fixture():
    def row(identifier, **changes):
        return {'id': identifier, 'variable': 'history', 'value': text_value(), 'stance': 'support',
                'reason': 'Literal reason:\n' + text_value(), 'source_ids': [identifier],
                'parent_versions': {'parent': 2}, 'entity_scope': 'user',
                'time_scope': 'past', 'extra': [False, 0, 0.0, True, 1, 1.0], **changes}
    failed = {'id': 'slot-failure', 'observation': row('failed'),
              'slot': {'observations': [row('slot')], 'unresolved': ['keep']},
              'observations': [row('duplicate')],
              'replacement_observations': [row('replacement')],
              'required_counter_observations': [row('counter', stance='contradiction')]}
    whole = {'variable': 'history', 'draft_observations': [row('whole-draft')],
             'known_observations': [row('whole-known')], 'known_observation_ids': ['known']}
    return {'supported_parents': {}, 'known_observations': [row('known')],
            'failed_batch_reviews': [{'id': 'batch', 'variable': 'history', 'review_allowed': False,
                                      'failed': [failed], 'validated_observations': [row('validated')],
                                      'whole_value_reviews': [whole],
                                      'invalid_response': ' {"a":1,"a":2, "broken": "\\u4e2d"...'}],
            'repair_scope': {'attempt': 1, 'failed': [deepcopy(failed)],
                             'whole_value_reviews': [deepcopy(whole)]},
            'state': {'pending': ['history'], 'budget': {'llm': 16, 'json_repairs': 4}},
            'source_spans': [{'text': text_value(), 'value': text_value()}]}


def independent_inverse(wire):
    """Enumerate the documented protocol independently of production helpers."""
    out = deepcopy(wire)
    pool = out.pop('shared_review_texts', None)
    if pool is None:
        return out
    rows = [(row, ('value',)) for row in out.get('supported_parents', {}).values()]
    if 'known_observation_columns' not in out:
        rows.extend((row, ('value', 'reason')) for row in out.get('known_observations', []))
    containers = out.get('failed_batch_reviews', []) + ([out['repair_scope']] if 'repair_scope' in out else [])
    for container in containers:
        if container is not out.get('repair_scope'):
            rows.extend((row, ('value', 'reason')) for row in container.get('validated_observations', []))
        for failure in container.get('failed', []):
            if failure.get('observation') is not None:
                rows.append((failure['observation'], ('value', 'reason')))
            if failure.get('slot') is not None:
                rows.extend((row, ('value', 'reason')) for row in failure['slot'].get('observations', []))
            for name in ['observations', 'replacement_observations', 'required_counter_observations']:
                rows.extend((row, ('value', 'reason')) for row in failure.get(name, []))
        for whole in container.get('whole_value_reviews', []):
            for name in ['draft_observations', 'known_observations']:
                rows.extend((row, ('value', 'reason')) for row in whole.get(name, []))
    for row, fields in rows:
        restored = {}
        for key, value in row.items():
            field = next((field for field in fields if key == 'review_' + field + '_ref'), None)
            if field:
                restored[field] = pool[value]
            else:
                restored[key] = value
        row.clear()
        row.update(restored)
    return out


def verify(original, applied=True):
    before = exact(original)
    encoded, actual = codec.pack_with_status(original)
    assert actual is applied
    assert exact(original) == before
    assert exact(codec.unpack(encoded)) == before
    assert exact(independent_inverse(encoded)) == before
    assert exact(codec.pack_with_status(original)[0]) == exact(encoded)
    if applied:
        assert len(exact(encoded).encode()) + len(codec.INSTRUCTION.encode()) < len(before.encode())
    else:
        assert exact(encoded) == before
    return encoded


def test_no_parent_all_registered_review_paths_roundtrip_with_roles_intact():
    data = review_fixture()
    encoded = verify(data)
    assert len(encoded['shared_review_texts']) == 2
    assert 'review_value_ref' in encoded['known_observations'][0]
    for container in [*encoded['failed_batch_reviews'], encoded['repair_scope']]:
        assert container.get('review_allowed', False) is False
        failure = container['failed'][0]
        rows = [failure['observation'], *failure['slot']['observations'],
                *failure['observations'], *failure['replacement_observations'],
                *failure['required_counter_observations']]
        for whole in container['whole_value_reviews']:
            rows += whole['draft_observations'] + whole['known_observations']
        rows += container.get('validated_observations', [])
        assert all('review_value_ref' in row and 'review_reason_ref' in row for row in rows)
    assert exact(encoded['source_spans']) == exact(data['source_spans'])
    assert encoded['failed_batch_reviews'][0]['invalid_response'] == data['failed_batch_reviews'][0]['invalid_response']
    encoded['known_observations'][0]['extra'].append('mutated')
    assert exact(data) == exact(review_fixture())


def test_parent_value_only_and_exact_unicode_no_normalization():
    data = review_fixture()
    data['supported_parents']['history'] = {'before': True, 'value': text_value(),
                                           'reason': text_value(), 'after': 1.0}
    data['known_observations'].append({'value': text_value().replace('e\u0301', 'é')})
    encoded = verify(data)
    assert list(encoded['supported_parents']['history']) == ['before', 'review_value_ref', 'reason', 'after']
    assert encoded['supported_parents']['history']['reason'] == text_value()
    assert 'value' in encoded['known_observations'][-1]


def test_nonstring_values_nested_refs_and_unregistered_rows_remain_literal():
    data = review_fixture()
    values = [None, False, True, 0, 1, 0.0, 1.0, ['r0'],
              {'review_value_ref': 'r0', 'nested': {'value': text_value()}},
              {'shared': 0, 'review_allowed': True}]
    for value in values:
        data['known_observations'].append({'value': value, 'reason': deepcopy(value)})
    data['frozen_observation_reviews'] = [{'review_value_ref': 'literal', 'value': text_value()}]
    data['application'] = {'observations': [{'review_reason_ref': 'literal', 'reason': text_value()}]}
    encoded = verify(data)
    assert exact(encoded['known_observations'][1:]) == exact(data['known_observations'][1:])
    assert exact(encoded['frozen_observation_reviews']) == exact(data['frozen_observation_reviews'])
    assert exact(encoded['application']) == exact(data['application'])


@pytest.mark.parametrize('field', ['review_value_ref', 'review_reason_ref'])
@pytest.mark.parametrize('where', ['parent', 'known', 'counter', 'validated', 'repair'])
def test_reserved_collision_in_noncandidate_row_falls_back_everywhere(field, where):
    data = review_fixture()
    row = {'value': 1, field: 'caller-owned', 'extra': False}
    if where == 'parent':
        data['supported_parents']['other'] = row
    elif where == 'known':
        data['known_observations'].append(row)
    elif where == 'counter':
        data['failed_batch_reviews'][0]['failed'][0]['required_counter_observations'].append(row)
    elif where == 'validated':
        data['failed_batch_reviews'][0]['validated_observations'].append(row)
    else:
        data['repair_scope']['whole_value_reviews'][0]['draft_observations'].append(row)
    encoded, applied = codec.pack_with_status(data)
    assert not applied and exact(encoded) == exact(data)
    with pytest.raises(ValueError):
        codec.unpack(encoded)


def test_existing_pool_and_repeated_encoding_do_not_activate_again():
    encoded = verify(review_fixture())
    twice, applied = codec.pack_with_status(encoded)
    assert not applied and exact(twice) == exact(encoded)
    data = review_fixture()
    data['shared_review_texts'] = {'r0': 'preexisting literal'}
    result, applied = codec.pack_with_status(data)
    assert not applied and exact(result) == exact(data)


@pytest.mark.parametrize('change', ['parents', 'known', 'batch', 'failed', 'slot', 'observation',
                                   'validated', 'whole', 'repair', 'table'])
def test_malformed_registered_containers_fall_back_without_partial_change(change):
    data = review_fixture()
    batch = data['failed_batch_reviews'][0]
    failure = batch['failed'][0]
    if change == 'parents': data['supported_parents'] = []
    elif change == 'known': data['known_observations'].append([])
    elif change == 'batch': data['failed_batch_reviews'].append(None)
    elif change == 'failed': batch['failed'] = {}
    elif change == 'slot': failure['slot'] = []
    elif change == 'observation': failure['observation'] = 'broken'
    elif change == 'validated': batch['validated_observations'] = None
    elif change == 'whole': batch['whole_value_reviews'].append(False)
    elif change == 'repair': data['repair_scope'] = []
    else: data['known_observation_columns'] = ['value', 'value']
    encoded, applied = codec.pack_with_status(data)
    assert not applied and exact(encoded) == exact(data)
    with pytest.raises(ValueError):
        codec.unpack(encoded)


def test_nullable_failure_observation_and_slot_are_real_supported_producer_shapes():
    data = review_fixture()
    data['failed_batch_reviews'][0]['failed'].append({'observation': None, 'slot': None})
    verify(data)


@pytest.mark.parametrize('change', ['pool_list', 'pool_empty', 'pool_value', 'dangling',
                                   'bool_ref', 'number_ref', 'both_value', 'both_reason', 'parent_reason'])
def test_decoder_rejects_invalid_pool_and_reference_shapes(change):
    encoded = verify(review_fixture())
    row = encoded['known_observations'][0]
    if change == 'pool_list': encoded['shared_review_texts'] = []
    elif change == 'pool_empty': encoded['shared_review_texts'] = {}
    elif change == 'pool_value': encoded['shared_review_texts']['r0'] = {'review_value_ref': 'r0'}
    elif change == 'dangling': row['review_value_ref'] = 'absent'
    elif change == 'bool_ref': row['review_value_ref'] = True
    elif change == 'number_ref': row['review_value_ref'] = 0
    elif change == 'both_value': row['value'] = text_value()
    elif change == 'both_reason': row['reason'] = text_value()
    else: encoded['supported_parents']['parent'] = {'review_reason_ref': 'r0'}
    with pytest.raises(ValueError):
        codec.unpack(encoded)


@pytest.mark.parametrize('raw', [' {"a":1,"a":2,"value":"\\u0061"} ',
                                 '{"value":"unterminated', 'not JSON\n\t\\"', None])
def test_raw_invalid_response_is_never_parsed_or_reserialized(raw):
    data = review_fixture()
    data['failed_batch_reviews'][0]['invalid_response'] = raw
    data['repair_scope']['invalid_response'] = raw
    encoded = verify(data)
    assert exact(encoded['repair_scope']['invalid_response']) == exact(raw)


def test_old_known_table_remains_byte_exact_even_with_literal_reference_shapes():
    data = review_fixture()
    data['known_observation_columns'] = ['id', 'value', 'reason', 'metadata']
    data['known_observation_values'] = [text_value(), {'shared': 0}, {'review_value_ref': 'literal'}]
    data['known_observations'] = [['known', {'shared': 0}, {'shared': 1}, {'shared': 2}],
                                  ['other', {'review_value_ref': 'literal'}, False, 1.0]]
    encoded = verify(data)
    for name in ['known_observations', 'known_observation_columns', 'known_observation_values']:
        assert exact(encoded[name]) == exact(data[name])


def test_no_net_benefit_including_instruction_falls_back():
    verify({'known_observations': [{'value': 'tiny', 'reason': 'tiny'}]}, applied=False)
    value = 'x' * (len(codec.INSTRUCTION.encode()) // 2)
    verify({'known_observations': [{'value': value}, {'value': value}]}, applied=False)


@pytest.mark.parametrize('use_history,use_review', [(False, False), (False, True), (True, False), (True, True)])
def test_prepare_keeps_independent_flags_and_inverse_layer_order(monkeypatch, use_history, use_review):
    reader, _, _, _ = fixture()
    monkeypatch.setattr(reader, '_wire_data', deepcopy)
    monkeypatch.setattr(reader, '_system', lambda wire: 'BASE')
    data = review_fixture() if use_review else {'supported_parents': {}, 'known_observations': []}
    data['grounded_subquestion'] = 'Read history'
    if use_history:
        value = 'Another history string. ' * 120
        data['supported_parents']['history'] = {'type': 'text', 'value': value}
        data['grounded_subquestion'] = 'Read ' + value + ' now.'
    old_wire, old_applied = history_wire.pack_with_status(data)
    assert old_applied is use_history
    wire, system = reader._prepare_wire(data)
    assert (history_wire.INSTRUCTION in system) is use_history
    assert (codec.INSTRUCTION in system) is use_review
    assert exact(codec.unpack(wire)) == exact(old_wire)
    # History's older codec contract allows key reordering; this layer does not.
    assert history_wire.unpack(codec.unpack(wire)) == history_wire.unpack(old_wire)
    assert system == 'BASE' + (history_wire.INSTRUCTION if use_history else '') + (codec.INSTRUCTION if use_review else '')
    assert exact(reader._prepare_wire(data)[0]) == exact(wire)


@pytest.mark.parametrize('collision', ['shared_history_texts', 'grounded_subquestion_parts',
                                      'parent_value_ref', 'known_value_ref'])
def test_history_collision_leaves_history_literal_while_independent_review_can_apply(monkeypatch, collision):
    reader, _, _, _ = fixture()
    monkeypatch.setattr(reader, '_wire_data', deepcopy)
    monkeypatch.setattr(reader, '_system', lambda wire: 'BASE')
    data = review_fixture()
    data['grounded_subquestion'] = text_value()
    data['supported_parents']['history'] = {'type': 'text', 'value': text_value()}
    if collision == 'shared_history_texts':
        data[collision] = {'t0': 'caller-owned literal'}
    elif collision == 'grounded_subquestion_parts':
        data[collision] = [{'text_ref': 'caller-owned literal'}]
    elif collision == 'parent_value_ref':
        data['supported_parents']['history']['value_ref'] = 'caller-owned literal'
    else:
        data['known_observations'][0]['value_ref'] = 'caller-owned literal'
    before = exact(data)
    wire, system = reader._prepare_wire(data)
    assert history_wire.INSTRUCTION not in system
    assert codec.INSTRUCTION in system
    assert exact(independent_inverse(wire)) == before
    assert exact(codec.unpack(wire)) == before
    assert exact(data) == before


def test_estimate_receives_same_prepared_wire_without_state_schema_or_budget_changes(monkeypatch):
    reader, engine, state, calls = fixture()
    data = reader._payload(engine.steps[0], 'Read history', ['history'], ['s1'])
    reviews = review_fixture()
    data['failed_batch_reviews'] = reviews['failed_batch_reviews']
    data['known_observations'] = reviews['known_observations']
    before = exact((data, state.public_dict(), engine.mapper.sources, engine.ledger.used))
    schema = reader._schema(data)
    captured = []
    def estimate(operation, system, wire, actual_schema, **options):
        captured.append((system, deepcopy(wire), actual_schema, options))
        return 123
    monkeypatch.setattr(engine.reasoner, 'estimate', estimate)
    assert reader._estimate('joint_read_repair', data, output_token_cap=127) == 123
    system, wire, actual_schema, options = captured[0]
    assert codec.INSTRUCTION in system and 'shared_review_texts' in wire
    assert exact(wire) == exact(reader._prepare_wire(data)[0])
    assert system == reader._prepare_wire(data)[1]
    assert exact(actual_schema) == exact(schema) and options == {'output_token_cap': 127}
    assert exact((data, state.public_dict(), engine.mapper.sources, engine.ledger.used)) == before
    assert not calls.requests


@pytest.mark.parametrize('field', ['review_value_ref', 'review_reason_ref'])
@pytest.mark.parametrize('replace_literal', [True, False])
def test_model_output_cannot_return_input_references(field, replace_literal):
    reader, _, state, calls = fixture()
    row = observation()
    if replace_literal:
        del row['value' if field == 'review_value_ref' else 'reason']
    row[field] = 'r0'
    before = exact(state.public_dict())
    with pytest.raises(ProtocolError):
        reader._observation(row, 'history', visible(reader), reader.e.steps[0])
    assert exact(state.public_dict()) == before and not calls.requests
