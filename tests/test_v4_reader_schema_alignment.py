"""Mirror established obligation and frozen-row guards in the offered schema.

These examples exercise request-specific offers, not answer quality. Existing
runtime validators remain authoritative when a plain-mode model ignores schema.
"""
from copy import deepcopy

import pytest
from jsonschema import Draft202012Validator

from dagbt.joint_reading import JointReader
from test_personamem_joint_protocol import fixture, observation, response


def offered_data(*, frozen=False):
    reader, engine, _, _ = fixture()
    data = reader._payload(engine.steps[0], 'Review scoped history', ['history'], ['s1'])
    data['evidence_obligations'] = [{'id': 'issue', 'variable': 'history',
        'source_ids': ['s1'], 'descriptions': ['A real scoped issue remains.']}]
    if frozen:
        data['repair_scope'] = {'failed': [{'variable': 'history', 'id': '*',
            'slot': response(observation('frozen'))['slots'][0],
            'frozen_observation_reviews': {'slot_observation_ids': ['frozen'],
                                         'observations': []}}]}
    return data


def accepts(data, result):
    schema = JointReader._schema(data)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema).is_valid(result)


def test_claimed_obligation_resolution_cannot_keep_an_unresolved_issue_in_schema():
    data = offered_data()
    result = response(observation())
    result['slots'][0].update(resolved_obligation_ids=['issue'], unresolved=['Still missing a premise.'])
    assert not accepts(data, result)
    result['slots'][0]['unresolved'] = []
    assert accepts(data, result)  # Source-coverage validation still runs later.


@pytest.mark.parametrize('use_reference', [False, True])
def test_offered_frozen_id_cannot_be_resubmitted_as_an_observation(use_reference):
    data = offered_data(frozen=True)
    result = response(observation('frozen'))
    if use_reference:
        result['slots'][0]['reviewed_observation_ids'] = ['frozen']
    assert not accepts(data, result)
    result['slots'][0]['observations'] = []
    result['slots'][0]['reviewed_observation_ids'] = ['frozen']
    assert accepts(data, result)


@pytest.mark.parametrize('resolution', [None, []])
def test_conservative_unresolved_slot_remains_legal_without_a_resolution_claim(resolution):
    data = offered_data()
    result = response()
    result['slots'][0]['unresolved'] = ['Still missing a premise.']
    if resolution is not None:
        result['slots'][0]['resolved_obligation_ids'] = resolution
    assert accepts(data, result)


def test_frozen_exclusion_does_not_apply_to_other_variables_or_known_routes():
    data = offered_data(frozen=True)
    data['variables']['other'] = deepcopy(data['variables']['history'])
    data['requested_slots'].append('other')
    data['requested_scopes']['other'] = deepcopy(data['requested_scopes']['history'])
    result = response(observation('new-review'))
    result['slots'].append({'variable': 'other', 'observations': [observation('frozen')], 'unresolved': []})
    assert accepts(data, result)
    data['repair_scope']['failed'] = []
    data['known_observations'] = [{**observation('frozen'), 'variable': 'history'}]
    result['slots'][0]['observations'] = [observation('frozen')]
    assert accepts(data, result)


def test_complete_frozen_rows_have_the_same_exclusion_as_slot_references():
    data = offered_data(frozen=True)
    offer = data['repair_scope']['failed'][0]['frozen_observation_reviews']
    offer.update(slot_observation_ids=[], observations=[observation('frozen')])
    assert not accepts(data, response(observation('frozen')))


def test_failed_batch_resolution_may_preserve_a_real_semantic_gap():
    data = offered_data()
    data['failed_batch_reviews'] = [{'id': 'failed', 'variable': 'history', 'review_allowed': True}]
    row = observation('corrected-draft', stance='partial')
    row['value'] = None
    result = response(row)
    result['slots'][0].update(resolved_failed_batch_ids=['failed'], unresolved=['The corrected partial still needs evidence.'])
    assert accepts(data, result)


def test_schema_build_does_not_modify_offers_or_original_rows():
    data = offered_data(frozen=True)
    before = deepcopy(data)
    JointReader._schema(data)
    assert data == before
