"""Summary revision permission is a frozen, explicit Persona history contract."""
from copy import deepcopy

import pytest

from dagbt.engine import Engine, legacy_modules
from dagbt.methods import RESIDUAL_V2
from dagbt.personal_task import validate_personal_plan
from dagbt.reasoning import ProtocolError
from dagbt.residual_plan import planner_contract, validate_plan
from test_v4_personal_task import OPTIONS, QUESTION, plan
from test_residual_runtime import demo


def summary_plan():
    value = plan()
    value['program']['variables']['history']['evidence_semantics'] = 'personal_history_summary'
    return value


def personal_validate(value):
    _, repair, _ = legacy_modules()
    return validate_personal_plan(
        validate_plan(value, repair.validate_plan, personal_history=True), QUESTION)


def test_declared_summary_roundtrips_without_granting_missing_or_atomic_permission():
    value = summary_plan()
    before = deepcopy(value)
    accepted = personal_validate(value)
    assert accepted == before == value
    assert 'evidence_semantics' not in accepted['program']['variables']['response']
    atomic = plan()
    atomic['program']['variables']['history']['evidence_semantics'] = 'personal_atomic_fact'
    assert personal_validate(atomic)['program']['variables']['history']['evidence_semantics'] == 'personal_atomic_fact'
    missing = plan()
    del missing['program']['variables']['history']['evidence_semantics']
    with pytest.raises(ProtocolError, match='must explicitly choose'):
        personal_validate(missing)
    assert 'evidence_semantics' not in missing['program']['variables']['history']


def test_generic_planner_neither_offers_nor_accepts_personal_summary_marker():
    legacy, repair, _ = legacy_modules()
    generic = planner_contract(legacy.PLAN_SCHEMA)
    personal = planner_contract(legacy.PLAN_SCHEMA, personal_history=True)
    declaration = lambda schema: schema['properties']['program']['properties']['variables']['additionalProperties']
    assert 'evidence_semantics' not in declaration(generic)['properties']
    assert declaration(personal)['properties']['evidence_semantics']['enum'] == ['personal_history_summary', 'personal_atomic_fact']
    with pytest.raises(ProtocolError, match='explicit PersonaMem task'):
        validate_plan(summary_plan(), repair.validate_plan)


@pytest.mark.parametrize('kind', ['bool', 'entity', 'number', 'set', 'record', 'rule'])
def test_non_text_fact_cannot_claim_summary_revision(kind):
    value = summary_plan()
    value['program']['variables']['history']['type'] = kind
    with pytest.raises(ProtocolError, match='unrestricted text'):
        personal_validate(value)


@pytest.mark.parametrize('mutation', ['finite_domain', 'unit', 'deterministic', 'terminal', 'no_scope', 'unknown_marker'])
def test_summary_marker_cannot_relax_other_frozen_contracts(mutation):
    value = summary_plan()
    history = value['program']['variables']['history']
    if mutation == 'finite_domain':
        history['domain'] = ['one possible fact']
    elif mutation == 'unit':
        history['unit'] = 'day'
    elif mutation == 'deterministic':
        value['steps'].insert(0, demo.step('raw', 'Find an exact historical statement'))
        value['steps'][1].update(execution='compose', inputs=['raw'])
        value['program']['variables']['raw'] = {'type': 'text', 'demand': 'raw',
            'scope': {'entity_scope': 'user', 'time_scope': 'recorded history'}}
        value['program']['task_expressions'] = {'history': {'op': 'var', 'id': 'raw'}}
    elif mutation == 'terminal':
        value['program']['variables']['response']['evidence_semantics'] = 'personal_history_summary'
    elif mutation == 'no_scope':
        del history['scope']
    else:
        history['evidence_semantics'] = 'overwrite_anything'
    with pytest.raises(ProtocolError, match='[Pp]ersonal evidence semantics'):
        personal_validate(value)


def test_actual_personal_engine_plan_preserves_explicit_summary_permission():
    import numpy as np
    value = summary_plan()
    services = demo.SourcesOnly(value)
    engine = Engine({'id': 'summary-plan', 'question': QUESTION},
        ({}, [], np.empty((0, 3), np.float32), None, demo.Tokenizer()), services,
        {'_test_transport': True, 'llm_base_url': 'http://synthetic/v1', 'llm_model': 'script',
         'fusion': {'max_repairs_per_request': 0}}, RESIDUAL_V2,
        reader_question=QUESTION, output_options=OPTIONS, task_kind='personal_response_selection')
    engine.plan()
    assert engine.fact_state.initial_program == value['program']
    assert all(not request['url'].endswith('/embeddings') for request in services.requests)
