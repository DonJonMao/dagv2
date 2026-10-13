"""Public response-selection objective and exact label/text binding regressions."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest

from dagbt import runner
from dagbt.engine import Engine, legacy_modules
from dagbt.personal_task import validate_personal_plan, PERSONAL_PLAN_GUIDANCE
from dagbt.residual_plan import validate_plan
from dagbt.reasoning import ProtocolError
from dagbt.methods import RESIDUAL_V2
from test_residual_runtime import demo

QUESTION = 'I made a coffee table from reclaimed wood.'
OPTIONS = ['You enjoy hands-on DIY projects; tell me about the build.',
           'That sounds nice.', 'You dislike making things.', 'You prefer buying every item new.']


def plan():
    return {'steps': [demo.step('history', 'Find the user\'s DIY interests and experiences'),
                      demo.step('response', QUESTION, ['history'], 'compose')],
            'final_node_id': 'response',
            'output_contract': {'mode': 'semantic', 'result_type': 'text', 'explanation': 'semantic',
                                'entity_scope': 'user', 'time_scope': 'reply now'},
            'program': {'variables': {
                'history': {'type': 'text', 'demand': 'history',
                            'scope': {'entity_scope': 'user', 'time_scope': 'recorded history'}},
                'response': {'type': 'text', 'demand': 'response'}},
                'expression': {'op': 'var', 'id': 'response'}}}


def validate(value):
    _, repair, _ = legacy_modules()
    return validate_personal_plan(validate_plan(value, repair.validate_plan), QUESTION)


def test_personal_contract_accepts_evidence_then_response_without_mutation():
    value = plan()
    assert validate(value) == value
    assert 'exact original user utterance' in PERSONAL_PLAN_GUIDANCE


@pytest.mark.parametrize('mutation', ['fact_certificate', 'narrow_terminal', 'retrieval_terminal',
                                      'parent_expression', 'multiple_outputs', 'orphan', 'unbound_task'])
def test_personal_contract_rejects_known_modeling_mismatches(mutation):
    value = plan()
    if mutation == 'fact_certificate':
        value['output_contract'].update(mode='value', explanation='none')
    elif mutation == 'narrow_terminal':
        value['steps'][-1]['question'] = 'What furniture did the user make?'
    elif mutation == 'retrieval_terminal':
        value['steps'][-1]['execution'] = 'retrieval'
    elif mutation == 'parent_expression':
        value['program']['expression']['id'] = 'history'
    elif mutation == 'multiple_outputs':
        value['program']['variables']['extra'] = {'type': 'text', 'demand': 'response'}
    elif mutation == 'orphan':
        value['steps'].insert(0, demo.step('extra', 'Find an unrelated event'))
        value['program']['variables']['extra'] = {'type': 'text', 'demand': 'extra'}
    else:
        del value['program']['variables']['history']
    with pytest.raises(ProtocolError):
        validate(value)


def test_personal_engine_rejects_old_symbolic_objective_before_retrieval():
    value = plan(); value['output_contract'].update(mode='value', explanation='none')
    services = demo.SourcesOnly(value)
    engine = Engine({'id': 'no-fact-shortcut', 'question': QUESTION},
                    ({}, [], np.empty((0, 3), np.float32), None, demo.Tokenizer()), services,
                    {'_test_transport': True, 'llm_base_url': 'http://synthetic/v1',
                     'llm_model': 'script', 'fusion': {'max_repairs_per_request': 0}}, RESIDUAL_V2,
                    reader_question=QUESTION, output_options=OPTIONS, task_kind='personal_response_selection')
    with pytest.raises(ProtocolError, match='semantic/text'):
        engine.run()
    assert all(not r['url'].endswith('/embeddings') for r in services.requests)
    assert OPTIONS[0] not in str(services.requests)


@pytest.mark.parametrize('scope', [None, {'entity_scope':'user', 'time_scope':'reply now'}])
def test_history_cannot_silently_inherit_or_claim_response_scope(scope):
    value = plan()
    if scope is None:
        del value['program']['variables']['history']['scope']
    else:
        value['program']['variables']['history']['scope'] = scope
    with pytest.raises(ProtocolError, match='explicit evidence scope'):
        validate(value)


def test_explicit_smoke_ids_are_public_and_cover_seven_types():
    args = SimpleNamespace(datasets=['personamem'], limit=None,
                           question_ids_file=runner.ROOT / 'configs/personamem-v4-smoke-ids.json')
    selected = runner.experiment_questions(args)['personamem']
    from collections import Counter
    assert len(selected) == 14
    assert set(Counter(q['question_type'] for q in selected).values()) == {2}
    assert all('correct_answer' not in q for q in selected)


@pytest.mark.parametrize('ids', [['unknown'], [], ['known', 'known']])
def test_bad_explicit_scope_rejected(tmp_path, monkeypatch, ids):
    path = tmp_path / 'ids.json'; runner.save(path, {'personamem': ids})
    monkeypatch.setattr(runner, 'load_questions', lambda *args: [{'id': 'known'}])
    with pytest.raises(ValueError):
        runner.experiment_questions(SimpleNamespace(datasets=['personamem'], limit=None, question_ids_file=path))


class PersonalServices:
    """Transport fixture checks every actual request boundary, no remote calls."""
    def __init__(self):
        self.requests = []

    def get(self, stage, url, payload):
        import json
        self.requests.append({'stage': list(stage), 'url': url, 'payload': deepcopy(payload)})
        if url.endswith('/embeddings'):
            return {'response': {'data': [{'index': 0, 'embedding': [1., 0., 0.]}]}}
        data = json.loads(payload['messages'][1]['content'])
        if stage[0] == 'planner':
            assert data == QUESTION and OPTIONS[0] not in str(payload)
            value = plan()
        else:
            if 'source_span_columns' in data:
                columns = data.pop('source_span_columns')
                data['source_spans'] = [dict(zip(columns, row)) for row in data['source_spans']]
            assert data['personal_response_selection']
            audit = stage[0].startswith('audit_joint')
            terminal = stage[0].startswith('semantic_compose')
            assert ('public_options' in data) == (audit or terminal)
            rows = []
            for variable in data['requested_slots']:
                obs = []
                if not audit and (variable == 'response' or not terminal):
                    obs = [{'id': variable + '-source', 'stance': 'support',
                            'value': OPTIONS[0] if variable == 'response' else 'The user enjoys hands-on DIY projects.',
                            'source_ids': [data['source_spans'][0]['id']],
                            **data['requested_scopes'][variable],
                            'parent_versions': ({'history': data['supported_parents']['history']['version']}
                                                if variable == 'response' else {}),
                            'retract_ids': [], 'reason': 'The user explicitly describes enjoying DIY.'}]
                rows.append({'variable': variable, 'observations': obs, 'unresolved': []})
            value = {'slots': rows}
            if terminal:
                value.update(final_prediction='a', semantic_answer=OPTIONS[0])
            if audit:
                assert data['terminal_candidate'] == {'value': OPTIONS[0], 'final_prediction': 'a'}
                value['audit_verdict'] = 'supported'
        return {'response_ref': 'fixture-' + str(len(self.requests)), 'response': {'choices': [
            {'finish_reason': 'stop', 'message': {'content': json.dumps(value)}}]}}


def test_actual_engine_personal_task_ends_in_audited_response_not_fact_certificate():
    text = 'I enjoy hands-on DIY projects, including making furniture from reclaimed wood.'
    doc = SimpleNamespace(doc_id='diy', passage=text, metadata={'source_segments': [
        {'role': 'user', 'start': 0, 'end': len(text), 'source_message_indices': [0]}]})
    services = PersonalServices()
    config = {'_test_transport': True, 'llm_base_url': 'http://synthetic/v1', 'llm_model': 'script',
              'embedding_base_url': 'http://synthetic/v1', 'embedding_model': 'script',
              'fusion': {'initial_width': 1, 'proposal_width': 1}}
    engine = Engine({'id': 'personal-diy', 'question': QUESTION},
        ({'diy': doc}, ['diy'], np.array([[1, 0, 0]], np.float32), None, demo.Tokenizer()),
        services, config, RESIDUAL_V2, reader_question=QUESTION, output_options=OPTIONS,
        task_kind='personal_response_selection')
    row = engine.run()
    assert runner.validate_result(row, 'personal-diy') is row
    assert row['answer']['status'] == 'ok'
    assert row['answer']['semantic_answer'] == OPTIONS[0]
    assert row['answer']['completion_kind'] == 'semantic_assessment'
    assert row['answer']['certificate'] is None
    assert row['diagnostics']['audit_complete']
    assert engine.fact_state.scope('history')['time_scope'] == 'recorded history'
    assert engine.fact_state.scope('response')['time_scope'] == 'reply now'
    assert not any(r['stage'][0] in ('reader', 'select') for r in services.requests)
