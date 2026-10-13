"""Controller-to-reader source scope, with real state and scripted services."""
from copy import deepcopy
import json

import numpy as np
import pytest

from test_v4_control import advice_plan, decoded_request, demo


class ScopedServices(demo.SourcesOnly):
    """Interpret one synthetic preference; other spans supply no new premise."""

    def __init__(self, plan):
        super().__init__(plan, scenario='advice')
        self.reads = []

    def get(self, stage, url, payload):
        result = super().get(stage, url, payload)
        if stage[0] == 'planner' or url.endswith('/embeddings'):
            return result
        data = decoded_request(payload)
        self.reads.append((stage[0], deepcopy(data)))
        primary = next(source for source in data['source_spans']
                       if 'Preference 0: quiet places.' in source['text'])
        audit = 'audit_instruction' in data
        values = {'x': 0, 'middle': 'The visitor prefers quiet places.',
                  'advice': 'Choose a quiet visit based on the sourced preference.'}
        slots = []
        for variable, declaration in data['variables'].items():
            scope = data['requested_scopes'][variable]
            parents = next(step['inputs'] for step in self.plan['steps']
                           if step['output_slot'] == declaration['demand'])
            rows = [] if audit else [{
                'id': variable + '-proof', 'stance': 'support', 'value': values[variable],
                'source_ids': [primary['id']], **scope,
                'parent_versions': {name: binding['version']
                    for name, binding in data['supported_parents'].items()
                    if binding['demand'] in parents},
                'retract_ids': [], 'reason': 'The cited synthetic preference establishes this value.'}]
            slots.append({'variable': variable, 'observations': rows, 'unresolved': []})
        response = {'slots': slots}
        if audit:
            response['audit_verdict'] = 'supported'
        elif stage[0] == 'semantic_compose':
            response.update(semantic_answer=values['advice'], final_prediction=values['advice'])
        result['response']['choices'][0]['message']['content'] = json.dumps(response)
        return result


def run_scoped(*, intermediate=False):
    plan = advice_plan()
    if intermediate:
        plan['steps'].insert(1, demo.step('middle', 'Describe {preference}', ['preference'], 'compose'))
        plan['steps'][-1]['inputs'] = ['middle']
        plan['steps'][-1]['question'] = 'Recommend from {middle}'
        plan['program']['variables']['middle'] = {'type': 'text', 'demand': 'middle'}
    service = ScopedServices(plan)
    # All spans are mapped and read during retrieval. Only the first establishes
    # the synthetic preference; sharing its document does not make others premises.
    passage = 'Preference 0: quiet places.\n' + ('Decorative context with 0 new preference facts. ' * 40)
    row, _, engine = demo.run(service, {'x': passage}, np.array([[1, 0, 0]], np.float32),
        settings={'max_quote_chars': 120, 'max_repairs_per_request': 0})
    return row, engine, service


@pytest.mark.parametrize('intermediate', [False, True])
def test_controller_composition_and_audit_reproduce_proofs_without_reselecting_parent_documents(intermediate):
    row, engine, service = run_scoped(intermediate=intermediate)
    assert row['answer']['status'] == 'ok'
    assert engine.audit_complete and not engine.fact_state.pending
    assert not engine.residual_control.reader.unread_obligations
    primary = next(alias for alias, source in engine.mapper.sources.items()
                   if 'Preference 0: quiet places.' in source['text'])
    assert len(engine.mapper.sources) > 5
    retrieval = [data for operation, data in service.reads
                 if data['task']['output_slot'] == 'preference' and operation == 'joint_read']
    assert retrieval and any(len(data['source_spans']) > 1 for data in retrieval)
    composed = [(operation, data) for operation, data in service.reads
                if data['task']['execution'] == 'compose' or operation == 'audit_joint']
    assert len(composed) == (3 if intermediate else 2)
    assert sum(operation == 'audit_joint' for operation, _ in composed) == 1
    for _, data in composed:
        assert {source['id'] for source in data['source_spans']} == {primary}
        assert data['known_observations'] and data['supported_parents']
        assert all(binding['source_ids'] == [primary] for binding in data['supported_parents'].values())
    if intermediate:
        middle = [data for operation, data in composed
                  if operation == 'joint_read' and data['task']['output_slot'] == 'middle']
        assert len(middle) == 1  # The nonterminal composition really called the model.
        assert engine.fact_state.versions['middle'] == 1
        audit = next(data for operation, data in composed if operation == 'audit_joint')
        assert set(audit['task']['inputs']) == {'preference', 'middle'}
        assert set(audit['supported_parents']) == {'x', 'middle'}

