"""Phase boundaries during recovery after a rejected terminal answer."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from test_v4_joint_reading import prepare_terminal


@pytest.mark.parametrize('phase', ['history_evidence', 'response_selection', 'response_audit'])
def test_recovery_answer_is_only_exposed_to_response_stages_without_losing_history(phase):
    reader, engine, state, _ = prepare_terminal(personal=True)
    feedback = {'previous_semantic_answer': engine.output_options[1],
                'previous_prediction': 'b', 'reason': 'audit_rejected',
                'correction_round': 1}
    engine.residual_control = SimpleNamespace(
        rules_completion_needed=lambda: False, task_projection=lambda task: None,
        audit_feedback=deepcopy(feedback))
    before = deepcopy(state.public_dict())
    reader.current_terminal_output = phase == 'response_selection'
    data = reader._payload(engine.steps[0] if phase == 'history_evidence' else engine.steps[1],
                           'Check historical activity', ['history'], ['s1'],
                           semantic=phase == 'response_selection', audit=phase == 'response_audit')
    wire = reader._wire_data(data)
    assert data['phase'] == phase
    assert data['known_observations'] and data['source_spans']
    if phase == 'history_evidence':
        assert not {'audit_correction', 'public_options', 'terminal_candidate'} & data.keys()
        assert feedback['previous_semantic_answer'] not in json.dumps(wire)
    else:
        assert data['audit_correction'] == feedback
        assert data['public_options'] == engine.output_options
    assert engine.residual_control.audit_feedback == feedback
    assert state.public_dict() == before
