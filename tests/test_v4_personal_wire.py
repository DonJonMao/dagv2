"""Only a duplicate system instruction may be omitted from the user wire."""
from copy import deepcopy

import pytest

from dagbt.joint_reading import JointReader
from dagbt.personal_task import (
    PERSONAL_HISTORY_GUIDANCE, PERSONAL_READING_GUIDANCE, PERSONAL_AUDIT_GUIDANCE,
)
from dagbt.personal_wire import deduplicate_personal_instruction


@pytest.mark.parametrize('phase,instruction,audit', [
    ('history_evidence', PERSONAL_HISTORY_GUIDANCE, False),
    ('response_selection', PERSONAL_READING_GUIDANCE, False),
    ('response_audit', PERSONAL_AUDIT_GUIDANCE, True),
    ('history_evidence', PERSONAL_HISTORY_GUIDANCE, True),
], ids=['history', 'selection', 'audit', 'history_precedes_audit'])
def test_exact_duplicate_is_removed_only_from_user_with_system_and_evidence_unchanged(phase, instruction, audit):
    evidence = {'source_spans': [{'text': 'Exact original source', 'start': 3, 'end': 24}],
                'known_observations': [{'value': {'personal_task_instruction': instruction},
                    'stance': 'contradiction', 'retract_ids': ['old-route'], 'source_ids': ['s1']}],
                'failed_batch_reviews': [{'failed': [{'observation': {'id': 'failed-counter'}}]}]}
    data = {**evidence, 'variables': {}, 'personal_response_selection': True, 'phase': phase,
            'personal_task_instruction': instruction}
    if audit:
        data['audit_instruction'] = None  # _system branches on presence, not truthiness.
    before = deepcopy(data)
    system = JointReader._system(data)
    wire = deduplicate_personal_instruction(data)
    assert data == before
    assert wire == {key: value for key, value in before.items() if key != 'personal_task_instruction'}
    assert JointReader._system(wire) == system and instruction in system
    for key in evidence:
        assert wire[key] is data[key]  # No traversal or rewrite of evidence/typed values.


@pytest.mark.parametrize('personal,phase,audit,instruction', [
    (False, 'history_evidence', False, PERSONAL_HISTORY_GUIDANCE),
    (True, 'history_evidence', False, PERSONAL_HISTORY_GUIDANCE + '\nKeep this custom rule.'),
    (True, 'response_audit', False, PERSONAL_AUDIT_GUIDANCE),
    (True, 'response_selection', True, PERSONAL_READING_GUIDANCE),
    (True, 'history_evidence', True, PERSONAL_AUDIT_GUIDANCE),
], ids=['nonpersonal', 'custom', 'audit_phase_without_audit_branch',
        'selection_phase_with_audit_branch', 'audit_text_with_history_branch'])
def test_nonpersonal_custom_or_different_actual_system_branch_is_preserved(personal, phase, audit, instruction):
    data = {'personal_response_selection': personal, 'phase': phase,
            'personal_task_instruction': instruction}
    if audit:
        data['audit_instruction'] = None
    before = deepcopy(data)
    assert deduplicate_personal_instruction(data) == before
    assert data == before


def test_absent_instruction_is_not_created():
    data = {'personal_response_selection': True, 'phase': 'history_evidence'}
    assert deduplicate_personal_instruction(data) == data
