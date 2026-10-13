"""Remove only the exact personal instruction already sent in the system role."""


def deduplicate_personal_instruction(data):
    """Return a shallow wire copy, preserving all canonical evidence objects.

    This mirrors JointReader._system's actual branch selection, including the
    history-phase precedence and audit_instruction *presence* check. A custom
    instruction, or one different from the instruction selected for the system
    message, must remain in the user payload. No typed value, observation,
    source, counter, route, or nested object is inspected or changed.
    """
    packed = dict(data)
    if not data.get('personal_response_selection'):
        return packed
    from .personal_task import (
        PERSONAL_HISTORY_GUIDANCE, PERSONAL_READING_GUIDANCE, PERSONAL_AUDIT_GUIDANCE,
    )
    if data.get('phase') == 'history_evidence':
        instruction = PERSONAL_HISTORY_GUIDANCE
    elif 'audit_instruction' in data:
        instruction = PERSONAL_AUDIT_GUIDANCE
    else:
        instruction = PERSONAL_READING_GUIDANCE
    if data.get('personal_task_instruction') == instruction:
        packed.pop('personal_task_instruction', None)
    return packed
