"""Structural proof status must not be confused with successful annotation."""
from dagbt.runner import normalized_semantic_evidence, summarize_semantic_evidence


def test_validated_but_incomplete_raw_answer_has_explicit_support_status():
    semantic = normalized_semantic_evidence({'semantic_evidence': {
        'version': 'dagbt_semantic_evidence_v4', 'phase': 'selection_complete',
        'evidence_state': 'raw_only', 'selected_doc_ids': ['raw'],
        'selected_mapped_doc_ids': [], 'selected_raw_only_doc_ids': ['raw'],
        'coverage_validation_complete': True, 'structural_validation_complete': True,
        'complete_required': False, 'review_complete': True}})
    assert semantic['coverage_state'] == 'complete'  # Historical annotation field.
    assert semantic['support_state'] == 'incomplete'
    row = {'valid': True, 'em': 1., 'modules': {'semantic_evidence': semantic}}
    report = summarize_semantic_evidence([row], 'em')
    assert report['all_task_support_state_counts'] == {'complete': 0, 'incomplete': 1, 'unknown': 0}
    assert report['support_completion_cohorts']['incomplete']['em'] == 1.
    assert report['support_completion_cohorts']['complete']['tasks'] == 0


def test_legacy_or_pending_annotations_do_not_acquire_a_complete_proof():
    old = normalized_semantic_evidence({'semantic_evidence': {
        'coverage_validation_complete': True, 'evidence_state': 'mapped_only'}})
    assert old['support_state'] == 'unknown'
    pending = normalized_semantic_evidence({'semantic_evidence': {
        'phase': 'selection_pending', 'coverage_validation_complete': True,
        'structural_validation_complete': True, 'complete_required': True}})
    assert pending['support_state'] == 'unknown'
