"""Reject settings that silently disable safeguards or fail only after model calls."""
import pytest

from dagbt.config import DEFAULTS, resolve


@pytest.mark.parametrize('name,value', [
    ('reserved_gap_ann_calls', -1), ('reserved_audit_calls', -1),
    ('max_refinement_nodes', -1), ('max_feedback_rounds', -1),
    ('max_feedback_rounds', 1.5), ('max_feedback_rounds', '2'),
    ('json_repairs', -1), ('json_repairs', True),
    ('input_margin', -1), ('input_margin', True), ('max_repairs_per_request', -1),
    ('max_quote_chars', -10), ('max_quote_chars', 0), ('max_quote_chars', True),
    ('initial_width', 0), ('proposal_width', False), ('pair_rescue_width', -1),
])
def test_invalid_loop_chunk_and_reservation_settings_rejected_before_execution(name, value):
    with pytest.raises(ValueError, match=name):
        resolve({'fusion': {name: value}})


@pytest.mark.parametrize('settings', [
    {'condition_audit': 'false'}, {'reader_chain': 1}, {'invalidation': None},
    {'selection': 'dependncy'}, {'retrieval': 'bt'}, {'ann_call': 12},
    {'search': []}, {'search': {'coverage_roots': 0}},
    {'search': {'quantum_new_sets': 3}}, {'search': {'nonexistent_switch': True}},
    {'response_format': 'auto'},
])
def test_module_settings_do_not_silently_select_another_method(settings):
    with pytest.raises(ValueError):
        resolve({'fusion': settings})


@pytest.mark.parametrize('settings,method', [
    ({'ann_calls': 2, 'reserved_gap_ann_calls': 2}, 'fusion'),
    ({'llm_calls': 2, 'reserved_audit_calls': 1}, 'bt_flat'),
    ({'context_tokens': 4104}, 'fusion'),
    ({'reader_output_tokens': 16376}, 'fusion'),
    ({'map_batch_tokens': 13000}, 'fusion'),
])
def test_impossible_reservations_and_contexts_fail_during_configuration(settings, method):
    with pytest.raises(ValueError):
        resolve({'fusion': settings}, method)


def test_explicit_zero_optional_budgets_are_supported_without_mutating_defaults():
    supplied = {'fusion': {'reserved_gap_ann_calls': 0, 'reserved_audit_calls': 0,
                          'max_refinement_nodes': 0, 'max_feedback_rounds': 0,
                          'json_repairs': 0, 'pair_rescue_width': 0,
                          'search': {'max_pivots': 0}}}
    result = resolve(supplied, 'fusion_fixed_dag')
    assert result['refinement'] is False
    assert result['pair_rescue_width'] == result['max_feedback_rounds'] == 0
    result['search']['max_pivots'] = 1
    assert supplied['fusion']['search']['max_pivots'] == 0
    assert DEFAULTS['search'] == {} and DEFAULTS['max_feedback_rounds'] == 2


def test_named_ablation_overrides_valid_user_defaults():
    result = resolve({'fusion': {'retrieval': 'bridge', 'selection': 'dependency'}}, 'dense_flat')
    assert result['retrieval'] == 'dense' and result['selection'] == 'flat'


@pytest.mark.parametrize('margin', [0, 256])
def test_map_context_boundary_includes_configured_safety_margin(margin):
    result = resolve({'fusion': {'map_batch_tokens': 12280 - margin, 'input_margin': margin}})
    assert result['map_batch_tokens'] + result['reasoning_output_tokens'] + 8 + margin == result['context_tokens']
    with pytest.raises(ValueError, match='Map batch'):
        resolve({'fusion': {'map_batch_tokens': 12281 - margin, 'input_margin': margin}})


@pytest.mark.parametrize('value', ['true', 'false', 1, None])
def test_navigation_closure_must_be_explicit_boolean(value):
    with pytest.raises(ValueError, match='navigation_closure must be boolean'):
        resolve({'fusion': {'navigation_closure': value}})


@pytest.mark.parametrize('method', ['bt_flat', 'dense_flat', 'fusion_navigation_closure'])
def test_navigation_closure_cannot_silently_run_flat_selection(method):
    with pytest.raises(ValueError, match='navigation_closure requires dependency selection'):
        resolve({'fusion': {'navigation_closure': True, 'selection': 'flat'}}, method)


def test_navigation_closure_named_arm_sets_only_explicit_selection_intervention():
    result = resolve({}, 'fusion_navigation_closure')
    assert result['navigation_closure'] is True and result['selection'] == 'dependency'
    assert result['retrieval'] == 'bridge' and result['reader_chain'] is False
