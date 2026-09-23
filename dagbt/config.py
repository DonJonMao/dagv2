"""Explicit experimental factors and shared per-question budgets."""
from __future__ import annotations
from collections.abc import Mapping
from copy import deepcopy

DEFAULTS = {
    'ann_calls': 36, 'set_score_calls': 512, 'llm_calls': 24, 'reader_calls': 1,
    'reserved_gap_ann_calls': 2, 'reserved_audit_calls': 1,
    'json_repairs': 2, 'max_initial_nodes': 6, 'max_refinement_nodes': 2,
    'max_alternatives': 2, 'max_enumeration_states': 10000,
    'initial_width': 12, 'proposal_width': 4, 'pair_rescue_width': 4,
    'context_tokens': 16384, 'map_batch_tokens': 6144,
    'reasoning_output_tokens': 4096, 'reader_output_tokens': 1024,
    'max_quote_chars': 400, 'max_feedback_rounds': 2,
    'selection': 'dependency', 'retrieval': 'bridge', 'proxy_mode': 'activation',
    'condition_audit': True, 'allow_alternatives': True,
    'invalidation': True, 'refinement': True, 'reader_chain': False,
    'navigation_closure': False,
    'search': {},
}
# Full factorial discovery x selection has identical solver and raw reader.
# Legacy dagv2 is a separate unchanged method, never called a matched control.
METHODS = {
    'fusion': {},
    'fusion_proxy_free': {'proxy_mode': 'none'},
    'dense_dependency': {'retrieval': 'dense'},
    'bt_flat': {'selection': 'flat'},
    'dense_flat': {'retrieval': 'dense', 'selection': 'flat'},
    'fusion_no_conditions': {'condition_audit': False},
    'fusion_single_support': {'allow_alternatives': False},
    'fusion_no_invalidation': {'invalidation': False},
    'fusion_fixed_dag': {'refinement': False},
    'fusion_chain': {'reader_chain': True},
    'fusion_navigation_closure': {'navigation_closure': True},
}

def resolve(config, method='fusion'):
    if method not in METHODS:
        raise ValueError(f'Unknown fusion method: {method}')
    if not isinstance(config, Mapping) or not isinstance(config.get('fusion', {}), Mapping):
        raise ValueError('Configuration and fusion settings must be mappings')
    supplied = config.get('fusion', {})
    unknown = set(supplied) - set(DEFAULTS)
    if unknown:
        raise ValueError('Unknown fusion settings: ' + ', '.join(sorted(map(str, unknown))))
    result = {**deepcopy(DEFAULTS), **deepcopy(dict(supplied)), **METHODS[method]}
    for name in ('ann_calls','set_score_calls','llm_calls','reader_calls','context_tokens',
                 'map_batch_tokens','reasoning_output_tokens','reader_output_tokens',
                 'max_initial_nodes','max_alternatives','max_enumeration_states',
                 'initial_width','proposal_width','max_quote_chars'):
        if not isinstance(result[name], int) or isinstance(result[name], bool) or result[name] < 1:
            raise ValueError(f'{name} must be positive integer')
    for name in ('reserved_gap_ann_calls','reserved_audit_calls','max_refinement_nodes',
                 'max_feedback_rounds','json_repairs','pair_rescue_width'):
        if not isinstance(result[name], int) or isinstance(result[name], bool) or result[name] < 0:
            raise ValueError(f'{name} must be nonnegative integer')
    for name in ('condition_audit','allow_alternatives','invalidation','refinement','reader_chain','navigation_closure'):
        if not isinstance(result[name], bool):
            raise ValueError(f'{name} must be boolean')
    if result['selection'] not in ('dependency','flat') or result['retrieval'] not in ('bridge','dense'):
        raise ValueError('selection must be dependency/flat and retrieval must be bridge/dense')
    if result['navigation_closure'] and result['selection'] != 'dependency':
        raise ValueError('navigation_closure requires dependency selection; flat-navigation combination is undefined')
    if result['proxy_mode'] not in ('activation','none'):
        raise ValueError('proxy_mode must be activation or none')
    if not isinstance(result['search'], Mapping):
        raise ValueError('search must be a mapping of EvidenceSearchConfig settings')
    if result['search']:
        # Reuse the actual vendored search contract rather than maintaining a
        # second validator that could drift from its scheduling implementation.
        from vendor.bridgetree.evidence_config import EvidenceSearchConfig
        try:
            EvidenceSearchConfig(**dict(result['search']))
        except (TypeError, ValueError) as exc:
            raise ValueError(f'Invalid Evidence BridgeTree search settings: {exc}') from exc
    result['search'] = deepcopy(dict(result['search']))
    if result['max_initial_nodes'] > 6 or result['max_refinement_nodes'] > 2:
        raise ValueError('Initial/refinement nodes exceed the audited 6+2 finite graph')
    if result['max_alternatives'] > 2:
        raise ValueError('At most two support alternatives per node')
    if result['reserved_gap_ann_calls'] >= result['ann_calls']:
        raise ValueError('ANN gap reservation leaves no discovery budget')
    final_calls = result['reserved_audit_calls'] + int(result['selection'] == 'flat')
    if final_calls >= result['llm_calls']:
        raise ValueError('LLM audit/selection reservation leaves no reasoning budget')
    if result['reasoning_output_tokens'] + 8 >= result['context_tokens']:
        raise ValueError('Reasoning output reserve leaves no input context')
    if result['reader_output_tokens'] + 8 >= result['context_tokens']:
        raise ValueError('Reader output reserve leaves no input context')
    if result['map_batch_tokens'] + result['reasoning_output_tokens'] + 8 > result['context_tokens']:
        raise ValueError('Map batch plus reasoning output reserve exceeds context_tokens')
    return result
