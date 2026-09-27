from copy import deepcopy

import pytest

from test_engine import setup, FakeCalls, Engine, step, answer, unknown
from dagbt.reasoning import ProtocolError
from dagbt.support import make_span, validate_span, SupportError, compile_graph, select_support


def test_non_object_resolver_neighbor_does_not_discard_valid_support(setup):
    bridge, resources, config = setup
    bridge.routes = {'answer': ['a']}
    def resolver(data):
        result = answer(data, 'fixture_answer', ['a'])
        result['alternatives'] = (['malformed'] if 'local_repair' in data
                                  else result['alternatives'] + ['malformed'])
        return result
    engine = Engine({'id': 'q', 'question': 'Q?'}, resources,
                    FakeCalls([step('answer')], resolver), config, 'fusion')
    result = engine.run()
    assert result['budgets']['20']['complete_required']
    recovery = result['diagnostics']['reliability']['row_recoveries'][0]
    assert recovery['retained_counts']['alternatives'] == 1
    assert not recovery['complete'] and recovery['repairs'] == 2
    assert result['ranking']['status'] == 'partial'


def test_valid_audit_conflict_survives_non_object_neighbor(setup):
    _, resources, config = setup
    def audit(data):
        aid = data['nodes'][0]['alternatives'][0]['id']
        contrary = next(s['id'] for s in data['evidence'] if s['doc_id'] == 'x')
        conflicts = ['bad'] if 'local_repair' in data else [
            {'alternative_ids': [aid], 'span_ids': [contrary], 'reason': 'Opposing claim'}, 'bad']
        return {'conflicts': conflicts, 'unresolved_guards': []}
    engine = Engine({'id': 'q', 'question': 'Q?'}, resources,
                    FakeCalls([step('answer')], lambda data: answer(data, 'A', ['a']), audit), config, 'fusion')
    engine.plan(); engine.add_candidates(['a', 'x']); engine.map_pending()
    engine.resolve_node(engine.steps[0], 'Q?'); engine.audit()
    assert engine.conflicts
    assert engine.node_map()['answer']['status'] != 'supported'
    assert not engine.row_recoveries[-1]['complete']


def test_visible_parent_assessment_cannot_be_reused_as_direct_support_for_other_node(setup):
    _, resources, config = setup
    def mapper(data):
        return {'units': [{'unit_id': u['unit_id'], 'irrelevance_reason': '', 'assessments': [
            {'span_ids': [u['source_span_id']], 'node_id': 'parent', 'kind': 'explicit',
             'claim': 'Only supports parent', 'stance': 'support', 'entity_scope': 'fixture',
             'event_time': None, 'time_span_ids': [], 'reason': 'No child support'}]} for u in data['units']]}
    def resolver(data):
        if data['node_id'] == 'parent':
            return answer(data, 'A', ['a'])
        value = answer(data, 'B', [], alternatives=[([], [])])
        value['alternatives'][0]['source_span_ids'] = [data['evidence'][0]['id']]
        return value
    engine = Engine({'id': 'q', 'question': 'Q?'}, resources,
                    FakeCalls([step('parent'), step('child')], resolver, mapper=mapper), config, 'fusion')
    engine.plan(); engine.add_candidates(['a']); engine.map_pending(2)
    engine.resolve_node(engine.steps[0], 'Parent?')
    with pytest.raises(ProtocolError):
        engine.resolve_node(engine.steps[1], 'Child?')
    assert engine.node_map()['child']['status'] == 'unknown'


def test_input_window_omission_is_reported_as_truncated_cohort(setup):
    _, resources, config = setup
    engine = Engine({'id': 'q', 'question': 'Q?'}, resources,
                    FakeCalls([step('answer')], lambda data: unknown()), config, 'fusion')
    engine.plan(); engine.add_candidates(['a', 'b']); engine.map_pending()
    # An oversized semantic record must be dropped whole, never clipped into
    # apparently complete evidence. Source fragments remain untouched.
    first = next(iter(engine.spans.values()))
    first['reason'] = '语义说明' * 20000
    engine.resolve_node(engine.steps[0], 'Q?')
    reliability = engine.reliability()
    assert reliability['cohort'] == 'truncated'
    assert first['id'] in reliability['input_views'][-1]['omitted_span_ids']
    assert reliability['mapping_complete']


def test_multifragment_assessment_is_one_graph_premise_and_tampering_is_rejected():
    docs = {'d': 'First fact. Unrelated middle. Last fact.'}
    fragments = [make_span('f1', 'd', 'First fact.', docs, start=0),
                 make_span('f2', 'd', 'Last fact.', docs, start=30)]
    item = validate_span({'id': 'assessment', 'doc_id': 'd', 'fragments': fragments}, docs)
    graph = compile_graph([{'id': 'answer', 'answer': 'A', 'status': 'supported', 'version': 0,
        'alternatives': [{'id': 'route', 'source_span_ids': ['assessment'], 'semantic_status': 'supported'}]}],
        [item], [{'id': 'r', 'necessary': True, 'terminal_node_ids': ['answer'], 'terminal_mode': 'all'}], docs)
    assert len(graph['spans']) == 1 and len(graph['spans'][0]['fragments']) == 2
    assert select_support(graph, lambda ids: {'feasible': True, 'token_count': len(ids)})['selected_doc_ids'] == ['d']
    corrupt = deepcopy(item); corrupt['fragments'][1]['exact_quote'] = 'Fabrication'
    with pytest.raises(SupportError, match='quote_not_exact'):
        validate_span(corrupt, docs)
