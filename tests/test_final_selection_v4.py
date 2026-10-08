"""Support-review input bounds, transactional fallback and final Reader gates.

All model responses are explicit fixtures. Source validation and graph selection
are production code; the final-gate tests deliberately corrupt a real selection.
"""
from copy import deepcopy

import pytest

from dagbt.engine import Engine
from dagbt.final_selection import FinalSelector
from dagbt.proof_review import apply_review
from dagbt.reasoning import ProtocolError, RefusalError
from dagbt.support import SupportError, compile_graph, make_span
from dagbt.transport import ServiceError
from test_engine import Document, FakeCalls, answer, setup, step, unknown
from test_final_selection_v3 import make_selector
from test_support_review_v4 import fact, no_mapping, proof, raw_span, review, run_engine, update


def support_selector(responses=(), **kwargs):
    legacy, engine, calls, events = make_selector(responses, **kwargs)
    engine.errors = []
    engine.nodes = deepcopy(legacy.graph['nodes'])
    engine.conflicts = deepcopy(legacy.graph.get('conflicts', []))
    engine.graph = deepcopy(legacy.graph)
    return FinalSelector(engine, legacy.graph, legacy.proposal), engine, calls, events


def supported_selector(responses=(), **kwargs):
    return support_selector(responses, supported=True,
        facts=[('s1', 'a', 'answer', 'support', 'explicit')], **kwargs)


def actual_engine(fixture, *, calls=None, parents=False, guard=False, settings=None):
    bridge, resources, original = fixture
    config = deepcopy(original)
    pool = ['a', 'b'] if parents or not guard else ['a', 'c']
    config['fixed_candidate_pools'] = {'q': pool}
    config['fusion'].update(selection_review=True, condition_audit=False, **(settings or {}))
    steps = ([step('author'), step('answer', 'Birthplace of {author}?', ['author'])]
             if parents else [step('answer')])
    def resolve(data):
        if parents:
            first = data['node_id'] == 'author'
            return answer(data, 'fixture_author' if first else 'fixture_answer',
                          ['a'] if first else ['b'], [] if first else ['author'])
        result = answer(data, 'fixture_answer', ['a'])
        if guard:
            result['alternatives'][0]['guard_span_ids'] = [
                next(span['id'] for span in data['evidence'] if span['doc_id'] == 'c')]
        return result
    calls = calls or FakeCalls(steps, resolve, selector=lambda data: review())
    engine = Engine({'id': 'q', 'question': 'Which relation is supported?'}, resources,
                    calls, config, 'fusion')
    return engine, calls


def test_support_review_keeps_full_raw_records_and_source_roles():
    docs = {'a': Document('a', 'User explicitly described the whole experience.'),
            'b': Document('b', 'Assistant suggested an alternative.')}
    docs['a'].metadata = {'source_segments': [{'start': 0, 'end': len(docs['a'].passage),
        'role': 'user', 'source_message_indices': [3]}], 'observation_order': 3}
    docs['b'].metadata = {'source_segments': [{'start': 0, 'end': len(docs['b'].passage),
        'role': 'assistant', 'source_message_indices': [4]}], 'observation_order': 4}
    selector, engine, _, _ = support_selector(docs=docs)
    view = selector.prepare_view()
    raw = {row['doc_id']: row for row in view.data['raw_memory_candidates']}
    assert set(raw) == {'a', 'b'}
    for doc_id, role, order in [('a', 'user', 3), ('b', 'assistant', 4)]:
        assert raw[doc_id]['passage'] == docs[doc_id].passage
        metadata = raw[doc_id]['metadata']
        assert metadata['source_segments'][0]['role'] == role
        assert metadata['source_segments'][0]['source_message_indices'] == [order]
        assert metadata['observation_order_is_event_time'] is False
    assert engine.input_views[-1]['policy_version'] == 'full_raw_proof_revision_v4'


@pytest.mark.parametrize('attempt_hidden_update', [False, True])
def test_overflow_hides_whole_proof_from_review_but_keeps_it_selectable(attempt_hidden_update):
    # The evidence tokenizer counts UTF-8 bytes; the fixture Reader tokenizer
    # counts words. A long unspaced record fits Reader but not the review input.
    docs = {'a': Document('a', '中' * 20000), 'b': Document('b', 'Small raw record.')}
    def respond(data):
        if not attempt_hidden_update:
            return review()
        return review(node_updates=[{'node_id': 'answer', 'answer': None, 'status': 'unknown',
            'applicable_scope': 'fixture', 'retained_alternative_ids': [], 'alternatives': [],
            'unresolved_inputs': ['The reviewer did not see the source'], 'unresolved_guards': [],
            'reason': 'Attempt to remove an unreviewed route'}])
    selector, engine, calls, _ = supported_selector([respond], docs=docs,
                                                  config={'max_repairs_per_request': 0})
    # Only the original record is large; the existing mapped quotation remains
    # a legal short span, as it would in the actual mapping pipeline.
    span = make_span('s1', 'a', docs['a'].passage[:30], docs, start=0,
        node_ids=['answer'], node_id='answer', stance='support', kind='explicit',
        source_role='document', claim='Fixture claim', assessment_id='s1', premise_group_ids=['s1'])
    selector.graph = compile_graph(selector.graph['nodes'], [span], selector.graph['requirements'], docs)
    engine.spans = {'s1': deepcopy(span)}
    engine.graph = deepcopy(selector.graph)
    before = deepcopy(selector.graph)
    view = selector.prepare_view()
    assert view.audit['omitted_raw_review_doc_ids'] == ['a']
    assert view.audit['raw_review_visible_doc_ids'] == ['b']
    assert view.audit['omitted_alternative_ids'] == ['answer_alt']
    assert view.data['nodes'][0]['alternatives'] == []
    assert view.data['nodes'][0]['omitted_alternative_ids'] == ['answer_alt']
    assert view.data['evidence'] == []
    assert view.data['raw_memory_candidates'][0]['passage'] == docs['b'].passage
    result = selector.run()
    assert result['20']['complete_required'] and result['20']['selected_doc_ids'] == ['a']
    assert selector.graph['nodes'] == before['nodes']
    assert engine.semantic_evidence['review_complete'] is not attempt_hidden_update
    if attempt_hidden_update:
        assert 'cannot_drop_invisible_alternative' in engine.semantic_evidence['review_error']['error']
    assert len(calls.requests) == 1


@pytest.mark.parametrize('responses,error_type,attempts', [
    ([{}, {}, {}], 'ProtocolError', 3),
    ([{'choices': [{'message': {'content': '{'}, 'finish_reason': 'length'}]}], 'OutputTruncated', 1),
])
def test_bounded_review_failure_preserves_a_complete_existing_closure(responses, error_type, attempts):
    selector, engine, calls, events = supported_selector(responses)
    before = deepcopy(selector.graph)
    result = selector.run()
    assert len(calls.requests) == attempts
    assert selector.graph == before
    for choice in result.values():
        assert choice['selected_doc_ids'] == ['a'] and choice['complete_required']
        assert choice['review_complete'] is False
    assert not engine.semantic_evidence['review_complete']
    assert engine.semantic_evidence['review_error']['error_type'] == error_type
    assert any(event['event'] == 'support_review_incomplete' for event in events)


def test_invalid_patch_never_publishes_a_new_span_or_changes_engine_nodes():
    def invalid(data):
        bad = raw_span(data, 'fabricated', 'b', 'answer')
        bad['quote'] = 'This text never appeared in the original record.'
        return review(new_spans=[bad])
    selector, engine, calls, _ = supported_selector([invalid], config={'max_repairs_per_request': 0})
    graph_before, spans_before = deepcopy(selector.graph), deepcopy(engine.spans)
    result = selector.run()
    assert selector.graph == graph_before and engine.spans == spans_before
    assert engine.nodes == graph_before['nodes']
    assert len(calls.requests) == 1 and not result['20']['review_complete']
    assert result['20']['selected_doc_ids'] == ['a'] and result['20']['complete_required']


def test_final_enumeration_overrides_too_small_exploratory_limit():
    selector, _, _, _ = supported_selector([lambda data: review()],
                                           config={'max_enumeration_states': 1})
    result = selector.run()
    for choice in result.values():
        assert choice['selected_doc_ids'] == ['a'] and choice['complete_required']
        assert choice['configured_exploratory_enumeration_limit'] == 1
        assert choice['final_enumeration_states'] == 2


def test_failed_review_with_no_proof_preserves_actual_raw_reader_context(setup):
    calls = FakeCalls([step('answer')], lambda data: unknown(), mapper=no_mapping,
                      selector=lambda data: {})
    engine, calls = actual_engine(setup, calls=calls, settings={'max_repairs_per_request': 1})
    result = engine.run()
    choice = result['budgets']['20']
    assert choice['selected_doc_ids'] == ['a', 'b']
    assert not choice['complete_required'] and not choice['review_complete']
    assert result['diagnostics']['spans'] == []
    assert result['diagnostics']['semantic_evidence']['evidence_state'] == 'raw_only'
    reader = next(row for row in calls.requests if row['operation'] == 'reader')
    passages = reader['payload']['messages'][1]['content'].split(
        'Context passages:\n', 1)[1].split('\n\nCommitted demand-state evidence:', 1)[0]
    for doc_id in ('a', 'b'):
        assert engine.docs[doc_id].passage in passages and 'source_doc_id=' + doc_id in passages
    assert calls.counts['select'] == 2 and calls.counts['reader'] == 1


def test_strict_review_failure_raises_but_valid_missing_support_does_not():
    selector, engine, calls, _ = supported_selector([{}],
        config={'allow_unassessed_coverage': False, 'max_repairs_per_request': 0})
    before = deepcopy(selector.graph)
    with pytest.raises(ProtocolError):
        selector.run()
    assert selector.graph == before and len(calls.requests) == 1
    assert engine.semantic_evidence['phase'] == 'selection_pending'

    selector, _, _, _ = support_selector([lambda data: review()],
        config={'allow_unassessed_coverage': False})
    choice = selector.run()['20']
    assert choice['review_complete'] and not choice['complete_required']


@pytest.mark.parametrize('failure', [ServiceError('offline service failure'), RefusalError()])
def test_service_and_refusal_are_fatal_instead_of_graph_fallback(failure):
    selector, engine, calls, _ = supported_selector([failure])
    before = deepcopy(selector.graph)
    with pytest.raises(type(failure)):
        selector.run()
    assert selector.graph == before and len(calls.requests) == 1
    assert engine.semantic_evidence['phase'] == 'selection_pending'


@pytest.mark.parametrize('kind,removed', [('parent', 'a'), ('guard', 'c')])
def test_reader_gate_rejects_parent_or_guard_removed_after_real_selection(setup, monkeypatch, kind, removed):
    engine, calls = actual_engine(setup, parents=kind == 'parent', guard=kind == 'guard')
    original_select = engine.select
    def corrupt_selected_context():
        result = original_select()
        assert result['20']['complete_required'] and removed in result['20']['selected_doc_ids']
        result['20']['selected_doc_ids'].remove(removed)
        return result
    monkeypatch.setattr(engine, 'select', corrupt_selected_context)
    with pytest.raises(SupportError):
        engine.run()
    assert calls.counts['reader'] == 0
    assert not any(event['event'] == 'reader_input' for event in engine.events)


def test_engine_recompile_preserves_invalidated_proof_history_and_blocks_later_revival(setup):
    previous = {}
    def revise(data):
        old = data['nodes'][0]['alternatives'][0]
        previous['sources'] = list(old['source_span_ids'])
        contrary = {**raw_span(data, 'disprove_a', 'x', 'answer'), 'stance': 'contradiction'}
        return review(new_spans=[contrary], invalidations=[{
            'alternative_ids': [old['id']], 'source_span_ids': ['disprove_a'],
            'reason': 'The former source refers to the wrong entity', 'disputed': False}],
            node_updates=[update('answer', 'fixture_answer', [proof([fact(data, 'c', 'answer')])])])

    result, engine, _, _ = run_engine(setup, [step('answer')],
        lambda data: answer(data, 'fixture_answer', [], alternatives=[(['a'], []), (['c'], [])]), revise)
    graph = result['diagnostics']['support_graph']
    assert result['budgets']['20']['selected_doc_ids'] == ['c']
    assert graph['proof_review']['invalidated_proof_signatures']
    assert graph['proof_review']['dropped_alternative_ids']
    assert graph['proof_review'] == result['diagnostics']['semantic_evidence']['proof_review']

    recompiled = engine.compile()
    assert recompiled['proof_review'] == graph['proof_review']
    assert recompiled['revision'] == graph['revision']
    assert [node['version'] for node in recompiled['nodes']] == [node['version'] for node in graph['nodes']]
    assert recompiled['supplemental_doc_ids'] == graph['supplemental_doc_ids']
    before = deepcopy(recompiled)
    with pytest.raises(SupportError, match='cannot_recreate_invalidated_proof'):
        apply_review(recompiled,
            review(node_updates=[update('answer', 'fixture_answer', [proof(previous['sources'])])]),
            engine.docs, visible_doc_ids=set(engine.docs),
            visible_alternative_ids={alt['id'] for node in recompiled['nodes'] for alt in node['alternatives']},
            visible_span_ids={span['id'] for span in recompiled['spans']})
    assert recompiled == before
