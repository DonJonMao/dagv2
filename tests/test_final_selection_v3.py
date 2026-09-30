"""V3 selection uses real protocol/closure code and explicitly scripted models."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from dagbt.budget import BudgetExceeded, Ledger
from dagbt.config import resolve
from dagbt.final_selection import FinalSelector, SELECT_SCHEMA
from dagbt.model_runtime import BTTokenAccounting
from dagbt.reasoning import InputOverflow, OutputTruncated, ProtocolError, Reasoner, RefusalError
from dagbt.support import compile_graph, invalidate_support, make_span, with_navigation_closure
from dagbt.transport import ServiceError, StubMeter
from dagbt import prompts
from test_engine import Document, Tokenizer, FakeCalls, step, unknown, setup


class SelectionCalls:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def get(self, stage, url, payload):
        data = json.loads(payload['messages'][1]['content'])
        self.requests.append({'stage': stage, 'data': deepcopy(data), 'payload': deepcopy(payload)})
        item = next(self.responses)
        if isinstance(item, BaseException):
            raise item
        item = item(data) if callable(item) else item
        if isinstance(item, str):
            body = {'choices': [{'message': {'content': item}, 'finish_reason': 'stop'}]}
        elif 'choices' in item:
            body = item
        else:
            body = {'choices': [{'message': {'content': json.dumps(item)}, 'finish_reason': 'stop'}]}
        return {'response_ref': f'fixture_{len(self.requests)}', 'response': body}


def missing(data, selected=None):
    fixed = data.get('repair_scope', {}).get('fixed_header')
    return {**(fixed or {'selected_doc_ids': list(selected if selected is not None else data['candidate_doc_ids']),
                         'reason': 'Fixture review, no semantic coverage claim', 'conflicts': []}),
            'coverage': [{'requirement_id': requirement['id'], 'status': 'missing', 'source_span_ids': [],
                          'kind': 'explicit', 'reason': 'Fixture has not established this need'}
                         for requirement in data['requirements']]}


def coverage(rid, refs, status='covered', kind='explicit'):
    return {'requirement_id': rid, 'source_span_ids': list(refs), 'status': status, 'kind': kind,
            'reason': 'Scripted source relationship'}


def make_selector(responses=(), *, docs=None, facts=(), node_ids=('answer',), supported=False,
                  baseline=None, proposal=(), config=None):
    docs = docs or {'a': Document('a', 'First genuine historical evidence.'),
                    'b': Document('b', 'Different genuine historical evidence.')}
    s = resolve({'fusion': {'selection_review': True, 'context_tokens': 8192,
                            'reasoning_output_tokens': 256, 'reader_output_tokens': 128,
                            **(config or {})}})
    spans = {}
    for index, fact in enumerate(facts):
        fid, doc, node, stance, kind = fact[:5]
        quote = docs[doc].passage
        spans[fid] = make_span(fid, doc, quote, docs, start=0, node_ids=[node], node_id=node,
                              stance=stance, kind=kind, source_role='document', claim='Fixture claim',
                              assessment_id=fid, premise_group_ids=[fid], **(fact[5] if len(fact) > 5 else {}))
    nodes = []
    for nid in node_ids:
        sources = [sid for sid, fact in spans.items() if nid in fact['node_ids']]
        alternatives = [{'id': f'{nid}_alt', 'source_span_ids': sources, 'guard_span_ids': [],
                         'used_parent_ids': [], 'used_parent_versions': {}, 'applicable_scope': 'fixture',
                         'semantic_status': 'supported'}] if supported and sources else []
        nodes.append({'id': nid, 'answer': 'Fixture answer' if alternatives else None,
                      'status': 'supported' if alternatives else 'unknown', 'version': 0, 'alternatives': alternatives})
    requirements = [{'id': 'final', 'necessary': True, 'terminal_node_ids': [node_ids[-1]], 'terminal_mode': 'all'}]
    graph = compile_graph(nodes, list(spans.values()), requirements, docs)
    events = []
    ledger = Ledger({'llm': 24, 'json_repairs': 6, 'reader': 1}, events.append)
    model_config = {'model_profile': 'bridgetree', 'llm_model': 'fixture',
                    'llm_base_url': 'http://invalid.fixture/v1', 'fusion': s}
    calls = SelectionCalls(responses)
    r = Reasoner(StubMeter(calls, ledger, model_config), BTTokenAccounting(), model_config, s, ledger, events.append)
    tokenizer = Tokenizer()
    def feasible(ids, max_docs=20):
        count = 20 + sum(len(tokenizer.encode(docs[d].passage)) for d in ids)
        return {'feasible': len(ids) <= max_docs and count <= s['context_tokens'], 'token_count': count,
                'budget': s['context_tokens'], 'context_hash': 'fixture'}
    e = SimpleNamespace(q={'id': 'q', 'question': 'Which history is relevant?'}, docs=docs,
                        spans=spans, s=s, tokenizer=tokenizer, candidates=list(docs),
                        baseline_ids=list(docs) if baseline is None else list(baseline),
                        steps=[step(nid) for nid in node_ids], reasoner=r, ledger=ledger,
                        mapper=SimpleNamespace(diagnostics=lambda: {'mapping_incomplete': not bool(spans)}),
                        chunks=[{'doc_id': d} for d in docs], mapped_chunks=set(), event=events.append,
                        input_views=[], feasible=feasible)
    selector = FinalSelector(e, graph, {'selected_doc_ids': list(proposal)})
    return selector, e, calls, events


def test_unmapped_original_documents_can_be_selected_without_fabricating_closure():
    selector, e, calls, _ = make_selector([lambda data: missing(data, ['b'])])
    result = selector.run()
    assert result['20']['selected_doc_ids'] == ['b']
    assert not result['20']['complete_required'] and result['20']['verified_support_doc_ids'] == []
    assert result['20']['coverage'][0]['status'] == 'missing'
    assert e.semantic_evidence['evidence_state'] == 'raw_only'
    assert calls.requests[0]['data']['evidence'] == []
    assert calls.requests[0]['data']['raw_memory_candidates'][1]['passage'] == e.docs['b'].passage


def test_raw_view_preserves_full_text_role_offsets_and_order_metadata():
    doc = Document('a', 'User experience, not an assistant suggestion.')
    doc.metadata = {'source_segments': [{'start': 0, 'end': len(doc.passage), 'role': 'user',
                                        'source_message_indices': [2]}], 'observation_order': 3}
    selector, e, _, _ = make_selector(docs={'a': doc})
    view = selector.prepare_view()
    raw = view.data['raw_memory_candidates'][0]
    assert raw['passage'] == doc.passage
    assert raw['metadata']['source_segments'][0]['role'] == 'user'
    assert raw['metadata']['source_segments'][0]['source_message_indices'] == [2]
    assert raw['metadata']['observation_order_is_event_time'] is False
    assert e.reasoner.estimate('select', prompts.SELECT_V3, view.data, SELECT_SCHEMA) <= view.audit['input_token_limit']


def test_raw_overflow_omits_whole_record_and_preserves_exact_later_small_record():
    docs = {'huge': Document('huge', '中' * 20000), 'small': Document('small', 'Short history.')}
    selector, e, _, _ = make_selector(docs=docs)
    view = selector.prepare_view()
    assert view.audit['omitted_raw_review_doc_ids'] == ['huge']
    assert view.audit['raw_review_visible_doc_ids'] == ['small']
    assert view.data['raw_memory_candidates'][0]['passage'] == docs['small'].passage
    assert 'huge' not in view.visible_doc_ids
    assert len(e.input_views) == 1


def test_raw_review_ablation_does_not_expose_unmapped_candidate_ids():
    selector, _, _, _ = make_selector(config={'raw_memory_review': False})
    view = selector.prepare_view()
    assert not view.visible_doc_ids and view.data['raw_memory_candidates'] == []
    with pytest.raises(ProtocolError, match='undisplayed'):
        selector._header({'selected_doc_ids': ['a'], 'reason': 'Claim', 'conflicts': [], 'coverage': []})


@pytest.mark.parametrize('bad_refs,selected,rid', [(['invented'], ['a'], 'n1'),
    (['s1'], ['a'], 'n1'), (['e1'], ['b'], 'n1'), (['e1'], ['a'], 'n2')])
def test_coverage_requires_current_alias_same_node_and_selected_document(bad_refs, selected, rid):
    selector, _, _, _ = make_selector(facts=[('s1', 'a', 'n1', 'support', 'explicit')], node_ids=('n1', 'n2'))
    selector.prepare_view()
    selector.header = {'selected_doc_ids': selected, 'reason': 'Fixture', 'conflicts': []}
    with pytest.raises(ProtocolError):
        selector._coverage(coverage(rid, bad_refs, status='partial'))


def test_implicit_support_is_conservatively_normalized_after_reference_validation():
    selector, _, _, _ = make_selector(facts=[('s1', 'a', 'answer', 'support', 'implicit')], supported=True)
    selector.prepare_view()
    selector.header = {'selected_doc_ids': ['a'], 'reason': 'Fixture', 'conflicts': []}
    row = selector._coverage(coverage('answer', ['e1']))
    assert row['declared_kind'] == 'explicit' and row['kind'] == 'inference'
    assert row['normalization_reason'] == 'cited_implicit_assessment'
    assert row['validation_complete']


def test_joint_independent_partial_is_inference_but_single_or_repeated_source_cannot_cover():
    selector, _, _, _ = make_selector(facts=[('s1', 'a', 'answer', 'partial', 'explicit'),
                                            ('s2', 'b', 'answer', 'partial', 'explicit')], supported=True)
    selector.prepare_view(); selector.header = {'selected_doc_ids': ['a', 'b'], 'reason': 'Fixture', 'conflicts': []}
    row = selector._coverage(coverage('answer', ['e1', 'e2']))
    assert row['kind'] == 'inference' and row['normalization_reason'] == 'joint_partial_inference'
    with pytest.raises(ProtocolError, match='independent'):
        selector._coverage(coverage('answer', ['e1']))
    with pytest.raises(ProtocolError):
        selector._coverage(coverage('answer', ['e1', 'e1']))


def test_overlapping_partial_judgments_and_contradiction_do_not_create_covered():
    for stance in ('partial', 'contradiction'):
        selector, _, _, _ = make_selector(facts=[('s1', 'a', 'answer', stance, 'explicit'),
                                                ('s2', 'a', 'answer', stance, 'explicit')], supported=True)
        selector.prepare_view(); selector.header = {'selected_doc_ids': ['a'], 'reason': 'Fixture', 'conflicts': []}
        with pytest.raises(ProtocolError, match='independent'):
            selector._coverage(coverage('answer', ['e1', 'e2']))


def test_covered_requires_actual_selected_dag_route_and_fresh_certificate():
    selector, _, _, _ = make_selector(facts=[('s1', 'a', 'answer', 'support', 'explicit')], supported=False)
    selector.prepare_view(); selector.header = {'selected_doc_ids': ['a'], 'reason': 'Fixture', 'conflicts': []}
    with pytest.raises(ProtocolError, match='DAG support closure'):
        selector._coverage(coverage('answer', ['e1']))


def test_coverage_repair_can_finish_unassessed_without_changing_legal_selection():
    bad = lambda data: {**missing(data, ['a']), 'coverage': [coverage('answer', ['invented'])]}
    selector, e, calls, _ = make_selector([bad, bad, bad])
    result = selector.run()
    assert result['20']['selected_doc_ids'] == ['a']
    assert not result['20']['coverage_validation_complete']
    row = result['20']['coverage'][0]
    assert row['status'] == 'unassessed' and row['source_span_ids'] == [] and row['kind'] is None
    assert e.ledger.used['json_repairs'] == 2
    assert all('raw_memory_candidates' not in request['data'] for request in calls.requests[1:])


@pytest.mark.parametrize('failure,expected', [
    ('not JSON', ProtocolError),
    ({'choices': [{'message': {'content': '{}', 'refusal': 'refused'}, 'finish_reason': 'stop'}]}, RefusalError),
    ({'choices': [{'message': {'content': '{}'}, 'finish_reason': 'length'}]}, OutputTruncated),
    (ServiceError('service unavailable'), ServiceError),
    ({'selected_doc_ids': ['future'], 'reason': 'changed', 'conflicts': [], 'coverage': []}, ProtocolError),
])
def test_global_fault_after_legal_header_never_becomes_unassessed(failure, expected):
    bad = lambda data: {**missing(data, ['a']), 'coverage': [coverage('answer', ['invented'])]}
    selector, _, _, _ = make_selector([bad, failure])
    with pytest.raises(expected):
        selector.run()


def test_invalid_initial_selection_ids_cannot_degrade_to_unassessed():
    bad = {'selected_doc_ids': ['future'], 'reason': 'Unseen ID', 'conflicts': [], 'coverage': []}
    selector, _, calls, _ = make_selector([bad, bad, bad])
    with pytest.raises(ProtocolError, match='undisplayed'):
        selector.run()
    assert len(calls.requests) == 3


def test_unassessed_disabled_propagates_coverage_error():
    bad = lambda data: {**missing(data, ['a']), 'coverage': [coverage('answer', ['invented'])]}
    selector, _, _, _ = make_selector([bad, bad, bad], config={'allow_unassessed_coverage': False})
    with pytest.raises(ProtocolError, match='unverified'):
        selector.run()


def test_compact_repair_keeps_valid_coverage_and_only_pending_selected_ledger():
    def first(data):
        value = missing(data, ['a'])
        value['coverage'][1] = coverage('n2', ['invented'])
        return value
    selector, e, calls, _ = make_selector([first, missing],
        facts=[('s1', 'a', 'n1', 'support', 'explicit'), ('s2', 'b', 'n2', 'support', 'explicit')],
        node_ids=('n1', 'n2'), supported=True)
    result = selector.run()
    repair = calls.requests[1]['data']
    assert [r['id'] for r in repair['requirements']] == ['n2']
    assert repair['candidate_doc_ids'] == ['a']
    assert repair['evidence'] == []  # n2's b-source is outside the fixed selected set.
    assert 'raw_memory_candidates' not in repair
    assert result['20']['coverage_validation_complete']
    assert [r['status'] for r in result['20']['coverage']] == ['missing', 'missing']
    assert e.ledger.used['json_repairs'] == 1


def test_coverage_repair_budget_failure_retains_real_cause_in_unassessed_rows():
    bad = lambda data: {**missing(data, ['a']), 'coverage': [coverage('answer', ['invented'])]}
    selector, e, _, _ = make_selector([bad])
    preflight = e.reasoner._preflight
    def bounded(operation, *args, **kwargs):
        if operation == 'select_repair':
            raise InputOverflow('actual compact repair exceeds context')
        return preflight(operation, *args, **kwargs)
    e.reasoner._preflight = bounded
    result = selector.run()
    row = result['20']['coverage'][0]
    assert row['status'] == 'unassessed' and row['failure_type'] == 'InputOverflow'
    assert row['failure_detail'] == 'actual compact repair exceeds context'
    assert e.semantic_evidence['recovery_actions'][-1]['status'] == 'not_sent'
    assert not e.ledger.used['json_repairs']


@pytest.mark.parametrize('extra', [
    coverage('unknown_requirement', [], status='missing'),
    None, 'not an object', ['not', 'an', 'object'],
    {'requirement_id': ['answer']}, {'requirement_id': {'id': 'answer'}},
])
def test_unknown_nonobject_and_unhashable_coverage_rows_use_clean_envelope_repair(extra):
    def first(data):
        value = missing(data, ['a'])
        value['coverage'].append(deepcopy(extra))
        return value
    selector, e, calls, _ = make_selector([first, missing])
    result = selector.run()
    repair = calls.requests[1]['data']
    assert repair['requirements'] == [] and repair['evidence'] == []
    assert repair['repair_scope']['pending_requirement_ids'] == []
    assert repair['repair_scope']['coverage_envelope_errors']
    assert 'coverage=[]' in repair['repair_scope']['instruction']
    assert result['20']['selected_doc_ids'] == ['a']
    assert [r['requirement_id'] for r in result['20']['coverage']] == ['answer']
    assert result['20']['coverage'][0]['status'] == 'missing'
    assert result['20']['coverage_validation_complete']
    assert e.semantic_evidence['unassessed_requirement_ids'] == []
    assert not e.semantic_evidence['coverage_envelope_unassessed']
    assert e.semantic_evidence['coverage_failures'][0]['coverage_envelope_errors']
    assert e.ledger.used['json_repairs'] == 1


def test_unchanged_valid_repeat_does_not_pay_another_requirements_missing_row():
    saved = {}
    def first(data):
        value = missing(data, ['a'])
        saved['valid'] = deepcopy(value['coverage'][0])
        value['coverage'] = value['coverage'][:1]
        return value
    def second(data):
        return {**missing(data), 'coverage': [deepcopy(saved['valid'])]}
    selector, e, calls, _ = make_selector([first, second, missing], node_ids=('n1', 'n2'))
    result = selector.run()
    assert len(calls.requests) == 3
    assert [r['id'] for r in calls.requests[1]['data']['requirements']] == ['n2']
    assert [r['id'] for r in calls.requests[2]['data']['requirements']] == ['n2']
    assert calls.requests[2]['data']['repair_scope']['coverage_envelope_errors'] == []
    assert [r['requirement_id'] for r in result['20']['coverage']] == ['n1', 'n2']
    assert result['20']['coverage_validation_complete']
    assert e.ledger.used['json_repairs'] == 2


def test_unchanged_repeat_compares_raw_row_before_alias_and_kind_normalization():
    saved = {}
    def first(data):
        value = missing(data, ['a'])
        saved['row'] = coverage('n1', [data['evidence'][0]['id']], kind='explicit')
        value['coverage'] = [deepcopy(saved['row'])]
        return value
    def second(data):
        value = missing(data)
        value['coverage'].insert(0, deepcopy(saved['row']))
        return value
    selector, e, _, _ = make_selector([first, second], node_ids=('n1', 'n2'), supported=True,
        facts=[('s1', 'a', 'n1', 'support', 'implicit')])
    result = selector.run()
    accepted = result['20']['coverage'][0]
    assert accepted['source_span_ids'] == ['s1'] and accepted['kind'] == 'inference'
    assert accepted['declared_kind'] == 'explicit' and saved['row']['source_span_ids'] == ['e1']
    assert result['20']['coverage_validation_complete']
    assert not e.semantic_evidence['coverage_envelope_unassessed']


def test_changed_valid_repeat_cannot_replace_frozen_coverage_or_become_normal():
    saved = {}
    def first(data):
        value = missing(data, ['a'])
        saved['valid'] = deepcopy(value['coverage'][0])
        value['coverage'] = value['coverage'][:1]
        return value
    def second(data):
        value = missing(data)
        changed = {**saved['valid'], 'reason': 'Changed already accepted explanation'}
        value['coverage'].append(changed)
        return value
    selector, e, calls, _ = make_selector([first, second], node_ids=('n1', 'n2'),
                                        config={'max_repairs_per_request': 1})
    result = selector.run()
    assert len(calls.requests) == 2
    assert result['20']['coverage'][0]['reason'] == saved['valid']['reason']
    assert all(row['validation_complete'] for row in result['20']['coverage'])
    assert not result['20']['coverage_validation_complete']
    assert e.semantic_evidence['coverage_envelope_unassessed']
    assert e.semantic_evidence['unassessed_requirement_ids'] == []
    assert 'Previously validated coverage was changed' in e.semantic_evidence['coverage_envelope_errors'][0]


def test_envelope_exhaustion_keeps_real_requirements_and_records_incomplete_container():
    def noisy(data):
        value = missing(data, ['a'])
        value['coverage'].append({'requirement_id': 'not_a_real_need'})
        return value
    selector, e, calls, _ = make_selector([noisy, noisy, noisy])
    result = selector.run()
    assert len(calls.requests) == 3
    assert result['20']['status'] == 'coverage_unassessed'
    assert not result['20']['coverage_validation_complete']
    assert result['20']['coverage'][0]['validation_complete']
    assert [r['requirement_id'] for r in result['20']['coverage']] == ['answer']
    assert e.semantic_evidence['coverage_envelope_unassessed']
    assert e.semantic_evidence['unassessed_requirement_ids'] == []
    assert len(e.semantic_evidence['coverage_failures']) == 3


def test_strict_mode_rejects_pure_envelope_exhaustion_despite_valid_coverage():
    def noisy(data):
        value = missing(data, ['a'])
        value['coverage'].append(None)
        return value
    selector, e, calls, _ = make_selector([noisy, noisy],
        config={'allow_unassessed_coverage': False, 'max_repairs_per_request': 1})
    with pytest.raises(ProtocolError, match='unverified'):
        selector.run()
    assert len(calls.requests) == 2
    assert selector.valid['answer']['validation_complete']
    assert selector.envelope_errors
    assert e.semantic_evidence['coverage_validation_complete'] is False


@pytest.mark.parametrize('failure', [BudgetExceeded('llm', 'physical retry', 1, 0),
                                    InputOverflow('physical compact envelope context')])
def test_pure_envelope_repair_budget_failure_keeps_explicit_incomplete_state(failure):
    def noisy(data):
        value = missing(data, ['a'])
        value['coverage'].append(None)
        return value
    selector, e, _, _ = make_selector([noisy, failure])
    result = selector.run()
    assert result['20']['coverage'][0]['validation_complete']
    assert not result['20']['coverage_validation_complete']
    assert e.semantic_evidence['coverage_envelope_unassessed']
    assert e.semantic_evidence['unassessed_requirement_ids'] == []
    assert e.semantic_evidence['recovery_actions'][-1]['error_type'] == type(failure).__name__


def test_disputed_and_navigation_groups_remain_hard_final_constraints():
    selector, _, _, _ = make_selector(facts=[('s1', 'a', 'answer', 'support', 'explicit'),
                                            ('s2', 'b', 'answer', 'contradiction', 'explicit')], supported=True)
    selector.graph = invalidate_support(selector.graph, ['answer_alt'], conflict_span_ids=['s2'], reason='Conflict', disputed=True)
    selector.prepare_view()
    with pytest.raises(ProtocolError, match='disputed source group'):
        selector._header({'selected_doc_ids': ['a'], 'reason': 'Drop counterevidence', 'conflicts': [], 'coverage': []})
    selector.graph['conflicts'] = []
    selector.graph = with_navigation_closure(selector.graph, {'a': [], 'b': ['a']}, selector.e.docs)
    with pytest.raises(ProtocolError, match='source closure'):
        selector._header({'selected_doc_ids': ['b'], 'reason': 'Drop navigation', 'conflicts': [], 'coverage': []})


def test_subbudget_report_must_not_split_navigation_closure():
    docs = {f'd{i}': Document(f'd{i}', 'Raw history.') for i in range(6)}
    selector, e, _, _ = make_selector([lambda data: missing(data)], docs=docs)
    selector.graph = with_navigation_closure(selector.graph,
        {doc: ['d5'] if doc == 'd0' else [] for doc in docs}, docs)
    result = selector.run()
    for limit in ('5', '10', '20'):
        chosen = set(result[limit]['selected_doc_ids'])
        assert 'd0' not in chosen or 'd5' in chosen, 'Reported subbudget set must retain required navigation ancestors'


def test_subbudget_report_must_not_split_disputed_source_group():
    docs = {f'd{i}': Document(f'd{i}', 'Raw history.') for i in range(6)}
    selector, _, _, _ = make_selector([lambda data: missing(data)], docs=docs,
        facts=[('support', 'd0', 'answer', 'support', 'explicit'),
               ('counter', 'd5', 'answer', 'contradiction', 'explicit')], supported=True)
    selector.graph = invalidate_support(selector.graph, ['answer_alt'], conflict_span_ids=['counter'],
                                        reason='Opposing evidence', disputed=True)
    result = selector.run()
    for limit in ('5', '10', '20'):
        chosen = set(result[limit]['selected_doc_ids'])
        assert not (chosen & {'d0', 'd5'}) or {'d0', 'd5'} <= chosen


class EngineV3Calls(FakeCalls):
    def get(self, stage, url, payload):
        if stage[0].startswith('select') and payload['messages'][0]['content'].startswith('Review source documents'):
            self.counts[stage[0]] += 1
            self.requests.append({'operation': stage[0], 'payload': deepcopy(payload)})
            data = json.loads(payload['messages'][1]['content'])
            value = missing(data, [r['doc_id'] for r in data.get('raw_memory_candidates', [])])
            return {'response_ref': 'final_fixture', 'response': {'choices': [
                {'message': {'content': json.dumps(value)}, 'finish_reason': 'stop'}]}}
        return super().get(stage, url, payload)


def test_formal_engine_unmapped_baseline_history_reaches_final_raw_reader(setup):
    from dagbt.engine import run_question
    bridge, resources, config = setup
    config['fusion']['selection_review'] = True
    config['fixed_candidate_pools'] = {'q': ['a', 'b']}
    calls = EngineV3Calls([step('answer')], lambda data: unknown(),
                          mapper=lambda data: {'units': [{'unit_id': unit['unit_id'], 'assessments': [],
                                                         'irrelevance_reason': 'Scripted mapper missed this history'}
                                                        for unit in data['units']]})
    result = run_question({'id': 'q', 'question': 'Which history is relevant?'}, resources, calls, config)
    assert result['budgets']['20']['selected_doc_ids'] == ['a', 'b']
    assert not result['budgets']['20']['complete_required'] and result['diagnostics']['spans'] == []
    reader_request = next(record for record in calls.requests if record['operation'] == 'reader')
    reader = json.dumps(reader_request['payload'], ensure_ascii=False)
    assert resources[0]['a'].text in reader and resources[0]['b'].text in reader


@pytest.mark.parametrize('failed_stage,empty_selection,expected_state', [
    ('select', False, 'unknown'),
    ('reader', False, 'raw_only'),
    ('reader', True, 'empty_context'),
])
def test_formal_failure_artifact_refreshes_attempt_counts_without_erasing_semantic_phase(
        setup, tmp_path, failed_stage, empty_selection, expected_state):
    """Actual Engine failure path persists the last failed attempt as a cost."""
    from dagbt.engine import run_question
    _, resources, config = setup
    config['fusion']['selection_review'] = True
    config['fixed_candidate_pools'] = {'q': ['a', 'b']}

    class FailureCalls(EngineV3Calls):
        def get(self, stage, url, payload):
            operation = stage[0]
            if operation == failed_stage:
                self.counts[operation] += 1
                self.requests.append({'operation': operation, 'payload': deepcopy(payload), 'failed': True})
                raise ServiceError('Scripted ' + operation + ' service failure')
            if operation == 'select' and empty_selection:
                self.counts[operation] += 1
                self.requests.append({'operation': operation, 'payload': deepcopy(payload)})
                data = json.loads(payload['messages'][1]['content'])
                return {'response_ref': 'empty_selection_fixture', 'response': {'choices': [
                    {'message': {'content': json.dumps(missing(data, []))}, 'finish_reason': 'stop'}]}}
            return super().get(stage, url, payload)

    calls = FailureCalls([step('answer')], lambda data: unknown(),
        mapper=lambda data: {'units': [{'unit_id': unit['unit_id'], 'assessments': [],
                                       'irrelevance_reason': 'Scripted mapper leaves history unmapped'}
                                      for unit in data['units']]})
    calls.output = tmp_path / 'actual_engine_failure'
    with pytest.raises(ServiceError, match=failed_stage + ' service failure'):
        run_question({'id': 'q', 'question': 'Which history is relevant?'}, resources, calls, config)

    partial = json.loads((calls.output / 'fusion_partial.json').read_text())
    diagnostics = partial['diagnostics']
    semantic = diagnostics['semantic_evidence']
    llm_attempts = sum(request['operation'] != 'reader' for request in calls.requests)
    reader_attempts = sum(request['operation'] == 'reader' for request in calls.requests)
    assert partial['error_type'] == 'ServiceError'
    assert calls.requests[-1]['operation'] == failed_stage and calls.requests[-1]['failed']
    assert semantic['evidence_logical_calls'] == diagnostics['reasoning_call_count'] == llm_attempts
    assert semantic['budgeted_llm_attempts'] == partial['ledger']['used']['llm'] == llm_attempts
    assert semantic['budgeted_reader_attempts'] == partial['ledger']['used'].get('reader', 0) == reader_attempts
    assert semantic['evidence_state'] == expected_state
    assert partial['events'][-1]['event'] == 'task_failed'
    assert not (calls.output / 'fusion_snapshot.json').exists()

    if failed_stage == 'select':
        assert semantic['phase'] == 'selection_pending'
        assert semantic['selected_doc_ids'] == [] and semantic['coverage_validation_complete'] is None
        pending = next(event for event in partial['events'] if event['event'] == 'selection_progress'
                       and event['phase'] == 'selection_pending')
        assert pending['evidence_logical_calls'] + 1 == semantic['evidence_logical_calls']
        assert reader_attempts == 0
    else:
        assert semantic['selected_doc_ids'] == ([] if empty_selection else ['a', 'b'])
        assert semantic['coverage_validation_complete'] is True
        accepted = next(event for event in partial['events'] if event['event'] == 'semantic_selection_complete')
        assert accepted['budgeted_reader_attempts'] == 0 and semantic['budgeted_reader_attempts'] == 1
