"""Exercise the actual DAG-Resolve flow; model semantics are scripted fixtures.

These tests establish execution, grounding, provenance and resource contracts,
not NLP accuracy. The bundled tokenizer and frozen Reader are used unchanged.
"""
from copy import deepcopy
import json
import re
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from dagresolve import flow
from dagresolve.runtime import import_originals


QUESTION = 'In which city was the director of the 1998 film Homeward born?'
DIRECTOR = 'Who was the director of the 1998 film Homeward?'
CHILD = 'In which city was {director} born?'
BINDING = 'The 1998 film Homeward was directed by Lin Zhou alone, not Chen Hai.'
PRIMARY_SUPPORT = 'Chen Hai directed the 1998 film Homeward.'
CO_SUPPORT = 'Chen Hai and Lin Zhou jointly directed the 1998 film Homeward.'
TEXTS = {
    'chen': 'Chen Hai directed Homeward.',
    'lin': 'Lin Zhou directed Homeward.',
    'binding': BINDING,
    'lin_birth': 'Lin Zhou was born in Suzhou.',
    'chen_birth': 'Chen Hai was born in Wuhan.',
}
USAGE = {'prompt_tokens': 11, 'completion_tokens': 7, 'total_tokens': 18}


@pytest.fixture(scope='module')
def frozen_runtime():
    runtime = import_originals('hotpotqa')
    sentinel = object()
    previous = {name: getattr(runtime.flow, name, sentinel)
                for name in ('CHAT_URL', 'COMPLETION_URL', 'cross', 'archive', 'dense')}
    runtime.e.native.configure(runtime.e.CONFIG)
    try:
        yield runtime
    finally:
        for name, value in previous.items():
            if value is sentinel:
                delattr(runtime.flow, name)
            else:
                setattr(runtime.flow, name, value)


@pytest.fixture(scope='module')
def real_tokenizer(frozen_runtime):
    return frozen_runtime.e.AutoTokenizer.from_pretrained(
        str(frozen_runtime.e.TOKENIZER), local_files_only=True)


def make_corpus(runtime, mode='switch'):
    # Twenty ordinary hits outrank all three unseen evidence passages initially.
    ids = ['chen', 'lin'] + [f'filler_{i}' for i in range(18)]
    ids += ['binding', 'lin_birth', 'chen_birth']
    documents = {doc_id: runtime.e.Document(doc_id, '', TEXTS.get(
        doc_id, f'Unrelated catalogue entry {doc_id}.')) for doc_id in ids}
    if mode == 'primary_supported':
        documents['chen'] = runtime.e.Document('chen', '', TEXTS['chen'] + '\n' + PRIMARY_SUPPORT)
    elif mode in ('joint_judge', 'invalid_primary_quote'):
        documents['binding'] = runtime.e.Document('binding', '', CO_SUPPORT)
    elif mode == 'legal_multivalue':
        for doc_id in ('chen', 'lin'):
            documents[doc_id] = runtime.e.Document(doc_id, '', TEXTS[doc_id] + '\n' + CO_SUPPORT)
    vectors = np.zeros((len(ids), 4096), dtype=np.float32)
    for i in range(20):
        vectors[i, 0] = 1 - i * .01
        vectors[i, 4 + i] = np.sqrt(1 - vectors[i, 0] ** 2)
    vectors[20, 1] = vectors[21, 2] = vectors[22, 3] = 1
    index = SimpleNamespace(vectors=dict(zip(ids, vectors)), lock=threading.Lock())
    return documents, ids, vectors, index


def plan(node_count=2):
    steps = [{'question': DIRECTOR, 'output_slot': 'director',
              'answer_type': 'person', 'inputs': []}]
    for i in range(1, node_count):
        previous = steps[-1]['output_slot']
        slot = 'birthplace' if i == 1 else 'detail_' + str(i)
        question = CHILD if i == 1 else 'What is the recorded location of {' + previous + '}?'
        steps.append({'question': question, 'output_slot': slot,
                      'answer_type': 'city', 'inputs': [previous]})
    return {'steps': steps}


def make_row(ids, node_count=2):
    return {'unit_id': 'scripted-binding-case', 'question': QUESTION, 'plan': plan(node_count),
            'candidate_doc_ids': list(ids[:20]), 'archive_trace': []}


def query_vector(query):
    vector = np.zeros(4096, dtype=np.float32)
    if (query.startswith(('Chen Hai ', 'Lin Zhou '))
            and all(term in query for term in ('director', 'Homeward', '1998'))):
        vector[1] = 1
    elif 'In which city was Lin Zhou born?' in query:
        vector[2] = 1
    elif 'In which city was Chen Hai born?' in query:
        vector[3] = 1
    else:
        vector[0] = 1
    return vector.tolist()


def displayed_index(prompt, quote):
    matches = list(re.finditer(r'Passage \[(\d+)\]\n', prompt))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(prompt)
        if quote in prompt[match.end():end]:
            return int(match.group(1))
    raise AssertionError('Fixture evidence was not displayed: ' + quote)


def reference(prompt, quote):
    return {'panel_index': displayed_index(prompt, quote), 'quote': quote}


def assessment(status='unknown', evidence=()):
    return {'requirement_id': 'r1', 'status': status, 'evidence': list(evidence)}


def completion_value(payload, mode='switch'):
    """Use visible passages/indices, never an invented source or new candidate."""
    prompt = payload['prompt']
    schema = payload['structured_outputs']['json']['properties']
    if 'candidates' in schema and 'answer' not in schema:
        if mode in ('joint_judge', 'invalid_primary_quote'):
            quote = reference(prompt, CO_SUPPORT)
            primary_quote = {**quote, 'quote': 'Invalid primary citation.'} if mode == 'invalid_primary_quote' else quote
            return {'legal_multivalue': True, 'candidates': [
                {'id': 'h1', 'assessments': [assessment('supported', [primary_quote])]},
                {'id': 'h2', 'assessments': [assessment('supported', [quote])]}]}
        quote = reference(prompt, BINDING)
        primary = 'contradicted' if mode in ('switch', 'contra_without_alt') else 'unknown'
        alternative = 'supported' if mode in ('switch', 'quote_invalid') else 'unknown'
        if mode == 'quote_invalid':
            quote = {**quote, 'quote': 'Invented binding proof that is not in this passage.'}
        return {'legal_multivalue': False, 'candidates': [
            {'id': 'h1', 'assessments': [assessment(primary, [quote] if primary != 'unknown' else [])]},
            {'id': 'h2', 'assessments': [assessment(alternative, [quote] if alternative != 'unknown' else [])]}]}
    count = schema['sources']['minItems']
    if 'candidates' in schema:
        result = {'answer': 'Chen Hai', 'sources': [False] * count,
                  'legal_multivalue': False,
                  'requirements': [{'id': 'r1', 'kind': 'year_version',
                      'question_quote': 'the director of the 1998 film Homeward',
                      'relation_quote': 'director', 'subject_quote': 'Homeward', 'scope_quotes': ['1998']}],
                  'candidates': []}
        result['sources'][displayed_index(prompt, TEXTS['chen'])] = True
        for i, name in enumerate(('Chen Hai', 'Lin Zhou')):
            quote = reference(prompt, TEXTS['chen' if i == 0 else 'lin'])
            result['candidates'].append({'id': 'h' + str(i + 1), 'answer': name, 'name_quote': name,
                'evidence': [quote], 'assessments': [assessment()]})
        if mode == 'single':
            result['candidates'] = result['candidates'][:1]
        elif mode == 'invalid_anchor':
            result['candidates'][1]['evidence'][0]['quote'] = 'No passage says this.'
        elif mode == 'invalid_proposal_assessment':
            result['candidates'][0]['assessments'] = [assessment('supported',
                [{'panel_index': 0, 'quote': 'No passage says this.'}])]
        elif mode == 'primary_supported':
            result['candidates'][0]['assessments'] = [assessment('supported',
                [reference(prompt, PRIMARY_SUPPORT)])]
        elif mode == 'legal_multivalue':
            for candidate in result['candidates']:
                candidate['assessments'] = [assessment('supported', [reference(prompt, CO_SUPPORT)])]
            result['legal_multivalue'] = True
        return result
    grounded = prompt.split('Task with the available parent assignments:\n', 1)[1].split(
        '\n\nParent results', 1)[0]
    chosen = 'lin_birth' if 'Lin Zhou' in grounded else 'chen_birth'
    sources = [False] * count
    try:
        sources[displayed_index(prompt, TEXTS[chosen])] = True
    except AssertionError:
        # The unresolved-parent branch follows the original parent-question
        # grounding. It is deliberately left unresolved in this fixture.
        return {'answer': '', 'sources': sources}
    return {'answer': 'Suzhou' if chosen == 'lin_birth' else 'Wuhan', 'sources': sources}


class ScriptedCalls:
    def __init__(self, mode='switch'):
        self.mode, self.requests = mode, []

    def get(self, stage, url, payload):
        self.requests.append({'stage': stage, 'url': url, 'payload': deepcopy(payload)})
        if url.endswith('/embeddings'):
            if self.mode == 'embedding_failure' and tuple(stage)[:2] == ('resolve', 'clarify'):
                raise ConnectionError('Optional embedding fixture failure')
            query = payload['input'][0].split('\nQuery: ', 1)[1]
            response = {'data': [{'index': 0, 'embedding': query_vector(query)}], 'usage': USAGE}
        elif 'prompt' in payload:
            value = completion_value(payload, self.mode)
            finish = 'length' if self.mode == 'judge_length' and tuple(stage) == ('resolve', 'clarify', 'judge') else 'stop'
            response = {'choices': [{'text': json.dumps(value), 'finish_reason': finish}], 'usage': USAGE}
        else:
            response = {'choices': [{'message': {'content': 'Answer: Suzhou'}, 'finish_reason': 'stop'}], 'usage': USAGE}
        return {'response': response, 'response_ref': 'fixture-' + str(len(self.requests))}


def execute(runtime, tokenizer, mode='switch', node_count=2):
    documents, ids, _, index = make_corpus(runtime, mode)
    row, calls = make_row(ids, node_count), ScriptedCalls(mode)
    before = deepcopy(row)
    result = flow.solve(row, documents, tokenizer, index, calls, runtime=runtime)
    assert row == before  # Both the initial archive and caller's plan are immutable.
    return result, calls, row, documents


@pytest.mark.parametrize('mode,reason', [
    ('single', 'no_competing_candidate'), ('invalid_anchor', 'proposal_protocol_invalid'),
    ('invalid_proposal_assessment', 'proposal_protocol_invalid'),
    ('primary_supported', 'primary_supported'), ('legal_multivalue', 'legal_multivalue')])
def test_no_unresolved_sourced_competition_spends_no_optional_requests(
        frozen_runtime, real_tokenizer, mode, reason):
    result, calls, _, _ = execute(frozen_runtime, real_tokenizer, mode)
    ranking = result['ranking']
    assert ranking['status'] == 'ok'
    assert ranking['proposal_count'] == 1 and ranking['clarify_count'] == 0
    assert ranking['trace'][0]['clarify']['reason'] == reason
    assert ranking['embedding_requests'] == 2 and ranking['logical_requests'] == 2
    assert not any(tuple(r['stage'])[:2] == ('resolve', 'clarify') for r in calls.requests)
    assert ranking['nodes'][0]['answer'] == 'Chen Hai'
    assert ranking['trace'][1]['query'] == 'In which city was Chen Hai born?'


def test_new_binding_evidence_switches_only_the_committed_child_and_its_provenance(
        frozen_runtime, real_tokenizer):
    result, calls, row, documents = execute(frozen_runtime, real_tokenizer)
    ranking = result['ranking']
    assert ranking['status'] == 'ok' and result['answer']['status'] == 'ok'
    first, child = ranking['trace']
    assert 'binding' not in first['panel_doc_ids']
    queries = first['clarify']['queries']
    assert len(queries) == 2
    assert all(q.startswith(name + ' ') for q, name in zip(queries, ('Chen Hai', 'Lin Zhou')))
    assert all(all(term in q for term in ('director', 'Homeward', '1998')) and 'born' not in q for q in queries)
    assert first['clarify']['status'] == 'judged'
    assert 'binding' in first['clarify']['panel_doc_ids']
    assert ranking['embedding_requests'] == 4 and ranking['logical_requests'] == 3
    assert ranking['nodes'][0]['answer'] == 'Lin Zhou'
    assert ranking['nodes'][0]['source_doc_ids'] == ['lin', 'binding']
    assert 'chen' not in ranking['nodes'][0]['closure_doc_ids']
    assert child['query'] == 'In which city was Lin Zhou born?'
    assert ranking['nodes'][1]['answer'] == 'Suzhou'
    assert set(ranking['nodes'][1]['closure_doc_ids']) == {'lin', 'binding', 'lin_birth'}
    child_embeddings = [r for r in calls.requests if tuple(r['stage']) == ('resolve', 'node', '1', 'embedding')]
    assert len(child_embeddings) == 1
    assert 'Chen Hai born' not in child_embeddings[0]['payload']['input'][0]
    reader = next(r for r in calls.requests if tuple(r['stage']) == ('reader', 'resolve', 'first'))
    visible = ranking['reader']['panel_doc_ids']
    origin = frozen_runtime.v6.reader_input(row, visible, documents, real_tokenizer, ranking['nodes'])
    assert reader['payload']['messages'] == origin['messages']
    assert reader['payload']['max_tokens'] == 1024
    assert ranking['response_usage']['total_tokens'] == 54
    assert ranking['response_usage_complete']


@pytest.mark.parametrize('mode', ['unknown', 'quote_invalid', 'judge_length', 'embedding_failure'])
def test_uncertain_or_failed_optional_clarification_preserves_primary_path(
        frozen_runtime, real_tokenizer, mode):
    result, calls, _, _ = execute(frozen_runtime, real_tokenizer, mode)
    ranking = result['ranking']
    assert ranking['status'] == 'ok' and ranking['clarify_count'] == 1
    assert ranking['nodes'][0]['answer'] == 'Chen Hai'
    assert ranking['nodes'][0]['source_doc_ids'] == ['chen']
    assert ranking['trace'][1]['query'] == 'In which city was Chen Hai born?'
    assert len([r for r in calls.requests if tuple(r['stage']) == ('reader', 'resolve', 'first')]) == 1
    assert ranking['logical_requests'] <= 3 and ranking['embedding_requests'] <= 4
    if mode == 'embedding_failure':
        assert ranking['trace'][0]['clarify']['status'] == 'failed'
        assert not ranking['embedding_usage_complete']
        assert ranking['logical_requests'] == 2
    elif mode == 'judge_length':
        assert ranking['trace'][0]['clarify']['status'] == 'failed'
    elif mode == 'quote_invalid':
        clarify = ranking['trace'][0]['clarify']
        assert clarify['status'] == 'failed' and clarify['reason'] == 'judge_protocol_invalid'
        assert not clarify['judgment']['protocol_valid'] and clarify['judgment']['diagnostic_errors']


@pytest.mark.parametrize('mode', ['joint_judge', 'invalid_primary_quote'])
def test_primary_citation_corruption_preserves_jointly_supported_primary_child(
        frozen_runtime, real_tokenizer, mode):
    result, calls, _, _ = execute(frozen_runtime, real_tokenizer, mode)
    first, child = result['ranking']['trace']
    assert first['state']['answer'] == 'Chen Hai'
    assert child['query'] == 'In which city was Chen Hai born?'
    assert len([r for r in calls.requests if tuple(r['stage']) == ('resolve', 'node', '1', 'embedding')]) == 1
    if mode == 'joint_judge':
        assert first['clarify']['status'] == 'judged'
        assert first['binding_decision']['binding_status'] == 'legal_multivalue'
        assert first['state']['source_doc_ids'] == ['chen', 'binding']
    else:
        assert first['clarify']['status'] == 'failed'
        assert first['clarify']['reason'] == 'judge_protocol_invalid'
        assert [c['status'] for c in first['clarify']['judgment']['candidates']] == ['invalid', 'supported']
        assert first['binding_decision']['binding_status'] == 'invalid'
        assert first['state']['source_doc_ids'] == ['chen']


def test_explicit_primary_refutation_without_supported_alternative_uses_origin_unresolved_grounding(
        frozen_runtime, real_tokenizer):
    result, _, _, _ = execute(frozen_runtime, real_tokenizer, 'contra_without_alt')
    ranking = result['ranking']
    assert ranking['status'] == 'ok'
    assert not ranking['nodes'][0]['resolved']
    assert ranking['nodes'][0]['answer'] == '' and ranking['nodes'][0]['source_doc_ids'] == []
    assert ranking['trace'][1]['query'] == 'In which city was (' + DIRECTOR + ') born?'
    assert ranking['trace'][1]['parents'][0]['answer'] == ''


def test_six_node_chain_with_multiple_bridges_has_only_one_proposal_and_repair(
        frozen_runtime, real_tokenizer):
    result, calls, _, _ = execute(frozen_runtime, real_tokenizer, node_count=6)
    ranking = result['ranking']
    assert ranking['status'] == 'ok' and len(ranking['nodes']) == 6
    assert ranking['proposal_count'] == ranking['clarify_count'] == 1
    assert ranking['logical_requests'] == 7 and ranking['embedding_requests'] == 8
    assert sum(r['payload'].get('max_tokens') == 1024 and 'prompt' in r['payload'] for r in calls.requests) == 2
    assert sum('messages' in r['payload'] for r in calls.requests) == 1
    assert [e['proposal_node'] for e in ranking['trace']] == [True, False, False, False, False, False]
    assert ranking['solver_limits'] == {'node_and_judge_llm': 7, 'embedding': 8, 'reader_reserved': 1}
    for event in ranking['trace']:
        assert event['prompt_tokens_local'] + event['output_tokens_reserved'] + 8 <= 16384


def test_single_terminal_question_does_not_add_a_proposal(frozen_runtime, real_tokenizer):
    result, _, _, _ = execute(frozen_runtime, real_tokenizer, node_count=1)
    ranking = result['ranking']
    assert ranking['status'] == 'ok'
    assert ranking['proposal_count'] == ranking['clarify_count'] == 0
    assert ranking['logical_requests'] == ranking['embedding_requests'] == 1


def test_disjoint_binding_hits_do_not_overfill_judge_panel_with_archive_fillers():
    anchors = ['primary_raw', 'alternative_raw']
    sides = [[side + str(i) for i in range(12)] for side in ('left_', 'right_')]
    panel, _, _ = flow._joint_panel(anchors, sides, ['neutral_' + str(i) for i in range(20)])
    assert len(panel) == len(set(panel)) == 20
    assert set(anchors) <= set(panel)
    assert sum(doc.startswith('left_') for doc in panel) == sum(doc.startswith('right_') for doc in panel)
    assert not any(doc.startswith('neutral_') for doc in panel)


def test_context_pressure_preserves_anchors_and_equal_retrieval_depth(real_tokenizer):
    anchors = ['primary_raw', 'alternative_raw']
    sides = [[side + str(i) for i in range(12)] for side in ('left_', 'right_')]
    passages = {doc: doc + '\n' + ('Complete raw evidence. ' * 700)
                for side in sides for doc in side}
    passages.update({doc: doc + ' exact protected quotation.' for doc in anchors})

    def messages(ids):
        return [{'role': 'user', 'content': '\n\n'.join(passages[doc] for doc in ids)}]

    prepared, visible, removed = flow._prepare_joint(messages, anchors, sides, [], real_tokenizer)
    assert set(anchors) <= set(visible) and removed
    left = [doc for doc in visible if doc.startswith('left_')]
    right = [doc for doc in visible if doc.startswith('right_')]
    assert 0 < len(left) == len(right) < 9
    assert left == sides[0][:len(left)] and right == sides[1][:len(right)]
    assert prepared['native_prompt_tokens_local'] + 1024 + 8 <= 16384
    assert prepared['messages'] == messages(visible)  # Complete retained documents, no quote clipping.


def test_oversized_protected_evidence_is_rejected_without_truncating_its_quotation(real_tokenizer):
    raw = 'Protected exact quotation. ' * 10000
    with pytest.raises(flow.ContextBudgetError):
        flow._prepare_joint(lambda _: [{'role': 'user', 'content': raw}],
                            ['protected'], [[], []], [], real_tokenizer)
