"""Offline checks for source grounding, optional proposals and fixed adoption rules."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from dagresolve import protocol as p
from package.core import NodeOutputError


QUESTION = 'In which city was the director of the 1998 film Homeward born?'
STEP = {'question': 'Who directed the 1998 film Homeward?', 'output_slot': 'director',
        'answer_type': 'person', 'inputs': []}
PANEL = ['chen', 'lin']
DOCUMENTS = {key: SimpleNamespace(passage=text) for key, text in {
    'chen': 'Chen Hai directed Homeward.',
    'lin': 'Lin Zhou directed Homeward.',
    'birth': 'Chen Hai was born in Wuhan. Lin Zhou was born in Suzhou.',
    'old': 'Chen Hai directed the 1986 film Homeward.',
    'new': 'The 1998 film Homeward was directed by Lin Zhou alone, not Chen Hai.',
    'chen1998': 'Chen Hai directed the 1998 film Homeward.',
    'co': 'Chen Hai and Lin Zhou jointly directed the 1998 film Homeward.',
}.items()}


def ref(index, quote):
    return {'panel_index': index, 'quote': quote}


def assessment(status='unknown', evidence=(), rid='r1'):
    return {'requirement_id': rid, 'status': status, 'evidence': list(evidence)}


def raw_proposal():
    return {'answer': 'Chen Hai', 'sources': [True, False], 'legal_multivalue': False,
            'requirements': [{'id': 'r1', 'kind': 'year_version',
                              'question_quote': 'the director of the 1998 film Homeward',
                              'relation_quote': 'director', 'subject_quote': 'Homeward',
                              'scope_quotes': ['1998']}],
            'candidates': [
                {'id': 'h1', 'answer': 'Chen Hai', 'name_quote': 'Chen Hai',
                 'evidence': [ref(0, DOCUMENTS['chen'].passage)], 'assessments': [assessment()]},
                {'id': 'h2', 'answer': 'Lin Zhou', 'name_quote': 'Lin Zhou',
                 'evidence': [ref(1, DOCUMENTS['lin'].passage)], 'assessments': [assessment()]}]}


def decode(obj=None):
    return p.decode_proposal(json.dumps(raw_proposal() if obj is None else obj), PANEL, DOCUMENTS, QUESTION)


def judgment(first='unknown', second='supported', *, panel=None, evidence=None):
    panel = PANEL + ['new'] if panel is None else panel
    quote = ref(panel.index('new'), DOCUMENTS['new'].passage) if 'new' in panel else None
    evidence = evidence if evidence is not None else ([quote] if quote else [])
    values = [{'id': 'h1', 'assessments': [assessment(first, evidence if first != 'unknown' else [])]},
              {'id': 'h2', 'assessments': [assessment(second, evidence if second != 'unknown' else [])]}]
    return p.decode_judgment(json.dumps({'candidates': values, 'legal_multivalue': False}),
                             decode(), panel, DOCUMENTS)


def test_schema_does_not_require_a_second_candidate_or_nonempty_conditions():
    schema = p.proposal_schema(20)
    assert schema['properties']['candidates']['minItems'] == 0
    assert schema['properties']['candidates']['maxItems'] == 2
    assert schema['properties']['requirements']['minItems'] == 0
    assert schema['properties']['sources']['minItems'] == schema['properties']['sources']['maxItems'] == 20


def test_two_sourced_competitors_trigger_deterministic_binding_queries():
    proposal = decode()
    assert proposal['answer'] == 'Chen Hai' and proposal['sources'] == ['chen']
    assert proposal['trigger'] == {'eligible': True, 'reason': 'primary_binding_unverified'}
    assert p.binding_queries(proposal) == [
        'Chen Hai the director of the 1998 film Homeward',
        'Lin Zhou the director of the 1998 film Homeward']
    anchor = proposal['candidates'][0]['evidence'][0]
    assert anchor == {'doc_id': 'chen', 'start': 0, 'end': len(DOCUMENTS['chen'].passage),
                      'quote': DOCUMENTS['chen'].passage}


def test_query_retains_binding_modifiers_outside_separate_scope_fields():
    question = 'In which city was the first director of film Homeward born?'
    obj = raw_proposal()
    obj['requirements'][0].update(kind='relation', scope_quotes=[],
        question_quote='the first director of film Homeward')
    proposal = p.decode_proposal(json.dumps(obj), PANEL, DOCUMENTS, question)
    assert proposal['trigger']['eligible']
    assert p.binding_queries(proposal) == [
        'Chen Hai the first director of film Homeward',
        'Lin Zhou the first director of film Homeward']


def test_initial_complete_binding_support_is_retained_without_extra_retrieval():
    obj = raw_proposal()
    panel = PANEL + ['chen1998']
    obj['sources'] = [True, False, False]
    obj['candidates'][0]['assessments'] = [assessment(
        'supported', [ref(2, DOCUMENTS['chen1998'].passage)])]
    proposal = p.decode_proposal(json.dumps(obj), panel, DOCUMENTS, QUESTION)
    assert proposal['trigger']['reason'] == 'primary_supported'
    committed = p.adopt(proposal, None)
    assert committed['sources'] == ['chen', 'chen1998']
    assert 'lin' not in committed['sources']


@pytest.mark.parametrize('count', [0, 1])
def test_zero_or_one_candidate_preserves_primary_without_trigger(count):
    obj = raw_proposal()
    obj['candidates'] = obj['candidates'][:count]
    proposal = decode(obj)
    assert not proposal['trigger']['eligible']
    assert p.binding_queries(proposal) == []
    result = p.adopt(proposal, None)
    assert result['answer'] == obj['answer'] and result['sources'] == ['chen']


def test_primary_supported_alternative_unknown_does_not_trigger():
    obj = raw_proposal()
    obj['candidates'][0]['assessments'] = [assessment('supported', [ref(0, DOCUMENTS['chen'].passage)])]
    proposal = decode(obj)
    assert proposal['trigger']['reason'] == 'primary_supported'


@pytest.mark.parametrize('status', ['contradicted', 'conflict'])
def test_initial_refutation_or_conflict_can_trigger_repair(status):
    obj = raw_proposal()
    evidence = [ref(0, DOCUMENTS['chen'].passage)]
    if status == 'conflict':
        evidence.append(ref(1, DOCUMENTS['lin'].passage))
    obj['candidates'][0]['assessments'] = [assessment(status, evidence)]
    proposal = decode(obj)
    assert proposal['trigger']['eligible']
    # A skipped clarification still follows the ordinary primary path.
    assert p.adopt(proposal, None)['answer'] == 'Chen Hai'


def test_two_fully_supported_candidates_are_legal_multivalue_not_an_error():
    obj = raw_proposal()
    for i, candidate in enumerate(obj['candidates']):
        candidate['assessments'] = [assessment('supported', [ref(i, DOCUMENTS[PANEL[i]].passage)])]
    proposal = decode(obj)
    assert proposal['legal_multivalue']
    assert proposal['trigger']['reason'] == 'legal_multivalue'
    assert p.adopt(proposal, None)['binding_status'] == 'legal_multivalue'


def test_multivalue_flag_without_support_is_not_accepted():
    obj = raw_proposal()
    obj['legal_multivalue'] = True
    proposal = decode(obj)
    assert proposal['trigger']['eligible']
    assert 'legal_multivalue:not_established' in proposal['diagnostic_errors']


@pytest.mark.parametrize('change', [
    lambda obj: obj['candidates'][1].update(name_quote='Invented Alias'),
    lambda obj: obj['candidates'][1].update(answer='Chen Hai', name_quote='Chen Hai'),
    lambda obj: obj['candidates'][1]['evidence'][0].update(quote='Text not in any source'),
    lambda obj: obj['candidates'][0].update(answer='Different Primary'),
])
def test_unsupported_name_duplicate_or_primary_mismatch_cannot_trigger(change):
    obj = raw_proposal()
    change(obj)
    proposal = decode(obj)
    assert not proposal['trigger']['eligible']
    assert proposal['answer'] == 'Chen Hai' and proposal['sources'] == ['chen']
    assert proposal['diagnostic_errors']


@pytest.mark.parametrize('field,value', [
    ('question_quote', 'a fabricated condition'), ('relation_quote', 'author'),
    ('subject_quote', 'Another Movie'), ('scope_quotes', ['1986']), ('scope_quotes', []),
])
def test_original_question_conditions_must_be_exact_connected_quotes(field, value):
    obj = raw_proposal()
    obj['requirements'][0][field] = value
    proposal = decode(obj)
    assert not proposal['trigger']['eligible'] and proposal['requirements'] == []


def test_requirements_sorted_by_fixed_kind_and_original_order():
    obj = raw_proposal()
    first = obj['requirements'][0]
    second = {**first, 'id': 'r2', 'kind': 'identity'}
    obj['requirements'] = [second, first]
    for candidate in obj['candidates']:
        candidate['assessments'].append(assessment(rid='r2'))
    assert [r['id'] for r in decode(obj)['requirements']] == ['r1', 'r2']


@pytest.mark.parametrize('obj', [None, [], {'answer': 'A', 'sources': [1, False]},
                               {'answer': 'A', 'sources': [True]}, {'answer': 12, 'sources': [True, False]}])
def test_illegal_primary_or_overall_json_fails_without_a_repair_call(obj):
    with pytest.raises(NodeOutputError):
        p.decode_proposal(json.dumps(obj), PANEL, DOCUMENTS, QUESTION)


def test_illegal_auxiliary_fields_leave_valid_ordinary_primary_available():
    obj = raw_proposal()
    del obj['requirements']
    proposal = decode(obj)
    assert proposal['answer'] == 'Chen Hai' and proposal['sources'] == ['chen']
    assert not proposal['trigger']['eligible'] and proposal['diagnostic_errors']


def test_primary_with_no_ordinary_source_stays_origin_unresolved():
    obj = raw_proposal()
    obj['sources'] = [False, False]
    proposal = decode(obj)
    assert proposal['trigger']['reason'] == 'primary_unresolved'


def test_one_valid_quote_does_not_rescue_an_invalid_cited_conjunction():
    obj = raw_proposal()
    obj['candidates'][0]['assessments'] = [assessment('supported', [
        ref(0, DOCUMENTS['chen'].passage), ref(1, 'missing part of the proof')])]
    proposal = decode(obj)
    assert proposal['candidates'][0]['status'] == 'unknown'
    assert proposal['trigger']['eligible']


@pytest.mark.parametrize('rid,status', [([], 'supported'), ({}, 'supported'), ('r1', []), ('r1', {})])
def test_malformed_assessment_types_fall_back_instead_of_crashing(rid, status):
    obj = raw_proposal()
    obj['candidates'][0]['assessments'] = [assessment(status, rid=rid)]
    proposal = decode(obj)
    assert proposal['candidates'][0]['status'] == 'unknown'
    assert proposal['answer'] == 'Chen Hai'


@pytest.mark.parametrize('section,field,value', [
    ('candidates', 'id', []), ('candidates', 'id', {}),
    ('candidates', 'answer', []), ('candidates', 'name_quote', {}),
    ('candidates', 'evidence', {}), ('candidates', 'assessments', {}),
    ('requirements', 'id', []), ('requirements', 'id', {}),
    ('requirements', 'kind', []), ('requirements', 'kind', {}),
    ('requirements', 'question_quote', []), ('requirements', 'scope_quotes', {}),
])
def test_untrusted_optional_json_fields_never_break_a_valid_primary(section, field, value):
    obj = raw_proposal()
    obj[section][0][field] = value
    proposal = decode(obj)
    assert proposal['answer'] == 'Chen Hai' and proposal['sources'] == ['chen']
    assert proposal['diagnostic_errors']


def test_candidate_requires_every_necessary_requirement_to_be_supported():
    obj = raw_proposal()
    obj['requirements'].append({**obj['requirements'][0], 'id': 'r2', 'kind': 'identity'})
    obj['candidates'][0]['assessments'] = [assessment('supported', [ref(0, DOCUMENTS['chen'].passage)])]
    proposal = decode(obj)
    assert proposal['candidates'][0]['status'] == 'unknown'


def test_unknown_primary_supported_alternative_adopts_only_alternative_proof():
    result = p.adopt(decode(), judgment())
    assert result['answer'] == 'Lin Zhou'
    assert result['sources'] == ['lin', 'new']
    assert 'chen' not in result['sources']
    assert result['unexcluded_competitor'] is True
    assert result['binding_status'] == 'supported'


def test_refuted_primary_supported_alternative_is_excluded_not_merely_unknown():
    result = p.adopt(decode(), judgment('contradicted'))
    assert result['answer'] == 'Lin Zhou' and not result['unexcluded_competitor']


def test_refuted_primary_without_supported_alternative_becomes_unresolved():
    result = p.adopt(decode(), judgment('contradicted', 'unknown'))
    assert result['answer'] == '' and result['sources'] == []
    assert result['selected_candidate_id'] is None


def test_supported_primary_retains_all_ordinary_sources_and_new_support():
    proposal = decode()
    proposal['sources'] = ['chen', 'birth']  # Ordinary proof may be a conjunction.
    result = p.adopt(proposal, judgment('supported', 'unknown'))
    assert result['answer'] == 'Chen Hai'
    assert result['sources'] == ['chen', 'birth', 'new']


def test_same_scope_support_and_refutation_is_conflict_without_source_voting():
    proposal = decode()
    panel = PANEL + ['new', 'chen1998']
    values = [{'id': 'h1', 'assessments': [
        assessment('supported', [ref(3, DOCUMENTS['chen1998'].passage)]),
        assessment('contradicted', [ref(2, DOCUMENTS['new'].passage)])]},
        {'id': 'h2', 'assessments': [assessment('supported', [ref(2, DOCUMENTS['new'].passage)])]}]
    result = p.decode_judgment(json.dumps({'candidates': values, 'legal_multivalue': False}),
                               proposal, panel, DOCUMENTS)
    assert result['candidates'][0]['status'] == 'conflict'
    adopted = p.adopt(proposal, result)
    assert adopted['answer'] == 'Chen Hai' and adopted['binding_status'] == 'conflict'


@pytest.mark.parametrize('raw', ['truncated {', '{}', json.dumps({'candidates': [], 'legal_multivalue': False}),
                               json.dumps({'candidates': [{'id': [], 'assessments': []},
                                                          {'id': 'h2', 'assessments': []}],
                                           'legal_multivalue': False})])
def test_failed_judge_is_unknown_and_keeps_primary(raw):
    proposal = decode()
    result = p.decode_judgment(raw, proposal, PANEL, DOCUMENTS)
    assert all(c['status'] == 'unknown' for c in result['candidates'])
    assert p.adopt(proposal, result)['answer'] == 'Chen Hai'
    assert result['diagnostic_errors']


def test_invalid_judge_reference_does_not_prune_unknown_candidate():
    result = judgment('contradicted', 'unknown', evidence=[ref(99, 'not displayed')])
    assert result['candidates'][0]['status'] == 'unknown'
    assert p.adopt(decode(), result)['answer'] == 'Chen Hai'


def test_jointly_supported_judgment_keeps_primary_without_forcing_uniqueness():
    panel = PANEL + ['co']
    result = judgment('supported', 'supported', panel=panel, evidence=[ref(2, DOCUMENTS['co'].passage)])
    assert result['legal_multivalue']
    assert p.adopt(decode(), result)['answer'] == 'Chen Hai'
    assert p.adopt(decode(), result)['binding_status'] == 'legal_multivalue'
    assert p.adopt(decode(), result)['sources'] == ['chen', 'co']


def test_prompts_keep_original_task_raw_text_and_semantic_unknown_guard():
    original = deepcopy(DOCUMENTS)
    proposed = p.proposal_messages(QUESTION, STEP, STEP['question'], PANEL, [], DOCUMENTS)
    text = proposed[1]['content']
    assert QUESTION in text and DOCUMENTS['chen'].passage in text
    assert 'Return exactly two fields:' not in text
    assert 'TWO is a maximum, never a target' in text
    judged = p.judge_messages(QUESTION, STEP, decode(), PANEL + ['old'], DOCUMENTS)[1]['content']
    assert 'different year/version' in judged and 'same scope' in judged
    assert DOCUMENTS['old'].passage in judged
    assert DOCUMENTS == original
