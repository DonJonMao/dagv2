"""Small, source-grounded protocol for one binding clarification before a DAG commit.

JSON validity and exact quotations are checked here. Whether a quotation entails
a relation remains a model judgment; the prompts deliberately distinguish absence
of evidence from contradiction. The ordinary primary answer/source contract is
kept separate from optional clarification fields.
"""
import json

from package import core


KINDS = ('year_version', 'identity', 'relation')
STATUSES = ('supported', 'contradicted', 'unknown', 'conflict')
MAX_ANSWER = 256
MAX_QUOTE = 600
MAX_CONDITION = 512


def _array(items, maximum, minimum=0):
    return {'type': 'array', 'items': items, 'minItems': minimum, 'maxItems': maximum}


def _text(maximum=MAX_QUOTE):
    return {'type': 'string', 'maxLength': maximum}


def _object(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties),
            'additionalProperties': False}


def _evidence_schema(n):
    return _object({'panel_index': {'type': 'integer', 'minimum': 0, 'maximum': max(0, n - 1)},
                    'quote': _text()})


def _assessment_schema(n, requirement_ids=('r1', 'r2')):
    return _object({'requirement_id': {'type': 'string', 'enum': list(requirement_ids)},
                    'status': {'type': 'string', 'enum': list(STATUSES)},
                    'evidence': _array(_evidence_schema(n), 4)})


def proposal_schema(n):
    """Allow zero/one candidates: the model must never manufacture a second one."""
    candidate = _object({'id': {'type': 'string', 'enum': ['h1', 'h2']},
                         'answer': _text(MAX_ANSWER), 'name_quote': _text(MAX_ANSWER),
                         'evidence': _array(_evidence_schema(n), 4),
                         'assessments': _array(_assessment_schema(n), 4)})
    requirement = _object({'id': {'type': 'string', 'enum': ['r1', 'r2']},
                           'kind': {'type': 'string', 'enum': list(KINDS)},
                           'question_quote': _text(MAX_CONDITION),
                           'relation_quote': _text(100), 'subject_quote': _text(200),
                           'scope_quotes': _array(_text(200), 3)})
    return _object({'answer': _text(MAX_ANSWER),
                    'sources': _array({'type': 'boolean'}, n, n),
                    'candidates': _array(candidate, 2),
                    'requirements': _array(requirement, 2),
                    'legal_multivalue': {'type': 'boolean'}})


def proposal_messages(question, step, query, panel, parents, documents):
    """Reuse the ordinary task/evidence prompt, replacing its output instructions."""
    messages = core.messages(question, step, query, panel, parents, documents)
    marker = 'Return exactly two fields:'
    ordinary = messages[-1]['content'].split(marker, 1)[0]
    messages[-1]['content'] = ordinary + (
        'Return a JSON object with answer, sources, candidates, requirements, legal_multivalue. '
        f'answer and the exactly {len(panel)} sources booleans retain the ordinary meaning above: '
        'preserve ALL passages needed for the primary answer. Empty answer has all-false sources. '
        'Only list intermediate answers actually found in the displayed sources; do not guess '
        'another candidate or an alias. candidates may be empty or contain ONE candidate; TWO is '
        'a maximum, never a target. If candidates are listed, h1 must exactly match answer; h2 '
        'must be a different source-grounded answer to this same current task. candidate.answer '
        'and name_quote must name the same literal entity, and name_quote must occur verbatim '
        'inside at least one of its evidence quotations. Each quotation has panel_index (zero '
        'based displayed passage number) and quote (exact, contiguous raw text). '
        'requirements contains at most TWO necessary binding conditions from the ORIGINAL '
        'question, not the requested downstream attribute. Each requirement has id r1/r2, kind '
        'year_version/identity/relation, question_quote, relation_quote, subject_quote, scope_quotes. '
        'All of these strings must be exact original-question text; relation_quote, subject_quote '
        'and every scope_quote must be inside question_quote. Select year/version first, then '
        'identity, then relation. question_quote must retain EVERY original modifier constraining '
        'this binding, including first/former/sole and year/version/role; do not select a shorter '
        'substring that drops a necessary condition. Preserve original-question order within a kind. '
        'Do not invent '
        'a constraint or use the downstream task as a binding condition. If no such binding can '
        'be quoted, return requirements=[]. For each candidate assess EVERY requirement using '
        'supported, contradicted, unknown, or conflict and evidence quotations. supported means '
        'the complete required binding is established, not merely the candidate identity or a '
        'downstream attribute. contradicted requires explicit exclusion/incompatibility for the '
        'SAME scope; another year/version or missing facts is unknown. Where supported and '
        'contradicted evidence coexist, use conflict with both quotations, or separate supported '
        'and contradicted assessments for the same requirement. Do not vote by source count. '
        'unknown has no supporting/refuting evidence. Mark legal_multivalue true only if source '
        'text establishes that BOTH candidates legitimately satisfy all original binding '
        'conditions (for example, co-directors of the specified version). This is not an error '
        'to resolve. Do not add explanations outside these fields.')
    return messages


def _string(value, maximum, *, empty=False):
    return isinstance(value, str) and len(value) <= maximum and (empty or bool(value.strip()))


def _name(value):
    return value.strip().casefold()


def _references(values, panel, documents, errors, label):
    if not isinstance(values, list) or len(values) > 4:
        errors.append(label + ':evidence_contract')
        return []
    found, seen = [], set()
    for item in values:
        if (not isinstance(item, dict) or set(item) != {'panel_index', 'quote'}
                or type(item['panel_index']) is not int
                or not 0 <= item['panel_index'] < len(panel)
                or not _string(item['quote'], MAX_QUOTE)):
            errors.append(label + ':invalid_reference')
            continue
        doc_id = panel[item['panel_index']]
        passage = documents[doc_id].passage
        start = passage.find(item['quote'])
        if start < 0:
            errors.append(label + ':quote_not_in_passage')
            continue
        key = (doc_id, start, item['quote'])
        if key not in seen:
            seen.add(key)
            found.append({'doc_id': doc_id, 'start': start, 'end': start + len(item['quote']),
                          'quote': item['quote']})
    return found


def _requirements(values, question, errors):
    if not isinstance(values, list) or len(values) > 2:
        errors.append('requirements:contract')
        return []
    result, seen = [], set()
    fields = {'id', 'kind', 'question_quote', 'relation_quote', 'subject_quote', 'scope_quotes'}
    for value in values:
        if (not isinstance(value, dict) or set(value) != fields
                or value['id'] not in ('r1', 'r2') or value['id'] in seen
                or value['kind'] not in KINDS
                or not _string(value['question_quote'], MAX_CONDITION)
                or not _string(value['relation_quote'], 100)
                or not _string(value['subject_quote'], 200)
                or not isinstance(value['scope_quotes'], list) or len(value['scope_quotes']) > 3
                or any(not _string(q, 200) for q in value['scope_quotes'])):
            errors.append('requirements:contract')
            return []
        spans = [value['relation_quote'], value['subject_quote'], *value['scope_quotes']]
        if value['question_quote'] not in question or any(q not in value['question_quote'] for q in spans):
            errors.append('requirements:quote_not_in_question')
            return []
        if _name(value['relation_quote']) == _name(value['subject_quote']):
            errors.append('requirements:relation_subject_identical')
            return []
        if value['kind'] == 'year_version' and not value['scope_quotes']:
            errors.append('requirements:missing_year_version_scope')
            return []
        seen.add(value['id'])
        result.append({**value, 'scope_quotes': list(dict.fromkeys(value['scope_quotes']))})
    return sorted(result, key=lambda r: (KINDS.index(r['kind']), question.find(r['question_quote'])))


def _assessments(values, requirements, panel, documents, errors, label):
    groups = {r['id']: [] for r in requirements}
    if not isinstance(values, list) or len(values) > 4:
        errors.append(label + ':assessment_contract')
        values = []
    for value in values:
        if (not isinstance(value, dict) or set(value) != {'requirement_id', 'status', 'evidence'}
                or not isinstance(value['requirement_id'], str)
                or value['requirement_id'] not in groups or not isinstance(value['status'], str)
                or value['status'] not in STATUSES):
            errors.append(label + ':assessment_contract')
            continue
        previous_errors = len(errors)
        evidence = _references(value['evidence'], panel, documents, errors, label)
        status = value['status']
        if len(errors) != previous_errors:
            # Protocol failure is not semantic absence of evidence, even for unknown.
            status = 'invalid'
        if status in ('supported', 'contradicted') and not evidence:
            errors.append(label + ':label_without_evidence')
            status = 'invalid'
        if status == 'conflict' and len(evidence) < 2:
            errors.append(label + ':conflict_without_two_references')
            status = 'invalid'
        groups[value['requirement_id']].append({'status': status, 'evidence': evidence})
    assessments = []
    for requirement in requirements:
        records = groups[requirement['id']]
        labels = {r['status'] for r in records}
        if not records:
            errors.append(label + ':missing_assessment:' + requirement['id'])
            status = 'invalid'
        elif 'invalid' in labels:
            status = 'invalid'
        elif 'conflict' in labels or {'supported', 'contradicted'} <= labels:
            status = 'conflict'
        elif 'contradicted' in labels:
            status = 'contradicted'
        elif 'supported' in labels:
            status = 'supported'
        else:
            status = 'unknown'
        evidence, seen = [], set()
        for record in records:
            if record['status'] in ('unknown', 'invalid'):
                continue
            for reference in record['evidence']:
                key = (reference['doc_id'], reference['start'], reference['end'])
                if key not in seen:
                    seen.add(key)
                    evidence.append(reference)
        assessments.append({'requirement_id': requirement['id'], 'status': status, 'evidence': evidence})
    return assessments


def _status(assessments):
    labels = {a['status'] for a in assessments}
    if not assessments:
        return 'unknown'
    if 'invalid' in labels:
        return 'invalid'
    if 'conflict' in labels:
        return 'conflict'
    if 'contradicted' in labels:
        return 'contradicted'
    return 'supported' if labels == {'supported'} else 'unknown'


def _support_ids(assessments):
    return list(dict.fromkeys(e['doc_id'] for a in assessments if a['status'] == 'supported'
                              for e in a['evidence']))


def decode_proposal(raw, panel, documents, question):
    """Invalid primary fails; invalid optional fields cannot cause clarification."""
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise core.NodeOutputError('proposal_json') from exc
    if not isinstance(obj, dict) or not _string(obj.get('answer'), MAX_ANSWER, empty=True):
        raise core.NodeOutputError('proposal_primary_contract')
    answer, sources = core.decode(json.dumps({'answer': obj['answer'], 'sources': obj.get('sources')}), panel)
    errors = []
    expected = {'answer', 'sources', 'candidates', 'requirements', 'legal_multivalue'}
    extras_valid = set(obj) == expected and type(obj.get('legal_multivalue')) is bool
    if not extras_valid:
        errors.append('proposal:optional_fields_contract')
    requirements = _requirements(obj.get('requirements'), question, errors) if extras_valid else []
    candidates, seen_ids, seen_names = [], set(), set()
    values = obj.get('candidates')
    if not extras_valid or not isinstance(values, list) or len(values) > 2:
        errors.append('candidates:contract')
        values = []
    for value in values:
        fields = {'id', 'answer', 'name_quote', 'evidence', 'assessments'}
        if (not isinstance(value, dict) or set(value) != fields or value['id'] not in ('h1', 'h2')
                or value['id'] in seen_ids or not _string(value['answer'], MAX_ANSWER)
                or not _string(value['name_quote'], MAX_ANSWER)
                or _name(value['answer']) != _name(value['name_quote'])
                or _name(value['answer']) in seen_names
                or (value['id'] == 'h1' and _name(value['answer']) != _name(answer))
                or (value['id'] == 'h2' and _name(value['answer']) == _name(answer))):
            errors.append('candidates:contract')
            continue
        anchors = _references(value['evidence'], panel, documents, errors, value['id'] + ':anchor')
        if not anchors or not any(value['name_quote'] in e['quote'] for e in anchors):
            errors.append(value['id'] + ':name_not_in_anchor')
            continue
        previous_errors = len(errors)
        assessments = _assessments(value['assessments'], requirements, panel, documents, errors, value['id'])
        assessment_valid = len(errors) == previous_errors
        candidates.append({**value, 'evidence': anchors, 'assessments': assessments,
                           'status': _status(assessments) if assessment_valid else 'invalid',
                           'support_doc_ids': _support_ids(assessments) if assessment_valid else []})
        seen_ids.add(value['id'])
        seen_names.add(_name(value['answer']))
    candidates.sort(key=lambda c: c['id'])
    if candidates and candidates[0]['id'] != 'h1':
        errors.append('candidates:missing_primary')
        candidates = []
    legal_multivalue = len(candidates) == 2 and all(c['status'] == 'supported' for c in candidates)
    if obj.get('legal_multivalue') is True and not legal_multivalue:
        errors.append('legal_multivalue:not_established')
    if not answer.strip() or not sources:
        reason = 'primary_unresolved'
    elif errors:
        reason = 'proposal_protocol_invalid'
    elif len(candidates) < 2:
        reason = 'no_competing_candidate'
    elif not requirements:
        reason = 'no_original_binding_condition'
    elif legal_multivalue:
        reason = 'legal_multivalue'
    elif candidates[0]['status'] == 'supported':
        reason = 'primary_supported'
    else:
        reason = 'primary_binding_unverified'
    return {'answer': answer, 'sources': sources, 'candidates': candidates,
            'requirements': requirements, 'legal_multivalue': legal_multivalue,
            'trigger': {'eligible': reason == 'primary_binding_unverified', 'reason': reason},
            'protocol_valid': not errors, 'diagnostic_errors': errors}


def binding_queries(proposal):
    """Use one unresolved condition, with fixed kind/question-order priority."""
    if not proposal['trigger']['eligible']:
        return []
    primary = proposal['candidates'][0]
    by_id = {a['requirement_id']: a['status'] for a in primary['assessments']}
    requirement = next(r for r in proposal['requirements'] if by_id[r['id']] != 'supported')
    result = []
    for candidate in proposal['candidates']:
        # The full binding quote retains modifiers such as "first", "film"
        # and version/role qualifiers that separate extracted fields may omit.
        result.append(candidate['name_quote'] + ' ' + requirement['question_quote'])
    return result


def judge_schema(proposal, panel_size):
    ids = [c['id'] for c in proposal['candidates']]
    requirements = [r['id'] for r in proposal['requirements']]
    candidate = _object({'id': {'type': 'string', 'enum': ids},
                         'assessments': _array(_assessment_schema(panel_size, requirements), 4, len(requirements))})
    return _object({'candidates': _array(candidate, len(ids), len(ids)),
                    'legal_multivalue': {'type': 'boolean'}})


def judge_messages(question, step, proposal, panel, documents):
    candidates = [{'id': c['id'], 'answer': c['answer']} for c in proposal['candidates']]
    data = {'original_question': question, 'current_task': step['question'],
            'candidates': candidates, 'requirements': proposal['requirements']}
    text = (json.dumps(data, ensure_ascii=False) + '\n\nFull raw passages:\n'
            + '\n\n'.join(f'Passage [{i}]\n{documents[d].passage}' for i, d in enumerate(panel))
            + '\n\nJudge EACH candidate against EVERY complete original-question binding requirement. '
            'The proposals are fallible and are not evidence. Do not answer the downstream task, '
            'vote by passage counts, or infer exclusion merely because another candidate is supported. '
            'supported requires evidence of the complete binding; contradicted requires explicit '
            'exclusion/incompatibility for that same scope. Missing evidence, a different year/version '
            'or an unrelated attribute is unknown. If raw support and refutation coexist for the '
            'same requirement, return conflict with both quotations, or separate supported and '
            'contradicted assessments. Evidence uses zero-based panel_index and exact contiguous '
            'quote from that displayed passage. unknown may have evidence=[]. Return candidates '
            'with their original ids and assessments (requirement_id,status,evidence), and '
            'legal_multivalue. Set legal_multivalue true only when both candidates legitimately '
            'satisfy ALL conditions; joint/co-roles are allowed. Do not force a single candidate.')
    return [{'role': 'system', 'content': 'Assess candidate bindings only from the full raw source text.'},
            {'role': 'user', 'content': text}]


def decode_judgment(raw, proposal, panel, documents):
    """Keep protocol validity separate from semantic uncertainty in the evidence."""
    errors = []
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        obj = None
    ids = {c['id'] for c in proposal['candidates']}
    valid = (isinstance(obj, dict) and set(obj) == {'candidates', 'legal_multivalue'}
             and type(obj['legal_multivalue']) is bool and isinstance(obj['candidates'], list)
             and len(obj['candidates']) == len(ids))
    values = obj['candidates'] if valid else []
    if not valid:
        errors.append('judge:output_contract')
    mapped = {}
    for value in values:
        if (not isinstance(value, dict) or set(value) != {'id', 'assessments'}
                or not isinstance(value['id'], str)
                or value['id'] not in ids or value['id'] in mapped):
            errors.append('judge:candidate_contract')
            mapped = {}
            break
        mapped[value['id']] = value
    if set(mapped) != ids:
        if valid:
            errors.append('judge:candidate_coverage')
        mapped = {}
    candidates = []
    for candidate in proposal['candidates']:
        previous_errors = len(errors)
        assessments = _assessments(mapped.get(candidate['id'], {}).get('assessments', []),
                                   proposal['requirements'], panel, documents, errors, candidate['id'])
        assessment_valid = len(errors) == previous_errors
        candidates.append({'id': candidate['id'],
                           'status': _status(assessments) if assessment_valid else 'invalid',
                           'assessments': assessments,
                           'support_doc_ids': _support_ids(assessments) if assessment_valid else []})
    legal_multivalue = len(candidates) == 2 and all(c['status'] == 'supported' for c in candidates)
    if valid and obj['legal_multivalue'] and not legal_multivalue:
        errors.append('judge:legal_multivalue_not_established')
    return {'candidates': candidates, 'legal_multivalue': legal_multivalue,
            'protocol_valid': not errors, 'diagnostic_errors': errors}


def adopt(proposal, judgment):
    """Submit one value using the fixed MVP policy, preserving ordinary primary proof."""
    primary = next((c for c in proposal['candidates'] if c['id'] == 'h1'), None)
    if (proposal.get('protocol_valid') is not True
            or (judgment is not None and judgment.get('protocol_valid') is not True)):
        return {'answer': proposal['answer'], 'sources': list(proposal['sources']),
                'binding_status': 'invalid', 'unexcluded_competitor': False,
                'selected_candidate_id': 'h1' if primary else None,
                'decision_reason': 'proposal_protocol_invalid' if not proposal.get('protocol_valid')
                                   else 'judge_protocol_invalid'}
    outcomes = {c['id']: c for c in judgment['candidates']} if judgment else {}
    first = outcomes.get('h1', {'status': primary['status'] if primary and judgment is None else 'unknown',
                                'support_doc_ids': primary['support_doc_ids'] if primary and judgment is None else []})
    second = outcomes.get('h2', {'status': 'unknown', 'support_doc_ids': []})
    answer, sources = proposal['answer'], list(proposal['sources'])
    status = first['status'] if primary else 'not_checked'
    selected, unexcluded = 'h1' if primary else None, False
    if (judgment and judgment['legal_multivalue']) or (judgment is None and proposal['legal_multivalue']):
        status = 'legal_multivalue'
        sources = list(dict.fromkeys(sources + first['support_doc_ids']))
    elif second['status'] == 'supported' and first['status'] in ('unknown', 'contradicted'):
        alternative = next(c for c in proposal['candidates'] if c['id'] == 'h2')
        answer = alternative['answer']
        sources = list(dict.fromkeys([e['doc_id'] for e in alternative['evidence']]
                                     + second['support_doc_ids']))
        status, selected = 'supported', 'h2'
        unexcluded = first['status'] == 'unknown'
    elif judgment is not None and first['status'] == 'contradicted':
        answer, sources, status, selected = '', [], 'contradicted', None
    elif first['status'] == 'supported':
        sources = list(dict.fromkeys(sources + first['support_doc_ids']))
    return {'answer': answer, 'sources': sources, 'binding_status': status,
            'unexcluded_competitor': unexcluded, 'selected_candidate_id': selected}
