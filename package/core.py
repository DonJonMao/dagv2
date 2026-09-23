"""Execute saved query dependencies and retain conjunctions of cited sources."""
from graphlib import TopologicalSorter
from itertools import combinations
import json


class NodeOutputError(ValueError):
    pass


def ordered_steps(plan):
    steps = {s['output_slot']: s for s in plan['steps']}
    return [steps[slot] for slot in TopologicalSorter(
        {slot: s['inputs'] for slot, s in steps.items()}).static_order()]


def ground(step, states):
    query = step['question']
    for slot in step['inputs']:
        parent = states[slot]
        value = parent['answer'] if parent['resolved'] else '(' + parent['question'] + ')'
        placeholder = '{' + slot + '}'
        if placeholder in query:
            query = query.replace(placeholder, value)
        else:
            query += '\n' + slot + ': ' + value
    return query


def source_panel(pool, retrieved, parents, k=20):
    inherited = {d for p in parents if p['resolved'] for d in p['source_doc_ids']}
    ordered = [d for d in pool if d in inherited]
    ordered.extend(d for d in retrieved if d not in inherited)
    return ordered[:k]


def messages(question, step, query, panel, parents, documents):
    assignments = [{k: p[k] for k in ('output_slot', 'question', 'answer', 'resolved')}
                   for p in parents]
    text = ('Original question:\n' + question + '\n\nCurrent task:\n' + step['question']
            + '\n\nTask with the available parent assignments:\n' + query
            + '\n\nParent results (fallible proposals, not independent evidence):\n'
            + json.dumps(assignments, ensure_ascii=False)
            + '\n\nAvailable source passages in full:\n' + '\n\n'.join(
                f'Passage [{i}]\n{documents[d].passage}' for i, d in enumerate(panel))
            + '\n\nAnswer the CURRENT TASK using these sources. Resolve the actual relation and '
            'entity identity; do not substitute a topic-related entity. Parent answers can be wrong: '
            'use the source text to assess them. If the task cannot be answered from the available '
            'evidence, leave answer empty. Otherwise give its concise answer. '
            'Identify the passages that together establish your answer; retain all parts of a '
            'multi-passage argument, but do not select passages merely for topic overlap. '
            'Return exactly two fields: answer (a short string, or an empty string if unresolved) '
            f'and sources (exactly {len(panel)} booleans, one for EACH displayed passage in order). '
            'True means that passage is used as evidence for this answer. These are not passage '
            'identifiers. An unresolved answer has an all-false sources array. Do not include '
            'a rationale in the final JSON object.')
    return [{'role': 'system', 'content': 'Solve the current evidence-grounded task and identify its source passages.'},
            {'role': 'user', 'content': text}]


def schema(n):
    return {'type': 'object', 'properties': {
        'answer': {'type': 'string'},
        'sources': {'type': 'array', 'items': {'type': 'boolean'}, 'minItems': n, 'maxItems': n}},
        'required': ['answer', 'sources'], 'additionalProperties': False}


def decode(raw, panel):
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as error:
        raise NodeOutputError('node_json') from error
    if (not isinstance(obj, dict) or set(obj) != {'answer', 'sources'}
            or not isinstance(obj['answer'], str) or not isinstance(obj['sources'], list)
            or len(obj['sources']) != len(panel) or any(type(v) is not bool for v in obj['sources'])):
        raise NodeOutputError('node_output_contract')
    return obj['answer'], [d for d, cited in zip(panel, obj['sources']) if cited]


def node_state(step, query, answer, sources, parents, pool):
    # A model answer without a cited source is not an assignment for its children.
    resolved = bool(answer.strip() and sources)
    closure = set(sources) if resolved else set()
    if resolved:
        closure.update(d for p in parents if p['resolved'] for d in p['closure_doc_ids'])
    return {**step, 'grounded_question': query, 'answer': answer, 'resolved': resolved,
            'source_doc_ids': sources, 'closure_doc_ids': [d for d in pool if d in closure]}


def select_sources(pool, nodes, k):
    """Maximize fully retained node proofs, then prefer the original retrieval order.

    At most six nodes: enumerate proof unions exactly, not independent document scores.
    Each proof includes the sources of the parent assignments used to solve that node.
    Fill unused slots from the original archive order; optimize each global k separately.
    """
    proofs = [set(n['closure_doc_ids']) for n in nodes if n['resolved']]
    best, best_key = None, None
    for count in range(len(proofs) + 1):
        for subset in combinations(proofs, count):
            chosen = set().union(*subset)
            if len(chosen) > k:
                continue
            for d in pool:
                if len(chosen) == min(k, len(pool)):
                    break
                chosen.add(d)
            indices = tuple(i for i, d in enumerate(pool) if d in chosen)
            covered = sum(p <= chosen for p in proofs)
            key = (-covered, indices)
            if best_key is None or key < best_key:
                best, best_key = chosen, key
    return {'selected_doc_ids': [d for d in pool if d in best],
            'retained_node_proofs': -best_key[0], 'available_node_proofs': len(proofs)}
