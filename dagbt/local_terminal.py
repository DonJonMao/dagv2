"""Versioned terminal contract; no retrieval, model call, or answer fallback."""
from copy import deepcopy
from itertools import product
import re

from .reasoning import ProtocolError

VERSION = 'dagbt_local_terminal_v1'


def planner_contract(base_schema):
    schema = deepcopy(base_schema)
    schema['properties']['final_node_id'] = {'type': 'string'}
    schema['required'].append('final_node_id')
    item = schema['properties']['steps']['items']
    item['properties']['execution'] = {'enum': ['retrieval', 'compose']}
    item['required'].append('execution')
    return schema


PLAN_SYSTEM = '''Plan an executable dependency DAG of 1 to 6 tasks using only the question.
Return steps with question, output_slot, answer_type, inputs, execution, plus final_node_id.
Do not answer, guess intermediate entities, or use outside knowledge. Preserve all entity,
year, relation and scope constraints. Ground later queries with {parent_slot} placeholders.
execution=retrieval obtains missing factual evidence; execution=compose combines supported
parent conclusions and their necessary sources. Compose tasks must declare their necessary
parents. Include the final comparison or synthesis in this same plan when needed.
final_node_id identifies the existing task that answers the requested question completely.
final_node_id MUST exactly equal that task's output_slot string. Never use a positional
name such as step2 unless step2 is literally the declared output_slot. For example,
if the final task has output_slot="city", return final_node_id="city".
For a natural chain such as director -> university -> city, city is the final task; do not
append a redundant answer task. The final task has no children. Independent tasks are allowed.
Use unique simple output slots, no cycles, no hidden extra nodes. Do not include extra fields.'''


def validate_plan(value, legacy_validate):
    if not isinstance(value, dict) or set(value) != {'steps', 'final_node_id'}:
        raise ProtocolError('Local plan requires steps and final_node_id')
    steps = value['steps']
    if not isinstance(steps, list) or not 1 <= len(steps) <= 6:
        raise ProtocolError('Local plan size must be 1..6')
    pending = deepcopy(steps)
    slots = [s.get('output_slot') for s in pending if isinstance(s, dict)]
    if len(slots) != len(pending) or any(not isinstance(s, str) for s in slots) or len(set(slots)) != len(slots):
        raise ProtocolError('Local plan has invalid or duplicate slots')
    final = value['final_node_id']
    if final not in slots:
        raise ProtocolError('Designated final node is absent')
    kinds = {}
    for step in pending:
        if set(step) != {'question', 'output_slot', 'answer_type', 'inputs', 'execution'}:
            raise ProtocolError('Local plan step fields malformed')
        kind = step.pop('execution')
        if kind not in ('retrieval', 'compose') or (kind == 'compose' and not step['inputs']):
            raise ProtocolError('Compose requires parents; execution must be retrieval/compose')
        if not isinstance(step['inputs'], list) or any(not isinstance(p, str) for p in step['inputs']):
            raise ProtocolError('Invalid plan inputs')
        referenced = re.findall(r'\{([^{}]+)\}', step.get('question', '') if isinstance(step.get('question'), str) else '')
        if final in step['inputs'] + referenced:
            raise ProtocolError('Final node must have no children')
        step['inputs'] = list(dict.fromkeys(step['inputs'] + referenced))
        kinds[step['output_slot']] = kind
    # Storage order is not semantic. Normalize to a valid dependency order before
    # applying the frozen validator, which also validates placeholders and fields.
    ordered, seen = [], set()
    while pending:
        ready = [s for s in pending if set(s['inputs']) <= seen]
        if not ready:
            raise ProtocolError('Cyclic or missing local plan parents')
        for step in ready:
            ordered.append(step); seen.add(step['output_slot']); pending.remove(step)
    cleaned = legacy_validate({'steps': ordered})
    for step in cleaned['steps']:
        step['execution'] = kinds[step['output_slot']]
    return {**cleaned, 'final_node_id': final}


TERMINAL_INSTRUCTION = '''
This is the designated DAG terminal task. In THIS SAME response, return the supported
semantic answer and final_prediction that satisfies public_output_question, if supplied.
For multiple choice, final_prediction must be one option label only, with no explanation.
Use public options only to format the supported answer. Never guess an option when evidence
is insufficient. Return final_prediction=null for unknown, partial or ambiguous conclusions.
No later Reader or option-matching model will run. Preserve the normal evidence contract.'''


def provenance(graph, final_node_id):
    """Find an actual eligible route, preserving OR alternatives and AND premises.

    This is program-only provenance collection, not document selection for a model.
    No disputed-document expansion or unrelated search history becomes an answer source.
    """
    nodes = {n['id']: n for n in graph['nodes']}
    spans = {s['id']: s for s in graph['spans']}
    memo = {}

    def routes(nid):
        if nid in memo:
            return memo[nid]
        node = nodes[nid]
        result = []
        if node['status'] == 'supported':
            for alt in node['alternatives']:
                if not alt.get('eligible'):
                    continue
                parents = alt['used_parent_ids']
                parent_routes = [routes(p) for p in parents]
                for chosen in product(*parent_routes):
                    route = {'node_versions': {nid: node['version']},
                             'alternative_ids': {nid: alt['id']},
                             'span_ids': set(alt['source_span_ids'] + alt['guard_span_ids'])}
                    compatible = True
                    for parent in chosen:
                        if any(k in route['alternative_ids'] and route['alternative_ids'][k] != v
                               for k, v in parent['alternative_ids'].items()):
                            compatible = False; break
                        route['node_versions'].update(parent['node_versions'])
                        route['alternative_ids'].update(parent['alternative_ids'])
                        route['span_ids'].update(parent['span_ids'])
                    if compatible:
                        result.append(route)
        memo[nid] = result
        return result

    candidates = routes(final_node_id)
    if not candidates:
        return None
    def key(route):
        return (len({spans[s]['doc_id'] for s in route['span_ids']}), len(route['span_ids']),
                sorted(route['alternative_ids'].items()))
    route = min(candidates, key=key)
    route['span_ids'] = sorted(route['span_ids'])
    route['doc_ids'] = list(dict.fromkeys(spans[s]['doc_id'] for s in route['span_ids']))
    return route
