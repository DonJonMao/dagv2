"""One optional binding clarification, followed by the unchanged single-value DAG.

Candidate detection replaces one ordinary bridge-node call. Only an observed,
source-anchored competition can spend the two retrievals and one joint judgment.
No speculative children, rollback, extra planner, or new source selector.
"""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from package import core
from . import protocol


PANEL = 20
TOPK = 50
NODE_TOKENS = 512
PROPOSAL_TOKENS = 1024
JUDGE_TOKENS = 1024
READER_TOKENS = 1024
CONTEXT_TOKENS = 16384
CONTEXT_MARGIN = 8
ANCHOR_LIMIT = 4


class ContextBudgetError(ValueError):
    pass


def _prepare(make_messages, panel, tokenizer, output_tokens, protected=()):
    """Remove whole, unprotected documents; never truncate a cited quotation."""
    panel = list(dict.fromkeys(panel))
    original = list(panel)
    protected = set(protected)
    while True:
        messages = make_messages(panel)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        count = len(tokenizer.encode(prompt, add_special_tokens=False))
        if count + output_tokens + CONTEXT_MARGIN <= CONTEXT_TOKENS:
            return {'messages': messages, 'native_prompt': prompt,
                    'native_prompt_tokens_local': count}, panel, [d for d in original if d not in panel]
        removable = next((d for d in reversed(panel) if d not in protected), None)
        if removable is None:
            raise ContextBudgetError('full protected passages exceed context/output capacity')
        panel.remove(removable)


def _matrix(index, documents):
    with index.lock:
        if not hasattr(index, '_resolve_matrix'):
            ids = list(index.vectors)
            if not ids or len(set(ids)) != len(ids) or set(ids) != set(documents):
                raise ValueError('invalid corpus vector identifiers')
            matrix = np.asarray(np.stack([index.vectors[d] for d in ids]), dtype=np.float32)
            if (matrix.ndim != 2 or matrix.shape[1] == 0 or not np.isfinite(matrix).all()
                    or not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=2e-5)):
                raise ValueError('corpus vectors must be finite normalized rows')
            index._resolve_ids, index._resolve_matrix = ids, matrix
    return index._resolve_ids, index._resolve_matrix


def _embedding(response, dimension):
    data = response.get('data') if isinstance(response, dict) else None
    if (not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict)
            or type(data[0].get('index', 0)) is not int or data[0].get('index', 0) != 0):
        raise ValueError('query embedding must have exactly one index-zero result')
    values = data[0].get('embedding')
    if not isinstance(values, list) or any(type(v) not in (int, float) for v in values):
        raise ValueError('query embedding must contain numeric values')
    vector = np.asarray(values, dtype=np.float32)
    if vector.shape != (dimension,) or not np.isfinite(vector).all():
        raise ValueError('query embedding shape or finite-value contract')
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError('query embedding is zero or nonfinite')
    return vector / norm


def _choice(record, *, chat=False):
    response = record.get('response') if isinstance(record, dict) else None
    choices = response.get('choices') if isinstance(response, dict) else None
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError('one completion choice is required')
    choice = choices[0]
    if choice.get('finish_reason') != 'stop':
        raise ValueError('finish_reason=' + str(choice.get('finish_reason')))
    value = choice.get('message', {}).get('content') if chat and isinstance(choice.get('message'), dict) else None if chat else choice.get('text')
    if not isinstance(value, str):
        raise ValueError('completion text contract')
    return value


def _usage(records, *, embedding=False):
    keys = ('prompt_tokens', 'total_tokens') if embedding else ('prompt_tokens', 'completion_tokens', 'total_tokens')
    result = {key: 0 for key in keys}
    complete = True
    for record in records:
        usage = record.get('response', {}).get('usage')
        if not isinstance(usage, dict):
            complete = False
            continue
        for key in keys:
            value = usage.get(key)
            if type(value) is int and value >= 0:
                result[key] += value
            else:
                complete = False
    return result, complete


def _query_key(query):
    return ' '.join(query.split()).casefold()


def _anchors(proposal, initial_panel):
    required = set(proposal['sources'])
    for candidate in proposal['candidates']:
        required.update(e['doc_id'] for e in candidate['evidence'])
        required.update(e['doc_id'] for a in candidate['assessments'] for e in a['evidence'])
    return [d for d in initial_panel if d in required]


def _joint_panel(anchors, sides, initial):
    panel = list(anchors)
    seen = set(panel)
    sides = [[d for d in dict.fromkeys(side) if d not in seen] for side in sides]
    rounds, fillers = [], []
    # Match the per-candidate rank window before deduplicating shared documents.
    quota = (PANEL - len(anchors)) // 2
    for rank in range(quota):
        group = []
        for side in sides:
            if rank < len(side) and side[rank] not in seen:
                panel.append(side[rank])
                group.append(side[rank])
                seen.add(side[rank])
        rounds.append(group)
    for doc_id in initial:
        if len(panel) >= PANEL:
            break
        if doc_id not in seen:
            panel.append(doc_id)
            fillers.append(doc_id)
            seen.add(doc_id)
            if len(panel) == PANEL:
                break
    return panel, rounds, fillers


def _prepare_joint(make_messages, anchors, sides, initial, tokenizer):
    panel, rounds, fillers = _joint_panel(anchors, sides, initial)
    original = list(panel)
    while True:
        try:
            prepared, _, _ = _prepare(make_messages, panel, tokenizer, JUDGE_TOKENS, panel)
            return prepared, panel, [d for d in original if d not in panel]
        except ContextBudgetError:
            # Remove neutral archive fill first; then shorten BOTH query windows
            # by one rank at a time. Protected binding evidence is never removed.
            if fillers:
                panel.remove(fillers.pop())
            elif rounds:
                for doc_id in rounds.pop():
                    panel.remove(doc_id)
            else:
                raise


def _reader_messages(row, ids, documents, nodes, runtime):
    """Origin v6 Reader message construction; only capacity preparation differs.

    Kept separate from the frozen helper, whose preparation reserves no output.
    Integration tests compare these messages with that helper byte for byte.
    """
    evidence = [SimpleNamespace(
        demand_id=n['output_slot'], state_kind='resolved', state_label=n['grounded_question'],
        state_value=n['answer'], source_doc_id=(n['source_doc_ids'][0] if n['source_doc_ids'] else ''),
        source_span='') for n in nodes if n['resolved']]
    messages = runtime.flow.reader.reader_messages(question=row['question'], selected_doc_ids=ids,
        documents=documents, grounded_spans=(), lineage_evidence=evidence, required_count=0)
    if evidence:
        messages[-1]['content'] += (
            "\n\nThe committed demand-state evidence above is the intermediate answer chain "
            "already established for this question, in dependency order; its final state "
            "answers the question. Start from it: give a final answer consistent with the "
            "chain's final state, after checking it against the context passages. Only if "
            "the passages clearly contradict the chain, answer from the passages instead.")
    return messages


def controller(row, documents, tokenizer, index, calls, runtime):
    steps = core.ordered_steps(row['plan'])
    if not 1 <= len(steps) <= 6 or len({s['output_slot'] for s in steps}) != len(steps):
        raise ValueError('requires one to six unique DAG nodes')
    pool = list(row['candidate_doc_ids'])
    if not pool or len(set(pool)) != len(pool) or not set(pool) <= set(documents):
        raise ValueError('invalid initial archive')
    all_ids, matrix = _matrix(index, documents)
    pool_index = {d: i for i, d in enumerate(pool)}
    states, trace, generations, embeddings = {}, [], [], []
    llm_requests, embedding_requests = 0, 0
    status, error = 'ok', None
    requested = {_query_key(item['query']) for item in row.get('archive_trace', []) if isinstance(item, dict) and isinstance(item.get('query'), str)}
    parent_slots = {slot for step in steps for slot in step['inputs']}
    proposal_index = next((i for i, step in enumerate(steps) if step['output_slot'] in parent_slots), None)
    proposal_count, clarify_count = 0, 0

    def retrieve(query, stage):
        nonlocal embedding_requests
        if embedding_requests >= len(steps) + 2:
            raise ValueError('embedding request budget exhausted')
        embedding_requests += 1
        requested.add(_query_key(query))
        payload = {'model': 'nvidia/NV-Embed-v2',
                   'input': ['Instruct: ' + runtime.flow.INSTRUCTION + '\nQuery: ' + query]}
        with index.lock:
            record = calls.get(stage, runtime.flow.dense.EMBED_URL, payload)
        embeddings.append(record)
        vector = _embedding(record['response'], matrix.shape[1])
        scores = matrix @ vector
        if not np.isfinite(scores).all():
            raise ValueError('nonfinite corpus retrieval scores')
        order = np.argsort(-scores, kind='stable')[:TOPK]
        for i in order:
            doc_id = all_ids[int(i)]
            if doc_id not in pool_index:
                pool_index[doc_id] = len(pool)
                pool.append(doc_id)
        hits = [{'doc_id': all_ids[int(i)], 'archive_index': pool_index[all_ids[int(i)]],
                 'score': float(scores[int(i)]), 'rank': rank}
                for rank, i in enumerate(order, 1)]
        return hits, record.get('response_ref')

    def generate(prepared, schema, output_tokens, stage):
        nonlocal llm_requests
        if llm_requests >= len(steps) + 1:
            raise ValueError('node/judge request budget exhausted')
        llm_requests += 1
        record = calls.get(stage, runtime.flow.cross.COMPLETION_URL,
            {'model': 'qwen3-8b', 'prompt': prepared['native_prompt'], 'temperature': 0,
             'max_tokens': output_tokens, 'structured_outputs': {'json': schema},
             'stop': ['<|im_end|>']})
        generations.append(record)
        return record

    for node_index, step in enumerate(steps):
        parents = [states[slot] for slot in step['inputs']]
        query = core.ground(step, states)
        event = {'node_index': node_index, 'output_slot': step['output_slot'], 'query': query,
                 'parents': deepcopy(parents), 'response_refs': []}
        trace.append(event)
        try:
            hits, ref = retrieve(query, ('resolve', 'node', str(node_index), 'embedding'))
            event.update(hits=hits[:PANEL], embedding_response_ref=ref)
            panel = core.source_panel(pool, [h['doc_id'] for h in hits[:PANEL]], parents, PANEL)
            is_proposal = node_index == proposal_index
            output_tokens = PROPOSAL_TOKENS if is_proposal else NODE_TOKENS
            message_fn = protocol.proposal_messages if is_proposal else core.messages
            inherited = {d for p in parents if p['resolved'] for d in p['source_doc_ids']}
            prepared, panel, removed = _prepare(
                lambda ids: message_fn(row['question'], step, query, ids, parents, documents),
                panel, tokenizer, output_tokens, inherited)
            event.update(panel_doc_ids=panel, context_removed_doc_ids=removed,
                         prompt_tokens_local=prepared['native_prompt_tokens_local'],
                         output_tokens_reserved=output_tokens, proposal_node=is_proposal)
            if is_proposal:
                proposal_count += 1
            schema = protocol.proposal_schema(len(panel)) if is_proposal else core.schema(len(panel))
            record = generate(prepared, schema, output_tokens, ('resolve', 'node', str(node_index), 'sources'))
            event['response_refs'].append(record.get('response_ref'))
            raw = _choice(record)
            event.update(thinking_finish_reason='not_applicable', final_finish_reason='stop')
            if is_proposal:
                proposal = protocol.decode_proposal(raw, panel, documents, row['question'])
                event['proposal'] = proposal
                result = protocol.adopt(proposal, None)
                clarify = {'status': 'not_triggered', 'reason': proposal['trigger']['reason'],
                           'queries': [], 'retrievals': [], 'response_refs': []}
                event['clarify'] = clarify
                if proposal['trigger']['eligible']:
                    queries = protocol.binding_queries(proposal)
                    anchors = _anchors(proposal, panel)
                    remaining = len(steps) - node_index - 1
                    # The final Reader and all ordinary descendants are reserved.
                    reserved_llm = llm_requests + 1 + remaining + 1
                    reserved_embed = embedding_requests + 2 + remaining
                    if len(anchors) > ANCHOR_LIMIT:
                        clarify['reason'] = 'anchor_capacity'
                    elif (len(queries) != 2 or len({_query_key(q) for q in queries}) != 2
                          or any(_query_key(q) in requested for q in queries)):
                        clarify['reason'] = 'duplicate_or_missing_query'
                    elif reserved_llm > len(steps) + 2 or reserved_embed > len(steps) + 2:
                        clarify['reason'] = 'reserved_downstream_budget'
                    else:
                        try:
                            # Do not spend retrievals if even the protected raw evidence cannot fit.
                            _prepare(lambda ids: protocol.judge_messages(row['question'], step, proposal, ids, documents),
                                     anchors, tokenizer, JUDGE_TOKENS, anchors)
                        except ContextBudgetError:
                            clarify['reason'] = 'protected_context_capacity'
                        else:
                            clarify_count += 1
                            clarify.update(status='started', reason='binding_disagreement', queries=queries,
                                reserved_remaining_node_calls=remaining, reserved_reader_calls=1)
                            try:
                                sides = []
                                for i, binding_query in enumerate(queries):
                                    clarify['active_stage'] = 'embedding'
                                    clarify['active_query'] = binding_query
                                    new_hits, new_ref = retrieve(binding_query, ('resolve', 'clarify', str(i), 'embedding'))
                                    clarify['retrievals'].append({'query': binding_query, 'hits': new_hits,
                                                                  'response_ref': new_ref})
                                    sides.append([h['doc_id'] for h in new_hits])
                                prepared, joint, removed = _prepare_joint(
                                    lambda ids: protocol.judge_messages(row['question'], step, proposal, ids, documents),
                                    anchors, sides, panel, tokenizer)
                                clarify.update(panel_doc_ids=joint, context_removed_doc_ids=removed,
                                    protected_doc_ids=anchors, prompt_tokens_local=prepared['native_prompt_tokens_local'],
                                    output_tokens_reserved=JUDGE_TOKENS)
                                clarify['active_stage'] = 'judge'
                                judged = generate(prepared, protocol.judge_schema(proposal, len(joint)), JUDGE_TOKENS,
                                                  ('resolve', 'clarify', 'judge'))
                                clarify['response_refs'].append(judged.get('response_ref'))
                                judgment = protocol.decode_judgment(_choice(judged), proposal, joint, documents)
                                result = protocol.adopt(proposal, judgment)
                                clarify.update(status='judged', judgment=judgment)
                            except Exception as failure:
                                # Optional retrieval/judgment failure cannot fabricate contradiction.
                                judgment = protocol.decode_judgment('', proposal, panel, documents)
                                result = protocol.adopt(proposal, judgment)
                                clarify.update(status='failed', error_type=type(failure).__name__,
                                               reason='clarification_failed', judgment=judgment)
                event['binding_decision'] = result
                answer, sources = result['answer'], result['sources']
            else:
                answer, sources = core.decode(raw, panel)
            state = core.node_state(step, query, answer, sources, parents, pool)
            states[step['output_slot']] = state
            event.update(status='ok', state=state)
        except Exception as failure:
            status, error = 'node_failed', type(failure).__name__ + ': ' + str(failure)
            event.update(status=status, error=error)
            break
    usage, complete = _usage(generations)
    embedding_usage, embedding_complete = _usage(embeddings, embedding=True)
    return {'status': status, 'error': error, 'nodes': list(states.values()), 'trace': trace,
            'pool_doc_ids': pool, 'logical_requests': llm_requests, 'response_usage': usage,
            'response_usage_complete': complete and len(generations) == llm_requests,
            'embedding_requests': embedding_requests, 'embedding_usage': embedding_usage,
            'embedding_usage_complete': embedding_complete and len(embeddings) == embedding_requests,
            'proposal_count': proposal_count, 'clarify_count': clarify_count,
            'solver_limits': {'node_and_judge_llm': len(steps) + 1, 'embedding': len(steps) + 2,
                              'reader_reserved': 1}}


def solve(row, documents, tokenizer, index, calls, runtime=None):
    if runtime is None:
        from .runtime import import_originals
        runtime = import_originals('hotpotqa')
    ranking = controller(row, documents, tokenizer, index, calls, runtime)
    if ranking['status'] != 'ok':
        selections = {str(k): {'selected_doc_ids': [], 'retained_node_proofs': 0,
                               'available_node_proofs': 0} for k in (5, 10, 20)}
        answer = {'status': 'node_failed', 'prediction': '', 'error': ranking['error']}
    else:
        selections = {str(k): core.select_sources(ranking['pool_doc_ids'], ranking['nodes'], k)
                      for k in (5, 10, 20)}
        selected = selections['20']['selected_doc_ids']
        protected = {d for n in ranking['nodes'] if n['resolved'] and set(n['closure_doc_ids']) <= set(selected)
                     for d in n['closure_doc_ids']}
        reader_event = {'response_refs': [], 'output_tokens_reserved': READER_TOKENS}
        ranking['reader'] = reader_event
        try:
            # Preserve origin text, including its known unknown-primary bias.
            prepared, visible, removed = _prepare(
                lambda ids: _reader_messages(row, ids, documents, ranking['nodes'], runtime),
                selected, tokenizer, READER_TOKENS, protected)
            reader_event.update(panel_doc_ids=visible, context_removed_doc_ids=removed,
                                prompt_tokens_local=prepared['native_prompt_tokens_local'])
            record = calls.get(('reader', 'resolve', 'first'), runtime.flow.CHAT_URL,
                               runtime.v6.chat_request(prepared['messages'], False))
            reader_event['response_refs'].append(record.get('response_ref'))
            raw = _choice(record, chat=True)
            usage, complete = _usage([record])
            answer = runtime.flow.parsed_answer(raw, 'stop', usage, False)
            answer.update(response_refs=reader_event['response_refs'], continued_after_length=False,
                          response_usage_complete=complete)
            reader_event['status'] = answer['status']
        except Exception as failure:
            reader_event.update(status='reader_failed', error_type=type(failure).__name__)
            answer = {'status': 'reader_failed', 'prediction': '',
                      'error': type(failure).__name__ + ': ' + str(failure),
                      'response_refs': reader_event['response_refs']}
    return {'unit_id': row['unit_id'], 'ranking': ranking, 'budgets': selections, 'answer': answer}
