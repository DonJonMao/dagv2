"""No-thinking variant of the corpus-wide DAG v1 controller.

Changes relative to frozen_flow_v2.controller:
- Node stage: single non-thinking call (guided JSON output) instead of
  thinking(2048) + sources(512); retrieval stays corpus-wide.
- Reader: non-thinking chat completion via patched reader_input/chat_request
  and a solve() that requests readout 'final_answer_only'.
Everything else (planning, archive, corpus-wide node retrieval, source
selection, budgets) is identical to the v2 fixed run.
"""
import numpy as np
from types import SimpleNamespace
import core as execution
import frozen_flow as flow

PANEL = flow.PANEL
NODE_TOPK = 50  # corpus-wide union window per node query, matches archive top-50 rule
NODE_OUT_TOKENS = 512
READER_OUT_TOKENS = 1024


def controller(row, documents, tokenizer, arm, calls, index):
    pool = row['candidate_doc_ids']
    pool_index = {d: i for i, d in enumerate(pool)}
    with index.lock:
        if not hasattr(index, '_v2_all_ids'):
            index._v2_all_ids = list(index.vectors)
            index._v2_matrix = np.stack([index.vectors[d] for d in index._v2_all_ids])
    all_ids, matrix = index._v2_all_ids, index._v2_matrix
    plan = row['plan'] if arm == 'solve_dag' else {'steps': [
        {'question': row['question'], 'output_slot': 'answer', 'answer_type': 'answer', 'inputs': []}]}
    states, trace, generations, embeddings = {}, [], [], []
    status, error = 'ok', None
    for node_index, step in enumerate(execution.ordered_steps(plan)):
        parents = [states[s] for s in step['inputs']]
        query = execution.ground(step, states)
        payload = {'model': 'nvidia/NV-Embed-v2',
                   'input': ['Instruct: ' + flow.INSTRUCTION + '\nQuery: ' + query]}
        with index.lock:
            embedding = calls.get((arm, 'node', str(node_index), 'embedding'), flow.dense.EMBED_URL, payload)
        embeddings.append(embedding)
        vector = np.asarray(embedding['response']['data'][0]['embedding'], dtype=np.float32)
        scores = matrix @ (vector / np.linalg.norm(vector))
        order = [int(i) for i in np.argsort(-scores, kind='stable')]
        for i in order[:NODE_TOPK]:
            d = all_ids[i]
            if d not in pool_index:
                pool_index[d] = len(pool)
                pool.append(d)
        hits = [{'doc_id': all_ids[i], 'archive_index': pool_index[all_ids[i]],
                 'score': float(scores[i]), 'rank': rank}
                for rank, i in enumerate(order[:PANEL], 1)]
        panel = execution.source_panel(pool, [h['doc_id'] for h in hits], parents, PANEL)
        messages = execution.messages(row['question'], step, query, panel, parents, documents)
        prepared = flow.prepared(messages, tokenizer, False, 0, 0)
        event = {'node_index': node_index, 'output_slot': step['output_slot'], 'query': query,
                 'parents': parents, 'hits': hits, 'panel_doc_ids': panel, 'response_refs': [],
                 'embedding_response_ref': embedding['response_ref']}
        trace.append(event)
        final_payload = {'model': 'qwen3-8b', 'prompt': prepared['native_prompt'], 'temperature': 0,
                         'max_tokens': NODE_OUT_TOKENS,
                         'structured_outputs': {'json': execution.schema(len(panel))},
                         'stop': ['<|im_end|>']}
        final = calls.get((arm, 'node', str(node_index), 'sources'), flow.cross.COMPLETION_URL, final_payload)
        generations.append(final)
        event['response_refs'].append(final['response_ref'])
        choice = final['response']['choices'][0]
        event['thinking_finish_reason'] = 'not_applicable'
        event['final_finish_reason'] = choice['finish_reason']
        try:
            if choice['finish_reason'] != 'stop':
                raise execution.NodeOutputError('node_finish=' + choice['finish_reason'])
            answer, sources = execution.decode(choice['text'], panel)
        except execution.NodeOutputError as failure:
            status, error = 'node_failed', str(failure)
            event.update(status=status, error=error)
            break
        state = execution.node_state(step, query, answer, sources, parents, pool)
        states[step['output_slot']] = state
        event.update(status='ok', state=state)
    return {'status': status, 'error': error, 'nodes': list(states.values()), 'trace': trace,
            'logical_requests': len(generations), 'response_usage': flow.cross.tokens(generations),
            'embedding_requests': len(embeddings),
            'embedding_usage': {'prompt_tokens': sum(c['response']['usage']['prompt_tokens'] for c in embeddings)}}


def reader_input(row, ids, documents, tokenizer, nodes):
    evidence = [SimpleNamespace(
        demand_id=n['output_slot'], state_kind='resolved', state_label=n['grounded_question'],
        state_value=n['answer'], source_doc_id=(n['source_doc_ids'][0] if n['source_doc_ids'] else ''),
        source_span='') for n in nodes if n['resolved']]
    messages = flow.reader.reader_messages(question=row['question'], selected_doc_ids=ids,
        documents=documents, grounded_spans=(), lineage_evidence=evidence, required_count=0)
    if evidence:
        messages[-1]['content'] += (
            "\n\nThe committed demand-state evidence above is the intermediate answer chain "
            "already established for this question, in dependency order; its final state "
            "answers the question. Start from it: give a final answer consistent with the "
            "chain's final state, after checking it against the context passages. Only if "
            "the passages clearly contradict the chain, answer from the passages instead.")
    return flow.prepared(messages, tokenizer, False, 0, 0)


def chat_request(messages, thinking):
    return {'model': 'qwen3-8b', 'messages': messages, 'temperature': 0,
            'max_tokens': 2048 if thinking else READER_OUT_TOKENS,
            'chat_template_kwargs': {'enable_thinking': thinking}}


def solve(row, documents, tokenizer, index, calls):
    ranking = flow.controller(row, documents, tokenizer, 'solve_dag', calls, index)
    if ranking['status'] == 'ok':
        selections = {str(k): execution.select_sources(row['candidate_doc_ids'], ranking['nodes'], k)
                      for k in (5, 10, 20)}
        prepared = reader_input(row, selections['20']['selected_doc_ids'], documents, tokenizer,
                                ranking['nodes'])
        def get(stage, url, payload):
            return calls.get(('solve_dag', 'finalized_thinking', '20', stage), url, payload)
        answer = flow.evaluate(prepared, 'final_answer_only', get, None)
    else:
        selections = {str(k): {'selected_doc_ids': [], 'retained_node_proofs': 0,
                               'available_node_proofs': 0} for k in (5, 10, 20)}
        answer = {'status': 'node_failed', 'prediction': '', 'error': ranking['error']}
    return {'unit_id': row['unit_id'], 'ranking': ranking, 'budgets': selections, 'answer': answer}
