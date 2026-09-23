from __future__ import annotations
import numpy as np
from types import SimpleNamespace
import core as execution
import reader
PANEL=20
THINK_TOKENS=2048
FINAL_TOKENS=512
TOKENS=('prompt_tokens','completion_tokens','total_tokens')
CHAT_URL=''
COMPLETION_URL=''
INSTRUCTION='Given a question, retrieve relevant documents that best answer the question.'
def score_answer(out, answers):
    return out  # Scoring is a separate command; generation never loads labels.


def chat_request(messages, thinking):
    return {'model': 'qwen3-8b', 'messages': messages, 'temperature': 0,
        'max_tokens': 2048 if thinking else 128, 'chat_template_kwargs': {'enable_thinking': thinking}}

def continuation(prompt, raw):
    prefix = raw if '</think>' in raw else raw + '\n</think>\n\n'
    return prefix, {'model': 'qwen3-8b', 'prompt': prompt + prefix, 'temperature': 0,
        'max_tokens': 128, 'stop': ['<|im_end|>']}

def parsed_answer(raw, reason, usage, thinking):
    out = {'status': 'reader_failed', 'prediction': '', 'error': None,
        'finish_reason': reason, 'response_usage': usage}
    if reason != 'stop':
        out['error'] = 'finish_reason=' + reason
    elif thinking and '</think>' not in raw:
        out['error'] = 'missing_thinking_end'
    else:
        final = raw.rsplit('</think>', 1)[1] if thinking else raw
        prediction = reader.parse_answer(final)
        if prediction:
            out.update(status='ok', prediction=prediction)
        else:
            out['error'] = 'empty_final_answer'
    return out

def tokens(calls):
    return {key: sum(c['response']['usage'][key] for c in calls) for key in TOKENS}

def evaluate(prepared, readout, get_call, answers):
    thinking = readout == 'finalized_thinking'
    first = get_call('first', CHAT_URL, chat_request(prepared['messages'], thinking))
    response = first['response']
    choice = response['choices'][0]
    raw, reason, usage = choice['message']['content'], choice['finish_reason'], response['usage']
    first_outcome = score_answer(parsed_answer(raw, reason, usage, thinking), answers)
    continued = thinking and reason == 'length'
    out = dict(first_outcome)
    refs = [first['response_ref']]
    if continued:
        prefix, payload = continuation(prepared['native_prompt'], raw)
        second = get_call('finalization', COMPLETION_URL, payload)
        c = second['response']['choices'][0]
        usage = {key: usage[key] + second['response']['usage'][key] for key in TOKENS}
        out = score_answer(parsed_answer(prefix + c['text'], c['finish_reason'], usage, True), answers)
        refs.append(second['response_ref'])
    out.update(continued_after_length=continued, response_refs=refs, first_stage=first_outcome)
    return out

def prepared(messages, tokenizer, thinking, first_tokens, final_tokens=0):
    native = tokenizer.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=thinking)
    count = len(tokenizer.encode(native, add_special_tokens=False))
    assert count + first_tokens + final_tokens + 8 <= 16384, ('context_budget', count)
    return {'messages': messages, 'native_prompt': native, 'native_prompt_tokens_local': count}

def reader_input(row, ids, documents, tokenizer):
    messages = cross.reader.reader_messages(question=row['question'], selected_doc_ids=ids,
        documents=documents, grounded_spans=(), lineage_evidence=(), required_count=0)
    return prepared(messages, tokenizer, True, 2048, 128)

def controller(row, documents, tokenizer, arm, calls, index):
    pool = row['candidate_doc_ids']
    matrix = np.stack([index.vectors[d] for d in pool])
    plan = row['plan'] if arm == 'solve_dag' else {'steps': [
        {'question': row['question'], 'output_slot': 'answer', 'answer_type': 'answer', 'inputs': []}]}
    states, trace, generations, embeddings = {}, [], [], []
    status, error = 'ok', None
    for node_index, step in enumerate(execution.ordered_steps(plan)):
        parents = [states[s] for s in step['inputs']]
        query = execution.ground(step, states)
        payload = {'model': 'nvidia/NV-Embed-v2',
                   'input': [f'Instruct: {dense.INSTRUCTION}\nQuery: {query}']}
        with index.lock:
            embedding = calls.get((arm, 'node', str(node_index), 'embedding'), dense.EMBED_URL, payload)
        embeddings.append(embedding)
        vector = np.asarray(embedding['response']['data'][0]['embedding'], dtype=np.float32)
        scores = matrix @ (vector / np.linalg.norm(vector))
        order = [int(i) for i in np.argsort(-scores, kind='stable')[:PANEL]]
        hits = [{'doc_id': pool[i], 'archive_index': i, 'score': float(scores[i]), 'rank': rank}
                for rank, i in enumerate(order, 1)]
        panel = execution.source_panel(pool, [h['doc_id'] for h in hits], parents, PANEL)
        messages = execution.messages(row['question'], step, query, panel, parents, documents)
        prepared = archive.prepared(messages, tokenizer, True, THINK_TOKENS, FINAL_TOKENS)
        event = {'node_index': node_index, 'output_slot': step['output_slot'], 'query': query,
                 'parents': parents, 'hits': hits, 'panel_doc_ids': panel, 'response_refs': [],
                 'embedding_response_ref': embedding['response_ref']}
        trace.append(event)
        first = calls.get((arm, 'node', str(node_index), 'thinking'), cross.COMPLETION_URL,
            {'model': 'qwen3-8b', 'prompt': prepared['native_prompt'], 'temperature': 0,
             'max_tokens': THINK_TOKENS, 'stop': ['</think>']})
        generations.append(first)
        event['response_refs'].append(first['response_ref'])
        choice = first['response']['choices'][0]
        event['thinking_finish_reason'] = choice['finish_reason']
        _, final_payload = cross.continuation(prepared['native_prompt'], choice['text'])
        final_payload.update(max_tokens=FINAL_TOKENS, structured_outputs={'json': execution.schema(len(panel))})
        final = calls.get((arm, 'node', str(node_index), 'sources'), cross.COMPLETION_URL, final_payload)
        generations.append(final)
        event['response_refs'].append(final['response_ref'])
        choice = final['response']['choices'][0]
        event['final_finish_reason'] = choice['finish_reason']
        try:
            if event['thinking_finish_reason'] not in ('stop', 'length') or choice['finish_reason'] != 'stop':
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
            'logical_requests': len(generations), 'response_usage': cross.tokens(generations),
            'embedding_requests': len(embeddings),
            'embedding_usage': {'prompt_tokens': sum(c['response']['usage']['prompt_tokens'] for c in embeddings)}}

def configure(chat_url, completion_url, embedding_url):
    global CHAT_URL, COMPLETION_URL, cross, archive, dense
    CHAT_URL, COMPLETION_URL = chat_url, completion_url
    import sys
    cross = sys.modules[__name__]
    archive = cross
    dense = SimpleNamespace(EMBED_URL=embedding_url, INSTRUCTION=INSTRUCTION)
