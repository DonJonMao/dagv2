"""New result, APC and complete synthetic-flow integration checks."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import demo_local_terminal as demo
import benchmark_bt_reranker_prefix_cache as apc
from dagbt import runner
from dagbt.local_terminal import VERSION
from dagbt.methods import LOCAL, UNSCORED


def test_source_driven_director_university_city_executes_real_BT():
    row, requests = demo.run_demo()
    assert row['answer']['prediction'] == 'River City'
    traces = [d['trace'] for d in row['ranking']['trace']]
    assert [t['node_id'] for t in traces] == ['director','university','city']
    assert 'Lin Zhou' in traces[1]['scoring_query']
    assert 'North University' in traces[2]['scoring_query']
    measurement = next(a for a in traces[1]['search_archive']['activations']
                       if a['target_id']=='interview' and set(a['group_ids'])=={'class','school'} and a['premise_ids']==[])
    assert measurement['target_marginal_after'] == pytest.approx(.85)
    assert measurement['activation'] == pytest.approx(.75)
    assert measurement['accepted']
    assert set(row['answer']['sources']['doc_ids']) == set(demo.PASSAGES)
    assert sum(r['stage'][0]=='resolve' for r in requests) == 3
    assert all(r['stage'][0] not in ('select','reader') for r in requests)
    for r in requests:
        if r['stage'][:2] == ['dagbt','set_reranker']:
            assert r['payload']['query'] in {t['scoring_query'] for t in traces}
    cost = row['diagnostics']['cost']
    assert cost['node_resolve_calls'] == 3
    assert cost['independent_reader_calls'] == cost['global_final_selector_calls'] == 0
    assert cost['ann_calls'] <= 36 and cost['set_score_calls'] <= 512


@pytest.mark.parametrize('method', [LOCAL, UNSCORED])
def test_personamem_options_are_only_at_terminal_without_reader_instruction(method):
    from dagbt.engine import Engine
    from dagbt.personamem import ANSWER_INSTRUCTION, TERMINAL_CHOICE_INSTRUCTION, format_question, parse_choice
    question = 'Which city contains the university from which the director of Ash Wind (2012) graduated?'
    options = ['River City', 'Lake City', 'Hill City', 'Forest City']
    full_question = format_question(question, options)

    class ChoiceServices(demo.SourcesOnlyServices):
        def get(self, stage, url, payload):
            result = super().get(stage, url, payload)
            if stage[0] == 'resolve':
                data = json.loads(payload['messages'][1]['content'])
                if data['node_id'] == 'city':
                    assert data['public_options'] == options
                    assert data['public_output_question'] == question
                    value = json.loads(result['response']['choices'][0]['message']['content'])
                    if value['status'] == 'supported':
                        assert value['answer'] == options[0]
                        value['final_prediction'] = '(a)'
                        result['response']['choices'][0]['message']['content'] = json.dumps(value)
            return result

    docs = {d: SimpleNamespace(doc_id=d, title=d, text=text, passage=d+'\n'+text)
            for d, text in demo.PASSAGES.items()}
    vectors = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,.99,.01],[0,0,0,1]], dtype=np.float32)
    services = ChoiceServices()
    config = {'_test_transport': True, 'llm_base_url': 'http://synthetic/v1', 'llm_model': 'source-script',
              'embedding_base_url': 'http://synthetic/v1', 'embedding_model': 'source-script',
              'fusion': {'initial_width': 1, 'proposal_width': 2, 'pair_rescue_width': 2,
                         'max_feedback_rounds': 0, 'refinement': False}}
    if method == LOCAL:
        config['reranker'] = {'url': 'http://synthetic/rerank', 'model': 'source-script', 'batch_size': 4}
    engine = Engine({'id': 'personamem-synthetic-city', 'question': question},
                    (docs, list(docs), vectors, SimpleNamespace(), demo.Tokenizer()), services, config, method,
                    reader_question=question, output_options=options)
    row = engine.run()
    assert row['answer']['status'] == 'ok' and parse_choice(row['answer']['prediction'], options) == 'a'
    terminal = []
    for request in services.requests:
        payload = request['payload']
        data = json.loads(payload['messages'][1]['content']) if 'messages' in payload else None
        if request['stage'][0] == 'resolve' and data['node_id'] == engine.final_node_id:
            assert data['public_output_question'] == question and data['public_options'] == options
            assert data['original_question'] == question
            assert TERMINAL_CHOICE_INSTRUCTION in payload['messages'][0]['content']
            terminal.append(request)
        else:
            wire = json.dumps(payload, ensure_ascii=False)
            assert 'public_options' not in wire
        wire = json.dumps(payload, ensure_ascii=False)
        assert ANSWER_INSTRUCTION not in wire and 'Candidate answers:' not in wire
        assert full_question not in wire
    assert terminal and engine.output_options == options
    assert json.loads(services.requests[0]['payload']['messages'][1]['content']) == question
    assert not any(r['stage'][0] in ('reader', 'select') for r in services.requests)


def test_new_results_score_without_fabricating_reader_or_k_metrics(tmp_path):
    row,_ = demo.run_demo()
    unit = row['unit_id']
    assert runner.validate_result(row,unit) is row
    label={'id':unit,'answers':['River City'],'gold_groups':[['city']]}
    modules = runner.module_metrics(row,label,'hotpotqa',tmp_path)
    assert modules['legacy_reader_metrics'] == 'not_applicable'
    assert modules['reader_prompt_tokens_local'] is None
    assert modules['selected_doc_count_at20'] is None
    assert modules['gold_terminal_source_title_group_recall'] == 1
    runner.save(runner.result_path(tmp_path,'hotpotqa',VERSION,unit),row)
    summary = runner.score_all(tmp_path,{'hotpotqa':[{'id':unit,'question':'Synthetic question'}]},[VERSION],
                               label_loader=lambda dataset:[label])
    metrics=summary['hotpotqa']['arms'][VERSION]['all_task_metrics_percent']
    assert metrics == {'f1':100.,'em':100.}
    scores=runner.load(tmp_path/'hotpotqa'/VERSION/'scores.json')
    assert scores[0]['legacy_selection_metrics']=='not_applicable'
    assert 'r@20' not in scores[0]


def test_terminal_failures_remain_readable_to_runner_and_export(tmp_path):
    import export_results
    row=runner.failure_row({'id':'q'},RuntimeError('synthetic failure'),VERSION)
    row['answer']['prediction']=None
    assert runner.validate_result(row,'q') is row
    path=tmp_path/'result.json';runner.save(path,row)
    assert export_results.read_answer(path)['prediction'] is None


def test_APC_context_costs_are_aggregated_once_per_namespace(tmp_path):
    events=[{'event':'bridge_discovery_completed','trace':{'scoring_context_id':context,
        'scorer_cost_cumulative':{'memory_cache_hits':hits,'persistent_cache_hits':0,
                                'scored_sets':sets,'logical_input_tokens_estimate':sets*10}}}
            for context,hits,sets in [('one',1,4),('two',2,5),('one',3,6)]]
    path=tmp_path/'events.jsonl';path.write_text(''.join(json.dumps(e)+'\n' for e in events))
    _,cost=apc.scorer_metadata(path)
    assert cost=={'memory_cache_hits':5,'persistent_cache_hits':0,'scored_sets':11,'logical_input_tokens_estimate':110}


def test_APC_score_identity_separates_same_payload_context_and_algorithm():
    p={'query':'same task','documents':['same raw passage']}
    legacy=apc.score_bank_key(p,{})
    one=apc.score_bank_key(p,{'algorithm_version':VERSION,'scoring_context_id':'one'})
    two=apc.score_bank_key(p,{'algorithm_version':VERSION,'scoring_context_id':'two'})
    assert len({legacy,one,two})==3
    with pytest.raises(ValueError,match='context missing'):
        apc.score_bank_key(p,{'algorithm_version':VERSION})


def test_APC_refuses_cross_algorithm_trajectory_comparison(tmp_path):
    for name,version in [('off',apc.LEGACY_ALGORITHM),('on',VERSION)]:
        directory=tmp_path/name;directory.mkdir()
        (directory/'trajectory.json').write_text(json.dumps({'algorithm_version':version}))
    with pytest.raises(ValueError,match='Cross-algorithm'):
        apc.compare_search(tmp_path/'off',tmp_path/'on',tmp_path/'comparison')
