"""Smoke acceptance binds rows to the actual task, source and deployment."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import pytest
from dagbt import runner
from dagbt.methods import RESIDUAL_V2

spec = importlib.util.spec_from_file_location('v4_release_cli', Path(__file__).resolve().parents[1] / 'scripts/personamem_v4.py')
cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)


@pytest.fixture
def release(tmp_path, monkeypatch):
    root=tmp_path/'release'; out=tmp_path/'output'; root.mkdir()
    ids=root/'ids.json'; runner.save(ids,{'personamem':['q']})
    data=root/'data/personamem/manifest.json'; runner.save(data,{'synthetic':'data'})
    question={'id':'q','options':['A response','B response','C response','D response']}
    source={'synthetic':'sourcehash'}
    monkeypatch.setattr(cli,'ROOT',root); monkeypatch.setattr(cli,'SMOKE_IDS',ids)
    monkeypatch.setattr(cli,'load_questions',lambda root:[question])
    monkeypatch.setattr(runner,'frozen_sources',lambda:source)
    monkeypatch.setattr(runner,'status',lambda output:{'live_writer':False,'progress':{'state':'complete'}})
    # Structural production validation is separately exercised through Engine;
    # this gate fixture isolates release identity and publication conditions.
    monkeypatch.setattr(runner,'validate_result',lambda row,unit:row)
    manifest={'arms':cli.ARMS,'question_ids':{'personamem':['q']},'source_hashes':source,
              'questions_digest':{'personamem':runner.digest([question])},
              'dataset_manifests':{'personamem':runner.file_hash(data)},'config':{}}
    runner.save(out/'manifest.json',manifest)
    base={'unit_id':'q','answer':{'status':'ok','prediction':'a'}}
    runner.save(runner.result_path(out,'personamem','original','q'),base)
    row={**deepcopy(base),'method':RESIDUAL_V2,'algorithm_version':RESIDUAL_V2,
         'diagnostics':{'audit_complete':True,'terminal_input':{'validation_complete':True}}}
    row['answer'].update(completion_kind='semantic_assessment',certificate=None,semantic_answer='A response')
    path=runner.result_path(out,'personamem',RESIDUAL_V2,'q'); runner.save(path,row)
    return out,path,row,manifest


def test_valid_release_gate_passes(release):
    assert cli.check_smoke(release[0])['status']=='passed'


@pytest.mark.parametrize('change',['old_method','fact_certificate','no_audit','wrong_text','invalid_label','failed','incomplete_input'])
def test_release_gate_rejects_invalid_publication(release,change):
    out,path,row,_=release
    if change=='old_method': row['algorithm_version']='dagbt_residual_memory_v1'
    elif change=='fact_certificate': row['answer'].update(completion_kind='certified_symbolic',certificate={})
    elif change=='no_audit': row['diagnostics']['audit_complete']=False
    elif change=='wrong_text': row['answer']['semantic_answer']='Just a fact summary'
    elif change=='invalid_label': row['answer']['prediction']='a or b'
    elif change=='failed': row['answer']['status']='budget_exhausted'
    else: row['diagnostics']['terminal_input']['validation_complete']=False
    runner.save(path,row)
    with pytest.raises(ValueError): cli.check_smoke(out)


def test_release_gate_rejects_code_changed_after_smoke(release):
    out,_,_,manifest=release; manifest['source_hashes']={'synthetic':'different'}
    runner.save(out/'manifest.json',manifest)
    with pytest.raises(ValueError,match='代码'): cli.check_smoke(out)


def test_status_stop_never_guess_experiment_directory(monkeypatch):
    for action in ('status','stop'):
        with pytest.raises(SystemExit) as exc: cli.main([action])
        assert exc.value.code==2


def test_interim_accuracy_uses_paired_rows_and_retains_failed_answers(release, monkeypatch, capsys):
    from dagbt import personamem
    out, _, _, manifest = release
    questions = [{'id': unit, 'options': ['A response', 'B response', 'C response', 'D response']}
                 for unit in ('q', 'r', 's')]
    monkeypatch.setattr(cli, 'load_questions', lambda root: questions)
    monkeypatch.setattr(personamem, 'validate_dataset', lambda *a, **k: None)
    manifest['question_ids']['personamem'] = ['q', 'r', 's']
    manifest['questions_digest']['personamem'] = runner.digest(questions)
    runner.save(out / 'manifest.json', manifest)
    runner.save(cli.ROOT / 'data/personamem/evaluation_only.json',
                [{'id': unit, 'correct_answer': '(a)'} for unit in ('q', 'r', 's')])
    def row(arm, unit, status, prediction):
        runner.save(runner.result_path(out, 'personamem', arm, unit),
                    {'unit_id': unit, 'answer': {'status': status, 'prediction': prediction}})
    row('original', 'q', 'ok', '(a)')
    row('original', 'r', 'planner_failed', None)
    row('original', 's', 'ok', 'a')
    row(RESIDUAL_V2, 'q', 'ok', 'b')
    row(RESIDUAL_V2, 'r', 'ok', '(a)')
    cli.accuracy(out)
    output = capsys.readouterr().out
    assert '已记录 3/3，有效 2，损坏 0，正确 2，准确率 66.67%' in output
    assert '已记录 2/3，有效 2，损坏 0，正确 1，准确率 50.00%' in output
    assert '同题（含失败） 2 题：original 1/2=50.00%' in output
    assert '同题（双方有效） 1 题：original 1/1=100.00%' in output
    assert RESIDUAL_V2 + ' 0/1=0.00%' in output


def test_interim_accuracy_rejects_changed_dataset_manifest_before_reading_labels(release):
    out, _, _, _ = release
    runner.save(cli.ROOT / 'data/personamem/manifest.json', {'synthetic': 'different-data'})
    with pytest.raises(ValueError, match='数据清单'):
        cli.accuracy(out)


def test_explicit_index_preparation_uses_model_host_proxy_exclusions(release, monkeypatch):
    monkeypatch.chdir(cli.ROOT)
    config = cli.ROOT / 'configs/personamem-v4.json'
    runner.save(config, {'llm_base_url': 'http://chat.test/v1',
                         'embedding_base_url': 'http://embedding.test/v1'})
    monkeypatch.setenv('NO_PROXY', 'existing.test')
    monkeypatch.setenv('no_proxy', 'lower.test')
    called = []
    def prepare(args):
        called.append(args)
        assert {'chat.test', 'embedding.test', 'existing.test', 'lower.test'} <= set(cli.os.environ['NO_PROXY'].split(','))
        assert cli.os.environ['no_proxy'] == cli.os.environ['NO_PROXY']
        return 0
    monkeypatch.setattr(runner, 'main', prepare)
    assert cli.main(['prepare-index', '--config', str(config)]) == 0
    assert called == [['prepare-index', '--config', str(config), '--datasets', 'personamem',
                       '--output', str(cli.ROOT / 'outputs/personamem_v4_index')]]


@pytest.mark.parametrize('shared', ['root_link', 'nested_link', 'external_directory'])
def test_explicit_index_preparation_never_writes_a_shared_old_index(release, monkeypatch, tmp_path, shared):
    monkeypatch.chdir(cli.ROOT)
    old = tmp_path / 'old_index'; old.mkdir()
    marker = old / 'existing.npy'; marker.write_bytes(b'preserve existing index')
    derived = cli.ROOT / 'data/derived'
    if shared == 'root_link':
        derived.symlink_to(old, target_is_directory=True)
    elif shared == 'nested_link':
        derived.mkdir(); (derived / 'cache').symlink_to(old, target_is_directory=True)
    config = cli.ROOT / 'configs/personamem-v4.json'
    runner.save(config, {'llm_base_url': 'http://chat.test/v1',
                         'embedding_base_url': 'http://embedding.test/v1',
                         'derived_index_dir': str(old) if shared == 'external_directory' else 'data/derived'})
    called = []
    monkeypatch.setattr(runner, 'main', lambda args: called.append(args))
    with pytest.raises(ValueError, match='共享|符号链接'):
        cli.main(['prepare-index', '--config', str(config)])
    assert not called
    assert marker.read_bytes() == b'preserve existing index'
