#!/usr/bin/env python3
"""One-command PersonaMem v4 smoke, gate, full run, status and interim scoring."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dagbt import runner
from dagbt.methods import RESIDUAL_V2
from dagbt.personamem import load_questions, parse_choice

ARMS = ['original', RESIDUAL_V2]
SMOKE_IDS = ROOT / 'configs/personamem-v4-smoke-ids.json'


def check_smoke(output, config=None):
    output = Path(output).resolve()
    status = runner.status(output)
    if status['live_writer'] or (status.get('progress') or {}).get('state') != 'complete':
        raise ValueError('Smoke 尚未全部成功结束；查看 status / launcher.log，不要启动全量。')
    manifest = runner.load(output / 'manifest.json')
    if manifest.get('arms') != ARMS or manifest.get('question_ids') != runner.load(SMOKE_IDS):
        raise ValueError('Smoke 的方法或 14 道固定公开题目不匹配')
    if manifest.get('source_hashes') != runner.frozen_sources():
        raise ValueError('代码在 smoke 后发生变化，必须使用新目录重新测试')
    if config is not None and manifest.get('config') != runner.validate_config(runner.load(config)):
        raise ValueError('全量配置与 smoke 配置不同，必须重新测试')
    qs = {q['id']: q for q in load_questions(ROOT)}
    ids = manifest['question_ids']['personamem']
    if runner.digest([qs[u] for u in ids]) != manifest['questions_digest']['personamem']:
        raise ValueError('公开题目与 smoke 清单不同')
    if runner.file_hash(ROOT / 'data/personamem/manifest.json') != manifest['dataset_manifests']['personamem']:
        raise ValueError('数据清单与 smoke 不同')
    for arm in ARMS:
        for unit in ids:
            row = runner.validate_result(runner.load(runner.result_path(output, 'personamem', arm, unit)), unit)
            answer = row['answer']; choice = parse_choice(answer.get('prediction'), qs[unit]['options'])
            if answer['status'] != 'ok' or choice is None:
                raise ValueError(f'Smoke 存在失败或非法选项：{arm} / {unit}')
            if arm == RESIDUAL_V2:
                diag = row['diagnostics']
                if (row.get('method') != RESIDUAL_V2 or row.get('algorithm_version') != RESIDUAL_V2
                        or answer.get('completion_kind') != 'semantic_assessment'
                        or answer.get('certificate') is not None or not diag.get('audit_complete')
                        or answer.get('semantic_answer') != qs[unit]['options'][ord(choice) - 97]
                        or not (diag.get('terminal_input') or {}).get('validation_complete')):
                    raise ValueError(f'终端任务、选项原文或审计未通过：{unit}')
    return {'status': 'passed', 'questions': len(ids), 'arms': ARMS,
            'note': '执行可靠性检查通过；不代表全量无失败或准确率优于基线。'}


def accuracy(output):
    """Current rows only; explicit evaluation, never a generation input."""
    output = Path(output).resolve(); manifest = runner.load(output / 'manifest.json')
    qs = {q['id']: q for q in load_questions(ROOT)}
    ids, arms = manifest['question_ids']['personamem'], manifest['arms']
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError('运行清单题目 ID 不合法')
    if runner.digest([qs[u] for u in ids]) != manifest['questions_digest']['personamem']:
        raise ValueError('当前题目与实验不匹配')
    if runner.file_hash(ROOT / 'data/personamem/manifest.json') != manifest['dataset_manifests']['personamem']:
        raise ValueError('当前数据清单与实验不匹配，不能混用不同版本的评价标签')
    from dagbt.personamem import validate_dataset
    validate_dataset(ROOT, include_labels=True)
    labels = {r['id']: r for r in runner.load(ROOT / 'data/personamem/evaluation_only.json')}
    gold = {u: parse_choice(labels[u]['correct_answer'], qs[u]['options']) for u in ids}
    if any(v is None for v in gold.values()):
        raise ValueError('标准答案格式非法')
    rows, valid = {}, {}
    print(f'计划 {len(ids)} 题；以下只比较已发布的当前结果，失败计 0。')
    for arm in arms:
        rows[arm], valid[arm], statuses, bad = {}, {}, Counter(), 0
        for unit in ids:
            path = runner.result_path(output, 'personamem', arm, unit)
            if not path.exists():
                continue
            try:
                ans = runner.validate_result(runner.load(path), unit)['answer']
            except (OSError, ValueError, TypeError, KeyError, AttributeError):
                bad += 1; continue
            choice = parse_choice(ans.get('prediction'), qs[unit]['options'])
            ok = ans['status'] == 'ok' and choice is not None
            statuses[ans['status']] += 1
            rows[arm][unit] = int(ok and choice == gold[unit])
            if ok:
                valid[arm][unit] = rows[arm][unit]
        count = len(rows[arm]); correct = sum(rows[arm].values())
        score = f'{100 * correct / count:.2f}%' if count else '暂无'
        print(f'\n{arm}: 已记录 {count}/{len(ids)}，有效 {len(valid[arm])}，损坏 {bad}，正确 {correct}，准确率 {score}')
        print('状态：' + json.dumps(statuses, ensure_ascii=False))
    for arm in arms[1:]:
        for title, table in [('同题（含失败）', rows), ('同题（双方有效）', valid)]:
            common = table[arms[0]].keys() & table[arm].keys(); n = len(common)
            left = sum(table[arms[0]][u] for u in common); right = sum(table[arm][u] for u in common)
            if n:
                print(f'\n{title} {n} 题：{arms[0]} {left}/{n}={100*left/n:.2f}%；'
                      f'{arm} {right}/{n}={100*right/n:.2f}%；差值 {100*(right-left)/n:+.2f} 个百分点')
    return {'labels_read': True, 'scope': 'current rows only; missing and malformed excluded, failed answers scored zero'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['smoke', 'check', 'full', 'status', 'stop', 'accuracy', 'prepare-index'])
    parser.add_argument('--config', default=str(ROOT / 'configs/personamem-v4.json'))
    parser.add_argument('--output')
    parser.add_argument('--smoke-output', default=str(ROOT / 'outputs/personamem_v4_smoke'))
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    if args.action in ('status', 'stop') and not args.output:
        parser.error('status / stop 必须用 --output 指明 smoke 或 full 目录，避免操作错实验。')
    default = ('personamem_v4_index' if args.action == 'prepare-index' else
               'personamem_v4_full' if args.action in ('full', 'accuracy') else 'personamem_v4_smoke')
    output = Path(args.output or ROOT / 'outputs' / default).resolve()
    if args.action in ('smoke', 'full', 'prepare-index'):
        config = runner.load(args.config)
        hosts = [urlsplit(config[k]).hostname for k in ('llm_base_url', 'embedding_base_url')]
        os.environ['no_proxy'] = os.environ['NO_PROXY'] = ','.join(filter(None,
            [os.environ.get('NO_PROXY', ''), os.environ.get('no_proxy', ''), *hosts]))
        if args.action == 'prepare-index':
            derived = Path(config.get('derived_index_dir', 'data/derived')).expanduser()
            derived = derived if derived.is_absolute() else ROOT / derived
            try:
                relative = derived.relative_to(ROOT)
                derived.resolve().relative_to(ROOT.resolve())
            except ValueError:
                raise ValueError('索引构建必须位于当前新版目录中，不能写入共享旧索引') from None
            cursor = ROOT
            for part in relative.parts:
                cursor = cursor / part
                if part in ('.', '..') or cursor.is_symlink():
                    raise ValueError('拒绝通过符号链接构建索引；共享旧索引只能读取')
            if derived.exists() and any(path.is_symlink() for path in derived.rglob('*')):
                raise ValueError('索引目录含符号链接，停止构建以保护共享旧索引')
            # The runner validates config and frozen originals, takes its writer
            # lock, then explicitly builds/resumes only PersonaMem embeddings.
            return runner.main(['prepare-index', '--config', args.config, '--datasets', 'personamem',
                                '--output', str(output)])
        from dagbt.resources import inspect_index
        if inspect_index(config, 'personamem')['status'] != 'ready':
            raise ValueError('缺少已验证的 PersonaMem 索引，请先完成 install；不会自动写入共享旧索引。')
        if args.action == 'full':
            check_smoke(args.smoke_output, args.config)
        if (output / 'manifest.json').exists() or runner.lock_is_held(output):
            raise ValueError('此目录已经有实验。请用 status 查看，或用 --output 指定新目录。')
        result = runner.launch(SimpleNamespace(config=args.config, output=str(output), datasets=['personamem'],
            arms=ARMS, limit=None, question_ids_file=str(SMOKE_IDS) if args.action=='smoke' else None,
            retry_failed=False, offline_preflight=False))
    elif args.action == 'check':
        result = check_smoke(output, args.config)
    elif args.action == 'accuracy':
        result = accuracy(output)
    else:
        result = getattr(runner, args.action)(output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps(runner.redacted_error(exc), ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
