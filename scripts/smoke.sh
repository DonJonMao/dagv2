#!/usr/bin/env bash
# 本地冒烟检查：模块导入 + 三数据集 prepare-only 干跑（校验全部数据路径与哈希，无模型调用）。
# 每个数据集单独进程执行（prepare 会改写模块级 e.CONFIG）。
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"

echo "== import check =="
$PY -c "
import sys
sys.path.insert(0, 'dagv2')
import experiment_v6, frozen_flow_v6  # patches frozen_flow/run
import experiment, repair_planner, metrics, core, frozen_flow, reader, run
import pipeline_dagv2, pipeline_dagv2_musique
print('all modules import ok')
"

echo "== prepare-only dry runs =="
$PY dagv2/pipeline_dagv2.py hotpotqa --prepare-only
$PY dagv2/pipeline_dagv2.py 2wikimultihopqa --prepare-only
$PY dagv2/pipeline_dagv2_musique.py --prepare-only
echo "SMOKE OK: data paths, hashes, tokenizer and assets all verified (no model calls made)."
echo "Next with model services up: NOHUP=1 bash scripts/run_hotpotqa.sh  (etc.)"
