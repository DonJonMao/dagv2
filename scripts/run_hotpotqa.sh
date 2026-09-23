#!/usr/bin/env bash
# 跑 HotpotQA 全量（先 2 题冒烟，再 full1000；评分在 full1000 完成时自动生成）。
# 前台运行：bash scripts/run_hotpotqa.sh
# 后台运行：NOHUP=1 bash scripts/run_hotpotqa.sh   （日志 outputs/logs/hotpotqa.log）
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
if [ "${NOHUP:-0}" = "1" ]; then
  mkdir -p outputs/logs
  nohup "$PY" dagv2/pipeline_dagv2.py hotpotqa > outputs/logs/hotpotqa.log 2>&1 &
  echo "started hotpotqa pipeline in background, pid $!, log: outputs/logs/hotpotqa.log"
else
  exec "$PY" dagv2/pipeline_dagv2.py hotpotqa
fi
