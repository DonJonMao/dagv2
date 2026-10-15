#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$TASK_ROOT"
TASK_MODE="${1:-smoke}"
TASK_PERSONA="${2:-${PERSONAMEM_OUTPUT:-$HOME/dagbt_runs/20261014_personamem_v4_experiment/dagv2/outputs/personamem_v4_experiment_full}}"
TASK_CONFIG="${DAGBT_WEBWALKER_CONFIG:-configs/webwalker-v4.json}"
TASK_JUDGE_CONFIG="${DAGBT_WEBWALKER_JUDGE_CONFIG:-$TASK_CONFIG}"
TASK_PYTHON="${DAGBT_PYTHON:-}"
if [[ -z "$TASK_PYTHON" ]]; then
  if [[ -x "$TASK_ROOT/.venv/bin/python" ]]; then
    TASK_PYTHON="$TASK_ROOT/.venv/bin/python"
  elif [[ -x "$HOME/dagbt_runs/20260930_v3/dagv2/.venv/bin/python" ]]; then
    TASK_PYTHON="$HOME/dagbt_runs/20260930_v3/dagv2/.venv/bin/python"
  else
    TASK_PYTHON="python3"
  fi
fi
case "$TASK_MODE" in
  offline)
    "$TASK_PYTHON" scripts/webwalker_offline_smoke.py
    "$TASK_PYTHON" -m dagbt.webwalker_runner preflight --offline --config "$TASK_CONFIG"
    ;;
  preflight)
    "$TASK_PYTHON" -m dagbt.webwalker_runner preflight --config "$TASK_CONFIG" \
      --after-personamem "$TASK_PERSONA" --output outputs/webwalkerqa_v4_access_v3_preflight
    ;;
  smoke)
    TASK_OUTPUT="${WEBWALKER_OUTPUT:-outputs/webwalkerqa_v4_access_v3_smoke}"
    "$TASK_PYTHON" scripts/webwalker_offline_smoke.py
    "$TASK_PYTHON" -m dagbt.webwalker_runner preflight --config "$TASK_CONFIG" \
      --after-personamem "$TASK_PERSONA" --output outputs/webwalkerqa_v4_access_v3_preflight
    "$TASK_PYTHON" -m dagbt.webwalker_runner run --config "$TASK_CONFIG" \
      --after-personamem "$TASK_PERSONA" --limit 2 --output "$TASK_OUTPUT"
    "$TASK_PYTHON" -m dagbt.webwalker_runner score --config "$TASK_JUDGE_CONFIG" --output "$TASK_OUTPUT"
    ;;
  full)
    TASK_OUTPUT="${WEBWALKER_OUTPUT:-outputs/webwalkerqa_v4_access_v3_full}"
    "$TASK_PYTHON" -m dagbt.webwalker_runner preflight --config "$TASK_CONFIG" \
      --after-personamem "$TASK_PERSONA" --output outputs/webwalkerqa_v4_access_v3_preflight
    "$TASK_PYTHON" -m dagbt.webwalker_runner status --output "$TASK_OUTPUT"
    mkdir -p "$TASK_OUTPUT"
    DAGBT_PYTHON="$TASK_PYTHON" WEBWALKER_OUTPUT="$TASK_OUTPUT" nohup bash "$0" _foreground "$TASK_PERSONA" \
      >> "$TASK_OUTPUT/launcher.log" 2>&1 < /dev/null &
    TASK_PID=$!
    echo "WebWalker coordinator launched: PID=$TASK_PID; log=$TASK_OUTPUT/launcher.log"
    echo "Use: bash scripts/run_webwalker.sh status"
    ;;
  _foreground)
    TASK_OUTPUT="${WEBWALKER_OUTPUT:-outputs/webwalkerqa_v4_access_v3_full}"
    "$TASK_PYTHON" -m dagbt.webwalker_runner run --config "$TASK_CONFIG" \
      --after-personamem "$TASK_PERSONA" --output "$TASK_OUTPUT"
    "$TASK_PYTHON" -m dagbt.webwalker_runner score --config "$TASK_JUDGE_CONFIG" --output "$TASK_OUTPUT"
    ;;
  status)
    "$TASK_PYTHON" -m dagbt.webwalker_runner status --output "${WEBWALKER_OUTPUT:-outputs/webwalkerqa_v4_access_v3_full}"
    ;;
  stop)
    if [[ -z "${2:-}" ]]; then echo "Usage: $0 stop /path/to/webwalker/output" >&2; exit 2; fi
    "$TASK_PYTHON" -m dagbt.webwalker_runner stop --output "$2"
    ;;
  score)
    "$TASK_PYTHON" -m dagbt.webwalker_runner score --config "$TASK_JUDGE_CONFIG" \
      --output "${WEBWALKER_OUTPUT:-outputs/webwalkerqa_v4_access_v3_full}"
    ;;
  *) echo "Usage: $0 {offline|preflight|smoke|full|status|stop|score} [personamem_output]" >&2; exit 2 ;;
esac
