#!/usr/bin/env bash
# Recommended local-terminal method; legacy configs remain available explicitly.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -n "${DAGBT_PYTHON:-}" ]]; then
  PYTHON="$DAGBT_PYTHON"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
else
  PYTHON="python3"
fi
ACTION="start"
if [[ $# -gt 0 && "$1" != -* ]]; then
  ACTION="$1"
  shift
fi
OUTPUT="$ROOT/outputs/paired_local_terminal_v1"
case "$ACTION" in
  start|resume)
    exec "$PYTHON" -m dagbt.runner launch --config "$ROOT/configs/local-terminal.example.json" --output "$OUTPUT" --arms original dagbt_local_terminal_v1 "$@"
    ;;
  status|stop|diagnostics)
    exec "$PYTHON" -m dagbt.runner "$ACTION" --output "$OUTPUT" "$@"
    ;;
  preflight|prepare-index)
    exec "$PYTHON" -m dagbt.runner "$ACTION" --config "$ROOT/configs/local-terminal.example.json" "$@"
    ;;
  help)
    echo "Usage: $0 [start|resume|status|stop|diagnostics|preflight|prepare-index] [options]"
    echo "Default output: $OUTPUT"
    echo "resume skips completed tasks; retry failures only with explicit --retry-failed."
    ;;
  *)
    echo "Usage: $0 [start|resume|status|stop|diagnostics|preflight|prepare-index] [options]" >&2
    exit 2
    ;;
esac
