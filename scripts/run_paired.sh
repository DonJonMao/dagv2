#!/usr/bin/env bash
# Portable macOS/Linux launcher. All added artifacts stay in this repository.
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
ACTION="launch"
if [[ $# -gt 0 && "$1" != -* ]]; then
  ACTION="$1"
  shift
fi
case "$ACTION" in
  launch|run)
    exec "$PYTHON" -m dagbt.runner "$ACTION" --config "$ROOT/configs/paired.example.json" --output "$ROOT/outputs/paired" "$@"
    ;;
  preflight)
    exec "$PYTHON" -m dagbt.runner preflight --config "$ROOT/configs/paired.example.json" "$@"
    ;;
  status|stop)
    exec "$PYTHON" -m dagbt.runner "$ACTION" --output "$ROOT/outputs/paired" "$@"
    ;;
  *)
    echo "Usage: $0 [launch|run|preflight|status|stop] [options]" >&2
    exit 2
    ;;
esac
