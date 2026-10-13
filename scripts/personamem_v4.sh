#!/usr/bin/env bash
set -euo pipefail
DAGBT_RELEASE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DAGBT_RELEASE_PYTHON="${DAGBT_PYTHON:-$HOME/dagbt_runs/20260930_v3/dagv2/.venv/bin/python}"
if [[ ! -x "$DAGBT_RELEASE_PYTHON" ]]; then
  printf '%s\n' '找不到 Python 环境。请 export DAGBT_PYTHON=已有 dagbt 环境的 python 完整路径。' >&2
  exit 1
fi
cd "$DAGBT_RELEASE_ROOT"
if [[ "${1:-}" == install ]]; then
  shift
  exec "$DAGBT_RELEASE_PYTHON" scripts/install_personamem_v4.py "$@"
fi
exec "$DAGBT_RELEASE_PYTHON" scripts/personamem_v4.py "$@"
