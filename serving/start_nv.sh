#!/usr/bin/env bash
set -euo pipefail
: "${NV_EMBED_MODEL_PATH:?Set NV_EMBED_MODEL_PATH to the local NV-Embed-v2 checkpoint directory}"
export NV_EMBED_MODEL_NAME=nvidia/NV-Embed-v2
export NV_EMBED_MAX_LENGTH=4096
export NV_EMBED_BATCH_SIZE=4
export NV_EMBED_INSTRUCTION=''
exec python "$(dirname "$0")/serve_nv_embed.py"
