#!/usr/bin/env bash
# DAG v2 LLM 服务启动脚本（生产实测配置，2026-09 源服务器）。
# 模型权重不在本包内：设置 QWEN38_MODEL_PATH 指向本地 Qwen3.8-27B-AWQ-INT4 权重目录。
# 显存需求：TP2，两张 24G 显卡（如 2×RTX 4090）。
set -euo pipefail
: "${QWEN38_MODEL_PATH:?Set QWEN38_MODEL_PATH to the local Qwen3.8-27B-AWQ-INT4 checkpoint directory}"
exec vllm serve "$QWEN38_MODEL_PATH" \
  --served-model-name "${QWEN38_MODEL_NAME:-qwen3.8-27b}" \
  --host "${QWEN38_HOST:-127.0.0.1}" \
  --port "${QWEN38_PORT:-8020}" \
  --tensor-parallel-size "${QWEN38_TP_SIZE:-2}" \
  --disable-custom-all-reduce \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --gpu-memory-utilization "${QWEN38_GPU_MEM:-0.80}" \
  --max-model-len 32768 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 4096 \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --override-generation-config '{"temperature":0.7,"top_p":0.8,"top_k":20,"presence_penalty":1.5}'
