#!/bin/bash
# Start vLLM server with Qwen3-VL-32B-Instruct
set -e
cd "$(dirname "$0")/.."
source .env
mkdir -p logs
LOG_FILE="logs/vllm_$(date +%Y%m%d_%H%M%S).log"

source $HOME/miniconda3/bin/activate && conda activate vllm
exec python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 --port 8000 \
  --model Qwen/Qwen3-VL-32B-Instruct \
  --served-model-name Qwen3-VL-32B-Instruct \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 32768 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --api-key "$VLLM_API_KEY" \
  2>&1 | tee "$LOG_FILE"

