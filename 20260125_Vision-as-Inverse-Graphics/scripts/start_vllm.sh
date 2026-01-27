#!/bin/bash
# Start vLLM server with Qwen3-VL-32B-Instruct (background)
set -e
cd "$(dirname "$0")/.."
source .env
mkdir -p logs
LOG_FILE="logs/vllm_$(date +%Y%m%d_%H%M%S).log"

# Kill existing vllm server if running
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true

nohup bash -c "source $HOME/miniconda3/bin/activate && conda activate vllm && \
python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 --port 8000 \
  --model Qwen/Qwen3-VL-32B-Instruct \
  --served-model-name Qwen3-VL-32B-Instruct \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 32768 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --api-key $VLLM_API_KEY" > "$LOG_FILE" 2>&1 &

echo "vLLM server starting in background (PID: $!)"
echo "Log: $LOG_FILE"
echo "Tail log: tail -f $LOG_FILE"

