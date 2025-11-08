#!/bin/bash
set -e

# Kimi-K2-Thinking vLLM Launch Script (Docker Nightly Build)
# Optimized for: Long-context (up to 256k), Low-throughput workload
# Hardware: 8× NVIDIA H100 80GB (HGX system)

# Configuration
MODEL_PATH="/home/ubuntu/models/Kimi-K2-Thinking"
HOST="0.0.0.0"
PORT=8000
CONTAINER_NAME="kimi-k2-thinking"
LOG_DIR="/home/ubuntu/fs2/kimi_K2_thinking/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/vllm_server_${TIMESTAMP}.log"
LATEST_LOG_LINK="${LOG_DIR}/vllm_server_latest.log"

# Generate API key if not already set
API_KEY_FILE="/home/ubuntu/fs2/kimi_K2_thinking/.api_key"
if [ ! -f "${API_KEY_FILE}" ]; then
  echo "Generating new API key..."
  API_KEY="sk-$(openssl rand -hex 32)"
  echo "${API_KEY}" > "${API_KEY_FILE}"
  chmod 600 "${API_KEY_FILE}"
else
  API_KEY=$(cat "${API_KEY_FILE}")
fi

# Create log directory if it doesn't exist
mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "Kimi-K2-Thinking vLLM Server (Docker)"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "Docker Image: vllm/vllm-openai:nightly"
echo "Tensor Parallelism: 8 GPUs"
echo "Context Length: 256k tokens"
echo "Optimization: Long-context, low-throughput"
echo "API: OpenAI-compatible on ${HOST}:${PORT}"
echo "API Key: ${API_KEY}"
echo "API Key saved to: ${API_KEY_FILE}"
echo "=========================================="

# Check if container is already running
if sudo docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "Stopping existing container..."
  sudo docker stop ${CONTAINER_NAME} 2>/dev/null || true
  sudo docker rm ${CONTAINER_NAME} 2>/dev/null || true
fi

# Launch vLLM with Docker nightly build (includes kimi_k2 parser support)
# Run in detached mode (-d) so it persists after SSH disconnection
echo "Starting vLLM server in background..."
sudo docker run -d \
  --name ${CONTAINER_NAME} \
  --gpus all \
  --restart unless-stopped \
  -v "${MODEL_PATH}:/model" \
  -p ${PORT}:8000 \
  --ipc=host \
  -e NCCL_DEBUG=INFO \
  -e NCCL_IB_DISABLE=0 \
  -e NCCL_NET_GDR_LEVEL=5 \
  vllm/vllm-openai:nightly \
  --model /model \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key "${API_KEY}" \
  --tensor-parallel-size 8 \
  --trust-remote-code \
  --served-model-name "kimi-k2-thinking" \
  --reasoning-parser kimi_k2 \
  --tool-call-parser kimi_k2 \
  --enable-auto-tool-choice \
  --max-num-batched-tokens 32768 \
  --disable-log-requests
  # --dtype auto \
  # --max-model-len 131072 \
  # --max-num-seqs 4 \
  # --gpu-memory-utilization 0.90 \
  # --enable-chunked-prefill \
  # --enable-prefix-caching \

echo ""
echo "Container started successfully!"
echo ""
echo "Logs are being saved to: ${LOG_FILE}"
echo ""
echo "To view logs in real-time:"
echo "  tail -f ${LOG_FILE}"
echo "  OR"
echo "  sudo docker logs -f ${CONTAINER_NAME}"
echo ""
echo "To stop the server:"
echo "  sudo docker stop ${CONTAINER_NAME}"
echo ""
echo "To check status:"
echo "  sudo docker ps | grep ${CONTAINER_NAME}"
echo ""
echo "Starting log capture in background..."

# Capture logs to file in background
(sudo docker logs -f ${CONTAINER_NAME} >> "${LOG_FILE}" 2>&1) &
LOG_PID=$!
echo "Log capture started (PID: ${LOG_PID})"
echo ""
echo "Server is starting (this may take 2-5 minutes)..."
echo "You can disconnect from SSH - the server will continue running."
echo ""
echo "Monitor startup progress with:"
echo "  tail -f ${LOG_FILE}"

# Configuration explanation:
# -d: Run container in detached mode (background)
# --name ${CONTAINER_NAME}: Name the container for easy management
# --gpus all: Enable all NVIDIA GPUs in container
# --restart unless-stopped: Auto-restart container on reboot (unless manually stopped)
# -v "${MODEL_PATH}:/model": Mount model directory into container
# -p ${PORT}:8000: Map container port 8000 to host port 8000
# --ipc=host: Use host IPC namespace for better multi-GPU performance
# -e NCCL_*: NCCL environment variables for HGX system optimization
# --tensor-parallel-size 8: Use all 8 H100 GPUs
# --dtype auto: Let vLLM auto-detect INT4 compressed-tensors format
# --max-model-len 262144: Support full 256k context (262144 = 256 * 1024)
# --max-num-batched-tokens 32768: Conservative batch size for long contexts
# --max-num-seqs 4: Low concurrency (4 parallel requests max) for low-throughput use case
# --gpu-memory-utilization 0.95: Use 95% of GPU memory (640GB total)
# --enable-chunked-prefill: Essential for long-context processing
# --enable-prefix-caching: Cache common prefixes to speed up repeated queries
# --disable-log-requests: Reduce log verbosity for low-throughput use case
# --trust-remote-code: Required for custom Kimi-K2 model code
# --reasoning-parser kimi_k2: Required for correctly processing reasoning content (nightly build)
# --tool-call-parser kimi_k2: Required for tool usage (nightly build)
# Note: vLLM nightly auto-detects compressed-tensors format from model config

