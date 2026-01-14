#!/bin/bash
# DeepSpeed benchmark with GPU monitoring
# Usage: ./run_benchmark.sh [steps]

set -e

STEPS=${1:-100}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

# Output files
GPU_METRICS_FILE="$SCRIPT_DIR/gpu_metrics_deepspeed_$TIMESTAMP.csv"
TRAINING_LOG="$SCRIPT_DIR/training_$TIMESTAMP.log"

echo "DeepSpeed Benchmark"
echo "==================="
echo "Steps: $STEPS"
echo "GPU Metrics: $GPU_METRICS_FILE"
echo "Training Log: $TRAINING_LOG"
echo ""

# Check if monitor_gpu_v2 exists
MONITOR_BIN="$PARENT_DIR/gpu_fast_metrics/monitor_gpu_v2"
if [ ! -x "$MONITOR_BIN" ]; then
    echo "Error: $MONITOR_BIN not found or not executable"
    echo "Compile it with: gcc -o monitor_gpu_v2 monitor_gpu_v2.c -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lnvidia-ml"
    exit 1
fi

# Activate venv
source "$PARENT_DIR/.venv/bin/activate"

# Start GPU monitoring in background
echo "Starting GPU monitoring (10ms interval)..."
$MONITOR_BIN 10 "$GPU_METRICS_FILE" &
MONITOR_PID=$!
echo "GPU monitor PID: $MONITOR_PID"

# Cleanup function
cleanup() {
    echo ""
    echo "Stopping GPU monitoring..."
    kill $MONITOR_PID 2>/dev/null || true
    wait $MONITOR_PID 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT

# Wait for monitor to start
sleep 1

# Run DeepSpeed training
echo ""
echo "Starting DeepSpeed training..."
echo ""

deepspeed --num_gpus=2 "$SCRIPT_DIR/train_simple.py" \
    --deepspeed_config "$SCRIPT_DIR/ds_config.json" \
    --n_layer 12 \
    --n_embd 768 \
    --batch_size 8 \
    --seq_len 1024 \
    --steps $STEPS \
    2>&1 | ts '%Y-%m-%d %H:%M:%.S' | tee "$TRAINING_LOG"

echo ""
echo "Training complete."
echo "GPU metrics saved to: $GPU_METRICS_FILE"
echo "Training log saved to: $TRAINING_LOG"

