#!/bin/bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/da3_${TIMESTAMP}.log"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <images_dir> [options]"
    echo "Example: $0 /path/to/images --process-res 1024"
    exit 1
fi

INPUT_DIR="$1"
shift

if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Directory not found: $INPUT_DIR"
    exit 1
fi

echo "Starting DA3 processing..."
echo "Input: $INPUT_DIR"
echo "Log: $LOG_FILE"
echo ""

source setup_env.sh > /dev/null 2>&1

START_TIME=$(date +%s)

if da3 images "$INPUT_DIR" "$@" 2>&1 | tee "$LOG_FILE"; then
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    echo ""
    echo "✓ Completed in ${ELAPSED}s"
    echo "✓ Log: $LOG_FILE"
    echo "✓ Output: workspace/gallery/scene/"
    exit 0
else
    echo ""
    echo "✗ Failed - check log: $LOG_FILE"
    exit 1
fi

