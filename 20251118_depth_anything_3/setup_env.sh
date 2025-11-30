#!/bin/bash
# Setup script for Depth Anything V3 environment
# This script activates the virtual environment and sets up model caching

# Activate virtual environment
source /home/ubuntu/GitHub/PlayGround/20251118_depth_anything_3/.venv/bin/activate

# Set Hugging Face cache to local directory
export HF_HOME=/home/ubuntu/GitHub/PlayGround/20251118_depth_anything_3/models
export TRANSFORMERS_CACHE=/home/ubuntu/GitHub/PlayGround/20251118_depth_anything_3/models
export HF_DATASETS_CACHE=/home/ubuntu/GitHub/PlayGround/20251118_depth_anything_3/models

# Set CUDA environment
export CUDA_HOME=/usr/local/cuda

echo "Environment configured:"
echo "  - Virtual environment: activated"
echo "  - Model cache directory: $HF_HOME"
echo "  - CUDA_HOME: $CUDA_HOME"
echo ""
echo "DA3 CLI is ready. Try: da3 --help"

