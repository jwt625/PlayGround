#!/usr/bin/env python3
"""
vLLM server for Qwen3-VL-32B-Instruct with token-based authentication.

This server wraps vLLM's OpenAI-compatible API with token authentication.
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set CUDA_VISIBLE_DEVICES if specified
cuda_devices = os.getenv("CUDA_VISIBLE_DEVICES")
if cuda_devices:
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    print(f"Setting CUDA_VISIBLE_DEVICES={cuda_devices}")

# Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-VL-32B-Instruct")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
API_KEY = os.getenv("API_KEY")

# vLLM configuration for H100
TENSOR_PARALLEL_SIZE = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.95"))
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "128000"))
DTYPE = os.getenv("DTYPE", "bfloat16")

# Validate API key
if not API_KEY:
    print("ERROR: API_KEY environment variable is not set!")
    print("Please set it in the .env file or export it as an environment variable.")
    sys.exit(1)

print("=" * 80)
print("Qwen3-VL-32B-Instruct vLLM Server")
print("=" * 80)
print(f"Model: {MODEL_NAME}")
print(f"Host: {HOST}")
print(f"Port: {PORT}")
print(f"Tensor Parallel Size: {TENSOR_PARALLEL_SIZE}")
print(f"GPU Memory Utilization: {GPU_MEMORY_UTILIZATION}")
print(f"Max Model Length: {MAX_MODEL_LEN}")
print(f"Data Type: {DTYPE}")
print(f"API Key: {'*' * (len(API_KEY) - 4)}{API_KEY[-4:]}")
print("=" * 80)
print()

# Build vLLM command
cmd = [
    "vllm", "serve", MODEL_NAME,
    "--host", HOST,
    "--port", str(PORT),
    "--api-key", API_KEY,
    "--tensor-parallel-size", str(TENSOR_PARALLEL_SIZE),
    "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
    "--max-model-len", str(MAX_MODEL_LEN),
    "--dtype", DTYPE,
    "--trust-remote-code",
]

# Optional: Enable async scheduling for better performance
if os.getenv("ASYNC_SCHEDULING", "true").lower() == "true":
    cmd.append("--async-scheduling")
    print("Async scheduling: ENABLED")

# Optional: Limit video inputs if only using images
if os.getenv("DISABLE_VIDEO", "false").lower() == "true":
    cmd.extend(["--limit-mm-per-prompt.video", "0"])
    print("Video inputs: DISABLED")

print()
print("Starting vLLM server...")
print(f"Command: {' '.join(cmd)}")
print()
print("=" * 80)
print()

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Generate timestamped log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = logs_dir / f"vllm_server_{timestamp}.log"

print(f"Logging to: {log_file}")
print("=" * 80)
print()

# Run vLLM server with output redirected to log file
try:
    with open(log_file, "w") as f:
        # Write header to log file
        f.write("=" * 80 + "\n")
        f.write("Qwen3-VL-32B-Instruct vLLM Server Log\n")
        f.write(f"Started at: {datetime.now().isoformat()}\n")
        f.write("=" * 80 + "\n")
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write("=" * 80 + "\n\n")
        f.flush()

        # Run vLLM and tee output to both stdout and log file
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Stream output to both console and file
        for line in process.stdout:
            print(line, end='')
            f.write(line)
            f.flush()

        process.wait()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)

except KeyboardInterrupt:
    print("\n\nServer stopped by user.")
    with open(log_file, "a") as f:
        f.write(f"\n\nServer stopped by user at {datetime.now().isoformat()}\n")
except subprocess.CalledProcessError as e:
    print(f"\n\nError running vLLM server: {e}")
    with open(log_file, "a") as f:
        f.write(f"\n\nError at {datetime.now().isoformat()}: {e}\n")
    sys.exit(1)

