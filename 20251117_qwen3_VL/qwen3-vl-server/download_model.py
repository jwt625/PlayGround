#!/usr/bin/env python3
"""Download and cache the Qwen3-VL-32B-Instruct model locally."""

from huggingface_hub import snapshot_download
import os

MODEL_NAME = "Qwen/Qwen3-VL-32B-Instruct"
CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")

print(f"Downloading {MODEL_NAME} to {CACHE_DIR}...")
print("This may take a while depending on your internet connection.")
print("Model size: ~33B parameters (~62GB in BF16)")

snapshot_download(
    repo_id=MODEL_NAME,
    cache_dir=CACHE_DIR,
    resume_download=True,
    local_files_only=False,
)

print(f"\n✓ Model {MODEL_NAME} successfully downloaded and cached!")
print(f"Cache location: {CACHE_DIR}")

