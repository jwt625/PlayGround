#!/usr/bin/env python3
"""
Script to download and cache the EdgeTAM model locally.
This will download the model weights and configuration files to the local cache.
"""

import os
from transformers import EdgeTamModel, Sam2Processor, EdgeTamVideoModel, Sam2VideoProcessor

# Set cache directory (optional - defaults to ~/.cache/huggingface)
# os.environ['HF_HOME'] = './models_cache'

# Disable authentication requirement for public models
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'

MODEL_NAME = "yonigozlan/EdgeTAM-hf"

def download_models():
    """Download and cache EdgeTAM models and processors."""
    print(f"Downloading EdgeTAM model from {MODEL_NAME}...")
    print("This may take a few minutes depending on your internet connection.\n")

    # Download image model
    print("1. Downloading EdgeTAM image model...")
    model = EdgeTamModel.from_pretrained(MODEL_NAME, token=False)
    print(f"   ✓ Model downloaded successfully")
    print(f"   Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters\n")
    
    # Download processor
    print("2. Downloading Sam2Processor...")
    processor = Sam2Processor.from_pretrained(MODEL_NAME, token=False)
    print(f"   ✓ Processor downloaded successfully\n")

    # Download video model
    print("3. Downloading EdgeTAM video model...")
    video_model = EdgeTamVideoModel.from_pretrained(MODEL_NAME, token=False)
    print(f"   ✓ Video model downloaded successfully\n")

    # Download video processor
    print("4. Downloading Sam2VideoProcessor...")
    video_processor = Sam2VideoProcessor.from_pretrained(MODEL_NAME, token=False)
    print(f"   ✓ Video processor downloaded successfully\n")
    
    print("=" * 60)
    print("All models and processors have been downloaded and cached!")
    print("=" * 60)
    
    # Show cache location
    from huggingface_hub import scan_cache_dir
    cache_info = scan_cache_dir()
    print(f"\nCache location: {cache_info.cache_dir}")
    print(f"Total cache size: {cache_info.size_on_disk / 1e9:.2f} GB")
    
    # Find EdgeTAM in cache
    for repo in cache_info.repos:
        if "EdgeTAM" in repo.repo_id or "edgetam" in repo.repo_id.lower():
            print(f"\nEdgeTAM model cached:")
            print(f"  Repository: {repo.repo_id}")
            print(f"  Size: {repo.size_on_disk / 1e6:.2f} MB")
            print(f"  Last accessed: {repo.last_accessed}")
            print(f"  Last modified: {repo.last_modified}")

if __name__ == "__main__":
    download_models()

