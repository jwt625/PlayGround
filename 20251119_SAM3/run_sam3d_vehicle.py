#!/usr/bin/env python3
"""
Run SAM 3D Objects on a single image with best quality settings.

Usage:
    python run_sam3d_vehicle.py <image_path> [output_path]

Example:
    python run_sam3d_vehicle.py images/IMG_0092.jpeg
    python run_sam3d_vehicle.py images/IMG_0092.jpeg output.ply
"""
import sys
import os
from pathlib import Path

# Set environment variables needed by the inference code
# CUDA_HOME is needed for some CUDA operations
os.environ["CUDA_HOME"] = "/usr/local/cuda"
os.environ["LIDRA_SKIP_INIT"] = "true"

# Add notebook to path
sys.path.append("sam-3d-objects/notebook")

from inference import Inference, load_image
from PIL import Image
import numpy as np

# Parse command line arguments
if len(sys.argv) < 2:
    print("Error: Image path required")
    print("Usage: python run_sam3d_vehicle.py <image_path> [output_path]")
    sys.exit(1)

IMAGE_PATH = sys.argv[1]

# Generate output path if not provided
if len(sys.argv) >= 3:
    OUTPUT_PATH = sys.argv[2]
else:
    # Auto-generate output name from input filename
    input_name = Path(IMAGE_PATH).stem
    OUTPUT_PATH = f"{input_name}_reconstruction.ply"

# Configuration
CHECKPOINT_TAG = "hf"
CONFIG_PATH = f"sam-3d-objects/checkpoints/{CHECKPOINT_TAG}/pipeline.yaml"

print("=" * 80)
print("SAM 3D Objects - 3D Reconstruction")
print("=" * 80)

# Load the image
print(f"\n[1/4] Loading image: {IMAGE_PATH}")
image = Image.open(IMAGE_PATH).convert("RGB")
image_np = np.array(image)
print(f"  Image size: {image.size}")

# Create a full mask (reconstruct the entire image)
# For best results, we should segment just the vehicle, but for now use full image
print("\n[2/4] Creating mask (full image)")
mask = np.ones((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)
print(f"  Mask shape: {mask.shape}")

# Load the model
print(f"\n[3/4] Loading SAM 3D Objects model from: {CONFIG_PATH}")
print("  This may take a moment...")
inference = Inference(CONFIG_PATH, compile=False)
print("  Model loaded successfully!")

# Run inference with best settings
print(f"\n[4/4] Running 3D reconstruction (this will take some time)...")
print("  Using seed=42 for reproducibility")
output = inference(image_np, mask, seed=42)

# Save the output
print(f"\n[5/5] Saving reconstruction to: {OUTPUT_PATH}")
output["gs"].save_ply(OUTPUT_PATH)

print("\n" + "=" * 80)
print("✓ SUCCESS! Reconstruction complete!")
print("=" * 80)
print(f"\nOutput saved to: {OUTPUT_PATH}")
print("\nYou can view the .ply file with:")
print("  - CloudCompare")
print("  - MeshLab")
print("  - Blender")
print("  - Online viewers like https://3dviewer.net/")
print("\n")

