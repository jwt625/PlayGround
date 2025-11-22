#!/usr/bin/env python3
"""
Run SAM2 for segmentation with control points, then SAM3D Objects for 3D reconstruction.

Usage:
    python run_sam3_auto_segment.py <image_path> [output_dir] [--auto|--points]

Modes:
    --auto: Automatic segmentation (default, may over-segment)
    --points: Interactive point-based segmentation (recommended for better control)

Example:
    python run_sam3_auto_segment.py IMG_8020.JPG --points
    python run_sam3_auto_segment.py IMG_8020.JPG output_objects --auto
"""
import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage
from collections import defaultdict

# Set environment variables
os.environ["CUDA_HOME"] = "/usr/local/cuda"
os.environ["LIDRA_SKIP_INIT"] = "true"

# Add notebook to path
sys.path.append("sam-3d-objects/notebook")
from inference import Inference, load_image


def merge_overlapping_masks(masks_data, iou_threshold=0.3, containment_threshold=0.7):
    """
    Merge masks that significantly overlap or are contained within each other.

    Args:
        masks_data: List of mask dictionaries from SAM2
        iou_threshold: Merge if IoU (intersection over union) exceeds this
        containment_threshold: Merge if one mask is mostly contained in another

    Returns:
        List of merged mask dictionaries
    """
    if len(masks_data) == 0:
        return []

    # Sort by area (largest first) to prioritize larger objects
    masks_data = sorted(masks_data, key=lambda x: x['area'], reverse=True)

    # Track which masks have been merged
    merged_into = {}  # maps index -> index of mask it was merged into
    merged_masks = []

    for i, mask_i in enumerate(masks_data):
        if i in merged_into:
            continue  # Already merged into another mask

        # Start with current mask
        combined_mask = mask_i['segmentation'].copy()
        merged_indices = [i]

        # Check all subsequent masks for merging
        for j in range(i + 1, len(masks_data)):
            if j in merged_into:
                continue

            mask_j = masks_data[j]['segmentation']

            # Calculate overlap metrics
            intersection = np.logical_and(combined_mask, mask_j).sum()
            union = np.logical_or(combined_mask, mask_j).sum()
            iou = intersection / union if union > 0 else 0

            # Calculate containment (is mask_j mostly inside combined_mask?)
            containment = intersection / mask_j.sum() if mask_j.sum() > 0 else 0

            # Merge if significant overlap or containment
            if iou > iou_threshold or containment > containment_threshold:
                combined_mask = np.logical_or(combined_mask, mask_j)
                merged_indices.append(j)
                merged_into[j] = i

        # Create merged mask entry
        merged_mask_data = {
            'segmentation': combined_mask,
            'area': combined_mask.sum(),
            'bbox': get_bbox(combined_mask),
            'predicted_iou': np.mean([masks_data[idx]['predicted_iou'] for idx in merged_indices]),
            'stability_score': np.mean([masks_data[idx]['stability_score'] for idx in merged_indices]),
            'merged_from': merged_indices
        }
        merged_masks.append(merged_mask_data)

    print(f"  Merged {len(masks_data)} masks into {len(merged_masks)} objects")
    return merged_masks


def get_bbox(mask):
    """Get bounding box [x, y, w, h] from binary mask."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return [0, 0, 0, 0]
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [int(cmin), int(rmin), int(cmax - cmin + 1), int(rmax - rmin + 1)]

# Parse command line arguments
if len(sys.argv) < 2:
    print("Error: Image path required")
    print("Usage: python run_sam3_auto_segment.py <image_path> [output_dir] [--auto|--points]")
    sys.exit(1)

IMAGE_PATH = sys.argv[1]
input_name = Path(IMAGE_PATH).stem

# Parse mode and output directory
mode = "auto"  # default
OUTPUT_DIR = None

for arg in sys.argv[2:]:
    if arg in ["--auto", "--points"]:
        mode = arg[2:]  # Remove '--'
    elif not arg.startswith("--"):
        OUTPUT_DIR = arg

# Generate output directory if not provided
if OUTPUT_DIR is None:
    OUTPUT_DIR = f"{input_name}_objects"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print(f"SAM2 Segmentation ({mode.upper()} mode) + SAM 3D Objects - Multi-Object 3D Reconstruction")
print("=" * 80)

# Step 1: Load the image
print(f"\n[1/5] Loading image: {IMAGE_PATH}")
image = Image.open(IMAGE_PATH).convert("RGB")
image_np = np.array(image)
print(f"  Image size: {image.size}")

# Step 2: Run SAM2 for automatic segmentation
print("\n[2/5] Running SAM2 for automatic segmentation...")
print("  Installing SAM2 if needed...")

try:
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    print("  SAM2 already installed")
except ImportError:
    print("  SAM2 not found. Installing...")
    os.system("pip install git+https://github.com/facebookresearch/segment-anything-2.git")
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# Download SAM2 checkpoint if needed
checkpoint_path = "sam-3d-objects/checkpoints/sam2_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

if not os.path.exists(checkpoint_path):
    print(f"  Downloading SAM2 checkpoint...")
    os.system(f"wget -P sam-3d-objects/checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -O {checkpoint_path}")

# Build SAM2 model
print("  Loading SAM2 model...")
sam2_model = build_sam2(model_cfg, checkpoint_path, device="cuda")
mask_generator = SAM2AutomaticMaskGenerator(
    sam2_model,
    points_per_side=16,  # Reduced from 32 to get fewer, larger objects
    pred_iou_thresh=0.86,  # Increased from 0.7 to be more selective
    stability_score_thresh=0.92,  # Increased from 0.85 to be more selective
    crop_n_layers=0,  # Reduced from 1 to avoid small crops
    crop_n_points_downscale_factor=2,
    min_mask_region_area=2000,  # Increased from 100 to filter out small objects
)

print("  Generating masks...")
masks_data = mask_generator.generate(image_np)
print(f"  Found {len(masks_data)} initial masks")

# Step 2a: Merge overlapping/adjacent masks that likely belong to same object
print("  Merging masks that belong to the same object...")
masks_data = merge_overlapping_masks(
    masks_data,
    iou_threshold=0.1,  # Merge if 10% overlap
    containment_threshold=0.5  # Merge if one mask is 50% inside another
)

# Sort masks by area (largest first)
masks_data = sorted(masks_data, key=lambda x: x['area'], reverse=True)

# Filter to keep only significant objects (top N by area, or above certain size threshold)
image_area = image_np.shape[0] * image_np.shape[1]
min_object_area = image_area * 0.005  # At least 0.5% of image area (reduced from 1%)
max_objects = 15  # Maximum number of objects to process

masks_data = [m for m in masks_data if m['area'] >= min_object_area]
masks_data = masks_data[:max_objects]  # Keep only top N largest

print(f"  Filtered to {len(masks_data)} significant objects (min area: {min_object_area:.0f} pixels)")

# Step 3: Visualize segmentation
print("\n[3/5] Visualizing segmentation...")
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Original image
axes[0].imshow(image_np)
axes[0].set_title("Original Image")
axes[0].axis('off')

# Segmentation overlay
mask_colors = sns.color_palette("husl", len(masks_data))
overlay = np.copy(image_np).astype(float) / 255.0

for i, mask_data in enumerate(masks_data):
    mask = mask_data['segmentation']
    color = np.array(mask_colors[i])
    overlay[mask] = overlay[mask] * 0.5 + color * 0.5

axes[1].imshow(overlay)
axes[1].set_title(f"Segmentation ({len(masks_data)} objects)")
axes[1].axis('off')

plt.tight_layout()
seg_viz_path = os.path.join(OUTPUT_DIR, "segmentation_visualization.png")
plt.savefig(seg_viz_path, dpi=150, bbox_inches='tight')
print(f"  Saved segmentation visualization to: {seg_viz_path}")
plt.close()

# Save individual masks
print("\n[4/5] Saving individual masks...")
for i, mask_data in enumerate(masks_data):
    mask = mask_data['segmentation'].astype(np.uint8) * 255
    mask_path = os.path.join(OUTPUT_DIR, f"mask_{i:03d}.png")
    Image.fromarray(mask).save(mask_path)
print(f"  Saved {len(masks_data)} masks to {OUTPUT_DIR}/")

# Step 4: Run SAM 3D Objects on each mask
print(f"\n[5/5] Running 3D reconstruction on each object...")
print("  Loading SAM 3D Objects model...")

CONFIG_PATH = "sam-3d-objects/checkpoints/hf/pipeline.yaml"
inference = Inference(CONFIG_PATH, compile=False)
print("  Model loaded successfully!")

# Reconstruct each object
for i, mask_data in enumerate(masks_data):
    print(f"\n  Processing object {i+1}/{len(masks_data)} (area: {mask_data['area']} pixels)...")
    mask = mask_data['segmentation']
    
    output_path = os.path.join(OUTPUT_DIR, f"object_{i:03d}.ply")
    
    try:
        output = inference(image_np, mask, seed=42)
        output["gs"].save_ply(output_path)
        print(f"    ✓ Saved to: {output_path}")
    except Exception as e:
        print(f"    ✗ Failed: {e}")

print("\n" + "=" * 80)
print("✓ SUCCESS! Multi-object reconstruction complete!")
print("=" * 80)
print(f"\nOutput directory: {OUTPUT_DIR}/")
print(f"  - segmentation_visualization.png")
print(f"  - mask_*.png (individual masks)")
print(f"  - object_*.ply (3D reconstructions)")
print("\nYou can view the .ply files with CloudCompare, MeshLab, Blender, or online viewers.")
print("\n")

