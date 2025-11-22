#!/usr/bin/env python3
"""
Run SAM2 with interactive point-based segmentation, then SAM3D Objects for 3D reconstruction.
This gives you precise control over which objects to segment by clicking on them.

Usage:
    python run_sam3_points.py <image_path> [output_dir]

Example:
    python run_sam3_points.py IMG_8020.JPG
"""
import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Set environment variables
os.environ["CUDA_HOME"] = "/usr/local/cuda"
os.environ["LIDRA_SKIP_INIT"] = "true"

# Add notebook to path
sys.path.append("sam-3d-objects/notebook")
from inference import Inference

# Parse command line arguments
if len(sys.argv) < 2:
    print("Error: Image path required")
    print("Usage: python run_sam3_points.py <image_path> [output_dir]")
    sys.exit(1)

IMAGE_PATH = sys.argv[1]
input_name = Path(IMAGE_PATH).stem

# Generate output directory if not provided
if len(sys.argv) >= 3:
    OUTPUT_DIR = sys.argv[2]
else:
    OUTPUT_DIR = f"{input_name}_objects_points"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("SAM2 Point-Based Segmentation + SAM 3D Objects")
print("=" * 80)

# Step 1: Load the image
print(f"\n[1/4] Loading image: {IMAGE_PATH}")
image = Image.open(IMAGE_PATH).convert("RGB")
image_np = np.array(image)
print(f"  Image size: {image.size}")

# Step 2: Load SAM2 for point-based segmentation
print("\n[2/4] Loading SAM2 for point-based segmentation...")

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    print("  SAM2 already installed")
except ImportError:
    print("  SAM2 not found. Installing...")
    os.system("pip install git+https://github.com/facebookresearch/segment-anything-2.git")
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

# Download SAM2 checkpoint if needed
checkpoint_path = "sam-3d-objects/checkpoints/sam2_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

if not os.path.exists(checkpoint_path):
    print(f"  Downloading SAM2 checkpoint...")
    os.system(f"wget -P sam-3d-objects/checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -O {checkpoint_path}")

print("  Loading SAM2 model...")
sam2_model = build_sam2(model_cfg, checkpoint_path, device="cuda")
predictor = SAM2ImagePredictor(sam2_model)
predictor.set_image(image_np)

print("\n" + "=" * 80)
print("INTERACTIVE POINT-BASED SEGMENTATION")
print("=" * 80)
print("\nInstructions:")
print("  1. The image will be displayed")
print("  2. LEFT-CLICK on objects you want to segment (one click per object)")
print("  3. RIGHT-CLICK to undo last point")
print("  4. CLOSE the window when done")
print("\nTip: Click on the CENTER of each distinct object you want to reconstruct")
print("     Each click will segment one complete object")
print("=" * 80)

# Interactive point selection
points = []
point_markers = []
point_texts = []

fig, ax = plt.subplots(figsize=(14, 10))
ax.imshow(image_np)
ax.set_title("LEFT-CLICK on objects | RIGHT-CLICK to undo | CLOSE window when done", fontsize=14)
ax.axis('off')

def onclick(event):
    global points, point_markers, point_texts
    
    if event.xdata is None or event.ydata is None:
        return
    
    x, y = int(event.xdata), int(event.ydata)
    
    if event.button == 1:  # Left click - add point
        points.append([x, y])
        marker, = ax.plot(x, y, 'r+', markersize=20, markeredgewidth=4)
        text = ax.text(x, y-30, f'{len(points)}', color='red', fontsize=14, fontweight='bold',
                      bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        point_markers.append(marker)
        point_texts.append(text)
        fig.canvas.draw()
        print(f"  Point {len(points)}: ({x}, {y})")
    
    elif event.button == 3 and len(points) > 0:  # Right click - undo
        points.pop()
        point_markers[-1].remove()
        point_texts[-1].remove()
        point_markers.pop()
        point_texts.pop()
        fig.canvas.draw()
        print(f"  Removed last point. Now have {len(points)} points")

cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.tight_layout()
plt.show()

if len(points) == 0:
    print("\nNo points selected. Exiting.")
    sys.exit(0)

print(f"\n[3/4] Generating masks for {len(points)} selected points...")
masks_data = []

for i, point in enumerate(points):
    # Generate mask for this point
    masks, scores, logits = predictor.predict(
        point_coords=np.array([point]),
        point_labels=np.array([1]),  # 1 = foreground point
        multimask_output=True,
    )
    
    # Take the best mask (highest score)
    best_idx = np.argmax(scores)
    mask = masks[best_idx]
    
    masks_data.append({
        'segmentation': mask,
        'area': mask.sum(),
        'point': point,
        'score': scores[best_idx]
    })
    print(f"  Mask {i+1}/{len(points)}: area={mask.sum():,} pixels, score={scores[best_idx]:.3f}")

# Visualize segmentation
print("\nVisualizing segmentation...")
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Original image with points
axes[0].imshow(image_np)
for i, point in enumerate(points):
    axes[0].plot(point[0], point[1], 'r+', markersize=20, markeredgewidth=4)
    axes[0].text(point[0], point[1]-30, f'{i+1}', color='red', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
axes[0].set_title("Selected Points")
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
print("\nSaving individual masks...")
for i, mask_data in enumerate(masks_data):
    mask = mask_data['segmentation'].astype(np.uint8) * 255
    mask_path = os.path.join(OUTPUT_DIR, f"mask_{i:03d}.png")
    Image.fromarray(mask).save(mask_path)
print(f"  Saved {len(masks_data)} masks to {OUTPUT_DIR}/")

# Step 4: Run SAM 3D Objects on each mask
print(f"\n[4/4] Running 3D reconstruction on each object...")
print("  Loading SAM 3D Objects model...")

CONFIG_PATH = "sam-3d-objects/checkpoints/hf/pipeline.yaml"
inference = Inference(CONFIG_PATH, compile=False)
print("  Model loaded successfully!")

# Reconstruct each object
for i, mask_data in enumerate(masks_data):
    print(f"\n  Processing object {i+1}/{len(masks_data)} (area: {mask_data['area']:,} pixels)...")
    mask = mask_data['segmentation']

    output_path = os.path.join(OUTPUT_DIR, f"object_{i:03d}.ply")

    try:
        output = inference(image_np, mask, seed=42)
        output["gs"].save_ply(output_path)
        print(f"    ✓ Saved to: {output_path}")
    except Exception as e:
        print(f"    ✗ Failed: {e}")

print("\n" + "=" * 80)
print("✓ SUCCESS! Point-based multi-object reconstruction complete!")
print("=" * 80)
print(f"\nOutput directory: {OUTPUT_DIR}/")
print(f"  - segmentation_visualization.png")
print(f"  - mask_*.png (individual masks)")
print(f"  - object_*.ply (3D reconstructions)")
print("\nYou can view the .ply files with CloudCompare, MeshLab, Blender, or online viewers.")
print("\n")

