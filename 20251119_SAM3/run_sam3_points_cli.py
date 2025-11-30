#!/usr/bin/env python3
"""
Run SAM2 with command-line specified points for segmentation, then SAM3D Objects for 3D reconstruction.

Usage:
    python run_sam3_points_cli.py <image_path> <x1,y1> <x2,y2> ... [--output output_dir]

Example:
    python run_sam3_points_cli.py IMG_8020.JPG 1000,1500 2000,1500 3000,1500 --output my_objects
    
First run with --preview to see the image and decide where to click:
    python run_sam3_points_cli.py IMG_8020.JPG --preview
"""
import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
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
    print("Usage: python run_sam3_points_cli.py <image_path> <x1,y1> <x2,y2> ... [--output output_dir]")
    print("   or: python run_sam3_points_cli.py <image_path> --preview")
    sys.exit(1)

IMAGE_PATH = sys.argv[1]
input_name = Path(IMAGE_PATH).stem

# Check for preview mode
if "--preview" in sys.argv:
    print(f"Loading image: {IMAGE_PATH}")
    image = Image.open(IMAGE_PATH)
    width, height = image.size
    
    # Create preview with grid
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(image)
    ax.set_title(f"Image Preview: {IMAGE_PATH}\nSize: {width}x{height}\nClick on objects and note coordinates", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    
    # Add coordinate display on hover
    def on_move(event):
        if event.xdata is not None and event.ydata is not None:
            ax.format_coord = lambda x, y: f'x={int(x)}, y={int(y)}'
    
    fig.canvas.mpl_connect('motion_notify_event', on_move)
    
    preview_path = f"{input_name}_preview.png"
    plt.savefig(preview_path, dpi=150, bbox_inches='tight')
    print(f"\nPreview saved to: {preview_path}")
    print(f"\nImage dimensions: {width} x {height}")
    print("\nTo segment objects, run:")
    print(f"  python run_sam3_points_cli.py {IMAGE_PATH} <x1,y1> <x2,y2> ...")
    print("\nExample (click on 3 objects):")
    print(f"  python run_sam3_points_cli.py {IMAGE_PATH} {width//4},{height//2} {width//2},{height//2} {3*width//4},{height//2}")
    sys.exit(0)

# Parse points and output directory
points = []
OUTPUT_DIR = None

for arg in sys.argv[2:]:
    if arg == "--output":
        continue
    elif arg.startswith("--"):
        continue
    elif OUTPUT_DIR is None and sys.argv[sys.argv.index(arg)-1] == "--output":
        OUTPUT_DIR = arg
    else:
        # Try to parse as point
        try:
            x, y = map(int, arg.split(','))
            points.append([x, y])
        except:
            print(f"Warning: Could not parse '{arg}' as point (expected format: x,y)")

if len(points) == 0:
    print("Error: No valid points provided")
    print("Usage: python run_sam3_points_cli.py <image_path> <x1,y1> <x2,y2> ...")
    print("   or: python run_sam3_points_cli.py <image_path> --preview")
    sys.exit(1)

# Generate output directory if not provided
if OUTPUT_DIR is None:
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
print(f"  Points to segment: {len(points)}")
for i, point in enumerate(points):
    print(f"    Point {i+1}: ({point[0]}, {point[1]})")

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

