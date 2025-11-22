#!/usr/bin/env python3
"""
Interactive SAM2 segmentation with positive and negative control points.
Mimics the EdgeTAM interface for object segmentation using SAM2.

Usage:
    python run_sam2_interactive.py <image_path> [--port 7860]

Example:
    python run_sam2_interactive.py IMG_8020.JPG
    python run_sam2_interactive.py IMG_8020.JPG --port 8080
"""
import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import gradio as gr
import logging
import traceback
import cv2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set environment variables
os.environ["CUDA_HOME"] = "/usr/local/cuda"

# Parse command line arguments
if len(sys.argv) < 2:
    print("Error: Image path required")
    print("Usage: python run_sam2_interactive.py <image_path> [--port 7860]")
    sys.exit(1)

IMAGE_PATH = sys.argv[1]
input_name = Path(IMAGE_PATH).stem

# Parse port
port = 7860
if "--port" in sys.argv:
    port_idx = sys.argv.index("--port")
    if port_idx + 1 < len(sys.argv):
        port = int(sys.argv[port_idx + 1])

OUTPUT_DIR = f"{input_name}_sam2_interactive"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("SAM2 Interactive Segmentation with Positive/Negative Points")
print("=" * 80)
print(f"\nImage: {IMAGE_PATH}")
print(f"Output directory: {OUTPUT_DIR}")

# Note: normalize_image function is defined later in the file
# For initial load, we'll do basic conversion here and use normalize_image for uploads

# Load the image
print(f"\nLoading image...")

# Check file extension
ext = os.path.splitext(IMAGE_PATH)[1].lower()
print(f"File extension: {ext}")

if ext in ['.tif', '.tiff']:
    # Use OpenCV for TIFF files to preserve grayscale values
    print("Loading TIFF with OpenCV")
    img_cv = cv2.imread(IMAGE_PATH, cv2.IMREAD_UNCHANGED)
    print(f"OpenCV loaded - dtype: {img_cv.dtype}, shape: {img_cv.shape}, min: {img_cv.min()}, max: {img_cv.max()}, unique: {len(np.unique(img_cv))}")

    # Normalize if needed
    if img_cv.dtype == np.uint16:
        print("Converting 16-bit to 8-bit")
        img_cv = (img_cv / 256).astype(np.uint8)
    elif img_cv.dtype != np.uint8:
        print(f"Normalizing {img_cv.dtype} to uint8")
        img_min = img_cv.min()
        img_max = img_cv.max()
        if img_max > img_min:
            img_cv = ((img_cv.astype(np.float32) - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img_cv = np.full_like(img_cv, 128, dtype=np.uint8)

    print(f"After normalization - dtype: {img_cv.dtype}, min: {img_cv.min()}, max: {img_cv.max()}, unique: {len(np.unique(img_cv))}")

    # Convert to RGB
    if len(img_cv.shape) == 2:
        # Grayscale - convert to RGB
        print("Converting grayscale to RGB")
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB)
    elif img_cv.shape[2] == 4:
        # RGBA - convert to RGB
        print("Converting RGBA to RGB")
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2RGB)
    elif img_cv.shape[2] == 3:
        # BGR - convert to RGB
        print("Converting BGR to RGB")
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    print(f"Final OpenCV array - shape: {img_cv.shape}, dtype: {img_cv.dtype}, min: {img_cv.min()}, max: {img_cv.max()}")

    # Convert to PIL
    image = Image.fromarray(img_cv)
    print(f"Converted to PIL - mode: {image.mode}, size: {image.size}")
else:
    # Use PIL for other formats
    print("Loading with PIL")
    img = Image.open(IMAGE_PATH)
    print(f"Original image mode: {img.mode}, size: {img.size}")
    image = img.convert("RGB")

image_np = np.array(image)
print(f"Final image - size: {image.size}, mode: {image.mode}")
print(f"Final array - shape: {image_np.shape}, dtype: {image_np.dtype}, min: {image_np.min()}, max: {image_np.max()}")

# Load SAM2
print("\nLoading SAM2 model...")
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("Installing SAM2...")
    os.system("pip install git+https://github.com/facebookresearch/segment-anything-2.git")
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

checkpoint_path = "sam-3d-objects/checkpoints/sam2_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

if not os.path.exists(checkpoint_path):
    print("Downloading SAM2 checkpoint...")
    os.makedirs("sam-3d-objects/checkpoints", exist_ok=True)
    os.system(f"wget -P sam-3d-objects/checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -O {checkpoint_path}")

sam2_model = build_sam2(model_cfg, checkpoint_path, device="cuda")
predictor = SAM2ImagePredictor(sam2_model)
predictor.set_image(image_np)
print("SAM2 model loaded!")

# Global state
current_mask = None
mask_counter = 0
current_image = image
current_image_np = image_np


def create_overlay(image, mask, alpha=0.5, color=[30, 144, 255]):
    """Create an overlay of the mask on the original image."""
    img_array = np.array(image)
    
    # Create colored mask
    colored_mask = np.zeros_like(img_array)
    colored_mask[mask > 0] = color
    
    # Blend image and mask
    overlay = img_array.copy()
    overlay = (overlay * (1 - alpha) + colored_mask * alpha).astype(np.uint8)
    
    return Image.fromarray(overlay)


def visualize_points(image, points, labels):
    """Visualize points on the image - optimized to be fast."""
    # Use a shallow copy and draw directly - much faster than deep copy
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)

    for point, label in zip(points, labels):
        x, y = point
        # Positive points in green, negative in red
        color = (0, 255, 0) if label == 1 else (255, 0, 0)
        radius = 8
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color, outline="white", width=2)

    return img_copy


def segment_image(points_data, multimask_output):
    """
    Perform segmentation on the image using the provided points.

    Args:
        points_data: Dictionary containing point coordinates and labels
        multimask_output: Whether to output multiple masks

    Returns:
        Tuple of (annotated image, mask overlay, info text)
    """
    global current_mask, current_image

    logger.info("=" * 60)
    logger.info("Starting segmentation")
    logger.info(f"Points data: {points_data}")
    logger.info(f"Multimask output: {multimask_output}")

    # Parse points data
    if not points_data or "points" not in points_data or len(points_data["points"]) == 0:
        logger.warning("No points provided")
        return current_image, None, "Please click on the image to add points!"

    points = points_data["points"]
    logger.info(f"Number of points: {len(points)}")

    # Extract coordinates and labels
    coords = [[p[0], p[1]] for p in points]
    labels = [p[2] for p in points]  # 1 for positive, 0 for negative
    logger.info(f"Coordinates: {coords}")
    logger.info(f"Labels: {labels}")

    try:
        logger.info("Running SAM2 inference...")
        # Run inference with SAM2
        masks, scores, logits = predictor.predict(
            point_coords=np.array(coords),
            point_labels=np.array(labels),
            multimask_output=multimask_output,
        )
        logger.info("Inference completed")
        logger.info(f"Generated {len(masks)} mask(s)")
        logger.info(f"Scores: {scores}")

        # Select best mask (highest score)
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        best_score = scores[best_idx]
        current_mask = best_mask

        logger.info(f"Best mask index: {best_idx}, score: {best_score:.3f}")
        logger.info(f"Mask area: {best_mask.sum():,} pixels")

        # Create visualizations
        logger.info("Creating visualizations...")
        img_with_points = visualize_points(current_image, coords, labels)
        mask_overlay = create_overlay(current_image, best_mask)
        logger.info("Visualizations created")

        # Build info text
        info = f"Using {len(coords)} point(s)\n"
        info += f"Positive points: {sum(1 for l in labels if l == 1)}, "
        info += f"Negative points: {sum(1 for l in labels if l == 0)}\n"
        info += f"Generated {len(masks)} mask(s)\n"
        info += f"Scores: {', '.join([f'{s:.3f}' for s in scores])}\n"
        info += f"Best mask index: {best_idx} (score: {best_score:.3f})\n"
        info += f"Mask area: {best_mask.sum():,} pixels"

        logger.info("Segmentation completed successfully")
        logger.info("=" * 60)
        return img_with_points, mask_overlay, info

    except Exception as e:
        logger.error(f"Error during segmentation: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return current_image, None, f"Error during segmentation: {str(e)}"


def save_mask():
    """Save the current mask to disk."""
    global current_mask, mask_counter

    if current_mask is None:
        return "No mask to save. Please segment first!"

    try:
        # Save mask as PNG
        mask_path = os.path.join(OUTPUT_DIR, f"mask_{mask_counter:03d}.png")
        mask_img = Image.fromarray((current_mask * 255).astype(np.uint8))
        mask_img.save(mask_path)

        # Save mask as NPY
        npy_path = os.path.join(OUTPUT_DIR, f"mask_{mask_counter:03d}.npy")
        np.save(npy_path, current_mask)

        mask_counter += 1

        info = f"✓ Mask saved:\n"
        info += f"  PNG: {mask_path}\n"
        info += f"  NPY: {npy_path}\n"
        info += f"Total masks saved: {mask_counter}"

        logger.info(f"Saved mask to {mask_path} and {npy_path}")
        return info

    except Exception as e:
        logger.error(f"Error saving mask: {str(e)}")
        return f"Error saving mask: {str(e)}"


def update_points(current_points, label, evt: gr.SelectData):
    """Update points when user clicks on image - visualize points in real-time."""
    global current_image

    x, y = evt.index[0], evt.index[1]

    if current_points is None:
        current_points = {"points": []}

    current_points["points"].append([x, y, label])

    # Visualize points immediately (optimized to be fast)
    coords = [[p[0], p[1]] for p in current_points["points"]]
    labels = [p[2] for p in current_points["points"]]

    img_with_points = visualize_points(current_image, coords, labels)

    point_type = "positive" if label == 1 else "negative"
    info = f"Added {point_type} point at ({x}, {y})\n"
    info += f"Total points: {len(current_points['points'])} "
    info += f"(+{sum(1 for l in labels if l == 1)}, -{sum(1 for l in labels if l == 0)})\n"
    info += f"Click 'Segment' to generate mask"

    return current_points, img_with_points, info


def preprocess_image(image):
    """
    Preprocess uploaded image to ensure compatibility.
    Handles various image formats (TIFF, BMP, WebP, etc.) and converts to RGB.
    Uses OpenCV for better TIFF handling.

    Args:
        image: PIL Image or path to image file

    Returns:
        PIL Image in RGB format
    """
    logger.info("=" * 60)
    logger.info("preprocess_image() called")
    logger.info(f"Input type: {type(image)}")

    if image is None:
        logger.warning("Image is None, returning None")
        return None

    try:
        # If image is a path string, use OpenCV to load it
        if isinstance(image, str):
            logger.info(f"Image is a string path: {image}")

            # Check file extension
            ext = os.path.splitext(image)[1].lower()
            logger.info(f"File extension: {ext}")

            if ext in ['.tif', '.tiff']:
                # Use OpenCV for TIFF files to preserve grayscale values
                logger.info("Loading TIFF with OpenCV")
                img_cv = cv2.imread(image, cv2.IMREAD_UNCHANGED)
                logger.info(f"OpenCV loaded - dtype: {img_cv.dtype}, shape: {img_cv.shape}, min: {img_cv.min()}, max: {img_cv.max()}, unique: {len(np.unique(img_cv))}")

                # Normalize if needed
                if img_cv.dtype == np.uint16:
                    logger.info("Converting 16-bit to 8-bit")
                    img_cv = (img_cv / 256).astype(np.uint8)
                elif img_cv.dtype != np.uint8:
                    logger.info(f"Normalizing {img_cv.dtype} to uint8")
                    img_min = img_cv.min()
                    img_max = img_cv.max()
                    if img_max > img_min:
                        img_cv = ((img_cv.astype(np.float32) - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                    else:
                        img_cv = np.full_like(img_cv, 128, dtype=np.uint8)

                logger.info(f"After normalization - dtype: {img_cv.dtype}, min: {img_cv.min()}, max: {img_cv.max()}, unique: {len(np.unique(img_cv))}")

                # Convert to RGB
                if len(img_cv.shape) == 2:
                    # Grayscale - convert to RGB
                    logger.info("Converting grayscale to RGB")
                    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB)
                elif img_cv.shape[2] == 4:
                    # RGBA - convert to RGB
                    logger.info("Converting RGBA to RGB")
                    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2RGB)
                elif img_cv.shape[2] == 3:
                    # BGR - convert to RGB
                    logger.info("Converting BGR to RGB")
                    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

                logger.info(f"Final OpenCV array - shape: {img_cv.shape}, dtype: {img_cv.dtype}, min: {img_cv.min()}, max: {img_cv.max()}")

                # Convert to PIL
                image = Image.fromarray(img_cv)
                logger.info(f"Converted to PIL - mode: {image.mode}, size: {image.size}")
            else:
                # Use PIL for other formats
                logger.info("Loading with PIL")
                image = Image.open(image)
                logger.info(f"Opened image from path: {image.format}, {image.mode}, {image.size}")

                # Convert to RGB if needed
                if image.mode != "RGB":
                    logger.info(f"Converting image from {image.mode} to RGB")
                    image = image.convert("RGB")
        else:
            # PIL Image object
            logger.info(f"Original image - format: {getattr(image, 'format', 'N/A')}, mode: {image.mode}, size: {image.size}")

            # Check array values before conversion
            img_array = np.array(image)
            logger.info(f"Original array - dtype: {img_array.dtype}, shape: {img_array.shape}, min: {img_array.min()}, max: {img_array.max()}, unique: {len(np.unique(img_array))}")

            # Convert to RGB if needed (handles RGBA, L, CMYK, etc.)
            if image.mode != "RGB":
                logger.info(f"Converting image from {image.mode} to RGB")
                image = image.convert("RGB")
                logger.info(f"Conversion successful")
            else:
                logger.info("Image already in RGB mode, no conversion needed")

        # Final check
        final_array = np.array(image)
        logger.info(f"Final array - dtype: {final_array.dtype}, shape: {final_array.shape}, min: {final_array.min()}, max: {final_array.max()}")

        logger.info(f"Preprocessed image - mode: {image.mode}, size: {image.size}")
        logger.info("preprocess_image() completed successfully")
        logger.info("=" * 60)
        return image

    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"ERROR in preprocess_image()")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        logger.error("=" * 60)
        raise


def handle_image_upload(new_image):
    """Handle new image upload and reinitialize SAM2."""
    global current_image, current_image_np, current_mask

    logger.info("=" * 60)
    logger.info("handle_image_upload() called")
    logger.info(f"Received image type: {type(new_image)}")

    if new_image is None:
        logger.warning("No image uploaded")
        return None, {"points": []}, None, None, "No image uploaded"

    try:
        # Preprocess image using EdgeTAM's approach
        current_image = preprocess_image(new_image)

        if current_image is None:
            return None, {"points": []}, None, None, "Failed to process image"

        current_image_np = np.array(current_image)
        current_mask = None

        logger.info(f"Final image: size={current_image.size}, mode={current_image.mode}")
        logger.info(f"Numpy array: shape={current_image_np.shape}, dtype={current_image_np.dtype}, min={current_image_np.min()}, max={current_image_np.max()}")

        # Reinitialize SAM2 predictor with new image
        logger.info("Reinitializing SAM2 predictor with new image...")
        predictor.set_image(current_image_np)
        logger.info("SAM2 predictor reinitialized")

        logger.info("handle_image_upload() completed successfully")
        logger.info("=" * 60)

        return current_image, {"points": []}, None, None, f"New image loaded: {current_image.size[0]}x{current_image.size[1]}. Click to add points."

    except Exception as e:
        logger.error(f"Error handling image upload: {str(e)}")
        logger.error(traceback.format_exc())
        return None, {"points": []}, None, None, f"Error loading image: {str(e)}"


def clear_all():
    """Clear all points and outputs."""
    global current_mask, current_image
    current_mask = None
    return {"points": []}, None, None, "Points cleared. Click on the image to add points."


# Create Gradio interface
with gr.Blocks(title="SAM2 Interactive Segmentation") as demo:
    gr.Markdown(f"""
    # 🎯 SAM2 Interactive Segmentation

    Interactive object segmentation using SAM2 with positive and negative control points.

    **Image:** {IMAGE_PATH} | **Size:** {image.size[0]}x{image.size[1]}

    ## How to use:
    1. Click on the image to add positive points (green) - areas you want to include
    2. Switch to "Negative" and click to add negative points (red) - areas you want to exclude
    3. Click "Segment" to generate the mask
    4. Adjust parameters if needed and re-segment
    5. Click "Save Mask" to save the current mask to disk

    **Point Labels:**
    - **Positive points (green):** Click on the object you want to segment
    - **Negative points (red):** Click on areas you want to exclude from the mask
    """)

    with gr.Row():
        with gr.Column(scale=1):
            # Input image with interactive points
            input_image = gr.Image(
                label="Input Image (Click to add points)",
                type="pil",
                value=image,
                interactive=True
            )

            # Point selection interface
            points_data = gr.State(value={"points": []})

            with gr.Row():
                point_label = gr.Radio(
                    choices=[("Positive (include)", 1), ("Negative (exclude)", 0)],
                    value=1,
                    label="Point Type"
                )

            with gr.Row():
                segment_btn = gr.Button("Segment", variant="primary", size="lg")
                clear_btn = gr.Button("Clear Points")
                save_btn = gr.Button("Save Mask", variant="secondary")

        with gr.Column(scale=1):
            # Output visualizations
            output_points = gr.Image(label="Image with Points", type="pil")
            output_mask = gr.Image(label="Segmentation Result", type="pil")

    with gr.Row():
        with gr.Column():
            # Model parameters
            gr.Markdown("### Model Parameters")
            multimask_output = gr.Checkbox(
                label="Multi-mask Output",
                value=True,
                info="Generate multiple mask candidates and select the best one"
            )

        with gr.Column():
            # Information display
            info_text = gr.Textbox(
                label="Segmentation Info",
                lines=8,
                value="Click on the image to add points. Green = positive (include), Red = negative (exclude)"
            )

    with gr.Row():
        save_status = gr.Textbox(
            label="Save Status",
            lines=3,
            value=f"Masks will be saved to: {OUTPUT_DIR}/"
        )

    # Event handlers
    input_image.upload(
        fn=handle_image_upload,
        inputs=[input_image],
        outputs=[input_image, points_data, output_points, output_mask, info_text]
    )

    input_image.select(
        fn=update_points,
        inputs=[points_data, point_label],
        outputs=[points_data, output_points, info_text]
    )

    segment_btn.click(
        fn=segment_image,
        inputs=[points_data, multimask_output],
        outputs=[output_points, output_mask, info_text]
    )

    clear_btn.click(
        fn=clear_all,
        outputs=[points_data, output_points, output_mask, info_text]
    )

    save_btn.click(
        fn=save_mask,
        outputs=[save_status]
    )


print("\n" + "=" * 80)
print(f"Starting web server on port {port}...")
print("=" * 80)
print(f"\nOpen your browser and go to:")
print(f"  http://localhost:{port}")
print(f"\nor if accessing remotely:")
print(f"  http://<your-server-ip>:{port}")
print(f"\nMasks will be saved to: {OUTPUT_DIR}/")
print("\n" + "=" * 80)

demo.launch(server_name="0.0.0.0", server_port=port, share=False)

