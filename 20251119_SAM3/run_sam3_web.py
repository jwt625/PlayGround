#!/usr/bin/env python3
"""
Web-based SAM2 point selection + SAM3D Objects reconstruction.
Opens a web interface where you can click on objects to segment them.

Usage:
    python run_sam3_web.py <image_path> [--port 7860]

Example:
    python run_sam3_web.py IMG_8020.JPG
    python run_sam3_web.py IMG_8020.JPG --port 8080
"""
import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image
import gradio as gr
import json

# Set environment variables
os.environ["CUDA_HOME"] = "/usr/local/cuda"
os.environ["LIDRA_SKIP_INIT"] = "true"

# Add notebook to path
sys.path.append("sam-3d-objects/notebook")
from inference import Inference

# Parse command line arguments
if len(sys.argv) < 2:
    print("Error: Image path required")
    print("Usage: python run_sam3_web.py <image_path> [--port 7860]")
    sys.exit(1)

IMAGE_PATH = sys.argv[1]
input_name = Path(IMAGE_PATH).stem

# Parse port
port = 7860
if "--port" in sys.argv:
    port_idx = sys.argv.index("--port")
    if port_idx + 1 < len(sys.argv):
        port = int(sys.argv[port_idx + 1])

OUTPUT_DIR = f"{input_name}_objects_web"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("SAM2 Web-Based Point Selection + SAM 3D Objects")
print("=" * 80)
print(f"\nImage: {IMAGE_PATH}")
print(f"Output directory: {OUTPUT_DIR}")

# Load the image
print(f"\nLoading image...")
image = Image.open(IMAGE_PATH).convert("RGB")
image_np = np.array(image)
print(f"Image size: {image.size}")

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
    os.system(f"wget -P sam-3d-objects/checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -O {checkpoint_path}")

sam2_model = build_sam2(model_cfg, checkpoint_path, device="cuda")
predictor = SAM2ImagePredictor(sam2_model)
predictor.set_image(image_np)
print("SAM2 model loaded!")

# Load SAM3D Objects
print("\nLoading SAM 3D Objects model...")
CONFIG_PATH = "sam-3d-objects/checkpoints/hf/pipeline.yaml"
sam3d_inference = Inference(CONFIG_PATH, compile=False)
print("SAM 3D Objects model loaded!")

# Global state
selected_points = []
generated_masks = []
reconstruction_status = []

def process_click(image_with_points, evt: gr.SelectData):
    """Handle click events on the image"""
    global selected_points, generated_masks
    
    x, y = evt.index[0], evt.index[1]
    selected_points.append([x, y])
    
    print(f"\nPoint {len(selected_points)}: ({x}, {y})")
    
    # Generate mask for this point
    print(f"  Generating mask...")
    masks, scores, logits = predictor.predict(
        point_coords=np.array([selected_points[-1]]),
        point_labels=np.array([1]),
        multimask_output=True,
    )
    
    best_idx = np.argmax(scores)
    mask = masks[best_idx]
    score = scores[best_idx]
    
    generated_masks.append(mask)
    print(f"  Mask generated: area={mask.sum():,} pixels, score={score:.3f}")
    
    # Create visualization
    overlay = image_np.copy().astype(float) / 255.0

    # Draw all masks with different colors
    import matplotlib.pyplot as plt
    colors = plt.cm.hsv(np.linspace(0, 1, len(generated_masks) + 1))
    for i, m in enumerate(generated_masks):
        color = colors[i][:3]
        mask_bool = m.astype(bool)  # Convert to boolean for indexing
        overlay[mask_bool] = overlay[mask_bool] * 0.5 + color * 0.5
    
    # Draw all points
    overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(overlay_img)
    
    for i, pt in enumerate(selected_points):
        # Draw crosshair
        size = 15
        draw.line([(pt[0]-size, pt[1]), (pt[0]+size, pt[1])], fill='red', width=3)
        draw.line([(pt[0], pt[1]-size), (pt[0], pt[1]+size)], fill='red', width=3)
        # Draw number
        draw.text((pt[0]+20, pt[1]-20), str(i+1), fill='yellow', stroke_width=2, stroke_fill='black')
    
    status = f"Selected {len(selected_points)} objects. Click on more objects or click 'Generate 3D Reconstructions' when done."
    
    return overlay_img, status

def clear_points():
    """Clear all selected points"""
    global selected_points, generated_masks, reconstruction_status
    selected_points = []
    generated_masks = []
    reconstruction_status = []
    return image, "All points cleared. Click on objects to select them."

def generate_reconstructions():
    """Generate 3D reconstructions for all selected objects"""
    global reconstruction_status
    
    if len(selected_points) == 0:
        return "No objects selected. Please click on objects first."
    
    reconstruction_status = []
    status_text = f"Generating 3D reconstructions for {len(selected_points)} objects...\n\n"
    
    for i, (point, mask) in enumerate(zip(selected_points, generated_masks)):
        print(f"\nProcessing object {i+1}/{len(selected_points)}...")
        output_path = os.path.join(OUTPUT_DIR, f"object_{i:03d}.ply")
        mask_path = os.path.join(OUTPUT_DIR, f"mask_{i:03d}.png")
        
        try:
            # Save mask
            Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
            
            # Generate 3D reconstruction
            output = sam3d_inference(image_np, mask, seed=42)
            output["gs"].save_ply(output_path)
            
            reconstruction_status.append(f"✓ Object {i+1}: {output_path}")
            status_text += f"✓ Object {i+1}: Saved to {output_path}\n"
            print(f"  ✓ Saved to {output_path}")
        except Exception as e:
            reconstruction_status.append(f"✗ Object {i+1}: Failed - {str(e)}")
            status_text += f"✗ Object {i+1}: Failed - {str(e)}\n"
            print(f"  ✗ Failed: {e}")
    
    status_text += f"\n\nAll files saved to: {OUTPUT_DIR}/"
    status_text += f"\nYou can view .ply files with CloudCompare, MeshLab, Blender, or online viewers."
    
    return status_text

# Create Gradio interface
import matplotlib.pyplot as plt

with gr.Blocks(title="SAM2 + SAM3D Objects") as demo:
    gr.Markdown("# SAM2 Point Selection + SAM 3D Objects Reconstruction")
    gr.Markdown(f"**Image:** {IMAGE_PATH} | **Size:** {image.size[0]}x{image.size[1]}")
    gr.Markdown("**Instructions:** Click on objects in the image to select them. Each click will segment one object. When done, click 'Generate 3D Reconstructions'.")
    
    with gr.Row():
        with gr.Column():
            image_display = gr.Image(value=image, label="Click on objects to select", type="pil", interactive=True)
            status_text = gr.Textbox(label="Status", value="Click on objects to select them.", lines=3)
        
        with gr.Column():
            gr.Markdown("### Controls")
            clear_btn = gr.Button("Clear All Points", variant="secondary")
            generate_btn = gr.Button("Generate 3D Reconstructions", variant="primary", size="lg")
            result_text = gr.Textbox(label="Reconstruction Results", lines=15)
    
    # Event handlers
    image_display.select(process_click, inputs=[image_display], outputs=[image_display, status_text])
    clear_btn.click(clear_points, outputs=[image_display, status_text])
    generate_btn.click(generate_reconstructions, outputs=[result_text])

print("\n" + "=" * 80)
print(f"Starting web server on port {port}...")
print("=" * 80)
print(f"\nOpen your browser and go to:")
print(f"  http://localhost:{port}")
print(f"\nor if accessing remotely:")
print(f"  http://<your-server-ip>:{port}")
print("\n" + "=" * 80)

demo.launch(server_name="0.0.0.0", server_port=port, share=False)

