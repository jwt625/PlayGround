#!/usr/bin/env python3
"""
Gradio application for EdgeTAM interactive segmentation.
Provides a user-friendly interface for interactive image segmentation using EdgeTAM.
"""

import gradio as gr
import torch
import numpy as np
from PIL import Image, ImageDraw
from transformers import Sam2Processor, EdgeTamModel
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import io

MODEL_NAME = "yonigozlan/EdgeTAM-hf"

# Global variables for model and processor
model = None
processor = None
device = None


def initialize_model():
    """Initialize the EdgeTAM model and processor."""
    global model, processor, device
    
    if model is not None:
        return "Model already loaded"
    
    # Determine device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    try:
        model = EdgeTamModel.from_pretrained(MODEL_NAME, local_files_only=True).to(device)
        processor = Sam2Processor.from_pretrained(MODEL_NAME, local_files_only=True)
        return f"✓ Model loaded successfully on {device.upper()}"
    except Exception as e:
        return f"Error loading model: {str(e)}"


def create_overlay(image, mask, alpha=0.5, color=[30, 144, 255]):
    """Create an overlay of the mask on the original image."""
    # Convert image to numpy array
    img_array = np.array(image)
    
    # Create colored mask
    colored_mask = np.zeros_like(img_array)
    colored_mask[mask > 0] = color
    
    # Blend image and mask
    overlay = img_array.copy()
    overlay = (overlay * (1 - alpha) + colored_mask * alpha).astype(np.uint8)
    
    return Image.fromarray(overlay)


def visualize_points(image, points, labels):
    """Visualize points on the image."""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    for point, label in zip(points, labels):
        x, y = point
        # Positive points in green, negative in red
        color = (0, 255, 0) if label == 1 else (255, 0, 0)
        radius = 8
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color, outline="white", width=2)
    
    return img_copy


def segment_image(image, points_data, multimask_output, use_box):
    """
    Perform segmentation on the image using the provided points or box.
    
    Args:
        image: PIL Image
        points_data: Dictionary containing point coordinates and labels
        multimask_output: Whether to output multiple masks
        use_box: Whether to use bounding box mode
    
    Returns:
        Tuple of (annotated image, mask overlay, info text)
    """
    if model is None:
        return None, None, "Please initialize the model first!"
    
    if image is None:
        return None, None, "Please upload an image first!"
    
    # Parse points data
    if not points_data or "points" not in points_data or len(points_data["points"]) == 0:
        return image, None, "Please click on the image to add points!"
    
    points = points_data["points"]
    
    # Extract coordinates and labels
    coords = [[p[0], p[1]] for p in points]
    labels = [p[2] for p in points]  # 1 for positive, 0 for negative
    
    try:
        if use_box and len(coords) >= 2:
            # Use bounding box mode - create box from first two points
            x_coords = [p[0] for p in coords[:2]]
            y_coords = [p[1] for p in coords[:2]]
            box = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
            
            inputs = processor(
                images=image,
                input_boxes=[[[box]]],
                return_tensors="pt"
            ).to(device)
            
            info = f"Using bounding box: {box}\n"
        else:
            # Use point prompts
            input_points = [[coords]]
            input_labels = [[labels]]
            
            inputs = processor(
                images=image,
                input_points=input_points,
                input_labels=input_labels,
                return_tensors="pt"
            ).to(device)
            
            info = f"Using {len(coords)} point(s)\n"
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs, multimask_output=multimask_output)
        
        # Post-process masks
        masks = processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"]
        )[0]
        
        # Get IoU scores
        iou_scores = outputs.iou_scores.cpu().numpy()[0]
        
        # Select best mask (highest IoU)
        best_mask_idx = np.argmax(iou_scores)
        best_mask = masks[0, best_mask_idx].numpy()
        
        # Create visualizations
        img_with_points = visualize_points(image, coords, labels)
        mask_overlay = create_overlay(image, best_mask)
        
        # Add info
        info += f"Generated {masks.shape[1]} mask(s)\n"
        info += f"IoU scores: {', '.join([f'{score:.3f}' for score in iou_scores])}\n"
        info += f"Best mask index: {best_mask_idx} (IoU: {iou_scores[best_mask_idx]:.3f})"
        
        return img_with_points, mask_overlay, info
        
    except Exception as e:
        return image, None, f"Error during segmentation: {str(e)}"


def clear_points():
    """Clear all points and reset the interface."""
    return None, None, None, "Points cleared. Upload an image and click to add points."


# Create Gradio interface
with gr.Blocks(title="EdgeTAM Interactive Segmentation") as demo:
    gr.Markdown("""
    # 🎯 EdgeTAM Interactive Segmentation
    
    Interactive image segmentation using [EdgeTAM](https://huggingface.co/yonigozlan/EdgeTAM-hf) - 
    a lightweight variant of SAM 2 optimized for on-device execution.
    
    ## How to use:
    1. Click "Initialize Model" to load EdgeTAM
    2. Upload an image
    3. Click on the image to add positive points (green) or negative points (red)
    4. Adjust parameters if needed
    5. Click "Segment" to generate the mask
    
    **Point Labels:** 
    - Positive points (1): Click on the object you want to segment
    - Negative points (0): Click on areas you want to exclude
    """)
    
    with gr.Row():
        with gr.Column():
            init_btn = gr.Button("Initialize Model", variant="primary")
            init_status = gr.Textbox(label="Model Status", value="Model not loaded")
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input image with interactive points
            input_image = gr.Image(
                label="Input Image (Click to add points)",
                type="pil",
                sources=["upload"],
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
                segment_btn = gr.Button("Segment", variant="primary")
                clear_btn = gr.Button("Clear Points")
        
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
                info="Generate multiple mask candidates"
            )
            use_box = gr.Checkbox(
                label="Use Bounding Box Mode",
                value=False,
                info="Use first two points as bounding box corners"
            )
        
        with gr.Column():
            # Information display
            info_text = gr.Textbox(
                label="Segmentation Info",
                lines=5,
                value="Upload an image and click to add points"
            )
    
    # Event handlers
    def update_points(image, current_points, label, evt: gr.SelectData):
        """Update points when user clicks on image."""
        if image is None:
            return current_points, None, "Please upload an image first!"
        
        x, y = evt.index[0], evt.index[1]
        
        if current_points is None:
            current_points = {"points": []}
        
        current_points["points"].append([x, y, label])
        
        # Visualize points
        coords = [[p[0], p[1]] for p in current_points["points"]]
        labels = [p[2] for p in current_points["points"]]
        img_with_points = visualize_points(image, coords, labels)
        
        info = f"Added point at ({x}, {y}) with label {label}\n"
        info += f"Total points: {len(current_points['points'])}"
        
        return current_points, img_with_points, info
    
    def clear_all():
        """Clear all points and outputs."""
        return {"points": []}, None, None, "Points cleared. Upload an image and click to add points."
    
    # Connect event handlers
    init_btn.click(
        fn=initialize_model,
        outputs=init_status
    )
    
    input_image.select(
        fn=update_points,
        inputs=[input_image, points_data, point_label],
        outputs=[points_data, output_points, info_text]
    )
    
    segment_btn.click(
        fn=segment_image,
        inputs=[input_image, points_data, multimask_output, use_box],
        outputs=[output_points, output_mask, info_text]
    )
    
    clear_btn.click(
        fn=clear_all,
        outputs=[points_data, output_points, output_mask, info_text]
    )


if __name__ == "__main__":
    demo.launch(share=False)

