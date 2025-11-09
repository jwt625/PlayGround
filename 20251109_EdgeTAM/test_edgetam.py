#!/usr/bin/env python3
"""
Test script for EdgeTAM model.
This script demonstrates basic image segmentation using EdgeTAM.
"""

import torch
from PIL import Image
import requests
from transformers import Sam2Processor, EdgeTamModel
import time

MODEL_NAME = "yonigozlan/EdgeTAM-hf"

def test_basic_segmentation():
    """Test basic image segmentation with EdgeTAM."""
    print("=" * 60)
    print("EdgeTAM Basic Segmentation Test")
    print("=" * 60)
    
    # Check device availability
    if torch.backends.mps.is_available():
        device = "mps"  # Apple Silicon GPU
        print(f"✓ Using device: MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = "cuda"
        print(f"✓ Using device: CUDA GPU")
    else:
        device = "cpu"
        print(f"✓ Using device: CPU")
    
    print(f"\nPyTorch version: {torch.__version__}")
    print()
    
    # Load model and processor
    print("Loading EdgeTAM model and processor...")
    start_time = time.time()
    model = EdgeTamModel.from_pretrained(MODEL_NAME, local_files_only=True).to(device)
    processor = Sam2Processor.from_pretrained(MODEL_NAME, local_files_only=True)
    load_time = time.time() - start_time
    print(f"✓ Model loaded in {load_time:.2f} seconds")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print()
    
    # Load test image
    print("Loading test image...")
    image_path = "test_images/truck.jpg"
    raw_image = Image.open(image_path).convert("RGB")
    print(f"✓ Image loaded: {raw_image.size[0]}x{raw_image.size[1]} pixels")
    print()
    
    # Test 1: Single point segmentation
    print("Test 1: Single point click segmentation")
    print("-" * 60)
    input_points = [[[[500, 375]]]]  # Single point click
    input_labels = [[[1]]]  # Positive click
    
    inputs = processor(
        images=raw_image, 
        input_points=input_points, 
        input_labels=input_labels, 
        return_tensors="pt"
    ).to(device)
    
    print(f"  Input point: {input_points[0][0][0]}")
    print(f"  Running inference...")
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    inference_time = time.time() - start_time
    
    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(), 
        inputs["original_sizes"]
    )[0]
    
    print(f"  ✓ Inference completed in {inference_time*1000:.2f} ms")
    print(f"  Generated {masks.shape[1]} masks with shape {masks.shape}")
    print(f"  IoU scores: {outputs.iou_scores.cpu().numpy()}")
    print()
    
    # Test 2: Multiple points for refinement
    print("Test 2: Multiple points for refinement")
    print("-" * 60)
    input_points = [[[[500, 375], [1125, 625]]]]  # Two points
    input_labels = [[[1, 1]]]  # Both positive
    
    inputs = processor(
        images=raw_image, 
        input_points=input_points, 
        input_labels=input_labels, 
        return_tensors="pt"
    ).to(device)
    
    print(f"  Input points: {input_points[0][0]}")
    print(f"  Running inference...")
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    inference_time = time.time() - start_time
    
    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(), 
        inputs["original_sizes"]
    )[0]
    
    print(f"  ✓ Inference completed in {inference_time*1000:.2f} ms")
    print(f"  Generated {masks.shape[1]} masks with shape {masks.shape}")
    print(f"  IoU scores: {outputs.iou_scores.cpu().numpy()}")
    print()
    
    # Test 3: Bounding box input
    print("Test 3: Bounding box segmentation")
    print("-" * 60)
    input_boxes = [[[75, 275, 1725, 850]]]  # [x_min, y_min, x_max, y_max]
    
    inputs = processor(
        images=raw_image, 
        input_boxes=input_boxes, 
        return_tensors="pt"
    ).to(device)
    
    print(f"  Bounding box: {input_boxes[0][0]}")
    print(f"  Running inference...")
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    inference_time = time.time() - start_time
    
    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(), 
        inputs["original_sizes"]
    )[0]
    
    print(f"  ✓ Inference completed in {inference_time*1000:.2f} ms")
    print(f"  Generated {masks.shape[1]} masks with shape {masks.shape}")
    print(f"  IoU scores: {outputs.iou_scores.cpu().numpy()}")
    print()
    
    # Test 4: Multiple objects
    print("Test 4: Multiple objects segmentation")
    print("-" * 60)
    input_points = [[[[500, 375]], [[650, 750]]]]  # Two objects
    input_labels = [[[1], [1]]]  # Both positive
    
    inputs = processor(
        images=raw_image, 
        input_points=input_points, 
        input_labels=input_labels, 
        return_tensors="pt"
    ).to(device)
    
    print(f"  Object 1 point: {input_points[0][0][0]}")
    print(f"  Object 2 point: {input_points[0][1][0]}")
    print(f"  Running inference...")
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)
    inference_time = time.time() - start_time
    
    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(), 
        inputs["original_sizes"]
    )[0]
    
    print(f"  ✓ Inference completed in {inference_time*1000:.2f} ms")
    print(f"  Generated masks for {masks.shape[0]} objects")
    print(f"  Mask shape: {masks.shape}")
    print()
    
    print("=" * 60)
    print("All tests completed successfully! ✓")
    print("=" * 60)
    print(f"\nEdgeTAM is working correctly on your {device.upper()} device.")
    print(f"Average inference time: ~{inference_time*1000:.2f} ms per image")

if __name__ == "__main__":
    test_basic_segmentation()

