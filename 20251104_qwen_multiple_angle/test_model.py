#!/usr/bin/env python3
"""
Simple test script to verify Qwen-Image-Edit-2509 model works correctly.
Creates a test image and performs a basic edit operation.
"""

import os
import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import QwenImageEditPlusPipeline


def create_test_image(size=(512, 512), color='lightblue', text='Test Image 1'):
    """Create a simple test image with text."""
    img = Image.new('RGB', size, color=color)
    draw = ImageDraw.Draw(img)
    
    # Draw some shapes
    draw.rectangle([50, 50, 200, 200], fill='red', outline='black', width=3)
    draw.ellipse([250, 250, 450, 450], fill='yellow', outline='black', width=3)
    
    # Add text
    try:
        # Try to use a default font
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
    except:
        font = ImageFont.load_default()
    
    # Get text bounding box for centering
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    position = ((size[0] - text_width) // 2, 20)
    draw.text(position, text, fill='black', font=font)
    
    return img


def main():
    print("=" * 60)
    print("Qwen-Image-Edit-2509 Test Script")
    print("=" * 60)
    
    # Check GPU availability
    print(f"\nGPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    
    # Create output directory
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Create test images
    print("\nCreating test images...")
    test_image1 = create_test_image(color='lightblue', text='Bear 1')
    test_image2 = create_test_image(color='lightgreen', text='Bear 2')
    
    # Save test images
    test_image1.save(f"{output_dir}/input1.png")
    test_image2.save(f"{output_dir}/input2.png")
    print(f"  Saved: {output_dir}/input1.png")
    print(f"  Saved: {output_dir}/input2.png")
    
    # Load model
    print("\nLoading Qwen-Image-Edit-2509 model...")
    print("  This may take a few minutes on first run (downloading model)...")
    
    try:
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit-2509",
            torch_dtype=torch.bfloat16
        )
        print("  Model loaded successfully")
        
        # Move to GPU
        if torch.cuda.is_available():
            pipeline.to('cuda')
            print("  Model moved to GPU")
        else:
            print("  WARNING: Running on CPU (will be slow)")
        
        pipeline.set_progress_bar_config(disable=False)
        
    except Exception as e:
        print(f"\nERROR: Failed to load model: {e}")
        return 1
    
    # Test 1: Single image edit
    print("\n" + "=" * 60)
    print("Test 1: Single Image Edit")
    print("=" * 60)
    
    prompt1 = "A magical wizard bear with a purple hat and glowing staff"
    print(f"Prompt: {prompt1}")
    print("Running inference...")
    
    try:
        with torch.inference_mode():
            output1 = pipeline(
                image=test_image1,
                prompt=prompt1,
                generator=torch.manual_seed(42),
                true_cfg_scale=4.0,
                negative_prompt=" ",
                num_inference_steps=40,
                guidance_scale=1.0,
                num_images_per_prompt=1,
            )
            output_image1 = output1.images[0]
        
        output_path1 = f"{output_dir}/output_single.png"
        output_image1.save(output_path1)
        print(f"  Success! Saved: {output_path1}")
        print(f"  Output size: {output_image1.size}")
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Test 2: Multi-image edit
    print("\n" + "=" * 60)
    print("Test 2: Multi-Image Edit")
    print("=" * 60)
    
    prompt2 = "Two friendly bears having a picnic in a sunny park, one on the left wearing a red scarf, one on the right wearing a blue hat"
    print(f"Prompt: {prompt2}")
    print("Running inference...")
    
    try:
        with torch.inference_mode():
            output2 = pipeline(
                image=[test_image1, test_image2],
                prompt=prompt2,
                generator=torch.manual_seed(123),
                true_cfg_scale=4.0,
                negative_prompt=" ",
                num_inference_steps=40,
                guidance_scale=1.0,
                num_images_per_prompt=1,
            )
            output_image2 = output2.images[0]
        
        output_path2 = f"{output_dir}/output_multi.png"
        output_image2.save(output_path2)
        print(f"  Success! Saved: {output_path2}")
        print(f"  Output size: {output_image2.size}")
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print("All tests passed successfully!")
    print(f"\nGenerated files in '{output_dir}':")
    print(f"  - input1.png (test input)")
    print(f"  - input2.png (test input)")
    print(f"  - output_single.png (single image edit)")
    print(f"  - output_multi.png (multi-image edit)")
    print("\nModel is working correctly!")
    
    # Cleanup
    del pipeline
    torch.cuda.empty_cache()
    
    return 0


if __name__ == "__main__":
    exit(main())

