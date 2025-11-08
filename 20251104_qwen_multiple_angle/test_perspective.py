#!/usr/bin/env python3
"""Test perspective change using the Qwen-Image-Edit-2509 model with Multiple-angles LoRA."""

import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

print("=" * 60)
print("Qwen-Image-Edit-2509 with Multiple-angles LoRA Test")
print("=" * 60)

# Load the input image
input_path = "test_output/output_multi.png"
print(f"\nLoading input image: {input_path}")
input_image = Image.open(input_path)
print(f"  Input size: {input_image.size}")

# Load the base model
print("\nLoading Qwen-Image-Edit-2509 model...")
pipeline = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509",
    torch_dtype=torch.bfloat16
)
pipeline.to('cuda:0')
print("  Base model loaded and moved to GPU")

# Load the LoRAs
print("\nLoading LoRAs...")
print("  1. Loading Lightning LoRA (required dependency)...")
pipeline.load_lora_weights(
    "lightx2v/Qwen-Image-Lightning",
    weight_name="Qwen-Image-Edit-Lightning-4steps-V1.0-bf16.safetensors",
    adapter_name="lightning"
)
print("  2. Loading Multiple-angles LoRA...")
pipeline.load_lora_weights(
    "dx8152/Qwen-Edit-2509-Multiple-angles",
    weight_name="镜头转换.safetensors",
    adapter_name="multiple_angles"
)

# Set both LoRAs active with appropriate weights
print("  Setting LoRA adapters...")
pipeline.set_adapters(["lightning", "multiple_angles"], adapter_weights=[1.0, 1.0])
print("  LoRAs loaded successfully!")

# Test different camera angles
test_prompts = [
    ("将镜头向左旋转45度", "output_rotate_left_45.png", "Rotate camera 45 degrees to the left"),
    ("将镜头向右旋转45度", "output_rotate_right_45.png", "Rotate camera 45 degrees to the right"),
    ("将镜头转为俯视", "output_top_down.png", "Turn camera to top-down view"),
    ("将镜头向左移动", "output_move_left.png", "Move camera to the left"),
]

for i, (prompt_cn, output_filename, prompt_en) in enumerate(test_prompts, 1):
    print("\n" + "=" * 60)
    print(f"Test {i}/{len(test_prompts)}: {prompt_en}")
    print("=" * 60)
    print(f"Prompt (Chinese): {prompt_cn}")
    print(f"Prompt (English): {prompt_en}")
    print("Running inference...")

    # Run inference
    output = pipeline(
        image=[input_image],
        prompt=prompt_cn,
        negative_prompt=" ",
        num_inference_steps=40,
        guidance_scale=1.0,
        true_cfg_scale=4.0,
        generator=torch.Generator(device='cuda:0').manual_seed(456 + i)
    ).images[0]

    # Save the output
    output_path = f"test_output/{output_filename}"
    output.save(output_path)
    print(f"  Success! Saved: {output_path}")
    print(f"  Output size: {output.size}")

print("\n" + "=" * 60)
print("All Tests Complete")
print("=" * 60)
print("\nGenerated files:")
for _, output_filename, prompt_en in test_prompts:
    print(f"  - test_output/{output_filename} ({prompt_en})")

