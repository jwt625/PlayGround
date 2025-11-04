#!/usr/bin/env python3
"""Test camera angle change on optical table image."""

import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

print("=" * 60)
print("Optical Table Camera Angle Test")
print("=" * 60)

# Load the input image
input_path = "test_output/optical-table-Zhong2020.png"
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

# Test camera rotation down by 45 degrees
prompt_cn = "将镜头向下旋转45度"
prompt_en = "Rotate the camera down by 45 degrees"

print("\n" + "=" * 60)
print(f"Test: {prompt_en}")
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
    generator=torch.Generator(device='cuda:0').manual_seed(789)
).images[0]

# Save the output
output_path = "test_output/optical_table_rotate_down_45.png"
output.save(output_path)
print(f"  Success! Saved: {output_path}")
print(f"  Output size: {output.size}")

print("\n" + "=" * 60)
print("Test Complete")
print("=" * 60)
print(f"\nGenerated file: {output_path}")
print("The camera has been rotated down by 45 degrees.")

