#!/usr/bin/env python3
"""Analyze target image with Qwen3-VL to understand the layout."""

import sys
import os
import json
import base64
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add gdsfactory venv to path if specified
venv_python = os.getenv('GDSFACTORY_VENV_PYTHON')
if venv_python:
    venv_site_packages = Path(venv_python).parent.parent / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages'
    if venv_site_packages.exists():
        sys.path.insert(0, str(venv_site_packages))


def encode_image(image_path):
    """Encode image to base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def analyze_target_layout(image_path):
    """Analyze target layout image with Qwen3-VL."""
    
    # Initialize client
    api_key = os.getenv("VLLM_API_KEY")
    client = OpenAI(
        api_key=api_key,
        base_url="http://192.222.54.152:8000/v1"
    )
    
    # Encode image
    print(f"Analyzing: {image_path}")
    image_b64 = encode_image(image_path)
    
    # Analysis prompt
    analysis_prompt = """Please analyze this photonic integrated circuit (PIC) layout image in detail.

Identify and describe:
1. **Components**: What types of photonic components do you see? (e.g., waveguides, bends, splitters, couplers, rings, etc.)
2. **Topology**: How are components connected? What is the signal flow path?
3. **Dimensions**: Estimate relative sizes and proportions of components
4. **Layout structure**: Overall organization and arrangement
5. **Key features**: Any distinctive characteristics or design patterns

Provide a detailed technical description that would help a layout engineer recreate this circuit."""
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": analysis_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
            ]
        }
    ]
    
    print("Calling Qwen3-VL for analysis...")
    response = client.chat.completions.create(
        model="Qwen3-VL-32B-Instruct",
        messages=messages,
        max_tokens=2000,
        temperature=0.3
    )
    
    analysis = response.choices[0].message.content
    
    print(f"\n{'='*80}")
    print("TARGET LAYOUT ANALYSIS:")
    print(f"{'='*80}")
    print(analysis)
    print(f"\n{'='*80}")
    print(f"Tokens used: {response.usage.total_tokens}")
    
    return {
        "analysis": analysis,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
    }


if __name__ == "__main__":
    target_path = "source/chip1.png"
    
    if not Path(target_path).exists():
        print(f"Error: Target image not found: {target_path}")
        sys.exit(1)
    
    result = analyze_target_layout(target_path)
    
    # Save analysis
    output_path = Path("output/chip1_iterations/target_analysis.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nAnalysis saved: {output_path}")

