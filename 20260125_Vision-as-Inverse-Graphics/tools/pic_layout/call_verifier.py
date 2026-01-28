#!/usr/bin/env python3
"""Helper script to call Qwen3-VL verifier with target and current layout images."""

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

# Import prompts
from prompts.pic_layout.verifier import pic_layout_verifier_system


def encode_image(image_path):
    """Encode image to base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def call_verifier(target_image_path, current_image_path, target_description, current_code, 
                  model="Qwen3-VL-32B-Instruct", max_tokens=2000):
    """Call Qwen3-VL verifier to compare layouts.
    
    Args:
        target_image_path: Path to target layout image
        current_image_path: Path to current generated layout image
        target_description: Text description of target circuit
        current_code: The gdsfactory code that generated current layout
        model: Model name
        max_tokens: Maximum tokens in response
    
    Returns:
        dict with verifier response
    """
    # Initialize client
    api_key = os.getenv("VLLM_API_KEY")
    client = OpenAI(
        api_key=api_key,
        base_url="http://192.222.54.152:8000/v1"
    )
    
    # Encode images
    print("Encoding images...")
    target_b64 = encode_image(target_image_path)
    current_b64 = encode_image(current_image_path)
    
    # Construct message
    user_message = f"""**Target Description:**
{target_description}

**Current Code:**
```python
{current_code}
```

Please compare the two layout images (first is target, second is current) and provide detailed verification feedback following the specified JSON format.
"""
    
    messages = [
        {"role": "system", "content": pic_layout_verifier_system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_message},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{target_b64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{current_b64}"}}
            ]
        }
    ]
    
    print(f"Calling {model}...")
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.7
    )
    
    content = response.choices[0].message.content
    
    # Try to extract JSON from response
    try:
        # Look for ```json blocks
        if "```json" in content:
            json_start = content.find("```json") + 7
            json_end = content.find("```", json_start)
            json_str = content[json_start:json_end].strip()
            feedback = json.loads(json_str)
        else:
            # Try to parse entire content as JSON
            feedback = json.loads(content)
    except json.JSONDecodeError:
        # If parsing fails, return raw content
        feedback = {"raw_response": content}
    
    return {
        "feedback": feedback,
        "raw_response": content,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
    }


def save_verifier_result(result, output_path):
    """Save verifier result to file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Verifier result saved: {output_path}")


if __name__ == "__main__":
    # Example usage
    print("Qwen3-VL Verifier Test")
    print("=" * 60)
    
    # Check if example files exist
    target_path = "output/example_test/iteration_00/layout_iter00.png"
    current_path = "output/example_test/iteration_00/layout_iter00.png"
    
    if not Path(target_path).exists():
        print(f"Example files not found. Run manual_test_pic_layout.py first.")
        sys.exit(1)
    
    example_description = """
    A Mach-Zehnder Interferometer (MZI) with:
    - 1x2 MMI splitter at input
    - Two arms with 10 µm length difference
    - 2x2 MMI combiner at output
    - Two output ports
    """
    
    example_code = """
import gdsfactory as gf

c = gf.Component("test_mzi")
mzi = c << gf.components.mzi(delta_length=10)
c.add_ports(mzi.ports)
"""
    
    print("\nCalling verifier (using same image as both target and current for demo)...")
    result = call_verifier(
        target_image_path=target_path,
        current_image_path=current_path,
        target_description=example_description,
        current_code=example_code
    )
    
    print(f"\n{'='*60}")
    print("VERIFIER RESPONSE:")
    print(f"{'='*60}")
    print(json.dumps(result['feedback'], indent=2))
    print(f"\nTokens used: {result['usage']['total_tokens']}")
    
    save_verifier_result(result, "output/example_test/iteration_00/verifier_response.json")

