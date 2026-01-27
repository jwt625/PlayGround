#!/usr/bin/env python3
"""Detailed multi-query analysis for chip2.png - complex layout with multiple components."""

import sys
import os
import json
import base64
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def encode_image(image_path):
    """Encode image to base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def query_vlm(image_b64, prompt, model="Qwen3-VL-32B-Instruct"):
    """Query VLM with specific prompt."""
    api_key = os.getenv("VLLM_API_KEY")
    client = OpenAI(api_key=api_key, base_url="http://192.222.54.152:8000/v1")
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
        ]
    }]
    
    response = client.chat.completions.create(
        model=model, messages=messages, max_tokens=2000, temperature=0.3
    )
    
    return response.choices[0].message.content, response.usage.total_tokens


def analyze_chip2_detailed(image_path):
    """Multi-query analysis focusing on different aspects."""
    
    print(f"Analyzing: {image_path}")
    image_b64 = encode_image(image_path)
    
    results = {}
    
    # Query 1: Component count and identification
    print("\n[1/4] Identifying components...")
    prompt1 = """Focus on identifying and counting all distinct components in this PIC layout.

List:
1. How many ring resonators do you see? Describe their positions (left, center, right).
2. How many waveguide bends? Estimate bend angles.
3. Are there splitters or couplers? What type and where?
4. Are there grating couplers or edge couplers? How many and where?
5. Are there metal electrodes or contacts? Describe their shapes and positions.

Be specific with counts and positions."""
    
    response1, tokens1 = query_vlm(image_b64, prompt1)
    results['component_identification'] = {'response': response1, 'tokens': tokens1}
    print(f"Tokens: {tokens1}")
    
    # Query 2: Dimensional estimates
    print("\n[2/4] Estimating dimensions...")
    prompt2 = """Focus on estimating dimensions of key components.

For each component type, estimate:
1. Ring resonators: radius in micrometers
2. Waveguide widths
3. Coupling gaps between rings and bus waveguides
4. Bend radii
5. Splitter/coupler dimensions
6. Grating coupler sizes
7. Overall layout dimensions (width x height)

Provide numerical estimates in micrometers."""
    
    response2, tokens2 = query_vlm(image_b64, prompt2)
    results['dimensional_estimates'] = {'response': response2, 'tokens': tokens2}
    print(f"Tokens: {tokens2}")
    
    # Query 3: Topology and connectivity
    print("\n[3/4] Analyzing topology...")
    prompt3 = """Focus on the signal flow and connectivity.

Describe:
1. Where does the signal enter? (left, top, bottom?)
2. Trace the main bus waveguide path
3. How are the rings connected to the bus? (side-coupled, all-pass, add-drop?)
4. What happens after the rings? Where does the signal go?
5. How many output ports are there?
6. Are components in series or parallel?

Draw a simple signal flow diagram in text."""
    
    response3, tokens3 = query_vlm(image_b64, prompt3)
    results['topology_analysis'] = {'response': response3, 'tokens': tokens3}
    print(f"Tokens: {tokens3}")
    
    # Query 4: Critical design details
    print("\n[4/4] Extracting critical details...")
    prompt4 = """Focus on details needed for accurate recreation.

Specify:
1. Ring-to-bus coupling gap (critical for coupling strength)
2. Are the rings identical or different sizes?
3. Electrode placement relative to rings
4. Waveguide routing: Manhattan (90° only) or curved?
5. Any symmetry in the layout?
6. Spacing between components
7. Any visible layer differences (colors/shades)?

Prioritize information needed for gdsfactory code generation."""
    
    response4, tokens4 = query_vlm(image_b64, prompt4)
    results['critical_details'] = {'response': response4, 'tokens': tokens4}
    print(f"Tokens: {tokens4}")
    
    # Summary
    total_tokens = tokens1 + tokens2 + tokens3 + tokens4
    results['total_tokens'] = total_tokens
    
    print(f"\n{'='*80}")
    print(f"Total tokens used: {total_tokens}")
    print(f"{'='*80}")
    
    return results


if __name__ == "__main__":
    target_path = "source/chip2.png"
    
    if not Path(target_path).exists():
        print(f"Error: Target image not found: {target_path}")
        sys.exit(1)
    
    results = analyze_chip2_detailed(target_path)
    
    # Save detailed analysis
    output_path = Path("output/chip2_iterations/target_analysis_detailed.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed analysis saved: {output_path}")
    
    # Print all responses
    print(f"\n{'='*80}")
    print("DETAILED ANALYSIS RESULTS:")
    print(f"{'='*80}")
    
    for key, value in results.items():
        if key != 'total_tokens':
            print(f"\n{key.upper().replace('_', ' ')}:")
            print("-" * 80)
            print(value['response'])

