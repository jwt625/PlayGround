#!/usr/bin/env python3
"""
Test script to verify timestamp tracking and tokens/sec calculation
"""

import os
import json
import time
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

KIMI_API_BASE = os.getenv("KIMI_API_BASE")
KIMI_API_KEY = os.getenv("KIMI_API_KEY", "")
KIMI_MODEL_ID = os.getenv("KIMI_MODEL_ID", "kimi-k2")

if not KIMI_API_BASE:
    raise ValueError("KIMI_API_BASE environment variable is required")

def test_timing():
    """Test timing tracking with a complex problem that generates >10k tokens"""
    client = OpenAI(api_key=KIMI_API_KEY or "dummy", base_url=KIMI_API_BASE)

    # Complex multi-step geometry problem that requires extensive reasoning
    problem = """Consider a regular dodecahedron with edge length 1.

1. First, calculate the exact coordinates of all 20 vertices of the dodecahedron when centered at the origin with one vertex at (φ, 1, 0), where φ is the golden ratio.

2. Next, inscribe a sphere inside the dodecahedron. Calculate the exact radius of this insphere.

3. Now, consider all 30 edges of the dodecahedron. For each edge, construct a plane perpendicular to that edge passing through its midpoint. These 30 planes divide 3D space into regions. Calculate the total number of bounded regions created.

4. Select the 12 pentagonal faces of the dodecahedron. For each face, calculate the center point and construct a pyramid with apex at the origin and base as that pentagonal face. Calculate the total volume of all 12 pyramids combined.

5. Consider the dual polyhedron of the dodecahedron (which is an icosahedron). Calculate the edge length of this dual icosahedron.

6. Now imagine truncating each of the 20 vertices of the original dodecahedron by cutting off a small regular pentagonal pyramid at each vertex, such that the truncation removes exactly 1/10 of the distance from each vertex to the center. Describe the resulting polyhedron: how many faces does it have, how many edges, and how many vertices? What types of faces does it have?

7. Calculate the surface area of this truncated dodecahedron.

8. Calculate the volume of this truncated dodecahedron.

9. If you were to place identical spheres at each vertex of the original dodecahedron such that neighboring spheres are tangent to each other, what would be the radius of each sphere?

10. Finally, consider the convex hull of the centers of all faces of the original dodecahedron. What polyhedron is formed, and what is its volume?

Show all your work step by step, including all intermediate calculations, coordinate geometry, trigonometry, and algebraic manipulations. Verify your answers using multiple methods where possible. Provide your final answer for the volume of the truncated dodecahedron in \\boxed{} format."""

    print("="*80)
    print("TIMING TEST - COMPLEX PROBLEM (TARGET: >10K TOKENS)")
    print("="*80)
    print(f"\nProblem length: {len(problem)} characters")
    print(f"\nGenerating 2 completions with max_tokens=64000...")
    print("Expected: 10-step problem, ~2-4k tokens per step = 20-40k tokens total")

    # Track timing
    start_time = time.time()
    start_timestamp = datetime.now().isoformat()

    response = client.chat.completions.create(
        model=KIMI_MODEL_ID,
        messages=[{"role": "user", "content": problem}],
        max_tokens=64000,  # Set to 40k to allow full solution with verification
        temperature=0.8,
        n=2,  # Reduced to 2 to save time
        logprobs=True,
        top_logprobs=20
    )
    
    end_time = time.time()
    end_timestamp = datetime.now().isoformat()
    elapsed_time = end_time - start_time
    
    print(f"\n{'='*80}")
    print("TIMING RESULTS")
    print("="*80)
    print(f"Start time: {start_timestamp}")
    print(f"End time: {end_timestamp}")
    print(f"Duration: {elapsed_time:.2f} seconds")
    
    print(f"\n{'='*80}")
    print("PER-TRACE RESULTS")
    print("="*80)
    
    total_tokens = 0
    for i, choice in enumerate(response.choices):
        content = choice.message.content or ""
        reasoning = getattr(choice.message, 'reasoning', '') or ""
        
        # Count tokens from logprobs
        num_tokens = 0
        if choice.logprobs and hasattr(choice.logprobs, 'content') and choice.logprobs.content:
            num_tokens = len(choice.logprobs.content)
        
        total_tokens += num_tokens
        tokens_per_sec = num_tokens / elapsed_time if elapsed_time > 0 else 0
        
        print(f"\nTrace {i}:")
        print(f"  Tokens: {num_tokens}")
        print(f"  Tokens/sec: {tokens_per_sec:.2f}")
        print(f"  Reasoning length: {len(reasoning)} chars")
        print(f"  Content length: {len(content)} chars")
        print(f"  Content preview: {content[:100]}...")
    
    num_traces = len(response.choices)

    print(f"\n{'='*80}")
    print("AGGREGATE STATS")
    print("="*80)
    print(f"Total tokens (all traces): {total_tokens}")
    print(f"Average tokens per trace: {total_tokens / num_traces:.1f}")
    print(f"Total tokens/sec: {total_tokens / elapsed_time:.2f}")
    print(f"Time per trace (sequential): {elapsed_time / num_traces:.2f}s")

    # Check if any trace exceeded 10k tokens
    max_tokens_trace = max(
        len(choice.logprobs.content) if choice.logprobs and hasattr(choice.logprobs, 'content') and choice.logprobs.content else 0
        for choice in response.choices
    )
    print(f"\nMax tokens in a single trace: {max_tokens_trace}")
    if max_tokens_trace >= 10000:
        print("✓ SUCCESS: At least one trace exceeded 10,000 tokens!")
    else:
        print(f"✗ No trace exceeded 10k tokens. Max was {max_tokens_trace}.")
    
    # Save results to JSON
    results = {
        "timing": {
            "start": start_timestamp,
            "end": end_timestamp,
            "duration_seconds": elapsed_time
        },
        "traces": [
            {
                "trace_index": i,
                "num_tokens": len(choice.logprobs.content) if choice.logprobs and hasattr(choice.logprobs, 'content') and choice.logprobs.content else 0,
                "tokens_per_second": (len(choice.logprobs.content) / elapsed_time) if choice.logprobs and hasattr(choice.logprobs, 'content') and choice.logprobs.content and elapsed_time > 0 else 0,
                "reasoning_length": len(getattr(choice.message, 'reasoning', '') or ""),
                "content_length": len(choice.message.content or "")
            }
            for i, choice in enumerate(response.choices)
        ],
        "aggregate": {
            "total_tokens": total_tokens,
            "avg_tokens_per_trace": total_tokens / num_traces,
            "total_tokens_per_second": total_tokens / elapsed_time if elapsed_time > 0 else 0,
            "max_tokens_single_trace": max_tokens_trace,
            "exceeded_10k": max_tokens_trace >= 10000
        }
    }
    
    output_file = "timing_test_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    test_timing()

