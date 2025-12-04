#!/usr/bin/env python3
"""
Test script to verify timestamp tracking and tokens/sec calculation
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

KIMI_API_BASE = os.getenv("KIMI_API_BASE")
KIMI_API_KEY = os.getenv("KIMI_API_KEY", "")
KIMI_MODEL_ID = os.getenv("KIMI_MODEL_ID", "kimi-k2")

if not KIMI_API_BASE:
    raise ValueError("KIMI_API_BASE environment variable is required")

def test_timing(n_completions: int = 2, verbose: bool = True, output_file: Optional[str] = None) -> Dict[str, Any]:
    """Test timing tracking with a problem that generates ~10k tokens

    Args:
        n_completions: Number of completions to generate
        verbose: Whether to print progress and results
        output_file: Optional output filename. If None, auto-generates based on n and timestamp

    Returns:
        Dictionary containing timing results and trace statistics
    """
    client = OpenAI(api_key=KIMI_API_KEY or "dummy", base_url=KIMI_API_BASE)

    # Simpler problem targeting ~10k tokens total
    problem = """Solve the following math problem step by step. Show all your reasoning and calculations.

Problem: A rectangular garden has a length that is 3 meters more than twice its width. The perimeter of the garden is 54 meters.

1. Find the width and length of the garden.
2. Calculate the area of the garden.
3. If a diagonal path is built across the garden, what is the length of this path?
4. If the garden is divided into 6 equal rectangular plots, what are the dimensions of each plot?
5. If a fence costs $15 per meter, what is the total cost to fence the entire garden?

Show all your work step by step, including equations, substitutions, and calculations. Provide your final answers in \\boxed{} format."""

    if verbose:
        print("="*80)
        print(f"TIMING TEST - n={n_completions} COMPLETIONS")
        print("="*80)
        print(f"\nProblem length: {len(problem)} characters")
        print(f"\nGenerating {n_completions} completions with max_tokens=64000...")

    # Track timing
    start_time = time.time()
    start_timestamp = datetime.now().isoformat()

    response = client.chat.completions.create(
        model=KIMI_MODEL_ID,
        messages=[{"role": "user", "content": problem}],
        max_tokens=64000,
        temperature=0.8,
        n=n_completions,
        logprobs=True,
        top_logprobs=20
    )
    
    end_time = time.time()
    end_timestamp = datetime.now().isoformat()
    elapsed_time = end_time - start_time

    if verbose:
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
    trace_data = []

    for i, choice in enumerate(response.choices):
        content = choice.message.content or ""
        reasoning = getattr(choice.message, 'reasoning', '') or ""

        # Count tokens from logprobs
        num_tokens = 0
        reasoning_tokens = []
        if choice.logprobs and hasattr(choice.logprobs, 'content') and choice.logprobs.content:
            num_tokens = len(choice.logprobs.content)
            # Extract reasoning tokens (tokens from the reasoning field)
            reasoning_tokens = [
                {
                    "token": token_data.token,
                    "logprob": token_data.logprob
                }
                for token_data in choice.logprobs.content
            ]

        total_tokens += num_tokens
        tokens_per_sec = num_tokens / elapsed_time if elapsed_time > 0 else 0

        trace_data.append({
            "trace_index": i,
            "num_tokens": num_tokens,
            "tokens_per_second": tokens_per_sec,
            "reasoning_length": len(reasoning),
            "content_length": len(content),
            "reasoning_tokens": reasoning_tokens
        })

        if verbose:
            print(f"\nTrace {i}:")
            print(f"  Tokens: {num_tokens}")
            print(f"  Tokens/sec: {tokens_per_sec:.2f}")
            print(f"  Reasoning length: {len(reasoning)} chars")
            print(f"  Content length: {len(content)} chars")
            print(f"  Content preview: {content[:100]}...")
    
    num_traces = len(response.choices)
    max_tokens_trace = max(t["num_tokens"] for t in trace_data)

    if verbose:
        print(f"\n{'='*80}")
        print("AGGREGATE STATS")
        print("="*80)
        print(f"Total tokens (all traces): {total_tokens}")
        print(f"Average tokens per trace: {total_tokens / num_traces:.1f}")
        print(f"Total tokens/sec: {total_tokens / elapsed_time:.2f}")
        print(f"Time per trace (sequential): {elapsed_time / num_traces:.2f}s")
        print(f"\nMax tokens in a single trace: {max_tokens_trace}")

    # Build results dictionary
    results = {
        "metadata": {
            "n_completions": n_completions,
            "model": KIMI_MODEL_ID,
            "max_tokens": 64000,
            "temperature": 0.8
        },
        "timing": {
            "start": start_timestamp,
            "end": end_timestamp,
            "duration_seconds": elapsed_time
        },
        "traces": trace_data,
        "aggregate": {
            "total_tokens": total_tokens,
            "avg_tokens_per_trace": total_tokens / num_traces,
            "total_tokens_per_second": total_tokens / elapsed_time if elapsed_time > 0 else 0,
            "max_tokens_single_trace": max_tokens_trace,
            "exceeded_10k": max_tokens_trace >= 10000
        }
    }

    # Generate output filename if not provided
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"timing_test_results_n={n_completions}_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(f"\nResults saved to: {output_file}")

    return results

if __name__ == "__main__":
    # Parse command line arguments
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    test_timing(n_completions=n, verbose=True)

