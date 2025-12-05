#!/usr/bin/env python3
"""
Inspect question 2 to see why no answers were extracted
"""

import json
import sys

def inspect_question2(json_file: str):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    if "per_problem_results" in data:
        results = data["per_problem_results"]
    elif "results" in data:
        results = data["results"]
    else:
        print("Unknown JSON format")
        return
    
    # Find question 2
    q2 = None
    for r in results:
        if r.get("problem_id") == "AIME2025-I-2":
            q2 = r
            break
    
    if not q2:
        print("Question 2 not found")
        return
    
    print("="*80)
    print("QUESTION 2 ANALYSIS")
    print("="*80)
    
    print(f"\nProblem ID: {q2['problem_id']}")
    print(f"Ground Truth: {q2['ground_truth']}")
    print(f"\nProblem Text:")
    print(q2['problem_text'])
    
    print(f"\n{'='*80}")
    print("TRACE ANALYSIS")
    print("="*80)
    
    for i, trace in enumerate(q2['traces']):
        print(f"\nTrace {i}:")
        print(f"  Num tokens: {trace['num_tokens']}")
        print(f"  Extracted answer: {trace['extracted_answer']}")
        print(f"  Content length: {len(trace['content'])} chars")
        print(f"  Reasoning length: {len(trace['reasoning'])} chars")

        # Check both content and reasoning for \boxed
        content = trace['content']
        reasoning = trace['reasoning']

        full_text = reasoning + content

        if '\\boxed' in full_text:
            print(f"  Contains \\boxed: YES")
            # Show last 500 chars
            print(f"  Last 500 chars of full text (reasoning + content):")
            print(f"    ...{full_text[-500:]}")
        else:
            print(f"  Contains \\boxed: NO")
            print(f"  Last 500 chars of full text (reasoning + content):")
            print(f"    ...{full_text[-500:]}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_question2.py <results_json_file>")
        sys.exit(1)
    
    inspect_question2(sys.argv[1])

