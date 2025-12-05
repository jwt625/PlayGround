#!/usr/bin/env python3
"""
Analyze DeepConf results to check:
1. Token usage (are we hitting the 10k limit?)
2. Answer diversity (are problems too simple?)
3. Baseline vs DeepConf comparison
"""

import json
import sys
from collections import Counter
from typing import Dict, Any, List

def analyze_results(json_file: str):
    """Analyze the results JSON file"""
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Check if this is the summary format or the old format
    if "per_problem_results" in data:
        results = data["per_problem_results"]
    elif "results" in data:
        results = data["results"]
    else:
        print("Unknown JSON format")
        return
    
    print("="*80)
    print("DEEPCONF RESULTS ANALYSIS")
    print("="*80)
    print(f"\nTotal problems evaluated: {len(results)}")
    
    # 1. Token usage analysis
    print("\n" + "="*80)
    print("1. TOKEN USAGE ANALYSIS")
    print("="*80)
    
    token_stats = []
    for r in results:
        for trace in r.get("traces", []):
            num_tokens = trace.get("num_tokens", 0)
            token_stats.append(num_tokens)
    
    if token_stats:
        print(f"Total traces: {len(token_stats)}")
        print(f"Min tokens: {min(token_stats)}")
        print(f"Max tokens: {max(token_stats)}")
        print(f"Mean tokens: {sum(token_stats)/len(token_stats):.1f}")
        print(f"Median tokens: {sorted(token_stats)[len(token_stats)//2]}")
        
        # Check how many hit the limit (assuming 10k limit)
        near_limit = sum(1 for t in token_stats if t >= 9000)
        at_limit = sum(1 for t in token_stats if t >= 9900)
        print(f"\nTraces near limit (>=9000 tokens): {near_limit}/{len(token_stats)} ({near_limit/len(token_stats)*100:.1f}%)")
        print(f"Traces at limit (>=9900 tokens): {at_limit}/{len(token_stats)} ({at_limit/len(token_stats)*100:.1f}%)")
    
    # 2. Answer diversity analysis
    print("\n" + "="*80)
    print("2. ANSWER DIVERSITY ANALYSIS")
    print("="*80)
    
    all_same_count = 0
    diverse_count = 0
    no_answers_count = 0
    
    for r in results:
        problem_id = r.get("problem_id", "unknown")
        traces = r.get("traces", [])
        
        # Get all extracted answers
        answers = [t.get("extracted_answer") for t in traces if t.get("extracted_answer") is not None]
        
        if len(answers) == 0:
            no_answers_count += 1
            print(f"\n{problem_id}: NO ANSWERS EXTRACTED (0/{len(traces)} traces)")
        elif len(set(answers)) == 1:
            all_same_count += 1
            print(f"\n{problem_id}: ALL SAME - {len(answers)}/{len(traces)} traces all answered '{answers[0]}'")
        else:
            diverse_count += 1
            answer_counts = Counter(answers)
            print(f"\n{problem_id}: DIVERSE - {len(answers)}/{len(traces)} traces, answers: {dict(answer_counts)}")
    
    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"  All traces same answer: {all_same_count}/{len(results)} ({all_same_count/len(results)*100:.1f}%)")
    print(f"  Diverse answers: {diverse_count}/{len(results)} ({diverse_count/len(results)*100:.1f}%)")
    print(f"  No answers extracted: {no_answers_count}/{len(results)} ({no_answers_count/len(results)*100:.1f}%)")
    
    # 3. Baseline vs DeepConf comparison
    print("\n" + "="*80)
    print("3. BASELINE VS DEEPCONF COMPARISON")
    print("="*80)
    
    baseline_correct = 0
    deepconf_results = {
        "tail": {"eta_10": 0, "eta_90": 0},
        "lowest_group": {"eta_10": 0, "eta_90": 0},
        "bottom_10": {"eta_10": 0, "eta_90": 0}
    }
    
    for r in results:
        if r.get("baseline_majority_voting", {}).get("correct", False):
            baseline_correct += 1
        
        for metric in ["tail", "lowest_group", "bottom_10"]:
            for eta in ["eta_10", "eta_90"]:
                if r.get("deepconf", {}).get(metric, {}).get(eta, {}).get("correct", False):
                    deepconf_results[metric][eta] += 1
    
    total = len(results)
    print(f"\nBaseline (Simple Majority Voting): {baseline_correct}/{total} ({baseline_correct/total*100:.1f}%)")
    
    for metric in ["tail", "lowest_group", "bottom_10"]:
        print(f"\n{metric.upper()}:")
        for eta in ["eta_10", "eta_90"]:
            count = deepconf_results[metric][eta]
            print(f"  {eta}: {count}/{total} ({count/total*100:.1f}%)")
    
    # Check if baseline and deepconf have identical results
    print("\n" + "="*80)
    print("4. IDENTICAL RESULTS CHECK")
    print("="*80)
    
    for metric in ["tail", "lowest_group", "bottom_10"]:
        for eta in ["eta_10", "eta_90"]:
            identical_count = 0
            for r in results:
                baseline_ans = r.get("baseline_majority_voting", {}).get("answer")
                deepconf_ans = r.get("deepconf", {}).get(metric, {}).get(eta, {}).get("answer")
                if baseline_ans == deepconf_ans:
                    identical_count += 1
            
            print(f"{metric} {eta}: {identical_count}/{total} problems have same answer as baseline ({identical_count/total*100:.1f}%)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <results_json_file>")
        sys.exit(1)
    
    analyze_results(sys.argv[1])

