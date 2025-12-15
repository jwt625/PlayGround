#!/usr/bin/env python3
"""
Analyze existing timing test results to compute TPS statistics
"""

import json
import glob
import statistics
from collections import defaultdict

def analyze_existing_results():
    """Analyze all existing timing_test_results_n=*.json files"""
    
    # Find all timing result files
    files = glob.glob("timing_test_results_n=*.json")
    
    if not files:
        print("No timing result files found!")
        return
    
    print("="*80)
    print("ANALYSIS OF EXISTING TIMING RESULTS")
    print("="*80)
    print(f"Found {len(files)} result files\n")
    
    # Group by n value
    results_by_n = defaultdict(list)
    
    for file in sorted(files):
        with open(file, 'r') as f:
            data = json.load(f)
        
        # Extract n from filename (e.g., "timing_test_results_n=8.json" -> 8)
        n_str = file.split("n=")[1].split(".json")[0].split("_")[0]
        n = int(n_str)
        
        # Extract TPS for each trace
        tps_values = [trace["tokens_per_second"] for trace in data["traces"]]
        
        results_by_n[n].append({
            "file": file,
            "tps_values": tps_values,
            "duration": data["timing"]["duration_seconds"],
            "total_tokens": data["aggregate"]["total_tokens"]
        })
    
    # Analyze each n value
    print(f"{'N':<4} {'Runs':<6} {'Mean TPS':<12} {'Std TPS':<12} {'Min TPS':<12} {'Max TPS':<12} {'CV%':<8}")
    print("-" * 80)
    
    all_analysis = {}
    
    for n in sorted(results_by_n.keys()):
        runs = results_by_n[n]
        
        # Combine all TPS values across all runs for this n
        all_tps = []
        for run in runs:
            all_tps.extend(run["tps_values"])
        
        mean_tps = statistics.mean(all_tps)
        std_tps = statistics.stdev(all_tps) if len(all_tps) > 1 else 0.0
        min_tps = min(all_tps)
        max_tps = max(all_tps)
        cv = (std_tps / mean_tps * 100) if mean_tps > 0 else 0.0
        
        all_analysis[n] = {
            "n_completions": n,
            "num_runs": len(runs),
            "num_samples": len(all_tps),
            "mean_tps": mean_tps,
            "std_tps": std_tps,
            "min_tps": min_tps,
            "max_tps": max_tps,
            "cv": cv
        }
        
        print(f"{n:<4} {len(runs):<6} {mean_tps:<12.2f} {std_tps:<12.2f} {min_tps:<12.2f} {max_tps:<12.2f} {cv:<8.1f}")
    
    # Per-run analysis
    print(f"\n{'='*80}")
    print("PER-RUN DETAILS")
    print("="*80)
    
    for n in sorted(results_by_n.keys()):
        runs = results_by_n[n]
        
        if len(runs) > 1:
            print(f"\nN={n} ({len(runs)} runs):")
            
            # Compute mean TPS for each run
            run_means = [statistics.mean(run["tps_values"]) for run in runs]
            run_stds = [statistics.stdev(run["tps_values"]) if len(run["tps_values"]) > 1 else 0.0 for run in runs]
            
            mean_of_means = statistics.mean(run_means)
            std_of_means = statistics.stdev(run_means) if len(run_means) > 1 else 0.0
            
            print(f"  Mean TPS across runs: {mean_of_means:.2f} ± {std_of_means:.2f}")
            print(f"  Individual runs:")
            for i, (run, mean, std) in enumerate(zip(runs, run_means, run_stds)):
                print(f"    Run {i+1}: {mean:.2f} ± {std:.2f} TPS ({run['duration']:.1f}s, {run['total_tokens']} tokens)")
    
    # Save analysis
    output_file = "existing_timing_analysis.json"
    with open(output_file, "w") as f:
        json.dump(all_analysis, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Analysis saved to: {output_file}")
    print("="*80)

if __name__ == "__main__":
    analyze_existing_results()

