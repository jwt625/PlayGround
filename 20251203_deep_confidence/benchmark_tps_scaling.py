#!/usr/bin/env python3
"""
Benchmark script to study how TPS (tokens per second) scales with concurrent traces.

Runs test_timing for N = 2, 4, 6, 8, 10, 12, 14, 16 with 3 repetitions each.
Analyzes variance of mean and std across repetitions.
"""

import os
import json
import time
import statistics
from datetime import datetime
from typing import List, Dict, Any
from test_timing import test_timing

def save_summary(
    summary_file: str,
    timestamp: str,
    n_values: List[int],
    repetitions: int,
    all_results: List[Dict[str, Any]]
) -> None:
    """Save current benchmark summary to file"""
    analysis = analyze_results(all_results, verbose=False)

    summary = {
        "metadata": {
            "timestamp": timestamp,
            "n_values": n_values,
            "repetitions": repetitions,
            "completed_n_values": [r["n_completions"] for r in all_results]
        },
        "results": all_results,
        "analysis": analysis
    }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

def run_benchmark(
    n_values: List[int] = [2, 4, 6, 8, 10, 12, 14, 16],
    repetitions: int = 4,
    output_dir: str = "benchmark_results"
) -> Dict[str, Any]:
    """Run systematic benchmark of TPS scaling
    
    Args:
        n_values: List of n_completions values to test
        repetitions: Number of repetitions for each n value
        output_dir: Directory to save results
        
    Returns:
        Dictionary containing aggregated benchmark results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(output_dir, f"benchmark_summary_{timestamp}.json")

    print("="*80)
    print(f"TPS SCALING BENCHMARK - {timestamp}")
    print("="*80)
    print(f"N values: {n_values}")
    print(f"Repetitions per N: {repetitions}")
    print(f"Total runs: {len(n_values) * repetitions}")
    print(f"Summary file: {summary_file}")
    print("="*80)

    all_results = []
    
    for n in n_values:
        print(f"\n{'='*80}")
        print(f"Testing N={n} ({repetitions} repetitions)")
        print("="*80)
        
        n_results = []
        
        for rep in range(repetitions):
            print(f"\n  Run {rep+1}/{repetitions}...")
            
            # Generate unique filename
            run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(
                output_dir,
                f"timing_n={n}_rep={rep+1}_{run_timestamp}.json"
            )
            
            # Run test
            result = test_timing(
                n_completions=n,
                verbose=False,
                output_file=output_file
            )
            
            n_results.append(result)
            
            # Print summary
            avg_tps = statistics.mean([t["tokens_per_second"] for t in result["traces"]])
            print(f"    Duration: {result['timing']['duration_seconds']:.1f}s")
            print(f"    Avg TPS per trace: {avg_tps:.2f}")
            print(f"    Total tokens: {result['aggregate']['total_tokens']}")
            
            # Small delay between runs
            time.sleep(2)

        all_results.append({
            "n_completions": n,
            "repetitions": n_results
        })

        # Save progress after each N value
        save_summary(summary_file, timestamp, n_values, repetitions, all_results)
    
    # Final analysis with verbose output
    print(f"\n{'='*80}")
    print("FINAL ANALYSIS")
    print("="*80)

    analysis = analyze_results(all_results, verbose=True)

    # Save final summary
    save_summary(summary_file, timestamp, n_values, repetitions, all_results)

    print(f"\nBenchmark complete! Summary saved to: {summary_file}")

    return {
        "metadata": {
            "timestamp": timestamp,
            "n_values": n_values,
            "repetitions": repetitions
        },
        "results": all_results,
        "analysis": analysis
    }


def analyze_results(all_results: List[Dict[str, Any]], verbose: bool = True) -> Dict[str, Any]:
    """Analyze benchmark results to compute statistics across repetitions"""

    analysis = {}

    if verbose:
        print(f"\n{'N':<4} {'Mean TPS':<12} {'Std TPS':<12} {'Mean±Std':<20} {'CV%':<8}")
        print("-" * 80)
    
    for n_data in all_results:
        n = n_data["n_completions"]
        reps = n_data["repetitions"]
        
        # Extract TPS for each trace across all repetitions
        all_tps = []
        for rep in reps:
            all_tps.extend([t["tokens_per_second"] for t in rep["traces"]])
        
        # Compute mean TPS per repetition
        mean_tps_per_rep = [
            statistics.mean([t["tokens_per_second"] for t in rep["traces"]])
            for rep in reps
        ]

        # Overall statistics
        overall_mean = statistics.mean(all_tps)
        overall_std = statistics.stdev(all_tps) if len(all_tps) > 1 else 0.0
        
        # Variance of the mean across repetitions
        mean_of_means = statistics.mean(mean_tps_per_rep)
        std_of_means = statistics.stdev(mean_tps_per_rep) if len(mean_tps_per_rep) > 1 else 0.0
        
        # Coefficient of variation
        cv = (overall_std / overall_mean * 100) if overall_mean > 0 else 0.0
        
        analysis[f"n={n}"] = {
            "n_completions": n,
            "overall_mean_tps": overall_mean,
            "overall_std_tps": overall_std,
            "mean_of_means": mean_of_means,
            "std_of_means": std_of_means,
            "coefficient_of_variation": cv,
            "num_samples": len(all_tps)
        }

        if verbose:
            print(f"{n:<4} {overall_mean:<12.2f} {overall_std:<12.2f} {mean_of_means:.2f}±{std_of_means:.2f}{'':>8} {cv:<8.1f}")

    return analysis


if __name__ == "__main__":
    run_benchmark()

