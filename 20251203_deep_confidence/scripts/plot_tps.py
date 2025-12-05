#!/usr/bin/env python3
"""Parse tmp.log and plot TPS metrics vs N."""

import re
import matplotlib.pyplot as plt
import numpy as np

def parse_log(filename):
    """Parse the log file and extract TPS data."""
    data = {}
    current_n = None
    
    with open(filename, 'r') as f:
        for line in f:
            # Match N value
            n_match = re.match(r'Testing N=(\d+)', line)
            if n_match:
                current_n = int(n_match.group(1))
                data[current_n] = {'durations': [], 'avg_tps': [], 'total_tokens': []}
                continue
            
            # Match duration
            duration_match = re.match(r'\s+Duration: ([\d.]+)s', line)
            if duration_match and current_n is not None:
                data[current_n]['durations'].append(float(duration_match.group(1)))
                continue
            
            # Match avg TPS
            tps_match = re.match(r'\s+Avg TPS per trace: ([\d.]+)', line)
            if tps_match and current_n is not None:
                data[current_n]['avg_tps'].append(float(tps_match.group(1)))
                continue
            
            # Match total tokens
            tokens_match = re.match(r'\s+Total tokens: (\d+)', line)
            if tokens_match and current_n is not None:
                data[current_n]['total_tokens'].append(int(tokens_match.group(1)))
                continue
    
    return data

def calculate_metrics(data):
    """Calculate total TPS and average TPS for each N."""
    n_values = []
    total_tps_values = []
    avg_tps_values = []
    all_total_tps = []
    all_avg_tps = []
    all_n_for_total = []
    all_n_for_avg = []

    for n in sorted(data.keys()):
        if not data[n]['durations'] or not data[n]['total_tokens']:
            continue

        n_values.append(n)

        # Calculate total TPS for each run (total_tokens / duration)
        total_tps_per_run = [tokens / duration
                             for tokens, duration in zip(data[n]['total_tokens'],
                                                        data[n]['durations'])]

        # Remove the minimum TPS run from N=10 (outlier)
        if n == 10:
            min_idx = total_tps_per_run.index(min(total_tps_per_run))
            total_tps_per_run_filtered = [tps for i, tps in enumerate(total_tps_per_run) if i != min_idx]
            avg_tps_filtered = [tps for i, tps in enumerate(data[n]['avg_tps']) if i != min_idx]
        else:
            total_tps_per_run_filtered = total_tps_per_run
            avg_tps_filtered = data[n]['avg_tps']

        # Store all individual points
        all_total_tps.extend(total_tps_per_run_filtered)
        all_n_for_total.extend([n] * len(total_tps_per_run_filtered))

        all_avg_tps.extend(avg_tps_filtered)
        all_n_for_avg.extend([n] * len(avg_tps_filtered))

        # Average total TPS across runs
        avg_total_tps = np.mean(total_tps_per_run_filtered)
        total_tps_values.append(avg_total_tps)

        # Average of avg TPS per trace across runs
        avg_avg_tps = np.mean(avg_tps_filtered)
        avg_tps_values.append(avg_avg_tps)

    return (n_values, total_tps_values, avg_tps_values,
            all_n_for_total, all_total_tps, all_n_for_avg, all_avg_tps)

def plot_tps(n_values, total_tps, avg_tps, all_n_total, all_total, all_n_avg, all_avg,
             output_file='tps_plot.png'):
    """Create plot of TPS metrics vs N."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot individual points (smaller markers, semi-transparent)
    ax.plot(all_n_total, all_total, 'o', markersize=4, alpha=0.4, color='C0',
            label='Total TPS (individual runs)')
    ax.plot(all_n_avg, all_avg, 's', markersize=4, alpha=0.4, color='C1',
            label='Avg TPS per trace (individual runs)')

    # Plot averages (larger markers, solid lines)
    ax.plot(n_values, total_tps, 'o-', linewidth=2, markersize=8, color='C0',
            label='Total TPS (mean)', zorder=5)
    ax.plot(n_values, avg_tps, 's-', linewidth=2, markersize=8, color='C1',
            label='Avg TPS per trace (mean)', zorder=5)

    # Styling
    ax.set_xlabel('N', fontsize=14)
    ax.set_ylabel('TPS (tokens/second)', fontsize=14)
    ax.set_title('TPS Scaling vs N', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")

    # Print summary
    print("\nSummary:")
    print(f"{'N':<6} {'Total TPS':<12} {'Avg TPS/trace':<15}")
    print("-" * 35)
    for n, ttps, atps in zip(n_values, total_tps, avg_tps):
        print(f"{n:<6} {ttps:<12.2f} {atps:<15.2f}")

if __name__ == '__main__':
    # Parse log file
    data = parse_log('tmp.log')

    # Calculate metrics
    (n_values, total_tps, avg_tps,
     all_n_total, all_total, all_n_avg, all_avg) = calculate_metrics(data)

    # Create plot
    plot_tps(n_values, total_tps, avg_tps, all_n_total, all_total, all_n_avg, all_avg)

