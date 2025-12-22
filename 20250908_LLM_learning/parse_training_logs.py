#!/usr/bin/env python3
"""
Parse training logs and extract loss curves.
Usage: python parse_training_logs.py
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def parse_log_file(log_path):
    """
    Parse a training log file and extract step, loss, and time information.
    
    Returns:
        pd.DataFrame with columns: step, loss, grad_norm, lrm, dt_ms, tok_per_sec, mfu, total_time_min
    """
    # Pattern for lines like:
    # step 00000/12800 (0.00%) | loss: 11.090355 | grad norm: 0.3870 | lrm: 1.00 | dt: 9875.83ms | tok/sec: 53,087 | mfu: 5.40 | total time: 0.00m
    # or (without grad norm):
    # step 00001 (0.13%) | loss: 1.437552 | lrm: 1.00 | dt: 28942.46ms | tok/sec: 18,114 | mfu: 1.84 | total time: 0.00m
    
    pattern = re.compile(
        r'step\s+(\d+)(?:/\d+)?\s+\([^)]+\)\s+\|\s+loss:\s+([\d.]+)'
        r'(?:\s+\|\s+grad norm:\s+([\d.]+))?'
        r'\s+\|\s+lrm:\s+([\d.]+)'
        r'\s+\|\s+dt:\s+([\d.]+)ms'
        r'\s+\|\s+tok/sec:\s+([\d,]+)'
        r'\s+\|\s+mfu:\s+([\d.]+)'
        r'\s+\|\s+total time:\s+([\d.]+)m'
    )
    
    data = []
    with open(log_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                step = int(match.group(1))
                loss = float(match.group(2))
                grad_norm = float(match.group(3)) if match.group(3) else None
                lrm = float(match.group(4))
                dt_ms = float(match.group(5))
                tok_per_sec = int(match.group(6).replace(',', ''))
                mfu = float(match.group(7))
                total_time_min = float(match.group(8))
                
                data.append({
                    'step': step,
                    'loss': loss,
                    'grad_norm': grad_norm,
                    'lrm': lrm,
                    'dt_ms': dt_ms,
                    'tok_per_sec': tok_per_sec,
                    'mfu': mfu,
                    'total_time_min': total_time_min
                })
    
    return pd.DataFrame(data)


def plot_loss_curves_static(dfs_dict, output_path='loss_curves.png'):
    """
    Plot loss curves from multiple training runs (static matplotlib version).

    Args:
        dfs_dict: Dictionary mapping run names to DataFrames
        output_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Loss vs Step
    ax = axes[0, 0]
    for name, df in dfs_dict.items():
        ax.plot(df['step'], df['loss'], label=name, marker='o', markersize=2, alpha=0.7)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title('Loss vs Training Step')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Loss vs Time
    ax = axes[0, 1]
    for name, df in dfs_dict.items():
        ax.plot(df['total_time_min'], df['loss'], label=name, marker='o', markersize=2, alpha=0.7)
    ax.set_xlabel('Training Time (minutes)')
    ax.set_ylabel('Loss')
    ax.set_title('Loss vs Training Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: MFU vs Step
    ax = axes[1, 0]
    for name, df in dfs_dict.items():
        ax.plot(df['step'], df['mfu'], label=name, marker='o', markersize=2, alpha=0.7)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('MFU (%)')
    ax.set_title('Model FLOPs Utilization vs Step')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Throughput vs Step
    ax = axes[1, 1]
    for name, df in dfs_dict.items():
        ax.plot(df['step'], df['tok_per_sec'], label=name, marker='o', markersize=2, alpha=0.7)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Tokens/sec')
    ax.set_title('Throughput vs Step')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Static plot saved to {output_path}")
    plt.close()


def plot_loss_curves_interactive(dfs_dict, output_path='loss_curves.html'):
    """
    Plot interactive loss curves from multiple training runs using Plotly.

    Args:
        dfs_dict: Dictionary mapping run names to DataFrames
        output_path: Path to save the HTML plot
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Loss vs Training Step', 'Loss vs Training Time',
                       'Model FLOPs Utilization vs Step', 'Throughput vs Step'),
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )

    # Plot 1: Loss vs Step
    for name, df in dfs_dict.items():
        hover_text = [f"Run: {name}<br>Step: {step}<br>Loss: {loss:.6f}<br>Time: {time:.2f}m"
                     for step, loss, time in zip(df['step'], df['loss'], df['total_time_min'])]
        fig.add_trace(
            go.Scatter(x=df['step'], y=df['loss'], mode='lines+markers',
                      name=name, legendgroup=name, showlegend=True,
                      marker=dict(size=3), line=dict(width=2),
                      hovertext=hover_text, hoverinfo='text'),
            row=1, col=1
        )

    # Plot 2: Loss vs Time
    for name, df in dfs_dict.items():
        hover_text = [f"Run: {name}<br>Time: {time:.2f}m<br>Loss: {loss:.6f}<br>Step: {step}"
                     for step, loss, time in zip(df['step'], df['loss'], df['total_time_min'])]
        fig.add_trace(
            go.Scatter(x=df['total_time_min'], y=df['loss'], mode='lines+markers',
                      name=name, legendgroup=name, showlegend=False,
                      marker=dict(size=3), line=dict(width=2),
                      hovertext=hover_text, hoverinfo='text'),
            row=1, col=2
        )

    # Plot 3: MFU vs Step
    for name, df in dfs_dict.items():
        hover_text = [f"Run: {name}<br>Step: {step}<br>MFU: {mfu:.2f}%<br>Throughput: {tok:,} tok/s"
                     for step, mfu, tok in zip(df['step'], df['mfu'], df['tok_per_sec'])]
        fig.add_trace(
            go.Scatter(x=df['step'], y=df['mfu'], mode='lines+markers',
                      name=name, legendgroup=name, showlegend=False,
                      marker=dict(size=3), line=dict(width=2),
                      hovertext=hover_text, hoverinfo='text'),
            row=2, col=1
        )

    # Plot 4: Throughput vs Step
    for name, df in dfs_dict.items():
        hover_text = [f"Run: {name}<br>Step: {step}<br>Throughput: {tok:,} tok/s<br>MFU: {mfu:.2f}%"
                     for step, mfu, tok in zip(df['step'], df['mfu'], df['tok_per_sec'])]
        fig.add_trace(
            go.Scatter(x=df['step'], y=df['tok_per_sec'], mode='lines+markers',
                      name=name, legendgroup=name, showlegend=False,
                      marker=dict(size=3), line=dict(width=2),
                      hovertext=hover_text, hoverinfo='text'),
            row=2, col=2
        )

    # Update axes labels
    fig.update_xaxes(title_text="Training Step", row=1, col=1)
    fig.update_xaxes(title_text="Training Time (minutes)", row=1, col=2)
    fig.update_xaxes(title_text="Training Step", row=2, col=1)
    fig.update_xaxes(title_text="Training Step", row=2, col=2)

    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=2)
    fig.update_yaxes(title_text="MFU (%)", row=2, col=1)
    fig.update_yaxes(title_text="Tokens/sec", row=2, col=2)

    # Update layout
    fig.update_layout(
        height=800,
        width=1400,
        title_text="Training Metrics Dashboard",
        hovermode='x unified',
        template='plotly_white'
    )

    # Save to HTML
    fig.write_html(output_path)
    print(f"Interactive plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Parse training logs and plot loss curves')
    parser.add_argument('--log-dir', type=str, default='nanochat',
                       help='Directory containing log files (default: nanochat)')
    parser.add_argument('--output', type=str, default='loss_curves.html',
                       help='Output plot filename (default: loss_curves.html)')
    parser.add_argument('--csv', action='store_true', help='Also save parsed data as CSV files')
    parser.add_argument('--static', action='store_true',
                       help='Generate static PNG plot instead of interactive HTML')
    args = parser.parse_args()

    log_dir = Path(args.log_dir)

    # Check if log directory exists
    if not log_dir.exists():
        print(f"Error: Log directory '{log_dir}' does not exist")
        return

    log_files = sorted(log_dir.glob('training_*.log'))

    if not log_files:
        print(f"No training log files found in {log_dir}")
        return

    print(f"Found {len(log_files)} log files in {log_dir}:")
    for log_file in log_files:
        print(f"  - {log_file.name}")

    # Parse all log files
    dfs_dict = {}
    for log_file in log_files:
        print(f"\nParsing {log_file.name}...")
        df = parse_log_file(log_file)
        if len(df) > 0:
            run_name = log_file.stem  # filename without extension
            dfs_dict[run_name] = df
            print(f"  Extracted {len(df)} training steps")
            print(f"  Step range: {df['step'].min()} - {df['step'].max()}")
            print(f"  Loss range: {df['loss'].min():.4f} - {df['loss'].max():.4f}")
            print(f"  Total time: {df['total_time_min'].max():.2f} minutes")

            if args.csv:
                csv_path = log_file.with_suffix('.csv')
                df.to_csv(csv_path, index=False)
                print(f"  Saved to {csv_path}")
        else:
            print(f"  No training steps found in {log_file.name}")

    # Plot the curves
    if dfs_dict:
        if args.static:
            # Generate static PNG plot
            output_path = args.output if args.output.endswith('.png') else 'loss_curves.png'
            plot_loss_curves_static(dfs_dict, output_path)
            print(f"\n✓ Successfully created static plot: {output_path}")
        else:
            # Generate interactive HTML plot
            output_path = args.output if args.output.endswith('.html') else 'loss_curves.html'
            plot_loss_curves_interactive(dfs_dict, output_path)
            print(f"\n✓ Successfully created interactive plot: {output_path}")
    else:
        print("\n✗ No data to plot")


if __name__ == '__main__':
    main()

