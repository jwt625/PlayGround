#!/usr/bin/env python3
"""
Visualize GPU power and temperature metrics from v2 CSV file.
V2 format includes both instant and average power readings.
Automatically detects and cuts off idle periods at the end.
Creates interactive HTML visualization using Plotly.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import re

def extract_timestamp_from_filename(csv_file):
    """Extract timestamp from CSV filename like 'gpu_metrics_v2_20251222_210600.csv'"""
    match = re.search(r'(\d{8}_\d{6})', csv_file)
    if match:
        return match.group(1)
    return "unknown"

def find_idle_cutoff(df, power_threshold=200, window_size=100):
    """
    Find the time when GPU goes idle (power drops below threshold consistently).
    Uses average power for detection.
    """
    active = (df['gpu0_power_avg_w'] > power_threshold) | (df['gpu1_power_avg_w'] > power_threshold)
    active_indices = np.where(active)[0]
    
    if len(active_indices) == 0:
        return None
    
    last_active_idx = active_indices[-1]
    cutoff_idx = min(last_active_idx + window_size, len(df) - 1)
    cutoff_time = df.iloc[cutoff_idx]['elapsed_sec']
    
    return cutoff_time

def load_and_process_data(csv_file, power_threshold=200, downsample_factor=100, buffer_minutes=10):
    """
    Load CSV data, detect idle cutoff, and downsample for visualization.
    """
    print(f"Loading data from {csv_file}...")
    
    chunks = []
    chunk_size = 100000
    
    for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
        chunks.append(chunk)
    
    df = pd.concat(chunks, ignore_index=True)
    print(f"Loaded {len(df):,} rows")
    
    cutoff_time = find_idle_cutoff(df, power_threshold=power_threshold)
    
    if cutoff_time is not None:
        original_len = len(df)
        buffer_seconds = buffer_minutes * 60
        cutoff_with_buffer = cutoff_time + buffer_seconds
        df = df[df['elapsed_sec'] <= cutoff_with_buffer]
        print(f"Detected idle period starting at {cutoff_time:.1f} seconds ({cutoff_time/3600:.2f} hours)")
        print(f"Added {buffer_minutes} minute buffer, cutting at {cutoff_with_buffer:.1f} seconds ({cutoff_with_buffer/3600:.2f} hours)")
        print(f"Removed {original_len - len(df):,} rows ({(original_len - len(df))/original_len*100:.1f}%)")
    else:
        print("No idle period detected")
    
    if downsample_factor > 1:
        df_plot = df.iloc[::downsample_factor].copy()
        print(f"Downsampled to {len(df_plot):,} rows for plotting (factor: {downsample_factor})")
    else:
        df_plot = df.copy()
    
    return df_plot, cutoff_time

def plot_metrics(df, cutoff_time=None, output_file='gpu_metrics_plot.html'):
    """
    Create interactive visualization of GPU power and temperature over time using Plotly.
    Includes both instant and average power readings.
    """
    time_hours = df['elapsed_sec'] / 3600
    time_seconds = df['elapsed_sec'].values

    # Format time as HH:MM:SS.mmm for each data point
    time_formatted = []
    for sec in time_seconds:
        hours = int(sec // 3600)
        minutes = int((sec % 3600) // 60)
        secs = int(sec % 60)
        milliseconds = int((sec % 1) * 1000)
        time_formatted.append(f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}")

    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('GPU Power Consumption Over Time (Instant & Average)', 'GPU Temperature Over Time'),
        vertical_spacing=0.12
    )

    # Power subplot - both instant and average
    power_traces = [
        {'y': df['gpu0_power_instant_w'], 'name': 'GPU 0 Power (Instant)', 'color': '#1f77b4', 'unit': 'W', 'dash': 'solid'},
        {'y': df['gpu0_power_avg_w'], 'name': 'GPU 0 Power (Avg)', 'color': '#1f77b4', 'unit': 'W', 'dash': 'dot'},
        {'y': df['gpu1_power_instant_w'], 'name': 'GPU 1 Power (Instant)', 'color': '#ff7f0e', 'unit': 'W', 'dash': 'solid'},
        {'y': df['gpu1_power_avg_w'], 'name': 'GPU 1 Power (Avg)', 'color': '#ff7f0e', 'unit': 'W', 'dash': 'dot'},
    ]
    
    for trace_info in power_traces:
        fig.add_trace(
            go.Scatter(
                x=time_hours,
                y=trace_info['y'],
                name=trace_info['name'],
                mode='lines',
                line=dict(color=trace_info['color'], width=1, dash=trace_info['dash']),
                customdata=time_formatted,
                hovertemplate=f"<b>{trace_info['name']}</b><br>Time: %{{customdata}}<br>Value: %{{y:.2f}} {trace_info['unit']}<extra></extra>"
            ),
            row=1, col=1
        )
    
    # Temperature subplot
    temp_traces = [
        {'y': df['gpu0_temp_c'], 'name': 'GPU 0 Temp', 'color': '#2ca02c', 'unit': '°C'},
        {'y': df['gpu1_temp_c'], 'name': 'GPU 1 Temp', 'color': '#d62728', 'unit': '°C'}
    ]
    
    for trace_info in temp_traces:
        fig.add_trace(
            go.Scatter(
                x=time_hours,
                y=trace_info['y'],
                name=trace_info['name'],
                mode='lines',
                line=dict(color=trace_info['color'], width=1),
                customdata=time_formatted,
                hovertemplate=f"<b>{trace_info['name']}</b><br>Time: %{{customdata}}<br>Value: %{{y:.2f}} {trace_info['unit']}<extra></extra>"
            ),
            row=2, col=1
        )

    # Add cutoff line to both subplots
    if cutoff_time is not None:
        cutoff_hours = cutoff_time / 3600
        for row in [1, 2]:
            fig.add_vline(
                x=cutoff_hours,
                line_dash="dash",
                line_color="red",
                line_width=2,
                opacity=0.7,
                annotation_text=f"Idle Start ({cutoff_hours:.2f}h)",
                annotation_position="top",
                row=row, col=1
            )

    # Update axes
    fig.update_yaxes(title_text="Power (W)", row=1, col=1)
    fig.update_yaxes(title_text="Temperature (°C)", row=2, col=1)
    fig.update_xaxes(title_text="Time (hours)", row=1, col=1)
    fig.update_xaxes(title_text="Time (hours)", row=2, col=1)

    # Update layout
    fig.update_layout(
        height=900,
        showlegend=True,
        hovermode='x unified',
        title_text="GPU Metrics Visualization (V2 - Instant & Average Power)",
        title_font_size=20,
        template='plotly_white'
    )

    # Save as HTML
    fig.write_html(output_file)
    print(f"\nInteractive plot saved to: {output_file}")

    return fig

def print_statistics(df):
    """Print summary statistics of GPU metrics."""
    print("\n" + "="*70)
    print("GPU METRICS V2 SUMMARY")
    print("="*70)

    for gpu_id in [0, 1]:
        print(f"\nGPU {gpu_id}:")
        print(f"  Power Instant (W):")
        print(f"    Mean:   {df[f'gpu{gpu_id}_power_instant_w'].mean():.2f}")
        print(f"    Max:    {df[f'gpu{gpu_id}_power_instant_w'].max():.2f}")
        print(f"    Min:    {df[f'gpu{gpu_id}_power_instant_w'].min():.2f}")
        print(f"  Power Average (W):")
        print(f"    Mean:   {df[f'gpu{gpu_id}_power_avg_w'].mean():.2f}")
        print(f"    Max:    {df[f'gpu{gpu_id}_power_avg_w'].max():.2f}")
        print(f"    Min:    {df[f'gpu{gpu_id}_power_avg_w'].min():.2f}")
        print(f"  Temperature (°C):")
        print(f"    Mean:   {df[f'gpu{gpu_id}_temp_c'].mean():.2f}")
        print(f"    Max:    {df[f'gpu{gpu_id}_temp_c'].max():.2f}")
        print(f"    Min:    {df[f'gpu{gpu_id}_temp_c'].min():.2f}")

    print(f"\nTotal duration: {df['elapsed_sec'].max()/3600:.2f} hours")
    print("="*70 + "\n")

def export_summary(df, output_file='gpu_metrics_v2_summary.txt'):
    """Export summary statistics to a text file."""
    with open(output_file, 'w') as f:
        f.write("GPU METRICS V2 SUMMARY\n")
        f.write("="*70 + "\n\n")

        for gpu_id in [0, 1]:
            f.write(f"GPU {gpu_id}:\n")
            f.write(f"  Power Instant (W): Mean={df[f'gpu{gpu_id}_power_instant_w'].mean():.2f}, ")
            f.write(f"Max={df[f'gpu{gpu_id}_power_instant_w'].max():.2f}, ")
            f.write(f"Min={df[f'gpu{gpu_id}_power_instant_w'].min():.2f}\n")
            f.write(f"  Power Average (W): Mean={df[f'gpu{gpu_id}_power_avg_w'].mean():.2f}, ")
            f.write(f"Max={df[f'gpu{gpu_id}_power_avg_w'].max():.2f}, ")
            f.write(f"Min={df[f'gpu{gpu_id}_power_avg_w'].min():.2f}\n")
            f.write(f"  Temperature (°C): Mean={df[f'gpu{gpu_id}_temp_c'].mean():.2f}, ")
            f.write(f"Max={df[f'gpu{gpu_id}_temp_c'].max():.2f}, ")
            f.write(f"Min={df[f'gpu{gpu_id}_temp_c'].min():.2f}\n\n")

        f.write(f"Total duration: {df['elapsed_sec'].max()/3600:.2f} hours\n")
        f.write(f"Sample rate: {len(df)/df['elapsed_sec'].max():.2f} samples/sec\n")

    print(f"Summary exported to: {output_file}")

if __name__ == "__main__":
    # Configuration
    csv_file = "gpu_metrics_v2_20251222_210600.csv"

    # Extract timestamp from filename
    timestamp = extract_timestamp_from_filename(csv_file)
    output_html = f"gpu_metrics_plot-v2_{timestamp}.html"
    output_summary = f"gpu_metrics_v2_summary_{timestamp}.txt"

    power_threshold = 200
    downsample_factor = 1
    buffer_minutes = 10

    # Load and process data
    df_plot, cutoff_time = load_and_process_data(
        csv_file,
        power_threshold=power_threshold,
        downsample_factor=downsample_factor,
        buffer_minutes=buffer_minutes
    )

    # Print statistics
    print_statistics(df_plot)

    # Export summary
    export_summary(df_plot, output_file=output_summary)

    # Create visualization
    plot_metrics(df_plot, cutoff_time=cutoff_time, output_file=output_html)

