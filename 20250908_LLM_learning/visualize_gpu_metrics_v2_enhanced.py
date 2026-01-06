#!/usr/bin/env python3
"""
Visualize GPU power and temperature metrics from v2 CSV file.
V2 format includes both instant and average power readings.
Automatically detects and cuts off idle periods at the end.
Creates interactive HTML visualization using Plotly.
Optionally overlays training events from a parsed events CSV.

Usage:
    python visualize_gpu_metrics_v2_enhanced.py <csv_file> [downsample_factor]
    python visualize_gpu_metrics_v2_enhanced.py <csv_file> [downsample_factor] --events <events_csv>

Arguments:
    csv_file: Path to the GPU metrics CSV file
    downsample_factor: Optional downsampling factor (default: 10)
                      Use 1 for no downsampling, 10 to keep every 10th sample, etc.
    --events: Path to events CSV file (from parse_training_events.py)
    --time-offset: Time offset in seconds to align events with GPU metrics
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import re
import sys
import argparse


# Event type styling configuration
EVENT_STYLES = {
    'validation_bpb': {'color': 'rgba(0, 128, 0, 0.6)', 'dash': 'dash', 'width': 1},
    'validation_loss': {'color': 'rgba(0, 128, 0, 0.6)', 'dash': 'dash', 'width': 1},
    'core_metric': {'color': 'rgba(128, 0, 128, 0.8)', 'dash': 'solid', 'width': 2},
    'benchmark_eval': {'color': 'rgba(100, 100, 100, 0.3)', 'dash': 'dot', 'width': 1},
    'inline_benchmark': {'color': 'rgba(0, 0, 255, 0.6)', 'dash': 'dash', 'width': 1},
    'checkpoint_save': {'color': 'rgba(255, 0, 0, 0.8)', 'dash': 'solid', 'width': 2},
    'model_load': {'color': 'rgba(255, 165, 0, 0.7)', 'dash': 'dashdot', 'width': 1},
    'arc_easy_result': {'color': 'rgba(0, 0, 255, 0.7)', 'dash': 'solid', 'width': 2},
    'mmlu_result': {'color': 'rgba(0, 0, 255, 0.7)', 'dash': 'solid', 'width': 2},
    'humaneval_result': {'color': 'rgba(0, 128, 128, 0.8)', 'dash': 'solid', 'width': 2},
}

# Event types to show (set to None to show all, or list specific types)
DEFAULT_EVENT_FILTER = ['validation_bpb', 'validation_loss', 'core_metric', 'checkpoint_save',
                        'arc_easy_result', 'mmlu_result', 'humaneval_result']

def extract_timestamp_from_filename(csv_file):
    """Extract timestamp from CSV filename like 'gpu_metrics_v2_20251222_210600.csv'"""
    match = re.search(r'(\d{8}_\d{6})', csv_file)
    if match:
        return match.group(1)
    return "unknown"

def load_events(events_file, gpu_start_time=None, event_filter=None):
    """
    Load training events from CSV file and align with GPU metrics timeline.

    Args:
        events_file: Path to events CSV
        gpu_start_time: Start timestamp of GPU metrics (datetime) for auto-alignment
        event_filter: List of event types to include, or None for all

    Returns:
        DataFrame with events aligned to GPU metrics timeline
    """
    from datetime import datetime

    df = pd.read_csv(events_file)

    # Parse event timestamps and calculate elapsed time relative to GPU start
    if gpu_start_time is not None:
        # Parse ISO format timestamps from events
        df['event_datetime'] = pd.to_datetime(df['timestamp'])
        # Calculate elapsed seconds from GPU start time
        df['elapsed_sec'] = (df['event_datetime'] - gpu_start_time).dt.total_seconds()
        df['elapsed_hours'] = df['elapsed_sec'] / 3600
        print(f"Auto-aligned events to GPU metrics start time: {gpu_start_time}")

    if event_filter:
        df = df[df['event_type'].isin(event_filter)]

    print(f"Loaded {len(df)} events from {events_file}")
    return df


def get_gpu_start_time(csv_file):
    """Extract the start timestamp from GPU metrics CSV."""
    from datetime import datetime

    df_head = pd.read_csv(csv_file, nrows=1)
    timestamp_str = df_head['timestamp'].iloc[0]
    # Parse format: "2026-01-02 21:10:45.409"
    return datetime.strptime(timestamp_str.split('.')[0], '%Y-%m-%d %H:%M:%S')


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

def plot_metrics(df, cutoff_time=None, output_file='gpu_metrics_plot.html', events_df=None):
    """
    Create interactive visualization of GPU power and temperature over time using Plotly.
    Includes both instant and average power readings.
    Optionally overlays training events as vertical lines.

    Args:
        df: GPU metrics DataFrame
        cutoff_time: Time when idle period starts (optional)
        output_file: Output HTML file path
        events_df: DataFrame with training events (optional)
    """
    time_hours = df['elapsed_sec'] / 3600

    # Determine if we have events to show
    has_events = events_df is not None and len(events_df) > 0

    # Create figure with subplots (3 rows if events, 2 otherwise)
    if has_events:
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('GPU Power Consumption Over Time (Instant & Average)',
                            'GPU Temperature Over Time',
                            'Training Events'),
            vertical_spacing=0.08,
            shared_xaxes=True,
            row_heights=[0.4, 0.4, 0.2]
        )
    else:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('GPU Power Consumption Over Time (Instant & Average)',
                            'GPU Temperature Over Time'),
            vertical_spacing=0.12,
            shared_xaxes=True
        )

    # Power subplot
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
                hovertemplate=f"{trace_info['name']}: %{{y:.1f}} {trace_info['unit']}<extra></extra>"
            ),
            row=1, col=1
        )

    # Temperature subplot
    temp_traces = [
        {'y': df['gpu0_temp_c'], 'name': 'GPU 0 Temp', 'color': '#2ca02c', 'unit': 'C'},
        {'y': df['gpu1_temp_c'], 'name': 'GPU 1 Temp', 'color': '#d62728', 'unit': 'C'}
    ]

    for trace_info in temp_traces:
        fig.add_trace(
            go.Scatter(
                x=time_hours,
                y=trace_info['y'],
                name=trace_info['name'],
                mode='lines',
                line=dict(color=trace_info['color'], width=1),
                hovertemplate=f"{trace_info['name']}: %{{y:.1f}} {trace_info['unit']}<extra></extra>"
            ),
            row=2, col=1
        )

    # Add cutoff line to subplots
    if cutoff_time is not None:
        cutoff_hours = cutoff_time / 3600
        num_rows = 3 if has_events else 2
        for row in range(1, num_rows + 1):
            fig.add_vline(
                x=cutoff_hours,
                line_dash="dash",
                line_color="red",
                line_width=2,
                opacity=0.7,
                annotation_text=f"Idle Start ({cutoff_hours:.2f}h)" if row == 1 else None,
                annotation_position="top",
                row=row, col=1
            )

    # Add training events to separate subplot (row 3)
    if has_events:
        max_time_hours = df['elapsed_sec'].max() / 3600
        events_in_range = events_df[events_df['elapsed_hours'] <= max_time_hours]

        # Group events by type and assign y positions
        event_types = events_in_range['event_type'].unique()
        type_to_y = {et: i for i, et in enumerate(event_types)}

        # Track which types have been added for legend
        event_types_added = set()

        for _, event in events_in_range.iterrows():
            event_type = event['event_type']
            style = EVENT_STYLES.get(event_type, {'color': 'rgba(128,128,128,0.8)', 'dash': 'dot', 'width': 1})
            y_pos = type_to_y[event_type]

            show_legend = event_type not in event_types_added
            event_types_added.add(event_type)

            # Add marker for event in the events subplot
            fig.add_trace(
                go.Scatter(
                    x=[event['elapsed_hours']],
                    y=[y_pos],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=style['color'],
                        symbol='diamond',
                        line=dict(width=1, color='white')
                    ),
                    name=event_type.replace('_', ' ').title() if show_legend else None,
                    showlegend=show_legend,
                    legendgroup=event_type,
                    hovertemplate=f"<b>{event['label']}</b><br>Time: {event['elapsed_hours']:.2f}h<extra></extra>",
                ),
                row=3, col=1
            )

        # Configure events subplot y-axis with event type labels
        fig.update_yaxes(
            tickmode='array',
            tickvals=list(range(len(event_types))),
            ticktext=[et.replace('_', ' ').title() for et in event_types],
            row=3, col=1
        )

        print(f"Added {len(events_in_range)} event markers to plot")

    # Update axes
    fig.update_yaxes(title_text="Power (W)", row=1, col=1)
    fig.update_yaxes(title_text="Temperature (C)", row=2, col=1)
    if has_events:
        fig.update_yaxes(title_text="Events", row=3, col=1)
        fig.update_xaxes(title_text="Time (hours)", row=3, col=1)
    else:
        fig.update_xaxes(title_text="Time (hours)", row=2, col=1)

    # Update layout
    fig.update_layout(
        height=1000 if has_events else 800,
        showlegend=True,
        hovermode='x unified',  # Unified hover for GPU metrics
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
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Visualize GPU metrics from v2 CSV file with instant and average power readings.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process with default downsampling (factor=10)
  python visualize_gpu_metrics_v2_enhanced.py gpu_metrics_v2_20251222_210600.csv

  # Process with no downsampling
  python visualize_gpu_metrics_v2_enhanced.py gpu_metrics_v2_20251222_210600.csv 1

  # Process with heavy downsampling (every 100th sample)
  python visualize_gpu_metrics_v2_enhanced.py gpu_metrics_v2_20251222_210600.csv 100

  # Include training events overlay
  python visualize_gpu_metrics_v2_enhanced.py gpu_metrics_v2.csv 10 --events training_events.csv
        """
    )
    parser.add_argument('csv_file', type=str,
                        help='Path to the GPU metrics CSV file')
    parser.add_argument('downsample_factor', type=int, nargs='?', default=10,
                        help='Downsampling factor (default: 10). Use 1 for no downsampling.')
    parser.add_argument('--power-threshold', type=float, default=200,
                        help='Power threshold in Watts to detect idle state (default: 200)')
    parser.add_argument('--buffer-minutes', type=int, default=10,
                        help='Minutes to add after detected idle cutoff (default: 10)')
    parser.add_argument('--events', type=str, default=None,
                        help='Path to events CSV file (from parse_training_events.py)')
    parser.add_argument('--event-filter', type=str, nargs='*', default=None,
                        help='Event types to show (default: validation, core_metric, checkpoint, results)')

    args = parser.parse_args()

    # Configuration from arguments
    csv_file = args.csv_file
    downsample_factor = args.downsample_factor
    power_threshold = args.power_threshold
    buffer_minutes = args.buffer_minutes

    # Extract timestamp from filename
    timestamp = extract_timestamp_from_filename(csv_file)
    output_html = f"gpu_metrics_plot-v2_{timestamp}.html"
    output_summary = f"gpu_metrics_v2_summary_{timestamp}.txt"

    print(f"Processing: {csv_file}")
    print(f"Downsample factor: {downsample_factor}")
    print(f"Power threshold: {power_threshold}W")
    print(f"Buffer minutes: {buffer_minutes}")
    print()

    # Get GPU start time for event alignment
    gpu_start_time = get_gpu_start_time(csv_file)
    print(f"GPU metrics start time: {gpu_start_time}")

    # Load and process data
    df_plot, cutoff_time = load_and_process_data(
        csv_file,
        power_threshold=power_threshold,
        downsample_factor=downsample_factor,
        buffer_minutes=buffer_minutes
    )

    # Load events if provided (auto-aligned to GPU start time)
    events_df = None
    if args.events:
        event_filter = args.event_filter if args.event_filter else DEFAULT_EVENT_FILTER
        events_df = load_events(args.events, gpu_start_time=gpu_start_time, event_filter=event_filter)

    # Print statistics
    print_statistics(df_plot)

    # Export summary
    export_summary(df_plot, output_file=output_summary)

    # Create visualization
    plot_metrics(df_plot, cutoff_time=cutoff_time, output_file=output_html, events_df=events_df)

