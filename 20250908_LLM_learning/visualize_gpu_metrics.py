#!/usr/bin/env python3
"""
Visualize GPU power and temperature metrics from CSV file.
Automatically detects and cuts off idle periods at the end.
Creates interactive HTML visualization using Plotly.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path

def find_idle_cutoff(df, power_threshold=200, window_size=100):
    """
    Find the time when GPU goes idle (power drops below threshold consistently).
    
    Args:
        df: DataFrame with GPU metrics
        power_threshold: Power threshold in watts to consider as idle
        window_size: Number of consecutive samples below threshold to confirm idle
    
    Returns:
        Cutoff time in seconds, or None if no idle period detected
    """
    # Check if either GPU is above threshold
    active = (df['gpu0_power_w'] > power_threshold) | (df['gpu1_power_w'] > power_threshold)
    
    # Find last index where GPU was active
    active_indices = np.where(active)[0]
    
    if len(active_indices) == 0:
        return None
    
    last_active_idx = active_indices[-1]
    
    # Add some buffer to include the transition
    cutoff_idx = min(last_active_idx + window_size, len(df) - 1)
    cutoff_time = df.iloc[cutoff_idx]['elapsed_sec']
    
    return cutoff_time

def load_and_process_data(csv_file, power_threshold=200, downsample_factor=100, buffer_minutes=10):
    """
    Load CSV data, detect idle cutoff, and downsample for visualization.

    Args:
        csv_file: Path to CSV file
        power_threshold: Power threshold for idle detection
        downsample_factor: Factor to downsample data (e.g., 100 = keep every 100th row)
        buffer_minutes: Minutes to add after cutoff time to verify idling

    Returns:
        Processed DataFrame and cutoff time
    """
    print(f"Loading data from {csv_file}...")

    # Read CSV in chunks to handle large files efficiently
    chunks = []
    chunk_size = 100000

    for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    print(f"Loaded {len(df):,} rows")

    # Find idle cutoff
    cutoff_time = find_idle_cutoff(df, power_threshold=power_threshold)

    if cutoff_time is not None:
        original_len = len(df)
        # Add buffer time to show idle period
        buffer_seconds = buffer_minutes * 60
        cutoff_with_buffer = cutoff_time + buffer_seconds
        df = df[df['elapsed_sec'] <= cutoff_with_buffer]
        print(f"Detected idle period starting at {cutoff_time:.1f} seconds ({cutoff_time/3600:.2f} hours)")
        print(f"Added {buffer_minutes} minute buffer, cutting at {cutoff_with_buffer:.1f} seconds ({cutoff_with_buffer/3600:.2f} hours)")
        print(f"Removed {original_len - len(df):,} rows ({(original_len - len(df))/original_len*100:.1f}%)")
    else:
        print("No idle period detected")

    # Downsample for visualization
    if downsample_factor > 1:
        df_plot = df.iloc[::downsample_factor].copy()
        print(f"Downsampled to {len(df_plot):,} rows for plotting (factor: {downsample_factor})")
    else:
        df_plot = df.copy()

    return df_plot, cutoff_time

def plot_metrics(df, cutoff_time=None, output_file='gpu_metrics_plot.html'):
    """
    Create interactive visualization of GPU power and temperature over time using Plotly.
    """
    # Convert elapsed time to hours for better readability
    time_hours = df['elapsed_sec'] / 3600
    time_seconds = df['elapsed_sec']

    # Create subplots with secondary x-axis
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('GPU Power Consumption Over Time', 'GPU Temperature Over Time'),
        vertical_spacing=0.12,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )

    # Plot 1: Power consumption
    fig.add_trace(
        go.Scatter(
            x=time_hours,
            y=df['gpu0_power_w'],
            name='GPU 0 Power',
            mode='lines',
            line=dict(color='#1f77b4', width=1),
            hovertemplate='<b>GPU 0</b><br>Time: %{x:.3f} hrs<br>Power: %{y:.2f} W<extra></extra>'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=time_hours,
            y=df['gpu1_power_w'],
            name='GPU 1 Power',
            mode='lines',
            line=dict(color='#ff7f0e', width=1),
            hovertemplate='<b>GPU 1</b><br>Time: %{x:.3f} hrs<br>Power: %{y:.2f} W<extra></extra>'
        ),
        row=1, col=1
    )

    # Add cutoff line for power plot
    if cutoff_time is not None:
        cutoff_hours = cutoff_time / 3600
        fig.add_vline(
            x=cutoff_hours,
            line_dash="dash",
            line_color="red",
            line_width=2,
            opacity=0.7,
            annotation_text=f"Idle Start ({cutoff_hours:.2f}h)",
            annotation_position="top",
            row=1, col=1
        )

    # Plot 2: Temperature
    fig.add_trace(
        go.Scatter(
            x=time_hours,
            y=df['gpu0_temp_c'],
            name='GPU 0 Temp',
            mode='lines',
            line=dict(color='#2ca02c', width=1),
            hovertemplate='<b>GPU 0</b><br>Time: %{x:.3f} hrs<br>Temp: %{y:.1f} °C<extra></extra>'
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=time_hours,
            y=df['gpu1_temp_c'],
            name='GPU 1 Temp',
            mode='lines',
            line=dict(color='#d62728', width=1),
            hovertemplate='<b>GPU 1</b><br>Time: %{x:.3f} hrs<br>Temp: %{y:.1f} °C<extra></extra>'
        ),
        row=2, col=1
    )

    # Add cutoff line for temperature plot
    if cutoff_time is not None:
        fig.add_vline(
            x=cutoff_hours,
            line_dash="dash",
            line_color="red",
            line_width=2,
            opacity=0.7,
            annotation_text=f"Idle Start ({cutoff_hours:.2f}h)",
            annotation_position="top",
            row=2, col=1
        )



    # Update y-axes labels
    fig.update_yaxes(title_text="Power (W)", row=1, col=1)
    fig.update_yaxes(title_text="Temperature (°C)", row=2, col=1)

    # Calculate ranges for secondary axes
    max_seconds = df['elapsed_sec'].max()
    max_hours = max_seconds / 3600

    # Update x-axes for both subplots
    # Row 1 (Power) - bottom axis in hours
    fig.update_xaxes(title_text="Time (hours)", side='bottom', showgrid=True, row=1, col=1)

    # Row 2 (Temperature) - bottom axis in hours
    fig.update_xaxes(title_text="Time (hours)", side='bottom', showgrid=True, row=2, col=1)

    # Update layout to add secondary x-axes (seconds on top)
    # The secondary axes need to be in hours (same as primary) but display as seconds
    fig.update_layout(
        height=900,
        showlegend=True,
        hovermode='x unified',
        title_text="GPU Metrics Visualization",
        title_font_size=20,
        template='plotly_white',

        # First subplot - top axis (seconds)
        xaxis2=dict(
            title="Time (seconds)",
            overlaying='x',
            side='top',
            showgrid=False,
            range=[0, max_hours],  # Range in hours to match the data
            tickmode='array',
            anchor='y',
            # Convert hours to seconds for display
            tickvals=[i*max_hours/10 for i in range(11)],
            ticktext=[f'{int(i*max_seconds/10)}' for i in range(11)]
        ),

        # Second subplot - top axis (seconds)
        xaxis4=dict(
            title="Time (seconds)",
            overlaying='x3',
            side='top',
            showgrid=False,
            range=[0, max_hours],  # Range in hours to match the data
            tickmode='array',
            anchor='y2',
            # Convert hours to seconds for display
            tickvals=[i*max_hours/10 for i in range(11)],
            ticktext=[f'{int(i*max_seconds/10)}' for i in range(11)]
        )
    )

    # Save as HTML
    fig.write_html(output_file)
    print(f"\nInteractive plot saved to: {output_file}")

    return fig

def print_statistics(df):
    """Print summary statistics of GPU metrics."""
    print("\n" + "="*60)
    print("GPU METRICS SUMMARY")
    print("="*60)
    
    for gpu_id in [0, 1]:
        print(f"\nGPU {gpu_id}:")
        print(f"  Power (W):")
        print(f"    Mean:   {df[f'gpu{gpu_id}_power_w'].mean():.2f}")
        print(f"    Max:    {df[f'gpu{gpu_id}_power_w'].max():.2f}")
        print(f"    Min:    {df[f'gpu{gpu_id}_power_w'].min():.2f}")
        print(f"  Temperature (°C):")
        print(f"    Mean:   {df[f'gpu{gpu_id}_temp_c'].mean():.2f}")
        print(f"    Max:    {df[f'gpu{gpu_id}_temp_c'].max():.2f}")
        print(f"    Min:    {df[f'gpu{gpu_id}_temp_c'].min():.2f}")
    
    print(f"\nTotal duration: {df['elapsed_sec'].max()/3600:.2f} hours")
    print("="*60 + "\n")

if __name__ == "__main__":
    # Configuration
    csv_file = "gpu_metrics_20251221_070042.csv"
    output_file = "gpu_metrics_plot.html"
    power_threshold = 200  # Watts - threshold to detect idle state
    downsample_factor = 100  # Keep every Nth row for plotting
    buffer_minutes = 10  # Minutes to add after cutoff to verify idling

    # Load and process data
    df_plot, cutoff_time = load_and_process_data(
        csv_file,
        power_threshold=power_threshold,
        downsample_factor=downsample_factor,
        buffer_minutes=buffer_minutes
    )

    # Print statistics
    print_statistics(df_plot)

    # Create visualization
    plot_metrics(df_plot, cutoff_time=cutoff_time, output_file=output_file)

