#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "pandas",
#     "scipy",
#     "plotly",
# ]
# ///
"""
Generate waterfall spectrograms from GPU power metrics.

Creates two spectrograms with different time windows to capture both
short-term (batch-level) and long-term (checkpoint/validation) dynamics.

Usage:
    uv run gpu_spectrogram.py <csv_file>
    uv run gpu_spectrogram.py <csv_file> --gpu 0
    uv run gpu_spectrogram.py <csv_file> --metric power_avg
"""

import argparse
import numpy as np
import pandas as pd
from scipy import signal
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import sys


# Spectrogram configurations
SPECTROGRAM_CONFIGS = {
    'fine': {
        'window_sec': 60,      # 1 minute
        'overlap_frac': 0.75,  # 75% overlap
        'description': '1-min window (batch/step dynamics)',
    },
    'coarse': {
        'window_sec': 600,     # 10 minutes
        'overlap_frac': 0.75,  # 75% overlap
        'description': '10-min window (checkpoint/validation dynamics)',
    },
}


def extract_timestamp_from_filename(csv_file):
    """Extract timestamp from CSV filename."""
    match = re.search(r'(\d{8}_\d{6})', csv_file)
    return match.group(1) if match else "unknown"


def interpolate_nans(values):
    """
    Interpolate NaN values by averaging left and right neighbors.
    For edge cases or consecutive NaNs, use linear interpolation.
    """
    values = values.copy()
    nan_mask = np.isnan(values)
    nan_count = np.sum(nan_mask)

    if nan_count == 0:
        return values, 0

    # Use pandas interpolation for robustness with consecutive NaNs
    series = pd.Series(values)
    series = series.interpolate(method='linear', limit_direction='both')
    values = series.values

    # Check if any NaNs remain (e.g., all-NaN regions)
    remaining_nans = np.sum(np.isnan(values))
    if remaining_nans > 0:
        # Fill remaining with global mean
        global_mean = np.nanmean(values)
        values = np.where(np.isnan(values), global_mean, values)

    return values, nan_count


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


def load_data(csv_file, metric='power_instant', gpu_id=0,
              power_threshold=200, buffer_minutes=10):
    """
    Load GPU metrics and return time series.

    Args:
        csv_file: Path to CSV file
        metric: 'power_instant', 'power_avg', or 'temp'
        gpu_id: 0 or 1
        power_threshold: Power threshold in W to detect idle state
        buffer_minutes: Minutes to keep after idle detected

    Returns:
        elapsed_sec: numpy array of timestamps
        values: numpy array of metric values
        sample_rate: estimated sample rate in Hz
    """
    print(f"Loading {csv_file}...")

    # Load in chunks for large files
    chunks = []
    for chunk in pd.read_csv(csv_file, chunksize=500000):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)

    print(f"Loaded {len(df):,} samples")

    # Detect and cut idle period
    cutoff_time = find_idle_cutoff(df, power_threshold=power_threshold)
    if cutoff_time is not None:
        original_len = len(df)
        buffer_seconds = buffer_minutes * 60
        cutoff_with_buffer = cutoff_time + buffer_seconds
        df = df[df['elapsed_sec'] <= cutoff_with_buffer]
        print(f"Detected idle at {cutoff_time:.1f}s ({cutoff_time/3600:.2f}h)")
        print(f"Cut at {cutoff_with_buffer:.1f}s with {buffer_minutes}min buffer")
        print(f"Removed {original_len - len(df):,} rows ({(original_len - len(df))/original_len*100:.1f}%)")
    else:
        print("No idle period detected")

    # Select column
    col_map = {
        'power_instant': f'gpu{gpu_id}_power_instant_w',
        'power_avg': f'gpu{gpu_id}_power_avg_w',
        'temp': f'gpu{gpu_id}_temp_c',
    }
    col_name = col_map[metric]

    elapsed_sec = df['elapsed_sec'].values
    values = df[col_name].values

    # Interpolate NaN values
    values, nan_count = interpolate_nans(values)
    if nan_count > 0:
        print(f"Interpolated {nan_count:,} NaN values ({nan_count/len(values)*100:.3f}%)")

    # Estimate sample rate
    dt = np.median(np.diff(elapsed_sec[:10000]))
    sample_rate = 1.0 / dt

    print(f"Duration: {elapsed_sec[-1]/3600:.2f} hours")
    print(f"Sample rate: {sample_rate:.1f} Hz")
    print(f"Metric: {col_name}")

    return elapsed_sec, values, sample_rate


def compute_spectrogram(values, sample_rate, window_sec, overlap_frac):
    """
    Compute spectrogram using scipy.signal.spectrogram.
    
    Returns:
        freqs: frequency array (Hz)
        times: time array (seconds)
        Sxx: power spectral density (dB)
    """
    nperseg = int(window_sec * sample_rate)
    noverlap = int(nperseg * overlap_frac)
    
    print(f"  Window: {window_sec}s ({nperseg} samples)")
    print(f"  Overlap: {overlap_frac*100:.0f}% ({noverlap} samples)")
    
    freqs, times, Sxx = signal.spectrogram(
        values,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='density',
        mode='psd',
    )
    
    # Convert to dB
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    
    print(f"  Frequency range: {freqs[1]:.6f} - {freqs[-1]:.4f} Hz")
    print(f"  Time bins: {len(times)}")
    
    return freqs, times, Sxx_db


MIN_FREQ = 0.001   # 1000s period (~17 min)
MAX_FREQ = 0.5     # 2s period


def create_spectrogram_figure(freqs, times, Sxx_db, config_name, config, metric, gpu_id):
    """Create a single spectrogram figure."""

    times_hours = times / 3600

    # Same frequency range for both spectrograms
    freq_mask = (freqs >= MIN_FREQ) & (freqs <= MAX_FREQ)

    fig = go.Figure(data=go.Heatmap(
        x=times_hours,
        y=freqs[freq_mask],
        z=Sxx_db[freq_mask, :],
        colorscale='Viridis',
        colorbar=dict(title='Power (dB)'),
    ))

    fig.update_layout(
        title=f'GPU {gpu_id} {metric} Spectrogram - {config["description"]}',
        xaxis_title='Time (hours)',
        yaxis_title='Frequency (Hz)',
        height=500,
        template='plotly_white',
    )

    return fig


def create_combined_figure(spectrograms, metric, gpu_id, timestamp):
    """Create a combined figure with both spectrograms."""

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            f'Fine Scale: {SPECTROGRAM_CONFIGS["fine"]["description"]}',
            f'Coarse Scale: {SPECTROGRAM_CONFIGS["coarse"]["description"]}',
        ],
        vertical_spacing=0.12,
        shared_xaxes=True,
    )

    # Track global min/max for colorbar defaults
    all_z_values = []

    for idx, (config_name, data) in enumerate(spectrograms.items()):
        freqs, times, Sxx_db = data['freqs'], data['times'], data['Sxx_db']
        times_hours = times / 3600

        # Same frequency range for both
        freq_mask = (freqs >= MIN_FREQ) & (freqs <= MAX_FREQ)
        z_data = Sxx_db[freq_mask, :]
        all_z_values.append(z_data)

        fig.add_trace(
            go.Heatmap(
                x=times_hours,
                y=freqs[freq_mask],
                z=z_data,
                colorscale='Viridis',
                showscale=(idx == 0),
                colorbar=dict(title='Power (dB)', x=1.02) if idx == 0 else None,
                zmin=None,  # Will be set by JS
                zmax=None,
            ),
            row=idx+1, col=1
        )

        fig.update_yaxes(title_text='Frequency (Hz)', row=idx+1, col=1)

    fig.update_xaxes(title_text='Time (hours)', row=2, col=1)

    fig.update_layout(
        title=f'GPU {gpu_id} {metric} Spectrograms - {timestamp}',
        height=900,
        template='plotly_white',
    )

    # Compute default z range
    all_z = np.concatenate([z.flatten() for z in all_z_values])
    z_min_default = float(np.percentile(all_z[~np.isnan(all_z)], 5))
    z_max_default = float(np.percentile(all_z[~np.isnan(all_z)], 95))

    return fig, z_min_default, z_max_default


def main():
    parser = argparse.ArgumentParser(
        description='Generate waterfall spectrograms from GPU power metrics.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('csv_file', help='Path to GPU metrics CSV file')
    parser.add_argument('--gpu', type=int, default=0, choices=[0, 1],
                        help='GPU ID (default: 0)')
    parser.add_argument('--metric', type=str, default='power_instant',
                        choices=['power_instant', 'power_avg', 'temp'],
                        help='Metric to analyze (default: power_instant)')
    parser.add_argument('--downsample', type=int, default=10,
                        help='Downsample factor to reduce memory (default: 10)')
    parser.add_argument('--power-threshold', type=float, default=200,
                        help='Power threshold in W to detect idle state (default: 200)')
    parser.add_argument('--buffer-minutes', type=int, default=10,
                        help='Minutes to keep after idle detected (default: 10)')

    args = parser.parse_args()

    timestamp = extract_timestamp_from_filename(args.csv_file)

    # Load data
    elapsed_sec, values, sample_rate = load_data(
        args.csv_file, metric=args.metric, gpu_id=args.gpu,
        power_threshold=args.power_threshold, buffer_minutes=args.buffer_minutes
    )

    # Downsample to reduce computation
    if args.downsample > 1:
        elapsed_sec = elapsed_sec[::args.downsample]
        values = values[::args.downsample]
        sample_rate = sample_rate / args.downsample
        print(f"Downsampled to {len(values):,} samples ({sample_rate:.1f} Hz)")

    # Compute spectrograms
    spectrograms = {}
    for config_name, config in SPECTROGRAM_CONFIGS.items():
        print(f"\nComputing {config_name} spectrogram...")
        freqs, times, Sxx_db = compute_spectrogram(
            values, sample_rate, config['window_sec'], config['overlap_frac']
        )
        spectrograms[config_name] = {
            'freqs': freqs, 'times': times, 'Sxx_db': Sxx_db
        }

    # Create combined figure
    print("\nGenerating visualization...")
    fig, z_min_default, z_max_default = create_combined_figure(
        spectrograms, args.metric, args.gpu, timestamp
    )

    output_file = f'gpu_spectrogram_{args.metric}_gpu{args.gpu}_{timestamp}.html'

    # Generate HTML with custom colorbar controls
    html_content = generate_html_with_controls(fig, z_min_default, z_max_default)
    with open(output_file, 'w') as f:
        f.write(html_content)
    print(f"\nSaved: {output_file}")


def generate_html_with_controls(fig, z_min_default, z_max_default):
    """Generate HTML with interactive colorbar range controls."""

    # Get the plotly div
    plotly_div = fig.to_html(full_html=False, include_plotlyjs='cdn')

    html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>GPU Spectrogram</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .controls {{
            background: #f5f5f5;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 20px;
        }}
        .control-group {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .control-group label {{
            font-weight: bold;
            min-width: 80px;
        }}
        .control-group input {{
            width: 80px;
            padding: 5px;
            border: 1px solid #ccc;
            border-radius: 3px;
        }}
        button {{
            padding: 8px 16px;
            background: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }}
        button:hover {{ background: #45a049; }}
        .reset-btn {{ background: #666; }}
        .reset-btn:hover {{ background: #555; }}
    </style>
</head>
<body>
    <div class="controls">
        <div class="control-group">
            <label>Color Min (dB):</label>
            <input type="number" id="zmin" value="{z_min_default:.1f}" step="1">
        </div>
        <div class="control-group">
            <label>Color Max (dB):</label>
            <input type="number" id="zmax" value="{z_max_default:.1f}" step="1">
        </div>
        <button onclick="updateColorRange()">Apply</button>
        <button class="reset-btn" onclick="resetColorRange()">Reset</button>
    </div>

    <div id="plotly-div">
        {plotly_div}
    </div>

    <script>
        const defaultZmin = {z_min_default:.1f};
        const defaultZmax = {z_max_default:.1f};

        function updateColorRange() {{
            const zmin = parseFloat(document.getElementById('zmin').value);
            const zmax = parseFloat(document.getElementById('zmax').value);

            const plotDiv = document.querySelector('.plotly-graph-div');
            Plotly.restyle(plotDiv, {{
                zmin: zmin,
                zmax: zmax
            }});
        }}

        function resetColorRange() {{
            document.getElementById('zmin').value = defaultZmin.toFixed(1);
            document.getElementById('zmax').value = defaultZmax.toFixed(1);
            updateColorRange();
        }}

        // Apply defaults on load
        window.onload = function() {{
            setTimeout(updateColorRange, 500);
        }};
    </script>
</body>
</html>'''

    return html


if __name__ == '__main__':
    main()

