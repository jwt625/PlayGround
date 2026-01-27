#!/usr/bin/env python3
"""
Extract beat timestamps with stricter detection for main beats only.
"""

import librosa
import numpy as np
import json

# Load the audio file
audio_file = "extracted_segment.mp3"
print(f"Loading audio from {audio_file}...")
y, sr = librosa.load(audio_file, sr=None)

duration = len(y)/sr
print(f"Sample rate: {sr} Hz")
print(f"Duration: {duration:.2f} seconds")

# For 18 beats in ~15 seconds, expected interval is ~0.83s
expected_interval = duration / 17  # 17 intervals for 18 beats
print(f"Expected interval for 18 beats: {expected_interval:.3f}s")

# Detect onsets with MUCH stricter parameters
# Increase pre_max (longer lookback) to avoid double-detections
# Increase delta (threshold) to only catch strongest beats
onset_frames = librosa.onset.onset_detect(
    y=y,
    sr=sr,
    units='frames',
    backtrack=True,
    hop_length=512,
    pre_max=40,      # Increased from default 3 to suppress nearby onsets
    post_max=40,     # Increased to ensure we catch peaks
    pre_avg=100,     # Increased averaging window
    post_avg=100,
    delta=0.3,       # Much higher threshold (default ~0.07)
    wait=int(sr * 0.5 / 512)  # Minimum 0.5s between beats
)

onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)

print(f"\nDetected {len(onset_times)} strong beats")

if len(onset_times) > 0:
    # Calculate intervals
    if len(onset_times) > 1:
        intervals = [onset_times[i+1] - onset_times[i] for i in range(len(onset_times)-1)]
        print(f"Interval range: {min(intervals):.3f}s - {max(intervals):.3f}s")
        print(f"Mean interval: {np.mean(intervals):.3f}s")

    # If we got more than 18, try to select the most evenly spaced subset
    if len(onset_times) >= 18:
        print(f"\nUsing first 18 beats:")
        selected_beats = onset_times[:18]
    elif len(onset_times) < 18:
        print(f"\nDetected only {len(onset_times)} beats, may need to adjust parameters.")
        print("Trying alternative: energy-based peak detection...")

        # Alternative: Use RMS energy to find strong beats
        hop_length = 512
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

        # Find peaks in RMS energy
        from scipy.signal import find_peaks

        # Find peaks with minimum distance constraint
        min_distance = int(0.7 * sr / hop_length)  # Minimum 0.7s between peaks
        peaks, properties = find_peaks(
            rms,
            distance=min_distance,
            prominence=np.std(rms) * 1.5  # Only strong peaks
        )

        # Sort by peak height and take top 18
        peak_heights = rms[peaks]
        sorted_indices = np.argsort(-peak_heights)  # Descending order
        top_peaks = peaks[sorted_indices[:18]]
        top_peaks = np.sort(top_peaks)  # Sort by time

        selected_beats = librosa.frames_to_time(top_peaks, sr=sr, hop_length=hop_length)
        print(f"Found {len(selected_beats)} peaks using RMS energy")
    else:
        selected_beats = onset_times

    # Sort and display
    selected_beats = np.sort(selected_beats)

    print("\n" + "="*60)
    print(f"Final {len(selected_beats)} beats:")
    print("="*60)

    results = []
    for i, time in enumerate(selected_beats):
        timestamp_ms = int(time * 1000)
        beat_info = {
            "beat_number": i + 1,
            "timestamp_seconds": float(time),
            "timestamp_ms": timestamp_ms,
            "timestamp_formatted": f"0:{time:06.3f}"
        }
        results.append(beat_info)

        interval_str = ""
        if i > 0:
            interval = time - selected_beats[i-1]
            interval_str = f" | interval: {interval:.3f}s"

        print(f"Beat {i+1:2d}: {time:6.3f}s ({timestamp_ms:5d}ms){interval_str}")

    # Calculate interval statistics
    if len(selected_beats) > 1:
        intervals = [selected_beats[i+1] - selected_beats[i] for i in range(len(selected_beats)-1)]
        print(f"\nInterval statistics:")
        print(f"  Mean: {np.mean(intervals):.3f}s")
        print(f"  Std dev: {np.std(intervals):.3f}s")
        print(f"  Range: {np.min(intervals):.3f}s - {np.max(intervals):.3f}s")

    # Save to JSON
    output_file = "beat_timestamps_18_equal.json"
    with open(output_file, 'w') as f:
        json.dump({
            "audio_file": audio_file,
            "sample_rate": int(sr),
            "duration_seconds": float(duration),
            "total_beats": len(selected_beats),
            "average_interval": float(np.mean(intervals)) if len(selected_beats) > 1 else None,
            "beats": results
        }, f, indent=2)

    print(f"\n✓ Saved to {output_file}")

else:
    print("No beats detected. Try lowering the threshold.")
