#!/usr/bin/env python3
"""
Analyze beat spacing to find equally-spaced beats.
"""

import json
import numpy as np

# Load the beat timestamps
with open('beat_timestamps.json', 'r') as f:
    data = json.load(f)

beats = [b['timestamp_seconds'] for b in data['beats']]
print(f"Total beats detected: {len(beats)}")
print(f"Duration: {data['duration_seconds']:.2f}s\n")

# Calculate intervals between consecutive beats
intervals = [beats[i+1] - beats[i] for i in range(len(beats)-1)]

print("All intervals:")
for i, interval in enumerate(intervals):
    print(f"Beat {i+1} to {i+2}: {interval:.3f}s")

print(f"\nInterval statistics:")
print(f"Mean: {np.mean(intervals):.3f}s")
print(f"Std dev: {np.std(intervals):.3f}s")
print(f"Min: {np.min(intervals):.3f}s")
print(f"Max: {np.max(intervals):.3f}s")

# Look for clusters of similar intervals
print("\n" + "="*60)
print("Looking for equally-spaced beats...")
print("="*60)

# If there are 18 equally spaced beats across ~15 seconds
# That's 17 intervals, so approximately 15/17 ≈ 0.88 seconds per interval
expected_interval = data['duration_seconds'] / 17

print(f"\nFor 18 beats equally spaced: ~{expected_interval:.3f}s intervals")

# Try to find beats with intervals close to the expected interval
# Allow some tolerance (±20%)
tolerance = 0.20
min_interval = expected_interval * (1 - tolerance)
max_interval = expected_interval * (1 + tolerance)

print(f"Looking for intervals in range [{min_interval:.3f}s - {max_interval:.3f}s]")

# Find beats that match the expected spacing
matched_beats = [beats[0]]  # Start with first beat
for i in range(len(intervals)):
    current_beat = beats[i]
    next_beat = beats[i+1]
    interval = intervals[i]

    if min_interval <= interval <= max_interval:
        matched_beats.append(next_beat)

print(f"\nFound {len(matched_beats)} roughly equally-spaced beats")

if len(matched_beats) >= 18:
    # Take the first 18
    equal_beats = matched_beats[:18]
    print(f"\nTaking first 18 equally-spaced beats:")
else:
    # Try a different approach - find the longest sequence of similar intervals
    print(f"\nNeed to find a different pattern. Analyzing interval clusters...")

    # Group similar intervals (within 0.05s tolerance)
    interval_groups = {}
    for i, interval in enumerate(intervals):
        # Round to nearest 0.1s for grouping
        key = round(interval * 2) / 2  # Round to nearest 0.5s
        if key not in interval_groups:
            interval_groups[key] = []
        interval_groups[key].append((i, interval))

    print("\nInterval clusters:")
    for key in sorted(interval_groups.keys()):
        group = interval_groups[key]
        print(f"  ~{key:.2f}s: {len(group)} intervals")

    # Find the most common interval
    max_group = max(interval_groups.values(), key=len)
    common_interval = np.mean([iv[1] for iv in max_group])
    print(f"\nMost common interval: ~{common_interval:.3f}s ({len(max_group)} occurrences)")

# Alternative approach: look for beats at regular intervals starting from different offsets
print("\n" + "="*60)
print("Alternative approach: Regular grid search")
print("="*60)

best_match = None
best_count = 0

# Try different starting points and intervals
for start_idx in range(5):  # Try first few beats as starting points
    if start_idx >= len(beats):
        break

    start_time = beats[start_idx]
    remaining_time = data['duration_seconds'] - start_time

    # Try different intervals
    for num_beats in [18, 19, 20]:  # Try 18, 19, or 20 beats
        expected_interval = remaining_time / (num_beats - 1)

        # Find beats close to this regular grid
        regular_beats = [start_time]
        for i in range(1, num_beats):
            target_time = start_time + i * expected_interval
            # Find closest actual beat
            closest_idx = min(range(len(beats)), key=lambda idx: abs(beats[idx] - target_time))
            closest_beat = beats[closest_idx]

            # Only include if within reasonable tolerance (0.1s)
            if abs(closest_beat - target_time) < 0.15:
                regular_beats.append(closest_beat)

        if len(regular_beats) >= best_count:
            best_count = len(regular_beats)
            best_match = {
                'start_idx': start_idx,
                'num_beats': num_beats,
                'interval': expected_interval,
                'beats': regular_beats
            }

if best_match and len(best_match['beats']) >= 18:
    print(f"\nBest match: {len(best_match['beats'])} beats")
    print(f"Starting from beat {best_match['start_idx'] + 1} ({best_match['beats'][0]:.3f}s)")
    print(f"Average interval: {best_match['interval']:.3f}s")

    equal_beats = best_match['beats'][:18]  # Take exactly 18

    print("\n18 Equally-spaced beats:")
    print("-" * 60)
    for i, beat_time in enumerate(equal_beats):
        ms = int(beat_time * 1000)
        print(f"Beat {i+1:2d}: {beat_time:6.3f}s ({ms:5d}ms)")
        if i > 0:
            interval = beat_time - equal_beats[i-1]
            print(f"         (interval: {interval:.3f}s)")

    # Save to new JSON
    output_beats = []
    for i, beat_time in enumerate(equal_beats):
        output_beats.append({
            "beat_number": i + 1,
            "timestamp_seconds": float(beat_time),
            "timestamp_ms": int(beat_time * 1000),
            "timestamp_formatted": f"0:{beat_time:06.3f}"
        })

    output_data = {
        "audio_file": data['audio_file'],
        "sample_rate": data['sample_rate'],
        "duration_seconds": data['duration_seconds'],
        "total_beats": 18,
        "average_interval": float(np.mean([equal_beats[i+1] - equal_beats[i] for i in range(len(equal_beats)-1)])),
        "beats": output_beats
    }

    with open('beat_timestamps_18_equal.json', 'w') as f:
        json.dump(output_data, f, indent=2)

    print("\n✓ Saved 18 equally-spaced beats to beat_timestamps_18_equal.json")
else:
    print("\nCould not find 18 equally-spaced beats with the current detection.")
    print("Consider adjusting the onset detection parameters.")
