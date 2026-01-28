#!/usr/bin/env python3
"""
Extract beat timestamps from audio file.
Detects sudden onset events (volume pulses) and outputs their exact timestamps.
"""

import librosa
import numpy as np
import json

# Load the audio file
audio_file = "extracted_segment.mp3"
print(f"Loading audio from {audio_file}...")
y, sr = librosa.load(audio_file, sr=None)

print(f"Sample rate: {sr} Hz")
print(f"Duration: {len(y)/sr:.2f} seconds")

# Detect onsets (beats/pulses) - these are sudden increases in energy
onset_frames = librosa.onset.onset_detect(
    y=y,
    sr=sr,
    units='frames',
    backtrack=True,  # More accurate timing
    hop_length=512,
    delta=0.1  # Threshold for onset detection
)

# Convert frame indices to timestamps
onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)

print(f"\nDetected {len(onset_times)} beats/onsets:")
print("-" * 50)

# Output in multiple formats
results = []
for i, time in enumerate(onset_times):
    timestamp_ms = int(time * 1000)  # milliseconds
    minutes = int(time // 60)
    seconds = time % 60

    beat_info = {
        "beat_number": i + 1,
        "timestamp_seconds": float(time),
        "timestamp_ms": timestamp_ms,
        "timestamp_formatted": f"{minutes}:{seconds:06.3f}"
    }
    results.append(beat_info)

    print(f"Beat {i+1:2d}: {time:6.3f}s ({timestamp_ms:5d}ms) - {minutes}:{seconds:06.3f}")

# Save to JSON file
output_file = "beat_timestamps.json"
with open(output_file, 'w') as f:
    json.dump({
        "audio_file": audio_file,
        "sample_rate": int(sr),
        "duration_seconds": float(len(y)/sr),
        "total_beats": len(onset_times),
        "beats": results
    }, f, indent=2)

print(f"\n✓ Beat timestamps saved to {output_file}")

# Also create a simple text file for easy reference
with open("beat_timestamps.txt", 'w') as f:
    f.write(f"Beat timestamps from {audio_file}\n")
    f.write(f"Total beats detected: {len(onset_times)}\n")
    f.write("=" * 50 + "\n\n")
    for beat in results:
        f.write(f"Beat {beat['beat_number']}: {beat['timestamp_seconds']:.3f}s ({beat['timestamp_ms']}ms)\n")

print("✓ Beat timestamps saved to beat_timestamps.txt")
