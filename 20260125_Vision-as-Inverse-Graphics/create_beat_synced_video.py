#!/usr/bin/env python3
"""
Create a video synced to beat timestamps.
Each image switches at each beat.
"""

import json
import os
import subprocess
from pathlib import Path

# Load beat timestamps
with open('beat_timestamps_18_equal.json', 'r') as f:
    beat_data = json.load(f)

beats = [b['timestamp_seconds'] for b in beat_data['beats']]
duration = beat_data['duration_seconds']

print(f"Loaded {len(beats)} beat timestamps")
print(f"Audio duration: {duration:.2f}s")

# Find all images
image_dir = Path("output/chip2_iterations")
images = []
for i in range(1, 20):
    img_path = image_dir / f"iteration_{i:02d}" / f"layout_iter{i:02d}.png"
    if img_path.exists():
        images.append(str(img_path))
        print(f"Found: {img_path}")
    else:
        print(f"Missing: {img_path}")

print(f"\nTotal images found: {len(images)}")

if len(images) != 19:
    print(f"Warning: Expected 19 images, found {len(images)}")
    if len(images) < 19:
        print("Cannot proceed without all images.")
        exit(1)

# Calculate display duration for each image
# Image 1: 0s to beat[0]
# Image 2: beat[0] to beat[1]
# ...
# Image 19: beat[17] to end
image_durations = []
image_durations.append(beats[0])  # First image: 0 to first beat

for i in range(len(beats) - 1):
    image_durations.append(beats[i + 1] - beats[i])

# Last image: last beat to end of audio
image_durations.append(duration - beats[-1])

print("\nImage display durations:")
for i, dur in enumerate(image_durations):
    print(f"Image {i+1:2d} (iter{i+1:02d}): {dur:.3f}s")

print(f"\nTotal duration: {sum(image_durations):.3f}s")

# Create ffmpeg concat file
concat_file = "concat_list.txt"
with open(concat_file, 'w') as f:
    for i, (img_path, dur) in enumerate(zip(images, image_durations)):
        # FFmpeg concat demuxer format
        f.write(f"file '{img_path}'\n")
        f.write(f"duration {dur:.6f}\n")
    # Add last image again (ffmpeg concat requirement)
    f.write(f"file '{images[-1]}'\n")

print(f"\n✓ Created {concat_file}")

# Create video with ffmpeg
output_video = "chip2_iterations_synced.mp4"
audio_file = "extracted_segment.mp3"

print(f"\nCreating video: {output_video}")
print("Encoding with H.264 + AAC for web/mobile compatibility...")

# FFmpeg command:
# -f concat: use concat demuxer
# -safe 0: allow absolute paths
# -i concat_list.txt: input image sequence with durations
# -i audio: input audio
# -c:v libx264: H.264 video codec
# -preset medium: encoding speed/quality tradeoff
# -crf 23: quality (lower = better, 18-28 is good range)
# -pix_fmt yuv420p: compatible pixel format for web/mobile
# -r 30: 30 fps
# -c:a aac: AAC audio codec
# -b:a 128k: audio bitrate
# -shortest: match video length to shortest stream (audio)
# -movflags +faststart: enable streaming (web-friendly)

ffmpeg_cmd = [
    "ffmpeg",
    "-f", "concat",
    "-safe", "0",
    "-i", concat_file,
    "-i", audio_file,
    "-c:v", "libx264",
    "-preset", "medium",
    "-crf", "23",
    "-pix_fmt", "yuv420p",
    "-r", "30",
    "-c:a", "aac",
    "-b:a", "128k",
    "-shortest",
    "-movflags", "+faststart",
    "-y",  # overwrite output
    output_video
]

print("\nRunning ffmpeg...")
print(" ".join(ffmpeg_cmd))
print()

result = subprocess.run(ffmpeg_cmd, capture_output=False)

if result.returncode == 0:
    print(f"\n✓ Video created successfully: {output_video}")

    # Get video info
    probe_cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration,size",
        "-show_entries", "stream=codec_name,width,height,r_frame_rate",
        "-of", "default=noprint_wrappers=1",
        output_video
    ]

    print("\nVideo info:")
    subprocess.run(probe_cmd)

    # Get file size
    size_mb = os.path.getsize(output_video) / (1024 * 1024)
    print(f"\nFile size: {size_mb:.2f} MB")

else:
    print(f"\n✗ Error creating video (exit code: {result.returncode})")
    exit(1)

# Cleanup
print(f"\nCleaning up {concat_file}...")
os.remove(concat_file)

print("\n✓ Done!")
