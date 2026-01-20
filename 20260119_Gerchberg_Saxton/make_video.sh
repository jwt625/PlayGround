#!/bin/bash
# Combine frames with audio and apply 1.5x speedup (pitch preserved)
# Output: badapple_gs_demo.mp4

set -e

FPS=10
SPEEDUP=1.5

echo "Creating video from frames..."

# Step 1: Create video from frames at original FPS
ffmpeg -y -framerate $FPS -i frames/frame_%05d.png \
    -c:v libx264 -pix_fmt yuv420p \
    -preset medium -crf 23 \
    temp_video.mp4

echo "Speeding up audio (1.5x, pitch preserved)..."

# Step 2: Speed up audio 1.5x with pitch preservation using atempo
ffmpeg -y -i badapple_audio.mp3 \
    -filter:a "atempo=1.5" \
    -c:a aac -b:a 192k \
    temp_audio.m4a

echo "Combining video and audio with 1.5x speedup..."

# Step 3: Speed up video 1.5x and combine with sped-up audio
# setpts=PTS/1.5 = setpts=0.6667*PTS
ffmpeg -y -i temp_video.mp4 -i temp_audio.m4a \
    -filter:v "setpts=PTS/1.5" \
    -c:v libx264 -pix_fmt yuv420p \
    -preset medium -crf 23 \
    -c:a copy \
    -shortest \
    -movflags +faststart \
    badapple_gs_demo.mp4

# Cleanup
rm -f temp_video.mp4 temp_audio.m4a

echo ""
echo "Done! Output: badapple_gs_demo.mp4"
ls -lh badapple_gs_demo.mp4

