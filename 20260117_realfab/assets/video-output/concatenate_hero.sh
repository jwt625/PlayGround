#!/bin/bash
# Concatenate all segments + outro and add audio
# REVISED: Time-stretch clips to match audio exactly, preserving full outro
set -e
cd "$(dirname "$0")/.."

OUT_DIR="video-output"
AUDIO="audio-narrator/realfab-hero-final.m4a"
OUTRO_DUR=3.0

# Get audio duration
AUDIO_DUR=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$AUDIO")
echo "Audio duration: $AUDIO_DUR seconds"
echo "Outro duration: $OUTRO_DUR seconds"

# Target duration for clips (excluding outro)
TARGET_CLIPS_DUR=$(echo "$AUDIO_DUR - $OUTRO_DUR" | bc)
echo "Target clips duration: $TARGET_CLIPS_DUR seconds"

echo "=== Step 1: Concatenating clips (without outro) ==="
cat > "$OUT_DIR/concat_clips.txt" << EOF
file 'segments/01_micron.mp4'
file 'segments/02_asml.mp4'
file 'segments/03_rfgan.mp4'
file 'segments/04_belllabs.mp4'
file 'segments/05_siltronic.mp4'
file 'segments/06_ficontec.mp4'
file 'segments/07_smd.mp4'
file 'segments/08_awg.mp4'
file 'segments/09_nordson.mp4'
file 'segments/10_turbopump.mp4'
file 'segments/11_microscale.mp4'
file 'segments/12_ehla.mp4'
file 'segments/13_eplus3d.mp4'
file 'segments/14_trulaser.mp4'
file 'segments/15_5axis.mp4'
file 'segments/16_sparknano.mp4'
file 'segments/17_xtoolp3.mp4'
file 'segments/18_xtoolf2.mp4'
file 'segments/19_makera.mp4'
EOF

ffmpeg -y -f concat -safe 0 -i "$OUT_DIR/concat_clips.txt" \
  -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p \
  "$OUT_DIR/clips_raw.mp4"

# Get actual clips duration
CLIPS_DUR=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$OUT_DIR/clips_raw.mp4")
echo "Actual clips duration: $CLIPS_DUR seconds"

# Calculate speed factor needed (speed > 1 means speed up, < 1 means slow down)
SPEED_FACTOR=$(echo "scale=6; $CLIPS_DUR / $TARGET_CLIPS_DUR" | bc)
echo "Speed adjustment factor: $SPEED_FACTOR"

echo "=== Step 2: Time-stretching clips to exactly $TARGET_CLIPS_DUR seconds ==="
# setpts=PTS/SPEED_FACTOR speeds up by SPEED_FACTOR
ffmpeg -y -i "$OUT_DIR/clips_raw.mp4" \
  -vf "setpts=PTS/$SPEED_FACTOR" -r 30 \
  -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p \
  "$OUT_DIR/clips_adjusted.mp4"

# Verify adjusted duration
ADJUSTED_DUR=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$OUT_DIR/clips_adjusted.mp4")
echo "Adjusted clips duration: $ADJUSTED_DUR seconds"

echo "=== Step 3: Concatenating adjusted clips + outro ==="
cat > "$OUT_DIR/concat_final.txt" << EOF
file 'clips_adjusted.mp4'
file 'outro_3s.mp4'
EOF

ffmpeg -y -f concat -safe 0 -i "$OUT_DIR/concat_final.txt" \
  -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p \
  "$OUT_DIR/hero_video_silent.mp4"

# Verify silent video duration
SILENT_DUR=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$OUT_DIR/hero_video_silent.mp4")
echo "Silent video duration: $SILENT_DUR seconds (should equal audio: $AUDIO_DUR)"

echo "=== Step 4: Adding audio track ==="
ffmpeg -y -i "$OUT_DIR/hero_video_silent.mp4" -i "$AUDIO" \
  -c:v copy -c:a aac -b:a 192k \
  -map 0:v:0 -map 1:a:0 \
  -movflags +faststart \
  "$OUT_DIR/realfab-hero-final.mp4"

echo "=== Done! ==="
echo "Output: $OUT_DIR/realfab-hero-final.mp4"

# Show final duration
FINAL_DUR=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$OUT_DIR/realfab-hero-final.mp4")
echo "Final video duration: $FINAL_DUR seconds"
echo "Audio duration was: $AUDIO_DUR seconds"

# Cleanup intermediate files
rm -f "$OUT_DIR/clips_raw.mp4" "$OUT_DIR/clips_adjusted.mp4"
rm -f "$OUT_DIR/concat_clips.txt" "$OUT_DIR/concat_final.txt"
echo "Cleaned up intermediate files"
