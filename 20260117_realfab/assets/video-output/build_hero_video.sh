#!/bin/bash
# Hero Video Assembly Script - REVISED
# Target: 1920x1080 @ 30fps, H.264
# OLD FAB: clips 01-10, NEW FAB: clips 11-19

set -e
cd "$(dirname "$0")/.."

CLIPS_DIR="video-clips"
OUT_DIR="video-output"
mkdir -p "$OUT_DIR/segments"

# Common filter for scaling to 1920x1080 with black bars (letterbox/pillarbox)
SCALE_FILTER="scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black"

echo "=== Processing OLD FAB clips (Section 1: 0-30s) ==="

# 1. Inside Micron Taiwan - full 12.9s @ 6x = 2.15s
ffmpeg -y -i "$CLIPS_DIR/Inside Micron Taiwan.mov" \
  -vf "$SCALE_FILTER,setpts=PTS/6" -r 30 -an \
  -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p \
  "$OUT_DIR/segments/01_micron.mp4"

# 2. Unveiling High NA EUV ASML - 0-15s @ 5x = 3s
ffmpeg -y -ss 0 -t 15 -i "$CLIPS_DIR/Unveiling High NA EUV - ASML.mov" \
  -vf "$SCALE_FILTER,setpts=PTS/5" -r 30 -an \
  -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p \
  "$OUT_DIR/segments/02_asml.mp4"

# 3. RF GaN Fab Tour - 0-15s @ 5x = 3s
ffmpeg -y -ss 0 -t 15 -i "$CLIPS_DIR/RF GaN Experience - Fab Tour.mov" \
  -vf "$SCALE_FILTER,setpts=PTS/5" -r 30 -an \
  -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p \
  "$OUT_DIR/segments/03_rfgan.mp4"

# 4. Processing Bell Labs 1979 - 1:50-2:50 (60s) @ 20x = 3s
ffmpeg -y -ss 110 -t 60 -i "$CLIPS_DIR/Processing Integrated Circuits at Bell Labs (1979) - AT&T Archives.mp4" \
  -vf "$SCALE_FILTER,setpts=PTS/20" -r 30 -an \
  -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p \
  "$OUT_DIR/segments/04_belllabs.mp4"

# 5. Siltronic insights - 0-12s @ 4x = 3s
ffmpeg -y -ss 0 -t 12 -i "$CLIPS_DIR/Siltronic insights.mov" \
  -vf "$SCALE_FILTER,setpts=PTS/4" -r 30 -an \
  -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p \
  "$OUT_DIR/segments/05_siltronic.mp4"

# 6. ficonTEC Wafer-level - 0:53-1:08 (15s) @ 5x = 3s
ffmpeg -y -ss 53 -t 15 -i "$CLIPS_DIR/ficonTEC - Electro-optical Wafer-level Test Systems for PICs.mp4" \
  -vf "$SCALE_FILTER,setpts=PTS/5" -r 30 -an \
  -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p \
  "$OUT_DIR/segments/06_ficontec.mp4"

# 7. SMD Bestückung - 0:15-0:23 (8s) @ 4x = 2s
ffmpeg -y -ss 15 -t 8 -i "$CLIPS_DIR/SMD Bestueckung - SMD Bestückung.mp4" \
  -vf "$SCALE_FILTER,setpts=PTS/4" -r 30 -an \
  -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p \
  "$OUT_DIR/segments/07_smd.mp4"

# 8. AWG PLC Auto Align - 0-30s @ 10x = 3s (MOVED FROM NEW FAB)
ffmpeg -y -ss 0 -t 30 -i "$CLIPS_DIR/AWG PLC Auto Align and Cure.mp4" \
  -vf "$SCALE_FILTER,setpts=PTS/10" -r 30 -an \
  -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p \
  "$OUT_DIR/segments/08_awg.mp4"

# 9. Nordson ASYMTEK NexJet - 0-12s @ 4x = 3s (MOVED FROM NEW FAB)
ffmpeg -y -ss 0 -t 12 -i "$CLIPS_DIR/Nordson ASYMTEK NexJet.mov" \
  -vf "$SCALE_FILTER,setpts=PTS/4" -r 30 -an \
  -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p \
  "$OUT_DIR/segments/09_nordson.mp4"

# 10. Turbo pump explode - full 4.7s @ 1x (dramatic ending for old fab section)
ffmpeg -y -i "$CLIPS_DIR/Screen Recording 2026-01-18 at 11.48.12 - turbo pump explode.mov" \
  -vf "$SCALE_FILTER" -r 30 -an \
  -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p \
  "$OUT_DIR/segments/10_turbopump.mp4"

echo "=== Processing NEW FAB clips (Section 2: 30-68s) ==="

# 11. Microscale 3D printing spaceship - full 40s @ 8x = 5s
ffmpeg -y -i "$CLIPS_DIR/Microscale 3D printing spaceship.mov" \
  -vf "$SCALE_FILTER,setpts=PTS/8" -r 30 -an \
  -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p \
  "$OUT_DIR/segments/11_microscale.mp4"

# 12. EHLA Extreme High-speed Laser - 0-25s @ 5x = 5s
ffmpeg -y -ss 0 -t 25 -i "$CLIPS_DIR/EHLA Extreme High-speed Laser.mov" \
  -vf "$SCALE_FILTER,setpts=PTS/5" -r 30 -an \
  -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p \
  "$OUT_DIR/segments/12_ehla.mp4"

# 13. Eplus3D Metal 3D Printers - full 20s @ 8x = 2.5s
ffmpeg -y -i "$CLIPS_DIR/Eplus3D Metal 3D Printers Dual Lasers.mov" \
  -vf "$SCALE_FILTER,setpts=PTS/8" -r 30 -an \
  -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p \
  "$OUT_DIR/segments/13_eplus3d.mp4"

# 14. TruLaser Cell 7040 - full 15s @ 4x = 3.75s
ffmpeg -y -i "$CLIPS_DIR/TruLaser Cell 7040 Overview.mov" \
  -vf "$SCALE_FILTER,setpts=PTS/4" -r 30 -an \
  -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p \
  "$OUT_DIR/segments/14_trulaser.mp4"

# 15. 5-Axis Cutting Silicon Carbide - full 20s @ 5x = 4s
ffmpeg -y -i "$CLIPS_DIR/5-Axis Cutting Silicon Carbide.mov" \
  -vf "$SCALE_FILTER,setpts=PTS/5" -r 30 -an \
  -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p \
  "$OUT_DIR/segments/15_5axis.mp4"

# 16. SparkNano Omega - 0-25s @ 5x = 5s
ffmpeg -y -ss 0 -t 25 -i "$CLIPS_DIR/SparkNano Omega.mov" \
  -vf "$SCALE_FILTER,setpts=PTS/5" -r 30 -an \
  -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p \
  "$OUT_DIR/segments/16_sparknano.mp4"

# 17. xTool P3 - 0:02-0:14 (12s) @ 3x = 4s
ffmpeg -y -ss 2 -t 12 -i "$CLIPS_DIR/xTool P3 Sneak Peek ｜ The Next-Generation CO2 Laser Flagship.mp4" \
  -vf "$SCALE_FILTER,setpts=PTS/3" -r 30 -an \
  -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p \
  "$OUT_DIR/segments/17_xtoolp3.mp4"

# 18. xTool F2 - 0:16-0:28 (12s) @ 3x = 4s
ffmpeg -y -ss 16 -t 12 -i "$CLIPS_DIR/xTool F2 ｜ A New Star in Portable Lasers — Fast. Easy. Powerful..f616.mp4" \
  -vf "$SCALE_FILTER,setpts=PTS/3" -r 30 -an \
  -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p \
  "$OUT_DIR/segments/18_xtoolf2.mp4"

# 19. Makera Z1 Unicorn - full 10s @ 2.5x = 4s
ffmpeg -y -i "$CLIPS_DIR/Makera Z1 Unicorn.mov" \
  -vf "$SCALE_FILTER,setpts=PTS/2.5" -r 30 -an \
  -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p \
  "$OUT_DIR/segments/19_makera.mp4"

echo "=== All segments processed ==="
echo "Next step: Run concatenate script"

