#!/bin/bash
# Render and encode the presenter cam loops (anim_host_pat.py) into course/media/render/host_pat_cam_<clip>.mp4.
# H.264 yuv420p, constant 24 fps, +faststart, no audio track. One Blender process at a time (slot wrapper).
# Usage: encode_host_cam.sh <frames dir> [clip ...]      env: BSLOT (Blender slot wrapper), SAMPLES, NOISE, CRF
#   RENDER=0 skips rendering and only encodes existing frames.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/../../media/render"
FR="${1:?frames dir}"; shift
CLIPS=("$@"); [ ${#CLIPS[@]} -eq 0 ] && CLIPS=(idle talk gesture empathy_idle empathy_talk)
BSLOT="${BSLOT:-blender}"
mkdir -p "$FR"
for c in "${CLIPS[@]}"; do
  if [ "${RENDER:-1}" = "1" ]; then
    "$BSLOT" -b --factory-startup --python "$HERE/anim_host_pat.py" -- clip="$c" out_dir="$FR" \
      samples="${SAMPLES:-96}" noise="${NOISE:-0.02}"
  fi
  ffmpeg -hide_banner -loglevel error -y -framerate 24 -i "$FR/${c}_%04d.png" \
    -c:v libx264 -preset slow -tune animation -crf "${CRF:-18}" -pix_fmt yuv420p -r 24 -an \
    -movflags +faststart "$OUT/host_pat_cam_${c}.mp4"
  echo "ENCODED $c $(stat -f %z "$OUT/host_pat_cam_${c}.mp4") bytes"
done
