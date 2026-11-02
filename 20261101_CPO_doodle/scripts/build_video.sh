#!/bin/bash
# Rebuild the final videos: render silent loops, then mux the soundtrack crop (bar line 19.331 s, 8 bars, 10 ms fade at each end).
# Run from anywhere. Needs uv, ffmpeg, and a Playwright Chromium headless shell. About 1 minute per video.
set -euo pipefail
cd "$(dirname "$0")/.."
TRACK="Mittsies - Vitality.m4a"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP" cache' EXIT
ffmpeg -v error -y -ss 19.331 -t 12.8 -i "$TRACK" -af "afade=t=in:d=0.01,afade=t=out:st=12.79:d=0.01" -c:a pcm_s16le -ar 44100 "$TMP/crop.wav"
build() { # name, scene query
  (cd scripts && uv run --no-project --with playwright python render_loop.py --query "$2" --tag "$1" --crf 20 --out "$TMP/$1.mp4")
  ffmpeg -v error -y -i "$TMP/$1.mp4" -i "$TMP/crop.wav" -map 0:v -map 1:a -c:v copy -c:a aac -b:a 192k -movflags +faststart -t 12.8 "outputs/$1.mp4"
  echo "outputs/$1.mp4"
}
build cpo_line_loop_sway ""
build cpo_line_loop "dance=0"
