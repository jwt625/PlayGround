#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${1:-${ROOT_DIR}/still_image_shortlist.tsv}"
BUILD_DIR="${ROOT_DIR}/build"
CLIP_DIR="${BUILD_DIR}/still_shortlist_clips"
SEGMENTS_LIST="${BUILD_DIR}/still_shortlist_segments.txt"
OUT_MP4="${BUILD_DIR}/still_image_shortlist_preview.mp4"
OUT_CONTACT="${BUILD_DIR}/still_image_shortlist_contact_sheet.jpg"

WIDTH="${WIDTH:-1080}"
HEIGHT="${HEIGHT:-1350}"
FPS="${FPS:-30}"
START_SECONDS="${START_SECONDS:-0.6}"
MID_SECONDS="${MID_SECONDS:-0.1}"
END_SECONDS="${END_SECONDS:-0.1}"
SECOND_STAGE_COUNT="${SECOND_STAGE_COUNT:-0}"
CRF="${CRF:-19}"
PRESET="${PRESET:-medium}"

require_tool() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required tool: $1" >&2
    exit 1
  }
}

escape_for_concat() {
  printf "%s" "$1" | sed "s/'/'\\\\''/g"
}

require_tool ffmpeg
require_tool magick

mkdir -p "$CLIP_DIR"
: > "$SEGMENTS_LIST"

ENTRIES=()
while IFS= read -r line; do
  ENTRIES+=("$line")
done < <(awk -F '\t' 'NR>1 && $1 !~ /^#/ && NF>=3 {print}' "$MANIFEST")
total="${#ENTRIES[@]}"
count=0

for entry in "${ENTRIES[@]}"; do
  IFS=$'\t' read -r idx title input_path <<< "$entry"
  [[ -f "$input_path" ]] || {
    echo "Missing input image: $input_path" >&2
    exit 1
  }

  count=$((count + 1))
  clip_path="${CLIP_DIR}/$(printf '%03d' "$count").mp4"
  if [[ "$total" -le 1 ]]; then
    duration="$START_SECONDS"
  elif [[ "$SECOND_STAGE_COUNT" -gt 0 && "$SECOND_STAGE_COUNT" -lt "$total" ]]; then
    first_stage_count=$((total - SECOND_STAGE_COUNT))
    if [[ "$count" -le "$first_stage_count" ]]; then
      if [[ "$first_stage_count" -le 1 ]]; then
        duration="$START_SECONDS"
      else
        duration="$(awk -v i="$count" -v n="$first_stage_count" -v a="$START_SECONDS" -v b="$MID_SECONDS" 'BEGIN { printf "%.6f", a + (b-a) * ((i-1)/(n-1)) }')"
      fi
    else
      second_index=$((count - first_stage_count))
      if [[ "$SECOND_STAGE_COUNT" -le 1 ]]; then
        duration="$MID_SECONDS"
      else
        duration="$(awk -v i="$second_index" -v n="$SECOND_STAGE_COUNT" -v a="$MID_SECONDS" -v b="$END_SECONDS" 'BEGIN { printf "%.6f", a + (b-a) * ((i-1)/(n-1)) }')"
      fi
    fi
  else
    duration="$(awk -v i="$count" -v n="$total" -v a="$START_SECONDS" -v b="$END_SECONDS" 'BEGIN { printf "%.6f", a + (b-a) * ((i-1)/(n-1)) }')"
  fi

  ffmpeg -nostdin -y -v warning \
    -loop 1 -t "$duration" -i "$input_path" \
    -vf "scale=${WIDTH}:${HEIGHT}:force_original_aspect_ratio=decrease:flags=lanczos,pad=${WIDTH}:${HEIGHT}:(ow-iw)/2:(oh-ih)/2:black,fps=${FPS},format=yuv420p" \
    -an \
    -c:v libx264 \
    -preset "$PRESET" \
    -crf "$CRF" \
    -pix_fmt yuv420p \
    -movflags +faststart \
    "$clip_path"

  printf "file '%s'\n" "$(escape_for_concat "$clip_path")" >> "$SEGMENTS_LIST"
done

ffmpeg -nostdin -y -v warning -f concat -safe 0 -i "$SEGMENTS_LIST" -c copy "$OUT_MP4"

montage $(awk -F '\t' 'NR>1 {print $3}' "$MANIFEST") \
  -thumbnail '320x320>' -tile 5x6 -geometry +8+8 \
  -background black "$OUT_CONTACT"

printf "Built preview: %s\n" "$OUT_MP4"
printf "Built contact sheet: %s\n" "$OUT_CONTACT"
