#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${1:-${ROOT_DIR}/montage_manifest_final_mp4.tsv}"
BUILD_DIR="${ROOT_DIR}/build"
CLIP_DIR="${BUILD_DIR}/clips_final_mp4"
TITLE_DIR="${BUILD_DIR}/title_cards_final_mp4"
SEGMENTS_LIST="${BUILD_DIR}/segments_final_mp4.txt"
REPORT_PATH="${BUILD_DIR}/report_final_mp4.tsv"
SAMPLE_FRAME="${BUILD_DIR}/subtitle_style_final_sample.png"

WIDTH="${WIDTH:-1080}"
HEIGHT="${HEIGHT:-1350}"
FPS="${FPS:-30}"
BASE_SPEED="${BASE_SPEED:-3.0}"
LUMERICAL_SPEED="${LUMERICAL_SPEED:-20.0}"
SHORT_CLIP_SPEED_MID="${SHORT_CLIP_SPEED_MID:-2.0}"
SHORT_CLIP_SPEED_MIN="${SHORT_CLIP_SPEED_MIN:-1.5}"
SHORT_CLIP_THRESHOLD_MID="${SHORT_CLIP_THRESHOLD_MID:-0.75}"
SHORT_CLIP_THRESHOLD_MIN="${SHORT_CLIP_THRESHOLD_MIN:-0.50}"
CRF_MP4="${CRF_MP4:-19}"
PRESET_MP4="${PRESET_MP4:-medium}"
FONT_FILE="${FONT_FILE:-/System/Library/Fonts/Supplemental/Arial Black.ttf}"
POINTSIZE="${POINTSIZE:-42}"
STROKEWIDTH="${STROKEWIDTH:-2}"
TITLE_HEIGHT="${TITLE_HEIGHT:-190}"
LUMERICAL_MARKER="${LUMERICAL_MARKER:-/lumerical/}"

MANIFEST_STEM="$(basename "${MANIFEST}")"
MANIFEST_STEM="${MANIFEST_STEM%.*}"
base_speed_tag="$(printf '%s' "$BASE_SPEED" | sed 's/[^0-9]/_/g')"
lumerical_speed_tag="$(printf '%s' "$LUMERICAL_SPEED" | sed 's/[^0-9]/_/g')"
FINAL_MP4="${BUILD_DIR}/${MANIFEST_STEM}_base${base_speed_tag}x_lumerical${lumerical_speed_tag}x_subtitled_twoline_4x5.mp4"

require_tool() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required tool: $1" >&2
    exit 1
  }
}

escape_for_concat() {
  printf "%s" "$1" | sed "s/'/'\\\\''/g"
}

calc_duration() {
  local duration=""
  duration="$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$1" 2>/dev/null || true)"
  if [[ -z "$duration" ]]; then
    echo "Unable to determine duration for: $1" >&2
    exit 1
  fi
  printf "%s\n" "$duration"
}

split_title() {
  local title="$1"
  if [[ "$title" == *" - "* ]]; then
    printf "%s\t%s\n" "${title%% - *}" "${title#* - }"
  else
    printf "%s\t%s\n" "" "$title"
  fi
}

render_title_card() {
  local out="$1"
  local line1="$2"
  local line2="$3"
  local pointsize="$POINTSIZE"

  if [[ ${#line2} -gt 38 ]]; then
    pointsize=36
  fi

  if [[ -n "$line1" ]]; then
    magick -size "${WIDTH}x${TITLE_HEIGHT}" xc:none \
      -font "$FONT_FILE" \
      -gravity south \
      -fill white -stroke black -strokewidth "$STROKEWIDTH" -pointsize "$pointsize" \
      -annotate +0+95 "$line1" \
      -annotate +0+40 "$line2" \
      "$out"
  else
    magick -size "${WIDTH}x${TITLE_HEIGHT}" xc:none \
      -font "$FONT_FILE" \
      -gravity south \
      -fill white -stroke black -strokewidth "$STROKEWIDTH" -pointsize "$pointsize" \
      -annotate +0+40 "$line2" \
      "$out"
  fi
}

require_tool ffmpeg
require_tool ffprobe
require_tool awk
require_tool magick

if [[ ! -f "$MANIFEST" ]]; then
  echo "Missing manifest: $MANIFEST" >&2
  exit 1
fi

if [[ ! -f "$FONT_FILE" ]]; then
  echo "Missing font file: $FONT_FILE" >&2
  exit 1
fi

mkdir -p "$CLIP_DIR" "$TITLE_DIR"
: > "$SEGMENTS_LIST"
printf "index\tchapter\ttitle\tinput\tinput_seconds\tspeed\toutput_seconds\n" > "$REPORT_PATH"

clip_count=0
total_input=0
total_output=0
sample_done=0

while IFS=$'\t' read -r chapter title input_path; do
  [[ -z "${chapter}" ]] && continue
  [[ "${chapter}" =~ ^# ]] && continue

  if [[ ! -f "$input_path" ]]; then
    echo "Missing input media: $input_path" >&2
    exit 1
  fi

  clip_count=$((clip_count + 1))
  slug="$(printf '%03d_%s' "$clip_count" "${title}" | tr ' /' '__' | tr -cd '[:alnum:]_.-\n')"
  clip_path="${CLIP_DIR}/${slug}.mp4"
  title_png="${TITLE_DIR}/${slug}.png"

  speed="$BASE_SPEED"
  input_duration="$(calc_duration "$input_path")"
  base_output_duration="$(awk -v d="$input_duration" -v s="$BASE_SPEED" 'BEGIN { printf "%.6f", d / s }')"
  if [[ "$input_path" == *"${LUMERICAL_MARKER}"* ]]; then
    speed="$LUMERICAL_SPEED"
  elif awk -v d="$base_output_duration" -v t="$SHORT_CLIP_THRESHOLD_MIN" 'BEGIN { exit !(d < t) }'; then
    speed="$SHORT_CLIP_SPEED_MIN"
  elif awk -v d="$base_output_duration" -v t="$SHORT_CLIP_THRESHOLD_MID" 'BEGIN { exit !(d < t) }'; then
    speed="$SHORT_CLIP_SPEED_MID"
  fi
  pts_factor="$(awk -v s="$speed" 'BEGIN { printf "%.6f", 1.0 / s }')"

  output_duration="$(awk -v d="$input_duration" -v s="$speed" 'BEGIN { printf "%.6f", d / s }')"

  IFS=$'\t' read -r line1 line2 < <(split_title "$title")
  render_title_card "$title_png" "$line1" "$line2"

  ffmpeg -nostdin -y -v warning \
    -i "$input_path" \
    -i "$title_png" \
    -filter_complex "[0:v]setpts=${pts_factor}*PTS,scale=${WIDTH}:${HEIGHT}:force_original_aspect_ratio=decrease:flags=lanczos,pad=${WIDTH}:${HEIGHT}:(ow-iw)/2:(oh-ih)/2:black,fps=${FPS},format=yuv420p[base];[base][1:v]overlay=0:H-h:format=auto[v]" \
    -map "[v]" \
    -an \
    -c:v libx264 \
    -preset "$PRESET_MP4" \
    -crf "$CRF_MP4" \
    -pix_fmt yuv420p \
    -movflags +faststart \
    "$clip_path"

  if [[ "$sample_done" -eq 0 ]]; then
    ffmpeg -nostdin -y -v warning \
      -i "$input_path" \
      -i "$title_png" \
      -filter_complex "[0:v]scale=${WIDTH}:${HEIGHT}:force_original_aspect_ratio=decrease:flags=lanczos,pad=${WIDTH}:${HEIGHT}:(ow-iw)/2:(oh-ih)/2:black,select=eq(n\\,0),format=yuv420p[base];[base][1:v]overlay=0:H-h:format=auto[v]" \
      -map "[v]" \
      -frames:v 1 -update 1 \
      "$SAMPLE_FRAME"
    sample_done=1
  fi

  printf "file '%s'\n" "$(escape_for_concat "$clip_path")" >> "$SEGMENTS_LIST"
  printf "%s\t%s\t%s\t%s\t%.6f\t%s\t%.6f\n" \
    "$clip_count" "$chapter" "$title" "$input_path" "$input_duration" "$speed" "$output_duration" >> "$REPORT_PATH"

  total_input="$(awk -v a="$total_input" -v b="$input_duration" 'BEGIN { printf "%.6f", a + b }')"
  total_output="$(awk -v a="$total_output" -v b="$output_duration" 'BEGIN { printf "%.6f", a + b }')"
done < "$MANIFEST"

ffmpeg -nostdin -y -v warning -f concat -safe 0 -i "$SEGMENTS_LIST" -c copy "$FINAL_MP4"

printf "\nBuilt montage assets:\n"
printf "  MP4: %s\n" "$FINAL_MP4"
printf "  Report: %s\n" "$REPORT_PATH"
printf "  Sample frame: %s\n" "$SAMPLE_FRAME"
printf "\nStats:\n"
printf "  Clips: %s\n" "$clip_count"
printf "  Raw total duration: %s sec\n" "$total_input"
printf "  Planned output duration: %s sec\n" "$total_output"
printf "  Direct cuts: %s\n" "$(( clip_count > 0 ? clip_count - 1 : 0 ))"
