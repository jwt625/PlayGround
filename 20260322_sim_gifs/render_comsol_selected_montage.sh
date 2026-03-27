#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELECTION_JSON="${1:-${ROOT_DIR}/build/comsol_release_browser/export_20260327.json}"

WIDTH="${WIDTH:-1280}"
HEIGHT="${HEIGHT:-720}"
FPS="${FPS:-30}"
CRF="${CRF:-21}"
PRESET="${PRESET:-medium}"
VIDEO_CODEC="${VIDEO_CODEC:-libx264}"
PIX_FMT="${PIX_FMT:-yuv420p}"
PROFILE="${PROFILE:-high}"
LEVEL="${LEVEL:-4.1}"
OUTPUT_NAME="${OUTPUT_NAME:-comsol_release_selected_montage_1280x720_h264.mp4}"
OUTPUT_PATH="${ROOT_DIR}/build/${OUTPUT_NAME}"
FONT_FILE="${FONT_FILE:-/System/Library/Fonts/Supplemental/Arial Black.ttf}"
POINTSIZE="${POINTSIZE:-34}"
STROKEWIDTH="${STROKEWIDTH:-2}"
TITLE_HEIGHT="${TITLE_HEIGHT:-130}"

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
require_tool ffprobe
require_tool python3
require_tool magick

if [[ ! -f "$SELECTION_JSON" ]]; then
  echo "Missing selection JSON: $SELECTION_JSON" >&2
  exit 1
fi

if [[ ! -f "$FONT_FILE" ]]; then
  echo "Missing font file: $FONT_FILE" >&2
  exit 1
fi

mkdir -p "${ROOT_DIR}/build"

tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/comsol_montage.XXXXXX")"
cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

manifest_tsv="${tmp_dir}/selected_manifest.tsv"
segments_txt="${tmp_dir}/segments.txt"
stats_txt="${tmp_dir}/stats.txt"
subtitle_log_path="${ROOT_DIR}/build/$(basename "${OUTPUT_PATH%.*}")_subtitle_log.tsv"

python3 - "$SELECTION_JSON" "$manifest_tsv" <<'PY'
import json
import re
import sys
from pathlib import Path

selection_path = Path(sys.argv[1]).resolve()
manifest_path = Path(sys.argv[2]).resolve()

def speed_for(duration: float) -> int:
    if 1 <= duration < 2:
        return 2
    if 2 <= duration < 4:
        return 3
    if 4 <= duration < 10:
        return 4
    if 10 <= duration <= 30:
        return 5
    return 1

def caption_from_item(item: dict) -> str:
    src = item.get("src") or ""
    release = str(item.get("release", ""))
    page_slug = str(item.get("page_slug", ""))
    media_id = str(item.get("media_id", ""))
    basename = Path(src).stem
    prefix = f"{release}_{page_slug}_"
    if basename.startswith(prefix):
        basename = basename[len(prefix):]
    if media_id and basename.endswith(f"_{media_id}"):
        basename = basename[: -(len(media_id) + 1)]
    tokens = [token for token in basename.replace("_", " ").split() if token]
    cleaned = []
    for token in tokens:
        lower = token.lower()
        if lower in {"animation", "screen", "recording", "promotional", "movie"}:
            continue
        if lower == "rh":
            continue
        if re.match(r"^(?:\d+(?:\.\d+)?)?rh(?:\d+(?:\.\d+)?)?$", lower):
            continue
        if re.match(r"^rh\d+(?:\.\d+)?$", lower):
            continue
        cleaned.append(token)
    core = " ".join(cleaned).strip()
    return f"{core} (v{release})".strip()

data = json.loads(selection_path.read_text())
if not isinstance(data, list):
    raise SystemExit("Expected selection JSON to be a list")

with manifest_path.open("w", encoding="utf-8") as handle:
    handle.write("index\tid\trelease\tpage_slug\tduration\tspeed\tcaption\tpath\n")
    for index, item in enumerate(data, start=1):
        src = item.get("src")
        if not src:
            raise SystemExit(f"Missing src for item #{index}: {item}")
        absolute_src = (selection_path.parent / src).resolve()
        duration = float(item.get("duration", 0))
        speed = speed_for(duration)
        caption = caption_from_item(item)
        fields = [
            str(index),
            str(item.get("id", "")),
            str(item.get("release", "")),
            str(item.get("page_slug", "")),
            f"{duration:.6f}",
            str(speed),
            caption,
            str(absolute_src),
        ]
        handle.write("\t".join(fields) + "\n")
PY

: > "$segments_txt"
printf "index\trelease\tpage_slug\tcaption\tline1\tline2\tinput_file\n" > "$subtitle_log_path"

clip_count=0
raw_total=0
output_total=0

split_caption() {
  python3 - "$1" <<'PY'
import sys
text = sys.argv[1].strip()
if len(text) <= 34:
    print("")
    print(text)
    raise SystemExit
words = text.split()
mid = max(1, len(words) // 2)
best = None
for i in range(1, len(words)):
    left = " ".join(words[:i])
    right = " ".join(words[i:])
    score = abs(len(left) - len(right))
    if best is None or score < best[0]:
        best = (score, left, right)
print(best[1] if best else "")
print(best[2] if best else text)
PY
}

render_title_card() {
  local out="$1"
  local line1="$2"
  local line2="$3"
  local pointsize="$POINTSIZE"

  if [[ ${#line2} -gt 36 ]]; then
    pointsize=30
  fi

  if [[ -n "$line1" ]]; then
    magick -size "${WIDTH}x${TITLE_HEIGHT}" xc:none \
      -font "$FONT_FILE" \
      -gravity south \
      -fill white -stroke black -strokewidth "$STROKEWIDTH" -pointsize "$pointsize" \
      -annotate +0+68 "$line1" \
      -annotate +0+24 "$line2" \
      "$out"
  else
    magick -size "${WIDTH}x${TITLE_HEIGHT}" xc:none \
      -font "$FONT_FILE" \
      -gravity south \
      -fill white -stroke black -strokewidth "$STROKEWIDTH" -pointsize "$pointsize" \
      -annotate +0+24 "$line2" \
      "$out"
  fi
}

while IFS=$'\t' read -r index item_id release page_slug duration speed caption input_path; do
  [[ "$index" == "index" ]] && continue

  if [[ ! -f "$input_path" ]]; then
    echo "Missing input media: $input_path" >&2
    exit 1
  fi

  clip_count=$((clip_count + 1))
  pts_factor="$(awk -v s="$speed" 'BEGIN { printf "%.6f", 1.0 / s }')"
  output_duration="$(awk -v d="$duration" -v s="$speed" 'BEGIN { printf "%.6f", d / s }')"
  clip_path="${tmp_dir}/clip_$(printf '%03d' "$clip_count").mp4"
  title_png="${tmp_dir}/title_$(printf '%03d' "$clip_count").png"
  split_output="$(split_caption "$caption")"
  line1="$(printf '%s\n' "$split_output" | sed -n '1p')"
  line2="$(printf '%s\n' "$split_output" | sed -n '2p')"
  line1="${line1:-}"
  line2="${line2:-$caption}"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$index" "$release" "$page_slug" "$caption" "$line1" "$line2" "$(basename "$input_path")" >> "$subtitle_log_path"
  render_title_card "$title_png" "$line1" "$line2"

  ffmpeg -nostdin -y -v warning \
    -i "$input_path" \
    -i "$title_png" \
    -an \
    -filter_complex "[0:v]setpts=${pts_factor}*PTS,fps=${FPS},scale=${WIDTH}:${HEIGHT}:force_original_aspect_ratio=decrease:flags=lanczos,pad=${WIDTH}:${HEIGHT}:(ow-iw)/2:(oh-ih)/2:black,format=${PIX_FMT}[base];[base][1:v]overlay=0:H-h:format=auto[v]" \
    -map "[v]" \
    -c:v "$VIDEO_CODEC" \
    -preset "$PRESET" \
    -crf "$CRF" \
    -profile:v "$PROFILE" \
    -level:v "$LEVEL" \
    -movflags +faststart \
    -pix_fmt "$PIX_FMT" \
    "$clip_path"

  printf "file '%s'\n" "$(escape_for_concat "$clip_path")" >> "$segments_txt"

  raw_total="$(awk -v a="$raw_total" -v b="$duration" 'BEGIN { printf "%.6f", a + b }')"
  output_total="$(awk -v a="$output_total" -v b="$output_duration" 'BEGIN { printf "%.6f", a + b }')"
done < "$manifest_tsv"

if [[ "$clip_count" -eq 0 ]]; then
  echo "No clips found in selection JSON." >&2
  exit 1
fi

ffmpeg -nostdin -y -v warning \
  -f concat -safe 0 -i "$segments_txt" \
  -an \
  -c:v "$VIDEO_CODEC" \
  -preset "$PRESET" \
  -crf "$CRF" \
  -profile:v "$PROFILE" \
  -level:v "$LEVEL" \
  -movflags +faststart \
  -pix_fmt "$PIX_FMT" \
  "$OUTPUT_PATH"

{
  printf "clips\t%s\n" "$clip_count"
  printf "raw_seconds\t%s\n" "$raw_total"
  printf "output_seconds\t%s\n" "$output_total"
  printf "output_path\t%s\n" "$OUTPUT_PATH"
  printf "subtitle_log\t%s\n" "$subtitle_log_path"
} > "$stats_txt"

cat "$stats_txt"
