#!/usr/bin/env bash
# Re-runnable final assembly of film v4 (DevLog-014 sections 15-17). Run from the project root:
#   scripts/post/finalize_v4.sh [--audio]
# --audio : also snapshot scripts/audio/scene_events.json and re-run SFX + mix (use after S / M change event times).
# Steps: [SFX + mix] -> callouts + headers + beam note (labelled copies, fresh folder swapped in) -> silent picture
# (intro xfade) -> captions + mux -> stream / loudness check. Clean frames are never written.
set -euo pipefail
cd "$(dirname "$0")/../.."
T=$(date +%Y%m%d_%H%M%S)
LOG=outputs/logs/finalize_v4_$T.log
exec > >(tee "$LOG") 2>&1
echo "finalize_v4 $T"

if [[ "${1:-}" == "--audio" ]]; then
  SNAP=outputs/audio/scene_events_film_v4_$T.json
  cp scripts/audio/scene_events.json "$SNAP"
  echo "events snapshot $SNAP md5 $(md5 -q "$SNAP")"
  uv run --no-project --with numpy --with soundfile python scripts/audio/make_sfx.py --events "$SNAP" \
    --overrides scripts/audio/event_overrides_v4c.json --out outputs/audio/sfx_v4_20261010.wav
  uv run --no-project python scripts/audio/mix.py --vo outputs/audio/vo_20261010_rev6.wav --sfx outputs/audio/sfx_v4_20261010.wav \
    --music outputs/audio/music_plague_20261010.wav --music-lu 14 --out outputs/audio/mix_v4_20261010.wav
fi

NEW=outputs/film_v4_frames_lab.new
rm -rf "$NEW"
uv run --no-project --with pillow python scripts/post/callout_overlay.py --json outputs/film_v4_callouts.json \
  --texts outputs/film_v4_texts.json --frames-in outputs/film_v4_frames --frames-out "$NEW"
rm -rf outputs/film_v4_frames_lab
mv "$NEW" outputs/film_v4_frames_lab

uv run --no-project python scripts/post/assemble_film.py --out outputs/film_v4_1_20261010.mp4 --end 2012 --dur 67.067
uv run --no-project --with pillow python scripts/post/burn_captions.py --video outputs/film_v4_1_20261010.mp4 \
  --audio outputs/audio/mix_v4_20261010.wav --out outputs/film_v4_1_20261010_av.mp4

ffprobe -v error -show_entries stream=codec_name,pix_fmt,width,height,r_frame_rate,nb_frames,sample_rate,channels,bit_rate,duration:format=duration \
  -of compact outputs/film_v4_1_20261010_av.mp4
uv run --no-project python -c "
import sys; sys.path.insert(0, 'scripts/audio'); from audio_util import ebur128
print('loudness', ebur128('outputs/film_v4_1_20261010_av.mp4'))
h = open('outputs/film_v4_1_20261010_av.mp4', 'rb').read(65536)
print('faststart', h.find(b'moov') != -1 and (h.find(b'mdat') == -1 or h.find(b'moov') < h.find(b'mdat')))"
echo "done; log $LOG"
