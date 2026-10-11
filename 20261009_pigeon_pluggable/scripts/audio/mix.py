"""Final mix: VO + SFX + music bed, SFX and music sidechain-ducked under the VO, mastered to -16 LUFS / TP <= -1.5 dBTP.

Run (project root): uv run --no-project python scripts/audio/mix.py [--date YYYYMMDD] [--no-music]
Inputs: outputs/audio/vo_<date>.wav, sfx_<date>.wav, music_<date>.wav. Output: outputs/audio/mix_v1_<date>.wav
(48 kHz stereo PCM 24, 62.0 s).

Stem levels are set by measured loudness (ffmpeg ebur128), not by fixed gains: SFX bed SFX_LU below the VO, music
MUSIC_LU below the VO, before ducking. Ducking: ffmpeg sidechaincompress keyed by the VO (music deeper than SFX, so
sync hits still read under speech). Master: audio_util.master (gain to -16 LUFS, 4x-oversampled limiter, linear
loudnorm trim).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_util import OUT_AUDIO, TODAY, TOTAL, Log, ebur128, master, rel, run  # noqa: E402

SFX_LU = 7.0      # SFX stem integrated loudness, LU under the VO stem (before ducking)
MUSIC_LU = 13.0   # music stem, LU under the VO stem (before ducking)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=TODAY)
    ap.add_argument("--no-music", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--vo", default=None, help="VO wav (default outputs/audio/vo_<date>.wav)")
    ap.add_argument("--music", default=None, help="music bed wav (default outputs/audio/music_<date>.wav, procedural)")
    ap.add_argument("--sfx", default=None, help="SFX wav (default outputs/audio/sfx_<date>.wav)")
    ap.add_argument("--music-lu", type=float, default=MUSIC_LU, help="music stem loudness under the VO stem, LU")
    args = ap.parse_args()
    log = Log("mix")
    vo, sfx, mus = (os.path.join(OUT_AUDIO, "%s_%s.wav" % (k, args.date)) for k in ("vo", "sfx", "music"))
    vo = args.vo or vo
    sfx = args.sfx or sfx
    mus = args.music or mus
    use_music = not args.no_music and os.path.exists(mus)
    L = {k: ebur128(p) for k, p in (("vo", vo), ("sfx", sfx)) + ((("music", mus),) if use_music else ())}
    for k, m in L.items():
        log("stem %-5s I %6.1f LUFS  LRA %4.1f LU  TP %5.1f dBTP" % (k, m["I"], m["LRA"], m["TP"]))
    g_sfx = (L["vo"]["I"] - SFX_LU) - L["sfx"]["I"]
    g_mus = (L["vo"]["I"] - args.music_lu) - L["music"]["I"] if use_music else 0.0
    log("stem gains: sfx %+.1f dB (-> %.1f LU under VO), music %+.1f dB (-> %.1f LU under VO)" % (g_sfx, SFX_LU, g_mus, args.music_lu))
    pre = os.path.join(OUT_AUDIO, "_premix.wav")
    fc = ("[0:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,asplit=3[vo][sc1][sc2];"
          "[1:a]aformat=sample_fmts=fltp:channel_layouts=stereo,volume=%.2fdB[sfx];"
          "[sfx][sc1]sidechaincompress=threshold=0.04:ratio=2.5:attack=15:release=250:knee=4[sfxd];" % g_sfx)
    ins = ["-i", vo, "-i", sfx]
    if use_music:
        ins += ["-i", mus]
        fc += ("[2:a]aformat=sample_fmts=fltp:channel_layouts=stereo,volume=%.2fdB[mus];"
               "[mus][sc2]sidechaincompress=threshold=0.02:ratio=6:attack=40:release=500:knee=6[musd];"
               "[vo][sfxd][musd]amix=inputs=3:normalize=0:duration=longest,atrim=0:%g[out]" % (g_mus, TOTAL))
    else:
        fc += "[sc2]anullsink;[vo][sfxd]amix=inputs=2:normalize=0:duration=longest,atrim=0:%g[out]" % TOTAL
    run(["ffmpeg", "-hide_banner", "-nostats", "-y", *ins, "-filter_complex", fc, "-map", "[out]", "-c:a", "pcm_f32le", pre], log)
    out = args.out or os.path.join(OUT_AUDIO, "mix_v1_%s.wav" % args.date)
    master(pre, out, I=-16.0, TP=-1.5, log=log)
    os.remove(pre)
    m = ebur128(out)
    # stem gains as they sit in the final mix (stem gain + master gain), for plot_audio.py
    with open(os.path.splitext(out)[0] + ".json", "w") as f:
        json.dump({"stem_gain_db": {"vo": -3.01, "sfx": round(g_sfx, 2), "music": round(g_mus, 2)},
                   "note": "vo -3.01 dB = mono to stereo pan law; master trim not included", "loudness": m}, f, indent=1)
    log("stems: vo %s, sfx %s, music %s" % (rel(vo), rel(sfx), rel(mus) if use_music else "none"))
    log("mix: I %.1f LUFS, LRA %.1f LU, TP %.1f dBTP -> %s" % (m["I"], m["LRA"], m["TP"], rel(out)))
    if not (abs(m["I"] + 16.0) <= 0.5 and m["TP"] <= -1.5):
        log("WARNING: loudness/true-peak target missed")
    log("log", rel(log.path))


if __name__ == "__main__":
    main()
