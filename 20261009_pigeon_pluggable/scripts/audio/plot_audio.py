"""Check image for the mix: stem envelopes (VO, SFX, music, mix) in dBFS with the narration windows shaded,
short-term loudness (3 s) of the mix, and a spectrogram of the mix. Scene boundaries every 10 s.

Run (project root): uv run --no-project --with numpy --with soundfile --with matplotlib python scripts/audio/plot_audio.py [--date YYYYMMDD]
Writes outputs/audio/mix_v1_<date>_check.png.
"""
import argparse
import json
import os
import re
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_util import HERE, OUT_AUDIO, SCENE_BOUNDS, TODAY, TOTAL, rel  # noqa: E402


def env_db(path, hop=0.02, gain_db=0.0):
    a, sr = sf.read(path, always_2d=True)
    a = a * 10 ** (gain_db / 20)
    a = a.mean(1) if a.shape[1] > 1 else a[:, 0]
    h = int(hop * sr)
    n = len(a) // h
    r = np.sqrt(np.mean(a[:n * h].reshape(n, h) ** 2, axis=1))
    return np.arange(n) * hop, 20 * np.log10(r + 1e-9)


def short_term(path):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-loglevel", "verbose", "-i", path, "-af", "ebur128", "-f", "null", "-"],
                       capture_output=True, text=True)
    t, s = [], []
    for m in re.finditer(r"t:\s*([\d.]+)\s+TARGET.*?S:\s*(-?[\d.]+|-inf)", r.stderr):
        t.append(float(m.group(1)))
        s.append(float(m.group(2)) if m.group(2) != "-inf" else np.nan)
    return np.array(t), np.array(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=TODAY)
    ap.add_argument("--mix", default=None, help="mix wav (default outputs/audio/mix_v1_<date>.wav); sidecar json and png follow its name")
    ap.add_argument("--vo", default=None, help="VO wav (default outputs/audio/vo_<date>.wav)")
    ap.add_argument("--sfx", default=None)
    ap.add_argument("--music", default=None)
    args = ap.parse_args()
    d = args.date
    f = {k: os.path.join(OUT_AUDIO, "%s_%s.wav" % (k, d)) for k in ("vo", "sfx", "music", "mix_v1")}
    if args.mix:
        f["mix_v1"] = args.mix
    for k in ("vo", "sfx", "music"):
        if getattr(args, k):
            f[k] = getattr(args, k)
    base = os.path.splitext(f["mix_v1"])[0]
    lines = json.load(open(os.path.join(HERE, "narration.json")))["lines"]
    plt.rcParams.update({"font.size": 8, "axes.facecolor": "#111317", "figure.facecolor": "#0b0c0f", "text.color": "#ccc",
                         "axes.labelcolor": "#ccc", "xtick.color": "#aaa", "ytick.color": "#aaa", "axes.edgecolor": "#444"})
    fig, ax = plt.subplots(3, 1, figsize=(16, 9), sharex=True, gridspec_kw={"height_ratios": [2.2, 1, 1.6]})
    for L in lines:
        for a in ax[:2]:
            a.axvspan(L["t0"], L["t1"], color="#2a5d8a", alpha=0.35, lw=0)
        ax[0].text(L["t0"], 1.0, L["id"], fontsize=6, color="#8cf", va="top")
    side = base + ".json"
    gains = json.load(open(side))["stem_gain_db"] if os.path.exists(side) else {}
    cols = {"vo": "#9fd3ff", "sfx": "#ffb347", "music": "#b48cff", "mix_v1": "#e8e8e8"}
    for k, p in f.items():
        if os.path.exists(p):
            g = gains.get(k, 0.0)
            t, e = env_db(p, gain_db=g)
            lab = "mix (final)" if k == "mix_v1" else "%s stem at mix gain %+.1f dB, before ducking" % (k, g)
            ax[0].plot(t, e, lw=0.6 if k != "mix_v1" else 0.8, color=cols[k], label=lab, alpha=0.9)
    ax[0].set_ylim(-70, 2)
    ax[0].set_ylabel("RMS 20 ms (dBFS)")
    ax[0].legend(loc="lower right", fontsize=7, facecolor="#111317", edgecolor="#444")
    t, s = short_term(f["mix_v1"])
    ax[1].plot(t, s, color="#e8e8e8", lw=0.8)
    ax[1].axhline(-16, color="#6c6", lw=0.6, ls="--")
    ax[1].set_ylabel("mix short-term (LUFS)")
    ax[1].set_ylim(-40, -8)
    a, sr = sf.read(f["mix_v1"], always_2d=True)
    ax[2].specgram(a.mean(1), NFFT=2048, Fs=sr, noverlap=1536, cmap="magma", vmin=-130, vmax=-30)
    ax[2].set_yscale("symlog", linthresh=200)
    ax[2].set_ylim(30, 20000)
    ax[2].set_ylabel("mix (Hz)")
    ax[2].set_xlabel("film time (s)")
    for a_ in ax:
        for x in sorted({b for bb in SCENE_BOUNDS for b in bb}):
            a_.axvline(x, color="#888", lw=0.5)
        a_.set_xlim(0, TOTAL)
    fig.tight_layout()
    out = base + "_check.png"
    fig.savefig(out, dpi=110)
    print("wrote", rel(out))


if __name__ == "__main__":
    main()
