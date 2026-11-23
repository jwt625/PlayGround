"""Voice A/B samples: the same passage (host paragraph + wrong-answer explanation + "That's right!") in several Kokoro
voices and speeds, processed exactly like the course clips, plus measured metrics and a static A/B page.

Run (from course/):
  uv run --no-project --with kokoro-onnx --with pyloudnorm --with librosa --with soundfile \
    python tools/tts/voice_samples.py [--voices af_heart,am_michael] [--speeds 1.0] [--tmp DIR]
Writes ../research/voice_samples/<voice>_<speed>.mp3, metrics.json (merged across runs), index.html.
Metrics: speech seconds and words per minute (clips back to back, without the 0.35 s inter-clip gaps), F0 median and
F0 spread (pyin; std and 5-95 percentile range in semitones over voiced frames), Whisper small.en WER on the
per-clip round trip (EnglishTextNormalizer). Nobody listened; these are proxies.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tts_common as T  # noqa: E402

COURSE = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.abspath(os.path.join(COURSE, "..", "research", "voice_samples"))
WHISPER_PY = os.environ.get("WHISPER_PY", "/opt/homebrew/Cellar/openai-whisper/20250625_5/libexec/bin/python")
GAP = 0.35
GRADE = {"af_heart": "A", "af_bella": "A-", "af_nicole": "B-", "am_michael": "C+", "am_fenrir": "C+", "bf_emma": "B-",
         "bm_george": "C"}   # Kokoro-82M VOICES.md overall grade, fetched 2026-11-22
DEFAULT = ("af_heart", 1.0)   # chosen 2026-11-22; reasons in DevLog-003 voice section
PASSAGE = [
    "Hi. I'm Pat Reynolds, Director of the Office of Photonic Conduct here at Intracavity Photonics.",
    "Lithium niobate is a wonderful colleague. It is electro-optic, piezoelectric, pyroelectric, photorefractive, and "
    "anisotropic. It brings its whole self to work.",
    "Not quite. Let's take a moment to understand why.",
    "At zero and ninety degrees, many common rotation bugs cancel.",
    "The bug appears at a general angle like thirty or forty five degrees.",
    "So the passing tests did not test the thing that is most likely to be wrong.",
    "Please try again.",
    "That's right!",
]


def f0_stats(a):
    import librosa
    f0, vflag, _ = librosa.pyin(a.astype(np.float32), fmin=60, fmax=400, sr=T.KSR, frame_length=1024, hop_length=240)
    f = f0[vflag & np.isfinite(f0)]
    st = 12 * np.log2(f / np.median(f))
    return {"f0_median_hz": round(float(np.median(f)), 1), "f0_std_st": round(float(np.std(st)), 2),
            "f0_p5_p95_st": round(float(np.percentile(st, 95) - np.percentile(st, 5)), 2),
            "voiced_frac": round(float(len(f) / len(f0)), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", default="af_heart,af_bella,af_nicole,am_michael,am_fenrir,bf_emma,bm_george")
    ap.add_argument("--speeds", default="1.0")
    ap.add_argument("--tmp", default=os.environ.get("TTS_TMP", "/tmp/tts_tmp"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--index-only", action="store_true", help="rewrite index.html from metrics.json")
    a = ap.parse_args()
    if a.index_only:
        write_index(json.load(open(os.path.join(OUT, "metrics.json"))))
        return
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(a.tmp, exist_ok=True)
    import soundfile as sf
    from kokoro_onnx.tokenizer import Tokenizer
    k = T.load_kokoro(a.threads)
    tok = Tokenizer()
    mpath = os.path.join(OUT, "metrics.json")
    metrics = json.load(open(mpath)) if os.path.exists(mpath) else {}
    asr_in, rows = [], {}
    for v in a.voices.split(","):
        for sp in map(float, a.speeds.split(",")):
            key = "%s_%.2f" % (v, sp)
            clips, synth_s = [], 0.0
            for i, t in enumerate(PASSAGE):
                t0 = time.time()
                x, info, _ = T.render(k, tok, t, v, sp)
                synth_s += time.time() - t0
                clips.append(x)
                w = os.path.join(a.tmp, "%s_%d.wav" % (key, i))
                sf.write(w, np.clip(x, -1, 1).astype(np.float32), T.KSR)
                asr_in.append({"id": "%s|%d" % (key, i), "wav": w, "ref": t})
            speech = np.concatenate(clips)
            gap = np.zeros(int(GAP * T.KSR))
            full = np.concatenate([np.concatenate([c, gap]) for c in clips])[:-len(gap)]
            T.encode_mp3(full, os.path.join(OUT, key + ".mp3"), kbps=64)
            words = sum(len(t.split()) for t in PASSAGE)
            rows[key] = {"voice": v, "speed": sp, "speech_s": round(len(speech) / T.KSR, 2), "words": words,
                         "wpm": round(words / (len(speech) / T.KSR) * 60, 1), "rtf": round(synth_s / (len(speech) / T.KSR), 3),
                         "thats_right_s": round(len(clips[-1]) / T.KSR, 2), **f0_stats(speech)}
            print(key, rows[key], flush=True)
    jin, jout = os.path.join(a.tmp, "asr_in.json"), os.path.join(a.tmp, "asr_out.json")
    json.dump(asr_in, open(jin, "w"))
    subprocess.run([WHISPER_PY, os.path.join(HERE, "asr_whisper.py"), jin, jout, "--threads", str(a.threads)], check=True)
    res = json.load(open(jout))
    for key in rows:
        rr = [r for r in res if r["id"].split("|")[0] == key]
        rows[key]["wer"] = round(sum(r["errs"] for r in rr) / sum(r["n_ref"] for r in rr), 4)
        rows[key]["asr_misses"] = [{"ref": r["ref_n"], "hyp": r["hyp_n"]} for r in rr if r["errs"]]
        I, tp = T.measure_file(os.path.join(OUT, key + ".mp3"))
        rows[key]["mp3_lufs"], rows[key]["mp3_peak_dbfs"] = I, tp
    for it in asr_in:
        os.remove(it["wav"])
    os.remove(jin)
    os.remove(jout)
    metrics.update(rows)
    json.dump(metrics, open(mpath, "w"), indent=1)
    write_index(metrics)
    for key, r in rows.items():
        print("%-16s wpm %5.1f  F0 %5.1f Hz  std %.2f st  p5-95 %.2f st  WER %.3f  rtf %.3f  mp3 %s LUFS" % (
            key, r["wpm"], r["f0_median_hz"], r["f0_std_st"], r["f0_p5_p95_st"], r["wer"], r["rtf"], r["mp3_lufs"]))


def write_index(metrics):
    rows = sorted(metrics.values(), key=lambda r: (r["voice"], r["speed"]))
    tr = "\n".join(
        "<tr%s><td>%s</td><td>%.2f</td><td><audio controls preload=\"none\" src=\"%s_%.2f.mp3\"></audio></td>"
        "<td>%.1f</td><td>%.0f</td><td>%.2f</td><td>%.2f</td><td>%.1f</td><td>%.2f</td><td>%s</td></tr>" % (
            ' class="def"' if (r["voice"], r["speed"]) == DEFAULT else "",
            r["voice"] + (" (default)" if (r["voice"], r["speed"]) == DEFAULT else ""), r["speed"], r["voice"], r["speed"], r["wpm"], r["f0_median_hz"], r["f0_std_st"], r["f0_p5_p95_st"],
            100 * r["wer"], r["voiced_frac"], GRADE.get(r["voice"], "")) for r in rows)
    passage = "".join("<li>%s</li>" % t for t in PASSAGE)
    html = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Voice Samples</title>
<style>
:root{--bg:#fff;--fg:#1d2329;--muted:#5b6672;--line:#d5dbe1}
@media (prefers-color-scheme:dark){:root{--bg:#14181c;--fg:#dfe5ea;--muted:#8d99a5;--line:#2c343c}}
body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:0;padding:16px;max-width:1100px}
table{border-collapse:collapse;width:100%%}th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left}
th{color:var(--muted);font-weight:600}td:nth-child(n+4){font-variant-numeric:tabular-nums}audio{height:32px;max-width:300px}
ol{color:var(--muted)}tr.def td{font-weight:600}.n{color:var(--muted);font-size:12px}
</style></head><body>
<h2>Kokoro voice A/B: Pat Reynolds</h2>
<p class="n">Same passage, same processing as the course clips (trim, -18 LUFS, limiter -1.5 dBFS, MP3 64 kb/s mono 24 kHz), 0.35 s between clips.
wpm: words per minute of speech (gaps excluded). F0 spread: pyin, semitones over voiced frames. WER: Whisper small.en round trip, percent (the only miss common to all voices is the spelling "photo refractive").
voiced: fraction of pyin frames voiced (af_nicole is Kokoro's whispered voice). Kokoro grade: overall grade from the Kokoro-82M VOICES.md (hexgrad, Hugging Face).</p>
<table><tr><th>voice</th><th>speed</th><th>sample</th><th>wpm</th><th>F0 median Hz</th><th>F0 std st</th><th>F0 p5-p95 st</th><th>WER %%</th><th>voiced</th><th>Kokoro grade</th></tr>
%s
</table>
<p class="n">Passage:</p><ol>%s</ol>
</body></html>
""" % (tr, passage)
    open(os.path.join(OUT, "index.html"), "w").write(html)


if __name__ == "__main__":
    main()
