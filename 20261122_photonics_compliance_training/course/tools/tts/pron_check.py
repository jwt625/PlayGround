"""Pronunciation round trip: ~60 technical sentences (real narration lines with the tricky terms + written-form
sentences such as "r33 of 30.8 pm/V at 1550 nm"), synthesized with and without the pronunciation map in one voice,
transcribed by Whisper small.en (asr_whisper.py), scored by WER and key-term regexes.

Run (from course/):
  uv run --no-project --with kokoro-onnx --with pyloudnorm --with soundfile python tools/tts/pron_check.py \
    [--voice af_heart] [--n 60] [--no-raw] [--tmp DIR]
Log: tools/tts/logs/pron_check_<YYYYmmdd_HHMMSS>.jsonl (one row per sentence and variant) + summary on stdout.
Scratch wavs go to --tmp and are deleted at the end.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pronounce  # noqa: E402
import tts_common as T  # noqa: E402

WHISPER_PY = os.environ.get("WHISPER_PY", "/opt/homebrew/Cellar/openai-whisper/20250625_5/libexec/bin/python")

# label, regex on the original text (activates the check), regex on Whisper's normalized transcript
KEYS = [
    ("niobate", r"niobate", r"\bniobate\b"),
    ("tantalate", r"tantalate", r"\btantalate\b"),
    ("x-cut", r"\bx-cut", r"\b(x|ex) ?cut\b"),
    ("z-cut", r"\bz-cut", r"\b(z|zee|zed) ?cut\b"),
    ("V pi", r"\bV pi\b", r"v ?(pi|pie|pai)\b|vpi\b"),
    ("TE", r"\bT ?E\b", r"\b(te|t e|tee)\b"),
    ("TM", r"\bT ?M\b", r"\b(tm|t m)\b"),
    ("DC", r"\bD ?C\b", r"\b(dc|d c)\b"),
    ("RF", r"\bR ?F\b", r"\b(rf|r f|rfv)"),
    ("r33", r"\br ?(33|three three)\b", r"\br ?(33|3 3|three three)\b"),
    ("pyroelectric", r"pyroelectric", r"\bpyro ?electric\b"),
    ("photorefractive", r"photorefracti", r"\bphoto ?refracti"),
    ("Fabry-Perot", r"Fabry", r"\bfabr[iy] ?p[ea]rr?ot\b"),
    ("isolator", r"isolator", r"\bisolators?\b"),
    ("FSR", r"\bFSR\b|F S R", r"\b(fsr|f s r)\b"),
    ("dB", r"\bdB\b|\bd B\b", r"\bdecibels?\b|\bdb\b"),
    ("nm", r"\bnm\b|nanometer", r"\bnanometers?\b|\bnm\b"),
    ("pm/V", r"pm/V|picometers per volt", r"\bpicometers? per volt\b"),
    ("Euler", r"Euler", r"\beuler\b"),
    ("kHz", r"kHz|kilohertz", r"\bkilohertz\b|\bkhz\b"),
    ("electro-optic", r"electro-optic", r"\belectro ?optic"),
    ("uniaxial", r"uniaxial", r"\buniaxial\b"),
    ("birefringence", r"birefringen", r"\bbirefringen"),
    ("quadrature", r"quadrature", r"\bquadrature\b"),
    ("Lorentzian", r"Lorentzian", r"\blorentzian\b"),
    ("10^N", r"\de\d", r"\b10 to the\b|\bten to the\b"),
    ("square root", r"sqrt", r"\bsquare root\b"),
]

WRITTEN = [
    "At 633 nm, Weis and Gaylord list r33 of about 30.8 pm/V clamped.",
    "On x-cut lithium niobate at 1550 nm, TE sees the extraordinary index.",
    "A Fabry-Perot ripple with an FSR of 0.2 nm points to a 2 mm cavity.",
    "Gary measured V pi with a 1 kHz triangle and quoted it as the RF V pi.",
    "The isolator gave 30 dB of isolation, and half a dB of insertion loss.",
    "A quarter model gives Q = 1e13; the full model gives Q around 1e3.",
    "Ripple depth grows with sqrt(R1 R2), even for weak mirrors.",
    "Z-cut wafers build a pyroelectric field during every DC bake.",
    "The Euler angles were (pi/4, pi/2, pi/2) in the z-x-z convention, per Voigt notation.",
    "Xu et al. (Optics Express, 2021) removed the oxide cladding from the TFLN rings.",
    "The photorefractive blue shift competes with the thermal red shift in lithium tantalate.",
    "InP lasers on a III-V platform can see 20 mW of back-reflection.",
]


def pick(lines, n):
    """Greedy: prefer lines that cover a not-yet-covered key; then lines with the most keys; 10-40 words."""
    keyed = []
    for t in lines:
        w = len(t.split())
        ks = {k for k, a, _ in KEYS if re.search(a, t)}
        if ks and 6 <= w <= 45:
            keyed.append((t, ks))
    out, cover = [], {}
    while len(out) < n and keyed:
        keyed.sort(key=lambda x: (-sum(1 for k in x[1] if cover.get(k, 0) < 3), -len(x[1]), x[0]))
        t, ks = keyed.pop(0)
        out.append(t)
        for k in ks:
            cover[k] = cover.get(k, 0) + 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--no-raw", action="store_true", help="skip the raw-text (no map) variant")
    ap.add_argument("--only", default="", help="regex: only sentences matching")
    ap.add_argument("--tmp", default=os.environ.get("TTS_TMP", "/tmp/tts_tmp"))
    a = ap.parse_args()
    import soundfile as sf
    from kokoro_onnx.tokenizer import Tokenizer
    os.makedirs(a.tmp, exist_ok=True)
    ext = subprocess.run(["node", os.path.join(HERE, "extract.js")], capture_output=True, check=True, text=True)
    lines = [x["text"] for x in json.loads(ext.stdout)["voice_lines"]]
    sents = pick(lines, a.n - len(WRITTEN)) + WRITTEN
    if a.only:
        sents = [s for s in sents if re.search(a.only, s)]
    k, tok = T.load_kokoro(4), Tokenizer()
    variants = ["map"] + ([] if a.no_raw else ["raw"])
    asr_in = []
    for i, t in enumerate(sents):
        for var in variants:
            x, info, sayinfo = T.render(k, tok, t, a.voice, a.speed, use_map=(var == "map"))
            w = os.path.join(a.tmp, "pc_%03d_%s.wav" % (i, var))
            sf.write(w, np.clip(x, -1, 1).astype(np.float32), T.KSR)
            asr_in.append({"id": "%d|%s" % (i, var), "wav": w, "ref": t, "say": sayinfo["say"]})
    jin, jout = os.path.join(a.tmp, "pc_in.json"), os.path.join(a.tmp, "pc_out.json")
    json.dump(asr_in, open(jin, "w"))
    subprocess.run([WHISPER_PY, os.path.join(HERE, "asr_whisper.py"), jin, jout], check=True)
    res = {r["id"]: r for r in json.load(open(jout))}
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    logp = os.path.join(HERE, "logs", "pron_check_%s.jsonl" % time.strftime("%Y%m%d_%H%M%S"))
    summ = {v: {"errs": 0, "n": 0, "errs_say": 0, "n_say": 0, "key_hit": 0, "key_n": 0, "key_miss": []} for v in variants}
    with open(logp, "w") as f:
        for it in asr_in:
            r = res[it["id"]]
            i, var = it["id"].split("|")
            keys = []
            for lab, act, hyp in KEYS:
                if re.search(act, it["ref"]):
                    ok = bool(re.search(hyp, r["hyp_n"]))
                    keys.append([lab, ok])
                    summ[var]["key_n"] += 1
                    summ[var]["key_hit"] += ok
                    if not ok:
                        summ[var]["key_miss"].append(lab)
            summ[var]["errs"] += r["errs"]
            summ[var]["n"] += r["n_ref"]
            summ[var]["errs_say"] += r.get("errs_say", r["errs"])
            summ[var]["n_say"] += r.get("n_say", r["n_ref"])
            row = {"i": int(i), "variant": var, "voice": a.voice, "ref": it["ref"], "say": it["say"], "hyp": r["hyp"],
                   "wer": r["wer"], "keys": keys}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if var == "map" and (r["errs"] or not all(ok for _, ok in keys)):
                print("[%d] WER %.2f %s\n   ref: %s\n   say: %s\n   hyp: %s" % (int(i), r["wer"], [l for l, ok in keys if not ok],
                                                                         it["ref"], it["say"], r["hyp"]))
    for it in asr_in:
        os.remove(it["wav"])
    os.remove(jin)
    os.remove(jout)
    for v, s in summ.items():
        print("%s: %d sentences, WER vs original %.3f (%d/%d words), WER vs spoken text %.3f, key terms %d/%d heard; misses: %s" % (
            v, len(sents), s["errs"] / max(1, s["n"]), s["errs"], s["n"], s["errs_say"] / max(1, s["n_say"]), s["key_hit"], s["key_n"],
            sorted(set(s["key_miss"]))))
    print("log", os.path.relpath(logp, os.path.join(HERE, "..", "..")))


if __name__ == "__main__":
    main()
