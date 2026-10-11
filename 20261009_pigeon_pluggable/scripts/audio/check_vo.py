"""Whisper round trip for the voice-over: pronunciation of parody names / acronyms, and caption alignment.

Run (project root): uv run --no-project python scripts/audio/check_vo.py [--model small.en] [--whisper /opt/homebrew/bin/whisper]
Needs outputs/audio/vo_lines/<id>.wav and outputs/audio/vo_<date>.wav from make_vo.py.

1. Per line: transcribe each take, normalise, check the key-term patterns (KEYS) and the word match vs `show`.
2. Full track with word timestamps: every transcribed word must fall inside some narration line [t0 - 0.15, t1 + 0.15]
   (listen-proxy for "VO segments align to captions"); reports words outside and per-line first/last word times.
Whisper json outputs go to a scratch dir under outputs/audio/_whisper (removed at the end); the report goes to the log.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_util import HERE, OUT_AUDIO, TODAY, Log, rel  # noqa: E402

# key terms: (label, regex on the line's `show` text that activates the check, regex on the normalised transcript)
KEYS = [
    ("O-S-F-P", r"OSFP", r"\bo\s?s\s?f\s?p\b"),
    ("backronym", r"Overhead Surveillance", r"overhead surveillance feathered platform"),
    ("heatsink", r"[Hh]eatsink", r"\bheat\s?sink\b(?!\w)"),
    ("transmit(s)", r"transmits?\b", r"\btransmits?\b"),
    ("D-F-B", r"DFB", r"\bd\s?f\s?b\b"),
    ("Mach-Zehnder", r"Mach-Zehnder", r"\b(mach|mock)\s?ze(h)?nder\b"),
    ("T-I-A", r"\bTIA\b", r"\bt\s?i\s?a\b"),
    ("transimpedance", r"transimpedance", r"trans\s?impedance"),
    ("D-S-P", r"\bDSP\b", r"\bd\s?s\s?p\b"),
    ("Narrow-com", r"NARROWCOM", r"\bnarrow\s?com\b"),
    ("Mer-vall", r"MERVALL", r"\bmer\s?v[ao]l+\b"),
    ("Out-no-light", r"OUTNOLIGHT", r"\bout\s?no\s?light\b"),
    ("Oo-no-light", r"OUNOLIGHT", r"\b(oo|ooh|ou|u|you)\s?no\s?light\b|\buno\s?li(gh)?te?s?\b"),
    ("Dim-en-tum", r"DIMENTUM", r"\bdim\s?m?en\s?t[ue]m\b"),
    ("In-co-here-ent", r"INCOHERENT", r"\bin\s?co\s?here?\s?ent\b|\bincoherent\b"),
    ("Sem-trek", r"SEMTREK", r"\bsem\s?tre[kc]\b"),
    ("May-combo", r"MACOMBO", r"\b(may|ma|mac|mack)\s?combo\b"),
    ("heat sink", r"heat sink", r"\bheat\s?sink\b(?!\w)"),
    ("snitches", r"snitches", r"\bsnitches\b"),
    ("handoff", r"handoff", r"\bhand\s?off\b"),
    ("Again x2", r"Again", r"\bagain\b.*\bagain\b"),
    ("and again", r"And again", r"\band again\b"),
    ("Hi, Gary", r"Gary", r"\bhi\s?gary\b"),
    ("Parody", r"Parody", r"\bparody\b"),
]


NUM = {"2": "two", "8": "eight"}
COMPOUND = {"photo diodes": "photodiodes", "groundlands": "ground lands", "heat sink": "heatsink"}


def norm(s):
    s = s.lower().replace("-", " ").replace(".", " ")
    for a, b in NUM.items():
        s = re.sub(r"\b%s\b" % a, b, s)
    for a, b in COMPOUND.items():
        s = s.replace(a, b)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\b([a-z]) (?=[a-z]\b)", r"\1", s)   # join spelled letters: "o s f p" -> "osfp"
    return re.sub(r"\s+", " ", s).strip()


def norm_keys(s):
    s = s.lower().replace("-", " ").replace(".", " ")
    s = re.sub(r"(\d)([a-z])", r"\1 \2", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def word_match(ref, hyp):
    """fraction of reference words found in order (LCS / len(ref))."""
    r, h = norm(ref).split(), norm(hyp).split()
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r)):
        for j in range(len(h)):
            dp[i + 1][j + 1] = dp[i][j] + 1 if r[i] == h[j] else max(dp[i][j + 1], dp[i + 1][j])
    return dp[-1][-1] / max(1, len(r))


def whisper(files, model, outdir, exe, words=False):
    cmd = [exe, *files, "--model", model, "--language", "en", "--output_format", "json", "--output_dir", outdir,
           "--fp16", "False", "--verbose", "False", "--temperature", "0", "--condition_on_previous_text", "False"]
    if words:
        cmd += ["--word_timestamps", "True"]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small.en")
    ap.add_argument("--whisper", default="/opt/homebrew/bin/whisper")
    ap.add_argument("--vo", default=os.path.join(OUT_AUDIO, "vo_%s.wav" % TODAY))
    args = ap.parse_args()
    log = Log("check_vo_%s" % args.model.replace(".", ""))
    doc = json.load(open(os.path.join(HERE, "narration.json")))
    lines = doc["lines"]
    scratch = os.path.join(OUT_AUDIO, "_whisper")
    os.makedirs(scratch, exist_ok=True)
    files = [os.path.join(OUT_AUDIO, "vo_lines", "%s.wav" % L["id"]) for L in lines]
    whisper(files, args.model, scratch, args.whisper)
    log("Whisper %s, per line (match = fraction of caption words recovered in order):" % args.model)
    bad = 0
    for L in lines:
        hyp = json.load(open(os.path.join(scratch, "%s.json" % L["id"])))["text"].strip()
        m = word_match(L["show"], hyp)
        keys = []
        for lab, act, rx in KEYS:
            if not re.search(act, L["show"]):
                continue
            ok = re.search(rx, norm_keys(hyp)) is not None
            keys.append("%s:%s" % (lab, "OK" if ok else "MISS"))
            bad += 0 if ok else 1
        log("%-4s match %.2f  %-28s | %s" % (L["id"], m, " ".join(keys), hyp))
    log("key-term misses: %d" % bad)

    whisper([args.vo], args.model, scratch, args.whisper, words=True)
    js = json.load(open(os.path.join(scratch, os.path.splitext(os.path.basename(args.vo))[0] + ".json")))
    words = [w for seg in js["segments"] for w in seg.get("words", [])]
    out = []
    per = {L["id"]: [] for L in lines}
    for w in words:
        # Whisper word starts are often pulled back to the previous word end; assign by overlap with the caption window
        ov = [(min(w["end"], L["t1"] + 0.15) - max(w["start"], L["t0"] - 0.15), L) for L in lines]
        best = max(ov, key=lambda x: x[0])
        if best[0] > 0:
            per[best[1]["id"]].append(w)
        else:
            out.append(w)
    log("full track: %d words, %d outside every caption window" % (len(words), len(out)))
    for w in out:
        log("  outside: %.2f-%.2f %s" % (w["start"], w["end"], w["word"]))
    for L in lines:
        ws = per[L["id"]]
        if ws:
            log("%-4s caption %6.2f-%6.2f  heard %6.2f-%6.2f  (%d words)" % (L["id"], L["t0"], L["t1"], ws[0]["start"], ws[-1]["end"], len(ws)))
        else:
            log("%-4s caption %6.2f-%6.2f  heard NOTHING" % (L["id"], L["t0"], L["t1"]))
    log("full transcript:", js["text"].strip())
    shutil.rmtree(scratch)
    log("log", rel(log.path))


if __name__ == "__main__":
    main()
