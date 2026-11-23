"""Whisper spot check of generated clips: N random clips (current lines only), transcribed from the shipped MP3s by
Whisper small.en (asr_whisper.py), WER vs the original text and vs the spoken (map-rewritten) text.

Run (from course/): uv run --no-project python tools/tts/spot_check.py [--n 40] [--seed 20261122]
Log: tools/tts/logs/spot_check_<YYYYmmdd_HHMMSS>.jsonl
"""
import argparse
import json
import os
import random
import subprocess
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
COURSE = os.path.abspath(os.path.join(HERE, "..", ".."))
WHISPER_PY = os.environ.get("WHISPER_PY", "/opt/homebrew/Cellar/openai-whisper/20250625_5/libexec/bin/python")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20261122)
    ap.add_argument("--tmp", default=os.environ.get("TTS_TMP", tempfile.gettempdir()))
    a = ap.parse_args()
    ext = json.loads(subprocess.run(["node", os.path.join(HERE, "extract.js")], capture_output=True, text=True, check=True).stdout)
    index = json.load(open(os.path.join(HERE, "clips_index.json")))
    cur = [x for x in ext["voice_lines"] if x["hash"] in index and os.path.exists(os.path.join(COURSE, "audio", x["hash"] + ".mp3"))]
    pick = random.Random(a.seed).sample(cur, min(a.n, len(cur)))
    items = [{"id": x["hash"], "wav": os.path.join(COURSE, "audio", x["hash"] + ".mp3"), "ref": x["text"], "say": index[x["hash"]]["say"]}
             for x in pick]
    os.makedirs(a.tmp, exist_ok=True)
    jin, jout = os.path.join(a.tmp, "spot_in.json"), os.path.join(a.tmp, "spot_out.json")
    json.dump(items, open(jin, "w"))
    subprocess.run([WHISPER_PY, os.path.join(HERE, "asr_whisper.py"), jin, jout], check=True)
    res = json.load(open(jout))
    os.remove(jin)
    os.remove(jout)
    logp = os.path.join(HERE, "logs", "spot_check_%s.jsonl" % time.strftime("%Y%m%d_%H%M%S"))
    E = N = ES = NS = 0
    with open(logp, "w") as f:
        for it, r in zip(items, res):
            E += r["errs"]; N += r["n_ref"]; ES += r["errs_say"]; NS += r["n_say"]
            f.write(json.dumps({"hash": it["id"], "ref": it["ref"], "say": it["say"], "hyp": r["hyp"], "wer": r["wer"],
                                "errs_say": r["errs_say"], "n_say": r["n_say"]}, ensure_ascii=False) + "\n")
            if r["errs_say"]:
                print("[%s] errs vs spoken %d/%d\n   say: %s\n   hyp: %s" % (it["id"], r["errs_say"], r["n_say"], it["say"], r["hyp"]))
    print("clips %d (of %d current): WER vs original %.3f (%d/%d), WER vs spoken %.3f (%d/%d); clips with any error vs spoken: %d"
          % (len(items), len(cur), E / max(N, 1), E, N, ES / max(NS, 1), ES, NS, sum(1 for r in res if r["errs_say"])))
    print("log", os.path.relpath(logp, COURSE))


if __name__ == "__main__":
    main()
