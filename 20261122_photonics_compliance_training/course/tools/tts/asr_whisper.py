"""Whisper round trip (openai-whisper, CPU, bounded threads). Runs under an interpreter that has openai-whisper,
e.g. the Homebrew one: /opt/homebrew/Cellar/openai-whisper/<ver>/libexec/bin/python (see WHISPER_PY in pron_check.py).

  python asr_whisper.py in.json out.json [--model small.en] [--threads 4]
in.json: [{"id", "wav", "ref", optional "say"}]; out.json: [{"id", "hyp", "ref_n", "hyp_n", "wer", "n_ref"}].
WER on Whisper's EnglishTextNormalizer output (numbers to digits, punctuation dropped), word-level Levenshtein.
"""
import argparse
import json


def wer(r, h):
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--model", default="small.en")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    import torch
    import whisper
    from whisper.normalizers import EnglishTextNormalizer
    torch.set_num_threads(a.threads)
    m = whisper.load_model(a.model, device="cpu")
    norm = EnglishTextNormalizer()
    out = []
    for it in json.load(open(a.inp)):
        r = m.transcribe(it["wav"], language="en", temperature=0.0, fp16=False, condition_on_previous_text=False)
        hyp = r["text"].strip()
        rn, hn = norm(it["ref"]).split(), norm(hyp).split()
        e = wer(rn, hn)
        extra = {}
        if it.get("say"):   # WER against the spoken (map-rewritten) text: discounts symbol renderings like R1 vs "R one"
            sn = norm(it["say"]).split()
            extra = {"errs_say": wer(sn, hn), "n_say": len(sn)}
        out.append({**extra, "id": it["id"], "hyp": hyp, "ref_n": " ".join(rn), "hyp_n": " ".join(hn), "errs": e, "n_ref": len(rn),
                    "wer": round(e / max(1, len(rn)), 4)})
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
