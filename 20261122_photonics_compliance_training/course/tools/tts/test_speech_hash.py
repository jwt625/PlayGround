"""JS / Python hash parity: engine/speech_hash.js vs tools/tts/speech_hash.py.

Run (from course/): uv run --no-project python tools/tts/test_speech_hash.py   (or: uv run --no-project --with pytest pytest tools/tts)
Strings: 200 real narration strings (extract.js, evenly sampled), every non-ASCII text field found in the course
objects, and hand-written edge cases (Unicode whitespace, astral characters, lone surrogate, empty).
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COURSE = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from speech_hash import lms_hash, norm_text  # noqa: E402

NODE_HASH = """
const h = require(process.argv[1]);
const xs = JSON.parse(require('fs').readFileSync(0, 'utf8'));
process.stdout.write(JSON.stringify(xs.map((t) => [h.LMS_normText(t), h.LMS_hash(t)])));
"""
NODE_NONASCII = """
global.window = {};
const fs = require('fs'), path = require('path');
const dir = path.join(process.argv[1], 'courses');
fs.readdirSync(dir).filter((f) => f.endsWith('.js')).forEach((f) => require(path.join(dir, f)));
const out = new Set();
(function walk(o) { if (typeof o === 'string') { if (/[^\\x00-\\x7f]/.test(o)) out.add(o); }
  else if (Array.isArray(o)) o.forEach(walk); else if (o && typeof o === 'object') Object.values(o).forEach(walk); })(window.LMS_COURSES);
process.stdout.write(JSON.stringify([...out]));
"""
EDGE = [
    "", " ", "That's right!", "  leading and trailing  ", "tab\tnewline\nCR\r\nform\x0cfeed\x0bvtab",
    "nbsp here", "thin space and en space", "ideographic　space", "bom﻿inside", "line sep para",
    "x-cut LiNbO₃, r₃₃ ≈ 30.8 pm/V", "µm, Ω, Å, 90°, λ/4", "Fabry–Pérot — café",
    "math chi \U0001d712 and \U0001f600 astral", "lone \ud800 surrogate", "中文 铌酸锂",
    "file sep\x1c and next line\x85 (not JS whitespace)",
]


def node(script, arg, stdin=None):
    r = subprocess.run(["node", "-e", script, arg], input=stdin, capture_output=True, check=True,
                       encoding="utf-8", errors="surrogatepass")
    return json.loads(r.stdout)


def corpus():
    ext = subprocess.run(["node", os.path.join(HERE, "extract.js")], capture_output=True, check=True, text=True)
    lines = [x["text"] for x in json.loads(ext.stdout)["voice_lines"]]
    step = max(1, len(lines) // 200)
    real = lines[::step][:200]
    nonascii = node(NODE_NONASCII, COURSE)
    return real, nonascii


def test_parity():
    real, nonascii = corpus()
    # real strings with typographic non-ASCII spliced in (the course text is ASCII today; authors may add some)
    sub = [("-", "\u2013"), (" degrees", "\u00b0"), ("'", "\u2019"), (" ", "\u00a0"), ("e", "\u00e9")]
    spliced = [t.replace(*sub[i % len(sub)]) + " \u03bb\U0001d712" for i, t in enumerate(real[:50])]
    xs = real + spliced + nonascii + EDGE
    js = node(NODE_HASH, os.path.join(COURSE, "engine", "speech_hash.js"),
              json.dumps(xs, ensure_ascii=True))
    bad = []
    for t, (jn, jh) in zip(xs, js):
        pn, ph = norm_text(t), lms_hash(t)
        if jh != ph or (jn != pn and "\ud800" not in t):
            bad.append((t, jn, pn, jh, ph))
    print("strings: %d real narration, %d real with non-ASCII spliced, %d non-ASCII course strings, %d edge cases;"
          " mismatches: %d" % (len(real), len(spliced), len(nonascii), len(EDGE), len(bad)))
    for b in bad[:10]:
        print("MISMATCH", repr(b))
    assert len(real) == 200 and not bad
    assert lms_hash("a") == "e40c292c70772d5a"   # FNV-1a 32 of "a" = 0xe40c292c (published test vector)


if __name__ == "__main__":
    test_parity()
    print("OK")
