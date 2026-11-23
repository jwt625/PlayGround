"""Python twin of engine/speech_hash.js (COURSE_SPEC.md, "Recorded voice (Kokoro)"). Must stay byte-identical.

norm_text(t): JS String(t).replace(/\\s+/g, ' ').trim(), with the exact JS \\s set (Python's \\s differs:
it adds U+001C..U+001F and U+0085 and drops U+FEFF).
lms_hash(t): FNV-1a 32 (prime 0x01000193) over UTF-8 of norm_text(t), offsets 0x811c9dc5 and 0x050c5d1f, 8+8 hex.
Lone surrogates are encoded as U+FFFD (as JS TextEncoder does).
"""
import re

JS_WS = "\t\n\x0b\x0c\r    -     　﻿"
_WS_RE = re.compile("[%s]+" % JS_WS)
_LONE = re.compile("[\ud800-\udfff]")


def norm_text(t) -> str:
    return _WS_RE.sub(" ", str(t)).strip(" ")


def _fnv1a(b: bytes, h: int) -> str:
    for v in b:
        h = ((h ^ v) * 0x01000193) & 0xFFFFFFFF
    return "%08x" % h


def lms_hash(t) -> str:
    s = norm_text(t)
    # well-formed pairs never appear as separate code points in a Python str from json.loads, so any surrogate is lone
    b = _LONE.sub("�", s).encode("utf-8")
    return _fnv1a(b, 0x811C9DC5) + _fnv1a(b, 0x050C5D1F)
