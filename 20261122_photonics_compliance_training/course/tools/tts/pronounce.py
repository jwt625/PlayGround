"""Pronunciation map for Kokoro (espeak-ng en-us G2P). Applied at synthesis time only: the clip key stays
LMS_hash(original text); captions show the original text.

Two layers:
1. TEXT_RULES (regex -> text) rewrite symbols, units, math and acronyms into speakable words.
2. WORD_IPA (whole word -> IPA) replaces espeak's guess for words it gets wrong (niobate, author names). The text is
   split around those words, the rest is phonemized by espeak (kokoro_onnx Tokenizer), the IPA is spliced in, and the
   phoneme string goes to Kokoro with is_phonemes=True.
Checked with a Whisper round trip (tools/tts/pron_check.py); see DevLog-003 voice section for results.
"""
import re

DIG = "zero one two three four five six seven eight nine".split()

# acronyms that stay as words (everything else of 2-5 capitals is spelled letter by letter)
KEEP_CAPS = {"OK", "I", "A", "LASER", "NASA", "IEEE"}
SPELL_MIXED = {"InP": "I N P", "GaAs": "gallium arsenide", "SiN": "silicon nitride", "SiPh": "silicon photonics",
               "LiNbO3": "lithium niobate", "LiTaO3": "lithium tantalate", "TFLN": "thin film lithium niobate"}


def _spell(m):
    w = m.group(0)
    return w if w in KEEP_CAPS else " ".join(w)


def _sci(m):
    a, b = m.group(1), m.group(2)
    return ("ten to the %s" % b) if a == "1" else ("%s times ten to the %s" % (a, b))


TEXT_RULES = [
    # symbols
    (r"π", " pi "), (r"°", " degrees"), (r"Ω", " ohms"), (r"≈", " about "), (r"[–—]", ", "),
    (r"[µμ]m\b", " micrometers"), (r"[‘’]", "'"), (r"[“”]", '"'),
    # abbreviations
    (r"\bet al\.", "et al"), (r"\be\.g\.", "for example"), (r"\bi\.e\.", "that is"), (r"\bvs\.", "versus"),
    (r"\bFabry-Perot\b", "Fabry Perot"),
    # x-cut: glued "x-cut" is heard as "X-Gut" (s+k cluster loses the k aspiration); a space fixes it (Whisper check)
    (r"\b([xyzXYZ])-(cut|axis|propagating|polarized)\b", lambda m: m.group(1).upper() + " " + m.group(2)),
    (r"\b([xyz])-([xyz])-([xyz])\b", lambda m: " ".join(g.upper() for g in m.groups())),
    (r"\bpi/(\d)", r"pi over \1"),
    # document codes: "PHC-1550.4" -> "P H C fifteen fifty point four" (espeak reads 1550.4 as "one thousand five hundred")
    (r"\b([A-Z]{2,5})-(1[0-9])([0-9]{2})\.(\d)\b", lambda m: "%s %s %s point %s" % (" ".join(m.group(1)), m.group(2), m.group(3), m.group(4))),
    (r"\b([A-Z]{2,5})-(\d)", lambda m: " ".join(m.group(1)) + " " + m.group(2)), (r"\bIII-V\b", "three five"),
    # tensor components r33, d33, p41 -> "r three three"
    (r"\b([rdpse])(\d)(\d)\b", lambda m: "%s %s %s" % (m.group(1), DIG[int(m.group(2))], DIG[int(m.group(3))])),
    (r"\bV_?pi\b|\bVπ", "V pi"),
    # math
    (r"\b(\d+(?:\.\d+)?)e(\d+)\b", _sci),
    (r"\bsqrt\(([^)]*)\)", r"the square root of \1"),
    (r"\b([A-Z])(\d)\b", lambda m: "%s %s" % (m.group(1), DIG[int(m.group(2))])),
    (r"\s=\s", " equals "), (r"\s/\s", " over "),
    # units (after a number or standalone)
    (r"\b(1[0-9])([0-9]{2})\s?(?:nm|nanometers?)\b", r"\1 \2 nanometers"), (r"\bnm\b", "nanometers"),
    (r"\bpm/V\b", "picometers per volt"), (r"\bV/(?:µ|u)m\b", "volts per micrometer"),
    (r"\b([kMGT])Hz\b", lambda m: {"k": "kilo", "M": "mega", "G": "giga", "T": "tera"}[m.group(1)] + "hertz"),
    (r"\bdBm\b", "D B M"), (r"\b(a|one) dB\b", r"\1 decibel"), (r"\bdB\b", "decibels"),
    (r"(\d)\s?mW\b", r"\1 milliwatts"), (r"(\d)\s?K\b", r"\1 kelvin"),
    # acronyms
    (r"\b(%s)\b" % "|".join(SPELL_MIXED), lambda m: SPELL_MIXED[m.group(1)]),
    (r"\b[A-Z]{2,5}\b", _spell),
    (r"\s{2,}", " "),
]
TEXT_RULES = [(re.compile(p), r) for p, r in TEXT_RULES]

# word -> IPA (espeak/Kokoro symbol set). Case-sensitive whole words.
WORD_IPA = {
    "niobate": "naɪˈoʊbeɪt",       # espeak: nɪˈɑːbeɪt ("nee-AH-bate")
    "niobium": "naɪˈoʊbiəm",
    "Xu": "ʃˈuː",                  # espeak: zˈuː
    "Xie": "ʃjˈɛ",                 # espeak: zˈaɪ
    "Liu": "ljˈoʊ",                # espeak: lˈɪuː
    "Zhang": "dʒˈɑːŋ",             # espeak: ʒˈæŋ
    "Zheng": "dʒˈʌŋ",              # espeak: ʒˈɛŋ
    "Wang": "wˈɑːŋ",               # espeak: wˈæŋɡ
    "Li": "lˈiː",                  # espeak: lˈaɪ
    "Shi": "ʃˈɜː",
    "Liang": "ljˈɑːŋ",
    "Maeder": "mˈeɪdɚ",            # espeak: mˈiːdɚ
    "Gimzewski": "ɡɪmzˈɛfski",     # espeak: ɡɪmzjˈuːski
    "Naranjo": "nɐɹˈɑːnhoʊ",       # espeak: næɹˈændʒoʊ
    "Tkach": "tkˈɑːtʃ",            # espeak: tˈiːkˈætʃ
    "Voigt": "vˈɔɪt",              # espeak: vˈɔɪɡt
    "Koninck": "kˈoʊnɪŋk",
    "Fabry": "fɑːbɹˈiː",
    "Perot": "pɛɹˈoʊ",
}
_IPA_RE = re.compile(r"\b(%s)\b" % "|".join(map(re.escape, WORD_IPA)))


def say_text(text: str) -> str:
    s = text
    for p, r in TEXT_RULES:
        s = p.sub(r, s)
    return s.strip()


def to_phonemes(text: str, tok) -> str:
    """text (already say_text'ed) -> phoneme string with WORD_IPA spliced in. tok = kokoro_onnx Tokenizer."""
    parts, last = [], 0
    for m in _IPA_RE.finditer(text):
        seg = text[last:m.start()]
        if seg.strip():
            parts.append(tok.phonemize(seg, "en-us").strip())
        parts.append(WORD_IPA[m.group(1)])
        last = m.end()
    seg = text[last:]
    if seg.strip():
        ph = tok.phonemize(seg, "en-us").strip() if re.search(r"\w", seg) else seg.strip()
        parts.append(ph)
    s = " ".join(p for p in parts if p)
    return re.sub(r"\s+([,.;:!?)])", r"\1", s)


def needs_ipa(text: str) -> bool:
    return bool(_IPA_RE.search(text))


if __name__ == "__main__":
    import sys
    for t in sys.argv[1:]:
        print(say_text(t))
