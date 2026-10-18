"""Map ECTC Xplore records to program session number/title/committee using the ectc.net final program PDFs.

Block k = text between the (k-1)-th and k-th "Session Co-Chairs" marker region; blocks are in session order.
Titles are located by normalized-prefix search in the program text.
Usage: uv run python scripts/ectc_sessions.py
"""

from __future__ import annotations

import bisect
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xplore_common import ROOT  # noqa: E402

PROG = {2022: "72", 2023: "73", 2024: "74", 2025: "75", 2026: "76"}
TXT = ROOT / "cache" / "ectc_programs" / "txt"


def norm_with_map(s: str) -> tuple[str, list[int]]:
    out, idx = [], []
    for i, ch in enumerate(s):
        if ch.isalnum():
            out.append(ch.lower())
            idx.append(i)
    return "".join(out), idx


def clean(s: str | None) -> str | None:
    if s is None:
        return None
    s = s.replace("\ufffe", "").replace("\u00ad", "")
    s = re.split(r"\s+Time:", s)[0]
    return re.sub(r"\s+", " ", s.replace("Assembly and Manufacturing", "Assembly & Manufacturing")).strip()


def norm(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isalnum())


def session_titles(text: str) -> dict[int, str]:
    titles: dict[int, str] = {}
    lines = text.splitlines()
    last: int | None = None
    for ln in lines:
        parts = re.split(r"(Session (\d+): )", ln)
        if len(parts) > 1:
            # parts: [pre, full, num, body, full, num, body, ...]
            for j in range(1, len(parts), 3):
                n, body = int(parts[j + 1]), parts[j + 2].strip()
                if n not in titles:
                    titles[n] = body
                    last = n
                else:
                    last = None
        elif last is not None:
            s = ln.strip()
            if s and not re.match(r"^(Refreshment|Lunch|<<PAGE|PROGRAM|\d+$|Committee)", s) and len(s) < 70:
                titles[last] = f"{titles[last]} {s}".strip()
            last = None
    return titles


def main() -> None:
    out = ROOT / "cache" / "ectc_sessions.jsonl"
    rows = []
    for year, num in PROG.items():
        f = next(TXT.glob(f"{num}-*.txt"), None)
        xp = ROOT / "cache" / "xplore" / f"ECTC_{year}.jsonl"
        if f is None or not xp.exists():
            print(f"{year}: program or xplore missing, skipped")
            continue
        text = f.read_text()
        titles = session_titles(text)
        marks = [m.start() for m in re.finditer(r"Session Co-Chairs", text)]
        committees = []
        for m in marks:
            window = text[max(0, m - 400):m]
            c = window.rfind("Committee:")
            lines = [ln.strip() for ln in window[c + len("Committee:"):].split("\n")] if c >= 0 else []
            lines = [ln for ln in lines if ln]
            # committee name lines sit between "Committee:" and the room line just before "Session Co-Chairs"
            committees.append(" ".join(lines[:-1]) if len(lines) >= 2 else None)
        ntext, nmap = norm_with_map(text)
        recs = [json.loads(ln) for ln in xp.read_text().splitlines() if '"_done"' not in ln]
        hit = 0
        for r in recs:
            t = norm(r["articleTitle"] or "")
            pos = -1
            for a, b in ((0, 60), (0, 40), (0, 25), (15, 45), (30, 60)):
                if len(t) >= b:
                    pos = ntext.find(t[a:b])
                    if pos >= 0:
                        break
            row = {"articleNumber": r["articleNumber"], "year": year, "session_no": None,
                   "session_title": None, "committee": None}
            if pos >= 0:
                k = bisect.bisect_right(marks, nmap[pos])  # number of blocks started before title
                if 1 <= k <= len(marks):
                    st = titles.get(k)
                    cm = "Interactive Presentations" if st and "Interactive Presentations" in st else committees[k - 1]
                    row.update(session_no=k, session_title=clean(st), committee=clean(cm))
                    hit += 1
            rows.append(row)
        print(f"{year}: {hit}/{len(recs)} titles mapped to sessions; {len(marks)} blocks; {len(titles)} titles")
    with out.open("w") as fo:
        for row in rows:
            fo.write(json.dumps(row) + "\n")
    print(out)


if __name__ == "__main__":
    main()
