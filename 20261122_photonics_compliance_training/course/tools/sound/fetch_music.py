# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Download the source music tracks (Kevin MacLeod, incompetech.com, CC BY 4.0) into a local cache.

Run from the project root:  uv run course/tools/sound/fetch_music.py
Writes course/tools/sound/.cache/<file>.mp3 (git-ignored; only the edited course files are shipped).
Polite: skips files already present, waits 2 s between requests, generic User-Agent, stops on HTTP 429 / 403.
License verified 2026-11-22 on https://incompetech.com/music/royalty-free/licenses/ ("Creative Commons: By
Attribution 4.0"; required credit block: "Title" Kevin MacLeod (incompetech.com) / Licensed under Creative
Commons: By Attribution 4.0 / http://creativecommons.org/licenses/by/4.0/). Track facts from the site catalog
https://incompetech.com/music/royalty-free/pieces.json (same date).
"""
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache"
BASE = "https://incompetech.com/music/royalty-free/mp3-royaltyfree/"
UA = "Mozilla/5.0 (compatible; course-builder/1.0)"

# title -> (catalog filename, ISRC); catalog facts: BPM, instruments, length
TRACKS = {
    "Life of Riley": ("Life of Riley.mp3", "USUAN1400054"),   # 102 BPM; glockenspiel, ukulele, guitar, percussion; 3:55
    "Carefree": ("Carefree.mp3", "USUAN1400037"),             # 96 BPM; ukulele, guitar, marimba, glockenspiel; 3:25
    "Cheery Monday": ("Cheery Monday.mp3", "USUAN1700065"),   # 102 BPM; bass, piano, percussion, glockenspiel; 1:20
    "Easy Lemon (30 second)": ("Easy Lemon 30 second.mp3", "USUAN1200078"),  # 82 BPM; 0:31 edit with ending
}


def main() -> int:
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / ".gitignore").write_text("*\n")
    first = True
    for title, (fname, _isrc) in TRACKS.items():
        out = CACHE / fname
        if out.exists() and out.stat().st_size > 10000:
            print(f"skip  {fname}")
            continue
        if not first:
            time.sleep(2.0)
        first = False
        url = BASE + urllib.parse.quote(fname)
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
        except urllib.error.HTTPError as e:
            print(f"HTTP {e.code} for {url}; stopping")
            return 1
        out.write_bytes(data)
        print(f"got   {fname}  {len(data)} B")
    return 0


if __name__ == "__main__":
    sys.exit(main())
