"""Crop the downloaded Commons bird photos to a consistent 4:3 card size and write references/birds/INDEX.md.

Usage (project root): uv run --with pillow python -I scripts/intro/crop_cards.py [sheet.png]
Inputs: references/birds/raw/<slug>.jpg + references/birds/fetch_log.jsonl (latest record per slug).
Outputs: assets/components/intro/photos/<slug>.jpg (800x600, sRGB JPEG q92); references/birds/INDEX.md;
optional contact sheet of the crops. Crops only (plus a mild auto-contrast where noted); no other edits.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "references" / "birds" / "raw"
OUT = ROOT / "assets" / "components" / "intro" / "photos"
LOG = ROOT / "references" / "birds" / "fetch_log.jsonl"
INDEX = ROOT / "references" / "birds" / "INDEX.md"
CW, CH = 800, 600

# slug: (centre x, centre y, crop width) as fractions of the source image width/height; crop is 4:3
# or ("fit", x0, y0, x1, y1): that region (fractions) fitted inside 4:3 and padded with the region's border colour
CROPS = {
    "theropod": (0.42, 0.52, 0.80),
    "archaeopteryx": (0.50, 0.40, 0.92),
    "moa": ("fit", 0.0, 0.02, 1.0, 1.0),
    "elephant_bird": ("fit", 0.0, 0.0, 1.0, 1.0),
    "emu": ("fit", 0.0, 0.0, 1.0, 1.0),
    "ostrich": (0.48, 0.52, 0.95),
    "chicken": (0.52, 0.50, 0.80),
    "hummingbird": (0.50, 0.52, 0.80),
    "albatross": (0.52, 0.52, 0.90),
    "parrot": (0.42, 0.55, 0.80),
    "cuckoo": (0.37, 0.40, 0.27),
    "passenger_pigeon": (0.42, 0.50, 0.70),
    "rock_pigeon": (0.50, 0.55, 0.90),
    "dodo": (0.68, 0.50, 0.80),
}
ORDER = ["theropod", "archaeopteryx", "moa", "elephant_bird", "emu", "ostrich", "chicken", "hummingbird",
         "albatross", "parrot", "cuckoo", "passenger_pigeon", "rock_pigeon", "dodo"]
SPECIES = {
    "theropod": "Feathered dromaeosaur (non-avian theropod), museum reconstruction",
    "archaeopteryx": "Archaeopteryx lithographica, Berlin specimen (fossil)",
    "moa": "South Island giant moa (Dinornis robustus), skeleton",
    "elephant_bird": "Elephant bird (Aepyornis), skeleton",
    "emu": "Emu (Dromaius novaehollandiae)",
    "ostrich": "Common ostrich (Struthio camelus), male",
    "chicken": "Chicken (Gallus gallus domesticus), rooster",
    "hummingbird": "Ruby-throated hummingbird (Archilochus colubris), male",
    "albatross": "Wandering albatross (Diomedea exulans)",
    "parrot": "Scarlet macaw (Ara macao)",
    "cuckoo": "Common cuckoo (Cuculus canorus)",
    "passenger_pigeon": "Passenger pigeon (Ectopistes migratorius), taxidermy specimens",
    "rock_pigeon": "Feral rock pigeon (Columba livia domestica)",
    "dodo": "Dodo (Raphus cucullatus), painting after Roelant Savery",
}


def crop(slug: str) -> Image.Image:
    im = Image.open(RAW / f"{slug}.jpg")
    im = ImageOps.exif_transpose(im).convert("RGB")
    w, h = im.size
    if CROPS[slug][0] == "fit":
        _, a, b, c, d = CROPS[slug]
        reg = im.crop((round(a * w), round(b * h), round(c * w), round(d * h)))
        rw, rh = reg.size
        border = list(reg.crop((0, 0, rw, 4)).get_flattened_data()) + list(reg.crop((0, 0, 4, rh)).get_flattened_data()) + \
            list(reg.crop((rw - 4, 0, rw, rh)).get_flattened_data())
        col = tuple(sorted(px[k] for px in border)[len(border) // 2] for k in range(3))  # per-channel median
        sc = min(CW / rw, CH / rh)
        reg = reg.resize((round(rw * sc), round(rh * sc)), Image.LANCZOS)
        canvas = Image.new("RGB", (CW, CH), col)
        canvas.paste(reg, ((CW - reg.size[0]) // 2, CH - reg.size[1]))  # sit on the bottom edge
        return canvas
    cx, cy, fw = CROPS[slug]
    cw = fw * w
    ch = cw * 3 / 4
    if ch > h:  # portrait sources: limit by height
        ch = h * min(1.0, fw)
        cw = ch * 4 / 3
    x0 = min(max(cx * w - cw / 2, 0), w - cw)
    y0 = min(max(cy * h - ch / 2, 0), h - ch)
    return im.crop((round(x0), round(y0), round(x0 + cw), round(y0 + ch))).resize((CW, CH), Image.LANCZOS)


def records() -> dict:
    recs = {}
    for line in LOG.read_text().splitlines():
        r = json.loads(line)
        recs[r["slug"]] = r  # latest wins
    return recs


def clean_author(a: str) -> str:
    a = re.sub(r"\(?[\w.+-]+@[\w-]+\.[\w.]+\)?", "", a)  # drop e-mail addresses from credits
    a = re.sub(r"\(https?://[^)]*\)?", "", a)
    return re.sub(r"\s+", " ", a).strip(" ,;")


def write_index(recs: dict) -> None:
    rows = []
    for s in ORDER:
        r = recs[s]
        raw = RAW / f"{s}.jpg"
        crop_p = OUT / f"{s}.jpg"
        rows.append(
            f"| {s} | {SPECIES[s]} | [{r['title'][5:]}]({r['file_page']}) | {clean_author(r['author'])} | {r['licence']} | "
            f"{r['width']}x{r['height']} | `raw/{s}.jpg` ({Image.open(raw).size[0]}x{Image.open(raw).size[1]}) | "
            f"`{hashlib.sha256(raw.read_bytes()).hexdigest()}` | "
            f"`{hashlib.sha256(crop_p.read_bytes()).hexdigest()[:16]}...` | {r['date_fetched']} |"
        )
    INDEX.write_text(
        "# Bird photos for the tree intro (F8)\n\n"
        "Source: Wikimedia Commons, via the MediaWiki API (`scripts/intro/fetch_bird_photos.py`; one request at a time, "
        ">= 6 s apart after one HTTP 429 at 2.5 s spacing on 2026-10-10). Stored file = the Commons thumbnail at the listed "
        "width (or the original). Licence and author from the file page's extmetadata (LicenseShortName, Artist); e-mail "
        "addresses removed from author strings. No AI-generated images.\n\n"
        "Card crops: `assets/components/intro/photos/<slug>.jpg`, 800x600 (4:3), crop + resize only; moa, elephant bird "
        "and emu are fitted whole and padded with their own median border colour (`scripts/intro/crop_cards.py`). Derivatives of CC BY-SA files are shared under the same licence; credit each "
        "author on the S7 credits card as \"<author> / Wikimedia Commons, <licence>\".\n\n"
        "| Slug | Subject | Commons file page | Author | Licence | Original px | Stored (raw) | SHA-256 (raw) | SHA-256 (crop, prefix) | Fetched |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n\n"
        "Hero card image: render of the project's hero GLB (`assets/hero/pigeon_osfp_dr8.glb`), not a photo.\n"
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    crops = {}
    for s in ORDER:
        crops[s] = crop(s)
        crops[s].save(OUT / f"{s}.jpg", quality=92)
    write_index(records())
    if len(sys.argv) > 1:
        sheet = Image.new("RGB", (5 * 410, 3 * 330), "white")
        d = ImageDraw.Draw(sheet)
        for i, s in enumerate(ORDER):
            t = crops[s].resize((400, 300))
            x, y = (i % 5) * 410, (i // 5) * 330
            sheet.paste(t, (x + 5, y + 25))
            d.text((x + 6, y + 6), s, fill="black")
        sheet.save(sys.argv[1])


if __name__ == "__main__":
    main()
