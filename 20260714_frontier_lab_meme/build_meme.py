#!/usr/bin/env python3
"""Build the Bell Labs roster meme from the original 5x5 template.

The script deliberately uses ordinary image processing only:

1. Detect the white tile borders by projection over near-white pixels.
2. Fetch/cache curated Wikipedia/Wikimedia portrait images and source metadata.
3. Crop each portrait into the detected tile interior.
4. Replace only the portrait interiors; preserve all original text and borders.

Requires Pillow. It does not call any image-generation service.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "image.png"
DEFAULT_OUTPUT = ROOT / "bell_labs_meme.png"
DEFAULT_NAMED_OUTPUT = ROOT / "bell_labs_meme_named.png"
DEFAULT_HISTORIC_OUTPUT = ROOT / "bell_labs_meme_historic.png"
DEFAULT_HISTORIC_NAMED_OUTPUT = ROOT / "bell_labs_meme_historic_named.png"
CACHE_DIR = ROOT / "cached_pfps"
ATTRIBUTION_FILE = ROOT / "assets" / "portrait_sources.json"
CONTACT_SHEET = ROOT / "assets" / "portrait_contact_sheet.jpg"
HISTORIC_CACHE_DIR = ROOT / "cached_pfps_historic"
HISTORIC_ATTRIBUTION_FILE = ROOT / "assets" / "portrait_sources_historic.json"
HISTORIC_CONTACT_SHEET = ROOT / "assets" / "portrait_contact_sheet_historic.jpg"
USER_AGENT = "frontier-lab-meme/1.0 (local research-image compositor)"


# Matrix order is row-major: $5 row through $1 row.
# Wikipedia titles are explicit so identity comes from source metadata, not face matching.
ROSTER = [
    ("Oppenheimer", "J._Robert_Oppenheimer"),
    ("von Neumann", "John_von_Neumann"),
    ("Curie", "Marie_Curie"),
    ("Shannon", "Claude_Shannon"),
    ("Moore", "Gordon_Moore"),
    ("Bush", "Vannevar_Bush"),
    ("Turing", "Alan_Turing"),
    ("Rutherford", "Ernest_Rutherford"),
    ("Bardeen", "John_Bardeen"),
    ("Land", "Edwin_H._Land"),
    ("Kelly", "Mervin_Kelly"),
    ("Liskov", "Barbara_Liskov"),
    ("Doudna", "Jennifer_Doudna"),
    ("Mead", "Carver_Mead"),
    ("Su", "Lisa_Su"),
    ("Taylor", "Robert_Taylor_(computer_scientist)"),
    ("Lamport", "Leslie_Lamport"),
    ("Arnold", "Frances_Arnold"),
    ("Ritchie", "Dennis_Ritchie"),
    ("Church", "George_Church_(geneticist)"),
    ("Gilbreth", "Lillian_Moller_Gilbreth"),
    ("Perlman", "Radia_Perlman"),
    ("Dresselhaus", "Mildred_Dresselhaus"),
    ("Hamilton", "Margaret_Hamilton_(software_engineer)"),
    ("Karikó", "Katalin_Karikó"),
]


# All-deceased edition. The five columns retain the same roles as the mixed-era
# board: lab architect, theory/computation, experiment, devices/systems, scale.
HISTORIC_ROSTER = [
    ("Oppenheimer", "J._Robert_Oppenheimer"),
    ("von Neumann", "John_von_Neumann"),
    ("Curie", "Marie_Curie"),
    ("Shannon", "Claude_Shannon"),
    ("Moore", "Gordon_Moore"),
    ("Bush", "Vannevar_Bush"),
    ("Turing", "Alan_Turing"),
    ("Rutherford", "Ernest_Rutherford"),
    ("Bardeen", "John_Bardeen"),
    ("Land", "Edwin_H._Land"),
    ("Kelly", "Mervin_Kelly"),
    ("Hopper", "Grace_Hopper"),
    ("Franklin", "Rosalind_Franklin"),
    ("Kilby", "Jack_Kilby"),
    ("Noyce", "Robert_Noyce"),
    ("Taylor", "Robert_Taylor_(computer_scientist)"),
    ("Dijkstra", "Edsger_W._Dijkstra"),
    ("Elion", "Gertrude_B._Elion"),
    ("Ritchie", "Dennis_Ritchie"),
    ("Sanger", "Frederick_Sanger"),
    ("Gilbreth", "Lillian_Moller_Gilbreth"),
    ("Spärck Jones", "Karen_Spärck_Jones"),
    ("Dresselhaus", "Mildred_Dresselhaus"),
    ("Vaughan", "Dorothy_Vaughan"),
    ("Borlaug", "Norman_Borlaug"),
]


# Stable institutional sources used where Wikimedia's thumbnail service is
# especially rate-limited. The page URL is retained for attribution/review.
HISTORIC_SOURCE_OVERRIDES = {
    "Karen_Spärck_Jones": {
        "image_url": "https://www.cl.cam.ac.uk/misc/obituaries/sparck-jones/photos/CU%20KSJ-002.tif",
        "page_url": "https://www.cl.cam.ac.uk/misc/obituaries/sparck-jones/photos/",
    },
    "Dorothy_Vaughan": {
        "image_url": "https://www.nasa.gov/wp-content/uploads/2017/03/dorothy-vaughan-10people.jpg",
        "page_url": "https://www.nasa.gov/people/dorothy-vaughan/",
    },
    "Norman_Borlaug": {
        "image_url": "https://www.nobelprize.org/images/borlaug-13223-landscape-medium.jpg",
        "page_url": "https://www.nobelprize.org/prizes/peace/1970/borlaug/",
    },
}


# Per-portrait framing corrections. ImageOps.fit uses these as the focal point
# when cropping: a lower y keeps more of the source top (moving content down),
# while a higher x keeps more of the source right (moving content left).
CROP_CENTER_OVERRIDES = {
    "Kelly": (0.5, 0.12),
    "Noyce": (0.78, 0.38),
}


def request_bytes(url: str) -> bytes:
    for attempt in range(5):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == 4:
                raise
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"Failed to fetch {url}")


def fetch_page_metadata(roster: list[tuple[str, str]]) -> dict[str, dict]:
    """Resolve all roster pages and images in one MediaWiki API request."""
    parameters = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "prop": "pageimages|info",
        "inprop": "url",
        "piprop": "original|thumbnail",
        # 640 px is a standard Wikimedia thumbnail width and is ample for 214 px tiles.
        "pithumbsize": "640",
        "redirects": "1",
        "titles": "|".join(page_title for _, page_title in roster),
    }
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(parameters)
    payload = json.loads(request_bytes(url))
    query = payload["query"]
    aliases: dict[str, str] = {}
    for item in query.get("normalized", []) + query.get("redirects", []):
        aliases[item["from"]] = item["to"]
    pages = {page["title"]: page for page in query["pages"]}

    resolved: dict[str, dict] = {}
    for _, requested in roster:
        title = requested.replace("_", " ")
        visited = set()
        while title in aliases and title not in visited:
            visited.add(title)
            title = aliases[title]
        if title not in pages:
            raise RuntimeError(f"Could not resolve Wikipedia page {requested!r}; got {title!r}")
        resolved[requested] = pages[title]
    return resolved


def safe_stem(text: str) -> str:
    text = text.lower().replace("ó", "o")
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def fetch_portraits(
    roster: list[tuple[str, str]],
    cache_dir: Path,
    attribution_file: Path,
    source_overrides: dict[str, dict[str, str]] | None = None,
    force: bool = False,
) -> list[dict]:
    """Fetch Wikipedia page images and cache normalized JPEGs plus provenance."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    metadata = fetch_page_metadata(roster)
    source_overrides = source_overrides or {}

    for index, (label, page_title) in enumerate(roster, start=1):
        destination = cache_dir / f"{index:02d}_{safe_stem(label)}.jpg"
        page = metadata[page_title]
        override = source_overrides.get(page_title)
        image_info = page.get("thumbnail") or page.get("original")
        if not image_info or "source" not in image_info:
            raise RuntimeError(f"No page image returned for {page_title}")

        image_url = override["image_url"] if override else image_info["source"]
        if force or not destination.exists():
            try:
                raw = request_bytes(image_url)
            except urllib.error.HTTPError as error:
                original_url = page.get("original", {}).get("source")
                if error.code != 429 or not original_url or original_url == image_url:
                    raise
                image_url = original_url
                raw = request_bytes(image_url)
            with Image.open(io.BytesIO(raw)) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                # Retain enough resolution for recropping without caching huge originals.
                image.thumbnail((1400, 1400), Image.Resampling.LANCZOS)
                image.save(destination, "JPEG", quality=93, optimize=True)
            # Avoid tripping Wikimedia's image-resize/download rate limits.
            time.sleep(1.25)

        records.append(
            {
                "index": index,
                "label": label,
                "page_title": page["title"],
                "page_url": (
                    override["page_url"]
                    if override
                    else page.get("fullurl", f"https://en.wikipedia.org/wiki/{page_title}")
                ),
                "image_url": image_url,
                "cached_file": str(destination.relative_to(ROOT)),
            }
        )

    attribution_file.parent.mkdir(parents=True, exist_ok=True)
    attribution_file.write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return records


def is_near_white(pixel: tuple[int, int, int]) -> bool:
    return min(pixel) >= 235 and max(pixel) - min(pixel) <= 20


def contiguous_runs(values: Iterable[int]) -> list[tuple[int, int]]:
    runs: list[list[int]] = []
    for value in values:
        if not runs or value > runs[-1][-1] + 1:
            runs.append([value])
        else:
            runs[-1].append(value)
    return [(run[0], run[-1]) for run in runs]


def border_runs(image: Image.Image, axis: str) -> list[tuple[int, int]]:
    """Detect long white lines by 1D pixel projection.

    The lower/right 85% of the canvas is used so title lettering and dollar labels
    do not dominate. A candidate border coordinate must be near-white along at
    least 60% of the relevant projection span.
    """
    rgb = image.convert("RGB")
    width, height = rgb.size
    pixels = rgb.load()

    if axis == "x":
        outer_range = range(width)
        inner_range = range(int(height * 0.14), height)
        point = lambda outer, inner: (outer, inner)
    elif axis == "y":
        outer_range = range(height)
        inner_range = range(int(width * 0.13), width)
        point = lambda outer, inner: (inner, outer)
    else:
        raise ValueError("axis must be x or y")

    threshold = int(len(inner_range) * 0.60)
    candidates = []
    for outer in outer_range:
        count = sum(is_near_white(pixels[point(outer, inner)]) for inner in inner_range)
        if count >= threshold:
            candidates.append(outer)
    return contiguous_runs(candidates)


def detect_cells(image: Image.Image) -> list[tuple[int, int, int, int]]:
    """Return 25 interior rectangles inferred from ten x and ten y borders."""
    x_runs = border_runs(image, "x")
    y_runs = border_runs(image, "y")
    if len(x_runs) != 10 or len(y_runs) != 10:
        raise RuntimeError(
            "Expected ten border runs per axis; "
            f"detected x={x_runs!r}, y={y_runs!r}"
        )

    x_interiors = [(x_runs[i][1] + 1, x_runs[i + 1][0]) for i in range(0, 10, 2)]
    y_interiors = [(y_runs[i][1] + 1, y_runs[i + 1][0]) for i in range(0, 10, 2)]
    cells = [(left, top, right, bottom) for top, bottom in y_interiors for left, right in x_interiors]

    if len(cells) != 25 or any(r <= l or b <= t for l, t, r, b in cells):
        raise RuntimeError(f"Invalid detected cells: {cells!r}")
    return cells


def find_font(bold: bool = False) -> str:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    raise RuntimeError("Could not find a usable system font")


def make_tile(
    portrait: Image.Image,
    size: tuple[int, int],
    label: str | None = None,
    centering: tuple[float, float] = (0.5, 0.38),
) -> Image.Image:
    """Create a top-biased cover crop, optionally with a review label."""
    width, height = size
    portrait = ImageOps.exif_transpose(portrait).convert("RGB")
    tile = ImageOps.fit(
        portrait,
        size,
        method=Image.Resampling.LANCZOS,
        centering=centering,
    )
    if label:
        overlay = Image.new("RGBA", size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        strip_height = max(27, int(height * 0.15))
        overlay_draw.rectangle(
            (0, height - strip_height, width, height), fill=(0, 0, 0, 178)
        )
        tile = Image.alpha_composite(tile.convert("RGBA"), overlay).convert("RGB")

        draw = ImageDraw.Draw(tile)
        font_path = find_font(bold=True)
        font_size = 20
        while font_size > 10:
            font = ImageFont.truetype(font_path, font_size)
            box = draw.textbbox((0, 0), label.upper(), font=font)
            if box[2] - box[0] <= width - 12:
                break
            font_size -= 1
        text_width = box[2] - box[0]
        text_height = box[3] - box[1]
        x = (width - text_width) // 2
        y = height - strip_height + (strip_height - text_height) // 2 - box[1]
        draw.text((x, y), label.upper(), font=font, fill="white")
    return tile


def create_contact_sheet(records: list[dict], destination: Path) -> None:
    thumb_w, thumb_h = 220, 250
    sheet = Image.new("RGB", (thumb_w * 5, thumb_h * 5), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(find_font(bold=True), 18)
    for index, record in enumerate(records):
        with Image.open(ROOT / record["cached_file"]) as portrait:
            thumb = ImageOps.fit(
                ImageOps.exif_transpose(portrait).convert("RGB"),
                (thumb_w, thumb_h - 30),
                Image.Resampling.LANCZOS,
                centering=(0.5, 0.38),
            )
        x = (index % 5) * thumb_w
        y = (index // 5) * thumb_h
        sheet.paste(thumb, (x, y))
        label = f"{index + 1:02d} {record['label']}"
        draw.text((x + 6, y + thumb_h - 26), label, font=font, fill="black")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, "JPEG", quality=91, optimize=True)


def render_pair(
    original: Image.Image,
    cells: list[tuple[int, int, int, int]],
    records: list[dict],
    output_path: Path,
    named_output_path: Path,
) -> None:
    unlabeled = original.copy()
    named = original.copy()

    for cell, record in zip(cells, records):
        left, top, right, bottom = cell
        size = (right - left, bottom - top)
        centering = CROP_CENTER_OVERRIDES.get(record["label"], (0.5, 0.38))
        with Image.open(ROOT / record["cached_file"]) as portrait:
            tile = make_tile(portrait, size, centering=centering)
            named_tile = make_tile(
                portrait,
                size,
                label=record["label"],
                centering=centering,
            )
        unlabeled.paste(tile, (left, top))
        named.paste(named_tile, (left, top))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    named_output_path.parent.mkdir(parents=True, exist_ok=True)
    unlabeled.save(output_path, "PNG", optimize=True)
    named.save(named_output_path, "PNG", optimize=True)


def build(
    input_path: Path,
    output_path: Path,
    named_output_path: Path,
    historic_output_path: Path,
    historic_named_output_path: Path,
    force_fetch: bool = False,
) -> None:
    records = fetch_portraits(
        ROSTER, CACHE_DIR, ATTRIBUTION_FILE, force=force_fetch
    )
    historic_records = fetch_portraits(
        HISTORIC_ROSTER,
        HISTORIC_CACHE_DIR,
        HISTORIC_ATTRIBUTION_FILE,
        source_overrides=HISTORIC_SOURCE_OVERRIDES,
        force=force_fetch,
    )
    create_contact_sheet(records, CONTACT_SHEET)
    create_contact_sheet(historic_records, HISTORIC_CONTACT_SHEET)

    original = Image.open(input_path).convert("RGB")
    cells = detect_cells(original)
    render_pair(original, cells, records, output_path, named_output_path)
    render_pair(
        original,
        cells,
        historic_records,
        historic_output_path,
        historic_named_output_path,
    )

    print(f"Detected cells: {cells}")
    print(f"Portrait cache: {CACHE_DIR}")
    print(f"Contact sheet: {CONTACT_SHEET}")
    print(f"Output: {output_path}")
    print(f"Named output: {named_output_path}")
    print(f"Historic portrait cache: {HISTORIC_CACHE_DIR}")
    print(f"Historic contact sheet: {HISTORIC_CONTACT_SHEET}")
    print(f"Historic output: {historic_output_path}")
    print(f"Historic named output: {historic_named_output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--named-output", type=Path, default=DEFAULT_NAMED_OUTPUT)
    parser.add_argument("--historic-output", type=Path, default=DEFAULT_HISTORIC_OUTPUT)
    parser.add_argument(
        "--historic-named-output", type=Path, default=DEFAULT_HISTORIC_NAMED_OUTPUT
    )
    parser.add_argument("--force-fetch", action="store_true", help="redownload cached portraits")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        args = parse_args()
        build(
            args.input,
            args.output,
            args.named_output,
            args.historic_output,
            args.historic_named_output,
            force_fetch=args.force_fetch,
        )
    except (RuntimeError, OSError, urllib.error.URLError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
