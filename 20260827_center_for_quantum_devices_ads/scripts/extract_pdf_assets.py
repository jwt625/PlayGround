#!/usr/bin/env python3
"""Extract reusable equipment photos and audit page samples from the source PDF."""

from pathlib import Path

import pymupdf


ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "source" / "CQD-Daily-responsibility-August21-2026.pdf"
PHOTO_DIR = ROOT / "public" / "assets" / "equipment"
SAMPLE_DIR = ROOT / "extraction" / "sample-pages"


def main() -> None:
    PHOTO_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(PDF)

    # The room-by-room equipment audit occupies pages 39–84. Small images are
    # mostly logos/icons; the threshold keeps equipment and facility photos.
    extracted = 0
    seen: set[int] = set()
    for page_number in range(39, 85):
        page = doc[page_number - 1]
        page_index = 0
        for image in page.get_images(full=True):
            xref, width, height = image[0], image[2], image[3]
            if width * height < 40_000 or xref in seen:
                continue
            seen.add(xref)
            page_index += 1
            payload = doc.extract_image(xref)
            suffix = payload["ext"].lower().replace("jpeg", "jpg")
            output = PHOTO_DIR / f"p{page_number:03d}-{page_index:02d}-x{xref}.{suffix}"
            output.write_bytes(payload["image"])
            extracted += 1

    for page_number in (1, 12, 30, 35, 39, 44, 50, 61, 69, 75, 84):
        pixmap = doc[page_number - 1].get_pixmap(
            matrix=pymupdf.Matrix(1.4, 1.4), alpha=False
        )
        pixmap.save(SAMPLE_DIR / f"page-{page_number:03d}.jpg")

    print(f"Extracted {extracted} source images from {len(doc)} pages")


if __name__ == "__main__":
    main()
