# Cray patent figure viewer

Interactive gallery for two Seymour Cray patents, built from the USPTO scans with one pipeline:

| Patent | Title | Figures |
|---|---|---|
| US 4,128,880 (1978) | Computer Vector Register Processing | 12 (FIG. 1, 2, 3A-3D, 4-9) |
| US 5,131,233 (1992) | Gas-Liquid Forced Turbulence Cooling | 5 |

Open `index.html` (self-contained, works from `file://`, no network).

## Views

- **Gallery**: one card per patent.
- **Figures**: figure list, pan/zoom stage, part panel. Every reference numeral is a traced
  vector element: hover or click its body, label or leader. View modes: Redraw (vector trace),
  Scan (original sheet), Overlay (trace over scan). Parts tour steps through parts in order of first
  mention; timing figures get a clock-period slider. Keys: Left/Right part, Up/Down figure, F fit, Esc clear.
- **Description / Claims**: verbatim text with column:line citations (click a citation for the scanned
  page). Numerals and FIG. references are links; hovering a numeral highlights the part in the
  figure pane beside the text.
- **About**: context note, bibliographic data, glossary, parts index, references, sources.

Patent text is quoted verbatim (typos as printed). Context added by us is always in a box labeled "Note".

## Pipeline

See `pipeline/DATA_SCHEMA.md` and `pipeline/TRACE_SCHEMA.md`. The whole workflow, including agent
briefs and lessons learned, is the Claude skill `.claude/skills/patent-figure-viewer/`.

Fresh checkout: `index.html` is committed and works as is. To rebuild or trace, regenerate `work/`
first: `pipeline/extract.sh <pn>` and `python3 -I pipeline/crop.py <pn>` for each patent.
Tools: poppler (`pdfimages`), tesseract, Python 3 with Pillow (numpy + scipy for `coverage.py`),
ImageMagick, Google Chrome (overlay, screenshots, self-test).

```
pipeline/extract.sh <pn>                 # 1 page PNGs + OCR       -> work/<pn>/
(agents)                                 # 2 text.json: OCR corrected against scans + Google Patents
(agents) + python3 -I pipeline/crop.py <pn>   # 3 figures.json + upright crops
(agents) + python3 -I pipeline/overlay.py <pn> <fig> [--region x0,y0,x1,y1]   # 4 traces, overlay-checked
python3 -I pipeline/coverage.py <pn> <fig>   # 4 covered % + largest missing/extra ink regions
(agents)                                 # 5 notes.json
python3 -I pipeline/build.py             # 6 -> index.html
pipeline/selftest.sh                     # headless interaction self-test
pipeline/shoot.sh '#/<pn>/figures/<fig>' out.png   # headless screenshot
```

`work/` is regenerable (`extract.sh` + `crop.py`) and git-ignored. `data/` holds every hand-checked
artifact. Adding a patent: drop `sources/<pn>.pdf`, add `data/<pn>/meta.json` (`col1_page` = 0-based page
index of column 1), run the stages.
