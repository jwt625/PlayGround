# Patent figure viewer

Interactive gallery of classic patents, built from the USPTO scans with one pipeline:

| Group | Patent | Title | Figures |
|---|---|---|---|
| Cray | US 4,128,880 (1978) | Computer Vector Register Processing | 12 |
| Cray | US 5,131,233 (1992) | Gas-Liquid Forced Turbulence Cooling | 5 |
| Impact and royalties | US 4,018,121 (1977) | Method of Synthesizing a Musical Sound (Chowning, FM synthesis) | 18 |
| Impact and royalties | US 4,200,770 (1980) | Cryptographic Apparatus and Method (Hellman, Diffie, Merkle) | 6 |
| Impact and royalties | US 4,859,016 (1989) | Fiber Optic Amplifier (Shaw, Digonnet; 3 reexamination certificates) | 14 |
| Impact and royalties | US 6,285,999 B1 (2001) | Method for Node Ranking in a Linked Database (Page, PageRank) | 3 |
| Classic products | US 3,138,743 (1964) | Miniaturized Electronic Circuits (Kilby, integrated circuit) | 14 |
| Classic products | US 3,659,915 (1972) | Fused Silica Optical Waveguide (Maurer, Schultz, Corning) | 2 |
| Classic products | US 3,821,715 (1974) | Memory System for a Multi-Chip Digital Computer (Hoff, Mazor, Faggin, Intel) | 5 |
| Classic products | US 5,061,049 (1991) | Spatial Light Modulator and Method (Hornbeck, TI DMD) | 73 |
| Industry legends | US 2,524,035 (1950) | Three-Electrode Circuit Element (Bardeen, Brattain, transistor) | 18 |
| Industry legends | US 3,387,286 (1968) | Field-Effect Transistor Memory (Dennard, DRAM cell) | 9 |
| Industry legends | US 3,541,541 (1970) | X-Y Position Indicator (Engelbart, mouse) | 7 |

Open `index.html` (gallery, under 1 MB); each patent is a self-contained `p/<pn>.html` (1~10 MB) that
works from `file://` with no network. Term status (`meta.json` `status`) is computed from the statutory
rules and printed disclaimers, cross-checked against Google Patents legal status; it is not legal advice.

## Views

- **Gallery** (`index.html`): search (number, title, people, places, part names), Cards / Table views,
  12 sort keys (table headers sort too), facet chips (group, decade, assignee kind, country). State is in the
  hash (`#/?v=table&s=issued&d=desc&q=stanford`). Each patent opens its own page `p/<pn>.html`.
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
