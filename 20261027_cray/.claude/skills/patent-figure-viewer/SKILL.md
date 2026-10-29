---
name: patent-figure-viewer
description: Turn scanned patent PDFs into a self-contained interactive gallery and viewer - verbatim text with column:line citations, every figure retraced as vector elements keyed to its reference numerals, linked to the description and claims. Use when asked to build an interactive explainer, viewer or gallery for patents (or similar scanned technical documents with numbered drawings), to add another patent to this viewer, or to re-run or repair any stage of this pipeline.
---

# Patent figure viewer

Pipeline and viewer live in this project (`20261027_cray/`). Every patent goes through the same
stages and produces the same files; `pipeline/build.py` turns all of `data/*` into one `index.html`.
Read before working: `README.md`, `pipeline/DATA_SCHEMA.md`, `pipeline/TRACE_SCHEMA.md`, and the
user's `~/.claude/VIZ-STYLE-GUIDE.md` (dark default, sharp corners, no explainer copy, self-contained).

## Stages

| # | stage | who | command / output |
|---|---|---|---|
| 1 | page images + OCR | script | `pipeline/extract.sh <pn>` -> `work/<pn>/pages/p-NNN.png`, `work/<pn>/ocr/` |
| 2 | verbatim text | agent per patent | `data/<pn>/text.json` |
| 3 | figure table, parts, notes | agent per patent | `data/<pn>/figures.json`, `data/<pn>/notes.json`, then `python3 -I pipeline/crop.py <pn>` |
| 4 | traces | agents, split by complexity | `data/<pn>/traces/fig-<id>.json`, checked with `overlay.py` + `coverage.py` |
| 5 | build | script | `python3 -I pipeline/build.py [--only <pn>...]` -> `index.html` gallery + `p/<pn>.html` per patent |
| 6 | verify | script + eyes | `pipeline/selftest.sh`, `pipeline/shoot.sh '#/<pn>/figures/<fig>/<ref>' out.png` |

Setup for a new patent: put the PDF at `sources/<pn>.pdf` (USPTO image PDF), write
`data/<pn>/meta.json` = `{"col1_page": <0-based page index of text column 1>, "google": "https://patents.google.com/patent/US<pn>A/en", "group": "<gallery section>"}`
(groups and their order: `GROUPS` in `build.py`). PDF links: Google Patents page HTML has the `patentimages.storage.googleapis.com/...pdf` URL;
B1 grants (no pre-grant publication) use `US<pn>B1`.
Find `col1_page` from the OCR: the first page whose text starts with the patent number then `1`.

`work/` is regenerable and untracked: on a fresh checkout run stage 1 and `crop.py` for each patent
before tracing or running overlay/coverage. `build.py` needs `work/` too (it embeds crops and pages);
the committed `index.html` does not.

## Orchestration

- Stages 2 and 3 are independent: run a text agent and a figure-table agent per patent in parallel.
- Stage 4 starts when `figures.json` + crops exist (text is not needed; tracers read raw OCR).
  Split figures across tracing agents using the complexity estimates the figure agent reports:
  one agent for the single hardest block diagram, pair medium figures, batch simple/timing/table
  figures. 2-3 tracers per patent was right for 5-12 figures.
- Write the viewer/build while agents run; test the build as soon as one `text.json` lands.
- Agent briefs that worked are in `references/agent-briefs.md`. Brief each agent with the goal
  (faithful interactive rebuild, near visual match), the schema files, inputs, outputs, the
  overlay iteration loop, validation rules, and what to report. Tell agents not to edit shared
  files (renderer, figures.json) but to report needs; extend the renderer yourself, update
  TRACE_SCHEMA.md, and message running agents about new options.

## Faithfulness rules (non-negotiable)

- The scan is the authority. Google Patents text only resolves OCR misreads; it has its own errors.
- Keep printed typos and misprints ("correpondingly", "FIG. 6 ... of FIG. 6"). Only undo line-break
  hyphenation and drop gutter numbers/running headers. Every correction goes in `ocr_fixes`.
- Part names are the noun phrase the description uses when introducing the numeral, verbatim.
- Anything not quoted from the patent is a Note (`notes.json`), shown in a labeled box. Check notes
  against the drawing: tracers caught a note saying V_{j} where the figure prints V_{i}.
- Traces copy what is printed: every line, symbol, string, numeral and leader, in place.

## Trace quality loop

1. Generator script per figure in `work/<pn>/tmp/trace-<fig>/` (regenerate after every fix).
   Measure from the scan with PIL/numpy/scipy (dark-run projections, connected components,
   skeletonisation for freeform strokes); snap coordinates; deskew if the scan is rotated.
2. `python3 -I pipeline/overlay.py <pn> <fig> [--region x0,y0,x1,y1]`: grey scan, red trace. Read it.
3. `python3 -I pipeline/coverage.py <pn> <fig>`: covered %, off-scan %, largest MISSING/EXTRA
   clusters as `--region` boxes. Typical finished figures: 90-98 % covered, under 3 % off-scan;
   residual MISSING is mostly pattern fills and hand-lettering glyph shapes.
4. Repeat until no geometry is missing or misplaced; also read the `.trace.png` alone.

Renderer features that exist because scans needed them (see TRACE_SCHEMA.md): `sub` scale/drop for
full-size dropped subscripts, hatch `pitch`/`angle`/`hw`, `stipple`/`dashes` fills for gas/liquid
regions, `paper` ink/fill for white knockouts, `w: 0` invisible hit areas for unoutlined spaces,
`step` + `steps` for timing figures (viewer slider), text `len` to force printed width.

## Viewer behaviour (viewer/app.js)

Gallery card per patent (front-page figure as trace). Figures tab: list, pan/zoom stage with
Redraw/Scan/Overlay, hover/pin a part highlights all elements sharing its `ref`, side panel with
verbatim mentions + citations, parts tour, step slider. Description/Claims tabs: precomputed numeral
and FIG. links (`build.py link_spans`), hovering a numeral drives a linked figure pane, citations
open the scanned page. About tab: note, bibliography, glossary, parts index, references.

## Gotchas learned

- Numeral linking (`build.py`): patents say "reference number 20", "data line 80", "transmitted to 161",
  "(134, FIG. 5)". Block only true non-references (FIG./claim/column/Ser. No., numeric ranges
  "0 to 63", unit words after the number). After a build, list parts with zero mentions and grep
  their contexts; zero should mean drawing-only numerals or misprints.
- Figure ids: sub-figures (3A-3D) share the parent's brief description; the front page repeats one
  figure (flag `front_page`, don't duplicate). Rotated sheets: `rotate` is CCW degrees.
- Headless Chrome lays out at >= 500 px wide even with a 390 px window; check phone width by
  framing the page in a 390 px iframe (`--allow-file-access-from-files`).
- Chrome throttles hundreds of rapid `location.hash` changes; the self-test routes internally.
- `?selftest`, `?step=N`, `?theme=light|dark` are headless review hooks.
- Parent repo `.gitignore` ignores `*.json/*.pdf/*.html`; the project `.gitignore` re-includes them.
- Size: full-page 1-bit scans are ~0.1 MB each base64; 13 patents of pages alone were ~34 MB, so the build
  is split (light gallery `index.html`, one self-contained `p/<pn>.html` per patent, each under ~20 MB).
- Check drawings exist before proposing a patent: Cohen-Boyer US 4,237,224 is "No Drawings".
- Batch runs: a Workflow pipeline per patent (text + figure agents in parallel, tracers grouped by the
  figure agent's complexity score: >=7 solo, else packed to a load of 10, max 6 figures) ran 11 patents /
  208 figures in ~2 h with 97 agents at ~12 concurrent. Tracer median 12~18 min, dense logic sheets ~25.
- Numeral linking also covers uppercase suffixes (12D), primes (3', 320′ -> ASCII in parts), comma lists
  without spaces (42,48) and range ends (51-54). Letter labels (A, B, C) stay drawing-only.
- Drawing-only designators (R1, C1, Q1) used as refs get a part with an empty name, never an invented one.
- Term status: 17 years from grant; 20 from filing after 1995-06-08; the later of the two if in force on
  1995-06-08; 20 years counts from the earliest U.S. non-provisional parent (front page [63]); a printed
  terminal disclaimer ("[*] Notice") caps it. Google's "anticipated expiration" missed a disclaimer (4,859,016).
- Tracer wishlist (not built): hatch phase/offset, centred hatch line, text skew/weight/outline for hand
  lettering, arrowhead length/width, dash offset, hit-only fill, multi-ref elements, ignore-mask in
  coverage.py for neighbouring-figure ink. Hand lettering vs Helvetica is the main residual off-scan ink.
- Certificates (correction, reexamination) are appended pages, sometimes a different page size
  (2560x3300): text paras carry `page`; the scan modal reads per-page sizes (`page_wh`).
