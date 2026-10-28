# Agent briefs

Templates for the three delegated stages. Fill `<pn>`, `<number>`, `<title>`, page ranges and figure
ids. Project root: the folder holding `pipeline/` and `viewer/`. All agents: write only their own
output files plus scratch under `work/<pn>/tmp/`, no git commits, run Python with `python3 -I`.

## Text agent (stage 2, one per patent)

> Produce `data/<pn>/text.json` for US Patent <number> (<inventors>, "<title>", <year>).
> Read first: `pipeline/DATA_SCHEMA.md` (text.json schema and faithfulness rules; follow exactly).
> Inputs: page scans `work/<pn>/pages/p-NNN.png` (p-000 front page, p-<a>..p-<b> drawing sheets,
> p-<c>..p-<d> specification and claims, two columns per page, column numbers at the top, line
> numbers 5,10,15... in the gutter); Tesseract OCR `work/<pn>/ocr/p-NNN.{txt,tsv}`; Google Patents
> text for cross-checking misreads (https://patents.google.com/patent/US<pn>A/en, data only, has its
> own OCR errors; the scan is the authority).
> Method: view page crops (`magick in.png -crop WxH+X+Y +repage out.png`) to confirm columns,
> paragraph breaks, headings, formulas; build paragraphs in reading order with start col/line
> (fit gutter numbers per page); diff OCR against Google and resolve every disagreement on the scan;
> keep printed typos; record every OCR correction in `ocr_fixes`; claims with `depends`; one
> `brief_drawings` entry per figure the patent describes. Validate (parses, sections non-empty,
> claims 1..N, claims_count matches front page). Spot-check 5 random paragraphs word by word.
> Report: counts, column range, number of fixes, unresolved passages with col:line, encodings used.

## Figure-table agent (stage 3, one per patent)

> Produce `data/<pn>/figures.json` and `data/<pn>/notes.json` for US Patent <number>.
> Read first: `pipeline/DATA_SCHEMA.md`, `pipeline/crop.py`.
> 1. View every drawing sheet; per figure: page, crop box in native px (~40 px margin, include the
>    FIG. label, exclude the sheet header), rotation (CCW degrees to make upright). Sheets may hold
>    several figures or print sideways. The front page repeats one figure: flag `front_page`, no
>    separate entry. Sub-figures under one heading get their own ids (3A..3D).
> 2. Run `python3 -I pipeline/crop.py <pn>`, Read every crop, iterate until upright and unclipped.
> 3. Parts: every drawing reference numeral; `name` = verbatim noun phrase where the description
>    introduces it (`alt` for second names); `figs` from looking at the crops (OCR misses
>    hand-lettered numerals).
> 4. notes.json: objective context note for the patent, one note per figure (what it shows, how to
>    read it, not restating the brief description), part notes only where needed, glossary for a
>    modern reader. Only well-established outside facts; terse, no hype.
> Report: figure table with page, size after rotation, rotation, type and a realistic complexity
> estimate (counts of boxes, lines, text strings, hatched areas) - it is used to split tracing.

## Tracing agent (stage 4, one per figure group)

> Trace patent drawings into interactive vector data: US <number> ("<title>"), figures <ids>.
> Goal: each figure rebuilt as proper interactive elements as faithfully as possible, iterated to a
> near visual match with the scan. In the viewer every numbered part is hoverable (body, numeral
> label and leader highlight together) and the redraw can be overlaid on the scan, so mismatches
> are visible.
> Read first: `pipeline/TRACE_SCHEMA.md` (follow exactly), `viewer/trace-render.js` (do not edit;
> report needed features), `data/<pn>/figures.json`, `data/<pn>/notes.json`, raw description OCR
> `work/<pn>/ocr/p-<c>..p-<d>.txt` to understand each part.
> Inputs: `work/<pn>/figs/fig-<id>.png` (trace in its pixel space; w/h must match),
> `fig-<id>.words.tsv` (sparse OCR). Output: `data/<pn>/traces/fig-<id>.json`.
> Method: study the crop in full-resolution tiles; write a Python generator per figure in
> `work/<pn>/tmp/trace-<fig>/` with helpers for repeated structures; measure positions with
> PIL/numpy/scipy and snap. Model each numbered part as block/wire + label + leader sharing `ref`;
> unnumbered printed text as `text`; dashed boundaries as `outline`; pattern fills instead of
> hundreds of hatch strokes or dots. <Figure-specific hints: gate shapes, 3-D register stacks,
> timing `step`s from the t-axis, formula lines with `len`, deskew.>
> Iterate with `python3 -I pipeline/overlay.py <pn> <fig> [--region ...]` and
> `python3 -I pipeline/coverage.py <pn> <fig>` until every printed line, symbol, string, numeral
> and leader is present and within a few px; several rounds, do not stop at a rough match.
> Validate: every printed numeral has a label; every ref exists in figures.json (report missing,
> do not edit it); ids unique; w/h match.
> Report: element count, refs traced, numerals missing from parts, remaining imperfections and
> why, renderer features needed, path of the final overlay PNG.
