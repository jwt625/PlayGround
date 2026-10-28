# Data schema

Every patent goes through the same stages and produces the same files under `data/<pn>/`.

| stage | tool / owner | output |
|---|---|---|
| 1 extract | `pipeline/extract.sh <pn>` | `work/<pn>/pages/p-NNN.png`, `work/<pn>/ocr/p-NNN.{txt,tsv}` |
| 2 text | OCR corrected against the scans, cross-checked with Google Patents | `data/<pn>/text.json` |
| 3 figures | figure table, then `pipeline/crop.py <pn>` | `data/<pn>/figures.json`, `work/<pn>/figs/fig-<id>.png` |
| 4 traces | traced against crops, checked with `pipeline/overlay.py` | `data/<pn>/traces/fig-<id>.json` (see TRACE_SCHEMA.md) |
| 5 notes | labeled context notes | `data/<pn>/notes.json` |
| 6 build | `pipeline/build.py` | `index.html` (gallery + viewers, self-contained) |

## Faithfulness rules (text and names)

- The scan is the authority. Google Patents text is only for resolving OCR misreads.
- Keep the patent's own spelling, typos, capitalization, punctuation and wording. A word the
  printed patent misspells stays misspelled. Only undo line-break hyphenation
  ("arith-\nmetic" -> "arithmetic"), keeping true hyphenated compounds.
- Drop the gutter line numbers (5, 10, 15 ...) and running headers; keep everything else.
- Formulas, subscripts, register names: render as printed using `_{}` / `^{}`, e.g. `V1_{3}`.

## text.json

```json
{
  "pn": "4128880",
  "number": "4,128,880",
  "title": "COMPUTER VECTOR REGISTER PROCESSING",
  "inventors": ["Seymour R. Cray, Jr."],
  "assignee": "Cray Research, Inc., Chippewa Falls, Wis.",
  "filed": "1976-06-30", "issued": "1978-12-05", "appl_no": "701,119",
  "int_cl": "...", "us_cl": "...",
  "refs_patents": [{"no": "3,462,744", "date": "8/1969", "name": "Tomasulo et al.", "cl": "340/172.5"}],
  "refs_other": ["Anderson et al.; Microelectronic Cooling by ...; 1988."],
  "examiners": "Primary Examiner--...; Attorney, Agent, or Firm--...",
  "claims_count": 12, "drawing_figures": 13,
  "abstract": "Vector processing in a computer ...",
  "sections": [
    {"heading": "BACKGROUND OF THE INVENTION",
     "paras": [{"col": 1, "line": 4, "text": "The present invention pertains to ..."}]}
  ],
  "brief_drawings": [{"fig": "1", "col": 2, "line": 60, "text": "FIG. 1 is a block diagram of ..."}],
  "claims": [{"n": 1, "col": 18, "line": 40, "depends": null, "text": "Vector processing apparatus ..."}],
  "ocr_fixes": [{"col": 1, "line": 47, "ocr": "IIliac", "fixed": "Illiac", "basis": "scan + Google Patents"}]
}
```

- `col`/`line`: where the paragraph starts in the printed patent (column number printed at the
  top of each text column, line from the gutter numbering; interpolate between the 5-line marks).
- Paragraphs: one entry per printed paragraph. Sub-headings in the description become new sections.
- `brief_drawings`: one entry per figure from the BRIEF DESCRIPTION OF THE DRAWINGS (also kept
  in `sections`). Tables, if any, go in a section para with `"table": [[cells], ...]` instead of `text`.
- `ocr_fixes`: every place where OCR was wrong and corrected (audit trail).

## figures.json

```json
{
  "pn": "4128880",
  "figures": [
    {"id": "1", "label": "FIG. 1", "sheet": 1, "page": 1,
     "crop": [x0, y0, x1, y1], "rotate": 0,
     "front_page": false}
  ],
  "parts": [
    {"ref": "20", "name": "vector registers", "figs": ["1", "2"]}
  ]
}
```

- `page` is the 0-based page index (`work/<pn>/pages/p-NNN.png`), `crop` in native page pixels
  (2320 x 3408), generous margin (about 40 px) around the figure including its `FIG. n` label.
- `rotate`: degrees counter-clockwise that make the figure read upright (PIL `rotate`).
- One entry per figure, even when a sheet holds several. Figure ids are strings ("1", "5A").
- `parts`: every reference numeral used in the drawings or the description. `name` is the noun
  phrase the description uses when it introduces the numeral, verbatim ("vector register V0",
  "cooling liquid"), lower case kept as printed. A numeral used for two things gets the first
  introduced name and an `"alt": ["..."]` list.

## notes.json

Context written by us, always shown in the viewer as a labeled note, never mixed with quotes.

```json
{
  "patent": "Short objective context: what machine/product this describes, era, why it matters.",
  "figs":  {"1": "One to three sentences: what to look at, how the parts interact."},
  "parts": {"20": "Optional one-liner for parts where the patent's term needs context."},
  "glossary": [{"term": "chaining", "def": "..."}]
}
```

Notes are objective and terse: no hype, no audience lines, no restating the quoted text.
Any factual claim not in the patent (e.g. "this is the Cray-1") must be well established.
