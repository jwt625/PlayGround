# Trace schema

One file per figure: `data/<pn>/traces/fig-<id>.json`. Coordinates are pixels of the upright crop
`work/<pn>/figs/fig-<id>.png` (origin top-left, y down), produced by `pipeline/crop.py`.
Rendered by `viewer/trace-render.js`, the same code the viewer ships.

```json
{
  "fig": "1",
  "w": 1830, "h": 2410,            // crop size in px (must equal the PNG size)
  "stroke": 3,                     // default stroke width in px, match the scan line weight
  "elements": [ Element, ... ]
}
```

## Element

```json
{
  "id": "vreg",                    // unique within the figure, kebab/snake case
  "ref": "20",                     // patent reference numeral this element IS (omit if none)
  "kind": "block",                 // see kinds below
  "name": "vector registers",      // optional override; default comes from figures.json parts
  "step": 3,                       // optional, timing/sequence figures: clock or step index
  "shapes": [ Shape, ... ]
}
```

Kinds (drive styling and hit-testing only):

| kind | use |
|---|---|
| `block` | a physical or functional part: register, unit, module, pump, valve, housing |
| `wire` | signal path, bus, pipe, duct, flow line (usually `l` shapes with arrows) |
| `label` | a reference numeral as printed (`t` shape); give it the same `ref` as the part |
| `leader` | the lead line from a numeral to its part; same `ref` |
| `text` | any other printed text: legends, `FIG. 1`, names inside blocks, axis labels |
| `outline` | dashed group boundaries, frames, cross-section hatching areas |
| `mark` | timing bars, ticks, element bars, any repeated glyph without a numeral |

A part with numeral 40 is normally three elements: the `block` (ref 40), the `label` (ref 40) and
its `leader` (ref 40). All three highlight together in the viewer.

## Shape

| t | data | notes |
|---|---|---|
| `r` | `"d": [x0, y0, x1, y1]` | rectangle |
| `c` | `"d": [cx, cy, r]` | circle |
| `e` | `"d": [cx, cy, rx, ry]` | ellipse |
| `p` | `"d": [[x, y], ...]` | closed polygon |
| `l` | `"d": [[x, y], ...]` | open polyline; `"arrow": "end" \| "start" \| "both"` |
| `path` | `"d": "M x y L ... C ... A ... Z"` | SVG path, ABSOLUTE commands only (M L H V C S Q T A Z) |
| `t` | `"x", "y", "s", "size", "anchor", "rot", "italic", "bold", "len"` | text; `y` = baseline; `anchor` start/middle/end; `rot` degrees; `len` forces text length in px; `"V1_{3}"` subscript, `"t^{2}"` superscript |

Subscript style: `"sub": {"scale": 1.0, "drop": 12, "rise": 12}` (scale x font size, drop/rise in px)
on a `t` shape, or at the trace top level as the figure default. Default: scale 0.68, typographic offsets.

Optional on any shape: `"w"` stroke width px (`0` = no outline, fill only), `"dash": true` (or `[on, off]` px),
`"fill": "none" | "light" | "solid" | "hatch" | "stipple" | "dashes"` (default none; `solid` = ink, e.g. filled
dots/arrows, `light` = tinted part body, `hatch` = 45 deg cross-section hatching, `stipple` = dot fill for
gas/vapor regions, `dashes` = short-dash fill for liquid regions). `"fill": "paper"` and `"ink": "paper"` draw in the
background colour (white knockouts such as slashes inside a solid block). Hatch spacing:
`"pitch": px, "angle": deg, "hw": px` on a `hatch` shape (default 14 px, 45 deg, 2 px line) to match printed hatching. Use a pattern fill instead of tracing
hundreds of individual hatch strokes or dots; trace individual strokes only where they carry meaning.
Known renderer behaviour, kept on purpose: the hatch line sits on the pattern tile edge, so it renders
`hw/2` wide, offset `hw/4`; the pattern phase is fixed to the figure origin. All existing traces were
fitted against the overlay with this behaviour (most pass `hw` = 2x the printed width), so changing it
needs a re-fit of every hatched figure. Phase-match hatching by tuning `pitch` per region, or trace
measured strokes.

## Sequences (timing diagrams, step-by-step figures)

Give elements a `step` (integer, 0-based) when the figure shows a progression in time (clock periods,
element transfers, issue/go/free events). The viewer then offers a step slider: elements with
`step` > current fade, `step` == current is highlighted. Elements without `step` always show.
Name the steps at the top level, verbatim from the figure where possible:

```json
"steps": {"label": "Clock period", "names": ["t_{0}", "t_{1}", "t_{2}"]}
```

## Rules

- Trace what is printed: every line, box, arrow, numeral, leader, and text string, in place.
  No invented parts, no beautification that moves geometry. Straighten hand-drawn wobble only.
- Text strings are copied exactly as printed (case, abbreviations, subscripts).
- Every numeral printed in the figure gets a `label` element with its `ref`.
- Match the scan to within a few px: check with
  `python3 -I pipeline/overlay.py <pn> <fig> [--region x0,y0,x1,y1]`, read the PNGs, fix, repeat.
- Keep file size sane: repeated structures (register stacks, element bars) as separate small
  elements are fine; do not trace halftone noise or scan specks.
