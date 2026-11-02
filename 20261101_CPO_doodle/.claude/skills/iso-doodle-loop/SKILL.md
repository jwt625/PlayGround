---
name: iso-doodle-loop
description: Build a looping isometric hand-drawn machine animation in code (Canvas2D, no Blender, no image generation), such as a factory or assembly line in a Japanese doodle style, cut to a soundtrack. Use for a looping process or manufacturing animation, a doodle or rakugaki style, an isometric line drawing, a seamless GIF or MP4 loop, fixing wrong or flickering layering in a 2D isometric scene, or matching station pace to a track's tempo.
---

# Isometric doodle loop

Distilled 2026-11-01 from the CPO assembly line loop (`cpo-doodle` in `~/.claude/skills/video-library/library.md`, session 30d7f7f2, 2026-11-01). Read the video-library skill first for his general standards. This file covers what is specific to this kind of work.

Reference build: `PlayGround/20261101_CPO_doodle` (github.com/jwt625/PlayGround). Rebuild with `scripts/build_video.sh`. Copy these rather than rewriting them:

| File | What it gives you |
|---|---|
| `lib/doodle.js` | True isometric projection, prisms, panels, tubes, decals, hand line, fill styles, line boil, painter sort, `DD.violations`, `DD.overlaps` |
| `lib/line_anim.js` | Stateless periodic line: beat clock, seeded rejects, pile slots, arm ripple, hinged carton |
| `lib/models.js`, `scenes/oe.html` | Exploded part stills for a style sheet |
| `scripts/render_loop.py` | Capture to PNG, encode by `--out` extension; about 23 s for 384 frames |
| `scripts/build_video.sh` | Render plus soundtrack crop, the full rebuild |
| `scripts/make_style_sheet.py` | Style stills and comparison sheet |
| `scripts/audio_analysis/` | Tempo grid, subdivisions, near-loop crops |
| `scripts/check_order.py` | Pairs drawn in the wrong order, the check that matters |
| `scripts/check_layers.py` | Solids whose boxes intersect |
| `scripts/check_blink.py` | Frame-by-frame run lengths of an indicator |
| `scripts/bench_render.py` | Time one frame by stage across browser flags |

## Order of work

1. Confirm the style in words, then show it: one sheet of 4-5 styles, each with a part close-up and a small line, using one shared model. He picked in one word from a sheet. Name the option you recommend.
2. Ask few numbered questions. Bare numbered fragments are final; record them verbatim in the DevLog.
3. Build the loop. Show a whole frame early.
4. Take feedback by region and by station. Render a fresh cut after each round; at about 20 s per render there is no reason to batch.
5. Compare before changing when he asks for numbers, and offer 2 or 4 versions when he asks for choices. Each version plays its loop twice so the seam can be heard.

## Rendering speed

- Time one frame by stage before any full render. Do not copy browser flags between projects.
- `--use-angle=swiftshader` is for WebGL. On Canvas2D it cost 2993 ms per frame raster against 9 ms without it, which made 360 frames take 564 s instead of 17 s. He caught this before I did.
- Read frames with `canvas.toDataURL`, not page screenshots. `DD.frame` alone returns before the raster is done, so its timing misleads; force a readback when timing.
- Never run many software-rendering browsers at once: 6 of them pushed load average to 74.

## Layering, the part that went wrong

A 2D painter sort of 3D boxes is exact if built this way, and the hand-drawn line comes free. Moving to a 3D engine was not needed.

- Compare only pairs whose **projected hexagons** overlap, using ranges on `x+y`, `y+z`, `x-z` for a view along (1, -1, 1). Screen bounding rectangles are not enough: long diagonal belts have huge rectangles, so unrelated pairs add arbitrary order constraints, which form cycles. Breaking a cycle draws parts wrongly, and moving parts change the cycles, so arms and legs flip frame to frame.
- Add an edge only when a plane separates the pair. Intersecting pairs give no constraint, so a part in flight through a machine cannot create a cycle.
- Geometry must not intersect. Fix collisions rather than tie-break them: items wider than the belt pitch, tool heads entering a closed part, spawning inside a box, overlapping trays and belts, parts stacked with interpenetrating layers. Build piles as pyramids: layers separated in z, cells in a layer separated in xy.
- `check_layers` at zero does **not** mean correct. I said fixed on that basis and he still saw wrong parts. Use `check_order` on every frame of the output, then look at region crops at 2x. State what each check covers.
- Delete stale crops before making a review sheet. One sheet mixed old and new stills.

## Seamless loops with randomness

- Derive everything from `t mod T`. No state, no `Math.random`, no `Date`.
- A part's fate is a hash of (line, gate, index mod N), so random events repeat exactly. Keep N even when anything alternates by parity.
- A growing pile cannot loop. Keep each reject's slot fixed and always drawn, so a landing part lands on its identical copy and the pile looks constant.
- Every line's phase is a whole number of beats, or the neighbors drift off the grid.
- Confirm closure by comparing frames at 0 and T, byte for byte.
- A moving solid needs a stable seed (`DD.dyn(key)`) or its wobble changes as it moves.

## Comedy and pace

- Move fast, then idle. Keep the period and shorten the duty cycle: a 16th down, a 16th up, parts crossing in about an 8th. Slow steady motion reads as dull at any period.
- Make arms hit one per beat, left to right, as a ripple. A station with a fast job keeps its own faster tap; say so explicitly and do not flatten it into the ripple.
- Dancing machines: study the reference frames and measure them first. A rigid lean was rejected ("no curvature at all"). What worked after the Helltaker idle: feet pinned, hips one way, shoulders the other, head only translating, holds on the beat, a quick swing between, a small stretch in transit, and a per-station amplitude for character (`drawBent`, `DD.dance`). Attached arms, lamps and heads ride with the head.
- Exaggerate scale at the joke: tiny dies popping to about 3x in flight read better than dies that keep their true size.
- He named real mechanics when a shortcut was not enough: fold the flaps and roll the tape rather than drop a finished box. Hinged panels (`DD.panel`) do this.
- Every failed part in view carries the mark, including parts that were always in the pile. Put the mark on walls too when tops are covered.
- Status lights must stay on long enough to read: 6 frames minimum, with the idle state between blinks.

## Fitting to a soundtrack

Agents cannot hear. Report numbers, and say so.

- Do not trust `beat_track` alone: it said 152 BPM on a track that is 150.00. Score a fixed grid across candidate tempos; the true tempo stands out (contrast 5.24 against about 1.1 at neighbors). Then check phase stability per 30 s window.
- Onset autocorrelation at musical lags shows which subdivisions are real. A cycle can sit exactly on an anti-pulse (0.3 s at -0.12 here) while the even multiple is strong.
- Pick a clock where a beat is a whole number of frames: 150 BPM at 30 fps is 12 frames. Express every time in beats from one constant.
- Near-loop search: on bar lines, correlate what follows the cut with what follows the start. Remove per-band means first, or every score reads about 0.99.
- He chooses the crop start by musical feel ("before the main theme kicks in"), so search from his start rather than arguing for the best score.
- Give the pace table with period, beats, nearest grid value and percent off **before** changing anything when he asks to compare.

## GIF or video

GIF at 1280 wide with boil reached 30+ MB, and he moved to MP4 at that point. Palette 64, no dither and no paper grain help. Halving the boil rate cut about a third. Offer that number before he has to ask.

## Framing

- "Single frame" means one fixed camera for the whole loop.
- A 30 degree belt runs corner to corner on about 3:2, so set the aspect ratio from the line's extents and fit the rectangle exactly (`DD.cameraRaw`).
- Parallel lines are rows along one axis. Added rows go in the same array; ask if a stated position does not fit the array, because rows below center leave through the bottom edge before reaching the right corner.
- Keep named features in frame by computing their projected position, and check them at the crop.
- When a layout changes, derive mounts from shared constants. A hardcoded `y0` left over from a draft put an indicator behind its column.

## Modeling from references

- Facts on screen come from sources; ask what the source does not state rather than asserting it. Exaggeration is fine when he approves it and it is recorded.
- Ingest references through the atlas skills with one agent per atlas, then a fresh-context audit before relying on them. Merge steps can touch shared registries: stop and ask if a tool would rewrite unrelated rows.
- A public PlayGround folder must not name private sources. Redact them even in quoted requests in the DevLog.

## Keeping it reproducible

One-off mux and analysis commands get lost. Put them in tracked scripts as you go, and rerun the build before committing. The parent `.gitignore` drops `*.mp4 *.png *.ttf`, so canonical outputs and fonts need `git add -f`.

## Subagents on a shared file

- Give each agent its own copy with a baseline commit, a narrow scope, and a patch as the deliverable. Merge in the coordinator.
- Agree the clock first. A clock change mid-run wasted the package-line half of both patches here, so tell running agents when the base changes.
- "Discuss" and "compare" mean no render. If he asks while agents run, say what has changed and what has not.
