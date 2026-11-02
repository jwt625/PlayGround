# CPO assembly line, doodle loop

A looping isometric doodle animation of a co-packaged optics assembly line and its test gates, cut to a soundtrack. Everything is drawn in code on Canvas2D: no image generation, no Blender, no libraries.

Plan, decisions and feedback: `DevLog/DevLog-000-kickoff-style-drafts.md`. Lessons for similar work: `.claude/skills/iso-doodle-loop/SKILL.md`.

## Outputs

| File | What |
|---|---|
| `outputs/cpo_line_loop_sway.mp4` | Final. 12.8 s seamless loop, 8 bars at 150 BPM, 1280x844, 30 fps, stations dancing |
| `outputs/cpo_line_loop.mp4` | Same loop with the stations still |
| `outputs/style_drafts/` | Early style comparison and OE model stills |

## Rebuild

Needs `uv`, `ffmpeg`, and a Playwright Chromium headless shell (`uvx playwright install chromium-headless-shell`).

```sh
scripts/build_video.sh
uv run --no-project --with playwright --with pillow python scripts/make_style_sheet.py
```

## Checks

Run from `scripts/` with `uv run --no-project --with playwright python <script>`:

| Script | Checks |
|---|---|
| `check_order.py --step 0.0666666667` | Pairs drawn in the wrong order, on every video frame |
| `check_layers.py` | Solids whose bounding boxes intersect |
| `check_blink.py` | Known Good Die FAIL run lengths in frames |
| `bench_render.py` | One frame's time by stage, per browser flag set |

`scripts/audio_analysis/` holds the tempo, subdivision and near-loop measurements for the track.

## Layout

| Path | What |
|---|---|
| `lib/doodle.js` | Isometric projection, solids, hand line, fills, dance curve, painter sort |
| `lib/line_anim.js` | The line: beat clock, stations, rejects, boxing |
| `lib/models.js` | OE models and the style-draft line |
| `scenes/` | HTML entry points. `line_loop.html` takes `bars`, `dance`, `pace`, `w`, `h` |

## Third-party material

Used here without any license from the rights holders; rights not checked.

- `Mittsies - Vitality.m4a`: the soundtrack, by Mittsies, from the game Helltaker.
- `Factorio - Base Game Complete Soundtrack [16.59-21.32].m4a`: a soundtrack candidate, not in the outputs.
- `lib/fonts/xkcd-script.ttf`: xkcd-font, CC BY-NC per its repo.
- The flow and test points follow slides photographed at OFC 2026. Illustrative only, with no vendor names.

The Helltaker motion reference video is not in this repo.
