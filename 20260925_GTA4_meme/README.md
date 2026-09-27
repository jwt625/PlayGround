# Grand Theft Alignment IV

An AI-industry meme cast and reusable loading-screen template, built from inspection of the cached GTA IV video. Six generated transparent portraits, six backgrounds based on photographs of real AI-company sites, independent scene-specific motion, fades through black, and the cached soundtrack.

## Watch and edit

- `output/v3/ai-industry-clean.mp4`: revised 54 seconds, 1440×900, 30 fps, no character labels; closest to the reference's spare composition.
- `output/v3/ai-industry.mp4`: same revised sequence with names, GTA counterparts, and meme dialogue.
- Earlier exports remain in `output/` and `output/v2/` for comparison.
- `index.html`: editable browser preview. Open directly for playback, or use the local server for PNG export.
- `output/v3/card-*.png` and `output/v3/clean-*.png`: revised individual composed stills.
- `assets/*.png`: original generated portraits with alpha and background plate.
- `prompts.json`: original portrait/background prompts; `prompts-corporate.json`: revised background prompts and reference inputs. Generated with the built-in image_gen tool.
- `reference/corporate/SOURCES.md`: real-site photo sources, credits, identification checks and artistic changes.
- `CAST.md`: casting rationale, limitations, and sources.

From this directory:

```sh
npm install
npm run preview
```

Open http://127.0.0.1:8765. Play enables music; choose a character to jump to its card. Edit names, counterparts, taglines, and dialogue; replace a portrait; choose each card's background and motion profile; change side, duration, and soundtrack start; toggle labels. Save the template JSON to preserve edits, including per-card background/motion choices. Replacement portraits are embedded as data URLs. Use `config.js` to change the default sequence, add characters, or change title, audio, and fade length. The browser editor currently edits the existing cast rather than adding or reordering cards.

```sh
npm run render
npm run render:clean
node render.mjs --config my-template.json --output output/my-meme.mp4
node render.mjs --config my-template.json --clean --output output/my-meme-clean.mp4
```

The renderer requires Node and `ffmpeg` on PATH. Preview and export share `draw.js`, so composition and timing use the same equations. Default browser/system fonts may differ slightly across operating systems. `--width 1920` exports 1920×1200, retaining the source's 8:5 aspect ratio. Exports longer than the remaining soundtrack (about 125 seconds at the default 13-second offset) need a longer audio source; the renderer does not loop the music.

## What was observed

The supplied file is 1152×720, 30 fps, about 138.13 seconds. It contains an extended black opening and ending; the current edit omits both and begins immediately on Ilya. `reference/sparse.jpg` samples the whole clip at one frame per 8 seconds; `reference/dense.jpg` samples 12–18 seconds at 4 fps; `reference/transition.jpg` samples 21–26 seconds at 4 fps. Contact sheets read left to right, top to bottom; trailing unused cells are black. FFmpeg's fps sampling chooses representative frames within its sampling windows rather than exact labeled seek times.

The essential grammar is saturated, angular painted figures against pale monochrome urban plates; figures remain rigid; layers translate/scale slowly; cuts fade briefly through black. Sparse samples show several distinct settings: downtown street canyon, bridge-side block, industrial corner, and elevated bridge/skyline. The denser `motion-25-33.jpg`, `motion-41-49.jpg`, and `motion-49-57.jpg` sheets show that camera and figure motion vary by shot, including small vertical components and scale changes. The template approximates that grammar; it is not a traced reconstruction of the original motion curves. The labeled version adds typography that is absent from the sampled reference. The clean version omits it.

Timing: Ilya 0–9 s; Sam 9–18; Jensen 18–27; Elon 27–36; Dario 36–45; Mark 45–54. There are no opening or ending screens. Internal cuts retain 0.38-second fades on either side; the first and last frames remain fully visible. Audio starts immediately at source 0:13 (config `audioStart: 13`) and fades out over the last two seconds. Preview playback, seeking, restart, and export use the same offset. Older saved templates without `audioStart` retain a zero offset. Portraits deliberately crop below the waist/thigh to keep faces large.

## Scene-specific backgrounds and motion

| Character | Photo-based background | Foreground travel | Background zoom, relative to cover |
|---|---|---|---|
| Ilya | Pioneer Building, historical OpenAI allusion | Up-left, 6% slow zoom-in | 1.065 → 1.090 |
| Sam | Stargate / Crusoe Abilene | Down-right, slight shrink | 1.095 → 1.070 |
| Jensen | NVIDIA Voyager / Endeavor | Down-left, 6% slow zoom-in | 1.060 → 1.082 |
| Elon | xAI Colossus 1, Memphis | Up-right, 5% slow zoom-in | 1.090 → 1.070 |
| Dario | Anthropic, 500 Howard / Foundry Square IV | Mostly upward, 4% slow zoom-in | 1.075 → 1.091 |
| Mark | Meta MPK 21 garden terraces | Down-right, slight shrink | 1.100 → 1.073 |

Every layer has its own x/y/scale endpoints and restrained easing, with foreground translation around 7–33 pixels horizontally and 10–17 pixels vertically over nine seconds at 1440×900. Background translation and scaling run independently. This avoids a repeated alternating horizontal slide; no bobbing, limb animation or oscillation is added.

`config.js` contains `backgrounds`, `motionProfiles`, and per-card `background`/`motion` assignments. Motion `x` and `y` are `[start,end]` fractions of frame width/height; portrait `y` is the image top. Background `zoom` is relative to cover; portrait `height` is relative to frame height. A card can supply an inline `motion` object instead of a profile name. Oversized background pans are clamped to keep the frame covered. Templates saved before this revision still use their original single background and motion.

## Artwork and audio

The portraits and backgrounds were generated with the built-in image_gen tool. Original portrait PNGs and alpha are preserved. New corporate plates are saved as `assets/background-{pioneer,stargate,nvidia,colossus,anthropic,meta}.png`. Architecture follows the downloaded photos; small identifying signs and camera changes are artistic additions. Pioneer is a historical OpenAI allusion, not an SSI campus. Props and poses are fictional satire; captions are invented, not quotations. Source-photo credits and links are in `reference/corporate/SOURCES.md`. The extracted `assets/theme.m4a` is from the supplied cached video; no music rights are granted by this project. Source video and metadata are untouched.

## Validation

Checked portrait alpha, composed stills, browser asset loading and editing, playback, MP4 duration/codecs, and sparse/dense exported frames. The strongest likenesses are Sam, Jensen, Elon, and Mark; Ilya and Dario remain stylized interpretations rather than identity-locked portraits. Casting quality and possible alternatives are documented in `CAST.md`.

Revision 2: verified all twelve image assets, all six background/motion selections, visible movement on every card, JSON save/load of those choices, and browser playback with no page errors. Sampled 4,878 motion/size combinations to check full-frame background coverage and checked legacy-template fallback. Both exports contain 1,800 H.264 frames at 1440×900/30 fps and 60 seconds of AAC audio. `output/v2/review-export.png` samples the encoded clean video; `review-transition.png` shows its first inter-card fade; `review-motion.png` compares 0.6, 4.5, and 8.4 seconds within each scene.

Revision 3: removes both title screens, offsets the soundtrack by 13 seconds, and adds visible 4–6% foreground zooms to four of the six portraits while retaining their translation. The sequence lasts 54 seconds; previous exports are preserved. Both exports were verified as 1,620 H.264 frames at 1440×900/30 fps with 54-second AAC streams starting at time zero. The first five seconds of exported audio match reference-video seconds 13–18 (waveform correlation 0.9977). Browser checks passed for immediate first/last compositions, playback/seek/restart audio alignment, offset editing and JSON roundtrip, and four foreground zooms with simultaneous drift. `output/v3/review-export.png` samples all six scenes; `editor-preview.png` shows the updated editor.
