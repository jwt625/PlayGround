# DevLog-017: Tree intro (F8) - real bird photos + form-factor name tags (v3; v1 had simple models)

| Field | Value |
|---|---|
| Date | 2026-10-10 |
| Status | v3 delivered 2026-10-10 (models removed per owner; photo + bird name + form-factor tag cards); API and hook names unchanged |
| Workstream | Tree (DevLog-015 item F8) |
| Owns | `scripts/intro/`, `assets/components/intro/`, `references/birds/`, `outputs/assets/intro/`, this file |
| Inputs | DevLog-015 (owner feedback), DevLog-006 (rev 3 tree: leaves, numbered lineage), DevLog-001 section 4 (mapping), DevLog-002 (backronyms), `research/birds.md`, `research/form_factors_and_companies.md`, `research/hero_800g_osfp_dr8.md`, `outputs/film_v1_20261010.mp4` 0-5 s |
| Consumer | S (scene assembly; fade into the street) |

## 1. Brief

Owner (verbatim, 2026-10-10): "opening scene should show actual picture of different kinds of birds with a simple model, and fade in the actual scene instead of on top of or inside the scene, contrast is horrible."

Coordinator: one Commons photo per leaf (free licence, recorded), simple bpy form-factor models (true proportions where research has them, else labelled approximate), a separate clean high-contrast intro set laid out as cards with clade frames and the connected pluggable lineage, hero card emphasized, ~4.5 s camera move ending on the hero card, cross-dissolve into the street; 4:5 framing, bottom 18% free for captions; append API for S; previews.

## 2. Plan (written before implementation)

### 2.1 Content (from DevLog-006 rev 3; no years)

| Leaf (bird) | Form factor | Branch |
|---|---|---|
| Theropod (root, fossil/reconstruction) | 1x9 soldered | lineage (0) |
| Archaeopteryx | GBIC | lineage (1) |
| Moa | XENPAK | other |
| Elephant bird | CFP | coherent |
| Emu | CFP2-DCO | coherent |
| Ostrich | XPO | lineage (6) |
| Chicken | SFP | lineage (2) |
| Hummingbird | QSFP-DD | lineage (5) |
| Wandering albatross | 400ZR | coherent |
| Parrot | compatible | other |
| Cuckoo | AEC | copper |
| Passenger pigeon | QSFP+ | lineage (3) |
| Rock pigeon | QSFP28 | lineage (4) |
| Dodo | CPO | co-packaged |
| Cyber pigeon (hero, render of the hero GLB) | 800G OSFP DR8 | lineage (5) |

Lineage (gold, numbered): 1x9 -> GBIC -> SFP -> QSFP+ -> QSFP28 -> QSFP-DD / OSFP -> XPO. Coherent (magenta): CFP -> CFP2-DCO -> 400ZR. Clades (faint frames, Stiller 2024 / Prum 2015 agreement only): Palaeognaths; Galloanserae; Neoaves (unresolved); Columbaves (inside Neoaves).

### 2.2 Layout (cards on a vertical wall, root at the bottom, hero at the top)

| Row | Cards (left to right) | Frame |
|---|---|---|
| 4 | HERO (1.5x card) | Columbaves |
| 3 | passenger pigeon, rock pigeon, cuckoo, dodo | Columbaves |
| 2 | chicken / hummingbird, albatross, parrot | Galloanserae / Neoaves (unresolved) |
| 1 | moa, elephant bird, emu, ostrich | Palaeognaths |
| 0 | theropod, archaeopteryx (+ legend) | root / stem |

Card: light neutral face, 4:3 photo on top, the simple model standing in front of the card (common mm scale so relative size is true), label (species / form factor) below, lineage number badge. Background dark neutral; lights inside the intro collection (no dependence on the scene world). Connectors routed in gutters (polylines), drawn behind cards.

### 2.3 Deliverables

- `scripts/intro/fetch_bird_photos.py` (Commons API, rate-limited, writes `references/birds/INDEX.md` rows) and `scripts/intro/crop_cards.py` (uv, Pillow).
- `scripts/intro/form_factor_models.py` (bpy builders, dims table with source / APPROX flag).
- `scripts/intro/build_intro.py`: builds `assets/components/intro/intro_tree.blend` (collection `INTRO_tree`, root empty, hooks `HOOK_intro_cam_start` / `HOOK_intro_cam_mid` / `HOOK_intro_cam_end` + targets, preview camera keyed 4.5 s) and exposes `append_intro(...)` + `key_camera(...)` for S; recommended fade window.
- Previews in `outputs/assets/intro/`: start / mid / end stills, 540x674 fast-EEVEE mp4.

## 3. As built

### 3.1 Files

| Path | What |
|---|---|
| `scripts/intro/fetch_bird_photos.py` | Commons API search / info / get (one request at a time, >= 6 s apart; stops on any HTTP error); appends `references/birds/fetch_log.jsonl` |
| `scripts/intro/crop_cards.py` | 4:3 card crops (800x600) + writes `references/birds/INDEX.md` (file page, author, licence, SHA-256 raw + crop) |
| `scripts/intro/render_hero_card.py` | hero card image = studio render of `assets/hero/pigeon_osfp_dr8.glb` (leash hidden) -> `assets/components/intro/photos/cyber_pigeon.png` |
| `scripts/intro/form_factor_models.py` | 15 bpy form-factor builders, dims table with source and grade (A/B/C/APPROX) |
| `scripts/intro/build_intro.py` | builds the set, hooks, preview camera, stills, preview mp4; API `append_intro`, `key_camera`, `offset_animation`, `FADE` |
| `scripts/intro/test_append.py` | smoke test / usage example: append at (0, -800, 0), intro starting at frame 300 |
| `assets/components/intro/intro_tree.blend` (+ `.json` sidecar: hooks, card rectangles, dims, fade) | collection `INTRO_tree` (240 objects) under empty `INTRO_root`; photos and font packed; scene `Scene` with preview camera `INTRO_cam` in collection `INTRO_preview` |
| `assets/components/intro/photos/*.jpg|png` | 14 crops + hero render |
| `references/birds/raw/*.jpg`, `references/birds/INDEX.md` | downloaded photos (git-ignored) and their provenance |
| `outputs/assets/intro/intro_tree_{start,mid,end}.png` | stills at frames 0, 68, 135 (540x674, Standard view transform) |
| `outputs/assets/intro/intro_tree_preview_20261010.mp4` | 540x674, 30 fps, 160 frames (0-159), H.264 yuv420p CRF 18 +faststart, EEVEE 8 samples |

Rebuild: `scripts/bslot.sh -b --factory-startup --python scripts/intro/build_intro.py` (about 40 s incl. renders; flags `--no-render`, `--stills-only`, `--agx`). Photos: `uv run --with pillow python -I scripts/intro/crop_cards.py`.

### 3.2 Set

- Vertical wall of cards (XZ plane, cards face -Y), root at the bottom, hero at the top; dark neutral backdrop (emission 0.016) 60 x 60 m; cards off-white emission (0.80), black text, so contrast is independent of the host scene's lighting.
- Card (1.0 x 1.42 m): 4:3 photo; species (black); form-factor bar coloured by branch (gold = lineage, magenta = coherent, copper = AEC, grey = other / CPO); dark lower band with the model; gold square badge with the lineage number (0-6); "EXTINCT" chip on extinct birds (theropod, Archaeopteryx, moa, elephant bird, passenger pigeon, dodo); "APPROX." where the model dimensions are approximate.
- Rows: 0 theropod, Archaeopteryx + title/legend; 1 Palaeognaths (moa, elephant bird, emu, ostrich); 2 Galloanserae (chicken) | Neoaves unresolved (hummingbird, albatross, parrot); 3 Columbaves (passenger pigeon, rock pigeon, cuckoo, dodo); 4 hero (1.5x card, red frame) inside Columbaves.
- Gold lineage routed in the gutters (behind the cards): 0 1x9 -> 1 GBIC -> 2 SFP -> 3 QSFP+ -> 4 QSFP28 -> 5 QSFP-DD and 5 OSFP (hero) -> 6 XPO (right margin down to the ostrich). Magenta: CFP -> CFP2-DCO -> 400ZR. Clade frames faint grey with labels; Neoaves outline encloses Columbaves.
- Models: one common scale (x4.6 real size), tilted 50 deg toward the camera, slow yaw +-18 deg over frames 0-159; one neutral satin grey body material + one dark material (latch, ports, cable); no logos. Three area lights inside the collection with a 40 m cutoff.

### 3.3 Form-factor dimensions (W x H x L, mm)

| Model | Dims | Source | Grade |
|---|---|---|---|
| OSFP (hero) | 22.58 x 13.00 x 100.40 | OSFP MSA Rev 5.2 Fig 3-2 (`research/hero_800g_osfp_dr8.md`); fins drawn open-top for readability | A |
| QSFP+ / QSFP28 / compatible | 18.35 x 8.5 x 72.4 (nose 19 x 13.5 x 20 max) | QSFP112 MSA Rev 2.1.2 Fig 12 + sec 6.2 ("Type 1 (same as QSFP28) 72.4 mm max") (`references/specs/`) | A |
| SFP | 13.7 x 8.95 x 56.5 | Phoenix Contact SFP module datasheet (phoenixcontact.com/us/products/2891754/pdf), vendor | B |
| CFP | 82 x 13.6 x 144.8 | Cisco 100GBASE CFP data sheet c78-633027; 13.6 mm height also in CFP MSA Rev 1.4 revision history (approvednetworks.com/content/msa/CFP_MSA.pdf, read 2026-10-10) | B |
| XENPAK | 36 x 18 x 121 | Cisco XENPAK-10GB-ZR product listing (icecat); other listings 35.6 x 12.7 x 122 (CX4) | C |
| GBIC | 30 x 12 x 65 | listings disagree (ZyXEL 65 x 30 x 10; D-Link ~57 x 30 x 12) | APPROX |
| QSFP-DD, 400ZR | 18.35 x 8.5 x 89 | width/height from the QSFP cage; length not in project research; 400ZR drawn as a QSFP-DD body with a nose heatsink (stylized) | APPROX |
| CFP2-DCO | 41.5 x 12.4 x 107.5 | not verified (CFP2 MSA hardware spec not fetched) | APPROX |
| XPO | 45 x 16 x 120 + cold plate + 2 coolant fittings | "about two OSFP footprints" [2nd], integrated liquid cold plate (research B); no primary dimensions | APPROX |
| 1x9 | 25 x 10 x 39 + 9 pins + SC snouts | no spec found | APPROX |
| AEC | QSFP-DD-like plug + cable | stylized | APPROX |
| CPO (dodo) | 34 x 34 substrate, die + 4 engines + fiber stub | stylized | APPROX |

### 3.4 Photos (details, SHA-256: `references/birds/INDEX.md`)

| Leaf | Commons file | Author | Licence |
|---|---|---|---|
| Theropod | Milwaukee Public Museum November 2022 019 (Exploring Life on Earth--Feathered Dromaeosaur).jpg | Michael Barera | CC BY-SA 4.0 |
| Archaeopteryx | Berlin Archaeopteryx.jpg | Emily Willoughby | CC BY-SA 4.0 |
| Moa | Dinornis robustus skeleton.jpg | Mike Dickison | CC BY-SA 2.0 |
| Elephant bird | Elephant Bird skeleton.jpg | shankar s. | CC BY 2.0 |
| Emu | Emu Side.JPG | Quartl | CC BY-SA 3.0 |
| Ostrich | Struthio camelus - male - Rulzheim.jpg | H. Zell | CC BY-SA 3.0 |
| Chicken | Rooster - Gallus gallus domesticus.jpg | MrPanyGoff | CC BY-SA 3.0 |
| Hummingbird | Archilochus colubris-male hovering.jpg | Blake Ross | CC BY 4.0 |
| Albatross | Diomedea exulans - SE Tasmania.jpg | JJ Harrison | CC BY-SA 3.0 |
| Parrot | Scarlet macaw (Ara macao cyanopterus) Copan.jpg | Charles J. Sharp | CC BY-SA 4.0 |
| Cuckoo | 20150419 025 Kessel Weerdbeemden koekoek ... (17014010968).jpg | Jac. Janssen | CC BY 2.0 |
| Passenger pigeon | Ectopistes migratorius ULaval 2.jpg (taxidermy) | Cephas | CC BY-SA 3.0 |
| Rock pigeon | Feral pigeon (Columba livia domestica), 2017-05-27.jpg (same file as `references/figures/`) | Lewis Hulbert | CC BY-SA 4.0 |
| Dodo | After Roelant Savery - The Dodo.jpg (painting) | Roelant Savery | Public domain |

### 3.5 API for S

```python
import sys; sys.path.insert(0, "scripts/intro")    # relative to the project root
import build_intro as BI
root = BI.append_intro(location=(0.0, -800.0, 0.0))  # appends INTRO_tree; returns INTRO_root (move/scale it freely)
BI.offset_animation(root, f0)                        # only if the intro does not start at frame 0 (model yaw keys)
BI.key_camera(cam, root, f_start=f0)                 # keys cam + Track-To target: start f0, a f0+45, b f0+92, end f0+135 (hero)
BI.FADE  # {'camera_move': (0, 135), 'hold_on_hero': (135, 159), 'cross_dissolve': (135, 159), ...}
```

- Hooks (children of `INTRO_root`): `HOOK_intro_cam_{start,a,b,end}` (camera positions), `HOOK_intro_tgt_{start,a,b,end}` (look-at points); custom props `lens_mm` (50, sensor fit horizontal 36 mm) and `frame`. End = hero card centred in the top 82% of a 4:5 frame.
- Recommended fade: camera move 0.0-4.5 s (frames 0-135), hold on the hero card, cross-dissolve to the street 4.5-5.3 s (frames 135-159). Render the intro as its own pass (its own camera, Standard view transform) and dissolve in the compositor or ffmpeg (`xfade=transition=fade:duration=0.8:offset=4.5`), not overlaid in the street. Under the film's AgX the emission colours get flatter and less saturated (checked with `--agx` stills, 2026-10-10); a separate Standard pass keeps the intended contrast.
- Smoke test: `scripts/bslot.sh -b --factory-startup --python scripts/intro/test_append.py -- <out_dir>` (verified 2026-10-10: 240 objects appended, camera keyed at frames 300/435, start and end frames match the previews).

### 3.6 v3 update (DevLog-018 wave, 2026-10-10)

Owner (verbatim, via coordinator): "initial tree is good. Remove actual pictures of the real pluggables, and just use bird pictures, bird names, and pluggable names."

| Change | Detail |
|---|---|
| Models removed | No form-factor model objects on any card (incl. the hero's OSFP); no model band, no "APPROX." tags; no lights left in `INTRO_tree` (all emission). `form_factor_models.py` kept in the repo but no longer imported; sections 3.2 (model bullets) and 3.3 are history |
| Card re-balanced | Card 1.0 x 1.30 m (was 1.42): 4:3 photo 0.92 x 0.69 at the top; bird name below it (auto-fit, max 0.12 m); form-factor tag bar 0.92 x 0.19 m with text up to 0.115 m (was 0.115 m bar, 0.082 m text); lineage badge 0.19 m; EXTINCT chip 0.32 x 0.09 m. Hero card 1.5x with red frame, photo = hero GLB render |
| Kept | Lineage numbers, gold lineage lines, magenta coherent branch, clade frames, EXTINCT tags, title; legend lines enlarged (0.054 -> 0.064 m), "MODELS: ONE SCALE" removed |
| Camera | Hook names unchanged (`HOOK_intro_cam/tgt_{start,a,b,end}`); hook z positions now scale with the row pitch (1.92 m, was 2.04 m), so values differ slightly; `key_camera` reads them from the blend, no code change for S. `offset_animation` kept (no keys left; no-op). `FADE` unchanged (move 0-135, dissolve 135-159) |
| OUNOLIGHT rename | "OUTNOLIGHT" does not appear in any intro file (checked with grep); nothing to rename |
| Renders | 540x674 stills + mp4 and 1080x1350 stills (`intro_tree_{start,mid,end}_1080x1350.png`); the saved blend's preview scene is set to 1080x1350 (last build). Rebuild at 540: `INTRO_W=540 INTRO_H=674` (default) |
| Checks | Stills at both resolutions and a 2.67 fps sheet of the mp4 viewed: all names and tags legible at 1080x1350; at 540x674 bird names about 10-14 px and tags about 10-13 px cap height ("PASSENGER PIGEON" smallest, auto-fit 0.078 m); append smoke test: 184 objects, end frame matches the preview (mean abs diff 0.41/255) |

## 4. Open issues

- Credits: 13 of 14 photos need attribution (CC BY / BY-SA); S7 credits card must list "<author> / Wikimedia Commons, <licence>" (INDEX.md). CC BY-SA crops are derivatives shared under the same licence.
- (v3) APPROX model dimensions no longer relevant (models removed).
- Legibility (v3): at 540x674 bird names 10-14 px and tags 10-13 px cap height, EXTINCT chip and legend about 7-8 px; at 1080x1350 double that.
- The moa, elephant bird and emu crops are letterboxed with the photo's own border colour (portrait sources); the elephant-bird photo includes a museum sign ("MAGNI... BIRDS").
- Ostrich photo is a head close-up (whole-body alternative: "Masai ostrich (Struthio camelus massaicus) male Serengeti.jpg", CC BY-SA 4.0, not downloaded).
- No years on screen; lane counts dropped from the labels (short labels per brief); lineage numbers 0-6 as in film v1.

## 5. TODO

- [x] Plan (this file)
- [x] Photos: 14 leaves from Commons, licences, SHA-256, INDEX.md
- [x] Card crops (consistent size)
- [x] Form-factor dims: primary sources where available, else APPROX
- [x] Models (bpy)
- [x] Intro set + hooks + preview camera + append API
- [x] Previews: stills + mp4; reviewed; iterated for legibility/contrast (4 rounds)
- [x] v3: models removed, cards re-balanced, previews at 540x674 and 1080x1350, append test
- [ ] S: integrate (append or separate pass), cross-dissolve, credits on S7
- [ ] Owner review

## 6. Progress log

- 2026-10-10 11:30: inputs read; film v1 0-5 s contact sheet checked (hologram tree over the street: thin cyan labels on a busy background, low contrast); plan written.
- 2026-10-10 11:25: Commons searches; HTTP 429 at 2.5 s spacing on the 8th search -> stopped, waited ~2 min, raised spacing to 6 s; no further errors.
- 2026-10-10 11:31: 14 photos downloaded (fetch log `outputs/logs/intro_fetch_20261010_112802.log`); cuckoo re-fetched at full resolution (bird small in frame), moa replaced by a full-skeleton photo (first pick showed only skull and neck).
- 2026-10-10 11:34: crops reviewed on a contact sheet (theropod head cut, portrait skeletons cut) -> fit-and-pad mode; hero card render (first pass overexposed, relit).
- 2026-10-10 11:39-11:44: set built; stills rounds 1-4: perspective skew and off-centre framing (lens 35 -> 50, smaller lift/side offsets), overexposed models (lights 2600 -> 520 W, satin material), title overflowing the frame (moved next to the root cards), models overlapping the form-factor bar (scale 5 -> 4.6, tilt 62 -> 50 deg, dark model band), path pulling back (3 -> 4 camera keys); AgX check stills (then deleted); final preview mp4 rendered and checked on a contact sheet.
- 2026-10-10 11:45: append smoke test at (0, -800, 0) with a frame-300 start: OK.
- 2026-10-10 11:55: v3 request (DevLog-018): models and APPROX tags removed, lights removed, card re-balanced (bigger name and tag), legend enlarged; built, previews rendered at 540x674 (stills + mp4) and 1080x1350 (stills), viewed; append smoke test OK. No "OUTNOLIGHT" in intro files.
