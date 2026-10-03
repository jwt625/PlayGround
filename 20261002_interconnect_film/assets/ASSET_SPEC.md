# Asset specification for the v1.0 model library

Project: `20261002_interconnect_film` (PlayGround). Date: 2026-10-02. Audience: every agent that builds an asset for the film. This is the shared contract; read it fully before building anything.

## 1. Goal

v1.0 replaces the crude primitives of v0.2 with models that are as accurate and complete as possible, organized as a reusable library for future similar videos (it will be spun off later). The storyboard is fixed and approved: `DevLog/DevLog-001-story-scenes-assets-proposal.md` Section 4 (rev 5) lists every scene, and `scripts/build_crude_film.py` shows what each scene needs and roughly where. Wentao will review modeling details after v1.0 and send feedback, so each asset must be documented well enough for targeted corrections.

## 2. Style

- Hardware (packages, boards, modules, cables, racks, equipment, fab tools): realistic geometry from real dimensions, realistic PBR materials (silicon, gold, copper, FR4, solder mask, mold compound, anodized metal, glass, plastics). Accurate small features where visible in a close-up (balls, pads, pins, traces, bond pads, rings, fiber cores).
- Characters and gag props (Gary, Manager, vendors, customers, shotgun, eggs): clay look (matte, subtle thumbprint bump, rounded forms), simple but expressive.
- No emoji anywhere (code, comments, names, docs).
- No real company logos or trademarks on any model. Parody marks only, drawn from scratch: shirt wordmarks MOLEXX, NUBISS, TERAHOP, AYARR (vendors), NVYDIA, OPENAY / ANTHROPY (customers); chip and sign brand NARROWCOM with a valley mark (an arch with the tip flipped upside down, over a wave line). Real part numbers or names of public products may be used as text labels in docs and metadata, not as logos on models. Public specs (MSA documents, standards) may be used for dimensions.
- Do not invent numbers. Every dimension that matters must come from a source (spec, datasheet, public teardown, photo scale, or the reference images in `references/`) or be flagged as an estimate with a range. Say so in the asset metadata.

## 3. Units, axes, origin

- 1 Blender unit = 1 m. Model at real size; use `C.mm()` for millimetres. The assembly step scales hardware up for macro shots; do not bake scene scale into assets.
- Z up, -Y is the front of the asset (the side a viewer faces), +X to the viewer's right. Apply scale and rotation (identity transforms on mesh objects).
- Origin of the root empty: bottom center of the asset's footprint (so it sits on z = 0), unless the asset is something that mounts onto another part (document the mounting origin in the metadata, e.g. an OE module's origin is the center of its underside).
- One top-level collection per asset named `ASSET_<asset_id>`, with a root empty `ROOT_<asset_id>`; all parts parented under the root. Sub-collections allowed for variants (`VARIANT_<name>`) and detail levels (`LOD_high`, `LOD_low` if you build them).

## 4. Files and folders

For a category `<cat>` (lowercase, e.g. `characters`):

- Blend files: `assets/components/<cat>/<asset_id>.blend` (one .blend per asset or tightly coupled family; a family may hold several `ASSET_` collections, document it).
- Build scripts: `scripts/assets/<cat>/build_<asset_id>.py`, runnable headless and deterministic: `/Applications/Blender.app/Contents/MacOS/Blender -b --python scripts/assets/<cat>/build_<asset_id>.py -- assets/components/<cat>`. No downloads inside build scripts; the blend must open with no missing external files (procedural textures or small packed images only).
- Metadata: `assets/components/<cat>/<asset_id>.json` (written by `C.finish`, then extended by you): sources (URL or file path, access date, what it was used for), dimension table with provenance and accuracy level per item, simplifications, animation hooks, material slot names, custom properties, intended scene usage.
- Previews: `assets/components/<cat>/previews/<asset_id>_<view>.png`, 900x675, EEVEE, at least front, three-quarter and top; add close-ups for small features (keep each PNG under about 400 KB, total per category under 25 MB).
- Category devlog: `DevLog/v1/DevLog-003-<cat>.md` with your plan, checklist of items with status, timestamped progress log, sources, decisions, deviations, open questions for Wentao, and an audit-corrections section. Update it as you go; the coordinator reads it for the master tracker. Only edit your own files.
- Reference material you download or extract: `references/<cat>/` (small files only; add a metadata sidecar with url, access date, SHA-256; never commit or redistribute vendor PDFs beyond Wentao's own provided files).

## 5. Naming and hooks

- Objects: `<asset_id>_<part>` (lowercase, underscores). Materials: `MAT_<cat>_<name>`. Node groups: `NG_<name>`.
- Animation hooks are empties named `HOOK_<name>` parented to the part they drive (examples: `HOOK_fiber_attach`, `HOOK_muzzle`, `HOOK_head_top`). List every hook in the metadata.
- Custom properties on the root for anything the assembler may drive: documented in the metadata and prefixed `p_` (e.g. `p_heat`, `p_hole_1_radius`).
- Parts that must be animated or recolored individually (individual chips, rings, pins) get their own object and their own material slot, not a shared material.
- Image-sequence screens (oscilloscope, whiteboard): a plane object named `<asset_id>_screen` with a material `MAT_<cat>_screen` whose emission shader has an image texture node left empty (the assembler plugs in the sequence); UV 0..1 over the visible area.

## 6. Quality and size

- EEVEE Next friendly: instancing (linked duplicates or Geometry Nodes instances) for arrays of identical parts (balls, pads, pins, MLCCs, cables); target under 600k triangles per asset in the high-detail state; provide a reduced variant if larger.
- Bevels on visible hard edges, correct normals, no n-gons on curved parts, no overlapping coplanar faces.
- Check every asset visually: render the previews and look at them (Read the PNGs) before reporting; fix what looks wrong (scale, missing parts, flipped normals, floating parts, z-fighting).
- Compare against the reference images/dimensions you used; list what still differs.

## 7. Machine and process rules

- Blender 4.2.3 at `/Applications/Blender.app`. Headless only (`-b`). Use Blender's bundled Python and numpy; do not use pip, conda or the system Python for Blender work. Shell Python (`python3`) is fine for simple file text edits.
- Run one Blender process at a time per agent, quit it when done, never kill processes you did not start. Previews: EEVEE, 16 samples, 900x675.
- Disk: about 12 GiB free on the system volume, shared by all agents. Keep each category under about 300 MB total; write temporary files only to your session scratchpad directory (given in your prompt), never `/tmp` directly.
- Never delete or overwrite files you did not create. Do not touch other categories' folders or the storyboard scripts (`scripts/build_crude_film.py`, `scripts/blender_lib.py`, `scripts/tex_gen.py`, `assets/generated_textures/`, `scenes/`, `outputs/`).
- Do not run git commands. Do not commit.
- Web research: primary sources first (MSA specifications, standards task-force slides, manufacturer datasheets, public teardowns); few requests, small waits, stop on any anti-bot signal; no bulk downloads; cache only small files with metadata. Wentao's local corpora are valid sources: blog images at `/Users/wentaojiang/Documents/GitHub/jwt625.github.io/assets/images/` (his own teardown photos of Innolight/Intel/Cisco modules in `2025/20251219_CPO/`, `2025/20250117_20250127_long_thread/`, `2025/20250425_20250425_long_thread/`, `2025/20250913_20250913_long_thread/`), OFC 2026 slide photos in `2026/OFC2026/`, the OFC/ISSCC/IEEE caches under `/Users/wentaojiang/Documents/GitHub/PlayGround/20260320_OFC`, `20260614_ISSCC_2026`, `20260521_ieee802.3dj`, his 3D Gaussian-splat captures and photos in `/Users/wentaojiang/Documents/3DGS/` (innolight, intel100G; PLY splats cannot be imported into Blender directly, but their photo sets are valid references).
- Dates are ISO (YYYY-MM-DD). Log files get timestamps in their names.
- You cannot ask Wentao questions during the build; record open questions in your category devlog and your final report, choose a sensible default, and say which.

## 8. Shared helper module

`scripts/assets/_common/common.py` (import as shown in its docstring) provides: `reset()`, `new_asset()`, `add()`, `sub_collection()`, `hook()`, `principled()`, `assign()`, primitives (`cube`, `cylinder`, `sphere`, `text_mesh`), `bevel()`, `join()`, `bbox_mm()`, `count_tris()`, `write_meta()`, `save()`, `preview()` and `finish()` (save + JSON metadata + previews). Extend it only by adding your own helpers inside your category folder; do not edit the shared module (tell the coordinator if you need a change).

## 9. Final report (your last message)

Return: a table of assets built (id, file, size in mm, triangles, accuracy level A/B/C), key sources, what is simplified or missing, animation hooks and custom properties, anything the assembler must know (scale, origin, materials to plug), open questions for Wentao, and the paths of your devlog and previews. Keep it under about 900 words; details belong in the devlog and metadata.
