# DevLog-003: materials_vfx (materials, shaders, effects, lighting, render presets)

Owner: materials_vfx agent. Date: 2026-10-02. Blender 4.2.3, EEVEE Next. All scripts in `scripts/assets/materials_vfx/`, outputs in `assets/components/materials_vfx/`.

## Plan
Shared lib `vfx_lib.py` (node-group expression DSL, drivers, preview helpers). One build script per asset; final `build_library.py` merges node groups/materials/collections into `materials_vfx_library.blend`.

## Checklist
- [x] materials_clay (NG_clay + 47 MAT_vfx_clay_*)  built, previewed (2026-10-02)
- [x] materials_pbr_hardware (NG_pbr_surface/silicon/led/fiber_core + ~60 MAT_vfx_*)  built, previewed (2026-10-02)
- [x] shader_heat_wave (NG_heat_wave, NG_heat_glow)  built, contact sheet (2026-10-02)
- [x] shader_bullet_hole_note (NG_cutout_tube)  built, preview (2026-10-02)
- [x] shader_egg_fry  built, 6-frame contact sheet verified (2026-10-02); p_cook via root property + drivers (bind_egg)
- [x] vfx_particles_and_effects  15 effects built (smoke_puff, steam, ear_steam, sparks_burst, dust_cloud, impact_stars, muzzle_flash, smoke_ring, shockwave, confetti, egg_splatter, glow_trail, motion_streaks, heat_shimmer, bang_starburst), contact sheets checked (2026-10-02)
- [x] lighting_and_world  5 rigs (daylight, macro_studio, data_hall, lab, hot_chip_rim), 5 worlds, NG_comp_post (bloom/vignette/CA/grain), DOF rig; contact sheet checked (2026-10-02)
- [x] render_presets + benchmark  module render_presets.py + render_presets.blend; measured 1080x1350 per-frame (M4 Pro, Blender 4.2.3 EEVEE Next Metal, headless): draft 8spp 0.43 s; standard 24spp 1.7 s (1.75 s with comp); hero 48spp+ray tracing+motion blur 3.5 s (4.2 s with comp); first frame +6-9 s shader compile (2026-10-02)
- [x] library merge (materials_vfx_library.blend: 26 node groups, 136 materials, 21 collections, 5 worlds; append test OK, no missing files); audit skipped per coordinator (2026-10-02)

## Decisions
- Colours authored in sRGB hex, converted to linear (`V.srgb`).
- Hardware material micro-textures use Object coordinates (needs applied scale per spec); frequencies per metre.
- Heat/egg shaders expose spec-named sockets (p_wave_phase, p_wave_speed, p_wave_k, p_heat, p_cook).
- Contact sheets are larger than 900x675 when tiled (documented deviation).

## Progress log
- 2026-10-02: lib + clay + PBR + heat wave + cutout built and visually checked. Run cut off by API limit; resumed.
- 2026-10-02: egg fry verified (white glassy->opaque, rim browning, yolk firming); heat wave and cutout previews re-rendered and checked.
- 2026-10-02: effects: stateless GN particles (closed-form in time), driven by root p_t; heat shimmer true refraction dropped (renders dark on one-sided plane), haze streaks only.
- 2026-10-02: lighting_and_world built; world uses raw direction z; comp vignette blur factor is in percent.
- 2026-10-02: render_presets built and benchmarked. 1860 frames at standard = about 54 min; at hero = about 2.2 h (single process).

## Sources
- Metal F0 values: typical Filament PBR reflectance table, recalled not re-fetched (level C). No other external sources; all looks are art-directed.
- Style reference: frames extracted from references/JE3LSo1Rp54.mp4 (saturated sky blue, glossy hardware, glow bursts).

## Deviations / known limits
- Contact sheets are tiled (e.g. 1350x676) rather than 900x675.
- Heat shimmer: true refraction does not work on a one-sided plane in EEVEE Next (renders dark); only the haze-streak plane is provided. A compositor Displace node is the suggested route for real distortion.
- Egg: white needs a point attribute `egg_rim` (radial fallback otherwise). Per-egg p_cook via bind_egg().
- Clay thumbprint fingerprint ridges are subtle and only visible in macro shots.
- Effects sizes are human scale; effects JSON has no triangle counts (particles are realized instances, a few thousand triangles).
- NG_comp_post grain uses the legacy Noise texture (not frame-animated unless Grain Seed is driven by frame).
- Fast GI / AO only in the 'hero' preset (ray tracing on).

## Open questions for Wentao
1. Which quality preset for the final render (standard 1.7 s/frame vs hero 3.5-4.2 s/frame)?
2. Glow style: more bloom (hero_glow) or the subtler cartoon comp preset?
3. Egg browning amount (Brown Extent 0.42) and frill: keep or push darker?

## Audit corrections
(skipped by coordinator instruction)
