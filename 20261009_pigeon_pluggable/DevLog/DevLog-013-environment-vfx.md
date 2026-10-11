# DevLog-013: Environment + VFX library (workstream E)

| Field | Value |
|---|---|
| Date | 2026-10-10 |
| Status | v4 W8 (DevLog-020: parody AI-era SF billboards, company signs, robotaxi) delivered 2026-10-10; v3 wave (DevLog-018: SF backdrop, satellites 0.5x / 2x speed, MIMO uplinks, street beam choreography) delivered 2026-10-10; interim items F4/F6/F7 (DevLog-015) delivered 2026-10-10; v2 fixes delivered 2026-10-10 (DevLog-010 section 9 items 9, 10, 13: hatchery motion + hop-off + append API, fast-tier puddles, close satellite, tree alternatives); v1 assets otherwise unchanged |
| Workstream | E (DevLog-010 section 3) |
| Owns | `scripts/env/`, `scripts/vfx/`, `assets/components/environment/`, `outputs/assets/environment/`, this file |
| Inputs | DevLog-010 (owner input, decisions P1-P6, interfaces, resource rules), DevLog-006 (storyboard rev 3: S1 street opening, S6 street ending, S5 hatchery), `scripts/build_crude_film.py` v0.2.3 (placements), frames of `outputs/crude_v0_2_3_20261010.mp4`, R's `bird_flap_vent` (DevLog-007), H's hero GLB (hatchery stage meshes) |
| Consumer | S (scene assembly, DevLog-011) |

## 1. Brief (coordinator, 2026-10-10, summary)

1. Street kit at production detail (true-scale metres, bpy generators, EEVEE-friendly, cyberpunk night, parody signage only): facades (windows, fire escapes, signage, parapets, rooftop AC units and antennas); the datacenter facade "DOVECOTE DC" (loading door, entrance, security camera, mount for R's bird flap vent); street furniture (lamp posts with arms, utility poles with crossarms and catenary wires, railings, bins, parked scooter, hanging sign, ledges, CCTV on poles/walls, puddles); sky with stars, light haze, satellite prop. Two layouts: `street_opening.blend` (in front of the DC facade, space for the hologram tree), `street_ending.blend` (natural scatter of perches near the current flock positions). Hooks `HOOK_perch_*` with facing and camera-relevant landmarks.
2. Hatchery "OUTNOLIGHT HATCHERY": conveyor, pick-and-place robots, inspection camera, stations adding PCB, DSP, optics, shell/feathers, eyes lighting up; seamless ~2 s loop.
3. VFX library `scripts/vfx/`: hologram, beam/link (fade keys), cheap volumetric haze, compositor glare/bloom + vignette + grain (Blender 4.2 EEVEE), wet ground with puddle masks, neon with flicker, eye glow for the lenslets.
4. Previews: both street layouts, hatchery, VFX test sheet; look, iterate once; API documented here.

## 2. Plan (written before implementation)

### 2.1 Coordinates and conventions

- World frame identical to `build_crude_film.py` so S drops the sets in without offsets: hall end wall (DC facade) at y = 13.0 (wall 12.94 ... 13.06), street along +y, road |x| < 3.5, sidewalks 3.5 < |x| < 6 (top z = 0.12), building fronts at |x| = 6, buildings 3.6 m wide at y = 15 + 4k (k = 0..4), heights from the crude `bldg_h`. The hall side (y < 12.94, z < 7.85) stays empty for S / R.
- Prop anchors copied from the crude constants (LAMPS, POLES, wire sag, CCTV, RAIL_X/Z, AC units, ledge, hanging sign, bin, scooter, roof edges, SAT0/SAT1); production props keep every crude perch surface height.
- Hooks: empties `HOOK_<name>`; perch hooks: location = bird feet origin, rotation Z = yaw (local +X = beak direction, as `bird_at`); custom props `kind`, `yaw_deg`, `crude_name`, `source`. JSON sidecar per blend lists every hook (location, facing vector, yaw).
- Bird flap: hole 0.7 x 0.5 m centred (0, 13.0, 3.25); `HOOK_bird_flap_vent` at (0, 12.94, 3.25), rotation Z = -90 deg = R's vent frame (origin on the inner wall face, +X into the hall); R's `ASSET_bird_flap_vent` appended under the hook, not rebuilt.
- Door 1.4 x 2.5 m at x = 0 (S1 walk-in, S6 walk-out); sliding leaves separate objects (open).
- Tree clear zone (opening): |x| < 1.3, 13.1 < y < 20.5, z < 4.6 (crude S1 camera, tree, subject path).

### 2.2 Files

| Path | Role |
|---|---|
| `scripts/vfx/__init__.py` | public API re-exports |
| `scripts/vfx/nodes.py` | node-graph helper `NB`, node groups, time driver, blended-material setup |
| `scripts/vfx/materials.py` | hologram, beam, neon, wet ground, puddle, eye glow + key helpers |
| `scripts/vfx/atmosphere.py` | night sky (world / dome), world-volume haze, haze cards, fake light cones |
| `scripts/vfx/post.py` | EEVEE presets, compositor bloom + vignette + grain (+ streaks, chroma) |
| `scripts/vfx/build_vfx_test.py` | VFX test sheet scene + render |
| `scripts/env/env_lib.py` | mesh builder MB (copied from R's `rk_lib.py`, + transform stack), material palette `M()`, hooks, text-to-mesh, catenary, lights, IO |
| `scripts/env/street_kit.py` | prop generators |
| `scripts/env/build_street.py` | `street_opening` / `street_ending` / `street_kit` blends + JSON |
| `scripts/env/build_hatchery.py` | `hatchery.blend` + JSON |
| `scripts/env/render_previews.py` | preview stills |

## 3. TODO

- [x] VFX library (hologram, beam, neon, wet ground, puddle, eye glow, haze world/cards, light cones, sky, compositor, EEVEE presets)
- [x] Street kit props (lamp post, utility pole + catenary wires, railing, wall/roof AC, antennas, dish, bin, hydrant, cones, vending machine, scooter, hanging sign, CCTV pole/wall, satellite, holo projector, neon/blade signs, puddles, buildings with windows/shops/awnings/fire escapes/parapets/roof clutter)
- [x] DC facade with R's flap vent on its hook, door, loading door, cameras, sign, louvres, roof coolers
- [x] Street layouts opening + ending with hooks and JSON; 21 crude perches reproduced (max delta 0.3 mm)
- [x] Hatchery with seamless 2 s loop (stage meshes from the hero GLB)
- [x] Previews (street x2, hatchery, VFX sheet); looked; 3 iteration rounds
- [x] API documentation (section 5)
- [ ] S integration feedback (scale/visibility windows/clearance against S's camera keys)
- [x] v4 W8 (DevLog-020): parody AI-era billboards (4 rooftop boards + 2 bus-shelter ads), company signs, robotaxi on both street blends (`env/billboards.py`); perch/landmark hooks unchanged (diffed); previews from the S6 camera keys at 540 x 674 + 1080 x 1350
- [x] v3 (DevLog-018): San Francisco backdrop in both street layouts (`env/sf_landmarks.py`), satellites 0.5x scale and 2x-speed pass (`CN.V3_SCALE`, `CN.S6_PASS_V3`), `vfx.mimo_links`, `vfx.street_beam_choreo`; previews from S's S6 camera keys at 540 x 674 (+ one 1080 x 1350)
- [x] Interim F4 HUD (`vfx/hud.py`), F6 constellation train (`env/constellation.py`) + beam track/sweep/hops (`vfx/beams.py`), F7 data emitters / data beams (`vfx/datafx.py`); test renders checked
- [x] v2 (DevLog-010 section 9 items 9/10/13): continuous conveyor + tracking robots, finished bird hops off, `hatchery_api.py` (append / shift / bake); fast-tier wet ground + puddles (switchable); close satellite pass 1.6x; tree projector collection + 5 alternative bases with clear-zone check

## 4. As built

### 4.1 Assets (`assets/components/environment/`)

| File | Content | Objects / tris |
|---|---|---|
| `street_opening.blend` + `.json` | root collection `ENV_street_opening` (children `ENV_ground`, `ENV_buildings`, `ENV_dc_facade`, `ENV_signs`, `ENV_furniture`, `ENV_skyline`, `ENV_sky`, `ENV_lights`, `ENV_vfx`, `ENV_hooks`, `EXT_R_bird_flap_vent`); holo projector at the tree base; `CLEAR_tree_zone` cube empty; 39 hooks (21 perch) | 222 / 177 k |
| `street_ending.blend` + `.json` | same street; 79 hooks (21 crude perches + 41 natural extras `HOOK_perch_x_*`) | 259 / 176 k |
| `street_kit.blend` + `.json` | each prop at the origin in `ASSET_env_<prop>` (lamp_post, utility_pole, railing_5m, ac_wall, bin, hydrant, scooter, hanging_sign, cctv_pole, cctv_wall, satellite, holo_projector, vending_machine, building_9m, neon_sign) | - |
| `hatchery.blend` + `.json` | root empty `hatchery_root`; collections `HATCH_set`, `HATCH_line`, `HATCH_lights`, `HATCH_hooks`; frames 1-60 loop | 176 / 566 k (shared meshes; internals hidden once the shell is on) |

Rebuild (one Blender process each, about 1-15 s):
```text
scripts/bslot.sh -b --factory-startup --python scripts/env/build_street.py -- opening|ending|kit [out.blend]
scripts/bslot.sh -b --factory-startup --python scripts/env/build_hatchery.py -- [out.blend] [loop_frames=60] [frame_start=1]
scripts/bslot.sh -b --factory-startup --python scripts/vfx/build_vfx_test.py -- [out.png] [frame]
MARKERS=0|1 TAG=<suffix> PREVIEW_FRAME=<f> scripts/bslot.sh -b <blend> --python scripts/env/render_previews.py -- opening|ending|hatchery [view ...]
```

### 4.2 Street details

- Buildings (10): plaster/concrete/tile walls, floor bands, 2 windows per upper floor (frame, mullion, transom, sill, glass quad with lit-room shader: dark 42 %, warm, cool, TV flicker, pink; half-drawn blinds), window AC units (30 %), ground-floor lit shopfront or roller shutter, sign band, awnings (L0, R2), fire escapes (L4, R4: slatted platforms, railings, stairs, drop ladder), parapet + coping (top = h + 0.1 at the facade line, crude roof-edge perch), roof clutter (condensers, yagi/whip antennas with beacon, dishes, water tanks, vents). Back walls close the 0.4 m gaps.
- Signage (parody only): flat neon BIT ERROR BAR, KARAOKE (flicker), NOODLES, PACKET INN, ARCADE, OPEN 24H; channel letters LINK UP (crude position) and NO SIGNAL (flicker; crude position lowered to z 3.9 to clear the AC_2 bracket); blade signs HOTEL, BAR, PHO, LIVE; hanging lightbox RAMEN / 24H (crude position, double-sided); DOVECOTE DC; end block OUTNOLIGHT, PIGEON POST, LATENCY LOUNGE; street name PIGEON ST; scooter plate OSFP 800. Each sign has a coloured spill light.
- DOVECOTE DC facade: cladding panels, plinth, slit control-room windows, LED band, door frame + canopy (LED edge, 2 downlights, dome camera), card reader, sliding leaves `dc_door_leaf_L/R` (props `open_x`, `closed_x`), flap hood + cyan guide LEDs, loading door (roller shutter, hazard jambs, hood, bollards, wall pack, DOCK 1, KEEP CLEAR box), exhaust louvre bank, two wall bullet cameras, plates DC-01 and AUTHORIZED BIRDS ONLY, roof dry coolers + mast with red beacon (roof slab y 10.4-12.94 at z 7.85-8.0, above the hall). R's vent world bbox after appending: x +-0.41, y 12.916-13.1, z 2.925-3.56 (its frame is the exterior trim; no duplicate trim built).
- Street: wet asphalt (VFX wet ground, puddle coverage 0.15), sidewalks with paving joints, curbs, drain grates, manholes, lane dashes (none at y < 19, the tree spot), zebra crossing near the end, DC forecourt + trench drain, 10 puddle decals; 3 lamp posts (cobra heads, spot lights, fake light cones); 2 utility poles (crossarm, pin insulators, step bolts, transformer on pole 0); catenary wires (3 main spans, telecom cable, cross-street wire to L1, service drops, DC drop, spans out of view) and a festoon of 13 warm bulbs across the street at z ~7.4-7.8; railing (crude span); 3 wall ACs (crude); ledge (crude); bin, scooter (crude); hydrant, vending machine, traffic cones; CCTV on pole + 2 wall (wall cameras moved 0.385 m off the wall, the crude centres were inside the wall); satellite at SAT0 with key light and blinking LED.
- Skyline: end block across a cross street (y 37.5-45, 16 m) with lit ground floor and windows, 26 far towers kept under 31 deg elevation from the street so the satellite (z 26, y 38) stays in open sky (checked in `street_ending_sky.png`), red beacons; sky dome radius 240 m + world `WORLD_night_street` (same shader), world haze density 0.0035, 4 haze cards along the street.

### 4.3 Hatchery details

- Line along +X: 9 pallet slots (positions -1..7, pitch 0.75 m; -1 and 7 inside strip-curtain tunnels), belt top z 0.85, belt stripe texture moving with the index; stations 0-4 each with a SCARA robot (2-link IK, J4 keeps the part aligned) and a feeder table holding the next part; station 5 = inspection gantry (camera, ring light, red laser sweep, PASS display) where the eyes light up; station 6 = finished bird; header beam with labels CHASSIS, PCB, DSP, OPTICS, SHELL + FEATHERS, EYES ON / QA; neon OUTNOLIGHT HATCHERY on the back wall; epoxy floor with walkway lines.
- Stage meshes from the hero GLB by part prefix (decimated): chassis = `leg_`, `foot_` (4.0 k faces); PCB = `pcb_`, `elec_`, `flex_`, `silk_`, `conn_`, `mark_` (9.0 k); DSP = `dsp_`, `thermal_` (5.2 k); optics = `tx_`, `rx_`, `clip_` (14.7 k, beam viz dropped); shell = `skin_`, `wing_`, `tail_fan`, `beak_`, `leash_`, eye bezels/iris rings (14.0 k); eyes TX/RX = lenslet arrays + iris glow rings with `VFX_EyeGlow` (red/cyan). Bird scale 2.6 (hall scale, = `S_BIRD`).
- Loop (frames in the loop): robot down 6, grip 8, up 11, over 21, down 25, release 26 (part appears on the bird, carried copy disappears), clear 29, home 50; conveyor index 30-56; eyes on 16-22 at station 5; PASS 22-56. Every F-curve has a Cycles modifier (REPEAT); slot k at the loop end equals slot k+1 at the start. Check: frame 1 vs frame 61 render PSNR 38.9 dB (difference = per-frame film grain).

### 4.4 Previews (`outputs/assets/environment/`, 540 x 674, EEVEE draft, AgX Punchy, compositor on)

| File | View |
|---|---|
| `street_opening_wide.png`, `_s1_start.png`, `_flap.png`, `_facade.png` | down the street to the facade; crude S1 start camera (tree absent); crude S1 5.2 s flap camera; three-quarter facade |
| `street_opening_plan_hooks.png` | top ortho with perch markers (preview-only magenta spheres + facing stub) |
| `street_ending_door_out.png`, `_crane.png`, `_perches.png`, `_wires.png`, `_sky.png` | S6 out of the door; crane; rail/vending/ledge side; wires + HOTEL; tilt to the satellite |
| `street_ending_crane_hooks.png`, `street_ending_plan_hooks.png` | perch markers (62) |
| `hatchery_front.png` (f16), `hatchery_station.png` (f16), `hatchery_eyes.png` (f24) | line overview; stations 0-1 with robots placing; inspection with lit eyes |
| `vfx_test_sheet.png` (960 x 600) | hologram tree, beams (full and 35 % fade), neon (steady + flicker), eye glow (TX, RX pulse, RX off), haze cards + light cone, wet ground + puddles |

## 5. API

Import inside Blender: `sys.path.insert(0, "<repo>/scripts")`, then `import vfx`; for the kit also `sys.path.insert(0, "<repo>/scripts/env")`, `import env_lib, street_kit`. All functions create data in the current .blend; node groups are created once and shared.

### 5.1 `vfx` materials (`scripts/vfx/materials.py`)

```python
TX_RED = (1.0, 0.08, 0.05); RX_CYAN = (0.1, 0.85, 1.0); HOLO_CYAN = (0.2, 0.9, 1.0)
BEAM_NOTE = "* beams invisible (1311 nm); rendered for viewing"   # the scene puts it on screen
hologram_material(name="VFX_holo", color=HOLO_CYAN, strength=3.0, base=0.3, rim_power=2.0, scan_density=45.0,
                  scan_speed=0.35, scan_mix=0.45, flicker=0.25, flicker_rate=5.0, glitch=0.4, alpha=1.0, seed=0.0) -> Material
    # additive (emission + transparent, BLENDED): fresnel rim, world-Z scanlines (lines/m), noise flicker, rare glitch bands;
    # base=1 for flat text/labels; fade with key_input(m, "Alpha", f, v) or per object with key_obj_alpha
beam_material(name="VFX_beam_tx", color=TX_RED, strength=8.0, core_power=2.0, flow_mix=0.4, flow_density=4.0,
              flow_speed=2.5, fade=1.0, seed=0.0) -> Material      # bright view-facing core, pulses travelling source -> target
make_beam(name, p0, p1, radius=0.004, material=None, color=TX_RED, collection=None, seg=8) -> Object
    # shared unit mesh along local +Z from p0, scaled to |p1 - p0|; not shadow casting; object colour alpha = fade
set_beam(o, p0, p1)                                   # re-aim
key_beam_endpoints(o, frame, p0, p1, interp="LINEAR")  # follow moving birds (key every few frames)
key_beam(o, f_on, f_off, ramp=6, grow=True, length=None)  # fade in (and grow from the source) / hold / fade out
neon_material(name, color, strength=12.0, flicker=0.0, flicker_rate=2.5, buzz=0.06, core=0.5, on=1.0, seed=0.0) -> Material
    # tube emission with white-hot core; flicker 0..1 = dropout amount (0.3 = failing sign); per-object random phase
wet_ground_material(name="VFX_wet_asphalt", base_color=(0.03, 0.032, 0.04), wetness=0.6, coverage=0.3, scale=0.35,
                    rough_dry=0.8, rough_wet=0.3, rough_puddle=0.03, grain=0.25, grain_scale=40.0, seed=0.0) -> Material
    # world-space (no UVs); group output "Puddle Mask" available; needs setup_eevee(raytracing=True) for reflections
puddle_material(name="VFX_puddle", tint=(0.012, 0.013, 0.016), roughness=0.02) -> Material   # decal meshes 2-5 mm up
eye_glow_material(name, color=TX_RED, strength=5.0, power=1.0, rim=0.5, pulse=0.0, pulse_rate=1.5) -> Material
    # dark coated glass when off; core + rim emission when on; effective power = power x object alpha
key_eye_power(o, frame, power, interp="LINEAR")       # per bird / per lens object (0 dark .. 1 lit)
key_obj_alpha(o, frame, value, interp="LINEAR")       # generic per-object fade read by every VFX group
key_input(m, group_input, frame, value, interp="LINEAR")  # key a VFX group input on a material
```

Node groups: `VFX_Hologram`, `VFX_Beam`, `VFX_Neon`, `VFX_WetGround`, `VFX_EyeGlow`, `VFX_NightSky`, `VFX_HazeCard`, `VFX_LightCone`. Time inside groups = Value node driven by `frame/30` (simple expression, runs without auto-exec).

### 5.2 `vfx` atmosphere and post (`atmosphere.py`, `post.py`)

```python
night_sky_world(scene=None, name="WORLD_night_street", strength=1.0, stars=1.0, moon=0.0, city_glow=(0.05, 0.022, 0.03),
                zenith=(0.0015, 0.002, 0.006), horizon=(0.012, 0.011, 0.022)) -> World
sky_dome(name="ENV_sky_dome", center=(0, 25, 0), radius=240.0, collection=None, stars=1.0, moon=0.0, strength=1.0,
         city_glow=(0.05, 0.022, 0.03)) -> Object      # for one-scene films with a neutral world; clip_end >= 300
world_haze(scene, density=0.004, color=(0.55, 0.6, 0.75), anisotropy=0.25, absorption=0.0) -> volume node
haze_cards(name, center, size, n=6, color=(0.5, 0.55, 0.7), density=0.05, normal_axis="Y", noise_scale=0.6, drift=0.08,
           height_falloff=1.2, collection=None, seed=0.0) -> [Object]   # additive mist planes, fade edge-on
light_cone(name, apex, direction=(0, 0, -1), length=4.0, angle_deg=55.0, color=(1.0, 0.78, 0.55), strength=0.25,
           collection=None, seg=24, dust=0.35) -> Object          # fake volumetric lamp shaft
setup_eevee(scene, preset="draft"|"review"|"final", raytracing=True, volumetrics=True, vol_start=0.1, vol_end=80.0,
            motion_blur=False, view_transform=None, look=None, exposure=None, gamma=None)
    # draft 12 / review 32 / final 64 TAA samples; screen-space ray tracing (reflections); colour management only if given
setup_compositor(scene, bloom=True, bloom_threshold=1.0, bloom_size=7, bloom_mix=0.0, streaks=0.0, vignette=0.28,
                 vignette_radius=0.92, grain=0.035, chroma=0.0) -> {name: node}
    # Render Layers -> Glare BLOOM (4.2) -> [Glare STREAKS add] -> vignette (blurred ellipse multiply) -> grain (NOISE texture
    # overlay, new pattern every frame) -> [lens dispersion] -> Composite + Viewer. Node names: bloom, streaks, vignette, grain, chroma
key_compositor(scene, node_name, input_name, frame, value)   # e.g. key_compositor(scn, "vignette", "Fac", 900, 0.4)
```

Previews use `view_transform="AgX", look="AgX - Punchy"`: neon keeps saturation with soft highlight roll-off; Standard clips (compared in the first VFX sheet round). S currently renders with Standard; the choice is S's (section 6).

### 5.3 Street kit (`scripts/env/street_kit.py`; all return dicts with the object/root and world points)

```python
lamp_post(c, loc, yaw, name="lamp", H=4.6, arm=1.05, street_sign=None)   # arm toward local +X; keys obj, head, top, arm(d)
utility_pole(c, loc, yaw=0.0, name="pole", H=6.4, arm_z=5.85, transformer=False)  # attach{-0.5, 0, 0.5, "telecom"}, arm_top(dx)
wires(c, spans, name="wires")                          # spans = [(p0, p1, sag_m, radius_m)], true catenary tubes
railing(c, x, y0, y1, H=1.0, name="railing")          # top-rail top = H + 0.025
ac_wall(c, center, yaw, name="ac")                     # 0.55 x 0.8 x 0.55, front local +X; top
bin_prop(c, loc), hydrant(c, loc), vending_machine(c, loc, yaw, name, label="DRINKS"), scooter(c, loc, yaw, name, plate)
hanging_sign(c, center, yaw, lines, neon="neon_orange", name, wall_off=0.6)   # top = center.z + 0.3
cctv(c, center, yaw, mount="pole"|"wall", name="cctv", tilt=0.0)           # lens, lens_dir, top
satellite(c, loc, name="satellite")                    # root (animate), dish
holo_projector(c, loc, name)                           # tree-base puck + upward cone; top
neon_sign(c, loc, yaw, text, neon="neon_mag", size=0.42, name, backer=True, border=True, light=None)  # text faces local -Y
blade_sign(c, loc, yaw, text, neon="neon_cyan", size=0.32, name)          # vertical, projects along local +X
puddles(c, specs=[(x, y, r, z)], name, seed)
building(c, base, yaw, width, depth, h, wall, rng, name, shop="lit"|"shutter"|..., fire_escape=(xa, xb), roof=[(kind, x, y, extra)],
         awning=False, skip_windows=(), win_ac=0.25, sidewalk_z=0.12)  # facade plane y = 0 faces local +y; perches list
window_grid_material(name, seed)                       # far skyline windows on box-UV metres
env_lib.M(name)   # palette: asphalt, sidewalk, puddle, concrete(_light), plaster_a/b/c, tile_dark, metal_dark, metal_paint, galv,
                  # rail, rust, wood, rubber, plastic_white/dark, ac_white, grille, glass_dark, win_warm/cool/tv/pink, shopfront,
                  # shutter, hazard, cladding(_light), paint_white/yellow, red_body, chrome, seat, bin_green, sign_back,
                  # lamp_lens(_cool), led_red (pulsing), led_green/cyan/warm, camera_white, lens_black, foil, solar, dish,
                  # fabric_awning, cable, insulator, transformer, curb, water_tank, neon_<mag|cyan|orange|red|green|violet|warm|white>
                  # and flickering neon_mag_fl / neon_cyan_fl / neon_orange_fl
env_lib.catenary(p0, p1, sag, n=24), env_lib.catenary_z(p0, p1, sag, u), env_lib.text(...), env_lib.MB (mesh builder)
```

### 5.4 Hooks (world coordinates; full list with facing in the JSON sidecars)

| Hook | Where | Notes |
|---|---|---|
| `HOOK_perch_<crude name>` | 21 spots = crude `street_perches()` + `FLYER_PERCHES` (`lamp1_top`, `lamp1_arm`, `lamp2_arm`, `wire_18p2`, `wire_19p0`, `wire_20p7`, `rail_a`, `rail_b`, `ac_1`, `ac_2`, `ledge_a`, `ledge_b`, `sign`, `bin`, `scooter`, `roof_L`, `roof_R`, `cctv_pole`, `hero`, `comp0`, `comp1`) | both layouts; max deviation from the crude coordinates 0.3 mm (`.` -> `p` in names, `crude_name` keeps the original); yaw = crude yaw |
| `HOOK_perch_x_*` | 41 natural extras (ending): lamp tops/arms, crossarms, more wire points, ac_3, hydrant, vending top, cctv_2, scooter bar, DC canopy and dock hood, rail post, window AC tops, fire-escape rails, parapets, roof condensers, tank top, blade-sign tops | facing toward the street |
| `HOOK_bird_flap_vent` | (0, 12.94, 3.25), rot Z -90 deg | R's vent frame; R's objects parented to it |
| `HOOK_flap_outside`, `HOOK_door_dc`, `HOOK_door_dc_inside`, `HOOK_loading_door`, `HOOK_dc_sign` | facade landmarks | +X out to the street (inside: into the hall) |
| `HOOK_tree_base` (opening) | (0, 17, 0.07) on the projector | crude `TREE_C`; plus `CLEAR_tree_zone` empty |
| `HOOK_lamp_<i>_head`, `HOOK_cctv_<i>_lens`, `HOOK_cctv_dc_L/R_lens` | light heads, camera lenses | lens facing = optical axis (S6 CCTV links) |
| `HOOK_satellite`, `HOOK_satellite_pass_end`, `HOOK_satellite_dish`, `HOOK_satellite_far` (ending) | v2: pass start (-5, 27, 17) -> end (1, 28.8, 19), scale 1.6, collection `ENV_satellite`; far = v1 SAT0 | see section 7; the opening has no satellite since v2 |
| `HOOK_street_vista` | (0, 33, 1.6) | end-of-street look target |
| Hatchery (local to `hatchery_root`): `HOOK_hatchery_sign`, `HOOK_station_<chassis|pcb|dsp|optics|shell>`, `HOOK_inspection_camera`, `HOOK_eyes_on`, `HOOK_exit`, `HOOK_cam_overview` (prop `target`) | | |

## 6. Open issues / for S and the critic

1. Colour management: previews use AgX Punchy; S's crude build uses Standard (neon and lamps clip to flat colour under Standard). Recommend AgX Punchy (or Medium High Contrast) film-wide; emission strengths were tuned for AgX.
2. Visibility: S uses one continuous scene with time windows; the street root collection has 222/259 objects and about 30 lights (each sign has a spill light). Windowing per collection (`ENV_lights` included) is cheapest; the world (`WORLD_night_street` + world haze) is global, so in the hall S should use `sky_dome` + haze cards instead of the world haze, or key the world volume density.
3. Camera clearance: props were placed from the crude anchors, but new items (festoon wire z 7.4-7.8 across y 24.6-26, DC canopy to y 14.26 at z 2.65-2.75, flap hood to y 13.4 at z 3.66, bollards, vending machine, cones) need S's clearance checker against the final camera keys. The S1 flap pass (camera z 3.2-3.5 at y 12.35-14.0) clears the canopy (top 2.75) and runs under the hood edge (z ~3.6 at y 13.06-13.4): verify.
4. NO SIGNAL lowered to z 3.9 (crude 4.2); wall CCTV 1 and 2 centres moved 0.385 m off the wall (crude centres were inside the wall); hooks give the new lens positions.
5. Hatchery scale: birds at hall scale 2.6 (0.32 m); robots and feeders are sized for that. Triangle count 566 k counted per object (shared meshes); internals are hidden once the shell is on. The PASS display sits above the `eyes` preview frame.
6. Puddle decals read as dark holes where they reflect only the sky; they read well where neon is in the reflection. Remove or move individual puddles per shot if needed (`puddles` object, 10 blobs).
7. Hologram tree content is S's; the library gives the material and the projector. Hologram on thin geometry reads mostly as rim; use `base` 0.5-1.0 for leaves/labels.
8. Not done: animated rain, steam, flying litter; satellite motion (S keys it); traffic.

## 7. v2 fixes (coordinator work order, DevLog-010 section 9 items 9, 10, 13)

### 7.1 Hatchery: motion that survives the append, hop-off, conveyor always moving

- Diagnosis: read-only probe of `scenes/film_v1.blend` (`outputs/logs/E_probe_film_hatchery_*.log`): after S's append, the robot joints, quills and pallets animate at the S5 frames 1441-1491 (actions + Cycles modifiers intact; S's hide keys sit on extra F-curves). The "static" read came from the v1 indexing conveyor (pallets still for half the loop) and small arm moves in a 1.5 s wide shot.
- v2 loop (60 frames, frame 1 start):
  - The conveyor moves continuously (one pitch per loop, linear; belt texture in step).
  - Robot bases are shifted +0.31 m so the arm can reach the moving pallet. Arms key every frame from 21 to 29, so the part lands on the pallet while it moves.
  - The QA gantry sits where the bird is during the eyes-on window (f16-22). The PASS display hangs from the gantry, off the hop path.
  - Finished bird (`hopper`, full shell + lit eyes) leaves the pallet at the line end. Hop 1 is f4-13 onto the DISPATCH ledge at (2.5, -0.62, 0.62). Hop 2 is f19-28 to the floor. Hops 3-4 run f33-41 and f45-53 toward the camera-right, out of frame. It is hidden from f54 and back on the arriving pallet at the loop start.
  - Slot 6 parts are hidden, because the hopper is that bird.
  - New hooks: `HOOK_hop_start`, `HOOK_dispatch_ledge`, `HOOK_cam_hop` (camera (1.45, -2.7, 1.35) looking at (2.55, -0.7, 0.65)).
- How S appends it (either way keeps the loop):
  1. S's current `append_hatchery()` (load the `HATCH_*` collections; the root comes along as the parent) works unchanged. Do not clear or replace the objects' actions; S's visibility keys may be added to the same actions.
  2. `scripts/env/hatchery_api.py`:
     - `append_hatchery(path, loc, rot_z=0, collection=None)` loads the whole `ENV_hatchery` collection and places `hatchery_root`.
     - `shift_loop(objs, offset)` and `phase_offset(event, film_frame)` put an event on a film frame. Events: robot_down 6, grip 8, place 25, release 26, eyes_on 16, eyes_full 22, qa_pass 22, hop1 4, ledge 13, hop2 19, floor 28, hop3 33, hop4 45, hopper_gone 54. Example: `shift_loop(objs, phase_offset("hop1", F(4, 8.4)))`.
     - `bake_hatchery_loop(objs, f0, f1)` writes explicit per-frame keys and removes the modifiers. It also covers the belt material's node-tree action, and leaves hide_render / hide_viewport alone.
- Previews (fast tier):
  - `hatchery_front_f1/f21/f41.png`: robot poses differ.
  - `hatchery_hop_f9/f16/f24.png`: mid-hop, on the ledge, hopping down.
  - `hatchery_eyes.png` (f21) and `hatchery_station.png` (f21).

### 7.2 Fast-tier wet ground and puddles (switchable)

- Cause of "flat magenta polygons": without ray tracing the near-mirror puddles show the coloured AREA spill lights as large specular blobs.
- `VFX_WetGround` and the new `VFX_Puddle` group have a `Fast` input (1 = fast tier, the default in the sets). Fast mode does the following:
  - Gloss but no mirror: roughness >= 0.3 (puddles 0.32) and specular 0.15-0.2, so there are no light blobs.
  - A probe-free fake reflection: colour graded across x from `Refl A` (magenta, -x side) to `Refl B` (cyan, +x side) around `Refl Split X`, plus warm `Refl C` patches along y.
  - Streaks stretched along the street axis, a fresnel term (grazing views reflect more) and a slow drift.
  - Defaults: `Refl Strength` 0.35 on the ground (puddle mask + 20 % of damp patches) and x1.4 on the decals.
- API: `wet_ground_material(..., fast=True, refl_strength=0.35, refl_a, refl_b, refl_c)` and `puddle_material(..., fast=True, refl_strength=0.35, refl_a, refl_b, refl_c, split_x=0.0, width=7.0)`. `vfx.set_fast_reflections(on=True, materials=None)` switches every VFX ground/puddle material; use `on=False` with `setup_eevee(raytracing=True)`.
- Previews `street_ending_puddles_low.png`, `street_opening_wide.png`, `street_ending_door_out.png` (fast tier: 1 sample, no ray tracing, shadows or compositor) show magenta/cyan wet streaks instead of flat polygons. `render_previews.py` gained `FAST=1`.

### 7.3 Satellite

- `street_kit.satellite(c, loc, name, scale=1.0, yaw=0.0)`.
- The ending set now has the satellite in its own collection `ENV_satellite`. It is the 1.6x prop (wingspan about 10 m) at the start of a close pass over the street, `HOOK_satellite` (-5, 27, 17) to `HOOK_satellite_pass_end` (1, 28.8, 19). The pass runs through S's S6 6.5 s roof-camera view ((-4.3, 18.3, 8.3) looking at (-2, 26, 16), lens 22).
- To animate, key the instancer of `ENV_satellite` from (0, 0, 0) to `pass_end - location`. `HOOK_satellite_dish` is the dish at the pass start; move it with the same offset.
- The v1 far position is kept as `HOOK_satellite_far` (SAT0); a satellite there reads about 20 px.
- Preview `street_ending_sat_roofcam.png` (from S's camera): the satellite spans about 40 % of the frame width.

### 7.4 Tree area

- The projector and its cone are in their own collection `ENV_tree_projector` (opening), built at the crude base (0, 17). To move it, instance that collection with location = new base - (0, 17, 0). The `offset` property on each alternative hook holds that vector.
- Alternative hooks `HOOK_tree_base_<alt>` were each checked against every set mesh AABB in the zone |dx| < 1.3, |dy| < 1.6, 0.05 < z < 4.6. The results are in the JSON (`tree_zones`, and `clear_hits` on each hook):

| Alt | Base | Clear-zone hits |
|---|---|---|
| crude | (0, 17) | none |
| `y18p5` | (0, 18.5) | none |
| `y19p0` | (0, 19.0) | none |
| `xp1p5` | (1.5, 17) | none |
| `xm1p5_y18p5` | (-1.5, 18.5) | none |
| `xm1p5` | (-1.5, 17) | lamp_0 head/arm, scooter |

- "1.5 m off the DC sign": the sign is at (0, 13.13, 4.55), and the crude tree at y = 17 is already 3.9 m from the sign plane. A base 1.5 m further out from the crude position is `y18p5`; a 1.5 m sideways shift is `xp1p5`. Both are clear.
- `CLEAR_tree_zone` still marks the crude zone. The S1 camera / subject corridor (|x| < 1.3, y 13.1-20.5) is unchanged and clear.

## 8. Owner interim items F4, F6, F7 (DevLog-015)

### 8.1 F4: POV surveillance HUD (`scripts/vfx/hud.py`)

```python
vfx.hud_overlay(scene, cam, target_obj, f0, f1, style="intel"|"stealth", seed=0, blips=None, card=None, ticker=None,
                target_name="GARY", step=2, lock_at=None, every=4, collection=None, radar_range=8.0,
                feed_label="RX-L // SURVEILLANCE", alert_text="ALERT // SUBJECT IN VIEW", typing_cps=45.0, scanlines=True)
    -> [holder, *objects]
```

- **Rendering:** everything is emission/transparent geometry and text curves on `HUD_holder`, an empty parented to `cam` 5 cm in front of the lens (camera `clip_start` must be < 0.05). The holder scale is driven by the camera lens and sensor width, so the layout holds through zooms; it assumes `sensor_fit` HORIZONTAL, as in S's camera.
  - HUD units: x is -1..1 across the frame width; y is -A..A, with A = res_y / res_x.
  - The bottom 18 % of the frame stays empty; target brackets are clamped above it.
  - All HUD objects are hidden outside [f0, f1] with hide_render keys.
- **Elements:**
  - Boot-up flicker.
  - Target brackets: a large acquire box shrinks onto the target's projected bounding box, then turns into red LOCK corners at `lock_at` (default f0 + 14), with a "!" flash and a blinking LOCKED tag.
  - TGT label that types in.
  - TARGET TRACK panel: RNG / BRG / ELV / ALT values measured from the scene every `every` frames.
  - Heading compass tape (clipped, follows the camera yaw) with an HDG readout.
  - TARGET PROFILE card that types line by line after the lock. Default parody lines: GARY / COFFEE ONLY / CAFFEINATED / HUMAN (PROB.) / 96 BPM HIGH / OBSERVED; override with `card=`.
  - Heading-up proximity radar: rotating sweep, bird blips from `blips` (objects or points), red target blip, contact count.
  - Link bars and SNR readout, REC dot and clock, a scrolling telemetry ticker (clipped), a blinking alert banner, and a scanline + vignette multiply layer.
- **Text and font:** text is keyed rather than edited per frame. Typing is done by revealing along the text's Generated X using object colour R; changing values are swapped through visibility windows of separate text objects.
  - Font: system Courier New Bold, scaled 1.3x.
  - Andale Mono and SF Mono drop whole glyph stems (B, H, L) in Blender's text fill because their glyph contours overlap; this was seen in the test renders.
- **Cost:** about 150-250 objects for a 4 s window. Values are sampled with `frame_set` every `step` frames, so in the full film call it once for the S4 window only.
- **Previews:** `fx_hud_f008.png` (boot / acquire), `fx_hud_f024.png` (lock, alert, card typing), `fx_hud_f070.png` (full). Test scene: stand-in Gary walking down a rack aisle, 8 bird blips, fast tier 540 x 674. Text is legible at 540 px at 1 sample, although thin strokes alias; it will be cleaner at review resolution.

### 8.2 F6: LEO constellation train + beam motion

- **Train** (`scripts/env/constellation.py`):
  - `constellation_train(c, n=10, spacing=4.5, scale=1.0, name="cn_train", seed=7, jitter=0.25) -> [sat objects]`. The root is `sats[0].parent`.
  - Each satellite is a generic flat-panel bus (2.8 x 1.4 m) with a phased-array tile face, one sideways solar wing (8.4 x 1.3 m) and a "sunlit" glint. Bus and wing are faintly emissive so the train reads at night, like the real sunlit trains. No logos or markings.
  - `constellation_pass(root, p_start, p_end, f0, f1, roll=0.0)` gives a linear pass with the train axis aligned to the direction of travel.
  - `CN.S6_PASS = ((-16, 33, 25.5), (14, 24, 27))` runs across S's S6 roof-camera view, left to right on screen, so satellites enter the frame one after another.
  - `sat_position(start, end, f0, f1, index, spacing, scale)` returns an analytic per-satellite position callable, for when the train is instanced.
  - Recommended: S calls the generator in its build rather than instancing, so the satellites are real objects that beams can track.
- **Beams** (`scripts/vfx/beams.py`). Sources and targets can be points, objects, (object, local offset) pairs or callables f(frame).
  - `beam_track(beam, source, targets, f0, f1, handoffs=None, step=1, jump=3, ramp=4, scene=None) -> [(frame, target index)]`: locks on and follows each target. At every handoff the beam blinks off for one frame and re-grows onto the next target over `jump` frames.
  - `beam_sweep(beam, source, f0, f1, yaw0, yaw1, pitch0=0, pitch1=0, length=12, step=1, wobble=0, ease=True, path=None, scene=None, seed=0)`: searchlight sweep by angles, or through `path` points.
  - `beam_hops(beam, nodes, f0, hop=6, hold=8, step=1, scene=None) -> end frame`: a beam jumping bird to bird.
- **Previews:**
  - `fx_sky_f020.png`, `fx_sky_f070.png`: train pass near S's roof camera, with a sweep beam.
  - `fx_sky_side_f031/f034/f037/f063.png`, from above the DC roof: the red track beam from the roof_L perch on satellite 1, re-growing toward satellite 2 at the f33 handoff, locked on satellite 2, then on satellite 3 after the f61 handoff. The cyan sweep crosses the street.
  - At sky distances use beam radius 0.04-0.06 m to read at 540 px; 0.01-0.02 m is too thin.

### 8.3 F7: high-speed data look (`scripts/vfx/datafx.py`)

```python
data_emitter_material(name, color=TX_RED, strength=6, rate=20, levels=4, base=0.08, pitch=1.0, offset=(0,0,0), seed=0, power=1)
data_beam_material(name, color=TX_RED, strength=9, symbol_length=0.06, speed=3.0, levels=4, gap=0.3, blink_rate=15,
                   blink_depth=0.35, core_power=2, fade=1, seed=0)
make_data_beam(name, p0, p1, radius=0.004, material=None, color=TX_RED, collection=None, seg=8, **material_kw)
apply_data_look(objs, material)                     # object-linked slots; shared meshes untouched
key_blink(objs, f0, f1, rate=12, duty=0.6, seed=0, fps=30, levels=(0, .33, .66, 1))   # seeded per-object steps
```

- **Emitter:** each lane (one cell of a `pitch` grid in object space, shifted by `offset`; align it so each lenslet sits in its own cell) shows a new random PAM4 level (0, 1/3, 2/3, 1) `rate` times per second. Object Info Random varies the pattern per object.
  - Hero-GLB lenslets: pitch 0.0023 in GLB units (2.3 mm, `LENS_PITCH` in the crude build).
  - x1000 macro sets: pitch 2.3.
- **Data beam:** PAM4 symbols scroll from the source at `speed` m/s, with dark gaps and random fast blinking. Symbols are measured in metres from the beam origin (`make_beam` convention; `beam_material` pulses are now in metres too, previously per normalised length).
- Everything is shader-driven through the `frame/30` time driver, with no per-frame keys. `key_blink` is for separate emitter objects (laser facets, LEDs).
- **Previews:** `fx_data_f010.png` and `fx_data_f011.png` (consecutive frames): 2x4 lenslets with per-lane levels and 8 dashed PAM4 beams; lanes and dashes change between the two frames.

## 9. v3 wave (DevLog-018: V4 satellites, V5 San Francisco, S6 beam counts)

### 9.1 San Francisco backdrop (`scripts/env/sf_landmarks.py`, built into both street blends)

- **Layout** (generic low-poly geometry, window emission, no logos or text marks; distances compressed so everything rises above the hill line at about 15-20 deg from the street):
  - The cross street (y 33-37.5) now has two corner blocks (y 37.5-46.5, |x| 5.6-15). The OUNOLIGHT neon sign (V1 rename) is on the left block and LATENCY LOUNGE on the right; the single 16 m end block is gone.
  - Between the corner blocks, a steep hill street climbs straight ahead from y 37.5 (z 0) to y 118 (z 18, about 13 deg), then a plateau to y 170. It has sidewalks, cable-car rails and slot, 5 lamp posts with spot lights, and 24 Victorian row houses stepping up both sides (pastel paint, bay windows, gables or flat cornices, lit windows, garage plinths).
  - A frontal "painted ladies" row of 7 houses sits at the hilltop (y 124) facing the street.
  - Cable car: classic two-section car (8.5 m) on the slope with lit windows, a headlight and a tail light, in its own collection `ENV_cable_car` (hook `HOOK_cable_car` with `path_start` / `path_end`). To move it, key the instancer location by (path point - hook location); `sf_landmarks.cable_car_path(u)` gives the path.
  - Downtown cluster: 14 window-grid towers, 110-230 m.
  - Pyramid tower at (-7, 530): 260 m, two side wings, spire, floodlit white with lit windows.
  - Tall rounded tower at (125, 600): 326 m, glowing crown.
  - Fluted tower on a lit hill at (28, 340): top 159 m, warm floodlight.
  - Red-orange suspension bridge on the left (towers at (-56, 590) and (-110, 800), 284 m, portal struts, catenary cables, suspenders, deck lights) with fog cards.
  - Grey suspension bridge on the right (towers (210, 590) and (320, 780), X-bracing) with white cable lights.
  - A low fog band, and dark far hills against the sky glow.
- **Hooks:** `HOOK_sf_<pyramid|round_tower|hill_tower|gg_tower_1|gg_tower_2|bay_tower_1|bay_tower_2>` (base + height) and `HOOK_sf_ladies`.
- **Sky dome:** radius raised from 240 m to 1200 m so it encloses the backdrop; camera `clip_end` must be >= 1300 (S uses 2000).
- **Counts:** ending 299 objects / 197 k triangles (was 261 / 176 k); opening 260 / 197 k.
- **Previews:** fast tier, from S's S6 v2.1 camera keys (2.0 / 4.75 / 5.05 / 7.15 s) in `render_previews.py` (`s6_door`, `s6_sidewalk`, `s6_up`, `s6_cc`, `sf_overview`):
  - `street_ending_s6_door.png`, `_s6_sidewalk.png`, `_s6_up.png`, `_s6_cc.png`, `street_ending_s6_cc_1080.png` (1080 x 1350), `street_ending_sf_overview.png`.
  - From the street, the hill street with Victorians and the cable car fills the vista. Above it, the pyramid spire, the fluted tower, the crowned tower and the red-orange bridge tower read at 540 px. The grey bridge is in frame in the up-shot and the overview.

### 9.2 Satellites: 0.5x and 2x speed (`scripts/env/constellation.py`)

- `constellation_train(...)` now defaults to `scale=V3_SCALE` (0.5); spacing scales with the root.
- `S6_PASS_V3 = ((-29, 28.5, 12.75), (27, 22.5, 15.75))` doubles S's v2.1 pass ((-15, 27, 13.5) -> (13, 24, 15)) about its centre. Over the same S6 3.6-8.2 s window the train therefore moves 2x faster.
- `scale_pass(p_start, p_end, factor)` lengthens any pass for other speed factors.

### 9.3 MIMO uplinks + street choreography (`scripts/vfx/beams.py`)

```python
vfx.mimo_links(sources, sats, f0, f1, k=2, step=2, min_elev_deg=20.0, reconfig_every=(12, 20), fade=4, cam=None, seed=0,
               radius=0.05, material=None, color=None, collection=None, isl=False, isl_material=None, isl_radius=0.02,
               scene=None, name="mimo") -> {"sessions": [(src_i, sat_j, f_start, f_end, beam)], "isl": [beam]}
vfx.street_beam_choreo(birds, f0, f1, n_sweep=4, n_p2p=3, switch_every=(15, 26), seed=0, sweep_len=(5, 9), sweep_period=(45, 80),
                       sweep_amp_deg=40, pitch_deg=(-20, 15), radius_sweep=0.015, radius_p2p=0.012, mat_sweep=None, mat_p2p=None,
                       step=2, fade=4, collection=None, scene=None, name="street") -> {"sweeps": [beam], "p2p": [(a, b, f_start, f_end, beam)]}
```

- **MIMO:** each source keeps `k` simultaneous links.
  - A link drops when its satellite goes below `min_elev_deg` or leaves `cam`'s frame.
  - Free slots fill with the best visible satellite: elevation + seeded per-pair preference - load from other birds, so links spread over the array.
  - Each bird re-configures one link every 12-20 frames (seeded, staggered, never re-picking the satellite just dropped).
  - New links grow in over `fade` frames; dropped links fade out.
  - `isl=True` adds inter-satellite links along the train.
  - The session list is returned for SFX timing.
- **Choreo:** `n_sweep` distinct birds sweep searchlights back and forth for the whole window: sinusoidal yaw +-40 deg about a centre (default toward the street axis), pitch bob, seeded period, phase and length, staggered fade-in. Alternating sweeps default to TX red / RX cyan materials; pass `mat_sweep` to override.
  - `n_p2p` concurrent point-to-point links run between the other birds. Each switches partner every `switch_every` frames (fade out, re-grow onto the new pair), never repeating the previous pair or reusing a busy bird.
  - `birds` items are sources, or (source, yaw_centre_rad) pairs.
- **Previews** (fast, 540 x 674):
  - `fx_v3sky_f045/f070/f095/f120.png`, from S's 5.05 s up-shot: the 0.5x train on the 2x pass, 4 parapet/roof birds with k = 2 links each, 24 link sessions over 100 frames, ISLs. The link set changes between frames.
  - `fx_choreo_s6_sidewalk_f020/f050/f080.png` and `fx_choreo_s6_cc_f035/f065.png`: 4 sweeps + 3 switching links (15 p2p sessions over 120 frames), with the SF backdrop behind.
- The v2-scale train strips (`fx_sky_*`) were removed as superseded. `build_fx_tests.py` modes: `hud`, `sky` (v2 scale 1), `data`, `v3sky`, `choreo` (`S6CAM=s6_sidewalk|s6_cc|...`).
- **Known gaps:**
  - In `fx_v3sky_f045`, links from the roof_L bird to the train's lead satellites start at the left frame edge and are not yet visible. S may start the MIMO window a few frames after the train enters.
  - The cyan sweep beams read near-white at the core under AgX.

## 10. v4 W8: parody AI-era SF billboards (DevLog-020)

Owner (verbatim, DevLog-020 W8): "also for the final scenes, use famous AI era SF startup billiboard or companies for the street signs." The coordinator's reference search and draft parody set are in DevLog-020 W8. The coordinator had already renamed the DC sign to EQUONOX (`sign_equonox`); it is kept.

### 10.1 Built (`scripts/env/billboards.py`, called from `build_street.py` for both layouts)

| Item | Where (world, m) | Art |
|---|---|---|
| ARTISAND rooftop board 6.0 x 2.4 | R3 roof, centre (7.6, 27.0, 10.9), facing (-0.55, -0.83) | off-white lit panel, black "STOP HIRING HUMANS.", red "HIRE BIRDS.", red side bar, brand |
| NONLINEAR rooftop board 7.0 x 2.8 | R4 roof, centre (7.8, 31.0, 15.4), facing (-0.5, -0.87); staggered above ARTISAND | navy panel, white "AGENTS." / "AT YOUR COMMAND.", violet bar, brand |
| SIGMOID rooftop board 5.6 x 2.6 | L3 roof, centre (-7.4, 27.4, 7.4), facing (0.55, -0.83) | green panel, white "WELCOME TO" / "AI COUNTRY", gold disc, brand |
| bare "AGENTS." board 3.0 x 1.5 | R2 roof, centre (7.0, 24.2, 6.7), facing (-0.6, -0.8) | black panel, white "AGENTS." |
| VANTANOT bus-shelter ad (portrait lightbox) | left sidewalk shelter at (-4.75, 28.2) | white panel, "COMPLIANCE / THAT DOESN'T / SOC 2 (purple) / MUCH", brand |
| SNOOKS bus-shelter ad | right sidewalk shelter at (4.75, 28.6) | yellow panel, "AI WON'T / TAKE YOUR JOB. / A BIRD USING / SNOOKS WILL.", brand |
| Company neon signs (shop bands, z 3.05) | L2 KARAOKE -> CLOSEDAI (white), R3 ARCADE -> MISANTHROPIC (warm orange), L4 NOODLES -> PERPLEXED (cyan) | neon tube text on a backer (the existing neon_sign kit) |
| WAYMORE robotaxi | `ENV_robotaxi`, parked behind the crosswalk at (2.35, 35.2), front toward -y | generic white car, roof lidar dome with a spinning teal band (cyclic keys), corner sensors, "WAYMORE" plain type on the doors |

- **Rules followed:** parody names only; system type (Arial Black / Arial Bold); one accent colour plus at most one simple shape; no image generation. Panels are emissive (lit at night), with uplight heads and a soft area-light wash per rooftop board. Text is converted to mesh.
- **Hooks:** new `HOOK_billboard_<artisand|nonlinear|sigmoid|agents|vantanot|snooks>` and `HOOK_robotaxi` (`path_start` (2.35, 35.2, 0) -> `path_end` (2.35, 26.5, 0)). To drive the car slowly, key the `ENV_robotaxi` instancer.
- **Hook check:** every existing hook (perches incl. `HOOK_perch_x_scooter_bar`, landmarks, SF) was diffed against the pre-W8 JSON for both layouts: 0 changed, 0 missing, 7 added.
- **Other geometry changes:**
  - `bldg_backfill` walls behind the building rows lowered from 14 m to 6 m; they rose above the roofs and hid half of each rooftop board.
  - R4's roof water tank removed to make room for NONLINEAR (R4's tank had no hook).
  - The billboard code uses its own placement data and no RNG, so building geometry and perches are otherwise unchanged.
- **Counts:** ending 361 objects / 211 k triangles (was 299 / 197 k); opening 322 / 212 k. Extras: about 60 small objects and 6 area lights.

### 10.2 Placement vs the S6 framing

- S's S6 cameras sit near x = -3 looking +y, so the side buildings hide the street end beyond |x| = 6.
- Rejected positions:
  - Freeway pole boards behind the corner blocks: not visible from any S6 key.
  - Boards over the end-of-street flanks: hidden or over the landmark vista.
  - Robotaxi at the right curb near y 22-28: hid the SNOOKS shelter.
- The boards therefore sit on the right-side roofs (R2/R3/R4), angled toward the camera side and staggered in height so they do not overlap. SIGMOID is on the L3 roof, and the ads are at sidewalk level.
- None of them covers the SF landmarks: pyramid, fluted tower, crowned tower and red bridge tower stay clear in every S6 view. All are above the bottom caption band.
- **Known framing limits:**
  - SIGMOID sits at the left frame edge in the sidewalk / CCTV views (fully visible in the door view and the up-shot).
  - NONLINEAR's brand line is partly behind ARTISAND from x = -3.
  - The bare AGENTS. board is at the right edge in the CCTV view.

### 10.3 Previews (fast tier; `render_previews.py` views)

- `street_ending_s6_door.png`, `street_ending_s6_sidewalk.png`, `street_ending_s6_cc.png`, `street_ending_s6_wide.png`, `street_ending_s6_up.png`
- `street_ending_s6_cc_1080.png` (1080 x 1350)
- Close checks: `street_ending_bb_front_artisand.png`, `street_ending_bb_front_nonlinear.png`, `street_ending_bb_left.png`, `street_ending_bb_right.png`

### 10.4 For S

- Re-run the film build: `append_env` picks up the rebuilt street blends and instances the new child collections `ENV_billboards` and `ENV_robotaxi` with the street windows.
- No code change is required. Optional: key the `ENV_robotaxi` instancer along `HOOK_robotaxi`'s path.

## 11. Progress log

- 2026-10-10 08:40: read DevLog-010, DevLog-006, `build_crude_film.py` (street, facade, perches, hatchery), R's flap vent JSON and hall layout; extracted 10 frames of crude v0.2.3 (S1, S5 hatchery, S6) to the scratch dir. Plan written (section 2).
- 2026-10-10 08:45: VFX library v1 (nodes, materials, atmosphere, post) + test sheet; first render OK; neon/eyes blown out under AgX MHC, framing clipped. Compared AgX Punchy / Standard / Khronos PBR Neutral: chose AgX Punchy for previews; neon 14 -> 6, eye 12 -> 6, hologram 3 -> 2.
- 2026-10-10 08:59: env_lib, street_kit, build_street; opening built (221 objects, 177 k tris, 39 hooks); R's vent appended (bbox x +-0.41, y 12.916-13.1, z 2.925-3.56); duplicate flap trim removed (R's frame is the trim).
- 2026-10-10 09:00: previews round 1 (opening 5 views, ending 6 views): puddles too large/black, roof condensers and lit windows too bright, satellite unlit in the sky view, markers dominate. Fixed: fewer/smaller puddles, AC housing darker, window/shop emission about halved, satellite key light, clean renders without markers + separate `_hooks` renders.
- 2026-10-10 09:04: hatchery v1 (stage meshes from the hero GLB, 5 SCARA robots, inspection, loop); preview: robots oversized vs the 0.32 m birds, cyan rails blinding, haze milky. Fixed: robots smaller (column r 0.08, links 0.32/0.30, arm z 1.3), rails steel with a thin LED strip, haze 0.01 -> 0.004, bigger labels, closer cameras; internals hidden after the shell; loop seam check f1 vs f61 PSNR 38.9 dB (grain only).
- 2026-10-10 09:07: round 3: skyline windows floated as bright tiles (towers invisible against the sky): smaller, dimmer, 20 % lit, lighter tower base; asphalt puddle coverage 0.28 -> 0.15; wire perches set to +1 cm over the wire centre (crude).
- 2026-10-10 09:09: round 4 (final): shopfront shelves pattern, warmer, end block uses lit-window glass; eye glow white core 0.25 -> 0.1; all blends rebuilt; 16 previews + VFX sheet rendered and checked. Perch check (uv run): 21 crude perches within 0.3 mm in both layouts.
- 2026-10-10 10:50: v2 work order received (DevLog-010 section 9, items 9, 10, 13). Read-only probe of `scenes/film_v1.blend`: the hatchery animation survives S's append (joints/pallets move at frames 1441-1491).
- 2026-10-10 10:59: wet ground + puddle fast mode (`Fast` input, fake reflection, `set_fast_reflections`); satellite scale + close pass in `ENV_satellite`; `ENV_tree_projector` + 5 alternative bases with the clear-zone check; hatchery continuous conveyor, tracking robots, hop-off bird, DISPATCH ledge, `hatchery_api.py`. All blends rebuilt (opening 222 objects / 42 hooks; ending 261 / 81; hatchery 186 objects).
- 2026-10-10 11:00: fast-tier previews (opening wide / s1_start / tree_alt, ending door_out / puddles_low / sat_roofcam, hatchery front f1/f21/f41, hop f9/f16/f24, eyes, station). Fixes after the look: QA display moved from a floor post (blocked the hop) to a hanger on the gantry. Removed the stale `street_ending_sky.png` (far satellite). The v1 previews without `_f`/v2 names are the ray-traced look of 09:09; street geometry is unchanged except the satellite.
- 2026-10-10 11:20: interim owner items F4/F6/F7 (DevLog-015) received. Wrote `vfx/hud.py`, `vfx/beams.py`, `vfx/datafx.py`, `env/constellation.py`; beam flow is now measured in metres from the source; `vfx/build_fx_tests.py` holds the tests.
- 2026-10-10 11:29: HUD round 1: text too small (8 px), panels and the scan tint washed the frame green, filled radar box. Fixed: sizes 0.034-0.046, darker panels, tint 0.25 -> 0.03, radar outline only.
- 2026-10-10 11:31: HUD round 2: glyph stems missing (B/H/L) with Andale Mono and SF Mono (overlapping contours). Switched to Courier New Bold x1.3; all glyphs present. Feed label and link block moved off the compass tape.
- 2026-10-10 11:35: sky round 1: train invisible against the end block (unlit) and the track beam was head-on. Fixed: emissive "sunlit" bus/wing + larger glint, pass raised to z 25.5-27, hand-off strip camera above the DC roof. Data test: per-lane levels and PAM4 dashes change frame to frame. All previews checked.
- 2026-10-10 15:40: v3 wave (DevLog-018) received; read S's S6 camera keys and train/beam staging in `scripts/build_film.py`.
- 2026-10-10 15:46: `sf_landmarks.py`, hill street, Victorians, cable car, landmarks, fog; dome 1200 m. Round 1 from S cameras: landmarks too high in frame (tops cut) and the pyramid unreadable (dark). Fixed: hill lowered to z 18, landmarks moved 1.4x further out, pastel paint emission 0.04 -> 0.16, floodlit pyramid (window grid `base_emit`).
- 2026-10-10 15:50: constellation v3 defaults (0.5x, `S6_PASS_V3`); `mimo_links`, `street_beam_choreo`; test renders (MIMO from the up-shot, choreo from the sidewalk and CCTV cameras) and a 1080 x 1350 check of the CCTV camera view; both street blends rebuilt.
- 2026-10-10 17:19: v4 W8 started. `billboards.py` (flat parody boards, bus-shelter ads, robotaxi), company signs in the SIGNS table; first build.
- 2026-10-10 17:20: round 1 from S6 cameras: rooftop boards cut at the frame top, pole boards invisible (hidden by the side buildings), text overflowing (fit used the wrong text axis), brand mis-placed (stale matrix). Fixed: text fit on local Y, view-layer update before parenting, boards moved to R2/R3/R4/L3 roofs, ads to bus shelters.
- 2026-10-10 17:25: front-on checks: half of each rooftop board hidden by the 14 m backfill wall; lowered to 6 m. Railing crossing the art removed; disc accent reserves text width; robotaxi moved behind the crosswalk (it hid the SNOOKS shelter). Hooks diffed: unchanged.
- 2026-10-10 17:28: final build of both blends; S6 previews + 1080 check viewed.
