# DevLog-007: Datacenter racks (GB300 NVL72 compute rack, Q3400-style bird switch rack, hall layout)

| Field | Value |
|---|---|
| Date | 2026-10-10 |
| Status | v2 bird rack (tiered caged farm, owner P2 in DevLog-010) delivered with hall layout v2; GB300, real Q3400, vent, overhead unchanged since the audit round |
| Workstream | R (DevLog-005 section 2) |
| Owns | `scripts/assets/datacenter/`, `assets/components/datacenter/`, `outputs/assets/datacenter/`, `research/racks_refs.md`, this file |
| Inputs | Owner feedback 2026-10-10 (DevLog-005 section 1), DevLog-004 (hall frame, bird scale 2.6, cells), interconnect film rack generators (read-only, copied and adapted) |
| Research | `research/racks_refs.md` (sources, grades, local reference files in `references/racks/`) |

## 1. Brief (owner, verbatim excerpt)

```text
Reuse rack assets from the prev interconnect film project and make the bird racks about 1/3 of the racks. The birds should also fill the bird rack from top to bottom, and the rack should be built to higher details similar to the interconnect film racks, and they both probably needs further improvements after you inspect more rack refs. Use GB300 as the compute rack ref. And mimic the bird rack as a switch rack with e.g. nvidia QD-3400.
```

"QD-3400" = NVIDIA Quantum-X800 **Q3400-RA** (research section 1).

## 2. Plan

1. Research (done, `research/racks_refs.md`): Lenovo GB300 NVL72 user guide figures (rack front/rear, tray fronts/rears), NVIDIA Q3400-RA datasheet, user-manual spec and front/rear figures, CC BY GB200 NVL72 photo for colours, 800 mm network-cabinet cable managers.
2. Reuse: copy `common.py`, `dc_common.py` (MeshBuilder, materials, perforation, curves, previews) and `Part/instantiate` from the interconnect film into `scripts/assets/datacenter/`; extend with box-projected UVs and small generated image tiles (perforation, cable bundle) so the look survives glTF export.
3. Build (all true-scale metres, one generator per asset, `.blend` + `.glb` + `.json`):
   - `rack_gb300_nvl72`: 600 x 1068 x 2236 mm, 48U EIA, U layout measured from Lenovo Fig. 11 (6-shelf configuration); compute tray, NVLink switch tray, power shelf, management switch fronts per Lenovo Figs. 2/5/7/9; drip pan; rear busbar, cable-cartridge frame (2 rails, 9 bolted cross bars, 5 cable columns), side manifolds with per-tray QDs, tray rear connectors, rack-level manifold couplings 2 x 2 at the bottom rear, power whips.
   - `switch_q3400_real`: true-scale Q3400-RA chassis (438 x 177.8 x 850 mm), 72 cages laid out per the NVIDIA front figure; rear 8 PSUs + 10 fans.
   - `rack_bird_q3400`: 800 x 1068 x 2236 mm network cabinet; top: 1U patch panel, 1U management switch, 1U horizontal manager (true scale); below: three bird-scale Q3400-style chassis, each one block row of 2 x 3 portrait cells; vertical finger cable managers both sides; rack-top landing pad with charging/perch rail. Hooks `HOOK_cell_<unit>_<row>_<col>` (+X out of the cell, origin = bird feet origin), `HOOK_landing_pad`, `HOOK_perch_<i>`, fill order top to bottom.
   - `bird_flap_vent`: louvered wall vent with a swinging flap (entrance).
   - `hall_layout.json`: 2 rows x 15 racks, bird racks 5 per row (1/3) at the same indices on both sides, entrance and landing pad positions, every cell in world coordinates with fill index.
4. Previews (EEVEE draft) front / rear / 3-4 of each rack, bird rack with bird-envelope proxies, hall top + aisle view; compare with references; iterate at least once.

## 3. Scale statement (bird rack)

| Item | Real Q3400-RA (B, measured on the NVIDIA figure) | Bird chassis (as built) | Ratio |
|---|---|---|---|
| Chassis face | 438 W x 177.8 H mm (4U, A) | 700 W x 640 H mm, 850 mm deep (same depth as the real) | 1.6x W; one bird chassis = one real block row (57.5 mm pitch) -> 11.1x H |
| Cage grid | 72 cages = 3 block rows x 3 groups x (2 x 4) | 6 cells = 1 block row x 3 groups x (2 x 1) | - |
| Cage opening | modelled 22.8 W x 13.3 H mm, 0.3 mm lips, outline 23.4 x 13.9 mm (landscape, aspect 1.71); an OSFP module is 22.58 x 13.00 mm (OSFP MSA Fig 3-2) | 180 W x 270 H x 360 D mm (portrait, aspect 0.67) | 7.9x W, 20.3x H (anisotropic, stylized) |
| Label strip | about 3 mm, champagne, between cage pairs | 34 mm, champagne, "1 ^v 2" style numbers | 11x |
| Perforation | square holes, about 3 mm pitch | square holes, 9 mm pitch (image tile) | 3x (kept fine so it reads as grille) |
| True-scale cues kept | - | FNM OSFP port, USB, RJ-45 on the chassis side panels; 1U patch panel, 1U mgmt switch, 1U brush panels, finger managers at true scale | 1x |

Bird envelope at hall scale 2.6 (crude film primitive x 2.6, DevLog-004): 0.32 m beak to tail, 0.245 m tall, about 0.13 m wide; beak 0.16 m ahead of the feet origin, tail 0.17 m behind. In a cell: 25 mm side clearance, 25 mm head clearance, feet origin 60 mm behind the faceplate, beak about 0.10 m out of the faceplate, tail 0.23 m deep (cell 0.36 m). A uniform 2.6x scale of a real OSFP cage (61 x 37 mm) could not hold a standing bird, hence the portrait cells.

## 4. As built (2026-10-10)

| Asset | Files (`assets/components/datacenter/`) | Size (m, X depth x Y width x Z) | Tris | Accuracy |
|---|---|---|---|---|
| GB300 NVL72 compute rack | `rack_gb300_nvl72.blend/.glb/.json` | 1.068 x 0.600 x 2.236 (upper power whips end at 2.376 under the busway tap-offs) | 163.8k (incl. 8 reused cartridge bands) | envelope A, U layout B (+-1U), tray fronts A inventory / B positions, tray depth C, rear B |
| Bird switch rack | `rack_bird_q3400.blend/.glb/.json` | 1.069 x 0.800 x 2.236 (perch rail to 2.332) | 37.5k | cabinet C, chassis grammar B, scale stylized |
| Q3400-RA true scale | `switch_q3400_real.blend/.glb/.json` | 0.863 x 0.483 x 0.178 | 26.2k | envelope A, front layout B, rear B |
| Bird flap vent | `bird_flap_vent.blend/.glb/.json` | 0.184 x 0.82 x 0.635 | small | C (design) |
| Overhead infrastructure | `overhead_infrastructure.blend/.glb/.json` (world coordinates) | 5.0 x 10.2 x 1.95 (z 2.30-4.25) | small | C (design) |
| Hall layout | `hall_layout.json` | 30 racks + overhead runs | - | design |

Generators (`scripts/assets/datacenter/`): `rk_lib.py` (helpers adapted from the interconnect film `common.py`, `dc_common.py`, `dc_parts.py`), `rk_q3400.py` (shared real/bird chassis), `build_rack_gb300_nvl72.py`, `build_switch_q3400_real.py`, `build_rack_bird_q3400.py`, `build_hall_layout.py` (layout JSON + vent + hall previews; reads `rack_bird_q3400.json`, so build the racks first), `check_rack_glbs.py` (re-imports every GLB, reports hooks/bounds). Each runs as `scripts/bslot.sh -b --python scripts/assets/datacenter/<script>.py -- assets/components/datacenter` in about 5-10 s including previews.

Common frame for every rack: origin at the floor centre of the footprint, FRONT = +X, width along Y (+Y = right seen from the front), Z up; glTF export Y-up. Materials are plain Principled BSDF plus small packed image tiles (perforation, cable bundle, fan grille, rail holes) on box-projected UVs, so the look survives glTF. Meshes are shared between identical trays/chassis (cheap instancing). Collections are toggles (e.g. hide SIDE_PANELS for cutaways).

### 4.1 GB300 NVL72 rack (vs Lenovo Figs. 2-12)
- Front, U1 at z = 82 mm (82 mm plinth + 48U + 20.4 mm top frame = 2236 mm): U7-9 power shelves (3), U10 drip pan, U11-18 compute (8), U19-27 NVLink switch (9), U28-37 compute (10), U39-41 power shelves (3), U45-46 management switches (2); open bays elsewhere (as in Fig. 11; the busbar shows through).
- Compute tray front per Fig. 2: OSFP card (2 OSFP) + RJ-45 lower left, perforated PCIe blank upper left, 8 E1.S carriers with LEDs, control panel (3 buttons, RJ-45 BMC, USB, micro-USB, mini-DP, NVL/fault LEDs), BlueField-3 bracket (RJ-45 + 2 QSFP) upper right, OSFP card (2 OSFP) lower right, ears with latch levers.
- NVLink switch tray per Fig. 7: control area (buttons, status + 4 NVLink LEDs, USB, 5 RJ-45) and the plain light-grey panel. Power shelf per Fig. 9: PMC module + 6 PSUs with round fan grilles, handles, 2 LEDs each. Management switch: 48 RJ-45 + 4 QSFP28.
- Rear per Fig. 12 / Fig. 3: copper busbar in a black shroud (full height, hidden behind the cartridges in the tray zone), cable cartridges (cable-bundle texture) behind a frame of 2 side rails, 9 bolted cross bars and column dividers; per-tray rear connectors (4 cartridge blocks with guide pins, busbar clip, colour-ringed QDs); side manifolds (supply blue -Y, return red +Y) with a hose per tray; rack-level couplings 2 x 2 at the bottom rear; 6 power whips (upper shelves exit through the roof brush, lower ones down through the plinth).
- Hooks (31 in the default state): `HOOK_compute_NN_front` (18), `HOOK_nvswitch_NN_front` (9), `HOOK_rack_front_center`, `HOOK_rack_top_center`, `HOOK_busbar_top`, `HOOK_manifold_couplings`; plus `HOOK_pulled_compute_14`, `HOOK_pulled_nvswitch_05` in the hidden pulled-tray variant (animate location.x 0.492 seated -> 1.042 pulled).
- Toggle collections: `BLANKS` (1U blank panels for every open front bay, hidden by default because Fig. 11 shows open bays), `REAR_CARTRIDGE_DETAIL` (reused cartridge bands, on), `VARIANT_seated_slots` (on) / `VARIANT_pulled_trays` (off): swap them to show compute tray 14 and NVLink switch tray 5 slid out 0.55 m with the lid off. The .glb holds the default state only; use the .blend for variants.

### 4.2 Bird switch rack
- Cabinet 800 x 1068 x 2236 mm (black, closed sides, open front and rear). From the top: 1U fibre patch panel (24 duplex LC, blue adapters, yellow patch cords into the left manager), 1U management switch (48 RJ-45 + 4 QSFP28), 1U horizontal brush manager, then bird chassis u1, u2, u3 (each 640 mm), 1U brush panel, plinth. Perforated fillers beside the 19 in units.
- Each bird chassis: perforated black grille, 6 cells with dark-nickel cage lips, zinc walls, EMI fingers top and bottom, green host PCB floor with a 14-finger gold contact field under the feet, back-wall host connector with gold contacts and an in-cell status LED; champagne label strips with port numbers; left panel = LED bar + port LED beside each cell row (cols 0-1), true-scale FNM OSFP port, black badge with the parody name "NVYDIA QX-3400B"; right panel = LED bars for col 2, 5 status LEDs, USB/console, 2 x RJ-45 (Cat6 cables run into the right finger manager and up to the mgmt switch). Rear: 2 PSUs per side, 5 fans with champagne handles, power cords to two 0U PDUs.
- Finger managers (2 in class) both sides at true scale (one finger per U).
- Rack top: landing deck 0.60 x 0.76 m with cyan glow ring, amber cross marks and approach chevrons (pointing to the aisle), copper/gold charging coil, edge LED strips; charging/perch rail along the front edge with 3 lanes (front brass = GND for the claws, middle gold = power, rear gold = signal), 4 perch spots with LEDs at 0.21 m spacing, rail posts between the spots. Glow objects use `MAT_dc_pad_glow_cyan` / `MAT_dc_pad_glow_amber` (animate emission for "charging").
- Hooks (25): `HOOK_cell_<unit>_<row>_<col>` x 18 (unit 1..3 top to bottom, row 0 upper / 1 lower, col 0..2 left to right seen from the front; location = bird feet origin on the cell floor 60 mm behind the faceplate, x = 0.445 m; rotation identity so +X = out of the cell = bird beak direction; custom props `fill_index` 0..17 top-left to bottom-right, `unit`, `row`, `col`, `cell_label`, `cell_opening_m`). Cell floor z (m): 1.786, 1.470 (u1), 1.146, 0.830 (u2), 0.506, 0.190 (u3); y (m): -0.208, 0, +0.208. `HOOK_landing_pad` (0.22, 0, 2.256), `HOOK_perch_0..3` (0.492, y = -0.315 / -0.105 / 0.105 / 0.315, 2.332), `HOOK_rack_front_center`, `HOOK_rack_top_center`.

### 4.3 Hall layout (`hall_layout.json`)
- Frame = crude v0.1 hall: aisle along +Y, rack fronts at x = +-1.1, end wall with the door at y = 13.0, ceiling 4.2 m.
- 2 rows x 15 racks, rows y = 1.0 .. 11.0; per row C B C repeating (bird at index 1, 4, 7, 10, 13), same indices on both rows so bird racks face each other across the aisle: 10 bird racks / 30 = 0.333; 180 cells. Left row unrotated at x = -1.634; right row rotated 180 deg at x = +1.634.
- Entrance: bird flap vent centre (0, 12.94, 3.25), opening 0.70 x 0.50 m, in the end wall above the door, inward normal -Y, vent rotation_z -90 deg; flap object rotates about its local Y.
- Primary landing pad: rack L13 top, (-1.414, 10.0, 2.256), facing +X (aisle). All 10 pads and 40 perch spots listed; `overhead` lists the busway / fibre raceway / ladder runs, 20 tap-off boxes (with the 3 GB300 whip end points each) and 10 raceway drops; every cell listed in world coordinates with `fill_index` and bird yaw.

### 4.4 Previews (`outputs/assets/datacenter/`, EEVEE 8 samples)
- GB300: `rack_gb300_nvl72_front.png`, `_rear.png` (orthographic), `_three_quarter.png`, `_three_quarter_rear.png`, `_front_closeup.png`, `_rear_closeup.png`; `_rear_couplings.png`, `_variant_pulled_trays.png`, `_variant_blanks_front.png`; `compare_gb300_vs_lenovo_fig11_fig12.png` (render vs Lenovo figures, local reference only).
- Bird rack (with preview-only bird envelope proxies): `rack_bird_q3400_front.png`, `_three_quarter.png`, `_rear.png`, `_three_quarter_rear.png`, `_cells_closeup.png`, `_cells_empty_closeup.png`, `_landing_pad.png`.
- Q3400 true scale: `switch_q3400_real_front.png`, `_rear.png`, `_three_quarter.png`; `compare_q3400_real_vs_nvidia_front.png`.
- Hall: `hall_layout_top.png`, `hall_layout_aisle_from_entrance.png`, `hall_layout_aisle_eye.png`, `hall_layout_landing_pad.png`, `hall_layout_overhead_whips.png` (top view hides the overhead asset).

## 5. Sourced vs estimated

- Sourced (A): GB300 rack 600 x 1068 x 2236 mm, tray/shelf/switch counts, tray front/rear inventories, rear inventory (busbar, cartridges, manifolds), Q3400-RA 4U 438 x 177.8 x 850 mm, 72 OSFP cages, separate FNM port, front management ports, 8 PSUs, 10 fans.
- Measured on primary figures (B): GB300 U positions (+-1U), element positions on the tray fronts and the rack rear, Q3400 cage pitch/gaps/panels, rear PSU/fan arrangement; tray colours from the CC BY GB200 photo (assumed same for GB300).
- Estimated (C): plinth 82 mm (80-140) + top frame 20.4 mm, tray depth 880 mm (800-950), rear connector/manifold/busbar sizes, bird cabinet 800 mm width (common network cabinet, matched depth/height), true-scale top units, landing pad, perch rail, flap vent (design).
- Stylized: bird chassis and cell scale (section 3).

## 6. Deviations and open issues

- GB300 shown in the 6-power-shelf configuration of Lenovo Fig. 11; NVIDIA's reference architecture lists 8 shelves (142 kW). Two more shelves would fill two of the open bays; tell me if wanted.
- No front cabling on the GB300 (OSFP/QSFP cages empty, as in the exhibit photo); a deployed rack would have transceivers and fibre bundles. Could be added as a toggle collection.
- Bird cells have no beak "latch lip" (DevLog-004 open point); the cage top edge could take a lip bar if S2 needs the beak bite on crowd cells.
- Bird racks are 0.8 m wide vs 0.6 m compute racks, so cell columns do not line up with the old 0.8 m pitch of crude v0.1 (S reads positions from the JSON).
- Leash (tail pull tab) is not represented in the proxies; cell depth leaves about 0.13 m behind the tail.
- Text on the front (port numbers, parody badges) is converted to meshes; no logos anywhere; real product names only in docs.
- Fresh-context audit: run by the coordinator 2026-10-10; corrections in section 9.

## 7. TODO

- [x] Research GB300 NVL72, Q3400-RA, switch-rack practice; refs file
- [x] Read interconnect film rack generators and DevLogs
- [x] Copy/adapt helper library (`rk_lib.py`)
- [x] GB300 NVL72 rack generator + previews
- [x] True-scale Q3400-RA chassis
- [x] Bird switch rack generator + hooks + previews with bird proxies
- [x] Bird flap vent
- [x] Hall layout JSON + hall preview
- [x] Iteration round after comparing previews with references (section 8)
- [ ] Owner/coordinator review; optional: 8-shelf variant, GB300 front cabling toggle, cage lip bar
- [x] Fresh-context audit (coordinator) and corrections (section 9)
- [x] v2 bird rack: tiered caged farm (P2), fully filled from the top, hero front slot empty, per-slot hooks with head/neck data (section 10)
- [x] hall_layout.json v2 (240 slots, tiers, occupant, compat aliases) + previews
- [ ] Owner review of the v2 bird rack (end gaps beside the middle rows, neck extension look)
- [x] Hero mating cell for the S2 cutaway (DevLog-010 section 9 item 3): `hero_cell_mating` asset, hooks, hall_layout `hero_cell` (section 11)
- [x] v4: GB300 front ports populated (CX-8 OSFP, BF3 QSFP112, RJ-45), fibre bundles to the overhead raceway, Cat6 to the mgmt switches, bird-rack patch-panel trunks (section 12)
- [x] v4 fresh-context audit against the refs (section 12.6)

## 8. Progress log

- 2026-10-10 01:20: read DevLog-005/004, crude v0.1 frames, interconnect rack generators and previews (generic trays, order from recollection, no cartridges).
- 2026-10-10 01:30: research: Lenovo GB300 NVL72 UG downloaded (figures give the real front order and rear layout), NVIDIA Quantum-X800 datasheet, XDR switch UM specs + Q3400-RA front/rear figures, 2 CC BY GB200 photos, cable-manager listings; `research/racks_refs.md` written; plan above.
- 2026-10-10 01:42: `rk_lib.py` + GB300 generator; first previews overexposed and too small in frame. Fixed lights, ortho front/rear.
- 2026-10-10 01:46: `rk_q3400.py` + true-scale Q3400; layout matched the NVIDIA figure (groups, blocks, panels) on the first try; materials too bright.
- 2026-10-10 01:49: bird rack generator; previews OK structurally; Bezier cables overshot through the side panel -> Catmull-Rom cable helper; rear fan handles toned down.
- 2026-10-10 01:52: hall layout JSON (30 racks, 10 bird, 180 cells), bird flap vent, hall previews (collection instances + bird proxies).
- 2026-10-10 01:54: iteration vs references: removed invented vent slots on the NVLink switch tray (Fig. 11 shows a plain panel) and an invented drip-pan label; centre cartridge column now hides the busbar (Fig. 12); found the chassis body box sat 2.5 mm behind the faceplate and hid every cage interior (real cages read grey, bird cells were flat) -> body now starts behind the cages; cage backs black on the real chassis; real Q3400 front now matches the NVIDIA figure closely.
- 2026-10-10 01:58: final rebuild of all assets; GLB re-import check: hooks and custom props survive (bird rack 26 hooks, `HOOK_cell_1_0_0` facing +X, fill_index 0); no .blend1 backups; logs `outputs/logs/racks_*_2026101*.log`.
- 2026-10-10 02:20: audit corrections applied (section 9); all generators re-run in one Blender at a time; previews checked (Q3400 vs NVIDIA figure, bird cells, landing pad, pulled trays, rear couplings, overhead whips, aisle); GLB re-import check clean.

## 9. Audit corrections (fresh-context audit, 2026-10-10)

| # | Finding | Correction |
|---|---|---|
| 1 | Badges carried real model numbers (Q3400-RA / Q3400-B) | Parody badges only: "NVYDIA QX-3400" (true-scale reference), "NVYDIA QX-3400B" (bird chassis); real name kept in docs only |
| 2 | True-scale cage opening 21.0 x 11.9 mm smaller than an OSFP module (22.58 x 13.00 mm) | Opening 22.8 x 13.3 mm, 0.3 mm lips, pitch 23.4 mm; `racks_refs.md` and section 3 updated |
| 3 | Group gap 17.2 mm vs measured about 16 +-1 mm | Panels set to the measured 62 / 64 mm -> gap 15.6 mm, inside the tolerance |
| 4 | Plinth statement inconsistent | Both docs: 82 mm plinth + 20.4 mm top frame = 102.4 mm = 2236 - 48 x 44.45 |
| 5 | Two "search summary" sources graded A | Regraded B (Lenovo DS0207 "standard 19-inch", Lenovo lp2357 135/155 kW) |
| 7 | Owner asked to reuse the interconnect rack assets | Reused by read-only append from `../20261002_interconnect_film/assets/components/datacenter/`: `nvl72_backplane_cartridges_band` x 8 on the GB300 rear cartridge bays (scaled 0.95 in width; now at the real rear location), `tray_dummy_gb300_compute_dressing` and `tray_dummy_nvlink_switch_dressing` (scale 0.8) in the pulled compute / NVLink switch trays (`VARIANT_pulled_trays`). Helper code was already adapted from `common.py`, `dc_common.py`, `dc_parts.py` (rk_lib.py header). Recorded in `rack_gb300_nvl72.json` `reused_from_interconnect_film` |
| 8 | Grainy metallic cage walls at low samples | Cage walls non-metallic (metallic 0, roughness 0.75); frame, dark steel and dark nickel less metallic / rougher; previews at 16 samples |
| 9 | GB300 rack-level couplings were chrome spheres; open bays | Couplings = quick-disconnect bodies (female body, knurled colour sleeve, male nipple/hex) on grey plates, facility hoses down through the plinth, all inside the footprint; `BLANKS` collection (1U blanks for every open bay, hidden by default to match Fig. 11) |
| 10 | Rear panels flat | Fan grille tile (staggered openings, lighter land) on the fans; PSUs with IEC inlet surround, recess, 3 pins, LED, handle, red latch tab |
| 11 | Preview bird proxies overlapping each other and the rail | 4 perch spots at 0.21 m spacing, rail posts moved between spots, preview birds on the outer spots only |
| 12 | GB300 whips ended in mid-air | New `overhead_infrastructure` asset + `hall_layout.json` `overhead`: busway per row over the rack rear (rack-local x = -0.28) with a tap-off box over every GB300 at the whip ends (z = 2.376), yellow fibre raceway (z 2.95-3.05) with drops to the rear of every bird-rack top (behind the landing deck), galvanised ladder tray over the hot-aisle side (z 3.30), hanger rods; lower whips go down through the plinth |
| 13 | LED groups / port recesses | Real chassis: LED matrix as in the NVIDIA figure (6 rows in the upper side panels, groups of 4 + 2 bars per row; the figure does not place them per block row, so that layout was kept). Bird chassis: one LED bar + port LED beside each cell row. RJ-45, USB and OSFP ports are now recessed openings (shell walls + floor behind) on both chassis and the GB300 trays |
- 2026-10-10 03:05: production wave (DevLog-010, P2): v1 bird generator frozen as `build_rack_bird_q3400_v1.py` (writes `rack_bird_q3400_v1`, never v2); new `rk_loft.py` (tiered chassis); `build_rack_bird_q3400.py` v2; first v2 previews: stepped tiers read, but the cells read as plain boxes -> added cage wires flanking each opening (shared slot mesh) for the caged-farm look; hall layout v2 (240 slots); aisle view checked.

## 10. v2 bird rack: tiered caged farm (owner P2, 2026-10-10)

Owner (verbatim, DevLog-010): "bird racks are full filled from the top and can be tiered in depth to fill the gaps like a caged far[m], where their heads stick out for observation."

### 10.1 Layout (per chassis, chassis-local mm; 3 chassis per rack, same cabinet, top units, managers, landing deck, perch rail)

| Tier | Row | Slots | Face plane x | Feet origin x | Floor z | Columns y |
|---|---|---|---|---|---|---|
| 0 | bottom | 3 | 0 (front) | -30 | 30 | -190, 0, 190 |
| 1 | middle | 2 (offset half a column, brick bond) | -80 | -110 | 218 | -95, 95 |
| 2 | top | 3 | -160 | -190 | 406 | -190, 0, 190 |

- Slot interior 172 W x 176 H x 300 D mm (bird 0.13 m wide -> 21 mm each side; 176 since the hero-model check in section 11, was 173); row pitch 188 mm, column pitch 190 mm; floors/treads 12 mm (was 15) with the champagne label strip on the front edge (Q3400 strip between cage rows); grille between openings; cage wires (2 vertical at |y| 72 / 80 mm + 2 short horizontal per side; were 62 / 75) flank each opening; side panels (LED bar + port LED beside every slot row, true-scale FNM OSFP port, USB, 2 x RJ-45, badge "NVYDIA QX-3400B") at the front plane; rear PSUs + fans as v1. Stair-step per chassis (each row recessed further back, like bleachers), sawtooth over the 3 chassis.
- 24 birds per rack (v1: 18); 240 slots in the hall (90 / 60 / 90 per tier), 239 crowd + the hero's empty front slot.
- Render-fast: one shared slot mesh (walls, floor board with gold contact fingers, host connector, LED, lips, EMI fingers, cage wires) instanced 24 times per rack; one shared chassis-shell mesh for the 3 chassis; 41.9k triangles per rack.

### 10.2 Geometry checks (crude-film bird primitive x 2.6, bird-local mm x 2.6)

- Feet 30 mm behind the own face: at the face plane the bird cross-section is only the breast/body (top 0.167 m, half-width 0.051 m), below the 0.173 m opening; neck base and neck are fully in front of the face (x >= +7 mm); toes reach about 30 mm over the floor edge.
- Head centre 0.114 m ahead of and 0.212 m above the feet: head 0.084 m out of its own face for every tier. Tier 0 heads (x +0.053..+0.115) are in front of the tier-1 tread edge (x = 0); tier 1 heads (x -0.027..+0.035) are in front of the tier-2 tread edge (x = -80 mm); tier 2 head top reaches chassis-local z 651, below the next chassis' tier-0 floor slab (local 15 = 655), 4 mm clearance, and its x (-0.107..-0.045) is in front of that chassis' slot backs only under open space.
- Suggested neck extension (head centre 0.05 m in front of the tier-0 face): tier 0: 0, tier 1: 0.046 m, tier 2: 0.126 m; the extended heads pass in front of the recessed faces and below the next chassis (checked: tier 2 extended head z 587-649 local at x +0.02..+0.08, in front of the next chassis bezel). Neutral (0) is also collision-free. Preview proxies show the extension.

### 10.3 Hooks and data

- Per slot `HOOK_bird_<unit>_<tier>_<col>` (unit 1..3 top to bottom, tier 0 front/bottom row, 1 middle, 2 back/top row, col left to right seen from the front): location = feet origin, +X = beak; props `fill_index` (0 = highest slot; fully filled from the top), `unit`, `tier`, `col`, `slot_label`, `head_out_own_face_m` (0.0844), `head_x_rel_front_face_m`, `neck_extension_m`.
- Compatibility aliases (extra empties at the same transform, prop `alias_of`): `HOOK_cell_<u>_1_<c>` = tier-0 slot (v1 lower-row name; feet within 3 mm in z and 30 mm deeper-forward in x than v1: x 0.475 vs 0.445 rack-local, y +-0.19 vs +-0.208), `HOOK_cell_<u>_0_<c>` = tier-2 slot (v1 upper-row name, now the recessed top row). The scene's `HOOK_cell_1_1_1` (hero), `HOOK_cell_1_0_1`, `HOOK_cell_2_0_1` keep resolving.
- Rack hooks: 24 bird + 18 alias + landing pad + 4 perch + 2 rack = 49 (glb re-import check: 49).
- `hall_layout.json` v2: `cells.list` = 240 entries (one per slot; `hook` = compat alias if any else the v2 name, `bird_hook`, `compat_alias`, `occupant` crowd / hero_empty, `tier`, `unit`, `col`, `fill_index`, `slot_label`, `feet_origin`, `head_center_neutral`, `bird_yaw_deg`, `head_out_own_face_m`, `neck_extension_m`); `counts` adds `bird_slots` 240, `crowd_birds` 239, `per_rack` 24, `per_tier`. Hero slot: L07 `HOOK_bird_1_0_1` (= `HOOK_cell_1_1_1`), feet (-1.159, 6.0, 1.4727) in layout coordinates (before the scene's +1 m y shift).

### 10.4 Previews (`outputs/assets/datacenter/`, 16 samples, small sizes)

`rack_bird_q3400_front.png`, `_three_quarter.png`, `_rear.png` (720x900); `_tier_section_side.png`, `_tier_section_front.png`, `_tier_section_empty.png`, `_landing_pad.png` (960x640); hall `hall_layout_aisle_bird_racks.png` (new), `hall_layout_aisle_eye.png`, `_aisle_from_entrance.png`, `_landing_pad.png`, `_overhead_whips.png` (960x540), `hall_layout_top.png` (600x1000).

### 10.5 Open points

- The middle rows have 2 slots, leaving a 0.10 m grille gap at each end of the bay (a third offset slot would need a 0.74 m wide bay and would remove the Q3400 side panels). Could hold a small feeder/vent detail if the owner wants it closed.
- Neck extension is a suggestion for the rig (C/S); the preview proxies only approximate it with stacked neck segments.
- 2026-10-10 11:30: hero mating cell (DevLog-010 section 9 item 3): first cutaway preview with the real hero GLB showed (a) the cage wires at |y| 62 mm cut the hero body (+-66 mm) and (b) only 1 mm between the hero back (172 mm at the face) and the tread; measured the hero envelope (584k vertices, cell frame x 2.6) and fixed the v2 slot for every rack: wires to |y| 72 / 80 mm, floor/tread 12 mm so the interior is 176 mm (hook positions unchanged); lip brackets moved to |y| 60 mm; lip bar placed from the measured beak (lowest beak point 196.6 mm at x 150-170 mm). Rebuilt rack, hero cell, hall layout; glb check clean.

## 11. Hero mating cell (S2 cutaway, 2026-10-10)

Asset `assets/components/datacenter/hero_cell_mating.blend/.glb/.json` (generator `build_hero_cell_mating.py`), 0.445 x 0.208 x 0.212 m, about 1.5k triangles.

- Frame = the hero's slot hook frame (rack L07 `HOOK_bird_1_0_1` = `HOOK_cell_1_1_1`): origin = feet origin on the cell floor, +X = beak / out of the cell (tier-0 face at x = +0.030), +Y = bird's left, Z up. World transform in `hall_layout.json` `hero_cell` (location (-1.159, 6.0, 1.4727), yaw 0, before the scene's +1 m y shift).
- Collections: `CELL_BASE` (floor board, back wall + host connector + LED, +Y wall, tread with label strip, grille slice, lips, cage wires), `CUTAWAY_hero_cell` (the -Y wall, its grille slice, lip and wires: hide it for the cutaway), `HERO_LIP`, `HOST_PADS`.
- Cage lip (beak latch): nickel bar 0.126 m wide on two brackets from a hinged pivot at the tread front edge (`hero_cell_mating_lip_pivot`; rotation_euler.y = -1.2 swings it up out of the hop-out path); bar top 0.191 m above the floor at x = 0.158 m, i.e. 5.6 mm under the lowest beak point (0.197 m, measured on the hero GLB x 2.6) and 21 mm below the eye centre (0.212 m); champagne catch tab, rubber bite pad; brackets at |y| 60 mm clear the hero (<= 53 mm there).
- Host contact plate: dark FR4 plate flush with the floor (protrudes past the face under the toes) with all 60 pads of the hero pin map (`assets/hero/pigeon_osfp_dr8_pinmap.json`, OSFP MSA 60-contact edge mapped to toes and belly) projected x 2.6: TX 16 under the right foot (-Y), RX 16 under the left foot, GND 20 (talon tips and separators), VCC 4 + LS 4 on a belly pogo post at the belly-pad height (36 mm). Host pads are 15 % longer than the module pads, min 1.6 mm wide for visibility. One material per class (`MAT_dc_hero_pad_TX/RX/GND/VCC/LS`, gold, emission colour red / cyan / grey / orange / green, strength 0): animate the strength to pulse (item 5: right pads red at 14.6 s, left cyan at 16.4 s). Verified in the preview with the hero GLB: toes land on their pads.
- Hooks: `HOOK_hero_host_pads` (plate centre top), `HOOK_hero_lip` (bar top centre = beak bite point, child of the pivot), `HOOK_hero_cutaway_cam` (suggested camera 0.56 m to the -Y side, x 0.095, z 0.205 m, looking at (0.035, 0, 0.095); lens 35 mm; Blender camera axes, -Z looks at the target).
- Use in the shot: hide `CUTAWAY_hero_cell` and the L07 rack's unit-1 chassis (or the whole L07 instance) in front of the camera; the asset supplies the cell around the hero.
- Hero-model fit (crowd rigs derived from the hero should fit too): hero envelope x 2.6: 0.341 m long (x -0.173..+0.168), 0.244 m tall, +-0.066 m wide; height at the face plane 0.172 m vs the 0.176 m interior; width +-0.066 vs wires at +-0.072 and walls at +-0.086 m.
- Previews (`outputs/assets/datacenter/`, 960x640, 16 samples): `hero_cell_mating_cutaway.png` (from the suggested camera, wall hidden, hero GLB), `_cutaway_pads_lip.png`, `_lip_closeup.png`, `_pads_top.png` (hero hidden; tread covers the rear pads in plan view), `_front_closed.png`.
- 2026-10-10 17:10: v4 wave (DevLog-020, owner: "find and inspect GB300 racks better and put in proper OSFP ports and fiber cable bundles"): re-researched (Lenovo UG Fig. 2/7/8 re-read, Gigabyte XN15-CB0 spec, NVIDIA MMS4A20 flat-top OSFP page + figure); new `rk_cabling.py`; GB300 FRONT_CABLING; first previews: fibre bundles ended in yellow drops over the cold-aisle edge (40 yellow "arches" over the aisle, heavy and in the flight path) -> bundles now turn back over the roof into drops directly under the raceway; fibre exit control points smoothed; bird-rack patch-panel trunks; hall raceway drops for every GB300; rebuilt GB300, bird rack, hall; GLB check clean.
- 2026-10-10 18:40: v4 audit corrections applied (12.6); GB300, real Q3400 (FNM cage), bird rack, hall rebuilt; GLB check clean.

## 12. v4: GB300 front ports and fibre cabling (2026-10-10)

Owner (verbatim, DevLog-020): "find and inspect GB300 racks better and out in proper OSFP ports and fiber cable bundles." Refs: `research/racks_refs.md` section 6.

### 12.1 What the compute-tray front carries (refs -> model)

| Port (per compute tray) | Count | Ref | Model |
|---|---|---|---|
| ConnectX-8 800G OSFP (scale-out) | 4 (2 OSFP cards x 2) | Lenovo UG Fig. 2 (A), Gigabyte XN15-CB0 (B) | cages at true OSFP size (opening 22.7 x 13.1 mm, fits the 22.58 x 13.0 mm module; QSFP112 cage opening 18.4 x 9.3 mm); populated with flat-top (RHS) single-port modules, yellow pull tab, green MPO-12/APC + black boot, 3.0 mm yellow SMF jumper; compute trays 5 and 16 carry black OSFP dust caps instead |
| BlueField-3 B3240 QSFP112 | 2 | Lenovo UG (A) | port 1 module + MPO + jumper (front-end network), port 2 dust cap |
| RJ-45 (OS mgmt, BMC, BF3 OOB) | 3 | Lenovo UG (A) | blue slim Cat6 with plugs and boots |
| NVLink switch tray | no optics | Lenovo UG Fig. 7/8 (A) | unchanged: no front OSFP; 2 RJ-45 (BMC, Eth mgmt 0) cabled with Cat6; NVLink stays on the rear cartridges |

Per rack (after the audit fixes): 60 OSFP populated + 8 capped on trays 5 and 16 + 4 capped on the pull-out tray 14; 17 QSFP112 populated + 17 capped (+ 2 capped on tray 14); 77 fibre jumpers (left bundle 30, right bundle 47); 67 Cat6.

### 12.2 Dressing

- Fibres leave each connector with a straight run, a true quarter arc sideways (radius up to 35 mm), a horizontal run in front of the ports and a quarter arc up (radius up to 30 mm) into two vertical bundles at the front sides (rack-local x 0.587, y +-0.258, outside the tray ears); measured minimum bend radius 22-24 mm (fibre), 30 mm (Cat6), recorded in `front_cabling.min_bend_radius_mm`; hex-packed in joining order, black velcro straps every 0.25 m sized to the members below each strap. At the top (z 2.261) both bundles turn back over the roof (radius about 29 mm) and end inside the overhead raceway drops at rack-local (0.10, +-0.20, 2.42) (drop bottom 2.30 m, directly under the yellow raceway; `hall_layout.json` `overhead.raceway_drops` kind `gb300_front_left/right`).
- Cat6: two outboard bundles (x 0.632, y +-0.262) up to just below the management switches (U45 left, U46 right), then fanned out to individual ports with plugs.
- Bird racks: three yellow MPO trunks from the patch-panel rear through the cabinet top and roof (behind the landing deck) into the existing bird-rack raceway drop. The bird chassis themselves get no fibre: the story is that the birds link optically through free space ("no fibre required"), so fibre appears only on the compute racks and on the conventional switch side (patch panel, uplinks). The mgmt Cat6 inside the bird racks stays (switch management), as in v1.

### 12.3 Interface for S (stable)

- Same asset paths, collection names, hooks and `hall_layout.json` schema. New: collection `FRONT_CABLING` in `rack_gb300_nvl72` (visible by default; hide it for a clean rack); `rack_gb300_nvl72.json` `front_cabling` (counts, bundle positions, end points); `overhead.raceway_drops[*].kind` (`bird_rack_rear`, `gb300_front_left`, `gb300_front_right`; 50 drops total); the overhead asset now has 40 more short drops above the GB300 roofs.
- The cabling sticks out of the rack front by up to 0.12 m (x <= 0.652 rack-local), leaving 1.96 m of the 2.2 m cold aisle clear; the rack bounding box grows accordingly (1.192 m deep, z to 2.42 where the fibres enter the drops). The `HOOK_compute_NN_front` hooks (x 0.495) now sit inside the plugged modules (x up to 0.55).
- S needs to rebuild the hall (re-link `rack_gb300_nvl72`, `rack_bird_q3400`, `overhead_infrastructure`); nothing else changes.

### 12.4 Render cost

All cabling is merged into 4 meshes per rack (modules/connectors, fibres, Cat6, straps): 5-sided tubes, no modifiers; the GB300 goes from 163.8k to 258.6k triangles (+94.8k for modules, connectors, pull-tab loops, fibres, Cat6 and straps). The hall instances the rack collection, so the 20 GB300 racks share one copy of the mesh data.

### 12.5 Previews (`outputs/assets/datacenter/`)

`rack_gb300_nvl72_front.png`, `_three_quarter.png` (720x900); `_ports_macro.png`, `_cable_dressing_left.png`, `_cable_fanout_top.png`, `_front_closeup.png` (960x640); hall `hall_layout_gb300_front_cabling.png`, `hall_layout_fiber_drops.png`, `hall_layout_aisle_from_entrance.png` (960x540).

### 12.6 Fresh-context audit (one agent, 2026-10-10) and corrections

| # | Finding | Correction |
|---|---|---|
| 1 | Fibre bend radius in the code about 10-20 mm, not the claimed 35 mm; hooks at the boots in the macro | New `arc_route()`: straight, true quarter arc sideways, run, true quarter arc up; bundles moved forward (fibre x 0.587, Cat6 x 0.632); measured minimum 22-24 mm fibre, 30 mm Cat6 (recorded in the JSON); docs corrected |
| 2 | Pulled-tray variant broken (tray 14's connectors stayed at the seated face) | Compute tray 14 (and NVLink switch tray 5) are the pull-out trays: capped / unplugged; tray 14's caps are a small mesh parented to the seated tray (VARIANT_seated_slots) and to the pull hook (VARIANT_pulled_trays) |
| 3 | OSFP cage opening 22.2 x 12.8 mm smaller than the module | OSFP cage 23.9 x 14.3 (opening 22.7 x 13.1), QSFP cage 19.6 wide (opening 18.4); also the chassis FNM ports |
| 4 | Port y positions off by up to 22 mm vs Fig. 2 | OSFP -190 / -144, +114 / +160; BF3 QSFP 156 / 178; BF3 RJ-45 120 |
| 5 | NVLink switch tray RJ-45 assignment untraceable | stated as a design choice |
| 6 | Pull tab a flat strap, NVIDIA figure shows a wide loop | 20 mm wide loop sweeping 38 mm forward and down |
| 7 | JSON still said "no front cabling"; sources/accuracy cited section 2 only; BF3 module type unsourced | simplification removed; sources add section 6; `accuracy.front_cabling` added; BF3 module listed as a design choice |
| 8 | Diagonal dressing across trays above; Cat6 wide loops | horizontal run first (rise < 10 mm before the up-arc); Cat6 runs in its own plane in front of the fibres |
| 9 | Interface side effects | noted in 12.3 (aisle protrusion, front hooks inside the modules) |
