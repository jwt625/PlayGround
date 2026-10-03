# DevLog-003 fab_test: semiconductor test and packaging equipment, wafers (v1.0 library)

Owner: fab_test agent. Contract: `assets/ASSET_SPEC.md`. Date started: 2026-10-02.

## 1. Plan

Build seven asset families as separate .blend files in `assets/components/fab_test/` using one local helper module (`scripts/assets/fab_test/ft.py`, wraps `_common/common.py`). Hardware at real size in mm; every hook is an empty `HOOK_*`; assembler-drivable values are `p_*` custom properties on the root empty with drivers on the hooks.

| # | asset_id | file(s) | notes |
|---|---|---|---|
| 1 | probe_station_cm300_style | probe_station_cm300_style.blend | CM300xi-ULN style, real size from FormFactor facility planning guide drawings (PN 188-101-6) |
| 2a | wafer_300mm | wafer_300mm.blend | VARIANT_blank (mirror Si, notch, bevel) and VARIANT_siph (reticle/die grid, per-die objects) |
| 2b | wafer_tape_frame | wafer_tape_frame.blend | VARIANT_whole, VARIANT_diced |
| 2c | foup | foup.blend | SEMI E47.1 style |
| 2d | wafer_cassette | wafer_cassette.blend | open 25-slot cassette |
| 3 | probe_card | probe_card.blend | VARIANT_full_needles, VARIANT_light |
| 4 | dicing_saw_station | dicing_saw_station.blend | dual-spindle style generic |
| 5 | die_bonder_station | die_bonder_station.blend | plus ASSET eic_pic_bonding_close in its own file |
| 6 | fau_attach_station | fau_attach_station.blend | |
| 7 | reflow_oven_line | reflow_oven_line.blend | plus conveyor modules and WIP carrier (own files) |

## 2. Checklist

- [x] probe_station_cm300_style: built, saved, photo comparison done (see section 8)
- [x] wafer_300mm (blank): built, saved
- [x] wafer_300mm_siph (668 dies, die_map_tools.py, diemap JSON): built, saved
- [x] probe_card (VARIANT_full_needles 384, VARIANT_light 24): built, saved
- [x] wafer_tape_frame (whole + diced), foup (25 wafers, door), wafer_cassette: built, saved
- [x] reflow_oven_line (10 zones, per-zone glow materials, board), conveyor_modules, wip_carrier: built, saved
- [x] die_bonder_station, eic_pic_bonding_close: built, saved (generic; glass variant visible in previews)
- [x] fau_attach_station: built, saved
- [x] dicing_saw_station: built, saved (p_spin, HOOK_cut_line)
- [x] station vs photo comparison (section 8), metadata JSON complete (dimension tables, hooks, material slots, props)
- [ ] fresh-context audit: NOT run (token budget); open

## 3. Progress log

- 2026-10-02: read spec, storyboard S5, crude `cm300`/`s5`, common.py; fetched CM300xi-ULN facility planning guide (PN 188-101-6) and read the station drawings.

- 2026-10-02 13:25: helper layer ft.py (+ wafer_lib.py, die_map_tools.py) written; station, wafer_300mm, wafer_300mm_siph built and saved. Resumed after API limit; remaining order per coordinator.

- 2026-10-02 17:50: station compared with reference photo (layout matches: fascia, drawer with chuck+wafer, deck, 4 positioners, arch+column, monitor). Streak artifacts in wafer previews traced to depth precision at far camera: preview camera near clip fixed in ft.render_previews.

- 2026-10-02 18:05: probe_card built and checked (needle close-up fine; PCB/stiffener/window OK).

- 2026-10-02 18:25: wafer_tape_frame, foup, wafer_cassette built (FOUP shell uses alpha-blend, EEVEE transmission rendered opaque).

- 2026-10-02 18:50: reflow_oven_line, conveyor_modules, wip_carrier built.

- 2026-10-02 19:30: die_bonder_station, eic_pic_bonding_close, fau_attach_station, dicing_saw_station built; all 14 blends open headless with no missing files; category size 69 MB (previews 19 MB).

## 4. Sources
Per-asset JSON lists sources with access dates. Key: FormFactor CM300xi-ULN Facility Planning Guide PN 188-101-6 (station drawings); ePAK eFOUP300 listing (FOUP dims); DISCO tape frames DTF2-12 (400/350/1.5 mm); DISCO DFD6361 (1200x1550x1800); Heller 1936 MK5 listing (oven envelope); SEMI M1 wafer geometry; Wentao's reference photos (CM300 photo, bonder photo, Broadcom flow slide).

(see per-asset JSON; summary filled in at the end)

## 5. Decisions
- Wafer variants as separate .blend files (wafer_300mm blank, wafer_300mm_siph with 668 die objects) instead of VARIANT sub-collections; tape frame holds VARIANT_whole / VARIANT_diced; probe card holds VARIANT_full_needles / VARIANT_light.
- Die recolouring through object colour (rgb = state, alpha = glow) on one shared material; scripts/assets/fab_test/die_map_tools.py gives exactly round(N/10) green dies.
- Reticle 26x33 mm holds 3x3 dies at 7.1x9.1 mm pitch plus test-structure bands (layout choice, level C).
- Hooks are driven by p_* properties on the root (drivers); keyframe the p_* props.
- Preview shadows disabled for wafer previews and near clip raised for far shots (EEVEE depth-precision streaks on 20 um features).

## 6. Open questions for Wentao
1. Die size 7 x 9 mm with a 26 x 33 mm field: 3 x 3 dies per field is my layout; confirm or give the real PIC layout.
2. Probe station chuck/drawer travel (500 mm) and positioner radius are estimates from the photo.
3. Oven length 4.5 m, 10 zones (8 heat + 2 cool) chosen inside the 3-5 m brief; confirm.
4. FAU, bonder and saw are generic (no vendor machine copied); give a reference photo if a specific look is wanted.

## 7. Audit corrections

## 8. Station vs reference photo (differences)
- Matches: black cabinet with silver deck and front lip, two round front posts, left knob + thumb-lever slot, right knob + e-stop + label, central drawer opening with silver chuck and wafer on a tray, four positioners around the platen with arms to the centre, black arch gantry with column, monitor at top right, keyboard shelf.
- Differences: no vendor logos (generic text WAFER PROBE / CM300-STYLE ULN / VUE-STYLE); chuck lacks the clamp hardware, tubing and side plates seen in the photo; positioners are simplified (no micrometer rings/detail, arms are plain rods); microscope column is a plain black block (no optics detail, label plate generic); deck has no brushed texture; deck rails and screws simplified; cabinet interior empty; photo shows a left-side door and a keyboard tray partly in frame that are only approximated; drawer tray handle and latches simplified; probe needle tips are tiny and barely visible at preview scale.

## 9. Status at hand-off (exact)
- Done and checked visually (contact sheet): probe_station_cm300_style (+photo comparison), wafer_300mm, wafer_300mm_siph, wafer_tape_frame, probe_card, foup (alpha shell), wafer_cassette, reflow_oven_line, conveyor_modules, wip_carrier, die_bonder_station, eic_pic_bonding_close, fau_attach_station, dicing_saw_station.
- Partial: needle tips on the station are not visible in previews (micron-scale tips, close-up framing poor); eic bumps close-up framing poor; foup shell washed out (alpha blend); no fresh-context audit.
- Not started: LOD_low variants, cantilever probe-card variant, dual spindle on the saw.
