# DevLog-003 lab_office

Category agent: lab_office. Spec: `assets/ASSET_SPEC.md`. Outputs: `assets/components/lab_office/`, `scripts/assets/lab_office/`. Shared local helpers: `scripts/assets/lab_office/lo_common.py` (all geometry helpers take mm).

## Plan and checklist

| # | asset | status |
|---|---|---|
| 1 | whiteboard_big | built, previewed, checked (2026-10-02) |
| 2 | bench_oscilloscope | built, checked (probe in ITEM_probe_ch1) |
| 3 | sampling_scope_dca | built, checked |
| 4 | lab_bench (with scope placement) | built, checked (preview appends the scope at HOOK_scope_slot; blend itself has no external link) |
| 5 | conference_room | built, checked (38.6k tris; 9 chairs + 6 hidden realistic-variant chairs, rigged) |
| 6 | lab_clutter | built, checked (12 items, 30.9k tris) |
| 7 | office_cubicle_set (stretch) | built, checked (27.9k tris) |

## Decisions

- Screen materials (both scopes): `MAT_lab_office_screen`, node `SCREEN_IMAGE` (empty Image Texture), Value node `SCREEN_FAC` default 0 (idle dark screen, so the empty slot is not magenta). Assembler: plug the image, set `img.source = 'SEQUENCE'`, `node.image_user.frame_duration = N`, `frame_start = 1`, `frame_offset` for the start frame, `use_auto_refresh = True`, then set `SCREEN_FAC` to 1.0. Emission strength in node `SCREEN_EMISSION`. UV 0..1 over the active area, 16:10 (matches the 384x240 eye sequences).
- Whiteboard: `MAT_lab_office_whiteboard`, same recipe with `BOARD_FAC`; the image is multiplied over the smudged white board; writing area is exactly 16:9 so the 640x360 graph is not stretched (board outer is 3.0 x 1.71 m, the spec said about 3.0 x 1.5 class).
- Screen planes are kept 1.5 to 2 mm in front of the backing geometry (0.2 mm gave z-fighting in previews) and do not cast shadows.
- All hooks are empties in the parent's local frame; parts are meshes with baked rotation, location only.

## Progress log

- 2026-10-02: whiteboard_big built (3.9k tris), eraser, 5 markers, tray; graph_s2 plugged in previews only.
- 2026-10-02: bench_oscilloscope built (26k tris), eye_s1 plugged in previews; panel, rear, probe checked.
- 2026-10-02: sampling_scope_dca built (18.7k tris).
- 2026-10-02: run resumed after an API usage limit; remaining order lab_bench, conference_room, lab_clutter, cubicle set.

- 2026-10-02: rebuilt bench_oscilloscope (probe now in sub-collection) and DCA previews; lab_bench built (9.3k tris, width parametric via 2nd argv, default 1800).
- 2026-10-02: conference_room built (rigged chairs in lo_furniture.py; hidden variants keep hide_viewport on so matrices evaluate).
- 2026-10-02: lab_clutter built; microscope head re-centred over the stage after the first preview.
- 2026-10-02: office_cubicle_set built; common area re-spaced after the first preview.

- 2026-10-02: verification pass: every .blend opens headless with no images/libraries (no external files), identity transforms on all mesh objects except the 60 linked plant leaves (intentional instances in office_cubicle_set); .blend1 backups removed; JSON extended with accuracy_level and preview list.

## Assembler notes

- Scale: all assets real size in metres; origins: instruments and clutter items at the feet (z = 0), lab_bench worktop top at z = 0.9 m with HOOK_scope_slot, whiteboard_big at the frame bottom centre on the wall plane (-Y into the room, suggested bottom edge height 0.9 m), conference_room at the floor centre, office_cubicle_set at floor level at the centre of the shared partition.
- Screens: bench_oscilloscope_screen and sampling_scope_dca_screen (MAT_lab_office_screen, 16:10), whiteboard_big_screen (MAT_lab_office_whiteboard, 16:9, node BOARD_FAC), conference_room_tv_screen (16:9), and per-instrument LCDs in lab_clutter (MAT_lab_office_screen_<item>). Plug-in recipe is in Decisions above. Because each .blend has its own MAT_lab_office_screen, link/append them per asset (they are different datablocks after appending: .001 suffix is expected).
- Eye sequences are 384x240 (16:10) so they map 1:1; graph_s2 640x360 maps 1:1 on the whiteboard; spec_s5 480x270 (16:9) fits the TV.
- Hooks: scopes HOOK_screen_center, HOOK_probe_tip (bench scope probe in ITEM_probe_ch1), HOOK_ch1_bnc, HOOK_trigger_level_knob, HOOK_power_button; DCA HOOK_screen_center, HOOK_optical_in, HOOK_electrical_in_1/2, HOOK_dust_cap; bench HOOK_scope_slot, HOOK_clutter_left/right, HOOK_worktop_center, HOOK_esd_mat_snap, HOOK_monitor_screen_center, HOOK_operator_stand, drawer handle hooks; conference HOOK_seat_far_1..4, HOOK_seat_head, HOOK_light_1..6, HOOK_tv_screen_center, HOOK_door_hinge, HOOK_camera_front, table edge hooks; whiteboard HOOK_board_center, HOOK_board_top_left, HOOK_tray_center, HOOK_marker_grip.
- Variants: conference_room VARIANT_table_storyboard (default, visible) and VARIANT_table_realistic (hide_render on), VARIANT_front_wall (hide_render on), ITEM_chairs_near; sampling_scope_dca ITEM_variant_rack_ears (hide_render on).
- Chair rigs: one armature per chair under its chair_* empty; bones base, swivel (pose rotation_euler Y), back_tilt (pose rotation_euler X). Yaw the chair_* empty.
- Lights: no Blender lights are stored; use HOOK_light_N (ceiling panels) for area lights. Previews use temporary preview lights only.

## Deviations from the brief

- whiteboard_big outer size is 3.0 x 1.71 m (16:9 writing area), spec said about 3.0 x 1.5 m.
- Screen idle default: SCREEN_FAC = 0 (not magenta); empty image node is still present.
- lab_bench blend does not contain the scope (no external links); the preview appends the scope at HOOK_scope_slot to show placement.
- Some preview PNGs are 400-700 KB (noisy render); category total is about 18 MB, under the 25 MB limit.
- bench_oscilloscope bbox in the JSON (756 mm deep) includes the probe lying on the bench; hide ITEM_probe_ch1 for the bare instrument (about 400 x 150 x 245 mm incl. handle).

## Sources

See each asset JSON (`sources`, `dimension_table`). Main: Keysight 86100D datasheet (221 x 426 x 530 mm body), Tektronix 4 Series MSO spec page (405 mm width, 155 mm depth, 13.3 in screen 289 x 165 mm), Keysight InfiniiVision 4000 X page (12.1 in screen), public catalogue sizes for ESD benches (72 x 30 / 36 in, 30-36 in height).

## Open questions for Wentao

- Whiteboard height 1.71 m instead of 1.5 m (16:9 writing area): OK, or letterbox the graph on a 1.5 m board?

## Audit corrections

(no fresh-context audit run in this session; coordinator may request one)
