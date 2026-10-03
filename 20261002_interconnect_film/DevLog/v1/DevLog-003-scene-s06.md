# DevLog-003-scene-s06: scene S6 "Fiber: the tray" (film 50-60 s)

| Field | Value |
|---|---|
| Date | 2026-10-02 |
| Owner | scene S6 assembly agent |
| Files | `scripts/film_v1/s06_fiber.py`, `scenes/v1/s06_fiber.blend` (about 63 MB), `scenes/v1/s06_fiber_contact.png` (8 stills, 4x2 at 540x675) |
| Build | `/Applications/Blender.app/Contents/MacOS/Blender -b --python scripts/film_v1/s06_fiber.py -- scenes/v1/s06_fiber.blend` (env `S06_LOD=high` builds the 9,072-strand variant, cost test only) |
| Status | complete: all storyboard beats built, contact sheet viewed and fixed once |

## Plan / checklist

- [x] hall (datahall_environment + data_hall rig), tray p_slide 1.5-3.2 s bezier
- [x] glowing patch cord dolly + jammed connector macro (0-1.5 s)
- [x] fiber pile grown 2.0-6.0 s (600-strand LOD, GN visible-length growth), squirm via p_squirm
- [x] Gary kneel_and_tie (kneel 0-24, loop 24-84), tie gun in hand, tie box, tied bundles appearing 4.6 / 5.8 / 7.0 s
- [x] fiber threaded through hole 2 (healed radius), hole radii on the film schedule
- [x] Manager walk-in, AOC whip on HOOK_whip_grip_L, arm_whip, CRACK FX, Gary jolt_hit
- [x] NVYDIA customer slap (contact 9.147 s), Manager stagger + head snap + shock face, OPENAY / ANTHROPY hug_leg
- [x] captions, narration chunks, labels, cards, FX notes from crude s6 + timecode(6)
- [ ] not done: end-face macro (x40 LC ferrule item) not used; no compositor bloom (glow is plain emission); no fresh-context audit

## Layout and scale (documented)

- World: aisle along x, cold aisle y in [-0.6, 0.6], row b racks at y > 0.6, fronts facing -y. Characters face -y at yaw 0 (Gary yaw pi faces the tray).
- Hall changes (own scene only): row a racks hidden for x in [-3.6, 4.8] (the real 1.2 m aisle cannot hold a 2.4 m pile, four characters and the camera); hall rack b16 (x = 0.3) hidden and replaced by the pull-out tray rack stub (1.0 x 0.6 x 1.5 m, so it is shorter than its 2.236 m neighbours); containment roof, under-floor, signage and racks outside x in [-6, 7.5] hidden (shadowing / render cost). Data_hall rig rotated 90 deg about z, centred (1.4, -0.9).
- Tray: ROOT at (0.3, 1.1, 0); `p_slide` 0 -> 1 keyed 1.5 -> 3.2 s, bezier (drawer front travels y 0.67 -> 0.12 m, 0.55 m travel).
- Fiber pile: `fiber_spaghetti_generator` LOD_low (600 strands, profile 3), strand radius 4 mm (real cord 1 mm radius: x4 scale-up; the asset's own LOD default is 3.5 mm). Root parented to `HOOK_slide` of the tray (offset z -0.7) so it follows the drawer. Needed z stretch x2.12 (asset tray lip is 0.35 m, drawer top is 0.744 m) is applied inside the modified node group. Growth: added inputs Fill / Strands / Z Stretch / Spread to `NG_fiber_spaghetti` (in this scene only): points beyond each strand's grown length are deleted; Fill keyed 0 (2.0 s) -> 1 (6.0 s), strands staggered (Spread 3, each strand grows over about 1 s). `p_squirm` 0 -> 0.1 (2.0 s) -> 0.8 (4.4 s) -> 1.0 (8.0 s), `p_squirm_speed` 3; measured mean vertex motion 4.4 mm per 0.2 s at 6 s.
- 9,072-strand variant measured at 17-22 s per frame at 540x675 (GN evaluation, 2.25 M tris): rejected, 600-strand LOD used. The 9,072 count stays a footnote caption (own estimate, as in the crude).
- Intro cord and connectors: `fiber_cables_and_trays` VARIANT_patch_cords (2.0 mm cord, glowing: jacket material replaced by a cyan emission material) and VARIANT_patch_panel at x5 (cord 2.0 m, plug about 8 cm), placed (-1.7, -1.2, 1.3), yaw -90 deg; macro needs scale-up because the overlay plane is 0.8 m in front of the camera. Jam: from 0.9 s the whole cord moves 8 cm back and the plug yaws 0.22 rad and rattles (+-0.06 rad every 0.06 s) in front of the first LC port, small impact_stars burst. Visible only 0-1.55 s.
- Tied bundles: `bundle_cords` + `cable_tie` meshes of the same asset cloned x8.
- Whip: x2.5 cable thickness (3 mm real), length 3.0 m unscaled; bone wave = library `ACT_whip_crack_demo` as an NLA strip with its crack frame (36) at 8.0 s; attached to `HOOK_whip_grip_L` with rotation z = 180 deg (whip -y along the hand +y). Manager at (2.55, -1.55), 2.3 m from Gary.
- Thread fiber: NURBS curve, 6 mm bevel radius (real: 1 mm), hole 2 radius 0.3 x 0.0406 m = 12.2 mm, fits. Three Hook modifiers: middle points follow an empty parented to `HOLE_hole_2` (slides along the hole axis 0.6 m -> 0 between 5.4 and 6.6 s: the threading), end points static on the pile. Hole axis is tilted up toward the back (Gary leans forward), so the through section follows the axis and the outside runs horizontally.
- Holes: `r = max(0.3, 0.9 ** floor((50 + t - T_shot) / 1.5))` keyed in 1.5 s steps (CONSTANT) for T_shot 9.2 / 19.2 / 28.95 / 38.9 / 49.1 s: holes 1, 2, head at 0.3 for all of S6; hole 4 steps 0.46 -> 0.41 -> ... (floor reached at 56 s); hole 5 steps 0.94 -> 0.85 -> ... (about 0.5 at 60 s).

## Timeline (scene-local seconds)

| t | content |
|---|---|
| 0.0-0.9 | dolly along glowing cord (lens 35); lab FIBER, SMF-28 card, narration |
| 0.9-1.5 | push-in on jam (lens 50), CONNECTOR JAM label, stars |
| 1.5-3.2 | tray slides (camera in front of rack), fibers appear from 2.0 s |
| 3.2-5.6 | Gary kneels (3.2-4.0) then tying loop; 9,072 footnote card 3.2-8.0 |
| 5.6-7.4 | camera behind Gary on hole 2 (35 -> 50 mm); thread 5.4-6.6 s; Manager walks 3.2-7.2 along the aisle (x 7.55 -> 2.55) |
| 7.2-8.0 | arm_whip at speed 0.79 (crack at action frame 19 = 8.0 s) |
| 8.0-8.6 | CRACK 8.1-8.7, impact_stars + shockwave at Gary, Gary jolt_hit (5 frame blend from kneel), dread face |
| 8.6-10 | wide (lens 21) from y = -4.05; NVYDIA walks in, slap contact 9.147 s, SLAP 9.1-9.7, Manager stagger + shock + head yaw snap 0.7 rad, OPENAY walks in and hug_leg from 9.0 s; labels THE CUSTOMER 8.6-10, CUSTOMER 8.9-10 |

## Measured cost (M-series, Blender 4.2.3 EEVEE Next, standard preset 24 spp)

- 540x675: 1.6-3.5 s per frame; first frame of a process +5 s shader compile.
- 1080x1350 steady state: tray/macro frames 4-5 s, wide frames 6.5-7.7 s, pile + characters frames 7.6 s. Hall = about 6 s of 7.6 s (hall hidden: 1.7 s; characters hidden 6.4 s; pile hidden 6.5 s). Culling 75 rack/under-floor objects saved about 1 s. Most expensive single hall pieces: floor tiles (about 1.3 s), rack doors and perforated tiles (dithered alpha). Whole scene stays under 8 s per frame, above the 1-5 s target.

## Asset problems found

- `fiber_cables_and_trays` patch cord: plug b group is built with boot and cord pointing away from the cord body (cord stub 0.6 m beyond the jackets mesh at x5; gap to the S-section). Worked around by negative y scale on `cord_2p0mm_plug_b_group` (mirror). Plug b is also exactly on the panel centre, not on a port (HOOK_panel_port_lc_1 is 0.95 m (x5) off to the side); I moved the cord root to the port.
- Same asset: panel port hook z (1.41 m at x5 placement) is above the plate centre z; ports sit near the top edge.
- `fiber_spaghetti_generator`: pile default is 2.4 m wide at real size, strand start points are inside a 0.35 m high tray (needs a z stretch to sit on a 0.74 m drawer); LOD_low radius 3.5 mm. No growth input exists (added here).
- `tray_fiber_pullout`: dark steel on dark racks reads poorly; added one spot light. Rack stub is 1.5 m tall vs 2.236 m hall racks.
- `datahall_environment`: cold aisle is 1.2 m wide, so S6 needs racks hidden (above).
- `asm.show()` fails at finalize (`Collection.hide_render` is not animatable in 4.2). Worked around by monkeypatching `asm.show` in this script to key per-object visibility via `blender_lib.V` (skipping objects hidden on purpose). Needed framework change: asm.show should key the objects of the collection (and skip hidden ones).
- `aoc_whip`: thin (3 mm) at real size; demo bone action is a generic wave, tip path not tuned to Gary's body (whip passes at belt height at the crack, sags to the floor afterwards).
- Characters: `jolt_hit` is an upright pose, so Gary stands up from the kneel on the whip hit (5-frame NLA blend); NPC `hug_leg` position relative to the NVYDIA leg set by eye (hugger placed 0.40 m left, 0.10 m ahead of NVYDIA).

## Deviations from the storyboard / crude

- FX caption 4.4-8.0 shortened to 4.4-7.4 s to avoid overprinting the 7.4-8.7 caption at the same position (crude overlaps them).
- No end-face macro (x40 LC ferrule item from fiber_connectors); the jam is shown with the patch cord LC plug and the panel LC ports of fiber_cables_and_trays.
- Manager walk-in is a real walking gait over 4 s (3.2-7.2 s, 5 m at the gait speed), so he is visible in the far background of the 3.2-7.4 shots before 6.4 s; crude had him cover 2.9 m in 0.8 s.
- Gary is hidden before 3.2 s (as in the crude); Gary's fiber hook-through slides in rather than appearing at 5.6 s.
- Hall modified as above (racks hidden, roof hidden).

## Open questions

1. Is a hall with a cleared bay acceptable, or should the 1.2 m aisle be kept and the action moved to the open floor beyond the end doors?
2. Bloom: no compositor glow applied (the framework does not set it); glow reads as flat cyan. Apply `RP.setup_comp` in the final render?
3. Whip tuning (tip hits Gary's back at belt height at 8.0 s; thickness x2.5) and the NVYDIA / OPENAY staging want a look at full resolution.

## Progress log

- 2026-10-02: read brief, storyboard, assets; first build (hall, tray, pile, Gary, Manager) rendered; found `asm.show` failure and worked around it.
- 2026-10-02: intro cord and jam macro built; found plug b orientation defect and mirrored it.
- 2026-10-02: whip, customers, slap and crack FX, hole thread, props, bundles built; thread rebuilt with hook modifiers (rigid curve parented to the hole had huge lever arms).
- 2026-10-02: cost profiled (hall dominates), high-count variant rejected (17-22 s per frame), culling added, contact sheet (8 stills) viewed and fixed (labels, light). Scene finished.

## Audit corrections

(no fresh-context audit run)

## v1.1 update (2026-10-02)

See `DevLog/v1/DevLog-004-s06-new-assets.md` for the full record. Summary: intro uses the new MPO assets (plug jams crooked into an adapter plate); whip is a baked verlet chain (crack at 8.067 s, contact 8.300 s, 3-frame hit-stop, SFX markers `SFX_whip_crack` / `SFX_whip_hit_thump`), bundle x7 thicker; Manager moved 0.5 m closer to Gary (x 2.05), visible from 3.0 s with a 6.7-7.4 s cutaway; NVYDIA customer is `npc_nvydia_leather`; eased camera with handheld noise and hit shake; glow emission reduced. Draft p50 cost 0.5-1.4 s per frame, up to 2.1 s on dense whip frames. Contact sheet `scenes/v1/s06_fiber_contact.png`, whip strip `scenes/v1/s06_fiber_whip_strip.png`.
