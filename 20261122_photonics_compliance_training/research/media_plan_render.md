# Media plan: Blender renders (2026-11-22)

All files: `course/media/render/*.jpg`, 1920 x 1080, JPEG q88, 58-149 KB. Registry: `course/media/credits_render.js` (kind `render`, credit "Rendered in Blender for this course", CC BY 4.0, project asset). Scenes are built only from scripts in `course/tools/blender/` (Blender 4.2.3 LTS, Cycles, Metal GPU, adaptive sampling, OpenImageDenoise); run them through the slot wrapper with `-b --factory-startup --python <script> -- out=<path> [args]`. Lighting HDRIs: Poly Haven `studio_small_09` and `hospital_room` (CC0, 2k), cached outside the repo (`~/.cache/photonics_compliance_hdri/`).

Coordinates below are normalized to the image (x right, y down, 0..1). Points for the perspective shots were projected through the scene camera with `project_points.py`; points for the orthographic micrographs are computed from the camera field.

## 1. Orientation choice cards (Ferroelectric, module "Crystal Orientation Awareness")

| File | Shows |
|---|---|
| `tfln_die_fixture_0deg.jpg` | Correct: die +Z arrow parallel to fixture +Z engraving, waveguide on the IN/OUT slot axis, pad sets toward RF |
| `tfln_die_fixture_90deg.jpg` | Die rotated 90 degrees counterclockwise: facets against the pocket walls, +Z arrow points along the fiber axis |
| `tfln_die_fixture_180deg.jpg` | Die rotated 180 degrees: +Z antiparallel, label upside down, pad sets face away from the probe side, waveguide 0.8 mm off the slot axis |

Die layout (all die renders): x-cut LNOI Mach-Zehnder modulator after Wang et al. 2018 / Xu et al. 2022: edge coupler, 1x2 MMI, two arms that pass around the west pad set and dive under their own ground into the two gaps of a straight 3.2 mm traveling-wave GSG electrode (ground 300 um / signal 40 um / ground 300 um, 8 um gaps), mirror image to the 2x1 MMI and output; the electrode tapers in line at both ends to rectangular GSG pad sets at 150 um pitch (RF input west, termination east), so no waveguide is routed into a pad. Fixture: 40 x 28 x 6 mm, slots on the waveguide line, IN/OUT at the slots, M3 screws, vacuum port and hose barb. Camera, lights, fixture and dust are identical across the three (only the die rotates).

- Step type: `cards`, `style: 'grid'`, `mode: 'choose'`, prompt "Which die has been loaded in accordance with the crystal orientation policy?" Correct card: 0 degrees.
- Shared hotspot points (all three images): fixture +Z engraving (0.377, 0.279); fiber slot IN, mouth at the pocket (0.419, 0.557); fiber slot OUT, mouth at the pocket (0.653, 0.650); IN label (0.225, 0.598); OUT label (0.813, 0.835). Die +Z arrow: 0 deg (0.548, 0.516), 90 deg (0.501, 0.559), 180 deg (0.529, 0.633). Modulator electrode centre: 0 deg (0.535, 0.595), 90 deg (0.552, 0.580), 180 deg (0.542, 0.554).
- 0 degree image only: RF input pad set (0.474, 0.570); termination pad set (0.597, 0.620); input MMI splitter (0.449, 0.560); output MMI combiner (0.622, 0.629); arm passing around the RF pads (0.477, 0.550).
- Caption (0 deg): "Figure 1. A die in its approved orientation. Both arrows agree. This is what alignment looks like."
- Card `why` lines: 90 deg: "The die's z axis now runs along the waveguide. Light will still go in. The electro-optic coefficient you designed for will not." 180 deg: "Plus z is now minus z. The die did not complain. Your bias polarity, your poling direction and your RF pads might."

## 2. Probe station (Ferroelectric, "Crystal Orientation Awareness", follow-up to the cards; also usable in "Bias Point Retention")

`tfln_probe_station_closeup.jpg`: the same die, correctly loaded, two GSG probes (150 um pitch) landing on the pad sets at the two ends of the straight traveling-wave electrode (RF input west, termination east), lensed fibers at both facets in the fixture slots. Framed tighter than the fixture cards (field about 10.5 mm).

- Step type: `hotspots`, `style: 'callout'`.
- Points: GSG probe on RF input pads (0.449, 0.724); GSG probe on termination pads (0.574, 0.701); lensed fiber at input facet (0.413, 0.730); lensed fiber at output facet (0.610, 0.695); die +Z arrow (0.503, 0.661); probe bodies with connectors and cables at the left and right edges (about 0.08, 0.35 and 0.92, 0.25); microscope objective above (top edge, 0.51, 0.02).
- Caption: "Figure 2. A correctly oriented device under test. Every probe is where the layout said it would be. Please take a moment to appreciate this."

## 3. Gel tray, one die rotated (Ferroelectric, "Crystal Orientation Awareness")

`tfln_gel_tray_one_rotated.jpg`: 4 x 3 identical dies on a gel-film tray; row 2, column 3 is rotated 90 degrees.

- Step type: `spot`. Target (rotated die): `{ x: 0.508, y: 0.471, w: 0.121, h: 0.198 }` (image bbox of the die; centre 0.569, 0.570). Dies are labelled MZM-01 to MZM-12 and hand-placed (about 2 degrees of random yaw), so only the 90 degree die is the hazard.
- `missWhy`: "That die is fine. It is facing the same way as its colleagues. Keep looking."
- Caption: "Figure 3. Twelve dies were placed in this tray. Eleven of them followed the procedure."

## 4. Lithium niobate wafers (Ferroelectric, "Crystal Orientation Awareness" and "Pyroelectric Awareness")

| File | Module | Step type |
|---|---|---|
| `ln_wafer_carrier.jpg` | Crystal Orientation Awareness | `photo` with `kb` push-in toward the flats, or `hotspots` (`pins`) |
| `ln_wafer_hotplate.jpg` | Pyroelectric Awareness (cold open replacement keyframe, or `check` image for "What would you do?") | `video` keyframe image / `check` image / `hotspots` |

- Carrier points (double-side polished, black carrier): wafer centre (0.475, 0.504); primary flat (0.288, 0.845); secondary flat (0.260, 0.225); carrier floor mark seen through the wafer (0.501, 0.708); window reflection band on the right (about 0.66, 0.45).
- Hot plate points (single-side polished device wafer): wafer (0.574, 0.199); primary flat (0.547, 0.326); temperature display (0.221, 0.92, near the bottom edge). The caution label is now below the frame.
- Ken Burns (hot plate): `kb: [0.5, 0.5, 1.0, 0.58, 0.25, 1.6]` (from full frame to the wafer).
- Captions: carrier: "Figure 4. A 100 mm lithium niobate wafer. The flats are there so that nobody has to guess. Nobody should guess." Hot plate: "Figure 5. A routine bake at 115 degrees C. No voltage has been applied. Charge is being generated anyway."

## 5. Pyroelectric damage (Ferroelectric, "Pyroelectric Awareness")

`pyro_discharge_comb_micrograph.jpg`: reflected-light micrograph, interdigitated Au electrodes (4 um fingers and gaps) on x-cut LNOI, one main discharge site and one minor site; 20 um scale bar.

- Step type: `hotspots`, `style: 'numbered'` (or `spot` with the two damage sites as targets).
- Points / spot targets: main discharge site (0.49, 0.45), target `{ x: 0.42, y: 0.31, w: 0.15, h: 0.28 }`; minor notched finger tip (0.21, 0.22), target `{ x: 0.18, y: 0.17, w: 0.06, h: 0.10 }`; intact fingers for contrast (0.80, 0.50); scale bar (0.89, 0.92).
- Caption: "Figure 6. Electrodes after an unscheduled thermal event. The fingers on the left completed the cooldown. The fingers in the middle did not."

## 6. Butterfly laser, isolator and connectors (Feedback, "Reciprocity Is Not Optional"; connectors also "Every Interface Is a Cavity")

| File | Shows |
|---|---|
| `butterfly_isolator_fc_bench.jpg` | Correct: isolator arrow points away from the laser |
| `butterfly_isolator_reversed_bench.jpg` | Same bench, isolator spliced in backwards (arrow toward the laser) |
| `fc_apc_vs_pc_ferrules.jpg` | Inset: FC/APC (green, 8 degree polished tip) above FC/PC (blue, domed tip), side view |

- Bench, step type `hotspots` (`pins`): butterfly package (0.685, 0.180); M6 cap screw in a grid hole (0.498, 0.308); laser D-sub and cable (0.809, 0.300); inline isolator (0.333, 0.446), laser-side end (0.403, 0.501), output-side end (0.264, 0.391); splice sleeves (0.485, 0.601) and (0.184, 0.334); FC/APC green, capped (0.684, 0.824); FC/PC blue, capped (0.678, 0.953).
- New `butterfly_isolator_reversed_closeup.jpg` (same scene as the reversed bench, camera on the isolator; arrow about 300 px long): isolator box `{ x: 0.13, y: 0.48, w: 0.38, h: 0.23 }` (ends at (0.148, 0.518) output side and (0.492, 0.683) laser side); arrow points toward the laser side, lower right; laser mount in the upper right third for context; splice sleeve (0.704, 0.850). Use it with the forward bench (0.333, 0.446 isolator) for a `cards` pair, or as the `spot` image.
- Ferrules, step type `photo` or `cards`: APC tip (0.854, 0.385); PC tip (0.876, 0.573); key tabs (0.736, 0.305) and (0.751, 0.490); knurled coupling nut (0.577, 0.373); green boot (0.295, 0.431); blue boot (0.282, 0.622); jacket leaving the boot at the left edge.
- Captions: bench: "Figure 7. A laser, protected. The arrow on the isolator is not decorative." Reversed: "Figure 8. The same laser, protected from the wrong direction." Ferrules: "Figure 9. Green means angled. Angled means the reflection leaves through the cladding instead of coming home. Please do not mate green with blue."

## 7. PANDA fiber end face (Feedback, "Every Interface Is a Cavity" or "Your Instruments Are Averaging"; usable in any polarization incident)

| File | Shows |
|---|---|
| `pm_panda_endface_0deg.jpg` | Stress rods at 12 and 6 o'clock, aligned with the KEY marker |
| `pm_panda_endface_90deg.jpg` | Same framing, stress rods at 3 and 9 o'clock (fiber rotated 90 degrees in the ferrule) |

- Step type: `cards` (`choose`, "Which connector has its slow axis aligned to the key?") or `hotspots` on the 0 degree image.
- Points: KEY marker (now an annotation just outside the scope field, 0.50, 0.04); core (0.50, 0.50); stress rods 0 deg (0.50, 0.436) and (0.50, 0.564); stress rods 90 deg (0.464, 0.50) and (0.536, 0.50); ferrule (0.69, 0.50). The two images use different debris (two different connectors).
- Caption: "Figure 10. Stress rods at twelve o'clock. The key is also at twelve o'clock. Everyone is on the same page." 90 deg: "Figure 11. Stress rods at three o'clock. The key did not move. Your extinction ratio did."

## 8. Silicon nitride microring (Feedback, "Backscatter in High-Q Resonators")

`sin_microring_lensed_fiber.jpg`: top-down brightfield micrograph illustration (objective NA 0.28), field 520 um: lensed fiber tip at the chip edge on a bus waveguide with an inverse taper (0.4 um at the facet) coupling to a 100 um radius add-drop ring (2.5 um wide, 800 nm thick SiN, 0.8 um gaps to the bus and to the drop waveguide; the gap is below the optical resolution at NA 0.28 and shows as a dark line, not as contact); 50 um scale bar.

- Step type: `hotspots` (`numbered`) or `photo` with `kb` from the fiber tip to the coupling point.
- Points: lensed fiber tip at the facet (0.148, 0.842); chip edge (0.154, 0.50); inverse taper (0.288, 0.842); ring-bus coupling point (0.692, 0.83); ring centre (0.692, 0.489); drop bus coupling point (0.692, 0.14); bus downstream of the ring (0.865, 0.842); layout label (0.692, 0.917); scale bar (0.913, 0.952).
- `kb: [0.2, 0.8, 1.8, 0.69, 0.75, 1.8]`
- Caption: "Figure 12. A high-Q resonator awaiting light. Some of that light will be returned to the sender. This is not a courtesy."

## Engineering audit (2026-11-22, after review)

| Render | Nonsense found | Fix |
|---|---|---|
| Butterfly benches (both) | Mount base not on the breadboard grid, yet four bolts drawn through its holes into nothing; breadboard holes on a 12.5 mm offset grid | Hole grid moved to multiples of 25 mm; base plate 112 x 62 mm with four counterbored M6 clearance holes at +-50 / +-25 mm from the centre, all on grid holes; M6 socket head cap screws (head 10 x 6 mm) seated in the counterbores |
| Butterfly benches | Package pins bent down over a loose white bar; package held by two floating clamp bars | Straight 14 pins (7 per long side, 2.54 mm pitch) lying in black pin sockets under clamp bars, each bar held by two M2 screws into the socket; package flange held by four M2 screws through its tabs into a gold seat plate on the TEC |
| Butterfly benches | Mount had no electrical connection | D-sub receptacles on both end faces (LD and TEC) with mated plugs, backshells and grey cables running off the board |
| Butterfly benches | Pigtail bent sharply down right at the boot | Fiber leaves the boot on axis and droops to the board with bend radii above 25 mm |
| Butterfly benches | Breadboard holes rendered as flat rings | Tapped-hole shading: dark bore getting darker with depth, bare metal thread crests at 1 mm pitch, chamfer ring |
| Butterfly benches | Loose FC connectors with bare ferrules on the bench | Dust caps on both (colour-matched); connectors rest on nut and boot (slight tilt) instead of floating at the boot |
| FC/APC vs PC close-up | Two connectors floating in mid-air | Both rest on an anti-static mat, caps off for inspection, keys up so the 8 degree face shows in profile |
| TFLN die renders | (earlier review) single U band did not read as an MZM | Straight GSG traveling-wave electrode with GSG pad sets at both ends, MMI splitter/combiner and fan-out visible |
| SiN microring | (earlier review) ring and bus nearly invisible | 520 um field, 2.5 um wide ring, cladding-bump outlines, true stack colours |
| Die fixture cards, probe close-up | (first audit) fixture screws, die seating | Superseded by the critique fixes below (fixture rebuilt at realistic scale) |
| Gel tray, wafer carrier, hot plate | Checked: tray, carrier and lid, and hot plate rest on the bench; wafer rests on the carrier ledge / plate; hot plate power cord is behind the unit (not in view) | No change |
| Micrographs (pyro, PANDA, SiN) | Checked: scale bars match the fields; PANDA zone rings and key marker are inspection-software overlays | No change |

### Fresh-eyes critique v1 (research/render_critique_v1.md) and user notes: outcomes

| Render | Critique / note | Outcome |
|---|---|---|
| Butterfly benches | Bolts not in table holes; holes flat rings; no LD/TEC cables | Fixed earlier (grid, tapped-hole shading, D-subs); re-verified in the new renders |
| Butterfly benches | SHCS heads read as spacers; leads round and arcing; stray crescent | Fixed earlier (10 mm SHCS with hex sockets; flat straight leads under clamp bars; the crescent is now one of four flange screws) |
| Butterfly benches | No splice sleeves although "spliced in" | Added a 40 mm heat-shrink splice sleeve with steel rod on each side of the isolator |
| Butterfly benches | Isolator arrow a 10 px chevron; reversed arrow at a different place | Printed white label band with a 21 mm arrow (60 % of the body) and text; the reversed variant is the same decal turned end for end. The isolator now lies across the view and its forward direction points visually away from the laser |
| Butterfly benches | Mount label "TEC 14-PIN" | Replaced earlier by connector labels LD and TEC |
| Butterfly benches, ferrules | FC connector housing wrong (user): smooth nut, loose key | Rebuilt as the layered FC plug from references (below): axially knurled coupling nut with smooth front section and M8 x 0.75 opening, plug-body barrel with the key tab at its root next to the ferrule, split bronze retaining ring, chamfered zirconia ferrule, boot tapering to the cable with flex slots, dust caps on the bench connectors |
| fc_apc_vs_pc_ferrules | Connectors float; key a loose tab | Both rest on a mat (contact shadows); key tab sits on the barrel root, touching the barrel |
| ln_wafer_carrier | Opaque matte disc plus a separate flatted band; wafer intersects the carrier wall; flats wrong | Paper interleaf removed (it was the "inner disc"); wafer is one double-side-polished body, clear with a faint yellow tint, floor ribs visible through it, reflection band; carrier pocket 102 mm with a ledge, 1 mm clearance; flats 32.5 mm and 18 mm at 90 degrees |
| ln_wafer_hotplate | Opaque decal, no reflection or thickness; two matte rectangles of stripes | Single-side-polished device wafer (milky body from the lapped back), glossy highlight band from a strip light, visible edge; grid of the 6 mm modulator dies in metallic gold with 3 mm edge exclusion and rows parallel to the primary flat; 7-segment display behind a smoked window |
| pyro_discharge_comb_micrograph | Decal over intact fingers; craggy 5-10 um nuggets; bubble spatter; scorched halo; pale gold | Finger segments removed (bare substrate), smooth resolidified spheres about 1 um at the broken ends, one fused bridge across a gap, sub-um solid spatter, a 5 um crater with radial cracks in the LN, a subtle thin-film discoloration under the gold only; plain sRGB camera response so the gold is the brightest saturated feature. Now described as a rendered illustration of an optical micrograph |
| sin_microring_lensed_fiber | Top bus uncoupled; ring touching bus; widths; no taper; label | Top bus is now a drop port at the same 0.8 um gap; gap renders as a sub-resolution dark line; ring and bus both 2.5 um with narrower slopes; inverse taper to 0.4 um over 150 um at both facets; label below the bus with cladding-bump edges; smoother facet |
| tfln_die_fixture_* | Slots not on the waveguide line and running under the die; waveguides appear routed into RF pads; fixture text and screws too small; die sidewall thin; no hold-down | Fixture rebuilt at 40 x 28 x 6 mm: two 1.2 mm slots on the waveguide line ending at the pocket, IN/OUT engraved at the slots, 2 mm laser-marked text, M3 cap screws, vacuum port under the die with a hose barb and tubing; pocket 0.2 mm deep so the die stands 0.3 mm proud. Die redesigned: straight GSG electrode with in-line tapered pad sets at both ends; the arms go around the pads and pass under their own ground only, so no waveguide meets a pad |
| tfln_die_fixture_90deg | Alt text claimed facets against closed walls | Kept the square die (the stated hazard: it fits in all four rotations) and made the alt text honest: facets now face the closed north and south pocket walls and the slots line up with no waveguide; visible in the render |
| tfln_gel_tray_one_rotated | All dies MZM-07; glossy gel; identical angles | Dies MZM-01 to MZM-12, about 2 degrees of random yaw and 0.25 mm of offset, satin gel film inset from the wall |
| tfln_probe_station_closeup | Bare tubes from above; no connectors, cables or arms; shadow; sparkly gold; black fiber | GSG probes come in at 30 degrees from left and right, each with a body, gold 2.92 mm-class connector, blue cable and positioner arm; compact three-contact tips on the pad sets; objective above; die gold without micro-bump (no sparkle); fibers held from above with the coating visible |
| pm_panda_endface_* | Identical debris; outlined rods | Two different connectors: different debris seeds; rod outline discs removed; KEY marker moved outside the scope field as an annotation |
| credits_render.js | "micrograph" for renders | Alt texts now say "Rendered illustration of an optical micrograph" / "of a fiber inspection scope view" |

Second pass (coordinator review): FC boot was a long cylinder at nearly the nut diameter -> now a 2 mm crimp sleeve behind the nut, then a 26.5 mm boot tapering from 6 mm to 2.7 mm with flex slots over its rear two thirds and the jacket leaving it (plug 24 mm + boot 28 mm); knurl kept over about 70 % of the nut length, as in the references. ln_wafer_carrier still read as a white grooved plastic disc -> black conductive carrier, no floor ribs; the DSP wafer now shows the floor mark through it (refracted, warm tint), a window reflection on part of its face, and side lighting that makes the 0.5 mm edge and both flats glint; flats turned to the visible side of the pocket.

FC connector references viewed (images only, not copied into the course):
- Wikimedia Commons, "FCPC 001.jpg" and "FCPC 002.jpg" (FC/PC patch cord on a breadboard): https://commons.wikimedia.org/wiki/File:FCPC_001.jpg , https://commons.wikimedia.org/wiki/File:FCPC_002.jpg
- Wikimedia Commons, "Connecteurs FC-APC - 1.JPG" (side view of an FC/APC plug: knurled nut, smooth front ring, barrel with key tab, ferrule): https://commons.wikimedia.org/wiki/File:Connecteurs_FC-APC_-_1.JPG
- Wikimedia Commons, "Optical-patch-fiber-FC-APC-connectors.jpg", "Cordons FC-APC FC-PC- 1.JPG", "Optical-fiber-coupler-FC-APC 1.jpg" (context, caps, boots)
- Optical Cable Corporation drawing C112948 "Fiber conn kit, SM/MM, FC conn" (exploded view: metal housing with knurled nut, crimp ring, boot, dust cap): https://www.occfiber.com/wp-content/uploads/2018/01/C112948A1.pdf
- Wikipedia, "FC connector" (plug key width vs receptacle keyway width table, type N about 2.09-2.14 mm, type R about 1.97-2.02 mm; spring-loaded 2.5 mm ferrule): https://en.wikipedia.org/wiki/FC_connector
- Coupling thread M8 x 0.75: reported for vendor FC drawings in search results (Molex FC/APC drawing; L-com FOA-PMFCPN 2D drawing, which also gives key width 2.15). I did not open a drawing that shows the thread; the model uses an M8 opening and a dark internal thread band, not modelled flanks.
- Finding on the key: every reference shows the plug's key as a radial tab (protrusion) on the plug-body barrel at the nut opening, and the keyway (slot) is on the receptacle/adapter. The "notch gap" visible on some plugs is the gap in the split bronze retaining ring at the barrel root (FCPC 002.jpg); both are modelled. No receptacle appears in these scenes.

Wafer flat convention used (stated in the scripts and alt text): primary flat 32.5 mm, secondary flat 18 mm, secondary 90 degrees clockwise from the primary seen from the polished face (SEMI-style 100 mm layout). LN vendors vary; not a vendor drawing.

## Not made

- Dicing saw blade over a wafer on a tape frame (optional item): not rendered.
- Wide probe-station view with the objective in frame: rendered as a test, rejected (probe bodies and fiber positioners read as CG blocks); the objective is out of frame in the close-up.

## References used for accuracy (viewed read-only, not copied)

- Wang, Zhang, Stern, Lipson, Lončar, "Nanophotonic lithium niobate electro-optic modulators", Optics Express 26, 1547 (2018): GSG layout with both MZ arms in the electrode gaps, x-cut orientation (die, probe, gel tray renders).
- Kharel et al., "Breaking voltage-bandwidth limits in integrated lithium niobate modulators using micro-structured electrodes", Optica 8, 357 (2021), Fig. 1: straight GSG traveling-wave section and arm placement.
- Xu et al., "Dual-polarization thin-film lithium niobate in-phase quadrature modulators for terabit-per-second transmission", Optica 9, 61 (2022), Fig. 1: x-cut TFLN chip electrode routing to side pads.
- Sarabalis, "Acousto-optics in thin-film lithium niobate", PhD thesis, Stanford (2021), p. 239, and McKenna et al., "Cryogenic packaging of an optomechanical crystal", Optics Express (2019), as collected in the post "How to hold fibers" (outside5sigma.com, 2025-12-25): appearance of TFLN dies with gold electrodes on a mount, fiber approach.
- Franken et al., "High-power and narrow-linewidth laser on thin-film lithium niobate enabled by photonic wire bonding" (2025), doi:10.1063/5.0231827, same post: dark TFLN chip photographs.
- Wilson et al., "Integrated gallium phosphide nonlinear photonics", arXiv:1808.03554 (2018), same post: top-down micrograph look of rings with lensed fibers entering from both sides (microring render).
- Liu et al., "Wafer-Scale Squeezed-Light Chips", arXiv:2509.10445 (2025), and Kramnik et al., Nature Electronics (2025), same post: objective above the chip, side-coupled fibers in V-groove holders (probe-station close-up).
- Thiele et al., "Cryogenic electro-optic modulation in titanium in-diffused lithium niobate waveguides", arXiv:2202.00306 (2022): pyroelectric charging and discharging during cooldown (context only; no damage micrograph found in the atlas caches).
- Lithium niobate wafer flats: the `lithium-niobate` project notes (open question Q4) record that flat lengths and the flat-to-axis convention vary by vendor (one listing: 32 mm primary flat at 100 mm; another: secondary flat 90 degrees clockwise). The render uses those values as a representative layout, not a vendor drawing.

## Physical simplifications (known)

- LNOI and SiN chip surfaces are rendered as coloured mirrors whose colour is the normal-incidence reflectance of the layer stack, from a transfer-matrix calculation (`course/tools/blender/color_stack.py`). Modulator die: trench-defined rib process (600 nm LN film on 4.7 um SiO2 on Si, mauve-grey field; each waveguide is a 1.5 um rib between two 3 um trenches etched to a 300 nm slab, green-grey, given a rougher ion-milled finish), so a waveguide shows as a faint ~7.5 um line at true size. SiN chip: field (0.226, 0.245, 0.304) vs over-core (0.260, 0.272, 0.325), about 11 % brighter; a partially coherent model (thick oxide added incoherently) gives the same values. The visible waveguide outlines come from the conformal cladding bump (0.8 um high, 34 degree slopes) whose sides reflect outside the objective NA. No angle-dependent thin-film colour shift.
- Electrode gaps (8 um), waveguides and the 150 um GSG pad pitch are to scale, so the 8 um gaps are about one pixel or less in the die-level shots. The arms cross only under their adjacent ground conductor (cladding crossing, not drawn).
- LN birefringence is ignored (single IOR 2.2). Transparent wafers and fibers do not cast shadows (caustics off).
- Depth of field is thin-lens; the microscope shots use an orthographic camera, an illuminator limited to the objective NA, and compositor vignette and lateral colour.
- Pyroelectric damage morphology (beaded finger ends, molten bridge, spatter, darkened halo) is modelled on generic electrostatic-discharge damage of thin Au films; no micrograph of pyroelectric damage on LN electrodes was found in the local corpora, so treat it as plausible, not documented.
- PANDA geometry is typical PM1550 (125 um cladding, 33 um rods 28 um off-axis, 8.5 um core); grey levels mimic an LED inspection scope; zone rings approximate the IEC 61300-3-35 single-mode inspection zones (drawn at 25, 120, 130, 250 um diameter).
- Butterfly package, mount and isolator are generic geometry (no vendor model). FC connectors follow the references listed above; the M8 x 0.75 internal thread is shown as a dark band, not modelled flanks.
- Fixture die pocket is 6.12 mm square, so a square die fits in any of four rotations; this is the hazard the course describes.

## Timing (M4 Pro GPU, Metal, 1920 x 1080)

| Shot | Samples | Seconds per frame (alone) |
|---|---|---|
| Die in fixture (x3) | 256 | 80 each, three in parallel |
| Probe station close-up | 256 | 75 (three in parallel) |
| Gel tray | 256 | 68 (three in parallel) |
| Wafer carrier / hot plate | 256 | 77 / 45 (in parallel) |
| Pyro micrograph | 192 | 44 (in parallel) |
| PANDA end face (x2) | 128 | 33 (in parallel) |
| SiN microring | 192 | 21 (three in parallel) |
| Butterfly benches (x2) + isolator close-up / ferrules | 256 | 60 each (three in parallel) / 37 |

Benchmark: first full-size die frame at 96 samples, 10.5 s.
