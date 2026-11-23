# Render critique v1 (2026-11-22)

Scope: the 14 Blender renders in `course/media/render/` and `course/media/fe/sim_solver_window_drawn.jpg`, checked against their alt text in `course/media/credits_render.js` and `course/media/credits_fe.js`. Read-only review from the point of view of an experimental photonics engineer. Every image was viewed in full and the details were cropped at full resolution.

Format: `[severity] (x, y) what is wrong -> what the real thing looks like -> fix`. Coordinates are normalized (0,0 = top-left, 1,1 = bottom-right) and refer to the 1920x1080 frame. Severity is must-fix, should-fix or nit.

Measurements below come from pixel counts on the frames. Where a scale is inferred (scale bar, 125 um cladding, 2.5 mm ferrule), the inference is stated.

---

## 1. butterfly_isolator_fc_bench.jpg

- [must-fix] (0.59,0.27), (0.835,0.195), (0.795,0.345), (0.62,0.40 approx.): **already caught by the owner.** The four mount bolts are not in breadboard holes; the mount base-plate holes sit off the 25 mm grid. -> A real mount is either bolted into tapped M6 holes on the grid (holes in the base line up with holes in the table) or held down by clamping forks. -> Move the mount so its four holes land exactly on grid holes, or swap in two clamping forks.
- [must-fix] Whole breadboard: the "holes" are thin dark rings, and inside each ring is the same grey as the table top. There is no depth, no thread and no dark bore (crop at 0.0-0.4, 0.5-0.8). -> A real tapped M6 hole is a black bore with a small chamfer/countersink highlight on the rim. -> Model actual bores (about 6 mm diameter, about 0.5 mm 45-degree chamfer, interior near black), or use a displacement/alpha texture with a dark interior.
- [must-fix] Mount (0.57-0.85, 0.08-0.40): no electrical cables. A 14-pin butterfly with a TEC cannot run without a laser-current cable and a TEC-controller cable. -> Commercial butterfly mounts have two DB9-type connectors (LD and TEC) on the side or back, with round cables running off the table to the drivers. -> Add two D-sub connectors on the mount body and two black cables leaving the frame.
- [should-fix] Bolt heads (e.g. 0.59,0.27): they read as hollow cylindrical spacers or nuts. The top is a plain annulus with a dark centre, there is no hex socket, and the head is about the same diameter as a table hole. -> An M6 socket-head cap screw has a 10 mm knurled head, about 1.7x the 6 mm hole, with a visible 5 mm hex socket. Usually a washer sits under it on a slotted base. -> Model SHCS heads at 10 mm with a hex socket. Optionally add washers, and slots in the base plate.
- [should-fix] Butterfly leads (0.67-0.75, 0.17-0.21): the 7 visible leads are round wires that arc down onto a white strip. -> Butterfly leads are flat gold-plated Kovar ribbons that exit straight and horizontal from the package base at 2.54 mm pitch. In a mount they are pressed down by a hinged lead clamp or ZIF bar (zero insertion force) onto contact pads. -> Make the leads flat and straight, and put a clamp bar over them.
- [should-fix] (0.632,0.117): an unexplained shiny crescent (a partial screw head or clamp) floats beside the package on the black block. -> Every visible part should be identifiable: the package flange hold-down screws (4, through the flange ears) or a clamp. -> Either remove it or make it one of the 4 flange screws, with flange ears visible on the package.
- [should-fix] Pigtail between laser and isolator (0.30-0.58, 0.08-0.30), and isolator to connector: an inline isolator is fusion-spliced into the pigtail, but there are no splice protection sleeves. -> Each splice has a 40-60 mm clear heat-shrink sleeve with a steel strength rod, or sits in a splice tray. -> Add a sleeve on each side of the isolator. This matters most for the reversed variant, whose alt text says "spliced in backwards".
- [should-fix] Isolator arrow (0.24,0.42): the direction arrow is a single 10 px chevron, and at normal viewing size its direction cannot be read. The arrow is the whole teaching point of this image pair. -> Real inline isolators carry a printed label: a long arrow, the wavelength, and often the part number. -> Make the arrow a full-length printed line-and-head (about 60% of the body length) on a label band.
- [nit] Mount body text "TEC 14-PIN" (0.69,0.26): this is a descriptor, not something a vendor prints. -> Real mounts carry a model number and connector labels ("LD", "TEC"). -> Use a neutral fictitious model number and connector labels, with no real vendor names.
- [nit] FC connectors (0.62-0.75, 0.82-0.93): the coupling nut is a smooth cylinder with no knurl and no key, and there are no dust caps on unused connectors. -> Real FC nuts are knurled. Unused connectors on a bench wear dust caps (and the APC one usually has a green cap). -> Add knurling. Optionally put a cap on the blue patch cord.

## 2. butterfly_isolator_reversed_bench.jpg

Same scene, so every item in section 1 applies. Specific to this image:
- [should-fix] (0.24,0.37): the reversed arrow is again tiny and is printed higher on the body than in the forward variant. -> Use the same label position and orientation, rotated 180 degrees. -> Use one label decal and flip it.
- [should-fix] Alt text says "spliced in backwards", but no splice is visible (see the sleeve item in section 1).

## 3. fc_apc_vs_pc_ferrules.jpg

APC angle checks out. The tip-face offset is about 17 px over a 104 px ferrule height (ferrule = 2.5 mm), which is about 9 degrees, close to 8 degrees, and it lies in the key plane. PC tip is fine.
- [should-fix] Key (0.76,0.15) and (0.76,0.58): the key is a loose rectangular tab sticking out of the front face of the coupling nut at about 75% of the nut radius, with a gap between it and the ferrule. -> On an FC connector the key (about 2 mm wide) is part of the connector body or ferrule holder, right next to the ferrule, recessed inside the nut opening. -> Add a ferrule-holder barrel inside the nut and put the key on it, touching the ferrule side.
- [should-fix] Both connectors float. The upper one has no support. The lower one's contact shadow on the floor (y≈0.93) is separated from the nut, so it hovers. -> Product shots put the part flat on a surface, or in a holder or bulkhead. -> Lay both on the surface with contact shadows, or show them held in a clip.
- [nit] Nuts are flawless mirror chrome with no knurl. -> Real FC nuts are satin nickel-plated brass with knurling, and usually show small machining marks. -> Add knurling and roughness of about 0.2-0.3.
- [nit] Boots (0.0-0.24): they get wider toward the cable end. -> Strain-relief boots taper down toward the cable. -> Reverse the taper.

## 4. ln_wafer_carrier.jpg

- [must-fix] Wafer appearance (0.22-0.72, 0.18-0.85): the alt text says "clear, faintly yellow". The render shows an opaque, matte, off-white disc with no specular reflection, surrounded by a darker translucent band. A 100 mm LN wafer (n≈2.2, about 14% front reflectance) is mirror-like. A double-side-polished wafer is water-clear with a faint yellow tint and shows the carrier floor through it. A single-side-polished wafer has a lapped back and looks frosty or milky, but its front is still a mirror. -> Choose one. For DSP: glass-like material (IOR 2.2, slight yellow absorption) over the black carrier floor, plus a visible reflection of the room or light. For SSP: a frosted back, but a mirror front.
- [must-fix] The inner light disc (0.21-0.72) is perfectly round with no flats, while the flats appear only on an outer translucent band, so the image reads as two objects (a round pad plus a ring with flats). The real wafer is one piece: its outline has the flats, and it has a single uniform look. -> Fix the material/geometry so the flats cut the visible wafer itself.
- [must-fix] Wafer interpenetrates the carrier wall. At the right (0.66,0.24)-(0.72,0.50) and lower left (0.24,0.70)-(0.39,0.84), the wafer edge disappears into the carrier's inner wall, and the wall shows facets where the flats cut through it. The wafer diameter is larger than the pocket. -> A single-wafer carrier pocket is about 1-2 mm larger than the wafer, and the wafer sits on a ledge or ring with clearance all round. -> Make the pocket ID about 102 mm and set the wafer on a ledge.
- [should-fix] Flats: measured in the plane, both flats are about 38-44 mm long, and they are roughly 150-160 degrees apart. -> On a 100 mm wafer the primary flat is about 32.5 mm and the secondary is clearly shorter, about 18 mm. In SEMI-style layouts the secondary is 90 degrees from the primary (some LN vendors use other angles; check the vendor drawing for the cut). -> Make the primary about 32 mm and the secondary about 18 mm, at the angle the vendor spec gives.
- [nit] The bench is described as stainless steel but renders as plain matte light grey. -> Brushed stainless shows directional anisotropic highlights. -> Use an anisotropic brushed-metal shader, or change the alt text.

## 5. ln_wafer_hotplate.jpg

- [must-fix] Wafer (0.45-0.72, 0.08-0.30): opaque cream-white with no reflection and no visible thickness. It looks like a paper decal on the plate. -> An LN wafer on an aluminium plate is either clear with a yellow tint (showing brushed Al through it) or frosted, and it has a polished front that reflects. Its 0.5 mm edge catches a highlight. -> Use the same material fix as the carrier image, plus edge thickness and a contact shadow.
- [should-fix] Gold pattern (0.46-0.71, 0.13-0.23): two large rectangles of evenly spaced tan stripes, with no pads, no die outlines, no dicing streets and no alignment marks. The gold is matte beige, not metallic. -> A modulator wafer shows a grid of dies, each with long CPW electrodes ending in pad groups, dicing lanes and corner marks. Evaporated gold looks bright, saturated and specular. -> Tile a die layout (the same MZM die as the fixture images) across the wafer, and use metallic gold with roughness of about 0.15.
- [nit] Display (0.2-0.33, 0.77-0.88): the digits are a rounded vector font printed straight onto the panel. -> Hot plates use 7-segment LEDs behind a smoked window. -> Use a 7-segment font with a slight glow behind a dark glass inset.
- [nit] No power cord, and the top plate sits off-centre on the housing (left margin much larger than right). -> Center the plate, or accept as is.

## 6. pm_panda_endface_0deg.jpg

Geometry checks out. Taking cladding = 125 um, the scale is 2.5 px/um. Stress rods are about 33 um in diameter, rod inner edges are about 11.6 um from the core centre, and the core is about 8 um. The IEC-style zone rings measure about 24 / 120 / 130 / 246 um, which matches the 25 / 120 / 130 / 250 um core / cladding / adhesive / contact zones. The outer grey disc is the scope's field of view, not the ferrule edge, which is correct for a 400x scope. The slow axis (the line through the rods) is aligned to KEY, as the alt text says.
- [should-fix] Stress rods (0.48,0.43) and (0.48,0.57): drawn as flat discs with crisp dark outlines, like vector art. -> In a reflected-light or backlit inspection image, the boron-doped rods show as slightly different-contrast discs with soft edges and no outline stroke. -> Drop the outline and use a slight brightness offset with about 1-2 px blur.
- [nit] KEY label and triangle (0.5,0.08-0.13): an inspection scope does not print "KEY" on the image. This is fine as a course overlay, but if it is meant to look like scope output, move it outside the field-of-view circle as an annotation.

## 7. pm_panda_endface_90deg.jpg

Same geometry checks pass, and the rods are at 3 and 9 o'clock.
- [should-fix] All dust, pits and scratches are pixel-identical to the 0-degree image (e.g. the defect at 0.522,0.478, and the scratches at 0.63-0.66, 0.2-0.5). If these are two different connectors, identical contamination is a giveaway when the images are shown side by side. If it is the same fiber rotated, the defects on the glass (inside the cladding) should rotate with it. -> Use a different defect seed for the second image, or rotate the in-cladding defects with the fiber.
- [should-fix] Same stress-rod outline issue as in section 6.

## 8. pyro_discharge_comb_micrograph.jpg

Scale checks out. The 20 um bar is 193 px, so 9.65 px/um. Fingers measure 4.1 um, gaps 3.8 um, and finger-tip to opposite busbar is about 10 um. This is consistent with the alt text.
- [must-fix] Central damage site (0.43-0.57, 0.30-0.58): the "melted away" area is a flat, hard-edged dark-brown polygon laid over the fingers, and the finger stripes stay continuous under the airbrushed halo. The gold has not actually gone anywhere. -> Real arc or discharge damage on an IDT (interdigitated transducer) has missing finger segments where the bare substrate shows, fused bridges across a gap, a crater or pit in the LN, and often conchoidal or radial cracks in the LN from thermal shock. -> Cut the finger geometry where it melted, add 1-2 bridged gaps, and add a small cratered or cracked LN region instead of a brown decal.
- [must-fix] Beads (same area): the debris is craggy, faceted "rock" or cornflake shapes 5-10 um across, bright orange with white highlights. -> Re-solidified gold forms smooth spheres. Under brightfield reflected light a sphere appears dark with a small bright specular dot at its top. Mass balance: melting about 3 fingers x 15 um x 4 um x 0.25 um is about 45 um^3 of gold, which is enough for a few 2-3 um beads and many sub-micron droplets, not several 5-10 um nuggets. -> Use smooth spheres, mostly under 2 um, rendered dark with a point highlight.
- [should-fix] Spatter (scattered over 0.38-0.65, 0.2-0.75): the droplets are ring outlines with transparent centres, so they read as soap bubbles. -> Make them small solid dark dots with specular points.
- [should-fix] Halo: a soft, smoke-like gradient with green-brown hue. Gold and LN do not char, so "scorched" is the wrong look. What really appears is a thin redeposited-film discoloration (often interference-coloured), with fairly defined edges. -> Tighten it and keep it subtle, or replace it with a thin-film colour shift.
- [should-fix] Gold colour: pale tan, darker than a polished metal would be. -> In brightfield, gold is the brightest, most saturated yellow in the frame, brighter than the dielectric. -> Raise the gold value and saturation.
- [nit] Finger corners are perfectly sharp and the edges are ideal. -> Lift-off gold at 4 um shows about 0.3-0.5 um corner rounding and a thin dark edge line. -> Bevel the corners slightly.
- Second site (0.21,0.22): the location (finger tip next to the opposite busbar) is physically sensible. It needs the same bead and halo fixes.

## 9. sin_microring_lensed_fiber.jpg

Scale checks out. The 50 um bar is 185 px, so 3.7 px/um. The ring diameter is 756 px, about 204 um, so R is about 102 um, consistent with "R100".
- [must-fix] Upper "bus" waveguide (y≈0.05-0.06): it is about 24 um above the ring top (ring top y≈0.14). At that gap it does not couple to a SiN ring, so it is not a drop port and serves no purpose. -> In an add-drop ring the drop bus passes at about the same gap as the through bus (about 0.5-1 um). -> Move it to within about 1 um of the ring top, or remove it and update the alt text.
- [should-fix] Coupling region (0.62-0.76, 0.83-0.84): the ring's outer edge line merges into the bus's top edge line, so at 3.7 px/um the ring appears to touch the bus. A 0.8 um gap is 3 px, which would be resolvable as a thin bright line between two dark edges. -> Show a 2-3 px light gap at the point of closest approach.
- [should-fix] Widths: the ring band measures about 3.2 um and the bus about 3.8 um (centre to centre of the edge lines), but the label says W2.5, and the ring and bus are drawn differently. Also, the waveguides are outlines only, with no fill contrast. -> In reflected light a SiN core shows as a filled stripe with a slightly different tone from the cladding, with edge lines. -> Use the same width for both (2.5 um = 9 px), and give the core a fill a few percent lighter or tinted.
- [should-fix] Facet (0.15,0.84): the bus runs at full 2.5 um width right up to the facet. -> An edge-coupled SiN chip used with a lensed fiber has an inverse taper (down to about 0.2-0.5 um over 100-300 um) or a dedicated mode converter at the facet. -> Narrow the bus over the last 100+ um.
- [nit] Label "R100 W2.5 G0.8" (0.63-0.76, 0.49): a faint, watermark-like text floating inside the ring. -> Layout text is in SiN or metal, so it has the same edge contrast as the waveguides, and it is usually placed outside the device. -> Render it with waveguide-like edges and move it below the bus.
- [nit] Lensed fiber (0.0-0.14, 0.78-0.90): a smoky dark cone with a second faint diverging edge line. -> A glass taper top-lit shows a bright axial specular streak and crisp outer edges near the tip, and is defocused further back. -> Add the axial highlight and drop the extra edge.
- [nit] The facet edge is wavy (chipping). -> A coupling facet is usually polished or deep-etched and smooth. -> Use a smoother edge with small chips only.

## 10. tfln_die_fixture_0deg.jpg

On the X-cut convention: the waveguide runs left-right (along Y), and +Z is in-plane and perpendicular to it. That is correct.
- [must-fix] Fiber slots versus waveguide: the only slot-like feature is a dark channel that crosses the fixture under the die. On the left (x≈0.3) it is at y≈0.30-0.45. On the right (x≈0.75-0.85) it is at y≈0.51-0.59. The waveguide ends are at IN (0.355,0.535) and OUT (0.72,0.65), so OUT is about 0.06 of the frame (roughly 0.5 mm) below the channel and IN is below it too. The channel also runs under the die's body, where no fiber could reach a facet. The "IN" and "OUT" engravings (0.155,0.72) and (0.82,0.97) sit beside screw counterbores, not at slot ends. -> A real die fixture has V-grooves or slots that end at the two facets, collinear with the waveguide, with IN and OUT engraved at the slot mouths. -> Put two short slots ending at the left and right die facets on the waveguide line, move the IN/OUT labels to the slot mouths, and take the channel out from under the die.
- [must-fix] Waveguide/electrode layout (pad region 0.38-0.46, 0.42-0.56): the splitter arms come in from the upper left and stop at the outer edge of the electrode bend. The only lines inside the gold are the CPW (coplanar waveguide) gaps, and these curve 90 degrees up into the RF pads. So the drawing reads as the optical arms turning into the RF pads. -> In a TFLN MZM the two arms run straight and sit in the two signal-ground gaps along the straight section. At each end the CPW bends away toward the pads while the arms carry straight on, under the outer ground or through a gap in it, to the splitter or combiner. The waveguides are never routed into pads. -> Draw the arms as straight lines through the full electrode length, with the splitter and combiner outside the bends, and draw the CPW gaps separately as they curve up.
- [should-fix] Scale: the fixture's engraved letters ("RF", "IN") are about 60-70 px tall, and the counterbores are about 300 px across. If the die is 6 mm (about 125 px/mm), the letters are about 0.5 mm tall and the counterbores about 2.4 mm, which suits M1.4 screws. Machined fixtures engrave 2-5 mm text and use M2-M4 screws. -> Either scale the die up (a real TFLN MZM die is long and narrow, e.g. 10-20 mm x 1-3 mm), or scale the fixture details up about 3-5x.
- [should-fix] CPW proportions: the whole GSG band is about 200 um wide (about 48 um signal and 60-80 um grounds). Real TFLN CPWs have grounds hundreds of um to mm wide. Pads are fan-shaped with uneven widths, and the gaps between them are only about 16 um. -> GSG pads are rectangles about 60-100 um wide at 100-150 um pitch, with gaps of about 25-50 um. The grounds are wide. -> Widen the grounds and make the pads rectangular. The measured pitch of about 145 um is fine.
- [should-fix] There is no die hold-down. The die just sits there, with no vacuum port, clamp or spring finger. -> Fixtures that hold a die for fiber alignment use vacuum through a hole or a small clamp. -> Add a vacuum hole under the die, or a clamp tab.
- [nit] The die's sidewall is very thin (5-10 px). At this viewing angle a 0.5 mm substrate should show about 20-30 px of edge. -> Thicken the die to 0.5 mm.

## 11. tfln_die_fixture_90deg.jpg

The die's internal rotation is consistent: +Z points left, the label reads bottom to top, pads are on the left side of the vertical electrode, IN is at the bottom and OUT at the top.
- [must-fix] The alt text says "polished facets face the closed walls of the pocket". The die is square and occupies exactly the same footprint as at 0 degrees, and there are no pocket walls next to the facets, so the image cannot show that failure. -> With a realistic long, narrow die, a 90-degree rotation would not fit the pocket at all, or would visibly sit across it. -> Either use a rectangular die with a matching pocket, or change the alt text.
- Inherits the slot/channel, layout and scale issues from section 10.

## 12. tfln_die_fixture_180deg.jpg

Rotation is consistent: +Z points down, the label is upside down, pads are at the bottom, IN is on the right, REF is at the top.
- Inherits the slot/channel, layout and scale issues from section 10. The alt text's "no longer lines up with the fiber slots" is not a contrast with the 0-degree image, because the 0-degree image does not line up either.

## 13. tfln_gel_tray_one_rotated.jpg

The rotated die is in the middle row, as the alt text says, and its rotation is internally consistent.
- [should-fix] All 12 dies carry the same ID "MZM-07". -> Dies singulated from a wafer carry distinct IDs (row/column or serial). -> Use different IDs (e.g. 01-12).
- [nit] The gel floor (whole tray) is glossy black acrylic with mirror reflections. -> Gel-Pak-type gel film is satin to matte (clear or grey, or black on ESD versions) and shows a film edge inset from the box wall. Boxes have a hinged lid. -> Lower the floor's specular, add the film border, and optionally show the hinged lid.
- [nit] Square dies: same comment as section 10 (real MZM dies are long and narrow).
- [nit] Every die sits at exactly the same angle. Hand-placed dies show about 1-3 degrees of random yaw. -> Add small random yaw to the 11 "correct" dies, keeping it well below the 90-degree error.

## 14. tfln_probe_station_closeup.jpg

The left probe lands on RF IN and the right probe on TERM. The IN lensed fiber (tip at 0.249,0.686) meets the IN waveguide at the facet (0.254,0.683). Both are correct.
- [must-fix] Probes (left 0.0-0.33, 0.0-0.60; right 0.63-0.72, 0.0-0.42): bare tubes floating in space, with no coaxial connector, no cable and no positioner arm. -> A microwave GSG probe is a body with a coax connector (2.92 mm / 1.85 mm / 1.0 mm class) on top, bolted to a micropositioner arm and coming in at about 30-45 degrees from the side. A semi-rigid or flexible coax runs to the VNA (vector network analyzer) or signal source, and the termination probe goes to a 50-ohm load or VNA port 2. -> Add probe bodies with connectors, coax cables and arm stubs going off-frame.
- [must-fix] The right probe comes straight down vertically from above (0.65, 0.0-0.40). -> That space is taken by the microscope. Probes always approach from the side, at a shallow angle. -> Tilt it to mirror the left probe, coming from the right or back side.
- [should-fix] Tip geometry: the blades are about 1.4 mm long and splay apart. On the right probe there is a strange black and white curved shroud (0.66-0.72, 0.25-0.38) and loose gold rectangles near the contact. -> Real GSG tips are a compact planar or coplanar structure, a few hundred um long, with three contacts converging at the pad pitch. -> Shorten them to a compact tip and delete the shroud.
- [should-fix] Left probe (0.19-0.25, 0.48-0.62): a black blob with three curved blades below the real tips reads as a second, deformed probe. If it is meant as the die-surface reflection, it is too sharp and dark, and it extends past the die edge. -> Reduce the die's specular, or clip the reflection at the die boundary.
- [should-fix] A hard-edged black shadow (0.43-0.75, 0.15-0.55) from an off-frame object crosses the die and partly covers the +Z arrow and label. -> A probe-station microscope with coaxial or ring illumination gives soft, nearly shadowless light from above. -> Remove the occluder or soften it.
- [should-fix] Electrode material (0.55-0.70, 0.42-0.55): the right half of the electrode has sparkly, granular, noisy gold, while the left half is smooth. This is a sampling or material artifact. -> Use one consistent gold material, and raise the sample count or denoise.
- [should-fix] OUT lensed fiber (0.78-1.0, 0.30-0.47): an opaque glossy black rod, like a stylus. -> A lensed fiber is clear glass, 125 um across, tapering to the tip, with bright edge highlights and a 250 um acrylate coating further back. -> Use a glass shader with a coating section.
- Inherits the waveguide/electrode layout issue from section 10 (the splitter arms end at the electrode bend, 0.30-0.35, 0.62-0.68).

## 15. fe/sim_solver_window_drawn.jpg (drawn mock-up)

- [must-fix] PML placement versus physics: the physics node is "solid mechanics, piezoelectric", yet the hatched PML wraps the beam's +y side and +x end as if the beam were embedded in an absorbing medium. A suspended nanobeam in vacuum has no medium to radiate into at its sides. A mechanical PML only goes in the solid that the beam is clamped to (substrate or anchors). An air PML only makes sense for an optical or electromagnetic problem. -> Show the beam clamped at its +x end into a substrate or anchor block with the PML beyond it, and leave vacuum (unhatched, or an air domain with no PML) beside the beam.
- [should-fix] The displacement field is a smooth monotonic left-to-right colour ramp, uniform across the beam width and ignoring the holes. -> A localized nanobeam mode has cell-periodic lobes that peak at the defect (symmetry plane 1), decay through the mirror cells, and vary across the width around each hole. -> Draw a decaying oscillating envelope with per-cell structure.
- [nit] The colour bar shows only "max/min", with no values or units. Real solvers show a numeric scale (e.g. displacement in arbitrary units or nm). -> Add a tick scale.
- [nit] Every model-tree node has a checkbox. Real model trees use node-type icons; checkboxes appear only for enable/disable states. -> Optional.
- Q = 7.1e13 for the quarter model, with "Crystal symmetry about these planes: not checked", reads as the intended lesson (symmetry boundary conditions on a rotated anisotropic crystal). No change needed if that is deliberate.

---

## Images that are fine or nearly fine

- pm_panda_endface_0deg.jpg: geometry, zone rings and key alignment are correct. Only should-fix and nit items.
- fc_apc_vs_pc_ferrules.jpg: APC angle and polish orientation are correct. The key and the floating are should-fix items.
- tfln_die_fixture_180deg.jpg and tfln_gel_tray_one_rotated.jpg: the internal rotation logic is correct. The remaining problems are inherited or should-fix items.

## Summary of must-fix items per image

| Image | Must-fix |
|---|---|
| butterfly_isolator_fc_bench | bolts not in table holes (owner); table holes are rings with no bore; no LD/TEC cables or connectors on mount |
| butterfly_isolator_reversed_bench | same three as fc_bench |
| fc_apc_vs_pc_ferrules | none |
| ln_wafer_carrier | wafer opaque matte off-white instead of clear/mirror LN; round inner disc plus a separate flatted band (reads as two objects); wafer interpenetrates carrier wall |
| ln_wafer_hotplate | wafer opaque cream decal with no reflection or thickness |
| pm_panda_endface_0deg | none |
| pm_panda_endface_90deg | none |
| pyro_discharge_comb_micrograph | "melted" area is a decal over intact fingers (no missing gold, bridges or LN crater/cracks); beads are faceted 5-10 um nuggets instead of small smooth spheres |
| sin_microring_lensed_fiber | top bus 24 um from ring, so uncoupled and meaningless |
| tfln_die_fixture_0deg | fiber channel not aligned with waveguide IN/OUT and runs under the die, IN/OUT labels not at slots; MZM arms end at electrode bend while CPW gaps curve into pads |
| tfln_die_fixture_90deg | square die in the same footprint cannot show "facets against closed pocket walls" (plus inherited items) |
| tfln_die_fixture_180deg | inherited from 0deg only |
| tfln_gel_tray_one_rotated | none |
| tfln_probe_station_closeup | probes have no connector, cable or positioner; right probe comes vertically from above |
| sim_solver_window_drawn | PML wrapped around a suspended mechanical beam instead of on the clamped anchor/substrate |
