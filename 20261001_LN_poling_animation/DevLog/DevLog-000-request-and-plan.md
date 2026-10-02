# DevLog-000: LN poling animation - request, decisions, progress

| Field | Value |
|---|---|
| Date | 2026-10-01 |
| Status | v1 + revisions 2-3 built and screenshot-verified (software GL only); user-approved look, committed 2026-10-01 |
| Scope | Animated visualization of periodic poling of x-cut (z in-plane) LiNbO3 |
| Authors | Claude, at Wentao Jiang's request |

## Request (verbatim, 2026-10-01)

Reference figure supplied: electrical poling measurement, panels (a)/(c) = current (nA, left axis) and voltage (V, right axis) vs time (0-10 ms) for Q = 45 pC and Q = 90 pC; panels (b)/(d) = PFM-type top-view images of the resulting domains (+Z / -Z) between +V and ground finger electrodes, scale bars 7.5 um and 10 um. Source publication not identified.

> i want a animated visualization of the poling process for LN, including the voltage and current graph, with a moving vertical dashed line and points on the curve moving accordingly, and the top view of the progressing domain, and a 3d crystal lattice with the proper atoms getting poled to flip the crystal z. Look up the LiNbO3 crystal structure from trusted sources, and show both a large multiple unitcell big lattice in which the domain front propagation is visible (exaggerated to be slow), as well as a zoomed in unitcell showing which atom get pulled to tunnel thru the lattice potential and flip the builtin polarization. Use proper different coloring for different atoms, and try your best to add additional visual effect for the unitcell rendering to show the crystal fields, the applyed fields ramp up and down, and slowly orbit the camera (for both overview of the lattice and the unitcell). Ask me any clarification questionos you have, Read memory on UI/UX style guideline and apply them as fit. Create a new folder under this playground repo and log the request and progress as you go.

## Clarification answers (2026-10-01)
1. Deliverable: interactive HTML only (no MP4).
2. Pulse: 90 pC only.
3. Domain model: hexagonal facets (3m symmetry).
4. Unit cell flip: Li through O plane + Nb in octahedron + double-well potential inset.

## Requirements
- [x] R1 Voltage + current vs time; moving vertical dashed cursor; markers on both curves.
- [x] R2 Top view of progressing domain, same time state.
- [x] R3 3D overview: multi-unit-cell lattice; domain front visible (playback slowed during switching); slow camera orbit.
- [x] R4 3D unit cell: atoms colored by species; Li and Nb flip path; crystal-field / applied-field effects; applied field ramps with V; slow camera orbit.
- [x] R5 Crystal structure from trusted sources, cached in `references/`.
- [x] R6 One shared simulation state (pulse time t) for all panels.
- [x] R7 UI per global taste: dark, sharp corners, dense, details in hover tooltips, no emoji.
- [ ] R8 User review of look and timing (pending).

## Crystal structure (sources in `references/README.md`)
- R3c, hexagonal setting, Z = 6 (30 atoms: 6 Li, 6 Nb, 18 O). Coordinates from Hsu et al., Acta Cryst. B53, 420 (1997), synchrotron X-ray 293 K, COD 2101845: a = 5.148, c = 13.863 A; Li (0,0,0.2806), Nb (0,0,-0.0010), O (0.04751, 0.34301, 0.0625). Cross-checked against Abrahams et al. 1966 (COD 1541936: a = 5.14829, c = 13.8631) and Parlinski et al. (cond-mat/9902274).
- Computed by `scripts/gen_lattice.py` from the R3c operators: Nb-O 1.876 / 2.130 A and Li-O 2.063 / 2.245 A (match Hsu 1997 bond table).
- Flip construction: down state = inversion of the up state about the centre of the Nb O6 octahedron (the paraelectric R-3c inversion centre); each atom paired with the nearest same-species atom of the inverted structure. Results in the O-fixed frame: Nb offset from octahedron centre 0.275 A (total flip travel 0.55 A); Li offset from its O3 triangle plane 0.713 A (total travel 1.426 A, through the triangle into the adjacent octahedral site); O in-plane twist 0.086 A, no z motion.
- Spontaneous polarization 0.71 C/m2 (Weis and Gaylord, via secondary literature; primary not opened). Not displayed numerically in the UI.
- Inbar and Cohen 1996 (arXiv mtrl-th/9508005): LiNbO3 double-well depth 18.3 mRy (2858 K) per 10-atom cell for coupled O + Li displacement; displacement of Li alone hardly changes the energy and Nb d0-O hybridization drives the instability. Only quoted in the unit-cell tooltip. The animation highlights Li (hop through the O plane) and Nb (octahedron) as requested; O relaxation is only the 0.09 A twist in the O-fixed frame.

## Model (all in `js/`)
- Waveform (`waveform.js`): voltage trapezoid read from figure panel (c): 0 V until 1.5 ms, ramp to 560 V at 2.7 ms, hold to 3.5 ms, linear fall to 0 V at 8.5 ms (50 us box smoothing). Current = C dV/dt (C = 0.1 pF; gives ~47 nA on the up-ramp and ~-11 nA on the down-ramp, matching the figure) + switching pulse (Gaussian rise sigma 0.06 ms, peak at 2.43 ms, exponential tail; peak 280 nA; area normalized to Q = 90 pC; total peak ~327 nA vs ~330 nA in the figure). Synthesized to the figure's features, not digitized; no noise added.
- Switched-area fraction f(t) = integral(I_sw dt) / Q.
- Geometry (corrected 2026-10-01, see Corrections): x-cut film. Image vertical in (b) = crystal z (polar axis, along the electrode-to-electrode field), horizontal = crystal y (period direction), surface normal = crystal x. Initial Ps = +z (toward the +V finger), E points from +V to ground (-z), poled domains have Ps = -z.
- Domain (`domain.js`, current form after Revisions 2-3): normalized period 1, gap 2.2 periods, 5 stripes parallel to z, attached to domed electrode edges, tapered wedge front with random needles, capped at 0.66 period width, reaching the far electrode; seeded jitter. Each point has arrival parameter g; the domain at time t is the set of points with g <= quantile(f) of the sorted g values, so switched area tracks the integrated current exactly.
- Atoms flip individually: trigger time = time at which the area fraction reaches the atom's rank; flip duration 0.05 ms (overview) or 0.08 ms (unit cell) of pulse time; smoothstep path. Overview: displacement from the paraelectric site drawn x2; deeper layers switch up to 0.04 later (schematic).
- Playback: 0.625 ms/s outside the switching window, slowed by 8x while the switching current exceeds 4% of its peak (domain front exaggerated slow). Pulse takes ~30 s at 1x.
- Overview (c): crystal (x,y,z) -> world (Xw,Yw,Zw) = (S - y, Lz - z, x); film 16 cells along y (period), 13 cells along z (180 A, electrode to electrode), 3 cells along x (surface normal); ~21k atoms. Per-cell Ps arrows lie along z (amber toward +V, blue after flip). Scale is schematic (real period ~10^4 cells), stated in tooltip.
- Unit cell: shown in the O-fixed frame at true scale; tracked Li and Nb are the in-cell atoms nearest the cell centre; tracked cell sits at normalized (0.109, 1.164) and flips at 2.45 ms (V = 444 V).
- Double-well inset: U(q) = (q^2 - 1)^2 + 1.2 (V/Vmax) q, schematic. The barrier is not removed at 560 V, so the hop is drawn as nucleation-assisted, not intrinsic switching (intrinsic switching would need a far larger field than the coercive field).

## Files
- `index.html`, `js/{waveform,domain,lattice,graph,topview,overview,unitcell,app}.js`
- `data/lattice_data.js` generated by `scripts/gen_lattice.py` (`uv run --with numpy python scripts/gen_lattice.py`)
- `vendor/` three.js r147 + OrbitControls (MIT), `references/` CIFs + README
- Run: open `index.html` directly (file://, no build, no server). URL params for testing: `?t=2.45&paused=1`.

## Verification (2026-10-01)
- Headless Chrome (software GL) screenshots at many pulse times across all revisions (final check t = 2.35-5.0 ms); no console or page errors; `node --check` passes on all JS; orbit and arrow toggles clicked and verified.
- Checked: cursor and markers follow t; domain nucleates near t ~ 2.3 ms and grows with the current pulse (Q readout 89.8 pC at 3.86 ms); overview domain overlay and per-cell arrows flip together; unit cell P/Ps goes +1 -> -1 at 2.45 ms with Li 1.43 A and Nb 0.55 A shifts; scrub by dragging the graph; hover readout; tooltips.
- Bug found and fixed: negative frame dt at start made t negative; near-zero f produced a stray domain pixel.
- Not verified: frame rate on a real GPU (software GL gave ~1.5 fps; scene is 21,866 instanced atoms + 702 arrows in (c) after the x-cut re-orientation, 47 atoms in (d)); the visual quality of the camera orbit over time (static frames only).

## Corrections
- 2026-10-01 (user): panel (c) was built as a z-cut slab with z out of plane, but the supplied figure's +Z/-Z arrows lie in the image plane along the electrode-to-electrode direction, i.e. x-cut (or y-cut) with z along/opposite to E. Fixed: panel (c) re-oriented as above; axis indicator added to (b); tooltips updated. Panel (d) was already z-up with the applied field along -z, so it is now consistent with (b) and (c). Verified by screenshots: flipped arrows inside the domain point along E (blue), the rest point toward +V (amber).
- Consequence: the "hexagonal facets" choice (answer 3) no longer applies literally to the top view; the 3m hexagonal prism cross-section would appear in the x-y plane (film thickness x period), which panel (c) does not resolve. Option: add an end-face cross-section view.

## Revision 2 (2026-10-01, user request with SHG microscope reference image)
- [x] Toggle for the per-cell built-in polarization arrows in (c) (button in the (c) header, default on).
- [x] Single run/pause toggle for the camera orbit of (c) and (d) (button in the top bar, default on).
- [x] Panel (b) restyled as an SHG microscope image: grayscale SH intensity (white = 1), dark domain walls (blurred edge map), poled domains slightly darker, grain, intensity darkening toward the electrodes, black electrode blocks (bars aligned with the stripes, light slits between), SH colorbar (0 black top, 1 white bottom), arrow row above the image showing local Ps per stripe and per gap (amber up, blue down), axis indicator. The contrast values (0.70 poled, 0.86 unpoled, wall darkening) are cosmetic, not simulated SHG.
- [x] Domain geometry: stripes now attach to the electrode edge (bus side, y = 0) instead of the finger tips; vertical walls start at the electrode edge and the front is a rounded (elliptical) cap, stopping just short (0.97 of the gap) of the far electrode. Finger caps removed. The same g(x,y) drives (b), the (c) overlay and all atom/cell trigger times.

## Revision 3 (2026-10-01, user request with PFM reference image of tapered spiky domains)
- [x] Domains now reach the far electrode (clipped at the plot range edge, y = LG) in the final state.
- [x] Final poled width 0.66 period (half-width 0.33 +-6% per stripe) vs electrode width 0.55 period, i.e. ~20% wider; wall wobble +-4% (smooth, per stripe).
- [x] Electrodes drawn with domed tips (half-ellipse, height 0.19 period, half-width 0.275); the domain starts at the dome surface (and ends at the far dome).
- [x] Sharp, randomly spiky front: front profile P(dx,s) = A1 s - 1.5 |dx| + needles (4-6 random triangular needles per stripe, advance 0.2-0.7 period, half-width 0.035-0.085, seeded RNG), so the domain is a tapered wedge with spikes that fills out to the capped width. Same g(x,y) feeds (b), the (c) overlay and the atom flip times. Grid raised to 140 px per period.
- Arrow row in (b) now samples just below the dome. Spike statistics are cosmetic, not fitted to the reference.

## Open items / possible next steps
- Review look and timing with user; tune slow-down factor, flip durations, atom sizes, exaggeration.
- Optional: 45 pC case, MP4 export (both declined for v1).

## Progress log
- 2026-10-01: folder created, request logged, clarifications answered.
- 2026-10-01: structure research, CIFs cached, `gen_lattice.py` run (displacements above).
- 2026-10-01: built all panels (initially z-cut, corrected above); fixed hex-skew index range in overview, domain front shape, glow intensity; added domain overlay plane on the lattice top surface; highlighted tracked Li/Nb with labels; fixed dt bug and stray pixel.
