# TX optics + PIC references (workstream T)

Date: 2026-10-10. Owner: workstream T (DevLog-008). Grades: A spec/datasheet, B vendor page/paper/patent, C photo-measured or photo-observed, D estimate (method and range stated).
Blog photos: `$BLOG` = the owner's `jwt625.github.io` repo (sibling of PlayGround). Web lookups were done one at a time (search engine + 2 page fetches), no bulk downloads; nothing was saved into `references/`.

## 1. Owner teardown photos (read directly)

| Photo | Module | What is visible (used for) | Grade |
|---|---|---|---|
| `$BLOG/assets/images/2025/20241215_20250915/20250913_031929_0.jpg` | Innolight 800G OSFP PSM8, top view (31.5 px/mm from the 22.58 mm shell, research/hero_component_dimensions.md) | Central TX optics block under a clear yellowish encapsulant/lid: upper half = laser sub-block (~9.2 W x 4.9 L) with two laser sites, each with gold pads, a small die and a round optic; a narrow glass strip runs down the middle between the two laser sites to the PIC edge (read as the TX fiber/FAU path: lasers and the TX output share the same PIC edge); lower half = PIC (~8.3 W x 4.4 L) with dense dark row structures, wire bonds along its left, right and bottom edges, two round features on top. RX: two fiber pigtail blocks left and right, each ending in a glass block over a PD/TIA area with gold bonds. | C |
| `.../20250913_031929_1.jpg` | same, oblique | Long rows of wire bonds along the PIC long edge; dark periodic rows on the PIC (electrode / fill structures); circular etched fiducials with "X" and "V" glyphs on the clear lid; round bumps over the laser sites. Basis for the fiducial circles, edge bond rows, and the glass lid. | C |
| `.../20250913_031929_2.jpg` | same, RX side | Glass block with an angled, AR-coated (green reflection) face over a die with a fan of short gold bonds. Confirms angled glass optics at a die edge. | C |
| `.../20250913_030659_0..3.jpg`, `20250913_030604_0.jpg` | Innolight 400G OSFP DR4+ | TX under a yellow gel dome: a row of 4 round lenses (dark circles with red cores) in front of 4 laser dies, a second row of 4 elongated lenses, dies with gold bond fans and pads; RX: 4 cyan AR-coated chips at ~0.75 mm pitch bonded to a TIA with orange metal routing, TIA output bonds to gold pads. Basis for "one lens per laser" and short paired bonds. | C |
| `$BLOG/assets/images/2025/20250425_20250425_long_thread/20250425_044131_1.jpg` | Cisco (Finisar) 400G DR4 SiPh | Ribbon fiber on a FAU with a glass lid sitting on the PIC edge; CW laser on a gold-pad submount with bonds, dark lens blocks and a cube between laser and PIC (owner's caption: "two CW lasers, going thru lens and two isolators, and then onto a modulator chip"). Basis for the lens + isolator + lens laser train. | C |
| `.../20250425_044131_2.jpg` | same | CW laser die on a submount, ~5 parallel gold bonds per pad, clear cube lens in front of the die. Basis for laser bias bond multiplicity and lens placement. | C |
| `.../20250425_044131_3.jpg` | same | SiPh PIC with a flip-chip EIC; 4 straight waveguides visible; double seal ring; fine fill texture; one edge of wire-bond pads to PCB pads with traces in pairs (differential) fanning out. Basis for the seal ring, fill texture and paired RF routing. | C |
| `$BLOG/assets/images/2025/20250117_20250127_long_thread/20250122_052349_0..2.jpg` | Intel-era 100G TOSA (EML) | Dark AR-coated silicon lens chip standing in front of the laser on a spacer; AlN submount with GSG lines, laser bias pad and capacitor. Basis for a standing Si lens chip on a pedestal. | C |

Notes: the blog captions say "mostly IIIV for the lasers and detectors, and some isolators and lens chips, hard to know" (2025-09-15 post). The 800G PSM8 round optics are not identified (lens or isolator).

## 2. Literature / vendor facts

| # | Topic | Fact used | Source | Grade |
|---|---|---|---|---|
| 1 | Si microlens, collimator | Focuslight 18-01342: silicon, L 1.2 x W 0.6 x center thickness 0.5 mm, clear aperture 0.500 mm, R 0.83 mm, conic -3.7 to -1.7, 1310 nm design, EFL 0.34 mm, working distance 0.194 mm, T > 98 %, R <= 1 % | focuslight.com datasheet "Rev01-Coupler-Collimator-18-01342" (Rev 01, 2024-08-05), text extracted from the PDF | A |
| 2 | Si microlens, focuser | Focuslight 18-01339: 1.0 x 0.6 x 0.5 mm, CA 0.500, R 1.1, EFL 0.45 mm, WD 0.30 mm, R < 0.5 % | focuslight.com "Rev01-Coupler-Collimator-18-01339" datasheet (search-result summary; not re-fetched) | B |
| 3 | Si lens physics | Silicon (n ~3.5) allows plano-convex lens chips with low spherical aberration for 1310 nm DFB/EML coupling; standard and custom sizes | Broadcom silicon lens product brief (docs.broadcom.com/doc/12398001, search summary) | B |
| 4 | DFB -> Si PIC edge coupler with lens + isolator | Lens between DFB-LD and the PIC input edge coupler to match spot sizes; isolator or silica spacer bonded to the edge-coupler facet with index-matching (UV) fluid, AR-coated toward the lens; tilt-polished facets and a silica cover lid for low return loss; FAU bonded to the output edge coupler with the same fluid | US 9,787,054 (SiFotonics, filed 2016-04-29, granted 2017-10-10), patents.google.com/patent/US9787054 | B |
| 5 | Isolator build | 0 deg polarizer + Faraday rotator + 45 deg polarizer (+ wave plate) stack | same patent family (US 2017/0219783) | B |
| 6 | Isolator in shipping DR8 | none found for DR8 (one TFLN DR8 patent lists lens + isolator); Credo/DustPhotonics DR8 is lensless butt coupling | research/hero_800g_osfp_dr8.md claim 1 | B / N/A |
| 7 | Laser-to-PIC miniaturization limit | lens working distance needed to magnify the laser mode limits miniaturization (grating-coupled laser alternative) | US 11,435,522 background | B |
| 8 | TW-MZM, differential GSSG | two signal lines over the two arms with lateral grounds (balanced coplanar stripline + lateral grounds), series push-pull, common doped return; GSSG/GSGSG give electrical isolation between channels | US 9,069,223 ; US 9,008,469 (search summaries) | B |
| 9 | TW-MZM termination | termination resistor matched to the line impedance; common-mode reflection depends on the GSSG termination structure | USPTO 11,287,720 / 9,448,426 (search summaries) | B (qualitative) |
| 10 | TW-MZM dimensions | silicon series push-pull TW-MZM, 35 GHz, 112 Gb/s PAM4 (Samani et al., IEEE Photon. J. 2015, doi 10.1109/JPHOT.2015.2426875); arm length ~2-2.5 mm used before (arxiv 1811.11096) | as cited | B (low) |
| 11 | Edge coupler | inverse taper SSC, narrow tip at the facet, NA matched to SMF, 1-2 dB loss targets; O-band ECs are harder (polarization, tip size) | US 9,638,857 ; US 11,719,895 (angled facets) ; Tidy3D EdgeCoupler example (C-band: 0.19 um tip, 25 um taper, 0.5 um out) | B |
| 12 | FAU attach | fibers in V-grooves under a glass lid, UV epoxy cured through the glass | US 9,316,796 / 7,203,410 (search summaries) | B |
| 13 | Laser-written glass fan-out (1D <-> 2D) | 3D laser-written glass waveguides can convert between 1D and 2D arrays (pitch converter / furcation) | US 10,684,419 | B |
| 14 | Glass fan-in footprint | fs-written 4-core MCF fan-in/out 2.5 x 2.0 mm, 0.54 dB at 1310 nm (Adv. Photonics Res., doi 10.1002/adpr.70241, search summary) | as cited | B |
| 15 | Stepped-mirror beam rearrangement | diode-bar beam shapers split a 1D beamlet line and re-stack it with stepped mirrors or mirror pairs | rp-photonics.com/beam_shapers.html ; FFI report "Beam shaping of high power laser diode bars" (2001) | B (qualitative) |
| 16 | SMF mode | G.652 MFD at 1310 nm 8.6-9.5 um (nominal 9.2) | ITU-T G.652 (standard value) | A |
| 17 | FAU / edge pitch | 127 or 250 um; MT-16 one row at 0.250 | research/hero_component_dimensions.md | B |
| 18 | PIC die size | Credo Carmel8 7 x 7.5 mm; Innolight PSM8 PIC area ~8.3 x 4.4 (photo) | research/hero_component_dimensions.md | B / C |
| 19 | Material indices at 1310 nm | fused silica 1.447 (Malitson Sellmeier); N-BK7 ~1.504 (Schott Sellmeier) | standard Sellmeier values, not looked up this session | B |

## 3. Design decisions and derived numbers (grade D unless noted)

All numbers below are computed in `scripts/assets/optics_tx/build_optics_tx.py` and written to `assets/components/optics_tx/optics_tx_parts.json` (section `optics_design`).

### 3.1 Architecture (follows the 800G PSM8 photo + the owner's request)

- Lasers and the TX output share the PIC front edge (PSM8 photo: glass strip between the two laser sites). Two CW DFBs at +/-2.5 mm (photo C) from the PIC centre line, each with a Si collimator lens (18-01342 class), a micro isolator (Cisco 400G DR4 photo, US 9,787,054; flagged optional because none is documented for DR8), and a Si focusing lens (18-01339 class) into a laser-side SSC.
- 8 MZM lanes, two 1x4 splitter trees (one per laser), output SSCs in a 1x8 row at 0.25 mm pitch in the middle of the same edge.
- Output optics, all free space, no fiber (the film eye): (1) FAU-attach collimator block bonded to the edge facet (fused silica spacer + 1x8 lens array on its exit face, 0.25 pitch); (2) 45 deg TIR staircase turning prism: each lane has its own 45 deg facet in its own 0.25 mm band, placed at the target column, turning +Y -> +X (out of the eye); (3) row-shift rhomboid micro-prisms (periscopes) that move six lanes up or down by 0.25 or 0.5 mm so the lanes land on two rows; (4) 2x4 eye plate = per-lane Galilean beam expander (concave lenslet on the inner face, convex lenslet on the outer face).
- Why not a laser-written glass fan-out: the lateral moves are up to ~1.3 mm; low-contrast fs-written waveguides need bend radii of many mm (estimate, no source found for a specific minimum radius), so the fan-out would be ~10 mm long and would not fit behind the eye. Free-space stepped mirrors (fact 15) fit in ~1.5 mm.
- Why not divergent beams straight to the eye: an SMF-like 9.2 um mode diverges to w ~0.25-0.4 mm over the 3-4 mm paths, more than the 0.25 mm lane spacing at the staircase and more than the rhomboid apertures; collimating at the facet first is required.

### 3.2 Gaussian beam numbers (method: paraxial Gaussian optics, thin-surface lens formulas; ranges from the assumed mode sizes)

| Quantity | Value | Method / assumption |
|---|---|---|
| Output SSC mode | w0 = 4.6 um (MFD 9.2) | SSC matched to G.652 SMF (fact 16); range 4.3-4.75 |
| Collimator block | fused silica, 0.80 mm long; lens R = (n-1) L / n = 0.247 mm, aperture 0.22 | collimates the w0 source after L in glass (reduced distance L/n) |
| Collimated beam | w 0.050 mm (z_R 6.1 mm); 0.058 mm at the eye plate after the longest 3.6 mm path | w = w0 sqrt(1 + (L_red/z_R0)^2); fits the 0.25 mm band with > 2w margin |
| Staircase TIR | N-BK7 class: 45 deg incidence vs critical 41.7 deg (3.3 deg margin; fused silica would leave only 1.3 deg) | TIR holds; glass-air back faces |
| Rhomboid aperture | 0.20 mm | 1.7 x the 1/e^2 beam diameter at the rhomboids (0.117 mm); truncation ~0.1 % |
| Eye expander | M = 3.5 -> output w 0.176-0.204 mm (0.204 for the longest lane, which arrives at w 0.058); plate 1.2 mm N-BK7-class; R_in 0.161 (concave, aperture 0.22), R_out 0.563 (convex, aperture 0.70) from the paraxial vergence equations; truncation < 0.3 % | Galilean in one plate (concave in, convex out) |
| Rayleigh range of the eye beam | 74 mm | z_R = pi w^2 / lambda |
| DFB mode | w0 1.0 um assumed (0.8-1.1 um) | paraxial w0 = lambda / (pi theta_1/e2) for an assumed 25-35 deg FWHM far field; no DFB datasheet value found |
| Laser train | DFB -> 0.194 WD -> 18-01342 (EFL 0.34) -> isolator (pol / latching garnet / pol / half-wave plate) -> 18-01339 (EFL 0.45) -> 0.30 WD -> SSC | magnification 0.45/0.34 = 1.32 -> laser-side SSC MFD ~2.6 um; collimated w 0.142 mm inside the 0.5 mm CA (0.2 % truncation); the half-wave plate rotates the 45 deg isolator output back to TE |

### 3.3 PIC layout (texture + raised pads; all D unless noted)

| Feature | Value | Basis |
|---|---|---|
| Die | 7.0 (along the optical edge) x 7.5 x 0.75 mm | Carmel8 (B, a published DR8 SiPh PIC size) kept over the PSM8 photo estimate ~8.3 x 4.4 (C, +/-20 %, different vendor and a PSM8 not DR8) and for continuity with the hero; thickness D (0.15-0.78) |
| Lane pitch / MZM electrode length | 0.78 / 2.4 mm | hero D / papers 2-2.5 mm (B low) |
| RF electrode | GSSG per lane, light and RF co-propagate (RF enters from the back-edge pads), on-chip termination at the far end (2 x 50 ohm to G, 100 ohm differential, D) | facts 8-9 |
| Thermal phase shifters | TiN-style heater meanders on both arms after the TW section | standard bias control (D) |
| Monitor PD | ~2 % tap after each MZM to a Ge PD | Credo "monitor PDs" (research hero claim 2), tap ratio D |
| Splitter tree | 1x2 MMI -> 2x 1x2 MMI per laser | 2 lasers x 4 lanes (research) |
| Pads | back edge: 25 RF pads (G S S G ... shared G) 0.27 mm pitch; side edges: heater, MPD and common pads | photo C (3 bonded edges), counts derived |
| Alignment | corner fiducial crosses, vernier marks, two loopback SSC pairs on the 0.25 mm grid (positions 1-2 and 11-12 of a 1x12 lens array) for active alignment | common practice (D) |
| Dummy fill | metal fill squares with keep-out around waveguides and RF lines | foundry density rules (D) |
| Seal ring | double ring inset from the die edge, opened across the facet zone (no metal over the SSCs) | Cisco PIC photo (C) |
| Markings | parody only ("ROCKDOVE PHOTONICS", "PGN-8TX") | project rule |
| Drawn widths | waveguide cores 3.5 um and SSC tips 2.4 um on the texture (real ~0.4 / ~0.2 um) for render legibility | stated exception to true scale |
| Waveguides under RF feed metal | the splitter tree and laser feed pass under the wide top-metal RF feed tapers (v 6.55-7.2), separated by the BEOL oxide | layout simplification (D); a production layout would route them through feed gaps |
| Driver | no driver EIC on the PIC: research claim 2 locks DSP-integrated SiPh direct drive (Broadcom BCM87800, MaxLinear Keystone); the Cisco 400G DR4 photo shows a flip-chip EIC, which is a different architecture | research |

## 4. Open questions

1. PSM8 round optics: lens or isolator, and their exact count per laser (photo resolution).
2. Whether the PSM8 lasers really share the FAU edge (photo interpretation, C).
3. Real DR8 TX has no 2x4 output (16-fiber MPO); the 1x8 -> 2x4 free-space rearranger is a film device built from real elements.
4. No public SiPh DR8 PIC layout; the PIC layout is a typical-practice composite.

## 5. Audit corrections (2026-10-10)

See DevLog-008 Section 6.
