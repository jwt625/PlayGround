# Mining: long-form blog posts + PlayGround photonics logs

Date: 2026-11-22. Scope: `jwt625.github.io/_posts/*.md` excluding `weekly-OFS*`, `OFC2026-notes.md`, `2026-04-04-OFC2026.md`; plus PlayGround folder names and README/DevLogs of photonics projects. Read-only on sources.

Path shorthand: `P/` = `jwt625.github.io/_posts/`, `G/` = `PlayGround/`. Line numbers are source-file line numbers (1-based).

Confidence key: **S** = stated in source (his own log/post); **O** = his opinion/guess, flagged as such in source; **I** = my inference (not in source); **L** = PlayGround project log written by an agent at his direction (stated, but agent-authored).

Population estimates are my rough guesses (**I**), reasoned from the number of groups/companies doing the activity; treat them as order-of-magnitude only.

Biggest finding: the 2018-2020 LN logs are a near-continuous diary of the same family of mistakes (wrong crystal cut/Euler angle, symmetry shortcuts, tensor-notation factors). This is the richest and most personal material, and it is self-documented with timestamps and exclamation marks ("Idiot! You also didn't change the coord system!!!").

---

## Cluster A. Anisotropic Crystal Bookkeeping (LN/LT tensor and orientation hygiene)

- Experts: ~100-500 (I). People who rotate LN/LT elastic, piezo, photoelastic, EO tensors in FEM for X/Y-cut thin film: TFLN piezo-optomechanics, acousto-optics, EO modulators. Tens of groups, a few people each.
- Why niche: requires simultaneous Voigt notation, Euler conventions (ZXZ), solver-specific matrix ordering, and knowledge of which LN coefficients are "measured vs primary".
- Visual: Blender: square LN chip spinning on a tweezer tip, crystal axes (X, Y, Z arrows) swapping each time someone looks away; a wafer flat that nobody can find. Interactive: slider over in-plane angle theta drawing p22'(theta) for LNX vs LNY using his closed-form rotated tensor (P/2018-09-12 Final results); Euler ZXZ gimbal toy showing "rotate crystal" vs "rotate coordinates" giving opposite sign.

| # | Pitfall (precise) | Why experts miss it | Gag / visual | Source | Conf |
|---|---|---|---|---|---|
| A1 | Running months of optimization in the wrong material coordinate system (bender's Y-cut "coordB" instead of X-cut "coordA") silently gives plausible-looking but wrong Q and couplings. | The model still solves; nothing errors. Coordinate frame is set once in a template and copied. | Incident log: "Why are you keep simulating and optimizing in the wrong material coordinate???" | P/2018-01-19-lnnb-full-beam-simulations-1801-1805.md L343; L263, L288; P/2018-04-28-ln-properties.md L76 ("careful! all old simulations ... are with this coordinate!") | S |
| A2 | LN Euler angle sign error: (-pi/4, pi/2, pi/2) used instead of (pi/4, pi/2, pi/2). | 45 deg vs -45 deg (or 135 deg) in-plane looks symmetric to intuition; 3m symmetry is not mirror-symmetric under that flip. | "Spot the minus sign" chapter quiz | P/2018-06-01-lnnb-full-beam-simulations-1806-1812.md L890-892 | S |
| A3 | COMSOL Euler alpha = 135 deg corresponds to a 45 deg physical in-plane rotation; plots titled "45 deg" were actually 135 deg. | Solver angle convention differs from the lab-frame angle you think in. | Header literally reads "At 45(wrong) 135 deg" | P/2018-09-12-rotation-of-photoelastic-tensor-and-piezoelectric.md L198; P/2018-06-01-...1806-1812.md L972 | S |
| A4 | Euler angles in COMSOL are ZXZ and rotate the coordinates, so the same matrix maps rotated-frame quantities back to global; working in global frame needs the Euler matrix, working in material frame needs its inverse. MATLAB lacks ZXZ, so he built it manually. | Active vs passive rotation ambiguity. | Two trainees rotate the crystal and the room in opposite directions; result is identical apart from a sign | P/2018-09-12-... L126-129 | S |
| A5 | Reproducing a thesis' rotated photoelastic tensor agreed except for the sign of theta: the source meant rotating the crystal, not the coordinates. | Thesis convention not stated explicitly. Cited: Jasper Chan thesis p.97. | | P/2018-09-12-... L84 | S |
| A6 | Component transformation is the inverse (and transpose) of the basis-vector transformation; at special angles they coincide, hiding the bug. | Tests at 0/90 deg pass. | "Works at 90 deg" certificate | P/2018-01-03-ln-memos-simulations-2018.md L112 | S |
| A7 | Voigt notation: no extra factor 2 for elasticity c, but piezo d shear components carry a factor 2 (his dmat2dtensor uses coef=2 for shear); mixing them corrupts rotated tensors. | Each tensor family has its own convention; textbooks differ (strain vs engineering strain). | | P/2018-06-01-...1806-1812.md L726 ("No extra 2s for Voigt notation of elasticity tensor c", with "wrong" vs "corrected" figures); P/2018-09-12-... piezo section (~L218-245) | S |
| A8 | COMSOL scripting order for the 6x6 symmetric D matrix is {D11, D12, D22, D13, D23, D33, ..., D56, D66} (upper triangle column-wise), not row-major Voigt. | Pasting a row-major list gives a valid-looking but wrong stiffness. | | P/2018-04-28-ln-properties.md L99-100; repeated P/2019-06-13-piezoelectric-k-square.md L251 | S |
| A9 | Permittivity superscript T (unclamped, low frequency) vs S (clamped, high frequency) differ substantially for LN; using the wrong one at GHz mis-estimates fields and transducer admittance. | Tables list both without context. His transducer estimate uses eps_T~80 vs eps_S~40 (P/2025-04-02 L75). | | P/2018-04-28-ln-properties.md L84 (Weis and Gaylord, Appl. Phys. A 1985) | S |
| A10 | Measured LN photoelastic p13 and p33 include a significant secondary (piezoelectric + linear EO) contribution; "primary" values differ from measured ones. | People copy p-tables without the measured/primary distinction. | | P/2018-04-28-ln-properties.md L199-200 (K. K. Wong, Properties of LN) | S |
| A11 | At "45 deg" in-plane rotation, X-cut and Y-cut give different effective p22' (0.304 vs 0.223) because X-cut picks up p14/p41 terms. | "Same angle, same material" intuition. | Side-by-side tensor heatmaps | P/2018-09-12-... L213-216; P/2018-06-01-... L758-761 (values from Andrushchak et al., JAP 106, 073510 (2009)) | S |
| A12 | Roto-optic effect (rotation of an anisotropic index ellipsoid by local curl of displacement) only creates off-diagonal permittivity change proportional to birefringence; estimated g_RTO <~ 4% of g_PE for his nanobeam. | Usually ignored entirely; nonzero only for anisotropic media. | | P/2018-10-13-roto-optic-effect-estimation-on-lnnb-181013.md L20-54 | S |
| A13 | Simulating a phonon shield in Z-cut when the device is X-cut ("This is Z cut, need to redo"). Repeats in 2026: LN poling animation first built as z-cut, but reference figure was x-cut (z in-plane). | Cut is a single dropdown; figures rarely label it. | Recurring-offender badge | P/2018-07-18-ln-2d-phonon-shield-simulation-180718.md L48; G/20261001_LN_poling_animation/DevLog/DevLog-000-request-and-plan.md "Corrections" (2026-10-01) | S / L |
| A14 | "Do not use square or near-square chips for anisotropic materials." | Orientation is unrecoverable after dicing if chip is square and unmarked. | Square chip, four identical edges, crystal axes question mark | P/2024-10-13-yapping-on-ebeam-lithography.md L165 (parenthetical) | O |
| A15 | Comparing LN vs LiTaO3: LT has larger stiffness but much smaller piezo d (e.g. d15 2.6e-11 vs 6.8e-11 C/N in his COMSOL listing), so "LT is not as piezoelectric" despite similar permittivity. | LT sold as "LN with better optical damage"; transducer physics changes. | | P/2019-09-27-comparing-ln-and-litao3.md (d-matrix listing, conclusion line) | S |
| A16 | LN switching: displacing Li alone barely changes the energy; Nb d0-O hybridization drives the ferroelectric instability; at 560 V the double-well barrier is not removed, so poling is nucleation-assisted, not intrinsic. | Textbook cartoon shows Li hopping through the O plane as the whole story. | Unit-cell animation already exists | G/20261001_LN_poling_animation/DevLog/DevLog-000 (Inbar and Cohen 1996, arXiv mtrl-th/9508005) | L |

---

## Cluster B. Multiphysics Simulation Self-Deception (symmetry, BCs, PML, normalization)

- Experts: ~500-2,000 (I) for photonic/phononic FEM-FDTD users; the specific COMSOL piezo-eigen and Meep oblique-port items maybe ~100-300.
- Why niche: each failure produces numbers that look physical (Q 1e13, efficiency 7%), not errors.
- Visual: interactive: toggle "symmetry plane ON/OFF" and watch Q jump 10 orders; PML slider where Q goes up when PML is weakened; flux monitor dragged toward an interface reporting "gain". Blender: a quarter-model nanobeam whose missing three quarters are hidden behind a curtain ("trust me").

| # | Pitfall | Why experts miss it | Gag / visual | Source | Conf |
|---|---|---|---|---|---|
| B1 | Mechanical Q with symmetry planes: 7.1e13 / 2e12 / 2e10; full geometry, no symmetry, same (bender) coordinate: fundamental Q ~2200 (leaky). Anisotropic crystal + symmetry plane enforces symmetry the crystal does not have. | Symmetry planes are standard practice for isotropic Si OMCs. | Q-meter needle pinned at 1e13 | P/2018-01-19-...1801-1805.md L250-270 | S |
| B2 | Weakening PML made Q higher, not lower: Q insensitive to PML strength indicates the loss channel is not being captured at all. | "Q too high, PML too strong?" reflex. | | P/2018-01-19-... L258 | S |
| B3 | COMSOL cyclic symmetry behaved like reflection, not rotation, giving the wrong mode profile, even with copy-face meshing; full geometry gave the correct cyclic-symmetric mode (55 min vs 22 min). | Docs claim it should work. | | P/2019-01-09-lnnb-full-beam-simulation-201901.md L42-55 | S (COMSOL version not stated) |
| B4 | COMSOL piezo eigenfrequency study: a Voltage terminal is treated as short circuit (setpoint ignored) giving the resonance; a Charge terminal with Q0=0 is open circuit giving the anti-resonance, which is what a qubit/MW resonator couples to. Two eigen runs give k^2 (20.7% in his toy model). | You set V=1 and expect a driven bias; eigen solver ignores it silently. | Two identical buttons, one labeled "short", one "open" | P/2019-09-03-piezoelectric-eigen-mode-sim-with-different-es-boundary-on-metal.md L60-95; P/2019-06-13-piezoelectric-k-square.md L346-350 | S |
| B5 | Piezo energy bookkeeping: E*D already includes T*d*E; naive Berlincourt k^2 = Um^2/(Ue Ud) evaluation "does not work"; absolute energies are meaningless in eigenmode solutions. | Energy terms double count across constitutive forms (stress-charge vs strain-charge). | | P/2019-06-13-piezoelectric-k-square.md L270-300, L350 | S |
| B6 | Sidewall-angle geometry bug: bottom hole width computed as hx - t tan(theta) instead of hx - 2 t tan(theta) (both sides taper); invalidated a GA run. | Off-by-two in geometry, not physics. | | P/2018-01-11-lnnb-unitcell-simulations-1801-1805.md L188-190 | S |
| B7 | Using "nearest measured" sidewall angle per hole size made angle jumpy cell-to-cell, scattering light and capping optical Q < 1e4; smooth x^4 angle model fixed it. | Measured data inserted naively into an adiabatic taper breaks adiabaticity. | | P/2018-01-19-... L327-336 | S |
| B8 | Dataset plumbing: "select -> mirror -> join" gave zero interaction integrals; "mirror -> join" worked. Zero g_om is a pipeline artifact, not physics. | Integral silently returns 0. | | P/2018-06-01-...1806-1812.md L26-29, L196-199 | S |
| B9 | Radiation-pressure g_rp wrong because hole sidewalls were not selected as boundaries for the surface integral. | Surface selection set for outer walls only. | Highlighter missing the holes | P/2018-06-01-... L478 | S |
| B10 | Mode classification in symmetry-broken anisotropic unit cells (LNX 135 deg): TE/TM-like and sym/asym labels need ad-hoc semi-symmetry metrics; several attempts failed ("I don't understand mechanics"). | Band diagrams normally rely on parity labels. | | P/2018-06-23-lnnb-unitcell-simulations-1806-1812.md L417-450 | S |
| B11 | Overwrote a GA result by forgetting to change the output filename. | | "GA180508 got overwritten!!!!" | P/2018-01-19-... L342 | S |
| B12 | Meep flux convention: custom Poynting sum multiplied by 0.5 while `dft_flux` does not -> factor-of-2 efficiency error. | Time-average 1/2 vs DFT normalization. | | G/20260918_volumetric_edge_coupling/reports/R1_R4_diagnostics.md "R2 - flux convention" | L |
| B13 | Transfer-matrix slab solver in tan form returned spurious higher-branch roots (poles) reported as TE1/TM1 modes; fix: undivided residual and residual threshold. | Root finders find poles of divided residuals. | | same file, R1 and "Corrections" item 1 | L |
| B14 | Meep subpixel smoothing shifted n_eff by ~0.02 for a 262 nm Si layer at res 200; comparable to tolerance-sweep effects. | Smoothing is on by default and "improves accuracy". | | same file, R1 | L |
| B15 | Flux monitor near the interaction region sampled evanescent/TIR field: "net 7.4%" vs 1.4% forward modal power. | Monitor placement looks innocent. | Monitor dragged toward interface shows fake gain | same file, R4 | L |
| B16 | Meep stores kpoints as a flattened band*nfreq+freq table; misindexing reported two adjacent frequencies of band 1 as bands 1 and 2, producing a false "phase-matched" claim (retracted). | API detail. | | same file, Corrections item 2-3 | L |
| B17 | Paper's "TE" label may map to the Meep Hz branch (E in plane) for an inclined guide; polarization naming is geometry-dependent. | TE/TM names assume a slab orientation. | | same file, R3 | L |
| B18 | Run a 1x2 MMI backward with out-of-phase inputs: power goes "everywhere" (radiates) since the antisymmetric input has no matching single-mode output. | Reciprocity intuition stops at the forward case. | His own field animation exists ("invisible modes") | P/2025-02-27-SiPho.md L225; P/2026-08-20-silicon-photonics-and-co-packaged-optics.md L357 (quiz) | S |
| B19 | "You'll always forget something in your circuit level simulations, whether it is backreflection, other modes, or stray light ... it will never match your measurement." | | | P/2026-08-20-... L359 | O |
| B20 | Strict-OFL GRIN optimum sits just before a mode group crosses cutoff: sigma(alpha) is a sawtooth; the "optimum" is not robust to ~1% Delta or a few nm. | Optimizers find cliff edges. | Sawtooth plot with a cartoon optimizer standing at the cliff | G/20261106_uled_grin_profile/DevLog/DevLog-000 progress 2026-11-07 00:10 | L |

---

## Cluster C. Piezo Transduction and Phononic Waveguides ("How to (not) excite a tiny mechanical wave")

- Experts: ~50-300 (I). GHz phononic integrated circuits on TFLN/LNOS/AlN, quantum acoustics; perhaps 10-30 groups.
- Why niche: requires both acoustics boundary physics and RF impedance intuition; failure masquerades as success via SAW leakage.
- Visual: his own 2D toy-model animations (free vs clamped boundary taper) are directly reusable; interactive: taper width slider showing fundamental and first-order shear becoming degenerate edge modes; impedance calculator G ~ omega eps0 (epsT - epsS) A/d. Blender: wide IDT shouting into a narrow waveguide while the sound sneaks around the outside (SAW).

| # | Pitfall | Why experts miss it | Gag / visual | Source | Conf |
|---|---|---|---|---|---|
| C1 | Straight taper from a wide IDT to a wavelength-scale waveguide with free (soft) boundaries fails: as the guide widens, fundamental and first-order shear become nearly degenerate edge-localized modes with poor overlap to a centered IDT. | EM horn-antenna intuition: metal walls are "hard" boundaries; free solid surfaces are "soft". | Horn antenna vs. acoustic horn side by side | P/2025-04-02-how-to-excite-tiny-mech-wave.md L103-123, L146 | S |
| C2 | A longer, slower taper does not help; fixes are stopping before the guide gets too wide, or a non-adiabatic focusing IDT. | Adiabaticity reflex from photonics. | "Make it even slower" chapter | same, L148-152 (Dahmani2020 doi:10.1103/PhysRevApplied.13.024069; Peairs2020 doi:10.1103/PhysRevApplied.14.061001); critique of Deng2025 arXiv:2503.18113 at L172-173 | S/O |
| C3 | With a hard boundary (cladding with higher acoustic velocity, e.g. SiN on LN) the mode fills the wide guide and a straight taper works (~10% S21 cold, >30% single-sided incl. taper). | | | same L125-141 (Shao2022 doi:10.1038/s41928-022-00773-3) | S |
| C4 | Do not trust waveguide S21 unless the waveguide is released or bends; unreleased guides can show S21 dominated by SAW not in the guide (example -20 dB unreleased vs -50 dB suspended). | Transmission exists, so it "works". | Sound taking the scenic route | same L155-168 (Ding2024 PhysRevApplied.21.014034) | O |
| C5 | Wavelength-scale piezo transducer (~1 um^2, 1 um gap, LN) has conductance ~1/(450 kOhm): >99.9% reflected at 50 Ohm; need area, narrow band (Q), higher eps difference. | Small-is-better intuition from photonics. | 50-Ohm bouncer at the door | same L55-91 | S |
| C6 | Microwave-to-single-structure efficiency ~1e-8 when a big IDT illuminates an array of wavelength-scale resonators. | | | same L163 (Forsch2020 doi:10.1038/s41567-019-0673-7) | S |
| C7 | "Relative transmission" normalization can hide absolute insertion loss ("We 'cheated' ... shame on us"). | Normalized plots look great. | | same L42 | S |
| C8 | "Be wary of +-30 dB engineers/engineering." | | Closing slide | same L176 | O |
| C9 | Phononic waveguide mode leaks into the nearest available mirror-cell mode, converting mostly in-plane motion to mostly out-of-plane; frequency shifted +400 MHz when attaching a waveguide. | Bandgap of the mirror is for one polarization family only. | | P/2019-03-19-lnx-pnc-to-wg-20190319.md L39-61 | S |
| C10 | Phonon shield tethered at two points: "bad idea". Lamb mode at X bends down when a node crosses the tether. | | | P/2018-07-18-ln-2d-phonon-shield-simulation-180718.md L115-118 | S |
| C11 | Piezo bender mirror symmetry: mirroring the electrode wiring mirrors the field, flipping bending direction where the same bending was desired. | Geometric mirror is not a symmetry of a polar crystal + field. | | P/2019-09-19-tilt-mirror-with-bender.md L110 | S |
| C12 | Bender coupled to a nanobeam is "locked by the bender-NB joint": 1 V displacement drops from 3.73 nm (bender alone) to 0.19 nm. | Load stiffness ignored in standalone actuator sims. | | P/2018-01-03-ln-memos-simulations-2018.md L492-506 | S |

---

## Cluster D. E-beam Lithography Silent Failures (for nanophotonic devices)

- Experts: deep users of JEOL/EBPG aligned writes with PEC: ~1,000-5,000 worldwide (I); the specific tool quirks (JEOL MDRG/CHIPAL, EBPG back-referenced holders) a few hundred.
- Why niche: the scary errors are the ones the tool never reports. He logged ~420 sessions.
- Visual: Blender: cassette with a protruding reference chip and a clumsy hand; thermal-diffusion timer after finger touch; writefield grid with a stitching fault running through a photonic crystal. Interactive: PSF with two length scales and a nonlinear-development threshold slider showing why PEC can resolve a missing shot; DAC dwell-time calculator (dose x pitch^2 / current vs 1/50 MHz).

| # | Pitfall | Why experts miss it | Gag / visual | Source | Conf |
|---|---|---|---|---|---|
| D1 | Most dangerous errors produce no tool error: wrong pattern extent/center from fracturing (everything shifted by the same large amount), typo in offset/holder, wrong layer or writefield path. | Tool happily exposes. | "Watch where the beam is on the cassette" | P/2024-10-13-yapping-on-ebeam-lithography.md L470-474 | S |
| D2 | Forgetting to deselect the manual-writefield helper layer in BEAMER exposed the enclosing squares over the actual pattern. | Helper layers live in the same file. | Squares stamped over a ring resonator | same L271 | S |
| D3 | Pattern origin mismatch is "the leading cause of a failed aligned beamwrite". | | | same L269 | O |
| D4 | Multipass: the tool does not know you are doing multipass; you must divide the dose yourself. | | Double dose stamp | same L268 | S |
| D5 | PEC corrects mainly the long-range backscatter term; short-range PSF is not deconvolved; any PEC parameter change needs a new dose sweep. PEC works only because development is nonlinear. | PEC treated as a magic checkbox. | | same L285-294 | S |
| D6 | Touching the cassette: ~0.1 C stability needed for 100 nm over 200 mm (CTE 1-2e-5); thermalization 1-100 s scale by diffusivity (Al ~100 mm^2/s), wait ~10x. | Stage interferometer measures stage, not the warm holder. | Finger heat-map on cassette | same L214-219 | S (estimate in source) |
| D7 | Rarely used 100 nA+ beams may be out of calibration such that calibration itself fails. | | | same L172 | S |
| D8 | Titanium 5 nm discharge layer got hot enough to cross-link the top of the resist; prefer low-melting metals (Al 5-10 nm); water+IPA can partially develop some resists when removing discharge layers. | Discharge layer assumed inert. | | same L326-327 | S |
| D9 | Alignment marks near (not under) the clamp edge can show drift during beam imaging. | Charging/field near clamp. | | same L374 | S |
| D10 | "Rotation angle too big" often means the auto-align found the wrong marker from another set; label markers. | | Radar operator locking onto a decoy | same L469; L423-424 | S |
| D11 | Height detection (grazing LED onto CCD) fails on transparent chips or hits the clamp. | Transparent substrates (sapphire, LN, quartz) common in photonics. | | same L467 | S |
| D12 | Global coordinates have +y pointing down; JEOL optical alignment microscope flips y because cassette is upside down. | | | same L389 | S |
| D13 | JEOL low-current writefield is 62.5 um (1000 um / 2^4): stitching lines through small photonic devices. | | | same L529 | S |
| D14 | Single-layer CSAR liftoff works because exposure leaves a negative sidewall; narrow meandering gaps fail liftoff in the center where solvent cannot reach. | Single-layer liftoff believed impossible. | | same L308, L320 | S |
| D15 | Dolan-bridge angled evaporation hits the lower resist sidewall, leaving a vertical flap after liftoff. | | | same L302 | S |
| D16 | EBPG: reference chip protrudes from the holder ("so you'll bump into and crack it, and they could sell you more"); back-referenced holders need tilt/height reset when previous user had different chip thickness. | | Accident-prone protrusion | same L366 | O |

---

## Cluster E. Thin-film LN Fabrication and Material Side Effects

- Experts: TFLN fab people at universities/foundries: ~300-1,000 (I).
- Why niche: LN-specific (pyroelectric, photorefractive, anisotropic etch), invisible to Si photonics habit.
- Visual: Blender: a-Si hardmask crazing into cracks that transfer into LN like a dried riverbed; hotplate on Z-cut LN making resist glow (pyro charge); hexagonal domain growth. Interactive: thermal bistability triangle (cold vs hot detuning) with a power slider.

| # | Pitfall | Why experts miss it | Gag / visual | Source | Conf |
|---|---|---|---|---|---|
| E1 | a-Si hardmask on LN cracked and etched inhomogeneously under ion milling; mask eroded from edges inward; cracks transferred into LN edges. Proposed fix: anneal a-Si to poly-Si. | a-Si hardmask is standard in SOI-like flows. | Dry-riverbed crack animation | P/2018-08-20-lnnb17-fab-si-hardmask-pattern-test-180820.md L158-213 | S |
| E2 | "LNY is hard to etch"; HSQ-masked Y-cut sidewall tilt 15-20 deg depending on structure; holes 22 deg vs outer sidewall 12 deg; tilt differs for +x and -x directions. | One sidewall angle assumed per process. | | same L159; P/2018-07-18-... L44-46; P/2018-06-01-... L538-540; P/2018-01-19-... L293 | S |
| E3 | Heating Z-cut LN to 120 C could expose ZEP resist (pyroelectric charge). | Resist bake is routine; pyroelectric field invisible. | Hotplate "exposes" resist without a beam | P/2018-04-28-ln-properties.md L268 (Mostafa et al. 2006, no DOI in post) | S |
| E4 | Photorefractive "optical damage": photoexcited carriers trapped along polar axis create space-charge field, changing index via EO effect; reversible. MgO:LNOI quoted 170x higher damage threshold. | Name "damage" suggests permanent physical damage. | | P/2018-04-28-ln-properties.md L219-238 (Jermann et al., JOSA B 1995); P/2018-04-26-ln-photonics-literature.md L110-111 | S |
| E5 | Quadratic thermo-optic shift in LN nanobeam cavity (alpha/2pi = 34 Hz/n^2) with hot vs cold detuning; thermal relaxation rate depends on intracavity photon number. | Linear thermal-shift model assumed. | Triangle resonance interactive | P/2018-12-21-thermal-shift-numerical-calculation-20181221.md L28-50; P/2019-01-08-lnnb-paper-outline-and-figures-20190108.md L99-128 | S |
| E6 | 50 nm minimum feature mirror cells were abandoned because electrode alignment to them was too hard. | Optical optimizer ignores downstream alignment layers. | | P/2018-06-01-... L538 | S |
| E7 | Insulating substrates (sapphire, LN) charge up: LN-on-sapphire SEM shows bright charged region; discharge layer and clamp contact needed; evaporated discharge metal can be missing at edges due to evaporation-holder clamp shadows. | | | P/2024-10-13-... L312, L326, L349 | S |

---

## Cluster F. Getting Light Onto the Chip: Mode Overlap, Fiber Attach, Cryo Packaging

- Experts: fiber-to-PIC packaging engineers and lab experimentalists: ~2,000-10,000 (I); cryo photonic packaging subset ~100-300.
- Why niche: parity/phase overlap, epoxy CTE, facet-angle geometry and wirebond parasitics are learned by failure.
- Visual: interactive: two modes with identical |E|^2 but opposite parity, overlap readout 0; MFD definition toggle. Blender: fiber array glued at an angle, epoxy blob on all four sides cracking a chip at 4 K; 25 um wire vs a 15 g tweezer weight; long wirebond acting as an antenna.

| # | Pitfall | Why experts miss it | Gag / visual | Source | Conf |
|---|---|---|---|---|---|
| F1 | Two modes with nearly identical |E|^2 images but opposite parity have near-zero overlap; similar effective area does not imply identical mode shape; vertical mismatch can dominate. | People compare intensity images and spot sizes. | Twin portraits, one wearing a minus sign | G/20260913_mode_overlap/DevLog/000-initial-discussion.md section 9 (L550-590) | L |
| F2 | "1 - sum of guided projections" is not radiation loss unless the modal basis is complete and reflections are handled. | | | same section 10 | L |
| F3 | Fiber MFD in the model is the intensity 1/e^2 diameter; mixing field vs intensity radius definitions changes overlap. | Datasheet conventions vary. (I: the "varies" part is my inference) | | G/20260913_mode_overlap/README.md (units paragraph) | L / I |
| F4 | Fiber coming in at an angle because the chip facet is angled for return loss but the waveguide was not rotated to compensate. | Facet angle and refraction angle bookkeeping. | Fiber visibly bent toward chip | P/2025-12-25-how-to-hold-fibers.md L130 | O |
| F5 | At cryo, apply epoxy as a small drop on one side only; multi-side epoxy between chip and copper applies extreme CTE stress. | Room-temperature packaging habit. | Chip cracking on cooldown | same L276 (quote from Witt2025, doi:10.1063/5.0283650) | S (quoted) |
| F6 | Optical (index-matching) epoxy alone is not sufficient long-term; add a structural epoxy (TRA-BOND 2151 cited). | | | same L211 (quote from Klawson2025, Berkeley EECS-2025-19) | S (quoted) |
| F7 | Brass screw for wire thermalization because its CTE is larger than OFHC Cu, so it tightens on cooldown. | Material chosen for wrong reason (conductivity) otherwise. | | P/2024-12-08-SC-coil.md L204 | S |
| F8 | Above ~150 mA through the coil, normal-metal (PB) leads heat the 4 K stage; heating comes from leads, not the superconducting coil. | | | same L93, L236 | S |
| F9 | Wirebonds: ~1 nH/mm; ~6 mm length starts resonating near 6 GHz; long (~5 mm) bonds have low-Q resonances at 5-10 GHz; use the minimum number when part of a resonance, otherwise >=2-3 for signal. | | Wirebond as accidental resonator | P/2024-07-19-how-to-wirebond.md L70, L185-189 | S |
| F10 | Holding the fixture by hand absorbs the ultrasonic bond energy ("your hand is a very good vibration attenuator"); clamp instead. | | | same L34 | S |
| F11 | Always check bond force; previous user may have changed the spring. Old fab tools lack "reset to default". | | | same L56 | S |
| F12 | 25 um Al wire breaks at ~0.17 N (estimate), comparable to a 15 g tweezer's weight. | | | same L150 | S (estimate) |
| F13 | Polarization: there is no longitudinal polarization in the far field; if you see one it is from a different source; "it could bring you so much trouble". Also "Be wary of the 2 pi ... could screw you over". | | | P/2025-01-10-fiber2chip-history-concept.md L68, L77, L183 | S/O |
| F14 | TWPA/broadband amp: residual reflections cause gain ripple that worsens with pump power/gain; strong pump can saturate the room-T LNA. | | | P/2025-11-15-TWPA.md L256, L285 | S |
| F15 | Short-reach VCSEL links use multimode fiber for alignment tolerance; modal dispersion then limits baud rate. | | | P/2026-08-20-... L457 | S |
| F16 | For a broad uLED source at 430 nm in GRIN MMF, total bandwidth is far below modal bandwidth even for the optimized alpha profile (summary table: BW modal 115 vs BW total 2.28 for strict alpha, 50 um/NA 0.20); profile dispersion at 430 nm (y = 0.365 vs 0.051 at 850 nm) moves inner-group alpha_opt to ~2.36 vs ~2.03. | GRIN optimization intuition from 850 nm VCSEL world. | | G/20261106_uled_grin_profile/outputs/summary.md (first table); DevLog-000 progress 2026-11-07 00:45 | L (model output; BW units not stated in summary header) |
| F17 | Classical Olshansky-Keck alpha_opt describes only inner mode groups; with all guided modes (OFL) the optimum moves to 2.0-2.2 due to near-cutoff groups. | | | G/20261106_uled_grin_profile/DevLog/DevLog-000 progress 2026-11-07 00:05 | L |

---

## Cluster G. PIC and Transceiver Forensics / Accounting Crimes

- Experts: ~1,000-5,000 (I) (datacom optics engineers, analysts). Less niche than A-C; better as a lighter "bonus" course.
- Visual: Blender/3DGS: Intel 100G CWDM4 module with six InP coupons (he has Gaussian splats per teardown post L49); interactive: "expand the bandwidth number" calculator R_total = R_lane x N_lambda x N_fiber x N_direction, and energy-per-bit boundary checklist toggles.

| # | Pitfall | Why experts miss it | Gag / visual | Source | Conf |
|---|---|---|---|---|---|
| G1 | Six InP coupons on Intel 100G CWDM4 are four lasers + two monitor PDs, not redundant lasers. | Looks like 4+2 spares. | "Spot the spare" game | P/2026-08-20-... L130; P/2026-06-13-100G-200G-400G-800G.md L99 | S |
| G2 | Energy-per-bit claims: check whether laser wall-plug, driver (not just modulator capacitance), ring heaters/control, PD/TIA/ADC/DSP/CDR/FEC, one-way vs bidirectional, temperature/loss budget/BER are included. | Different boundary per vendor. | Shrinking box around a pJ/bit number | P/2026-08-20-... L287-304 | S |
| G3 | Symbol rate (Bd), bit rate (b/s) and analog modulator bandwidth (Hz) are related but different numbers (100G PAM4 ~ 50 GBd class). | | | same L170 | S |
| G4 | Infinera 1.6T PIC: CMP dishing fringes where dummy fill was removed around MMIs/detectors; extra MMI dummy ports (guess: local symmetry); electrical lines narrowed where crossing waveguides; curved tapered terminations dump residual light after PDs; no on-chip polarization rotator found (guess: rotated off chip, all on-chip TE). | Die-shot reading skill. | Interactive component map exists (/infinera-pic-map/) | P/2026-05-15-infinera-DP-64QAM.md L58, L65, L75, L82 | O (his guesses) |
| G5 | Facet looked cleaved, later judged laser stealth diced ("too clean to be cleaved"). | | | same L38, L86 | O |
| G6 | TIA output differential pairs did not appear length-matched. | | | P/2026-06-13-... L81 | O |
| G7 | Foundry feedback latency: "you may wait half a year to learn about a mistake". MPW chips come back as dies, so no wafer-scale backend. | | | P/2026-08-20-... L378-382 | S/O |
| G8 | Current CPO optical engines sit on organic substrate, not silicon interposer; bandwidth not high enough to compete with HBM for interposer area. | | | same L231, L340 | O |

---

## Cluster H (mini). Inverse-problem / Measurement Interpretation

- Experts: FBG sensing and fiber-component metrology: ~200-1,000 (I).
- Visual: interactive: heat profile f(z) and its mirror f(L-z) producing identical reflection power spectra.

| # | Pitfall | Why experts miss it | Gag / visual | Source | Conf |
|---|---|---|---|---|---|
| H1 | Power-only FBG reflection cannot distinguish a heating profile f(z) from its mirror f(L-z); optimizer converged to the mirrored solution. | Phase information discarded by OSA. | Mirror twins | G/20260802_nonuniform_FBG/MODEL_NOTES.md "Thermal parameterization" | L |
| H2 | Temperature conversion must use the spectral edge, not the labeled last local maximum of a chirped (nonuniform) FBG. | Peak-picking habit. | | same "What is measured" | L |
| H3 | A one-sided exponential interpretation of a quoted thermal FWHM leaves the far grating end warm and translates the cold edge, contradicting data; a centered Gaussian produces unrealistically regular short-wavelength fringes. | FWHM is ambiguous for asymmetric profiles. | | same | L |
| H4 | Source paper text (~460 C at 367 mW) disagrees with its plotted point (~479 C). | | | same (paper file oe-34-15-28164_parsed.md in folder) | L |

---

## Notes for course building

- Strongest courses by depth + personal evidence: A (crystal bookkeeping), B (simulation self-deception), C (phononic transduction). Each has 10+ sourced, dated, first-person failures.
- D (e-beam) is rich and personal but borders on general nanofab; fine as "Nanophotonic Fab Compliance".
- G is the least niche; useful as a crowd-pleaser module.
- Real-safety topics (ESD for TWPA, P/2025-11-15-TWPA.md L269; acids) appear in the corpus and are excluded per brief.
- PlayGround folders scanned by name with likely further material not mined here: 20250216_MZI_nerd_snipe, 20240912_tapered_fiber, 20250328_PIC_torch, 20260203_transfer_matrix_film_thickness, 20260612_fiber_bundle, 20250624_inv_design_filter, 20250210_cherenkov_fdtd, 20261001_eo_modulator_atlas (its DevLog-005 is data-viz correctness, not photonics physics).
- Items marked L come from agent-written project logs; verify with him before quoting as his own words.
- Not verified: COMSOL versions for B3/B4 behaviors (logs mention 5.0, 5.4 in 2018-2019); any claim may differ in current COMSOL.
