# Micrograph references for the code-drawn chip animations (2026-11-22)

Reference only: nothing here is shipped, no image from these papers is kept in the project (PDF pages were rendered to a temporary scratch folder to look at, then deleted). All entries read on 2026-11-22. "est." = read off the rendered figure against its own scale bar, not a stated value. "N/A" = not stated in the text that was read.

| Key | Paper | Figure | URL | License | What it shows |
|---|---|---|---|---|---|
| wang18 | C. Wang et al., "Integrated lithium niobate electro-optic modulators operating at CMOS-compatible voltages", Nature 562, 101 (2018) (author-hosted manuscript PDF) | Fig. 1c | https://doi.org/10.1038/s41586-018-0551-y (manuscript read: https://www.ee.cityu.edu.hk/~cwang/document/LNMOD2018_2_100Gmod_.pdf) | Nature, not open access; no CC license | Optical micrograph, 3 x-cut TW-MZMs side by side (20, 10, 5 mm), GSG strip lines, Y-junctions, "RF in" / "RF terminate" ends |
| xu20 | M. Xu et al., "High-performance coherent optical modulators based on thin-film lithium niobate platform", Nat. Commun. 11, 3911 (2020) | Fig. 1b (micrograph), 1a (schematic), 1e (SEM) | https://doi.org/10.1038/s41467-020-17806-0 | CC BY 4.0 | Full IQ modulator: 1x2 MMIs, two TW-MZMs sharing a GSGSG electrode, flared probe ends, Ti heaters with wire-bonded DC pads, grating couplers |
| renaud23 | D. Renaud et al., "Sub-1 Volt and high-bandwidth visible to near-infrared electro-optic modulators", Nat. Commun. 14, 1496 (2023) | Fig. 1d | https://doi.org/10.1038/s41467-023-36870-w | CC BY 4.0 | Micrograph of a 5 mm x-cut TFLN MZM: unbalanced arms, Y-splitters, "Contact Region" (probe flare with gap taper), "Interaction Region" |
| valdez22 | F. Valdez, V. Mere, S. Mookherjea, "100 GHz bandwidth, 1 volt near-infrared electro-optic Mach-Zehnder modulator", arXiv 2211.13348 (2022) | Fig. 1a (schematic), 1f (electrode micrograph), 1g (stitched chip micrograph) | https://arxiv.org/abs/2211.13348 | CC BY-NC-ND 4.0 (arXiv) | Bonded TFLN on SiN MZMs: edge couplers, 2x2 MMIs, TFLN bond edge, T-segmented CPW with flare-outs for GSG probes |
| nelan22 | S. Nelan et al., "Ultra-high extinction dual-output thin-film lithium niobate intensity modulator", arXiv 2207.02608 (2022) | Fig. 1a (render), 1b (micrograph) | https://arxiv.org/abs/2207.02608 | arXiv non-exclusive license (no CC) | Folded (U-turn) MZM, 7.8 x 1.5 mm, waveguide crossing at the fold, arms entering under the flared ground pad, heaters at the U-turn |
| xu22 | M. Xu et al., "Dual-polarization thin-film lithium niobate in-phase quadrature modulators for terabit-per-second transmission", Optica 9, 61 (2022) | Fig. 1a (schematic), 1b (CL-TWE micrograph), 1d (PRC micrograph) | https://doi.org/10.1364/OPTICA.449691 | Optica Open Access Publishing Agreement (not CC BY) | Four sub-MZMs with capacitively loaded TWE, flip-chip 50 ohm terminators, PRC, edge couplers; micrograph of T-segments |
| wangx22 | X. Wang et al., "Thin-film lithium niobate based dual-polarization IQ modulator for single-carrier 1.6 Tb/s transmission", arXiv 2201.09730 (2022) | Fig. 1a (schematic), 1b, 1c (micrographs) | https://arxiv.org/abs/2201.09730 | CC BY 4.0 (arXiv) | GSGSG DP-IQ on Si; micrographs of a Ti thermo-optic phase shifter and the polarization rotator-combiner |
| kharel21 | P. Kharel et al., "Breaking voltage-bandwidth limits in integrated lithium niobate modulators using micro-structured electrodes", Optica 8, 357 (2021) | Fig. 1a (artistic top view), 1b (SEM), 1e | https://arxiv.org/abs/2011.13422 | CC BY-NC-ND 4.0 (arXiv) | Segmented (T-rail) GSG electrode; no optical micrograph in the arXiv version |
| zhang19 | M. Zhang et al., "Broadband electro-optic frequency comb generation in a lithium niobate microring resonator", Nature 568, 373 (2019) | Fig. 2a | https://arxiv.org/abs/1809.08636 | arXiv non-exclusive license (no CC) | Micrograph: racetrack resonator with point-coupled bus, narrow gold strips along both straights, segmented G/S/G pad column at the racetrack center |
| wang18ppln | C. Wang et al., "Ultrahigh-efficiency wavelength conversion in nanophotonic periodically poled lithium niobate waveguides", Optica 5, 1438 (2018) | Fig. 2 (schematic + false-color SEM) | https://arxiv.org/abs/1810.09235 | arXiv non-exclusive license (no CC) | Poling finger electrodes on x-cut LNOI, poled region, ridge waveguides inside it |
| he19ec | L. He et al., "Low-loss fiber-to-chip interface for lithium niobate photonic integrated circuits", Opt. Lett. 44, 2314 (2019) | Fig. 1a (schematic), Fig. 3 (SEM) | https://arxiv.org/abs/1902.08969 | arXiv non-exclusive license (no CC) | Bilayer inverse-taper edge coupler (rib taper on top of slab taper), etched facets |

## Observations used for the drawings

### Layout topology (x-cut TW-MZM)

- Probe pad position, all reachable micrographs of straight x-cut TW-MZMs (wang18, xu20, renaud23, valdez22; xu22 and kharel21 schematics): the RF probe lands on the electrode ends, in line with the CPW axis, at both ends (input probe and terminating probe). No figure in this sweep shows a straight TFLN MZM whose CPW bends 90 degrees toward the long chip edge.
- Arm exit, inline pads: the optical arms stay inside the two S-G gaps through the whole electrode including the probe region and leave through the open end of the gap. They never cross metal. The splitter sits just outside the electrode end; the S-bends from splitter spacing to gap spacing are partly inside the flared gap.
  - wang18: no separate pad; the wide G, S, G rectangles end bluntly and the probes land on those ends. The Y-junction opens to the gap spacing about 0.1-0.2 mm (est.) before the electrode starts.
  - xu20: the GSGSG electrode ends in a short flare (about 0.2 mm, est.) where each conductor narrows into a finger and the gaps widen; the four arms pass between the fingers and S-bend to the MMIs outside.
  - renaud23: the "Contact Region" (about 1.3 mm long, est.) is a wide gold block with the S-G gaps tapering from wide (at the probe) to narrow (interaction); the arms S-bend inside those tapering gaps.
  - valdez22: "flare-outs at the ends to facilitate landing of standard GSG probe tips"; same inline topology.
- Arm under metal (folded device, nelan22): the electrode has a wide rounded pad at the chip end; the arms reach the gaps by S-bends that pass under the flared ground metal, and the arms also cross under the electrodes at the U-turn. A locally thick SiO2 buffer (2 um, laser-written mask) protects the mode wherever metal is overhead; the buffer is etched to 450 nm elsewhere. The thick-buffer patch is visible in the micrograph as a rectangle with a different tint around the pad region. An optical waveguide crossing sits at the fold apex.
- Air bridges: Xu et al., APL Photonics 8, 066104 (2023) put air bridges in the U-turns of a folded DP-IQ CPW (abstract only; the figure could not be reached).
- Termination: there is no on-chip thin-film resistor in any figure reviewed. wang18, xu20, renaud23 and valdez22 used a second probe with an external 50 ohm load. xu22 used flip-chip bonded 50 ohm terminators at the far end of each CL-TWE (drawn as grey blocks on the electrode ends). wangx22 used an external 38 ohm resistor on the packaged chip.
- Splitters: wang18 and renaud23 use Y-junctions. xu20, xu22 and wangx22 use 1x2 MMIs (in the xu20 micrograph they show as short, slightly wider purplish segments). valdez22 and nelan22 use 2x2 MMIs.
- Bias: xu20 has a Ti heater (160 um, 760 ohm) on each arm outside the electrodes, with square DC pads that are wire-bonded (ball bonds visible). wangx22 has Ti heater strips crossing the waveguide, with rectangular pads. nelan22 has heaters at the U-turn.
- Layout of several devices on one chip (wang18): the MZMs are stacked with roughly 50 um (est.) between electrode blocks. Output waveguides of the shorter devices wind around the electrode ends of their neighbours with large-radius bends. There is a small on-chip text label "Length 0.5 cm Width 700 nm".
- Ring (zhang19): racetrack with a vertical bus that comes down past the left U-bend, point coupled. The gold is two narrow strips per straight, one on each side of the waveguide, with the waveguide-to-gold gap at 3.3 um. All inner strips join a central signal bar and the outer strips join ground bars. The bar is cut into G / S / G segments by the two waveguides passing through it, so the waveguide runs through the gap between adjacent pad segments and never under metal. On-chip etched text: "wgcp 0.7", "wd 2.4, rad 80.0".
- PPLN (wang18ppln): two opposing comb electrodes (bus bars with fingers) on the x-cut film, with the fingers along z. The poled region is 75 um wide and 4 mm long and holds 3 ridge waveguides. The electrodes (15 nm Cr / 150 nm Au) are removed after poling, and the waveguides are written inside the poled region afterwards. Period 4.1 um, duty cycle about 39 %.
- Edge coupler (he19ec): the rib tapers to a tip over 120 um and the slab below tapers from 6 um to a 340 nm tip over 450 um, ending at an etched (not diced) facet. The waveguides are 600 nm LN on 2 um SiO2 with 1 um PECVD SiO2 cladding. wang18: chip edges diced and polished. wangx22: SSCs at polished facets, with a fiber array glued on.
- Kharel-type segmented electrode (kharel21, also xu22 and valdez22): T-shaped rails off the S and G edges toward the gap, with a short stem and a wide head, periodic along the line. The heads set the narrow gap next to the waveguide.

### Dimensions (as stated in the papers)

| Key | Signal width | S-G gap | Ground width | Electrode length | Au thickness | Other |
|---|---|---|---|---|---|---|
| xu20 | 19.5 um | 7 um (waveguide-electrode 1.5 um) | 80 um | 7.5 and 13 mm | 900 nm | rib top 4 um, 300 nm etch on 600 nm film; GSGSG probes |
| wangx22 | 40 um | 5 um | 50 um | 12 mm | 1200 nm | rib 1.5 um, 260 nm etch, 500 nm LN on 4.7 um BOX; 2 um SiO2 over electrodes |
| nelan22 | 24.5 um | 4 um | 75 um | 11 mm interaction (2 x 5.5 mm) | 1 um electroplated | 2x2 MMI 15 um wide, 618-640 um long; chip 7.8 x 1.5 mm |
| xu22 | 80 um | 3 um (between T heads) | N/A | 23.5 mm | 0.9 um on 0.7 um SiO2 | T period 50 um, gap between adjacent T heads along the line 5 um, t = 1 um, s = 1 um (Fig. 1b labels); rib top 2.5 um; quartz substrate |
| kharel21 | N/A | 5 um | N/A | 20 mm | 800 nm | (h, s, t, r, c) = (6, 2, 6, 45, 5) um; 600 nm LN on quartz; 1 um SiO2 cladding |
| valdez22 | N/A | 4 um | N/A | 8 mm | N/A | T segments h = 2, L = 20, t = 15 um; MMI 8 x 96 um |
| renaud23 | N/A | 3 and 5 um (sweep) | about 1 mm (est.) | 5 mm (micrograph), 10 mm (data) | N/A | contact region about 1.3 mm long (est.); probe GGB 40A-GSG-100-F (100 um pitch); 300 nm LN on 2 um SiO2 |
| wang18 | varied across devices (stated); top device about 90-100 um, others about 45 um (est.) | N/A in the text read | about 170 um (est.) | 20, 10, 5 mm | 500 nm | 600 nm LN; 800 nm PECVD SiO2 cladding |
| zhang19 (ring) | narrow strips, about 10-15 um (est.) | waveguide-gold 3.3 um | N/A | racetrack | N/A | waveguide 1.4 um wide; 600 nm LN on 2 um SiO2 |

- Arm separation equals S width + gap (center to center: S + gap), so 26.5 um (xu20), 29 um (nelan22), 45 um (wangx22) from the stated values. This is derived, not stated.
- Probe pitch: the only stated value is 100 um (renaud23 probe model). Pad sizes are not stated in any paper read.

### Colors in bright field (from the micrographs; they depend on stack, cladding, illumination and white balance)

- Gold electrodes: saturated yellow in all color micrographs (wang18 orange-yellow; xu20, renaud23 and zhang19 bright lemon yellow). Under strong illumination it looks pale yellow-white (nelan22) or pale beige-grey (xu22 CL-TWE micrograph).
- Film/background:
  - wang18 (600 nm LN, SiO2/Si, 800 nm PECVD cladding): medium blue.
  - xu20 (600 nm LN, NanoLN): pale teal / cyan-green.
  - renaud23 (300 nm LN on 2 um SiO2, cladded): olive green.
  - zhang19 (600 nm LN on 2 um SiO2): dark slate blue-grey.
  - wangx22 (500 nm LN on 4.7 um BOX, 2 um SiO2 on top): blue in one inset, grey-tan in the other at a different zoom.
  - nelan22 (300 nm LN on 4.7 um SiO2 with SiN and buffer): salmon / pink-tan.
  - valdez22 (bonded TFLN on SiN): neutral grey, with the TFLN region lighter.
  - xu22 (360 nm LN on quartz): pale green-teal.
  - The claim "600 nm X-cut LN on 4.7 um oxide appears purple/greenish" was not verified by any figure in this set.
- Waveguides: thin dark lines (dark grey or near black) on the film. Inside electrode gaps they often read as a bright thin line between two dark edges (wang18, xu20). Rib edges show as a double line at high magnification (wangx22 PRC). MMIs read as short, slightly wider bars with a purple tint (xu20).
- Annotation style: white scale bar with white sans-serif "500 um" / "1 mm" / "100 um" text in the lower corner (wang18, xu20, renaud23, zhang19). Red scale bars on the wangx22 insets; a black bar above the image in valdez22. Panel labels are white sans-serif ("Ground (G)", "Signal (S)", "Contact Region", "Y-splitter"). Crystal axes (z up, y right) are drawn as white arrows in wang18.

## Not reachable

- Xu et al., "Attojoule/bit folded thin film lithium niobate coherent modulators using air-bridge structures", APL Photonics 8, 066104 (2023), https://doi.org/10.1063/5.0146987: the publisher page redirected to a CDN minimal page, which was not followed; no arXiv version found. Air-bridge figure not seen.
- Zhang, Wang, Kharel, Zhu, Loncar, "Integrated lithium niobate electro-optic modulators: when performance meets scalability", Optica 8, 652 (2021): not fetched (fetch budget).
- He et al., Nat. Photon. 13, 359 (2019), hybrid Si/LN (arXiv 1807.10362): main-text Fig. 1 is a schematic plus SEMs; there is no optical micrograph of the MZM layout in the main text. Stated: signal 19.5 um, ground 30 um, waveguide-electrode gap 2.75 um, 600 nm Au, second probe with 50 ohm termination. arXiv non-exclusive license (no CC).
- Mercante et al., Opt. Express 26, 14810 (2018): this is a phase modulator, not an MZM; not fetched.
- A published micrograph of a straight x-cut TFLN TW-MZM with the CPW bent toward the long chip edge (off-axis GSG pads) was not found in this sweep.

## How the code-drawn chips use these (course/anims/fe_draw.js, 2026-11-22)

- Die layout follows the course renders (course/tools/blender/tfln_die.py): GSG 300/40/300 um with 8 um gaps, CPW bending north at the RF input end and south at the far end (0.45 mm centreline radius), GSG pads 80 um signal / 40 um gaps / 150 um pitch / 150 um long, pads about 1 mm inside the long edges. Published straight TW-MZMs mostly use inline end pads (wang18, xu20, renaud23, valdez22); the off-axis pads are kept for consistency with the renders and the probe/fiber layout.
- Arm routing at the bends: the arms run straight out of the gaps and pass under the bent CPW inside a thick-oxide window (drawn as a lighter rectangle visible in the gaps and beside the metal), after nelan22, where arms cross under a ground pad over a locally thick SiO2 buffer. The arm in the inner gap crosses the signal and the outer ground; the other arm crosses the outer ground only. Splitters (MMIs) sit outside the bends.
- On-chip termination (incident die only, 8 x 4 mm, 5 mm electrode): none of the papers above shows one; drawn as a short pad flare, a thin-film resistor from the signal end to a ground bar joining both grounds.
- Colours: film, slab and gold from the render stack (600 nm x-cut LN on 4.7 um SiO2 on Si, transfer-matrix colour in tfln_die.py), consistent with the range of film colours seen in the micrographs above; waveguides as 7.5 um slab-colour bands with a 1.5 um rib when the zoom resolves it; gaps never drawn narrower than 2 px.
- Probe and fibre silhouettes: research/ref_probes/README.md (GSG contacts at 150 um pitch, 1.2 mm coax) and 125 um lensed fibre held about 3 mm back in a V-groove block.
