# Mine: advanced-packaging-atlas, photonic packaging pitfalls

Date: 2026-11-22. Corpus: `~/Documents/GitHub/advanced-packaging-atlas` (read-only; nothing in that repo was modified).

## Method

- **Selection.** From `data/papers.csv` (299 papers), I kept the 35 with photonic or CPO content (CPO = co-packaged optics): 34 with photonic or CPO bucket/title keywords, plus `ectc51909.2023.00230`, found through `data/optical_interfaces.csv`. Non-photonic hits were dropped: ABF (Ajinomoto build-up film) laser drilling, LABC (laser-assisted bonding with compression), FLOLA (flux-less oxide-removal laser assembly), LDI (laser direct imaging), InP power amplifier, balun and TGV (through-glass via) etch papers.
- **Reading.** Every kept paper's `references/<paper_id>/text.md` was read in full by six parallel reader subagents, about 6 papers each.
- **Spot-check.** I checked about 30 of the key quotes against `text.md` with whitespace-normalized search, and all matched. One example: AIM's "1 5 mm" is OCR for 15 mm.
- **Locators.** "p.N" is the `<!-- page N -->` marker in `text.md`. This is the PDF page index, not the printed journal page.
- **OCR texts.** Four papers have `TEXT_LAYER_WARNING.md`, meaning their text is OCR: `ectc51909.2023.00048`, `ectc51909.2023.00324`, `eptc59621.2023.10457918` and `iedm50572.2025.11353491`. The readers checked their key numbers against page renders or `data/evidence/*.yaml`.
- **Labels.**
  - Columns "why experts miss it" and "gag/visual" are always my inference.
  - Conf `S` = stated in source.
  - Conf `S(sim)` = stated, but simulation only.
  - Conf `I` = my inference from stated facts.
  - Conf `S+I` = the mechanism is stated and the framing is mine.
- **Numbers.** Every number is quoted from the source. None is estimated by me.
- **Corpus bias.** The atlas is packaging-venue literature from 2022-2026: ECTC, EPTC, ESTC, TCPMT, IEDM, VLSI and ISSCC (Electronic Components and Technology Conference; Electronics Packaging Technology Conference; Electronics System-Integration Technology Conference; IEEE Transactions on Components, Packaging and Manufacturing Technology; International Electron Devices Meeting; VLSI Symposium; International Solid-State Circuits Conference). Mode hops, laser-physics pitfalls and pure PIC design are mostly outside it.
- **What the corpus lacks.** The readers found no measured data on:
  - optical return loss (ORL, back-reflection);
  - TSV (through-silicon via) stress keep-out effects on rings (only cited: Yang 2013);
  - measured EIC-to-ring thermal crosstalk;
  - AuSn laser-joint voids;
  - epoxy yellowing or aging spectra;
  - birefringence or stress-induced PDL (polarization-dependent loss) from attach.

  Do not build chapters on those from this source. (EIC = electronic IC; PIC = photonic IC.)

## Source key

All sources are full texts at `advanced-packaging-atlas/references/<paper_id>/text.md`. The DOI is `10.1109/<paper_id>`.

| key | paper_id | title (short) | year | org |
|---|---|---|---|---|
| GF22 | ectc51906.2022.00072 | Optical performance and reliability assessment from self-aligned single mode fiber attach for O-band silicon photonics platform | 2022 | GlobalFoundries, IBM |
| AIST23 | ectc51909.2023.00324 | Self-aligned optical connector assembly on polymer waveguide integrated package substrate for co-packaged optics | 2023 | AIST |
| LM26 | ectc51846.2026.00176 | Advancing Interconnect Performance and Reliability with Innovations in 3D Photonic Integration Packaging and Fiber Coupling | 2026 | Lightmatter, PHIX, Amkor |
| IZM26 | ectc51846.2026.00302 | Monolithic Passive Low-Loss Fiber Coupling to Photonic Glass Core Substrates using Ion-Exchanged Waveguides and SLE-Fabricated V-Grooves | 2026 | Fraunhofer IZM, TU Berlin |
| INT25a | ectc51687.2025.00061 | Demonstration of Co-Packaged Optics Assembly for Fiber-Based Optical Interconnect | 2025 | Intel |
| INT25b | ectc51687.2025.00012 | Heterogeneous Integration of Fiber-Based Co-Packaged Optics with EMIB technology: Assembly, Performance, and Reliability | 2025 | Intel |
| COR24 | ectc51529.2024.00023 | Ultra Low-Loss Ion-Exchange Waveguides in Optimized Alkali Glass for Co-Packaged Optics | 2024 | Corning |
| COR26 | ectc51846.2026.00106 | Low-Loss Optical Interconnect Designs in Optimized Glass for Co-Packaged Optics | 2026 | Corning |
| DNP26 | ectc51846.2026.00110 | High-Density Polymer Waveguide Integration on Glass Substrate for CPO | 2026 | Dai Nippon Printing |
| IBM25i | iedm50572.2025.11353491 | Co-packaged Optics Module with Single-Mode Polymer Waveguide | 2025 | IBM (OCR text) |
| AIST24 | ectc51529.2024.00229 | Scalable Fabrication of 3D Optical Re-distribution for Co-Packaged Optics using an Developed Nanoimprint Stepper | 2024 | AIST |
| CORK25 | ectc51687.2025.00063 | Advanced Glass Substrate Fabrication and Metallization Process Technology for Co-Packaged Optics | 2025 | Corning Korea |
| LSP23 | eptc59621.2023.10457918 | Overcoming Assembly Process Challenges in Glass Interposer: Development of Jigs and Fixtures | 2023 | LightSpeed Photonics (OCR text) |
| SEI25 | ectc51687.2025.00014 | Flip-Chip Photonic-Electronic Integration Platform for Co-Packaged Optics Using a Glass Substrate with Vertically-Coupled Beam Expanding Lens | 2025 | Sumitomo Electric, FICT, Science Tokyo |
| IBM25 | ectc51687.2025.00052 | Co-Packaged Optics (CPO) Technology Full Module Test Vehicle Demonstrations | 2025 | IBM |
| TSMC24 | ectc51529.2024.00144 | High Bandwidth and Energy Efficient Electrical-Optical System Integration Using COUPE Technology | 2024 | TSMC (simulation only) |
| TSMC25a | ectc51687.2025.00015 | Optical and Electrical Characterization of a Compact Universal Photonic Engine | 2025 | TSMC |
| IME26 | ectc51846.2026.00016 | High-Density, Energy-Efficient CPO Platform with PIC-in-Mold Interposer Architecture for AI/ML Data Centers | 2026 | A*STAR IME |
| TSMC25b | ectc51687.2025.00284 | Broadband Optical Engine System Integration by Wafer Level Process in HPC/AI Era | 2025 | TSMC |
| INT26v | vlsitechnologyandcir65830.2026.11577543 | An 800 Gbps/Fiber Silicon Photonic Microring-Based DWDM Transceiver in an Open-Cavity Package | 2026 | Intel |
| AIM23 | ectc51909.2023.00048 | AIM Photonics Demonstration of a 300 mm Si Photonics Interposer | 2023 | SUNY Poly / AIM (OCR text) |
| LETI23 | ectc51909.2023.00009 | Process Integration of Photonic Interposer for Chiplet-based 3D Systems | 2023 | CEA-Leti |
| MRVL26 | ectc51846.2026.00346 | A Low-Cost 2.5D Photonic Silicon Interposer Assembly Process for HPC Applications | 2026 | Marvell |
| IME24 | eptc62800.2024.10909836 | Fusion Bonding with Ultra-Thin Dielectric Layers for Photonic Interposer Applications | 2024 | A*STAR IME |
| NV26 | isscc49663.2026.11409081 | A 32Gb/s/lambda 256Gb/s/Fiber Half-Rate Bandpass-Filtered Clock-Forwarding DWDM Optical Link in a 3D-Stacked 7nm EIC/65nm PIC Technology | 2026 | Nvidia |
| IZM23 | ectc51909.2023.00230 | 3D silicon interposer for terabit/s transceivers based on high-speed TSVs | 2023 | Fraunhofer IZM, CEA-Leti |
| TYN24 | tcpmt.2025.3533388 | Thermal and Electrical Study of Glass Interposers in Co-Packaged Electronic-Photonic Systems | 2025 | Tyndall, Fraunhofer IZM, TU/e |
| SEI24 | estc60143.2024.10712002 | Over 100-GHz Bridge Chip Interconnection Between Photonic and Electrical ICs with a Heat-Insulating Stress-Relief Membrane Structure | 2024 | Sumitomo Electric, Tokyo Tech |
| IME25 | eptc67330.2025.11392620 | Characterization of Fine Line Width/Spacing RF Interconnects for Co-Packaged Optics with High I/O Density | 2025 | A*STAR IME |
| MEL26 | ectc51846.2026.00019 | High-Density Integration of III-V Devices and EICs using Vertical-Coupled Photonic Packages with Glass-IP and Redistribution Layers | 2026 | Mitsubishi Electric |
| TYN23 | eptc59621.2023.10457729 | Packaging strategies for 3D integration of Photonic and Electronic chips on a Glass substrate | 2023 | Tyndall, MIT |
| TSMC23 | ectc51909.2023.00108 | Integrated Optical Interconnect Systems (iOIS) for Silicon Photonics Applications in HPC | 2023 | TSMC |
| IMEC24 | ectc51529.2024.00227 | Collective die-to-wafer assembly process for optically interconnected System-on-wafer | 2024 | imec |
| CEL25 | ectc51687.2025.00062 | Optical Multi-Chip Interconnect Bridge (OMIB) Interposer Assembly Process to Enable High-Density Photonic Interconnects | 2025 | Celestial AI |

### How expert populations were judged

All population figures are my inference, not data.

- **Basis.** The 35 photonic papers come from 48 distinct affiliation strings across about 25 organizations, counted from `data/papers.csv` `orgs`.
- **Per-cluster count.** For each cluster I counted the organizations that publish on that topic in this corpus. I then assumed roughly 5-30 engineers per organization work at that depth, plus a similar number who never publish (OSATs (outsourced assembly and test houses), connector vendors, hyperscalers).
- **Precision.** Treat these as order-of-magnitude figures only.

---

## Cluster 1. "The Adhesive Is an Optical Component": glue, underfill and fillers in or near the optical path

- **Experts (inference):** about 100-400 people worldwide. Basis: 12+ organizations in the corpus touch this topic (IBM, Intel, Lightmatter/PHIX, A*STAR, AIST, Marvell, Celestial, Tyndall, Sumitomo, Mitsubishi, GF, Corning), plus adhesive and FAU (fiber array unit) vendors.
- **Why niche:** the material is chosen by packaging, but its refractive index, cure shrinkage and flow decide optical loss. Few people own both sides.
- **Visual potential: highest.**
  - Code interactive: an adhesive-index slider on an edge-coupler mode solver. The mode leaks into the cladding when n > 1.444 (IME26). This is a 1D/2D slab solver, quick to build.
  - Blender: an underfill fillet creeping onto a waveguide facet; a UV-cure shrink pulling a connector off-axis; a TIM (thermal interface material) blob flowing between EIC and PIC.

| # | Pitfall | Why experts miss it (inf.) | Gag / visual (inf.) | Source + locator | Conf |
|---|---|---|---|---|---|
| 1.1 | An FAU adhesive with index above SiO2 (1.444) makes the edge-coupler mode leak. Loss went from about 3.25 dB/facet in air to more than 20 dB/facet with an n = 1.5 adhesive. | "Index-matching epoxy" is assumed to help. Generic optical epoxies sit near 1.5. | Glue bottle "n = 1.50" with a skull; light pouring through the cladding. Mode-solver slider. | IME26 p.5 Sec. VI; p.6 Fig. 10; also `data/optical_interfaces.csv` row ectc51846.2026.00016-e-o2 | S |
| 1.2 | The good window is one-sided and narrow: n = 1.38-1.442 beats air, and the minimum (<1.5 dB/facet) is at n = 1.442, just 0.002 below silica. Lower index adds mode-field mismatch and reflection loss. | The optimum sits against the leakage cliff, so lot-to-lot index or cure shift can cross it. | Tightrope walker between 1.442 and 1.444 | IME26 p.5 Sec. VI; Fig. 10 | S (cliff-risk framing I) |
| 1.3 | Lightmatter/PHIX used two adhesives: a silica-filled low-shrinkage epoxy for mechanics and a low-index silicone (RI ~1.4) on all FAU facets, which "aided in beam expansion". The silicone needs 24 h at 25 C after batch UV initiation to finish curing. | The index-match material is part of the optical design. Cure time is hidden WIP (work in progress). | Calendar page for the glue | LM26 p.3 Sec. III | S |
| 1.4 | GF/IBM: the index-matching optical adhesive fills the undercut beneath the suspended SSC (spot-size converter). Its properties are matched to the coupler design, and three complementary adhesives split the optical, structural and strain-relief roles. | Glue selection is owned by packaging, yet it is a coupler design parameter. | Three glue bottles, each with a job title | GF22 p.3 Fig. 1 caption; p.5 Sec. IV | S |
| 1.5 | UV-adhesive cure shrinkage displaced an already self-aligned optical connector, so adhesive amount and adhesion points had to be optimized. | "Alignment done" is declared before the glue cures. | Connector drifting under a UV lamp | AIST23 p.3 Sec. II.C | S |
| 1.6 | Optical epoxy shrinks on cure. It helps coupling only if the shrinkage is accounted for and acts along the optical path. Otherwise the gap changes. | Glue is treated as a fixed spacer. | Wool sweater in the dryer | TYN23 p.1 Intro | S |
| 1.7 | Simulation assumed underfill n = 1.457, but the epoxy actually used was n = 1.5. Higher index relaxes Z tolerance but weakens confinement, so tapers must be longer to compensate scattering loss. | Epoxy is picked for viscosity and cure, and the index is assumed. | Epoxy "index-matched" to the wrong thing | TYN23 p.2 Simulation; p.3 Methodology | S (sim-vs-used mismatch I) |
| 1.8 | An underfill fillet forms on the waveguide facet. The fix needed a solder dam, lower-viscosity underfill dispensed perpendicular to the facet with controlled length, height and volume, and an O2/Ar plasma clean. | Underfill is an electrical step that silently kills optics. | Horror-film ooze creeping to the facet (Blender) | LM26 p.1 Abstract; p.2 Sec. II.C, Fig. 6 | S |
| 1.9 | Underfill droplets deposit on the optical facet *during underfill cure*, not during dispense, which is why the plasma clean was needed. | Dispense inspection looks clean. | Facet with polka dots | LM26 p.1 Abstract | S |
| 1.10 | In open-cavity CPO, the lid TIM can flow between the EIC and PIC dies and contaminate the PIC edge, a high risk for fiber attach. A polymer TIM was chosen over solder or liquid metal. | The thermal team's material is an optical risk. | Liquid metal heading for the facet | INT25a p.5 Sec. IV.C | S |
| 1.11 | Open-cavity underfill must not bridge between PIC dies, roll up onto restricted PIC areas or overrun the cavity edge. It needed jet dispense at several locations in multiple passes. | Fillet control becomes facet cleanliness. | Underfill surfing up the die | INT25a p.4 Sec. IV.B, Fig. 11 | S |
| 1.12 | Optical-adhesive overflow from the glass-block attach (over grating couplers) can reach the neighboring PIC-EIC microbump region. It needed a dam and a keep-out zone (KOZ). The critical controls are alignment, bond-line thickness and overflow. | Optical epoxy is not seen as a contaminant of electrical joints. | Flood barrier at the microbump city limits | MRVL26 p.4 Sec. III.B, Fig. 9; p.5 Fig. 10 | S |
| 1.13 | The grating-coupler opening must be guarded during bridge-die attach with a polyimide passivation (PSV) dam. The block-to-grating epoxy then replaces the grating's designed superstrate, so peak wavelength had to be checked against baseline. | The optical window is treated as just another via. A grating designed for one cladding gets another. | Grating shocked by its new epoxy hat | CEL25 p.3 Sec. II.B, Fig. 5; p.5 Fig. 10a | S (superstrate mechanism I) |
| 1.14 | Exposed resin lenses rule out fluxed solder (flux-less or lens protection needed). Underfill would invade the lens trench, so only sidefill was used, and larger chips will need underfill. | A standard OSAT flow is assumed. | Underfill flood heading for the lens | SEI25 p.5 Sec. V.B | S |
| 1.15 | IBM: adhesives had to be "extensively evaluated, modified and enhanced" for reflow and JEDEC (semiconductor reliability standards body) compatibility. Many materials needed replacement. The adhesive spec covers RI, viscosity, mechanical, thermal and modulus properties. | Glue is treated as a consumable, not a component. | Glue tube wearing a "Critical Component" lanyard | IBM25 p.5 Sec. V.B; Table 1 row 10 | S |
| 1.16 | The polymer-waveguide-to-PIC adhesive loses peel strength within the first 24 h of temperature/humidity exposure, then plateaus through 200 h. The value depends on whether it is measured moist or oven-dry. | Only the plateau gets called "stable". | "Now 50% less sticky" | IBM25i p.1 Sec. III; p.3 Fig. 4 (axis unitless; the "about half" reading is I) | S (magnitude I) |
| 1.17 | The Na+ needed for ion exchange in glass waveguides can migrate into optical adhesive and into attached ICs, degrading transistors, so a diffusion barrier is mandatory. | Optics people don't think about CMOS mobile-ion contamination. | Sodium ions escaping to sabotage the switch ASIC | COR24 p.3 Sec. V.A | S |
| 1.18 | Transparent resin fills the EML-to-prism optical path, with mold elsewhere. No resin loss, index-stability or aging data is given. (EML = electro-absorption modulated laser.) | The encapsulant is treated as optically inert. | "Trust me, it's clear" | MEL26 p.2 Sec. II.A | S (design); risk I |

## Cluster 2. "Reflow, Humidity and the Connector Nobody Qualified": reliability traps of fiber-attached CPO

- **Experts (inference):** about 50-200 people. Basis: only Intel, IBM, GF/IBM, Lightmatter/PHIX and TSMC publish module-level reliability in this corpus, plus FAU and MT vendor qualification engineers.
- **Why niche:** telecom GR-468 (Telcordia optoelectronic reliability standard) and semiconductor JEDEC cultures collide. Failures are bimodal, and some have no root cause.
- **Visual potential: high.** Blender: an MT ferrule in an oven mitt; two identical chiplets, one failing, with a shrugging FA (failure analysis) team. Code: an animated histogram that goes bimodal over uHAST hours, from Table VIII values (uHAST = unbiased highly accelerated stress test).

| # | Pitfall | Why experts miss it (inf.) | Gag / visual (inf.) | Source + locator | Conf |
|---|---|---|---|---|---|
| 2.1 | Under uHAST, mean IL/facet (insertion loss) went -3.03 (25 h) to -6.44 dB (275 h), with std rising to 3.69 dB. Some chiplets fell to about -15 dB/facet while others held about -3 dB. FA found no assembly-related difference (adhesive delamination or crack, fiber crack) between pass and fail. | A bimodal humidity failure with no visible cause; the mean hides it. | FA team shrugging at identical twins; animated bimodal histogram | INT25b p.4 Sec. V, Table VIII, Fig. 8 | S |
| 2.2 | Industry-standard FAUs are not qualified above 100 C, largely because of the MT connectors. A standard FAU alone degraded about 0.5-0.75 dB IL/facet after SAC (tin-silver-copper solder) reflow. Workarounds were thermal insulation of the MT or a supplier reflow-compatible FAU. | The connector, not the chip interface, is the reflow weak link. | MT ferrule in a tiny oven mitt | INT25b p.3 Sec. IV | S |
| 2.3 | HTS 150 C (high-temperature storage) gave about 0.4 dB change at 1008 h. The authors suggest polymer aging of adhesives or MT connectors causing fiber shift or MTP-MT contact issues. | Slow drift is blamed on the chip. | Aging glue with a walking cane | INT25b p.4-5 Sec. V, Table IX | S (hedged "could be") |
| 2.4 | Mean IL through TCB (temperature cycling B) is non-monotonic: -2.38 (end of line), -2.51 (3x reflow), -2.69, -2.83, -2.89, then -2.73 dB at TCB1500 (n = 16). The text calls it "very stable". | A partial recovery hides the mid-test drift. | IL improving with more abuse | INT25b p.4 Table VII | S (numbers); I (interpretation) |
| 2.5 | At 20 G RMS vibration, fibers were severely damaged without proper board-level fiber support. With support, IL change was minimal. | Pigtail routing is a board/system responsibility. | Fiber pigtail as a guitar string | INT25b p.5 Sec. VI, Fig. 10, Table XI | S |
| 2.6 | IBM's early full-module builds showed IL increases after lead-free reflow in both optics-first and optics-last flows. Early PIC-to-SMF IL was 3.2 to more than 5.1 dB, improved to below 1.5-3.0 dB only after "numerous changes to the materials, processes, and quality integration". | The abstract reads as a clean pass. | "First-time-right" award with sticky note "iteration 7" | IBM25 p.6 Sec. V.E, Fig. 8; p.8 Sec. VI | S |
| 2.7 | In IBM's earlier PIC-to-polymer-waveguide-to-ferrule samples, thermal cycling and 85 C storage gave small or no IL change. Damp heat 85/85 gave 0-0.75 dB increase in most samples. | Thermal cycling is assumed to be the killer, but moisture was. | Humidity chamber as the final boss | IBM25 p.3 Sec. IV | S |
| 2.8 | Earlier Telcordia-passing optical-link test vehicles had no base package and no combined electrical-optical evaluation, so the full CPO module needed separate qualification. | "GR-468 passed" gets carried forward to a different structure. | Certificate "valid for a different product" | IBM25 p.1 Sec. I | S |
| 2.9 | In open-cavity large organic substrates, about 50% delaminated at a solder-temperature pre-screen bake. Yield loss reached about 90% through TCB (thermocompression bonding), lid and ball attach. A lower-thermal-exposure process fixed it. | The PIC cavity is the substrate's weak spot. | Substrate peeling like a croissant | INT25a p.4-5 Sec. IV.C, Fig. 13 | S |
| 2.10 | Photonics thermal sensitivity forced low-temperature solder for the BGA (ball grid array). | The thermal budget is set by the optics, not the silicon. | Photonics on a cold-solder diet | INT25a p.5 Sec. IV.C | S |
| 2.11 | The PWG (polymer waveguide) without a substrate cut-out must be bonded to the PIC before the rest of assembly and ride through every later thermal step. The cut-out (optics-last) exists to cut that budget. | Assembly order is set by the electrical team. | Waveguide forced to attend every reflow party | IBM25i p.1 Sec. II; p.2 Sec. V; IBM25 p.5 Fig. 6 | S |
| 2.12 | GF/IBM: "moisture ingress has been one of primary challenges in industry". Their 260 C reflow and -40/125 C TC (temperature cycling) passes used only "a handful" of modules with limited sample size. LTS (low-temperature storage) duration is 72 h in Table 1 but "LTS 500hrs" in Fig. 5. | The CPO-relevant reflow evidence base is thin. | Tiny sample-size font; two clocks disagreeing | GF22 p.6 Sec. V.C, Table 1 vs Fig. 5 | S |
| 2.13 | LM26 thermal cycling covered "a few modules" up to 15 cycles, plus an 80 C bake as infant-mortality screen, with process monitoring on TE0 IL only. The abstract claims robustness to temperature cycling and polarization changes. | Qualification scope is far smaller than the abstract implies. | 15-cycle reliability trophy | LM26 p.3 Sec. III; p.1 Abstract | S (scope); I (gap) |
| 2.14 | TSMC COUPE optical reliability (TC, damp heat 85/85 1000 h, HTS 175 C 1000 h; delta IL < 0.25 dB) is wafer-level with no FAU attached. That leaves out the interface where Intel and IBM saw failures. | "Passes reliability" is read as module-level. | Crash-test certificate for the engine block alone | TSMC25a p.4 Sec. V, Fig. 11 | S (scope); I (gap) |
| 2.15 | Intel used JEDEC rather than Telcordia conditions because JEDEC is harsher. | Two qualification cultures. | Two rulebooks at a duel | INT25b p.4 Sec. V | S |
| 2.16 | The bridge-chip membrane relieves stress only if left free, so underfill could not be used on membrane samples. Die shear was tested only on non-membrane bridges, and the design was not yet in an optical module. | The underfill reflex voids the design. | Checkbox "Underfill applied" voiding warranty | SEI24 p.4 Sec. IV, Fig. 9; p.1-4 | S |

## Cluster 3. "Warpage Is an Alignment Error": temperature-dependent shape vs sub-micron optics

- **Experts (inference):** about 100-300 people at the warpage-meets-optics interface. Basis: 9 organizations here (Lightmatter, Intel, TSMC, A*STAR, imec, Celestial, AIM, GF, IBM). Warpage specialists in electronics number in the thousands, but few connect warpage to coupling loss.
- **Visual potential: very high.**
  - Blender: a package doing smile-frown-smile through the reflow profile (LM26 Fig. 5 sign changes); a potato-chip PIC die washed off a carrier (IMEC24).
  - Code: shadow-moire-style warpage vs temperature animation driven by the stated sign sequence. Only the stated magnitudes may be used.

| # | Pitfall | Why experts miss it (inf.) | Gag / visual (inf.) | Source + locator | Conf |
|---|---|---|---|---|---|
| 3.1 | Module warpage changes sign with temperature: convex at room temperature, maximum concave at 150 C (mold Tg (glass transition temperature)), convex again by 260 C. Fig. 5 labels 58.9 um at 260 C; the text says about 56 um at reflow. | A room-temperature flatness check says nothing about reflow. | Smile-frown-smile package (Blender) | LM26 p.2 Sec. II.B, Fig. 5 (conflict logged in `data/evidence/ectc51846.2026.00176.yaml`) | S |
| 3.2 | Warpage appears as C4 solder height spread 51.7-85.6 um center to edge. PIC and substrate warpage is named a major source of coupling-efficiency variation: 0.43 dB/facet mean, 1.48 dB worst case (mean + 3 sigma). | Electrical yield >95% makes the spread look harmless. | Bumps lined up by height like a class photo | LM26 p.3 Sec. III, Table 2, Fig. 7 | S |
| 3.3 | The model flags the center PIC of three as highest risk for front-edge warpage and tilt at the FAU attach edge. | Package position changes optical risk. | Middle child of three PICs | INT25a p.3 Sec. III, Fig. 7 | S(sim) |
| 3.4 | As fiber count rises, PIC warpage and fiber-array overlay control become critical for edge-coupler loss. A 40-fiber iFAU (integrated FAU) averaged 1.28 dB loopback (sigma 0.18), with fiber-to-fiber variation the main factor. | Single-fiber hero numbers hide array overlay. | 40-fiber array on a frowning chip | TSMC25b p.1-2 Sec. I; p.3 Sec. III.A, Fig. 4 | S |
| 3.5 | A stressed SiN waveguide layer warps thinned PIC dies (7.2 x 7.2 mm, 100 um): 10 um side to side, 18 um diagonal. A die-to-wafer flow already proven on electrical dies failed on the first dies with optical waveguides. | Waveguide films are seen as optical layers, not stress layers. | Potato-chip die; recipe stamped "electrical only" | IMEC24 p.3 Sec. II, Fig. 4b; p.4 Sec. III | S |
| 3.6 | On the carrier, warped dies (about 8 um) touched only at the edges and voided underneath. The protective-layer strip "flushed off" 12% of dies on one carrier and 0% on the other ("marginal process window"). Only 74% transferred; particles were ruled out by SAM (scanning acoustic microscopy). | The cleaning step is not seen as a yield step, and contamination is the default suspect. | Dies going down the drain (Blender) | IMEC24 p.3-4 Sec. II, Fig. 4d, Fig. 6 | S |
| 3.7 | The imec fix was process-side, not lower wafer stress: coat the release layer on the thinned wafer backside before singulation. Result: 99.9% carrier yield, <1 um warpage, 96% transfer. Die edges still bend up and void after bonding. | The wafer stack is outside the assembler's control. | Die curling its edges like old paper | IMEC24 p.4-5 Sec. III, Figs. 7-9 | S |
| 3.8 | Removing oxide from the back of the top SOI wafer during bond/thin causes an abrupt bow change of "usually on the order of -400 um", which can break wafers. The fix is a pre-built counter-bow plus stress relief. | One-sided dielectric removal is an unbudgeted stress step. | "Pre-warp it the other way" memo | AIM23 p.4 Sec. III.A, Fig. 5 (OCR) | S |
| 3.9 | Grating-loss outliers through a through-mold glass block were attributed to unoptimized test-wafer warpage. Reconstituted-wafer warpage depends on EMC (epoxy molding compound) shrinkage, CTE (coefficient of thermal expansion), Tg, cure, reflow, temporary bond/debond and underfill cure. | Warpage is seen as a yield issue, not an optical one. | Wafer taco through process steps | CEL25 p.4 Sec. III.B, Fig. 7; p.5 Sec. III.C, Fig. 10b | S |
| 3.10 | A 79 x 52 mm molded interposer needs four-field stitching. Predicted board-level warpage is about 230 um without a stiffener vs about 116 um with stainless steel. Local flatness at the PIC is about 4 um. | Good local flatness hides global warpage. | Potato-chip interposer with a steel belt | IME26 p.2 Sec. III; p.4 Sec. V, Figs. 6-7 | S(sim) |
| 3.11 | A V-groove array lowers die flexural strength, so bare-die handling (shipping, pick-and-place, die attach) is the exposure window. | Grooves are treated as optics, not stress concentrators. | Pick-and-place nozzle as villain | GF22 p.3 Sec. II.B, Fig. 2 (small sample, worst case per authors) | S |
| 3.12 | A glass interposer's low lateral thermal conductivity can create nonuniform gradients, stress, warping or cracking: the same property sold as thermal isolation. | The isolation benefit hides the mechanical cost. | "Our insulator is also our crack initiator" | TYN24 p.8 Sec. IV-D | S (qualitative) |

## Cluster 4. "Tolerance Is Conserved": alignment budgets, gap budgets, and who actually sets them

- **Experts (inference):** about 100-500 people (coupler designers plus precision-assembly engineers). Basis: nearly every corpus organization; the topic is broad, but the specific traps below are niche.
- **Why niche:** the tolerance gets moved, not removed. Lateral turns into angular, the gap is set by pad size, height is set by a deposition recipe.
- **Visual potential: very high for code interactives.**
  - A Gaussian-beam overlap explorer with sliders for MFD (mode field diameter), offset, tilt and gap, showing lateral tolerance rising while angular tolerance falls (SEI25, IME26). Textbook overlap integrals; only paper-stated values on labels.
  - A solder-bump-height vs pad-size toy (TYN23).
  - Blender: a laser beam clipping its own chip edge off a 45-degree prism (MEL26).

| # | Pitfall | Why experts miss it (inf.) | Gag / visual (inf.) | Source + locator | Conf |
|---|---|---|---|---|---|
| 4.1 | Standard flip-chip placement (+/-3 um) is fine electrically but "catastrophic" for waveguide-to-waveguide coupling, which needs about 0.1 um. | The electrical team owns the bonder spec. | Bonder brochure "+/-3 um, great accuracy!" | SEI25 p.2 Sec. I | S |
| 4.2 | A beam-expanding lens relaxes lateral 1-dB tolerance to +/-11 um but tightens angular tolerance. Chip tilt is assumed negligible. Coupling was deliberately traded to 1.6 dB (0.8 dB considered ideal); uncoated lens surfaces add 2 x 0.2 dB Fresnel loss. | Tolerance moves, it does not vanish. | Seesaw labelled lateral / angular; overlap interactive | SEI25 p.3 Sec. II.B, Fig. 8; p.4 Sec. IV.C; p.5 Sec. V.A | S |
| 4.3 | Bare edge coupling has a +/-1.1 um 1-dB tolerance. A lensed expanded beam (ROC 300 um, MFD ~42 um) measured +/-6 um; >+/-15 um is simulation only, because the current lens still diverges the beam. | The simulated number gets quoted. | "Measured +/-6; Simulated +/-15; Marketing +/-inf" | IME26 p.6 Sec. VI, Fig. 11 | S |
| 4.4 | The coupler choice sets the gap, and the gap sets the bonding process. Free-form reflectors need <= 14 um, so solder works; inverse tapers need <= 3 um, so epoxy is required. | Coupler and assembly are owned by different teams. | Solder ball too fat for a 3 um door | TYN23 p.1 Intro, Fig. 1 | S |
| 4.5 | The bump-set gap scales inversely with pad size: a 50 um pad gives a 26.8 um bump, a 100 um pad 12.14 um. The bonded gap (8.07 um) is about 4 um below the pre-bond bump height. Text says +/-0.177 um, Fig. 7 label says +/-0.107. | Pad size is drawn for electrical reasons. Designing from single-bump height. | Pad-size dial labelled "optical gap" | TYN23 p.3 Results, Figs. 5-7; p.4 Fig. 7 | S (collapse interpretation I) |
| 4.6 | Evanescent taper tolerances: +/-2.7 um lateral, 2.8 um Z, >+/-150 um X, +/-2.3 deg twist, and only +/-0.4 deg tilt. The X/Y values were simulated at 0 um gap, but the built epoxy gap was 1.618 um. | Pick-and-place specs focus on XY, and tolerances are read without their boundary conditions. | Die bonder perfect in XY, wobbling 0.5 deg | TYN23 p.2 Simulation, Fig. 2; p.4 | S (consequence I) |
| 4.7 | Evanescent coupling avoids sub-micron alignment but needs tapers several mm long, which inflates PIC die area. | Silicon area cost gets ignored. | Taper longer than the chip | SEI25 p.2 Sec. I | S |
| 4.8 | A flip-chipped EML sits so close to the glass that the beam reflected by the 45-degree prism clips at the glass surface or at the EML's own chip edge. Small prism setback clips the chip edge, large setback clips the glass. Raising the emitter (bump height) helps. A 0.7 um spot diverges and clips; a 1.5 um spot is nearly loss-free at h = 20 um. | Bump height is an electrical parameter; spot size is a laser parameter. | Laser bonking its own chip edge (Blender) | MEL26 p.3 Sec. III.A, Figs. 3-4 | S(sim) |
| 4.9 | Collimating-lens in-plane tolerance for IL <= 1 dB is +/-0.77 um. The "passive" vision-placed prism leaves axis error that has to be absorbed by actively aligning the lens array. | "Passive assembly" still contains an active step. | "Passive" sticker on an active-alignment robot | MEL26 p.2 Sec. II.A; p.3 Fig. 5 | S(sim) |
| 4.10 | For a III-V laser edge-coupled into an interposer trench, sub-2 dB coupling needs lateral alignment <1 um (observed <0.4 um). Vertical alignment is set by trench standoffs in the interposer dielectric stack, not by the bonder. | The laser's height was fixed by a deposition recipe months earlier. | "Your laser's height was decided at tapeout" | AIM23 p.4 Sec. II.C; p.5 Sec. III.C, Fig. 8c (OCR) | S |
| 4.11 | V-groove width sets *vertical* fiber position. Too narrow or too wide puts the fiber core above or below the suspended SSC, and IL rises steeply on both sides of the optimum. Once seated there is no correction: Monte Carlo gives "up to 1.3 um" from fiber and groove tolerances. | A lateral dimension sets the vertical error, so passive alignment becomes incoming-part metrology. | Goldilocks fiber in a groove | GF22 p.4 Sec. III, Fig. 3; p.5 Sec. V.A | S |
| 4.12 | Self-alignment error is anisotropic: x +1 um, y +3.5 um. The y axis caused the ~1 dB penalty, and the first self-aligned version lost 3 dB vs active alignment. | A guide constrains one axis much better than the other. | Connector centered sideways, floating vertically | AIST23 p.2 Sec. I; p.3 Sec. III; p.4 Sec. IV (OCR) | S |
| 4.13 | A 12-fiber array's yield is limited by its worst link. | Averages look fine. | One bad fiber sinking the ribbon | GF22 p.5 Sec. IV | S |
| 4.14 | The die-to-carrier pick-and-place (m+3s about 6 um X, 5.15 um Y) is worse than the final 2.4 um. The final number comes from pre-correcting the average misalignment at the collective wafer-to-wafer bond. EVC (evanescent coupler) loss correlates with Y misalignment. | The single-die placement spec gets read as the system spec. | Wafer of dies nudged together | IMEC24 p.4-5 Sec. II, IV, Figs. 5-6, 10 | S |
| 4.15 | A nanoimprinted 45-degree mirror came out at 46.7 deg. Imprint copied the mold's own angle spread (sigma 0.736 deg vs mold 0.773 deg). The mirror coupler lost 10.2 dB (vs 4 dB grayscale-litho). A >50 um mirror-to-waveguide gap makes the path sensitive to angle. | Under 2 deg sounds negligible. The stamp is assumed perfect. | Photocopier reproducing a coffee stain; protractor in the kit | AIST24 p.3 Sec. III.A; p.4 Sec. III.B; p.5 Sec. IV | S |
| 4.16 | In a free-form reflector pair, rotation about the propagation axis steers the beam off the output facet. That is the tightest tolerance. | Roll about the beam axis is assumed harmless. | Beam missing the facet | TYN23 p.2-3, Fig. 3 | S(sim) |
| 4.17 | An index-matching oil's capillary motion misaligns the fiber, so a mass had to be placed on the fiber during measurement. | The index liquid itself moves the fiber. | Paperweight on a fiber | IZM26 p.3 Sec. II.E | S |

## Cluster 5. "Glass and Polymer Are Not Inert": waveguide materials that drift, diffuse, scatter and guide where they should not

- **Experts (inference):** about 30-150 people. Basis: few organizations in the corpus (Corning x3 papers, DNP, AIST, IBM, Sumitomo, IZM, Tyndall, LightSpeed), and glass-waveguide CPO is a small community.
- **Why niche:** ion-exchange diffusion kinetics, barrier-layer guiding and imprint residual layers are known to a handful of groups.
- **Visual potential: high.**
  - Code: a 1D diffusion (erfc profile) interactive showing the IOX (ion-exchange) index profile flattening, MFD growing and coupling loss rising over time at ASIC temperature. The mechanism is from COR24; any numbers shown should be only those in COR24 Table 2.
  - Code: a two-waveguide crosstalk toy with a residual-layer slider (DNP26).
  - Blender: a waveguide melting like a popsicle next to a GPU.

| # | Pitfall | Why experts miss it (inf.) | Gag / visual (inf.) | Source + locator | Conf |
|---|---|---|---|---|---|
| 5.1 | IOX glass waveguides keep diffusing in service. ASIC heat far below the 350 C process temperature still degrades the index profile "over a period of weeks". | "Process 350 C, use 85 C, so it is frozen." | Popsicle waveguide by a GPU | COR24 p.2 Sec. II.A | S |
| 5.2 | The failure is gradual: lower core index, wider mode, higher fiber coupling loss. A 50% core index drop adds 1.12 dB, against a 4 dB 400G-FR4 budget. Hot spots and load cycling are excluded from the 85 C model. | The loss budget is checked at T0 only, at an average temperature. | Mode field inflating like a balloon; diffusion interactive | COR24 p.2 Sec. II.B, Tables 1-2 | S |
| 5.3 | Commercial IOX glasses developed for other applications "fail in the first month" at CPO temperature. The 2026 paper puts computing chips near 110 C, versus 85 C in the 2024 lifetime model. Only "a few" of more than 200 compositions meet the 110 C diffusivity target. | The telecom PLC (planar lightwave circuit) glass worked before, and the spec moved between papers. | Calendar "Day 30: link down" | COR26 p.2 Sec. II, Fig. 2; p.3 Fig. 3; vs COR24 p.2 | S (spec-drift comparison I) |
| 5.4 | The Na+ diffusion barrier fights evanescent coupling. Above 200 nm SiO2, IOX-to-SiN loss rises sharply; thinner barriers need 2.5-3 mm couplers. Barriers with index >1.6 support their own guided modes. TE and TM curves differ. | The barrier is specified by chemists. A "thin film" secretly becomes a waveguide. | Film that is moonlighting as a waveguide | COR24 p.4 Sec. V.B, Figs. 7-8 | S |
| 5.5 | Measured IOX waveguide loss (0.034 dB/cm) is four times the intrinsic glass loss (0.008 dB/cm). Reaching low intrinsic loss required cutting tramp metals (Pt, Rh, Fe) and drying the glass. | "Low-loss glass" is assumed to mean "low-loss waveguide". | Crucible metals as stowaways | COR24 p.3 Sec. III-IV, Fig. 3 | S |
| 5.6 | Mode-matched low-contrast glass waveguides have severe bend loss: 4 dB/cm at 10 mm radius before redesign. Outer fan-out channels (1 and 16) lost up to 1.3 dB more until mask-width-varied contrast was used. | The design is optimized for fiber match, then routing area runs out. | Bus turning in an alley | COR26 p.1 Abstract; p.3-4 Sec. III, Figs. 6-7 | S |
| 5.7 | Imprint lithography leaves a thin residual resin layer between polymer cores. It cannot be fully removed and raises crosstalk. Etching it off raises sidewall roughness and propagation loss. | Each fix creates the next failure. | Whack-a-mole; waveguides holding hands under the table | DNP26 p.1-3 Figs. 2-3, Sec. III.A, Fig. 6 | S |
| 5.8 | -30 dB crosstalk at 10 um pitch needs high index contrast (about 3.0% simulated, 2.3% measured). The result holds only for the 18 mm section tested; crosstalk grows with length. Polymer contrast fell 3.02% to 2.87% after 500 h HTS 150 C, measured on a blanket film. | A short test coupon, a long product; reliability measured on a film. | Ruler extending until alarm | DNP26 p.2-5 Figs. 3, 5, 7, 8 | S (film caveat I) |
| 5.9 | Polymer waveguides came out 10.3 x 8.4 um against an 8 um design and went multimode. Residual air bubbles left one mirror unformed, and that site was quietly excluded from the histogram. Grayscale-litho mirrors are sensitive to cleanroom temperature and humidity. | Thick-polymer litho tolerance; statistics drop the defect. | Single-mode waveguide with an identity crisis | AIST24 p.2 Sec. II; p.3 Fig. 4(d) caption; p.5 Sec. IV | S |
| 5.10 | Laser-assisted-etched glass trench sidewalls are rough (Ra 400 nm, Rz 2 um). Simulated, 2 um roughness costs 14 dB from air, but 0.3 dB when filled by an index-matched resin lens. | The trench is judged by shape, not optical finish. | Frosted bathroom window in the beam | SEI25 p.4 Sec. IV.B, Fig. 6 | S(sim) |
| 5.11 | Through-mold glass blocks become optical surfaces after mold grind and needed CMP (chemical-mechanical polishing) to about 1 nm Ra. | Grinding is a mold-reveal step, not optics. | Optician polishing a mold-grind face | CEL25 p.5 Sec. III.C, Fig. 8; MRVL26 p.4 Sec. III.B | S |
| 5.12 | The SLE (selective laser etching) V-groove leaves an exposed vertical edge next to the IOX waveguide. Possible local index effects are uncharacterized. | Machining sits next to a graded-index guide. | Waveguide on a cliff edge | IZM26 p.3 Sec. III; p.4 Outlook | S (open issue) |
| 5.13 | Copper adheres poorly to glass ("one of most critical challenges"), so a Ti adhesion layer is needed. Spin-coat resist pools on deep cavity walls, and even spray coat flows down the sidewall. Contact-mask exposure cannot focus on a 100 um cavity floor. | Organic-substrate habits get carried over to 3D glass. | Resist pooling in a bathtub; projector focused on the ceiling | CORK25 p.2-3 Sec. III, Figs. 5-6 | S |
| 5.14 | The Cu-filled TGV vs glass CTE mismatch dictated a 150 um wafer. A yield-packed irregular die layout made machine and laser dicing infeasible; manual scribing gave +/-40-1200 um dimensional variation, and dies chipped even with the SOP followed. | The mask layout never asked the dicing team. | Tetris wafer, hand saw | LSP23 p.1 Intro; p.3 Results; p.4 Tables 2-4 (OCR) | S |
| 5.15 | The PIC optical edge must survive thinning to 50 um and singulation. That required frontside plasma trenching plus a first-of-a-kind temporary carrier. | Dicing quality is optical quality. | Facet surviving the grinder | INT25a p.2 Sec. I; p.3-4 Sec. IV.A | S |

## Cluster 6. "Your Fix Is Someone Else's Bug": EIC-PIC thermal and electrical co-design traps

- **Experts (inference):** about 100-300 people who work across ring tuning, SerDes (serializer/deserializer) circuits and package thermal/RF. Basis: Nvidia, Intel, TSMC, Sumitomo, A*STAR, Tyndall/IZM, CEA-Leti and Mitsubishi in the corpus.
- **Why niche:** each fix is locally right and globally wrong. The bridge that fixes RF leaks heat; the ground plane that fixes crosstalk leaks heat; the ring-variation fix changes the clock architecture.
- **Visual potential: high.**
  - Code: a ring-resonance drift interactive driven by heater and neighbor power, using only stated budgets (NV26 0.76 pJ/b heaters). A thermal resistance network comparing wire bond and bridge, using SEI24 Table I values.
  - Blender: a silicon bridge as a heat highway with a toll booth at the TEC (thermoelectric cooler).

| # | Pitfall | Why experts miss it (inf.) | Gag / visual (inf.) | Source + locator | Conf |
|---|---|---|---|---|---|
| 6.1 | Replacing wire bonds with a Si bridge (for >100 GHz) creates a heat path from EIC into the TEC-controlled PIC and raises TEC power. The bridge measured 104.4 K/W vs about 150 K/W for 16 Au wires. 50 um and 100 um etched gaps give 202.6 and 232.4 K/W. | Wire bonds are never counted as thermal insulators. | Wire bonds win "best insulator" award | SEI24 p.1 Sec. I; p.2-3 Sec. II, Table I, Fig. 4 | S(sim) |
| 6.2 | The bridge-metal resonances cause transmission dips and adjacent-channel crosstalk. The fix (a ground layer) costs thermal insulation. | The default RF fix is also a heat spreader. | Ground plane labelled both "fix" and "leak" | SEI24 p.4 Sec. III-B, Fig. 8 | S |
| 6.3 | The PIC on a TEC, the EIC on a heat-transfer block and a CuW spreader all have different CTEs. They give 3D relative shifts that stress the bridge bumps. | A rigid bridge spans two independently moving carriers. | Bridge between drifting ice floes | SEI24 p.2 Sec. II, Fig. 2(b) | S |
| 6.4 | Lateral heat from adjacent xPUs sets PIC temperature. The peak is at the PIC corners nearest the xPU, making the PIC the most stringent thermal constraint. Holding it <65 C at about 1 kW took four 120 mm fans and a 248 x 258 mm vapor chamber. | The optical engine is about 10 W, so it gets budgeted in isolation. | Tiny PIC sunbathing by a 1 kW bonfire | IME26 p.5 Sec. V, Fig. 9 | S(sim) |
| 6.5 | TX+RX ring heaters cost 0.76 pJ/b, equal to the laser's 0.76 pJ/b. The laser figure assumes 10% fiber-coupled wall-plug efficiency; the total is 2.78 pJ/b. | Heater power is booked as overhead. | Pie chart: "keeping rings warm" ties the laser | NV26 p.2 body text | S |
| 6.6 | Wafer-to-wafer ring resonance variation makes fixed TX-RX ring mapping cost high heater power. Flexible mapping fixes that, but the forwarded clock can then land on any RX ring, so an injection-locked-oscillator clock distribution replaced a 9:1 mux. | A process-variation problem becomes a clocking-architecture change. | Musical chairs for wavelengths | NV26 p.1 RX section, Fig. 23.1.4 | S |
| 6.7 | The MRM (microring modulator) needs 1.5 V reverse bias, i.e. a 1.95 V cathode. To avoid CMOS overstress the path is ac-coupled, which causes baseline wander and needs a dc correction loop. A large bias resistor (~80 kOhm in prior work) would drop about 0.8 V from uA-level ring photocurrent. | Photonics sets the bias; the transistor rating belongs to another team. | Ring demanding volts the transistor cannot give; modulator stealing its own bias | INT26v p.1 Sec. B, Fig. 3 | S |
| 6.8 | Each MRM and add-drop filter needs its own thermal loop locked to a unique laser using photocurrent. The TX metric must subtract the baseline-wander correction current. The TX lock setpoint comes from a pre-characterization step before live data. | Control loops interact through shared bias paths and hidden calibration. | 32 thermostats in one apartment | INT26v p.1 Sec. E, Fig. 4; NV26 p.1 Fig. 23.1.3 | S |
| 6.9 | The TX level shifter must tolerate up to 300 uA of MRM leakage, linked to carrier photogeneration. | A lit diode is not a dark reverse-biased diode. | Diode that leaks when you look at it | NV26 p.1 TX section, ref [16] | S |
| 6.10 | RX rings are placed in radius-descending order to partly cancel optical crosstalk. | Ring order on the bus is not a free layout choice. | Rings lined up by height | NV26 p.1 | S |
| 6.11 | Each MRR (microring resonator) sits over a 40 um backside cavity etched to the BOX to cut tuning power. That forced a dedicated spin-coat process to passivate over cavities. | Heater efficiency turns into a backside process change. | Ring floating over a hole, "do not step here" | LETI23 p.2 Sec. II; p.5 Sec. III.E.2, Fig. 18 | S |
| 6.12 | An ASIC on an active photonic interposer needs a thermal solution against "thermal contamination" of PIC devices. Micro-bump EIC-PIC stacking hits a parasitics wall. | The photonic interposer is treated as passive wiring. | Hazmat suit for heat | TSMC23 p.4 Sec. III.D; p.1 Sec. II.A | S |
| 6.13 | Solder-bump capacitance pulls PD 3 dB bandwidth down to 32 GHz. TIA (transimpedance amplifier) input noise scales with Cin^2, where Cin includes PD, interconnect and ESD capacitance. Lower Cin gives >2.2 dB sensitivity and 32-40% less laser power. | ESD capacitance ends up on the laser power bill. | ESD diode invoicing the laser | TSMC24 p.3 Sec. III.B-C, Figs. 5, 7 | S(sim) |
| 6.14 | Smaller rings for 32/64 wavelengths improve heater efficiency and thermal crosstalk, enabling tighter MRM pitch. That forces EIC-PIC interconnect pitch scaling. | Wavelength count becomes a bump-pitch problem. | Rings shrinking faster than bumps | TSMC24 p.2 Sec. III.A, Fig. 2 | S |
| 6.15 | Thermal vias in glass saturate below 100 um pitch because the FR4 PCB is the bottleneck: 93% fewer vias raised chip temperature only 2.9 C. All-copper vias gave at most 1.5 C; a PCB-embedded Cu heatsink gave about 15 C. Test-chip power was only 0.13-0.5 W vs 5-50 W real. | People optimize the part they designed, not the board below. | Drilling more vias against a PCB wall | TYN24 p.5-6 Sec. III-B, Figs. 6-9, Table I | S (largely sim) |
| 6.16 | Uneven underfill between glass interposer and PCB leaves air pockets that cause IL resonances above 20 GHz. Geometry-only redesign (via, pad, solder sizes) cut IL by about 11.83 dB at 40 GHz in simulation. | Underfill is treated as mechanical, not an RF dielectric. | Bubbles appearing as S21 peaks | TYN24 p.9 Sec. IV-D, Figs. 15-16 | S |
| 6.17 | Glass dielectric loss measured about 0.0435 dB/GHz/cm, "very close" to FR4. Package vertical transitions (TGV, microvia, BGA pads) dominated, with 3-dB bandwidth only 19.56-25.55 GHz. | Glass is assumed to beat laminate on RF. | Premium glass on a scale reading "FR4" | TYN24 p.7-8 Sec. IV-B/C, Figs. 13-14 | S |
| 6.18 | Pads >300 x 300 um in polyimide build-up need holes, or outgassing cracks them during fabrication and soldering. The second-level Bi-Sn reflow oven set-point must exceed 155 C because the interposer lags the oven. | Big solid pads look like good ground practice; the oven readout is trusted. | Swiss-cheese bond pad | TYN24 p.2 Sec. II; p.4 Sec. III-A | S |
| 6.19 | Metal-density meshing of wide metal adds inductance and loss, worst on signal traces. M2 lines lose more than M4 because lossy Si sits about 6 um below. Fine lines cost 1.35-1.40 dB/mm at 56 GHz (2 um) vs 0.441 (25 um). | Fab fill is applied after the RF design. | DRC auto-perforating the signal line | IME25 p.3-4 Sec. II-B/C, Table I, Figs. 9-11 | S |
| 6.20 | A larger-than-designed package-substrate pad caused an impedance mismatch dip near 60 GHz on an EML path. | Pad size is left to the substrate vendor. | One fat pad | MEL26 p.4 Sec. III.B, Fig. 7 | S |
| 6.21 | Offsetting C4 bumps away from open (unfilled) TSVs avoids topography issues but creates an impedance discontinuity. The thin-RDL fix is "more lossy", to be compensated by equalization. | Signal-integrity fixes push cost into the SerDes. | "Equalization will fix it" mantra | MRVL26 p.3 Sec. II.B, Fig. 5; p.6 Sec. III.D, Figs. 14-16 | S |

## Cluster 7. "The Measurement Passed": test, KGD and metrology blind spots in photonic packaging

KGD = known-good die.

- **Experts (inference):** about 100-300 people (photonic test engineers, process integration). Basis: blind spots appear in nearly every paper, so the gags transfer widely.
- **Why niche:** each item is a small reporting or measurement convention that changes what a number means.
- **Visual potential:** medium-high. Code:
  - a Fabry-Perot ripple interactive for a dry polymer-waveguide-to-SMF gap (AIST23);
  - an "attenuator reference" TDECQ (transmitter dispersion eye closure quaternary) demo where the eye stays clean while power vanishes (AIST24);
  - a stray-light floor demo (DNP26).

  Blender: an inspector photographing a stunt-double chip (SEI25).

| # | Pitfall | Why experts miss it (inf.) | Gag / visual (inf.) | Source + locator | Conf |
|---|---|---|---|---|---|
| 7.1 | The per-facet pass/fail limit (-0.71 dB) is an RSS of a 0.5 dB device spec and 0.5 dB measurement-system variation. Measurement noise is as large as the spec. | The noise floor sits inside the spec. | Ruler as wide as the tolerance | GF22 p.6 Sec. V.B, Fig. 5 | S (blind-spot framing I) |
| 7.2 | Foreign-material contamination on V-groove and MT ferrule surfaces is "a major source" of IL variation. Outliers were removed if FM was found or IL recovered on retest. | Dust beats alignment; the published sigma is post-cleanup. | One dust speck with an org chart | INT25b p.2-3 Sec. III | S |
| 7.3 | The HVM (high-volume manufacturing) tester matched the lab mean but had worse std (0.34 vs 0.25 dB). Causes: FAU length and MT variation within and across suppliers, and interference between adjacent MTP connectors when two FAUs were tested at once. | The fixture adds variance blamed on the product. | Two connectors elbowing | INT25b p.3 Sec. III, Table III, Fig. 5 | S |
| 7.4 | Passive alignment matched active in IL, but active was kept because the loopback power reading is an early in-process excursion detector. | The alignment method doubles as in-line test coverage. | Passive fiber with no tattletale | INT25b p.3 Sec. III, Table II | S |
| 7.5 | IL/facet is loopback/2 and includes on-chip circuitry and MTP-MT loss, so it "cannot be simply compared" with other papers' facet loss. | Headline dB/facet is not like-for-like. | Apples vs waveguides chart | INT25b p.2 Sec. III | S |
| 7.6 | The dry polymer-waveguide-to-SMF interface (no contact force, no index match) showed O-band loss ripple from multiple reflections. | Ripple gets read as wavelength-dependent loss. | Accidental Fabry-Perot certificate; FP interactive | AIST23 p.4 Sec. IV, Fig. 10 (OCR) | S |
| 7.7 | Glass-waveguide coupling numbers (0.28 dB, 1.756 dB) were measured with index-matching oil on laser-cut faces or by hand-held fiber. Return loss is unquantified, and angled placement is planned to cut back-reflection. | Lab coupling gets quoted as product coupling. | "Results valid only when oiled" | COR26 p.3 Sec. III; IZM26 p.3 Sec. II.E-F; p.4 Outlook | S |
| 7.8 | Crosstalk measurement floor: with the output fiber moved 7000 um away, about -50 dB was still detected (stray light and scattering). | The lab gets measured instead of the device. | Fiber pointed at nothing still "sees" | DNP26 p.3 Fig. 5 discussion | S |
| 7.9 | 112G PAM4 TDECQ penalty (0.1 dB) is measured against an attenuator set to the same 14.2 dB fiber-to-fiber IL, so a clean eye says nothing about the loss budget. | A beautiful eye masks link-budget failure. | Clean eye, empty power meter | AIST24 p.4-5 Sec. III.C, Fig. 7 | S (implication I) |
| 7.10 | COUPE high-speed optical test injected an external 106.25 Gb/s PAM4 signal into a loopback "due to equipment limitations". The EIC-driven path was not exercised. | The TDECQ reads like a system result. | Car "road tested" by towing | TSMC25a p.4 Sec. IV.A, Fig. 9 | S (blind spot I) |
| 7.11 | "Net 0 dB" grating insertion loss results from a 0.5 dB metal-reflector gain offsetting the stack's inherent COUPE-process loss. The 800 um support Si makes the GC-to-lens path long. | The net number hides two offsetting terms. | Accountant balancing books with a mirror | TSMC25a p.3 Sec. IV.A, Figs. 5, 7(a) | S |
| 7.12 | Buried resin lenses cannot be imaged once bonded. The SEM "proof" of lens alignment was taken on a different chip without trenches. | Unverifiable buried optics. | Inspector photographing a stunt double | SEI25 p.3 Sec. IV.A, Fig. 5(b) | S |
| 7.13 | Laser coupling loss was computed by subtracting setup, waveguide and grating losses from on-chip GC output. The wafer-level statistical variation is still unknown. | A hero-die number gets read as yield. | "Loss = measured minus everything we didn't like" | AIM23 p.5 Sec. III.C; p.6 Sec. IV (OCR) | S |
| 7.14 | Ring-modulator data came from separate "pathfinder lot" wafers. Passive and PD measurements after TSV were only at metal-1 level, yet the abstract says no deviation after TSV/BEOL (back end of line). | Rings certified by their cousins. | Family resemblance certificate | LETI23 p.6 Sec. IV.A; p.8 Sec. V | S (framing I) |
| 7.15 | Bond-oxide thickness stats exclude the outermost 10 mm ring: within 20 nm for >90% of dies, 35 nm for all. | The edge exclusion is in the caption, not the headline. | "100% compliant (edge excluded)" | IME24 p.2 Fig. 4A/B | S |
| 7.16 | RF characterized only to 65 GHz (probe limit). The 150 GHz / 448G PAM4 claims extrapolate HFSS (electromagnetic simulator) assuming stable dielectric properties. | Nyquist sits beyond the measured range. | Ruler ending at 65, "trust me" to 150 | IME26 p.3-4 Sec. IV, Fig. 5 | S |
| 7.17 | The 40-fiber edge array is dominated by fiber-to-fiber process variation. The micro-lens free-space hop varied 1.30-1.66 dB over 3 samples. The ideal 0.3 dB holds only "in a perfect condition with no process variation". The 2 dB system total is conditional on "fine process variation control", with active alignment at both attaches. | Nominal simulation is presented as the expected value. | "Perfect condition" unicorn on the spec sheet | TSMC25b p.3-4 Sec. III.A-D; p.4 Conclusion; p.2 Sec. II.A | S |
| 7.18 | TSMC iOIS: the headline 0.015 dB OE-to-interposer coupling loss is labeled "estimated", with no measurement basis given. | Three significant figures read as measured. | Three-sig-fig estimate | TSMC23 p.5 Fig. 9 caption | S (basis gap I) |

## Cluster 8. "Void-Free Is Not Bonded": wafer bonding and TSVs on photonic wafers

- **Experts (inference):** about 30-150 people. Basis: AIM, CEA-Leti, Fraunhofer IZM, A*STAR IME, Marvell and imec in the corpus. Photonic-wafer bonding with optical gap control is a small subset of the hybrid-bonding community.
- **Why niche:** the bonding oxide is simultaneously glue and an optical gap, and TSV decisions must be made at PIC tapeout.
- **Visual potential:** medium-high.
  - Blender: a "VOID-FREE" certificate shredded by a back-grinder; a 15 mm edge delamination ring.
  - Code: an evanescent-coupling-vs-oxide-gap curve. The physics is generic; only gap ranges stated in IME24 should appear.

| # | Pitfall | Why experts miss it (inf.) | Gag / visual (inf.) | Source + locator | Conf |
|---|---|---|---|---|---|
| 8.1 | Void-free bonded wafers with sufficient bond energy still delaminated during back grind, especially near the edge (about a 15 mm band). The hypothesis is long-range product topography lowering local bond energy. The fix was edge-compensating TEOS (tetraethyl orthosilicate oxide) plus CMP. | A clean SAM scan reads as pass/fail, and blanket test wafers lack topography. | "VOID-FREE" certificate shredded | AIM23 p.5 Sec. III.B, Fig. 7 (OCR "1 5 mm") | S |
| 8.2 | Voids and bond strength are separate axes: baseline was about 0.9 J/m2 and about 1.3 J/m2 with surface treatment. Weak void-free bonds fail at grind. | Two failure axes, one inspection. | Razor-blade test as annual quiz | AIM23 p.4 Sec. III.B, Table I | S |
| 8.3 | Evanescent interlayer coupling needs a gap typically <500 nm, so bonding-oxide thickness variation becomes an optical yield parameter. CMP must remove topography (oxide thicker than device height) while the gap stays thin. | Oxide is "glue" in bonding culture. | "Deposit more, polish to less" | IME24 p.1-2 Intro, Fig. 3 | S (conflict I) |
| 8.4 | Nanometer-scale within-die non-uniformity alone caused W2W (wafer-to-wafer) air voids. In C2W (chip-to-wafer), die oxide thickness was 140-180 nm depending on the die's original center, middle or edge position. | Wafer bow and roughness pass, while the die-scale failure goes unnoticed. KGD binning ignores die origin. | Die passport stamped "EDGE-BORN" | IME24 p.3 Figs. 5A-B | S |
| 8.5 | Standard 8-inch PIC fabs offer no TSV. The low-cost flow requires a through-dielectric via formed during PIC fab, so it is committed at tapeout. Via-last TSVs need keep-out zones in the photonics FEOL, and pads cannot sit on top of TSVs. | "Post-process TSV later" sounds foundry-agnostic. | "Your packaging decision was due at mask order" | MRVL26 p.1 Abstract; p.3 Sec. II.B; IZM23 p.3-4 Sec. III.B | S |
| 8.6 | Dielectric CMP after TSV fill is hard because the TSV Cu opens during polish. PVD (physical vapor deposition) seed needed an extra electro-grafted Cu layer in 12x100 um vias. Cu nail reveal after thinning varied 3-6 um. | Photonics fabs don't run high-AR (aspect ratio) seed and CMP. | Copper popping up through the floor | LETI23 p.4-5 Sec. III.B, III.E.1, Figs. 12, 14 | S |
| 8.7 | Laser and EIC attach were deliberately done at die level after singulation, to avoid de-bonding a fully populated thin wafer. | Wafer-level assembly looks cheaper on paper. | Carrier de-bond as bomb disposal | AIM23 p.3 Sec. II.B, Fig. 4 | S |
| 8.8 | The die-to-carrier temporary adhesive must survive collective cleaning, yet be much weaker than die strength and the final die-to-wafer bond. A Si carrier matches CTE (better alignment) but allows only mechanical debond. | Temporary adhesive is treated as a commodity. | Glue that must hold on and let go | IMEC24 p.3 Sec. I(ii) | S |

---

## Ranking for course selection

This ranking is my inference: pitfall density x non-obviousness x visual potential.

1. **Cluster 1 (adhesive = optical component).** Densest, most sourced and most visual. The index-cliff mode-solver interactive is a natural centerpiece, and chapters map cleanly: index cliff, cure shrink, underfill fillet, TIM flow, overflow into bumps, Na+ migration.
2. **Cluster 3 (warpage = alignment).** Strong Blender material (smile-frown-smile; dies flushed off the carrier) with measured numbers.
3. **Cluster 4 (tolerance is conserved).** Best fit for code physics interactives (Gaussian overlap, gap vs pad size, prism clipping).
4. **Cluster 2 (reflow / humidity / connector).** Best "corporate qualification" satire: bimodal uHAST with no root cause, unqualified FAU, 15-cycle reliability, wafer-level-only qualification.
5. **Cluster 6 (fix is someone else's bug).** Rich for an EIC/PIC co-design course. It is more circuit-flavored, and the ring heater versus laser energy split is a good opener.
6. **Cluster 5 (glass/polymer not inert).** The most niche community. The IOX diffusion interactive is strong.
7. **Cluster 7 (measurement passed).** Works best as a cross-course "Module 0: Reporting Integrity" or as end-of-chapter quiz gags.
8. **Cluster 8 (void-free is not bonded).** Smaller; could merge into cluster 3 or 6.

**Not supported by this corpus:** mode hops, PIC-level polarization management, laser RIN/feedback, ORL measurements, attach-induced birefringence, and measured EIC-to-ring thermal crosstalk. Source those elsewhere.
