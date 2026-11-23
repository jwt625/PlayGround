# Mining: OFS #61-#124 and OFC 2026 notes, pitfall candidates

| Field | Value |
|---|---|
| Date | 2026-11-22 |
| Corpus | `jwt625.github.io/_posts/*weekly-OFS-*.md` for OFS 61-124 (53 posts; 89, 90, 97-101, 111, 112, 116, 119 do not exist), `2026-04-04-OFC2026.md` (7645 lines), `OFC2026-notes.md` |
| Method | Keyword grep plus a full heading index (1145 headings). About 200 photonics-relevant sections read in full. The OFC file was read by a subagent, and 15 of its quotes were spot-checked by grep (all matched). |
| Mode | Read-only on the sources. This file is the only output. |

## 0. Conventions

- **Locators.** `OFS-79 L176` means the `# heading` at line 176 of `2025-12-29-weekly-OFS-79.md`. The section runs to the next `# heading`. The file map is in section 9. `OFC L5641 IMG_4158` means line 5641 of `2026-04-04-OFC2026.md`, under the heading `### IMG_4158.JPG`.
- **Image ids.** Given where useful, under `/assets/images/<year>/<folder>/`.
- **Confidence.**
  - `stated`: stated in the cited paper, slide or vendor text, as quoted in the post.
  - `his-comment`: his own comment in the post.
  - `inference`: my reasoning, not in the source. Verify before scripting.
- **OCR.** OFC slide text is OCR. Numbers are quoted as OCR'd, and some are garbled (for example "deliciated", "Tu31.6").
- **Corpus character.** Most entries are one-line quips over images. The physics is often in the image, not the text. Where the answer to one of his quizzes is not in the text, this file says so.

## 1. Cluster summary

All expert-population figures in this table are my estimates, from the number of active groups and companies I know of. None comes from a source.

| # | Cluster | Est. expert population | Pitfalls found | Fit to DevLog-001 shortlist |
|---|---|---|---|---|
| A | Lithium niobate / tantalate: the crystal fights back (pyroelectric, anisotropy, thickness, power handling, RF) | 100-400 (TFLN/TFLT device and foundry people) | 13 | Strengthens T3. Feeds T1 ch3 |
| B | Coupling, facets, back-reflection and stray light ("the garden hose") | 300-1000 (PIC packaging and coupler people). Cleaning-adjacent items removed | 14 | New candidate course. Supports T2 ch1 |
| C | Wavelength and thermal control of rings and DWDM laser arrays | 100-300 (MRM/DWDM link architects at a handful of companies) | 13 | Strengthens T5 |
| D | Simulation, measurement and reporting artifacts ("data integrity") | Broad, but each item is niche. 500+ device researchers | 15 | New candidate: a meta course or a final-assessment pool |
| E | Surfaces, cryo and hybrid quantum photonics ("surfaces are the work of the devil") | 20-100 (cryo photonics, transduction, atom-photonics) | 11 | Overlaps T6. His PhD home turf |
| F | Ultrastable reference cavities (mounting, CTE zero, coating noise) | 10-40 groups (clock and cavity people) | 7, from one dense post | New. Most niche, strong 3D visuals |
| G | CPO / ELS fleet reliability and link-level gotchas | 300-1000+ (less niche, but the numbers are concrete) | 14 | Hold, or use as a "management edition" |
| H | Polarization and mode purity (supplement to T1) | as T1 | 11 | Direct inputs for course 1 |

Gaps (searched, nothing usable):

- Photorefractive effect: no hits.
- Laser mode-hop mechanics beyond spec numbers: only "50 GHz mode hop free" (OFS-91 L177).
- Microring locking algorithms and wrong-FSR lock: not discussed.
- MPI (multipath interference), SBS (stimulated Brillouin scattering), RIN (relative intensity noise) mechanics: only one-line mentions.
- T2 laser feedback and T5 locking-algorithm details need other corpora.

## 2. Cluster A: lithium niobate / tantalate, the crystal fights back

- **Why niche:** the pitfalls come from LN's pyroelectric, ferroelectric and anisotropic nature, and from thin-film stack choices. Only people who fabricate and measure TFLN meet them. Several are his own lab experiences.
- **Visual potential:**
  - Blender: a chip on a hotplate charging up and arcing; a released LN membrane collapsing at 4 K; a disk glowing SHG only at two azimuths.
  - Code: a field-arrow animation around an x-cut disk vs crystal z; a TE1 slab-mode cutoff slider vs film thickness.

| # | Pitfall (one sentence) | Why experts miss it | Gag / visual | Source locator | Confidence |
|---|---|---|---|---|---|
| A1 | Heating or cooling an LN chip makes a pyroelectric field large enough to fry devices or crack the substrate. The usual fix is a conductive coating or wrapping the chip in aluminum foil during processing for a discharge path. | Bakes are treated as thermally benign. The field comes from dT, not from any applied bias | Hotplate bake; sparks; Gary wraps his chip in kitchen foil "per procedure" | OFS-79 L176 "Pyroelectric effect challenges in lithium niobate devices", img 20251224_183552_0/1 | his-comment |
| A2 | The pyroelectric effect is cut-dependent and usually worse on z-cut. Literature from the 1980s reports channel-waveguide MZIs drifting more on z-cut LN. | Drift gets blamed on bias electronics or DC drift mechanisms | Bias point walking with room temperature; z-cut vs x-cut side by side | OFS-79 L176. Ref linked as https://doi.org/10.1063/1.96640 (title not given in the post) | his-comment (citing paper) |
| A3 | During a cooldown to 4 K, the field from an LN chip ESD'd a device on a *different* chip. Shielding electrodes saved those devices, but IDTs on the LN chip still fried. | Nobody expects cross-chip damage from a passive neighbor in a fridge | Neighbor chip zapped across the gap during cooldown | OFS-79 L176. Ref is his tweet https://x.com/jwt0625/status/1806362394639372773 | his-comment |
| A4 | Released x-cut TFLN-on-Si devices started collapsing at 4 K. He ruled out CTE mismatch because the onset was sudden. His suspect is pyroelectric charge plus electrostatic pull-in, possibly after a wafer change to better crystal with worse discharge. Aluminum electrodes shorting the LN to the Si substrate stopped it. | A better-quality wafer making things worse is counterintuitive | Membranes snapping down one by one as the fridge cools; fix = tiny ground straps | OFS-79 L176 | his-comment (his own hypothesis, flagged "could be") |
| A5 | In an x-cut LN disk, SHG lights up only at two azimuthal regions (left and right). The quiz asks what the mode polarization is and why. He adds that TM should radiate much worse out of plane to a top-view camera. | Ring and disk designers treat the disk as isotropic | Disk glowing at two spots; the "test your NLO colleague" quiz | OFS-66 L265 "Thin Film Lithium Niobate SHG", img 20250927_185658_0, 20250927_190754_0 | his-comment. Answer not in text. My inference: the TE (in-plane radial) field is parallel to crystal z only near two azimuths, so d33 coupling is azimuth-dependent |
| A6 | With 300 nm SiN on 300 nm LN, the stack is too thick: a TE1 slab mode appears at about 470 nm total thickness, and the paper's simulation shows higher-order modes along the thickness. | Loaded-LN designs inherit thicknesses without checking slab-mode cutoff | Slider over film thickness; a ghost TE1 appears at 470 nm | OFS-118 L161 "Dual-Band TFLN Modulator and Grating Coupler". Wang2026, https://doi.org/10.1364/OPTICA.609834 | his-comment |
| A7 | The same paper's CL-TWE has 12 dB/cm microwave loss at 100 GHz. His fix is a thick BOX or a quartz substrate. | RF loss into the Si substrate is ignored when the optics look good | RF wave fading into the substrate | OFS-118 L161. Also OFS-107 L91: 12 um buried oxide to cut RF loss and RF group index from the Si substrate, Xue2026 https://doi.org/10.1364/OE.588103 | his-comment + stated |
| A8 | Which handles more optical power at similar mode area, a proton-exchange LN waveguide or an etched/diced ridge (RWG)? Per the vendor, the RWG is better. He calls this counterintuitive. | The intuition is that the buried, smoother PE guide is more robust | Two waveguides in a power-ramp race | OFS-113 L164 "Lithium Niobate Waveguide Power Handling". HC Photonics note https://uploads.strikinglycdn.com/files/5fbc03f8-dad1-4980-b5cf-e1cee8132314/PPLN%20Waveguide%20type.pdf | stated (vendor) + his-comment. Mechanism not given |
| A9 | An MZI's extinction ratio gets worse when heated. He poses this as an interview question about a TFLT modulator with stable DC bias at 120 C. | ER is treated as a static, fab-set number | ER readout degrading as the heater ramps | OFS-102 L85 "MZI Extinction Ratio Temperature Dependence". Sayem2026 https://arxiv.org/abs/2604.27285, img 20260501_053420_0 | his-comment. Answer not in text. Do not script a mechanism without a source |
| A10 | An odd grating-coupler waveguide shape on LN-on-sapphire is "very likely resist adhesion issue", and the paper does not discuss it. | Fab artifacts get published as design | Waveguide outline peeling like a sticker | OFS-83 L86 "Grating coupler efficiency on lithium niobate". Chen2025 https://arxiv.org/abs/2510.02089, img 20260122_075751_0 | his-comment |
| A11 | DFB-to-LNOI coupling loss exceeds 2 dB with a double-taper edge coupler: 1.4 dB modal with perfect alignment plus 0.6 dB from the air-LN index mismatch. Total IL is about 7 dB, and the LI curve after coupling "looks too nice". | The Fresnel term at the LN facet is forgotten in budgets | LI curve before/after; a 0.6 dB toll booth at the facet | OFS-121 L310 "Hybrid InP-LNOI Transmitter LI Curve". Bassil2026 https://doi.org/10.1063/5.0337394 | stated + his-comment |
| A12 | A WDM TFLT transmitter was tested with only two channels concurrent and at most 0 dBm laser power. He would test all 8 channels at 10 dBm or more to see whether weird TWM/FWM (three/four-wave mixing) appears. | Nonlinear crosstalk in chi(2) modulators only shows at full channel count and power | Channels lighting up one by one, then spurious tones | OFS-122 L182 "Thin-Film Lithium Tantalate WDM Modulators". Hulyal2026 https://arxiv.org/abs/2610.26674 | his-comment |
| A13 | Ellipsometry of TFLN needs optical anisotropy and Euler angles in the fit model, with angles chosen near Brewster for sensitivity. Glare corrupts color-based thickness guesses. | Users fit isotropic Cauchy models by default | Fitting software slowly drifting into a wrong thickness | OFS-85 L294 "Thin Film Thickness via Ellipsometry" | his-comment |

Also seen, weaker:

- OFS-76 L127: "lithium niobate is not created by god", so use it only where necessary, for example transduction, especially for microwave and quantum phononics (his opinion).
- OFS-124 L79: bulk-LN-era poling inspection by pouring HF on a wafer resting on paper, relying on the +z/-z etch-rate difference (anecdote).
- OFS-77 L118: his "conspiracy theory" that people put low-pass filters on modulator S21 measurements to shift the curve up. Listed under D.

## 3. Cluster B: coupling, facets, back-reflection and stray light

- **Why niche:** these are about the geometry of how light leaves a chip, and where the reflected and scattered part goes. Experts in coupler design know the peak number but not the system effects.
- **Exclusions:** connector cleaning and dust are excluded topics, so they are kept out.
- **Visual potential:** very high.
  - A 2D FDTD-style animation of an angled facet refracting the beam and the reflection re-interfering.
  - His line "picture a garden hose spraying water everywhere" (OFS-117 L399) as the chapter mascot.
  - A fiber-fuse bubble train crawling back toward the laser.

| # | Pitfall | Why experts miss it | Gag / visual | Source locator | Confidence |
|---|---|---|---|---|---|
| B1 | Transmitter output waveguides are tilted to cut back-reflection, like APC connectors. Because of refraction, the waveguide must be rotated so the *exiting* beam is horizontal, so most drawings of these chips are wrong. Some reflected light still escapes and interferes with the main beam. | People copy the 8-degree APC habit without the refraction correction | 2D toy sim (with AR layer) showing reflected beam and up-tilted far field | OFS-120 L73 "Tilted output waveguide facets on transmitter chips", img 20250127_005007_2..5, 20250127_013347_0..3 | his-comment |
| B2 | His follow-up quiz: why does an angled facet have an extra perpendicular portion instead of being angled all the way through? (QCL on Ge-on-Si.) | Not stated | Zoom on the facet's kink | OFS-120 L286 "Waveguide facet angle with perpendicular portion". Mitchell2026 https://doi.org/10.1364/OE.600656 | his-comment. Answer not in text |
| B3 | "58 dB in COMSOL is an isolator" is mocked: simulated isolation is not a working isolator. | Sim-only nonreciprocity claims are taken at face value | Simulation trophy vs a lab bench where light walks back in | OFS-121 L217 "Optical Isolator Alignment Chart". Mikhail2026 https://arxiv.org/abs/2610.17862, img 20261015_054813_0 | his-comment |
| B4 | Even single-mode PCSEL external light sources put an isolator in every TOSA (transmitter optical sub-assembly) path. | Isolator-free integration is assumed to be fine for high-power DFB/PCSEL sources | Each laser in the ELS wearing a one-way turnstile | OFC L5633-5661 IMG_4158 "PCSEL-based 8-channel DWDM ELSFP" (Sumitomo Electric / Kyoto Univ.) | stated. Feedback-sensitivity reading is inference |
| B5 | Grating-coupled detachable fiber interfaces bring limited bandwidth, large PDL (polarization-dependent loss) and high optical return loss together. | Coupler papers report peak efficiency only | Three red flags on one coupler | OFC L4377-4404 IMG_4120 "Vertical Detachable Couplers" (GlobalFoundries talk, OCR "OFC 2026 Tu31.6") | stated |
| B6 | You can optimize a grating coupler for peak efficiency at exactly 1310 nm, one polarization, one fiber angle, on one wafer. The system cares about bandwidth, wafer variation, polarization, back-reflection, fiber height, angular tolerance, process bias, testability and 16-fiber attach. | The publishing metric differs from the product metric | Trophy coupler vs production line | OFS-117 L341 "Dario Amodei Parody on Fiber to Chip Coupling" (GPT-6 parody text he posted) | his-comment (parody he posted and endorsed) |
| B7 | Lateral and angular alignment tolerance trade off against each other. When he asked, the answer was that industry has settled on a 30-40 um beam size. The slide's overlap plot assumes no angular offset. | Expanded-beam designs quote lateral tolerance only | Slider: beam waist vs lateral and angular tolerance curves | OFC L4162/L4710 (his comment), L4311-4342 IMG_4119 "Mode Expansion and Collimation" | his-comment + stated. The Gaussian tradeoff is inference |
| B8 | Simulated vs measured beam from an inverse-designed "universal" grating emitter (tantala, visible) show a "brutal difference". | Inverse design is trusted to the fab | Sim beam vs measured beam side by side | OFS-94 L64. Spektor2023 https://doi.org/10.1364/OPTICA.486747, img 20260407_033820_0 | his-comment |
| B9 | In cryo photonics, low grating-coupler efficiency means light "spilling everywhere" and heating the fridge. | Coupler loss is treated as a dB budget item, not a heat load | Photons spraying onto the mixing chamber; the thermometer rises | OFS-108 L110 "Grating coupler efficiency in cryogenic photonics". Bisang2026 https://pubs.acs.org/doi/10.1021/acsphotonics.6c00869 | his-comment |
| B10 | The fiber fuse: about 3 W is enough to start a catastrophic self-propelled damage front that leaves periodic hollow voids and travels toward the source at about 0.5 m/s. It has been seen in up to 1.5 km of SMF. | High-power ELS and CPO laser people rarely meet fiber-damage physics. Fuse papers live in fiber-sensor literature | Bubble train racing back up the fiber toward the laser | OFS-72 L141 "Fiber fuse effect micro-cavity fabrication". Antunes2014 https://doi.org/10.1109/LPT.2013.2288930, Kashyap1988 https://doi.org/10.1049/el:19880032 | stated (in papers as he summarized) |
| B11 | Index-matching epoxy near a TIR (total internal reflection) mirror surface is a risk: the glass coupler showed epoxy delamination and imperfect coverage, possibly from fear of epoxy reaching the TIR surface. | TIR only works with an air (low-index) interface behind it | Epoxy creeping onto the mirror; light leaks out | OFS-117 L287 "Detachable Edge-Coupling Glass Coupler". Psaila2024 https://doi.org/10.1109/JLT.2024.3386033 | his-comment (speculative "might be") |
| B12 | The ELSFP (external laser small form-factor pluggable) has known pain points: IL to the optical engine can exceed 3 dB, PMF (polarization-maintaining fiber) volume is too low and too expensive, and high power is "too easy to burn shit". | Laser power budgets assume datasheet IL | Laser power draining through a leaky pipe | OFS-121 L236 "ELSFP Pain Points". Also OFS-122 L79: "ELS IL, I've heard 3 dB from a few different places" | his-comment |
| B13 | Large weakly etched gratings on Si/SiN can yield one chip per 4-inch wafer. Coupling and heterogeneous integration stay the bottleneck. | Area cost is invisible in single-device papers | Wafer map with one green die | OFS-122 L66. Isichenko2026 https://arxiv.org/abs/2610.17918 | his-comment |
| B14 | Fiber-tip tip/tilt alignment from SMF power gives one number only. Telescopes surround the SMF with six MMFs plus a 3D-printed microlens ring for six feedback signals. | Single-scalar feedback alignment is underdetermined | Six satellite fibers lighting up as the spot drifts | OFS-110 L57. Hottinger2019 https://arxiv.org/abs/1912.09537 | stated |

## 4. Cluster C: wavelength and thermal control of rings and DWDM laser arrays

- **Why niche:** only a few teams (MRM CPO, DWDM ELS) carry the full chain. Ring-to-laser grid alignment, heater budgets and hot-ASIC proximity are system-level and invisible at device level.
- **Visual potential:**
  - Code: a comb of laser lines and a comb of ring resonances that drift at different pm/K; channel-spacing crosstalk eye diagrams.
  - Blender: a DFB array next to a GPU die whose temperature swings 40 C.

| # | Pitfall | Why experts miss it | Gag / visual | Source locator | Confidence |
|---|---|---|---|---|---|
| C1 | NVIDIA's ISSCC clock-forwarded DWDM chip budgets about 0.57/0.49 pJ/b for thermal tuning, out of 2.59/2.51 pJ/b total, sized to cover "3 of ring and laser variation" (likely 3 sigma). The thermal figure was measured with a bench laser. | Device papers quote junction efficiency and omit heater energy | Energy pie with a heater slice about 20% | OFC L5247-5354 IMG_4147 "CLK-FWD DWDM Test Chip Reported at ISSCC" | stated. The 20% share and "3 sigma" are inference |
| C2 | DWDM DFB-array spacing controllers lock only *relative* wavelengths, not absolute. Progress was from 200 +- 15 GHz bare array (locked to reference ring FSR +- 4 GHz) to +- 12.5 GHz to +- 3.5 GHz. He asks why no power curves are shown when temperatures are tuned. | Relative grid lock is mistaken for grid compliance | A comb that keeps its spacing while sliding off the grid | OFS-120 L294 "DFB laser array wavelength spacing controller". Levy2026 https://doi.org/10.1109/JLT.2026.3739658, img 20261008_051856_0 | stated + his-comment |
| C3 | You do not want a DFB laser close to GPU dies that can swing 40 C. He agrees with the link-flap concern. | Laser placement is chosen for density, not thermal stability | Laser grid shifting as the GPU workload spikes | OFS-122 L79 "Nvidia CPO Roadmap and Laser Sourcing" | his-comment |
| C4 | With 70 GHz FWHM MRMs at 100 GBd NRZ, crosstalk depends on channel spacing (150/200/400 GHz) and power. "You better interleave the channels." His BER also needed multi-step sampling, not a naive Gaussian fit. | Adjacent-channel beat crosstalk in ring banks is underestimated | Victim channel eye closing as the aggressor walks closer | OFS-114 L78 "Crosstalk Analysis in 70 GHz MRM Arrays", img 20260822_014812_0 | his-comment (his own GPT-assisted sim) |
| C5 | The study's assumption of a baseline MRM temperature around 55 C under stacked GPU/memory is "very optimistic". Measure it with thermoreflectance imaging. | Thermal models are simplified | Thermal camera reveals a much hotter ring | OFS-114 L280 "Thermal Tuning Overhead in Optical Interconnects". Yoon2026 https://arxiv.org/abs/2608.24637 | his-comment |
| C6 | Si and SiN MZIs drift at 80 pm/K and 18 pm/K. Combining both (opposite-sign design) gives -0.3 pm/K. | Athermal design via mixed thermo-optic coefficients is less known than heaters | Two seesaws balancing | OFS-96 L187 "Temperature Insensitive Wavelength Detector". Sang2025 https://doi.org/10.1364/OE.574680 | stated |
| C7 | Chaotic-response (speckle/microdisk) spectrometers have chaotic temperature behavior too, with a 50 pm/K shift: TEC or recalibrate. | Calibration is assumed portable across temperature | Reconstructed spectrum scrambling when the room warms | OFS-102 L43 "Chaotic Response Microdisk Spectrometer". Chen2026 https://pubs.acs.org/doi/10.1021/acsphotonics.5c02990 | stated + his-comment |
| C8 | A silicon pulse shaper with six pairs of racetrack resonators: the authors admit results could depend on the "phase of the moon". | Multi-ring circuits multiply drift sources | Moon phase dial on the lab wall | OFS-120 L122. Su2026 https://doi.org/10.1364/OE.592918 | his-comment (paraphrasing a paper admission; exact wording in the image only) |
| C9 | Photonic neural-network chips need stringent temperature control because "too many physical things [are] drifting on the chip". | Demo stability is not product stability | Weights melting as the room warms | OFS-83 L190. Xiang2026 https://arxiv.org/abs/2601.15989 | his-comment |
| C10 | Uncooled, transfer-printed O-band QD FP lasers shift from 1300 nm to 1330 nm between 20 C and 70 C. Coupling efficiency also moves, via index, mode size and position. | CWDM-agnostic thinking on FP sources | Spectrum walking 30 nm | OFC L5132-5137 IMG_4143 (UCSB poster, Diya Hu et al.) | stated. About 0.6 nm/K is inference |
| C11 | Competing vendors grade MRR thermal stability "Bad" or "Very Bad", needing "deliciated [sic] thermal control". The slide notes the specs are estimated. | Vendor bias | Ring wearing a fever thermometer | OFC L6154-6176 IMG_4173, L6533-6610 IMG_4180 | stated (vendor-biased) |
| C12 | A 1024x1024 AWGR-based OCS needs explicitly athermal 64x64 AWGRs (256 of them). | Passive mux drift is forgotten until scale | 256 AWGRs sweating | OFC L1687-1817 IMG_4021/4025, Paper M4F.1 (UC Davis) | stated. Drift reasoning is inference |
| C13 | Ring resonance placement accounting and heater budgeting is a first-class design task (in a MRM co-design paper). "2nd year PhD loves rings; 2nd year system architect hates rings." | Ring placement is treated as layout, not budget | Accounting ledger of resonances | OFS-120 L160. Huang2026 https://arxiv.org/abs/2609.36690. OFS-102 L250 | stated + his-comment |

## 5. Cluster D: simulation, measurement and reporting artifacts

- **Why niche:** each item is a specific way that numbers and figures mislead in photonics papers. Experts reading the paper miss it because it looks normal.
- **Visual potential:**
  - A "spot the artifact" quiz UI (mirrored chip photo, boundary mode, S21 filtering).
  - A code-generated mode solver whose "1st-order mode" is a box mode.

| # | Pitfall | Why experts miss it | Gag / visual | Source locator | Confidence |
|---|---|---|---|---|---|
| D1 | A thermal/mode simulation of a DFB laser array appears to report a mode formed by the simulation-domain boundary as the "1st order mode". He notes it is a 2D eigenmode solve, "a lot of stuff can do it". | Eigen solvers return box modes that look physical | Mode solver sliders; the mode hugs the PML / box edge | OFS-78 L140 "Thermal simulation artifact in DFB laser array". Yang2025 https://doi.org/10.1364/OE.580766, img 20251218_045259_0, 20251218_045529_0/1 | his-comment |
| D2 | His "conspiracy theory": some groups put low-pass filters on modulator or transceiver S21 measurements to shift the whole curve up. | Fixture de-embedding choices are not reported | S21 curve lifted by an invisible hand | OFS-77 L118 "BEOL integration of TFLN on active silicon photonics". Wu2025 https://arxiv.org/abs/2512.07196 | his-comment (explicitly a speculation) |
| D3 | A vendor's chip photo was mirrored, twice, so it looked like 12 lasers fanned into two waveguide sets. He caught it via a perfectly symmetric liquid residue. | Symmetric layouts hide image manipulation | Residue blob that is suspiciously symmetric | OFS-117 L383 "Mirrored Chip Photo in Dream Photonics" | his-comment |
| D4 | JMP-style one-click fits make it easy to stop asking which parameters to extract. | Tooling hides physics | Drag, drop, "bada bing" fit through garbage | OFS-106 L67. OFS-104 L202 | his-comment |
| D5 | A BER from a naive Gaussian fit to a simulated eye is not trustworthy. It needed multi-step sampling and threshold optimization. | Gaussian Q-factor shortcuts | BER meter jumping orders of magnitude | OFS-114 L78 | his-comment |
| D6 | PDL by polarization scrambling (a 5 s triangular sweep, 50,000 Stokes samples, max/min) is only as good as the Poincare-sphere coverage. | Coverage is not reported | Sparse dots on a Poincare sphere | OFC L588-612 IMG_3987 | stated method. The coverage caveat is inference |
| D7 | "SiPh optics are most reliable" (Alibaba RMA data: SiPho 400G DR4 0.17% vs 0.23%) can be bookkeeping, since the SiPh PIC carries no laser. | Laser failures get attributed elsewhere | Reliability trophy handed to a chip with no laser on it | OFC L6979-6993 IMG_4194, his comment L7001 | stated + his-comment |
| D8 | Root-cause statistics for optical failures are filtered to "failures restored by a single maintenance intervention". Reseat-fixed and NTF (no trouble found) categories dominate. | Filtering choices shape conclusions | Pie chart with a hidden slice | OFC L7053-7076 IMG_4196 | stated. Bias point is inference |
| D9 | Very low-loss nitride claims often "cheat" with thin waveguides where the mode is mostly in oxide. A cited "high density SiN at 0.1 dB/cm" used a 2 um x 0.8 um waveguide with 25 um bend radius. | Density and loss are quoted separately | Waveguide that is mostly oxide | OFS-114 L88 "Co-Packaged Optics Review Paper Critique". Kim2026 https://doi.org/10.1038/s41928-026-01681-6 | his-comment |
| D10 | Multi-core-fiber interface IL was measured between a single-core fiber and one core of the MCF, in C band only. | IL definitions vary | Fine print magnifier | OFS-108 L83. Radic2026 https://doi.org/10.1364/OE.601126 | stated |
| D11 | Dynamic range: typical power meters and APDs give 60-70 dB, and SNSPDs a similar number. A good attenuator adds 40-60 dB. Pulling a fiber slightly out of its connector is an effective attenuator if you measure it; he used it for about 90 dB measurements. | Detector range limits fake an extinction floor | Extinction "floor" that is the detector | OFS-87 L79 "Ultra-high dynamic range fiber Bragg grating filter". Field2024 https://arxiv.org/abs/2412.00981 | his-comment |
| D12 | "Normalized" plots in a space-PIC review do not say what they are normalized to. | Normalization hides range | Axis label "normalized (to what?)" | OFS-113 L126. Terrasanta2025 https://doi.org/10.1002/sat.1552 | his-comment |
| D13 | Benchmarks compile metrics that are not achieved simultaneously on the same chip. | Leaderboard aggregation | Frankenstein chip assembled from best numbers | OFS-124 L98 (his own Photonics Atlas benchmark caveat) | his-comment |
| D14 | Component papers omit the non-trivial step ("The chip was then diced and polished to expose the taper input facet"), and some hide facet coupling loss. | The method section is where the value is | Redacted methods section | OFS-113 L300 "Spot Size Converter Papers and Cognitive Effort" | his-comment |
| D15 | A detachable-coupler paper says only that the DLW chip is coupled to the PIC "through high precision assembly process such as thermal compression bonding (TCB)". "What do you mean 'such as'?" | Vague process language | Highlighted "such as" | OFS-117 L287. Zhang2026 https://doi.org/10.1109/ECTC51846.2026.00304 | his-comment |

## 6. Cluster E: surfaces, cryo and hybrid quantum photonics

- **Why niche:** cryogenic transduction, atom-photonics and phononic-quantum groups (a few dozen labs) meet effects that room-temperature photonics never sees. This is his PhD domain.
- **Visual potential:**
  - Blender: Rb atoms raining onto a waveguide; Ga ions implanting; a phononic crystal with rounded corners.
  - Code: Q vs power with TLS-like saturation (the sources do not give a model, so use it only as a generic visual).

| # | Pitfall | Why experts miss it | Gag / visual | Source locator | Confidence |
|---|---|---|---|---|---|
| E1 | Attaching an Rb vapor cell to a PIC deposits Rb onto the exposed waveguide and "pretty much kills light propagation". The exposed length had to shrink to about 100 um. The follow-up fixed it with a desorption laser through the waveguide, and used the loss as a vapor-density monitor. | Cell activation is treated as separate from optics | Atoms snowing onto the waveguide; a laser "defrosts" it | OFS-79 L159 "Rb vapor cell integration with photonic waveguides". Shrestha2025 https://arxiv.org/abs/2512.19852 | stated + his-comment |
| E2 | "Surfaces are the work of the devil": interface piezoelectricity on crystalline silicon makes superconducting devices lossy. Moving boundaries away from the field maximum reduces it. | Silicon is assumed non-piezoelectric | Silicon surface quietly buzzing | OFS-77 L61, https://arxiv.org/abs/2409.10626. OFS-103 L87, Zhou2026 https://arxiv.org/abs/2605.15554 | stated + his-comment |
| E3 | The oxide-silicon interface in SOI traps charges and causes microwave loss, so photonics-plus-RF on SOI pays in Q. People use sapphire or high-resistivity Si instead. | Photonics people pick SOI by default | Charges hopping in the BOX interface | OFS-113 L60. Messelot2026 https://arxiv.org/abs/2607.29154 | stated + his-comment |
| E4 | Ga+ FIB "killed quite a few of my devices likely because of doping/contamination". Ga+ gives more damage than Xe+ at the same voltage. | FIB trims look clean in SEM | Ga ions as tiny saboteurs | OFS-88 L179. Kelley2013 https://doi.org/10.1017/S1431927613006302, Balduini2026 https://doi.org/10.1063/5.0295819 | his-comment + stated |
| E5 | A final wet etch rounds "supposedly straight" corners in a MHz phononic-crystal shield. | Layout is not geometry | Square holes melting into rounded ones | OFS-76 L215. Matsuyama2025 https://doi.org/10.1063/5.0300550 | stated (quiz answer given) |
| E6 | Smart-cut thin-film Nd:YAG cracks from CTE mismatch during the exfoliation anneal, and gets "really messed up" at higher anneal temperatures. | Smart-cut success in LN does not transfer | Film shattering in the furnace | OFS-117 L231. Chen2026 https://doi.org/10.1016/j.matdes.2026.115969 | stated + his-comment |
| E7 | CTE matching for SiN on sapphire must hold across the temperature range, because CTE itself depends on temperature. Otherwise the film cracks. | Single-number CTE matching | CTE curves crossing at one temperature only | OFS-121 L208. Ma2026 https://arxiv.org/abs/2610.15754 | his-comment |
| E8 | Below-bandgap UV absorption from defects: LN (3.8 eV, 326 nm) and SiN (4.6 eV, 270 nm) absorb well before the bandgap, which is why alumina (> 6 eV) is used, with sub 1 dB/cm at 360 nm. | Bandgap is taken as the transparency edge | Absorption edge creeping in early | OFS-92 L48. Castillo2026 https://doi.org/10.1364/OPTICAQ.582412 | stated + his-comment |
| E9 | LN is a poor choice for quantum phononic circuits at microwave frequencies unless needed for transduction. Metal mirrors are "crap" for confined acoustic waves. | Reuse of LN process steps | LN wearing a "not created by god" badge | OFS-76 L127. Wang2025 https://arxiv.org/abs/2512.04953, Hitchcock2025 https://arxiv.org/abs/2502.16426 | his-comment (personal opinion) |
| E10 | Pyroelectric collapse at 4 K and cross-chip ESD on cooldown (A3, A4) belong here too. | see A3/A4 | see A3/A4 | OFS-79 L176 | his-comment |
| E11 | He used an EMI room mainly to detect weak optical sidebands without RF crosstalk: detectors, amplifiers and spectrum analyzer inside, fibers run in from the setup outside. | RF pickup mimics optical sidebands | Fake sideband that disappears when the door closes | OFS-62 L254 "Photonic links for distributed RF antenna arrays" | his-comment (anecdote). That pickup mimics sidebands is inference |

## 7. Cluster F: ultrastable reference cavities

- **Source:** one dense post. Every item below cites refs that he linked.
- **Why niche:** order 10-40 groups worldwide build 1e-17-class cavities (my estimate).
- **Visual potential:** excellent in Blender:
  - a double-cone spacer sagging under gravity;
  - 3-fold-symmetric silicon mounts;
  - a tilted mirror under longitudinal acceleration;
  - a CTE-vs-T curve with the zero crossing marked at 124 K and 17 K.

| # | Pitfall | Why experts miss it | Gag / visual | Source locator | Confidence |
|---|---|---|---|---|---|
| F1 | Away from the CTE zero crossing, a cavity may need sub-uK temperature control. Even at zero CTE it needs few-mK control. This is why silicon cavities run at 124 K and 17 K. | Cryo is assumed to fix everything | Temperature dial with a microkelvin wobble | OFS-84 L55 "Engineering ultrastable optical cavities for 1e-18 accuracy". Refs as linked, including Kessler2012 https://doi.org/10.1038/nphoton.2012.217 | his-comment summarizing refs |
| F2 | 4 K closed-cycle cryostats fluctuate about 20 mK and need heavy thermal damping. One paper claims "we now require only mK level control of the room temperature enclosure". | The cryostat is not a thermal bath | Pulse-tube heartbeat | OFS-84 L55. Zhang2017 and Robinson2019 https://doi.org/10.1364/OPTICA.6.000240 | his-comment summarizing refs. Note: the Zhang2017 DOI as linked in the post duplicates the Numata2004 DOI. Verify |
| F3 | The mechanical, optical and crystal axes must be aligned, or longitudinal acceleration tilts the mirrors and changes the length. The optical axis should be the stiffest crystal axis, since silicon's Young's modulus is anisotropic. | Isotropic mechanics assumed | Mirror nodding under acceleration | OFS-84 L55. Kessler2012, Hopcroft2010 https://doi.org/10.1109/JMEMS.2009.2039697 | his-comment summarizing refs |
| F4 | A double-cone spacer sags less under transverse acceleration. | Cylinders look fine | Cylinder vs double cone sag exaggerated | OFS-84 L55. Millo2009 (DOI as linked duplicates Numata2004; verify) | his-comment summarizing refs |
| F5 | Use PEEK instead of PTFE for the mount supports, and push the lowest mechanical resonance high. The support has 3-fold symmetry to match silicon's crystal symmetry. | Mount material is treated as trivial | PTFE mount wobbling like jelly | OFS-84 L55. Matei2016 https://doi.org/10.1088/1742-6596/723/1/012031 | his-comment summarizing refs |
| F6 | Amorphous dielectric mirror coatings set a thermal-noise floor. Crystalline GaAs/AlGaAs coatings reach 2.5e-17. | The coating is the last thing considered | Mirror coating jittering atomically | OFS-84 L55. Lee2026 https://doi.org/10.1103/zgrm-cjbb, Harry2002 http://doi.org/10.1088/0264-9381/19/5/305. OFS-84 L177, Penn2019 https://doi.org/10.1364/JOSAB.36.000C15 | stated |
| F7 | Temperature gradients matter too, which is why silicon (high thermal conductivity) beats glass. | Only uniform temperature is considered | Heat-map gradient across the spacer | OFS-84 L55 | his-comment |

## 8. Cluster G: CPO / ELS fleet reliability and link-level gotchas

- **Why less niche:** CPO is a large community, but these specific numbers are not widely internalized.
- **Overlap:** dust and contamination items overlap the excluded cleaning topic, so they are omitted here (OFC IMG_4078, IMG_4085 dust test, IMG_4196 contamination).
- **Visual potential:** fleet-scale animation (100k nodes, MTBF clock), and a link-flap heartbeat monitor.

| # | Pitfall | Why experts miss it | Gag / visual | Source locator | Confidence |
|---|---|---|---|---|---|
| G1 | 200 FIT per node gives a 50 h MTBF at 100k nodes and 5 h at 1M nodes: "model may never finish training". The slide's formula is written "Availability = MTBF/(MTTF+MTTR)". | Per-module FIT feels tiny | Countdown clock shrinking as nodes are added | OFC L5391-5396 IMG_4149 (Google, Zuowei Shen) | stated. Formula inconsistency (MTBF vs MTTF) is inference, possibly an OCR or slide error |
| G2 | A BER of 1e-7 to 1e-8 costs about 40% of training throughput: RDMA Go-back-N, possible 4.3 s timeouts, and at >15 s tasks cannot be recovered. | Telecom BER intuition | Throughput gauge collapsing at "acceptable" BER | OFC L6068-6127 IMG_4166, citing "Y. Liu, et al. OFC 2025, paper Th1A.3" | stated |
| G3 | A single link down costs about 6.25% for a 256-GPU training task but about 30% for a 32-GPU inference task. Flapping and CRC errors silently degrade jobs. | Link-down is thought of as binary | Heartbeat monitor flapping | OFC L7017-7035 IMG_4195 (Alibaba Cloud) | stated |
| G4 | XPO-LPO first results are 1e-8 vs 1e-10 for the retimed (FRO) version. | Linear drive margin | Two eyes, one squinting | OFC L3114-3125 IMG_4067 (Arista) | stated |
| G5 | Laser FIT claims conflict by vendor and temperature (Arista "<1 FIT at 45C per laser"; Lumentum VCSEL sparing; Broadcom "<0.1 FIT"). He flags the vendor biases. | FIT without an acceleration model is not comparable | Three salespeople with three numbers | OFC L3140-3151 IMG_4068, L3906-3914 IMG_4088, L3772 IMG_4085. His comments L3155, L3921 | stated + his-comment |
| G6 | CPO assembly needs known-good-die at every step: "so many bad dies, bad lasers, bad optical engines". | Multiplicative yield | One bad laser scrapping a whole switch package | OFC L3686-3734 IMG_4084, his comment L3736. OFC L4619 IMG_4125 | stated + his-comment |
| G7 | Delamination in a CPO assembly: the "nice 3D surface plot" is the actual delamination height profile. | Warpage maps look like decoration | Package surface heaving | OFS-122 L91. Baker2025 https://doi.org/10.1109/ECTC51687.2025.00061, img 20261019_060910_0 | his-comment |
| G8 | 0.4 dB connector IL looks great until the insertion losses of a full data-center channel add up. | Per-connector thinking | IL ledger stacking up | OFS-107 L208. Stone2026 https://www.ieee802.org/3/400GPL/public/260630/stone_400GPL_01a_260630.pdf | his-comment |
| G9 | An optical backplane must bring IL down and yield up so individual trays can be serviced without breaking other good connections. NVL72 gold fingers are finicky even electrically. The OCI-MSA wants overall fiber-link IL < 2.5 dB. | Rack makers are not used to optical tolerances | Pulling one tray, five others go dark | OFS-96 L150 "Optical Backplane Insertion Loss". OFC L3447 his comment | his-comment |
| G10 | Permanent V-groove fiber attach is "not favored for high fiber count CPO: N > 100". | It works for pluggables | V-groove line overflowing | OFC L4201-4248 IMG_4117 (GF) | stated |
| G11 | OTDM serializers need path-length drift and pulse overlap held to sub-10s of fs, plus optical phase, branch skew and pulse shape. | Moving SerDes (serializer/deserializer) burden into optics moves it into the picosecond domain | Pulses missing each other by femtoseconds | OFS-103 L156 "Optical Time-Division Multiplexing" | his-comment |
| G12 | Stealth dicing by shooting along lanes and pulling the die apart "feels risky". The chips had no dummy pads. | Dicing is an afterthought | Wafer pulled apart like chocolate | OFS-118 L85. Akazawa2026 https://arxiv.org/abs/2609.24611 | his-comment |
| G13 | VCSIL (VCSEL transfer-printed onto a PIC grating coupler) optical degradation starts at the top p-metal contact and has two phases. | Transfer printing adds new failure sites | Dark spot creeping from the contact | OFS-104 L277. Zenari2023 https://doi.org/10.1109/JQE.2023.3283514 | stated |
| G14 | Filamentation in broad-area lasers has three nonlinear mechanisms: gain-saturation index change via the linewidth-enhancement factor, thermal self-focusing, and self-defocusing from intensity-dependent cladding index. | High-power pump people assume a smooth near field | Beam breaking into filaments | OFS-96 L198. Marciante, DOI as linked https://doi.org/10.1109/3.488831 (post labels it 2026; DOI format suggests 1996, verify) | stated |

## 9. Cluster H: polarization and mode purity (supplement to T1)

Direct inputs for DevLog-001 course 1. These support existing chapters rather than adding a new course.

| # | Pitfall | Fits T1 ch | Source locator | Confidence |
|---|---|---|---|---|
| H1 | x-cut LN disk SHG lights only at two azimuths. The polarization depends on position relative to crystal z (see A5) | ch3 crystal orientation | OFS-66 L265 | his-comment + inference |
| H2 | A grating coupler gives polarization-selective feedback that makes the TE mode lase first in transfer-printed VCSELs | ch5 / laser crossover | OFS-104 L277 | stated |
| H3 | "When you want TM polarization but you don't want to make a polarization rotator" (on-chip PPLN QFC; the paper includes a non-adiabatic taper design) | ch1 | OFS-64 L101. Wang2025 https://arxiv.org/abs/2509.03869 | his-comment. Image-dependent, check the image before using |
| H4 | Grating-coupled interfaces bring large PDL (B5). The GlassBridge coupler specs "< 1.5 dB/facet TE" with "< 0.3 dB PDL" (TE-only loss quoted) | ch6 | OFC L4377 IMG_4120. OFC L4266-4290 IMG_4118 | stated. TE-only reading is inference |
| H5 | PDL by scrambler sweep depends on sphere coverage (D6). Measured all-port PDL <= 0.44 dB | ch6 | OFC L588-612 IMG_3987 | stated + inference |
| H6 | Too-thick SiN-on-LN supports a TE1 slab mode (about 470 nm cutoff) | ch1 | OFS-118 L161 | his-comment |
| H7 | Straight directional couplers (no bends) minimize coupling to higher-order modes | ch2 | OFS-95 L229. Wang2026 https://arxiv.org/abs/2604.12017 | stated |
| H8 | Multimode spiral waveguides trade loss for mode purity. They need a good mode filter and stray-light management | ch4 | OFS-122 L174. Hashemi2026 https://ieeexplore.ieee.org/document/11652868 | his-comment |
| H9 | Bulk LN for AR glasses: makers "eat the birefringence" rather than use SiC | ch3 aside | OFS-122 L182 | his-comment |
| H10 | ELSFP lacks PMF volume, and PMF is too expensive (B12). Laser complexity (polarizations, modes) raises sensitivity, and PM fiber price constrains choices | ch5 | OFS-121 L236. OFS-123 L94 | his-comment |
| H11 | Silicon's high index contrast enables dense OCS but makes "Polarization management" a listed commercialization requirement | ch0 framing | OFC L428-443 IMG_3983 | stated + inference |

## 10. Ranking for course building (my judgment)

1. **A (LN crystal fights back).** His own lab stories carry strong authority (A1-A4). Visuals are distinctive (sparks, foil wrap, collapsing membranes, two-spot SHG glow). About 13 sourced items, enough for 6 chapters. Merges well with T3 (bias drift).
2. **B (garden hose: coupling and back-reflection).** 14 items. The facet-refraction animation (B1) and the fiber fuse (B10) are signature visuals. Avoid cleaning content.
3. **D (data integrity).** Best as a final-assessment pool or a "Research Integrity" course: spot the boundary mode, the mirrored photo, the S21 filter.
4. **F (ultrastable cavity).** The most niche, with excellent Blender potential. Only 7 items from one post, so it needs a second source for chapters 5-7.
5. **C (rings / DWDM thermal).** Concrete numbers, but overlaps T5 and the Gary film's CPO scene.
6. **E (surfaces / cryo).** Strong for a T6-style course. Several items are opinion.
7. **G (fleet reliability).** Good numbers, less niche. Could be a "Management Edition" of the training.

## 11. File map (OFS posts cited)

| OFS | File |
|---|---|
| 62 | 2025-09-01-weekly-OFS-62.md |
| 64 | 2025-09-15-weekly-OFS-64.md |
| 66 | 2025-09-28-weekly-OFS-66.md |
| 72 | 2025-11-09-weekly-OFS-72.md |
| 76 | 2025-12-07-weekly-OFS-76.md |
| 77 | 2025-12-15-weekly-OFS-77.md |
| 78 | 2025-12-22-weekly-OFS-78.md |
| 79 | 2025-12-29-weekly-OFS-79.md |
| 83 | 2026-01-25-weekly-OFS-83.md |
| 84 | 2026-02-02-weekly-OFS-84.md |
| 85 | 2026-02-08-weekly-OFS-85.md |
| 87 | 2026-02-22-weekly-OFS-87.md |
| 88 | 2026-03-02-weekly-OFS-88.md |
| 92 | 2026-03-30-weekly-OFS-92.md |
| 94 | 2026-04-13-weekly-OFS-94.md |
| 95 | 2026-04-20-weekly-OFS-95.md |
| 96 | 2026-04-27-weekly-OFS-96.md |
| 102 | 2026-06-07-weekly-OFS-102.md |
| 103 | 2026-06-14-weekly-OFS-103.md |
| 104 | 2026-06-22-weekly-OFS-104.md |
| 106 | 2026-07-07-weekly-OFS-106.md |
| 107 | 2026-07-15-weekly-OFS-107.md |
| 108 | 2026-07-21-weekly-OFS-108.md |
| 110 | 2026-08-03-weekly-OFS-110.md |
| 113 | 2026-08-23-weekly-OFS-113.md |
| 114 | 2026-08-31-weekly-OFS-114.md |
| 117 | 2026-09-21-weekly-OFS-117.md |
| 118 | 2026-09-27-weekly-OFS-118.md |
| 120 | 2026-10-14-weekly-OFS-120.md |
| 121 | 2026-10-20-weekly-OFS-121.md |
| 122 | 2026-10-26-weekly-OFS-122.md |
| 123 | 2026-11-04-weekly-OFS-123.md |
| 124 | 2026-11-09-weekly-OFS-124.md |

## 12. Verify-before-script list

- A9 and B2: quiz answers are not in the text. Do not script a mechanism without a source.
- A5: the azimuthal-SHG mechanism is my inference.
- F2 and F4: the DOIs as linked for Zhang2017 and Millo2009 duplicate Numata2004's DOI (10.1103/PhysRevLett.93.250602). Find the correct DOIs.
- G14: the post labels the paper Marciante2026, but the DOI 10.1109/3.488831 suggests an IEEE JQE paper from about 1996. Verify the year.
- C8: the "phase of the moon" wording may be his paraphrase. Check image text before quoting the authors.
- OFC items: OCR numbers. Re-read the slide images before putting any number on screen.
