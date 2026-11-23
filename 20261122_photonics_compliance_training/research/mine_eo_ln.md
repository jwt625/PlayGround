# Mining: EO modulator and lithium niobate corpora for compliance-training pitfalls

Date: 2026-11-22. Read-only pass over two local corpora:

- `eo-modulator-atlas` (ATLAS): `references/<id>/text.md` (PDF text extracts; locator `p.N` = the `<!-- page N -->` marker, i.e. PDF page index, not journal page; extracts are local-only, not tracked in git), `data/devices.csv`, `data/schema/devices.schema.yaml` (conventions (a)-(gg)), `DevLog/`.
- `lithium-niobate` (LNR): `content/chapters/04-quirks/manuscript.md` (section-level locators), `content/chapters/06-devices/manuscript.md`, `content/chapters/05-making/manuscript.md`, `research/R06`, `R08`, `R14`.

Confidence codes: **S** = stated in the cited primary source text (read here); **C** = curation note (ATLAS DevLog / schema / devices.csv notes, or LNR chapter text with its own status code: P, P-comp, reported = abstract-level, derived, practitioner report); **I** = my inference (not in any source). Counts marked "my count" were computed by me from `eo-modulator-atlas/data/devices.csv` on 2026-11-22 (521 device rows).

Source keys in the tables resolve in the bibliography at the end (title, year, DOI/arXiv, repo path).

Expert-population estimates are **I** throughout: anchored on ATLAS co-author counts (`data/paper_authors.csv` joined to `data/devices.csv`; co-author counts include students and coauthors and miss non-publishing industry engineers), then widened by judgment. Treat as order-of-magnitude.

---

## Cluster ranking (my judgment, I)

| Rank | Cluster | Pitfall count | Why it ranks |
|---|---|---|---|
| 1 | C1 DC is not RF: EO relaxation, bias drift, hysteresis | 11 | Best-sourced, real numbers, strong physical story, built-in RC interactive spec exists in LNR |
| 2 | C3 Metrics that mean different things (Vpi, bandwidth, IL, ER) | 16 | ATLAS curation is literally a compliance manual; many counts; very gag-friendly |
| 3 | C2 Light and heat that write back: photorefraction, pyroelectricity | 12 | Visual (scars, sparks), solid numbers |
| 4 | C4 Traveling-wave electrodes and the substrate under them | 10 | Strong interactive (TW response), inversion-layer and quartz traps |
| 5 | C5 Anisotropy and constants: crystal axes are not optional | 11 | 3D index ellipsoid / bend mode crossing visuals; deep niche |
| 6 | C6 Microring and EAM thermal and detuning traps (SiPh) | 9 | Industry-relevant; ring bistability interactive |
| 7 | C7 Cryogenic EO | 8 | Smallest audience, funny failure modes, fewer sources |

---

## C1. "Your DC Is Not Your RF": low-frequency EO relaxation, bias drift, hysteresis (TFLN, TFLT)

- Expert population (I): about 100-500. ATLAS: 17 papers that discuss drift/photorefraction/relaxation have 104 distinct co-authors; add bulk-LN modulator and bias-controller engineers in industry.
- Why niche: the mechanisms (interface traps, cladding conduction, Poole-Frenkel emission) sit between device physics and electrochemistry; most modulator papers report RF metrics and avoid DC bias entirely.
- Visual potential: code interactive of a 2-3 layer Maxwell-Wagner RC stack (LNR fig-04-08 spec: bias step, LN field vs time, |H(f)| 1 mHz-1 MHz); Vpi(f) high-pass curve with the >40 V low-frequency wall; Blender: charges drifting to the metal-LN contact while a ball (bias point) rolls off the cos^2 transfer curve; EO-V curve morphing from linear to diode-like.

| # | Pitfall (one sentence) | Why experts miss it | Gag / visual | Source + locator | Conf. |
|---|---|---|---|---|---|
| 1.1 | A Vpi measured with a kHz-MHz triangle is not the RF Vpi: in visible/NIR TFLN MZMs dielectric relaxation with tau 0.85-2.41 us (20-50 C, 488 nm) pushed Vpi above 40 V at <= 10 kHz, and the authors extracted the usable high-frequency Vpi by fitting a first-order high-pass model in f and tau(T). | "DC Vpi" is a routine 1-minute bench measurement; nobody sweeps it vs frequency. | Slider: drive frequency; Vpi readout climbs to 40 V as you go below 10 kHz. "Please certify the frequency of your DC." | liu2026d p.5 (Fig. 2(c), Table 1); p.2 | S |
| 1.2 | The relaxation time constant is not a material constant: reported values span about 1 ms (1550 nm) and 0.1-1 ms (456 nm) to 0.9-6 us, and it shortens with temperature (surface conductivity) and is expected to shorten with optical power. | People quote one "relaxation time" per platform. | Thermometer and optical-power knobs both bend the same curve. | liu2026d p.2 (cites prior 1 ms, 0.1-1 ms), p.5 | S |
| 1.3 | Relaxation can make the low-frequency response larger rather than smaller (relaxation-enhanced band 1-20 kHz), so a kHz Vpi can be optimistic or pessimistic depending on the device. | Sign of the error is not intuitive; depends on which layer conducts more. | Two devices, same cross-section, one drifts up and one down. | LNR ch04 sec. 4.7 table (holzgrafe2024, status reported); LNR ch06 L3 | C |
| 1.4 | The drift direction flips with cladding: an RC fit gave 11.9 ms with SiO2 cladding (positive drift) vs 88.6 us uncladded (negative drift), consistent with a two-layer Maxwell-Wagner divider whose sign depends on C2/(C1+C2) - G2/(G1+G2). | Teams treat "cladding" as an optical choice only. | Toggle "cladding" in the RC stack; the bias point walks the other way. | LNR ch04 sec. 4.7 (li2026photonics, reported; formula in L2) | C |
| 1.5 | Bulk LN dark conductivity predicts relaxation times of 6.9 h to 29 d; ms/us TFLN times imply an effective conductivity 2e6-3e10 times higher, so the screening path is interfaces, cladding or contacts, not the crystal. | Intuition from bulk modulators (hours) does not transfer to thin film. | A "bulk LN" clock reading days next to a TFLN stopwatch reading microseconds. | LNR ch04 sec. 4.7 "Derived numbers" (eps33 = 28, sigma_d 1e-14 to 1e-16 S/m from volk2008 p.60) | C (derived) |
| 1.6 | Long DC bias leaves a memory: after 30 V for 36 h the EO-V response dropped 4.5x and became rectifying (diode-like), recovered thermally only over about 3 weeks, and a -60 V reverse ramp restored magnitude and linearity while the drift curve was only partly reset. | Drift is treated as a stationary nuisance, not history-dependent hysteresis. | "Your modulator remembers what you did last month." EO-V line bends into a diode curve. | yeh2026 p.2-5 (Figs. 1-4) | S |
| 1.7 | The initial drift rate grows exponentially with bias voltage and fits Poole-Frenkel trap emission (f(V) proportional to sqrt(V)), so a higher-voltage bias test is not a scaled copy of a low-voltage one. | Drift is usually characterized at one bias. | Staircase voltage; drift slope explodes on a log axis. | yeh2026 p.3-4 (Fig. 3, Eqs. 1-5) | S |
| 1.8 | Many "no drift observed" results never applied a DC bias: devices were set to quadrature by tuning the laser wavelength on an unbalanced MZI (valdez2023; wang2025), which avoids the effect instead of measuring it. | Methods sections bury the biasing scheme. | Auditor stamp: "DRIFT TEST: NOT PERFORMED (laser tuned instead)". | valdez2023 p.15; wang2025 p.5 | S (facts) / I (generalization) |
| 1.9 | Drift numbers are reported in incompatible windows and powers (e.g. < 2 dB vs 8 dB over 16 min at 4.3 dBm on-chip, 737 nm, TFLT vs TFLN; 0.4 dB over 15 h at 1.75 mW; constant switching voltage down to 10 mHz), so they cannot be ranked without normalization. | Each paper picks its own metric. | Leaderboard where every row has a different unit. | powell2024 p.1-2; lin2025 p.8; li2026ba p.1 | S (numbers) / I (comparability claim) |
| 1.10 | Cladding and electrode metal change TFLT drift: SiO2 cladding + Ag electrodes drifted about 3 dB vs about 1 dB for air cladding + Au over 60 min (authors suspect cladding-interface traps); bulk-LN history already showed buffer-layer leakage (seconds) and photorefraction (hours) as two drift components. | Stack choices are made for loss and bandwidth, then drift is blamed on the crystal. | Two chips, identical except cladding, diverging traces. | wang2024a p.4; LNR ch04 sec. 4.7 (yamada1981, gee1985, reported) | S / C |
| 1.11 | Thermo-optic biasing is the usual escape, but a "drifting heater" can be probe contact resistance: under constant-voltage drive ER decayed 25 -> 21 dB over 100 min, under constant-current drive it stayed 24-28 dB (median 25) with no systematic drift over 95 min. | Heater drift gets attributed to the chip. | "The heater is innocent. The probe needle did it." | liu2026d p.8 (Fig. S10) | S |

Process lore worth a slide (not a separate pitfall): all MZMs "hotplate-heated at 300 C for 5 h to remove trapped charges which can negatively impact DC drift effects" (powell2024 p.2, S); Ar+ milling pre-treatment and LN-metal interface control plus anneal reported to slow relaxation (LNR ch04 sec. 4.7: shi2025, holzgrafe2024, puma2022; reported).

---

## C2. "Light and Heat That Write Back": photorefraction, pyroelectric charge, power handling

- Expert population (I): about 200-800 for TFLN photorefraction (nonlinear, quantum, visible-TFLN groups); pyroelectric device-level effects: about 50-200.
- Why niche: effects are slow (seconds to weeks), sign-opposite to thermal shifts, and depend on fab history (cladding, e-beam dose, anneal).
- Visual potential: Blender: a beam writing a glowing index "scar" into a crystal slab that fades with light or heat (LNR fig-04-01 spec); a wafer lifted off a hotplate arcing; code: ring-sweep simulator with thermal red shift (triangle) vs photorefractive blue shift (LNR fig-04-04 spec); pyroelectric calculator (fig-04-07).

| # | Pitfall | Why experts miss it | Gag / visual | Source + locator | Conf. |
|---|---|---|---|---|---|
| 2.1 | At high optical power photorefraction in an LN waveguide redistributes space charge, distorts the guided mode and can excite higher-order modes, giving ER degradation, IL increase and operating-point instability; a TFLT modulator ran stably up to 28 dBm inside the modulator. | Power-handling tests are rarely in modulator papers. | Eye diagram melting as the laser knob turns up. | wang2025 p.1, p.5 | S |
| 2.2 | PECVD SiO2 cladding is reported to cause significant photorefraction; spin-on, annealed HSQ cladding brought it down to air-clad levels. | Cladding is chosen by habit. | "Approved cladding list" with PECVD crossed out. | shen2024 p.21 (SI VIII.A) | S |
| 2.3 | Giving trapped charge a path helps: one 780 nm TFLN modulator etched trenches in the cladding and put electrodes in direct contact with LN to reduce photorefraction (screening of the applied field also lowers EO efficiency). | Electrodes are normally kept away from the core for optical loss. | Trench as "charge emergency exit" signage. | celik2022 p.3 | S |
| 2.4 | E-beam lithography dose itself damages LNOI: ring Q fell from 1.3 M (air clad) and 2.0 M (HSQ) to about 0.6 M and 0.5 M; a second anneal recovers Q but strongly amplifies the photorefractive resonance shift. | EBL is assumed optically harmless after anneal. | The cure (anneal) makes the disease worse. | shen2024 p.22-23 (SI VIII.B) | S |
| 2.5 | Photorefraction blue-shifts a TFLN ring while heating red-shifts it, with a different time constant, so a power-dependent resonance can move either way and a lock can run away until traps saturate. | Thermal bistability training covers only the red-shift case. | Laser sweep where the triangle dip turns into an oscillation. | LNR ch04 sec. 4.3 (lin2021, xu2021oe, reported; model template); multani2025 p.16 | C / S |
| 2.6 | An unscreened pyroelectric field in z-cut LN gives a per-kelvin index shift (2.5e-5 to 5.9e-5 /K at 1550 nm) of the same size as the thermo-optic dn_e/dT (3.3e-5 to 3.8e-5 /K), so part of a "thermal" shift can be pyroelectric, depending on screening. | Temperature coefficients are fitted as one number. | Two arrows of equal length labeled "heat" and "charge from heat". | LNR ch04 sec. 4.6 derived table (p3 spread 4e-5 to 9.5e-5 C/(m^2 K)) | C (derived) |
| 2.7 | A 95 K hotplate step gives an unscreened 7.5-17.8 kV across a 0.5 mm z-cut wafer and 15-36 kV/mm inside, of the order of the congruent coercive field (21 kV/mm); practitioners report fried devices and cracked substrates, fixed with a conductive coating or aluminum-foil discharge path. | Silicon bake habits. | "Remove wafer from hotplate slowly, grounded." Spark animation. | LNR ch04 sec. 4.6 (derived) and sec. 4.10 row 1 (practitioner report) | C |
| 2.8 | Cut matters for pyro: in z-cut the charge sits on top/bottom faces and the field overlaps the mode; in x-cut it sits on sidewalls and chip edges; Ti-indiffused z-cut interferometers showed large pyroelectric instabilities while x-cut interferometers were stable. | Cut is chosen for r33 access, not thermal stability. | Two wafers, charges painted on different faces. | LNR ch04 sec. 4.6 (bulmer1986, reported; geometry derived) | C |
| 2.9 | Near-stoichiometric LN without Mg is less damage-resistant than congruent LN at low and moderate intensities (threshold 0.1 vs 1 kW/cm^2 in one tabulation); MgO helps only above a threshold (about 4.5-5 mol% congruent; > 0.78 mol% near-stoichiometric) by raising photoconductivity about 100x, not by stopping the photovoltaic current. | "Stoichiometric = better" folklore. | Threshold step plot with a cliff at 5 mol%. | LNR ch04 sec. 4.2 (volk2008 Table 4.3, P-comp; bryan1984, furukawa2000, reported) | C |
| 2.10 | A pyroelectric field drives photorefraction even in above-threshold MgO:LN: index change up to 0.8e-4 for a 3 C temperature change. | MgO doping is treated as a complete fix. | Thermostat wobble writes a scar in "damage-proof" crystal. | LNR ch04 sec. 4.6 L3 (schwesyg2011 thesis pp. 99-100, P) | C |
| 2.11 | "Damage threshold" names three different things at three intensity scales: photorefraction from as low as 10 mW/cm^2 (Fe-doped, no threshold), coloration near 100 MW/cm^2 pulsed, breakdown in the GW/cm^2 range; a vendor 100-300 MW/cm^2 (10 ns, 1064 nm) figure is not a photorefraction limit. | Datasheet number is read as a cw power limit. | One log axis from 1e-2 to 1e10 W/cm^2 with three unrelated cliffs. | LNR ch04 sec. 4.5 | C |
| 2.12 | LiTaO3 is the usual escape: birefringence about 10x lower and photorefractive resonance shift nearly 5x smaller than LN in identical racetracks, and authors push TFLT for UV because TFLN "faces severe limitations due to the photorefractive effect" toward the UV. | Material switch assumed to cost EO efficiency (r33 30.5 vs 30.9 pm/V per the same source). | Side-by-side ring scans, LN drifting, LT calm. | wang2024b p.2, p.4, p.8; lin2026a p.11 | S |

---

## C3. "Numbers That Mean Different Things": Vpi, VpiL, bandwidth, IL, ER reporting

- Expert population (I): about 100-300 who apply these consistently (reviewers, benchmark compilers, transceiver component engineers); a few thousand read the numbers.
- Why niche: each convention is reasonable in isolation; the trap is cross-paper comparison. The ATLAS schema (conventions (a)-(gg)) is effectively a compliance manual.
- Visual potential: code interactive "metric converter" where one simulated device's headline numbers swing by 2x as convention toggles flip; animated EO S21 with a draggable normalization marker and moving -3/-6 dB lines; histogram of bandwidth bounds piling up at 50/67/110 GHz (instrument limits); Blender: rubber stamps (EXTRAPOLATED, BOUND, UNSTATED REFERENCE) on a datasheet.

| # | Pitfall | Why experts miss it | Gag / visual | Source + locator | Conf. |
|---|---|---|---|---|---|
| 3.1 | Vpi and VpiL are quoted per arm, push-pull MZM, single-arm MZM, series push-pull, or differential (V+ - V-) and per-side amplitude, differing by factors of 2; in ATLAS, of 293 rows with a Vpi value, 177 are push-pull, 54 unspecified, 19 per-arm, 17 resonance-derived, 11 single-arm, 10 differential, 5 series push-pull. | Papers rarely state which; factor 2 hides inside "Vpi". | Toggle panel; VpiL halves and doubles. Example: aimone2026-a quotes 1.1 V cm (differential reference) while 3.26 V x 7.5 mm gives 2.45 V cm. | schema conventions (a), (q); my count; LNR ch06 sec. 6.1 L3 | C |
| 3.2 | "DC Vpi" frequency is often unstated: of 337 Vpi cells audited, 81 were consistent, 167 had a stated frequency to record, 86 had no stated frequency; ATLAS splits DC and RF Vpi at 1 GHz. | Readers assume DC means DC. | Frequency field reads "UNSTATED" in red. | DevLog-020 progress log 2026-10-05 (W4) and rule (m) | C |
| 3.3 | Papers contradict their own measurement frequency: one caption says Vpi "at 1 GHz" while Methods say "500 kHz triangular voltage sweep"; another text says f = 4.18 MHz while the figure spans 0.1-4 GHz. | Captions and methods are written by different coauthors. | Spot-the-difference exercise. | DevLog-020 (liu2025b-a, taghavi2024-b); liu2025b p.4; taghavi2024 p.11 | C / S |
| 3.4 | EO S21 in dB is 20 log10 of modulation amplitude ("electrical dB"): -3 dB EO means the drive power must double, -6 dB means Vpi has doubled (50 % of the integrated phase), so "3 dB bandwidth" and "Vpi bandwidth" differ; one TFLT paper measured a 3 dB EO roll-off near 5 GHz while its 6 dB line was beyond the 20 GHz detector. | Optical vs electrical dB confusion, plus marketing preference for the larger number. | "Which 3 dB? Select: optical / electrical / marketing." | DevLog-036 "Decibel and bandwidth convention" (Alferness 1982 eqs. 20-22); valdez2023 p.10 Eq. (3); powell2024 p.2 | S / C |
| 3.5 | The 3 dB bandwidth depends on the normalization frequency: between 100 MHz and 1 GHz the EO response changed +0.11 dB (0.54 cm) and -0.77 dB (1.0 cm) on two devices; 1 GHz is common, 5 GHz has been used, and one 738 nm MZM quoted 35 GHz "w.r.t. 3 GHz" because of rapid impedance-mismatch roll-off and low-frequency photorefractive instability. | The reference is in a caption if anywhere. | Drag the reference marker; the 3 dB point slides 20 GHz. | valdez2023 p.15; renaud2023 p.3 | S |
| 3.6 | In ATLAS, of 302 rows with a 3 dB bandwidth, 154 are normalized to an unstated low-frequency plateau and 31 have no stated reference; 65 are referenced to DC, 25 to 1 GHz, 4 to 10 GHz, 23 other. | Nobody counts this. | Pie chart titled "Where is your zero?" | my count (devices.csv bw3db_reference); convention (t) | C |
| 3.7 | Bandwidths are frequently lower bounds set by the instrument: 106 of 302 ATLAS 3 dB values carry a "greater than" qualifier, and in the TFLN/TFLT subset 67 of 145 are bounds piling up at 110 GHz (15), 67 GHz (13), 50 GHz (12); medians over bounded data understate. | Bounds plotted as points. | Histogram spikes exactly at VNA catalog frequencies. | my count; LNR ch06 sec. 6.1 L3 | C |
| 3.8 | Headline bandwidths can be extrapolations: a paper titled with an "Extrapolated Bandwidth of 170 GHz" measured a 2 dB roll-off at 100 GHz (entered as > 100 GHz), and a peaking-model bandwidth "exceeds 200 GHz" while measurement shows > 110 GHz. | Titles and abstracts carry the model number. | Stamp: EXTRAPOLATED. | ATLAS devices.csv notes arabjuneghani2022-b; chen2025 p.2 | C / S |
| 3.9 | A claimed 29 GHz measured resonator bandwidth could not be confirmed as a measured crossing when the Methods list a 7.5 GHz VNA and a 12 GHz photodetector; it equals the Q-limited estimate (about 30.7 GHz), so the curated row was cleared. | Reviewers do not cross-check instrument specs. | "Measuring 29 GHz with a 7.5 GHz VNA violates Policy 4.2." | ATLAS devices.csv notes kari2025-a; DevLog-020 g1 follow-ups | C |
| 3.10 | A load impedance below the line impedance (e.g. 30 ohm) "peaks" the response when normalized at 1 GHz at the cost of back-reflection to the source; an under-terminated SiPh MZM showed slight peaking at -2 V bias. | Peaking reads as bandwidth. | Bandwidth trophy with an S11 skeleton in the closet. | valdez2023 p.14; gong2026 p.3 | S |
| 3.11 | Insertion-loss scope varies: of 164 ATLAS rows with on-chip IL, 64 are whole device minus couplers, 31 phase-section only, 20 excess over an on-chip reference, 3 free-space, 46 undefined; one paper's main text (3.1/6.6/4.0 dB) and SI (2.95/6.43/3.77 dB) disagree. | "IL" without scope is the norm. | Four rulers of different lengths all labeled "IL". | my count; convention (s); DevLog-020 g3 (rahman2025) | C |
| 3.12 | ER measured with an on-chip photodiode was capped at 20 dB by light in a substrate mode, while fiber collection gave 31 dB; and a null-biased MZM with a 2x1 MMI combiner dumps light into slab modes that partly recouple downstream, limiting ER (fix: 2x2 MMI). | ER is assumed to be an interferometer property. | Light leaking into the slab like water under a door. | gupta2023 p.5; nelan2022 p.7 | S |
| 3.13 | Array-level values get copied onto channel rows and a "statistic" is rarely stated: ATLAS removed an array-level 25 dB ER from 8 channel rows, and only 40 of 521 rows state whether headline values are single, best, mean or median. | Hero devices are not labeled as such. | "Best of 200 devices" fine print magnified. | DevLog-020 g1 (hu2026a); convention (z); my count | C |
| 3.14 | "L" in VpiL is ambiguous: ATLAS uses electrode length per arm, two-arm totals go to notes, and ring VpiL quoted with circumference stays in notes; electrode-gap tapers in the probe region (about 0.8 mm per end on a 5 mm device) add an estimated 3-8 % effective length on 1 cm. | Length is "obvious". | Ruler with three tick marks all labeled L. | convention (q); DevLog-029 renaud2023 table (reader estimate) | C |
| 3.15 | Figure geometry can contradict the label: one paper's own simulated field plot digitizes to a 4.84 um electrode gap while the text says 6.0 um; with the labeled gap the measured VpiL cannot be reproduced. | Nobody digitizes field plots. | Caliper on a colour map. | DevLog-029 mao2022 table | C |
| 3.16 | RF loss units: amplitude vs power (1 Np = 8.686 dB power; Alferness converts dB/cm with 4.3) and sqrt(f) vs f scaling are often unstated, and only 11 % of TFLN rows report RF loss at all (Z0 10 %, n_RF 10 %). | Factor-2 hides in Np vs dB. | Unit-conversion boss fight. | convention (cc); DevLog-036 (Alferness note); DevLog-008; LNR ch06 L3 disclosure stats | C |

---

## C4. Traveling-wave electrodes and the substrate under them

- Expert population (I): about 150-500 (ATLAS: 41 papers with capacitively loaded / slow-wave / segmented electrodes, 311 co-authors; plus RF-photonics packaging engineers).
- Why niche: requires simultaneous microwave transmission-line and optical group-index intuition; loaded lines break cross-section intuition.
- Visual potential: code interactive TW response (n_RF, n_g, alpha, Z0, Z_S, Z_L, L sliders; LNR F6.04 spec; ATLAS engine `eo_response` contract); Blender: an optical pulse and a microwave pulse racing along a T-rail electrode, falling out of step; a glowing inversion sheet under the buried oxide eating the microwave.

| # | Pitfall | Why experts miss it | Gag / visual | Source + locator | Conf. |
|---|---|---|---|---|---|
| 4.1 | Velocity mismatch alone gives f_3dB L = 0.443 c / abs(n_RF - n_g); RF loss alone reaches 3 dB at a total electrode loss of 6.4 dB; for one loaded device (n_RF 2.35, n_g 2.2, 10 mm) the velocity term alone sets 88.5 GHz vs 317 GHz from loss, so a 0.15 index mismatch, not loss, is the limiter. | Loss is the usual suspect. | Two sliders; the "culprit" light switches. | LNR ch06 sec. 6.2 (derived; chen2022-c values from ATLAS) | C (derived) |
| 4.2 | A cross-section solve does not give the n_RF or Z0 of a periodically loaded line; capacitive loading raises n_RF and lowers Z0 together, and analytic T-rail capacitance formulas proved unreliable (a x23 fudge factor in one regression). | 2D solvers are trusted for 3D periodic structures. | Engineer measuring one slice of a centipede. | ATLAS README "Simulation status"; LNR ch06 sec. 6.2 L3; DevLog-023 "Pitfalls recorded"; DevLog-036 loaded line | C |
| 4.3 | Quartz handles are too fast: conventional electrodes on TFLN-on-quartz give RF index about 1.8 vs optical group index about 2.2, so quartz needs slow-wave loading, whereas on Si with thick oxide the slow-wave effect is undesirable. | Quartz is sold as the low-loss fix. | Microwave sprinting ahead of light on quartz. | kharel2021 p.3 | S |
| 4.4 | Si handle loss: microwave loss tangent 0.043 for Si vs 1e-5 for the quartz handle in one volume-manufacturing paper; capacitively loaded electrodes on SiO2/high-resistivity Si reached about 0.23 dB/cm/sqrt(GHz), comparable to quartz. | Substrate tan delta not in optical design flows. | Substrate swap gameshow. | liu2025b p.3; sayem2026 p.1 | S |
| 4.5 | "High-resistivity silicon" is not resistive at its oxide interface: a low-resistivity inversion layer can give coplanar-waveguide losses above 10 dB/cm at GHz, and a TFLN/SiPh modulator attributed unmodeled RF loss to sheet charge at the HR-Si/SiO2 interface. | Bulk resistivity spec (> 3 kOhm cm) is read as the whole story. | Hidden conductive sheet revealed by an X-ray scan. | witmer2020 p.14 (citing Wu et al. 1999); boynton2020 p.11; vanackere2023 p.4 | S |
| 4.6 | A fast initial roll-off followed by a flat high-frequency response signals impedance mismatch, not velocity mismatch (consistent with strong S11). | Every roll-off gets blamed on velocity. | Diagnosis flowchart. | powell2024 p.2; renaud2023 p.3 | S |
| 4.7 | Loss hides reflections: slow-wave lines at about 40 ohm showed the mismatch effect more clearly on 0.54 cm devices than on 1.0 cm devices because extra RF loss on the longer line damps the back-reflection. | Shorter should be cleaner. | "Loss is a feature." | valdez2023 p.14 | S |
| 4.8 | Pad and stub parasitics create LC peaking (a small resonance near 75 GHz in a BTO plasmonic device; peaking between 500 and 800 GHz from pad inductance in a THz-class device) that can extend or distort an apparent bandwidth. | Pads are excluded from the model. | Pad drawn as a spring. | kohli2025 p.4; horst2025 p.2-3 | S |
| 4.9 | A BTO device's Vpi drops from MHz to low GHz then flattens (6.4 V at 40 GHz, r_eff about 180 pm/V) because the Pockels coefficient itself disperses; RF Vpi cannot be predicted from a MHz Vpi on BTO. | Assumes a frequency-flat material. | See C5.6. | kohli2025 p.4 | S |
| 4.10 | In a sub-THz cavity the microwave mode hybridizes with many parasitic sapphire substrate modes (avoided crossings), so the substrate is part of the resonator. | Substrate is treated as a passive holder. | Resonance chart with a forest of substrate lines. | multani2025 p.5 | S |

---

## C5. Anisotropy and material constants: crystal axes are not optional

- Expert population (I): x-cut TFLN layout and mode designers about 200-600; tensor sign and clamping subtleties a few dozen to about 100.
- Why niche: commercial mode solvers accept a scalar or diagonal tensor; textbook constants are quoted without boundary conditions (clamped/unclamped) or frames.
- Visual potential: Blender: index ellipsoid rotating relative to the propagation direction as light runs around an x-cut ring, TE and TM modes swapping at the crossing angle; BTO domains with 45 deg routing; code: n_eff vs propagation angle (LN vs LT) with the crossing marked; BTO r(f) dispersion curve; clamped vs unclamped toggle on a VpiL calculator.

| # | Pitfall | Why experts miss it | Gag / visual | Source + locator | Conf. |
|---|---|---|---|---|---|
| 5.1 | In x-cut LN the fundamental TE and TM modes cross at a propagation angle of 25 deg (for the studied geometry), so a bend sweeping through all angles passes a hybridization point; LTOI of the same dimensions shows no crossing (birefringence LN -0.07 vs LT +0.004). | Bends are designed with isotropic intuition. | Ring with a "polarization toll booth" at 25 deg. | wang2024b p.3-4 (Fig. 2e-g) | S |
| 5.2 | Mode hybridization in sharp bends and adiabatic tapers comes from LN anisotropy plus angled sidewalls together. | Sidewall angle is treated as a loss parameter only. | Trapezoid rib tilting the mode. | liu2023 p.2 | S |
| 5.3 | Sidewall angle is rarely reported and moves VpiL: a 60 deg sidewall lowered a simulated VpiL by 3.5 % in one case; only 45 of 521 ATLAS rows carry a sidewall angle, and the reference (from plane vs from normal) must be stated. | Considered a fab detail. | "Is 70 deg from the plane or from the normal? Mandatory training." | DevLog-029 renaud2023 table; my count; convention (u) | C |
| 5.4 | Clamped vs unclamped: a static-field measurement corresponds to unclamped r^T, an RF measurement to clamped r^S; LN r33 is 30.8 (clamped, Turner 1966 row; 28.8 in the Kaminow row) vs 32.2-34.0 (unclamped) pm/V at 633 nm and eps11 is 44.3 vs 85.2; using clamped constants against kHz-MHz Vpi data explains -4.3 % to -6.7 % of VpiL gaps. | Simulations use one "r33". | Toggle that silently moves VpiL 5 %. | DevLog-038 (Weis and Gaylord 1985 Table 1 p.197, Table 4 p.199); DevLog-029 conclusions | C |
| 5.5 | r22 and r51 roughly halve when clamped (high-speed condition). | Rarely used components, rarely questioned. | | LNR research/R06 sec. 3 pitfall 7 | C |
| 5.6 | Thin-film BTO's effective Pockels coefficient is strongly frequency dependent: 1/2(r13 + 2 r42) is 481 pm/V at 100 MHz and 191 pm/V at 330 GHz, while r33 is flat above 1 GHz; LN shows no such dispersion. | A single "r_eff = 300+ pm/V" circulates. | Coefficient deflating like a balloon along a log-frequency axis. | chelladurai2025 p.2, p.6 | S |
| 5.7 | BTO r_eff (45 deg alignment, r42) treats the response as pure phase modulation and ignores polarization rotation from off-diagonal tensor terms in multidomain films. | r_eff hides the tensor. | Light exiting slightly rotated, nobody notices. | lin2026b p.26 | S |
| 5.8 | Two papers' own "agreement" checks used the ordinary index (2.21) where the x-cut TE r33 interaction needs n_e (about 2.14), and one paper's simulated VpiL (4.88 V cm) is not reproducible from its own stated equation inputs (3.87 V cm). | Index tables are mixed up; consistency checks are trusted. | n_o and n_e name tags swapped. | DevLog-029 mao2022 and boynton2020 tables, "Inputs vs physics" | C |
| 5.9 | Two mechanisms can hide in one device: a ferroelectric nematic LC on Si gives a slow orientational DC response and a fast Pockels AC response with different efficiency (AC VpiL about 25.7 V mm with r33 about 24 pm/V at f = 4.18 MHz; the curated DC row has Vpi 0.5 V; the authors attribute the DC-AC difference to the phase-shift mechanism); ATLAS splits such devices into two rows. | One "Vpi" per device assumed. | Two animals in one costume. | taghavi2024 p.11; convention (q) | S / C |
| 5.10 | LN thermo-optic response is strongly anisotropic (dn_e/dT about 3.75e-5 vs dn_o/dT about 0.41e-5 /K at 1550 nm, 25 C, from Sellmeier recomputation; dn_o/dT disagrees up to 2.5x between sources; one MgO:LN Sellmeier set gives anomalous dn/dT and is flagged "do not use"). | One dn/dT per material in heater design. | Two thermometers, one barely moves. | LNR ch04 sec. 4.8 table (derived from jundt1997, cti1999; R14 rows 36c, 38) | C |
| 5.11 | Tensor frames bite: never mix photoelastic p14/p41 from one paper with c14/e22 from another without checking the c14 sign; a c14 < 0 frame is non-standard for LN under IEEE 176-1987 / IRE rules; and HF etches the -c face much faster than +c, so x/y-cut sidewalls facing -c and +c recede differently. | Signs are copied from tables. | Mirror-image crystal duel. | LNR research/R06 sec. 3 pitfall 2; R14 rows 78-81 (inference flagged there); LNR ch04 sec. 4.10 row 8 (volk2008 p.168, P-comp) | C |

---

## C6. Microring and EAM thermal and detuning traps (silicon photonics)

- Expert population (I): about 300-2000 Si MRM / Ge EAM designers and characterization engineers, mostly industry; the operating-point-placement subtleties maybe a few hundred.
- Why niche: small-signal bandwidth, ER and resonance position all depend on optical power and laser detuning, which papers fix (or do not report) at characterization.
- Visual potential: code: ring transmission vs laser wavelength with self-consistent heating (scan-direction-dependent fold, bistability), detuning slider showing optical peaking vs efficiency; Blender: ring resonance "running away" from a laser line as the package heats from 15 to 85 C.
- Corpus gap: the corpora contain almost nothing on plasma-dispersion vs Pockels mixing in SiPh (one mention of combined plasma dispersion + DC Kerr explaining linearity, shawon2024 p.9). Not enough for a chapter.

| # | Pitfall | Why experts miss it | Gag / visual | Source + locator | Conf. |
|---|---|---|---|---|---|
| 6.1 | In a Si MRM model with stated assumptions, self-heating dominates resonance displacement: at 5 mW bus input the thermal red shift is 668 pm vs carrier blue shifts of 5.7 pm (1 ns residence) and 0.1 pm (10 ps); the thermal shift equals one linewidth at 2.7 mW. | Carrier effects get the attention. | Elephant (heat) and mouse (carriers) on a seesaw. | huang2026a p.27 (model, assumed absorbed fraction) | S (model values) |
| 6.2 | Above a calculated 1.8 mW threshold the CW transmission folds: one laser wavelength supports two stable hot resonances, and the measured spectrum depends on scan direction. | Static spectra are taken at one sweep direction. | Ring that refuses to come back. | huang2026a p.28; carmon2004 (via LNR ch04 sec. 4.8, title) | S |
| 6.3 | Si rings shift about 80 pm/K while CPO optical engines without integrated lasers must work across 15-85 C case temperature, so cold-resonance placement and heater reserve are system specs, not device afterthoughts. | Device papers measure on a TEC at one temperature. | 70 K x 80 pm/K = 5.6 nm (my arithmetic) of travel drawn as a highway. | huang2026a p.5 (80 pm/K, OIF 3.2 Tb/s CPO 15-85 C) | S / I (arithmetic) |
| 6.4 | Bandwidth measurements are deliberately run at low optical power (e.g. bus power limited to 500 uW) because TPA, FCA and self-heating shift the resonance, and high power can give asymmetric resonances and apparent EO bandwidth extension. | High-power S21 looks better and gets published. | "Results valid only below 0.5 mW" disclaimer font size 4. | hu2026 p.20; geravand2025 p.32 | S |
| 6.5 | Laser detuning trades efficiency for bandwidth via optical peaking; intensity-modulation peaking varies abruptly with detuning near resonance, so a single MRM bandwidth number is an operating-point choice. | Bandwidth treated as a device property. | Detuning knob; bandwidth and OMA move opposite. | hu2026 p.7; geravand2025 p.6; hu2023 p.1, p.3; huang2026a p.6 | S |
| 6.6 | Peaking plus a low normalization frequency inflates bandwidth: one MOSCAP MRM normalized to 500 MHz and reported peaking that "enhanced" its 3 dB bandwidth to 11 GHz. | Same as 3.5 and 3.10. | | hsu2024 p.9 | S |
| 6.7 | In Ge EAMs high optical power self-heating increases static ER without a matching increase in dynamic ER; one link budget books 4 dB to self-heating. | Static ER is the headline. | Static ER trophy, empty eye diagram. | steckler2025 p.3 (citing its ref. 23); li2026b p.2 | S |
| 6.8 | "ER" of a ring is often the resonance notch depth at fixed bias, not a voltage-switched contrast (21 ATLAS rows), and tuning signs flip with the reverse-bias sign convention (several sign corrections recorded during curation). | Positive reverse-bias axis vs negative voltage. | Two plots, mirror-image, both labeled "pm/V". | conventions (o), (v); DevLog-020 g1-g4 (geravand2025-a +0.032 -> -0.032 nm/V; bao2026b-a, hu2026-a, tan2024-a sign checks) | C |
| 6.9 | The standard 1.55 um plasma-dispersion table (Nedeljkovic 2011 Table 2) lists 0.552 dB absorption over L_pi while its own alpha and L_pi give 0.518 dB; implementers should gate on the formula, not that cell. | Tables are copied as ground truth. | Calculator vs printed table standoff. | DevLog-039 gate G8 note | C |

---

## C7. Cryogenic electro-optics

- Expert population (I): about 50-200 (ATLAS: 10 cryogenic papers, 98 co-authors; transduction and superconducting-readout groups).
- Why niche: few groups run EO devices at 4 K; effects mix superconductivity, pyroelectricity and photorefraction.
- Visual potential: Blender: dilution-fridge cutaway with a chip sparking during cooldown; code: Vpi(T) scatter showing the -10 % to +74 % spread; S21 step when Nb crosses Tc.

| # | Pitfall | Why experts miss it | Gag / visual | Source + locator | Conf. |
|---|---|---|---|---|---|
| 7.1 | Vpi at 4 K is not predictable from room temperature: one TFLN device went 220 mV -> 380 mV (+70 %), and literature reports range from -10 % to +20 %, +74 %, and +15 % (LNOI ridge); r33 is cited as dropping 20 % at 7 K. | Cryogenic EO assumed to be "the same, but colder". | Vpi thermometer with arrows pointing both ways. | shen2024 p.4, p.17 (SI IV) | S |
| 7.2 | Superconducting electrodes are lossy above Tc (signal electrode 13.8 kOhm above about 8 K for Nb), and compound superconductors (NbN, TiN, NbTiN) have kinetic inductance up to an order of magnitude higher, which changes the line. | RF design done at room temperature. | Bandwidth appears only below Tc. | shen2024 p.4 | S |
| 7.3 | Superconductor and anneal are incompatible: a 200 C vacuum anneal dropped Nb Tc from 8 K to about 2 K, so e-beam-induced loss (0.8 dB/cm) could not be annealed out after metallization. | Process order chosen for the optical part. | Two process cards that cannot both be played. | shen2024 p.23 (SI VIII.B) | S |
| 7.4 | Photorefraction at 4 K depends on process history: a device that received metal lift-off plus anneal drifted strongly even at 0 dBm and was stable only near -20 dBm, while a sister device drifted < 5 % over minutes at 13 dBm. | Cryo assumed to freeze out drift. | Two chips in the fridge, one wandering. | shen2024 p.23 (SI VIII.C) | S |
| 7.5 | Cryogenic TFLT transducers still showed bias drift accelerated by prolonged on-chip optical power above about -20 dBm. | "Low temperature = stable" folklore. | | axline2026 p.6 | S |
| 7.6 | Pyroelectric charge from cooldown: practitioner notes report an LN chip damaging a device on a neighboring chip during a 4 K cooldown (shielding electrodes protected it) and released x-cut structures collapsing after a wafer change, stopped by shorting LN parts to the Si substrate with Al. | Pyroelectricity forgotten for cooling (it is usually discussed for heating). | Chip zapping its neighbor in the cold plate. | LNR ch04 sec. 4.10 rows 3-4 (practitioner report; mechanism is a hypothesis); R08 sec. 3.4 (Zhu 2021 notes discharge damage to superconducting nanowires on bulk LN, secondary) | C |
| 7.7 | A documented remedy is counterintuitive: auxiliary co-propagating light reduces pyroelectric effects in cryogenic LN waveguides (raise photoconductivity rather than block light); bulk remedies include flooding with blue light. | Light is usually the enemy in photorefraction. | "Fight light with light." | LNR ch04 sec. 4.3 (lange2026, reported) and sec. 4.6 L3 (schwesyg2011 p.101, P) | C |
| 7.8 | DC-stable cryogenic operation has been reported (12 h drift-free LNOI MZI with SNSPDs; DC-stable TFLN at liquid-nitrogen temperature), so stability is device- and power-specific rather than guaranteed or excluded. | Both camps generalize. | | LNR ch04 sec. 4.7 table (lomonte2021, warner2025, reported) | C |

---

## Cross-cutting notes

- Best single "chapter" sources for a script writer: `lithium-niobate/content/chapters/04-quirks/manuscript.md` (sections 4.1-4.10 already structured as L1 lay / L2 numbers / L3 measurement and error sources, with figure specs in `figures.md` that map to interactives) and `eo-modulator-atlas/data/schema/devices.schema.yaml` header (conventions (a)-(gg) read like policy clauses).
- ATLAS DevLog-029 (VpiL discrepancy investigation) is a ready-made "case study" chapter: five papers, engine vs measured, each gap traced to conventions, constants or undisclosed geometry.
- Not found in these corpora (do not invent): quantitative TE/TM hybridization vs sidewall angle numbers; heater drift aging; Si-substrate conductivity numbers beyond the cited > 3 kOhm cm and > 10 dB/cm statements; GRIIRA in thin film (LNR ch04 sec. 4.4 L3 says no TFLN paper found); plasma-dispersion/Pockels mixing data.

---

## Bibliography (keys used above)

ATLAS paths are `eo-modulator-atlas/references/<key>/text.md` unless noted. LNR keys are cited inside the LNR chapter files; their DOIs come from `content/chapters/04-quirks/refs.yaml` (or 06-devices/refs.yaml for holzgrafe2024).

| Key | Title | Year | DOI / arXiv | Where read |
|---|---|---|---|---|
| liu2026d | Visible and Near-Infrared Thin-Film Lithium Niobate Modulators with Thermo-Optic Bias | 2026 | 10.1021/acsphotonics.6c00668 | ATLAS references |
| yeh2026 | Hysteresis and trap emission in dc-biased integrated lithium niobate electro-optic modulators | 2026 | 10.1021/acs.nanolett.6c03406; arXiv:2609.17489 | ATLAS references |
| valdez2023 | Integrated O- and C-band silicon-lithium niobate Mach-Zehnder modulators with 100 GHz bandwidth, low voltage, and low loss | 2022 (OE 2023) | 10.1364/oe.480519; arXiv:2211.05208 | ATLAS references |
| powell2024 | A sub-volt near-IR lithium tantalate electro-optic modulator | 2024 | 10.1364/cleo_si.2024.sm2d.2; arXiv:2505.00906 | ATLAS references (arXiv copy) |
| lin2025 | Copper damascene process-based high-performance thin-film lithium tantalate modulators | 2025 | 10.1038/s41467-026-69588-6; arXiv:2505.04755 | ATLAS references |
| li2026ba | Low voltage and high-bandwidth thin-film lithium tantalate modulator on a silicon dioxide substrate | 2026 | arXiv:2604.14836 | ATLAS references |
| wang2024a | Ultrabroadband thin-film lithium tantalate modulator for high-speed communications | 2024 | 10.1364/optica.537730; arXiv:2407.16324 | ATLAS references |
| wang2024b | Lithium tantalate photonic integrated circuits for volume manufacturing | 2024 | 10.1038/s41586-024-07369-1 | ATLAS references |
| wang2025 | Thin-Film Lithium Tantalate Modulator Operating at High Optical Power | 2025 | 10.1021/acsphotonics.5c00159 | ATLAS references |
| lin2026a | Thin-film lithium tantalate for ultraviolet integrated electro-optic modulator | 2026 | arXiv:2605.02758 | ATLAS references |
| celik2022 | High-bandwidth CMOS-voltage-level electro-optic modulation of 780 nm light in thin-film lithium niobate | 2022 | 10.1364/oe.460119; arXiv:2204.03138 | ATLAS references |
| shen2024 | Photonic link from single-flux-quantum circuits to room temperature | 2023 (Nat. Photon. 2024) | 10.1038/s41566-023-01370-2; arXiv:2309.03284 | ATLAS references (incl. SI pages) |
| multani2025 | Integrated sub-terahertz cavity electro-optic transduction | 2025 | arXiv:2504.01920 | ATLAS references |
| axline2026 | Stable, bidirectional electro-optic transduction in thin film lithium tantalate | 2026 | arXiv:2606.12726 | ATLAS references |
| renaud2023 | Sub-1 Volt and high-bandwidth visible to near-infrared electro-optic modulators | 2022 (NC 2023) | 10.1038/s41467-023-36870-w; arXiv:2210.13521 | ATLAS references |
| liu2025b | Volume manufacturing of thin-film lithium niobate modulators with bandwidth > 110 GHz based on 4-inch wafer with a quartz handle | 2025 | 10.37188/lam.2025.047 | ATLAS references |
| taghavi2024 | GHz-rate optical phase shift in light matter interaction-engineered, silicon-ferroelectric nematic liquid crystals | 2024 | 10.1038/s41467-025-63924-y; arXiv:2405.08833 | ATLAS references |
| arabjuneghani2022 | Thin-Film Lithium Niobate Optical Modulators with an Extrapolated Bandwidth of 170 GHz | 2022 | 10.1002/adpr.202200216 | ATLAS devices.csv notes |
| kari2025 | Enabling High-Bandwidth Coherent Modulation Through Scalable Lithium Niobate Resonant Devices | 2025 | arXiv:2502.10846 | ATLAS devices.csv notes |
| chen2025 | Ultra-Compact and Ultra-High Speed Topological Photonic Crystal Modulator Based on Thin Film Lithium Niobate | 2025 | 10.1002/lpor.202500380 | ATLAS references |
| gong2026 | A 60 GHz EO Bandwidth Mach-Zehnder Modulator for 200G/lambda O-band Datacom in 300-mm Monolithic CMOS Silicon Photonics Foundry | 2026 | 10.1364/ofc.2026.w1a.2 | ATLAS references |
| gupta2023 | Implementation, Modelling and Verification of High-Speed Mach-Zehnder Phase Modulators in an Open Access InP Foundry Platform | 2023 | 10.1109/jlt.2023.3244129 | ATLAS references |
| nelan2022 | Ultra-high extinction dual-output thin-film lithium niobate intensity modulator | 2022 | 10.1109/access.2022.3207764; arXiv:2207.02608 | ATLAS references |
| kharel2021 | Breaking voltage-bandwidth limits in integrated lithium niobate modulators using micro-structured electrodes | 2020 (Optica 2021) | 10.1364/optica.416155; arXiv:2011.13422 | ATLAS references |
| sayem2026 | Performance of Capacitively Loaded RF Electrodes on SiO2/Si Substrates for High-Speed Electro-Optic Modulators | 2026 | arXiv:2609.21830 | ATLAS references |
| witmer2020 | A silicon-organic hybrid platform for quantum microwave-to-optical transduction | 2019 (QST 2020) | 10.1088/2058-9565/ab7eed; arXiv:1912.10346 | ATLAS references |
| boynton2020 | A heterogeneously integrated silicon photonic/lithium niobate travelling wave electro-optic modulator | 2020 | 10.1364/oe.28.001868 | ATLAS references |
| vanackere2023 | Heterogeneous integration of a high-speed lithium niobate modulator on silicon nitride using micro-transfer printing | 2023 | 10.1063/5.0150878 | ATLAS references |
| kohli2025 | The plasmonic BTO-on-SiN platform - beyond 200 GBd modulation for optical communications | 2025 | 10.1038/s41377-025-02116-1 | ATLAS references |
| horst2025 | Ultra-wideband MHz to THz plasmonic EO modulator | 2025 | 10.1364/optica.544016 | ATLAS references |
| liu2023 | Ultra-compact lithium niobate photonic chip for high-capacity and energy-efficient wavelength-division-multiplexing transmitters | 2023 | 10.37188/lam.2023.013 | ATLAS references |
| chelladurai2025 | Barium titanate and lithium niobate permittivity and Pockels coefficients from megahertz to sub-terahertz frequencies | 2025 | 10.1038/s41563-025-02158-1 | ATLAS references |
| lin2026b | Integrated photonic devices in thin-film barium titanate: Opportunities and challenges | 2026 | 10.1063/5.0325359; arXiv:2609.16616 | ATLAS references |
| huang2026a | Co-design of Silicon Microring Modulator beyond 200 Gb/s per Lane: Device Physics, Operating Point, and Compact Models for Scale-Up and Scale-Out Optical I/O | 2026 | arXiv:2609.36690 | ATLAS references |
| hu2026 | 400-Gbps/lambda Ultrafast Silicon Microring Modulator for Scalable Optical Compute Interconnects | 2025 | 10.1002/lpor.202502840; arXiv:2509.01555 | ATLAS references |
| hu2023 | 300-Gbps optical interconnection using neural-network based silicon microring modulator | 2021 (2023) | 10.1038/s44172-023-00115-x; arXiv:2111.05331 | ATLAS references |
| geravand2025 | Ultrafast coherent dynamics of microring modulators | 2024 (Nat. Photon. 2025) | 10.1038/s41566-025-01686-1; arXiv:2412.17986 | ATLAS references |
| hsu2024 | Sub-volt high-speed silicon MOSCAP microring modulator driven by high-mobility conductive oxide | 2023 (NC 2024) | 10.1038/s41467-024-45130-4; arXiv:2308.16255 | ATLAS references |
| steckler2025 | Monolithic electro-optic platform on silicon with bandwidth of 100 GHz and beyond | 2025 | 10.1038/s41467-025-66566-2 | ATLAS references |
| li2026b | 300-Gb/s/lambda PAM4 IM/DD Link Enabled by GeSi Electro-absorption Modulator and BU-GRU Equalization | 2026 | 10.1364/ofc.2026.w2a.42 | ATLAS references |
| shawon2024 | Optical Linearization of Silicon Photonic Ring-Assisted Mach-Zehnder Modulator | 2023 (JLT 2024) | 10.1109/jlt.2023.3345350; arXiv:2308.15763 | ATLAS references |
| holzgrafe2024 | Relaxation of the electro-optic response in thin-film lithium niobate modulators | 2024 | 10.1364/OE.507536 | LNR ch04/ch06 (abstract-level) |
| li2026photonics | Low-Frequency Time-Domain Response of Thin-Film Lithium Niobate Electro-Optic Modulator | 2026 | 10.3390/photonics13040339 | LNR ch04 (abstract) |
| yamada1981 | DC Drift Phenomena in LiNbO3 Optical Waveguide Devices | 1981 | 10.1143/JJAP.20.733 | LNR ch04 (abstract) |
| gee1985 | Minimizing dc drift in LiNbO3 waveguide devices | 1985 | 10.1063/1.96220 | LNR ch04 (abstract) |
| bulmer1986 | Pyroelectric effects in LiNbO3 channel-waveguide devices | 1986 | 10.1063/1.96640 | LNR ch04 (abstract) |
| xu2021oe | Mitigating photorefractive effect in thin-film lithium niobate microring resonators | 2021 | 10.1364/OE.418877 | LNR ch04 (abstract) |
| lin2021 | Turnkey generation of Kerr soliton microcombs on thin-film lithium niobate on insulator microresonators powered by the photorefractive effect | 2021 | 10.1364/OE.446527 | LNR ch04 (abstract) |
| lange2026 | Photorefraction management in lithium niobate waveguides: High-temperature vs cryogenic solutions | 2026 | 10.1063/5.0324002; arXiv:2601.15817 | LNR ch04 (abstract) |
| lomonte2021 | Single-photon detection and cryogenic reconfigurability in lithium niobate nanophotonic circuits | 2021 | 10.1038/s41467-021-27205-8 | LNR ch04 / R08 |
| warner2025 | DC-stable thin-film lithium niobate modulator at liquid nitrogen temperatures | 2025 | 10.1364/OL.571632 | LNR ch04 (abstract) |
| shi2025 | Alleviation of DC drift in a thin-film lithium niobate modulator utilizing Ar+ ion milling | 2025 | 10.1364/OL.549975 | LNR ch04 (abstract) |
| puma2022 | Mitigating Anomalous Sub-Megahertz Frequency Response of Electro-Optic Phase Modulators in X-Cut Lithium Niobate on Insulator | 2022 | 10.1364/CLEO_SI.2022.SF2O.1 | LNR ch04 (abstract) |
| carmon2004 | Dynamical thermal behavior and thermal self-stability of microcavities | 2004 | 10.1364/OPEX.12.004742 | LNR ch04 (title) |
| volk2008 | Lithium Niobate: Defects, Photorefraction and Ferroelectric Switching (book) | 2009 | 10.1007/978-3-540-70766-0 | LNR ch04 (selected chapters, P-comp) |
| schwesyg2011 | Interaction of light with impurities in lithium niobate crystals (dissertation, Univ. Bonn) | 2011 | none | LNR ch04 (held copy) |
| weis1985 | Weis and Gaylord, LiNbO3 property review (Table 1, Table 4 used) | 1985 | see `lithium-niobate/content/chapters/04-quirks/refs.yaml` | ATLAS DevLog-038; LNR R14 |
| Alferness 1982 | Alferness, waveguide electrooptic modulators review (eqs. 18-22) | 1982 | not recorded in the files read | ATLAS DevLog-036 |
| Nedeljkovic 2011 | Nedeljkovic, Soref, Mashanovich, IEEE Photonics J. 3, 1171 | 2011 | not recorded in the files read | ATLAS DevLog-039 |

Curation files cited: `eo-modulator-atlas/DevLog/DevLog-020-data-convention-decisions.md`, `DevLog-029-vpil-discrepancy-investigation.md`, `DevLog-036-s7-eo-response-loaded-line.md`, `DevLog-038-library-primary-constants.md`, `DevLog-039-s8-silicon-drift-diffusion.md`, `DevLog-023-simulator-tfln-roadmap.md`, `DevLog-008-e3-rf-line.md`, `data/schema/devices.schema.yaml`, `data/devices.csv`, `README.md`; `lithium-niobate/content/chapters/04-quirks/manuscript.md`, `06-devices/manuscript.md`, `05-making/manuscript.md`, `research/R06-material-properties-data.md`, `research/R08-tfln-state-of-art.md`, `research/R14-property-verification.md`.
