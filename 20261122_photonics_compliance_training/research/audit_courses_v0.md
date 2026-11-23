# Audit v0: ferroelectric.js (FE-0042) and feedback.js (FB-1550)

Date: 2026-11-22. Fresh-context audit, technical and citation accuracy only (parody tone out of scope).

Method:
- All 52 unique DOIs in both files fetched from `api.crossref.org/works/<doi>` at about 1 request per 1.2 s; all returned HTTP 200, no rate limiting. The 3 arXiv ids were checked in one arXiv API query (all resolve, titles and first authors match).
- Claims checked against `research/mine_*.md` and, where a local extract exists, against the primary text: photonics-atlas `references/{lapointe2025, luke2020, luo2018, marinins2022, liu2024, liang2017, weigel2018, chowdhury2024, bao2024, koninck2025, franken2025, rees2020, kurczveil2013}/text.md`, eo-modulator-atlas `references/wang2024b/text.md`, lithium-niobate `research/R06-material-properties-data.md` and `R14-property-verification.md`.
- No local full text was available for Tkach 1986, Jalas 2013, Myers 1995, Weis and Gaylord 1985, Naranjo 2005, Xu 2022 (Nature), Kashyap 1988. Statements about them rest on Crossref metadata, the R06/R14 compilations, and general knowledge where noted.
- Formulas re-derived by hand (index ellipse, Airy, ripple depth, pyro field, coupled-mode doublet).

Verdicts: OK / wrong / unverifiable / citation mismatch. "Minor" marks items that are correct in substance but imprecise.

---

## 1. Ferroelectric course (ferroelectric.js)

### 1a. References (refs steps and course.references)

| location | claim | verdict | evidence | suggested fix |
|---|---|---|---|---|
| R.zelmon | Zelmon et al., JOSA B 1997, 10.1364/JOSAB.14.003319 | OK | Crossref: Zelmon, Small, Jundt, JOSA B 14, 3319 (1997), "Infrared corrected Sellmeier coefficients for congruently grown lithium niobate and 5 mol% MgO-doped lithium niobate". Paraphrased title acceptable | Optional: use the real title |
| R.weis | Weis and Gaylord, Appl. Phys. A 1985, 10.1007/BF00614817 | OK | Crossref: Appl. Phys. A 37, 191-203 (1985), title matches | none |
| R.ieee176 | ANSI/IEEE Std 176-1987, 10.1109/IEEESTD.1988.79638 | OK | Crossref: "IEEE Standard on Piezoelectricity" (no issued date in record; standard published 1988) | none |
| R.wangLT | Wang et al., Nature 2024, 10.1038/s41586-024-07369-1 | OK | Crossref: Nature 629, 784 (2024), title matches | none |
| R.liu2024 | Liu et al., OE 2024, 10.1364/OE.515173 | OK | Crossref: OE 32, 9433; real title "...based on the free-form curves on a thin-film lithium niobate platform" | none |
| R.liu2023 | Liu et al., Light Adv. Manuf. 2023, 10.37188/lam.2023.013 | OK | Crossref match | none |
| R.xie2026 | Xie et al., OE 2026, 10.1364/OE.588499 | OK | Crossref: OE 34, 20647; real title "...folded AWG on thin-film lithium niobate with 20 GHz fine channel spacing" | none |
| R.luke2020 | Luke et al., OE 2020, 10.1364/OE.401959 | OK | Crossref match (OE 28, 24452) | none |
| R.xu2021 | Xu et al., OE 2021, 10.1364/OE.418877 | OK | Crossref match (OE 29, 5497) | none |
| R.liang2017 | Liang et al., Optica 2017, 10.1364/OPTICA.4.001251 | OK | Crossref match | none |
| R.li2019 | Li et al., LPR 2019, 10.1002/lpor.201800228 | OK | Crossref match | none |
| R.lu2019 | Lu et al., Optica 2019, 10.1364/OPTICA.6.001455 | OK | Crossref match | none |
| R.shams2022 | Shams-Ansari et al., APL Photonics 2022, 10.1063/5.0095146 | OK | Crossref match | none |
| R.shen2024 | Shen et al., Nat. Photon. 2024, 10.1038/s41566-023-01370-2 | OK | Crossref match (18, 371) | none |
| R.herrmann2022 | Herrmann et al., Nat. Photon. 2022 | OK, but orphan | DOI matches; listed in ch3 refs but no ch3 text cites it | Cite it in ch3 (PR shows up as DC drift in coupled rings, per mine_photonics_atlas) or drop from ch3 refs |
| R.zhang2026 | Zhang et al., arXiv 2606.06637 | OK | arXiv API: "Reconfigurable Single-Ring Photonic Molecule on Lithium Niobate", Zhang, Garcia Primo, Liu, 2026-06-04 | none |
| R.wang2025 | Wang et al., ACS Photonics 2025, 10.1021/acsphotonics.5c00159 | OK, but orphan | Crossref match; not referenced in any chapter text or refs step (only in course.references) | Drop or use |
| R.luo2018 | Luo et al., Optica 2018 | OK | Crossref title exactly "Highly tunable efficient second-harmonic generation in a lithium niobate nanophotonic waveguide" | none |
| R.weigel2018 | Weigel et al., OE 2018, 10.1364/OE.26.023728 | OK (truncated title) | Crossref title continues "...exceeding 100 GHz 3-dB electrical modulation bandwidth" | Optional: full title |
| R.naranjo2005 | Naranjo, Gimzewski, Putterman, Nature 2005 | OK | Crossref: Nature 434, 1115 | none |
| R.bulmer1986 | Bulmer et al., APL 1986, 10.1063/1.96640 | OK | Crossref: Bulmer, Burns, Hiser, APL 48, 1036 | none |
| R.xu2022 | Xu et al., Nature 2022, 10.1038/s41586-022-05042-z | OK | Crossref: Nature 609, 496 | none |
| R.myers1995 | Myers et al., JOSA B 1995, 10.1364/JOSAB.12.002102 | OK (metadata) | Crossref: JOSA B 12, 2102, title matches | see 1d for the 21 kV/mm attribution |
| R.maeder2022 | Maeder et al., OL 2022 | OK | Crossref match (OL 47, 4375) | none |
| R.zheng2024 | Zheng et al., LPR 2024 | OK | Crossref match; real title "...for High-Fidelity and Power-Efficient Matrix Computation" | none |
| R.liu2026 | Liu et al., ACS Photonics 2026, 10.1021/acsphotonics.6c00668 | OK | Crossref match | none |
| R.yeh2026 | Yeh et al., Nano Lett. 2026 | OK | Crossref match (26, 13450) | none |
| R.holzgrafe2024 | Holzgrafe et al., OE 2024 | OK | Crossref match (32, 3619); only cited inside the biasDrift widget note | none |
| R.valdez2023 | Valdez et al., OE 2023 | OK | Crossref match | none |
| R.powell2024 | Powell et al., CLEO 2024 | OK, but orphan | Crossref match; in ch5 refs but no ch5 text uses it | Use (e.g. 300 C 5 h trapped-charge anneal, mine_eo_ln C1) or drop |
| ch6 refs | refs([R.ieee176, R.weis]) | minor | Chapter 6 (symmetry planes, PML, terminals) cites neither source | Replace with "unpublished lab notes" label, or drop the refs step content |
| ch0 refs narration | "All of them are real." | OK | All DOIs and arXiv ids resolve to the stated papers | none |

### 1b. Chapter content

| location | claim | verdict | evidence | suggested fix |
|---|---|---|---|---|
| ch0 reveal "The crystal" | Congruent LN, 3m, optic axis z, negative uniaxial | OK | standard | none |
| ch1 slide | x-cut: x normal, y and z in plane; z-cut: z normal | OK | standard | none |
| ch1 reveal "Rotate the crystal or the room?" | Some FE packages define ZXZ Euler angles as coordinate rotation; component transform is inverse of basis transform | OK | mine_blog_longform A4, A6 | none |
| ch1 reveal "Plus 45 is not minus 45" | 3m mirror planes contain z and are perpendicular to the x family; +45/-45 on x-cut not related by a crystal symmetry | OK | Consistent with standard LN setting (r22, d22 nonzero, r11 = 0 implies mirror perpendicular to x). Neither z -> -z nor y -> -y is in 3m or Laue -3m, so p14/p41 terms differ | none |
| ch1 reveal | Euler (-pi/4, pi/2, pi/2) used for (pi/4, pi/2, pi/2) | OK | mine_blog_longform A2 | none |
| ch1 reveal "Tests at 0 and 90" | Solver Euler 135 deg was the physical 45 deg device | OK | mine_blog_longform A3 | none |
| ch1 reveal "Notation and constants" | Voigt: no factor 2 for c, factor 2 on piezo d shear | OK | mine_blog_longform A7; standard | none |
| ch1 reveal | Solver expects upper triangle column-wise | OK | mine_blog_longform A8 | none |
| ch1 reveal | Low frequency sees unclamped, RF sees clamped; W&G tabulate both | OK | mine_eo_ln 5.4; R14 eps table | none |
| ch1 reveal | "Sign conventions for lithium niobate tensors follow IEEE Std 176-1987" | unverifiable / overclaim | lithium-niobate R06 item 3: "IEEE 176-1987 vs IRE 1949 wording not read (U)"; mine_eo_ln 5.11 is a curation note. Many LN tables use other frames | "Signs depend on the axis convention (for example IEEE Std 176-1987 vs the 1949 IRE rules); check that two sources use the same one" |
| ch1 check option 3 | x-cut vs y-cut photoelastic differs at the same angle (logged example) | OK | mine_blog_longform A11 (0.304 vs 0.223, Andrushchak 2009 values) | none |
| ch1 widget eulerSign | Active vs transpose: agree at 0 deg, differ in sign of y component elsewhere; at 90 deg only the sign differs | OK | Code: y = -s sin(a), z = cos(a). Correctly notes that a which-axis test passes at 90 deg | none |
| ch1 narration vs widget | "At zero and ninety degrees, many sign and transpose errors cancel" | OK (minor) | True for even-rank tensors (index, elastic, photoelastic); for odd-rank (piezo, EO) the 90 deg case flips sign, which the widget shows | Optional: "for even-rank tensors" |
| ch2 slide + widget xcutIndex | 1/n_TE^2 = cos^2(theta)/n_e^2 + sin^2(theta)/n_o^2, theta from y; TM = n_o | OK | Extraordinary-wave index with angle to optic axis phi = 90 deg - theta; TM along x is ordinary | none |
| ch2 slide, pool Q1, widget | n_o 2.211, n_e 2.138 at 1550 nm (Zelmon 1997) | OK | lithium-niobate R06 row: Zelmon CLN 1550 nm, 21 C: 2.2111 / 2.1376 (recomputed from Sellmeier). The 2026 Zelmon erratum affects only the MgO5 rows | none |
| ch2 chat, narration | Bulk TE index spans about 0.07 | OK | 2.2111 - 2.1376 = 0.0735 | none |
| ch2 reveal "A crossing at about 25 degrees" | Wang et al. 2024: TE/TM cross at 25 deg on x-cut LN; LT no crossing | OK | eo-modulator-atlas wang2024b text: "crossing of the fundamental TE and TM modes at an angle of 25 deg, whereas no mode crossing is found" | none |
| ch2 reveal "Gentler can be worse" | Liu et al. 2024 note larger bend radii can increase conversion and loss | OK (minor) | liu2024 text p.2 states it in the introduction, citing its ref. [27]; it is not Liu's own measurement | "Liu et al. (citing earlier work) note..." |
| ch2 reveal, ch2 check 2 correct option, pool Q3 correct answer | Larger radius = slower, adiabatic passage through the anticrossing, so more conversion | unverifiable (course inference stated as fact) | The reveal labels it "one way to picture it", but the check and exam correct answers state the mechanism as fact. Not in Liu 2024 or the notes | Keep the picture in the reveal; in the check and exam answers use "consistent with" / "a plausible mechanism is" or test only the sourced fact |
| ch2 reveal "Sidewalls are involved" | Liu 2023: anisotropy plus angled sidewalls cause hybridization | OK | mine_eo_ln 5.2 (liu2023 p.2, S) | none |
| ch2 reveal "Arrays of angles" | Xie 2026: in-plane anisotropy as obstacle for narrow-spacing AWGs | OK | mine_photonics_atlas cluster 1 (xie2026 p.1-2) | none |
| ch2 widget offset | "Illustrative waveguide offset (TM below TE)" shifts only TM | OK | Labeled illustrative and "not computed" | none |
| ch3 slide | PR blue-shifts LN resonances; heating red-shifts | OK | luke2020 text; standard | none |
| ch3 slide narration | Luke 2020 scanned long to short at under 20 uW to follow the blue shift | OK | luke2020 text: "< 20 uW ... scanned the laser from long to short wavelength, so that the laser scan would follow the power dependent photorefractive blue shift" | none |
| ch3 slide, widget prSweep | Scanning against the shift narrows, with it widens | OK | Standard thermal-triangle logic; widget model correct for both signs | none |
| ch3 widget note | PR response much slower than thermal | OK (general) | Notes C2: PR effects slow (seconds to weeks). No specific number quoted | none |
| ch3 reveal "Cladding" | Xu 2021: oxide cladding worsens PR; removal plus heat treatment suppressed relative shift by > 2 orders | OK | mine_photonics_atlas cluster 1 (xu2021d p.1) | none |
| ch3 reveal | Shen 2024 SI: PECVD oxide significant PR, annealed HSQ to air-clad levels | OK | mine_eo_ln 2.2 (shen2024 p.21) | none |
| ch3 reveal "Small cavities" | Liang 2017: about 84 MHz per photon | OK | liang2017 text: "~0.64 GHz/aJ, or equivalently ~84 MHz/photon" | none |
| ch3 reveal | Li 2019: PR quench at 2 mW | OK | mine_photonics_atlas (li2019 p.4) | none |
| ch3 reveal | Shams-Ansari 2022 pump-probe faster than PR | OK | notes (p.1-2) | none |
| ch3 reveal | Lu 2019 pump/SH PR mismatch; Zhang 2026 PR grating TE/TM, hour lifetimes | OK | notes | none |
| ch4 slide, widget, pool | E = p dT / (eps0 eps33) | OK | Open-circuit slab, D = 0: eps0 eps33 E + dPs = 0. Use eps33^T (static); 28 is inside the 28.1-30 range (R14) | none |
| ch4 widget pyroCalc note + buttons | p range 4.0e-5 to 9.5e-5 C/(m^2 K) "from literature compilations (see Weis and Gaylord, 1985)" | citation mismatch | R14 row 28: W&G give -4e-5 (Savage 1966). The 9.5e-5 is Popescu et al., J. Appl. Phys. 113, 043101 (2013) (R06, secondary page); cti1999 gives -8.3e-5 | "4e-5 (Weis and Gaylord 1985, after Savage 1966) to about 9.5e-5 (Popescu et al., J. Appl. Phys. 2013)" |
| ch4 widget, takeaway | Coercive field about 21 kV/mm for congruent LN (Myers et al. 1995) | value OK, attribution unverifiable | R14 row 18: 21 kV/mm from Volk and Wohlecke 2008 Table 7.1 p.185. R06: "claimed: Myers et al. 1995 ... Abstract does not state it" (U). Myers body text not held locally | Cite Volk and Wohlecke 2008 (10.1007/978-3-540-70766-0, Table 7.1), or confirm the page in Myers 1995 |
| ch4 numbers | 95 K step: 15-36 kV/mm, 7.5-17.8 kV across 0.5 mm | OK | Recomputed: 4e-5 x 95 / (8.854e-12 x 28) = 15.3 kV/mm; 9.5e-5 gives 36.4 kV/mm (matches mine_eo_ln 2.7) | none |
| ch4 check option | 10s of K give kV/mm | OK | 10 K: 1.6-3.8 kV/mm | none |
| ch4 check option | LN thermal conductivity lower than Si | OK | standard (about 4-5 vs about 150 W/(m K)) | none |
| ch4 reveal "Hotplates" | Fried devices, cracked substrates, foil discharge path | OK | mine_ofs_61 A1 (practitioner report) | none |
| ch4 reveal "Hotplates" | Heating z-cut LN reported to expose e-beam resist | unverifiable (no citation on screen) | mine_blog_longform E3: his note citing "Mostafa et al. 2006, no DOI in post" | Add "(practitioner report)" or a verified citation |
| ch4 reveal "Cryostats", chat 3 | Cooling to 4 K is nearly 300 K of change | OK (minor caveat) | Arithmetic correct; but p3 falls with temperature, so the charge from a cooldown is less than p(300 K) x 296 K | Optional: avoid implying room-temperature p over the whole range |
| ch4 reveal "Cryostats" | Cross-chip discharge; shielding electrodes; transducers on LN failed; x-cut membrane collapse hypothesis; Al straps fixed it | OK | mine_ofs_61 A3, A4 (his hypothesis, labeled as such on screen) | none |
| ch4 reveal "Measurements" | Bulmer 1986: Ti-indiffused z-cut instabilities, x-cut stable | OK | mine_eo_ln 2.8 (abstract-level) | none |
| ch4 reveal | Luo 2018: SHG thermal tuning slope above simulation attributed to pyroelectric and thermal expansion | OK | luo2018 text: "by pyroelectric [42] and thermal expansion effects [43]" | none |
| ch4 reveal | Weigel 2018: pyroelectric charge build-up a concern of etching LN | OK | weigel2018 text: "heat and pyroelectric charge build-up" | none |
| ch4 reveal | Naranjo 2005: warmed LiTaO3, tungsten tip, deuterium fusion | OK | Crossref title; mine_ofs_01 B1; tungsten tip from general knowledge (not in notes) | none |
| ch4 reveal | Xu 2022: fs pulses write domains; attributed to thermoelectric field from local heating | OK (not verified locally) | mine_ofs_01 B2 is his comment; consistent with the Nature abstract as recalled; no local text | Optional: check abstract wording |
| ch5 slide, check, pool | Liu 2026: Vpi > 40 V at <= 10 kHz; first-order high-pass extraction | OK | mine_eo_ln 1.1 (liu2026d p.5) | none |
| ch5 slide, chat | Yeh 2026: 30 V, 36 h, 4.5x drop, diode-like, about 3 weeks | OK | mine_eo_ln 1.6 | none |
| ch5 reveal | Maeder 2022: TO within 0.5 dB over 33.3 min | OK | notes | none |
| ch5 reveal | Liu 2026 heater: ER 25 to 21 dB over 100 min (constant V), no systematic drift over 95 min (constant I) | OK | mine_eo_ln 1.11 | none |
| ch5 reveal | Zheng 2024: 1 kHz refresh | OK | notes | none |
| ch5 reveal | Valdez 2023 laser-tuned quadrature | OK | mine_eo_ln 1.8 | none |
| ch5 reveal "Which V pi?" | factors of 2; -3 dB EO = drive power x2; -6 dB = Vpi x2 | OK | mine_eo_ln 3.1, 3.4; arithmetic correct | none |
| ch5 reveal "Clamped and unclamped" | W&G list r33 about 30.8 pm/V clamped vs 32.2-34.0 unclamped at 633 nm; several percent | OK (minor) | mine_eo_ln 5.4: 30.8 is the Turner row, 28.8 the Kaminow row; gaps -4.3 to -6.7 % | "28.8-30.8 pm/V clamped" |
| ch5 widget biasDrift note | "Real tau ranges from microseconds to hours ... (see Liu 2026; Holzgrafe 2024)" | unverifiable (upper end) | Notes: TFLN tau 0.85-2.41 us (Liu 2026), 0.1-1 ms, 88.6 us-11.9 ms (li2026photonics); hours appear only for bulk LN dark conductivity (6.9 h-29 d) and bulk-LN PR drift. Neither cited source supports hours | "microseconds to milliseconds in thin film (Liu 2026); hours to days predicted for bulk LN" with matching cites, or drop "hours" |
| ch5 widget | Maxwell-Wagner phi0 [r + (1 - r) exp(-t/tau)] | OK | Labeled illustrative; sign flip with stack per mine_eo_ln 1.4 | none |
| ch6 slide, video, pool | Quarter model Q 2e10 to 7.1e13; full model about 2200 | OK | mine_blog_longform B1 | none |
| ch6 slide, reveal, check | Weakening PML raised Q | OK | mine_blog_longform B2 | none |
| ch6 reveal "Cyclic symmetry" | 55 vs 22 min | OK | B3 | none |
| ch6 reveal, chat, pool | Voltage terminal = short (setpoint ignored) = resonance; Charge terminal Q0 = 0 = open = anti-resonance; k^2 from both | OK | B4; physically correct | none |
| ch6 reveal "Pipelines" | "mirror-and-join dataset operation in the wrong order returned zero" | minor wording | B8: "select -> mirror -> join" gave zero, "mirror -> join" worked (an extra step, not order) | "an extra selection step before a mirror-and-join returned zero" |
| ch6 reveal | Surface integral missed hole sidewalls; output overwritten; taper one side instead of two | OK | B9, B11, B6 | none |
| ch6 check 1 | Quarter model of an "x-cut" resonator | minor | B1 records the frame as the "bender" coordinate, not a stated cut | Drop "x-cut" or say "rotated-frame" |

### 1c. On-screen text hygiene (ferroelectric)

| location | claim | verdict | evidence | suggested fix |
|---|---|---|---|---|
| ch5 subtitle, objectives, slide bullets, callout, chats, check text, takeaways, pool items, biasDrift button "Hold constant D C voltage" | "D C", "R F" spelled with a space on screen | minor (TTS spelling leaked to screen) | lines 998-1001, 1027-1032, 1064, 1098-1099, 1112, 1119-1121, 1316-1321, 1564 | Write "DC", "RF" in on-screen fields; let tts() expand narration only |
| ch2 keyframe captions and animRing canvas | "T E", "T M" on screen ("x-cut ring, T E launched", "At one angle, T E meets T M.", canvas "now mostly T M") | minor | lines 164, 603, 605 | Use "TE", "TM" on screen |
| all on-screen fields | file paths, DevLog refs, cluster ids, emoji | OK | grep: none found. Fictional codes (FE-0042, PHC-17.3, "Cryostat run 42") are parody. "Logged case" items are uncited private lab notes | Optional: "(unpublished lab notes)" |

---

## 2. Feedback course (feedback.js)

### 2a. References

| location | claim | verdict | evidence | suggested fix |
|---|---|---|---|---|
| R.tkach | Tkach and Chraplyvy, JLT 4, 1655 (1986), 10.1109/JLT.1986.1074666 | OK | Crossref: JLT 4, 1655-1661 (1986), "Regimes of feedback effects in 1.5-um distributed feedback lasers" | none |
| R.marinins | Marinins et al., JSTQE 2022, 10.1109/jstqe.2022.3223641 | OK (truncated title) | Crossref title ends "...with sub-300nm Alignment Precision"; print issue is vol 29 no 3, 2023 (text header) | Optional: full title, 2023 |
| R.kurczveil | IEEE Photon. J. 2013 | OK | Crossref match | none |
| R.moscoso | CLEO-PR 2017 | OK | Crossref match | none |
| R.dekoninck | Nature 2025 | OK | Crossref: Nature 637, 63 | none |
| R.bao | LPR 2024 | OK | Crossref match | none |
| R.celo | GFP 2017, no title | OK | Crossref: "Optical proximity correction in geometry sensitive silicon photonics waveguide crossings" | Optional: add title |
| R.tian | Opt. Quantum Electron. 2025 | OK | Crossref: "Silicon-based star coupler power splitter with enhanced non-uniformity" | Optional: add title |
| R.cherchi | OE 2014 | OK | Crossref: "Unconstrained splitting ratios in compact double-MMI couplers" | Optional: add title |
| R.vaerenbergh | OE 2021 | OK | Crossref: OE 29, 37021 (vertical grating couplers) | Optional: add title |
| R.contreras | Meas. Sci. Technol. 2023 | OK | Crossref match | none |
| R.vanrees | OE 2020 | OK | Crossref match (OE 28, 5669) | none |
| R.xu2021 | Sci. Rep. 2021 | OK | Crossref match | none |
| R.franken | APL Photonics 2025 | OK | Crossref match | none |
| R.voloshin | arXiv 2411.19264 | OK metadata, orphan | arXiv API: title and first author match. No sentence in ch3 (or anywhere) uses it, and it is not in any research note | Drop from ch3 refs, or add the sourced claim it supports |
| R.jalas | Nat. Photon. 2013, "What is - and what is not - an optical isolator" | OK | Crossref: Nat. Photon. 7, 579 (2013), Jalas, Petrov, Eich + 10 | none |
| R.lapointe | Nat. Photon. 2025 | OK (metadata) | Crossref match | see 2b for the numbers |
| R.kashyap | Electron. Lett. 1988 | OK | Crossref: 24, 47-49 | none |
| R.ji2017, R.hausmann, R.raja, R.white, R.chen2026, R.chowdhury | as cited | OK | Crossref matches all | none |
| R.kramnik | arXiv 2411.05921 | OK | arXiv API match | none |
| ch2 refs, ch6 refs | include 'marinins' | minor orphan | Neither chapter mentions Marinins | Drop |
| ch4 refs | include 'moscoso' | minor orphan | Moscoso is used in ch1 only | Drop from ch4 |

### 2b. Chapter content

| location | claim | verdict | evidence | suggested fix |
|---|---|---|---|---|
| ch0 reveal "Parasitic cavity", ch2 slide, pool | FSR = c / (2 n L), n group index | OK | standard | none |
| ch0 check, ch1 reveal | Tkach and Chraplyvy: five regimes on 1.5 um DFB; I phase-dependent narrowing/broadening; II external-cavity mode splitting/hopping; III narrow single-mode window, phase-insensitive; IV coherence collapse, distance-insensitive; V AR-coated facet, stable ECL | OK (not in research notes) | Crossref confirms paper; regime descriptions match the paper's widely cited abstract. Course correctly avoids threshold numbers | Optional: add Tkach to the research notes with page locators |
| ch1 slide, check, pool | Marinins: L-I ripple period 25-30 mA; 4 % (-14 dB) from far-end SiN grating coupler; gap < 1 um too short | OK | marinins2022 text: "one L-I curve ripple period (25-30 mA) ... A backreflection strength of 4% (-14 dB) is estimated" | none |
| ch1 cold open caption, ch1 check 1, pool Q2 | "flip-chip ... onto a silicon nitride chip" / "In the published case" | minor mismatch | Marinins bonded InP DFBs onto imec 300 mm Si photonics; the waveguide and grating coupler were SiN on that platform (ch1 slide states this correctly) | "onto a silicon photonics chip (SiN waveguide)" |
| ch1 policy slide, pool | Kurczveil: cited requirement < -50 dB, measured < -41 dB | OK | kurczveil2013 text: "reflections below 41 dB"; "have to be below 50 dB [9]" (cited, as the course says) | none |
| ch1 policy slide | Moscoso: sensitivity depends on distance; isolator at chip output | OK | notes (p.1) | none |
| ch1 chat 2 | Bare glass-air end reflects a few percent | OK | Fresnel n = 1.45: 3.4 % | none |
| ch1 check option | Feedback effects set by returned fraction, not absolute power | OK (first order) | standard for linear feedback regimes | none |
| ch2 slide, widget fabryPerot | Lossless Airy T = (1-R1)(1-R2) / (1 + R1 R2 - 2 sqrt(R1 R2) cos phase) | OK | Re-derived; phase = 2 pi f / FSR is the round-trip phase | none |
| ch2 slide, widget | Peak-to-valley 20 log10[(1+r)/(1-r)], r = sqrt(R1 R2); d-lambda = lambda^2/(2 n L) | OK | Tmax/Tmin = (1+r)^2/(1-r)^2 | none |
| widget fabryPerot | Worst-case reflection (r1+r2)^2/(1+r1 r2)^2 | OK | 1 - Tmin, re-derived | none |
| ch2 check 1 | 0.1 nm at 1550 nm, n = 4: about 3 mm; 3 um gives about 100 nm; 30 cm gives about 1 pm | OK | 2.4e-12 / (8 x 1e-10) = 3.0 mm; others recomputed | none |
| ch2 reveal, check 2, pool | Bao 2024: parasitic lasing at 1531.5 nm from facet back-reflections | OK | bao2024 text: "parasitic lasing at the peak fluorescence wavelength of 1531.5 nm sets in" | none |
| ch2 reveal | Celo 2017 crossings; Tian 2025 splitter trees | OK | notes (stated in intros) | none |
| ch2 reveal | Cherchi 2014: > 30 modes needed | OK | notes (p.4) | none |
| ch2 reveal | Vertical GC large back-reflection (Van Vaerenbergh 2021) | OK | notes (p.1) | none |
| ch2 reveal | Offset splice MZI strain sensor; accidental-offset sentence labeled as course inference | OK | mine_ofs_01 A3 | none |
| ch2 reveal "The dark port" | 1x2 combiner: no second guided port; light radiated, part reflected | OK | mine_ofs_01 A1; mine_blog B18 | none |
| ch3 slide, widget modeHop, takeaways | Comb vs filter; hop at about half FSR; hysteresis margin | OK | Labeled illustrative; physically reasonable | none |
| ch3 reveal, check 2, pool | Mode-hop-free bounded by about one cavity FSR; van Rees 0.22 nm (28 GHz), six-fold | OK | rees2020 text: "six-fold increased continuous and mode-hop-free tuning range of 0.22 nm (28 GHz) as compared to the free-spectral range" | none |
| ch3 reveal | Xu 2021 cavity-phase setting | OK | notes (p.7) | none |
| ch3 reveal | Franken 2025: 58 h, 4.4 MHz/h | OK | franken2025 text | none |
| ch4 policy slide, check, pool | An isolator must break Lorentz reciprocity (Jalas 2013); LTI non-magnetic structures reciprocal, S symmetric | OK (minor wording) | Jalas condition is linear, time-invariant, symmetric permittivity and permeability tensors; "non-magnetic" is shorthand (an unbiased magnetic material is reciprocal) | Optional: "no magneto-optic bias" instead of "non-magnetic" |
| ch4 policy slide, ch4 check option | "magneto-optic ... time modulation, or nonlinearity" as routes to isolation | misleading to an expert | Nonlinearity breaks reciprocity but nonlinear isolators only work in a power window and fail with simultaneous counter-propagating signals (dynamic reciprocity; Shi, Yu, Fan, Nat. Photon. 2015). Not checked against Jalas full text | Add "(nonlinear devices only partially: power-dependent, and limited by dynamic reciprocity)" |
| ch4 policy slide bullet 5, narration | "launch the time-reverse of the forward output, in all modes" | misleading to an expert | Conflates time-reversal symmetry with Lorentz reciprocity; Jalas distinguishes them (lossy devices are reciprocal but not time-reversal symmetric). The correct reciprocity test is backward excitation in the forward output mode(s), counting all ports and modes | "send light back in the mode(s) the forward light exits in, count all ports and modes, and see what reaches the source" |
| ch4 reveal "Why the reflection comes home", pool | Backward launch in the forward output mode returns to the input mode | OK | S_ij = S_ji | none |
| ch4 reveal "Real isolators" body and narration | Lapointe 2025: "1 dB insertion loss and 25 dB isolation" | wrong | lapointe2025 abstract: ">25 dB isolation ratio and <1.5 dB insertion loss over the entire optical telecom C-band"; the Faraday rotator alone < 0.15 dB. The "~1 dB" in the text describes a target prior work missed. mine_ofs_01 C10 carried the error | "> 25 dB isolation and < 1.5 dB insertion loss across the C-band" |
| ch4 reveal, check 2, pool | Snell: exit angle larger than waveguide tilt | OK | n_eff > n_out | none |
| ch4 reveal | Kashyap and Blow 1988: fiber fuse travels toward the source | OK | Crossref title; mine_ofs_61 B10 | none |
| ch5 slide | Splitting = twice the coupling rate | OK | With da/dt coupling i beta, eigenvalues +/- beta | none |
| ch5 widget doublet | t = 1 - kappa_e D/(D^2 + beta^2), r = i beta kappa_e/(D^2 + beta^2), D = kappa/2 - i Delta; eta = 0.5 critical | OK | Re-derived from CW/CCW coupled-mode equations; code arithmetic checked | Minor: "Single-Lorentzian Q / true loaded Q" is a half-depth-width estimate, not a fit; label it so |
| ch5 reveals, checks, pool | Ji 2017 doublets; Hausmann 2014 EBL seams; Raja 2019 SIL resonances split; White 2024 nonlinear detuning; Kramnik 2024 thermal cliff, hot-to-cold sweep; Chen 2026 Fano vs doublet | OK | notes (all "stated", with page locators) | none |
| ch5 check 2 option | Backscatter fraction does not grow with power | OK (first order) | Linear Rayleigh backscatter | none |
| ch6 reveal, check 2 | De Koninck 2025: no isolator, linewidth overestimated | OK | koninck2025 text: "yielding to an overestimate of the laser linewidth" | none |
| ch6 reveal, check 1 | Chowdhury 2024: 160 vs 120 pm vs 110 pm simulated; heating red shift | OK | chowdhury2024 text: FSRs as stated; heating was the authors' hypothesis, then tested by current and stage-temperature experiments | none |

### 2c. On-screen text hygiene (feedback)

| location | claim | verdict | evidence | suggested fix |
|---|---|---|---|---|
| all on-screen fields | file paths, DevLog refs, cluster ids, emoji | OK | grep: none. Fictional codes (FB-1550, PHC-1550.4) are parody. The hint label uses a Unicode left arrow (not an emoji) | none |

---

## 3. Top-10 must-fix

1. Feedback ch4 "Real isolators" (body and narration): Lapointe 2025 numbers are wrong. Paper: > 25 dB isolation and < 1.5 dB insertion loss across the C-band, not "1 dB".
2. Ferroelectric pyroCalc note and buttons: the 9.5e-5 C/(m^2 K) upper bound is not from Weis and Gaylord (they give -4e-5, after Savage 1966). Cite Popescu et al., J. Appl. Phys. 2013 for 9.5e-5.
3. Ferroelectric pyroCalc and ch4: the coercive field of 21 kV/mm is correct (Volk and Wohlecke 2008, Table 7.1), but the Myers 1995 attribution is unverified (its abstract does not state it; local notes mark it U). Cite Volk and Wohlecke, or confirm the page in Myers.
4. Feedback ch4 policy slide and narration: the "time-reverse of the forward output" test conflates time reversal with Lorentz reciprocity. Reword as a backward launch into the forward output mode(s), counting all modes.
5. Feedback ch4 policy slide and check: nonlinearity is listed as an equal route to isolation. Add the caveat: power-window only, limited by dynamic reciprocity.
6. Ferroelectric biasDrift note: "tau ranges from microseconds to hours" is not supported by the cited Liu 2026 or Holzgrafe 2024. Thin film is us to ms; hours to days applies only to bulk LN.
7. Feedback ch3 refs: Voloshin et al. (arXiv 2411.19264) is an orphan. No claim uses it and no research note contains it. Remove it or add the claim it supports.
8. Ferroelectric ch2 check 2 and exam Q3 correct answers state the slow-adiabatic-passage mechanism as fact. It is course inference; Liu 2024 only notes the effect, citing earlier work. Soften the wording or test only the sourced fact.
9. Feedback ch1 cold open, check 1 and exam Q2 describe Marinins as a "silicon nitride chip". It was InP DFBs on imec Si photonics with SiN waveguides and grating couplers.
10. Ferroelectric on-screen text shows TTS spellings "D C", "R F", "T E", "T M" (ch5 subtitle, objectives, bullets, callout, chats, checks, takeaways, exam, widget button; ch2 captions and canvas). Use DC, RF, TE, TM on screen and expand only in narration.

Also worth fixing, lower priority:
- Ferroelectric ch1: "LN sign conventions follow IEEE Std 176-1987" is an overclaim; the frames used in the literature differ.
- Orphan refs: herrmann2022 (ch3), powell2024 (ch5), wang2025 (course list only), ieee176 and weis in ch6; marinins in feedback ch2 and ch6; moscoso in feedback ch4.
- Ferroelectric ch5: the clamped r33 is 28.8-30.8 pm/V in W&G, not just 30.8.
- Ferroelectric ch6: the "mirror-and-join in the wrong order" bug was really an extra select step.
- Ferroelectric ch4: the resist-exposure-on-heating claim has no citation.
- Ferroelectric ch4: the "nearly 300 K" cooldown ignores that p3 falls at low temperature.
