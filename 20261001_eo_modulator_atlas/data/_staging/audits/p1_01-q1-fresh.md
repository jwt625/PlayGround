# Audit Q1 (fresh context): batch p1_01 (deng2026, qiu2026, ogiso2024, porto2026, tanaka2026)

Auditor: independent subagent, 2026-10-01. Read-only: no data or staging file edited; only this report written. No network. Sources used: references/<id>/text.md, crossref.json, source.pdf (re-rendered at 3-6x for Figs. 4 of deng2026, 1-4 of ogiso2024, Fig. 3 of porto2026, Figs. 4-6 of tanaka2026) and figures/*.png; qiu2026 is text-only (checked: nothing in the rows, evidence or notes is claimed from a figure image; Fig. 1b/2b/3d statements come from captions and running text only).
Mechanical check (script, uv + pyyaml): every non-empty evidence:true cell has an evidence entry with the same value; no evidence entry points at an empty cell or an unknown column; no qualifier sits on an empty field; no absolute paths, home paths or emoji in any staging file or in sims/deng2026/config.yaml. All clean.
Severity: high = wrong/misleading in a way a plot or comparison would show; medium = convention or completeness defect that changes interpretation; low = cosmetic, locator, judgment call.

---------------------------------------------------------------------
## 1. deng2026 (Light Sci. Appl. 15, 21)

Identity: title, 19 authors in order (match Crossref), journal/volume/article 21, DOI, published_on 2026-01-02 (Crossref issued/published-online/created all 2026-01-02), CC-BY-4.0 (Crossref vor+tdm licenses and the p.1 notice), received/revised/accepted dates, affiliations 1-6 (p.9 Author details) all verified. PDF page index = journal page (no cover sheet); locators for Fig. 4 (p.8), Sec. On-chip EO modulator (p.7), Methods (p.9) are correct. Orgs: Tsinghua University, Zhejiang University (existing), Institute of Physics, Chinese Academy of Sciences (research_institute, CN): correct per affiliations 1-6; Taizhou Institute handling in notes is fine.

### Findings

| # | Sev | Field | Data says | Source says | Suggested correction |
|---|-----|-------|-----------|-------------|----------------------|
| D1 | medium | evidence basis of drive=single_ended and vpi_convention=mzm_single_arm | basis `measured`, locator p.7 / Fig. 4d | p.7: GSG arm carries RF + DC, other arm GS capacitive electrode with low DC bias (stated). The paper never says which arm the 100 kHz triangular sweep (Methods p.9) drives, nor that the quoted 7 V is a single-arm Vpi | Convention (f)/(h): inferred conventions need basis `derived` plus note "authors do not state which arm the Vpi sweep drives". `drive` can stay measured-from-text for the layout; `vpi_convention` is the inference. |
| D2 | low | sims config sin_strip width 1.4 um (figure_digitized) | 1.4 um from the Fig. 4c sketch scaled by the 5.5 um gap | Fig. 4b SEM: strip top 257 px vs 1 um bar 155 px, i.e. about 1.65 um (tilted view, approximate); Fig. 4c with the 5.5 um gap as scale: strip 73 px / gap 250 px x 5.5 = about 1.6 um. I cannot reproduce 1.4 | Use 1.6-1.7 um or state the range 1.4-1.7 and name the SEM bar as a second scale; add Fig. 4b scale bar to `missing` item text (currently says "Fig. 4c sketch only"). |
| D3 | low | config r_pm_per_v r42=358, class paper_exact | exact value | Abstract/p.5/Discussion: "exceeding 358 pm/V", calculated from an effective coefficient on a separate 180 nm film (p.5 Fig. 3c: 253 pm/V along <110>) | Bound from a different film: class project_inference (or paper_exact with an explicit ">" and film note). |
| D4 | low | config gold sigma 4.1e7, class standard_reference | note says "not verified against a source" | n/a | Same defect as pilot S2/X9: standard_reference with no citation. Use an unverified flag or cite a source. |
| D5 | low | config provenance for Fig. 4c geometry | only the Fig. 4a scale inconsistency is flagged | Fig. 4c 2 um bar is about 65 px while the 5.5 um electrode gap is about 250 px (implies about 7.7 um), and the drawn 0.5 um electrode thickness would read about 0.8 um at the bar scale | Add the 4c scale-bar inconsistency to the electrode/strip notes; the 5.5 um text value was correctly preferred. |
| D6 | low | evidence epitaxy_or_stack / device notes "approx 40 nm self-buffer layer" attached to the 300 nm device film | stack line for the device | p.4/p.5 (Fig. 2a caption): ~40 nm self-buffer region observed in the 180 nm characterization film; Methods p.9: "initial 40 nm of growth" at modified conditions | State that the 40 nm figure is from the 180 nm film and presumed for the device film. |
| D7 | low | evidence vpi_dc_v note "100 kHz triangular drive" | locator "p.7; Fig. 4d" | the 100 kHz detail is on p.9 Methods ("Electro-optic modulator characterization") | Add "p.9 Methods" to the locator. |

### Judgment calls checked and accepted
- vpi_dc_v=7 / vpil_dc_vcm=0.7 in the DC columns with mzm_single_arm: Vpi 7 V and 0.7 V*cm are explicit on p.7 and Fig. 4d (dashed lines at about -1 and +6 V on the x axis, 7 V span, verified). 7 V x 0.1 cm = 0.7 correct; basis derived for the product correct; simulated 0.73 correctly kept out of the columns.
- bw3db_ghz=12 (approx) and bw6db_ghz=28 (no qualifier): Fig. 4e re-read: response about 0 dB to 12 GHz, step to about -6 dB at about 12.5 GHz, plateau -5 to -6 dB to about 27 GHz, -9 dB at 30 GHz; dashed lines at -3 and -6 dB; text "approximately 12 GHz" and "6 dB EO bandwidth located at 28 GHz". Both crossings are observed inside the plot (30 GHz); not entering bw_measured_to_ghz is correct (the 40 GHz bias-T limit is Methods p.9, plot ends at 30 GHz). eo_rolloff 9 dB at 30 GHz (approx, extracted_from_figure) matches the plot end.
- Fields correctly left empty: IL, ER, propagation loss, RF loss, Z0, n_rf, ng, Si3N4 width, signal/ground widths, bias, energy, vpi_rf_v (grep for loss/extinction/dB shows only bandwidth/S11 statements).

### Sim config (provenance and `missing`)
Provenance classes are honest overall: eps_r for BTO/LSAT/Si3N4, Si3N4 index, strip width, electrode widths, signal side, LSAT thickness, BCB extent, DC bias, temperature are all in `missing` and omitted rather than guessed; the paper-stated values used (300 nm film, 260 nm Si3N4, 5.5 um gap, 0.5 um Au, 1 mm, 50 ohm load, n=2.26/1.99, <110> field) are all verified on p.2, p.7, p.9. Targets are correctly tied to evidence entries (0.7 measured/derived, 12, 28) and the paper's 0.73 FEM value is labelled basis simulated. Only D2-D5 above need attention. The claim "n = 2.26 paper_exact" is the introduction's bulk-type value (p.2) and the note already says isotropic.

### Sampled values verified directly (>=12)
Vpi 7 V and VpiL 0.7 V*cm (p.7, Fig. 4d); 0.73 V*cm and 45% overlap (p.7); 1 mm arms (p.7); 300 nm BTO (p.7); 5.5 um gap (p.7); 12 GHz and 28 GHz (p.7-8, Fig. 4e); S11 about -15 dB (p.8); 260 nm Si3N4 PECVD, >2 um BCB, 10 nm Ti + 500 nm Au (p.9); (001) LSAT, PLD (p.9); 67 GHz VNA with bias-T limited to 40 GHz (p.9); 100 kHz triangular sweep (p.9); n=2.26/1.99 (p.2); 40 nm self-buffer (p.4-5); r42 > 358 pm/V (abstract); Tc 200 C (abstract); Fig. 4a axes <110> / <1-10>; Fig. 4c "Air" label; received/revised/accepted dates; 19 authors; 6 affiliations.

### Verdict: ACCEPT WITH CORRECTIONS (no wrong data value; D1 should be fixed, D2-D7 cosmetic/config).

---------------------------------------------------------------------
## 2. qiu2026 (OFC 2026 Th4B.3, text-only)

Identity: title, 15 authors in order, venue string, page Th4B.3, DOI match Crossref. Crossref: issued/published year 2026 only, no license, no published-online: empty published_on and empty license with restricted_local_only are the correct reading. Affiliations (McGill ECE; Ciena Quebec City and Ottawa; Lumiphase AG, Stafa) match the text. Lumiphase AG already exists in data/organizations.csv, so not re-adding it is right; McGill and Ciena are new and correct.

### Findings

| # | Sev | Field | Data says | Source says | Suggested correction |
|---|-----|-------|-----------|-------------|----------------------|
| Q1 | low | papers.csv authors | "Francois Pelletier" | Crossref: "François Pelletier" (text.md extraction is garbled "Franc¸ois") | Use the Crossref spelling with the cedilla (authors "as in the source"); same check for any other diacritics. |
| Q2 | low | devices.csv tags `linear_driver` | tag present | Text: "differential RF drivers with up to 30 dB gain at 100 GHz"; never calls them linear; Sec. 3 says non-linear equalizers are needed because of driver roll-off | Drop the tag or replace with `rf_driver_packaged`. |
| Q3 | low | max_line_rate_gbps | empty | Title, Sec. 1 and Sec. 5 state "448 Gbps PAM4" per lane (225 GBd x 2 = 450 raw) | Either enter 448 (basis author-stated, note "title/Sec. 1/5 figure, not the 422 net") or add one sentence to notes that a stated 448 line-rate figure was deliberately not entered. Notes currently mention it only as a title/net mismatch. |
| Q4 | low | il_onchip_db basis `derived` | 3 dB, evidence entry says "equals 11.5 minus 6 minus 2 x 1.25; method not stated" | Sec. 2 states "3 dB on-chip IL" as one component of an itemised 11.5 dB PIC budget; the authors give no formula | The value is paper-stated (11.5 = 6 + 2.5 + 3 checks). If kept `derived`, the arithmetic belongs in the `derived` list, not in `entries`; otherwise basis `measured`/unspecified with the budget in the note. |
| Q5 | low | il_fiber_to_fiber_db empty | 12.5-14 dB in notes/context only | Sec. 2: connectorised fiber-to-fiber IL 12.5-14 dB across 4 channels (includes 1-2.5 dB splice loss) | Acceptable (a range cannot go in a float cell); keep, but say in notes that no single value was entered because it is a per-channel range. |

### Checked and correct (no value error found)
Vpi 6 V differential (Sec. 2) with mzm_differential/differential (Sec. 5 "high-speed differential-drive MZMs", "50 ohm differential terminations"); length 1.75 mm; bw3db 55 and bw6db 105 GHz, reference 1 GHz (Fig. 1b caption "normalized to 1 GHz"), no qualifier (stated values, not bounds), no bw_measured_to (cannot be determined without Fig. 1b: correct to leave empty); ER 35 with gt (">35 dB", DC); er_type static; 3 dB on-chip IL excludes 6 dB splitter, 2 x 1.25 dB couplers and 1-2.5 dB splices (matches the itemisation); 225 GBd, PAM4/PAM6/PAM8 nets 422/506/540 (Fig. 3 captions and Table 1 text), 540 as max net with derived basis, per-lane; PAM4 112 GBd BER < 1e-6 (DFE only); 85 C TEC limit and degradation above 50 C; Tp 13 mW, 20 dBm laser, 2 dBm Tx, 2 km, 5.1 x 3.5 mm2 die; eye BER 2.7e-3 (Fig. 3f caption). Rights: license empty, restricted_local_only, access unknown: consistent with the evidence. Net-rate arithmetic sanity: 225 x 2 / 422 = 1.066 (HD-FEC-like overhead), PAM8 540 = 675/1.25 (different FEC): the paper states the nets, none were recomputed into the CSV. Fab not named: foundry_or_fab empty correct. Text-only limits: bandwidth values are text-stated and the CSV says so; no figure-derived value was entered.

### Verdict: ACCEPT WITH CORRECTIONS (minor; no wrong data value; I attempted to break the bandwidth, IL budget, net-rate and convention cells and found them consistent with the text).

---------------------------------------------------------------------
## 3. ogiso2024 (OFC 2024 Tu2D.7)

Identity: title, 5 authors, venue string, page Tu2D.7, DOI match Crossref; Crossref year only and no license (published_on empty is right); license publisher-copyright from the p.1 footer. Org NTT Innovative Devices Corporation (Atsugi, JP) as in the paper; parent not stated in the paper (correctly left empty).

### Findings

| # | Sev | Field | Data says | Source says | Suggested correction |
|---|-----|-------|-----------|-------------|----------------------|
| O1 | medium | ogiso2024-b bw_measured_to_ghz | 100 (basis measured, "taken equal to the stated 3 dB lower bound") | Table 1 (p.3) for the C+L column: EO bandwidth ">100 / >110" at 3 / 6 dB. A 6 dB bandwidth over 110 GHz implies the measurement reached at least 110 GHz; no measurement limit is stated for this variant (no Fig. 4 curve for it) | The row contradicts itself (measured_to 100 < bw6db 110). Set bw_measured_to_ghz empty (limit not stated), or 110 with basis `derived` and a note that it is implied by the 6 dB bound. Do not label an equated bound as `measured`. |
| O2 | low | ogiso2024-c eo_rolloff_db | 7 at 100 GHz (approx) | Fig. 4(b) re-read at 6x: the orange trace crosses the -6 dB dashed line at about 99-100 GHz; the 100 GHz tick sits at about -6.0 to -6.3 dB. (3 dB crossing about 91 GHz, text "reached 90 GHz", matches) | Change to about 6 (keep `eo_rolloff_db:approx`). Rows a (about -2.2 dB) and d (about -13.5 dB) check out within the approx qualifier. |
| O3 | low | il_onchip_db basis `derived` (rows a, b, d) | evidence notes: "coupling is subtracted (method not stated)" | Abstract and Table 1: typical on-chip IL values "excl. optical coupling loss"; no derivation described | The subtraction story is the preparer's conjecture; use basis `measured` (typ. Table value) or leave basis unspecified; do not use `derived` without a stated computation. Separately, the lt qualifier on row a (abstract "less than 3.5") vs no qualifier on d (same Table typ. 3.5) is a defensible but inconsistent choice. |
| O4 | low | eo_material=inp_mqw (all rows) | enum closest | Paper says only "n-i-p-n heterostructure"; MQW never stated | Caveat is in BATCH_REPORT and epitaxy_or_stack but not in the evidence notes; use `other` or document in the row notes. |
| O5 | low | cached source rights | PDF footer | text.md/PDF footer: "Authorized licensed use limited to: Stanford University ... Downloaded from IEEE Xplore. Restrictions apply" | Not a data error; record that the cached copy is an institution-licensed download (never redistribute) next to restricted_local_only. |

### Judgment call requested by the coordinator: bw3db_ghz=100 (gt) with bw_measured_to_ghz=110 for row a
Reasonable reading. The paper claims "3-dB EO bandwidth exceeded 100 GHz" (abstract, Sec. 1, Sec. 3) and "6-dB ... exceeded 110 GHz"; Fig. 4(c) (red) stays between -1 and -2.5 dB to about 105 GHz, is noisy below -3 dB only in the last few GHz, and ends exactly at the 110 GHz axis limit with a spike to -6 dB. Entering 100 gt keeps the authors' claim; 110 as the plotted range is honest and labelled extracted_from_figure with "instrument limit not stated". Literal convention (c) would give bw3db_ghz=110 gt, which is a stronger claim than the paper makes for 3 dB (the trace actually appears to cross -3 dB near 106-107 GHz in the noisy tail, which no text states). Recommend amending (c) so that "paper-stated bound below the measured range" is explicitly allowed (see X1).

### Checked and correct (sampled directly)
Table 1 (p.3 render) columns: 130 Gbaud / >200 Gbaud C+L ver. / >200 Gbaud Low Vpi ver.; arrow resolution (left column) reproduces: IL 3.5 / 4.0 / 3.5; BW >67/>80, >100/>110, arrow; Vpi 2.0, arrow, 1.5; VpiL 0.72, arrow, 0.54 (1.5 x 0.36 = 0.54 and 2.0 x 0.36 = 0.72 verified); ER >25 all; Z0 60 all; die 5.0 x 2.5. Fig. 4: (a) blue crosses -3 dB at about 71 GHz and -6 dB at about 86 GHz ("around 70 GHz" text; row d approx 70, bw6db gt 80 from Table consistent), (c) is the 1.5 V (>200 Gbaud) device so row a is correct. Fig. 2(a) X/Y about 10.5-11.3 dB, X+Y about 7.5-8 dB; text "<11.5 dB each polarization" (row a per-polarization definition noted in includes/notes). Fig. 2(c) axis "Applied differential voltage", arrows at about -0.75/+0.75 V (1.5 V span). Fig. 1(c) Vpi 1.4 V with up to about 1 dB extra loss; wavelengths 1527/1550/1565 nm; Fig. 2(b)/3(b) ER over 25 dB (data 30-50 dB). Text: -10 V "around" (qualified), 4.5 um lensed fibre, 2.3 dB/facet, 3.6 mm TWE, period 150 to 120 um, Pi-heater <20 mW, Z0 "designed to be around 60 ohm" (design_target + approx correct). All locators match the PDF page index (p.1 Sec. 2 / Figs. 1; p.2 Figs. 2-3 and Sec. 3 start; p.3 Fig. 4, Table 1).

### Verdict: ACCEPT WITH CORRECTIONS (O1 should be fixed; O2-O5 minor).

---------------------------------------------------------------------
## 4. porto2026 (OFC 2026 Th1C.4)

Identity: title, 24 authors (initials as in the paper), venue, page, DOI match Crossref; Crossref no license/published_on; license from the p.1 footer ("(c) 2026 Optica Publishing Group") plus the abstract's "(c) 2025 Nokia Corporation". Org Nokia Corporation, Sunnyvale (US, from the paper's address) correct per convention (e).

### Findings

| # | Sev | Field | Data says | Source says | Suggested correction |
|---|-----|-------|-----------|-------------|----------------------|
| P1 | low | eo_material=inp_mqw | enum choice with no caveat in notes | Paper never mentions MQW, QCSE or any EO mechanism | Add "EO mechanism not stated; inp_mqw is the closest enum" to notes (as done for ogiso2024 in the report) or use `other`. |
| P2 | low | vpi_rf_v 1.5 with `lt` | evidence note quotes "less than 1.5 V" | p.2 Sec. 4: "RF Vpi less than 1.5 V"; abstract/Crossref abstract: "<= 1.5 V" | `lt` follows p.2; mention in the note that the abstract says "up to / <= 1.5 V" (an `le` operator does not exist). |

### Checked and correct (no value error)
Design-target treatment is right: Sec. 4 says "The modulator is designed for RF Vpi less than 1.5 V while providing over 25 dB of extinction and 53 GHz of bandwidth"; Vpi and ER entered with basis design_target and lt/gt qualifiers; 53 GHz not entered (bandwidth type undefined; also the SiGe driver peaking frequency) and kept in context. ER type unspecified: correct. Fig. 3 inset (3x zoom): RLM 0.969/0.971, outer ER 3.53/3.48, TDECQ 2.98/3.01, Ceq 0.05/0.06, avg P 2.93/4.92 dBm: notes and context match. 106.25 GBd PAM4, 212 Gbps (106.25 x 2 = 212.5, authors state 212: derived basis correct), 60 C (Sec. 3 and Conclusion), SOA 57-95 mA for +5 dBm, NF 6.5 dB, RIN 0.4% over 0.1-18 GHz, SMSR >50 dB, 30 ohm load resistors connecting the arms, four O-band DFBs each feeding two channels, OSFP LPO 2.8-3.0 dBm, FFE 15 taps with fewer than ten significant, driver peaking 5-16 dB at 53 GHz, AlN carrier: all verified in text. drive/vpi_convention left unspecified (not stated): correct. Locators match p.1-p.3 (Fig. 3 on p.3).

### Verdict: ACCEPT (two cosmetic notes; I checked 16 values and all conventions and found no data error).

---------------------------------------------------------------------
## 5. tanaka2026 (OFC 2026 M4D.2)

Identity: title, 15 authors, venue, page, DOI match Crossref; Crossref no license/date; license publisher-copyright from the p.1 footer. Orgs and affiliations 1-5 verified: PETRA (Bunkyo-ku), Sumitomo Electric Industries, Ltd. (Yokohama), Institute of Science Tokyo, 1FINITY Inc. (Kawasaki), The University of Tokyo, all JP. foundry_or_fab empty is correct (ARIM acknowledged, no site named).

### The Vpi*L = 0.43 V*cm vs Fig. 5 question (item 6): the stated value is right, the preparer's "mismatch" note is wrong
- Text: Sec. 2 "center bias voltage of -8 V, ... Vpi*L of 0.43 V*cm"; Fig. 5 label "VpiL = 0.43 Vcm"; Conclusion "0.43 V*cm". Three statements agree.
- Fig. 5 (4x zoom): x axis Vp_arm - Vn_arm; left dashed line at the transmission maximum at x about -3.6 V (tick spacing 42 px/V; line at about -3.57 V), right dashed line at the null at 0 V (about -15 dB), double arrow between them. So the figure's maximum-to-null span (the Vpi in the differential variable) is about 3.6 V. The batch's "about 3.5 V" visual read is right.
- The preparer concluded 3.5 V x 2 mm = 0.7 V*cm "does not match". But with the filling-factor-weighted length of the capacitively loaded section, L_eff = 0.60 x 2 mm = 1.2 mm: 3.6 V x 0.12 cm = 0.43 V*cm. The same holds for Fig. 6: Vpi 5.4 V x (0.80 x 1.0 mm = 0.08 cm) = 0.43 V*cm. Both devices land on 0.43 V*cm to within reading error, which strongly supports FF x length as the length used (the paper does not say so: this is an inference and must be labelled as such).
- Conclusion: figure and text are consistent; the 0.43 V*cm is right as stated. The "unreconciled" notes in papers.csv, devices.csv (tanaka2026-a) and BATCH_REPORT are wrong and would mislead an integrator.

### Findings

| # | Sev | Field | Data says | Source says | Suggested correction |
|---|-----|-------|-----------|-------------|----------------------|
| T1 | medium | papers.csv notes, tanaka2026-a notes, BATCH_REPORT: "0.43 V*cm not reconciled with Fig. 5 (about 0.7 V*cm at 2 mm)" | claims an internal inconsistency | Fig. 5 span about 3.6 V; 3.6 V x (0.6 x 2 mm) = 0.43 V*cm and 5.4 V x (0.8 x 1.0 mm) = 0.43 V*cm (Fig. 6 caption) | Replace the note with: "Fig. 5 shows about 3.6 V max-to-null; 0.43 V*cm is reproduced if L is FF x electrode length (1.2 mm); the authors do not state the length used (inference)." Remove the same wording from BATCH_REPORT. |
| T2 | low | tanaka2026-a vpi_dc_v (empty) | "no Vpi in volts is stated" | Fig. 5 annotates the span (about 3.6 V), axis is the differential arm voltage | Optionally enter 3.6 V, basis extracted_from_figure, approx, vpi_convention mzm_differential, bias -8 V; otherwise state in notes that the figure span was left out. |
| T3 | low | length_mm=2 with vpil_dc_vcm=0.43 (rows a and b) | 2 mm physical electrode length; row b 1.0 mm with Vpi 5.4 V | Fig. 5/6 captions give physical lengths; the Vpi*L product implies an FF-weighted length | Add a note that Vpi*L uses an effective length (inference) so that a view deriving Vpi x length (3.6 x 0.2 = 0.72 for a, 5.4 x 0.1 = 0.54 for b) is not read as a conflict with the stated 0.43. |
| T4 | low | tanaka2026-a drive/vpi_convention evidence basis | `measured` for both | Caption states "push-pull operation" (drive OK); the Vpi convention is inferred from the x-axis label "Vp_arm - Vn_arm" | vpi_convention basis `derived` with note "authors do not name the convention" (convention (f)/(h)). |

### Checked and correct
Fig. 6(a) re-read: all four sections cross -3 dB at about 55-60 GHz, near -11 dB at 110 GHz (range -10.5 to -11.8; row b eo_rolloff 11 at 110 approx OK); Fig. 6(b) peak about +4.3 dB at 43-47 GHz and steep drop through -3 dB at about 57 GHz (notes OK); Fig. 4: P_pi 20 mW (arrow about 24 to 44 mW), ER > 30 dB (0 to about -33 dB); Fig. 7 4x4 constellation = 16QAM; text values: -8 V centre bias, 64 GBd, MSB/LSB oDAC, driver IC and AWG, chip 3.8 x 3.3 mm, no wavelength/IL/BER stated (all left empty correctly); abstract "approximately 60 GHz (V_pi: 5.4 V)" matches bw3db 60 approx and vpi_dc_v 5.4 with convention unspecified (fine). ER 30 with gt and er_type static correct (heater sweep, noted). integration wafer_bonded_iii_v and eo_material inp_mqw (MQW and QCSE stated) supported. 256 Gb/s correctly not entered.

### Verdict: ACCEPT WITH CORRECTIONS (T1 should be fixed because the note asserts a non-existent inconsistency; no wrong cell value).

---------------------------------------------------------------------
## 6. Convention and skill ambiguities (for the coordinator)

| # | Issue | Evidence |
|---|-------|----------|
| X1 | Convention (c) literal text ("bw3db_ghz = measured-to value with gt") does not cover a paper that states a bound lower than the plotted range (ogiso2024-a: bound 100 GHz, plot to 110). The batch's reading is sensible; amend (c) to say the stated bound goes in bw3db_ghz:gt and the plotted/instrument range in bw_measured_to_ghz, and that bw_measured_to_ghz is left empty when no range is stated (ogiso2024-b). | O1 |
| X2 | Length for segmented/capacitively loaded electrodes: length_mm "active electrode length" is ambiguous when the paper's Vpi*L uses a loaded (FF-weighted) length. Add a field or a note rule (tanaka2026: 0.43 V*cm = Vpi x FF x L). | T1, T3 |
| X3 | `derived` vs paper-stated: values the paper states without a stated formula (qiu2026 3 dB on-chip; ogiso2024 typ. on-chip IL) get labelled `derived` by the preparer from guessed arithmetic. Rule: `derived` only when the paper's own computation is given or implied in the text; arithmetic checks go in the `derived` list. | Q4, O3 |
| X4 | Inferred conventions (vpi_convention, drive) basis: (f) says `derived` + note; the batch marks stated and inferred cases both `measured`. | D1, T4 |
| X5 | Vpi measured at an unspecified frequency is entered in vpi_dc_v with a caveat (qiu2026, tanaka2026-b). Still no field to say "frequency unspecified" (also raised in BATCH_REPORT). | Q-notes |
| X6 | eo_material has no value for undoped/n-i-p-n InP or "mechanism not stated"; inp_mqw is used when MQW is not stated (ogiso2024, porto2026). | O4, P1 |
| X7 | Multinational orgs: country comes from the paper's address (Ciena CA, Nokia US) but the organizations table is global, so the country may differ for another paper's address; org_type values `consortium` and `research_institute` are not enumerated anywhere. | org rows |
| X8 | Sim provenance still lacks an "unverified standard value" state (standard_reference used with "not verified"), same as pilot X9. | D4 |
| X9 | Rights: a cached PDF carrying an institution-licensed-use footer (ogiso2024) is not covered by the license rule. | O5 |
| X10 | Table-with-arrows convention (resolve to the left column) worked here and reproduced all Table 1 values; worth writing into the skill. | ogiso2024 |

---------------------------------------------------------------------
## Counts

| Paper | high | medium | low | verdict |
|-------|------|--------|-----|---------|
| deng2026 (rows, papers, config) | 0 | 1 | 6 | accept with corrections |
| qiu2026 | 0 | 0 | 5 | accept with corrections |
| ogiso2024 | 0 | 1 | 4 | accept with corrections |
| porto2026 | 0 | 0 | 2 | accept |
| tanaka2026 | 0 | 1 | 3 | accept with corrections |
| Total | 0 | 3 | 20 | none rejected |

Cross-paper convention/skill issues: 10 (X1-X10). Most important: tanaka2026 T1 (the "mismatch" note is wrong), ogiso2024-b O1 (invented measurement limit contradicting its own 6 dB bound), deng2026 D1 (inferred Vpi convention labelled measured).
