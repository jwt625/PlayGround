# Audit Q1 (fresh context): pilots chen2022, kohli2025, ogiso2016

Auditor: independent subagent, 2026-10-01. Read-only; no data edited. Sources used: references/<id>/text.md and page/figure PNGs (zoomed crops of Fig. 2/4/5 of chen2022, Fig. 2d/3b/4b of kohli2025, Fig. 2/4 of ogiso2016), schema, SKILL.md, data/papers.csv, data/devices.csv, data/organizations.csv, data/evidence/*.yaml, sims/chen2022/config.yaml, sims/SPEC.md (sign convention only).
Note: references/{chen2022,kohli2025,ogiso2016}/ contain NO crossref.json (other papers do). Crossref-derived claims (published_on for kohli2025, licenses, "Crossref queried" in ogiso2016 notes) could not be re-verified offline; the article's own front matter supports the chen2022 and kohli2025 CC-BY statements and the ogiso2016 paywall status.
Mechanical check (script): every non-empty evidence:true cell has an evidence entry or a `derived` item with the same value; no value mismatches; no absolute paths or private names in the three papers' files.

Severity: high = wrong/misleading in a way a plot or comparison would show; medium = convention or completeness defect that changes interpretation; low = cosmetic, locator, or judgment call.

---------------------------------------------------------------------
## 1. chen2022 (APL Photonics 7, 026103)

Identity: title, 11 authors (order), journal/volume/article number, DOI, published_on 2022-02-02 (Published Online 2 February 2022, p.1-2), CC-BY-4.0 (abstract footer p.2), affiliations/universities all match the source. repro_grade B plausible.

### Findings

| # | Sev | Field | Data says | Source says | Suggested correction |
|---|-----|-------|-----------|-------------|----------------------|
| C1 | high | chen2022-a/b/c extinction_ratio_db, qualifiers | 30 / 25 / 20, qualifiers column empty; notes say "reported as >30 dB" | p.7 Sec. III: "ERs of >30, >25, and >20 dB"; Fig. 4 insets "ER>30dB / >25dB / >20dB" (PDF p.6) | Add `extinction_ratio_db:gt` on all three rows (convention b); set er_type=static (text: "static voltage sweep"). Kohli and Ogiso rows do this; chen does not. |
| C2 | medium | il_onchip_db, prop_loss_db_per_cm, qualifiers | 0.2 (a,b,c), 0.15 (c), no approx qualifier; basis measured | p.7: "on-chip loss of ~0.2 dB for the longest device (10 mm) can be deduced"; "about 0.15 dB/cm". Fig. 4 insets do read -0.2 dB for all three lengths (verified) | Add `il_onchip_db:approx;prop_loss_db_per_cm:approx`; for c the paper itself deduces 0.2 from 8.2 dB total minus GC reference, so evidence basis should be `derived` (author-deduced), not `measured`. a/b values rest on the inset labels only (fine, keep with that locator). |
| C3 | medium | eo_rolloff_db / eo_rolloff_freq_ghz (chen2022-c) | empty (value only in notes) | Abstract, p.7 and Table I (p.8): roll-off 1.4 dB at 67 GHz (and 0.76 dB at 50 GHz) for the 10 mm device | Fill eo_rolloff_db=1.4, eo_rolloff_freq_ghz=67 (basis measured, locator abstract/Table I); the 50 GHz point stays in notes (single-point columns). Reference frequency is not stated in the paper: say so. |
| C4 | medium | vpi_convention=mzm_push_pull, drive=single_ended (a,b,c) | asserted with no evidence entry and no note in the CSV | The paper never says push-pull or differential (config.yaml provenance admits "paper never uses the words push-pull"). Fig. 1(c,d) show one centre signal electrode with one arm in each gap, so push-pull by geometry is a reasonable reading; Vpi is read as full-transmission swing (Fig. 4) | Add evidence entries for vpi_convention and drive with locator Fig. 1(c,d) and basis `derived`/inference note ("not stated by authors"); or use `unspecified` for Vpi convention. Matters for factor-of-2 comparisons. |
| C5 | medium | chen2022-a/b z0_ohm, n_rf, ng_opt, rf_loss_db_per_cm, rf_loss_freq_ghz | identical values copied onto 5 mm and 7 mm rows with evidence entries citing Fig. 5(d)-(f) | Fig. 5 caption: "Deduced (d) Zc, (e) alpha_m, (f) n_m" (a single curve each); p.7: "alpha0 = 0.36 ... for the measured device". Paper does not say the set applies to all three lengths or which device it is | Leave these RF-line fields empty on a/b, or keep them with a note that the assignment is an assumption. The "one deduced set used for all three lengths" remark appears only in the c-row notes. |
| C6 | low | rf_loss_db_per_cm=2.947 at 67 GHz | alpha0*sqrt(67) with alpha0=0.36 (arithmetic correct: 2.947) | Fig. 5(e) raw trace at 67 GHz reads about 3.3-3.5 dB/cm (noisy); 2.947 is the fitted law, and p.6 simulated alpha0 = 0.19 (1.555 dB/cm at 67 GHz, matches Fig. 2e) | Fine as a derived value (it is in `derived`), but add `rf_loss_db_per_cm:approx` and say "fit" in the row notes; basis is not "measured". |
| C7 | low | c-row notes: "Rib width 1.5 um (top)" | top | p.5: "a width of 1.5 um", top/bottom not specified; Fig. 2(a) sketch is a trapezoid | Drop "(top)" or mark as digitized assumption. |
| C8 | low | locators | e.g. "p.7 Sec. III; Fig. 4(a)" | Fig. 4/5 are on PDF p.6 (journal 026103-5); Sec. III text on PDF p.7 (026103-6). PDF page index used throughout; PDF p.1 is a cover sheet so journal page = PDF page - 1 | State the page convention in SKILL.md; use "p.6, Fig. 4(a); p.7 text". |
| C9 | low | papers.csv notes "Fab: in-house" | in-house | Paper does not state who fabricated (EBL, ICP-RIE, lift-off described; no facility named) | Reword to "fabricator/facility not named". foundry_or_fab empty is correct. |
| C10 | low | countries=CN for The Hong Kong Polytechnic University | CN | Affiliation reads "Hong Kong, China"; schema says ISO 3166 alpha-2 from affiliation (HK is a valid code) | Decide HK vs CN once for the DB; document in schema. |
| C11 | low | data/candidates.csv published_on | 2022-02-01 | Paper: 2 February 2022 (papers.csv has 2022-02-02, correct) | Stale candidate row; ignore or sync. |
| C12 | low | papers.csv notes cite Crossref VOR license record | claim | No crossref.json in references/chen2022 | License is independently supported by the article footer; add the cached Crossref file or reword. |

### Sim config (sims/chen2022/config.yaml)

| # | Sev | Item | Issue | Suggested correction |
|---|-----|------|-------|----------------------|
| S1 | low | provenance geometry.regions.sio2_cladding | class figure_digitized, note says sketch shows 0.86 um above slab; region uses y 0.2..1.1 (= 0.9 um, the text value "900 nm") | Class should be paper_exact for 0.9 um with project_inference for the reference surface; or use 0.86 if digitized is meant. |
| S2 | low | provenance materials.lithium_niobate n_o, n_e, eps_r, r_pm_per_v; silicon_dioxide | class standard_reference but notes say "value taken from SPEC.md example; not verified" | Use a distinct class/flag (unverified) until the citation is checked; standard_reference overclaims. |
| S3 | low | ln_rib_r sidewall | note says bottom 1.69 um, "~68 deg from horizontal"; polygon (+-0.84 at slab, +-0.75 at top, h=0.2) gives 65.8 deg from horizontal and bottom 1.68 | Align the note with the polygon (arithmetic check). "top width 1.5 um from text" is also an inference (text does not say top). |
| S4 | low | targets sign/label | eo_rolloff_db targets are negative (-1.4, -0.76), schema column is a positive drop; rf_loss 2.947 and n_rf/z0 are listed under "Measured" though they are deduced/fitted from measurements | Fine per SPEC example sign; state "deduced from measurement" in the block comment. |
| S5 | low | title "T-rail" | term not in paper (T-segment) | Cosmetic. |

Checked and correct in config: classes paper_exact for 400 nm film, 200 nm ridge, 3 um BOX, 725 um Si, 75 um signal, 1.1 um Au, 10 mm length, 50 ohm termination, TE mode; wavelength 1550 correctly project_inference (paper gives no wavelength); period 50 um and loaded length 36 um correctly project_inference; slot width hw=9, h=15, g=1.8 geometry arithmetic consistent (pads at +-0.9, slots 5.9..14.9, 3 um margins, mirror about -55.4 um, domain +-245 um); paper-simulated targets correctly labeled basis simulated/derived/extracted_from_figure and kept separate from measured: VpiL 2.10 (Fig. 2a), n_rf 2.16 (Fig. 2d plateau 2.15-2.17, verified), Z0 52.3 (Fig. 2c plateau, verified), rf loss 0.19*sqrt(67)=1.555 (Fig. 2e at 67 GHz about 1.55, verified), ng 2.2 (Fig. 2d label). Measured targets: VpiL 2.20, n_rf 2.35, Z0 50.5, roll-off -1.4/-0.76 are measured/deduced values, correctly separated.

### Sampled values verified directly (>= 10)
Fig. 4 Vpi 4.45 / 3.22 / 2.20 V; Fig. 4 insets -0.2 dB (x3) and ER > 30/25/20 (x3); VpiL 2.22/2.25/2.20 (4.45*0.5=2.225, 3.22*0.7=2.254, 2.20) correct; Fig. 5(d) Zc ~50 rising to ~52 noisy at 67 GHz; Fig. 5(f) n_m 2.35 flat (red dashed 2.2); Fig. 5(e) alpha_m scale; Fig. 2(a) 2.10 V*cm; Fig. 2(c-e); p.5 geometry (400/200/1.5/3/900/1.8/75/1.1/35 um); Table I 0.2 dB, 2.2 V, 10 mm, -0.76 dB@50, -1.4 dB@67; 8.2 dB total and 100 kHz sweep (p.7); 112 Gb/s = 56 GBd PAM-4, 100 Gb/s OOK, dynamic ER >9.7 dB, AWG ~35 GHz, no driver amplifier mentioned (p.7); EO S21 3 dB BW ">67 GHz beyond VNA limit" (p.7) correctly entered as bw_measured_to_ghz only (convention c).

Left empty correctly: wavelength, band, optical_input_power, sidewall angle, drive_vpp, foundry, wafer supplier, bw3db_ghz.

### Verdict: ACCEPT WITH CORRECTIONS (C1 must be fixed; C2-C5 should be fixed).

---------------------------------------------------------------------
## 2. kohli2025 (Light Sci. Appl. 14, 399)

Identity: title (en dash), 16 authors in order, journal, volume, article number 399, DOI, CC-BY-4.0 (p.1 notice), received/revised/accepted dates in notes (14 Aug 2024 / 14 Oct 2025 / 2 Nov 2025) all match. published_on 2025-12-16 is not in the text (only Crossref, not cached): unverifiable offline. Orgs: ETH Zurich, Ligentec SA, Lumiphase AG per affiliations 1-3; foundry_or_fab roles match author contributions (Ligentec SiN wafers, Lumiphase BTO growth/integration) and acknowledgements (BRNC cleanroom).

### Findings

| # | Sev | Field | Data says | Source says | Suggested correction |
|---|-----|-------|-----------|-------------|----------------------|
| K1 | medium | vpi_dc_v / vpi_convention (mzm, iq) | vpi_dc_v 3.6 / 4 as "headline", convention per_arm; push-pull 1.8 / 2 in vpi_mzm_pushpull_dc_v; evidence note for 1.8 calls it "paper headline convention" | p.4/p.8: "Vpi = 1.8 V at DC" is the paper's headline (Discussion); 3.6 V is the phase shifter | Schema (a) says the headline convention goes in vpi_dc_v; the rows and their own evidence note contradict each other. Resolve in schema (see X3); until then drop "paper headline convention" from the evidence note. |
| K2 | low | kohli2025-iq bw3db_ghz=70, bw_measured_to_ghz empty, no qualifier | 70 measured | p.5 text: "response starts to drop at ~70 GHz"; Fig. 3b caption: "cutoff at around 80 GHz. We find a 3-dB drop between 10 and 70 GHz". Fig. 3b (zoomed): about -4 to -5 dB at 10 GHz, about -7.5 at 70 GHz, -9 to -10 at 85 GHz; x axis runs to 110 | 70 is the caption value; add `bw3db_ghz:approx`; note the 80 GHz statement. Measurement range is plausibly 110 GHz (axis) but not stated: leave empty or enter 110 as extracted_from_figure. |
| K3 | low | optical_input_power_dbm (rt) | 13 dBm | Methods p.10: "input power (in the fiber before the device) of ~13 dBm"; MZ/IQ say "input power to the chip" (~20.3, ~21 dBm) | Definitions differ (fiber vs chip); schema asks launch/on-chip. Evidence note for rt already says fiber; add to row notes. |
| K4 | low | device_class | IQ = iq_mzm, MZ = plasmonic_mzm, both plasmonic | Both are plasmonic BTO devices | Class vs tag usage inconsistent (see X6). |
| K5 | low | papers.csv published_on / notes "License verified on Crossref" | 2025-12-16 | not in local sources; no crossref.json | Cache the Crossref record or mark unverified. |
| K6 | low | organizations.csv Binnig and Rohrer Nanotechnology Center | facility, country CH "entered from outside the paper" | Paper only thanks the BRNC cleanroom team; host not stated | Convention (e) wants host-institution derived country with a note; disclosed in notes, acceptable. Spelling "Stafa" vs paper "Stäfa" in org notes (cosmetic). |
| K7 | low | integration empty | empty | BTO on SiN is wafer-scale integrated by Lumiphase (p.2, Methods) but mechanism not given | Acceptable; `other` not warranted without text. |

Checked and correct (adversarial pass, no data-value error found): MZ: 15 um, 150 nm slot, Vpi 3.6 DC / 1.8 push-pull / 6.4 V at 40 GHz (phase-shifter context, p.4), 20.3 dB fiber-to-fiber at 1550 (Fig. 2c peak ~ -20), 0.5 dB/um = 5000 dB/cm, 2.8 dB/GC and 3.5 dB/PPC in notes, 20.3 dBm input, 256 GBd 2PAM / 170 GBd 4PAM / 96 GBd 8PAM, 340 Gb/s = 170*2, 30 fF and ~10 fJ/bit (basis author_estimate; 30 fF*(1.13 V)^2/4 = 9.6 fJ), 1.13 Vpp for all modulators. 3 dB BW 110 GHz = measurement limit, entered in both fields per convention (c); note that Fig. 2d normalization is unstated (plateau 0 dB at 40-70 GHz, about -2.5 dB at 110, about +2 dB at 10 GHz) and is correctly flagged. IQ: 17.5 um, 100 nm slot, 4 V/2 V, 23.9 dB at 1550 (22.5 at 1530), 0.7 dB/um = 7000 dB/cm, 21 dBm, 224 GBd 4QAM = 448 Gb/s (224*2), push-pull stated (p.10). RT: 5 um, 1315.7 nm, Q 1931, FSR 1.79, 0.3 nm/V, Vpi 3 V derived (FSR/2/0.3 = 2.98 V, vpi_basis derived and convention resonance_tuning_derived correct), IL <2 dB and ER >6 dB entered with lt/gt qualifiers (Fig. 4b labels verified), 9.4 dB fiber-to-fiber at 1310-1315 nm, 200 GBd 2PAM = 200 Gb/s derived (listed in `derived`), Fig. 4d axis +-70 GHz (bw_measured_to 70, basis extracted_from_figure, no 3 dB number: correct). Locators checked against page markers: all match (p.4 Vpi, p.5 IQ, p.6 RT, p.7 data, p.9-10 methods).
Left empty correctly: on-chip IL for MZ/IQ (not reported), RF Vpi for IQ/RT, ER for MZ/IQ, Z0/n_RF/ng, electrode thickness, slot width of RT, net rate. Pockels coefficient (~180 pm/V at 40 GHz) and BTO waveguide loss 4.5 dB/cm are in notes only (no column): fine.

### Sampled values verified directly (>= 10)
Fig. 2c 1550 nm peak ~ -20 dB; Fig. 2d shape/110 GHz axis; Fig. 3b response at 10/70/85 GHz; Fig. 3f -22.x at 1530 and ~ -24 at 1550; Fig. 4b Q label, IL<2 dB, ER>6 dB, 0.3 nm/V, dips at ~1313.7/1315.3 nm (FSR ~1.8); Fig. 4d axis; Fig. 5 captions (BERs 3.11e-2, 4.00e-2); text values listed above.

### Verdict: ACCEPT WITH CORRECTIONS (minor; no wrong numeric value found; K1 is a convention/schema issue).

---------------------------------------------------------------------
## 3. ogiso2016 (Electronics Letters 52(22), 1866-1867)

Identity: title, 7 authors, EL vol. 52 no. 22 pp. 1866-1867, 27 October 2016 issue, DOI, submitted 2016-08-16, E-first 2016-09-30 (published_on) all match page 2. paywalled/publisher-copyright/restricted_local_only consistent with the Wiley footer. Org: NTT Corporation expanded to Nippon Telegraph and Telephone Corporation, country JP, per convention (e), noted in organizations.csv. foundry_or_fab empty: correct (not stated).

### Findings

| # | Sev | Field | Data says | Source says | Suggested correction |
|---|-----|-------|-----------|-------------|----------------------|
| O1 | medium | bw_measured_to_ghz=67 with qualifier `bw_measured_to_ghz:gt`; bw3db_ghz empty | gt applied to the instrument limit | Abstract/p.1: "3 dB EO bandwidth of over 67 GHz"; Fig. 4 trace ends at 67 GHz at about -2.7 dB (no crossing) | The measurement limit is exactly 67 GHz; the bound belongs to the 3 dB bandwidth. Per convention (c) keep bw_measured_to_ghz=67 with no qualifier (as chen2022 does). If bounds are meant to be plotted via bw3db_ghz:gt, change the schema (X1). Rows are currently inconsistent across papers. |
| O2 | medium | eo_rolloff_db / eo_rolloff_freq_ghz | empty; "approx -2.7 dB" kept only in evidence context_values | Fig. 4 (zoomed): EO response at 67 GHz about -2.6 to -2.7 dB, referenced to 1.5 GHz (p.1 text) | Fill eo_rolloff_db about 2.7, eo_rolloff_freq_ghz 67, basis extracted_from_figure, bw3db_reference other/1.5 already set. |
| O3 | low | extinction_ratio_db=22 (gt) | "ER exceeded 22 dB for the entire C-band" | Fig. 2 (zoomed): 1540 nm trace (green) bottoms near -21 dB; others -24 to -27 (1560 nm -27 matches the 27 dB text) | Source internal inconsistency; keep the text value, add a note. |
| O4 | low | il_onchip_db=2, il_basis | row il_basis measured; evidence basis author_estimate; no approx qualifier | p.1: "on-chip loss was estimated to be 2 dB" (10 dB total - 2*4 dB/facet = 2, correct) | Add `il_onchip_db:approx`; row il_basis should read author_estimate for the on-chip number (evidence basis already wins by rule). |
| O5 | low | row granularity | one row, wavelength 1550 | Vpi=2.0 V and ER curves reported at 1530/1540/1550/1560 nm (Fig. 2); ER 27 dB at 1560 | Rule (d) allows a row per measured wavelength; one row with the C-band lower bound is a defensible simplification: state it in notes. |
| O6 | low | vpi_convention=mzm_series_push_pull | series push-pull | Paper: "differential voltage applied across the electrode", "conventional series push-pull drive (single-ended 50 ohm design)" | Both mzm_series_push_pull and mzm_differential fit (see X3). |
| O7 | low | papers.csv research_groups | "NTT Device Innovation Center; NTT Device Technology Laboratories" | As in the paper | Abbreviation NTT kept here while the org name is expanded; harmless. |
| O8 | low | notes "Crossref queried 2026-10-01" | claim | no crossref.json cached | Cache or reword. |

Checked and correct (adversarial pass): length 3 mm; Vpi 2.0 V (Fig. 2 arrow 0 to 2 V, axis "differential voltage"); bias_for_vpi_v correctly empty (bias value not stated); -5 V reverse bias only in notes; 1.5 GHz EO reference (other, 1.5); 10 dB fiber-to-fiber and 4 dB/facet exclusion text; ER type static; 1550 nm / +13 dBm / 2.3 Vpp demo values on p.1; NRZ-OOK 100 Gb/s with 100 GBd derived (NRZ = 1 bit/symbol, in `derived`); [011] stripe; epitaxy order top to bottom matches text; BCB, SI-InP; electrode CL-TWE; S21 6 dB electrical BW >67 GHz correctly not entered into an EO field; dynamic ER >18/>10 dB not in CSV (acceptable, in context_values).

### Sampled values verified directly (>= 10)
Fig. 2 Vpi arrow 2 V and ER at 1560 (-27 dB), 1540 (-21 dB); Fig. 3 S21 about -6 dB and S11 below -10 dB at 67 GHz, Vbias -5 V; Fig. 4 end of trace at 67 GHz about -2.7 dB, axis 0..-3; Fig. 1 axes [011]; text values listed above.

### Verdict: ACCEPT WITH CORRECTIONS (O1, O2 should be fixed; no wrong numeric value found).

---------------------------------------------------------------------
## 4. Cross-paper skill / schema issues

| # | Issue | Evidence |
|---|-------|----------|
| X1 | Convention (b) vs (c) conflict on "over X GHz" bandwidth. (b) says a bound is entered as number + qualifier; (c) says no crossing means only bw_measured_to_ghz. No rule says which field carries the gt qualifier. | chen2022: bw_measured_to=67, no qualifier; ogiso2016: same plus `bw_measured_to_ghz:gt`; kohli2025-mzm: crossing claimed at limit, both fields. |
| X2 | No validator/guidance that a text bound (">30 dB", "<2 dB", "~0.2 dB", "about") must produce a qualifier. Applied in kohli2025 and ogiso2016 ER, not in chen2022. | C1, C2, O4, K2 |
| X3 | Vpi convention: vpi_convention enum overlaps (mzm_differential vs mzm_series_push_pull vs mzm_push_pull) and rule (a) "headline convention" is ambiguous when the paper headlines the MZ value but quotes a per-arm value (kohli2025). `drive` mixes RF feed topology (chen2022 single_ended) with arm-field topology (kohli2025 push_pull, ogiso2016 series_push_pull) for what are all single-feed GSG devices. vpi_dc_v therefore mixes per-arm and MZ-level numbers across rows (factor 2); any chart must filter on vpi_convention. | C4, K1, O6 |
| X4 | eo_rolloff_db: single (value, freq) pair, reference frequency undefined, sign positive in schema but negative in sims/SPEC.md targets; papers report two points (chen2022) or only a figure read (ogiso2016); none filled in any of the three papers. | C3, O2, S4 |
| X5 | Basis granularity: il_basis/vpi_basis/bw_basis are per-group; evidence basis "wins". Author-deduced values (chen2022 0.2 dB, ogiso2016 2 dB, kohli2025 Vpi 3 V) are labeled inconsistently (measured / author_estimate / derived). Rule needed: author-deduced = `derived` or `author_estimate`, never `measured`. | C2, O4 |
| X6 | device_class vs tags for plasmonic MZ/IQ (iq_mzm vs plasmonic_mzm), and `integration` for BTO-on-SiN, have no tie-break rule. | K4, K7 |
| X7 | Page-locator convention undefined (PDF index vs journal page; figure page vs text page; cover sheets). | C8 |
| X8 | Country code for Hong Kong (CN vs HK) and facility-country when host is not named in the paper. | C10, K6 |
| X9 | Sim provenance classes lack an "unverified standard value" state; standard_reference is used for values explicitly marked not verified. figure_digitized used where the number is the text value. | S1, S2 |
| X10 | Crossref cache missing for the three pilots although notes cite Crossref checks; the skill does not require crossref.json to be kept as the evidence for published_on/license. | C12, K5, O8 |

---------------------------------------------------------------------
## Counts

| Paper | high | medium | low | verdict |
|-------|------|--------|-----|---------|
| chen2022 (rows + papers) | 1 | 4 | 7 | accept with corrections |
| chen2022 sim config | 0 | 0 | 5 | accept with corrections |
| kohli2025 | 0 | 1 | 6 | accept with corrections |
| ogiso2016 | 0 | 2 | 6 | accept with corrections |
| Total | 1 | 7 | 24 | none rejected |
Cross-paper schema/skill issues: 10 (X1-X10).
