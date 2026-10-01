# BATCH_REPORT p1_01 (BTO and InP papers)

Date: 2026-10-01. Validation: `uv run python scripts/merge_staging.py data/_staging/p1_01` (dry run) reports merge counts papers 5, devices 9, orgs 11, evidence 5; conflicts 0; validation errors 0. No network requests were made. The pre-existing `cr/` folder was left untouched.

Convention re-check: after the coordinator update (conventions (a)-(k), revised 2026-10-01) all nine rows, the evidence files and the sim config were re-checked against it and corrected. Changes made: (b) every over/below/about wording now has a qualifiers entry (added il_onchip_db:lt on ogiso2024-a, eo_rolloff_db:approx where read from plots); (c) ogiso2024-a carries the stated 3 dB bound (100 GHz, gt) plus bw_measured_to_ghz 110 from the plot axis (ogiso2024-b: see Audit corrections); (f) drive reviewed per row (deng2026 single_ended and mzm_single_arm now basis derived, tanaka2026-a push_pull per its caption, tanaka2026-b and porto2026 unspecified because not stated, no inferred drive values); (g) eo_rolloff_db and eo_rolloff_freq_ghz filled from plots (extracted_from_figure) for deng2026-a, ogiso2024-a/c/d, tanaka2026-b; (h) author-computed values are basis derived (Vpi*L products in deng2026-a, ogiso2024-a/b/d, tanaka2026-a; computed net rate in qiu2026-a; on-chip IL was first marked derived and is now measured (see Audit corrections); 212 Gb/s in porto2026-a), row il_basis/vpi_basis updated; (i) locators use the text.md page markers (qiu2026 is text-only without page markers, so its locators name sections); (j) device_class mzm/iq_mzm only, no plasmonic_mzm; (k) published_on left empty for qiu2026, porto2026 and tanaka2026 because the conference-schedule dates in the batch CSV are confirmed neither by Crossref (year only) nor by the papers (the integrator may restore them if a citable source is supplied); license basis recorded in each evidence file (context_values license_and_rights_basis). RF-line values (Z0 around 60 ohm) are entered only for the ogiso2024 variants the paper says share it.

Staging outputs: `papers.csv` (5), `devices.csv` (9), `organizations.csv` (11 new), `evidence/` (5 files), `SPEC_PROPOSALS.md`, this report. No `needs_download.md` (all five sources present).

## deng2026 - distilled
- Source: journal version of record, Light: Science & Applications 15, 21 (2026), CC-BY-4.0 (Crossref VOR/TDM records and article notice). Supplementary Information (Notes 1-8, Figs. S1-S5) is not in the cache and was not read.
- Rows: 1 (deng2026-a, 1 mm GSG-TWE BTO-on-LSAT MZI with Si3N4 strip loading). repro_grade B.
- Sim config: `sims/deng2026/config.yaml`. NOT RUNNABLE by design: the paper gives no RF permittivity for BTO, LSAT or Si3N4 and no Si3N4 index; these are omitted and listed under `missing` (the engine rejects dielectrics without eps_r). The coordinator should decide between supplying sourced constants or the placeholder proposal in SPEC_PROPOSALS.md. The app /sim route will list this config but the solve will error until then. Not run.
- Not reported: IL, ER, propagation loss, RF loss, Z0, n_rf, ng, Si3N4 strip width, signal/ground widths, system data, energy.
- Judgment calls: Vpi 7 V and Vpi*L 0.7 V*cm come from a 100 kHz triangular sweep, entered in the DC columns (single-arm convention, bias not stated). bw3db 12 GHz (approx) and bw6db 28 GHz both entered; Fig. 4e shows a 6 dB step at 12 GHz then a plateau; 0 dB reference unspecified. Measurement limit not entered because both crossings were observed (Methods say 40 GHz bias-T limit; plot ends at 30 GHz). Simulated Vpi*L 0.73 V*cm kept out of the columns (sim target, basis simulated). Film-level Pockels values (r42 > 358 pm/V, r_c 253 pm/V) are material properties from a separate 180 nm film, in evidence context only. Grade B rests on figure-digitized strip width and assumed electrode widths (Fig. 4a scale bar and the 5.5 um gap give inconsistent widths, so 20 um is a project_inference).
- CSV hints: platform barium_titanate / mzm / priority 1 / sim yes all consistent; platform enum bto_on_oxide_substrate, integration monolithic (no bonding, PLD on LSAT then PECVD Si3N4).

## qiu2026 - distilled (text-only)
- Source: pre-extracted text only; no PDF, no figures. Could not read Fig. 1b (EO S21 curve), Fig. 1c (process cross-section), Fig. 2, Fig. 3 plots or Table 1 layout. All values come from running text and captions; bandwidths (55/105 GHz) are therefore text-stated, not checked against the curve. No license on Crossref or in the text: license empty, redistribution restricted_local_only, access unknown.
- Rows: 1 (qiu2026-a, 1.75 mm differential MZM, DR4 chip). repro_grade C, no sim config (metrics only).
- Not reported: wavelength, Vpi frequency/bias, film thickness, electrode type and geometry, RF loss, Z0, n_rf, ng, energy per bit, IL of the device alone beyond the 3 dB on-chip statement.
- Judgment calls: Vpi 6 V ("differential Vpi of the bare die") entered in vpi_dc_v with convention mzm_differential although DC vs RF is not stated. Only the 3 dB on-chip IL is entered (excluding 6 dB splitter, 2 x 1.25 dB couplers, splices); connectorised fiber-to-fiber 12.5-14 dB and the 11.5 dB PIC budget are in notes/context. Max net rate 540 Gb/s is per lane (PAM8, 225 GBd, 2 km); title's 4x448 Gbps is not the results-text net rate (PAM4 net 422 Gb/s per lane). integration = other (MBE-grown BTO transferred; text also says monolithic).
- CSV hints: access guess unknown kept; license empty; published_on left empty (see convention re-check).

## ogiso2024 - distilled
- Source: preliminary OFC proceedings PDF, 3 pages; Table 1 and all figures read from page renders (arrows in Table 1 resolved to the column on the left). License publisher-copyright from the paper's own notice, restricted_local_only.
- Rows: 4 (a low-Vpi 1.5 V, b C+L 2.0 V, c electrode-modified variant, d 130 GBd-class reference). repro_grade C, no sim config (InP).
- Not reported: system demonstration (the beyond-200 GBd claim is bandwidth-based), RF loss, n_rf, ng, geometry, layer thicknesses/MQW composition, driver, energy, publication date.
- Judgment calls: Vpi convention mzm_series_push_pull following Table 1 "differential input / series push-pull drive" (Fig. 2c axis is differential voltage). Bandwidths are bounds (gt). Convention (c) asks for bw3db_ghz = measured-to value; the paper states "exceeds 100 GHz" (3 dB) while the Fig. 4 axis ends at 110 GHz, so bw3db_ghz keeps the stated 100 (gt) and bw_measured_to_ghz is 110 (instrument limit not stated); integrator to confirm. 0 dB reference unspecified. On-chip IL 3.5 dB is the Table typical value with an lt qualifier on row a (abstract: less than 3.5 dB), basis derived (coupling loss is excluded, so it is subtracted); per-polarization fiber-to-fiber < 11.5 dB entered only for row a. Z0 60 ohm entered with basis design_target (text: "designed to be around 60 ohm"). Rows c and d are extra devices characterized in the paper (Fig. 4b, 4a/Table 1); drop them if the integrator prefers headline-only rows. Row d bw3db 70 GHz (text) conflicts mildly with Table 1 ">67".
- CSV hints: platform inp_mqw / iq_mzm / priority 1 consistent; "MQW" is not stated in this paper (n-i-p-n heterostructure), inp_mqw is the closest eo_material enum.

## porto2026 - distilled (design-target row)
- Source: OFC 2026 paper PDF, 3 pages; Fig. 3 inset table read from the page render. License publisher-copyright from the paper's own notice, restricted_local_only.
- Rows: 1 (porto2026-a, one MZM channel of the 8-channel DFB/MZM/SOA PIC). repro_grade C, no sim config.
- MZM Vpi (< 1.5 V, RF), ER (> 25 dB) and the 53 GHz bandwidth are stated as design targets ("designed for"), not measurements. Vpi and ER are entered with basis design_target (vpi_basis design_target); the 53 GHz is not entered because it is not defined as a 3 dB bandwidth. Measured content: 106.25 GBd PAM4 (212 Gb/s) at 60 degC direct drive with a SiGe driver, SOA/DFB characterisation (context only).
- Not reported: measured Vpi, wavelength, length, geometry, IL of the MZM, bandwidth type, Z0, n_rf, energy per bit, fabrication site.
- CSV hints: platform/priority consistent; the paper is closer to a transmitter-PIC system paper than an EO modulator device paper, kept as a row because it reports modulator metrics and a system result.

## tanaka2026 - distilled
- Source: OFC 2026 paper PDF, 3 pages; Figs. 4-6 read from the page render. License publisher-copyright from the paper's own notice, restricted_local_only.
- Rows: 2 (a 2 mm / 60% FF section from Fig. 5; b 1.0 mm / 80% FF section, Vpi 5.4 V, EO response and 64 GBd 16QAM oDAC demo). repro_grade C, no sim config (InP).
- Not reported: wavelength, IL, Vpi in volts for row a, bandwidth of the 2 mm section, RF loss, Z0, n_rf, ng, layer thicknesses, fab site.
- Judgment calls: Vpi*L 0.43 V*cm entered as stated; it is reproduced as Fig. 5 span about 3.6 V x (0.6 x 2 mm) and 5.4 V x (0.8 x 1.0 mm), i.e. an FF-weighted length, which is an inference (the authors do not state the length used). Vpi 3.6 V for row a is read from Fig. 5. Row b Vpi 5.4 V entered in vpi_dc_v with convention unspecified (Fig. 6 caption "EO response at Vpi of 5.4 V"). 3 dB bandwidth about 60 GHz (approx qualifier), reference unspecified. integration = wafer_bonded_iii_v (chip-on-wafer direct bonding); electrode_type cl_twe with a segmented tag. 64 GBd x 4 bits/symbol = 256 Gb/s kept out of the CSV (context only).
- CSV hints: consistent.

## Schema/skill gaps hit
- No way to say "Vpi measurement frequency unspecified": papers give "Vpi 6 V" or "Vpi 5.4 V" without DC/RF; I used vpi_dc_v with notes. Suggest a `vpi_freq_note` or allow vpi_convention-style `vpi_freq_class: dc|rf|unspecified`.
- Design-target numbers (porto2026) are allowed by basis `design_target` but plots will mix them with measurements unless filtered on vpi_basis.
- bw3db_reference `unspecified` is used where the plot is normalised at low frequency without a stated frequency (ogiso2024, deng2026, tanaka2026).
- integration enum has no value for epitaxial-then-transferred BTO (qiu2026); `other` used.
- Evidence entries are limited to 25 words per note, so convention explanations moved to devices.csv notes and `context_values` (non-validated key kept per the pilot practice).
- Table-with-arrows readings (ogiso2024 Table 1) have no convention; resolved to the column on the left and said so in notes.
- Sim contract gaps are in SPEC_PROPOSALS.md (undisclosed eps_r, BTO crystal notation, single-arm drive factor, one-gap GSG).

## Audit corrections (audit data/_staging/audits/p1_01-q1-fresh.md, applied 2026-10-01)

Each finding was checked against the sources first. Dry-run merge afterwards: 0 conflicts, 0 validation errors.

- D1 changed: deng2026-a drive and vpi_convention evidence basis -> derived, note that the authors do not say which arm the Vpi sweep drives or name the convention; CSV notes reworded.
- D2 changed (partly): strip width in sims/deng2026/config.yaml 1.4 -> 1.6 um (region +-0.8 um), provenance gives the range 1.4-1.7 um with the Fig. 4(b) SEM bar as a second scale; `missing` text updated. My own 4(c) reading (1.4 um) used the gap as scale; the SEM and 4(c) readings differ within figure-reading error.
- D3 changed: r42 provenance class -> project_inference (paper gives a bound, "exceeding 358 pm/V", from a separate 180 nm film).
- D4 changed: gold sigma class -> project_inference with an "unverified, no citation" note (no source available offline).
- D5 changed: added the Fig. 4(c) scale-bar inconsistency (2 um bar vs 5.5 um gap) to the signal-electrode note; 5.5 um text value kept.
- D6 changed: 40 nm self-buffer noted as observed in the 180 nm film and presumed for the device film (evidence stack entry and note).
- D7 changed: vpi_dc_v locator now includes p.9 Methods.
- Q1 changed: authors spelling "Francois" -> "François" (Crossref); no other diacritics in the five Crossref author lists.
- Q2 changed: tag linear_driver -> rf_driver_packaged.
- Q3 changed: notes state that the 448 Gbps title/Sec. 1/Sec. 5 line-rate figure was deliberately not entered in max_line_rate_gbps.
- Q4 changed: qiu2026 3 dB on-chip IL basis derived -> measured (paper-stated budget item, no derivation given); il_basis measured.
- Q5 changed: notes say no single fiber-to-fiber value is entered because it is a per-channel range.
- O1 changed: ogiso2024-b bw_measured_to_ghz removed (empty; no measurement limit stated for this variant). Stated bounds bw3db >100 and bw6db >110 stay.
- O2 changed (partly): ogiso2024-c eo_rolloff_db 7 -> 6.5 at 100 GHz (approx). My re-read of Fig. 4(b) puts the 6 dB crossing near 99 GHz and the trace between -6 and -7 dB at 100 GHz; the audit's about 6 is inside that range.
- O3 changed: ogiso2024 on-chip IL basis derived -> measured (typical Table value, no derivation described) for rows a, b, d; il_basis measured. Kept the lt qualifier on row a only: the abstract (a) says "less than 3.5 dB" for the headline device; Table values for b and d are plain typicals.
- O4 changed: MQW-not-stated caveat added to the ogiso2024 paper notes (eo_material inp_mqw is the closest enum).
- O5 changed: ogiso2024 paper notes record that the cached copy carries an institution-licensed-use footer and must not be redistributed.
- P1 changed: porto2026 notes say the EO mechanism is not stated and inp_mqw is the closest enum.
- P2 changed: vpi_rf_v evidence note mentions the abstract's "up to 1.5 V"; lt kept.
- T1 changed: the "unreconciled" wording removed from papers.csv, devices.csv and this report; replaced by the FF x length reproduction (3.6 V x 0.6 x 2 mm = 0.43; 5.4 V x 0.8 x 1 mm = 0.43), labelled an inference.
- T2 changed: tanaka2026-a vpi_dc_v 3.6 V entered (extracted_from_figure, approx, mzm_differential), vpi_basis extracted_from_figure.
- T3 changed: notes on both tanaka2026 rows explain that Vpi x physical length (0.72, 0.54) differs from the stated 0.43 because of the inferred effective length.
- T4 changed: tanaka2026-a vpi_convention evidence basis -> derived (drive stays measured: the caption states push-pull).
- X1-X10 (convention/skill items) not actioned here; they are for the coordinator. For X1, ogiso2024-a keeps the stated bound as bw3db_ghz with the plotted range in bw_measured_to_ghz, as the coordinator confirmed.
