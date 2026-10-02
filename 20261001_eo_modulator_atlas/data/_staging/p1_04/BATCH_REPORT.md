# Batch p1_04 report (2026-10-01)

Validation: `uv run python scripts/merge_staging.py data/_staging/p1_04` (dry run) gives papers 5, devices 21, orgs 9, evidence files 4, conflicts 0, validation errors 0. Offline, no network, no engine run. A first attempt was interrupted by an API session limit; staging was empty afterwards and everything was redone from the cached sources.

| paper | status | device rows | repro_grade | sim config |
|---|---|---|---|---|
| valdez2022 | distilled | 1 | B | sims/valdez2022/config.yaml (not runnable, see below) |
| xu2022 | needs_download | 0 (metadata row only) | none | none |
| meng2023 | distilled | 3 | B | sims/meng2023/config.yaml (not runnable) |
| renaud2023 | distilled | 9 | B | sims/renaud2023/config.yaml (not runnable) |
| valdez2023 | distilled | 8 | B | sims/valdez2023/config.yaml (CN1 only, not runnable) |

All four configs omit the constants the papers do not give (LN indices and RF permittivity, SiO2, Si, gold conductivity) and list them under `missing`, like sims/deng2026. No recalled constants were inserted. The only non-paper numbers are two flagged geometry placeholders: BOX 3 um in valdez2023 (taken from the cached sister paper valdez2022 p.5) and BOX 2 um in meng2023 (borrowed from the cached renaud2023 p.2, a different group; weakest assumption in the batch). Pockels values used are the papers' own (r33 = 30.8 pm/V valdez2023 p.10; about 31 pm/V meng2023 p.2).

## valdez2022 (Sci. Rep. 12, 18611)
- Source: cached text is arXiv v1 (26 Oct 2022, shown on p.1), not the journal PDF. Numbers come from it; journal version not compared. published_on = 2022-10-26 (arXiv stamp, journal online 2022-11-03 per Crossref). License CC-BY-4.0 is the journal's (Crossref); the arXiv licence is not in the cache, so redistribution = restricted_local_only.
- Row valdez2022-a: 5 mm, 1550 nm, VpiL 3.1 V*cm (mean over 0.1-10 MHz, trapezoid overdrive), 3 dB BW 110 GHz (gt, abstract says greater than 110; body says 110; bw_measured_to 110; reference unspecified, curve normalized near 1 GHz), IL 1.8 dB on-chip (derived: 12.2 dB fiber-to-fiber minus 2 x 5.2 dB coupling), ER 28 dB (mean of passive fringes, static), 110 mW = 20.4 dBm optical power tested (gt, quasi-CW, not a damage limit), ng 2.32 and n_rf 2.34 (both simulated), gap 9 um, signal 55 um, 600 nm LN, 0.75 um Au.
- Judgment calls: drive push_pull and vpi_convention mzm_push_pull are inferred from the GSG geometry (one arm per gap) and marked derived in evidence; Z0 = 42 ohm not entered because the paper does not say whether simulated or measured (in notes and as a sim target with basis unspecified); hybrid propagation loss 0.6 dB/cm is cited from earlier work and not entered; optical_input_power_dbm left empty (IL at 1 mW, BW at 4 mW and 110 mW, Vpi at unstated power); fab not named, foundry_or_fab empty.
- Not reported: RF loss, ground electrode width, waveguide lateral position, bias, temperature, Vpi in volts (implied 6.2 V, notes only).
- Fig. 3(c) micrograph scale (200 um bar) is inconsistent with the stated 55 um signal and 9 um gap; text values used.

## xu2022
- needs_download: no source in references/xu2022 (only crossref.json). Entry in needs_download.md. papers.csv row is metadata only (identity, date, licence from Crossref, cache_status needs_download). The Crossref abstract metrics (sub-1 V, 110 GHz, 1.96 Tb/s net) are not entered anywhere as data.

## meng2023 (arXiv 2311.05119)
- Source: arXiv preprint only, version not shown, no Crossref record. published_on left empty (the arXiv date in the batch CSV cannot be verified offline); license "arXiv-nonexclusive" is the prefetch metadata, no notice in the text; restricted_local_only. No journal version identified.
- Row meng2023-a: 5 mm, 1310 nm, Vpi 2.04 V, VpiL 1.02 V*cm (abstract; Sec. 1 and Table 1 say 1.024), 3 dB roll-off at 108 GHz (bw_measured_to 110 read from the plot axis, reference unspecified), dynamic ER 1.98 dB at 224 Gb/s PAM4, ng about 2.29 (simulated), 224 Gb/s PAM4 at 112 GBd, ITO gap 3 um, Au gap 5 um, 100 nm buffer. Film thickness 500 nm is ridge + slab (derived). Rows meng2023-b (8 mm with TCO, ER 6.0 dB) and -c (8 mm without TCO, ER 1.3 dB) carry only the eye-diagram ER comparison; both flagged as minimal.
- IL 2.9 dB at 1310 nm is not entered: the paper does not define it (grating couplers, no on-chip/fiber split), and the text writes "3-mm TCO gap" (typo for 3 um).
- Not reported: signal/ground widths, BOX and cladding thickness, RF loss, Z0, n_rf, optical input power.

## renaud2023 (Nat. Commun. 14, 1496)
- Source: published article text (CC-BY-4.0, Crossref); supplementary figures/tables not available.
- Rows: 3 um gap, 1 cm MZM at 532/638/738/838/938 nm; 738 nm gap series 2.5/4/5 um; one phase modulator at 737 nm. Vpi 1 MHz triangle. Text values used for 532, 638, 738, 838 nm (0.42, 0.45, 0.55, 0.85 V); the plot points differ slightly (about 0.43, 0.47, 0.56, 0.81). 938 nm and the 2.5/4/5 um gap points are read from Fig. 2(a,b) (extracted_from_figure, approx).
- Headline row (3 um, 738 nm): VpiL 0.55 V*cm, 3 dB BW about 35 GHz (reference 3 GHz, other; abstract says in excess of 35; theory 36 GHz; bw_measured_to 40 read from axis), ER about 21 dB (static), prop loss about 0.7 dB/cm, about 15 dB fiber-to-fiber (wavelength of both not stated in main text), RF loss 7.99 dB/cm at 35 GHz derived from the stated 1.35 dB/cm/GHz^0.5 coefficient, n_rf 2.22 at 50 GHz and ng about 2.38 simulated, Z0 design target about 50 ohm.
- Phase-modulator row: Vpi about 1 V at 100 MHz in vpi_rf_v; geometry columns left empty (not restated); drive and convention inferred.
- Not reported: electrode signal/ground widths, whether electrodes contact the slab through the cladding, bandwidth of the phase modulator.

## valdez2023 (Opt. Express 31, 5273)
- Source: cached text is arXiv v2 (25 Jan 2023), not the journal PDF; journal version not compared. published_on = 2023-01-30 (Crossref; the arXiv v1 date 2022-11-09 in the batch CSV is not verifiable offline). License Optica-OA-License-v2 (Crossref) for the journal; restricted_local_only because the cached file is the arXiv manuscript.
- 8 rows named as in the paper (CW1, CW2, CN1, CN2, OW1, OW2, ON1, ON2); length/label mapping verified against the Fig. 5, 6, 7 legends and captions (note the CW/CN legends use 1 = 1.0 cm, the OW/ON legends use 1 = 0.54 cm for OW only). Vpi (1 kHz) from Fig. 7: 2.93, 5.59, 3.11, 5.78, 3.78, 2.01, 2.6, 4.37 V. VpiL not entered per device (Fig. 9(a) bars only); push-pull stated by the authors (Eq. 2).
- 3 dB bandwidth per device is read from the Fig. 9(b) bar chart by pixel measurement (about +-1 GHz): 54, 79, 69, 109, 94, 54, 66, 102 GHz, reference 1 GHz, measured to 110 GHz. This contradicts the text claims (greater than 100 GHz for 0.54 cm, greater than 60 GHz for 1.0 cm) for several devices (CW1 and OW2 about 54 GHz, CW2 about 79, OW1 about 94); both are recorded, the bars are used in the columns. Authors also note the traces never reach the -6 dB point and the response is gently sloped.
- RF loss (less than 9 dB/cm at 110 GHz) and Z0 (about 40 ohm) are text statements covering all eight devices, entered as such; per-device Fig. 4 curves not read. ng simulated per design (2.38, 2.31, 2.43, 2.30).
- IL 1.6 dB (0.54 cm) and 2.1 dB (1.0 cm) and phase-shifter loss 1.5 dB/cm are entered on the CN rows only (the text derives them from CN1 vs CN2 and does not say they apply to other designs); they are not mutually consistent ((2.1-1.6)/0.46 = 1.1 dB/cm) and the 2.1 dB conflicts with "less than 2 dB" in the introduction; kept as stated.
- Wavelength is the nominal band (1550/1310 nm); optical input power is laser output (+9 dBm C, +12 dBm O), not on-chip. Text elsewhere says OW/ON gap 6/8 um and CW 7 um; Sec. 3.1 labels T-rail G = 6 and 7 um consistently with Sec. 4.1.
- Not reported: ER, bias, temperature, signal width, BOX thickness, crystal cut for this paper (x-cut only inferred in the config from valdez2022).

## CSV hints
- valdez2022: the hint that this is a hybrid Si/LN MZM at 1550 nm is right; sim_candidate yes (but constants missing). valdez2023 platform_guess lithium_niobate is a hybrid Si/LN device (waveguide_platform lnoi_loaded_si). meng2023 priority/sim_candidate fine; the O-band claim is right (1310 nm). xu2022 and renaud2023 hints right (renaud2023 is not hybrid: etched TFLN rib, visible-NIR).

## Organizations
New in staging organizations.csv (9): University of California, San Diego; Sandia National Laboratories; San Diego Nanotechnology Infrastructure (facility, parent UCSD, from valdez2023 acknowledgements); Huazhong University of Science and Technology; Huawei Technologies; Harvard University; Center for Nanoscale Systems; California Institute of Technology; Agency for Science, Technology and Research (parent of the IMRE affiliation). UCSD, Sandia, Harvard and CNS rows are identical to the p1_02 staging rows (same names and attributes), so integration should dedupe cleanly. None of the 19 names in data/organizations.csv applies to this batch; no EPFL-type names occur.

## Schema and skill gaps
- Slot (inductive) slow-wave loading has no SPEC loading type; configs use `periodic_t_rail` as the closest form (SPEC_PROPOSALS.md). Non-conductor materials require `eps_r` in the JSON schema, so configs are not schema-valid until supplied.
- Bandwidth referenced to 3 GHz (renaud2023) fits only `bw3db_reference: other` plus the frequency column; SPEC targets have no reference-frequency field.
- No field for a text statement that applies to a group of devices (valdez2023 RF loss and Z0); used the per-row bound/approx qualifiers.
- For arXiv-only or arXiv-sourced rows the redistribution rule has no clean case when the journal licence is open but the cached copy is the arXiv manuscript (valdez2022, valdez2023): set restricted_local_only; coordinator may change after checking the arXiv licence.
- Could not read: supplementary material of renaud2023 (propagation-loss figure, tables), journal versions of valdez2022/valdez2023, and any source for xu2022.
