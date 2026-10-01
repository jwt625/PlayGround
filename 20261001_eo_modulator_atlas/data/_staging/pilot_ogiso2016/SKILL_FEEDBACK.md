# Skill feedback: pilot ogiso2016 (InP n-i-p-n CL-TWE MZM, Electronics Letters 2016)

Date: 2026-10-01. Paper chosen: 2016 Electronics Letters letter (2 pages). Judged adequate as a pilot (single device, headline Vpi / BW / loss / ER / 100G NRZ demo, plus all the convention traps: series push-pull, bandwidth with 1.5 GHz reference and no 3 dB crossing, fiber-to-fiber vs on-chip loss, bias-dependent Vpi). The 2025 O-band PIC paper was not processed.

## Time per step (wall clock, approximate; tool time only, reading/thinking not separated)

| Step | Time |
|---|---|
| Read SKILL.md, plan, schema, SPEC | ~1 min |
| Crossref lookup (2 calls, >= 3 s apart) | ~1 min |
| extract_source.py | < 5 s |
| Read text.md + 2 page PNGs | ~1 min |
| Write CSVs + evidence, dry-run merge | ~3 min |
| Total tool wall clock | ~2 min plus model time |

## Unclear in SKILL.md / schema / scripts

1. `figures.json` and the README say `figures/fig_<page>_<n>.png`; the script actually writes `img_pNN_n.png` and `page_NN.png`. Docs and code disagree.
2. Crossref step: SKILL.md gives `works/<doi>` but not the title-search query. The title in Crossref contains HTML (`<i>V<sub>pi</sub></i>`), newlines and a non-breaking space and a Unicode hyphen (U+2010) in "Mach-Zehnder" / "n-i-p-n"; titles must be normalized before writing papers.csv. Add a note (I wrote the title with the Unicode pi and ASCII hyphens).
3. Crossref `license` field is only a publisher T&C link; it cannot verify "paywalled" vs "open". SKILL.md rule 8 says verify license "on the source page", but rule 9 forbids workarounds and the task forbade other calls. For a paper where Crossref shows no CC license I set access=paywalled, redistribution=restricted_local_only; `access` is therefore inferred, not verified. Suggest: define `access` decision rule (no CC license in Crossref and no OA flag => paywalled).
4. `published_on`: "first public version if known". Crossref gives `created` 2016-09-20 (metadata deposit), `published-online` only month-precision (2016-10), the PDF prints E-first 2016-09-30. Rule needed on which wins (I used the printed E-first date).
5. `discovered_via` has no token for "local reading list / local PDF corpus supplied by the coordinator". The documented token for the private local cache would leak a private-project name into a file bound for a public repo (DevLog decision 1), so I used `landmark`. Suggest a neutral token such as `local_corpus`.
6. Organization name rule: "full, unabbreviated, from the paper's affiliation list" conflicts when the paper itself writes "NTT Corporation" (abbreviation of the legal name). I used "Nippon Telegraph and Telephone Corporation" (legal full name, which is not in the paper) and recorded the paper's wording in organizations.csv notes. Needs an explicit rule (legal name allowed? who verifies?). Sub-units (NTT Device Innovation Center, NTT Device Technology Laboratories) have no column; I put them in `research_groups`, which is defined as "PI / lab names". Acceptable but a `departments` column or rule would help.
7. `il_basis` is a single column for two IL fields. Here fiber-to-fiber 10 dB is measured while on-chip 2 dB is an author estimate (subtraction). I set il_basis=measured and gave the evidence entry for il_onchip_db basis=derived. Validator does not cross-check CSV basis columns against evidence bases, so the two can silently disagree. Also "derived" in the schema means project-derived (formula in `derived:` list), whereas here it is author-derived; a basis value `estimated_by_authors` is missing.
8. Evidence validator accepts only evidence:true columns; numbers that matter but have no column (drive amplitude 2.3 Vpp, input power +13 dBm, bias -5 V, excess absorption <0.5 dB, 27 dB ER at 1560 nm) cannot be given a locator. I added a non-validated `context_values:` block in the evidence yaml; the validator ignores it. Either bless that key or add columns (below).
9. `bw3db_reference` when `bw3db_ghz` is empty (no crossing): SKILL.md rule 3 implies filling `bw_measured_to_ghz` only. Enum has `1ghz`, `10ghz`, `dc`, `other`; this paper uses a 1.5 GHz reference -> `other`, and the exact 1.5 GHz has nowhere to go except the evidence note. Add `bw3db_reference_ghz` (float) or a `1.5ghz`-style catch-all.
10. Bound-valued results ("over 67 GHz", "exceeded 22 dB", "<0.5 dB"): the only schema mechanism is `bw_measured_to_ghz`. For ER I entered 22 (lower bound) as a plain number; a plotting script will read it as a value. Needs a convention, e.g. companion `*_bound` flag column or `>=` evidence field `bound: lower`. The validator's `values_match` would also need to ignore it.
11. `bw6db_ghz`: unclear whether it is EO or electrical 6 dB. The paper gives an electrical S21 6 dB BW "over 67 GHz" (Fig. 3). I left bw6db_ghz empty. The definition should say "EO 6 dB BW".
12. `max_baud_gbd` for NRZ: baud is not stated in the paper; 100 Gb/s NRZ implies 100 GBd. I entered 100 with an entry in `derived:` only (formula: rate / 1 bit per symbol). SKILL.md should say whether such values are left empty or derived; schema `derived` semantics are only described for vpil.
13. `z0_ohm`: paper says "single-ended 50 ohm design" and "impedance matching condition satisfied" but gives no measured/extracted Z0. There is no basis `design`. I left Z0 empty rather than enter a design target as a metric. Add basis `design_target` or explicitly say design values stay empty.
14. `wavelength_nm` with multiple wavelengths: Fig. 2 shows four wavelengths (1530-1560 nm) each at Vpi 2.0 V (bias-adjusted); the EO BW, S-parameters and IL wavelengths are not stated. Rule 5 suggests one row per wavelength, but the only wavelength-resolved number is ER (27 dB at 1560 nm); I kept one row at 1550 nm (the system-demo wavelength, with a Fig. 2 curve) and listed the others in notes. Guidance on when a wavelength sweep merits several rows would help. Wavelength for bandwidth is not stated, so wavelength_nm = 1550 is attached to the row, not to every metric.
15. `electrode_type`: enum `cl_twe` exists (good). `drive` vs `vpi_convention` for "series push-pull, single-ended 50 ohm design, differential voltage across the electrode": I used drive=push_pull, vpi_convention=mzm_push_pull. The paper's x-axis is "differential voltage" which could be read as mzm_differential; the SPEC meaning of `differential` (V+ minus V-) does not match a single-ended input driving a series push-pull pair. Needs enum `series_push_pull` (single RF input, two diodes in series) or a note in the schema.

## Missing columns needed (n-i-p-n heterostructure and general InP)

No column for epitaxy / layer stack. What was needed:
- `epi_stack` (string, ordered top to bottom: "n-InGaAs contact / n-InP / undoped InGaAlAs-InAlAs MQW / p-InAlAs EBL / n-InP / n-contact / SI-InP"), or structured: `junction_type` enum {p_i_n, n_i_p_n, n_p_i_n, other}.
- `core_material` / `mqw_composition` (here: InGaAlAs/InAlAs MQW; the paper gives no thicknesses, bandgap or PL wavelength).
- `waveguide_orientation` ([011] stripe direction; matters for the "synergistic EO effect", a first-order parameter for InP Pockels+QCSE devices) and `substrate_orientation` ((100) inferred only from the axes in Fig. 1, not stated in text; not entered). `crystal_cut` (examples "x-cut") does not fit; left empty.
- `bias_v` / `bias_for_vpi_v` and `bias_for_bw_v`: Vpi of InP MZMs depends on reverse bias (QCSE + linear EO); here Vpi 2.0 V is obtained "by adjusting the DC bias" (value not given) while EO-BW is measured at -5 V. Without a bias column, this is only recoverable from notes.
- `drive_amplitude_vpp`, `optical_power_in_dbm` (demo drive 2.3 Vpp, +13 dBm; `optical_power_handling_dbm` is a handling limit, not the same thing).
- `rf_input_config` or reuse `drive`: single RF feed vs differential vs dual-drive.
- Loaded-electrode parameters (period, loading capacitance): paper gives none, but for InP CL-TWE in general a place is needed (SPEC has `loading` but only for sim configs).
- `il_onchip_includes` (what the on-chip number includes); only `il_onchip_excludes` exists, and this paper defines nothing beyond "on-chip loss estimated to be 2 dB" after removing 4 dB/facet lensed-fibre loss.
- Static vs dynamic ER: `extinction_ratio_db` should be split (static DC ER >22 dB vs dynamic 100 Gb/s ER >10 dB); I entered static and put dynamic in context_values.
- `eo_material=inp_mqw` is fine for this device (MQW Pockels+QCSE in n-i-p-n), but a note on the physical mechanism would help comparisons.

## Pitfalls found

- Dates: `Submitted` / `E-first` / issue date / Crossref `created` all differ.
- The PDF's Wiley download banner states "OA articles are governed by the applicable Creative Commons License"; it is a generic footer and does not imply this article is OA. Do not use it for license.
- Page-render PNGs are essential: the pymupdf text of figures yields axis ticks only; the 3 dB non-crossing in Fig. 4 and the Vpi locus in Fig. 2 are visible only in the image.
- Fig. 4 EO trace ends near -2.7 dB (visual); "over 67 GHz" is the authors' wording, and the instrument limit is not stated. `bw_measured_to_ghz`=67 therefore means "authors' lower bound", not necessarily instrument limit.
- IL "10 dB entire insertion loss" with "4 dB/facet" coupling: 10 - 8 = 2 dB equals the stated on-chip loss; entering both is consistent but the 2 dB is derived by subtraction, so ratio-style fits should treat it as lower confidence. It also lacks wavelength and bias.
- Title and special-character handling: PDF filename contains non-ASCII hyphens; copying via shell glob worked (`ls | grep`), explicit quoting of the literal name would have failed.
- `extract_source.py` prints a uv warning about a foreign `VIRTUAL_ENV`; harmless, but unset `VIRTUAL_ENV` or use `--active` guidance in SKILL.md.

## Repro grade

C. The letter gives no numeric geometry (no ridge width/height, layer thicknesses, doping, electrode dimensions, loading period); Fig. 1 is a schematic without dimensions. Metrics-only. InP gets no sim config in any case.

## What the paper does not report (left empty)

vpi_rf_v, vpi_rf_freq_ghz, vpil_dc_vcm / vpil_rf_vcm (VpiL would be derived by build step from 2.0 V and 3 mm, flagged derived), bw3db_ghz (no 3 dB crossing observed), bw6db_ghz (EO), rf_loss_db_per_cm, rf_loss_freq_ghz, z0_ohm (50 ohm is a design statement), n_rf, ng_opt, prop_loss_db_per_cm, optical_power_handling_dbm, all geometry (film thickness, etch, rib width, slab, sidewall, gap, signal width, electrode thickness, buffer), electrode_metal, crystal_cut, temperature_class, net rate, energy per bit, fab/foundry, wafer supplier, process name.

## v2 revision (2026-10-01)

Staging re-done against schema v2; dry-run merge: 0 conflicts, 0 validation errors. Items 1-15 above that v2 addresses are now resolved as follows: epitaxy_or_stack and waveguide_orientation filled (text as stated, no thicknesses); drive=series_push_pull and vpi_convention=mzm_series_push_pull with evidence entries for the convention fields; bw_measured_to_ghz=67 with qualifier gt and bw3db_reference_freq_ghz=1.5 (bw3db_reference=other); extinction_ratio_db=22 with qualifier gt and er_type=static; il_onchip_db evidence basis author_estimate; drive_vpp_v=2.3 and optical_input_power_dbm=13 (both from the 1550 nm 100 Gb/s demo); discovered_via=local_corpus.

Remaining gaps:
- bias_for_vpi_v left empty: the paper sets Vpi = 2.0 V "by adjusting the DC bias" without a value; the -5 V bias is stated only for the S-parameter/EO measurement and is kept in notes and context_values. There is no column for bias at the bandwidth measurement (suggest bias_for_bw_v).
- il_onchip_includes left empty (paper defines the 2 dB only by subtracting 4 dB/facet coupling).
- Dynamic ER (over 18 dB at 50 Gb/s, over 10 dB at 100 Gb/s) and the 27 dB static ER at 1560 nm have no column; er_type holds one value per row. Kept in context_values.
- Electrical S21 6 dB bandwidth (over 67 GHz, Fig. 3) still has no clear column (bw6db_ghz is undefined as EO vs electrical); left empty.
- Qualifier on il_onchip_db (author "estimated to be 2 dB") not set; clarify whether author_estimate implies approx.
- Row basis columns (vpi_basis, bw_basis, il_basis) remain single-valued; il_basis=measured while the on-chip evidence basis is author_estimate (evidence wins per SKILL.md rule 11).
- context_values remains a non-validated block for numbers without a column (bias -5 V, excess absorption below 0.5 dB, 27 dB ER, S21 6 dB BW, dynamic ERs).
