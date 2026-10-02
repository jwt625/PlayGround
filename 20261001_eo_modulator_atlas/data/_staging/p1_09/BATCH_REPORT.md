# BATCH_REPORT p1_09 (2026-10-01)

Validation: `uv run python scripts/merge_staging.py data/_staging/p1_09` -> merge counts papers 5, devices 9, orgs 7, evidence 5; conflicts 0; validation errors 0. `uv run pytest -q`: 28 passed. Engine not run (per instructions); both sim configs checked against `engine/schema/sim.schema.json` with a YAML 1.2 parser and for electrode overlap / conductors inside the optical window.

Re-check against the revised conventions (a)-(k) (coordinator message): done on all rows. Applied: per-arm headline for niels2026 with push-pull value in `vpi_mzm_pushpull_dc_v`; qualifiers on every "about/approximately/over" statement; bandwidth without crossing as `bw3db_ghz:gt` plus `bw_measured_to_ghz`; `drive` with evidence entries (basis derived where the authors do not say push-pull); `eo_rolloff_db` filled where a plot shows it (lin2025-a/-b at 110 GHz, li2026ba-a near 110 GHz); author-computed values as `derived`; RF-line values only on the device they belong to (lin2025 test-CPW loss not copied to the MZM rows); locators are PDF page indices of `text.md` ("p.N"); `published_on`/license only where Crossref or the paper's own notice supports them.

## li2026 - status: distilled
- Rows: 1 device (li2026-a, 8 mm suspended LTOI MZM). repro_grade C; sim config: none.
- Version: arXiv v1 (2026-07-19), numbers from that PDF; no crossref.json (no DOI).
- Not reported: waveguide width, undercut width Wc (tuned per chip), ground width, modulator IL, propagation loss (ER only visible in Fig. 2(a), about 20 dB), Z0, drive Vpp, on-chip power (17.8 dBm is the laser). Fiber-to-fiber coupling loss (-5.2 dB) is coupling only; kept in notes.
- Judgment calls: G (T-bar tip-to-tip gap) entered as `electrode_gap_um`; rf_loss 5.0 dB/cm and n_rf/ng 2.24 read from Fig. 2 (text gives S21 -4.3 dB at 120 GHz = 5.4 dB/cm if all loss is conductor); VpiL = 4 V cm stated by the authors (`derived`); `drive` push-pull with `mzm_push_pull` as MZM-level; license left empty and redistribution `unknown` (no notice in the PDF).
- Source issues: caption panel letters (a)-(f) do not match rendered panels (a)-(e); text says "for the LiNbO3 platform" in the NDR sentence (typo).
- CSV hints: none wrong. platform `lithium_tantalate` ok, sim_candidate unknown -> C (no config).

## li2026a - status: distilled (text-only)
- Rows: 2 (li2026a-a C-band 1550 nm, li2026a-b O-band 1300 nm). repro_grade C; sim config: none.
- Source: pre-extracted text only, no figures; 67 GHz bandwidth, Vpi and net rate read from text. BW normalization reference unknown (`unspecified`). OFC version: numbers differ from li2026ba (67 vs 64 GHz, 1.56 vs 1.53 V, 437 vs 440.6 Gbit/s; 400 nm etch/200 nm slab vs 440/160).
- Not reported (text): electrode dimensions (figure only), metal thickness, ER, IL, S-parameters/Z0/n_RF, O-band bandwidth.
- Judgment calls: tip-to-tip gap 5 um derived from "1.5 um spacing" + 2 um waveguide; `integration: other` (film bonded to a fused-silica carrier); Vpi values are the authors' conversion of VpiL (`derived`).
- CSV hints: `published_on` (2026-03-18, conference schedule) not verifiable from Crossref (year only) -> left empty. Luxtelligence SA is an affiliation here (company org added). Access unknown; redistribution restricted (only "(c) 2025 The Author(s)").
- Could not read: all figures.

## li2026ba - status: distilled
- Rows: 2 (li2026ba-a C-band, li2026ba-b O-band, same 18 mm device). repro_grade B; sim config: `sims/li2026ba/config.yaml` (device li2026ba-a).
- Version: arXiv v1 (2026-04-16); no crossref.json.
- Not reported: rib width and sidewall (SEM-digitized: ~2 um mid-height), ground width, IL, propagation loss, drive Vpp, on-chip power, O-band bandwidth. Fiber-to-fiber coupling loss ~12 dB is coupling only (notes).
- Judgment calls: VpiL 2.76 / 2.19 V cm read from Fig. 2(c) field F11 (the Vpi device); T-segment parameter G identified as tip-to-tip gap (Fig. 3(a) schematic, cross-checked by SEM 4.9 um); `drive` push-pull is `derived` (not stated in this version); EO roll-off 7 dB near 110 GHz read from the smoothed curve; license empty, redistribution `unknown`.
- Sim: LT RF permittivity (41/43) is kept only as an UNVERIFIED placeholder with provenance class unknown (no source opened); r33 = 30.5 pm/V cited to niels2026 p.9; r13, r22, r51 omitted (listed in `missing`). Propagation direction, ground width and cladding profile are project inferences.
- CSV hints: id renamed from li2026b (batch builder); the merge should confirm no seed collision. sim_candidate yes confirmed.

## lin2025 - status: distilled
- Rows: 3 (lin2025-a 16 mm, lin2025-b 6 mm, lin2025-c IMDD demonstration device; the paper does not say which length was used for data transmission). repro_grade B; sim config: `sims/lin2025/config.yaml` (device lin2025-a).
- Version: accepted unedited Nature Communications manuscript (journal AM), not the arXiv preprint; Crossref 17(1), article 3211.
- Not reported: wavelength of the Vpi/EO measurements (main text), Vpi/ER for the 6 mm device, MZM signal width (27 um is a test-CPW dimension; used only in the sim config as project_inference), IL, propagation loss, Z0/n_RF per device (Fig. 2 shows only percent variations), on-chip power for the Vpi runs (1.17 W tested, device not stated; supplementary not available).
- Judgment calls: third row for the system demo; `ng_opt` 2.22 as `author_estimate` (method not stated); eo_rolloff 6.2 dB (16 mm) / 3.8 dB (6 mm) at 110 GHz read from Fig. 3(d); Fig. 1(f)/(h) do not scale to the stated 4 um/6 um (text values used, flagged in the config); VpiL 2.7 V cm `derived` (stated by the authors from 1.7 V x 16 mm).
- License CC-BY-NC-ND-4.0 (Crossref + paper notice) -> redistribution restricted_local_only. `published_on` 2026-02-26 from Crossref; the CSV arXiv date 2025-05-07 is not verifiable offline. Year 2026 (journal) with paper_id lin2025.
- Could not read: supplementary figures/tables (Fig. S1-S14, Table S1).

## niels2026 - status: distilled
- Rows: 1 (niels2026-a, 7 mm hybrid SiN/LiTaO3 MZM, 6.6 mm active). repro_grade C; sim config: none (electrode widths and oxide thicknesses not reported; FIB/schematics only).
- Version: the cached PDF is the arXiv v1 manuscript (PDF created 2025-03-14, includes supplementary), not the Nature Photonics 20(2) version; the journal numbers may differ. Crossref gives only publisher TDM license terms -> redistribution restricted_local_only.
- Not reported: electrode signal/ground widths, oxide thicknesses, HR vs standard Si for the Vpi/IL/data runs, drive Vpp, optical power, 3 dB bandwidth (>70 GHz, limited by setup), EO roll-off value (curve too noisy to read), standard-Si EOE number.
- Judgment calls: headline `vpi_dc_v` 7.0 V is per arm (authors' measurement) with the 3.5 V push-pull value in `vpi_mzm_pushpull_dc_v`; `vpil_dc_vcm` left empty because the paper's 2.3 V cm refers to the push-pull value (mixing conventions); IL 2.9 dB entered as on-chip with includes/excludes (1.6 dB term is simulated, basis `derived`); `drive` push_pull (stated by the authors; the feed is a differential GSSG signal, see Audit corrections); line rate 320 Gbit/s as `gt` (authors: "more than 320"); `electrode_type: other` for GSSG.
- CSV hints: priority/platform ok; `published_on` left empty (see Audit corrections N3).

## Organizations added (7)
Swiss Federal Institute of Technology Lausanne; Karlsruhe Institute of Technology; EPFL Center of MicroNano Technology (facility); EPFL Institute of Physics cleanroom (facility); Luxtelligence SA; Ghent University; imec (name as in the paper, not expanded). Other batches may spell EPFL/imec differently; the integrator should unify (the merge only compares type/country/region).

## Schema/skill gaps
- No verified LT RF permittivity / Pockels table in the repo (see SPEC_PROPOSALS item 1).
- `vpil_dc_vcm` when the headline convention is per-arm and the paper reports VpiL only for the push-pull value: left empty; a `vpil_mzm_pushpull_vcm` column would keep it.
- Pure system-demo row when the paper does not identify the device (lin2025-c): no rule in the skill.
- Conventions note: `bw3db_reference` has no value for "figure starts at 0 dB at the lowest plotted frequency" (used `unspecified`).

## Audit corrections (Q1 fresh audit, data/_staging/audits/p1_09-q1-fresh.md)
Each finding verified against the cached sources before changing. Dry-run merge after corrections: 0 conflicts, 0 validation errors; both sim configs re-validated against the JSON schema (YAML 1.2 parser).

High
- BC1 / LC1: changed. LT `eps_r` 41/43 now provenance class `unknown` with an explicit UNVERIFIED note and no citation (kept only so the config stays runnable); `r13` removed from both configs and listed in `missing`; `r33` kept, cited to niels2026 p.9 (verified in the cached text). Weis and Gaylord citation deleted from both LT entries.

Medium
- BC2: changed. Gold sigma now 3.9e+7 S/m, class `project_inference`, from the lin2025 thin-film Au resistivity 2.56 uOhm cm (text p.7, Fig. 2(c)-(d)); the SPEC-example origin of the old 4.1e7 is gone.
- L1: changed. li2026-a `bw_measured_to_ghz` = 110 (extracted_from_figure); note says the smoothed fit extends past the raw data.
- N1 (lin2025-a): changed. Text value 40 GHz kept; evidence note carries the Fig. 3(d) reading (-3 dB near 50 GHz from 0 dB at 1 GHz).
- N1m: changed. niels2026-a `drive` = push_pull (authors state push-pull, p.3, membranes at 180 deg); the differential GSSG feed is in the note; basis `design_target`. Added `vpi_convention` = per_arm_phase_shifter entry with locator p.4 / Fig. 5(b).
- N2m: partly changed. `vpil_dc_vcm` stays empty (convention (a)); the push-pull VpiL 2.3 V cm now also sits in the evidence note of `vpi_mzm_pushpull_dc_v`. A dedicated column remains a schema proposal.
- N3: changed. niels2026 `published_on` emptied; tag `numbers_from_arxiv_v1` on the device; arXiv v1 stated in the paper notes. (Crossref 2026-01-13 dates the journal object, the arXiv date is not verifiable offline.)

Low (changed)
- L2: li2026-a ER 20 dB approx (Fig. 2(a) dip, static, lower-bound caveat); report wording fixed.
- L4: evidence note for li2026 G now mentions the ground-only caption wording.
- A1, A2: evidence notes (etch depth read as 600 - 200; "around" attaches to the 1550 nm value).
- A4, A5: li2026a `slow_wave` tag dropped; tag `ofc_version_of_li2026ba` added.
- B2: li2026ba rib note and config polygon changed to base ~2.1 um (+-1.05), top 1.8 um.
- BC4: 15 nm Ti layer added to li2026ba limitations. BC5 / sigma syntax: both configs write `3.9e+7` / `4.9e+7`.
- LC2: silicon_dioxide n and silicon n cited to the engine library sources (Malitson, Salzberg and Villa); eps_r 3.9 flagged project_inference UNVERIFIED; silicon eps_r 11.7 cited to li2026 p.1.
- N2 (lin2025): `optical_power_handling_dbm` = 30.7 (gt, derived list) on lin2025-c as a tested level.
- N3 (lin2025): shared geometry copied onto lin2025-c.
- N4: `ng_opt:approx` removed on lin2025-a; note says the +-0.03 belongs to the microwave indices.
- N4 (niels), N8: notes added (trace touches -3 dB near 65 GHz, EOE definition not stated; ER is passive-MZI fringe contrast).
- N5: niels2026 license cell now "publisher TDM terms only; no open license verified".
- N6: notes say imec covers the Si/SiN wafers; the LT steps were done by the authors at Ghent University-imec. Org note for the IPHYS cleanroom states the EPFL prefix is inferred.

Disagreed / no change
- B1 (and the li2026 -5.2 dB): not entered in `il_fiber_to_fiber_db`. Both sentences define a coupling loss, not a modulator insertion loss; entering them in an IL column would mislead comparisons. They stay in notes.
- L3, N6 (basis `design_target` for stated architecture), L5 / B3 (integration monolithic vs other): left as is; no enum fits, listed as skill issues X6, X7.
- A3 (section locators for the text-only source): unavoidable; stays.
- N5 (published_on of lin2025), N7, N9, B4, LC3, BC3: audit confirms correct, no change.
- X-items: applied inside this batch only as stated above (X8 for L1, X10 for N1); X2, X3, X9, X11 remain open convention proposals.
