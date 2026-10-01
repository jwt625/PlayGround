# BATCH_REPORT p1_03 (TFLN: capacitively loaded / segmented electrodes plus three papers without source)

Date: 2026-10-01. Validation: `uv run python scripts/merge_staging.py data/_staging/p1_03` (dry run) reports merge counts papers 2, devices 3, orgs 1, evidence 2; conflicts 0; validation errors 0. No network requests; no git; no engine run. Tsinghua University already exists in `data/organizations.csv` (applied from p1_01), so only HyperLight is new.

Staging outputs: `papers.csv` (2), `devices.csv` (3), `organizations.csv` (1 new), `evidence/` (2 files), `SPEC_PROPOSALS.md`, `needs_download.md` (3), this report. Sim configs: `sims/kharel2021/config.yaml`, `sims/liu2021/config.yaml` (checked against `engine/schema/sim.schema.json` with jsonschema: only the `4.1e7` YAML-1.1-string artifact that chen2022 also has; no conductor inside the optical window).

## kharel2021 - distilled
- Source: the cached text is the arXiv v1 preprint (stamp 26 Nov 2020, text dated 30 Nov 2020), not the Optica journal version (Crossref: online 2021-03-09, print 2021-03-20, Optica 8(3) 357). Numbers may differ from the journal version; stated in papers.csv notes. published_on = 2020-11-26 (arXiv stamp). License: Crossref VOR record is Optica OA License v1; redistribution `unknown` (terms not checked; arXiv v1 license not in the PDF).
- Rows: 2. kharel2021-a (10 mm: Vpi 2.3 V at 1 GHz, 0.8 dB roll-off at 50 GHz, RF loss 2 dB/cm at 50 GHz) and kharel2021-b (20 mm headline: DC Vpi 1.35 V, Vpi 1.3 V at 1 GHz, 1.8 dB roll-off at 50 GHz, ER about 20 dB, on-chip loss below 1 dB, fiber-to-fiber 13 dB). repro_grade B.
- Sim config: `sims/kharel2021/config.yaml` (device -b, 20 mm; targets measured and Table I simulated kept separate).
- Not reported: DC Vpi for the 10 mm device, Vpi*L (not entered, derivable), propagation loss, Z0 (only simulated 42 ohm in Table I), ng, fabricated signal/ground widths, ridge (waveguide) width and sidewall angle, wafer-level temperature, optical power, system demonstration, energy per bit, fabricator.
- Judgment calls:
  - No 3 dB bandwidth entered. The authors give roll-offs at 50 GHz (0.8 and 1.8 dB) but never state a measured 3 dB bandwidth; the 20 mm trace in Fig. 3(b) touches the -3 dB line near 46 GHz (noise). So no (c)-style bound; `bw_measured_to_ghz` 50 is the Fig. 3(b) axis end (extracted_from_figure, instrument limit not stated). The 180 GHz figure is the authors' prediction (notes and sim target, basis predicted).
  - 1.35 V is called DC in the text but labelled 1 MHz in Fig. 3(a); entered in `vpi_dc_v` with that note. 1.3 V (1 GHz, Bessel method) in `vpi_rf_v`; 1.6 V at 50 GHz is author-stated and kept in notes (single-point column).
  - RF-line numbers assigned to one device only: loss 2 dB/cm at 50 GHz on the 10 mm row (Fig. 2 caption names a 10-mm electrode). The measured RF phase index 2.23 is not tied to a device in the text, so it is on no row (evidence `context_values`, sim target with inline source).
  - IL (<1 dB on-chip, 13 dB total) and ER: ER is explicitly for the 20 mm device; the IL statements name no device and are entered on the 20 mm headline row with a note. On-chip IL basis author_estimate (estimated from a coupler pair with and without the modulator), ER `approx`.
  - Electrode type `segmented` (authors' term); the structure is T-rail-like (fingers tied by stems), and Liu cites it as a CL-TWE. Integrator may switch to `cl_twe`. drive push_pull and vpi_convention mzm_push_pull are author-stated (p.1-2), not inferred. integration monolithic (commercial wafer), as in the chen2022 pilot.
  - slab 250 nm is derived (600 - 350) in `derived`.
- CSV hints wrong or changed: license hint "Optica-OA-License-v2" disagrees with Crossref (v1); access hint `arxiv` changed to `open_access` (Crossref license class VOR-OA); published_on kept at 2020-11-26 (arXiv).
- Sim config notes: ridge width is NOT disclosed and the Fig. 1(e) sketch is not to scale (gold drawn as tall as the ridge), so a 1.0 um rectangular placeholder with no source is used and flagged in `provenance`, `missing`, `limitations`; Vpi*L from the optical stage is not comparable until resolved. Signal width 100 um is the Table I simulation value, ground width assumed equal. Cell has three sections (finger-plus-slot 39 um, stem 6 um, gap 5 um); stem and gap lumped into the unloaded cut. A rough SEM scale reading suggests a slot height near 10 um against h = 6 um in the caption (low confidence, caption used). Segment symbols read from the Fig. 1(a)/(e) arrows: g = gap between facing finger edges, h = slot, s = finger width, t = stem width, r = finger length, c = finger gap.
- Could not read: journal version; Table I is read from the page render (verified), all plots read from page renders.

## liu2021 - distilled
- Source: arXiv preprint 2103.03684 (template placeholders for received/accepted dates), main text pp.1-4 and supplement pp.5-9 read; no Crossref record. License not shown in the PDF: `license` empty, redistribution `restricted_local_only` (prefetch says arXiv-nonexclusive, not verifiable offline).
- Rows: 1. liu2021-a (5 mm, 3 um T-rail gap: Vpi 3.4 V at 100 kHz, Vpi*L 1.7 V*cm, 1.3 dB roll-off at 67 GHz, ER >17 dB, fiber-to-fiber 17 dB, n_rf about 2.3, ng about 2.25). repro_grade B.
- Sim config: `sims/liu2021/config.yaml` (T-rail geometry with the symbol meanings of Table S1 read from Fig. S3(a) and the SEM inset; rail width 3 um and gap 3 um checked against the SEM scale bar).
- Not reported: on-chip IL, propagation loss, Z0 and RF loss in numbers (S21/S11 plots only), optical power, system demonstration, energy, fabricator, fabricated ridge width and sidewall angle, T-rail thickness, ground width.
- Judgment calls:
  - Vpi convention and drive are not stated by the authors; entered as push_pull / mzm_push_pull with basis derived (evidence notes say so). Support: one arm in each gap of the G-S-G line (Fig. 3(a)), x-cut, and Fig. 2(b) per-arm index change 2.4e-5 per V gives about 1.6 V*cm push-pull against 1.7 reported.
  - Vpi*L 1.7 V*cm entered with basis derived (authors' product of 3.4 V and 5 mm).
  - Convention (c) applied: no 3 dB crossing, "1.3 dB roll off ... limited by the bandwidth of the test system" (p.4), so bw3db 67 with `gt`, bw_measured_to 67, roll-off 1.3 dB at 67 GHz (reference frequency not stated, bw3db_reference unspecified). The authors' prediction "3 dB bandwidth over 110 GHz" (transmission-line model from S-parameters, Fig. S7) and the electrical 6 dB bandwidth over 110 GHz are in notes only (bw6db is not used for an electrical measurement).
  - ER: main text "beyond 17 dB" (qualifier gt); supplement says 17 dB from a DC scan; Fig. 4(c) normalised minimum reads about 0.03 (about 15 dB); noted.
  - n_rf 2.3 read from the Fig. 4(e) plateau (extracted_from_figure, approx; text only says "close to ng about 2.25"); ng 2.25 basis author_estimate (origin of the value not stated; red line "optical mode").
  - Ridge width 1 um is entered with basis design_target because the text states it for the Fig. 1(c) simulation, not the fabricated device.
  - A 4 um T-rail-gap device appears in Fig. 4(b) (loss only, about 18 dB read from the plot): no row.
- CSV hints wrong: published_on 2021-02-04 is not in the source and conflicts with the arXiv identifier 2103 (March 2021): left empty. venue arXiv, source_type arxiv_preprint, discovered_via local_corpus kept as in the batch CSV (its source_state tag `local_alias` is not a discovered_via token).
- Sim config notes: rail thickness is not stated (lift-off rails; main electrodes electroplated to 1.4 um); 1.4 um assumed as an upper bound and listed in `missing`. Ground width assumed equal to the 50 um signal. Silica buffer modelled as a 100 nm slab layer plus an outward-offset cap over the ridge, electrodes on the buffer. Optical window |x| < 1.4 um (rails at 1.5 um; scalar solver cannot model metal-induced loss, the paper's central point). Sidewall angle 68 deg is a low-confidence reading of the Fig. 1 sketch.

## pan2021, arabjuneghani2022, mao2022 - needs_download
- No `references/<id>/text.md` (only `crossref.json`); prefetch status `no_open_source` / `failed`. Recorded in `needs_download.md` with the Crossref abstract metrics. No rows written for them, no workaround attempted.
- Note for the coordinator: Crossref lists a CC-BY-4.0 version of record for arabjuneghani2022 and mao2022 (and Optica OA License v1 for pan2021), so a plain open-access fetch by the download owner may work.

## Schema/skill gaps hit
- SKILL/convention (c) vs absent bandwidth statements: kharel2021 gives roll-off at the axis end but no 3 dB statement, and the trace touches -3 dB once; no rule for "roll-off below 3 dB at the last measured point without an authors' bound". I entered no bound there and applied (c) for liu2021 where the authors say the roll-off is limited by the test system. Integrator may prefer a uniform rule.
- `vpi_dc_v` vs low-frequency labels: 1 MHz (kharel2021-b) and 100 kHz (liu2021) triangular/oscilloscope sweeps are entered as DC; no frequency column for the DC measurement.
- Electrode_type for segmented/finger-and-stem structures: `segmented` vs `cl_twe` has no tie-break rule (authors' term used).
- Single-point RF-line values that the paper does not attribute to a device (kharel2021 index 2.23): no way to attach a paper-level value; kept in `context_values` and the sim target.
- Author products (Vpi*L) are entered as basis derived per convention (h); entry-level basis wins over the row-level vpi_basis (measured, headline Vpi).
- Sim contract gaps: see `SPEC_PROPOSALS.md` (three-section cell, neutral loading type name, unverified standard-value class, bound/prediction targets, material loss tangents, device attribution, optical window versus nearby metal).
- Cleanup: the scratch builders and cropped figure crops are in the session scratchpad only; nothing temporary was left in the repo.
