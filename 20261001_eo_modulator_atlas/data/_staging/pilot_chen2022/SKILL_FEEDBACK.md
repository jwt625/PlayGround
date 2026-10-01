# SKILL_FEEDBACK (chen2022 pilot, 2026-10-01)

Time (wall clock, approximate; the tool clock showed ~8 min at the SPEC_PROPOSALS step, total about 10 min of tool time plus model reasoning):
- Read SKILL/PLAN/schema/SPEC: ~1 min.
- Crossref + extract_source: ~1 min (script worked first time).
- Reading text + figure inspection: the longest step, ~45% of effort. Geometry of the T-rail device needed 4 crops (Fig. 1(d) inset at 16x, Fig. 2(a), 2(b), 5(d-f)).
- Ledger + CSV/evidence writing: ~20%.
- Sim config (geometry coordinates, half-disc polygons, provenance): ~30%; the polygon/coordinate bookkeeping was done by a throwaway script in the scratchpad (not saved in the repo).
- Validation: 1 run, passed after the sim config existed.

## Unclear in SKILL.md
- Step 5 vs 7 order: papers.csv `sim_config` must point to an existing file or validation fails, so the config has to exist before the first clean dry run. State this order.
- "Verify the license on the publisher/Crossref record": the skill does not say which counts. I used the Crossref VOR license URL plus the article's own CC BY notice (publisher site not fetched, anti-bot risk). Say whether the page itself must be fetched.
- `discovered_via` allowed values list in the schema (drive_doc | tmp_eo_md | ...) has no value for "task assignment / local corpus"; I used `local_corpus`. Add it or a free-text escape.
- `repro_grade` A vs B: the T-rail device has all headline dims but position/margin/period details only in schematics. Added "A requires every dimension needed to build the 2D cut without digitizing" to my own reading; please make that explicit. Chose B.
- For a derived value (alpha0*sqrt(f)), the skill says Vpi*L may be derived by the build step but is silent on other derivations; I used the `derived:` evidence block for rf_loss and max_baud. Confirm that is intended.
- Evidence `basis` for non-headline fields (z0, n_rf, ng, geometry) is not covered by Rule 2's list per metric; I used per-entry basis (`simulated` for ng 2.2, `extracted_from_figure` for Z0, n_rf).

## Missing columns / enums I needed
- `eo_rolloff_db`, `eo_rolloff_at_ghz` (headline claim: 1.4 dB at 67 GHz; also 0.76 dB at 50 GHz, which is two points). Currently in notes only.
- `rf_loss_alpha0_db_per_cm_per_sqrtghz` (0.36 measured, 0.19 simulated). I stored the derived value at 67 GHz instead.
- Per-field basis for RF/geometry fields (only vpi/bw/il have *_basis columns). Evidence carries it, the CSV does not.
- Simulated companion values (VpiL 2.10, alpha0 0.19, Zc, n_m, ng from Fig. 2): no home in devices.csv. Options: a `sim_*` column set, or a second device row with basis simulated. I did neither; they live in sim targets only.
- `box_thickness_um` and `cladding_thickness_um`: `buffer_oxide_um` is ambiguous for LNOI (BOX 3 um under the film vs 0.9 um top cladding). I put the BOX in `buffer_oxide_um` and the cladding thickness inside the `cladding` text.
- `electrode_thickness_um` has two values here (1.1 um main, 0.2 um T-segments); also `ground_width_um`, `loading_period_um`, `undercut_um` for substrate-removed devices.
- `electrode_type` has `cl_twe`, good; no `drive` value for "single-ended G-S-G on a push-pull MZM". `vpi_convention: mzm_push_pull` + `drive: single_ended` was my combination; the paper never says "push-pull", it is inferred from the arm placement (one waveguide per gap, x-cut). Please define the pairing in the schema notes.
- `integration`: monolithic chosen for a commercial LNOI wafer; add a note.
- `band`/`wavelength_nm`: paper gives neither; both left empty.
- `bw3db_reference` has `unspecified`; used because the paper does not say what the curve is normalized to (curves start at ~0 dB near 0 GHz).
- `organizations`: Hong Kong has no row convention; `country` CN taken from "Hong Kong, China" in the affiliation, though ISO has HK. Decide.
- `research_groups`: paper names no PI labs, only key laboratories in affiliations. I listed those named units; define whether to leave empty.

## Pitfalls hit
- Ambiguity of which device Fig. 5(d)-(f) (Zc, alpha_m, n_m) belongs to: caption does not say; text says the calculated S21 for all three lengths use "the fitted data in Figs. 5(d)-5(f)", so one set was entered on all three rows with a note. Check fix for other papers.
- Insertion loss: 8.2 dB total is stated in a sentence that mixes it with "for the longest device"; entered fiber-to-fiber only on the 10 mm row. Fig. 4 insets label -0.2 dB for all three devices, entered 0.2 on each.
- ER given as ">30, >25, >20 dB"; numeric columns cannot carry the inequality; value stored with a note (bound). Same for bandwidth (>67 GHz): `bw3db_ghz` left empty, `bw_measured_to_ghz` = 67.
- Rule 3 plus the "100 Gb/s OOK" claim: baud = bit rate only for OOK; entered as derived.
- Text parameter list `(r, c, s, t, h, hw, hl)` has symbol meanings only in the Fig. 1(d) inset. `s` is the T-bar width (arrows bracket the bar), not a spacing; discovered only by 16x zoom. A trap: g = 1.8 um (text) vs s = 2 um (list) looks like a contradiction but is not.
- Figure geometry cross-checks that worked: Au 75 um span in Fig. 2(b) gave a 11.25 px/um scale; island 11.9 um, slot 9.07 um (= hw 9), pad gap 1.84 um (= g 1.8). Recommend always calibrating a digitization scale with two known dimensions.
- Sim config conflicts with the plan's `data/sims` assumption that the origin is on the optical axis: the mirror plane and the origin differ here (see SPEC_PROPOSALS).
- Two valid but different notions of "unloaded" (slot-free vs pad-free); the paper never names the cross-section for the unloaded RF cell. Documented, not resolved.
- Throwaway helper scripts (CSV/evidence builder, polygon generator) were written outside the repo to avoid typos in unit conversions; a generator for half-disc polygons might be worth adding to scripts/.

## Script notes
- extract_source.py: works. The page renders are 150 dpi PNGs; embedded figure PNGs are better for plots (full res). Warning about VIRTUAL_ENV from another repo is noise from the user environment, not the repo. figures.json lists captions truncated to the first line.
- merge_staging.py dry run: clear output; the sim_config path check fires before the file exists (see order note).
- Embedded PNGs opened through pymupdf have a 0.75 point scale (page rect 0.75 x pixels); crop coordinates needed that factor. Mention in the skill if figure cropping is expected.
