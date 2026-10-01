---
name: eo-modulator-distill
description: Distill an electro-optic modulator reference (paper, preprint, datasheet) into verified database rows (papers.csv, devices.csv, evidence yaml) and, when geometry allows, a browser-runnable simulation config. Use when adding or auditing an EO modulator paper in this repo.
---

# EO modulator distillation

Input: one reference (DOI, arXiv id, URL, or local PDF). Output: rows in `data/papers.csv` and `data/devices.csv`, `data/evidence/<paper_id>.yaml`, optionally `sims/<paper_id>/config.yaml`, plus an entry in the manual-download list when the full text is not accessible. Read `data/schema/devices.schema.yaml` (columns, enums, units) and `sims/SPEC.md` (sim config contract) before first use.

## Hard rules

1. Empty cell = not reported. Never estimate, interpolate from memory, or take a number from a tweet, blog, news item, review, or another paper's table of this paper. A number comes from the primary text or a figure/table of it (page + figure/table locator) or it stays empty.
2. Basis is explicit per headline metric: `measured`, `simulated`, `predicted`, `extracted_from_figure`, `derived`. A simulated or projected value must never be entered as measured.
3. Convention fields are mandatory when Vpi or Vpi*L is filled: `vpi_convention` (per-arm phase shifter, MZM push-pull, MZM differential drive, single-arm MZM, unspecified) and `drive`. State what the 3 dB bandwidth is referenced to (`bw3db_reference`: DC, 1 GHz, 10 GHz, other) and the measurement limit when no 3 dB crossing was observed (`bw_measured_to_ghz`). Insertion loss: fill `il_onchip_db` or `il_fiber_to_fiber_db` exactly as defined in the paper and write what is excluded in `il_onchip_excludes`. Do not convert between them.
4. Vpi*L: enter the reported value in `vpil_*`; if the paper only gives Vpi and length, leave `vpil_*` empty (the build step derives it and flags it `derived`). Never mix single-arm and push-pull definitions silently.
5. One row per device variant or operating point the paper characterizes (different length, wavelength, bias/frequency point, platform variant). A paper with a headline device plus a system demonstration uses one row per distinct device, with system-level results (baud, format, rate, driver) on the device they were measured with.
6. Organizations: names are full, unabbreviated, and come from the paper's affiliation list (universities, companies, institutes, foundries/fabs named as fabricating the device). Add every new org to `data/organizations.csv` (type, ISO alpha-2 country, region) taking country from the affiliation address; if a fab or foundry is not named, leave `foundry_or_fab` empty (do not guess from the company).
7. Primary-source URLs only (publisher or arXiv). Tweets and blog posts are never in `url`; record how a paper was found in `discovered_via`.
8. Rights: raw PDFs, extracted text and figure images stay in the git-ignored `references/<paper_id>/` cache. Only record `redistribution: open_license_ok` when the license (CC-BY family, etc.) is verified on the source page and written in `license`; otherwise `restricted_local_only`. Keep evidence notes to short factual remarks (<= 25 words), no long quotations.
9. Scraping etiquette: use `scripts/fetch_source.py` (>= 5 s between requests, stops on 403/429 or HTML-instead-of-PDF). Never fetch paywalled sources through workarounds; put them on the manual download list instead.
10. No emoji. ISO dates. N/A means empty cell.
11. Conventions block in `data/schema/devices.schema.yaml` (a-e) is binding: headline Vpi convention goes in `vpi_dc_v` / `vpi_rf_v` with `vpi_convention`, a separately reported push-pull MZM value goes in `vpi_mzm_pushpull_dc_v`; bounds ("over 67 GHz", "below 2 dB") and approximate values are entered as numbers plus the `qualifiers` column (`field:lt|gt|approx`); a claimed 3 dB crossing at the instrument limit fills `bw3db_ghz` and `bw_measured_to_ghz`; several baud/format points stay in `modulation_format` text with the extreme in `max_*`; layer stacks and orientations with no dedicated column go in `epitaxy_or_stack` / `waveguide_orientation`. Evidence entries are allowed for any column (use them to give convention choices a locator) but are required only for evidence-required columns. In the evidence file, `entries` hold values the paper states (basis may be `author_estimate` or `derived` when the paper itself derives it); the `derived` list holds your own arithmetic with its formula. If the row's basis column and a field's evidence basis disagree, the evidence basis wins. Org names follow convention (e) in the schema.

## Workflow

1. Identify: resolve DOI/arXiv via Crossref (`https://api.crossref.org/works/<doi>`) or the arXiv API; confirm title/authors/year/venue; pick `paper_id` = first-author surname (ascii, lowercase) + year (suffix a/b on collision). Check `data/papers.csv` for an existing row first (update instead of duplicating).
2. Acquire: check local corpora listed in the task prompt, then open-access sources (arXiv, publisher open access). For arXiv use the versioned PDF (`https://arxiv.org/pdf/<id>v<n>`). If paywalled and not local, set `cache_status: needs_download`, append to the manual download list (see Output formats) and stop; the row stays metadata-only with no devices.
3. Extract: `uv run python scripts/extract_source.py --paper-id <id> --pdf <file> --url <url> --doi <doi> --license <lic>`; creates `references/<paper_id>/{source.pdf,source.json,text.md,figures/}`. Page-delimited text may garble tables; open the page-render PNGs (`figures/page_NN.png`) and read the figure/table itself when a number matters. Look at images, do not guess.
4. Read the whole paper (abstract, device section, methods, results, supplementary pointers, figure captions). Build a per-device ledger: every metric with value, unit, basis, page/figure/table locator.
5. Fill `data/papers.csv` (all required columns; `verified_on` = today ISO date), `data/devices.csv` rows, and `data/evidence/<paper_id>.yaml` (every non-empty evidence-required cell has an entry whose value equals the CSV value). Parallel work: when the task names a staging directory `data/_staging/<batch>/`, write `papers.csv`, `devices.csv`, `organizations.csv` (every org your rows reference that is not already in `data/organizations.csv`; copy the header from `data/`) and `evidence/*.yaml` there instead of in `data/`, never editing the shared tables directly. Then run `uv run python scripts/merge_staging.py data/_staging/<batch>` (dry run) until it reports zero conflicts and zero validation errors; the coordinator applies it. Outside staged work run `uv run python scripts/validate_db.py` until zero errors.
6. Reproducibility grade (`repro_grade`): A = geometry and materials fully disclosed for a TWE-type device (film thickness, etch/rib, electrode gap/width/thickness, buffer, substrate, metal, cut, length, wavelength); B = most disclosed but some dimensions need figure digitization (SEM/cross-section image); C = metrics only. Only A/B get a sim config, and only for dielectric traveling-wave electrode devices (TFLN, TFLT, BTO, hybrid Si/SiN-LN/LT). InP, silicon, SOH and plasmonic papers get database rows and `repro_grade` but no sim unless the task says otherwise.
7. Sim config (grade A/B, dielectric TWE): write `sims/<paper_id>/config.yaml` per `sims/SPEC.md`. For each parameter record provenance class: `paper_exact`, `figure_digitized` (state the figure and how it was measured), `project_inference` (state the assumption), `standard_reference` (material constants: cite the standard source), `unknown` (leave out; list in `missing`). Put the paper-reported metrics in `targets` with tolerance and a `source` pointing at the evidence entry. List modeling limitations honestly (sidewall angle, scalar optics, quasi-static 2D, no pad/launch effects). If the engine CLI exists, run `node engine/cli.mjs sims/<paper_id>/config.yaml` and record pass/fail in the DevLog; do not tune parameters to hit targets, and do not store solver outputs.
8. Audit-ready summary: list what was read (pages), what could not be found (explicit `not reported` list), and every judgment call.

## Output formats

Manual download list: append one block per paywalled paper to `data/manual_downloads.md`:

```
- paper_id: <id>
  title: <title>
  doi: <doi>
  publisher_url: <url>
  save_as: <paper_id>.pdf        # exact filename to use
  drop_folder: references/_inbox/
  why_needed: <metrics expected>
```

Evidence yaml (one file per paper):

```yaml
paper_id: kohli2025
verified_on: 2026-10-01
source_files: [references/kohli2025/text.md]
entries:
  - {device_id: kohli2025-a, field: bw3db_ghz, value: 110, unit: GHz, basis: measured, locator: "p.4, Fig. 3b", note: "referenced to 10 GHz"}
derived: []
```

Common pitfalls (check before finishing): fiber-to-fiber vs on-chip loss; differential Vpi vs per-arm; EO bandwidth referenced to 1 or 10 GHz vs DC; "100 GHz" meaning measurement limit not crossing; simulated bandwidth quoted in the abstract; Vpi at DC vs at RF frequency; overlapping arXiv and journal versions with different numbers (record which version and its date).
