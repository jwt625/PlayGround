# Batch distillation instructions (shared contract, revised 2026-10-01 by D0)

You distill EO-modulator papers into database rows. Work from the project root (the directory that contains `data/`, `scripts/` and `references/`). No emoji anywhere. Never invent: an empty cell means not reported. Do not use git.

## What is already done for you (prefetch)

The coordinator ran `scripts/prefetch_batch.py` for your batch. For each paper `references/<paper_id>/` already holds, when obtainable:
- `crossref.json`: the Crossref work record (identity, authors, venue, license, links). Use it for identity; you do not call Crossref.
- `source.pdf`, `source.json`, `text.md` (page-delimited text, metadata header) and `figures/` (`page_NN.png` renders of pages with figure captions, `img_pNN_k.png` embedded images, `figures.json`).
- Text-only sources (header says "pre-extracted text copied from a local corpus"): no PDF and no figures; numbers that only appear in figures/tables you cannot see must stay empty, and you say so.
- `data/_staging/<batch>/prefetch_status.jsonl` and `needs_download.md` list what could not be obtained.

**You make no network requests.** No downloads, no Crossref, no arXiv, no publisher pages. If a paper has no `references/<paper_id>/text.md`, it is `needs_download` (add it to your `needs_download.md` in the format below) and you move on. If you believe a source is wrong (wrong paper, scanned, truncated), say so in your report instead of working around it.

## Read first (in order)
1. `.claude/skills/eo-modulator-distill/SKILL.md` (binding; includes rule 11 conventions)
2. `data/schema/devices.schema.yaml` (columns, enums, conventions a-e), `sims/SPEC.md` (sim config contract; read its Changelog and any later "Cross-section contract" addendum)
3. Worked examples that passed validation: `data/papers.csv`, `data/devices.csv`, `data/evidence/chen2022.yaml`, `data/evidence/kohli2025.yaml`, `data/evidence/ogiso2016.yaml`, `sims/chen2022/config.yaml`, and the pilot feedback `data/_staging/pilot_*/SKILL_FEEDBACK.md`.

## Your task
Your batch CSV (`data/_staging/batches/<batch>.csv`) lists the papers. Treat its priority/platform/sim_candidate columns as hints from titles/abstracts only; verify from the full text. Process the papers one at a time, completely, before starting the next. For each paper:
1. Identity from `crossref.json` (title, authors as listed, year, venue, license). If the paper text shows a different version than Crossref (preprint vs journal), say which one the numbers come from.
2. Read the whole `text.md`; open the page-render PNGs for every figure/table that supplies a number you use (read the numbers from the figure itself, not from memory). Record the page/figure/table locator for every value.
3. Fill papers/devices/organizations/evidence per the skill, schema v2 and the pilot examples. One row per distinct device/operating point; headline Vpi convention in `vpi_dc_v` + `vpi_convention`; bounds with qualifiers; fields the paper does not report stay empty; simulated/predicted values get basis `simulated`/`predicted`; sim-only papers are still rows if they report metrics. `discovered_via` comes from the batch CSV (private-corpus tags become `local_corpus`). `verified_on` = 2026-10-01. If the paper is not an EO modulator device paper with quantitative metrics (passive only, system only, material only), still write the papers.csv row (`cache_status: full_extract`, notes explaining) and no device rows, and say so.
4. `repro_grade` A/B/C as defined in the skill. For dielectric traveling-wave-electrode devices with grade A or B (TFLN, TFLT, BTO, hybrid Si/SiN-LN/LT) write `sims/<paper_id>/config.yaml` per `sims/SPEC.md` (provenance classes, targets with `source`, `missing`, `limitations`) following the conventions `sims/chen2022/config.yaml` established. If SPEC.md lacks something you need, use the closest valid form and add a bullet under a heading for your paper in your staging `SPEC_PROPOSALS.md`. Do not run the engine, do not tune parameters toward targets, do not store solver outputs. No sim config for InP, silicon, SOH, plasmonic, EAM or resonator papers.
5. Organizations: add every org referenced that is not yet in `data/organizations.csv` to your staging `organizations.csv` (header copied from `data/`; org_type in university|company|national_lab|research_institute|foundry|facility|consortium|other; ISO alpha-2 country; region in north_america|europe|east_asia|south_asia|southeast_asia|oceania|middle_east|other; names per schema convention e). Reuse existing org names exactly as written in `data/organizations.csv`.

## Output (staging dir `data/_staging/<batch>/`, e.g. `data/_staging/p1_01/`)
`papers.csv`, `devices.csv`, `organizations.csv` (new orgs only), `evidence/<paper_id>.yaml`, `SPEC_PROPOSALS.md` (only if needed), `needs_download.md` (only if needed), `BATCH_REPORT.md`.

`needs_download.md` format per paper:
```
- paper_id: <id>
  title: <title>
  doi: <doi>
  publisher_url: <url>
  save_as: <paper_id>.pdf
  drop_folder: references/_inbox/
  why_needed: <metrics the abstract reports>
```
`BATCH_REPORT.md` per paper: status (`distilled` | `no_device_rows` | `needs_download` | `failed`), rows written, repro_grade, sim config path or none, fields the paper does not report (short), judgment calls, any CSV hint that was wrong (identity, platform, priority), and what you could not read (for example figures in text-only sources). Partial access is not completion; state the concrete blocker.

## Rights and privacy
Source files, extracted text and figures are tracked in `references/<paper_id>/` under the project's reference-cache policy; record the license you can verify from `crossref.json` or the paper's own notice in `papers.csv` (`license`, `redistribution`). Never write absolute local paths, home-relative paths, or the names of private repositories or projects into any file you create (`--local-origin` values and notes included). Public-facing records never contain private paths.

## Validation and boundaries
Run `uv run python scripts/merge_staging.py data/_staging/<batch>` (dry run) until it reports 0 conflicts and 0 validation errors. Write only in `data/_staging/<batch>/`, `references/<paper_id>/` for your assigned papers (cache repairs only, no re-fetching), and `sims/<paper_id>/`. Never edit `data/*.csv`, `data/schema/`, `engine/`, `app/`, `sims/SPEC.md`, `sims/chen2022/`, or other batches' directories. Schema or contract changes are proposed in `SPEC_PROPOSALS.md` / the report, never applied.

## Final message
Short: per-paper status, validation output, number of rows, blockers, schema/skill gaps you hit (bullets).
