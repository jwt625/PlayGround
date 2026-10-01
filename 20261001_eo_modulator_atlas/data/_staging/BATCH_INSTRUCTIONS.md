# Batch distillation instructions (shared)

You distill EO-modulator papers into database rows. Project root: ~/Documents/GitHub/PlayGround/20261001_eo_modulator_atlas . No emoji anywhere. Never invent: an empty cell means not reported. Do not use git.

## Read first (in order)
1. .claude/skills/eo-modulator-distill/SKILL.md (binding; includes rule 11 conventions)
2. data/schema/devices.schema.yaml (columns, enums, conventions a-e), sims/SPEC.md (sim config contract; read the Changelog at the bottom for later changes)
3. Worked examples that already passed validation: data/papers.csv, data/devices.csv, data/evidence/chen2022.yaml, data/evidence/kohli2025.yaml, data/evidence/ogiso2016.yaml, sims/chen2022/config.yaml, and the pilot feedback files data/_staging/pilot_*/SKILL_FEEDBACK.md (pitfalls others already hit).

## Your task
Your batch CSV lists the papers (columns include paper_id, title, doi, arxiv_id, url, access_guess, local_source_abs, platform_guess, sim_candidate, notes). Treat the CSV's priority/platform/sim_candidate as hints from titles/abstracts only; verify from the full text. Process the papers one at a time, completely, before starting the next.

For each paper:
1. Identity: confirm via Crossref (https://api.crossref.org/works/<doi>; header User-Agent: eo-modulator-atlas (mailto:jwt625@gmail.com); >= 2 s between calls; one request at a time). Title, authors (as listed), year, venue, license. Do NOT call export.arxiv.org or the arXiv API. Do not scrape publisher landing pages.
2. Source: if local_source_abs is set, use that file (read-only input; copy via extract_source.py). Otherwise, if arxiv_id is set, `uv run python scripts/fetch_source.py --paper-id <id> --url https://arxiv.org/pdf/<arxiv_id>` (versioned id if known); otherwise for open-access publisher PDFs try the DOI's open-access PDF URL through fetch_source.py. fetch_source.py enforces >= 5 s between requests across all agents and stops on 403/429/HTML. If it fails, do NOT try workarounds: record the paper in your staging `needs_download.md` (format below) and move on. If you see HTTP 429 or any anti-bot page, finish the paper you are on if you have its text, then stop fetching entirely for the rest of the batch (still process papers whose files are local) and say so in your final message.
3. Extract with scripts/extract_source.py (`--local-origin local_corpus` for local files; never pass or write the private source paths anywhere; never mention the name of the repository a local file came from in any output file). Read the whole paper; look at page-render PNGs for every figure/table that supplies a number you use (read the actual numbers from the figure, not from memory).
4. Fill papers/devices/organizations/evidence per the skill, schema v2, and the pilot examples. Typical: one row per distinct device/operating point; headline Vpi convention in vpi_dc_v + vpi_convention; bounds with qualifiers; locator for every value; fields the paper does not report stay empty; simulated/predicted values get basis simulated/predicted; sim-only papers are still rows if they report metrics (basis simulated). discovered_via comes from the batch CSV (replace any private-corpus tag with local_corpus). verified_on = 2026-10-01. If after reading you find the paper is not an EO modulator device paper with quantitative metrics (e.g. passive only, system only), still write the papers.csv row (cache_status full_extract, notes explaining) and no device rows, and say so.
5. repro_grade A/B/C as defined. For dielectric traveling-wave-electrode devices with grade A or B (TFLN, TFLT, BTO, hybrid Si/SiN-LN/LT): write sims/<paper_id>/config.yaml per sims/SPEC.md exactly (provenance classes, targets with `source`, `missing`, `limitations`). Follow the conventions the chen2022 config established; if SPEC.md is missing something you need, use the closest valid form and write a bullet under a heading for your paper in your staging SPEC_PROPOSALS.md. Do not run the engine unless engine/cli.mjs exists and your task says so; do not tune parameters toward targets; do not store solver outputs. No sim config for InP, silicon, SOH, plasmonic, EAM or resonator papers.
6. Organizations: add every org referenced that is not yet in data/organizations.csv to your staging organizations.csv (header copied from data/; org_type in university|company|national_lab|research_institute|foundry|facility|consortium|other; ISO alpha-2 country; region in north_america|europe|east_asia|south_asia|southeast_asia|oceania|middle_east|other; full official names per schema convention e). Reuse existing org names exactly as written in data/organizations.csv (check before adding; match spelling) so merges do not conflict.

## Output (staging dir data/_staging/<batch>/ , e.g. data/_staging/p1_01/)
papers.csv, devices.csv, organizations.csv (new orgs only), evidence/<paper_id>.yaml, SPEC_PROPOSALS.md (only if needed), needs_download.md (only if needed), BATCH_REPORT.md.
needs_download.md format per paper:
```
- paper_id: <id>
  title: <title>
  doi: <doi>
  publisher_url: <url>
  save_as: <paper_id>.pdf
  drop_folder: references/_inbox/
  why_needed: <metrics the abstract reports>
```
BATCH_REPORT.md: per paper: status (distilled | no_device_rows | needs_download | failed), rows written, repro_grade, sim config path or none, fields the paper does not report (short), judgment calls, anything surprising or any CSV hint that was wrong (identity, platform, priority). Be honest about what you could not read.

## Validation
Run `uv run python scripts/merge_staging.py data/_staging/<batch>` (dry run) until it reports 0 conflicts and 0 validation errors. Never edit data/*.csv, data/schema, or other agents' staging dirs; write only in data/_staging/<batch>/, references/<paper_id>/, sims/<paper_id>/.

## Final message
Short: per-paper status, validation output, number of rows, blockers (rate limits, access failures), schema/skill gaps you hit (bullets).
