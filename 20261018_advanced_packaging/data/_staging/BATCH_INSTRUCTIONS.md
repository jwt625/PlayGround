# Batch distillation instructions (shared contract, 2026-10-18)

You distill advanced-packaging papers into table rows. Work from the project root (the directory containing `data/`,
`scripts/`, `references/`). No emoji. Never invent: an empty cell means not reported. Do not use git. No network.

## Read first
1. `.claude/skills/packaging-distill/SKILL.md` (binding)
2. `data/schema/packaging.schema.yaml` (columns, enums, rules a-l)
3. Worked examples once they exist: `data/_staging/pilot_*/` (and their `REVIEW.md` if present)

## Your task
Your batch file `data/_staging/batches/<batch>.csv` lists paper_ids (with title, bucket, priority as hints only).
For each paper, one at a time, completely:
1. If `references/<paper_id>/text.md` is missing: status needs_download; move on.
2. Read the whole text.md; open the page renders for every figure/table you take a number from.
3. Write rows into `data/_staging/<batch>/{papers,test_vehicles,components,stack_layers,interconnects,process_steps,electrical_interfaces,optical_interfaces,metrics}.csv`
   (copy headers from `data/_staging/TEMPLATE/`) and `data/_staging/<batch>/evidence/<paper_id>.yaml`.
4. The papers row: source_status full_text, pages_read, verified_on (today), primary_bucket from the batch file unless
   the full text clearly shows otherwise (say so in the report).

## Source quality
If `references/<paper_id>/TEXT_LAYER_WARNING.md` exists, the PDF text layer was garbled and `text.md` holds tesseract
OCR of `figures/page_NN_220dpi.png`: verify every number against those images and note "OCR text" in papers.notes.
Temporary files (crops, re-renders, helper scripts) go only in your own subfolder `<session scratchpad>/<batch>/`
(the scratchpad is shared by parallel agents).

## Validation and boundaries
Run `uv run python scripts/validate_db.py --staging data/_staging/<batch>` until it reports 0 errors (warnings are
listed in your report). Write only inside `data/_staging/<batch>/`. Never edit `data/*.csv`, `data/schema/`,
scripts, other batches, or `references/`.

## BATCH_REPORT.md (in your staging dir)
Per paper: status, rows per table, figures opened (file names), judgment calls, fields not reported, anything wrong
with the source (truncated, wrong paper, scanned). Final message: short per-paper status, validator output, blockers,
schema gaps you hit (bullets).
