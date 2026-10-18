---
name: packaging-distill
description: Distill an advanced-packaging paper (cached full text) into verified rows of the packaging tables (papers, test_vehicles, components, stack_layers, interconnects, process_steps, electrical_interfaces, optical_interfaces, metrics) with page/figure locators. Use when extracting or auditing a paper in this project.
---

# Advanced-packaging distillation

Input: one paper_id whose full text is cached in `references/<paper_id>/` (`text.md` page-delimited, `figures/page_NN.png`
page renders, `figures/img_pNN_k.png` embedded images, `source.json`). Output: rows for that paper in the staging
directory named by the task (`data/_staging/<batch>/<table>.csv`) plus `evidence/<paper_id>.yaml`.
Read `data/schema/packaging.schema.yaml` first (v2): its rules (a)-(m) and enums are binding and override this file if they disagree.

## Hard rules
1. Empty cell = not reported. Never estimate, never use memory, press, other papers, reviews or the triage abstract.
   Every number comes from this paper's text, a table, or a figure you actually opened.
2. Every row: `basis` (measured | design_target | simulated | derived | author_estimate | stated | unspecified),
   `source` (text | table | figure_image | schematic | caption | mixed) and `locator` (`p.N` = text.md page marker of
   the page showing the figure/text, plus `Fig. X`/`Table Y`/`Sec. Z`). Cells that differ from the row go in
   `overrides` ("pitch_um:schematic;yield_pct:measured"). Dimensions in the text are design_target unless the paper
   says measured; simulated values are simulated. Never present a design or simulated number as measured.
3. Bounds/approximations go in `qualifiers` (`pitch_um:lt`, `thickness_um:approx`). Ranges use *_min/*_max columns.
4. One test_vehicle per distinct physical build (different stack, die set, carrier, structure); split variables
   that leave the stack unchanged (dose, time, offset, side, sample split) go in metrics.variant, not new TVs.
   Coupons/sub-assemblies are TVs with tv_kind test_coupon/sub_assembly. Hang every row off a tv_id.
5. Stack layers: seq 1 = top (die side), increasing downward, per `structure`. Record material exactly as written
   (trade names included) plus material_class. Thickness, line/space, via diameter/pitch per layer when given.
6. Interconnects: every level the paper discloses (hybrid bond pads, microbumps, Cu pillars, C4, BGA, TSV/TGV/TDV,
   RDL vias): pitch, diameter, height/depth, count, metallurgy, bonding method and conditions (temp, pressure/force,
   time, anneal), alignment, resistance, yield.
7. Process steps: the ordered flow as the paper describes it (seq), with category, material, equipment (only if named)
   and conditions. Do not fill in "standard" steps the paper does not state.
8. Electrical interfaces (D2D/HBM PHY) only when the paper reports an electrical/protocol property: protocol as
   written (UCIe-S/A, BoW, AIB, HBM3/3E/4, custom), lane rate, lanes, aggregate BW, BW density (Tbps/mm shoreline,
   Tbps/mm2 areal), energy pJ/bit with `scope`, reach, bump pitch, channel line/space, insertion loss at frequency,
   BER, latency. A bare pad array is an interconnect only. Optical I/O (fiber attach, couplers, MFD, coupling loss,
   alignment) goes in optical_interfaces.
9. Metrics: warpage (with temperature and sign convention in scope), reliability (test_standard, condition incl.
   preconditioning, cycles/hours, n_samples, pass_criterion, result), thermal, yield, PDN, adhesion, geometry (gaps,
   composite lengths) as parameter/value/unit with variant and statistic.
10. technology_name only if the paper uses the name; mappings you infer go in notes ("inferred: ...").
11. No network access. No emoji. ISO dates. No absolute local paths or private names. Notes <= 30 words, no long quotes.
12. Unit conversions keep the original basis and are logged in the evidence `derived` list with the formula. Your
    own arithmetic beyond that (area = x*y, ratios) -> override "field:derived" plus a `derived` entry.
13. Conflicts inside the paper: measured cross-section/photo > result table > text > schematic; enter the winner and
    list all alternatives in the evidence `conflicts` list. Numbers attributed to cited prior work: not entered
    (list them in `excluded`). Suppliers named only in acknowledgments: allowed, note "ack".

## Workflow per paper
1. Read `source.json` and the whole `text.md`. Note the paper's TVs/builds.
2. Open the page renders (`figures/page_NN.png`, every page is rendered at 150 dpi; crop/upscale into your scratchpad if labels are small) for every figure/table that carries a dimension, stack drawing,
   process-flow schematic, cross-section or table you use. Read numbers from the image; do not guess.
3. Build a ledger per TV: components, layers (top to bottom), interconnects, process flow, interfaces, metrics.
4. Write rows to the staging CSVs (headers exactly as in `data/schema/packaging.schema.yaml`; copy the header lines
   from `data/_staging/TEMPLATE/`), and `evidence/<paper_id>.yaml`:
   ```yaml
   paper_id: ectc51909.2023.00174
   verified_on: 2026-10-18
   source_files: [references/ectc51909.2023.00174/text.md, references/ectc51909.2023.00174/figures/page_03.png]
   entries:            # optional value-level detail when a row locator is not enough
     - {table: interconnects, row_id: ectc51909.2023.00174-a-i1, field: pitch_um, value: 25, unit: um, basis: design_target, locator: "p.2 Fig. 1", note: "C4 pitch on LSI side"}
   derived:            # every derived cell
     - {table: test_vehicles, row_id: ectc51909.2023.00174-a, field: interposer_area_mm2, value: 3300, formula: "x*y = 55*60", inputs: "p.3 Table I"}
   conflicts:          # alternatives inside the paper (rule 13)
     - {table: components, row_id: ectc51909.2023.00174-a-c1, field: thickness_um, values: [750, 775], locators: ["p.3 Fig. 3(a)", "p.5 Fig. 7"], chosen: 775, why: "measured cross-section"}
   excluded: ["desmear roughness 300-500 nm attributed to ref [10]"]
   not_reported: [electrical_interfaces, "bonding temperature"]   # what the paper does not disclose (short)
   ```
5. Run `uv run python scripts/validate_db.py --staging data/_staging/<batch>` until 0 errors.
6. Report per paper: status (distilled | no_tv_rows | failed), row counts per table, figures opened, judgment calls,
   not-reported list.

## Pitfalls
Pitch vs line/space vs diameter confusion; "2/2 um" = line/space; "x um pitch" vs "x um bump"; um vs mm; panel vs unit
size; design vs measured; die thickness vs stack height; warpage sign convention (note it); TSV diameter vs depth;
"up to"/"sub-" bounds; numbers from a cited prior work (do not attribute to this TV); simulated values in abstracts.
