# Schema / skill feedback from pilot_1 (2026-10-18)

Paper: ectc51906.2022.00142 (UV-assisted SAP on prepreg; peel, FC-BGA mock-up warpage, comb HAST). A materials/process
paper with coupons, not a package build. Issues ordered by impact.

## 1. TV granularity vs split-lot variables (rule (f), skill rule 4) - unclear, high impact
- Rule says "different stack, pitch, size or process variant" = new TV. This paper has three split variables:
  foil-press condition (shiny VLP / matt VLP / matt ETS), UV dose (0, 315, 630, 945, 1260, 3145 mJ/cm2), Pd-removal time
  (0 / 60 / 120 s). Taken literally that is 12 peel TVs + up to 5 comb TVs.
- What I did: TV per sample type and dielectric (prepreg vs insulation film) = 5 TVs; split variables go into
  metrics.condition / parameter text.
- Proposal: add a `variant` (or `split`) column to metrics (and process_steps), or a `variants` child table
  (variant_id, tv_id, variable, value), and state in rule (f): "split-lot variables that do not change the layer stack
  are variants, not TVs".

## 2. Coupons and material-level samples have no fitting package_class
- Peel coupon (150x150 mm panel, no die) and comb HAST coupon are not packages. Used `organic_substrate`.
- Proposal: add `test_coupon` / `material_coupon` to package_class, or a `tv_kind` column (package | coupon | wafer |
  simulation).

## 3. Several features at one layer level
- The comb layer has two designs, L/S 10/10 and 8/8 um, on the same plating level. stack_layers holds one line/space
  per row; I used two rows with the same seq (layer_ids -L02 and -L02b). seq uniqueness is not validated.
- Proposal: allow `line_min_um`/`space_min_um`, or a `features` table (feature_id, layer_id, kind=comb|serpentine|
  daisy_chain|via_chain, line_um, space_um, count, length). Comb/serpentine test structures are common in substrate papers.

## 4. Process steps that the paper lists but skipped
- Fig. 4 shows half-etching, blackening and laser via + desmear as "(skipped)". No column says performed vs not.
  I put "skipped in this paper" in `result`.
- Proposal: add `performed: bool` (or `status: performed|skipped|proposed`).

## 5. step_category enum gaps
- No resist strip, no generic etch (wet/dry unstated), no "cover/encapsulate dielectric", no surface treatment.
  Used `other` for: blackening, resist removal, seed etching (method not stated), surface treatment, resin covering.
- Proposal: add `resist_strip`, `seed_etch` (or `etch` unspecified), `surface_treatment`, `preconditioning`
  (85/85 soak is currently `test`).

## 6. Basis rules conflict on unit conversions and mixed-basis rows
- Skill rule 12: "unit conversion ... the cell basis is derived"; schema rule (e): convert exact unit changes and record
  in evidence. Basis is per row, not per cell, so a 0.8 mm -> 800 um core row would become `derived` and lose
  design_target. I kept the source basis on the row and logged conversions in evidence `derived`.
- Rule (b) says "use the weakest" for mixed rows but gives no ordering. Example: component -c1 dims are design_target
  (text) but count=1 is read from the Fig. 3(b) photo. Need an explicit order, e.g.
  measured > design_target > stated > extracted_from_figure > simulated > author_estimate > derived (or per-field
  basis via evidence entries only; say so).

## 7. Basis for reported-but-unmeasured results
- "prepreg with surface roughness Ra of 0.2 um was obtained" (no method). Not a design target, not explicitly measured.
  Used `stated`. Rule (b) should say how to classify results whose measurement method is not given.

## 8. Missing columns for substrate/material papers
- stack_layers: no `roughness_um` (Ra/Rz is a key output for low-loss SAP); put in `properties` and a metric row.
- stack_layers: no `reinforcement` (glass cloth vs none) - the core variable of this paper (prepreg with glass cloth vs
  film without). Encoded in material text.
- metrics: no `value_min/value_max` or `uncertainty`/`n_samples`; warpage error bars, N=3 peel, N=6 HAST combs go in
  condition text.
- metrics: no `test_standard` column (JEDEC JESD22-A110 etc.); this paper gives none but most reliability papers do.
- process_steps: `time_min` forces 60 s -> 1 min and 168 h -> 10080 min conversions; add `time_s` or `time_h`.
- process_steps: no dose/energy column (UV 1260 mJ/cm2), no wavelength; put in `conditions`.

## 9. Warpage sign convention
- Fig. 6 plots RT negative and 260 C positive, viewed from BGA side, convention not defined. No column for it.
- Proposal: `sign_convention` text in metrics, or a rule that warpage metrics state view side + sign in condition.

## 10. Cross-TV comparison metrics
- "Warpage at 260 C of prepreg package was 45% of insulation film package" relates TV b to TV c. Put on TV b with
  "relative to TV c" in condition. Proposal: optional `ref_tv_id` column.

## 11. Pass/fail reliability rows
- HAST rows need both a number (200 h, or <20 h to short) and a verdict. Used value + value_text. Proposal: `pass_fail`
  enum (pass|fail|partial) and `criterion` (here: > 1.0 MOhm insulation resistance).

## 12. Minor
- interconnects: an undisclosed joint (Si chip on FC-BGA mock-up with underfill, no bump data) still needs ic_type;
  used `other`. Consider `unspecified` in interconnect_type and bonding_method (protocol already has it).
- structure enum has no underfill / die-attach level; used `bonding_interface` for underfill.
- component_type: plain Si mock-up chip mapped to `dummy_die` - fine, but say in the skill that mock-up chips are dummy_die.
- orgs: departments of one company (Packaging Solution Center, Laminate Materials R&D Dept) collapse to one entry;
  say whether departments belong in orgs.
- Table text with OCR-like glitches ("Glass cross", "VIP foil", "FBI") - rule for as-written vs corrected material
  names would help (kept as written, noted "likely cloth").
- Validator: does not check that seq is unique per (tv_id, structure), nor that evidence `entries`/`derived` row_ids
  exist (docstring says derived rows are checked; code checks only `formula`).
