# pilot_1 batch report (2026-10-18)

## ectc51906.2022.00142 - Fine Copper Lines with High Adhesion on High Rigidity Dielectrics (ECTC 2022)

Status: distilled. primary_bucket advanced_substrate (kept from batch file; full text agrees).

| table | rows |
|---|---|
| papers | 1 |
| test_vehicles | 5 |
| components | 4 |
| stack_layers | 33 |
| interconnects | 2 |
| process_steps | 48 |
| electrical_interfaces | 0 |
| metrics | 43 |

TVs: a peel coupon (prepreg); b FC-BGA mock-up (prepreg); c FC-BGA mock-up (insulation film ref); d comb coupon
(prepreg, L/S 10/10 and 8/8 um); e comb coupon (insulation film ref).

Figures opened: page_02.png, page_03.png, page_04.png, page_05.png, img_p03_1.png (Fig. 5), img_p03_2.png (Fig. 4),
img_p04_3.png (Fig. 10), img_p04_5.png (Fig. 8), img_p04_6.png (Fig. 9), img_p04_7.png and img_p04_8.png (Fig. 7),
img_p04_9.png (Fig. 6), img_p05_1.png (Fig. 12); Table III re-rendered from source.pdf at 250 dpi to resolve cell spans.

Judgment calls
- Split variables (foil press, UV dose, Pd-removal time) kept as metric conditions, not TVs; dielectric (prepreg vs
  insulation film) split into TVs. See SCHEMA_FEEDBACK.md item 1.
- Fig. 5 (peel), Fig. 6 (warpage) and Fig. 12 (insulation resistance at 200 h, log axis) values read by pixel
  calibration against gridlines; basis extracted_from_figure, qualifier value:approx. Fig. 6 check: 35/79 and 36/79
  match the text's 45%.
- Fig. 7 coplanarity labels illegible at native image resolution: not used. SEM/FIB scale bars (Figs. 9-11, 13) not
  used for dimensions (tilted FIB views; no value printed on features).
- Fig. 12 shiny-side 60 s curve hidden under other curves after about 30 h: no 200 h value taken.
- Warpage sign: as plotted (RT negative, 260 C positive), convention undefined in paper; noted on each row.
- Core Cu cladding rows (d/e L05, L07) come from the Fig. 4 schematic only (extracted_from_figure, no thickness).
- Insulation film reference TVs (c, e): core and plated Cu listed by analogy, flagged "stated" with notes; film
  thickness left empty ("same thickness as prepreg", ply count not given).
- Table III cover material "Prepreg without Glass cross" spans all rows incl. the insulation film reference; kept as
  written.
- Skipped Fig. 4 steps (half etch, blackening, laser via + desmear) recorded with result "skipped in this paper".
- Unit conversions (mm -> um, s/h -> min) logged in evidence `derived`; row basis kept as source basis.

Not reported by the paper: electrical interfaces (40 GHz microstrip result is prior work [22, 23]); product names and
CTE/modulus/Tg/Dk/Df of prepreg, core, film; seed and plated line thickness; die joint type/pitch and BGA; underfill
material; UV dose for warpage samples; reagent/etchant chemistries; cover-resin method; vias.

Source quality: text-layer extraction fine except mu glyph rendered "P" (e.g. "30-Pm") and figure label garbage on p.4;
caption typos in the paper itself ("VIP foil", "FBI", "Glass cross", Fig. 11 "(b) and (d)"). Not scanned, not truncated.

## v2 migration (2026-10-18)

Re-built all CSVs from the v2 TEMPLATE headers (new column order; optical_interfaces.csv added, 0 rows). Row counts
unchanged: papers 1, test_vehicles 5, components 4, stack_layers 33, interconnects 2, process_steps 48,
electrical_interfaces 0, optical_interfaces 0, metrics 43. Validator: 0 errors, 0 warnings.

What changed
- basis/source split: former extracted_from_figure rows -> basis measured + source figure_image (Fig. 5 peel,
  Fig. 6 warpage, Fig. 12 IR); core Cu rows -> stated + schematic; table-sourced rows -> source table.
- overrides: TV b/c `n_dies:figure_image`, component c1 `count:figure_image` (replaces evidence-only notes).
- test_vehicles.tv_kind: a, d, e test_coupon; b, c package.
- components: is_emulator yes / embedded no for the Si mock-up chip.
- stack_layers: side (front/core/back) with seq restarted per (structure, side); underfill moved to structure
  `underfill`; 10/10 and 8/8 combs are two rows with same seq and different `feature` (old -L02b id removed, ids
  renumbered); reinforcement filled (glass cloth / without glass cloth / "without glass cross (as written)").
  roughness_ra_um left empty: Ra 0.2 um applies to the shiny-VLP variant only, kept as a metric with variant.
- process_steps: performed=no on half-etch, blackening, laser via + desmear; time_min -> time_s (60 s native;
  168 h and 200 h converted, logged); categories resist_strip, etch_other, surface_treatment, preconditioning
  replace `other`; ack supplier (MEC COMPANY LTD.) noted on surface-treatment steps; numeric condition rows
  (UV dose, Pd time, HAST, reflow) set to design_target.
- metrics: variant holds foil side / UV dose / Pd time; statistic mean + n_samples 3 for peel; n_samples 6,
  pass_criterion "> 1.0 MOhm", result pass/fail for HAST; n_samples 192 for no-peeling; warpage sign convention in
  scope; 45% ratio gets reference_tv_id = TV c; peel -> category adhesion, line-width and Ra -> geometry;
  Ra basis -> unspecified (method not given).
- interconnects: die_to_substrate joint -> ic_type unspecified, bonding_method unspecified.
- evidence: added conflicts (Table III vs Sec. II.G cover material; 45% vs "reduced by half"), excluded (prior-work
  numbers from refs [17], [18], [22, 23]); derived list now mm->um and h->s only; entries carry basis + source.
