# BATCH_REPORT b003 (2026-10-18)

## ectc51906.2022.00244 - Buried Power Rails and Nano-Scale TSV (ECTC 2022, imec) - status: distilled

Source: full text, 8 pages, correct paper, not truncated, not scanned. **The text.md text layer is font-encoded:** letters
are shifted by +29 code points, and digits and spaces are dropped. Not a single number survives in text.md, so every
number was read from page renders of source.pdf. primary_bucket thermal_pdn is kept from the batch file (BSPDN/PDN paper).

| table | rows |
|---|---|
| papers | 1 |
| test_vehicles | 3 |
| components | 6 |
| stack_layers | 24 |
| interconnects | 3 |
| process_steps | 70 |
| electrical_interfaces | 0 |
| optical_interfaces | 0 |
| metrics | 61 |

TVs (rule f, split by metallization material = different stack): a W n-TSV on W BPR (Fig. 13 options 1-4);
b W n-TSV on Ru BPR (options 5-7); c CVD Co + Co ECD n-TSV on W BPR (option 8). Liner-etch time, SSE depth and W type A/B
are split variables and go in metrics.variant and process_steps.conditions.

Figures opened: page_01..08 (re-rendered from source.pdf at 220 dpi in quadrants), img_p03_1.png (Fig. 5, upscaled and
pixel-calibrated), img_p04_3.png (Fig. 6, upscaled, label crop), img_p04_1.png (Fig. 7), Fig. 13 panels and table
re-rendered at 450 dpi and pixel-calibrated against the axis frame.

Judgment calls
- Stack seq 1 = wafer backside surface after thinning (BSM1), going down to STI. Then the SiCN bond pair (bonding_interface)
  and the carrier (other). BPR is its own seq between the Si cap and STI because it is co-planar with both.
- Si thickness: I entered 350 nm (text p.2/p.3/p.7, Fig. 4 labels, Fig. 6 TEM 429 - 82.6 ~ 346 nm). The Fig. 13 table lists
  "500nm" for all options. This is logged as a conflict, and each metric condition notes it.
- Overlay: the text says <10 nm after correction, but Fig. 5 boxes read ~12 nm (X) and ~22 nm (Y). interconnects.alignment_um
  = 0.022 (measured plot, worst axis). The text claim stays as its own metric row, and the conflict is logged.
- Fig. 6 labels (122.8, 129.0, 429.x, 82.6 nm) are entered as annotated dimensions. The paper does not say which feature each
  one measures. The interpretation is in the notes, marked inferred. The 429 label is partly illegible (approx).
- Fig. 13 box plots: median entered only where a median line can be resolved. IQR, whiskers and outliers are in value_text.
  Medians hidden by dark fill or the cyan simulation band are left empty. Reading precision is about 1 Ohm (Kelvin), ~2 Ohm/um
  (BPR) and ~1 Ohm (chain).
- Simulated values: Kelvin and chain ranges come from the arrow labels (min/max rows, basis simulated). For BPR, the labels
  are 220 and 60 Ohm/um, but the dashed lines are plotted near ~240 and ~118. I chose the labels and logged the Ru case as a conflict.
- The 93 and 90 Ohm BPR-per-link values are the paper's own subtraction (basis derived).
- n-TSVs per chain: the Fig. 13 caption gives 1902 and the Fig. 10 table gives 3800 (blocks 1-5). Both are entered, and the
  conflict is logged (possibly per chain vs CH_A+CH_B; not stated).
- Process-generic metrics (TTV, overlay, Fig. 6) hang off TV a only. The full process flow is repeated for each TV, with
  per-TV metal, liner-etch and SSE conditions.
- SPTS-Newport appears only in the acknowledgments, so it is noted ("ack") on the n-TSV etch step. No equipment is named.
- package_class other / tv_kind wafer_level: no BSPDN or wafer-level PDN class exists.

Not reported: electrical/optical interfaces; n-TSV pitch and count per Kelvin; BPR height; BSM1 dimensions; SiCN thickness;
carrier material; chemistries and deposition temperatures; definition of W type A vs B; option 8 BPR and chain data
(panels empty); overlay sample count; reliability, thermal, yield.

Excluded (cited or background): Fig. 1 IR-drop maps (ARM, ref [2]); CVD-Ru 40% lower BPR R vs W (ref [7]); >20% of routing
resources for PDN (background).

Validator: `uv run python scripts/validate_db.py --staging data/_staging/b003` -> 0 errors, 0 warnings.

Schema gaps
- No BSPDN / backside-PDN package_class or wafer-level PDN test-vehicle class.
- stack_layers has line_um/space_um but no line pitch column. BPR 105 nm pitch went into the derived space and the notes.
- metrics has no IQR/quartile or whisker columns. Box-plot reads go into value_text.
- interconnects.alignment_um is a single value. Overlay per axis (X/Y) and before/after correction need metrics rows.
- No resistance-per-length unit column or field for line resistance (Ohm/um), so it is recorded only in metrics.
- interconnects has no non-circular via CD field (90x120 nm). The text is in notes.
- Source-quality flag: papers has no field for "text layer unusable". It is recorded in papers.notes and the evidence locator_notes.
