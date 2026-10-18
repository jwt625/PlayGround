# b004 batch report (2026-10-18)

Validator: `uv run python scripts/validate_db.py --staging data/_staging/b004` -> `0 errors, 0 warnings`.

| paper_id | status | TV | comp | layers | ic | steps | elec | opt | metrics |
|---|---|---|---|---|---|---|---|---|---|
| ectc32696.2021.00166 | distilled | 6 | 7 | 9 | 5 | 10 | 5 | 0 | 17 |
| iedm19573.2019.8993659 | distilled | 4 | 7 | 5 | 4 | 6 | 0 | 0 | 21 |
| ectc32696.2021.00125 | distilled | 3 | 13 | 16 | 6 | 3 | 0 | 0 | 16 |
| total (3 papers rows) | | 13 | 27 | 30 | 15 | 19 | 5 | 0 | 54 |

## ectc32696.2021.00166 - Intel hybrid bonding interconnect (ECTC 2021)

Status: distilled. primary_bucket hybrid_bonding (kept).

Source defect: the PDF text layer is font-encoded and unusable (text layer garbled; OCR used). text.md was swapped
for tesseract OCR mid-batch. No value was taken from either text layer: all pages were rendered from source.pdf
(200 dpi, page 4 at 400 dpi), which matches figures/page_NN_220dpi.png, and each number was read from the image.
The OCR was used only as a cross-check and agrees with every number entered.

TVs: a passive D2W HB assembly test chip (10 um hexagonal pitch, daisy chains, dies 15-190 mm2); b active test chip,
"Foveros [4] architecture", 4 top dies + 1 base die, Intel 10nm FinFET, 9 um pitch, >3 million HB connections
(still in fab, so no results); c/d F2F parasitic models (HB 10 um vs solder 40 um reference); e side-by-side
passive interposer channel model (pitch sweep); f Table 1 core IO-area analysis.

Figures opened: pages 1-6 at 200 dpi (column quadrants); page 4 at 400 dpi with Fig. 6(a) and 6(c) cropped.

Judgment calls
- Text and caption figure numbers disagree (text Fig. 3/2/4(b) = captions Figure 2/1/5(b)). Locators use the
  caption labels, and the mismatch is logged as a conflict.
- Fig. 6 read by pixel calibration into electrical_interfaces rows at pitches 10/20/30/40/50 um: reach = channel
  length (0.25-5.05 mm) and channel energy (0.005, 0.023, 0.07, >0.5 pJ/bit). Basis simulated.
- 6 GT/s per line not converted to lane_rate_gbps because the bits per transfer are not stated. It is kept in
  protocol_as_written.
- The >5X capacitance and >5X power reductions are derived ratios on TV c, with reference_tv_id d.
- carrier_format wafer_300mm comes from the fab-section text "within full 300mm wafer".
- Base-die metallurgy "Cu-Cu" is stated in Sec. II and Fig. 11 for the passive chip. It is not restated for the
  active chip, so it was left empty there.
- Fig. 1 (generic area-ratio vs pitch) is excluded. Fig. 6(b) has no y ticks, so only the text thresholds are used.

Not reported: die x/y/thickness, pad diameter/recess, dielectric material, bond/anneal temperatures and times,
alignment, daisy-chain resistance/yield, reliability, solder-version results, active-chip results.

## iedm19573.2019.8993659 - Intel ODI (IEDM 2019)

Status: distilled. primary_bucket microbump_3d (kept). technology_name ODI (paper's name; "Foveros Omni" not used).

TVs: a cavity-version proof of concept (2 top dies, bottom die in organic package cavity, microbumps + C4); b Cu
pillar version sub-assembly; c TSV resistance/area/cost model (Fig. 3); d 35x35 mm multi-reticle interposer cost
example.

Figures opened: page_03.png and page_04.png, both re-rendered at 300 dpi. Page 4 also at 800 dpi. Crops: Fig. 3(a),
3(b), 4(c), 5(c), 6 inset, 8(b). The embedded img_p04_6.png was upscaled.

Judgment calls
- package_class "other" for both ODI builds: the bottom die sits in a cavity with C4 down and microbumps up. That
  is neither si_bridge_organic nor a plain microbump_3d stack.
- Fig. 7(b) is labelled "ODI Type 2 - 1st Gen", but the Sec. II definition makes this geometry Type 1. No type is
  assigned and the mismatch is logged as a conflict.
- Process-window values (Z and XY) come from the shaded bands in Figs. 4(c) and 5(c). They are text ranges with
  basis derived, assigned to TV a (the paper uses the cavity version for illustration).
- XY-PW criterion: the caption says 80+%, but the plotted threshold is about 75%. 80 entered (ge); conflict logged.
- Fig. 8(b) SEM pillar-height labels are raster-limited (read tentatively as 137.4 / 135.9 um). They are recorded
  only as value_text with no numeric value.
- Fig. 3(b) cost reductions read at 100 and 500 mm2 for 0/0.1/0.5 defects/cm2. Cu pillar adder about 1.15 mOhm.

Not reported: any bump/pillar pitch, size or count; die and package dimensions; TCB conditions; numeric electrical
or reliability results; electrical/optical interfaces.

## ectc32696.2021.00125 - TSMC CoWoS-R reliability (ECTC 2021)

Status: distilled. primary_bucket organic_rdl_fanout (kept).

TVs: a 1 SOC (30x25) + 2 HBM2 package, 32x35 mm interposer, 55x55 substrate; b 2-SOC chiplet package, 53x34 mm
interposer (>2x reticle, non-stitching), 85x85 substrate; c flip-chip BGA FE reference (25x26 mm die). Both
packages: 4 RDLs, min L/S 2/2 um, Cu traces in low-k polymer, Cu vias in polymer, microbump min pitch 40 um, C4
Cu/solder, 5-2-5 low-CTE-core substrate.

Figures opened: pages 1-5 re-rendered at 300 dpi (column quadrants): Table 1-3, Figs. 1-8.

Judgment calls
- Reliability values are from Tables 2/3 (measured, pass). test_standard is JESD47K with JESD22-A113D MSL4
  preconditioning, as cited. JEDEC 500/850-cycle criteria go in pass_criterion.
- The SEM check after TCG 1000 is on TV a (stated). The FIB corner-C4-underfill check is on TV b (inferred from the
  two-SOC FE context, as noted).
- The FE results (ELK -53%, ubump dSED -34%, C4 dSED -43%, RDL P1 -35%) are stored as positive % reductions, basis
  simulated, on TV b. Where the comparison is against FCBGA, reference_tv_id is c. C4: text ">40%" vs figure
  "-43%"; 43 chosen and the conflict logged.
- The 5-2-5 substrate is split into build-up/core/build-up layer rows, as written.
- The SOC dimensions are given once for the chiplet package and applied to both SOCs (noted).
- interposer_area_mm2 is derived (1120, 1802).

Not reported: die/HBM thickness, RDL/via dimensions, microbump height/metallurgy, bonding conditions, C4/BGA pitch,
warpage, sample counts, absolute FE values, lid/ring (unlabeled blocks in Fig. 2(b)/3(b)).

## Schema gaps hit
- electrical_interfaces has no transfer-rate (GT/s) column and no signaling field (NRZ/PAM4, single-ended), so GT/s
  cannot be stored without inventing a conversion.
- stack_layers has no line-pitch column. "1 um routing line pitch" (line/space split unknown) went to notes.
- metrics has no min/max numeric columns, so process-window ranges (Z/XY PW) had to go in value_text.
- There is no package_class for ODI-like builds (cavity-embedded bottom die with bumps on both faces) or for an
  active-base-die chiplet over passive interposer. "other" was used.
- There is no field for a figure-vs-text numbering mismatch inside a paper. It was handled via conflicts on a
  TV-level row.
- A whole-paper source defect (garbled text layer, OCR substitution) has no structured field. It went to
  papers.notes and the evidence `source_defect` key.
- Illegible numeric labels (SEM measurements) have no "unreadable" flag. value_text was used.
