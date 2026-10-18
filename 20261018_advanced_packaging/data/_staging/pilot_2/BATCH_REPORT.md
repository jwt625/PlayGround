# BATCH_REPORT pilot_2 (2026-10-18)

## ectc51906.2022.00072 - status: distilled

Source: full text, 7 pages, not truncated, correct paper, text layer present (not scanned). primary_bucket cpo kept
from batch file; note the paper is a PIC fiber-attach sub-assembly study (CPO only as motivation, no ASIC co-package).

Rows: papers 1, test_vehicles 6, components 9, stack_layers 5, interconnects 2, process_steps 18,
electrical_interfaces 0, metrics 37.

TVs: a 12-Ch Vgroove passive fiber attach sub-assembly; b Vgroove width offset split; c DCB strip 90WG; d DCB strip
advanced CMOS reference; e BOR coupon with Vgroove; f BOR coupon without Vgroove.

Figures opened: page_03.png, page_04.png, page_05.png, page_06.png, page_07.png, img_p03_1.png, img_p03_2.png,
img_p03_3.png, img_p04_1.png, img_p04_2.png, img_p04_3.png, img_p04_4.png, img_p04_5.png, img_p04_6.png,
img_p05_1.png, img_p06_1.png, img_p06_2.png, img_p07_1.png (Fig. 2d, 3a, 5, 6 also upscaled in scratch for reading).

Judgment calls:
- 6 TVs per rule (f): controls (CMOS DCB, no-Vgroove BOR) split as own TVs.
- package_class other for all TVs (no PIC sub-assembly or coupon class).
- Optical I/O recorded as interconnect level optical / optical_coupling (250 um pitch, 12 ch, adhesive); alignment
  1.3 um (Monte-Carlo, simulated) moved to a metrics row to keep row basis honest.
- Fiber lid, Si-submount, three adhesives as component_type other.
- Vgroove crystallographic etch, UV tack, oven cure, cleave as step_category other; adhesive order unknown (noted).
- Figure reads (approx): Fig. 3a IL best/worst points; Fig. 5 per-condition IL shift ranges; Fig. 6 lognormal
  1/50/99% lines (log axis); Fig. 2d medians of Norm_Gc (1.79 vs 1.0) and sigma_max,norm (3.4 vs 1.0).
- SEM Fig. 1(c): scale bar not legible (only "x500 SE(LM)"); no Vgroove dimensions taken.
- Cited-work values (-20 to 150 C operation, reflow compatibility, ref [6]) excluded.

Not reported: electrical interfaces; Vgroove width/depth/angle; SSC dimensions; PIC die size/thickness; submount
size/attach; adhesive products, UV dose, cure temp/time; lid material; BEOL levels/materials; numeric assembly yield;
reflow count; extended TC cycles; DCB/BOR sample counts and absolute values.

Source issues:
- Text cites Fig. 2(c) for results; caption labels them (d).
- Fig. 5 panel labeled "LTS 500hrs" vs Table 1 LTS 72 h.
- Time-zero IL scope (per facet vs per loopback) not stated; IL sign convention differs across Fig. 3, 5, 6.
- No page_01/page_02 renders; Fig. 1(a) renders on page_03.png (located at p.3).

Validator: `uv run python scripts/validate_db.py --staging data/_staging/pilot_2` -> 0 errors, 0 warnings.
Schema issues: see SCHEMA_FEEDBACK.md.

## v2 migration (2026-10-18)

The schema v2 migration replaces everything above. The v1 row counts and TV letters above are superseded.

Rows (v2): papers 1, test_vehicles 4, components 7, stack_layers 5, interconnects 0, process_steps 17,
electrical_interfaces 0, optical_interfaces 2, metrics 34.

Restructuring:
- TVs 6 -> 4: a sub-assembly (tv_kind sub_assembly, package_class fiber_subassembly); b DCB strip 90WG; c DCB strip
  advanced CMOS reference; d BOR coupon 90WG (b-d tv_kind test_coupon, package_class other).
- v1 TV b (Vgroove width offset split) merged into a: the offset is a split variable on the same stack (rule f).
  The width-split IL values are now metrics.variant "Vgroove width near target/narrowest/widest, TE/TM", and the
  split's process steps are a-s13/s14 (flow "Vgroove width split").
- v1 TVs e/f (BOR with/without Vgroove) merged into d: same wafer stack, Vgroove presence is metrics.variant.
- DCB CMOS reference kept as its own TV c: a different BEOL stack (porous ULK). b-m2 (80% toughness gain) uses
  reference_tv_id = c.
- Optical I/O moved from interconnects (v1 a-i1, b-i1, now removed) to optical_interfaces a-o1 (TM 1.5 dB) and
  a-o2 (TE 1.7 dB): coupler_type edge (inferred, in-plane SSC; noted), alignment self_aligned, O-band, 12 channels,
  250 um pitch, SMF, MFD chip 0.35 um / fiber 9 um (design_target overrides), attach material index-matching adhesive.
  The time-zero IL metric rows were dropped so the values are not duplicated.
- extracted_from_figure removed: figure reads are now basis measured + source figure_image. Overrides mark the
  design_target cells in optical rows, and n_dies:figure_image on TV a.
- New enums used: component_type fiber_lid, submount, adhesive (x3); step_category etch_other (Vgroove), fiber_prep
  (cleave), dispense (x3 adhesives), cure (UV tack, oven cure); structure coupon (CMOS BEOL).
- Metrics carry statistic (median/percentile/min/max/single), n_samples (45/36), test_standard (Telcordia GR-468),
  pass_criterion (IL shift per facet > -0.71 dB), result pass, and scope (per facet + sign convention).
  The RSS criterion -0.71 dB is basis derived (the paper's own calculation).
- Evidence YAML: conflicts (LTS 72 h Table 1 vs "LTS 500hrs" Fig. 5 panel, chosen 72; Fig. 2(c) text vs (d)
  caption) and excluded (ref [6] -20 to 150 C operation and reflow compatibility; refs [2,3,6-8] throughput claims).
  Unit conversions (780 um -> 0.78 mm, 350 nm -> 0.35 um) are logged in derived with their original basis kept.
- page_01/page_02 renders now exist. page_02 has no figure image, which confirms Fig. 1(a) is on p.3.

Validator: `uv run python scripts/validate_db.py --staging data/_staging/pilot_2` -> 0 errors, 0 warnings.

Remaining schema problems (v2):
- optical_interfaces has no statistic, n_samples or condition column. The time-zero mean over 153 modules x 4
  loopbacks is only in notes; per-facet DH distributions stay in metrics, so optical loss is split across two tables.
- optical_interfaces.wavelength_nm is a single float. A 1270-1330 nm range has no min/max columns, and rule (d)
  "one row per value" does not fit a band; I used band "O-band" and kept the range in process step a-s12.
- optical_interfaces has no alignment_tolerance_um. The Monte-Carlo 1.3 um alignment (simulated) stays in metrics.
- coupler_type lacks a value for an in-plane suspended SSC butt-coupled in a V-groove; `edge` is an inference.
  alignment enum: passive vs self_aligned overlap (this paper uses both words).
- One test row per polarization duplicates every design field. A polarization-list form (TE;TM) with per-pol loss
  would be leaner.
- package_class still has no coupon value (tv_kind test_coupon + package_class other). package_x/y_mm still hold
  coupon dimensions.
- No component columns for fiber cladding, coating or MFD (fiber_type/mfd only exist on optical_interfaces).
- metrics.statistic `percentile` does not say which percentile (p1 vs p99); I put it in parameter. Suggest a
  percentile_value column.
- metrics.result has no value for qualitative "stable" outcomes without a stated criterion (reflow, extended TC,
  DH 2000 h); I left result empty.
- No source value for an axis-annotated plot vs a photo (both are figure_image); this is acceptable but limits audit.
