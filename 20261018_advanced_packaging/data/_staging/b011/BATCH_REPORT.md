# b011 batch report (2026-10-18)

Validator: `uv run python scripts/validate_db.py --staging data/_staging/b011` -> `0 errors, 0 warnings (data/_staging/b011)`.

| table | ectc51529.2024.00088 | ectc51529.2024.00144 | ectc32862.2020.00185 | total |
|---|---|---|---|---|
| papers | 1 | 1 | 1 | 3 |
| test_vehicles | 1 | 4 | 5 | 10 |
| components | 0 | 12 | 14 | 26 |
| stack_layers | 9 | 4 | 19 | 32 |
| interconnects | 0 | 4 | 2 | 6 |
| process_steps | 8 | 0 | 29 | 37 |
| electrical_interfaces | 0 | 6 | 0 | 6 |
| optical_interfaces | 0 | 0 | 0 | 0 |
| metrics | 39 | 31 | 45 | 115 |

## ectc51529.2024.00088 - Dry processes to form fine features on advanced substrates (ECTC 2024, Applied Materials)

Status: distilled. primary_bucket advanced_substrate (kept).

TV: a, an ABF-on-Cu test coupon. DFR (~15 um) is laminated, patterned by laser digital lithography and developed in Na2CO3
(30 C, 0.18 MPa spray), then post-exposure baked (80 C, 30 s). ABF is etched by ICP + CCP F-based plasma and the DFR is removed by O2 ICP ashing.

Figures opened: page_02 (Fig. 1, Fig. 2), page_03 (Figs. 3-5), page_04 (Fig. 6). Fig. 4 and Fig. 6 re-rendered at 400 dpi.

Judgment calls
- Followed the ectc.2019.00-29 convention for via openings without metal: design features (via CD/pitch, trench CD/pitch) go in stack_layers
  as separate `feature` rows at seq 2 (ABF). Measured dimensions go in metrics. No interconnect rows (unmetallized).
- Trench CD entered as line_um, with space_um = pitch - CD (derived, logged).
- Fig. 6: horizontal arrows read as top and bottom CD, the vertical arrow as depth. This matches the caption depths 6/10/10 um.
- Fig. 4 per-feature labels (14.3/14.1, 8.3/8.1, 5.9/6.0 um) entered instead of the overall text values (~14.2, ~8.2, ~6). Conflicts logged.
- The Fig. 6 caption panel ranges ("(a-b)" vias, "(d-e)" trenches) and the Fig. 4 caption "(b)" for panel (d) are inconsistent with the images.
  Image panel letters are used.
- Etch rates and selectivity have basis derived (the authors' calculation). Etch time is not reported.

Not reported: ABF grade and thickness, Cu thickness, coupon size, exposure dose, plasma power/gas/time, taper angle, metallization.
Source: clean text layer, 4 pages, not truncated. ABF supplier Ajinomoto Fine-Techno from the acknowledgment (in TV notes).

## ectc51529.2024.00144 - High Bandwidth and Energy Efficient Electrical-Optical System Integration Using COUPE Technology (ECTC 2024, TSMC)

Status: distilled (simulation only). primary_bucket cpo (kept). buckets adds si_interposer.

TVs (all simulation_model):
- a: SoIC-based COUPE OE (EIC on PIC).
- b: microbump OE baseline.
- c: enhanced interposer XPU-EIC / XPU-HBM3 link.
- d: conventional interposer baseline.

Figures opened: page_02 (Figs. 1-3), page_03 (Table I, Figs. 4-7), page_04 (Table II, Figs. 8-10, Table III). Fig. 3
re-rendered at 400 dpi, Figs. 5 and 7 at 300 dpi.

Judgment calls
- Nearly all results are normalized (x microbump, x conventional). They are entered as metrics with unit "x ..." and reference_tv_id.
  No absolute pitch, energy or laser power is given.
- The SoIC bond follows the b006 COUPE convention: level die_stack_internal, ic_type other, bonding_method unspecified. Hybrid bonding is not stated.
- MRM density: Table I ">10.7x" chosen over the text "10.7 times" (qualifier gt).
- PD 3 dB bandwidth: 32 GHz microbump (text; the figure reads ~31). SoIC ~40.7 GHz is read from the Fig. 5 crossing (approx).
- Fig. 7 bars are read against gridlines (approx). Table II values are normalized to SoIC at 50 Gbps and 1e-12 = 1.00x.
- Interposer package_class si_interposer_2p5d and channel_type si_interposer_rdl are inferred from the "Silicon" base in Fig. 9 (noted).
- The XPU-EIC link has 8 lanes, taken from the 8 GSGS groups / "8-sig" RLGC model. HBM3 link: 6.4 Gbps, 7 mm.
- No optical_interfaces rows: the paper gives no coupler type, loss or fiber data.
- Excluded: SoIC 85% lower capacitance (ref [13]), the Fig. 2 ring-radius trend (refs [22-31]), the PD RC from ref [18].

Not reported: process steps, absolute dimensions, die sizes, interposer L/S, absolute pJ/bit, coupler/fiber details, hardware measurements.

## ectc32862.2020.00185 - Mold-First FO-PLP on 600 mm x 600 mm panel (ECTC 2020, IME A*STAR)

Status: distilled. primary_bucket panel (kept). buckets adds organic_rdl_fanout.

TVs:
- a: mold-first FO-PLP package. 10 x 10 x 0.8 mm, 7 x 7 x 0.7 mm daisy-chain die, 1L Cu RDL 10/10 um, 392 I/O at 400 um,
  on a 600 x 600 mm panel with granular EMC G1.
- b: blanket granular-EMC compression-molded glass coupon.
- c: blanket liquid-EMC coupon (qualitative only).
- d: sheet-molded bare glass 650 x 550 mm (Table 4 runs 1-4).
- e: sheet-molded glass with mold tape and 700 um dies (Table 4 runs 5-6, Table 5, 3-step lamination).

Figures opened: page_02 to page_07 (Table 1, Figs. 1-3; Table 2, Figs. 4-6; Table 3, Table 4, Figs. 7-9; Fig. 10,
Table 5, Fig. 11; Figs. 12-16; Table 6, Fig. 17). Tables 2-5 and Figs. 10, 13, 14, 16 re-rendered at 300-400 dpi.

Judgment calls
- Molding parameter splits (Table 3 runs 2-4), die shift with/without compensation, and the mold tape N1/M1 choice are metric variants,
  not new TVs. Liquid vs granular vs sheet EMC are different builds and get separate TVs.
- Dielectric thickness: Table 1 gives 7 um and the text says "10 um coated". 7 um was entered (conflict logged). The 10 um is kept in the coating step.
- Max die shift with compensation: 66 um (body text and Fig. 14 labels) over ~60 um (abstract and conclusion).
- Max die shift without compensation: 324 um (text). An unlabeled Fig. 13 Y-axis point at about 365 um is logged as a conflict.
- Text says "singulation of 100mm x 100mm packages", likely a typo for 10 x 10 mm. It is kept as written in the step result and logged.
- Fig. 16 shows 0.5 mm warpage after molding, while Table 3 gives 3.5 mm (after PMC, de-bonded). Both are kept and logged.
- Table 1 "Bump size 200 um" is entered as diameter. The paper calls the bumps C4 made by ball drop, so ic_type is solder_ball.
- Reliability: Table 6 standards entered as written (no JEDEC number is cited). 10/10 pass each. The pass criterion is no void/delamination.
- Unit conversions logged: 50 kgf/cm2 -> 4.903 MPa, 200 kPa -> 0.2 MPa, 1000 A -> 0.1 um, 4 h -> 14400 s.

Not reported: dielectric/PR material names (only "polyimide", "positive tone"), solder alloy, die count per panel,
dielectric cure conditions, seed etch method, warpage sign convention, sheet lamination pressure/time values, electrical results.
Source: clean text layer, 7 pages, not truncated. All tables are images.

## Schema gaps
- No natural home for unmetallized etched features (via/trench openings in a dielectric): they are stuck in stack_layers feature rows with
  line/space semantics and geometry metrics.
- Simulation-only comparison papers give normalized results ("x baseline"). The metrics table holds them through unit plus
  reference_tv_id, but electrical_interfaces has no relative-value field, so energy/BW ratios cannot sit on the interface row.
- No component_type for a temporary carrier (glass panel carrier) or release tape; used other / adhesive.
- interconnects has no field for pad opening vs pad size (diameter_um used for opening, pad_x_um for pad).
- No etch-rate / selectivity metric category; used other.
