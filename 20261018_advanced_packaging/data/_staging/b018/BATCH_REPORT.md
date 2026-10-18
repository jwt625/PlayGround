# b018 batch report (2026-10-18)

Validator: `uv run python scripts/validate_db.py --staging data/_staging/b018` -> `0 errors, 0 warnings`.

| paper_id | status | TV | comp | layers | ic | steps | elec | opt | metrics |
|---|---|---|---|---|---|---|---|---|---|
| vlsitechnologyandcir46769.2022.9830182 | distilled | 10 | 16 | 6 | 8 | 1 | 2 | 0 | 17 |
| ectc51906.2022.00038 | distilled | 3 | 5 | 6 | 4 | 15 | 0 | 0 | 11 |
| total (2 papers rows) | | 13 | 21 | 12 | 12 | 16 | 2 | 0 | 28 |

Neither paper has a TEXT_LAYER_WARNING. The text layers were clean on every page. Each primary_bucket from the batch file was kept.

## vlsitechnologyandcir46769.2022.9830182 - TSMC SoIC_H (VLSI 2022, 2 pages)

Status: distilled. primary_bucket hybrid_bonding.

TVs: (a) SoIC_H architecture: chiplets and memory cubes SoIC-bonded face-to-face on a 5-metal Si interposer; (b) interposer
BEOL line coupon (5 mm single-ended, 1 mm differential), measured to 40 GHz; (c)/(d) D2D link models, SoIC 9 um vs
microbump 36 um interposer; (e)/(f) on-chip fanout models, 9 um vs 36 um; (g)/(h) 4-Hi vs 1-Hi SRAM models (N7, 64MB);
(i)/(j) 12-Hi SoIC_HBM vs microbump_HBM models.

Figures opened: page_02.png. From source.pdf: Fig. 1 and Fig. 8 at 400 dpi, Fig. 3 at 300 dpi.

Judgment calls
- Most of the paper is STCO, so each comparison gets a simulation_model TV for the SoIC case and one for the reference. The ratios go into metrics with reference_tv_id, following the ectc.2019.00095 precedent.
- The SoIC bond pitch range of 9 to 3 um is entered on TV a. The sentence cites refs [3][4], but the paper states the range as a property of its own architecture.
- The SoIC bond ic_type cu_cu_hybrid is inferred, because the paper never names the bond metallurgy.
- The contact resistivity "<10% of a reported microbump [5]" is characterized "by simulation and measurement" and is not tied to a build. It is entered once on TV a with basis unspecified.
- Fig. 4 table vs text: the text swaps the percentages (length 90% / area 60% lower), while the table gives area 0.1X and length 0.4X. The table values were entered and the swap is logged as a conflict.
- The abstract swaps the HBM numbers (30% BW / 28% EE). Fig. 8 and the body text give 1.28X BW and 1.30X EE, which were entered. Logged as a conflict.
- Fig. 8 lists a 36 um pitch for both the SoIC and microbump HBM. It is entered as printed.
- D2D electrical_interfaces: lane rate 32 Gbps is the top of the 2-32 Gbps sweep, and lanes = 256 bits (512 IO bonds). Energy per bit is only a ratio (about 66% reduction, figure label) because the Fig. 5(b) axis has no values.

Not reported: absolute energy, bandwidth and loss values; die and interposer sizes; layer thicknesses and line/space; bond
conditions; TSV and C4 dimensions; yield, reliability, warpage, thermal.

## ectc51906.2022.00038 - ASE/NTU MELI-Cu 30 um pitch Cu-Cu bonding (ECTC 2022, 5 pages)

Status: distilled. primary_bucket microbump_3d.

TVs: (a) chip-to-chip TV with 17,600 Cu pillars (25 um diameter) in three equal regions of 30/35/45 um pitch, 28 um
standoff, 5 um gap, corner Cu/Ni/SnAg to Cu/Ni/Au micro-bump spacers, underfilled after MELI-Cu; (b) 50 um diameter
pillar TV for the post-anneal void study; (c) horizontal side-by-side MELI-Cu concept.

Figures opened: page_02..04.png. Fig. 1 insets re-rendered at 600 dpi.

Judgment calls
- The three pitches share one die and the same other attributes, so they form one interconnect row with pitch_min/max of 30/45. The region-dependent plating results go in metrics variants.
- The micro-bump spacer layer heights (7/15/6 um) are schematic labels. Their sum, 28 um, matches the stated standoff. Only the SnAg cap (15 um) and the total length (28 um) are entered.
- The Table I LSV bath potentials are electrochemistry and not a build. They are entered on TV a with a note.
- Tensile strength: Sec. III.C says "above 80 MPa" and the abstract says "approach 80 MPa". It is entered as 80 with qualifier gt and logged as a conflict.
- The electroless step pressure_mpa is the >3 psi input flow pressure converted to MPa. The bond itself is pressureless, so bonding_pressure_mpa = 0 because the text says no pressure is applied.
- MELI-Cu bonding is entered with bonding_method other, because the enum has no electroless value.

Not reported: die sizes and thicknesses, pillar height, bath and pretreatment chemistry, plating time and flow rate,
underfill product, electrical resistance, yield %, reliability, warpage, n for the tensile test.

## Schema gaps
- bonding_method has no electroless or plating-based bonding value (MELI-Cu entered as other).
- electrical_interfaces lane_rate has no min/max columns, so a data-rate sweep (2-32 Gbps) can only store one value.
- STCO papers report mostly relative ratios. A metrics row with reference_tv_id works, but each comparison needs a reference simulation TV.
- There is no metric_category for bonding-interface quality such as void fraction (entered as other).
