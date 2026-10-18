# b019 batch report (2026-10-18)

Validator: `uv run python scripts/validate_db.py --staging data/_staging/b019` -> `0 errors, 0 warnings`.

| paper_id | status | TV | comp | layers | ic | steps | elec | opt | metrics |
|---|---|---|---|---|---|---|---|---|---|
| ectc51906.2022.00232 | distilled | 4 | 15 | 27 | 6 | 6 | 0 | 0 | 33 |
| ectc51906.2022.00135 | distilled | 3 | 5 | 16 | 2 | 24 | 0 | 0 | 41 |
| tcpmt.2023.3312037 | distilled | 5 | 25 | 8 | 2 | 21 | 0 | 0 | 26 |
| total (3 papers rows) | | 12 | 45 | 51 | 10 | 51 | 0 | 0 | 100 |

None of the papers has a TEXT_LAYER_WARNING. The primary_bucket from the batch file is kept for all three papers.

## ectc51906.2022.00232 - MediaTek large fan-out module reliability (ECTC 2022)

Status: distilled. primary_bucket organic_rdl_fanout.

TVs:
- (a) TV1 is the package on a 74x74 mm substrate.
- (b) TV2 is the package on a 91x91 mm substrate.
- (c) The fan-out module sub-assembly (41.3x33.6 mm, 1.6X reticle; SoC 31.5x23.8 mm at 7 nm plus 8 I/O dies of 15.5x4 mm; 60 um D2D gap; RDL1/2 at 2/2 um and RDL3 at 5/5 um; 130 um Cu pillar C4) is shared by TV1 and TV2. It holds the module-level shadow moire and simulation data.
- (d) The FEA RDL trace model (Tables III and IV, legs 1-7).

Figures opened: page_01..05.png. All numbers are in tables or legible cells.

Judgment calls
- Module size: the text says 41 x 33 mm and Tables I/II say 41.3 x 33.6 mm. The table value was used and the conflict is logged.
- The pre- and post-optimization RDL designs are entered as metric variants, not as separate TVs. The paper gives the optimized minimum trace width 2.2 um, thickness 2.3 um and via-to-turn length >18 um without naming a layer, so they are module metrics on TV c.
- Package warpage is reported only as ratios to TV1 (1.3X at 25 C, 1.6X at 250 C). TV2 carries the ratios with reference_tv_id = TV1. TV1 records only the warpage shape.
- The X-ray C4 check and the post-reliability cross-section FA are not tied to a TV. Each is entered once on TV1 with a note. The BGA appears only in the generic Fig. 1 and is entered on TV1.
- The package_class is fan_out_wafer (chip-first reconstituted wafer). The paper uses no technology name.

Not reported: die, mold and package thickness; wafer size; RDL dielectric material and thickness; via sizes; C4 diameter and height and the bonding conditions; substrate layer count; absolute package warpage; sample counts; the number of reflows in the failing pre-optimization test.

Source issue: in the text.md abstract (p.1), the '>=' in ">=65x65mm2" is extracted as 'ı'. I read that line from the page image. No other garbling.

## ectc51906.2022.00135 - Amkor Hybrid PLP on 650x650 mm glass panel (ECTC 2022)

Status: distilled. primary_bucket panel. technology_name: Hybrid PLP / HPLT, which is the paper's own name.

TVs:
- (a) Six-metal RDL-first TV on the 650x650 mm glass panel: UBM, M1-M4 and the die-side UBM. Nominal thicknesses are 7/5/5 um (UBM/RDL/dielectric), with minimum 5/5 um L/S. The die is attached by flip chip with mass reflow, then MUF, laser debond and blade dicing.
- (b) FEA panel warpage model: 4 RDL and 5 dielectric layers, 3 glass CTEs, 2 panel thicknesses.
- (c) The platform itself: 4x300, 9x200 or 16x150 mm wafers per panel, plus the cost models in Fig. 4 and Fig. 5.

Figures opened: page_01..04.png. From source.pdf: Fig. 9 at 600 dpi, Fig. 7 at 600 dpi, Fig. 8 at 400 dpi and Table 2 at 300 dpi.

Judgment calls
- "Six-metal" and "4 RDL" are consistent: 2 UBM layers plus M1-M4.
- Table 1 inline metrology goes in as 12 metrics: the mean as the value, with min, max and stdev in value_text and the spec in pass_criterion. The CD feature is not defined by the paper.
- The 11 thickness labels in the Fig. 9 cross-section are entered top to bottom. Their assignment to layers (metal vs dielectric) is my inference from image contrast, and each row is flagged.
- Fig. 7 and Fig. 8 have no readable axis or colorbar values, so warpage is qualitative only (measured well below simulated and below the handling limit).
- Reliability (Table 2): MRT 3x at 260 C (50 ea), TCT_B 1000 cycles, uHAST 192 h at 130 C/85% RH and HTST 1000 h at 150 C (25 ea each) all passed. n_samples is the per-lot count.
- The die-to-RDL joint is entered as cu_pillar with an override to schematic, because only the Fig. 6 drawing shows a pillar. pad_x_um is 85 um, the die-side UBM CD.
- The dielectric is labelled only "PI" in Fig. 7 and Fig. 8, so material_class is left unspecified.
- Cost values are % of wafer-line cost with basis simulated. The paper calls this a "cost simulation".

Not reported: die and package dimensions; bump pitch and solder; dielectric product and cure; glass thickness and CTE; debond conditions; absolute warpage; TCT_B temperatures; preconditioning.

Source issues
- Table 2 is image-only. text.md has the caption only.
- The Fig. 2 flow-chart text appears out of place at the end of p.2 in text.md.

## tcpmt.2023.3312037 - Intel PVC Co-EMIB underfill on die (TCPMT 2023)

Status: distilled. primary_bucket si_bridge. technology_name Co-EMIB.

TVs:
- (a) PVC Co-EMIB product package: die height delta up to 400 um, DtD spacing as small as 100 um, mold shelf about 250 um, UOD acceptance below 250 um. The HVM flow covers wafer-level UF and mold, BSM, laser trench, singulation, TCB, package underfill and solder TIM. The Table I comparison of the four solution paths is on this TV.
- (b) Trench coupon on a glass slide: about 350 um height delta. The Fig. 6(a) target is a 150 um trench, >20 um deep, with a 50 um ledge.
- (c) Hydrophobic top-coating coupon. Control: theta_e 10-20 deg, theta 11 deg. Treated: theta_e >90 deg, theta up to 85 deg.
- (d) Hydrophobic barrier coupon.
- (e) Sidewall-coating coupon: 90 / 0 deg, DtD gap 105.71 / 107.69 um.

Figures opened: page_01..05.png. From source.pdf: Fig. 6 at 600 dpi; Fig. 7(c), 7(d), 8 and 9(c) at 800 dpi.

Judgment calls
- Fig. 6(a): I assigned the labels to trench width (150 um) and outer ledge (50 um) from where the arrows sit. This is flagged.
- Many SEM and optical measurement labels in Fig. 7 and Fig. 8 are not explained in the text and were left out (listed in evidence `excluded`). The one exception is Fig. 8 label [1] = 308.29 um, entered as the die height delta from the vertical arrow (inferred).
- Table I (control, method, cost and risk per path) is entered as value_text cost metrics with basis author_estimate.
- The alternative HVM flows that were not selected are entered as process_steps with performed = no.
- HBM and transceiver counts from Fig. 1 were not entered, because that figure is a reprint of ref [1].

Not reported: package, die and EMIB dimensions; bump pitch; TCB conditions; material products; laser recipe; HVM UOD yield; thermal data.

Source issue: Table I is image-only; text.md has the caption only.

## Schema gaps
- There is no component_type for a Foveros 3-D stack, a transceiver die or a glass slide/carrier panel. I used `other` with descriptive names.
- metrics has no columns for min, max and stdev together. Inline metrology tables (mean/min/max/stdev plus spec) were put into value_text and pass_criterion.
- There is no field for a feature's location on the die (trench distance to die edge, mold-shelf width). These went into geometry metrics.
- There is no column for the carrier panel that holds the sub-panels (Hybrid PLP holds 4x300 mm sub-panels on 650x650 mm). Only carrier_size_mm exists, so the hierarchy is in a throughput metric.
- Contact angle and wetting have no metric_category. I used `other`.
