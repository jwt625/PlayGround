# b015 batch report (2026-10-18)

Validator: `uv run python scripts/validate_db.py --staging data/_staging/b015` -> `0 errors, 0 warnings`.

| paper_id | status | TV | comp | layers | ic | steps | elec | opt | metrics |
|---|---|---|---|---|---|---|---|---|---|
| ectc32696.2021.00174 | distilled | 6 | 3 | 18 | 5 | 20 | 0 | 0 | 48 |
| ectc51909.2023.00180 | distilled | 7 | 34 | 18 | 14 | 15 | 0 | 0 | 47 |
| ectc51906.2022.00108 | distilled | 2 | 8 | 13 | 1 | 5 | 0 | 0 | 53 |
| total (3 papers rows) | | 15 | 45 | 49 | 20 | 40 | 0 | 0 | 148 |

No paper had a TEXT_LAYER_WARNING, and all text layers were clean. Every primary_bucket from the batch file was kept.

## ectc32696.2021.00174 - nepes 600x600 mm FOPLP with 6-sided die protection (ECTC 2021)

Status: distilled. primary_bucket panel (buckets panel; organic_rdl_fanout).

TVs:
- (a) 4.5x5.5 mm single-die chip-first face-up package built on the 600x600 mm panel line (Fig. 9, Tables 3-5).
- (b) Table 2 panel-segmentation usable-area calculation (5x5 mm package, credited "Source: DECA"). It is kept as a
  simulation_model to match the related IWLPC 2020 paper in b005.
- (c-e) BLR simulation models of the 6.15x6.15 mm fan-out package with 2P2M, 1P2M and 2P1M layer sets.
- (f) 6x6 mm WLCSP 2P2M reference.

Figures opened: page_01..06.png. From source.pdf at 400-600 dpi: Fig. 3, 7, 9, 14 and Table 2.

Judgment calls
- Fig. 3 is captioned "BLR Simulation", but its chart title reads "Results". It is entered as simulated. Cycles to 1%
  failure were read off the bars by pixel calibration: about 1130, 900, 750 and 680 cycles (+-20). The text gives
  >60%, >30% and >10% improvement over WLCSP; these rows have reference_tv_id f.
- Each mask-layer variant is a different layer stack, so each one is its own simulation TV (rule f).
- Table 4 lists both "RDL Thickness 15um" and "RDL Height 7um". The stack layer uses 7 um, both values are in
  metrics, and the pair is logged as a conflict.
- Table 4 "TTV Post Top Grind 0.715mm" is entered as printed. It cannot be a TTV under the <50 um spec, which is noted.
- Warpage, TTV post-mold and TTV post-grind are entered as CpK >1.33 against their specs (5 panels, all pass). The
  Fig. 10 maps have no scale.
- Ball shear n_samples = 20 was counted from the Fig. 12 grid (derived).
- Reconstitution time, the 4-64 mm2 sweet spot, the 5x die gain and "5 x 200 mm wafers per panel" are entered as
  author_estimate.
- Organization: no affiliation name is printed (addresses only). nepes Corporation comes from the email domain and
  the acknowledgment.

Not reported: die size/thickness/function, mold and PM dielectric names, RDL/UBM metal, Cu stud dimensions, ball
count, RDL via size, warpage magnitude/sign, process temperatures, measured BLR.

Source issues: Fig. 14 is a 405x302 px embedded image, and its cross-section measurement labels are illegible.
Excluded as cited-source content: Fig. 1, Table 1, Fig. 18, Fig. 20 (Techsearch/Yole/JCET) and the market figures.

## ectc51909.2023.00180 - IBM surface-DBHi Si bridge (ECTC 2023)

Status: distilled. primary_bucket si_bridge.

TVs:
- (a) surface-DBHi package with T&R (tacking + fluxless formic-acid reflow) sub-assembly and capillary underfill.
  It has a 60 um bridge (3.5x2.5 mm), two 19x19 mm Cu-pillar chips and a 68.5 mm cavity-free laminate with LW/LS 9/12 um.
- (b) the same package with an NCP TCB sub-assembly.
- (c-g) FEM cases 1-5 from Table II (bridge thickness, cavity, C4 height).

Figures opened: page_01..06.png. From source.pdf at 400 dpi: Fig. 9, 14(d), 15(f), 16(c), 17, 18, 19.

Judgment calls
- T&R vs NCP is a different encapsulant in the uC4 layer (CUF vs NCP), so each is a separate TV. The qualitative
  results of Table I are 12 value_text rows per TV.
- uC4 pitch is entered as a 30-75 um range, as stated. Bump diameter and height are not given. The uC4 T&R bonding
  method is "other" (TCB tacking + reflow), with a note.
- FEM relative values are bar reads (+-0.02 to 0.05) with reference_tv_id = case 1. Fig. 17 plots case 3 twice
  (0.95 and about 0.99), which is logged as a conflict. The text says the cavity vs no-cavity uC4 strain differs by
  "less than 10%", but the read gives 0.70 vs 0.62. This is also logged.
- The FEM C4 ic_type is c4 for case 1 and cu_pillar for case 5 (Fig. 13). It is unspecified for cases 2-4.
- Table IV T0: 6 module rows (31 nets each). Modules 2 and 5 are a test-cable issue (entered as pass with a note).
  Module 6 is a genuine bridge-chip-A open (fail).

Not reported: electrical interface numbers (no data rate/energy), uC4/Cu-pillar dimensions for the build, TCB
temperature/force, materials, thermal cycling conditions/results, absolute FEM values.

## ectc51906.2022.00108 - Samsung 85x85 mm MIoS warpage and reliability (ECTC 2022)

Status: distilled. primary_bucket si_interposer (buckets si_interposer; thermal_pdn).

TVs:
- (a) the 85x85 mm MIoS package: 2 logic + 8 HBM on a 2800 mm2 Si interposer, with a ring stiffener.
- (b) the FEM substrate-leg model (Table I).

Figures opened: page_01..05.png. From source.pdf at 450 dpi: Fig. 4, 5, 6 and Table I. Fig. 7 at 300 dpi.

Judgment calls
- The stiffener material/thickness, press-cure temperature, adhesive and reflow-count splits leave the interposer
  and substrate stack unchanged, so they are metrics.variant on TV a, not new TVs. Likewise, the FEM legs are
  variants of TV b.
- Warpage values are the printed bar labels. The chart axes carry no unit; um is assumed from the text (400/250/200 um
  requirements) and noted on every row. The HT temperature is not given.
- "T" in stiffener labels was converted to mm per "Stiff A, 3mmT" (unit conversion logged).
- The text says the 1.4 mm core gives a 15% RT reduction, but Table I RT LEG2 = 0.82 (HT 0.85). The table was
  entered and the mismatch logged.
- The text calls the delamination figure Fig. 9, but its caption reads "Fig. 8" (a duplicate number). The caption
  label is cited with a note.
- The Adhesive A bars (267/134) equal Stiff C 1.5 mm, and Reflow x1 (225/193) equals Stiff B 1 mm 180 C. The builds
  used for these splits are not stated, so nothing was inferred.
- Table III adhesive properties are in normalized "n" units and are entered as such.

Not reported: interposer x/y, die sizes, bump pitches, substrate layer count, stiffener/adhesive materials, absolute
moduli/CTE, HT temperature, TC cycles and sample counts, outcomes of other reliability batches.

## Schema gaps hit
- No `value_min/value_max` in metrics. CTE ranges (3~4, 13~20 ppm) went into stack_layers.properties as text.
- Simulation sweep legs that change the substrate stack (FEM core thickness/material) do not fit rule (f) well. A
  `parent_tv_id`/`sweep_of` field would make variant vs TV explicit.
- Qualitative or relative material properties (Table II "+/++", Table III "n" multiples) have no home except
  metrics value_text or a normalized unit.
- Pass/fail per specification with CpK only (no measured value) needs `unit: CpK`. A dedicated capability field would
  be cleaner.
- Panel-level vs unit-level metrics on one TV (panel warpage vs package CLR) cannot be scoped except by text in `scope`.
