# b000 batch report (2026-10-18)

Validator: `uv run python scripts/validate_db.py --staging data/_staging/b000` -> `0 errors, 0 warnings`.

| paper_id | status | TV | comp | layers | ic | steps | elec | opt | metrics |
|---|---|---|---|---|---|---|---|---|---|
| ectc.2019.00095 | distilled | 6 | 12 | 0 | 6 | 11 | 2 | 0 | 36 |
| iedm19573.2019.8993637 | distilled | 2 | 3 | 5 | 7 | 22 | 0 | 0 | 13 |
| itherm45881.2020.9190542 | distilled | 2 | 11 | 7 | 2 | 6 | 0 | 0 | 68 |
| total (3 papers rows) | | 10 | 26 | 12 | 15 | 39 | 2 | 0 | 117 |

## ectc.2019.00095 - TSMC SoIC (ECTC 2019)

Status: distilled. primary_bucket hybrid_bonding (kept).

TVs: a SoIC-embedded InFO_PoP (logic-on-logic); b/c/d Fig. 3 photo builds (2-tier mixed chiplets, 2-tier large
chiplets, 3-tier Logic1-3); e typical 3DIC microbump stack (simulation_model, normalization reference for Figs. 7, 10,
11); f 2D side-by-side flip chip on substrate (Fig. 9 insertion-loss reference).

Figures opened: page_01..page_05.png; Fig. 5, Fig. 6, Fig. 9 cropped and upscaled 3x in scratchpad.

Judgment calls
- Platform paper: almost every benchmark is a normalized ratio from "our analysis" (basis derived, reference_tv_id e).
  The ratios come from the figure labels (0.23X, 16X, 0.08X/0.01X/0.08X, 0.09X, 0.5-0.8X); RC delay 156X and IR drop
  12.5X come from the text.
- SoIC bond pitch: "sub-10 um" (pitch_um 10, lt). Fig. 5 SoIC point reads ~8.7 um, ~1.2E4/mm2 (density approx).
  Consistency check: 0.23 x 40 um baseline = 9.2 um. Not entered as a pitch.
- ic_type cu_cu_hybrid inferred from the Fig. 8 legend "HB"; the paper never states Cu or "hybrid bonding" in text.
- Fig. 6 chain yield: 5 points at 100%, chain sizes read off a log axis (~41 to ~4.9E6 bonds/chain). Basis measured
  (text: "measured high bonding yield"). Assigned to TV a; the chain test structure is not described.
- Fig. 8 (normalized resistance) and Fig. 9 (insertion loss): paper does not say measured or simulated -> unspecified.
  Fig. 9 flip-chip IL ~1.05 dB at 30 GHz, SoIC ~0 dB. Entered as electrical_interfaces with protocol unspecified.
- Reliability: EM/SM/Vbd for TSV, TDV, SoIC bond = pass vs "TSMC internal requirements" (no conditions); Fig. 15
  PCT/HTS/TCC/uHAST and Table 1 package-level (MR, TCC, MSL3, TC-B, uHAST) all pass. HTS written "1000x" in text vs
  T1000 legend; kept as written (unit "x").
- TVs b/c/d have no numbers; kept as separate builds per rule (f).

Not reported: die sizes/thickness/nodes, number of tiers in the InFO_PoP TV, bond metallurgy/dielectric/pad size/
bonding and anneal conditions, TSV and TDV dimensions, any layer stack, absolute R/L/C/energy/thermal resistance,
reliability stress conditions and sample counts, D2D protocol/lane rate.

Excluded: >20 Tbps memory bandwidth (ref [13]); SoIC+ roadmap point; Fig. 5 generic flip-chip point; Fig. 3(d)
microbump stack photos; Fig. 1/Fig. 4 concept drawings.

Source quality: fine, 6 pages, not scanned, not truncated.

## iedm19573.2019.8993637 - Intel Foveros (IEDM 2019)

Status: distilled. primary_bucket microbump_3d (kept).

TVs: a Foveros F2F logic-on-logic stack on package (top die, 22FFL base die with TSV-mid, package); b TSV Kelvin and
comb test structures on a 12-inch wafer (D 7, 9, 11, 13 um).

Figures opened: page_03.png, page_04.png; pages 3-4 re-rendered at 300 dpi from source.pdf; Fig. 11 and Fig. 13
cropped from the 300 dpi renders.

Judgment calls
- Microbump: 50 um pitch, 25 um diameter, Cu pillar + SnAg on both dies, TCB (design_target, text).
- TSV: 9 um target diameter (Cu diameter "over 8 um"), array pitch 15.0-16.0 um (pitch_min/max). Base die ground to
  ~75 um (component thickness and grind step); TSV depth itself not stated, so depth_um left empty.
- TSV R: text mean 30 mOhm at 9 um (measured); Fig. 11 box centers read as medians (~59, ~28.6, ~19, ~12 mOhm for
  7/9/11/13 um) kept as separate median rows; the 30 vs 28.6 difference is logged as a conflict (mean vs median).
- TSV C: text 135 fF at 9 um; Fig. 13 medians ~100, ~136, ~175, ~216 fF.
- TSV diameter split kept in one wafer-level TV (rows per diameter in interconnects and metrics).
- 12-inch wafer mapped to carrier_format wafer_300mm (note says "12-inch as written").
- Three process flows (microbump + F2F bond, TSV-mid, TSV reveal) recorded with flow_name; relative order between
  flows not stated.

Not reported: die sizes, top die node, microbump height/cap/count/bond conditions/yield number, TSV depth/aspect
ratio/liner thicknesses, package substrate and backside bump pitch, reliability conditions, leakage/breakdown values,
any electrical interface.

Source quality: Figs. 2, 3 and 10 are truncated in the source PDF itself (only small fragments are drawn, confirmed by a
300 dpi re-render); text.md pages 3-4 contain captions only. No values were taken from those figures.

## itherm45881.2020.9190542 - MediaTek InFO_oS thermal (ITherm 2020)

Status: distilled. primary_bucket organic_rdl_fanout (kept); buckets add thermal_pdn because the paper is a thermal
characterization study.

TVs: a lidded InFO_oS thermal test vehicle (2 thermal dies 21.78x16.25 mm, InFO die 22.78x34.00 mm with 3 RDL W/S
2/2, 2/2, 5/5 um, C4 130 um, 62.50x62.50 mm substrate); b simulation_model: TTV package plus vapor-chamber heat sink
(Table 7/8, 14 power cases).

Figures opened: page_02..page_06.png; page 5 re-rendered at 300 dpi and Fig. 12 cropped.

Judgment calls
- TIM type (A-D) and BLT (60/140 um) treated as variants of one TTV (metrics.variant), not new TVs: the paper treats
  them as one TTV with different TIM. Stack layer TIM1 holds 60 um nominal, min/max 60/140.
- Icepak results for the TTV (Table 3/5 "Sim") kept on TV a with basis simulated; measured "Test" rows basis measured.
- Table 3 vs Table 5 disagree for the nominally same condition (6.0 W/mK, 60 um, Test 0.031 vs 0.034 C/W); both kept
  with the table named in the variant; logged as conflict.
- TIM letters differ between Table 2 (30 W/mK = C) and Tables 4-5 (30 W/mK = D); coverage variants keyed by thermal
  conductivity.
- Die size 16.26 (text) vs 16.25 mm (Table 1): Table 1 chosen.
- interposer_area_mm2 774.5 derived (22.78 x 34.00).
- Fig. 12 read: normalized ThetaJC B/A 2.05 (sim) and 2.10 (exp); die-to-die dT ~13.7 C (basis unspecified).
- TIM coverage 90% equals the ">90%" spec: result left na.
- Process-step order follows the text; the paper does not give the assembly sequence.

Not reported: die thickness, RDL materials/thicknesses/order, substrate layers, BGA pitch, lid metal and dimensions,
adhesive, TIM product names, test power for Tables 3/5, n_samples, ThetaCA, vapor chamber dimensions, die-to-InFO
joint type, warpage values, any electrical/optical interface.

Source quality: fine, 7 pages, not scanned. Table 4 and Table 5 text extraction interleaved; read from page_04.png.

## Schema gaps hit
- No metal-agnostic hybrid-bond ic_type: SoIC "HB" had to be cu_cu_hybrid although Cu is not stated.
- Normalized-only benchmarks (ratios vs a reference structure) fit metrics with reference_tv_id, but there is no field
  for the ratio direction; direction was put in parameter text ("X / Y") and scope.
- Material/thickness splits of one layer (TIM type, BLT) on one physical TTV: rule (f) is ambiguous (material swap
  changes the stack, but the paper calls it one TTV). A layer-level variant field would help.
- interconnects has no field for array configuration (single, 2x2, hexagon TSV clusters) or redundancy; went to notes.
- process_steps has no field for flow ordering between parallel flows (microbump vs TSV-mid vs reveal).
- Reliability rows with pass/fail only and no conditions: no column for the "internal requirement" owner beyond
  pass_criterion text.
- metrics has no min/max columns; a ratio range (0.5-0.8X) went into value_text with statistic range.
