# b005 batch report (2026-10-18)

Validator: `uv run python scripts/validate_db.py --staging data/_staging/b005` -> `0 errors, 0 warnings`.

| paper_id | status | TV | comp | layers | ic | steps | elec | opt | metrics |
|---|---|---|---|---|---|---|---|---|---|
| iwlpc52010.2020.9375887 | distilled | 2 | 5 | 8 | 4 | 20 | 0 | 0 | 35 |
| ectc51906.2022.00106 | distilled | 3 | 14 | 12 | 4 | 18 | 0 | 0 | 17 |
| total (2 papers rows) | | 5 | 19 | 20 | 8 | 38 | 0 | 0 | 52 |

## iwlpc52010.2020.9375887 - Nepes 600 mm M-Series FOPLP (IWLPC 2020)

Status: distilled. primary_bucket panel (kept).

TVs: a = 600x600 mm M-Series single-die panel (look-ahead build: 4-panel warpage/TTV evaluation, die shift, CSAM);
b = Table 3 segmentation usable-area calculation for a 5x5 mm package (tv_kind simulation_model, layout calculation).

Figures opened: page_01..page_06.png; img_p01_1.png (Table 1); img_p03_2.png; Fig. 5, Fig. 10 and Table 3
re-rendered from source.pdf at 5x zoom in the scratchpad.

Judgment calls
- Fig. 10 table: warpage 277/213/138/249 um, TTV 68/83/74/95 um. These average to the text values 219.25 um and 80 um.
  The text says "4 panels" but the figure labels the columns "Segment No."; variants use the figure label (conflict logged).
  The paper does not say which process step the values come from or what sign convention it uses. Warpage passes the
  <1 mm spec; TTV fails the <50 um spec.
- Table 2 ranges (Si BG 280-700 um, mold 405-815 um) are entered as thickness min/max on one TV, because the paper
  does not map thicknesses to individual panels.
- Table 3 is credited "Source: DECA" but cites no reference, so it is kept, with basis derived. Printed loss values
  4.50%/8.05% disagree with 100 minus the printed %-of-baseline (4.05/8.02). The "per segment" headers disagree with
  values that are per panel. Both are entered as printed and logged in conflicts.
- Stack order is top to bottom from the RDL/ball side: UBM, D2, RDL, D1, mold, wafer fab dielectric, Si, backside
  epoxy. The backside epoxy position comes from the Table 1 schematic.
- Cu stud is entered as ic_type cu_pillar with level other (die to fan-out RDL); the paper also calls them "pillars".
- Process flow is assembled from the Sec. 6-Sided text plus Fig. 10 stage labels (Via1 cure, sputter, RDL, Via2 cure,
  sputter). Backgrind, singulation and backside epoxy are listed without a stated position in the flow.

Not reported: die size/function, package size (TV a), Cu stud/via/RDL dimensions and L/S, all materials by product,
all process conditions, warpage method/temperature/sign, die shift magnitudes, void counts, reliability for this build.

Excluded: Fig. 1 Weibull TC data (6x6 mm WLCSP / 6.25x6.25 mm M-Series, 500 um pitch, SAC405, ref [1]/DECA);
Yole market Figs. 2-4 and conclusion forecasts; Table 1 eWLB/WLCSP columns; Fig. 9 generic CSAM schematic;
Fig. 12 300 mm KOZ example.

Source quality: fine, 7 pages (p.7 references only), not scanned, not truncated.

## ectc51906.2022.00106 - IBM DBHi Si-bridge jet CUF (ECTC 2022)

Status: distilled. primary_bucket si_bridge (kept).

Source defect: the original PDF text layer was font-encoded garbage (common-word ratio 0.008). text.md was later
replaced by tesseract OCR. Every value was read from page renders: source.pdf rendered at 216 dpi in quadrants, plus 7x crops of
Fig. 1(b) and Fig. 6, which is equivalent to figures/page_NN_220dpi.png. The OCR text was checked against these
afterwards. papers.notes records "Text layer garbled; OCR used".

TVs: a = DBHi module with jetted CUF bridge uC4 (new process); b = DBHi module with NCP/TCB reference process
(Fig. 1/Fig. 2a; no results in this paper); c = slit test sample for dot-size measurement (test_coupon).

Figures opened: all 4 pages (quadrant renders); Fig. 1(a)(b), Fig. 2, Tables I-III, Figs. 3-10.

Judgment calls
- Bridge size: text 2.5 x 3.2 mm vs Fig. 1(b) label 2.5 x 3.5 mm; text entered, conflict logged.
- Gap depth 800 um (abstract) vs 725 um in the Fig. 7 proportional drawing. The 725 um arrow spans the chip only, so
  it is entered as processor chip thickness (schematic override) and the conflict is logged.
- Table III: two parts, 0/29 broken nets at every readout to 1000 cycles of -40/125 C. One metric row per part.
  Result is pass, but the paper states no criterion or standard.
- Bridge uC4 is entered as ic_type c4, because the paper says "fine pitch C4"/uC4, at 75 um pitch. Formic-acid furnace
  reflow is entered as bonding_method mass_reflow. The NCP reference uses tcb (no NCP-specific enum value).
- Processor chips are entered as component_type cpu ("processor chip"). The paper does not say whether they are
  daisy-chain emulators (they are a test structure with nets).
- The first trial (jet from the C4 side on a tilted subassembly, which failed and contaminated the C4 and bridge
  backside) is entered as a separate flow on TV a plus a fail metric. It is not a separate TV.
- Flying-dot results (rod ~50 um, dot ~100 um, max 120 um, 7 good shots) go on TV c. The Fig. 6 annotations
  55/60 um are noted only.

Not reported: laminate size/layers, lid/TIM materials, bridge thickness, bump diameters/heights/counts/alloy, all
reflow/TCB/cure conditions, dispense parameters beyond nozzle size, TC standard/criterion, yield number, warpage.

Excluded: Table I and Table II (material properties and Gc, both credited to ref [3]); FEM shear-stress result (ref [3]);
DBHi advantages list (refs [1][2]).

## Schema gaps
- No bonding_method value for fluxless formic-acid reflow or for NCP (paste) TCB. Used mass_reflow and tcb, with notes.
- No interconnect_level for die-to-fan-out-RDL (Cu stud/pillar in chip-first FO). Used other.
- No metrics field for a measurement process stage (warpage after Via1 cure, RDL, ...). It goes in condition/notes.
- Panel-layout calculations (segmentation usable area) have no natural tv_kind. Used simulation_model.
- Dispense-equipment parameters (nozzle ID/OD, dot size) have no dedicated columns. They go in process_steps.conditions.
