# b012 batch report (2026-10-18)

Validator: `uv run python scripts/validate_db.py --staging data/_staging/b012` -> `0 errors, 0 warnings`.

| paper_id | status | TV | comp | layers | ic | steps | elec | opt | metrics |
|---|---|---|---|---|---|---|---|---|---|
| ectc51687.2025.00233 | distilled | 2 | 2 | 18 | 3 | 16 | 0 | 0 | 10 |
| eptc53413.2021.9663896 | distilled | 5 | 4 | 8 | 6 | 25 | 0 | 0 | 103 |
| ectc51909.2023.00175 | distilled | 3 | 16 | 19 | 10 | 22 | 0 | 0 | 41 |
| total (3 papers rows) | | 10 | 22 | 45 | 19 | 63 | 0 | 0 | 154 |

None of the three papers has a TEXT_LAYER_WARNING. All text layers were clean (no OCR needed).

## ectc51687.2025.00233 - DuPont/Unimicron DF-PID glass core TGV filling (ECTC 2025)

Status: distilled. primary_bucket is glass (kept).

TVs: a = DF-PID TGV filling test coupon (0.4 mm glass, 100 um metal-lined TGVs, one DF6000 20 um lamination);
b = 77x77 mm glass-core TV, 3 RDL + 3 DF-PID per side, conformal Cu TGV filled with DF-PID.

Figures opened: page_02.png (Table I, Table II), page_03.png (Figs. 1-3), page_04.png (Fig. 4, Fig. 5, Table III).
Fig. 2 and Fig. 3(a) were re-rendered at 5x in the scratchpad.

Judgment calls
- Layer count: p.2 says "two layer RDL structures ... on both side". p.4, the abstract and Fig. 3(a) (M1-M6) say 3 per side.
  3 per side is entered and the conflict is logged. The third DF-PID layer per side (at M1/M6) is inferred from
  "six layers of DF-PID" and the Fig. 3(a) drawing, and noted.
- M3/M4 sit on the glass surface and connect through the conformal Cu TGV liner (Fig. 3(a) schematic). AP9000C is placed
  between the glass/M3 and the first DF-PID. Back-side application is inferred from the double-side build and noted.
- The comb 9/12 um L/S cannot be assigned to an RDL layer, so it is a geometry metric (design_target), not a stack_layers cell.
- The daisy-chain via (30 um "drill size", 150 um pitch, chains of 50/100/200 vias) is one rdl_via row. The layer pair is not stated.
- TGV aspect ratio 4 = 400/100 (own arithmetic, override derived).
- The bHAST/HTS test structures (comb vs daisy chain) are not assigned in the paper. Noted, not asserted.
- "Good co-planarity" is entered as a qualitative warpage row (basis stated).

Not reported: glass type, TGV pitch/count and formation, per-layer thicknesses and L/S, lamination conditions, dose,
plasma settings, seed thickness, reliability n, measured resistance/R-shift values, warpage numbers.

## eptc53413.2021.9663896 - Deca/ASE M-Series and Adaptive Patterning to 600 mm (EPTC 2021)

Status: distilled. primary_bucket is panel (kept); organic_rdl_fanout is added in buckets.

TVs: a = 600 mm panel die-shift build at ASE (4.5x5.5 mm body; four 300 mm segments; ten panels, one plotted);
b = M-Series Gen 1 fully molded FO-WLP (generic Fig. 2 flow; custom PMIC cross-section Fig. 20: 30 um Cu stud,
15 um via); c = Gen 2 design for 20 um pitch (12 um stud, 6 um via, 3 um enclosure; design only); d = Gen 2 LDI demo
coupon with ADTEC (2 um L/S in positive-tone photoresist, 5 um vias in positive-tone polyimide; measured SEM);
e = simulation_model for format efficiency (Fig. 7), multi-die 3 sigma prediction (Table 2) and the large-die
displacement curves (Fig. 17).

Figures opened: page_01.png to page_07.png (all). Figs. 12/13 (fit labels) and Fig. 17 were re-rendered at 4-5x.

Judgment calls
- Die shift and rotation statistics are category geometry (placement accuracy). Table 1 is entered in full: per-segment
  slope/intercept X/Y, mean and sigma before and after compensation, averages, and the summary mean + 3 sigma (59 Table 1 rows; 73 metric rows on TV a in total).
- Y mean + 3 sigma before compensation: the text gives 27.5 um, but the Table 1 averages give 29.95 um (own arithmetic).
  The text value is entered and the conflict is logged. The X value (29.5) is consistent with the table.
- The Table 2 rotation columns (SR, 3SR) sit under a "(um)" header. The unit is entered as deg, flagged in notes.
- Fig. 17 values: the text "up to 32 um" at 36 mm is entered. The 45 mm end points are plot reads (approx). The 5 mm reads are in notes.
- Fig. 7: only the text-stated 36 mm values (92%/256 vs 73%/40, 6.4x) are numeric. The curve trend is qualitative value_text.
- The 300 mm production format is entered as wafer_300mm ("300mm round format" in the abstract; "SEMI standard 300 mm panel" on p.1).
- The Cu stud is entered as ic_type cu_pillar, level other (there is no Cu-stud enum). The Gen 2 LDI overlay <1 um is in alignment_um (lt).
- Fig. 18/19 (TSMC InFO, Apple A14, Prismark teardown ref [5]) are excluded.

Not reported: die sizes/thickness, EMC and polymer products/thicknesses, Gen 1 RDL L/S, stud height/pitch, ball
dimensions, LDI tool models, process conditions, electrical yield, warpage, reliability.

## ectc51909.2023.00175 - Amkor S-Connect chip-last bridge modules (ECTC 2023)

Status: distilled. primary_bucket is si_bridge (kept).

TVs: a = mTV1 1x reticle module (32.8x25.6 mm, 1 dummy ASIC 23x24, 2 dummy HBM2 8x12, 2 connect dies 7x6, on a
67x67 mm HDBU substrate); b = mTV2 3x reticle module (53.5x45.4 mm, 2 ASIC 24x26, 8 HBM 11x10, 10 connect dies
6x6 / 3x8, 85x85 mm, lidded); c = ANSYS APDL 16.1 warpage model of mTV1 (EMC legs 1-9, top die 450/550/650 um).

Figures opened: page_02.png (Figs. 1-2, Table 1), page_03.png (Tables 2-3, Figs. 3-5), page_04.png (Figs. 6-8),
page_05.png (Fig. 9, Table 4). Figs. 3, 4, 5, 6 and 9 were re-rendered at 4-5x.

Judgment calls
- Module size: the abstract gives 32.8x25.6 and 53.5x45.4. Table 1 gives 33x26 and 54x46 ("about"). Fig. 9 gives 840 and
  2400 mm2 (2.9X). Sec. I gives 850-2500 mm2. The precise abstract dimensions and the printed Fig. 9 areas are entered,
  with all alternatives logged. The module goes in interposer_x/y (S-Connect module = HDFO module).
- Warpage is only in a.u. or normalized. The Fig. 9 shadow moire 1.2x/0.9x (mTV1) and 3.2x/2.7x (mTV2) are entered with
  "normalization reference not stated". The simulated ratios mTV2/mTV1 3.4x/3.7x (text) are separate rows with reference_tv_id.
  The a.u. axes on Figs. 3/4 are numeric, so the plot reads are entered as approx values. The sign convention is from the
  Fig. 3 legend (+ crying, - smile).
- Abstract "<100 um warpage" does not name the module; entered on both mTV1 and mTV2 with a note.
- The Table 4 reliability (MRTL4 30 C/60% RH 96 h + 3x 245 C reflow; uHAST 130 C/85% RH 96 h; TCB -55/125 C 1000 cycles;
  HTS 150 C 1000 h; all Passed) is on mTV2. n is "fewer than JEDEC" (no number). The pass criterion is not stated.
- Tall Cu pillar 135 um height and 260 um pitch, connect die ~70 um and Cu stud ~5 um are mTV1 cross-section values (measured).
  The pillar is entered as a cu_pillar through_via. The paper names TMV only as a concept in the introduction.
- The process flow (Fig. 2) is entered on both mTV1 and mTV2 (same flow). Carrier removal is not described and is not added.
- The EMC DOE ranges (Tg 145-175 C, CTE 7-12 / 22-46 ppm/C, modulus 19/1.6 GPa) are on the model TV c layers. Table 2 gives
  only qualitative per-EMC levels. Text "22 C to 46 C" for CTE > Tg is read as ppm/C.

Not reported: build top-die thickness, which EMC was used in the builds, RDL details, ubump/C4 dimensions and bonding
conditions, IPD size/count, pillar diameter, substrate stack, lid material, absolute warpage, reliability n.

## Source issues
- ectc51909.2023.00175: module dimensions are inconsistent across the abstract, Table 1, Fig. 9 and the introduction.
  The text has the typo "22 C to 46 C" for CTE.
- eptc53413.2021.9663896: the Y mean + 3 sigma before compensation in the text (27.5) does not match the Table 1 averages.
- ectc51687.2025.00233: the RDL layer count per side differs between p.2 (two) and p.4/abstract/Fig. 3 (three).

## Schema gaps
- There is no home for a test-structure feature whose RDL layer is not stated (comb L/S): stack_layers needs a seq. It was entered as a geometry metric.
- There is no Cu stud / through-mold Cu post interconnect_type. cu_pillar (level other / through_via) was used.
- There is no field for via enclosure / capture-pad overlap (Deca 7.5/3 um). It went into metrics and notes.
- There is no metric_category for placement accuracy / die shift. geometry was used.
- There is no field for a normalization reference of normalized warpage. It went into scope.
