# b013 batch report (2026-10-18)

Validator: `uv run python scripts/validate_db.py --staging data/_staging/b013` -> `0 errors, 0 warnings`.

| paper_id | status | TV | comp | layers | ic | steps | elec | opt | metrics |
|---|---|---|---|---|---|---|---|---|---|
| ectc51909.2023.00091 | distilled | 2 | 15 | 23 | 8 | 0 | 0 | 0 | 10 |
| ectc32696.2021.00315 | distilled | 3 | 8 | 7 | 3 | 8 | 0 | 0 | 32 |
| ectc51909.2023.00274 | distilled | 4 | 9 | 30 | 8 | 30 | 0 | 0 | 21 |
| total (3 papers rows) | | 9 | 32 | 60 | 19 | 38 | 0 | 0 | 63 |

## ectc51909.2023.00091 - TSMC CoWoS-L_6x supercarrier RDL (ECTC 2023)

Status: distilled. primary_bucket si_bridge (kept). package_class si_bridge_fanout (LSI embedded in an RDL interposer).

TVs: a = TV0_1.6x (2 SoC + 4 HBM2E + 8 dummy Si, 6 LSI, lid 91x91x2 mm); b = TV0_4.8x (6 SoC + 12 HBM2E, 18 LSI,
stiffener ring, ~4000 mm2). Both have a 91x91 mm SC-RDL/package, 1 F/S + 6 B/S RDL (B/S 11/11 um Cu), TIV 50 um dia / 100 um
pitch, 45 IPDs (2.9 mm2), uBGA 0.5 mm, BGA 1 mm, and a 3-2-3 substrate with a 1.4 mm core.

Figures opened: page_02.png, page_03.png, page_04.png. Fig. 6 and Fig. 7 were re-rendered at 4x.

Judgment calls
- Warpage: the measured plot labels are entered (64 / -148 um for TV a, 186 / -107 um for TV b). The text values (65, ~150,
  "within 200", "within 120") are logged as conflicts. The sign convention is not stated.
- Fig. 5 insertion loss (SC-RDL vs SBT, 25-30% lower) is credited to ref [6] (SoIS). It is excluded, so there are no
  electrical_interfaces rows.
- TIV is mapped to ic_type tmv (inferred). The uBGA is entered as bga at 500 um pitch (Table 1); the text range is 400-500 um.
  Its height is ~200 um, from the uBGA layer total height.
- The dies are marked is_emulator yes, inferred from "mechanical TV". n_dies is the sum (derived).
- The ring label "18mm-12mm-2mm" is kept as text only, because the paper does not define the three values.
- The ubump pitch "35 um or lower" is a scheme capability, not a TV value. It is in notes only.
- No process flow is disclosed, so process_steps is empty.

Not reported: ubump pitch/size, F/S RDL L/S, dielectric product and thicknesses, die/LSI sizes, mold/TIM, package thickness,
reliability temperatures and preconditioning.

## ectc32696.2021.00315 - Samsung MIoS 85x85 mm (ECTC 2021)

Status: distilled. primary_bucket si_interposer (kept). The paper never uses the I-Cube name. technology_name is MIoS/MIP.

TVs: a = 85x85 mm MIoS package (2 ASIC + 8 HBM Flashbolt, 51x55 mm 110 um Si interposer, >2800 mm2); b = FEM MIP warpage
model; c = FEM EMC/underfill stress model.

Figures opened: page_01-05.png. Fig. 3 (6x), Fig. 5 (4x), Fig. 9 (4x) and Fig. 10 (6x) were re-rendered. Table I was read from the
render, because the circled failure-mode markers are lost in text.md.

Judgment calls
- Splits on one build are entered as metrics.variant: Process A/B joining, backside RDL volume phase1/phase2, and EMC-A/B/C.
- MIP warpage uses the text values: 165 um baseline, 80 um phase1, 50 um phase2. The plot reads (Fig. 5 peak about 155 um,
  Fig. 9 phase2 about -40 um at 240 C) are logged as conflicts. The Fig. 9 legend arrows are ambiguous. Some RT plot reads are
  entered as approx.
- The Cu pillar joint height is the measured joint gap, 72.2 um at the center and 69.5 um at the edge (schematic 0.07 mm).
  The Fig. 3 schematic thicknesses are entered as schematic (0.685 die, 0.035 ubump, 0.11 interposer, 0.07 bump, 0.71 ring,
  3.082 total).
- Table I gives 12 rows (MSL4 + TC 300/700/1000 for each EMC). When both failure modes occur, value is left empty and
  value_text holds "xF/n".
- FEM normalized stress (Fig. 12/13): Fig. 12 top has a readable x-axis and is entered as y ranges. The underfill-thickness
  plots have no x values, so they are qualitative value_text.

Not reported: ubump/Cu pillar pitch and diameter, solder alloy, reflow profile, EMC/underfill products and properties, substrate
layers, ring material, TSV details.

## ectc51909.2023.00274 - UCLA GaN-IF and dielet-side PDN (ECTC 2023)

Status: distilled. primary_bucket thermal_pdn (kept). buckets keep hybrid_bonding from the batch file, although the bond is a
Cu-Cu direct TCB with recessed dielectric (ic_type cu_cu_direct).

Source problem: the text.md text layer is garbled (per-font character shift). source.json says garbled_text_layer false and
there is no TEXT_LAYER_WARNING.md. I OCRed 300 dpi re-renders with tesseract in my scratchpad and checked every number
against the page renders. papers.notes says "OCR text". The OCR misread the abstract "<= 10 um" as "< 10 nm"; the render
was used. Suggest that the curator add a TEXT_LAYER_WARNING and re-extract this paper.

TVs: a = 2x2 mm Si dielets on a GaN-on-Si (GaN-IF) wafer, 10 um pitch Cu-Cu TCB (120 C / 200 N / 10 s tack, then 400 C 1 h
at 137.5 MPa); b = dielet-side PDN with four 100 um 5x5 mm dielets, SU8 TPVs and a 9.85 mm FR-4 PCB; c = the same flow with
200 um dielets (a different stack height, so a separate TV); d = the GaN-IF 3D-IVR concept (simulation_model, author estimates).

Figures opened: page_01, 03, 04, 05, 06, 07 .png. Figs. 6, 9, 10 and 11 were cropped at 4x. All pages were OCRed at 300 dpi.

Judgment calls
- package_class wafer_scale for all TVs. GaN-IF is the component_type interposer (Fig. 2(b) "TSV-less interposer").
- Two-stage TCB: the tack conditions are in the bonding_* columns and the anneal is in anneal_*. The 137.5 MPa anneal
  pressure is in notes and in process_steps.pressure_mpa (there is no interconnect anneal-pressure column).
- The TPV is ic_type other (through-polymer via). TPV and SU8 height = dielet thickness, entered as derived from the stated
  rule. Pad 250x480 um is from the Fig. 9(c) label.
- The "two different samples" with no opens/shorts are assigned one to each dielet thickness (inferred, noted).
- Fig. 11(c) bar values and Fig. 6(a) box ranges are plot reads (approx, in notes).
- The 48-fold current reduction, >2000x PDN saving and >=50 kW per wafer are author estimates on concept TV d.

Not reported: dielet thickness and count (TV a), wafer diameter, pillar/pad diameter and height, dielectric material,
TCB parameters for TV b/c, reflow profiles, solder volume, and resistance for the 200 um build.

## Schema gaps
- No interconnect column for anneal pressure, or for separate tack vs anneal pressure in two-stage TCB.
- interconnect_type has no through-polymer via (TPV) or TIV value (other / tmv used).
- interconnect_level has no generic die_to_wafer value for a non-hybrid Cu-Cu direct bond (die_to_interposer /
  die_to_substrate used).
- There is no structure enum for a PCB or power platform on top of a wafer assembly ("other" used).
- material_class has no compound-semiconductor value for GaN ("other" used).
- Reliability tables with two failure modes per readpoint do not fit one numeric value. value_text is used.
