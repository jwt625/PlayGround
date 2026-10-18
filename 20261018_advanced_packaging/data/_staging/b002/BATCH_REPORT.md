# BATCH_REPORT b002 (2026-10-18)

Schema v2. Validator: `uv run python scripts/validate_db.py --staging data/_staging/b002` -> 0 errors, 0 warnings.

Totals: papers 3, test_vehicles 5, components 15, stack_layers 60, interconnects 16, process_steps 112,
electrical_interfaces 0, optical_interfaces 1, metrics 58.

## ectc51909.2023.00150 - status: distilled

Source: full text, 6 pages, not truncated, correct paper, text layer present. primary_bucket advanced_substrate kept.

Rows: papers 1, test_vehicles 2, components 9, stack_layers 7, interconnects 6, process_steps 19,
electrical_interfaces 0, optical_interfaces 0, metrics 18.

TVs: a FPGA (29x12 mm, 40 um pitch, >188,000 pads) + HBM2 (55 um) + IBM AI chip (150 um) on 26x34 mm organic
interposer (4 thin-film layers, 2/2 um, ENEPIG) on 55x55 mm 7-2-7 build-up substrate, lid, BGA, evaluation board;
b FPGA-size dummy chip (no device layers, same bump array) used for bonding optimization.

Figures opened: page_01..page_06.png, img_p03_27.png, img_p03_31.png, img_p03_33.png; Fig. 3, 4, 5, 7 re-rendered
from source.pdf at 300-400 dpi in scratch.

Judgment calls:
- Dummy FPGA chip split as TV b (different die set). Paper does not say which figures (Fig. 5, 7, 8 failure
  cross-sections) used the dummy or the real FPGA; they are on TV a with a note.
- Both solder-deposition flows recorded (Process A solder transfer, Process B IMS). Which one built the final FPGA
  joint is not stated (abstract says IMS, conclusion says solder transfer): evidence conflict.
- Fig. 4 profile extremes read off the linear axis (approx): 200 C max 2.7 um (no vacuum) / 4.0 um (vacuum).
- Fig. 5/7 height labels (35, 37, 39 um) entered as geometry/yield metrics with scope "label in cross-section";
  what the arrow spans is not defined by the paper, so not put in interconnects.height_um.
- interposer_area_mm2 884 = 26 x 34 (derived).
- Fig. 12 temperatures (25/250 C) excluded: figure attributed to ref [12].

Not reported: electrical interface numbers; die/interposer/substrate thicknesses; RDL materials and vias; pillar
size, solder alloy, TCB conditions; UF material; interposer-to-substrate joint; BGA pitch; lid material;
reliability; Fig. 3 warpage magnitude (3D plot unreadable).

Source issues: abstract "4+1 7-2-7 layers" vs Table I "7-2-7" (kept 7-2-7, noted).

## ectc51909.2023.00009 - status: distilled

Source: full text, 8 pages, not truncated, correct paper, text layer present. primary_bucket cpo kept from batch
file; the paper is a SiPho interposer process-integration paper (si_interposer also listed). No fiber attach.

Rows: papers 1, test_vehicles 2, components 5, stack_layers 28, interconnects 9, process_steps 68,
electrical_interfaces 0, optical_interfaces 1, metrics 27.

TVs: a POPSTAR SiPho interposer wafer (200 mm SOI; FEOL photonics, TSV middle 12x100 um, 4-level Ti-TiN/AlCu BEOL,
Cu/Ni/Au u-pillars, backside RDL 10/10 um, 40 um thermal cavities, LTC polymer, Cu/Ni/SnAg bumps);
b POPSTAR ONoC system (4 chiplets + 6 E/O dies on the interposer, on BGA), design only, "currently in fabrication".

Figures opened: page_02..page_08.png; Fig. 11, 20, 21, 26, 27 re-rendered at 250-300 dpi in scratch.

Judgment calls:
- TV b kept as a design_target TV so the system architecture (10 dies, face-to-face) is captured; no build data.
- All test structures (spirals, PD, MRR, combs, Kelvin) on TV a (same stack); PN MRR data from "pathfinder lot
  wafers" noted in condition.
- Via 1/2/3 Kelvin medians read off Fig. 26 (Ohm -> mOhm conversion logged); TSV 11/13 um medians off Fig. 27.
- Backside bump pitch 80 um derived as diameter + space (40 + 40).
- u-pillar "CD 20 um and pitch 20 um" entered as written; physically CD = pitch is unusual, flagged in notes.
- SPGC entered in optical_interfaces (grating, 3.27 dB mean, 1308 nm); no fiber details.
- Seed/heater stack thicknesses summed (Ti + TiN, Ti + Cu) as derived.

Not reported: electrical interfaces (10 Gbit/s is a capability statement only); die sizes; implant/anneal
conditions other than TSV 400 C and LTC 230 C / 3 h; fiber coupling; warpage, reliability, thermal metrics.

Source issues:
- "estimated aluminum resistivity of 3.2 Ohm.m" (p.7): unit physically implausible; kept verbatim with a note.
- Text cites "shown in 0" for Fig. 22 (broken cross-reference).
- Abstract TSV "below 22 mOhm" vs text (21 +/- 3) mOhm; result text entered.
- Conclusion "ER in the range of 20 dB for 1.5 V" vs Sec. IV.A 18 dB at 1 V / 27 dB at 2 V; all entered as variants.
- No page_01.png render (page 1 is text only).

## ectc51909.2023.00298 - status: distilled

Source: full text, 5 pages, not truncated, correct paper, text layer present. primary_bucket glass kept.

Rows: papers 1, test_vehicles 1, components 1, stack_layers 25, interconnects 1, process_steps 25,
electrical_interfaces 0, optical_interfaces 0, metrics 13.

TV: a 8ML (M1-M4 on each face) double-side RDL on 300 um, 6x6 in Corning glass (CTE 7.8 ppm): SAP M1-M3 on
ABF GX92 (15 um), aSAP M4 2/2 um on ABF GX92P (5 um), femtosecond-laser microvias (~15 um design, 17 um min
measured), no TGV.

Figures opened: page_02..page_04.png; Fig. 1, 5, 8, 10 re-rendered at 200-400 dpi in scratch.

Judgment calls:
- package_class glass_core_substrate (paper uses both "glass substrate" and "glass core interposer").
- Stack per face: Cu / seed / dielectric for each Mn; ABF thicknesses from Fig. 1 schematic labels (override
  schematic), applied to every GX92 layer. M1 dielectric has no vias (vias M2-M4 only).
- M3 L/S: text target 5/5 entered over Fig. 1 label "7 um L/S" (conflict logged; 7 um is also the DFR thickness).
- Fig. 10 coupon map counted: front 8/10/2, back 8/8/4 (2 um / <=5 um / >5 um) of 20 coupons per face.
- Organization names corrected for obvious print typos ("Packaing", "Intitute"); noted in papers.notes.
- M4 Cu plating step not separately described; not added as a process step (rule 7). ABF cure and desmear entered
  from Fig. 2 flow chart (source schematic).
- carrier_size_mm 152.4x152.4 from 6 in (unit conversion logged).

Not reported: Cu thicknesses, seed thicknesses, via depth; lamination/cure/desmear/bake conditions; glass product;
numeric warpage; reliability (HAST future work).

Source issues: text cites "Figure 2" for Fig. 3 and "Figure 8" for Fig. 10; no page_01/page_05 renders.

## Schema gaps hit

- No column for solder-volume or deposition-method split per pad region (IMS variable volume): only in conditions.
- interconnects has no standoff/joint-height-as-labeled field distinct from Cu height; cross-section labels with
  undefined span went to metrics.
- No metal-thickness vs via-height distinction for BEOL vias with square contacts (via_diameter_um used for square
  width; shape only in notes).
- No column for TSV liner/barrier/seed stack thicknesses separately (dielectric/metallurgy free text).
- No on-wafer photonic device table (MRR Q, modulation efficiency, PD responsivity/dark current): all in metrics.
- optical_interfaces has no wavelength tolerance or n_samples/statistic (SPGC 1308 +/- 8 nm, mean of several wafers).
- No design-only TV kind: an unbuilt target system (POPSTAR ONoC) uses tv_kind package with basis design_target.
- metrics.value_text carries ratios like "8/20"; no numerator/denominator columns for coupon-map yields.
- stack_layers has no way to say a layer dimension applies "per layer" when only one schematic label exists.
