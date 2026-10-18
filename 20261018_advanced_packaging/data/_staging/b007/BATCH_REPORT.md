# b007 batch report (2026-10-18)

Validator: `uv run python scripts/validate_db.py --staging data/_staging/b007` -> `0 errors, 0 warnings (data/_staging/b007)`.

| table | ectc.2013.6575597 | ectc51906.2022.00303 | total |
|---|---|---|---|
| papers | 1 | 1 | 2 |
| test_vehicles | 4 | 5 | 9 |
| components | 12 | 13 | 25 |
| stack_layers | 5 | 20 | 25 |
| interconnects | 5 | 9 | 14 |
| process_steps | 18 | 25 | 43 |
| electrical_interfaces | 0 | 0 | 0 |
| optical_interfaces | 0 | 0 | 0 |
| metrics | 37 | 21 | 58 |

## ectc.2013.6575597 - Reliability characterization of CoWoS 3D IC integration technology (ECTC 2013, TSMC)

Status: distilled. primary_bucket si_interposer (kept). Clean text layer, 6 pages, not truncated.

TVs: a CoWoS component-level TV (28 nm 24x14 mm logic + two 40 nm 12x7 mm HBM-mimic dies on 26x24 mm TSV interposer,
40x40 mm organic FCBGA); b same component on IPC9701 8-layer 120x100x2.35 mm PWB (board TC, mechanical shock);
c power-cycling TTV (heater/sensor top dies, C4 underfill UF-1 vs UF-2); d CoW wafer (wafer-level TC, Kelvin resistance).

Figures opened: page_02 to page_06 (Figs. 1-17, Tables 1-2); Fig. 13 re-rendered at 300 dpi (scratch, not kept).

Judgment calls
- Interposer area 624 mm2 = 26x24 (derived); abstract 600 mm2 logged as conflict.
- Counts 270,000 ubumps / 8,700 C4 (abstract) entered on interconnect rows; 1024 I/O ubumps per D die in notes.
- TSV aspect ratio 10 derived (100/10). Interposer thickness 100 um = backside thinning target.
- Table 2 service-condition column repeats Table 1 labels (T/C-C 200X...), a source error; text says conditions F, A, D;
  only 200G/1.5ms = service D written as standard suffix (text states it). Others not mapped.
- Shock results: 9 rows (3 severities x 3 chains); no-failure = 30 with qualifier gt; 900G BGA = 1 read from Fig. 16.
- Power cycling (Fig. 13) has no axis values: entered rank of earliest failure per profile (own reading, value:derived)
  plus plotted point counts in value_text. TSV EM "one order MTTF improvement" entered as 10x approx.
- Separate TVs for board-level (adds PWB) and TTV (different top dies); CoW wafer as wafer_level TV.

Not reported: electrical/optical interfaces, die thicknesses, ubump diameter/height/solder, bond conditions, UBM stack,
substrate layer count, package thickness, EM numeric MTTF/current density, sample sizes for component/board tests,
wafer-level TC temperature range, warpage, thermal resistance.

## ectc51906.2022.00303 - TSV-less Power Delivery for Wafer-scale Assemblies and Interposers (ECTC 2022, UCLA CHIPS)

Status: distilled. primary_bucket thermal_pdn (kept); buckets add si_interposer (Si-IF). Clean text layer, 6 pages.

TVs: a DC PDN unit-cell simulation (dielet-side TPV vs substrate-side TSV; M1-M4 3 um, M1/M4 12/5 um, TPV 100 um dia /
200 um pitch); b analytic solder current-limit inter-die spacing model; c envisioned wafer-scale Si-IF assembly
(concept, Fig. 1/2/5 labels and proposed flow); d thin surrogate coupon (100 um mesas in 525 um Si, SU-8 200 um, Cu
~280 um bottom-up ECD); e thick-dielet coupon (500 um Si fusion-bonded, 3 nm ALD Al2O3, 500 um SU-8 TPVs unfilled).

Figures opened: page_02 to page_06 (Figs. 1-9); Fig. 1(a), 4, 7, 8 re-rendered at 300-450 dpi (scratch, not kept).

Judgment calls
- TPV has no interconnect_type enum; entered as `other` with notes.
- Fig. 7 cross-section labels (Cu 277, SU-8 203, Si 410, step 123, Cu 239, widths 1175/703, wafer 525 um) entered as
  measured; text design values (100 um mesa, 200 um SU-8, ~280 um Cu) logged as conflicts. Conclusion "100 um thick" Cu
  and abstract "500 um width TPVs" not reconciled with Fig. 7; listed in conflicts.
- Fig. 3(b) dielet-side normalized resistance ~26 at 20 mm and Fig. 4(b) ~1.13 mm spacing at 10 mm read off plots (approx).
- Envisioned assembly (TV c) recorded as tv_kind other with schematic basis (Fig. 1a: CPP <4 mm, dielet ~200 um,
  RDL <30 um, support >500 um, phi 100-300 mm); proposed flow steps stated, not performed.
- Fusion bond "room temperature" not converted to a number; anneal conditions not given.
- PDN results put in metrics (power_delivery), not electrical_interfaces (no protocol/link property).

Not reported: electrical/optical interfaces, Si-IF metal/dielectric materials, TSV dimensions for the reference model,
fusion anneal conditions, oxide thickness, seed thickness, plating chemistry/current, SU-8 cure, thermal-cycle count,
surrogate wafer diameter, IR drop values.

## Schema gaps
- No interconnect_type for through-polymer via (TPV); used `other`.
- No column for wire/line pitch when only pitch is given (Cu damascene <1 um, Al RDL 5 um); put in feature/notes.
- No capacitance-density field for MIM/decap layers (2.0 fF/um2 in notes).
- No field for mechanical-shock severity (G, ms); kept in variant.
- No way to mark a source-table labelling error other than evidence conflicts/notes.
- Ranking-only plot results (no axis values) forced into value with unit "rank".
