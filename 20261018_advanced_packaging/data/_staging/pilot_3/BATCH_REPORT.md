# pilot_3 batch report (2026-10-18)

## ectc51906.2022.00180 - Demonstration of Glass-based 3D Package Architectures with Embedded Dies for HPC (ECTC 2022)

Status: distilled. Source full_text, 7 pages (p.7 references only), clean extraction, correct paper.
primary_bucket glass (as batch); buckets glass;panel (100x100 mm glass panel processing in both TVs).

| table | rows |
|---|---|
| papers | 1 |
| test_vehicles | 2 (a = TV-1 3D GPE package 15x15 mm; b = TV-2 embedded glass interposer 50x50 mm) |
| components | 11 |
| stack_layers | 27 |
| interconnects | 13 |
| process_steps | 36 (TV-1 17 incl. die prep; TV-2 19) |
| electrical_interfaces | 4 (pitch/channel only; no electrical data in paper) |
| metrics | 2 |

Figures opened: page_01..page_06.png; high-dpi clips rendered from source.pdf (scratch only) of Fig. 2, 3(a), 3(b),
5(a), 6, 7 (+ 3D inset), 8, 9 (floorplan + cross-section), 11(b), 12(c), 13 inset.

Judgment calls:
- TV-1 package_class embedded_die_substrate (dies in glass cavity + ABF RDL + BGA pads); TV-2 glass_interposer.
- Text/figure conflicts resolved: package 15x15 mm (not 20 mm Fig. 3(a)); SoC emulator 5x10 mm (Fig. 2 10.1 mm noted);
  HBM 775 um (cross-section) over 750 um (schematic); TV-2 glass core 260 um (Fig. 9/13) over 200 um (text). All
  alternates recorded in evidence entries.
- TV-1 D2D pad pitch: text 10/20/30 um vs Fig. 2 20/40 um; conclusion "vias of 20 um pitch" -> pitch_um 20, pitch_min_um 10.
- 96 um annotation in Fig. 7 not entered as pitch (unclear endpoints).
- 10 um ABF bonding layer between cavity glass and bare glass taken from Fig. 5(a) label only.
- DL2 thickness 20 um from Fig. 11(b) "20 um ABF on each side" (as-laminated); DL3 30 um, DL4 25 um, solder mask 20 um from Fig. 13.
- Electrical_interfaces rows created from I/O labels (USR IO, HBM IO, D2D I/O) with pitch only; protocol usr/unspecified.
- Steps with unstated method (Ti/Cu deposition, seed etch, Cu differential etch) categorized `other`.

Not reported: RDL L/S, TCB conditions, underfill material, TV-2 via diameters, bump metallurgy for TV-2 dies, BGA alloy,
package thickness, glass grade/CTE, any electrical/thermal/warpage/reliability/yield data.

Source issues: internal inconsistencies listed above; PR supplier "TOK" in text vs "Hitachi Chemicals" in acknowledgment.

Validator: `0 errors, 0 warnings (data/_staging/pilot_3)`.

## v2 migration (2026-10-18)

Rebuilt all tables with v2 headers from `data/_staging/TEMPLATE/` (column order changed; optical_interfaces.csv added, header only).

| table | v1 rows | v2 rows | change |
|---|---|---|---|
| papers | 1 | 1 | |
| test_vehicles | 2 | 2 | tv_kind package; TV-1 package_class embedded_die_substrate -> glass_embedded_die; TV-2 area override derived |
| components | 11 | 11 | is_emulator/embedded set; cavity 10.4x10.4 (TV-1), 7.5x7.5x110 um (TV-2); gap_um 92 die-to-die (TV-1 dies), 150 cavity-to-die (TV-1 substrate); DAF -> die_attach_film; TV-2 embedded die -> sram + is_emulator; glass supplier AGC/Schott (ack) |
| stack_layers | 27 | 28 | side front/back/core; underfill and solder_mask structures; TV-1 15 um over-die ABF as separate row (feature); TV-2 front/back seq restarts per side; TV-1 Ti role left empty (not stated) |
| interconnects | 13 | 11 | TV-1 measured-diameter row merged into design row via override; 44 um cross-section row merged into pillar row as total_length_um; pillar 10 um Cu height + 20 um SnAg cap_height; pitch_max_um used; via_in_via type |
| process_steps | 36 | 36 | deposition_other, etch_other, laser_marking, die_attach, surface_finish (OSP), desmear, ball_attach; seq restarts per flow_name |
| electrical_interfaces | 4 | 0 | dropped (pad arrays only, no electrical property) |
| optical_interfaces | - | 0 | |
| metrics | 2 | 2 | category other -> geometry |

Basis/source: extracted_from_figure removed; cross-section photo labels (Fig. 7, 13) -> measured + figure_image; drawing labels (Fig. 3(a), 9, 10) -> schematic; panel/die photo labels (Fig. 5(a), 8, 11(b)) -> design_target + figure_image.
Conflict precedence applied: TV-2 glass core 260 um (cross-section) over 200 um (text); HBM 775 over 750 um; TV-1 die gap 92 (measured) over 100 um (design) in components.gap_um. All alternatives in evidence `conflicts`.
Excluded (cited prior work): desmear roughness 300-500 nm [10]; ~100x100 mm package size [8].

Validator: `0 errors, 0 warnings (data/_staging/pilot_3)`.
