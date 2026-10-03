# DevLog-003-photonics: photonics category (v1.0 model library)

| Field | Value |
|---|---|
| Date | 2026-10-02 |
| Owner | photonics agent (Claude) |
| Outputs | `assets/components/photonics/`, `scripts/assets/photonics/` (library `pl.py`, one `build_<asset>.py` each) |

## Plan

Build 10 assets (spec in agent brief) with a shared category library `pl.py` (mesh accumulator, material table MAT_photonics_*, Geometry-Nodes bump instancer, FAU / MT ferrule builders, section cutting, preview renderer with arbitrary camera shots). Micro-scale assets (cell, array) are built twice from one function: real size (1 um = 1e-6 m) and a scaled variant (identical proportions).

## Checklist

| # | Asset | Status | Notes |
|---|---|---|---|
| 1 | microring_modulator_cell | done (2026-10-02) | real + x10000 + cross-section variants, p_heat driver |
| 2 | microring_array_closeup | done (2026-10-02) | 24 ring materials, 28 slab materials, p_wave_* drivers |
| 3 | pic_die | done (2026-10-02) | 59k tris + 6300 GN-instanced pads |
| 4 | eic_die | done (2026-10-02) | microbump + hybrid-bond variants |
| 5 | pic_eic_stack | done (2026-10-02) | assembled / exploded / cutaway (TSV row, C4 row) / x500 inset |
| 6 | oe_module_cpo | done (2026-10-02) | closed + open variants, 110k tris, hooks lid_top/fiber_exit |
| 7 | oe_module_npo | done (2026-10-02) | HDI board, LGA underside, MT-16 board-edge connector, closed + open |
| 8 | fau_v_groove | done (2026-10-02) | 6 FAU variants, MT-12 ferrule, edge-coupling interface x200 (plain + section) |
| 9 | els_laser_source | done (2026-10-02) | ELSFP (OIF dims) closed/open/front-pigtail + 14-pin butterfly with FC/APC |
| 10 | grating_coupler_closeup | done (2026-10-02, stretch) | x5000 scaled + real, tilted fibre, mode cone |

## Sources (accessed 2026-10-02)

- OIF-ELSFP-01.0 (https://www.oiforum.com/wp-content/uploads/OIF-ELSFP-01.0.pdf): Fig. 7 body length 55.2 mm to latch datum, overall 100.0 mm with pull tab, heat-sink contact 59.29 x 17.03 mm, latch width 21.18 max; Fig. 9 end face 22.58 x 9.3 mm, module PCB interface 21.18 / 18.94 mm; optical classes Table 6; blind-mate dual MT-12 ferrules, PM fibers (accuracy A for envelope).
- Intel OCI (Hot Chips 2024 Fathololoumi, https://hc2024.hotchips.org/...62_HC2024.Intel.Fathololoumi.Final.pdf): 8 fiber pairs x 8 lambda, 64 lanes x 32 Gb/s, SMF-28, ring modulators + Ge PD, V-groove attach; no die dimensions published. Die size estimated from the blog photo `intel-oci-eic-pic-fiber.webp`: PIC about 8 x 8 mm (range 7-9), EIC about 2.1 x 4 mm, scaled from 16 fibers at assumed 250 um pitch (4 mm span) = 57 px/mm. Accuracy C.
- NVIDIA/OFC 2026 M4B.2 (local `20260320_OFC/extracted_text/M4B.2-*`): 7 nm EIC on 65 nm PIC, Cu-Cu hybrid bond, 1.33 Tb/s/mm^2 (256 Gb/s per fiber link), vertical grating couplers 1.3 dB, FAU + organic substrate.
- Marvell COUPE slide (blog `2026/OFC2026/IMG_4055.JPG`): EIC N3P on 65 nm PIC, hybrid bonding, TSV through PIC, C4 bumps to organic substrate, vertical grating couplers, integrated Si lenses.
- TSMC SoIC-X pitch roadmap 9 um (2023) -> 6 um (2025) (Tom's Hardware / TrendForce, search results); COUPE EIC-PIC spacing 5-15 um (search summary, unverified).
- AMF MRM paper PMC10831040: 500 x 220 nm rib, 90 nm slab, gap 180 nm, TiN heater ~500 ohm, 4 implant layers. OFC 2026 Th2A.14: 5 um radius, FSR 19.5 nm. Ranovus HC34 2022: RRM < 50 x 50 um^2, V-groove pitch 250 um, 16-fiber FVGA, GF 45SPCLO.
- MT ferrule: IEC 61754-5 / US Conec: 6.4 x 2.5 mm, 0.7 mm guide holes at 4.6 mm, 250 um pitch (search summary).
- Cheng 2025 NPO/CPO schematic (blog `pluggable-npo-cpo-comparison.webp`), Ranovus/MediaTek photo (`ranovus-optical-engines-on-asic.webp`): module count/arrangement only; OE module size is an estimate (C).

## Decisions

- Ring radius 7.5 um (public range 5-12 um). Ring objects keep origin at ring centre so the assembler can scale rings individually.
- Micro assets ship two variants in one .blend; the scaled variants (x10000 cell, x15000 array) are the ones to use in scenes.
- Dimension table lives in each asset JSON; accuracy levels A/B/C per spec.
- Previews are palette-quantised by `pl.shrink_png` to stay under 400 KB.

## Progress log

- 2026-10-02: library `pl.py` written; microring_modulator_cell built and previews checked (cutaway layers OK).
- 2026-10-02: microring_array_closeup built (149k tris), previews generated, under review.
- 2026-10-02: microring_array_closeup accepted (heat wave driver verified in preview; GC and seal ring clearance fixed).
- 2026-10-02: pic_die and eic_die built; previews checked (hybrid pad texture closeup OK). `dies.py` shared builders.
- 2026-10-02: fau_v_groove built (built early because OE modules reuse `pl.build_fau`); previews checked.
- 2026-10-02: oe_module_cpo and pic_eic_stack built and previews checked (cut now uses world-space plane, fixed parent-offset bug).
- 2026-10-02: oe_module_npo, els_laser_source, grating_coupler_closeup built; finalize_meta.py written and run; triangle budgets trimmed.
- 2026-10-02: run interrupted by usage limits twice; resumed on coordinator instruction.

## Comparison with reference images (differences that remain)

- Intel OCI photo (`intel-oci-eic-pic-fiber.webp`): real PIC has the EIC at the left and the fibre/V-groove strip at the right of a larger square die with pad frames; mine follows that arrangement (EIC site left of the ring banks, edge couplers on +X) but the PIC is 7 x 9 mm per the brief while the photo suggests about 8 x 8 mm plus the strip; waveguides/rings are drawn thicker than true; ring bank and fan-out are schematic.
- Ranovus/MediaTek photo: real OE cover is a flat metal plate with a fibre notch and engraved lettering (matched); real module also has mounting hardware around it and a different size ratio to the ASIC (estimated, C).
- Marvell COUPE slide: layer order matched (support Si, EIC, hybrid bond, PIC with TSV, C4, organic substrate); TSV only in cutaway, vertical grating coupler/Si lens not modelled in the stack.
- Cheng 2025 NPO schematic: NPO modelled as HDI module with connector at the board edge; the real HDI interposer sits under the whole ASIC package.

## Final status (2026-10-02)

All 10 assets built; JSON metadata complete (dimension tables, hooks, materials, variants, evaluated triangles). Triangle counts per file (evaluated incl. GN instances): cell 15k, array 156k, pic_die 264k, eic_die 404k, stack 838k over 3 variants (about 300k each), oe_cpo 156k, oe_npo 227k, fau 16k, els 29k, grating 3k. Category size 54 MB (previews 12 MB). Run `scripts/assets/photonics/finalize_meta.py` after any rebuild to refresh JSON.

## Open questions for Wentao

- OE module real size (Ranovus Odin 8P / MediaTek) is not published; used estimate (24 x 20 mm carrier, 18.5 x 16 mm lid), please correct if you have the photo scale.
- Die size: brief says about 7 x 9 mm; Intel OCI photo suggests about 8 x 8 mm: keep 7 x 9?
- Ring radius default 7.5 um (public range 5-12 um): different value wanted?
- TSMC SoIC-X 9 um / 6 um pitch and COUPE spacing come from secondary web summaries; verify against a primary source before putting on screen.

## Audit corrections

- Fresh-context audit: not run (token budget); the coordinator's audit should check (a) die/OE sizes vs sources, (b) the unverified TSMC pitch numbers, (c) cut-section previews.
- Self-corrections made during build: cut tool used local not world plane (fixed, affects pic_eic_stack/cell cutaways); hybrid-bond variant hid macros; OE modules and eic/pic variants reduced to hybrid face to stay under the triangle budget.
