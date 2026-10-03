# DevLog-003-packaging: packaging category (v1.0 model library)

Owner: packaging agent. Contract: assets/ASSET_SPEC.md. Helpers: scripts/assets/packaging/pk.py (mesh builder, GN instancing, materials, previews), hbm.py, parts.py.

## Checklist
- [x] hbm_stack (16-Hi/12-Hi/molded/cutaway) built, previews viewed (2026-10-02)
- [x] xpu_package_rubin_style: built, previews checked (lid frame / lid removed / lid plate / BGA balls / cutaway with p_z_exaggeration), 16 OE hooks HOOK_oe_L_0..7 / HOOK_oe_R_0..7
- [x] bga_lga_family: 9 packages (3 FCBGA 1.0, 3 FBGA 0.8, 3 LGA 1.0), counts 260-1676, previews checked
- [x] narrowcom_retimer_chip: built (bare, mounted, heat spreader, glow via p_glow), previews checked
- [x] passives_library: 33 parts (MLCC 0201-1206, resistors, ferrite/chip/power inductors 7x7 and 10x10, tantalum A-D, polymer, alu can, toroid, flyback, DrMOS 5x6, QFN controller, B2B connectors, 2x3 power header, test points, M3 screw/standoff/washer, LEDs), previews checked
- [x] hgx_baseboard: HGX variant (553 x 416, 17 holes, 19 edge connectors, 8 OAM modules + heatsinks, 4 NVSwitch, 8 retimers, VRM clusters, passives) and S4 variant (260 x 200 board, socket frame, 8 VRM clusters, passives, connectors, bolts, appended XPU package, hidden); p_density driver verified
- [x] pcb_generator: pcb_gen.py (PCB class) + demo board + tray board 440x350x2.4 (4 connector rows, 12 retimer footprints, hooks), previews checked

## Sources used so far
- HGX baseboard OCP contribution R1 V0.1: board 553.0 x 416.0 x 153.2 mm (565.4 with connectors), 17 captive screws and connector A1 coordinates (Fig 14), heatsink envelope 145.54 mm above PCB top, stiffener bottom 7.59 mm above PCB top.
- OAM spec v1.0: module 102 x 165 mm, connector pitch 102 mm, mount holes 90 x 102, 103 mm module pitch, 33.8 mm baffle gap, 8 modules = 2 rows x 4, 415 mm row width.
- TSMC CoWoS-L public numbers (5.5x: 100x100 mm; 9.5x: 120x150 mm), HBM4 775 um height, see hbm_stack.json.

## Progress log
- 2026-10-02: pk.py/hbm.py/parts.py written; hbm_stack built and visually checked (exposed/molded/cutaway; cutaway has p_z_exaggeration driver).
- 2026-10-02: build_xpu_package_rubin_style.py written (not yet run).

- 2026-10-02: xpu_package_rubin_style built (259k tris default variant, 13.7k balls in hidden variant); lid frame matches the slide; cutaway checked.
- 2026-10-02: bga_lga_family built (nine ASSET_ collections in one .blend, roots 60 mm apart); underside, ball/land close-ups viewed.
- 2026-10-02: narrowcom_retimer_chip built (260 balls, 15x15 mm, valley mark + wordmark + 4 rows).
- 2026-10-02: passives_library built (20.9k tris, 33 parts, display grid); sheet viewed.
- 2026-10-02: pcb_generator built (pcb_gen.py, 120x80 demo + 440x350 tray; 12 diff pairs / 64 diff pairs). NEXT: hgx_baseboard (S4 variant first) with p_density via NG_pk_scatter_density.

## Decisions
- XPU substrate widened to 180 x 100 mm so 8+8 OE sites fit outside the lid frame (matches S4 crude layout ratio).
- Cutaway variants are real-scale with a driver-scaled Z exaggeration empty.

## Open questions for Wentao
- OE footprint (14 x 10 mm at 11.5 mm pitch) is an estimate; align with the photonics OE asset.

## Audit corrections
(none yet)

- 2026-10-02: hgx_baseboard built (HGX 421k tris incl. instances, S4 152k); density driver needs pk.refresh_drivers() after changing p_density from Python; previews palette-quantized (shrink_previews.py) to 23 MB total.
- 2026-10-02: all seven assets saved; blends open headless with no libraries or missing images (chk). Category size 76 MB.

## Final status (exact)
Done: hbm_stack, xpu_package_rubin_style, bga_lga_family, narrowcom_retimer_chip, passives_library, pcb_generator, hgx_baseboard (both variants).
Partial / simplified: no fresh-context audit run (budget); only one contact sheet viewed per asset (not every preview); the retimer marking text was not re-viewed after the last tweak; pcb_generator tray close-up views 2 of 4 point at empty board areas (cosmetic); HGX connector type-to-position assignment is illustrative.
Not started: none of the listed assets. Not done: OAM grounding pads, OAM ASIC/HBM details, true heatsink fin geometry, PCB inner layers.

## Sources
- references/Open-Compute-Specification-HGX-Baseboard-Contribution R1 V0.1.pdf: outline, hole and connector coordinates, heatsink envelope, connector table.
- references/OAM Spec v1.0.zip (extracted to scratch only): module 102 x 165, KOZ, stack height, row layout (Fig. 54).
- references/rubin-ultra.png: XPU layout.
- https://www.techpowerup.com/336064/ and https://www.trendforce.com/news/2025/04/24/ (CoWoS-L 5.5x/9.5x reticle, 100x100 and 120x150 mm substrates); https://www.tomshardware.com/tech-industry/semiconductors/nvidia-enterprise-roadmap-rubin-rubin-ultra-feynman-and-silicon-photonics (Rubin Ultra 4 dies / 16 HBM4E); HBM4 775 um: https://www.techpowerup.com/320314/; HBM4 die thickness: patsnap; microbump 55 um: FormFactor SWTW2016; ball diameters: NXP AN10778, TI MicroStar guide, Analog Devices MO-275 outlines; blog image ieee-400g-package-electrical-path.jpg (cutaway layer order). Standard chip/tantalum/ISO 7045 sizes from memory (flagged in the JSON).

## Deviations
- XPU substrate 180 x 100 mm (wider than the public 100x100 / 120x150 numbers) to place 8+8 OE sites outside the lid; documented range.
- Interposer extends under the HBMs (physical) while the slide shows a teal band only; lid frame ring is the default (as in the slide) with a lid-plate variant.
- bga_lga_family: nine ASSET_ collections in one .blend (roots 60 mm apart); LGA uses 1.0 mm pitch lands (no 1.85 mm variant).
- hgx_baseboard.blend contains a nested copy of ASSET_xpu_package_rubin_style (appended, hidden variant VARIANT_s4_xpu_installed).

## Open questions for Wentao
1. Confirm the XPU package size (180 x 100 mm) and the OE site footprint (14 x 10 mm at 11.5 mm pitch) against the photonics OE asset.
2. Retimer part-number and marking rows are invented generic text; fine?
3. HGX layout (rows at +94 / -191 mm, NVSwitch band) is our own placement within the spec outline.

## Audit corrections
(none; audit not run)
