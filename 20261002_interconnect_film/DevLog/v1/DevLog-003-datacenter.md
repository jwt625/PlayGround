# DevLog-003-datacenter: racks, trays, rack interior, fiber tray, data hall

| Field | Value |
|---|---|
| Date | 2026-10-02 |
| Category | `datacenter` (assets/components/datacenter, scripts/assets/datacenter) |
| Spec | assets/ASSET_SPEC.md |

## Plan and checklist

- [x] tray_compute (cutaway default, lid collection) - 2026-10-02
- [x] tray_nvlink_switch - 2026-10-02
- [x] rack_nvl72_style (variants by collection toggles) - 2026-10-02
- [x] rack_interior_tray_stack (stylized + real; 96 retimer hooks each at N=8) - 2026-10-02, previews reviewed (interior camera views OK, backplane plain)
- [x] tray_fiber_pullout (p_slide drivers drive drawer, middle slides, CMA links) - 2026-10-02
- [x] rack_pair_for_cable_gag (p_gap drivers; 28 port hooks) - 2026-10-02
- [x] datahall_environment (L=20, 32 racks/row, 6 cut tiles, drivers p_door_a/b) - 2026-10-02
- [x] self-check (headless open of all 8 blends: no external images, one ROOT each, identity mesh scales except driven ladder rails), previews recompressed to 3.7 MB total - 2026-10-02
- [ ] fresh-context audit subagent (not run: coordinator asked for token economy)

## Sources (accessed 2026-10-02)

- Rack 600 x 1068 x 2236 mm (+132 mm optional extension), ~1.36 t, 18 compute + 9 switch trays, 8 x 33 kW shelves: Supermicro GB200 NVL72 page (https://www.supermicro.com/en/products/system/gpu/48u/srs-gb200-nvl72), OCP/Cheval ORV3 MGX listing (page 403, figures via search summary).
- 1RU compute/switch trays, hybrid liquid/air cooling: NVIDIA multi-node tuning guide (docs.nvidia.com/multi-node-nvlink-systems/...); SemiAnalysis GB200 hardware architecture (compute tray 1U, 2 Bianca boards, 5184 copper cables).
- Busbar, blind-mate liquid nozzles, side manifolds: ServeTheHome LITEON NVL72 rack article.
- EIA-310 1U = 44.45 mm; ORV3 OpenU 48 mm, 21 in (537 mm) opening: standards knowledge (spec PDFs not re-read).
- Not used: NVIDIA DSX content pack (license).

## Decisions and deviations

- Public sources say "48U" and 2236 mm; 48 x 44.45 mm does not fit in 2236 mm with a base frame. Modelled 42 OpenU (48 mm) slots between a 160 mm base and the top frame; trays 43 mm high; seven blank OU between the upper compute group and the top power shelves (spare height unexplained by sources). Order of trays (4 PS, 10 compute, 9 switch, 8 compute, ..., 4 PS) is from recollection of public photos: C.
- Interior/tray internals are generic: accuracy C. No logos.
- Variants are collection toggles (recipes in the JSON `variants`).
- Meshes shared via linked data; bbox ignores hidden variant collections (curve bound boxes of hidden objects are unreliable in headless).

## Progress log

- 2026-10-02: researched dimensions; built dc_common/dc_parts/dc_rack helpers, tray_compute, tray_nvlink_switch, rack_nvl72_style.
- 2026-10-02: built rack_interior_tray_stack (stylized s=4, pitch 0.7, depth 1.6; real s=1, pitch 44.45 mm).

- 2026-10-02: tray_fiber_pullout built; CMA solved analytically with chained custom-property drivers (no Python needed); previews checked (closed/open).

- 2026-10-02: datahall_environment built; fixed tile-grid alignment (hall half length snapped to tile multiple), safety line moved off the cut tiles.
- 2026-10-02: all previews palette-quantized (magick, 128 colours) to fit the 25 MB category budget; originals not kept.

## Assets, accuracy and files

| asset | blend | size mm (open/default state) | accuracy |
|---|---|---|---|
| tray_compute | tray_compute.blend | 526 x 875 x 44 | C (envelope B) |
| tray_nvlink_switch | tray_nvlink_switch.blend | 528 x 875 x 44 | C |
| rack_nvl72_style | rack_nvl72_style.blend | 616 x 1078 x 2236 | B (envelope), C (layout, internals) |
| rack_interior_tray_stack (stylized, s=4, pitch 0.7 m) | rack_interior_tray_stack.blend | 2300 x 1750 x 5332 | C |
| rack_interior_tray_stack_real (s=1, pitch 44.45 mm) | rack_interior_tray_stack_real.blend | 575 x 858 x 419 | C (pitch A) |
| tray_fiber_pullout | tray_fiber_pullout.blend | 600 x 1300 (open) x 1500 | C |
| rack_pair_for_cable_gag | rack_pair_for_cable_gag.blend | 2600 x 1086 x 2327 at p_gap=2 | C (rack envelope B) |
| datahall_environment | datahall_environment.blend | 25300 x 8500 x 4150 | B/C |

Differences vs public data (checked against numbers only; no licensed imagery was used and no photo-overlay comparison was possible): tray order and spare OU, tray depth, switch ASIC count (sources say 4 per tray, 2 modelled), compute-tray internals, connector count per tray, front door (real NVL72 often ships without), no rack-level CDU/PDU/rear cable cartridges (empty bay frames only).

## Open questions for Wentao

- Is the NVL72 tray order (10 compute / 9 switch / 8 compute) and spare OU layout what you want? Provide a photo if different.

## Audit corrections

- Self-check only (no subagent audit). Known gaps: rack_interior_tray_stack backplane plain; cable routing behind the backplane not modelled (conduits and bundles on the front face only); hose paths schematic.
