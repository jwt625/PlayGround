# DevLog-004 (S2 new assets): dummy board components, tray dressing layouts, NVL72 backplane cartridge band

Date: 2026-10-02. Owner: S2 agent of the v1.1 feedback wave (DevLog/DevLog-004-v1_1-feedback-wave.md). Scene use: `scripts/film_v1/s02_retimers.py` (rack-interior camera rise, local t 0.8-5.4 s).
All files are NEW (no existing asset was edited). Real-size features in mm (1 unit = 1 m); the rack interior applies the detail scale 4 (rack_interior_tray_stack p_detail_scale) at assembly. No logos, no text on any model.

## Checklist
- [x] references: component inventory from a primary source (Lenovo GB300 NVL72 user guide, text extracted once) and one press article for Helios; sidecar `references/datacenter/SOURCES_s02_dummy_parts.json`
- [x] `dc_board_parts.blend` (+ .json, 2 previews): 14 part assets (family)
- [x] `tray_dummy_layouts.blend` (+ .json, 6 previews): 3 tray dressing layouts (GB300 compute, NVLink switch, Helios compute)
- [x] `nvl72_backplane_cartridges.blend` (+ .json, 2 previews): cartridge band for the backplane
- [ ] fresh-context audit not run; previews were viewed in contact form only (the part grid is small; parts were checked in scene stills)
- [ ] assets/INDEX.md not regenerated (not owned by this agent): run `scripts/assets/make_index.py`

## Sources (accessed 2026-10-02)
- Lenovo NVIDIA GB300 NVL72 User Guide (https://pubs.lenovo.com/gb300-nvl72/gb300-nvl72_user_guide.pdf, PDF fetched once for text only, SHA-256 7d5ed603...341d9b): compute tray top view lists E1.S drive backplane, OSFP card, BMC card, TPM card, HMC card, CMOS battery (CR2032), two compute boards with cold plates (primary/secondary), busbar cable, M.2 drive, fans, power distribution board, BlueField-3 B3240 DPU adapter; front: 8 E1.S bays, 4 x 800G OSFP, 2 x QSFP; 960 GB on-board LPDDR5x; two liquid loops; "eight 40 mm x 56 mm dual-rotor fans". NVLink switch tray rear: coolant return, cable cartridge connectors, coolant supply, busbar connector; four hot-swap fan modules (N+1). Rack 600 x 1068 x 2236 mm.
- ServeTheHome, AMD Helios MI450 rack at OCP Summit (https://www.servethehome.com/amd-helios-mi450-rack-at-ocp-summit-with-a-different-version-from-meta/): two OCP NIC 3.0 slots (one populated), E1.S SSDs on both sides, fully liquid cooled, cable cartridges at the rear. Text only.
- Web search summaries (wccftech, igorslab, nextbigfuture, The Register; Lenovo Press GB300 guide): 4 x MI455X + 1 EPYC Venice per Helios compute tray; 2 GB300 boards (2 Grace + 4 B300) per GB300 tray. Inventory counts only.
- Not obtainable as primary material this session: board photographs and dimensions of GB300 / NVLink switch / Helios trays (vendor PDFs give lists, not layouts). No reference images were downloaded; the existing project references (`references/`, blog image folder) contain no GB300 / Helios tray photos (searched by name). Therefore every layout is accuracy C: the component INVENTORY follows the public lists, the positions and sizes are plausible generic.

## Assets
| id | file | size mm (real) | tris | accuracy |
|---|---|---|---|---|
| dc_board_parts (14 ASSET_ parts, listed below) | assets/components/datacenter/dc_board_parts.blend | per part | 104-1872 each | per part |
| tray_dummy_gb300_compute | tray_dummy_layouts.blend | 533 x 401 x 37 | 10568 | C |
| tray_dummy_nvlink_switch | tray_dummy_layouts.blend | 533 x 397 x 25 | 9632 | C |
| tray_dummy_helios_compute | tray_dummy_layouts.blend | 533 x 401 x 37 | 10828 | C |
| nvl72_backplane_cartridges | nvl72_backplane_cartridges.blend | 428 x 17.5 x 112 (4 modules) | 6448 | C |

Total new asset size about 5.8 MB (blend + json + 10 previews 4.1 MB). Build scripts: `scripts/assets/datacenter/dc_board_parts_lib.py` (shared part builders), `build_dc_board_parts.py`, `build_tray_dummy_layouts.py`, `build_nvl72_backplane_cartridges.py`; run each with `Blender -b --python <script> -- assets/components/datacenter`.

Parts (size mm, accuracy; full provenance in dc_board_parts.json `parts[*].provenance`): vrm_block 54 x 50 x 7 (C, estimate range 40-90 x 30-70); inductor_bank 95 x 14 x 8 (C); cap_bank 56 x 27 x 11 (C; 8 mm cans, range 6-10); heatsink_fin_stack 66 x 60 x 25 (C); socamm_module 90 x 14 x 3 (C; 14 x 90 mm recollected from the public JEDEC SOCAMM2 figures, NOT re-read, range 12-16 x 85-95); dimm_module 141 x 7.5 x 34 (B; JEDEC DIMM outline 133.35 x 31.25 recollected, not re-read); nic_dpu_card 76 x 117 x 16 (C; OCP NIC 3.0 SFF outline recollected, not re-read, +-5); coldplate_manifold 118 x 70 x 14 (C, range 60-130 x 50-100); qd_pair 61 x 42 x 23 (C, 14 mm bodies range 10-20); fan_module 40 x 56 x 40 (B: 40 x 56 from the Lenovo guide, 40 mm frame is an assumption); pcie_slot 89 x 7.6 x 11 (C; 89 mm x16 length from PCIe CEM); bmc_card 50 x 34 x 8 (C); coin_cell 25 x 22 x 8 (A for the CR2032 20 x 3.2 mm cell; holder estimated); e1s_bank 46 x 135 x 37 (C; E1.S 118.75 x 33.75 x 9.5 recollected, not re-read).

## Layout assets: placement and use
- Origin: tray centre at the board top surface (x = 0, y = 0 mid-depth, z = 0 board top; -y front). Instantiate as a linked-duplicate of the single joined mesh `<id>_dressing`, scale = detail scale 4, place at the stack's HOOK_tray<j>_surface. Free area used: x +-245 mm, y -183..-8 mm (front three quarters of the board); the rear quarter stays free for the connector and retimer rows.
- Fan modules are drawn at 60 percent height (24 mm instead of 40 mm) so the board stays visible from the low interior camera (documented in the JSON simplifications).
- Hoses are straight polylines from the rear QD pairs along the side walls to the cold-plate barbs (not routed physically).
- Material names: MAT_datacenter_* (same library palette as the other datacenter assets) plus new ones (choke_ferrite, cap_polymer_tan, cap_can_blue, cap_top_silver, pcb_black, pcb_card_blue, heatsink_alu, coldplate_copper, fan_frame, fan_rotor). Appending several of these files duplicates materials with .001 suffixes (asm.append behaviour).

## NVL72 backplane cartridge band (deviation)
The real NVL72 cable cartridges are at the rack REAR (Lenovo: NVLink switch tray rear has "cable cartridge connectors"). The film asked for "NVL72 backplane cartridges visible" in the rack-interior shots, so a stylised band (4 cartridge modules, dark frame, window with 56 bowed silver twinax strands, black end blocks with gold contact strips, blank label plate, pull handle) is mounted on the backplane FRONT face in each tray gap. Cable count per module is decorative (the sourced fact is >5,000 cables per rack, 4 cartridges). Existing asset `interconnect/nvl72_copper_cartridge_wall` (893,968 triangles, estimates) was not used inside the rack because of cost.

## Open questions for Wentao
1. Are front-face stylised cartridges acceptable, or should the interior show the rack rear?
2. Provide GB300 / NVLink switch / Helios tray photos if layouts should be closer than generic; none were found locally or in primary text sources.
3. Fan height shrink (60 percent) accepted?

## Progress log
- 2026-10-02 21:03 parts, layouts, band built; previews written.
- 2026-10-02 21:13 layouts rebuilt (fan height, Helios arrangement, preview exposure).
- 2026-10-02 21:16 used in scenes/v1/s02_retimers.blend (8 trays, 3 layouts cycling, 8 cartridge bands); checked in scene stills.
