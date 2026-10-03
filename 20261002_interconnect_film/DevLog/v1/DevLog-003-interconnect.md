# DevLog-003-interconnect: pluggables, cables, connectors, fiber (v1.0 model library)

| Field | Value |
|---|---|
| Date | 2026-10-02 |
| Owner | interconnect agent |
| Status | all 8 .blend files built, previewed once, JSON metadata written (resumed twice after usage-limit cut-offs) |

## Plan and checklist

Priority order after resume: nvl72 wall+bundle, osfp/qsfp-dd modules, fiber spaghetti, fiber connectors, DAC/AEC/AOC, fiber cables/trays, cage+faceplate.

- [x] nvl72_copper_cartridge_wall (+ 14-cable loose bundle, cartridge detail, bundle generator): built, 6 previews, level C (see json for verified vs estimated)
- [x] osfp_module, qsfp_dd_module: built, 8 previews each (closed, flat/finned variants, exploded, close-ups)
- [x] fiber_spaghetti_generator: built (9072 strands, GN tubes, squirm drivers, LOD_low 600; evaluated 2.25 M tris high / 0.11 M low)
- [x] fiber_connectors: built (MPO-12/16, adapters, mated pair, LC duplex x3, SC x2, dust caps, end-face macro x40)
- [x] dac_aec_aoc_cables: built (DAC OSFP, AEC QSFP-DD, AOC OSFP, openable plug A, cross-sections x1 and x20)
- [x] fiber_cables_and_trays: built (cords 2.0/3.0 mm, 6 trunk sections with true fibre counts, tie, velcro, splice tray, 1U panel 12 LC + 6 MPO)
- [x] osfp_cage_and_host_connector: built (1x1 cage, connector, belly-to-belly, 1U faceplate 2x16 populated)

## Sources (read 2026-10-02)

See `references/interconnect/SOURCES.json` and `dimension_extracts_2026-10-02.md` (OSFP MSA Rev 5.22, OSFP-XD Rev 1.11 (not modeled), QSFP-DD HW Rev 5.1, US Conec MTP-16 handout and white paper; the US Conec catalog URL returned HTTP 404). Teardown photos viewed: Innolight 800G PSM8 interior (20250913_030604_0), Innolight opened module (3DGS/innolight IMG_7985), Cisco/Finisar 400G DR4.
NVL72 facts (web, 2026-10-02): 4 rear NVLink cable cartridges, >5,000 copper cables, about 2 miles (SemiAnalysis, Lenovo Press GB300 guide, X post citing DGX GB200 user guide); 18 compute trays, 9 switch trays; each GPU has 18 NVLink5 links (one per in-rack NVSwitch ASIC). Derived (own arithmetic, unverified assumption of 4 differential pairs per link): 72 x 18 x 4 = 5,184 twinax pairs. An eBay listing title reads "Amphenol 9x2RU" (not fetched, 403): suggests a cartridge spans 9 compute-tray positions of 2RU; physical arrangement of the 4 cartridges is NOT verified.

## Decisions
- 2 x 16 "stagger" for the 1U faceplate interpreted as 2 stacked rows of 16 OSFP ports (stacked 2x1 cages, 14.9 mm vertical pitch from the spec); documented in the asset.
- Preview budget after resume: one contact sheet per asset.

## Progress log
- 2026-10-02: spec/source reading, MSA dimension extraction, references cached (sidecar), helper modules ic_common.py, ic_optics.py, ic_modules.py and build_osfp_module.py written (not yet run).
- 2026-10-02: resumed after usage-limit cut-off.
- 2026-10-02: nvl72_copper_cartridge_wall.blend saved (detail cartridge 1296 cables, real_4 = 5184 cables, 12x5 low-LOD wall, 14-cable NURBS bundle driven by hooks). Bezier+auto handles did not refresh under drivers (stale handles gave a zigzag); switched to NURBS control-point drivers.

- 2026-10-02: osfp_module.blend and qsfp_dd_module.blend saved (p_open explode driver, p_pull_mm). OSFP: MPO-16 APC receptacle, 7-fin open top (flat-cover variant), DSP+thermal pad+boss, 2 PIC/lens/potting engines, 16 fibers; QSFP-DD: dual LC, TOSA/ROSA, Type 1 flat / Type 2A finned.

- 2026-10-02: fiber_spaghetti_generator.blend saved; generator source embedded as text datablock fiber_spaghetti.py; previews 4.

- 2026-10-02: fiber_connectors.blend saved (US Conec catalog URL 404: MT 2.5 x 8.0 mm level B).

- 2026-10-02: dac_aec_aoc_cables.blend saved, 8 previews.

- 2026-10-02 (resume 2): done = nvl72 wall, osfp_module, qsfp_dd_module, fiber_spaghetti_generator, fiber_connectors, dac_aec_aoc_cables (all saved .blend + json + previews). Not started = fiber_cables_and_trays, osfp_cage_and_host_connector. Known: drivers on custom props need root.update_tag() before evaluated renders in headless; AOC opened close-up still shows plug closed in one shot (camera angle), driver verified working.

- 2026-10-02: fiber_cables_and_trays.blend saved, 6 previews. Only osfp_cage_and_host_connector remains.

- 2026-10-02: osfp_cage_and_host_connector.blend saved, 6 previews. Previews quantized to 128 colours (11 MB total, category 85 MB). All 8 blends open headless with no missing images or libraries.

## Final status (exact)
DONE (blend + json + previews): nvl72_copper_cartridge_wall, osfp_module, qsfp_dd_module, fiber_spaghetti_generator (9,072 and 600 variants), fiber_connectors, dac_aec_aoc_cables, fiber_cables_and_trays, osfp_cage_and_host_connector.
PARTIAL / known gaps: see "Deviations" below. NOT STARTED: OSFP-XD module (spec read, not modeled); SC adapter; a fresh-context audit subagent (not run, usage budget).

## Deviations and simplifications
- Previews: one contact sheet per asset was viewed (usage budget), not every PNG; defects seen and left: EMI finger comb on the cage sides looks stripe-like; faceplate pull-tab loops overlap in the stacked rows; AOC opened close-up angle shows the plug mostly closed; end-face macro ceramic is overexposed; trunk cross-section preview framing includes other items.
- osfp_module.json does not carry triangles_unique_meshes (built before that field existed); triangles = 31,202.
- OSFP/QSFP-DD interior layouts (DSP size, optical sub-assembly sizes, passives) are plausible generic (level C) based on the Innolight 800G PSM8 and Intel/Cisco photos; they are not measured.
- US Conec catalog URL (given in the task) returned 404: MT ferrule height 2.5 mm and length 8.0 mm are level B; MPO/LC/SC housing sizes are estimates (level C).
- NVL72 arrangement of the 4 cartridges, cartridge size and per-cartridge cable count are estimates; only 4 cartridges, >5,000 cables, about 2 miles, 18 compute + 9 switch trays and 18 NVLink5 links/GPU are sourced. 1296 x 4 = 5184 cables is own arithmetic (72 x 18 x 4 pairs).
- 14-cable bundle uses NURBS control points driven by hook empties (translation only; rotation of hooks is ignored); Bezier + AUTO handles do not refresh under drivers.
- Custom-property drivers require `root.update_tag()` before evaluated renders in headless scripts.
- Generator for the spaghetti is embedded as text datablock `fiber_spaghetti.py` in its blend; evaluated cost 2.25 M triangles for 9,072 strands at profile 4 (set Profile Resolution 3 for about 1.7 M).

## Open questions for Wentao
1. NVL72 spine: do you have a rear-rack photo or the exact cartridge layout (4 cartridges: 2 x 2 or 4 columns)? Modeled 2 columns x 2 stacked with a 400 mm switch band (guess).
2. "2 x 16 stagger": built as two stacked rows of 16 (stacked cages, 14.9 mm pitch). Did you mean laterally offset rows?
3. OSFP-XD (1.6T, 16 lanes) module not modeled; wanted?
4. Should the module DSP carry the NARROWCOM parody mark? Currently generic text "DSP-800G" / "DSP-400G".
5. Spaghetti default 2.0 mm cords at real size vs the film's scaled-up fibres: confirm the scale-up factor at assembly.

## Audit corrections
(no fresh-context audit was run; recommended next: check the dimension tables in each JSON against the cached MSA pages, and look at every preview PNG once)
