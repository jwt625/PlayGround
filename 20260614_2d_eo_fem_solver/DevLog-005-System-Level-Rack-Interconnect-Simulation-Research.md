# DevLog 005: System-Level Rack Interconnect Simulation Research

Date: 2026-07-10

## Decision Context

The long-term goal is an end-to-end system-level EM/SI simulation model that
can cover rack-scale AI hardware interconnect paths:

- Rack, tray, cartridge, board, and cable mechanical envelopes.
- PCB traces, vias, launches, gold fingers, and connector breakouts.
- RF/coax, twinax, CPC/copper cable, twisted-pair, and pluggable-module links.
- Board-to-board, board-to-cable, cable-to-cable, blind-mate, edge-card, and
  RF launcher interfaces.
- Material models for PCB stackups and high-speed dielectrics.

The key question is whether open datasets exist for rack CAD/dimensions, PCB
EM material properties, connector models, and NVL-style cartridge interfaces.

Current conclusion:

- Rack and tray mechanical data is the most open.
- Connector family information and generic connector models are partially open.
- PCB material properties are available as nominal vendor data, but not as a
  calibrated server-board material stack.
- True production board layouts, launch geometry, NVLink cartridge geometry,
  harness routing, and validated S-parameter channels remain mostly proprietary.

## Research Snapshot

### Open Data Availability

| Layer | Availability | Confidence | Notes |
| --- | --- | --- | --- |
| ORv3 / Open Rack dimensions | Medium-high | High | OCP publishes rack specifications and design pages. Product listings expose concrete dimensions for accepted racks. |
| GB200/NVL72 rack/tray public data | Medium | High | NVIDIA/OCP disclose rack architecture, tray counts, and coarse connector/topology information. Detailed CAD/model bundles may require OCP contribution access and may not include EM data. |
| Modern AI server PCB layouts | Low | High | Some older OCP hardware has board collateral, but modern 112G/224G server board layouts and stackups are generally not open. |
| PCB dielectric properties | Medium | High | Laminate vendors publish Dk/Df tables. These are usable priors, not calibrated channel models. |
| Connector STEP/CAD | Medium-high | Medium | Many vendors publish mechanical CAD after part selection or account login. |
| Connector S-parameters | Medium | Medium | Samtec, Molex, Amphenol, TE, and similar vendors often publish or provide Touchstone models; access can be gated or NDA. |
| NVL/NVLink cartridge implementation | Low | Medium | Public sources confirm architecture class and likely connector family class, but exact production models are not open. |
| Twinax/coax cable models | Medium | Medium | Vendor RLGC/S-parameter models are often available for selected cable assemblies; exact rack harness geometry is not. |

### Highest-Risk Missing Data

The main simulation credibility gap is not the rack envelope. It is the
high-speed channel detail:

- Connector launch and breakout geometry.
- Board stackup and via fields around BGA/connector escape.
- Cable/cartridge harness topology and lengths.
- De-embedded S-parameters for each interface.
- Retimer/equalizer/channel budget assumptions.
- Production tolerances, plating, roughness, skew, and glass-weave effects.

For a useful first simulator, these should be represented as hierarchical
parametric blocks with metadata and confidence annotations, not as one
monolithic geometry.

## Source Cache

Status meanings:

- `cached`: key facts have been extracted into this devlog.
- `follow_up`: source is relevant but needs file download, account access, or
  deeper extraction.
- `watch`: source is useful for current-market context but not primary enough
  for model parameters.

| ID | Topic | Source | Type | Accessed | Status | Cached facts |
| --- | --- | --- | --- | --- | --- | --- |
| S001 | Open Rack specs/designs | https://www.opencompute.org/wiki/Open_Rack/SpecsAndDesigns | OCP wiki | 2026-07-10 | cached | OCP hosts files for infrastructure creators and IT gear builders, including rack, power, interconnect, server/storage/switch context. |
| S002 | ORv3 base spec | https://www.opencompute.org/documents/open-rack-base-specification-version-3-pdf | OCP PDF | 2026-07-10 | follow_up | ORv3 base frame specification exists publicly; useful for base rack envelope and mechanical constraints. |
| S003 | Rittal ORv3 product | https://www.opencompute.org/products/440/rittal-open-rack-v3-orv3 | OCP product | 2026-07-10 | cached | Example accepted rack: 600 mm W x 1068 mm D x 2286 mm H, 44OU/47RU, 48 V DC busbar, 18 kW busbar rating, 1400 kg payload. |
| S004 | Pegatron GB200 NVL72 | https://www.opencompute.org/ai-marketplace/products/780/pegatron-ra4401-72n1-nvidia-gb200-nvl72 | OCP product | 2026-07-10 | cached | Per rack: 18 compute trays + 9 NVLink switch trays. Per tray: 4 Blackwell GPUs, 2 Grace CPUs, 4 NVLink connectors at 1.8 TB/s, 4 ConnectX-7 OSFP 400G ports, 2 BlueField-3 DPUs. |
| S005 | NVIDIA DGX GB hardware guide | https://docs.nvidia.com/dgx/dgxgb200-user-guide/hardware.html | NVIDIA docs | 2026-07-10 | cached | Compute trays are liquid cooled, include 2 Grace CPUs and 4 Blackwell GPUs, and interconnect through NVLink connectors at the back of the tray to NVSwitch trays. |
| S006 | NVIDIA OCP contribution post | https://developer.nvidia.com/blog/nvidia-contributes-nvidia-gb200-nvl72-designs-to-open-compute-project/ | NVIDIA blog | 2026-07-10 | cached | NVIDIA says GB200 NVL72 rack, compute tray, and switch tray liquid-cooled designs were contributed to OCP. |
| S007 | OCP AI systems blog | https://www.opencompute.org/blog/open-compute-project-foundation-expands-its-open-systems-for-ai-initiative | OCP blog | 2026-07-10 | cached | OCP says NVIDIA contributed MGX-based GB200-NVL72 rack, compute, switch tray, and liquid-cooling designs. |
| S008 | OCP deployment blog | https://www.opencompute.org/blog/the-open-compute-project-accelerating-deployment-of-next-gen-ai-clusters | OCP blog | 2026-07-10 | cached | Public description mentions reinforced OCP ORv3 rack architecture and 1RU liquid-cooled MGX compute/switch trays. |
| S009 | Amphenol Paladin HD | https://www.amphenol-cs.com/product-series/paladin-hd.html | Vendor product page | 2026-07-10 | cached | Paladin HD/HD2 class supports 112G/224G, up to 144 differential pairs in 1RU orthogonal configurations, board-to-board, board-to-cable, and cable-to-cable topologies. |
| S010 | Amphenol 224G solutions | https://www.amphenol-cs.com/224g-high-speed-solutions | Vendor product page | 2026-07-10 | cached | SkewClear EXD high-speed bulk cable technology supports 10G to 224G and multiple pair constructions. |
| S011 | Molex QSFP-DD model doc | https://www.molex.com/content/dam/molex/molex-dot-com/products/automated/en-us/electricalmodeldocumentpdf/214/214733/2147334000-000.pdf | Vendor PDF | 2026-07-10 | cached | Example of open electrical model documentation: QSFP-DD 1x1 SMT connector, 16 differential pairs, mated connector + CTV PCB model simulated to 60 GHz in HFSS. |
| S012 | OCP M-XIO spec | https://www.opencompute.org/documents/m-xio-r1-v1p0-rc4-pdf | OCP PDF | 2026-07-10 | cached | Defines a modular extensible I/O connector strategy for PCIe and sideband interfaces, but does not mandate specific connector choices. |
| S013 | PCI-SIG CEM overview | https://pcisig.com/blog/evolution-cem-connectors-almost-20-years-making | Standards org blog | 2026-07-10 | cached | PCIe CEM defines pinout, footprint, mating interface, and form-factor constraints while leaving connector design flexibility to vendors. |
| S014 | Isola IS420 Dk/Df tables | https://www.isola-group.com/wp-content/uploads/data-sheets/is420-high-performance-laminate-and-prepreg__Dk_Df_Tables.pdf | Vendor PDF | 2026-07-10 | cached | Core/prepreg Dk/Df tables include glass style, resin content, thickness, and values at 100 MHz, 1 GHz, 2 GHz, and 10 GHz. |
| S015 | IPC D-24C round robin | https://meridian.allenpress.com/jmep/article/13/3/77/36695/Round-Robin-of-High-Frequency-Test-Methods-by-IPC | Paper | 2026-07-10 | cached | High-frequency laminate characterization depends on measurement method; vendor nominal Dk/Df should be treated as priors. |
| S016 | Polar SI library | https://www.polarinstruments.com/support/si/si_index.html | App notes | 2026-07-10 | cached | Useful background on insertion loss, dielectric loss, and extracting loss tangent from transmission-line measurements. |
| S017 | Samtec connector models guidance | https://blog.samtec.com/post/best-practices-connector-models/ | Vendor blog | 2026-07-10 | cached | Samtec discusses S-parameter connector models, PCB stackup, return path, and breakout-region considerations. |
| S018 | Samtec high-speed cable assemblies | https://www.samtec.com/high-speed-cable/micro-coax-twinax/ | Vendor product page | 2026-07-10 | cached | Public examples of micro-coax/twinax high-speed assemblies with single-ended and differential routing options. |
| S019 | Amphenol RF SMA connector page | https://www.amphenolrf.com/en-us/products/rf-connectors/sma-connectors/ | Vendor product page | 2026-07-10 | cached | SMA board-launch and PCB variants exist; high-frequency end-launch products are available up to tens of GHz depending on series. |
| S020 | Amphenol PCB launch optimization | https://www.amphenolrf.com/en-us/engineering-center/custom-solutions/pcb-launch-optimization/ | Vendor engineering page | 2026-07-10 | cached | Vendor explicitly treats PCB launch refinement as a simulation/application-engineering problem. |

## Extracted Parameter Cache

### Rack and Tray Parameters

| Item | Parameter | Value | Source | Confidence | Use |
| --- | --- | --- | --- | --- | --- |
| ORv3 Rittal rack | Width | 600 mm | S003 | High | Rack envelope seed |
| ORv3 Rittal rack | Depth | 1068 mm | S003 | High | Rack envelope seed |
| ORv3 Rittal rack | Height | 2286 mm | S003 | High | Rack envelope seed |
| ORv3 Rittal rack | Height units | 44OU / 47RU | S003 | High | Tray placement grid |
| ORv3 Rittal rack | Busbar voltage | 48 V DC | S003 | High | Power architecture context |
| ORv3 Rittal rack | Busbar rating | 18 kW | S003 | High | Not representative of GB200 power; use only as ORv3 product example |
| GB200/NVL72 rack | Compute trays | 18 | S004 | High | Topology graph |
| GB200/NVL72 rack | NVLink switch trays | 9 | S004 | High | Topology graph |
| GB200 compute tray | GPUs | 4 Blackwell GPUs | S004, S005 | High | Compute node abstraction |
| GB200 compute tray | CPUs | 2 Grace CPUs | S004, S005 | High | Compute node abstraction |
| GB200 compute tray | NVLink connectors | 4 | S004 | Medium-high | Rear connector placeholder count |
| GB200 compute tray | NVLink connector bandwidth | 1.8 TB/s | S004 | Medium-high | Channel budget placeholder |
| GB200 compute tray | Cluster NICs | 4x ConnectX-7 OSFP 400G | S004, S005 | High | External network channel model |
| GB200 compute tray | DPUs | 2x BlueField-3 | S004, S005 | High | Management/storage network context |

### PCB Material Parameters

| Material/source | Parameter family | Cached values | Confidence | Simulation use |
| --- | --- | --- | --- | --- |
| Isola IS420 | Dk/Df vs frequency | Tables include values at 100 MHz, 1 GHz, 2 GHz, 10 GHz by glass style, resin content, and thickness | High for nominal data | Seed material database and interpolation/fitting flow |
| Generic FR-4 / low-loss laminate summaries | Dk/Df ranges | Public values are inconsistent by vendor/test method | Medium | Use only for coarse examples |
| IPC D-24C methods | Measurement sensitivity | Different extraction methods can disagree | High | Add metadata fields for method and uncertainty |

Minimum material metadata schema:

```yaml
material_id: isola_is420_106_72pct
vendor: Isola
source_id: S014
type: prepreg
glass_style: "106"
resin_content_pct: 72
nominal_thickness_mm: 0.051
dk_df:
  - freq_hz: 1.0e8
    dk: null
    df: null
  - freq_hz: 1.0e9
    dk: null
    df: null
  - freq_hz: 2.0e9
    dk: null
    df: null
  - freq_hz: 1.0e10
    dk: null
    df: null
model_status: nominal_vendor_table
missing:
  - anisotropic_dk
  - copper_roughness
  - press_thickness_tolerance
  - weave_orientation
```

Note: fill exact numeric values from the PDF table during the first structured
data-ingestion pass rather than hand-copying many rows here.

### Connector and Cable Parameters

| Family | Public parameters | Source | Confidence | Use |
| --- | --- | --- | --- | --- |
| Amphenol Paladin HD/HD2 | 112G/224G class, up to 144 differential pairs in 1RU orthogonal configuration, supports board/cable architectures | S009 | Medium-high | NVL-cartridge-class placeholder; not proof of exact NVIDIA implementation |
| Amphenol SkewClear EXD | High-speed bulk cable, multi-pair options, 10G to 224G class | S010 | Medium | Twinax/copper cable library seed |
| Molex QSFP-DD SMT | 16 differential pairs, mated connector + CTV PCB, HFSS to 60 GHz | S011 | High | Example open S-parameter documentation format |
| OCP M-XIO | PCIe/sideband modular I/O interface strategy, connector choice not fixed | S012 | High | Open interface abstraction example |
| PCIe CEM | Pinout, footprint, mating interface, form factor defined by PCIe spec | S013 | High | Gold-finger/card-edge modeling scope |
| RF board launch | SMA, 2.92 mm, 2.4 mm, 1.85 mm, SMP/SMPM class | S019, S020 | Medium | Board-launch model library |

## Proposed Simulation Data Model

Represent the system as a graph of components and interfaces:

```yaml
system:
  racks:
    - id: rack_0
      mechanical_model: ocp_orv3_or_mgx_placeholder
      trays:
        - id: compute_0
          type: gb200_compute_tray_placeholder
          interfaces:
            - id: nvlink_conn_0
              type: high_speed_blind_mate
              model_ref: paladin_hd_class_placeholder
            - id: osfp_0
              type: osfp_400g
              model_ref: vendor_connector_or_blackbox
channels:
  - id: ch_compute0_to_switch0
    path:
      - pcb_escape
      - connector_launch
      - blind_mate_connector
      - twinax_bundle
      - blind_mate_connector
      - connector_launch
      - pcb_escape
    models:
      pcb_escape: parametric_3d_or_2p5d
      connector: touchstone_or_vendor_blackbox
      cable: rlgc_or_touchstone
metadata:
  confidence: placeholder
  source_ids: [S004, S005, S009]
```

Model hierarchy:

| Level | Model type | Purpose |
| --- | --- | --- |
| L0 | Topology graph | Tray/switch/connector/cable connectivity and lengths |
| L1 | Behavioral channel | Cascaded S-parameters, RLGC lines, connector black boxes |
| L2 | Parametric EM submodel | PCB trace/via/launch geometry for sensitivity studies |
| L3 | Full extracted geometry | STEP/ECAD-derived 3D EM model where available |
| L4 | Calibrated production model | Measured/de-embedded S-parameters and fitted material/roughness parameters |

## Progress Tracker

### Completed

- [x] Identified OCP as the primary open source for rack/tray mechanical and
  contribution metadata.
- [x] Confirmed public GB200/NVL72 coarse topology: 18 compute trays and 9
  NVLink switch trays per rack.
- [x] Confirmed public NVIDIA/OCP statement that GB200 NVL72 rack, compute tray,
  and switch tray designs were contributed to OCP.
- [x] Confirmed public NVIDIA docs state rear tray NVLink connectors connect
  compute trays to NVSwitch trays.
- [x] Identified Amphenol Paladin HD/HD2 as a public 112G/224G backplane-class
  connector family relevant to NVL-style rack-scale copper architectures.
- [x] Identified Molex QSFP-DD model documentation as a concrete example of
  vendor-published connector S-parameter metadata.
- [x] Identified Isola Dk/Df tables and IPC D-24C papers as PCB material
  property starting points.

### In Progress

- [ ] Download and archive OCP ORv3 PDF and any accessible MGX/GB200 design
  collateral under a local `research_cache/` or equivalent directory.
- [ ] Convert source cache table into a machine-readable YAML/JSON file.
- [ ] Build a first material library from Isola/Rogers/Panasonic public tables.
- [ ] Build a first connector family registry for RF launchers, PCIe/gold
  fingers, OSFP/QSFP-DD, M-XIO/MCIO/SlimSAS, and Paladin/ExaMAX-class systems.
- [ ] Define a Touchstone metadata schema: port map, reference impedance,
  de-embedding plane, simulation/measurement method, fixture inclusion, and
  frequency range.

### Next Research Tasks

- [ ] Check whether OCP contribution S1029/MGX design collateral exposes
  downloadable CAD files without membership/account gating.
- [ ] Search vendor sites for downloadable STEP and Touchstone files for:
  Amphenol Paladin HD/HD2, ExaMAX/ExaMAX2, Samtec AcceleRate/Flyover, TE STRADA
  Whisper, Molex QSFP-DD/OSFP, and Amphenol/SV Microwave RF launches.
- [ ] Identify open board designs with nontrivial high-speed channels for
  solver validation, even if not AI-server-class.
- [ ] Collect public PCIe CEM connector fixture/test-procedure documents and
  extract reference-plane conventions.
- [ ] Define a synthetic NVL-like benchmark channel using public parameters:
  board escape + 224G-class connector + twinax segment + connector + board
  escape.
- [ ] Decide whether the local solver project should first target 2D/2.5D
  cross-section extraction, cascaded network modeling, or full 3D connector
  submodels.

## Working Assumptions

- Treat public NVL/NVLink cartridge details as topology hints, not as verified
  production EM data.
- Treat vendor Dk/Df tables as priors. Calibrated simulation requires measured
  coupons or vendor-specific fitted material models.
- Treat connector product pages as family-level evidence. Exact model files,
  pin maps, and S-parameters must be tied to explicit part numbers.
- Prefer cascaded S-parameter/channel modeling first. Full 3D geometry should
  be reserved for substructures where parametric sweeps or missing models matter.

## Recommended First Implementation Path

1. Add a `research_cache/` directory with source metadata and extracted
   parameter YAML files.
2. Build a small parser/validator for material and connector metadata.
3. Create a synthetic channel benchmark:
   `PCB trace -> via/launch -> connector S-parameter -> cable RLGC -> connector
   S-parameter -> PCB trace`.
4. Use public Touchstone files where available; otherwise insert placeholder
   analytic blocks with explicit confidence flags.
5. Connect this to the existing solver roadmap only after the 2D FEM backend has
   a clean material/geometry abstraction.

