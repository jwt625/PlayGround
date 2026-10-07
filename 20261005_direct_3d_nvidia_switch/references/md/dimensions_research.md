# Dimensions research: Spectrum-X CPO switch tray and Spectrum-6 CPO package

- Date: 2026-10-05
- Author: Claude subagent
- Purpose: metric scale references (published, citable) for the photo-based 3D model of the "Spectrum-X CPO Switch Tray" and the "Spectrum-6 CPO" package photographed 2026-08-24.
- Labels used: **FACT** = quoted from a source; **OBS** = what I saw in the project photos (`data/undistorted_s4/`, `source/images/`); **EST** = estimate (reasoning given); **ASSUMP** = assumption.

## 1. Product identification

- FACT: the booth sign reads "Spectrum-X CPO Switch Tray" (OBS, IMG_5735). The package sign reads "Spectrum-6 CPO" (OBS, IMG_5711).
- FACT: NVIDIA lists exactly one single-ASIC CPO Spectrum-6 system: **SN6810-LD**: "128 x MMC-16" connectors, "2RU liquid-cooled design", single Spectrum-6 ASIC, 102.4 Tb/s [S1]. The other CPO SKU, SN6800-LD, is 5RU with 4 ASICs and 512 MMC ports [S1][S2].
- OBS (IMG_5735 crop): the front panel has 8 groups of green connector blocks (4 left, 4 right). Each group is 2 columns x 4 rows. Center: 4 RJ45 jacks, one USB receptacle, and a row of LEDs, with cantilever levers at both ends.
  - This matches the Dell SN6810-LD front-panel illustration (4 port groups on each side, management ports in the center) [S3, p.2, "For illustration only"].
  - It also matches the port list: 2x 100/1000 management + 1 RJ45 serial + 1 RJ45 BMC = 4 RJ45 [S3 table]. The PNY datasheet lists 1 management port, which would give 3 RJ45 [S2 table], so Dell matches the photo.
- EST: 8 groups x 8 green blocks = 64 blocks. For 128 MMC ports that is 2 ports per block, which is consistent with US Conec's "MMC 2-Port Adapter, designed for an MPO (SC) cutout" [S8]. Not confirmed by NVIDIA.
- **Conclusion (EST, high confidence): the photographed tray is an SN6810-LD (2RU) with the top cover removed.**
- Discrepancies to keep in mind:
  - **Fiber connector name:** "MMC-16" [S1] vs "MMC-12" [S2][S3].
  - **USB type:** USB-C v3 [S2] vs USB-A v2 [S3].
    - OBS: the receptacle's width/height ratio in the photo is closer to USB-A than USB-C. This is tentative because of the oblique view.
  - **Coolant connectors:** 4x UQD8 v2 per datasheets, "2 Cold + 2 Hot per switch" [S2][S3][S4]. You saw 2 fittings at the rear, so 2 may be hidden or not fitted on the display unit. Not verified.
  - **Depth:** 776 mm body [S2][S3] vs 900 mm "including UQD and front levers" [S1].

## 2. Tray dimensions (official)

| Source | H | W | D | Note |
|---|---|---|---|---|
| NVIDIA SN6000 user manual v1.3, Specifications (last updated 2026-09-16) [S1] | 87 mm (3.43") | 438 mm (17.2") | 900 mm (35.4") | "*including UQD and front levers"; rack: "19" MGX v1.2 Rack"; weight 32.5 kg |
| NVIDIA/PNY SN6000 datasheet (JAN26), Technical Specifications p.4 [S2] | 87 mm (3.43"), 2RU | 438 mm (17.24") | 776 mm (30.55") | |
| Dell PowerSwitch SN6000 datasheet, p.5 [S3] | 87 mm (3.43"), 2RU | 438 mm (17.24") | 776 mm (30.55") | |

- FACT: the rail kit 930-9SKIT-00L0-00F is an "Installation kit for standard 2U switches in MGX v1.2 racks". The front has "cantilever handles" and ejector bars [S5].
- FACT: the MGX rack "adapts the ORV3 rack to support 19" EIA gear and deploy on a 1RU pitch" [S6, sec. 7.2.1]. So 1RU = 44.45 mm (EIA), not OCP OU = 48 mm.
  - 2RU nominal = 88.9 mm. The EIA panel-height convention gives 1.75n - 0.031 in = 88.11 mm for 2U [S7]. Both are consistent with 87-88 mm.
- FACT (context only, not this SKU): the OCP MGX 1RU reference chassis is 438.00 mm outer width and 431.40 mm internal width, with 766 / 808 mm depth dimensions, three 130 mm front bays with 3.7 mm dividers, and 41.50 mm internal height [S6, Fig. 7-1/7-2].
  - The 438 mm outer width matches the SN6810-LD. Whether the 2RU SN6810-LD reuses the other MGX 1RU chassis dimensions is **not verified**, so don't use those other numbers.
- Since 438 mm appears in every source, treat it as the body width without levers. ASSUMP: the levers/ears are outside this width. The datasheets give no ear or flange dimensions (N/A). In an MGX rack the tray mounts on rails, and EIA ear flanges (482.6 mm overall) are probably absent. Check in the photos before using 482.6 mm.

## 3. Package (Spectrum-6 CPO / Spectrum-X Photonics switch package)

- FACT (secondary, analyst): "For Nvidia's Spectrum-X Photonics switch ASIC package, the substrate will measure 110mm by 110mm." Also: "for Spectrum-X, 36 known good OEs need to be flip-chip bonded onto the substrate first" [S9, SemiAnalysis, 2026-01-01]. The future tense means this is pre-production analysis, not an NVIDIA spec.
- OBS (IMG_5711): **8 optical engines per side, 32 total** visible on the displayed package, which contradicts the 36 in [S9].
  - A search-engine snippet claimed "only 32 active, 4 for redundancy". I could not find this in a primary or fetched page, so it is unverified. Either way, count from your own photos.
- Package outer dimension from NVIDIA: N/A (not found in [S1][S2][S10][S11]).
- Die size, interposer size, stiffener size: N/A (not found).
- Optical engine count from NVIDIA: N/A for Spectrum-6. NVIDIA blogs [S10][S11] give bandwidth and port counts but no OE count or package size.
- Measuring the 110 mm in photos: the blue substrate shows at the notched stiffener corners (OBS). So the outer substrate edge, not the stiffener, should be the 110 mm square, if [S9] is correct. Uncertainty is unknown, because analyst numbers are often rounded.

## 4. Standard parts usable as cross-checks

### RJ45 (8P8C)
- FACT: 8P8C plug width 11.68 mm, height 8.00 mm, length 22.48 mm. Contact pitch 0.040 in (1.02 mm). Plug insertion-area height 0.260 in (6.60 mm). Governing standards: ANSI/TIA-1096-A and ISO 8877, with IEC 60603-7 for datacom [S12, dimensions table].
- EST: the jack opening is the plug width plus clearance, about 11.7-11.9 mm. Vendor drawings vary.
- The photo likely shows the **panel cutout/bezel** around each jack (OBS: a framed rectangle). That is vendor-specific and larger than the opening, often about 12.5-14 mm (EST, not sourced).
- Port pitch for these 4 discrete jacks: N/A. Pitch is layout-specific, not standardized. For example, ERNI ganged jacks use about 13.7-15.2 mm [S13], which shows how much it varies.

### USB receptacles
- FACT: Standard-A receptacle 12.5 mm x 5.12 mm (plug 12.0 x 4.5 mm) [S14, connector table]. The USB 2.0 spec tolerance is commonly cited as ±0.10 mm; I did not open the USB-IF spec, so treat the tolerance as unverified.
- FACT: USB-C receptacle opening 8.34 mm x 2.56 mm, 6.20 mm deep [S15].
- Which one is fitted: see the discrepancy in section 1 (OBS suggests USB-A).
- Use aspect ratio to tell them apart: A = 2.44, C = 3.26.

### MMC fiber adapters (green blocks)
- FACT: MMC is a VSFF (very small form factor) connector with a TMT ferrule. US Conec offers 1/2/4/6-port adapters, and "The 2 port adapter, designed for an MPO (SC) cutout" [S8]. TMT/MT ferrule width is 4.6 mm (12-fiber) and 5.3 mm (16-fiber) [S8, p.3]. The ferrule is not visible from outside.
- Adapter outer dimensions: N/A (no published drawing found). **Do not use as a scale reference.**

### Coolant quick disconnects
- FACT: SN6810-LD uses "UQD8 v2" [S1][S2][S3][S4]. The user manual describes them as "integrated Universal Quick Disconnect (UQD8) connectors" for "direct blind-mating" [S16].
- FACT: OCP UQD Spec rev 1.0 (2020-09-04) [S17]:
  - Table 2 (plug), UQD08: G = Ø21.8 ±0.025 mm (plug nose OD), H = Ø17.48, R = Ø17.4.
  - Table 1 (socket), UQD08: A = Ø22.05 (min), B = Ø17.56.
  - Plug termination is ORB -10 (7/8-14 UNF) per ISO 11926 (Table 3).
- FACT: OCP MGX spec: tray side carries a UQD insert in a float mechanism and the manifold side a UQDB socket. "Nominal OD of UQDB Socket of 24.89mm" (UQD-04 size in that revision) [S6, sec. 10.1].
- FACT, vendor bodies (not standardized):
  - Staubli UQD08 plug, UN thread: ØD 30 mm, hex 27 mm. Socket ØD 32.5 mm [S18, p.4].
  - Danfoss UQDB08 v2 socket: dims A/B/C/D = 50.66 / 35.56 / 33 / 29.8 mm. The meaning of each letter needs the figure, which I did not check. Thread 1-1/16-12 UN [S19, p.5].
- Use: low-reliability cross-check only. The external body is vendor-specific, and the photographed variant (UQD vs UQDB, plug vs socket) is not confirmed.

### EIA-310 / 19-inch rack (only if rack ears are visible)
- FACT: panel width 19 in (482.6 mm). Opening between rails 17.75 in (450.85 mm). Horizontal hole spacing 464.2-465.8 mm. 1U = 44.45 mm. Panel height = 1.75n - 0.031 in. Vertical hole pattern 12.70 / 15.88 / 15.88 mm per U. Standards: EIA-310-D, IEC 60297 [S7].
- OCP Open Rack: OU = 48 mm, 21-in equipment width [S7]. **Not applicable**: the MGX rack mounts 19-in EIA gear at 1RU pitch [S6].

### Other
- Screw heads, logo plates, acrylic signs: no standard size, skipped.
- Spectrum-6 die: N/A.

## 5. Sources

- [S1] NVIDIA, SN6000 Switch Systems User Manual v1.3. Specifications (last updated 2026-09-16): https://networking-docs.nvidia.com/sn6000hw/1.3/specifications. Introduction: https://networking-docs.nvidia.com/sn6000hw/1.3/introduction
- [S2] NVIDIA Spectrum SN6000 Ethernet Switch Series datasheet (PNY copy, doc 4664025, JAN26), p.4 Technical Specifications: https://www.pny.com/en-eu/File%20Library/Professional/DATASHEET/MELLANOX/PNY-ethernet-datasheet-spectrum-sn6000-switch.pdf
- [S3] Dell PowerSwitch SN6000 Ethernet Switch Series datasheet, p.2 illustrations, p.5 table: https://cdn.blueally.com/netsolutionworks/datasheets/powerswitch-sn6800-ld-spec-sheet.pdf
- [S4] NVIDIA SN6000 user manual, Liquid Cooling Specifications: https://networking-docs.nvidia.com/sn6000hw/1.3/liquid-cooling-specifications
- [S5] NVIDIA SN6000 user manual, SN6600-LD/SN6810-LD Rail Kit: https://networking-docs.nvidia.com/sn6000hw/1.3/sn6600-ld-sn6810-ld-rail-kit
- [S6] OCP, MGX Accelerated Computing Rack and Trays Specification rev 1.1 (dated 2024-01-14), sec. 7.1 Fig. 7-1/7-2, sec. 7.2.1, sec. 10.1: https://www.opencompute.org/documents/mgx-accelerated-computing-rack-and-trays-specification-1-1-pdf-1
- [S7] Wikipedia, 19-inch rack: https://en.wikipedia.org/wiki/19-inch_rack
- [S8] US Conec, MMC Connector Solutions brochure (SM-0043-0825), pp.1, 3, 4: https://www.usconec.com/media/hohfpwn1/mmc_brochure.pdf
- [S9] SemiAnalysis, "Co-Packaged Optics (CPO) Book" (2026-01-01): https://newsletter.semianalysis.com/p/co-packaged-optics-cpo-book-scaling
- [S10] NVIDIA Technical Blog, Scaling AI Factories with Co-Packaged Optics (2025-08-18): https://developer.nvidia.com/blog/scaling-ai-factories-with-co-packaged-optics-for-better-power-efficiency/
- [S11] NVIDIA Technical Blog, Scaling Power-Efficient AI Factories with Spectrum-X Ethernet Photonics (2026-01-06): https://developer.nvidia.com/blog/scaling-power-efficient-ai-factories-with-nvidia-spectrum-x-ethernet-photonics/
- [S12] Wikipedia, Modular connector (8P8C row of dimensions table; standards paragraph): https://en.wikipedia.org/wiki/Modular_connector
- [S13] ERNI Modular Jacks catalog MCMJ74a, pp.33, 37: https://pccomponents.com/datasheets/ERNI-MODJACK.PDF
- [S14] Wikipedia, USB hardware (connector dimensions table, Standard-A row): https://en.wikipedia.org/wiki/USB_hardware
- [S15] Wikipedia, USB-C: https://en.wikipedia.org/wiki/USB-C
- [S16] NVIDIA SN6000 user manual, Liquid-Cooled Systems Deployment: https://networking-docs.nvidia.com/sn6000hw/1.3/liquid-cooled-systems-deployment
- [S17] OCP, Universal Quick Disconnect (UQD) Specification rev 1.0 (2020-09-04), Tables 1-3: https://www.opencompute.org/documents/ocp-universal-quick-disconnect-uqd-specification-rev-1-0-2-pdf
- [S18] Staubli, UQD & UQDB thermal management brochure, p.4: https://www.staubli.com/content/dam/fcs/brochures/products/uqd/uqd-thermal-management-standard-staubli-en.pdf
- [S19] Danfoss Hansen UQDB fact sheet AM480054893377en (July 2026), p.5: https://assets.danfoss.com/documents/latest/598106/AM480054893377en-001001.pdf
- Checked, no physical dimensions: ServeTheHome Hot Chips 2026 Spectrum-X article (2026-08-25), https://www.servethehome.com/nvidia-spectrum-x-ethernet-multiplane-network-architecture-at-hot-chips-2026/

## 6. Summary table

Reliability rank: A = official NVIDIA spec or standard, multiple agreeing sources; B = official but definition ambiguous, or a single standard with a photo-feature ambiguity; C = secondary or vendor-specific.

| Feature | Dimension (mm) | Uncertainty | Source | How to measure in photos | Rank |
|---|---|---|---|---|---|
| Tray body width (front, excluding levers) | 438 | ±1 (EST; spec gives no tolerance) | S1, S2, S3 (all agree) | Side-wall outer face to outer face at the front panel, from a near-frontal or top view. Exclude the cantilever levers. | A |
| Tray front panel height (2RU) | 87 (88.11 EIA convention, 88.9 nominal 2U) | ±1.5 | S1, S2, S3, S7 | Front-face top edge to bottom edge. Check whether the open top lowers the visible height. | A |
| Tray depth, body | 776 | ±2 (EST) | S2, S3 | Front face to rear face of sheet metal, excluding UQDs and levers. Needs the full length in one view or the SfM model. | B |
| Tray depth incl. UQD and front levers | 900 | ±2 (EST) | S1 | Lever front to UQD tip. Conflicts in definition with 776; difference is 124 mm. | B |
| 1RU pitch (MGX rack is EIA) | 44.45 | exact | S6, S7 | Only if rack rails or a hole pattern are visible | A |
| RJ45 plug width / jack opening | 11.68 (plug); jack opening about 11.7-11.9 | ±0.2 for the opening; larger if measuring the bezel | S12 | Inner width of the jack opening, not the panel cutout frame. Average all 4 jacks. | B |
| RJ45 plug height | 8.00 | ±0.3 for the opening | S12 | Opening height including the latch slot | B |
| USB Standard-A receptacle | 12.5 x 5.12 | ±0.1 (commonly cited) | S14 | Inner opening of the USB port next to the RJ45s. Confirm type by aspect ratio (2.44). | B |
| USB-C receptacle (if fitted) | 8.34 x 2.56 | small | S15 | Same; aspect ratio 3.26 | B |
| CPO package substrate | 110 x 110 | unknown, analyst value, pre-production | S9 | Outer blue-substrate edges (visible at the stiffener notch corners), measured on the frontal package views | C |
| Optical engines per package | 32 seen (8 per side) vs 36 claimed | count | OBS; S9 | Count in IMG_5711 / 5712 / 5717 | n/a |
| UQD08 plug nose OD | 21.8 | ±0.025 (spec), but variant unconfirmed | S17 | Mating nose OD of the rear coolant fittings, if it is a UQD08 plug | C |
| UQD08 body hex (Staubli) | 27 across flats; body Ø30-32.5 | vendor-specific | S18 | Rear fitting hex or body | C |
| EIA panel width / hole spacing | 482.6 / 464.2-465.8 | standard | S7 | Only if EIA ears are present (likely absent on an MGX rail-mounted tray) | n/a |
| MGX 1RU reference chassis outer width | 438.00 | ±0.127 (spec edge-to-edge) | S6 | Same as tray width. Confirms the 438 value; other MGX 1RU dimensions do not apply to the 2RU SN6810-LD. | A (width only) |
| MMC adapter body | N/A | - | S8 (no drawing) | Do not use | - |
| Spectrum-6 die / interposer size | N/A | - | not found | - | - |

Recommended use: set the global scale from the 438 mm tray width. Cross-check with the 87 mm front height (independent axis) and the RJ45/USB openings (small features, same front plane). Check the package's photo-derived size against the analyst 110 mm only after scaling from objects at the same depth. In IMG_5711 the tray front and the package are both in frame.
