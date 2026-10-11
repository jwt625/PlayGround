# Research: datacenter rack references (GB300 NVL72 compute rack, Quantum-X800 Q3400 switch, switch-rack layout)

Date: 2026-10-10. Owner: workstream R (DevLog-007). Method: primary documents first (Lenovo GB300 NVL72 user guide, NVIDIA Quantum-X800 datasheet and XDR switch user manual), one CC BY photo pair, vendor cable-manager listings. Downloads one at a time with >= 2 s spacing; HTML error pages deleted. Accuracy grades: A = primary number, B = primary figure measured or consistent secondary sources, C = estimate (method and range given).

Name check: the owner wrote "QD-3400". The actual product is the **NVIDIA Quantum-X800 Q3400-RA** (4U, air-cooled, 144 x 800G ports in 72 OSFP cages, Quantum-3 XDR InfiniBand). A liquid-cooled sibling is listed as Q3400-LD (datasheet) and the CPO version as Q3450-LD (144 MPO, user manual). The bird rack mimics the Q3400-RA front.

## 1. Local files (`references/racks/`, binaries gitignored)

| File | What | Source URL | Date | SHA-256 (first 16) | Size (B) | License / use |
|---|---|---|---|---|---|---|
| lenovo_gb300_nvl72_user_guide.pdf | Lenovo NVIDIA GB300 NVL72 User Guide, 232 pp (front/rear figures of rack, compute tray, NVLink switch tray, power shelf, mgmt switch; specs) | https://pubs.lenovo.com/gb300-nvl72/gb300-nvl72_user_guide.pdf | 2026-10-10 | 4a8def2b1f285875 | 9924200 | (c) Lenovo 2026; reference only, do not redistribute |
| lenovo_gb300_fig11_rack_front.png | Crop of Fig. 11 (rack front view), rendered from the PDF at 300 dpi | same PDF, p. 21 | 2026-10-10 | fa8a088324c18f15 | 356870 | as above (reference only) |
| lenovo_gb300_fig12_rack_rear.png | Crop of Fig. 12 (rack rear view) | same PDF, p. 22 | 2026-10-10 | bbd1d3bf094ad995 | 313067 | as above |
| lenovo_gb300_fig2_fig9_fig4_tray_fronts_top.png | Fig. 2 compute tray front, Fig. 9 power shelf front, Fig. 4 compute tray top | same PDF, pp. 11, 18, 14 | 2026-10-10 | 37be6027f486b03c | 477866 | as above |
| lenovo_gb300_fig3_fig7_fig8_tray_rears.png | Fig. 3 compute tray rear, Fig. 7 NVLink switch tray front, Fig. 8 NVLink switch tray rear | same PDF, pp. 13, 16, 17 | 2026-10-10 | 3d0251c2066a6013 | 528782 | as above |
| nvidia_quantum_x800_datasheet_naddod_copy.pdf | NVIDIA Quantum-X800 InfiniBand Switches datasheet (NVIDIA doc 3231555, MAY24), reseller-hosted copy | https://resource.naddod.com/files/products-Switches/2025-03-28/nvidia-quantum-x800-infiniband-switches-datasheet-011078.pdf | 2026-10-10 | 430f7d77007695e5 | 250186 | (c) 2024 NVIDIA; reference only |
| nvidia_q3400ra_front.png | Q3400-RA front view figure (1296 x 520) from the NVIDIA XDR switch user manual, Introduction page | https://networking-docs.nvidia.com/xdrswitcheshw/introduction (attachment image-2024-11-18_12-23-30.png) | 2026-10-10 | eb8bd5c7156f0630 | 608582 | (c) NVIDIA; reference only, no redistribution, no texture use |
| nvidia_q3400ra_rear.png | Q3400-RA rear view figure (1058 x 421), same page | same page (attachment image-2024-11-18_12-23-6.png) | 2026-10-10 | 9f6473b3ac74767f | 504119 | (c) NVIDIA; reference only |
| commons_geekerwan_computex2024_gb200_nvl_19.png | ASUS ESC AI POD GB200 NVL72 rack front at COMPUTEX 2024 (1280 px thumb) | https://commons.wikimedia.org/wiki/File:Eindr%C3%BCcke_von_der_COMPUTEX_2024_(_%E6%9E%81%E5%AE%A2%E6%B9%BEGeekerwan)_19.png | 2026-10-10 | b2861b6a243dcdff | 891477 | CC BY 3.0, credit "极客湾Geekerwan / Wikimedia Commons" |
| commons_geekerwan_computex2024_gb200_nvl_20.png | Same rack, second view | https://commons.wikimedia.org/wiki/File:Eindr%C3%BCcke_von_der_COMPUTEX_2024_(_%E6%9E%81%E5%AE%A2%E6%B9%BEGeekerwan)_20.png | 2026-10-10 | 8cd96ba06a5d0de1 | 600443 | CC BY 3.0, same credit |

Not downloaded: Dell-hosted Q3200/Q3400 datasheet PDF (returned an HTML page; deleted); router-switch.com Q3400-RA datasheet (HTML, deleted). No Wikimedia Commons file found for GB300, Q3400 or Quantum-X800 (Commons API search 2026-10-10).

## 2. GB300 NVL72 rack

| Item | Value | Source | Grade |
|---|---|---|---|
| Rack external size | 600 W x 1068 D (1200 with extension frame) x 2236 H mm | Lenovo UG p. 6 | A |
| Weight | about 1580 kg, configuration dependent | Lenovo UG p. 6 | A |
| Rack | "48U MGX rack" with busbar, cable cartridges and manifolds; "standard 19-inch" | Lenovo UG package-contents list ("One 48U rack with a busbar, cable cartridges, and manifolds") (A); Lenovo DS0207 "standard 19-inch", search summary only, not fetched (B) | A / B |
| Contents | 18 compute trays (1U), 9 NVLink switch trays (1U), 6 or 8 power shelves (1U, 33 kW, six 5.5 kW PSUs, 50 V out, up to 660 A), 2 management switches (1U, 48 x RJ-45 1G + 4 x QSFP28), 1 leakage drip pan | Lenovo UG pp. 4-6, 21-22 | A |
| Power | 8 x 33 kW shelves, up to 142 kW per rack (NVIDIA RA); Lenovo product page: 135 kW TDP, 155 kW peak | docs.nvidia.com/enterprise-reference-architectures/nvl72-ai-factory/latest/components.html ; lenovopress lp2357 (search summary only, not fetched) | A (NVIDIA RA, fetched) / B (Lenovo 135/155 kW, search summary) |
| Front order, top to bottom (Fig. 11, 6-shelf configuration) | spare, 2 mgmt switches, spare, 3 power shelves, 1 spare, 10 compute trays, 9 NVLink switch trays, 8 compute trays, drip pan, 3 power shelves, open bays to the plinth | Lenovo UG Fig. 11 | B (order A; U numbers measured) |
| U positions modelled (U1 at the plinth top) | U45-46 mgmt switches; U39-41 power shelves; U28-37 compute (10); U19-27 NVLink switch (9); U11-18 compute (8); U10 drip pan (thin); U7-9 power shelves; U1-6 open | measured on Fig. 11 at 300 dpi: tray pitch 31.2 px = 1U (27 trays over 842 px); rack 1607 px tall = 2290 mm at that pitch vs 2236 mm spec (2.4 percent figure error) | B (+-1U) |
| Plinth (base) height | modelled 82 mm plinth + 20.4 mm top frame = 102.4 mm = 2236 - 48 x 44.45 | arithmetic on A numbers + Fig. 11 proportions | C (plinth 80-140 mm) |
| Compute tray front (Fig. 2, left to right) | left: PCIe/blank bracket (upper) over OSFP card with 2 x 800G OSFP (lower) + RJ-45 OS mgmt; centre-left: 8 E1.S hot-swap bays; centre: control panel (power, ID, reset buttons, RJ-45 BMC, USB 3.0, micro-USB, mini-DP, NVL and fault LEDs); right: BlueField-3 B3240 bracket (RJ-45 + 2 x QSFP) over OSFP card with 2 x 800G OSFP; ears both ends | Lenovo UG Fig. 2 + table | A (inventory), B (positions from the figure) |
| Compute tray rear (Fig. 3) | coolant return QD (left from rear), 4 cable-cartridge connector blocks (each 2 ports with guide pins), busbar connector (centre), coolant supply QD (right) | Lenovo UG Fig. 3 | A |
| Compute tray internals (Fig. 4) | front: E1.S backplane, OSFP cards, BMC/TPM/HMC cards; middle: 8 x 40 x 56 mm dual-rotor fans in two groups; rear half: 2 compute boards with cold-plate loops; busbar cable along the centre | Lenovo UG Fig. 4 + p. 4 | A |
| Tray depth | about 880 mm | C: Fig. 4 aspect ratio (length/width about 2.0) x 19 in body width about 440 mm; range 800-950 mm |
| NVLink switch tray front (Fig. 7) | left third: power/reset/ID buttons, status LED, 4 NVLink LEDs, USB 2.0, 5 x RJ-45 (leak detect, 2 Eth mgmt, BMC, console); rest: plain panel (light grey in Fig. 11) | Lenovo UG Fig. 7, Fig. 11 | A |
| NVLink switch tray rear (Fig. 8) | coolant return, 4 cable-cartridge connector blocks, busbar connector, coolant supply | Lenovo UG Fig. 8 | A |
| Power shelf front (Fig. 9) | PMC module (left, RJ-45 PoE, reset, status LED), 6 PSUs each with round fan grille, handle, 2 LEDs; ears | Lenovo UG Fig. 9 | A |
| Power shelf rear | power whip connector, busbar connector, 2 RJ-45 (daisy chain) | Lenovo UG Fig. 10 table | A |
| Rack rear (Fig. 12) | vertical busbar on the centre line over the full height; cable-cartridge frame over the tray zone (2 side rails, 9 horizontal bolted cross bars, cable panels in 5 columns); manifold QD columns along both sides; rack-level manifold connectors (4 large couplings, 2 x 2) at the bottom rear below the lower power shelves; power-shelf rears with busbar connectors top and bottom | Lenovo UG Fig. 12 | A (inventory), B (positions) |
| Front colours | black rack frame and posts; silver/light-grey metal tray fronts; power shelves silver with black fan grilles | CC BY photo (ASUS GB200 NVL72, COMPUTEX 2024) | B for GB200; assumed same for GB300 (same MGX tray form factor) |
| Doors | none modelled (photos and Lenovo figures show open front) | Lenovo Fig. 11, photo | B |

## 3. NVIDIA Quantum-X800 Q3400-RA

| Item | Value | Source | Grade |
|---|---|---|---|
| Form factor | 4U, 177.8 H x 438 W x 850 D mm, 19 in rack mount | NVIDIA XDR switch UM, Specifications (networking-docs.nvidia.com/xdrswitcheshw/specifications) | A |
| Weight / power | 60 kg (8 PSUs); typical 2900 W passive cables, max 7000 W active; 200-240 Vac per PSU | same | A |
| Ports | 144 x 800 Gb/s in 72 OSFP cages; separate 400G InfiniBand in-band management port (UFM / FNM) on the front, apart from the data ports | NVIDIA datasheet 3231555; UM Introduction | A |
| Cooling | air-cooled (Q3400-RA); liquid-cooled Q3400-LD option; 10 replaceable fans, 8 replaceable PSUs | datasheet; UM Introduction table | A |
| Front management | USB 2.0 type A, 2 x MGT, FNM, console, all front | UM Introduction table | A |
| Front layout (NVIDIA front figure) | 72 cages = 3 block rows x 3 column groups; each block = 2 stacked rows x 4 cages; a champagne/beige port-label strip between the upper and lower cage of each pair (labels like "1 ^v 2": one cage = 2 logical ports); square-hole perforated grille everywhere else; left panel: port LED matrix, FNM OSFP port, NVIDIA badge with "Q3400-RA"; right panel: LED matrix, status LEDs, USB + 2 x RJ-45 | NVIDIA front figure | A (inventory), B (positions) |
| Front dimensions measured on the figure | scale 0.338 mm/px (438 mm / 1296 px); cage pitch about 23.4 mm (69 px), cage opening about 13.7 mm high; label strip about 3 mm; block-row pitch about 57.5 mm; column-group pitch about 110.8 mm, gap between groups about 16 mm; left panel about 62 mm, right panel about 64 mm | measured on nvidia_q3400ra_front.png | B (+-1 mm) |
| Front dimensions modelled (switch_q3400_real) | cage opening 22.8 x 13.3 mm with 0.3 mm lips (outline 23.4 x 13.9 mm, pitch 23.4 mm) so an OSFP module fits (22.58 +-0.10 x 13.00 +0.10/-0.20 mm, OSFP MSA Rev 5.22 Fig 3-2, see research/hero_800g_osfp_dr8.md row 7); panels 62 / 64 mm -> group gap 15.6 mm (measured about 16 +-1 mm); block-row pitch 57.5 mm | model parameters in scripts/assets/datacenter/rk_q3400.py | B |
| Rear layout (NVIDIA rear figure) | 8 PSUs, 4 on each side, stacked (IEC inlet, latch, red LED); 10 fans in 2 rows of 5 in the middle, each with a beige/champagne pull handle | NVIDIA rear figure | A |
| Colour | black faceplate, beige/champagne label strips and fan handles | NVIDIA figures | B |

## 4. Switch-rack practice (bird rack frame)

| Item | Value | Source | Grade |
|---|---|---|---|
| Network cabinets 800 mm wide carry vertical finger cable managers beside the 19 in rails | Chatsworth 38646-733: for 42U 800 mm cabinet, 1821 mm tall, 127 mm deep, short fingers, about 24 cables per U; Vertiv VRA1016: 800 mm version has two rows of fingers; Eaton SRCABLEVRTL: 2 in wide, for 750-800 mm racks, 42U-48U | chatsworth.com product 38646-733; comms-express.com Vertiv VRA1016 listing; eaton.com SRCABLEVRTL (search summaries 2026-10-10) | B |
| Typical top-of-rack items | 1U fibre patch panel, 1U management switch (in-rack GB300 NVL72 uses two 1U mgmt switches with 48 RJ-45 + 4 QSFP28), horizontal cable manager | Lenovo UG pp. 5, 14 (mgmt switch); patch panel / horizontal manager = common practice, no single source fetched | B / C |

## 5. Design consequences (what is sourced vs stylized)

- GB300 compute rack: true scale, envelope A, U layout B, tray front inventory A with positions B, internals not modelled (trays are closed). Tray depth C (880 mm).
- Bird switch rack: STYLIZED. The cabinet is 800 W x 1068 D x 2236 H (width = common network cabinet; depth and height matched to the GB300 rack so the row fronts and tops line up; C/design). The bird chassis keep the Q3400-RA front grammar (black square-hole grille, cage pairs with a champagne label strip between them, 3 column groups, left badge/FNM/LED panel, right LED/USB/RJ-45 panel, rear with side PSUs and a row of fans with champagne handles) but each chassis is one block row (2 x 3 cells) and the cells are portrait to hold a standing 2.6x cyber-pigeon. Scale statement in DevLog-007 section 3.
- A true-scale Q3400-RA chassis is also built (reference and side-by-side scale shot); its 72 cages follow the measured front layout.
- Badges are parody names only (owner rule): "NVYDIA QX-3400" on the true-scale chassis, "NVYDIA QX-3400B" on the bird chassis. The real product name appears only in docs. No logos.

## 6. v4 wave (2026-10-10): GB300 compute-tray front ports, optics and fibre

| Item | Value | Source | Grade |
|---|---|---|---|
| Compute tray front network ports | 4 x 800 Gb/s OSFP on two "OSFP cards" (2 each), "direct GPU connections"; BlueField-3 B3240 with 2 x QSFP (400G Ethernet / NDR IB) + 1 x 1GbE RJ-45 OOB; RJ-45 BMC; RJ-45 OS mgmt; USB 3.0, micro-USB, mini-DP | Lenovo GB300 NVL72 UG Fig. 2 + Table 2 (section 2 above, local PDF) | A |
| Same, second vendor | Gigabyte XN15-CB0 (GB300 NVL72 1U compute tray): "4 x 800 Gb/s OSFP (NVIDIA ConnectX-8 SuperNIC)", 1 x BlueField-3 DPU; front I/O USB 3.2 Type-A, mini-DP, RJ-45, MLAN; rear I/O 4 x NVLink Switch connectors | gigabyte.com AI-DLC-Rack_NVIDIA-GB300-NVL72 (page returned HTTP 403 to fetch; search summary only) | B |
| NVLink switch tray front | no optics: buttons, LEDs, USB 2.0, 5 x RJ-45 (leak detect, 2 x Eth mgmt, BMC, console); NVLink goes through the rear blind-mate cable-cartridge connectors | Lenovo UG Fig. 7, Fig. 8, rack text p. 22 | A |
| NIC-side 800G module | NVIDIA MMS4A20 (order code 980-9IAT0-00XM00): single-port OSFP, riding heat sink = flat top (RHS); one MPO-12/APC optical connector (8 of 12 fibres used); 1310 nm single-mode, 500 m; links Quantum-X800 twin-port switch optics to the ConnectX-8 mezzanine in liquid-cooled systems and to CX-8 PCIe cards | networking-docs.nvidia.com/9iat0mosfp800sprhs (fetched 2026-10-10) | A |
| Same, details from search summaries of the NVIDIA pages | 18 W max; "Green plastic shell on the MPO-12/APC connector denotes Angled Polish Connector"; RHS used in ConnectX-8 cages of B300 / GB300; switch-side twin-port 2 x 800G DR4 (MMS4A00-XM) with two MPO-12/APC; multimode and AOC not used for XDR | docs.nvidia.com 800G XDR parts list and MMS4A20 pages (search summaries; the parts-list PDF downloaded on 2026-10-10 is a deprecation stub, not kept) | B |
| Pull tab | yellow loop in NVIDIA's MMS4A20 figure | `references/racks/nvidia_mms4a20_osfp_rhs_800g_dr4.png` | A (colour of the figure) |
| Fibre jacket colour | yellow = single-mode (TIA-598 colour convention) | industry convention, not re-read | B |

Local file added: `references/racks/nvidia_mms4a20_osfp_rhs_800g_dr4.png` (NVIDIA product figure, 1418 x 939, SHA-256 9e6b1e044181a6e1..., 105,248 B, from the page above; (c) NVIDIA, reference only).

Estimates used in the model (C): module nose protrudes 13.5 mm past the cage front (range 10-20); MPO housing 12.4 x 7.4 x 24 mm plus an 18 mm boot; MPO round jumper 3.0 mm OD; slim Cat6 4.2 mm OD (range 3.8-4.5); fibre bend radius: as built 22-24 mm minimum on the exit arcs (measured on the sampled paths; about 7-8 x OD, inside common long-term static limits for 3.0 mm bend-insensitive SMF jumpers, C), Cat6 30 mm; velcro straps every 0.25 m. Port y positions measured on Lenovo Fig. 2 at 2.47 px/mm (v4 audit): OSFP -190 / -144 and +114 / +160, BF3 QSFP +156 / +178, BF3 RJ-45 +120, BMC RJ-45 about +57 (model +54), OS RJ-45 -108 (B). Design choices (not from refs): all CX-8 OSFP cabled except compute trays 5 and 16 (dust caps, to show both states) and compute tray 14 (the pull-out tray, capped); BlueField-3 port 1 cabled with a QSFP112 + MPO module (module type not sourced), port 2 capped; which 2 of the 5 NVLink-switch-tray RJ-45 are cabled; fibres dressed to two strapped bundles at the front sides, up and back over the roof into the overhead raceway drops; Cat6 in two outboard bundles to the two in-rack management switches.
