# Research: hero 800G OSFP DR8 (silicon photonics variant)

Date: 2026-10-09. Method: research subagent, read-only. OSFP MSA facts read directly from Rev 5.2 (2025-04-17): https://www.osfpmsa.org/assets/pdf/OSFP_Module_Specification_Rev5_2.pdf

## Claims

| # | Claim | Value | Source | Conf |
|---|---|---|---|---|
| 1 | CW laser count (SiPh DR8) | 2 (Credo Carmel8 "2 integrated lasers"; Kfir 2 external CW inputs); 4 x 1311 nm CW DFB in Lumentum 2xDR4 OSFP | credosemi.com/?p=4823 ; credosemi.com/?p=4782 ; lumentum.com/en/products/800g-2dr4-osfp-transceiver-module | high |
| 1 | Laser power, cooling | Coherent 2023: 100 mW uncooled / 300 mW cooled for DR4/DR8; Coherent 2024: 70 mW, 1310 nm, uncooled, non-hermetic, Tj to 85 C | coherent.com press release; convergedigest.com | high |
| 1 | Laser-to-PIC coupling | Credo/DustPhotonics: discrete DFB butt-coupled to the PIC, sub-micron alignment, lensless | gazettabyte.com/tag/dustphotonics | high (Credo) |
| 1 | Isolator in CW path | not found for any DR8 product (one TFLN DR8 patent lists lens + isolator) | USPTO 12578596 | N/A |
| 2 | Modulator | MZM: Credo 8 MZMs + phase shifters + monitor PDs; InnoLight SiP MZM 3.5 dB loss, >35 GHz EO bandwidth; no shipping MRM DR8 found | Credo; lightwaveonline.com 14289317 | high |
| 2 | TW-MZM length | ~2-2.5 mm per arm (papers); Kfir die 4.7 x 5.4 mm, Carmel8 7 x 7.5 mm | arxiv.org/pdf/1811.11096 ; Credo | low-med |
| 2 | Driver | integrated in the DSP (direct drive): Broadcom BCM87800 (7 nm, integrated 112G driver, integrated regulators); MaxLinear Keystone (5 nm, SiPh differential drive to 4 Vpp, sub-15 W 800G, 12x13 mm); Marvell Spica Gen2 | docs.broadcom.com/doc/87800-PB ; businesswire 20230302005368 | high |
| 3 | Fiber coupling | edge (butt) coupling; edge-coupled fiber in InnoLight/Huawei 800G SiPh die teardowns; FAU pitch 127 or 250 um | Credo; yolegroup.com (InnoLight 800G SiPh die report) | med |
| 3 | MPO-16 lane map | one row, looking into the receptacle left to right: Tx1..Tx8 then Rx8..Rx1 (TX = left 8, RX = right 8) | OSFP MSA 5.2 sec 14.4.12, Fig 14-46 | high |
| 3 | Connector reality | many DR8s use dual MPO-12 APC (FS, Cisco OSFP-800G-DR8, Gigalight); MPO-16 APC versions: Cisco DR8P, NVIDIA MMS4X00-NM16 | Cisco DS; FS DS | high |
| 4 | RX | Gigalight SiPh DR8: SiPh + CW laser on TX only; RX = separate PIN array -> TIA -> DSP. Credo Kfir is TX-only PIC | gigalight.com/datasheets/800g-osfp-dr8-500m-siph.pdf | high (that vendor) |
| 4 | Octal TIA | Semtech GN1825: 8-ch, 56 GBd PAM4, 500 um pitch, flip-chip, for 800G DR8 | semtech.com press | high |
| 5 | Host interface | 8 x 106.25G PAM4 (53.125 GBd), 802.3ck Annex 120G C2M | FS DS; Lumentum | high |
| 6 | DDM | temperature, Vcc, per-lane TX bias / TX power / RX power; CMIS 5.0-5.3 | img-en.fs.com/file/datasheet/800g-osfp-dr8-datasheet.pdf | high |
| 6 | Electronics (SR8 analog, NVIDIA MMA4Z00-NS teardown) | ST STM32L452 MCU, Giantec memory, Renesas PMIC, MPS MP28164 buck-boost, step-down converter, Silergy SY6210 load switch, Marvell DSP + crystal, thermal potting on DSP, screwed metal shell | chargerlab.com/teardown-of-the-nvidia-800g-dual-port-osfp-optical-transceiver-mma4z00-ns/ | high (that part) |
| 7 | Edge connector | 60 contacts: 16 TX (8 pairs), 16 RX (8 pairs), 4 low-speed (SCL, SDA, LPWn/PRSn, INT/RSTn), 4 VCC (pins 15/16/45/46), 20 GND | MSA sec 15.1-15.3 | high |
| 7 | Mating order | 3 pad lengths: GND first, then power, then signal | MSA sec 3.5, 15.1 | high |
| 7 | TX/RX on card edge | TX groups (pins 2-12, 49-59) at the pin-1/60 end; RX groups (19-29, 32-42) at the pin-30/31 end; low-speed and VCC in the middle: TX and RX on opposite lateral halves | MSA Fig 15-1 | high |
| 7 | Orientation | seen from the module front, TX is on the viewer's left at both MPO and card edge (derived from MSA figures) | MSA figures | med-high |
| 7 | Dimensions | width 22.58 +/-0.10 mm, height 13.00 (+0.10/-0.20) mm, length 100.40 mm max (Type 1/2) | MSA Fig 3-2 | high |
| 7 | Heatsink | OSFP: integrated heatsink (closed or open top); OSFP-RHS: 9.5 mm, no heatsink, host riding heatsink; not mutually compatible | MSA sec 1, Table 12-1 | high |
| 7 | Power | classes 1-8 (1.5-14 W; class 8 >14 W); 3.3 V; 10 A max for OSFP800 | MSA Table 15-7/15-8 | high |
| 7 | DR8 module power | max ~16-16.5 W (datasheets) | approvednetworks OSFP-CLT-800G-DR8-M16-5; FS | high |
| 7 | Case temperature | standard 0-70 C | MSA Table 11-1 | high |
| 8 | Housing material | zinc die-cast shells on sale (supplier listings only) | listings | low |
| 9 | Teardowns | no public annotated SiPh DR8 internals; closest: Chargerlab SR8; paid Yole die reports; own Innolight 800G OSFP PSM8 photos (blog OFS-64) | - | - |
| 10 | Volume | LightCounting: single-mode 800G incl. DR8 likely to exceed forecast in 2024 and triple in 2025 | lightwaveonline.com 55037915 | med |
| 10 | NVIDIA twin-port | MMS4X00-NM: two 400G DR4 ports in one OSFP, 2x MPO-12 APC, finned top, 17 W max; EML in -NS-T and -NM16 variants; a DR8 is one 800G port | docs.nvidia.com/networking/display/mms4x00nm800g500m | high |

## Canonical typical internals (SiPh DR8)

TX inside to outside: host TX pairs 8 x 106.25G -> DSP (5/7 nm, 8:8, CDR + EQ, integrated SiPh driver) -> RF traces/wirebonds -> 2 x 1311 nm uncooled CW DFB (70-100 mW) butt-coupled -> on-chip 1:4 splitter per laser (inferred) -> 8 TW-MZMs with phase-shifter bias and monitor PDs -> edge couplers -> 8-fiber FAU -> MPO-16 APC Tx1-8.

RX outside to inside: MPO Rx1-8 -> fiber array / lens -> separate 8-ch PIN array -> 8-ch TIA (500 um pitch) -> DSP -> host RX pairs.

Electronics: MCU (CMIS firmware, ADC for DDM), memory/EEPROM, 3.3 V load switch, PMIC/buck rails (DSP core, TIA/laser bias), DSP reference crystal.

Mechanics: 2-piece metal shell with integrated finned heatsink, thermal gel/pad DSP to lid, paddle-card PCB with 60-pad edge, pull tab and latch, EMI fingers.

## Choices locked for the film (labelled "typical")

2 CW DFB butt-coupled (no isolator); 8 TW-MZMs on one TX PIC; separate PIN array + octal TIA for RX; single MPO-16 APC; closed-top finned OSFP shell; DSP with integrated driver.
