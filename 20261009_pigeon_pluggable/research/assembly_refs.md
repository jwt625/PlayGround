# Assembly equipment references (workstream M, DevLog-019)

Date: 2026-10-10. Grades: A spec/datasheet/primary SOP, B vendor page/press release/patent/paper, C photo-observed, D estimate or stylization (stated). Web lookups were done one at a time (search + single page fetch); Commons downloads with >= 3 s spacing. Vendor claims are marked as such. Nothing here is a spec of the film's parody equipment; it is what the set is modeled after.

## 1. Process chain used in the film (TX subassembly, then the module)

| Step | Real-world practice | Film station |
|---|---|---|
| 1 | Die attach of PIC / laser chips with a precision die bonder (vision alignment, bond head with collet, eutectic or epoxy) | S1 DIE ATTACH |
| 2 | Lens / isolator placement by pick-and-place, then active alignment with the laser powered and the coupled power measured; UV-curable (or thermal) epoxy; UV cure while the position is tracked (cure shrinkage) | S2 ACTIVE ALIGN + UV CURE |
| 3 | Ball-wedge gold wire bonding (first bond = ball, second = stitch/wedge) on a heated workholder | S3 WIRE BOND |
| 4 | Adhesive / thermal interface dispensing, housing halves closed and screwed | S4 DISPENSE + LID PRESS |
| 5 | Power-on test: transmitter eye / TDECQ on a sampling oscilloscope, pass/fail | S5 TEST / EYES ON |

## 2. Facts

| # | Topic | Fact | Source | Grade |
|---|---|---|---|---|
| 1 | Active alignment machine | ficonTEC AL2000: "six-degrees of independent axes for submicron precision alignment", pick-and-place "almost equally accurate", bonding by UV epoxy, thermal epoxy, soldering, laser welding, "about 1um post-bond-shift accuracy"; applications include FTTx transceiver assembly and micro-lens positioning with epoxy dispensing | https://www.electrooptics.com/press-releases/ficontec-al2000 ; https://www.photonics.com/pr32077/AL2000_Assembly_System | B (vendor) |
| 2 | PIC transceiver active alignment | ficonTEC AL2000-AA (PIC transceivers): pick-and-place of individual elements through active alignment and bonding with beam measurement; positional tracking during UV epoxy bonding because cure shrinkage shifts parts (search summary of the vendor material) | same press material (search summary, not re-fetched) | B (vendor) |
| 3 | Active alignment sequence | Passive vision placement first, then active alignment: laser on, coupling efficiency to the fiber measured while the lens position is adjusted; the isolator blocks light reflected back into the laser | US 6,896,422 (USPTO full text, search summary) | B |
| 4 | Epoxy types | after alignment a volume of epoxy is introduced; UV-curable, temperature-curable, time-curable | US 6,736,553 (search summary) | B |
| 5 | 6-DOF alignment engine | PI F-712 / F-713 photonics alignment: compact hexapod + NanoCube piezo scanner; firmware first-light detection, area scan, gradient search; vendor claims "typically 99%" time reduction, alignment "in less than one second" | https://www.physikinstrumente.com/en/products/fast-multi-channel-photonics-alignment-systems ; https://www.pi-usa.us/en/products/photonics-alignment-solutions/f-713h-compact-high-speed-6-dof-photonics-alignment-systems | B (vendor) |
| 6 | UV spot cure | Excelitas OmniCure LX500: UV LED spot curing, 2 or 4 channels, heads at 365 / 385 / 395 / 405 nm, light-guide heads, exposure time +/- 0.1 s, intensity 5-100 % | https://www.polytec.com/en/photonics/products/uv-technology/uv-curing/led-spot-curing-system/lx-500 ; https://prod.excelitas.com/product/omnicure-lx500-uv-led-spot-curing-system-controller | B (vendor) |
| 7 | Die bonder accuracy | MRSI-S-HVM: two modes +/-0.5 um @ 3 sigma and +/-1.5 um @ 3 sigma; eutectic, epoxy stamping and dispensing; silicon photonics, chip-on-wafer; MRSI-H3LD (diode lasers) < 3 um, on-the-fly tool changer, fast-ramp eutectic station | https://www.ofcconference.org/library/exhibits/OFC/2021/pdfs/914249-pb-productBrochure1.pdf ; https://www.semi.org/en/press/mrsi-launches-new-mrsi-h3ld-die-bonder-targeted-high-power-diode-laser-market | B (vendor) |
| 8 | Ball-wedge bonder anatomy | K&S 4124 SOP: first bond = ball, second = capillary wedge; heat, ultrasonic energy, force, time; workholder (chuck) set point "normally 150 - 200" C; bond head parts: EFO wand (moves under the wire tail, HV spark forms the ball), wire tension clamp, main wire clamp (sets tail length), ultrasonic transducer, capillary ("hollow high density ceramic tube"), wire spool, fiber lamp, microscope | University of Florida NRF SOP "Kulicke & Soffa Model 4124 Ball Wire Bonder", https://nrf.aux.eng.ufl.edu/_files/documents/144.pdf (text extracted 2026-10-10) | A (primary SOP) |
| 9 | Free air ball | EFO electrode near the wire tail, gas discharge melts the tail, surface tension makes a near-spherical free air ball (FAB) | US 6,898,849 ; US 7,523,848 (search summaries) | B |
| 10 | Capillary tip | example for ~1 mil (25 um) Au wire: hole ~1.0 mil, chamfer ~1.3 mil, tip diameter ~2.6 mil (66 um), face angle ~11 deg, cone angle ~50 deg; tip diameters 2.4-3.8 mil in use | US 7,407,080 (search summary) | B |
| 11 | Capillary hole vs wire | 23 um wire needs >= 28 um hole, >= 34 um chamfer for a reliable stitch | NTU paper, https://personal.ntu.edu.sg/mzwzhong/Analysis%20and%20experiments%20of%20ball%20deformation%20for%20ultra-fine-pitch%20wire%20bonding.pdf (search summary) | B |
| 12 | Capillary shank | 1/16 in (1.588 mm) shank and ~0.437 in (11.1 mm) length are the common size: NOT verified this session (SPT page had no numbers) | - | D (used for the model) |
| 13 | Dispensing | Nordson ASYMTEK jetting handles UV-cure adhesives; side-view LED example jets 0.1-0.2 mm dots through 0.4 mm windows (2011, silicone); needle dispensing needs a small gap (typically ~125 um) | https://www.nordson.com/en/divisions/electronics-solutions/your-process/fluid-types/adhesives ; https://www.led-professional.com/products/led-production-test-equipment/nordson-asymtek-fluid-dispensers-jet-precisely-into-narrow-cavities-to-improve-side-view-led-manufacturing-process | B (vendor) |
| 14 | Transceiver test | Keysight 800G manufacturing test: N1092X DCA-M sampling oscilloscope + N1078 clock recovery; TDECQ as the transmitter metric; pass/fail on few measurements for throughput | https://go.keysight.com/au/en/solutions/optimize-800g-optical-transceiver-manufacturing-tests.html ; https://www.keysight.com/dk/en/assets/3121-1474/demos/800G-Optical-Transceiver-Test.html | B (vendor) |
| 15 | Packaging review | Carroll et al., "Photonic Packaging: Transforming Silicon Photonic Integrated Circuits into Photonic Devices", Appl. Sci. 6(12):426 (2016), doi:10.3390/app6120426, CC BY: PIC packaging needs micron-level optical alignment, temperature control, electrical integration; laser-to-PIC integration covered. Full text not fetched (MDPI 403) | https://doi.org/10.3390/app6120426 | B (abstract level) |
| 16 | Laser train parts (film) | Si lens + micro isolator + Si lens between the DFB and the PIC edge (Focuslight 18-01342/-01339 class lenses; isolator = polarizer / garnet / polarizer / half-wave plate) | `research/tx_optics_refs.md` (T) | A/B/C |

## 3. Reference images (`references/assembly/`)

| File | What | Source | Licence | SHA-256 | Size (B) |
|---|---|---|---|---|---|
| commons_manual_wire_bonding_workplace.jpg | manual ball bonder: horn/transducer with capillary, wire spool, workholder heater with C display, microscope, fiber lamp | https://commons.wikimedia.org/wiki/File:Gold_wire_for_wire_bonding.JPG (User:Manfreeed) | Public domain | 5ef8f8c7e410c555d47c99883603bd8fb6b904dff5a42529edaa58b71358d8c1 | 1307369 |
| commons_hexapod_positioners.jpg | two hexapod (Stewart platform) positioners; leg/joint/platform shape for the alignment stage (industrial size, not a photonics hexapod) | https://commons.wikimedia.org/wiki/File:Hexapod_positioner_aka_Stewart_platform_x2.jpg (Mike1024, en.wikipedia) | Public domain | d90da90e8f4156f63404660fb63d2a54a111328c2e4c9ffd094965bbd716af67 | 899852 |
| commons_wire_bonding_diagram.png | ball-wedge wire bond diagram (chip to lead frame) | https://commons.wikimedia.org/wiki/File:Wire_bonding.png (Yu. A. Korotaev) | CC BY-SA 4.0 (credit author; share alike if modified) | c533443c8a94c1e3097a4ddf560456b659f6393898e3ad0daf015cc03b27e608 | 78557 |

Vendor product photos (MRSI, ficonTEC, PI, K&S, Excelitas) were viewed only through the pages above; none downloaded (reference only, all rights reserved).

## 4. Modeling decisions (D)

- Machines at human scale (bench 0.9 m, gantries 1.4-1.8 m); tool ends (collet, tweezers, capillary, horn, needle, UV light-guide tips) scaled x2.6 like the parts they handle (hall bird scale), so tool-to-part proportions stay real.
- Wire bonding speed in the film: ~4-5 frames per wire at 30 fps (6-8 wires/s) for readability; no throughput number is claimed.
- UV flash colour violet (365-405 nm is near-invisible to deep violet; the glow is a viewing aid).
- Parody names only: OUNOLIGHT HATCHERY; machine badges are generic model strings (DB-26 DIE BONDER, AA-6 ACTIVE ALIGNER, WB-26 BALL BONDER), no vendor names or logos.

## 5. v4 additions (2026-10-10): module test station, optical test heads, varying eye

| # | Topic | Fact | Source | Grade |
|---|---|---|---|---|
| 17 | Module test boards | EXFO module compliance board (MCB) tests transceivers / AOCs / DACs; host compliance board (HCB) tests host ports; 800G OSFP MCB carries the high-speed lanes to the BER tester (1x8 O-SMPM, ~67 GHz) plus low-speed signals and CMIS/DDMI over I2C; loopback modules serve as known-good references for host ports | https://www.exfo.com/en/resources/technical-documentation/spec-sheets/module-compliance-board-mcb/ (search summary) | B (vendor) |
| 18 | Optical probe heads on alignment stages | FormFactor CM300xi-SiPh: automated 6-axis fiber positioning above the wafer, coarse stage + piezo fine alignment, in-situ optical calibration (OptoVue); imec station: RF probe manipulator + two motorized fiber manipulators | https://psirep.com/system/files/2026-07/FormFactor%20CM300xi-SiPh%20Datasheet.pdf ; https://www.imec-int.com/en/imec-magazine/imec-magazine-april-2018/developing-silicon-photonics-technologies-with-a-wafer-level-test-station (search summaries) | B (vendor) |
| 19 | Electro-optical probing | PI: miniaturized fiber-array (FAU) alignment combined with electrical probing | https://www.physikinstrumente.com/en/expertise/future-zone/electro-optical-wafer-probing (search summary) | B (vendor) |
| 20 | Varying scope eye | approach copied from the interconnect film's eye generator (random symbols, Gaussian channel, noise, per-UI jitter, fold to 2 UI, histogram + glow); here PAM4, parameter wander and persistence | `20261002_interconnect_film/scripts/tex_gen.py` `eye_frame` (read only) | - |

Film framing (D): the birds have no fiber, so the S5 test heads are free-space optical heads (housing with a 2x4 lens-array window matching the bird's 2x4 lenslet eye, ring LED, motorized carriage on a gantry) that dock next to each eye, in the spirit of motorized optical probe heads (facts 18-19); the electrical side is the socket under the feet (MCB analogue, fact 17).

| File | What | Source | Licence | SHA-256 | Size (B) |
|---|---|---|---|---|---|
| commons_semiauto_probe_station.jpg | semi-automatic probe station (microscope, manipulators, controller rack, wafer map on screen) | https://commons.wikimedia.org/wiki/File:Micromanipulator_Semiautomatic_Probe_Station.jpg (Rgarrigan) | CC BY-SA 4.0 (credit author; share alike if modified) | 751973125d86b4686e70b06c04f68876883bac992ebec3f057b99a3826cd479e | 140642 |

## 6. v4.1 final-test stand (2026-10-10): how a DR8-class module is tested, and the film adaptation

### 6.1 Facts

| # | Topic | Fact | Source | Grade |
|---|---|---|---|---|
| 21 | Compliance fixtures | IEEE 802.3 C2M method: Host Compliance Board (HCB) and Module Compliance Board (MCB) are the reference fixtures for host and module electrical compliance (mated HCB/MCB specified in the 802.3ck drafts, subclause 162B.1.3 per the Nov 2019 contribution); module output verified at TP4 through the MCB, module input at TP1 (802.3bs-era method) | https://grouper.ieee.org/groups/802/3/ck/public/19_11/ghiasi_3ck_01a_1119.pdf ; https://grouper.ieee.org/groups/802/3/ck/public/19_03/sun_3ck_03_0319.pdf ; Keysight user manual https://www.keysight.com/ae/en/assets/9018-04702/user-manuals/9018-04702.pdf (search summaries) | B (drafts; final clause numbers not checked) |
| 22 | Commercial OSFP boards | MultiLane OSFP800: ML4064-MCB-112 module compliance board, ML4064-CHCB-112 cabled host compliance board, ML4064-ALB-112 active loopback (DSP inside, acts as a BERT in a host port); EXFO 800G OSFP MCB: BERT <-> module serial link, 1x8 O-SMPM (~67 GHz), CMIS/DDMI over I2C | https://www.oiforum.com/wp-content/uploads/OIF_PLL_Demo_OFC2023_MultiLane.pdf ; https://www.opencompute.org/marketplace/listing/multilane-osfp-active-loopback-for-800g-ml4064-alb-112-2/ ; https://www.exfo.com/en/resources/technical-documentation/spec-sheets/module-compliance-board-mcb/ (search summaries) | B (vendor) |
| 23 | TX optical test | 800G manufacturing test: DCA-M sampling oscilloscope (N1092X) + clock recovery (N1078); TDECQ is the primary PAM4 transmitter metric; few measurements per pass/fail for throughput (Keysight) | https://go.keysight.com/au/en/solutions/optimize-800g-optical-transceiver-manufacturing-tests.html ; https://www.keysight.com/dk/en/assets/3121-1474/demos/800G-Optical-Transceiver-Test.html | B (vendor) |
| 24 | TDECQ meaning / typical | TDECQ = power penalty of the transmitter vs an ideal reference transmitter at the target SER; does not correlate 1:1 with receiver sensitivity; values observed from transmitters typically 1-2 dB | https://www.electronicdesign.com/technologies/test-measurement/article/21284438/keysight-technologies-the-essentials-for-quality-tdecq-measurements ; https://www.ieee802.org/3/bs/public/adhoc/smf/17_06_27/king_02_0617_smf.pdf (search summaries) | B |
| 25 | RX test | receiver sensitivity is measured separately from TDECQ; a VOA sets the received optical power (IEEE 802.3dj discussion of VOA level calculation); BER measured by the BERT error detector on the host-side electrical output | https://ieee802.org/3/dj/public/26_07/issenhuth_3dj_01a_2607.pdf (search summary, not read) ; MCB/BERT link: fact 22 | B (search summary, not read) |
| 26 | BERT | Anritsu MP1900A: PAM4 PPG up to 64 GBaud, real-time FEC symbol error measurement for 400/800GbE devices and transceivers; Keysight M8040A 64 GBaud BERT with an N1060A DCA in the 112 Gb/s conformance solution | https://www.electronicdesign.com/markets/press-release/21211349/anritsu-anritsus-signal-quality-analyzer-r-mp1900a-bert-supports-real-time-measurement-of-fec-symbol-errors ; https://www.businesswire.com/news/home/20220317005664/en (search summaries) | B (vendor) |
| 27 | Pre-FEC BER limit | RS(544,514) "KP4" FEC: the optical sub-link pre-FEC BER requirement 2.4E-4 (802.3cd / B400G contributions; not checked in the normative text) | https://www.ieee802.org/3/cd/public/May16/wang_3cd_01a_0516.pdf ; https://grouper.ieee.org/groups/802/3/B400G/public/21_03/wang_b400g_01a_210315.pdf (search summaries) | B |
| 28 | Lanes | 800G DR8 = 8 optical lanes of 100G PAM4 (hero research) | `research/hero_800g_osfp_dr8.md` | B |
| 29 | Loopback | loopback modules (passive / DSP active) characterize and test host ports (MultiLane / EXFO kit descriptions) | fact 22 sources | B |

Not found / not used: a vendor photo or procedure of a complete production final-test station for OSFP DR8 (optical switch + VOA + DCA + BERT cabled to an MCB). The film station is an adaptation of the elements above, not a copy of a specific line.

### 6.2 Film adaptation (D)

| Real element | Film element (no fiber on the birds) |
|---|---|
| Module under test on an MCB (module compliance board); 8 high-speed lanes each way to the BERT over coax (1x8 O-SMPM on real boards) | Bird stands in a test socket on a green "MCB 800G MODULE COMPLIANCE BOARD": 52 pogo pins under the 2 x 26 foot contacts, a pogo pedestal under the belly plate (4 low-speed + 4 VCC contacts on 6 pads, the VCC pairs merged as on the hero), gold traces to 2 x 8 coax launches (1.85 mm style) on the board edges, blue coax cables to the BERT; LS/VCC traces to a header |
| TX fiber -> optical switch / reference receiver -> DCA (eye, TDECQ, OMA, ER) | Free-space "REF RX > DCA" head docks in front of the TX (right) eye; TX screen: live PAM4 optical eye, TDECQ 1.6 dB PASS (no limit shown), OMA PASS, ER PASS |
| Reference transmitter -> VOA -> module RX; BER at the host electrical output | Free-space "REF TX + VOA" head docks at the RX (left) eye; RX screen: BER vs RX OMA (dBm, no numeric ticks) waterfall, steeper with rising power, drawn as the VOA sweeps, KP4 limit line 2.4E-4, SENSITIVITY PASS |
| Full link / loopback: pre-FEC BER on all lanes | FULL LINK screen: 8 lanes x pre-FEC BER (illustrative 4.3E-09 .. 3.1E-08, all below 2.4E-4) filling in lane by lane, live host RX eye, PASS |
| Instruments | Parody names, generic grey boxes, no trade dress: KEYSIGHTLESS (Keysight) sampling scope; ANRITSOON (Anritsu) BERT PPG/ED with 16 coax; VIAVOID (VIAVI) VOA + optical switch; board label MCB (MultiLane / EXFO style board, unnamed) |

Screen styling: dark grey, light grey text, one accent (green) for PASS / OK, plus the amber eye trace.

### 6.3 Reference images

| File | What | Source | Licence | SHA-256 | Size (B) |
|---|---|---|---|---|---|
| commons_bert_vs_scope_eye_diagram.jpg | BERT-based vs oscilloscope-based eye diagram (instrument screen look) | https://commons.wikimedia.org/wiki/File:SyntheSys_Research_Eye_Diagram.jpg (Jimwaschura) | CC BY-SA 4.0 | d3c83c4e423a5a7025b87891400301a2a57e726499e21b6bef139cd9a7cd72c9 | 377275 |

Vendor product photos (MultiLane boards, EXFO MCB, Keysight DCA-M, Anritsu MP1900A) were viewed via the pages above only (reference only, not downloaded).

## 7. v4.1 board upgrade (2026-10-10): what is on a real 800G module compliance board

| # | Topic | Fact | Source | Grade |
|---|---|---|---|---|
| 30 | OSFP low-speed + power pins | SCL / SDA: 2-wire serial (3.3 V LVCMOS, pull-ups to 3.3 V on the host); LPWn/PRSn: multi-level low-power control + module presence; INT/RSTn: multi-level interrupt + reset; VCC: +3.3 V power from host (pins 45/46 etc.) | `references/specs/OSFP_Module_Specification_Rev5_22.pdf` Table 15-1/15-2, section 15.5 (read locally) | A |
| 31 | Wilder OSFP MCB | 800G OSFP MCB: 1.85 mm (70 GHz) or 2.4 mm (50 GHz) RF connector options; conforms to IEEE 802.3ck/bs/cd C2M AUI and OIF CEI-112G-VSR; module power from the included adapter or an external source (user selectable); CMIS access over USB with a GUI (direct I2C / CMIS memory-map read/write per a distributor listing) | https://www.oiforum.com/wp-content/uploads/OIF_PLL_Demo_Wilder_OFC2023.pdf ; https://oiforum.com/wp-content/uploads/OIF_PLL_Demo_Wilder_OFC2024.pdf ; https://etesters.com/catalog/64909079-9412-4208-B9B0-458E68EDA408 (search summaries) | B (vendor) |
| 32 | MultiLane OSFP MCB | ML4064-MCB-112 (8 x 112G) for transceiver / AOC / DAC testing; 224G MCB with high-bandwidth connectors and a USB interface to a CMIS GUI | https://oiforum.com/wp-content/uploads/OIF_PLL_Demo_MultiLane_ECOC24.pdf ; https://www.oiforum.com/wp-content/uploads/OIF_PLL_Demo_OFC2023_MultiLane.pdf (search summaries) | B (vendor) |
| 33 | Vertical launch connectors | Southwest Microwave 1.85 mm vertical launch jack, 2-hole bolt-on flange (screws from the board bottom), solderless and reusable, DC-67 GHz, for microstrip / GCPW | https://mpd.southwestmicrowave.com/docs/Vertical-Launch-Compression-Mount-PCB-Connectors.pdf ; https://mpd.southwestmicrowave.com/product-category/1-85-mm-connectors (search summaries) | B (vendor) |

Not confirmed from sources (used as generic board practice, D): banana jacks, LED indicators, DIP switches / jumpers, test points, DC-DC + inductor, current-sense monitor, fuse / TVS on a specific MCB; none of the vendor summaries listed these parts, and no datasheet with a board photo was read.

Film MCB (D, adaptation): 52 foot pogo pins + belly pedestal (2 VCC + 4 LS pads, 8 contacts); 16 differential pairs (G S S G pattern per foot) routed as matched pairs to 32 vertical-launch connectors (flange + 2 screws, hex body, threaded interface, cable nut) in staggered rows on both side edges (TX LANES 1-8 / RX LANES 1-8), 32 coax to the BERT; LS: U1 USB-I2C bridge + J1 USB-C (to a "CMIS HOST" box), U2 EEPROM, Y1 oscillator, J2 I2C header, SW1 DIP for LPWn/PRSn and INT/RSTn, JP1/JP2, D1-D3 LEDs (PRSn / INT / PG, on when seated), TP1-TP4, traces from the 4 LS pads to U1; power: J3/J4 banana jacks with red / black leads to a parody "RIGOLD" bench supply (readout 3.30 V, 3.8 A: 3.3 V per fact 30; 12.5 W, inside the 16 W max of the FS DR8 datasheet in `references/datasheets/`), F1 fuse, D4 TVS, R1 sense + U4 monitor, U3 DC-DC + L1, C1/C2 bulk + MLCC bank, wide pours to the 2 VCC pads; 4 mounting holes; silkscreen refdes.
