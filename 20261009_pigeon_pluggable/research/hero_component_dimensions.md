# Hero component dimensions: 800G OSFP DR8 (for the Blender model)

Date: 2026-10-09 (v1).
Grades: A spec/datasheet, B vendor page/paper, C photo-measured, D typical-industry estimate.
Axes: x = mm from front face toward card edge; y lateral, + = TX half (viewer's left looking into the front); z = mm above module bottom.
MSA = `references/specs/OSFP_Module_Specification_Rev5_22.pdf` (Rev 5.22, 2025-08-09).
Photos (`$BLOG/assets/images/2025/20241215_20250915/`): `20250913_031929_0.jpg` = Innolight 800G OSFP PSM8 microscope top view, scale shell width 22.58 mm = 670 px -> 31.5 px/mm (+/-8%); `20250913_030659_0.jpg` = Innolight 400G DR4+, scale from MLCCs ~40 px/mm. The 82 photos in `$DATA/3DGS/innolight/images/` ($DATA = owner data folder outside the repo) show an LC-duplex 4-lane module (likely the 200G QSFP FR4), not used for sizes.

## Anchor: forward stop B

B = forward stop (bottom-case side walls). Length 100.40 max (Fig 3-2); rear lip 10.61 behind B, PCB tip 9.80 behind B (Fig 3-8) -> B at x = 89.8 (derived, 89.6-90.0).

## 1. Shell

| Item | Value (mm) | Position | Source | Grade |
|---|---|---|---|---|
| Envelope | 100.40 x 22.58 +/-0.10 x 13.00 (+0.10/-0.20); long-edge R0.50 min | x 0-100.4 | MSA Fig 3-2 | A |
| Front nose (outside cage) | ~20 long; may be 22.93 W and drop 1.60 below bottom | x 0-~20 | Fig 3-2/3-3 (69.8 min body ahead of B) | A |
| EMI contact band (4 sides) | 56.4 max long | x ~33.4-89.8 | Fig 3-2 | A |
| Thermal flat top | flat 69.00 min; conductive patch 56.00 +/-0.50 x 16.60 min | patch ends 2.20 min before B | Fig 3-15 | A |
| Closed-top fins, example 1 | 10 channels 1.80 W x 3.20 H, ribs 0.30, side walls 1.00, top skin 0.30, channel floor z 9.20 | full flat length | Fig 3-16 | A (example) |
| Open-top fins (= EMI-finger pitch) | 6 fins 1.00 +/-0.10 thick, pitch 2.90, centres y = +/-1.45, +/-4.35, +/-7.25, height 3.80, slots 1.90, outer walls 1.64, draft 3 deg | same | Fig 3-20 | A |
| Clearance above PCB | 5.00 (PCB top to deck underside, example) -> z ~9.4 | - | Fig 3-16 | A (example) |
| Rear guard | top lip 11.50 long, 2.00 thick; bottom floor 1.00 max; 4 rear vents | x ~85-100.4 | Fig 3-8, 3-13 | A |
| Latch pockets (both sides) | z 2.95-6.10 band, depth 0.65, 2.00 min release-tab zone, 20 deg ramp | 44.59 ahead of B -> x ~45.2 | Fig 3-26 | A |
| Pull tab | tip 139.1 ahead of B -> ~49 beyond front face, runs under the bottom; yellow (Pantone 107U) for 1310 nm <=500 m | x ~-49 to ~20 | App. B Fig B-1; Table 3-3 | A (informative) |

## 2. Paddle-card PCB

| Item | Value (mm) | Source | Grade |
|---|---|---|---|
| Thickness / height | 1.00 +/-0.10; centre plane z 3.90 -> z 3.40-4.40 | Fig 3-8, 3-16 | A |
| Width | 18.77 +/-0.07 at card edge; 19.49 ref body; optional neck 16.25 starting 5.93 ahead of E | Fig 3-21, 3-24/25 | A |
| Signal-pad leading edge E | 8.40 +/-0.18 behind B -> x ~98.2 | Fig 3-14 | A |
| PCB tip | 1.40 behind E -> x ~99.6; corner chamfer C0.25, lead-in C0.30 x 45 deg top/bottom | Fig 3-21 | A |
| Pads | pitch 0.60; width 0.38 (28/side); power pads 15/16, 45/46 0.98 wide, tied, 0.15 gap; length >= 1.50 behind E | Fig 3-21/22/23 | A |
| 3 pad lengths | GND front edge 0.85 closer to tip than signal (0.55 from tip); power 0.55 closer (0.85 from tip); signal at E (1.40 from tip) | Fig 3-21 + sec 3.5 (offsets read from drawing) | A (interp.) |
| Rows | one row per side, 30 pads/side: top 31-60, bottom 1-30; pin 1 under pin 60 | Fig 15-1 | A |
| Lateral | top side: pin 60 centre at y +8.85 (TX side), pin 31 at y -8.55; bottom side offset 0.3 mm: pin 1 at y +8.55 (TX side), pin 30 at y -8.85 (RX side); 14-pad groups (13 x 0.60) either side of the power pads. Corrected 2026-10-10 (hero-model agent, per audit): was "pin 60/1 at +8.85, pin 31/30 at -8.55" | Fig 3-21 | A |
| TX/RX halves | +y: TX (top 49-59, bottom 2-12); -y: RX (top 32-42, bottom 19-29); middle: SDA/SCL, INT/RSTn, LPWn/PRSn, VCC; GND every third pad (G-S-S-G) | Fig 15-1 | A |
| Keepout | no components within 8.40 of E | Fig 3-24 | A |

## 3-8. Components

| Component | L x W x H (mm) | Position (x; y) | Source / method | Grade |
|---|---|---|---|---|
| DSP 8x100G, SiPh-driver variant | 14 x 13 (Credo Dove 800 family); 12 x 13 (MaxLinear Keystone); H 1.2-2.0 | x ~52-66; y 0 | credosemi.com/products/optical-dsp/dove-800 ; existing research | B (H: D) |
| DSP thermal gel | 0.3-1.0 thick; DSP+gel <= 5.0 above PCB | on DSP | Fig 3-16; photos | D |
| TX SiPh PIC | Credo Carmel8 7 x 7.5; Kfir 4.7 x 5.4; Innolight PSM8 PIC area ~8.3 W x 4.4 L (+/-20%); thickness 0.15-0.78 (D) | x ~43-51; y 0 | existing research (credosemi); photo 031929_0 | B / C |
| TW-MZM arm | ~2-2.5 long; 8 MZMs side by side | on PIC | arxiv 1811.11096 (existing research) | B (low) |
| PIC edge pitch to FAU | 127 or 250 um | PIC front edge | existing research | B |
| Laser sub-block (Innolight PSM8) | ~9.2 W x 4.9 L, 2 laser sites, each with a round optic ~1.3-1.4 dia (lens or isolator, unverified) | x ~38-43; y 0, sites at y ~ +/-2.5 | photo 031929_0 | C |
| CW DFB die | cavity 1 or 2 mm long (Casela high-power 1310 nm: 125-200 mW uncooled); width ~0.25-0.3; thick ~0.1 | on submount | lightwaveonline.com 14291423 (Casela); low-power DFB dies 0.265 x 0.25 x 0.10 (MACOM 131D) | B / D |
| DFB submount (AlN CoS) | ~1.5-2.5 x 1.0-1.5 x 0.2-0.3 | - | estimate from die size + bond pads | D |
| Isolator | none found in DR8; smallest catalogue 1310 nm isolator ~1.0 dia | - | agiltron.com (OISM) | B |
| TX FAU (Innolight PSM8, 8 fibres) | glass block ~4.4 L x 2.5 W | x ~33-38; y 0 | photo 031929_0 | C |
| V-groove block, 8 ch at 250 um | 10.4 L x 3.8 W x 1.0 T (Pyrex lid) | - | OZ Optics DTS0077 (VGC-8-250-10.4-3.8-1) | B |
| Fibre pitch (MT/FAU) | 0.250 (MT-16: 16 fibres in one row, 0.55 dia guide pins at 5.3 pitch) | - | usconec.com MT-16 SM product page | B |
| RX block (x2, each 4 ch, flanking TX) | ~3.2 W x 9 L total: lens/prism block ~1.6 W x 5 L + PD/TIA area ~3 x 3.8 | x ~42-51; y ~ +/-6.8 (+/-1) | photo 031929_0 | C |
| PIN PD array | 1x4 at 250 um: 0.985 x 0.235 x 0.150 (GaAs analog); 1x8 -> ~2.0 x 0.25 x 0.15; 400G DR4+ photo shows ~0.75 pitch PDs | on RX block | Lasermate PDA4-85A56G DS; photo 030659_0 | B / D / C |
| Octal TIA (GN1825) | 500 um channel pitch -> >= 4.0 wide; die ~4.2-4.5 x 1.0-1.5 x 0.15-0.25 | behind PD | Semtech (pitch); size N/A, estimated | A (pitch) / D |
| MCU STM32L452 | UFQFPN48 7 x 7 x ~0.55; UFBGA64 5 x 5 x 0.6; WLCSP64 3.4 x 3.7 x 0.6; LQFP64 10 x 10 x 1.4 | x ~70-85 | ST estore / Digi-Key listings | A |
| Buck-boost MP28164 | QFN-11 2 x 3 (~0.8 H) | power zone x ~66-88 | monolithicpower.com/mp28164 | A (H: D) |
| PMIC, EEPROM, load switch (SY6210) | PMIC 3 x 3 to 5 x 5 QFN; EEPROM ~2 x 3 DFN; SY6210 likely DFN 2 x 3 (if = SY20810DHC, unconfirmed) | power zone | parts not fully identified; silergy.com SY20810DHC | D |
| Power inductors | ~2.5 x 2 to 4 x 4, H 1-2 | behind DSP | photo 030604_1; typical | C/D |
| Crystal | 2016 = 2.0 x 1.6; 2520 = 2.5 x 2.0; H ~0.5-0.8 | next to DSP | EIA size codes | A (H: D) |
| MLCC | 0201 = 0.6 x 0.3 x 0.3; 0402 = 1.0 x 0.5 x 0.5 | everywhere | EIA size codes | A |

## 9. MPO-16 receptacle (reference only)

| Item | Value (mm) | Source | Grade |
|---|---|---|---|
| MT ferrule end face | 6.4 x 2.5 (length ~8, unverified, D) | IEC 61754-7 abstract (scc-ccn.ca 2036757) | A |
| Plug envelope at belly-to-belly | 6.20 max (keyed side) / 4.70 max | MSA Fig 14-49 | A |
| Lane map | one row: TX1-8 left, RX8-1 right (looking in) | MSA Fig 14-46 | A |
| Receptacle position | inside the front nose, x ~0-15 | Type 1 nose ~20 long | D |

## Layout summary (front to back; for the modeler)

1. x -49 to 0: yellow pull tab under the module.
2. x 0-15: MPO-16 receptacle in the nose (film: lenslet "eyes"; lenslet pitch 0.25 per MT, block ~6.4 x 2.5 face).
3. x 15-33: fibre routing over a metal bridge (8 TX + 2 x 4 RX); PCB front edge ~x 30.
4. x 33-38: TX FAU at y 0; RX lens/prism blocks start at y +/-6.8.
5. x 38-43: laser sub-block, 2 DFB on submounts (y ~ +/-2.5).
6. x 43-51: TX PIC (~8 x 4.5) with wirebond fan; RX PD + TIA at y +/-6.8.
7. x 52-66: DSP 13 x 14 under thermal gel, directly behind the optics.
8. x 66-89: power (buck, PMIC, inductors), MCU, EEPROM, crystal near DSP.
9. x 89.8: forward stop; x 98.2-99.6: 60-pad edge, TX on +y.
Vertical: PCB z 3.40-4.40; top components <= z 9.4; bottom-side parts <= ~2.4 tall.
Steps 2-8 x positions: grade C/D, +/-5 mm (one microscope view: optics ~12.9 long, PCB >= 20.6 ahead of DSP front edge; plus MSA limits).

## Open questions

1. Absolute x of optics/DSP: needs one full top-down photo of the 800G PSM8 board with a ruler.
2. Round optics at the PSM8 lasers suggest lens coupling (film locks butt coupling).
3. DSP height, GN1825 and PD die sizes: not public.
4. SY6210, EEPROM, PMIC part numbers unconfirmed.
5. Pad-stagger offsets (0.55/0.85) are read from the drawing; origins interpreted.
