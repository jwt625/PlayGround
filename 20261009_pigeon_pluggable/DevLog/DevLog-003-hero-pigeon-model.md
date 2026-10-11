# DevLog-003: Hero pigeon model (800G OSFP DR8 as a cyberpunk surveillance rock pigeon)

| Field | Value |
|---|---|
| Date | 2026-10-09 |
| Status | v2 pigeon assembly with audit corrections (2026-10-10) applied; stills and explode-orbit animation re-rendered |
| Scope | Script-generated hero asset (`scripts/assets/build_hero_pigeon.py`) -> `assets/hero/pigeon_osfp_dr8.{blend,glb}` + `pigeon_osfp_dr8_pinmap.json`; renders and animation in `outputs/hero/` |
| Depends on | DevLog-000 (decisions), DevLog-002 R3/R4 (fiber-free eyes), `research/hero_800g_osfp_dr8.md`, `research/hero_component_dimensions.md`, OSFP MSA Rev 5.22 (`references/specs/OSFP_Module_Specification_Rev5_22.pdf`) |
| Authors | Claude (hero-model subagent), at Wentao Jiang's request |

## 1. Requirement history (verbatim corrections, 2026-10-09)

1. Original brief (coordinator): OSFP closed-top module envelope at real scale, front face = head with lenslet eyes, beak latch, toe fairings over the 60-pad edge, internals placed per the research layout. Result: module-layout v1 (kept as reference: `outputs/hero/pigeon_osfp_dr8_exploded_module_layout.png`, script snapshot `scripts/assets/build_hero_module_layout_v1.py`).
2. Wentao (via coordinator): "why is the hero model not looking like a pigeon? This is nonsense. It should look like a pigeon, but with various components themselves faithfully modeled. They should get assembled into a pigeon shape, and then wired properly." The OSFP-envelope requirement was dropped.
3. Wentao: "the exploded png looks great. Now assemble it into actual pigeon shape please." Component models kept as in the module-layout exploded render.
4. Wentao: "iterate and polish the flex cable routing and the pins on the feet, should match the actual contact counts of the OSFP pluggable."
5. Wentao: "add a deliverable for the hero pigeon task: an explosive view with labels similar to the one you had originally (keep the original version), and a short animation of original half transparent pigeon slowly explode out to show inside components, and labels appearing after explosion animation done. During this whole sequence the camera should orbit around the pigeon."
6. Coordinator audit (2026-10-10): 13 findings (section 11). Wentao (2026-10-10), after a requested move of the pull tab to the front: "why is the pull tab coming out of the front? it is more natural coming out from the tail". The pull tab stays at the tail (stylized; real OSFP pull tab is at the front/optical end).
7. Wentao (2026-10-10) on the animation: "make it loopable i.e. continue orbiting same direction but hiding labels and un-explode it back to original. and then speed up to make the total length 15 s."

## 2. What was built (v2 pigeon assembly)

Two stages in one script:
- Component stage: every electronic / optical part at TRUE mm size, same models as module-layout v1 (TX SiPh PIC with splitter trees, 8 MZM arms + G-S-G electrodes, monitor PDs, edge couplers; 2 CW DFB dies on AlN submounts butt-coupled at the PIC back edge with wire bonds; 1x8 PIN PD array + octal TIA + 45 deg turning prism + fan-out bonds; DSP package/die/caps/thermal pad; MCU, EEPROM, PMIC, buck, load switch, 2 inductors, crystal, MLCCs; 2x4 lenslet eyes with iris rings). Added: TX/RX carriers (rigid boards the optics sit on), DSP BGA balls (0.8 pitch perimeter), TIA flip-chip bumps.
- Pigeon stage: rigid placements into a rock-pigeon anatomy, reshaped 2-piece skin, wings, tail, legs/feet carrying the real 60-contact edge, routed flex circuits, thermal path, leash.

True-scale statement: all electronic/optical components keep their modeled mm dimensions relative to each other; only their placement changed. The pigeon body is sized to enclose them: beak-to-tail 131.9 mm (beak tip Y -65.3 to tail-fan tip Y +66.6), standing height 93.8 mm, head ellipsoid 24 x 28 x 24 mm.

### Anatomy mapping

| Pigeon part | Component(s) | Notes |
|---|---|---|
| Head (cranium) | DSP (brain) on a head board, thermal pad + copper spreader on top; MCU (hypothalamus) and the DSP reference crystal (heartbeat, short REFCLK route) under the board | board 13.6 x 18.2 x 1.0 mm, narrowed at the rear to fit the skull |
| Right eye (TX, lateral, red) | 2x4 lenslet array + prism (1x8 edge row -> 2x4) + TX SiPh PIC on a vertical carrier, 2 CW DFB dies deeper inside (behind the PIC) | PIC face outward; edge couplers at the front edge feed the prism |
| Left eye (RX, lateral, cyan) | 2x4 lenslet array + prism + 45 deg turning prism + 1x8 PIN PD array + octal TIA on a vertical carrier | |
| Beak | retention latch: upper mandible + cere, lower mandible on `HOOK_beak_hinge` | scaled 1.4x to pigeon proportion |
| Neck (iridescent skin, thin-film shader) | 3 stacked flexes: host TX lanes (front), power + I2C (middle), host RX lanes (back), 2 clips; copper thermal strap down the back of the neck | |
| Chest | vertical sternum (keel) PCB with a copper GND ring: PMIC, buck converter + inductor L1, load switch on the right face; inductor L2 on the left face (lungs); 64 MLCCs | GND returns from the leg flexes via 2 tap flexes |
| Wings | finned heatsink split in two: 8 feather fins (0.30 mm) per wing layered outward + covert heat-spreader plate; copper straps from the spine plate | |
| Tail | 9-feather fan (fins), folded V; yellow pull tab = leash from under the tail (owner decision 2026-10-10; stylized, a real OSFP pull tab is at the front/optical end) | loop end `HOOK_leash_end` |
| Legs | tarsus struts; 26-trace leg flex down the front of each tarsus, 2 clips per leg | |
| Feet | right foot = 26 TX-side contacts, left foot = 26 RX-side contacts, 4 toes each (3 forward, 1 hallux) | section 4 |
| Belly | contact plate with the 4 low-speed + 4 VCC contacts (true lateral positions, merged VCC pads) | |
| Right leg band | EEPROM (DFN) on an orange band, text "OUTNOLIGHT"; 4-trace I2C flex (SCL, SDA, VCC, GND) up the tarsus to the sternum bus | |


## 3. Dimension table

Grades: A spec, B datasheet/paper, C photo-measured, D estimate. mm unless noted.

| Item | Value | Source | Grade |
|---|---|---|---|
| Edge pad width / pitch | 0.38 / 0.60 | MSA Rev 5.22 Fig 3-21 | A |
| VCC pad (pairs 15/16, 45/46 merged) | 0.98 wide | MSA Fig 3-22/3-23 | A |
| Pad lengths GND / VCC / signal | 2.35 / 2.05 / 1.50 (from 0.55 / 0.85 / 1.40 to a common 2.90 from the tip) | MSA Fig 3-21 (interpretation in `research/hero_component_dimensions.md`) | A (interp.) |
| Pin names and classes | 16 TX, 16 RX, 4 LS, 4 VCC, 20 GND | MSA Fig 15-1 | A |
| Belly pad lateral positions | top pins x = -8.85 + (60-n) 0.6; bottom x = -8.55 + (n-1) 0.6 | MSA Fig 3-21 | A |
| TX SiPh PIC | 7.0 x 7.5 x 0.75 | Credo Carmel8 (7 x 7.5); thickness range 0.15-0.78 | B / t D |
| MZM arm length / lane pitch | 2.2 / 0.78 | papers 2-2.5 mm / fit to PIC width | B (low) / D |
| Edge-coupler pitch | 0.25 | FAU 127/250 um | B |
| CW DFB die | 1.0 x 0.30 x 0.10 | Casela 1-2 mm cavity, MACOM class | B/D |
| DFB AlN submount / spacer | 1.5 x 1.0 x 0.25 / 1.7 x 1.4 x 0.40 | sized from die + bond pads | D |
| PIN PD array 1x8 | 2.0 x 0.25 x 0.15, 0.25 pitch | Lasermate 1x4 at 250 um, doubled | B/D |
| Octal TIA | 4.2 x 1.2 x 0.20, 0.50 pitch | Semtech GN1825 pitch; size not public | A pitch / D |
| RX turning prism | 2.4 W x 2.9 L x 1.0 H | photo block ~1.6 W x 5 L (research C); widened to 2.4 to span the 8 lanes at 0.25 pitch, shortened | D |
| DSP package | 13 x 14 (substrate 0.6) | Credo Dove 800 class 13 x 14 | B (height D) |
| DSP die | 9.0 x 9.5 x 0.75 | not public | D |
| DSP BGA | 0.8 pitch perimeter, 0.25 gap | typical | D |
| Thermal pad | 1.0 thick | 0.3-1.0 range | D |
| MCU | UFQFPN48 7 x 7 x 0.55 | ST STM32L452 package | A |
| Buck MP28164 | QFN 2 x 3 x 0.8 | MPS | A (H D) |
| PMIC / EEPROM / load switch | QFN 4 x 4 x 0.8 / DFN 2 x 3 x 0.55 / DFN 2 x 3 x 0.55 | parts unconfirmed | D |
| Power inductors | dia 4.0 x 2.0 | photo 030604_1, typical | C/D |
| Crystal | 2016: 2.0 x 1.6 x 0.6, on the head-board underside beside the MCU (research: crystal next to the DSP) | EIA size code | A (H D) |
| MLCC | 0402 1.0 x 0.5 x 0.5; 0201 0.6 x 0.3 x 0.3 | EIA | A |
| Wire bonds | 25 um Au | typical | D |
| Lenslet pitch / lens radius | 0.75 / 0.33 | stylized eye | D |
| Flex | polyimide 0.10 thick, signal traces 0.05-0.06 at 0.105-0.11 pitch; rails: DSP core 4.0, 3V3 1.2, GND returns 1.8 (neck), VCC 1.2 x2 (belly), GND taps 0.6 x2 | flex design class; rough IPC-2152 rule ~2.5 A per mm of 35 um outer copper at 10 C rise | D |
| Wing / tail feather fins | 0.30 thick | MSA Fig 3-16 fin (ref) | A (ref) |
| Pull tab | yellow Pantone 107U (A); strap 52.6 mm from under the tail base, loop end ~20 mm past the tail tip (D, stylized placement); the MSA App. B reference length (tip 139.1 mm from the forward stop, ~49 mm beyond a Type 1 front) is not applied at pigeon scale | MSA Table 3-3; App. B Fig B-1 | A colour / D length; width 6 D |
| Pigeon proportions | body, neck, head, tail section profiles and toe lengths (16 / 12.5 / 12 / 8.5) measured on the Commons side and feet photos, scaled so the photo bird is 124 mm beak to tail; head enlarged to 24 x 28 x 24 to enclose the DSP and side optics, beak scaled 1.4x; final 131.9 mm beak to tail | Commons photos (C), enlargements (D) | C / D |
| Head ellipsoid / board / carriers / sternum / skin | 24 x 28 x 24 / 13.6 x 18.2 x 1.0 / 9.2 x 10.4 and 5.2 x 5.6 / keel 30 x 33 / 0.6 skin | sized to enclose | D |

## 4. Feet = the exact OSFP 60-contact edge

Build-time assertions (`build_hero_pigeon.py`): every pin 1-60 used exactly once; counts TX 16 / RX 16 / LS 4 / VCC 4 / GND 20; each foot carries all 16 signals of its direction; every toe ends in a GND pad. Log line: `PIN_ASSERT_OK total=60 counts={'TX': 16, 'RX': 16, 'LS': 4, 'VCC': 4, 'GND': 20}`.

Mapping rules:
- Toe order follows the card position (audit fix 13): the card-edge end of each row goes to the outer toe, the card-centre end toward the inner toe. Right foot (TX): outer toe 60-54, middle toe 53-48, inner toe 8-13, hallux 1-7. Left foot (RX): outer 31-37, middle 38-43, inner 18-24, hallux 25-30.
- Pads run in a row ALONG each toe at 0.60 pitch in the true G-S-S-G-S-S-G order; each pad runs ACROSS the toe with its MSA length (GND 2.35, signal 1.50). Pads are coplanar as on the card (mating order comes from pad length only); the distal GND pad continues into the gold talon tip.
- A 13-pad MSA row (G S S G S S G S S G S S G) cannot be split into two independent 7-pad G-S-S-G-S-S-G toes without inventing GNDs, so one toe of each row holds 7 pads (G..G) and its neighbour 6 (S S G S S G), exactly as on the card where adjacent pairs share the GND between them. The boundary GNDs 48, 13, 43, 18 therefore stay on the feet (talon tips), not in the belly.
- Belly cluster: top-side row 47 SDA, 46/45 VCC (one merged pad), 44 INT/RSTn; bottom-side row 14 SCL, 15/16 VCC (merged), 17 LPWn/PRSn; all pads coplanar.
- Each toe has its own flex branch (7 or 6 traces) on the toe top; every trace ends in a via directly above its pad (via-in-pad). The 4 branches of a foot split from the 26-trace leg flex at the ankle (7 + 6 + 7 + 6 = 26 = 16 signal + 10 GND).
- Machine-readable map with 3D pad centres: `assets/hero/pigeon_osfp_dr8_pinmap.json`.

Pin map:

| Pin | Name | Class | Length class (mm) | MSA side | Pigeon location |
|---|---|---|---|---|---|
| 1 | GND | GND | long (mates 1st) (2.35) | bottom (1-30) | right foot (TX), hallux toe, pad 1 of 7 from base |
| 2 | TX2p | TX | short (mates last) (1.50) | bottom (1-30) | right foot (TX), hallux toe, pad 2 of 7 from base |
| 3 | TX2n | TX | short (mates last) (1.50) | bottom (1-30) | right foot (TX), hallux toe, pad 3 of 7 from base |
| 4 | GND | GND | long (mates 1st) (2.35) | bottom (1-30) | right foot (TX), hallux toe, pad 4 of 7 from base |
| 5 | TX4p | TX | short (mates last) (1.50) | bottom (1-30) | right foot (TX), hallux toe, pad 5 of 7 from base |
| 6 | TX4n | TX | short (mates last) (1.50) | bottom (1-30) | right foot (TX), hallux toe, pad 6 of 7 from base |
| 7 | GND | GND | long (mates 1st) (2.35) | bottom (1-30) | right foot (TX), hallux toe, pad 7 of 7 from base = talon tip GND |
| 8 | TX6p | TX | short (mates last) (1.50) | bottom (1-30) | right foot (TX), inner toe, pad 1 of 6 from base |
| 9 | TX6n | TX | short (mates last) (1.50) | bottom (1-30) | right foot (TX), inner toe, pad 2 of 6 from base |
| 10 | GND | GND | long (mates 1st) (2.35) | bottom (1-30) | right foot (TX), inner toe, pad 3 of 6 from base |
| 11 | TX8p | TX | short (mates last) (1.50) | bottom (1-30) | right foot (TX), inner toe, pad 4 of 6 from base |
| 12 | TX8n | TX | short (mates last) (1.50) | bottom (1-30) | right foot (TX), inner toe, pad 5 of 6 from base |
| 13 | GND | GND | long (mates 1st) (2.35) | bottom (1-30) | right foot (TX), inner toe, pad 6 of 6 from base = talon tip GND |
| 14 | SCL | LS | short (mates last) (1.50) | bottom (1-30) | belly cluster, bottom-side row |
| 15 | VCC | VCC | medium (mates 2nd) (2.05) | bottom (1-30) | belly cluster, bottom-side row (merged VCC pad 15/16) |
| 16 | VCC | VCC | medium (mates 2nd) (2.05) | bottom (1-30) | belly cluster, bottom-side row (merged VCC pad 15/16) |
| 17 | LPWn/PRSn | LS | short (mates last) (1.50) | bottom (1-30) | belly cluster, bottom-side row |
| 18 | GND | GND | long (mates 1st) (2.35) | bottom (1-30) | left foot (RX), inner toe, pad 1 of 7 from base |
| 19 | RX7n | RX | short (mates last) (1.50) | bottom (1-30) | left foot (RX), inner toe, pad 2 of 7 from base |
| 20 | RX7p | RX | short (mates last) (1.50) | bottom (1-30) | left foot (RX), inner toe, pad 3 of 7 from base |
| 21 | GND | GND | long (mates 1st) (2.35) | bottom (1-30) | left foot (RX), inner toe, pad 4 of 7 from base |
| 22 | RX5n | RX | short (mates last) (1.50) | bottom (1-30) | left foot (RX), inner toe, pad 5 of 7 from base |
| 23 | RX5p | RX | short (mates last) (1.50) | bottom (1-30) | left foot (RX), inner toe, pad 6 of 7 from base |
| 24 | GND | GND | long (mates 1st) (2.35) | bottom (1-30) | left foot (RX), inner toe, pad 7 of 7 from base = talon tip GND |
| 25 | RX3n | RX | short (mates last) (1.50) | bottom (1-30) | left foot (RX), hallux toe, pad 1 of 6 from base |
| 26 | RX3p | RX | short (mates last) (1.50) | bottom (1-30) | left foot (RX), hallux toe, pad 2 of 6 from base |
| 27 | GND | GND | long (mates 1st) (2.35) | bottom (1-30) | left foot (RX), hallux toe, pad 3 of 6 from base |
| 28 | RX1n | RX | short (mates last) (1.50) | bottom (1-30) | left foot (RX), hallux toe, pad 4 of 6 from base |
| 29 | RX1p | RX | short (mates last) (1.50) | bottom (1-30) | left foot (RX), hallux toe, pad 5 of 6 from base |
| 30 | GND | GND | long (mates 1st) (2.35) | bottom (1-30) | left foot (RX), hallux toe, pad 6 of 6 from base = talon tip GND |
| 31 | GND | GND | long (mates 1st) (2.35) | top (31-60) | left foot (RX), outer toe, pad 1 of 7 from base |
| 32 | RX2p | RX | short (mates last) (1.50) | top (31-60) | left foot (RX), outer toe, pad 2 of 7 from base |
| 33 | RX2n | RX | short (mates last) (1.50) | top (31-60) | left foot (RX), outer toe, pad 3 of 7 from base |
| 34 | GND | GND | long (mates 1st) (2.35) | top (31-60) | left foot (RX), outer toe, pad 4 of 7 from base |
| 35 | RX4p | RX | short (mates last) (1.50) | top (31-60) | left foot (RX), outer toe, pad 5 of 7 from base |
| 36 | RX4n | RX | short (mates last) (1.50) | top (31-60) | left foot (RX), outer toe, pad 6 of 7 from base |
| 37 | GND | GND | long (mates 1st) (2.35) | top (31-60) | left foot (RX), outer toe, pad 7 of 7 from base = talon tip GND |
| 38 | RX6p | RX | short (mates last) (1.50) | top (31-60) | left foot (RX), middle toe, pad 1 of 6 from base |
| 39 | RX6n | RX | short (mates last) (1.50) | top (31-60) | left foot (RX), middle toe, pad 2 of 6 from base |
| 40 | GND | GND | long (mates 1st) (2.35) | top (31-60) | left foot (RX), middle toe, pad 3 of 6 from base |
| 41 | RX8p | RX | short (mates last) (1.50) | top (31-60) | left foot (RX), middle toe, pad 4 of 6 from base |
| 42 | RX8n | RX | short (mates last) (1.50) | top (31-60) | left foot (RX), middle toe, pad 5 of 6 from base |
| 43 | GND | GND | long (mates 1st) (2.35) | top (31-60) | left foot (RX), middle toe, pad 6 of 6 from base = talon tip GND |
| 44 | INT/RSTn | LS | short (mates last) (1.50) | top (31-60) | belly cluster, top-side row |
| 45 | VCC | VCC | medium (mates 2nd) (2.05) | top (31-60) | belly cluster, top-side row (merged VCC pad 45/46) |
| 46 | VCC | VCC | medium (mates 2nd) (2.05) | top (31-60) | belly cluster, top-side row (merged VCC pad 45/46) |
| 47 | SDA | LS | short (mates last) (1.50) | top (31-60) | belly cluster, top-side row |
| 48 | GND | GND | long (mates 1st) (2.35) | top (31-60) | right foot (TX), middle toe, pad 6 of 6 from base = talon tip GND |
| 49 | TX7n | TX | short (mates last) (1.50) | top (31-60) | right foot (TX), middle toe, pad 5 of 6 from base |
| 50 | TX7p | TX | short (mates last) (1.50) | top (31-60) | right foot (TX), middle toe, pad 4 of 6 from base |
| 51 | GND | GND | long (mates 1st) (2.35) | top (31-60) | right foot (TX), middle toe, pad 3 of 6 from base |
| 52 | TX5n | TX | short (mates last) (1.50) | top (31-60) | right foot (TX), middle toe, pad 2 of 6 from base |
| 53 | TX5p | TX | short (mates last) (1.50) | top (31-60) | right foot (TX), middle toe, pad 1 of 6 from base |
| 54 | GND | GND | long (mates 1st) (2.35) | top (31-60) | right foot (TX), outer toe, pad 7 of 7 from base = talon tip GND |
| 55 | TX3n | TX | short (mates last) (1.50) | top (31-60) | right foot (TX), outer toe, pad 6 of 7 from base |
| 56 | TX3p | TX | short (mates last) (1.50) | top (31-60) | right foot (TX), outer toe, pad 5 of 7 from base |
| 57 | GND | GND | long (mates 1st) (2.35) | top (31-60) | right foot (TX), outer toe, pad 4 of 7 from base |
| 58 | TX1n | TX | short (mates last) (1.50) | top (31-60) | right foot (TX), outer toe, pad 3 of 7 from base |
| 59 | TX1p | TX | short (mates last) (1.50) | top (31-60) | right foot (TX), outer toe, pad 2 of 7 from base |
| 60 | GND | GND | long (mates 1st) (2.35) | top (31-60) | right foot (TX), outer toe, pad 1 of 7 from base |

## 5. Flex routing and wiring

| Flex | Traces (what they carry) | Route | Terminations |
|---|---|---|---|
| `flex_tx_rf_dsp_to_mzm` | 38 (8.8 mm wide): 8 RF lanes + 9 GND guards (G-S-G...) + 21 DC/bias: LD1, LD2, LD_RTN (2 CW DFB), HTR1-8 + HTR_RTN (MZM phase-shifter heaters), MPD1-8 + MPD_COM (monitor PDs) | TX carrier back edge -> 90 deg bend about the width axis behind the carrier -> head-board connector | stiffener + bond area on the carrier pads / vertical flex-to-board connector |
| `flex_rx_tia_to_dsp` | 23 (5.2 mm): 8 TIA outputs + 9 GND + 6 DC: VTIA, VPD_BIAS, TIA_SCL, TIA_SDA, TIA_RSTn, GND | mirror of TX | stiffener on the TIA output pads / connector |
| `flex_tx_host_lanes_neck_leg` | 26: TX1-8 pairs (16) + 10 GND, grouped per toe | head-board connector -> neck (front layer) -> chest (beside the sternum) -> right hip -> front of tarsus -> ankle -> 4 toe branches | board connector / via-in-pad on the 26 toe pads |
| `flex_rx_host_lanes_neck_leg` | 26: RX1-8 pairs + 10 GND | same, back layer, left leg | same |
| `flex_power_i2c_neck` | 8 (11.2 mm): 3V3 1.2 mm, DSP core 4.0 mm, 2x GND 1.8 mm, SCL, SDA, LPWn, INT | head-board connector -> neck (middle layer) -> sternum top-edge connector | connectors both ends |
| `flex_belly_lowspeed_vcc` | 6: SCL, SDA, LPWn/PRSn, INT/RSTn, VCC 15/16 and VCC 45/46 (1.2 mm each) | belly plate -> sternum bottom-edge connector | stiffener / connector |
| `flex_tx/rx_gnd_tap_to_sternum` | 2 x 0.6 mm GND each | leg flex GND traces -> sternum bottom connector (power return to the sternum loads; copper GND ring on both sternum faces) | connector |
| `flex_eeprom_band_i2c` | 4: SCL, SDA, VCC, GND | EEPROM on the right leg band -> up the outside of the tarsus -> sternum bottom connector (MCU I2C bus) | band / connector |
| Thermal straps (copper 0.5) | n/a | DSP spreader -> over the back of the skull -> back of the neck -> spine -> both wing coverts | |
| REFCLK | 1 trace + via | crystal (head-board underside) -> DSP BGA | |

Rules applied: constant width; bends only about the width axis (direction changes inside the ribbon plane are cut-outline curves), so no twist; Catmull-Rom routed curves, no straight runs through parts; neck flexes stacked 0.45 mm apart and clipped twice; leg flexes clipped twice to the tarsus. Component wiring kept from the component stage: PIC, laser, PD, TIA wire bonds; DSP BGA and TIA flip-chip bumps.

Deviation from the coordinator's flex brief: the neck carries the host-side lanes (8 TX + 8 RX pairs between the DSP and the feet) plus power/I2C. The line-side lanes (DSP driver -> MZMs, TIA -> DSP) never leave the head because the PIC and TIA sit in the head next to the DSP; they run on the two short head flexes.

## 6. Stylized vs real (deviations)

- No OSFP envelope: the parts are re-placed into a pigeon; real DR8 modules have no flexible feet, side-mounted PICs, or sternum board.
- Fiber-free eyes: 2x4 lenslet arrays and 1x8 -> 2x4 prisms are invented (real DR8: 16 fibers + MPO).
- The 60 contacts are on toes/belly instead of one card edge; pad sizes, pitch, order and length classes are real.
- Line-side RF flexes carry 8 single-ended lanes with guards because the PIC and TIA models have one RF pad per lane; real SiPh direct drive and TIA outputs are typically differential.
- Pull tab out of the tail (owner decision 2026-10-10: "why is the pull tab coming out of the front? it is more natural coming out from the tail"); on a real OSFP the pull tab is at the front (optical/faceplate end).
- Flex rail widths are estimates from a rough current-density rule (D), not a stack-up design.
- Laser coupling locked to butt coupling (per brief); the owner's 800G PSM8 photo shows a laser sub-block with round optics (research open question 2).
- Beak latch, cere, toe/claw shapes, skin, wings as feather fins, tail fan, leg band: stylized.

## 7. Rebuild

```bash
scripts/bslot.sh -b --factory-startup --python scripts/assets/build_hero_pigeon.py -- assets/hero
scripts/bslot.sh -b --factory-startup --python scripts/assets/check_hero_glb.py -- assets/hero/pigeon_osfp_dr8.glb
scripts/bslot.sh -b assets/hero/pigeon_osfp_dr8.blend --python scripts/assets/render_hero_pigeon.py -- outputs/hero profile,transparent,exploded,face,pins,flex
uv run --with pillow python scripts/assets/label_overlay.py outputs/hero/work/pigeon_osfp_dr8_exploded_raw.png outputs/hero/work/pigeon_osfp_dr8_exploded_labels.json outputs/hero/pigeon_osfp_dr8_exploded_labeled.png columns
uv run --with pillow python scripts/assets/pin_label_overlay.py outputs/hero/work/pigeon_osfp_dr8_feet_pins_raw.png outputs/hero/work/pigeon_osfp_dr8_feet_pins_labels.json outputs/hero/pigeon_osfp_dr8_feet_pins.png
uv run --with pillow python scripts/assets/compose_panels.py outputs/hero/work/flex_panel_neck.png outputs/hero/work/flex_panel_leg.png outputs/hero/pigeon_osfp_dr8_flex_routing.png "neck: TX / power+I2C / RX flex stacked, clipped" "leg: 26-trace flex, clips, 4 toe branches to pads"
scripts/bslot.sh -b assets/hero/pigeon_osfp_dr8.blend --python scripts/assets/animate_hero_explode.py -- outputs/hero 1 360
uv run --with pillow python scripts/assets/anim_label_overlay.py outputs/hero/frames_explode outputs/hero/work/anim_label_anchors_0001_0360.json
ffmpeg -framerate 30 -i outputs/hero/frames_explode/frame_%04d.png -c:v libx264 -pix_fmt yuv420p -crf 18 -movflags +faststart outputs/hero/pigeon_osfp_dr8_explode_orbit.mp4
```

## 8. Structure for animation

- Root `HERO_ROOT` (origin on the ground between the feet; -Y beak, +Z up, -X = pigeon's right = TX).
- Collections: `shell_top` (back skin, neck back, cranium, seams, screws), `shell_bottom` (belly, neck front, jaw), `fins` (wings, tail, thermal straps/spreader), `head_eyes`, `beak`, `leash`, `pcb` (head board, sternum, carriers, belly plate + pads), `dsp`, `tx_optics`, `rx_optics`, `electronics`, `feet` (tarsus, toes, talons, pads, leg band), `wiring` (all flexes, connectors, stiffeners, clips; added in v2).
- Hooks: `HOOK_eye_tx`, `HOOK_eye_rx` (local -Y of the lens = beam out), `HOOK_beak_hinge` (rotate +X to open the lower mandible), `HOOK_foot_tx`, `HOOK_foot_rx`, `HOOK_leash_end` (front end of the pull-tab loop under the chest), `HOOK_i2c_pads` (belly cluster).
- Every mesh has custom property `explode` (m, root frame; exported to glTF extras). Flexes, connectors, sternum and feet keep 0 so the exploded view stays connected; the renderer applies `explode(obj) - explode(nearest exploded ancestor)`.

## 9. Open issues

- Line-side lanes drawn single-ended (see section 6); upgrading means adding a second RF pad row to the PIC/TIA models.
- 233k triangles (flex traces dominate); decimate traces or drop per-trace geometry for crowd birds.
- Ribbons do not stretch in the animation (they stay in place); organs move up to 31 mm away from their flex ends in the exploded state.
- Toe-flex vias sit on the toe top while the pads are underneath (implied through-vias).
- DSP / TIA / PD heights and the PMIC, EEPROM, load-switch packages are estimates (D).

## 10. Deliverables (verified by viewing each render)

| Path | What |
|---|---|
| `assets/hero/pigeon_osfp_dr8.glb`, `.blend` | asset (157 meshes, 232,878 tris, 42 materials, extras `explode` on every mesh, 7 hooks; re-import check passed) |
| `assets/hero/pigeon_osfp_dr8_pinmap.json` | 60-contact pin map with 3D pad centres |
| `assets/hero/pigeon_osfp_dr8_parts.json` | per-part triangles, groups, explode offsets |
| `outputs/hero/pigeon_osfp_dr8_profile.png` | side profile (skin 72% opaque) |
| `outputs/hero/pigeon_osfp_dr8_shell_transparent.png` | 3/4 front-right-above, skin 50% |
| `outputs/hero/pigeon_osfp_dr8_face_closeup.png` | head close-up (DSP stack, TX eye, PIC, beak) |
| `outputs/hero/pigeon_osfp_dr8_exploded_labeled.png` | pigeon exploded view, callout labels on ray-tested visible anchors (23 labels, all visible) |
| `outputs/hero/pigeon_osfp_dr8_feet_pins.png` | bottom view, all 60 pin numbers by class |
| `outputs/hero/pigeon_osfp_dr8_flex_routing.png` | neck + leg routing close-up (2 panels) |
| `outputs/hero/pigeon_osfp_dr8_explode_orbit.mp4` | v2 seamless loop: 15.0 s, 450 frames, 1080x1080, 30 fps, H.264 yuv420p, +faststart (section 14). Contact sheet `outputs/hero/work/explode_orbit_contact_sheet.png` |
| `outputs/hero/pigeon_osfp_dr8_explode_orbit_v1_12s.mp4` | v1, 12 s explode-and-labels version (kept) |
| `outputs/hero/pigeon_osfp_dr8_exploded_module_layout.png` (and the identical original `pigeon_osfp_dr8_exploded.png`) | original labeled module-layout exploded view, kept untouched |

Render times (EEVEE, 1600x1200, 64 samples): profile 5.6 s, transparent 1.8 s, face 2.3 s, exploded 1.9 s, pins 1.7 s, flex panels 1.7 + 1.5 s. Animation v2 loop: 450 frames (12 samples, 1080x1080) in 166 s wall.

## 11. Audit corrections (fresh-context audit, 2026-10-10)

The auditor verified the pin map, orientation (TX = bird's right = -X), topology and part sizes. Findings and what was done:

| # | Finding | Action |
|---|---|---|
| 1 | No DC/bias wiring to the optics | TX line-side flex widened to 8.8 mm with 21 DC traces (LD1, LD2, LD_RTN; HTR1-8 + HTR_RTN; MPD1-8 + MPD_COM) beside the 17 RF traces; RX flex widened to 5.2 mm with 6 DC traces (VTIA, VPD_BIAS, TIA_SCL, TIA_SDA, TIA_RSTn, GND). Label "TX flex: 8 RF + 21 DC bias" added; section 5 trace counts updated |
| 2 | Crystal had no clock trace and sat on the sternum | moved to the head-board underside beside the MCU with a REFCLK trace + via up to the DSP BGA |
| 3 | No power return; rails too narrow | GND tap flexes (2 x 0.6 mm) from each leg flex to the sternum bottom connector; copper GND ring on both sternum faces; neck power flex 11.2 mm (DSP core 4.0, 3V3 1.2, 2 x GND 1.8 mm); belly VCC traces 1.2 mm each; widths listed as D estimates |
| 4 | EEPROM not wired | 4-trace flex (SCL, SDA, VCC, GND) from the leg band up the tarsus to the sternum bottom connector |
| 5 | Pull-tab length claim wrong | DevLog reworded: strap 52.6 mm from under the tail, loop ~20 mm past the tail tip; MSA App. B length not applied. (A coordinator request to move the tab to the front was cancelled by the owner, see section 1/6) |
| 6 | Exploded-label anchors on hidden parts | new `scripts/assets/anchor_vis.py` ray-tests sampled vertices (skipping glass in the still, skin + glass in the animation); PMIC and buck moved to the camera-side sternum face; RX carrier explodes behind its PD/TIA; PD label anchors the PD + submount; belly label anchors the contact plate + pads. Still: 23/23 anchors visible. Animation uses the same logic per frame |
| 7 | Research 'Lateral' row wrong vs MSA Fig 3-21 | `research/hero_component_dimensions.md` line 39 corrected (pin 30 at -8.85 RX side, pin 1 at +8.55 TX side, 0.3 mm row offset) with a correction note |
| 8 | GND/VCC pads raised 0.05 mm | all pads coplanar; mating order represented by pad length only |
| 9 | Tail-tip numbers | beak-to-tail 131.9 mm (beak tip -65.3, tail tip +66.6) |
| 10 | 0.2 mm overlaps | TX flex stiffener now starts 0.08 mm behind the DFB spacers; RX eye prism tab inner face moved to clear the RX carrier |
| 11 | Dimension-table issues | RX turning prism row (1.6 W research vs 2.4 W used, grade D), DSP die split out as D, pigeon-proportions row clarified |
| 12 | Home path in research | line 7 now uses `$DATA/3DGS/innolight/images/` (defined as outside the repo) |
| 13 | Toe order not monotonic with card position | toes reassigned so the card-edge end of each row is on the outer toe and the card-centre end toward the inner toe (TX: outer 60-54, middle 53-48, inner 8-13, hallux 1-7; RX: outer 31-37, middle 38-43, inner 18-24, hallux 25-30); assertions still pass |

## 12. TODO

- [x] Module-layout v1 (superseded, kept as reference)
- [x] Pigeon assembly: skin halves, head/neck/body/wings/tail/legs, organ placement at true component scale
- [x] Feet = exact 60-contact OSFP edge, pin assertions, pin-map JSON + table
- [x] Flex routing rules, terminations, clips, stacked neck
- [x] Renders: profile, transparent 3/4, face, exploded labeled, feet pins, flex routing
- [x] Explode-orbit animation (12 s, labels after explosion, orbit); camera v2 framing
- [x] GLB re-import check
- [x] Fresh-context audit (coordinator) and corrections (section 11)
- [x] 15 s seamless loop animation (section 14)
- [ ] Owner review of v2 + audit pass + loop

## 13. Progress log

- 2026-10-09 22:31: plan written; MSA figures read; component research file first pass (shell + PCB) available.
- 2026-10-09 22:57: module-envelope v1 built and rendered (GLB envelope verified 22.58 x 100.40 x 13.00 mm, 41,008 tris).
- 2026-10-09 23:00: CORRECTION 2 (see section 1): pigeon shape required; envelope dropped.
- 2026-10-09 23:02: CORRECTION 3: keep component models; module-layout exploded render saved as reference.
- 2026-10-09 23:15: pigeon assembly v2 built (136k tris): skin halves with solidify, seams, screws, side-mounted eye optics, DSP/MCU head board, sternum power organs, wing/tail fins, legs/feet, first flex pass.
- 2026-10-09 23:25: silhouette iteration (wings re-rooted along the body, thicker forward neck, head moved forward/down, beak 1.4x, lighter fins/beak); profile reads as a pigeon.
- 2026-10-09 23:30: ITERATION 4: exact 60-contact feet (assert OK 16/16/4/4/20), pin map JSON, rebuilt flex routing (constant width, no twist, stacked neck, clips, connectors, stiffeners, toe branches with via-in-pad), head 28 mm long, head board raised 1 mm.
- 2026-10-09 23:40: ITERATION 5: pigeon exploded labeled still (callout columns), explode offsets reworked for a side 3/4 view (flexes/feet/sternum fixed), explode-orbit animation rendered, labels as camera-facing 2D callouts, encoded, frames deleted, contact sheet checked.
- 2026-10-10 00:12: coordinator review: bird too small in the animation; camera v2 (tight start, eased dolly out with the explosion, pivot drift), off-frame label check (0 cut).
- 2026-10-10 00:20: audit corrections 1-13 applied (section 11). A request to move the pull tab to the front was implemented and then reverted on the owner's decision (verbatim, 2026-10-10): "why is the pull tab coming out of the front? it is more natural coming out from the tail". Rebuilt (232,878 tris), GLB re-import check passed, all stills and the animation re-rendered and viewed, frames deleted, contact sheet checked.
- 2026-10-10 00:45: loop request (section 1 item 7): v1 kept as `pigeon_osfp_dr8_explode_orbit_v1_12s.mp4`; `animate_hero_explode.py` retimed to a 450-frame cycle, assembled framing widened to 0.224 m so the side view fits; rendered, labelled, encoded, verified (section 14), frames and scratch files deleted.

## 14. Explode-orbit loop v2 (2026-10-10)

| t (s) | frames (1-based, cycle 450) | action |
|---|---|---|
| 0.0-1.0 | 1-31 | assembled hold, skin 50% |
| 1.0-5.0 | 31-151 | eased explosion along `explode`; camera dollies out 0.224 -> 0.425 m, elevation 0.058 -> 0.120 m, pivot drifts to the exploded centre |
| 5.0-5.6 | 151-169 | labels fade in (0.4-frame stagger, 9-frame fade) |
| 5.6-8.6 | 169-259 | labels hold |
| 8.6-9.2 | 259-277 | labels fade out |
| 9.2-13.2 | 277-397 | eased un-explode; camera dollies back in |
| 13.2-15.0 | 397-450 | assembled hold; distance, elevation and pivot back at their start values |

The orbit is one full turn at constant angular speed (pivot Z rotation linear, 0 at frame 1, 360 deg at frame 451), so frame 451 is the same pose as frame 1 and is not rendered. No labels in frames 1-150 or 277-450.

Loop verification: frame 451 was test-rendered and compared with frame 1 (mean absolute pixel difference 0.005 of 255, i.e. identical up to sampling noise), then deleted. The seam step frame 450 -> frame 1 (1.69 mean abs diff) matches a normal step frame 1 -> frame 2 (1.71). A 3x loop (`ffmpeg -stream_loop 2`, 45.0 s) was made in scratch; frames 446-453 and 896-903 around both seams show continuous rotation with no jump; the scratch file was deleted. Assembled-frame content spans x 175-1065 px and y 239-910 px (bird about 62% of frame height). Label edge check: 0 cut over 126 label frames; ray-tested anchors fell back to the part centroid for 571 of 2,921 per-frame evaluations (parts fully occluded at some orbit angles).

## Soundtrack for the 15 s loop (coordinator, 2026-10-10)

Request (verbatim): "Add /Users/wentaojiang/Desktop/Plague Inc OST - Plague Blossom (Main Theme, Evolved).mp3 as the loop's sound track. Pick a duration thats 15~20s starting around 1:36, analyze to pick a section that loops nicely, and speed it up (keep pitch) as needed by up to 1.25x. Keep encoding mobile/web compatible."

- Analysis (`scripts/audio/find_loop.py`, `scripts/audio/seam_check.py`, librosa via uv): tempo about 161.5 BPM; beat and downbeat grid around 1:30-1:42; candidates bar-aligned, 15-18.75 s (stretch <= 1.25x), scored by chroma + MFCC similarity of the windows after and before the cut and the RMS ratio across the seam.
- Pick: downbeat at 97.624 s (1:37.6), 40 beats (10 bars) = 15.00 s: similarity after/before the seam 0.966/0.975 (1 s windows), RMS ratio 1.01. Rejected: 1:36.1 start (after 0.98 but before 0.82, RMS drop 19%), 11 and 12 bar loops (1.10x, 1.19x; weaker seams).
- Build (`scripts/audio/make_loop_track.py`): loop length refined by onset-envelope correlation (+3.5 ms, 15.0035 s), atempo 1.00023 (pitch kept) to exactly 15.000 s, cut with margins, 50 ms equal-power circular crossfade, one linear gain to -16 LUFS (true peak -2.4 dBTP before gain). Seam step 0.0045 vs median sample step 0.0073.
- Outputs: `outputs/hero/audio/plague_blossom_loop_15s.wav` (48 kHz 24-bit), `outputs/hero/pigeon_osfp_dr8_explode_orbit_bgm.mp4` (video stream copied: H.264 High yuv420p 1080x1080 30 fps; AAC-LC 192 kb/s 48 kHz stereo; +faststart; 15.000 s; -16.1 LUFS integrated, -2.3 dBFS peak). Video-only loop kept unchanged.
- Verified: decoded audio starts at the music (edit list skips the 1024-sample AAC priming), matches the loop track to 2% RMS residual over 15 s. AAC pads the last frame (896 samples beyond 15.000 s); players that loop at the container duration skip it, while `ffmpeg -stream_loop` plays it (a ~19 ms dip at each seam in that test only).
- Rights: commercial OST, not cleared; fine for private review, not for public posting.

## 15. Hero v2 polish against the owner's teardown photos (workstream H, DevLog-005)

Owner (verbatim, 2026-10-10): "the explaining scenes around the bird is great. Next is to polish the model and improve accuracy and complexity of those scenes and components used. Inspect and model against proper references esp. the 400G and 800G example pluggable teardown pictures." TX and RX optics subassemblies are built by workstreams T (DevLog-008) and X (DevLog-009); H integrates them when delivered.

### 15.1 Reference observations (owner photos)

Sources: blog `2025-09-15-weekly-OFS-64.md` (Innolight 400G OSFP DR4+ and 800G OSFP PSM8: `20250913_030604_0/1`, `030659_0-3`, `031929_0-2`; the other `20250913_*` images in that folder are filters / heterogeneous integration, not pluggables) and `2026-06-13-100G-200G-400G-800G.md` (Cisco/Finisar 400G DR4: `20250425_044131_0-3`, `20250425_183530_0-1`).

| Item | What the photos show |
|---|---|
| DSP | Innolight 400G and 800G: package hidden under a light-grey thermal gel/putty with a torn, smeared surface; chamfered lid corners visible at the edge. Cisco 400G: nickel-plated metal lid with chamfered corners and laser-marked brand / part / date code on an FCBGA substrate |
| PCB | 800G PSM8: black solder mask; 400G DR4+ and Cisco: green. White silkscreen (rotated refdes text, maker / flammability / date-code block), fine curved traces visible under the mask, tented via dots, plated mounting holes with copper rings near the corners. Cisco: exposed gold (ENIG) RF traces fanning from the DSP to the optical engine |
| Passives / power | dense 0201/0402 clusters (tan bodies, tin ends) around ICs and the optics; round grey shielded power inductors (800G) and square molded inductors (Cisco); QFN/DFN ICs with laser-marked top text |
| Optics housings | PIC, lasers and PD/TIA under clear yellow-amber glob-top / gel; glass fiber-array blocks and lens/prism blocks; gold ball-wedge bonds in fans |
| Flex / ribbons | the white flat strips in the Cisco photos are fiber ribbons (no flex in these modules); flex look taken from standard amber polyimide practice |
| Shell interior | die-cast metal, bead-blasted matte grey with machined pockets and fiber channels; hex socket-head screws (Cisco); black spring clip on the 800G |

### 15.2 Plan (H)

- [x] DSP: chamfered nickel lid (11.6 x 12.2, C1.0) with laser-marking decal (NARROWCOM / NC-0800D), torn grey thermal-putty blob (noise-displaced) replacing the clean pad, 70 backside 0201 caps on the head-board underside
- [x] PCBs (head board top/bottom, both sternum faces, TX/RX carriers): black mask with silkscreen decals (refdes boxes with pin-1 dots, OUTNOLIGHT, 94V-0, date codes), tented via fields, GND stitching ring on the sternum, fiducials, test points, ENIG RF traces on the head-board bottom; textures from `scripts/assets/make_hero_textures.py` (belly plate left plain; plated mounting holes not modelled)
- [x] IC top markings (decals): MCU, EEPROM, PMIC, buck, load switch, crystal (generic text only)
- [x] More passives density (sternum 116 MLCCs, head-board underside 70 0201)
- [x] Skin: hex socket-head screws (head + hex socket) instead of domes, bead-blasted screw bosses behind each screw (visible through the x-ray skin)
- [x] Flex: translucent amber coverlay over the traces with bare ends (bond/contact fingers), gold contact rows on every flex-to-board connector
- [x] Integrate T and X subassemblies (interface DevLog-005 section 3): see 15.4
- [x] Re-render all stills and the 15 s loop (v1 copies kept with `_v1`, the loop as `_loop15_v1`)
- [ ] Re-run the integration build when T / X publish new versions (the build imports whatever is in `assets/components/optics_tx|rx/` at build time)

### 15.3 Progress

- 2026-10-10 01:30: v1 renders backed up (`*_v1.png`, `pigeon_osfp_dr8_explode_orbit_loop15_v1.mp4`, `assets/hero/v1/`); photos studied (15.1); T and X not delivered yet.
- 2026-10-10 01:36: polish pass built (251,194 tris, 202 meshes, 58 materials; pin assertion OK; GLB re-import OK, all meshes carry `explode`, 7 hooks); 13 decal textures generated; all stills re-rendered and viewed; 15 s loop re-render started. T and X not delivered yet (no `assets/components/optics_tx|rx`).
- 2026-10-10 01:41: polish loop rendered and encoded (pre-integration; superseded by the integration loop below).
- 2026-10-10 02:05-02:30: T (02:21 version) and X (02:02 version) integrated (15.4); textures regenerated; rebuilt (375,401 tris, 249 meshes, 95 materials, 18 hooks; pin assertion OK; GLB re-import OK, all meshes carry `explode`); all stills re-rendered and viewed.

- 2026-10-10 02:30-02:35: integration loop rendered (450 frames), labelled (24 labels, 126 label frames, 0 cut; 668 of 3,024 per-frame anchors fell back to the part centroid when occluded), encoded to `outputs/hero/pigeon_osfp_dr8_explode_orbit.mp4` (H.264 yuv420p, 1080x1080, 30 fps, 450 frames, 15.000 s, video only; audio and `_bgm.mp4` untouched). Seam: frame 451 vs frame 1 mean abs diff 0.004 of 255; step 450 -> 1 = 1.82 vs 1 -> 2 = 1.84; 3x `-stream_loop` (45.0 s) seam steps 2.00 vs neighbours 1.84-1.92. Contact sheet `outputs/hero/work/explode_orbit_contact_sheet.png` viewed; frames deleted.

- 2026-10-10 02:49: current renders kept as `*_v2.png` and `pigeon_osfp_dr8_explode_orbit_loop15_v2.mp4` before the X v3 rerun.
- 2026-10-10 02:54-03:06: X v3 (02:51 files; DevLog-009 section 7: differential PD -> TIA input, hooks and part positions unchanged) integrated. Fit re-checked: unchanged (TX 0.974, RX 0.710); RX eye block 4.4 x 1.7 mm (half-diagonal 2.36 mm) already clears the 2.45 mm iris rim. RX flex lanes are now measured at build time from X's `rx_flex_pads` mesh (32 RF: G 0.09 wide at lane +-0.19, S 0.055 wide at lane +-0.05, i.e. differential pairs 0.10 mm centre-to-centre; 6 DC/control 0.20 wide at +-2.30/2.62/2.94; build asserts 32 + 6 and prints `RX_PADS rf=32 dc=6 pair_cc=0.100`). X's `explode_gltf` copies are stripped on import (0 in the hero GLB). Rebuilt: 381,278 tris, 250 meshes, 95 materials, 18 hooks; pin assertion OK; GLB re-import OK. Stills and loop re-rendered and viewed: seam frame 451 vs 1 = 0.004 of 255, step 450 -> 1 = 1.82 vs 1 -> 2 = 1.84, 3x loop seam steps 2.00 vs neighbours 1.85-1.92, 0 labels cut, 668 of 3,024 anchors fell back to centroids; frames deleted.

### 15.4 T / X integration (2026-10-10)

What the build does now (`scripts/assets/build_hero_pigeon.py`, section "T / X subassembly integration"):

- Removes the hero's own v1 TX/RX optics groups, eye lenslets, eye prisms and lane rays, then appends every object of `assets/components/optics_tx/optics_tx.blend` and `assets/components/optics_rx/optics_rx.blend` (no edits to T / X files). Library-relative textures (T's `pic_albedo.png`, `pic_mr.png`) are made absolute and packed so the hero .blend is self-contained.
- Placement by the interface frame: sub origin (eye outer surface) at A = (s (X_S + EYE_OFF), E_Y, E_Z); TX rotated 180 deg about Z (sub +X -> pigeon -X, sub +Y -> pigeon -Y, as T specified), RX identity (sub +X -> pigeon +X). All imported objects are flattened to children of the hero root so the renderers' explode offsets act in the pigeon frame.
- Hooks renamed `HOOK_tx_*` / `HOOK_rx_*` (5 + 6); `HOOK_eye_tx|rx` kept, moved onto the aperture (0.0000 mm match printed by the build), local +X out of the eye.
- Explode: group offset TX (-12, -6, 4) mm, RX (12, -6, 4) mm plus 0.7 x each part's own `explode` vector rotated into the pigeon frame. Beam viz meshes (`tx_beam_*`, `rx_viz_beam_*`) carry `viz_only = 1` and `hide_render = True` (1311 nm beams are invisible; none of the hero renders show them, so no beam note is needed). The hero's smooth-by-angle pass skips imported meshes (their shading is kept).

Head fit (scripts/assets/fit_eye_subassy.py, numbers from the GLBs):

| item | v2 value | grade | note |
|---|---|---|---|
| eye centre | Y = HC.y - 5.0 = -49.0, Z = HC.z = 81.5 (was HC.y - 3.5, HC.z - 1.0) | D | grid search over E_Y, E_Z, offset, yaw |
| aperture offset | 1.0 mm inside the skin mid-surface (eye socket) | D | needed by T's 02:21 carrier (top face at sub X -1.75, 13.3 x 9.2 mm) |
| yaw | 0 (optical axis along +-X) | D | yaw toward the surface normal fitted worse |
| worst ellipsoid level outside the eye socket (r 4.2, 1.6 mm deep) | TX 0.974, RX 0.710 (inner skin minus 0.25 mm clearance; < 1 = inside) | - | binding: TX carrier corners; TX carrier inner face X -7.14 vs head-board edge -6.8 |
| head semi-axes | unchanged (12, 14, 12) | D | enlarging does not help: the aperture moves out with the surface |
| eye dressing | glow rim r 2.45-2.62, black iris r 2.45-3.25 (front 1.1 mm proud of the aperture), chrome eye cere r 3.25-4.40 (covers the RX carrier front edge, corner radius 3.69) | D | stylized |
| skin eye opening | boolean cut r 4.05 (cere outer - 0.35) in both head halves, baked | D | skin overlaps the cere rim |

Line-side flexes rerouted to the hooks (pad positions from T's parts JSON and measured from X's `rx_flex_pads`):

| flex | traces | width | route | connector (head-board underside) |
|---|---|---|---|---|
| `flex_tx_rf_dsp_to_pic` | 25 (GSSG x8 + shared G, 0.27 pitch = T's rf_pads) | 7.15 mm | on the carrier top-face pads at HOOK_tx_flex_rf, exits pigeon +Y past the carrier back edge, bends about its width axis (Z) to run inward behind the carrier | J1 `conn_head_tx_rf`, vertical riser X -6.75..-6.05 |
| `flex_tx_dc_pic_laser` | 32 (4 laser + 28 PIC DC = T's laser_nets + dc_nets) | 6.11 mm | on the -Z edge pads at HOOK_tx_flex_dc, exits down, wraps under the carrier (bend about Y), rises at X -6.1 | J6 `conn_head_tx_dc`, X -6.6..-5.6 |
| `flex_rx_tia_to_dsp` | 38 (8 differential GSSG at 0.50 lane pitch, S pair 0.10 c-c, + 6 DC/ctrl at +-2.30/2.62/2.94; measured from X's pads at build time) | 6.30 mm | on the pads at HOOK_rx_flex_rf/dc/ctrl (one flex covers all three groups), exits pigeon -X, bends up | J2 `conn_head_rx`, X 2.3..3.3 |

Knock-on changes (all D): head board front edge extended 0.7 mm (to HC.y - 8.7, chamfered corners) for J2; MCU moved to (-1.3, HC.y + 1.5) to clear J2 and J6; board-bottom decal refdes J1 / J2 / J6 and ENIG RF fan-out redrawn; the unused `silk_tx_carrier` / `silk_rx_carrier` decals dropped (T / X carry their own carrier artwork). The 60-pin feet, belly plate, pin assertion and GLB re-import check are unchanged and pass.

Labels (stills and loop): TX PIC -> `tx_pic_siph_die`; 2x CW DFB -> `tx_dfb_laser_die_LD1/LD2`; "TX eye: 2x4 plate + TIR prism"; "RX eye: 2x4 lenslets + Si plates"; PD -> `rx_pin_pd_array_1x8` + `rx_pd_submount_aln`; octal TIA -> `rx_octal_tia`; "TX flexes: 8 RF (GSSG) + 32 DC"; new "RX flex: 8 RF (GSSG) + 6 DC" (24 labels).

Notes for T / X (not changed by H):

- T: the hook empties in `optics_tx.blend` have local axes that do not match the documented directions (HOOK_eye_aperture local X lies along sub +Y; HOOK_flex_rf / HOOK_flex_dc local X along sub +X instead of -Y / -Z). H uses hook positions only.
- X: `optics_rx.blend` hooks carry no `explode` property (fine; H strips explode from hooks anyway).

## 16. Film v2 hero fixes (coordinator, DevLog-010 section 9 items 5, 6, 8 + misc; 2026-10-10)

- [x] (1) hero signature for crowd shots
- [x] (2) keyable emission channels per side
- [x] (3) switchable parody DSP stamp
- [x] (4) pull tab as webbing with a slight curl
- [x] (5) fadeable wing heatsink panels
- [x] rebuild, GLB re-import check, re-render the affected stills (previous ones kept as `*_v3.png`)

Material interface for the film scene (Blender material names; the scene keys the listed input of the Principled BSDF unless noted):

| purpose | material | objects | key | default |
|---|---|---|---|---|
| hero leg band glow (amber) | `hero_leg_band_glow_amber` | `leg_tx_band_eeprom_carrier` | Emission Strength | 1.8, colour (1.0, 0.36, 0.02) |
| neck rim accent (hero only) | `hero_neck_rim_accent` | `skin_back_neck`, `skin_front_neck` | node `rim_strength` (Math multiply) input 1; drives Emission Strength through a Layer Weight Fresnel (blend 0.25), colour (0.25, 0.95, 0.75) | 0.8 |
| right foot pads (TX, red) | `hero_foot_pads_R_tx` | `foot_tx_edge_pads_x26`, `foot_tx_talon_tip_gnd` | Emission Strength (colour preset red) | 0.0 (plain gold) |
| left foot pads (RX, cyan) | `hero_foot_pads_L_rx` | `foot_rx_edge_pads_x26`, `foot_rx_talon_tip_gnd` | Emission Strength (colour preset cyan) | 0.0 |
| right eye rim (TX, red) | `hero_eye_rim_R_tx` | `eye_tx_iris_glow_ring` | Emission Strength | 3.0 |
| left eye rim (RX, cyan) | `hero_eye_rim_L_rx` | `eye_rx_iris_glow_ring` | Emission Strength | 3.0 |
| DSP lid stamp, NARROWCOM | `hero_dsp_mark_NARROWCOM` | `dsp_lid_marking` (assigned) | swap `bpy.data.objects["dsp_lid_marking"].data.materials[0]` | assigned |
| DSP lid stamp, MERVALL | `hero_dsp_mark_MERVALL` | none (fake user; texture `assets/hero/textures/dsp_lid_marking_mervall.png`) | assign as above | not exported to the GLB (glTF drops unused materials) |
| wing heatsink fins | `hero_wing_fins_fadeable` | `wing_right/left_flight_feather_fins` | Alpha (render method Dithered; 0.3 checked) | 1.0 |
| wing covert / heat spreader | `hero_wing_covert_fadeable` | `wing_right/left_covert_heat_spreader` | Alpha (Dithered) | 1.0 |
| pull tab | `hero_pull_tab_webbing` | `leash_pull_tab_strap`, `leash_end_loop` | none (amber nylon webbing: roughness 0.78, sheen 0.3, wave-texture weave bump at about 0.4 mm period) | - |

Geometry changes (all D, styling):

- Pull-tab strap rebuilt as webbing: cross-section cupped 0.6 mm, mid-length twist up to 25 deg (zero at both ends, so the root and the loop stay put), lengthwise wave 1.0 mm.
- Leg band moved from 0.68 to 0.50 of the hip-to-foot length, between the two leg clips: at 0.68 it sat inside a clip and under the flex, so it was invisible. Radii are now 2.45-2.95 mm (was 1.25-1.95) so it rings the tarsus and the leg flex. The EEPROM chip, the OUTNOLIGHT text and the EEPROM I2C flex start moved with it.
- Tail-tab material and the tail fan are unchanged otherwise. The tail fan keeps `fin_metal`; only the wing panels got the fadeable materials.

Checks: the build prints `DSP_MARKS ['hero_dsp_mark_MERVALL', 'hero_dsp_mark_NARROWCOM']`, the pin assertion passes, and `RX_PADS rf=32 dc=6`. The GLB re-imports with 250 meshes, all carrying `explode`, 98 materials and 18 hooks (unchanged). A scratch test render confirmed the MERVALL swap, wing alpha 0.3 and the pads at emission strength 4 (red right, cyan left).

Re-rendered and viewed: `pigeon_osfp_dr8_profile.png`, `_shell_transparent.png`, `_face_closeup.png`, `_exploded_labeled.png` and `_flex_routing.png`. `_feet_pins.png` and the 15 s loop were not re-rendered because their content is unchanged at default material values. One exception: the loop still shows the old band position, the old tab and the old band colour.

Note: the x-ray renders swap the skin materials, so the neck rim accent shows only in opaque shots.

- 2026-10-10 11:05: the 15 s loop was re-rendered with the film v2 hero (new tab, new band position and colour; default emission and alpha, no pulses); the previous loop is kept as `pigeon_osfp_dr8_explode_orbit_loop15_v3.mp4`. Same timing (450 frames). Checks: seam frame 451 vs frame 1 = 0.004 of 255; step 450 -> 1 = 1.81 vs 1 -> 2 = 1.82; 3x `-stream_loop` (45.0 s) seam step 1.99 vs neighbours 1.84-1.91; 126 label frames, 0 cut; 668 anchor centroid fallbacks. Output: H.264 yuv420p, 1080x1080, 30 fps, 450 frames, 15.000 s, video only. Contact sheet viewed; frames deleted.

## 17. Parody brand rename OUTNOLIGHT -> OUNOLIGHT (DevLog-018 decision V1; 2026-10-10 15:46)

- [x] Renamed in `scripts/assets/build_hero_pigeon.py` (6 occurrences):
  - leg-band text mesh: body "OUNOLIGHT", object renamed `leg_tx_band_text_OUTNOLIGHT` -> `leg_tx_band_text_OUNOLIGHT`;
  - head-board top silk "OUNOLIGHT";
  - head-board bottom silk "OUNOLIGHT HB-01 REV B";
  - both sternum silks "OUNOLIGHT ST-02 REV A".
- [x] Decal layout rewritten and the 12 textures regenerated: 0 OUTNOLIGHT left in `decal_layout.json`, the parts JSON or the GLB.
- [x] Rebuilt and checked:
  - 381,858 tris; pin assertion OK; `RX_PADS rf=32 dc=6`.
  - Both DSP stamp materials present.
  - GLB re-import: 250 meshes, all with `explode`, 98 materials.
  - All 10 `hero_*` film materials of section 16 are in the GLB (`hero_dsp_mark_MERVALL` stays blend-only by design).
- [x] Re-rendered the stills in which the band or a board marking can appear: profile, shell_transparent, face_closeup, exploded_labeled, flex_routing. The previous versions are kept as `*_v4.png`. In flex_routing the leg panel was viewed: band "OUNOLIGHT" and sternum "OUNOLIGHT ST-02 REV A" render correctly.
- Not re-rendered:
  - feet_pins: no marking in view.
  - 15 s loop: at its framing the band text is about 4 px tall and illegible.
- Earlier sections of this DevLog keep the old spelling as history.
