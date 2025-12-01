# Electrical Port Routing Plan

**Date:** 2025-12-01  
**Project:** Photonic Integrated Circuit - Heater Bond Pad Routing  
**File:** test.py  

---

## Overview

This document describes the electrical routing plan for connecting heater ports to bond pads on the photonic chip. The design contains 11 heaters (4 standalone, 3 MZI heaters in circuits, 4 stacked MZI heaters), each requiring 2 electrical connections, resulting in 22 bond pads total.

---

## Chip Dimensions

**Pattern Extent (before bond pads):**
- X: [-321.5, 1140.2] μm
- Y: [-330.6, 584.6] μm

**Bond Pad Placement:**
- Edge buffer: 500 μm from pattern extent
- LEFT edge: x = -821.5 μm
- BOTTOM edge: y = -830.6 μm

---

## Bond Pad Specifications

**Parameters:**
- Pad size: 80 × 80 μm
- Pad pitch: 100 μm
- Layer: M3 (49/0)
- Port width: 40 μm

**Distribution:**
- LEFT edge: 12 pads (y: -160 to 940 μm)
- BOTTOM edge: 10 pads (x: 40 to 940 μm)

---

## Heater Port Grouping

Each heater has 8 electrical ports (4 on left end, 4 on right end) that are merged into 2 bond pads per heater.

### Standalone Heaters (4 heaters, 8 bond pads)

1. **Heater 1** (x ≈ -316 μm)
   - Left end: heater_1_l_e1, e2, e3, e4 (y ≈ 3 μm)
   - Right end: heater_1_r_e1, e2, e3, e4 (y ≈ 121 μm)

2. **Heater 2** (x ≈ -291 μm)
   - Left end: heater_2_l_e1, e2, e3, e4 (y ≈ 40 μm)
   - Right end: heater_2_r_e1, e2, e3, e4 (y ≈ 159 μm)

3. **Heater 3** (x ≈ 9 μm)
   - Left end: heater_3_l_e1, e2, e3, e4 (y ≈ 40 μm)
   - Right end: heater_3_r_e1, e2, e3, e4 (y ≈ 159 μm)

4. **Heater 4** (x ≈ 309 μm)
   - Left end: heater_4_l_e1, e2, e3, e4 (y ≈ 40 μm)
   - Right end: heater_4_r_e1, e2, e3, e4 (y ≈ 159 μm)

### Circuit MZI Heaters (3 circuits, 6 bond pads)

5. **Circuit 1 MZI** (y ≈ -21 μm)
   - Section 1: circuit_1_mzi_e1, e3, e6, e8 (x ≈ -35 μm)
   - Section 2: circuit_1_mzi_e2, e4, e5, e7 (x ≈ -206 μm)

6. **Circuit 2 MZI** (y ≈ -21 μm)
   - Section 1: circuit_2_mzi_e1, e3, e6, e8 (x ≈ 265 μm)
   - Section 2: circuit_2_mzi_e2, e4, e5, e7 (x ≈ 94 μm)

7. **Circuit 3 MZI** (y ≈ -21 μm)
   - Section 1: circuit_3_mzi_e1, e3, e6, e8 (x ≈ 565 μm)
   - Section 2: circuit_3_mzi_e2, e4, e5, e7 (x ≈ 394 μm)

### Stacked MZI Heaters (4 MZIs, 8 bond pads)

8. **Stacked MZI 1** (y ≈ 301 μm)
   - Section 1: stacked_mzi_1_e1, e3, e6, e8 (x ≈ -255 μm)
   - Section 2: stacked_mzi_1_e2, e4, e5, e7 (x ≈ -84 μm)

9. **Stacked MZI 2** (y ≈ 272 μm)
   - Section 1: stacked_mzi_2_e1, e3, e6, e8 (x ≈ -4 μm)
   - Section 2: stacked_mzi_2_e2, e4, e5, e7 (x ≈ 167 μm)

10. **Stacked MZI 3** (y ≈ 244 μm)
    - Section 1: stacked_mzi_3_e1, e3, e6, e8 (x ≈ 247 μm)
    - Section 2: stacked_mzi_3_e2, e4, e5, e7 (x ≈ 418 μm)

11. **Stacked MZI 4** (y ≈ 215 μm)
    - Section 1: stacked_mzi_4_e1, e3, e6, e8 (x ≈ 498 μm)
    - Section 2: stacked_mzi_4_e2, e4, e5, e7 (x ≈ 669 μm)

---

## Bond Pad Assignment

### LEFT Edge Pads (x = -781.5 μm)

| Pad # | Bond Pad Name           | Y Position | Source Ports                    | Avg Position      |
|-------|-------------------------|------------|---------------------------------|-------------------|
| 1     | circuit_1_mzi_1         | -160 μm    | circuit_1_mzi_e1,e3,e6,e8      | (-35, -21)        |
| 2     | circuit_1_mzi_2         | -60 μm     | circuit_1_mzi_e2,e4,e5,e7      | (-206, -21)       |
| 3     | circuit_2_mzi_2         | 40 μm      | circuit_2_mzi_e2,e4,e5,e7      | (94, -21)         |
| 4     | heater_1_left           | 140 μm     | heater_1_l_e1,e2,e3,e4         | (-316, 3)         |
| 5     | heater_2_left           | 240 μm     | heater_2_l_e1,e2,e3,e4         | (-291, 40)        |
| 6     | heater_3_left           | 340 μm     | heater_3_l_e1,e2,e3,e4         | (9, 40)           |
| 7     | heater_1_right          | 440 μm     | heater_1_r_e1,e2,e3,e4         | (-316, 121)       |
| 8     | heater_2_right          | 540 μm     | heater_2_r_e1,e2,e3,e4         | (-291, 159)       |
| 9     | heater_3_right          | 640 μm     | heater_3_r_e1,e2,e3,e4         | (9, 159)          |
| 10    | stacked_mzi_2_1         | 740 μm     | stacked_mzi_2_e1,e3,e6,e8      | (-4, 272)         |
| 11    | stacked_mzi_1_1         | 840 μm     | stacked_mzi_1_e1,e3,e6,e8      | (-255, 301)       |
| 12    | stacked_mzi_1_2         | 940 μm     | stacked_mzi_1_e2,e4,e5,e7      | (-84, 301)        |

### BOTTOM Edge Pads (y = -790.6 μm)

| Pad # | Bond Pad Name           | X Position | Source Ports                    | Avg Position      |
|-------|-------------------------|------------|---------------------------------|-------------------|
| 1     | stacked_mzi_2_2         | 40 μm      | stacked_mzi_2_e2,e4,e5,e7      | (167, 272)        |
| 2     | stacked_mzi_3_1         | 140 μm     | stacked_mzi_3_e1,e3,e6,e8      | (247, 244)        |
| 3     | circuit_2_mzi_1         | 240 μm     | circuit_2_mzi_e1,e3,e6,e8      | (265, -21)        |
| 4     | heater_4_left           | 340 μm     | heater_4_l_e1,e2,e3,e4         | (309, 40)         |
| 5     | heater_4_right          | 440 μm     | heater_4_r_e1,e2,e3,e4         | (309, 159)        |
| 6     | circuit_3_mzi_2         | 540 μm     | circuit_3_mzi_e2,e4,e5,e7      | (394, -21)        |
| 7     | stacked_mzi_3_2         | 640 μm     | stacked_mzi_3_e2,e4,e5,e7      | (418, 244)        |
| 8     | stacked_mzi_4_1         | 740 μm     | stacked_mzi_4_e1,e3,e6,e8      | (498, 215)        |
| 9     | circuit_3_mzi_1         | 840 μm     | circuit_3_mzi_e1,e3,e6,e8      | (565, -21)        |
| 10    | stacked_mzi_4_2         | 940 μm     | stacked_mzi_4_e2,e4,e5,e7      | (669, 215)        |

---

## Routing Strategy

**Edge Assignment Logic:**
- Ports with x < 100 μm route to LEFT edge
- Ports with x >= 100 μm route to BOTTOM edge

**Routing Method:**
- Metal layer: M3 (49/0)
- Trace width: 11-40 μm (to be determined)
- Routing approach: Manhattan routing with minimal bends
- Each bond pad will connect to all 4 ports in its group via metal traces

---

## Next Steps

1. Implement electrical routing from heater ports to bond pads
2. Add metal traces on M3 layer connecting grouped ports to their assigned bond pads
3. Verify design rules for metal trace width and spacing
4. Check for routing conflicts with optical waveguides
5. Generate final GDS with complete electrical routing

---

## Status

- [x] Heater port identification and grouping
- [x] Bond pad placement with 500 μm buffer
- [x] Bond pad assignment to heater groups
- [ ] Electrical trace routing implementation
- [ ] Design rule verification
- [ ] Final layout review

