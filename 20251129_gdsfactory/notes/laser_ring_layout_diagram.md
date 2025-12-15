# Laser to Ring Grid Layout Diagram

## Physical Layout (Top View)

```
Y-axis
  ↑
  │
1900 ┤
     │
1850 ┤  Laser 0 ────────────────────→ Ring 0                Ring 1
     │   (70, 1850)                    o3 ← → o4            o3 ← → o4
     │                                  ↑       ↑            ↑       ↑
1800 ┤  Laser 1 ─────────────────────────────────────────→  │       │
     │   (70, 1800)                     │       │            │       │
     │                                 o1 ← → o2            o1 ← → o2
1750 ┤  Laser 2 ────────────────────→ Ring 2                Ring 3
     │   (70, 1750)                    o3 ← → o4            o3 ← → o4
     │                                  ↑       ↑            ↑       ↑
1700 ┤  Laser 3 ─────────────────────────────────────────→  │       │
     │   (70, 1700)                     │       │            │       │
     │                                 o1 ← → o2            o1 ← → o2
1650 ┤  Laser 4 ────────────────────→ Ring 4                Ring 5
     │   (70, 1650)                    o3 ← → o4            o3 ← → o4
     │                                  ↑       ↑            ↑       ↑
1600 ┤  Laser 5 ─────────────────────────────────────────→  │       │
     │   (70, 1600)                     │       │            │       │
     │                                 o1 ← → o2            o1 ← → o2
1550 ┤  Laser 6 ────────────────────→ Ring 6                Ring 7
     │   (70, 1550)                    o3 ← → o4            o3 ← → o4
     │                                  ↑       ↑            ↑       ↑
1500 ┤  Laser 7 ─────────────────────────────────────────→  │       │
     │   (70, 1500)                     │       │            │       │
     │                                 o1 ← → o2            o1 ← → o2
1450 ┤
     │
     └────┴────────────────────────┴───────────────────┴───────────────────→ X-axis
          70                      300                 450


```

## Detailed Component Positions

### Lasers (8 emitters, 50 µm pitch)
```
Laser 0: (70, 1850) → outputs at 0° (pointing right)
Laser 1: (70, 1800) → outputs at 0° (pointing right)
Laser 2: (70, 1750) → outputs at 0° (pointing right)
Laser 3: (70, 1700) → outputs at 0° (pointing right)
Laser 4: (70, 1650) → outputs at 0° (pointing right)
Laser 5: (70, 1600) → outputs at 0° (pointing right)
Laser 6: (70, 1550) → outputs at 0° (pointing right)
Laser 7: (70, 1500) → outputs at 0° (pointing right)
```

### Ring Grid (4 rows × 2 columns, 100 µm row pitch, 150 µm column spacing)

**Row 0 (Y = 1850):**
```
Ring 0 @ (300, 1850)              Ring 1 @ (450, 1850)
  o3 (167, 1921.4) ← → o4 (313, 1921.4)      o3 (317, 1921.4) ← → o4 (463, 1921.4)
       180°              0°                        180°              0°
       ↑                 ↑                         ↑                 ↑
       │   TOP BUS       │                         │   TOP BUS       │
       │                 │                         │                 │
       │    [RING 0]     │                         │    [RING 1]     │
       │                 │                         │                 │
       │  BOTTOM BUS     │                         │  BOTTOM BUS     │
       ↓                 ↓                         ↓                 ↓
  o1 (167, 1814.3) ← → o2 (313, 1814.3)      o1 (317, 1814.3) ← → o2 (463, 1814.3)
       180°              0°                        180°              0°
```

**Row 1 (Y = 1750):**
```
Ring 2 @ (300, 1750)              Ring 3 @ (450, 1750)
  o3 (167, 1821.4) ← → o4 (313, 1821.4)      o3 (317, 1821.4) ← → o4 (463, 1821.4)
  o1 (167, 1714.3) ← → o2 (313, 1714.3)      o1 (317, 1714.3) ← → o2 (463, 1714.3)
```

**Row 2 (Y = 1650):**
```
Ring 4 @ (300, 1650)              Ring 5 @ (450, 1650)
  o3 (167, 1721.4) ← → o4 (313, 1721.4)      o3 (317, 1721.4) ← → o4 (463, 1721.4)
  o1 (167, 1614.3) ← → o2 (313, 1614.3)      o1 (317, 1614.3) ← → o2 (463, 1614.3)
```

**Row 3 (Y = 1550):**
```
Ring 6 @ (300, 1550)              Ring 7 @ (450, 1550)
  o3 (167, 1621.4) ← → o4 (313, 1621.4)      o3 (317, 1621.4) ← → o4 (463, 1621.4)
  o1 (167, 1514.3) ← → o2 (313, 1514.3)      o1 (317, 1514.3) ← → o2 (463, 1514.3)
```

## Ring Port Configuration (Counter-Clockwise Naming)

Each add-drop ring has 4 optical ports:
```
     o3 (left-top)  ← ─ ─ ─ → o4 (right-top)    [TOP BUS - Drop/Add]
        180°                      0°
         ↑                        ↑
         │                        │
         │      [RING with        │
         │       PIN on left]     │
         │                        │
         ↓                        ↓
     o1 (left-bottom) ← ─ ─ → o2 (right-bottom) [BOTTOM BUS - Input/Through]
        180°                      0°
```

- **o1**: Bottom-left, 180° (input to bottom bus)
- **o2**: Bottom-right, 0° (through from bottom bus)
- **o3**: Top-left, 180° (drop from top bus)
- **o4**: Top-right, 0° (add to top bus) ← **LASER CONNECTS HERE**

## CROSSING-FREE ROUTING STRATEGY

### Laser to Ring Connections (Avoiding Crossings)

**Key principle**: Even lasers cross over to right column (staying on top), odd lasers go to left column (staying below)

```
Laser 0 (70, 1850) ──→ Ring 1 o4 (463, 1921.4)  [crosses over, stays on top]
Laser 1 (70, 1800) ──→ Ring 0 o4 (313, 1921.4)  [goes to left, stays below laser 0]

Laser 2 (70, 1750) ──→ Ring 3 o4 (463, 1821.4)  [crosses over, stays on top]
Laser 3 (70, 1700) ──→ Ring 2 o4 (313, 1821.4)  [goes to left, stays below laser 2]

Laser 4 (70, 1650) ──→ Ring 5 o4 (463, 1721.4)  [crosses over, stays on top]
Laser 5 (70, 1600) ──→ Ring 4 o4 (313, 1721.4)  [goes to left, stays below laser 4]

Laser 6 (70, 1550) ──→ Ring 7 o4 (463, 1621.4)  [crosses over, stays on top]
Laser 7 (70, 1500) ──→ Ring 6 o4 (313, 1621.4)  [goes to left, stays below laser 6]
```

### Ring Bus Connections (Serpentine Multiplexing Chain)

The rings are connected in a serpentine pattern to avoid crossings:

```
                    ┌─────────────────────────────────┐
                    │                                 ↓
Ring 1 o1 → Ring 0 o2    Ring 0 o1 → Ring 2 o3    Ring 2 o4 → Ring 3 o3
(right)     (left)       (left)      (left)       (left)      (right)
                                         ↓                        ↓
                                     [to row 1]              [to row 2]

                    ┌─────────────────────────────────┐
                    │                                 ↓
Ring 3 o4 → Ring 5 o2    Ring 5 o1 → Ring 4 o2    Ring 4 o1 → Ring 6 o3
(right)     (right)      (right)     (left)       (left)      (left)
                                                                  ↓
                                                              [to row 3]

                    ┌─────────────────────────────────┐
                    │                                 ↓
Ring 6 o4 → Ring 7 o3    Ring 7 o4 → OUTPUT
(left)      (right)      (right)
```

**Detailed serpentine path:**
1. Ring 1 o1 → Ring 0 o2 (right to left, bottom bus)
2. Ring 0 o1 → Ring 2 o3 (left column, bottom to top, row 0 to row 1)
3. Ring 2 o4 → Ring 3 o3 (left to right, top bus)
4. Ring 3 o4 → Ring 5 o2 (right column, top to bottom, row 1 to row 2)
5. Ring 5 o1 → Ring 4 o2 (right to left, bottom bus)
6. Ring 4 o1 → Ring 6 o3 (left column, bottom to top, row 2 to row 3)
7. Ring 6 o4 → Ring 7 o3 (left to right, top bus)
8. Ring 7 o4 → OUTPUT (final multiplexed output)

**Signal flow summary:**
- Each laser adds its wavelength to its assigned ring via o4 port
- Wavelengths drop to the bus and follow the serpentine path
- All 8 wavelengths are multiplexed together
- Final output from Ring 7 o4 contains all wavelengths (λ0-λ7)

