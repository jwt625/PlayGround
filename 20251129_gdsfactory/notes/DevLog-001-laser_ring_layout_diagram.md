# Laser to Ring Grid Layout Diagram

## Physical Layout (Top View)

```
Y-axis
  ↑
  │
1900 ┤
     │
1853 ┤                                                      Ring 1 (+3µm shift)
1850 ┤  Laser 0 ────────────────────→ Ring 0                o3 ← → o4
     │   (50, 1850)                    o3 ← → o4            ↑       ↑
     │                                  ↑       ↑            │       │
1800 ┤  Laser 1 ─────────────────────────────────────────→  │       │
     │   (50, 1800)                     │       │            │       │
     │                                 o1 ← → o2            o1 ← → o2
1750 ┤  Laser 2 ────────────────────→ Ring 2
     │   (50, 1750)                    o3 ← → o4
     │                                  ↑       ↑
1747 ┤                                  │       │            Ring 3 (-3µm shift)
1700 ┤  Laser 3 ─────────────────────────────────────────→ o3 ← → o4
     │   (50, 1700)                     │       │            ↑       ↑
     │                                 o1 ← → o2            │       │
1650 ┤  Laser 4 ────────────────────→                       │       │
     │   (50, 1650)                                          │       │
1604 ┤                                                       │       │
1601 ┤                                Ring 4                o1 ← → o2
     │                                o3 ← → o4            Ring 5 (+3µm shift)
1600 ┤  Laser 5 ─────────────────────────────────────────→ o3 ← → o4
     │   (50, 1600)                    ↑       ↑            ↑       ↑
     │                                 │       │            │       │
1550 ┤  Laser 6 ────────────────────→ │       │            │       │
     │   (50, 1550)                    │       │            │       │
1500 ┤  Laser 7 ────────────────────→ o1 ← → o2            o1 ← → o2
     │   (50, 1500)                   Ring 6               Ring 7 (-3µm shift)
1497 ┤                                o3 ← → o4            o3 ← → o4
1450 ┤
     │
     └────┴────────────────────────┴───────────────────┴───────────────────→ X-axis
          50                      300                 450


```

## Detailed Component Positions

### Lasers (8 emitters, 50 µm pitch)
Laser die positioned at (50, 1850) after 180° rotation to face right.
```
Laser 0: Y=1850 → outputs at 0° (pointing right)
Laser 1: Y=1800 → outputs at 0° (pointing right)
Laser 2: Y=1750 → outputs at 0° (pointing right)
Laser 3: Y=1700 → outputs at 0° (pointing right)
Laser 4: Y=1650 → outputs at 0° (pointing right)
Laser 5: Y=1600 → outputs at 0° (pointing right)
Laser 6: Y=1550 → outputs at 0° (pointing right)
Laser 7: Y=1500 → outputs at 0° (pointing right)
```

### Ring Grid (4 rows × 2 columns, variable row spacing, 150 µm column spacing)

Grid configuration:
- Column 0: X = 300 µm
- Column 1: X = 450 µm (with ±3 µm Y-shift for routing optimization)
- Row spacing: Variable (1850, 1750, 1601, 1500) for routing clearance

**Row 0 (Y = 1850):**
```
Ring 0 @ (300, 1850)              Ring 1 @ (450, 1853)  [+3µm Y-shift]
  Left column, even row             Right column, even row
```

**Row 1 (Y = 1750):**
```
Ring 2 @ (300, 1750)              Ring 3 @ (450, 1747)  [-3µm Y-shift]
  Left column, odd row              Right column, odd row
```

**Row 2 (Y = 1601):**
```
Ring 4 @ (300, 1601)              Ring 5 @ (450, 1604)  [+3µm Y-shift]
  Left column, even row             Right column, even row
```

**Row 3 (Y = 1500):**
```
Ring 6 @ (300, 1500)              Ring 7 @ (450, 1497)  [-3µm Y-shift]
  Left column, odd row              Right column, odd row
```

**Y-shift pattern for column 1 (right column):**
- Even rows (0, 2): +3 µm shift upward
- Odd rows (1, 3): -3 µm shift downward
- Purpose: Provides additional routing clearance and reduces waveguide congestion

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
- **o4**: Top-right, 0° (add to top bus)

## CROSSING-FREE ROUTING STRATEGY

### Laser to Ring Connections

**Routing pattern**: Alternating port assignments by row to minimize crossings and optimize routing density.

**Row 0 (Lasers 0, 1):** Connect to o3 ports (top-left), route ABOVE rings
```
Laser 0 ──→ Ring 1 o3  [crosses to right column, 2 waypoints]
Laser 1 ──→ Ring 0 o3  [goes to left column, 2 waypoints]
```

**Row 1 (Lasers 2, 3):** Connect to o2 ports (bottom-right), route BELOW rings
```
Laser 2 ──→ Ring 2 o2  [goes to left column, 4 waypoints]
Laser 3 ──→ Ring 3 o2  [crosses to right column, 4 waypoints]
```

**Row 2 (Lasers 4, 5):** Connect to o3 ports (top-left), route ABOVE rings
```
Laser 4 ──→ Ring 5 o3  [crosses to right column, 2 waypoints]
Laser 5 ──→ Ring 4 o3  [goes to left column, 2 waypoints]
```

**Row 3 (Lasers 6, 7):** Connect to o2 ports (bottom-right), route BELOW rings
```
Laser 6 ──→ Ring 6 o2  [goes to left column, 4 waypoints]
Laser 7 ──→ Ring 7 o2  [crosses to right column, 4 waypoints]
```

### Routing Implementation Details

**Routing parameters:**
- `dx_laser_base = 155 µm`: Horizontal offset from laser ports (30 + 125 µm shift for compactness)
- `dx_spacing = 15 µm`: Spacing between parallel vertical segments
- `dx_ring_approach = 30 µm`: Horizontal offset when approaching ring ports
- `dx_0 = 155 µm`: Used by lasers 0, 3, 4, 7
- `dx_1 = 170 µm`: Used by lasers 1, 2, 5, 6

**Waypoint strategy for o3 connections (lasers 0, 1, 4, 5):**
- 2 waypoints per route
- Waypoint 1: Vertical segment from laser at (laser_x + dx_offset, laser_y)
- Waypoint 2: Horizontal segment at ring port height (laser_x + dx_offset, ring_port_y)
- Approach ring o3 port from the left side

**Waypoint strategy for o2 connections (lasers 2, 3, 6, 7):**
- 4 waypoints per route
- Waypoint 1: Vertical segment from laser at (laser_x + dx_offset, laser_y)
- Waypoint 2: Horizontal routing layer at calculated Y position
- Waypoint 3: Vertical descent at (ring_port_x + dx_ring_approach - 10, waypoint_y)
- Waypoint 4: Final approach to ring port (ring_port_x + dx_ring_approach - 10, ring_port_y)
- 10 µm reduction in final approach for tighter routing

**Key routing practices:**
1. Separate vertical segments by dx_spacing to prevent waveguide overlap
2. Route above rings for o3 connections, below rings for o2 connections
3. Align horizontal segments to ring port Y-coordinates when possible
4. Use manual waypoints instead of automatic routing to ensure crossing-free layout
5. Maintain minimum 20 µm clearance for bend radius (2× 10 µm bend radius)

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
  [+25µm X-shift]                                                ↓
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
4. Ring 3 o4 → Ring 5 o2 (right column, top to bottom, row 1 to row 2) **[+25 µm X-shift to avoid crossings]**
5. Ring 5 o1 → Ring 4 o2 (right to left, bottom bus)
6. Ring 4 o1 → Ring 6 o3 (left column, bottom to top, row 2 to row 3)
7. Ring 6 o4 → Ring 7 o3 (left to right, top bus)
8. Ring 7 o4 → OUTPUT (final multiplexed output)

**Special routing for Ring 3 o4 → Ring 5 o2:**
- Uses 2 waypoints with 25 µm horizontal offset to shift vertical segment away from other routes
- Waypoint 1: (r3_o4_x + 25, r3_o4_y)
- Waypoint 2: (r5_o2_x + 25, r5_o2_y)
- Prevents waveguide crossings with laser-to-ring routes

**Signal flow summary:**
- Each laser adds its wavelength to its assigned ring via o3 or o2 port
- Wavelengths couple to the bus and follow the serpentine path
- All 8 wavelengths are multiplexed together
- Final output from Ring 7 o4 contains all wavelengths (λ0-λ7)

## Routing Method Summary

**gdsfactory routing approach:**
- Use `gf.routing.route_single()` with explicit waypoints for all connections
- Avoid automatic routing algorithms that may introduce crossings
- Manual waypoint planning ensures deterministic, crossing-free layout
- Position and rotate components first, then route with waypoints (do not use `connect()` after rotation)

**Best practices applied:**
1. Separate routing parameters for laser-side and ring-side offsets
2. Use consistent dx_offset values across similar route types
3. Plan horizontal routing layers to avoid vertical overlap
4. Add strategic position shifts (Y-shift for rings, X-shift for bus routing) to create routing clearance
5. Minimize waypoint count where possible (2 waypoints for direct routes, 4 for complex routes)

