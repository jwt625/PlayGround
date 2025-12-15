# DevLog-002: Demux + Modulation Module Design

## Objective

Create a modularized demux and modulation module that will be repeated 4 times in the transceiver chip. This module takes one input waveguide carrying 8 wavelengths and outputs 8 separate modulated waveguides.

## Module Architecture

### Input/Output
- **Input**: 1 waveguide carrying 8 multiplexed wavelengths (from 1×4 power splitter)
- **Output**: 8 separate modulated waveguides (to AWG inputs)

### Components
1. **8 add-drop ring resonators with PIN** (demultiplexing)
2. **8 PIN modulators** (data encoding)

## Layout Specifications

### Ring Array Configuration (1×8 Horizontal)
- **Topology**: 1 row × 8 columns (horizontal arrangement)
- **X pitch**: 110 µm (center-to-center, matching PIN modulator length)
- **Y offset**: 7 µm alternating pattern between adjacent rings
  - Even index rings (0, 2, 4, 6): base Y position
  - Odd index rings (1, 3, 5, 7): base Y + 7 µm
- **Purpose of Y offset**: Provides routing clearance and reduces waveguide congestion

### PIN Modulator Array Configuration (1×8 Horizontal)
- **Topology**: 1 row × 8 columns (horizontal arrangement)
- **X pitch**: 110 µm (matching ring array)
- **Y offset**: 7 µm alternating pattern between adjacent modulators
  - Even index modulators (0, 2, 4, 6): base Y position
  - Odd index modulators (1, 3, 5, 7): base Y + 7 µm
- **Vertical separation from rings**: To be determined based on routing requirements

### Component Parameters

**Ring (ring_double_pin):**
```python
gap=0.2
radius=10.0
length_x=20.0
length_y=50.0
via_stack_width=10.0
pin_on_left=True
```

**PIN Modulator (straight_pin):**
```python
length=100.0
```

## Ring Port Configuration

Each add-drop ring has 4 optical ports (counter-clockwise naming):
```
     o3 (left-top)  ← ─ ─ ─ → o4 (right-top)    [TOP BUS - Drop/Add]
        180°                      0°
         ↑                        ↑
         │      [RING with        │
         │       PIN on left]     │
         ↓                        ↓
     o1 (left-bottom) ← ─ ─ → o2 (right-bottom) [BOTTOM BUS - Input/Through]
        180°                      0°
```

## Routing Strategy

### Bus Waveguide Routing (Top Bus)
- **Direction**: Right to left (input from right side)
- **Path**: Input → Ring 7 o4 → Ring 7 o3 → Ring 6 o4 → Ring 6 o3 → ... → Ring 0 o3 → termination
- **Routing method**: S-bends between rings to accommodate 7 µm Y offset
- **Configuration**: Wavelengths couple from top bus, drop to bottom bus

### Drop Waveguide Routing (Bottom Bus)
- **Direction**: Each ring's dropped wavelength exits from bottom bus
- **Ports**: o1 or o2 (to be determined based on ring configuration)
- **Path**: Ring bottom port → PIN modulator input
- **Routing method**: Direct connection or waypoint-based routing

### Modulator Output Routing
- **Path**: PIN modulator output → AWG input
- **Note**: This routing will be handled at the chip level, not within the module

## Physical Layout

```
Input (from right) ──→ o4   o3 ← o4   o3 ← o4   o3 ← ... ← o4   o3
                       Ring7   Ring6   Ring5           Ring1   Ring0
                         ↓       ↓       ↓               ↓       ↓
                       Mod7    Mod6    Mod5            Mod1    Mod0
                         ↓       ↓       ↓               ↓       ↓
                    To AWG inputs (0-7)
```

## Implementation Plan

### Step 1: Create Module Function
- Function signature: `demux_modulation_module(base_x, base_y) -> gf.Component`
- Parameters:
  - `base_x`: Starting X position for the module
  - `base_y`: Base Y position for the module
- Returns: Component with all rings, modulators, and internal routing

### Step 2: Place Ring Array
- Calculate positions for 8 rings with 110 µm X pitch and 7 µm Y offset
- Place rings from left to right (Ring 0 at leftmost position)
- Store ring references for routing

### Step 3: Place Modulator Array
- Calculate positions for 8 modulators with 110 µm X pitch and 7 µm Y offset
- Position below ring array with appropriate vertical clearance
- Store modulator references for routing

### Step 4: Route Bus Waveguide (Top Bus)
- Create input port on the right side
- Route through all rings using S-bends to accommodate Y offsets
- Connect: Input → Ring 7 o4 → Ring 7 o3 → Ring 6 o4 → ... → Ring 0 o3

### Step 5: Route Drop Waveguides
- Connect each ring's drop port to corresponding modulator input
- Use waypoints if needed to avoid crossings

### Step 6: Create Output Ports
- Expose modulator outputs as module output ports
- Name ports: mod_0_out, mod_1_out, ..., mod_7_out

### Step 7: Testing
- Create test script to instantiate module
- Verify component placement
- Verify routing (no crossings, proper connections)
- Generate GDS output for visual inspection

## Expected Dimensions

- **Module width**: ~8 × 110 µm = 880 µm (plus margins)
- **Module height**: Ring height + vertical separation + modulator height + routing clearance
- **Estimated total height**: ~200-300 µm

## Notes

- The 7 µm Y offset is chosen to match the doping width, minimizing wasted space
- S-bends are used for top bus routing to handle the Y offset between rings
- This horizontal layout is much simpler to route than the 4×2 grid used for TX mux
- The module will be instantiated 4 times at Y positions: [2100, 1700, 1300, 900]

## Implementation Status

### Completed

1. **Module function created** (`demux_modulation_module.py`)
   - Function signature: `demux_modulation_module(base_x, base_y, ...)`
   - Returns component with 1 input port and 8 output ports

2. **Ring array placement**
   - 8 rings in 1×8 horizontal arrangement
   - 110 µm X pitch
   - 7 µm cumulative Y offset (Ring i at Y = base_y + i × 7 µm)
   - Component: `ring_double_pin` with standard parameters

3. **Modulator array placement**
   - 8 modulators in 1×8 horizontal arrangement
   - 110 µm X pitch
   - 7 µm cumulative Y offset (Modulator i at Y = base_y - 50 µm + i × 7 µm)
   - Vertical separation: 50 µm from rings (reduced from initial 150 µm)
   - Component: `straight_pin` with length=100 µm

4. **Bus waveguide routing (top bus)**
   - Input port: `bus_input` (exposed at Ring 7 o4)
   - S-bend connections between rings to handle 7 µm Y offset
   - Path: Input → Ring 7 → Ring 6 → ... → Ring 0

5. **Drop waveguide routing**
   - Each ring's o2 port connected to corresponding modulator o1 port
   - Direct routing using `route_single`

6. **Output alignment**
   - All modulator outputs extended to same X position (x_max + 10 µm)
   - Straight waveguide extensions added where needed

7. **AWG integration**
   - AWG positioned 50 µm from modulator outputs
   - Direct S-bend bundle routing from modulator outputs to AWG inputs
   - AWG rotated 90° and mirrored to align E ports (inputs) with modulators
   - AWG recombines 8 modulated wavelengths into single output

### Module Specifications

- **Dimensions**: 1015.8 × 181.6 µm
- **Input port**: `bus_input` (8 multiplexed wavelengths)
- **Output port**: `awg_out` (8 recombined modulated wavelengths)
- **Total ports**: 2 (1 input, 1 output)

### Design Decisions

1. **Y offset**: 7 µm chosen to match doping width, minimizing space while providing routing clearance
2. **Ring-to-modulator separation**: 50 µm provides adequate clearance for drop waveguide routing
3. **S-bends for top bus**: Explicit S-bend components used instead of automatic routing for predictable layout
4. **Direct modulator-to-AWG routing**: S-bend bundle handles pitch transition from 110 µm to 1.357 µm in 50 µm length
5. **AWG pitch**: 1.357 µm determined by FPR geometry (width1=10 µm, inputs=8)

### Testing

- GDS file generated: `demux_modulation_module.gds`
- Visual inspection: Layout verified, no crossings detected
- Port count: 2 ports (1 input, 1 output) as expected
- AWG orientation verified: E ports face left at 180°

### Next Steps

1. Integrate module into transceiver chip layout
2. Connect 1×4 splitter outputs to 4 module inputs
3. Connect module AWG outputs to chip-level combiner
4. Add electrical routing for ring and modulator control signals

## Implementation Results

### Module Created Successfully
- **File**: `demux_modulation_module.py`
- **GDS Output**: `demux_modulation_module.gds`
- **Module Dimensions**: 901.5 × 259.6 µm

### Port Summary
- **Input**: `bus_input` (right side, aligned with Ring 7 o4)
- **Outputs**: `mod_0_out` to `mod_7_out` (8 modulator outputs)
- **Total Ports**: 9 optical ports

### Layout Verification
- 8 rings arranged horizontally with 110 µm pitch
- 8 modulators arranged horizontally below rings
- Alternating 7 µm Y offset for both rings and modulators
- Bus waveguide routed right-to-left through top bus (o4 → o3 chain)
- Drop waveguides connect ring o2 ports to modulator o1 ports
- Automatic S-bend routing handles Y offsets between components

### Next Steps
1. Integrate module into transceiver_chip_mvp.py
2. Replace existing vertical demux + modulator placement with 4 instances of this module
3. Route from 1×4 splitter outputs to module inputs
4. Route from module outputs to AWG inputs

