# Silicon Photonics Transceiver Chip Layout Specification

## Overview
Mock silicon photonics transceiver chip with 8 wavelength channels and 4 spatial channels for transmit and receive paths.

## Chip Organization

### Left Edge Components
- **Top-Left**: Laser die (8 lasers, wavelengths λ1-λ8)
- **Below Laser Die**: Detector die for laser power monitoring (8 detectors)
- **Bottom-Left**: Four detector chips for receive path (4 chips × 8 detectors = 32 detectors)

### Right Edge Components
- **Transmit**: 4 edge couplers for output to fiber array
- **Receive**: 4 edge couplers for input from fiber array

## Transmit Path (Left to Right)

### Stage 1: Laser Source
- **Component**: 8 lasers (different wavelengths λ1-λ8)
- **Location**: Top-left of chip
- **Output**: 8 separate waveguides

### Stage 2: Wavelength Multiplexing
- **Component**: 8 add-drop ring resonators with integrated PIN junctions
- **Configuration**: Add-drop (double bus) - each laser couples from side port, drops to through bus
- **Function**: Combine 8 wavelengths onto single bus waveguide
- **Rationale**: Avoid waveguide crossings when splitting to 4 spatial channels
- **Input**: 8 laser waveguides (side ports o4) + 1 bus waveguide (o1)
- **Output**: 1 bus waveguide (o2) carrying all 8 wavelengths
- **Control**: 8 PIN modulators for ring resonance tuning (wavelength matching)

### Stage 3: Spatial Channel Splitting
- **Component**: 1×4 power splitter
- **Input**: 1 bus waveguide (8 wavelengths multiplexed)
- **Output**: 4 waveguides (each carrying all 8 wavelengths)

### Stage 4: Wavelength Demultiplexing
- **Component**: 4 sets of 8 add-drop ring resonators with integrated PIN junctions (32 rings total)
- **Configuration**: Add-drop (double bus) - each ring drops one wavelength to separate waveguide
- **Function**: Demultiplex each spatial channel into 8 wavelength channels
- **Input**: 4 bus waveguides (each with 8 wavelengths, o1 ports)
- **Output**: 32 drop waveguides (o3 ports) + 4 through waveguides (o2 ports)
- **Control**: 32 PIN modulators for ring resonance tuning

### Stage 5: Data Modulation
- **Component**: 32 PIN modulators
- **Function**: Encode data onto each wavelength channel
- **Input**: 32 waveguides
- **Output**: 32 modulated waveguides
- **Control**: 32 electrical drive signals

### Stage 6: Wavelength Recombining
- **Component**: 4 AWGs (Arrayed Waveguide Gratings)
- **Function**: Recombine 8 wavelengths back to single waveguide per spatial channel
- **Input**: 32 waveguides (4 groups of 8)
- **Output**: 4 waveguides (one per spatial channel)

### Stage 7: Fiber Coupling
- **Component**: 4 edge couplers
- **Location**: Right edge of chip
- **Input**: 4 waveguides
- **Output**: Coupled to fiber array

## Receive Path (Right to Left)

### Stage 1: Fiber Coupling
- **Component**: 4 edge couplers
- **Location**: Right edge of chip
- **Input**: Fiber array (4 fibers)
- **Output**: 4 waveguides on chip

### Stage 2: Wavelength Demultiplexing
- **Component**: 4 AWGs (Arrayed Waveguide Gratings)
- **Function**: Demultiplex each input into 8 wavelength channels
- **Input**: 4 waveguides
- **Output**: 32 waveguides (4 spatial × 8 wavelength channels)

### Stage 3: Detection
- **Component**: 4 detector chips (8 detectors each)
- **Location**: Bottom-left of chip
- **Input**: 32 waveguides routed from AWGs
- **Output**: 32 electrical signals

## Component Summary

### Active Components
| Component Type | Quantity | Purpose |
|---------------|----------|---------|
| Laser emitters | 8 | Wavelength sources (λ1-λ8) |
| Add-drop ring resonators with PIN (TX mux) | 8 | Wavelength multiplexing onto bus |
| Add-drop ring resonators with PIN (TX demux) | 32 | Wavelength demultiplexing (4×8) |
| PIN modulators | 32 | Data encoding |
| Photodetectors (monitoring) | 8 | Laser power monitoring |
| Photodetectors (receive) | 32 | Signal detection (4 chips × 8) |

### Passive Components
| Component Type | Quantity | Purpose |
|---------------|----------|---------|
| 1×4 power splitter | 1 | Spatial channel splitting |
| AWG (TX recombining) | 4 | Wavelength recombining (8→1 each) |
| AWG (RX demultiplexing) | 4 | Wavelength demultiplexing (1→8 each) |
| Edge couplers (TX) | 4 | Transmit fiber coupling |
| Edge couplers (RX) | 4 | Receive fiber coupling |

### Total Component Count
- **Lasers**: 8
- **Add-drop ring resonators with PIN**: 40 (8 for mux + 32 for demux)
- **PIN modulators**: 32 (for data encoding)
- **Photodetectors**: 40 (8 monitoring + 32 receive)
- **Power splitters**: 1 (1×4)
- **AWGs**: 8 (4 TX + 4 RX)
- **Edge couplers**: 8 (4 TX + 4 RX)

## Electrical Connections

### Transmit Path
- 8 electrical connections for laser drive current
- 8 electrical connections for TX multiplexing ring tuning
- 32 electrical connections for TX demultiplexing ring tuning
- 32 electrical connections for data modulator drive
- 8 electrical connections for monitoring detector readout

### Receive Path
- 32 electrical connections for detector readout

### Total Electrical Connections
- **Transmit**: 80 connections
- **Receive**: 32 connections
- **Total**: 112 electrical connections

## Design Notes

1. **Add-Drop Ring Configuration**: All wavelength multiplexing and demultiplexing rings use add-drop (double bus) configuration:
   - TX Mux: Lasers couple from side port (o4), resonant wavelength drops to through bus (o1→o2)
   - TX Demux: Bus waveguide enters (o1), each wavelength drops to separate port (o3), remaining wavelengths exit (o2)
   - PIN modulator on left arm enables thermal tuning to match laser wavelengths

2. **Wavelength Multiplexing Strategy**: The initial multiplexing of 8 lasers onto a single bus waveguide using add-drop ring resonators avoids complex waveguide crossings when splitting to 4 spatial channels.

3. **Spatial Channels**: The 4 spatial channels allow parallel processing of the same 8 wavelength set, effectively creating a 4×8 = 32 channel system.

4. **Symmetry**: The transmit path uses AWGs to recombine wavelengths after modulation, while the receive path uses AWGs to demultiplex incoming signals. This creates a symmetric architecture.

5. **Monitoring**: Dedicated detector die next to laser die enables real-time power monitoring and wavelength stabilization feedback.

6. **Scalability**: The modular design with 4 spatial channels and 8 wavelength channels can be scaled by adjusting either dimension.

## Component Dimensions (Measured)

### Active Components
| Component | Width (µm) | Height (µm) | Notes |
|-----------|------------|-------------|-------|
| Laser bar (8 emitters) | 320 | 435 | 50 µm pitch between emitters |
| Single detector | 170 | 25 | Ge photodetector |
| Detector array (8×) | 170 | 375 | 50 µm pitch, for monitoring |
| Detector chip (8×) | 170 | 375 | 50 µm pitch, for receive |
| Add-drop ring with PIN | 55 | 72 | Radius 10 µm, double bus config |
| PIN modulator | 100 | 22 | Straight modulator |

### Passive Components
| Component | Width (µm) | Height (µm) | Notes |
|-----------|------------|-------------|-------|
| 1×4 Splitter tree | 206 | 76 | Binary tree topology |
| AWG (8 channels) | 104 | 65 | 20 arms, 8 outputs |
| Edge coupler | 100 | 0.5 | Fiber coupling interface |
| MMI 1×2 | 26 | 2.5 | Alternative splitter option |

## Placement Plan

### Chip Dimensions (Estimated)
- **Total width**: ~3000 µm (3 mm)
- **Total height**: ~2500 µm (2.5 mm)

### Left Side (X: 0-500 µm)
**Top-Left Section (Y: 1800-2300 µm)**
- Laser die: 320×435 µm at (50, 1850)
- Output ports facing right (0° orientation)

**Mid-Left Section (Y: 1200-1600 µm)**
- Monitoring detector array: 170×375 µm at (50, 1250)
- Input ports facing right to receive laser taps

**Bottom-Left Section (Y: 0-1250 µm)**
- RX Detector chip 1: 170×375 µm at (50, 50)
- RX Detector chip 2: 170×375 µm at (50, 450)
- RX Detector chip 3: 170×375 µm at (50, 850)
- RX Detector chip 4: 170×375 µm at (50, 1250)
- Spacing: 400 µm vertical
- Note: All chips aligned in same column (x=50) for parallel waveguide routing

### Transmit Path (X: 500-2800 µm)
**Stage 1: TX Multiplexing (X: 500-700 µm)**
- 8 add-drop ring resonators with PIN: arranged vertically
- Configuration: Lasers couple from side (o4), drop to bus (o1→o2)
- Vertical spacing: 80 µm pitch (larger rings)
- Position: (500, 1850) to (500, 2410)
- Bus waveguide runs through all rings (bottom coupler)

**Stage 2: Spatial Splitting (X: 800-1000 µm)**
- 1×4 splitter tree: 206×76 µm at (800, 2000)
- Input from multiplexed bus
- 4 outputs with 25 µm vertical spacing

**Stage 3: TX Demultiplexing (X: 1100-1300 µm)**
- 4 groups of 8 add-drop rings (32 total)
- Configuration: Bus enters o1, each wavelength drops to o3, through exits o2
- Group 1: (1100, 2100) - 8 rings vertical
- Group 2: (1100, 1700) - 8 rings vertical
- Group 3: (1100, 1300) - 8 rings vertical
- Group 4: (1100, 900) - 8 rings vertical
- Each group: 80 µm pitch, ~640 µm height

**Stage 4: Modulation (X: 1400-1600 µm)**
- 32 PIN modulators arranged in 4 groups
- Same vertical positions as demux rings
- Each modulator: 100×22 µm

**Stage 5: TX Recombining (X: 1700-1900 µm)**
- 4 AWGs (8→1 each): 104×65 µm
- AWG 1: (1700, 2100)
- AWG 2: (1700, 1700)
- AWG 3: (1700, 1300)
- AWG 4: (1700, 900)

**Stage 6: TX Edge Couplers (X: 2700-2800 µm)**
- 4 edge couplers: 100×0.5 µm
- Vertical positions: 2100, 1700, 1300, 900
- Right edge of chip for fiber array

### Receive Path (X: 2800-500 µm, right to left)
**Stage 1: RX Edge Couplers (X: 2700-2800 µm)**
- 4 edge couplers at right edge
- Vertical positions: 600, 400, 200, 0
- Separate from TX couplers

**Stage 2: RX Demultiplexing (X: 2400-2500 µm)**
- 4 AWGs (1→8 each): 104×65 µm
- AWG 1: (2400, 600)
- AWG 2: (2400, 400)
- AWG 3: (2400, 200)
- AWG 4: (2400, 0)

**Stage 3: Route to Detectors (X: 2400-250 µm)**
- 32 waveguides routed to 4 detector chips
- Routing requires careful planning to avoid crossings
- Each detector chip receives 8 waveguides

## Routing Considerations

1. **Waveguide Spacing**: Minimum 10 µm between parallel waveguides
2. **Bend Radius**: Use 10 µm radius Euler bends
3. **Crossing Avoidance**: Route RX path below TX path where possible
4. **Electrical Routing**: Not included in MVP, will be added later
5. **Alignment Marks**: Add at chip corners for fabrication alignment

## AWG Debug and Custom Implementation

### Problem with Built-in AWG

The gdsfactory built-in `awg()` component uses `route_bundle()` for automatic waveguide routing, which creates **irregular path length differences** between waveguides:

**Debug findings from default AWG (20 arms, 8 outputs):**
- Waveguide 0-2: ~178-190 µm (4 bends each)
- Waveguide 3-9: ~243-279 µm (8 bends, **+100 µm jump!**)
- Waveguide 10-18: ~206 µm (6 bends, **-36 µm drop!**)
- Waveguide 19: ~44 µm (2 bends, **-162 µm drop!**)

**Root cause:**
- `route_bundle()` is an automatic Manhattan router that changes routing strategies to avoid collisions with previously-routed waveguides
- With `arm_spacing=1.0` µm (default), waveguides are packed tightly, forcing later routes to take huge detours
- This creates the visual appearance of "waveguide overlaps" and inconsistent path lengths
- The irregular path lengths are problematic for AWG functionality, which requires precise phase relationships

**Why increasing `arm_spacing` didn't help:**
- Increasing `arm_spacing` (to 5, 10, 20 µm) just makes the detours larger
- The fundamental issue is the automatic routing algorithm, not the spacing parameter
- Larger spacing creates even longer waveguides (up to 856 mm for arm_spacing=20!)

### Custom AWG Solution

Created `awg_manual_route()` in `custom_awg_debug.py` with manual waveguide routing for consistent path lengths.

**Key design principles:**
1. **Port ordering**: Reverse input port order to avoid crossings (E0_out → E19_in, E1_out → E18_in, etc.)
2. **Detour assignment by horizontal distance**:
   - Sort waveguides by horizontal distance between ports
   - Shortest horizontal distance → lowest detour height (closest to FPR)
   - Longest horizontal distance → highest detour height
   - This ensures no waveguide crossings
3. **Small detour increments**: 1 µm increment between adjacent waveguides (by h_dist rank)
4. **Consistent routing pattern**: All waveguides follow same path: up → across → down

**Results for 20-arm AWG:**
- Consistent **~4 µm path length difference** between consecutive waveguides
- Waveguide lengths: 53.77 to 130.77 µm (much more reasonable than built-in AWG)
- Compact size: 70.0 x 54.2 µm
- No crossings or overlaps
- Clean, parallel waveguide routing

**Parameters:**
```python
awg_manual_route(
    arms=20,           # Number of waveguide arms
    outputs=8,         # Number of output channels
    fpr_spacing=50.0,  # Horizontal separation between FPRs (µm)
    delta_length=10.0, # Target path length difference (µm) - not fully achieved due to geometry
    cross_section="strip"
)
```

**Usage in MVP:**
The custom AWG is used in `transceiver_chip_mvp.py` for both TX and RX AWGs to ensure consistent, predictable waveguide routing without crossings.

