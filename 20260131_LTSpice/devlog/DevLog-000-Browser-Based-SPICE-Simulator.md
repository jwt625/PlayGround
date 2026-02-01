# DevLog-000: Browser-Based SPICE Circuit Simulator

Date: 2026-01-31

## Problem Statement

LTSpice is an excellent circuit simulator with a minimal, fast GUI. However, it is:
- Desktop-only (Windows native, Mac port available)
- Closed source (cannot be ported to web)
- Not shareable (circuits require LTSpice installation to view)

Goal: Investigate feasibility of a browser-based SPICE simulator with a minimal GUI similar to LTSpice.

## Research Summary

### LTSpice vs NGSpice

| Feature | LTSpice | NGSpice |
|---------|---------|---------|
| License | Proprietary freeware | Open source (BSD) |
| GUI | Built-in, excellent | Command-line (separate GUIs available) |
| Platform | Windows/Mac | Cross-platform |
| Speed | Very fast | Moderate |
| Web Portable | No | Yes (via WebAssembly) |

NGSpice is the only viable option for browser-based simulation due to its open source license.

### Existing Browser-Based Solutions

1. EasyEDA - Full-featured but bloated, not minimal
2. CircuitLab - Freemium, not open source
3. Falstad - Real-time but not SPICE-based
4. EEcircuit - NGSpice WASM, text-only input (no schematic GUI)

None of these provide a minimal LTSpice-like experience.

### EEcircuit Architecture Analysis

EEcircuit (https://github.com/eelab-dev/EEcircuit) demonstrates that NGSpice can run entirely client-side:

Architecture:
- NGSpice compiled to WebAssembly via Emscripten
- Published as npm package: eecircuit-engine
- Frontend: React + TypeScript + Chakra-UI
- Editor: Monaco (VS Code editor)
- Plotting: WebGL-Plot
- Web Workers: Comlink for async simulation
- Deployment: Static hosting (Vercel)

Key finding: No backend required. All simulation runs in browser.

## Technical Feasibility

### Confirmed Possible

1. NGSpice WASM compilation - Already done by EEcircuit team
2. Client-side simulation - Proven to work
3. Web Workers for non-blocking UI - Implemented via Comlink
4. WebGL plotting - Fast enough for real-time visualization

### Challenges

1. Schematic Editor - EEcircuit only has text input, no graphical schematic capture
2. File Size - NGSpice WASM is 2-5 MB, acceptable for modern web
3. Performance - WASM is slower than native, but adequate for most circuits

## High-Level Proposal

### Option A: Minimal GUI on Top of EEcircuit Engine

Use the existing eecircuit-engine npm package and build a minimal schematic editor.

Stack:
- Simulator: eecircuit-engine (NGSpice WASM)
- Frontend: Vanilla JS or Svelte (no React bloat)
- Editor: CodeMirror (lighter than Monaco) or custom textarea
- Schematic: HTML5 Canvas with minimal component library
- Plotting: WebGL-Plot or custom Canvas
- Build: Vite

Estimated effort: 2-4 weeks

### Option B: Full Custom Build

Compile NGSpice to WASM ourselves for full control over the build.

Additional work:
- Set up Emscripten toolchain
- Configure NGSpice build for WASM target
- Create JavaScript bindings
- Handle file system emulation for model files

Estimated effort: 4-6 weeks

### Recommended Approach

Start with Option A. Use eecircuit-engine to validate the concept, then consider Option B if customization is needed.

## Minimal Schematic Editor Requirements

To match LTSpice experience:

1. Component palette (R, L, C, V, I, diodes, transistors)
2. Wire drawing tool
3. Component rotation/mirroring
4. Node labeling
5. Netlist generation from schematic
6. Simulation control (.tran, .ac, .dc, .op)
7. Waveform viewer with zoom/pan

## File Size Estimates

| Component | Size |
|-----------|------|
| NGSpice WASM | 2-5 MB |
| Minimal JS framework | 50 KB |
| Schematic editor | 100-200 KB |
| Plotting library | 50-100 KB |
| Total | 2.5-5.5 MB |

Acceptable for modern web applications.

## Next Steps

1. Clone EEcircuit and run locally to understand the codebase
2. Extract eecircuit-engine usage patterns
3. Prototype minimal schematic editor with Canvas API
4. Integrate netlist generation
5. Connect to simulation engine
6. Build waveform viewer

## References

- EEcircuit: https://github.com/eelab-dev/EEcircuit
- EEcircuit Live: https://eecircuit.com
- NGSpice: https://ngspice.sourceforge.io/
- Emscripten: https://emscripten.org/
- WebGL-Plot: https://github.com/danchitnis/webgl-plot

---

## MVP Scope Decisions (2026-01-31)

### Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Framework | SvelteKit | Minimal, performant, reactive |
| Package Manager | pnpm | Fast, disk-efficient |
| Build Tool | Vite | Fast HMR, native ESM |
| Simulator | eecircuit-engine | NGSpice WASM, proven to work |
| Plotting | WebGL-Plot or custom Canvas | High performance, independent X/Y zoom |
| Editor | CodeMirror 6 | Lighter than Monaco, extensible |
| Styling | Vanilla CSS | Minimal footprint, LTSpice dark theme |

### UI Style Guidelines

**Sharp Corners Only**: All UI elements must have sharp corners (border-radius: 0) to match LTSpice's clean, technical aesthetic. This applies to:
- Tooltips
- Indicators (zoom mode, etc.)
- Legends
- Buttons
- Panels
- All other UI elements

No rounded corners anywhere in the application.

### Component Library

**MVP (No Import)**:
- Passives: R, L, C
- Sources: V (DC, AC, PULSE, SINE), I, GND
- Semiconductors: Diode, BJT (NPN/PNP), MOSFET (N/P)
- Use bundled models from eecircuit-engine (BSIM4, PTM, FreePDK45)

**Future**: Support user `.lib`/`.model` file import via virtual filesystem (Emscripten FS API, same approach as EEcircuit)

### Simulation Types

All simulation types are supported by NGSpice and passed through as netlist commands:

| Analysis | Command | MVP |
|----------|---------|-----|
| Transient | `.tran` | Yes |
| AC | `.ac` | Yes |
| DC Sweep | `.dc` | Yes |
| Operating Point | `.op` | Yes |
| Noise | `.noise` | Yes |
| Transfer Function | `.tf` | Yes |
| Parameter Sweep | `.step` | Yes |
| Measurement | `.meas` | Yes |

The frontend generates netlists; NGSpice handles all analysis types natively.

### File Format

**MVP**: LTSpice `.asc` file import/export

The `.asc` format is plain text with the following structure:
```
Version 4
SHEET 1 880 680
WIRE x1 y1 x2 y2
SYMBOL res x y R0
SYMATTR InstName R1
SYMATTR Value 1k
TEXT x y Left 2 !.tran 1m
```

Key elements:
- `WIRE x1 y1 x2 y2` - Wire segments
- `SYMBOL type x y rotation` - Component placement
- `SYMATTR` - Component attributes (name, value)
- `TEXT` with `!` prefix - SPICE directives

### Persistence

**MVP**: localStorage only
- Auto-save current schematic
- Named circuit slots
- No cloud, no URL sharing

### Waveform Viewer Features

| Feature | MVP | Notes |
|---------|-----|-------|
| Multiple traces | Yes | Color-coded |
| Zoom/Pan | Yes | Better than LTSpice |
| Independent X/Y zoom | Yes | Plotly-style controls |
| Cursors | Yes | Two cursors with delta readout |
| FFT view | Yes | For AC analysis |
| CSV export | Yes | Download button |
| Log scale | Yes | For frequency plots |

### Platform

**MVP**: Desktop only (1024px+ width)
**Future**: Mobile with component panel drawer

### Offline Support

**MVP**: Online only
**Future**: PWA with service worker

---

## LTSpice Keyboard Shortcuts (Target)

### Schematic Editor - Primary Keys

| Key | Function |
|-----|----------|
| R | Rotate component (while placing/selected) |
| Ctrl+R | Rotate selected |
| Ctrl+E | Mirror/flip horizontal |
| W | Wire mode |
| G | Ground |
| F2 | Place component |
| F3 | Wire mode |
| F4 | Net name/label |
| F5 | Delete |
| F6 | Copy/Duplicate |
| F7 | Move |
| F8 | Drag (move with wires) |
| F9 | Undo |
| Ctrl+Z | Undo |
| Ctrl+Y | Redo |
| Ctrl+S | Save |
| Ctrl+C | Copy |
| Ctrl+V | Paste |
| Ctrl+X | Cut |
| Delete | Delete selected |
| Escape | Cancel current operation |
| Space | Zoom to fit |

### Component Shortcuts

| Key | Component |
|-----|-----------|
| R (in component mode) | Resistor |
| C | Capacitor |
| L | Inductor |
| D | Diode |
| G | Ground |
| V | Voltage source |
| I | Current source |

### Simulation

| Key | Function |
|-----|----------|
| Ctrl+B / F5 (run) | Run simulation |
| Ctrl+H | Halt simulation |
| 0 | Reset simulation time |

### Waveform Viewer

| Key | Function |
|-----|----------|
| Space | Zoom to fit |
| Scroll | Zoom in/out |
| Click+Drag | Pan |
| Shift+Click | Add cursor |

---

## Implementation Plan

### Phase 1: Project Setup (Day 1)

1. Initialize SvelteKit project with pnpm
2. Configure Vite for WASM support
3. Install dependencies:
   - eecircuit-engine
   - webgl-plot
   - codemirror (v6)
   - comlink (for web workers)
4. Set up project structure:
   ```
   src/
     lib/
       components/     # Svelte components
       schematic/      # Schematic editor logic
       simulation/     # NGSpice wrapper
       waveform/       # Waveform viewer
       parser/         # ASC file parser
       stores/         # Svelte stores
     routes/
       +page.svelte    # Main app
   static/
     models/           # SPICE model files
   ```
5. Create dark theme CSS matching LTSpice

### Phase 2: Simulation Engine Integration (Day 2)

1. Create Web Worker wrapper for eecircuit-engine
2. Implement Comlink interface for async simulation
3. Test basic netlist execution
4. Parse simulation output (vectors, time series)
5. Handle simulation errors gracefully

### Phase 3: Netlist Editor (Day 3)

1. Integrate CodeMirror 6 with SPICE syntax highlighting
2. Create custom SPICE language mode
3. Add line numbers, error markers
4. Connect to simulation engine
5. Display simulation output/errors

### Phase 4: Schematic Canvas Foundation (Days 4-5)

1. Create Canvas-based schematic view
2. Implement coordinate system (grid snapping)
3. Pan and zoom controls
4. Selection system (click, box select)
5. Keyboard event handling

### Phase 5: Component System (Days 6-7)

1. Define component data model:
   ```typescript
   interface Component {
     id: string;
     type: ComponentType;
     x: number;
     y: number;
     rotation: 0 | 90 | 180 | 270;
     mirror: boolean;
     attributes: Record<string, string>;
     pins: Pin[];
   }
   ```
2. Create component renderers (SVG paths for each type)
3. Implement component placement
4. Rotation (R key) and mirroring (Ctrl+E)
5. Component property editing (double-click)

### Phase 6: Wire System (Days 8-9)

1. Wire data model with segments
2. Wire drawing mode (W key)
3. Auto-routing (Manhattan style)
4. Junction detection and rendering
5. Wire selection and deletion

### Phase 7: Netlist Generation (Day 10)

1. Build connectivity graph from schematic
2. Assign node numbers
3. Generate SPICE netlist from components
4. Handle subcircuits
5. Validate netlist before simulation

### Phase 8: Waveform Viewer (Days 11-13)

1. Create WebGL-based plot canvas
2. Implement trace rendering
3. X-axis zoom (scroll on X region)
4. Y-axis zoom (scroll on Y region)
5. Box zoom (click+drag)
6. Pan (right-click+drag or middle-click)
7. Cursor system with readouts
8. Legend with trace visibility toggles
9. CSV export

### Phase 9: ASC File Parser (Day 14)

1. Parse LTSpice .asc format
2. Map LTSpice symbols to internal components
3. Handle wire segments
4. Parse SPICE directives from TEXT elements
5. Export to .asc format

### Phase 10: Persistence and Polish (Day 15)

1. localStorage save/load
2. Auto-save on changes
3. Circuit naming and management
4. Undo/redo system
5. Error handling and user feedback
6. Performance optimization

### Phase 11: Testing and Refinement (Days 16-17)

1. Test with real circuits
2. Compare simulation results with LTSpice
3. Fix edge cases
4. Keyboard shortcut refinement
5. UI polish

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| eecircuit-engine API changes | Pin to specific version, fork if needed |
| WASM performance | Use Web Workers, optimize netlist size |
| Canvas performance | Use WebGL for waveforms, optimize redraws |
| ASC format complexity | Start with subset, expand as needed |
| Browser compatibility | Target modern browsers only (Chrome, Firefox, Safari) |

---

## Success Criteria

1. Can draw a simple RC circuit using keyboard shortcuts
2. Can run transient simulation and view waveform
3. Can zoom/pan waveform with independent X/Y control
4. Can save/load circuit from localStorage
5. Can import basic LTSpice .asc file
6. UI feels responsive and LTSpice-like

---

## Development Rules

1. **AI does NOT run the app** - The user runs the app and inspects results. AI writes code only.

---

## Implementation Status

### Phase 1-2: Complete (2026-01-31)
- Project setup: SvelteKit + pnpm + Vite
- Simulation engine: NGSpice WASM via eecircuit-engine, Web Worker + Comlink
- Waveform viewer: WebGL-based with webgl-plot
  - Pan (drag), zoom (scroll)
  - Box zoom (Z key)
  - Cursors A/B
  - Grid (G key toggle)
  - Autoscale (F key or double-click)
  - Tooltip on hover
  - Legend

### Code Cleanup (2026-01-31)
Reduced codebase by ~47% while maintaining all functionality:
- WaveformViewer.svelte: 905 -> 517 lines
- simulation/types.ts: 63 -> 36 lines
- styles/theme.css: 164 -> 41 lines
- waveform/types.ts: 69 -> 41 lines

### Phase 3: Netlist Editor - Complete (2026-02-01)
Integrated CodeMirror 6 with custom SPICE syntax highlighting:
- Created `src/lib/editor/spice-language.ts` - SPICE language mode
  - Comments (lines starting with `*` or `;`)
  - Directives (`.tran`, `.ac`, `.dc`, `.op`, `.model`, etc.)
  - Component names (`R1`, `C1`, `V1`, etc.)
  - Numbers with SI prefixes (`1k`, `10u`, `100n`, `1meg`)
  - Keywords (`PULSE`, `SINE`, `AC`, `DC`, etc.)
- Created `src/lib/editor/spice-theme.ts` - Dark theme matching LTSpice
  - Green comments, blue directives, yellow component names, light green numbers
- Created `src/lib/editor/NetlistEditor.svelte` - Svelte wrapper component
  - Line numbers, active line highlighting, undo/redo history
  - Two-way binding with `bind:value`

### Phase 4: Schematic Canvas Foundation - Complete (2026-02-01)
Created canvas-based schematic editor with pan/zoom/grid:
- Created `src/lib/schematic/types.ts` - Type definitions
  - `Point`, `ViewTransform`, `GridSettings`, `InteractionState`
  - `Component`, `Wire`, `Schematic`, `ComponentType`, `Rotation`, `Pin`
- Created `src/lib/schematic/SchematicCanvas.svelte` - Canvas component
  - Pan: Click and drag to pan the view
  - Zoom: Scroll wheel to zoom in/out (centered on mouse)
  - Grid: Dot grid at 10-unit spacing (toggle with Ctrl+G)
  - Origin crosshair: Visual reference at (0,0)
  - HUD: HTML overlay showing coordinates, zoom level, and mode hints
  - Keyboard shortcuts: Ctrl+G (grid), F/Home (reset view), +/- (zoom)
  - Coordinate conversion: screen to schematic and back
  - Grid snapping for component placement
  - ResizeObserver for proper canvas sizing without aspect ratio distortion

Added resizable and collapsible panel system:
- Created `src/lib/components/ResizablePanel.svelte`
  - Horizontal and vertical resize handles
  - Collapse/expand buttons with directional arrows
  - Panel headers with titles
- Updated main layout with three resizable panels:
  - Netlist panel (left, horizontal resize)
  - Schematic panel (top-right, vertical resize)
  - Waveform panel (bottom-right, vertical resize)

### Phase 5: Component System - Complete (2026-02-01)
Implemented full component rendering, placement, and manipulation:

#### Component Definitions (`src/lib/schematic/component-defs.ts`)
- Resistor (R key) - zigzag pattern
- Capacitor (C key) - parallel plates
- Inductor (L key) - coil pattern
- Ground (G key) - standard ground symbol
- Voltage Source (V key) - circle with +/- signs
- Current Source (I key) - circle with arrow
- Diode (D key) - triangle with bar
- NPN BJT (Q key) - transistor with arrow out
- PNP BJT (Shift+Q) - transistor with arrow in
- NMOS (M key) - MOSFET symbol
- PMOS (Shift+M) - PMOS with bubble
- Each component has: paths (draw commands), pins, label/value offsets, bounding box

#### Component Renderer (`src/lib/schematic/component-renderer.ts`)
- Renders components on canvas with rotation and mirror transforms
- Ghost preview for placement mode (semi-transparent)
- Selection highlighting (yellow)
- Labels counter-rotate to stay readable regardless of component rotation
- Hit testing for component selection

#### Component Placement
- Press component shortcut key to enter placement mode
- Ghost preview follows cursor (snapped to grid)
- Click to place component
- Can switch components while placing (e.g., R then C switches to capacitor)
- Auto-generated instance names (R1, R2, C1, etc.)
- Default values assigned (1k for resistors, 1u for capacitors, etc.)

#### Component Selection and Manipulation
- Click on component to select (yellow highlight)
- Shift+click for multi-select
- Ctrl+R to rotate (0 -> 90 -> 180 -> 270 degrees)
- Ctrl+E to mirror/flip horizontally
- Delete/Backspace to delete selected components
- Escape to cancel placement or clear selection

### Phase 6: Wire System - Complete (2026-02-01)
Implemented wire drawing with Manhattan-style routing:

#### Keyboard Shortcuts (LTSpice F-keys mapped to number keys)
- 3 or W: Wire mode (F3)
- 5: Delete mode (F5)
- 7: Move mode (F7)
- Space: Toggle wire direction (horizontal-first / vertical-first) while drawing
- Escape: Cancel current operation, return to select mode

#### Wire Drawing
- Click to start wire, click again to place segment(s)
- Manhattan routing: wires are always horizontal or vertical
- Space key toggles between horizontal-first and vertical-first routing
- Chained wire drawing: after placing, continues from endpoint
- Ghost preview shows wire path before placement

#### Wire Selection and Deletion
- Click on wire to select (yellow highlight)
- Shift+click for multi-select (components and wires)
- Delete/Backspace removes selected wires and components
- Delete mode (5): click to delete individual items

#### Junction Detection (Updated 2026-02-01)
- **Red junction dots** shown at:
  - Wire endpoints connecting to component pins (auto-connect)
  - Wire endpoints where 3+ segments meet (T-junction)
  - Explicit junctions created when drawing wires onto existing wires
- **Wires crossing visually are NOT connected** unless junction exists
- When drawing a wire and clicking on an existing wire segment, a junction is automatically created
- Junctions can be deleted in delete mode (5)

#### Duplicate Mode (Added 2026-02-01)
- Key 6: Duplicate mode (LTSpice F6)
- Click on component or wire to duplicate with 2-grid-unit offset
- Duplicated item is automatically selected

#### Cursor Feedback (Added 2026-02-01)
- Dynamic cursor based on mode:
  - Delete mode: `not-allowed`
  - Wire mode: `crosshair`
  - Move mode: `move`
  - Place/Duplicate mode: `copy`

#### Layout Improvements (Added 2026-02-01)
- Schematic panel: 1/2 of available height (50%)
- Waveform panel: 1/3 of available height (33%)
- Simulation info panel: 1/6 of available height (17%)
- Panel sizes calculated dynamically based on viewport

### Phase 7: Netlist Generation - Complete (2026-02-01)
Implemented connectivity analysis and SPICE netlist generation from schematic:

#### Netlist Module (`src/lib/netlist/`)
- `types.ts` - Type definitions for nets, pin connections, SPICE components
  - `COMPONENT_PREFIX` mapping: component type to SPICE prefix (resistor->R, inductor->L, etc.)
  - `PIN_ORDER` mapping: correct pin ordering for each component type
- `connectivity.ts` - Wire connectivity analysis using Union-Find algorithm
  - Groups connected wire segments and component pins into nets
  - Assigns node numbers (0 for ground, 1+ for other nodes)
  - Detects floating pins and unconnected components
- `netlist-generator.ts` - Converts schematic to SPICE netlist text
  - `generateNetlist()` - Main function returning structured netlist
  - `componentToSpice()` - Converts component to SPICE line with correct pin order
  - `netlistToText()` - Formats as SPICE text
  - `schematicToNetlist()` - Convenience wrapper
  - `generateNodeLabels()` - Creates node labels for display on schematic

#### UI Integration
- "Generate Netlist (Ctrl+N)" button in toolbar
- Populates netlist editor with generated SPICE code
- Status bar shows component/wire/node counts

### Phase 7.5: Node Labels and Probes - WIP (2026-02-01)
Added node label display and probe mode for voltage/current measurement:

#### Node Labels
- Node numbers displayed on schematic after netlist generation
- Green background for ground (node 0), blue for other nodes
- Labels positioned at first point of each net

#### Probe Mode (P key)
- Press P to enter probe mode
- Voltage probe: Click on wire to probe V(node)
- Current probe: Click on component to probe I(component)
- Differential voltage: Click and hold on first node (red + probe), drag to second node (black - probe)
- Probe icons: Pointed probe shape for voltage, current clamp shape for current
- Probes follow cursor position

#### Probe Integration (Debug WIP)
- `onprobe` callback prop on SchematicCanvas
- Probe events dispatched to parent component
- Probes stored in array, can be toggled on/off
- Waveform filtering by active probes (partial implementation)
- Known issues: Probe-to-trace matching needs refinement

### Phase 8: Waveform Viewer Enhancements (2026-02-01)

#### Plotly-Style Zoom Behavior
- Default left-click drag now performs directional zoom:
  - Horizontal drag (ratio > 3:1): X-only zoom with full-height selection band
  - Vertical drag (ratio < 1:3): Y-only zoom with full-width selection band
  - Diagonal drag (ratio between 1:3 and 3:1): Box zoom with rectangle selection
- Pan moved to middle-click (scroll wheel click + drag)
- Visual feedback: Dashed white outline with semi-transparent fill, mode indicator
- Minimum 10 pixel drag required before zoom type is determined

#### Tabbed Waveform Panels
- Multiple waveform panels as tabs
- Add (+) button to create new tabs
- Close (x) button on each tab (at least one tab always exists)
- Active tab highlighting
- Probes add traces to the active tab instead of replacing all traces

#### Delete Mode in Waveform (Key 5)
- Toggle delete mode with key 5
- Delete indicator shows instructions
- Click on trace names in legend to remove traces
- Escape exits delete mode

### Probe System Improvements (2026-02-01)

#### Fixed Probe-to-Trace Matching
- Refactored `addProbeTracesToActiveTab()` to only add traces for a specific probe
- Each probe click now adds only that specific trace to the active tab
- Traces are not duplicated if already present in the active tab
- Removed auto-trace generation on simulation run

#### Improved Wire-to-Node Detection
- Voltage probes now work anywhere on wires connected to a node, not just near node labels
- Implemented BFS traversal through wire connectivity:
  - Follows wire endpoints that touch each other
  - Follows through junctions (explicit T-connections)
  - Checks if junctions lie on wire segments
- Uses pre-built lookup structures for efficiency:
  - `endpointToWires`: Maps point keys to wires with that endpoint (O(1) lookup)
  - `pointToNode`: Maps node label positions to node names (O(1) lookup)
- Returns immediately when a node label is found (early exit)

#### Current Probe Handling
- Click on component to create current probe
- Uses `targetComponent` in probe state to track component clicks separately from wire clicks
- Works for all components (V, I, L, R, C, D, Q, M)

#### Client-Side Current Calculation (Fixed 2026-02-01)
NGSpice only outputs currents for voltage sources and inductors by default. The eecircuit-engine WASM build doesn't support `.save` directives, so we calculate currents client-side from node voltages:

**Implementation** (`src/lib/netlist/current-calculator.ts`):
- `parseValue()` - Parses component values with SI prefixes (1k, 1u, 1n, etc.)
- `calculateComponentCurrent()` - Computes current for a single component
- `calculateMissingCurrents()` - Batch calculates all missing currents

**Formulas used**:
- Resistor: `I = (V1 - V2) / R` (Ohm's law)
- Capacitor: `I = C * d(V1 - V2)/dt` (numerical derivative)

**Performance**: Negligible overhead - just array math on existing voltage data.

#### Known Issues (Still WIP)
- Voltage probe detection may still fail in some edge cases with complex wire networks
- Differential voltage probe needs further testing
- Diode/transistor currents cannot be calculated client-side (nonlinear)

### Phase 10: Persistence - Partial (2026-02-01)
Implemented schematic save/load to JSON files:

#### Save Schematic (Ctrl+S)
- Exports schematic as JSON file with timestamp filename
- Includes: components, wires, junctions, netlist text
- Format version for future compatibility

#### Open Schematic (Ctrl+O)
- File dialog to select .json file
- Restores components, wires, junctions
- Restores netlist if saved
- Clears existing probes

#### JSON Format
```json
{
  "version": 1,
  "schematic": {
    "components": [...],
    "wires": [...],
    "junctions": [...]
  },
  "netlist": "* SPICE netlist...",
  "savedAt": "2026-02-01T..."
}
```

### Not Started
- Phase 9: ASC file parser (LTSpice .asc import/export)
- Phase 10: localStorage auto-save, undo/redo
