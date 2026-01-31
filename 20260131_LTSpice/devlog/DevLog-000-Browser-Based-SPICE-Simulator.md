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

