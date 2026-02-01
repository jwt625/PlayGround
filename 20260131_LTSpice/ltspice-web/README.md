# WebSpice

A browser-based SPICE circuit simulator with a minimal GUI. Runs entirely client-side using NGSpice compiled to WebAssembly.

## Features

- **Schematic Editor**: Canvas-based editor with component placement, wire routing, and grid snapping
- **Component Library**: Resistor, Capacitor, Inductor, Voltage Source, Current Source, Ground, Diode, BJT (NPN/PNP), MOSFET (NMOS/PMOS)
- **Netlist Generation**: Automatic SPICE netlist generation from schematic
- **Simulation**: Transient analysis via NGSpice WASM (runs in Web Worker)
- **Waveform Viewer**: WebGL-based plotting with pan, zoom, cursors, and multiple tabs
- **Probing**: Voltage probes (click on wire), current probes (click on component), differential voltage (drag between nodes)
- **Persistence**: Save/load schematics as JSON files

## Keyboard Shortcuts

| Key | Function |
|-----|----------|
| R, C, L, V, I, G, D, Q, M | Place component |
| W or 3 | Wire mode |
| 5 | Delete mode |
| 6 | Duplicate mode |
| 7 | Move mode |
| P | Probe mode |
| Ctrl+R | Rotate selected |
| Ctrl+E | Mirror selected |
| Ctrl+N | Generate netlist |
| Ctrl+B | Run simulation |
| Ctrl+S | Save schematic |
| Ctrl+O | Open schematic |
| Escape | Cancel operation |

## Tech Stack

- **Framework**: SvelteKit 2 + Svelte 5
- **Build**: Vite
- **Language**: TypeScript
- **Simulation**: [eecircuit-engine](https://github.com/nickmitchko/eecircuit-engine) (NGSpice WASM)
- **Editor**: CodeMirror 6 (SPICE syntax highlighting)
- **Plotting**: webgl-plot
- **Web Workers**: Comlink

## Development

```sh
# Install dependencies
pnpm install

# Start development server
pnpm dev

# Build for production
pnpm build

# Preview production build
pnpm preview
```

## Project Structure

```
src/
  lib/
    components/     # UI components (ResizablePanel)
    editor/         # CodeMirror netlist editor
    netlist/        # Connectivity analysis, netlist generation, current calculation
    schematic/      # Canvas editor, component definitions, renderer
    simulation/     # NGSpice WASM wrapper, Web Worker
    waveform/       # WebGL waveform viewer
  routes/
    +page.svelte    # Main application
```

## Acknowledgments

- **LTSpice** by Analog Devices - Inspiration for UI design and keyboard shortcuts
- **NGSpice** - Open source SPICE simulator
- **EEcircuit** - Demonstrated NGSpice WASM feasibility; eecircuit-engine provides the simulation backend

## License

MIT
