# MVP Planning Document  
**Project:** Client-Side Isometric RF/Microwave Circuit Builder  
**Framework:** Svelte / SvelteKit  
**Status:** MVP Definition and Architectural Alignment  

---

## 1. Objective

Build a **client-side, interactive, isometric visualization tool** that allows RF/microwave engineers to construct physical-style schematics using dimensionally accurate components and cables. The tool prioritizes **physical intuition, true scale, and bench realism**, while remaining lightweight, offline-capable, and extensible.

The MVP explicitly targets **connectorized RF module chains on a plate or workspace**, rather than PCB layout or full 3D CAD.

---

## 2. Core Design Principles

- **True physical scale:** All placement and routing operate in millimeters.
- **Visual-first:** Components are represented using pre-generated 2.5D isometric textures (PNG/SVG), not runtime 3D models.
- **Client-only:** No backend, no cloud sync; all persistence is local.
- **Separation of concerns:** Rendering engine, editor core, and UI are strictly decoupled.
- **RF-aware primitives:** Ports, cables, and metadata reflect RF and DC realities (not generic diagram edges).

---

## 3. Technology Stack

### 3.1 Application Framework
- **SvelteKit** (TypeScript)
- App shell: toolbars, palettes, inspectors, file actions
- Editor state managed via Svelte stores

### 3.2 Rendering Engine
- **PixiJS** (2.5D isometric rendering)
- Imperative scene graph owned outside Svelte reactivity
- Isometric projection applied at render time; world space remains rectangular

### 3.3 Persistence
- **localStorage only** (MVP)
- Explicit JSON export/import for documents
- Autosave enabled

---

## 4. Rendering & Visual Assets

### 4.1 Component Representation
- Components rendered as **pre-generated 2.5D isometric textures** (PNG or SVG)
- Assets authored externally (e.g., Blender → isometric render → export)
- Renderer treats components as textured quads with metadata-driven dimensions

### 4.2 Scaling Model
- **True millimeter scale**
- Each component definition includes:
  - Physical dimensions (mm)
  - Texture scaling factor (mm → pixels)
- Grid size defined in mm (configurable)

---

## 5. Coordinate System & Grid

- **World space:** Rectangular Cartesian grid (mm units)
- **View:** Isometric projection (fixed camera)
- **Snapping:**
  - Grid snap enabled by default
  - Coarse / fine snap toggle
- **Rotation:**
  - 8 discrete orientations (0°, ±45°, 90°, ±135°, 180°)

---

## 6. Component Model

### 6.1 Component Definition (Library)
Each component includes:
- Unique ID, name, category
- Physical dimensions (mm)
- Texture reference + scale factor
- Ports:
  - Type: RF or DC
  - Connector family (e.g., SMA, 2.92mm, DC barrel, screw terminal)
  - Position in component-local mm coordinates
  - Orientation (one of 8 directions)

### 6.2 Component Instance
- Reference to component definition
- World position (mm)
- Rotation (8-way)
- Instance metadata (notes, part number, etc.)

---

## 7. Ports & Connectivity

- **Port-aware connectivity** is required in MVP
- Two port classes:
  - **RF ports**
  - **DC / power ports**
- Connectivity produces a **logical connection graph**, not simulation
- No electrical rule enforcement in MVP (connector mismatches allowed but typed)

---

## 8. Cable System

### 8.1 Routing Model
- **Manhattan-style routing on grid**
- Expanded to **8-direction routing** (orthogonal + ±45°)
- Routing operates in world space (mm)

### 8.2 Cable Features
- Cable types (initial):
  - Flexible RF coax
  - Semi-rigid RF coax
- Cables connect **port-to-port**
- Supports:
  - Intermediate waypoints (anchors)
  - Automatic length estimation
- Bend radius constraints tracked as metadata (warnings deferred)

---

## 9. Editor Capabilities (MVP)

### Required
- Component placement, move, rotate
- Multi-select
- Copy / paste
- Undo / redo (command-based)
- Port-to-port cable creation
- JSON export / import

### Explicitly Out of Scope
- Cloud sync
- Simulation or EM analysis
- Mounting hole enforcement
- Waveguide routing

---

## 10. Data Model & File Format

- **Single JSON document**
- Contains:
  - Document metadata
  - Component instances
  - Cable definitions
  - References to component library IDs
- Designed to be human-readable and versionable

---

## 11. Asset & Library Workflow (MVP)

- Component library is **developer-authored**
- Components defined via:
  - JSON metadata
  - Pre-rendered isometric textures
- Initial library size:
  - ~15–20 common RF blocks
  - Adapters included as components

---

## 12. Architectural Boundaries

### Editor Core (Framework-agnostic)
- Document model
- Command system (undo/redo)
- Snapping and routing logic
- Serialization

### Renderer (PixiJS)
- Scene graph
- Hit testing
- Visual transforms
- Texture management

### UI (SvelteKit)
- Toolbars, panels, inspectors
- File actions
- Store bindings to editor core

---

## 13. MVP Success Criteria

The MVP is successful if a user can:
1. Assemble a realistic RF module chain to scale
2. Connect RF and DC ports with correctly routed cables
3. Rotate components in 45° increments
4. Save, reload, and share designs via JSON
5. Extract a visually clear representation suitable for documentation

---

## 14. Deferred / MVP+ Considerations

- IndexedDB persistence
- In-app component authoring
- BOM CSV export
- Rule checks (bend radius, connector mismatch)
- Simulation export (scikit-rf, Touchstone references)
- Optional 3D preview mode

---

## 15. Open Decision (Confirmed Separately)

**Primary target environment:**  
Connectorized RF/microwave modules arranged on a plate or workspace.

This choice defines the primitives, routing assumptions, and library focus for the MVP.
