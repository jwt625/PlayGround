# DevLog-001: Detailed MVP Implementation Plan
**Date:** 2025-12-20
**Status:** Planning Phase
**Framework:** SvelteKit + PixiJS + TypeScript + pnpm

---

## 1. Overview

This document outlines the detailed step-by-step implementation plan for the RF/Microwave Circuit Builder MVP, based on the architectural decisions in DevLog-000.

### Key Decisions
- **Package Manager:** pnpm
- **Grid Size:** 5mm default
- **Asset Strategy:** Placeholder boxes initially, real textures added later
- **Testing:** Minimal critical tests only for MVP
- **Component Sources:** Mini-Circuits and Marki Microwave

---

## 2. MVP Component Library (Starter Set)

### 2.1 RF Active Components (6 items)
1. **Low Noise Amplifier (LNA)** - Single-stage, SMA connectors
2. **Power Amplifier (PA)** - Medium power, SMA connectors
3. **Mixer** - Frequency converter, 3-port (RF/LO/IF)
4. **Frequency Multiplier** - x2 or x3, 2-port
5. **Attenuator (Fixed)** - 3dB, 6dB, 10dB variants
6. **Attenuator (Variable)** - Adjustable, with control port

### 2.2 RF Passive Components (6 items)
7. **Bandpass Filter** - Fixed frequency range
8. **Lowpass Filter** - Fixed cutoff
9. **Highpass Filter** - Fixed cutoff
10. **Power Splitter/Combiner** - 2-way, Wilkinson type
11. **Directional Coupler** - 10dB or 20dB coupling
12. **Circulator/Isolator** - 3-port device

### 2.3 Connectors & Adapters (4 items)
13. **SMA Barrel Adapter** - Female-to-female
14. **SMA-to-2.92mm Adapter** - Connector transition
15. **Termination (50Ω)** - SMA load
16. **DC Block** - Inline RF component

### 2.4 Power & Support (4 items)
17. **DC Power Supply Module** - Voltage regulator with screw terminals
18. **Bias Tee** - Combines RF + DC
19. **RF Switch** - SPDT, electronically controlled
20. **Test Point** - SMA connector for measurement access

**Total: 20 components**

---

## 3. Implementation Phases

### Phase 1: Foundation (Days 1-2)
**Goal:** Working development environment with basic rendering

#### 1.1 Project Initialization
- [ ] Create SvelteKit project with TypeScript template
- [ ] Configure pnpm workspace
- [ ] Set up ESLint + Prettier
- [ ] Configure TypeScript strict mode
- [ ] Install dependencies: PixiJS, type definitions
- [ ] Create basic folder structure

#### 1.2 Core Data Models
- [ ] Define TypeScript interfaces:
  - `ComponentDefinition` - Library component schema
  - `ComponentInstance` - Placed component in document
  - `Port` - RF/DC port with position and type
  - `Cable` - Connection between ports
  - `Document` - Top-level data structure
- [ ] Implement coordinate system utilities
- [ ] Create isometric projection helpers

#### 1.3 Basic PixiJS Setup
- [ ] Initialize PixiJS application
- [ ] Set up isometric camera/projection
- [ ] Create scene graph structure
- [ ] Implement viewport pan/zoom
- [ ] Test rendering with simple shapes

---

### Phase 2: Component System (Days 3-4)
**Goal:** Place and manipulate components on canvas

#### 2.1 Component Library
- [ ] Create component definition JSON schema
- [ ] Implement component registry/loader
- [ ] Generate placeholder textures (colored boxes with labels)
- [ ] Define all 20 starter components with:
  - Physical dimensions (mm)
  - Port locations and types
  - Placeholder texture references

#### 2.2 Rendering Components
- [ ] Implement component sprite rendering
- [ ] Apply isometric projection to sprites
- [ ] Render port indicators
- [ ] Implement selection highlighting
- [ ] Add rotation visualization (8 orientations)

#### 2.3 Grid & Snapping
- [ ] Render isometric grid (5mm spacing)
- [ ] Implement grid snapping logic
- [ ] Add coarse/fine snap toggle
- [ ] Visual grid overlay with proper isometric perspective

---

### Phase 3: Editor Interactions (Days 5-6)
**Goal:** Interactive component placement and manipulation

#### 3.1 Placement & Selection
- [ ] Implement component drag-from-palette
- [ ] Click-to-place workflow
- [ ] Single selection (click)
- [ ] Multi-selection (shift-click, box select)
- [ ] Move selected components with snap
- [ ] Delete selected components

#### 3.2 Rotation & Transform
- [ ] Rotate component (R key or button)
- [ ] 8-way rotation (45° increments)
- [ ] Update port positions on rotation
- [ ] Visual rotation preview

#### 3.3 Copy/Paste
- [ ] Copy selected components (Ctrl+C)
- [ ] Paste with offset (Ctrl+V)
- [ ] Maintain relative positions in multi-select

---

## 4. Detailed File Structure

```
20251220_rf_bench/
├── src/
│   ├── lib/
│   │   ├── core/              # Framework-agnostic editor core
│   │   │   ├── models/        # Data models & interfaces
│   │   │   ├── commands/      # Command pattern for undo/redo
│   │   │   ├── geometry/      # Coordinate & isometric math
│   │   │   └── serialization/ # JSON import/export
│   │   ├── renderer/          # PixiJS rendering engine
│   │   │   ├── PixiRenderer.ts
│   │   │   ├── ComponentSprite.ts
│   │   │   ├── CableRenderer.ts
│   │   │   └── GridRenderer.ts
│   │   ├── library/           # Component definitions
│   │   │   ├── components.json
│   │   │   └── ComponentLibrary.ts
│   │   ├── stores/            # Svelte stores
│   │   │   ├── documentStore.ts
│   │   │   ├── selectionStore.ts
│   │   │   └── editorStore.ts
│   │   └── components/        # Svelte UI components
│   │       ├── Canvas.svelte
│   │       ├── Toolbar.svelte
│   │       ├── ComponentPalette.svelte
│   │       └── PropertyInspector.svelte
│   ├── routes/
│   │   └── +page.svelte       # Main editor page
│   └── assets/
│       └── textures/          # Component textures (placeholders)
├── static/
├── tests/                     # Minimal critical tests
├── package.json
├── tsconfig.json
├── vite.config.ts
└── .eslintrc.cjs
```

---

## 5. Implementation Phases (Continued)

### Phase 4: Command System (Day 7)
**Goal:** Undo/redo functionality for all operations

#### 4.1 Command Infrastructure
- [ ] Create `Command` interface with execute/undo
- [ ] Implement `CommandHistory` manager
- [ ] Create command classes:
  - `PlaceComponentCommand`
  - `MoveComponentCommand`
  - `RotateComponentCommand`
  - `DeleteComponentCommand`
  - `CreateCableCommand`
  - `DeleteCableCommand`
- [ ] Integrate with keyboard shortcuts (Ctrl+Z, Ctrl+Y)

---

### Phase 5: Cable Routing (Days 8-9)
**Goal:** Connect components with routed cables

#### 5.1 Port System
- [ ] Render port indicators on components
- [ ] Implement port hit detection
- [ ] Show port tooltip on hover (type, connector)
- [ ] Highlight compatible ports during cable creation

#### 5.2 Cable Creation
- [ ] Click port-to-port cable creation
- [ ] 8-direction Manhattan routing algorithm
- [ ] Automatic waypoint generation
- [ ] Manual waypoint editing (drag anchors)
- [ ] Cable length calculation and display

#### 5.3 Cable Rendering
- [ ] Render cables as segmented lines
- [ ] Different visual styles for RF vs DC cables
- [ ] Cable selection and deletion
- [ ] Show cable properties (length, type)

---

### Phase 6: UI Layer (Days 10-11)
**Goal:** Complete Svelte UI with all editor controls

#### 6.1 Main Layout
- [ ] Top toolbar with file operations
- [ ] Left component palette (categorized)
- [ ] Right property inspector
- [ ] Center canvas with zoom controls
- [ ] Status bar (cursor position, snap mode)

#### 6.2 Component Palette
- [ ] Categorized component list
- [ ] Search/filter components
- [ ] Drag-to-canvas interaction
- [ ] Component preview thumbnails

#### 6.3 Property Inspector
- [ ] Show selected component properties
- [ ] Edit instance metadata (name, notes, part number)
- [ ] Display port information
- [ ] Show cable properties when selected

#### 6.4 Toolbar Actions
- [ ] New document
- [ ] Save/Load (localStorage)
- [ ] Export JSON
- [ ] Import JSON
- [ ] Undo/Redo buttons
- [ ] Grid toggle
- [ ] Snap mode toggle

---

### Phase 7: Persistence (Day 12)
**Goal:** Save, load, and export designs

#### 7.1 Serialization
- [ ] Implement document-to-JSON serialization
- [ ] Implement JSON-to-document deserialization
- [ ] Validate imported JSON schema
- [ ] Handle version compatibility

#### 7.2 Storage
- [ ] localStorage autosave (every 30 seconds)
- [ ] Manual save/load from localStorage
- [ ] Export JSON file download
- [ ] Import JSON file upload
- [ ] Clear/reset document

---

### Phase 8: Integration & Polish (Days 13-14)
**Goal:** End-to-end testing and MVP validation

#### 8.1 Integration Testing
- [ ] Test complete workflow: place → connect → save → load
- [ ] Verify undo/redo for all operations
- [ ] Test multi-component selection and operations
- [ ] Validate JSON export/import round-trip
- [ ] Test with all 20 component types

#### 8.2 Critical Tests
- [ ] Unit test: Isometric coordinate conversion
- [ ] Unit test: Grid snapping logic
- [ ] Unit test: Command execute/undo
- [ ] Integration test: Component placement workflow
- [ ] Integration test: Cable routing

#### 8.3 MVP Validation
- [ ] ✓ Assemble realistic RF module chain to scale
- [ ] ✓ Connect RF and DC ports with routed cables
- [ ] ✓ Rotate components in 45° increments
- [ ] ✓ Save, reload, and share designs via JSON
- [ ] ✓ Extract visually clear representation

---

## 6. Technical Implementation Details

### 6.1 Isometric Projection
```typescript
// World space (mm) → Screen space (pixels)
const ISO_ANGLE = Math.PI / 6; // 30 degrees
const SCALE = 2; // pixels per mm

function worldToScreen(x: number, y: number): { x: number, y: number } {
  return {
    x: (x - y) * Math.cos(ISO_ANGLE) * SCALE,
    y: (x + y) * Math.sin(ISO_ANGLE) * SCALE
  };
}
```

### 6.2 Component Definition Schema
```typescript
interface ComponentDefinition {
  id: string;
  name: string;
  category: 'active' | 'passive' | 'connector' | 'power';
  dimensions: { width: number; height: number; depth: number }; // mm
  texture: string; // path to texture asset
  textureScale: number; // mm per pixel in texture
  ports: Port[];
}

interface Port {
  id: string;
  type: 'RF' | 'DC';
  connector: 'SMA' | '2.92mm' | 'DC_BARREL' | 'SCREW_TERMINAL';
  position: { x: number; y: number; z: number }; // mm, component-local
  orientation: 0 | 45 | 90 | 135 | 180 | 225 | 270 | 315; // degrees
}
```



### 6.3 Document Schema
```typescript
interface Document {
  version: string;
  metadata: {
    name: string;
    created: string;
    modified: string;
  };
  components: ComponentInstance[];
  cables: Cable[];
  gridSize: number; // mm
}

interface ComponentInstance {
  id: string; // unique instance ID
  definitionId: string; // reference to ComponentDefinition
  position: { x: number; y: number }; // mm, world space
  rotation: 0 | 45 | 90 | 135 | 180 | 225 | 270 | 315;
  metadata: {
    name?: string;
    partNumber?: string;
    notes?: string;
  };
}

interface Cable {
  id: string;
  type: 'RF_FLEXIBLE' | 'RF_SEMIRIGID' | 'DC_WIRE';
  fromPort: { componentId: string; portId: string };
  toPort: { componentId: string; portId: string };
  waypoints: { x: number; y: number }[]; // mm, world space
  length?: number; // calculated length in mm
}
```

---

## 7. Development Tools & Configuration

### 7.1 Package Dependencies
```json
{
  "dependencies": {
    "svelte": "^4.x",
    "@sveltejs/kit": "^2.x",
    "pixi.js": "^7.x"
  },
  "devDependencies": {
    "typescript": "^5.x",
    "vite": "^5.x",
    "eslint": "^8.x",
    "prettier": "^3.x",
    "@typescript-eslint/eslint-plugin": "^6.x",
    "@typescript-eslint/parser": "^6.x",
    "vitest": "^1.x"
  }
}
```

### 7.2 TypeScript Configuration
- Strict mode enabled
- Path aliases: `$lib/*`, `$core/*`, `$renderer/*`
- Target: ES2022
- Module: ESNext

### 7.3 Linting Rules
- ESLint with TypeScript plugin
- Prettier for formatting
- No unused variables
- Explicit return types for public APIs
- Consistent naming conventions

---

## 8. Success Metrics

### MVP Complete When:
1. ✅ All 20 components defined with placeholder textures
2. ✅ Components can be placed, moved, rotated (8-way), and deleted
3. ✅ Multi-select and copy/paste working
4. ✅ Undo/redo for all operations
5. ✅ Port-to-port cable routing with 8-direction Manhattan
6. ✅ Save/load from localStorage with autosave
7. ✅ Export/import JSON files
8. ✅ Grid snapping (5mm) with toggle
9. ✅ Property inspector shows component/cable details
10. ✅ Can build a complete RF chain (e.g., LNA → Filter → Mixer → PA)

---

## 9. Next Steps After MVP

### Immediate Post-MVP (MVP+)
- Replace placeholder textures with real isometric renders
- Add more components from Mini-Circuits/Marki catalogs
- Implement bend radius warnings for cables
- BOM export (CSV with part numbers)
- Enhanced visual styling

### Future Enhancements
- IndexedDB for larger projects
- Multiple document tabs
- Component parameter editing (frequency, gain, etc.)
- Export to PNG/SVG for documentation
- Simulation export (Touchstone references)
- Collaborative features

---

## 10. Risk Mitigation

### Technical Risks
- **PixiJS performance with many components:** Use sprite batching, limit initial library size
- **Complex cable routing:** Start with simple Manhattan, iterate based on user feedback
- **Isometric hit detection:** Implement proper depth sorting and bounding box tests

### Scope Risks
- **Feature creep:** Strictly adhere to MVP scope, defer enhancements
- **Asset creation bottleneck:** Use placeholders, parallelize real asset creation

---

## 11. Timeline Estimate

**Total: ~14 days for MVP**

- Phase 1 (Foundation): 2 days
- Phase 2 (Components): 2 days
- Phase 3 (Interactions): 2 days
- Phase 4 (Commands): 1 day
- Phase 5 (Cables): 2 days
- Phase 6 (UI): 2 days
- Phase 7 (Persistence): 1 day
- Phase 8 (Integration): 2 days

*Note: Timeline assumes focused development; adjust based on available time.*

---

## 12. Implementation Order Summary

The recommended implementation order prioritizes getting visual feedback early:

1. **Project setup** → See something on screen quickly
2. **Data models** → Foundation for everything
3. **PixiJS + basic rendering** → Visual confirmation system works
4. **Component library + placement** → Core interaction loop
5. **Selection + manipulation** → Make it interactive
6. **Command system** → Add undo/redo safety net
7. **Cable routing** → Connect components
8. **UI layer** → Polish the interface
9. **Persistence** → Save work
10. **Testing + validation** → Ensure quality

Each phase builds on the previous, with frequent opportunities to test and validate.

---

**End of DevLog-001**
