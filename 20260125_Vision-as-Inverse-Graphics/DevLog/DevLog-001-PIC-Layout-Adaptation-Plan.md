# DevLog-001: Adapting VIGA for Photonic Integrated Circuit Layout

**Date:** 2026-01-27  
**Author:** Planning Document  

---

## 1. Objective

Adapt the VIGA (Vision-as-Inverse-Graphics) workflow for **photonic integrated circuit (PIC) layout design** using gdsfactory instead of Blender.

**Workflow:**
- **Vision Agent (Qwen3-VL-32B):** Analyze target layouts, compare renders, provide structured feedback
- **Code Agent (Claude/Augment):** Generate gdsfactory Python code to create PIC layouts
- **Rendering:** Convert GDS layouts to PNG images for visual comparison

---

## 2. Key Differences: 3D Scenes vs PIC Layouts

| Aspect | 3D Scenes (Blender) | PIC Layouts (gdsfactory) |
|--------|---------------------|--------------------------|
| **Dimensionality** | 3D (x, y, z) | 2D (x, y) with layers |
| **Critical parameters** | Colors, materials, lighting, camera | Dimensions, lengths, gaps, widths |
| **Topology** | Spatial arrangement | Connectivity, routing, port alignment |
| **Physics** | Rigid body, animation | Optical propagation (not simulated in layout) |
| **Output format** | PNG renders from 3D | PNG renders from 2D GDS |
| **Precision** | Visual similarity (~10-20%) | Dimensional accuracy (±nm precision) |

**What matters for PICs:**
- **Dimensions:** Waveguide widths, coupling gaps, bend radii
- **Lengths:** Phase shifter lengths, delay lines, resonator circumferences
- **Topology:** Port connections, routing paths, component placement
- **Layer assignment:** Core, cladding, metal, vias
- **Colors:** Not relevant (layers have functional meaning, not aesthetic)
- **Materials:** Implicit in layer definitions (e.g., layer 1 = Si core)

---

## 3. gdsfactory Rendering to PNG

### 3.1 Research Findings

Based on gdsfactory documentation and examples:

**Method 1: `Component.plot()` (matplotlib-based)**
```python
import gdsfactory as gf
c = gf.components.mzi()
c.plot()  # Opens matplotlib window
```

**Method 2: `Component.plot()` with save**
```python
import matplotlib.pyplot as plt
c = gf.components.mzi()
c.plot()
plt.savefig('output.png', dpi=300, bbox_inches='tight')
plt.close()
```

**Method 3: `Component.show()` (KLayout viewer)**
```python
c.show()  # Opens KLayout GUI (not suitable for headless)
```

**Method 4: Export to image via KLayout Python API**
```python
import klayout.db as kdb
import klayout.lay as lay
# More complex but provides better control
```

**Recommended:** Use Method 2 (matplotlib) for headless rendering in VIGA workflow.

### 3.2 Rendering Script Template

```python
import gdsfactory as gf
import matplotlib.pyplot as plt

def render_gds_to_png(component, output_path, dpi=300):
    """Render gdsfactory component to PNG image."""
    fig = component.plot()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return output_path
```

---

## 4. Prompt Adaptation Strategy

### 4.1 Generator Prompt Changes

**Original (Blender 3D):**
```
You are DynamicSceneGenerator — an expert Blender coding agent that builds 3D dynamic 
scenes from scratch. You will receive (a) an image describing the target scene and 
(b) a text description about the dynamic effects in the target scene.
```

**Adapted (PIC Layout):**
```
You are PICLayoutGenerator — an expert gdsfactory coding agent that designs photonic 
integrated circuit layouts from scratch. You will receive (a) an image showing the 
target layout and (b) a text description of the circuit functionality and specifications.

Your goal is to reproduce the target PIC layout as accurately as possible by writing 
gdsfactory Python code.

[Critical Parameters]
Focus on these aspects in order of importance:
1. **Dimensions:** Waveguide widths, coupling gaps, bend radii (±10nm tolerance)
2. **Lengths:** Component lengths, delay lines, resonator perimeters
3. **Topology:** Port connectivity, routing paths, component placement
4. **Layer assignment:** Correct layer usage (WG, SLAB, METAL, etc.)

[Response Format]
Output a JSON object wrapped in ```json``` markers:
```json
{
  "thought": "Analysis of target layout and design strategy.",
  "code": "Complete gdsfactory Python code to create the layout."
}
```
```

### 4.2 Verifier Prompt Changes

**Original (Blender 3D):**
```
You are DynamicSceneVerifier — a 3D visual feedback assistant...
Your task is to comprehensively analyze discrepancies between the current scene 
and the target...

{
  "visual_difference": "Visual difference between current and target scene.",
  "edit_suggestion": "Edit suggestion for the current scene."
}
```

**Adapted (PIC Layout):**
```
You are PICLayoutVerifier — a photonic circuit layout review assistant...
Your task is to analyze dimensional and topological discrepancies between the 
current layout and the target.

[Analysis Focus]
Prioritize these aspects:
1. **Dimensional accuracy:** Measure and compare waveguide widths, gaps, lengths
2. **Topological correctness:** Verify port connections and routing paths
3. **Component placement:** Check relative positions and spacing
4. **Layer usage:** Confirm correct layer assignments

[Response Format]
Output a JSON object:
```json
{
  "dimensional_difference": {
    "critical": ["List of dimension mismatches >10% error"],
    "major": ["List of dimension mismatches 5-10% error"],
    "minor": ["List of dimension mismatches <5% error"]
  },
  "topological_difference": {
    "missing_connections": ["List of unconnected ports"],
    "incorrect_routing": ["List of routing path errors"],
    "placement_errors": ["List of component position errors"]
  },
  "edit_suggestion": [
    "Specific code changes to fix critical issues",
    "Dimensional adjustments needed",
    "Routing corrections required"
  ],
  "overall_accuracy": <0-100>,
  "next_focus": "What to prioritize in next iteration"
}
```
```

---

## 5. Implementation Plan

### 5.1 Directory Structure

```
20260125_Vision-as-Inverse-Graphics/
├── prompts/
│   └── pic_layout/              # NEW: PIC layout prompts
│       ├── __init__.py
│       ├── generator.py         # PICLayoutGenerator system prompt
│       └── verifier.py          # PICLayoutVerifier system prompt
├── runners/
│   └── pic_layout.py            # NEW: PIC layout runner script
├── tools/
│   └── gdsfactory/              # NEW: gdsfactory execution tools
│       ├── exec.py              # Execute gdsfactory code and render
│       └── inspector.py         # Inspect GDS layout properties
├── data/
│   └── pic_layout/              # NEW: PIC layout test cases
│       ├── mzi/
│       │   ├── target.png
│       │   └── description.txt
│       ├── ring_modulator/
│       └── awg_demux/
└── output/
    └── pic_layout/              # NEW: Generated layouts
```

### 5.2 Core Components to Create

**1. Rendering Utility (`utils/gds_render.py`)**
```python
import gdsfactory as gf
import matplotlib.pyplot as plt
from pathlib import Path

def render_gds_to_png(component, output_path, dpi=300):
    """Render gdsfactory component to PNG.

    Note: gdsfactory 9.23.0 plot() does not accept show_ports parameter.
    Ports are shown by default in the plot.
    """
    fig = component.plot()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    return output_path

def execute_and_render(code_str, output_dir):
    """Execute gdsfactory code and render to PNG."""
    # Execute code in isolated namespace
    namespace = {'gf': gf, '__name__': '__main__'}
    exec(code_str, namespace)

    # Find the component (assume last created or named 'c')
    component = namespace.get('c') or namespace.get('component')

    # Save GDS
    gds_path = Path(output_dir) / 'layout.gds'
    component.write_gds(gds_path)

    # Render PNG
    png_path = Path(output_dir) / 'render.png'
    render_gds_to_png(component, png_path)

    return {
        'gds_path': str(gds_path),
        'png_path': str(png_path),
        'component': component
    }
```

**Test Results:**
```bash
$ /path/to/.venv/bin/python test_gds_render.py
Creating MZI component...
GDS saved: output/test_render/test_mzi.gds
PNG saved: output/test_render/test_mzi.png

Component info:
  Name: mzi_gdsfactorypcomponentspmzispmzi_DL10_LY2_LX0p1_Bbend_1e16e3c9
  Size: 91.10 x 50.75 µm
  Ports: ['o1', 'o2']

Test completed successfully!
```

**2. Generator Prompt (`prompts/pic_layout/generator.py`)**
- Adapted from `dynamic_scene_generator_system_no_tools`
- Focus on dimensions, topology, connectivity
- Remove references to colors, materials, lighting

**3. Verifier Prompt (`prompts/pic_layout/verifier.py`)**
- Adapted from `dynamic_scene_verifier_system_no_tools`
- Add dimensional analysis fields
- Add topological verification fields
- Remove visual aesthetics feedback

**4. Runner Script (`runners/pic_layout.py`)**
- Adapted from `runners/dynamic_scene.py`
- Replace Blender execution with gdsfactory execution
- Use `utils/gds_render.py` for rendering

### 5.3 Execution Flow

```
1. Load target layout image (PNG)
2. Initialize Qwen3-VL vision agent (verifier)
3. Initialize Claude code agent (generator)

Loop for N iterations:
  4. Generator analyzes target + previous feedback
  5. Generator outputs gdsfactory Python code
  6. Execute code → generate GDS file
  7. Render GDS to PNG
  8. Verifier compares render vs target
  9. Verifier outputs dimensional/topological feedback
  10. Feedback → Generator for next iteration

11. Save final GDS + render + iteration history
```

---

## 6. Prompt Improvements for PIC Layouts

### 6.1 What to Remove from 3D Prompts

**Remove:**
- Color matching ("match the blue tint", "warm lighting")
- Material properties ("metallic roughness", "glass IOR")
- Camera angles ("eye-level view", "45-degree perspective")
- Lighting setup ("area light", "HDRI environment")
- Physics simulation ("rigid body", "collision detection")
- Animation ("keyframes", "physics baking")

### 6.2 What to Add for PIC Layouts

**Add:**
- **Dimensional specifications:**
  - "Waveguide width: 0.5 µm ± 0.01 µm"
  - "Coupling gap: 0.2 µm"
  - "Bend radius: 10 µm minimum"

- **Topological constraints:**
  - "Port 'o1' must connect to Port 'o2'"
  - "Route from component A to component B with Manhattan routing"
  - "Maintain 5 µm minimum spacing between waveguides"

- **Layer specifications:**
  - "Use layer (1, 0) for waveguide core"
  - "Use layer (2, 0) for metal heaters"
  - "Use layer (3, 0) for vias"

- **Component library references:**
  - "Use gf.components.mzi() for Mach-Zehnder interferometer"
  - "Use gf.components.ring_single() for ring resonator"
  - "Use gf.routing.get_route() for waveguide routing"

### 6.3 Example Feedback Format

**3D Scene Feedback (Original):**
```json
{
  "visual_difference": "The jug is too tall and lacks a handle. The pears are too
                        uniform in color. The lighting is too harsh.",
  "edit_suggestion": "Reduce jug height by 30%, add curved handle. Vary pear colors
                      from yellow-green to brown. Use softer area light."
}
```

**PIC Layout Feedback (Adapted):**
```json
{
  "dimensional_difference": {
    "critical": [
      "MZI arm length difference: measured 15.2 µm, target 10.0 µm (52% error)",
      "Ring radius: measured 12.3 µm, target 10.0 µm (23% error)"
    ],
    "major": [
      "Waveguide width: measured 0.48 µm, target 0.50 µm (4% error)"
    ],
    "minor": []
  },
  "topological_difference": {
    "missing_connections": ["MMI output port 'o2' not connected to ring input"],
    "incorrect_routing": ["Route from splitter to combiner uses diagonal, should be Manhattan"],
    "placement_errors": ["Ring resonator offset 2.5 µm from centerline"]
  },
  "edit_suggestion": [
    "Adjust MZI delta_length parameter from 15.2 to 10.0",
    "Change ring_single(radius=12.3) to ring_single(radius=10.0)",
    "Add route from mmi.ports['o2'] to ring.ports['o1']",
    "Use gf.routing.get_route_from_steps() for Manhattan routing"
  ],
  "overall_accuracy": 65,
  "next_focus": "Fix critical dimensional errors in MZI and ring, then address routing"
}
```

---

## 7. Testing Strategy

### 7.1 Test Cases (Increasing Complexity)

**Level 1: Single Component**
- Target: Straight waveguide (10 µm length, 0.5 µm width)
- Target: 90-degree bend (10 µm radius)
- Target: Y-branch splitter

**Level 2: Simple Circuits**
- Target: Mach-Zehnder Interferometer (MZI)
- Target: Ring resonator with bus waveguide
- Target: Directional coupler

**Level 3: Complex Circuits**
- Target: Ring modulator with heater
- Target: 1x4 AWG demultiplexer
- Target: Transceiver module (laser + modulator + detector)

### 7.2 Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Dimensional accuracy | >95% | Compare measured vs target dimensions |
| Topological correctness | 100% | All ports connected correctly |
| Iteration count | <10 | Number of rounds to converge |
| Code validity | 100% | No Python/gdsfactory errors |

---

## 8. Python Environment Setup

### 8.1 Decision: Reuse Existing gdsfactory venv

**Path:** `/Users/wentaojiang/Documents/GitHub/PlayGround/20251129_gdsfactory/.venv`

**Rationale:**
- Existing venv already has gdsfactory 9.23.0, matplotlib 3.10.7, numpy 2.3.5
- Only missing: `openai` and `pillow` packages
- Avoids duplication (~500MB gdsfactory installation)
- Both projects are in playground repo (experimental/development)
- No version conflicts expected
- Easy to create separate venv later if needed

**Action Taken:**
```bash
# Install pip first (venv was created without it)
.venv/bin/python -m ensurepip --upgrade

# Install missing packages
.venv/bin/python -m pip install openai pillow
```

**Installation Results:**
```
gdsfactory: 9.23.0
matplotlib: 3.10.7
openai: 2.15.0
Pillow: 12.0.0
```

**Usage in VIGA:**
All PIC layout scripts will use this venv:
```bash
/Users/wentaojiang/Documents/GitHub/PlayGround/20251129_gdsfactory/.venv/bin/python script.py
```

Or set as alias:
```bash
alias viga-python="/Users/wentaojiang/Documents/GitHub/PlayGround/20251129_gdsfactory/.venv/bin/python"
```

---

## 9. Next Steps

1. [DONE] Test remote Qwen3-VL server connection
2. [DONE] Setup Python environment with gdsfactory
3. [TODO] Create `prompts/pic_layout/generator.py` with adapted prompt
4. [TODO] Create `prompts/pic_layout/verifier.py` with adapted prompt
5. [TODO] Create `utils/gds_render.py` rendering utility
6. [TODO] Create `tools/gdsfactory/exec.py` execution tool
7. [TODO] Create `runners/pic_layout.py` runner script
8. [TODO] Prepare test case: simple MZI layout
9. [TODO] Run first iteration and validate workflow
10. [TODO] Iterate and refine prompts based on results

---

## 9. Open Questions

1. **Dimensional measurement:** How does Qwen3-VL measure dimensions from PNG images?
   - May need to add scale bars or dimension annotations to renders
   - Consider adding text labels with measurements

2. **Layer visualization:** How to distinguish layers in PNG renders?
   - Use different colors for different layers in matplotlib
   - Add legend showing layer assignments

3. **Port visualization:** How to show port locations/names?
   - gdsfactory `plot(show_ports=True)` shows port markers
   - May need to enhance with port labels

4. **Quantitative feedback:** Can VLM provide numerical measurements?
   - Test Qwen3-VL's ability to estimate dimensions from images
   - May need to provide reference scale in prompt

---

## 10. Conclusion

The VIGA workflow is well-suited for PIC layout generation with key adaptations:
- Replace Blender → gdsfactory
- Replace 3D rendering → 2D GDS rendering (matplotlib)
- Replace aesthetic feedback → dimensional/topological feedback
- Maintain iterative refinement loop with vision-language model

The hybrid architecture (Qwen3-VL for vision, Claude for code) should work effectively for PIC layouts, potentially with higher accuracy than 3D scenes due to the 2D nature and quantifiable metrics.

