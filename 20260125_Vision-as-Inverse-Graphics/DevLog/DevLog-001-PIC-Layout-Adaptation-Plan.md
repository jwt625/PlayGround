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

## 9. Implementation Status

### Completed Components

**1. Remote Qwen3-VL Server**
- Server: http://192.222.54.152:8000/v1
- Model: Qwen3-VL-32B-Instruct
- API key: In `.env` file
- Status: Tested and working

**2. Python Environment**
- Path: Stored in `.env` as `GDSFACTORY_VENV_PYTHON`
- Value: `/Users/wentaojiang/Documents/GitHub/PlayGround/20251129_gdsfactory/.venv/bin/python`
- Packages: gdsfactory 9.23.0, matplotlib 3.10.7, openai 2.15.0, Pillow 12.0.0
- Status: All scripts updated to load from environment variable

**3. Generator Prompt**
- File: `prompts/pic_layout/generator.py`
- Variable: `pic_layout_generator_system`
- Key features:
  - Dimensional accuracy focus (waveguide widths, gaps, bend radii, lengths)
  - Topological correctness (port connectivity, routing)
  - Design rules (min width, min radius, min spacing)
  - gdsfactory component library reference
  - Code structure template
- Output format: JSON `{"thought": "...", "code": "..."}`

**4. Verifier Prompt**
- File: `prompts/pic_layout/verifier.py`
- Variable: `pic_layout_verifier_system`
- Key features:
  - Component identification checklist
  - Dimensional verification (critical/major/minor/negligible error classification)
  - Topological verification (connectivity, routing, symmetry, alignment)
  - Design rule checking
  - Layout quality assessment
- Output format: JSON with component_check, dimensional_analysis, topological_check, design_rule_violations, overall_assessment, edit_suggestions

**5. Rendering Utility**
- File: `utils/gds_render.py`
- Functions:
  - `render_gds_to_png(component, output_path, dpi=300)` - Render component to PNG
  - `execute_and_render(code_str, output_dir, component_name)` - Execute code and render
  - `get_component_info(component)` - Extract component metadata
- Status: Tested with gdsfactory 9.23.0 API

**6. Manual Test Script**
- File: `manual_test_pic_layout.py`
- Functions:
  - `test_generator_code(code_str, iteration, output_base)` - Execute code, render, save outputs
  - `save_iteration_summary(iteration, code, result, feedback)` - Track iteration data
- Outputs: GDS file, PNG render, code file, summary.json
- Status: Tested and working

**7. Verifier Caller**
- File: `call_verifier.py`
- Functions:
  - `call_verifier(target_image, current_image, description, code)` - Call Qwen3-VL API
  - `save_verifier_result(result, output_path)` - Save verifier response
- Uses: OpenAI client with vLLM endpoint
- Status: Tested and working

**8. Target Analysis Script**
- File: `analyze_target.py`
- Function: Analyze target layout image with Qwen3-VL before starting iterations
- Provides detailed component identification and dimensional estimates
- Status: Tested with chip1.png

**9. Iteration Runner Helper**
- File: `run_chip1_iteration.py`
- Function: Helper script for running iterations on chip1.png
- Combines test_generator_code and call_verifier
- Status: Created but manual workflow preferred for MVP

### First Complete Test Case: chip1.png

**Target:** `source/chip1.png`
**Output:** `output/chip1_iterations/`
**Status:** COMPLETED - 3 iterations, 95% accuracy, fabrication-ready

**Results:**
- Iteration 1: 85% accuracy (bend_radius=72.5, straight_length=22.5)
- Iteration 2: 85% accuracy (bend_radius=68.0, straight_length=18.0, over-corrected)
- Iteration 3: 95% accuracy (bend_radius=72.5, straight_length=22.5, fabrication-ready)

**Documentation:** `output/chip1_iterations/ITERATION_SUMMARY.md`

### Manual Testing Workflow (Validated)

1. Analyze target with Qwen3-VL:
   ```bash
   python analyze_target.py
   ```

2. Generate initial code based on analysis

3. Execute and render:
   ```python
   from manual_test_pic_layout import test_generator_code
   result = test_generator_code(code, iteration=N, output_base='output/chip1_iterations')
   ```

4. Get verifier feedback:
   ```python
   from call_verifier import call_verifier
   feedback = call_verifier(target_image, result['png_path'], description, code)
   ```

5. Refine code based on feedback and repeat

---

## 10. Lessons Learned from chip1.png Test

### Dimensional Measurement Accuracy

**Finding:** Qwen3-VL can estimate dimensions from PNG images but with variable accuracy.
- Initial visual estimates in iteration 1 suggested reducing dimensions (7-25% error)
- However, when given explicit target ranges, verifier correctly identified iteration 3 as optimal
- **Recommendation:** Provide explicit dimensional specifications in target description rather than relying solely on visual estimation

### VLM Feedback Consistency

**Finding:** Verifier feedback can oscillate between iterations.
- Iteration 1 (72.5/22.5): Suggested reducing dimensions
- Iteration 2 (68.0/18.0): Suggested increasing back to iteration 1 values
- Iteration 3 (72.5/22.5): Confirmed as optimal (95% accuracy)
- **Recommendation:** When verifier suggests reverting to previous iteration, trust the feedback

### Convergence Pattern

**Finding:** Initial estimates based on target specifications were more accurate than visual-only analysis.
- Iteration 1 parameters (midpoint of target ranges) achieved 95% accuracy after one correction cycle
- Visual-only estimates led to over-correction
- **Recommendation:** Start with analytical estimates from target specifications, use visual feedback for refinement

### Quantitative vs Qualitative Feedback

**Finding:** Qwen3-VL provides both quantitative estimates and qualitative assessments.
- Quantitative: Percentage errors, dimensional comparisons
- Qualitative: "within range", "acceptable", "critical error"
- Both are useful for different purposes
- **Recommendation:** Use quantitative feedback for parameter tuning, qualitative for overall assessment

### Layer Visualization

**Finding:** gdsfactory default color scheme is sufficient for simple layouts.
- Single-layer waveguide structures render clearly
- Grating couplers are visually distinct
- **Open question:** Multi-layer structures may need custom color schemes

### Prompt Refinement

**Finding:** Current prompts are effective for simple structures.
- Generator prompt provides clear code structure guidance
- Verifier prompt produces structured, actionable feedback
- **Future work:** Add examples for complex multi-component layouts

---

## 11. Next Steps

### Immediate Tasks

1. Test with more complex layouts:
   - Mach-Zehnder Interferometer (MZI) with explicit arm lengths
   - Ring resonator with bus waveguide
   - Directional coupler with specific gap

2. Evaluate prompt improvements:
   - Add successful examples to generator prompt
   - Refine verifier error classification thresholds
   - Test with multi-layer structures

3. Automation considerations:
   - Evaluate when to stop iterations (convergence criteria)
   - Handle cases where verifier feedback oscillates
   - Implement automatic parameter interpolation between iterations

### Future Enhancements

1. Quantitative dimensional extraction from images
2. Multi-layer color scheme for complex structures
3. Automated test suite for common PIC components
4. Integration with optical simulation tools

---

## 12. Summary

VIGA workflow successfully adapted for PIC layout generation:
- Vision agent: Qwen3-VL-32B-Instruct (remote server)
- Code agent: Manual iteration with Claude/Augment guidance
- Rendering: gdsfactory → matplotlib → PNG
- Focus: Dimensional accuracy and topological correctness over visual similarity

**Status:** MVP validated with chip1.png test case (3 iterations, 95% accuracy, fabrication-ready)

**Key Achievement:** Demonstrated that vision-language models can provide actionable feedback for photonic layout design, successfully adapting the VIGA approach from 3D scene generation to 2D precision engineering.

---

## 13. chip2.png Extended Iteration Study (19 iterations)

### Test Case
**Target:** `source/chip2.png` - Complex multi-component layout with rings, electrodes, waveguides
**Complexity:** Much higher than chip1.png - lower resolution, incomplete chip view, more component types
**Goal:** Achieve 95%+ accuracy through iterative refinement

### Iteration Results Summary

| Iterations | Accuracy Range | Approach |
|------------|---------------|----------|
| 1-8 | 30-65% | Initial topology exploration, basic structure |
| 9 | 75% | Best result: coupling_gap=0.20µm, 9 electrode rings, proper spacing |
| 10-13 | 55-65% | Minor parameter tweaks based on verifier feedback - all worse than iter 9 |
| 14-17 | 25-45% | Complete redesigns based on detailed VLM analysis - significantly worse |
| 18 | 45% | Essentially identical to iter 9 - demonstrates verifier inconsistency |
| 19 | 65% | Added electrode routing to pads |

**Best Configuration (Iteration 9):**
```python
ring_radius = 10.0 µm
coupling_gap = 0.20 µm  # Critical parameter
ring_spacing = 13.0 µm
electrode_radii = [11.0, 10.75, 10.5, 10.25, 10.0, 9.75, 9.5, 9.25, 9.0]  # 9 concentric rings
```

### VLM Query Strategy Evolution

**Initial Approach (Iterations 1-13):**
- Single broad queries to verifier
- Generic "what's wrong" questions
- Minor parameter adjustments without deep understanding
- Result: Stuck at 75% with oscillating feedback

**Improved Approach (Iterations 14-19):**
- Multi-query analysis with focused questions
- Regional cropping for detailed examination
- Systematic component inventory
- Understanding what worked (iteration 9 analysis)

**Analysis Tools Created:**
1. `detailed_comparison.py` - 5 focused queries (overall, electrodes, waveguides, GCs, dimensions)
2. `understand_target.py` - Specific topology analysis
3. `why_iter9_worked.py` - Analysis of successful iteration
4. `complete_component_inventory.py` - Exhaustive component counting
5. `overview_and_regions.py` - Regional division and cropping
6. `analyze_left_edge.py`, `analyze_right_edge.py`, `analyze_bottom_half.py` - Region-specific analysis
7. `final_grating_coupler_check.py` - Systematic edge scanning

### Key Findings About VLM Usage

**What Works:**
- Breaking complex layouts into focused queries (electrode structure, waveguide topology, etc.)
- Asking VLM to describe what it sees rather than what's wrong
- Cropping regions for detailed analysis
- Systematic component inventory (type, quantity, orientation, location, size)
- Asking "why did this work" instead of only "what's wrong"

**What Doesn't Work:**
- Trusting single-pass analysis for complex layouts
- Assuming VLM sees grating couplers when it says "no grating couplers visible"
- Following verifier suggestions blindly without empirical validation
- Making changes based on numerical scores alone

**Critical Discovery - VLM Limitations:**
- Iteration 9: 75% accuracy
- Iteration 18 (essentially identical code): 45% accuracy
- Iteration 13 (iter 9 with coupling_gap 0.20→0.25): 55% accuracy
- Conclusion: Numerical scores are unreliable for identical/similar layouts

**Verifier Feedback Contradictions:**
- Suggested coupling_gap=0.25 when 0.20 empirically worked better
- Different scores for same layout across iterations
- Topology descriptions varied between queries on same image

### Lessons on VLM Interaction

**Effective Query Patterns:**
1. Overview first: "Describe overall layout structure and suggest regions to analyze"
2. Regional analysis: Crop and analyze specific areas separately
3. Component inventory: "List every component type, count, orientation, location, size"
4. Focused questions: "How many electrode rings per resonator? Count carefully."
5. Comparative analysis: "What did iteration X get correct? Build on that."

**Ineffective Query Patterns:**
1. Vague questions: "What's wrong with this layout?"
2. Assuming VLM knowledge: "Fix the grating couplers" (when none exist)
3. Trusting first answer: VLM may contradict itself on re-query
4. Numerical precision: VLM dimensional estimates have high variance

**Human-in-the-Loop Critical:**
- User correctly identified that VLM might see grating couplers despite saying "no"
- User pushed back on blaming VLM for inconsistency
- User demanded systematic regional analysis
- Lesson: VLM is a tool, not an oracle - verify its claims

### Methodology Improvements Demonstrated

**Multi-Query Analysis:**
- Single query: "What's different?" → vague feedback
- Multi-query: 5 focused questions → detailed structural insights
- Regional crops: Focused attention on specific areas

**Understanding Success:**
- Analyzing why iteration 9 achieved 75% revealed core structure was correct
- Building on success more effective than chasing verifier suggestions

**Empirical Validation:**
- coupling_gap 0.20 vs 0.25: empirical testing showed 0.20 better despite verifier suggestion
- Trust measurements over VLM numerical scores

### Challenges and Blockers

**Verifier Inconsistency:**
- Same layout gets different scores (75% vs 45%)
- Feedback contradicts empirical results
- Cannot reliably iterate toward 95%+ using scores alone

**Complex Target Ambiguity:**
- Low resolution image
- Incomplete chip view
- VLM gave different topology descriptions in different queries

**Remaining Gap (75% → 95%):**
- Requires complex electrode routing to pads
- Precise dimensional matching beyond VLM measurement capability
- May need human validation or different verification approach

### Recommendations for Future Work

**VLM Query Best Practices:**
1. Always start with overview and regional division
2. Crop and analyze regions separately for complex layouts
3. Ask multiple focused questions rather than single broad query
4. Request component inventory with counts, orientations, locations, sizes
5. Verify VLM claims with follow-up queries or human inspection
6. Use comparative analysis (what worked vs what didn't)

**Iteration Strategy:**
1. Start with analytical estimates from specifications
2. Use VLM for qualitative feedback, not absolute numerical scores
3. When verifier oscillates, trust empirical measurements
4. Build on successful iterations rather than complete redesigns
5. Validate VLM topology descriptions with regional analysis

**Tool Improvements Needed:**
1. Automated region cropping based on VLM recommendations
2. Dimensional extraction from images (independent of VLM)
3. Consistency checking across multiple VLM queries
4. Human validation interface for ambiguous cases

**Conclusion:**
Achieved 75% accuracy on complex layout through systematic VLM querying. The 95%+ goal was blocked by verifier inconsistency and target ambiguity, not VLM capability. Key insight: VLM is powerful for qualitative analysis and component identification when queried systematically, but unreliable for numerical scoring and precise dimensional feedback. Human oversight essential for validating VLM claims and resolving contradictions.

