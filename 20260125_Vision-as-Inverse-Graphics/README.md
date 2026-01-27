# Vision-as-Inverse-Graphics (VIGA) for Photonic IC Layout

Adaptation of VIGA workflow for photonic integrated circuit layout generation using gdsfactory.

## Project Structure

```
.
├── DevLog/                          # Development logs and planning documents
│   ├── DevLog-000-*.md             # Initial setup and Qwen model testing
│   └── DevLog-001-*.md             # PIC layout adaptation plan and results
│
├── tools/
│   └── pic_layout/                 # Core PIC layout workflow tools
│       ├── manual_test.py          # Execute code, render, save outputs
│       ├── call_verifier.py        # Qwen3-VL verifier API caller
│       ├── analyze_target.py       # Initial target analysis
│       └── run_iteration.py        # Iteration helper (chip1 example)
│
├── analysis_tools/
│   └── chip2_vlm_queries/          # VLM analysis scripts (chip2 study)
│       ├── detailed_comparison.py  # Multi-query focused analysis
│       ├── overview_and_regions.py # Regional division and cropping
│       ├── complete_component_inventory.py  # Systematic component counting
│       ├── understand_target.py    # Topology analysis
│       ├── why_iter9_worked.py     # Success analysis
│       └── analyze_*.py            # Regional analysis scripts
│
├── prompts/
│   └── pic_layout/                 # PIC layout prompts
│       ├── generator.py            # PICLayoutGenerator system prompt
│       └── verifier.py             # PICLayoutVerifier system prompt
│
├── utils/
│   ├── gds_render.py               # gdsfactory rendering utilities
│   └── common.py                   # Common utilities
│
├── source/                         # Target layout images
│   ├── chip1.png                   # Simple test case (95% achieved)
│   ├── chip2.png                   # Complex test case (75% achieved)
│   └── regions/                    # Cropped regions for analysis
│
├── output/                         # Generated layouts and iterations
│   ├── chip1_iterations/           # chip1 iteration history (3 iterations)
│   └── chip2_iterations/           # chip2 iteration history (19 iterations)
│
└── scripts/                        # Utility scripts
    └── test_gds_render.py          # Test rendering functionality

```

## Quick Start

### Environment Setup

```bash
# Python environment path stored in .env
GDSFACTORY_VENV_PYTHON=/path/to/gdsfactory/.venv/bin/python
VLLM_API_KEY=your_api_key_here
```

### Manual Iteration Workflow

```python
# 1. Analyze target
python tools/pic_layout/analyze_target.py

# 2. Generate code based on analysis

# 3. Test and render
from tools.pic_layout.manual_test import test_generator_code
result = test_generator_code(code, iteration=N, output_base='output/chip1_iterations')

# 4. Get verifier feedback
from tools.pic_layout.call_verifier import call_verifier
feedback = call_verifier(target_image, result['png_path'], description, code)

# 5. Refine and repeat
```

## Test Results

### chip1.png (Simple Layout)
- Iterations: 3
- Best accuracy: 95%
- Status: Fabrication-ready
- Key parameters: bend_radius=72.5µm, straight_length=22.5µm

### chip2.png (Complex Layout)
- Iterations: 19
- Best accuracy: 75% (iteration 9)
- Status: Demonstrates VLM querying methodology
- Key finding: Systematic multi-query analysis essential for complex layouts

## Key Learnings

### Effective VLM Querying
- Multi-query analysis with focused questions
- Regional cropping for detailed examination
- Component inventory (type, count, orientation, location, size)
- Comparative analysis (what worked vs what didn't)

### VLM Limitations
- Numerical scores unreliable (same layout: 75% then 45%)
- Dimensional estimates have high variance
- Topology descriptions may contradict across queries
- Human validation essential for complex cases

See DevLog-001 Section 13 for detailed methodology and lessons learned.

## Documentation

- **DevLog-000**: Initial setup, Qwen3-VL testing, environment configuration
- **DevLog-001**: PIC layout adaptation, prompts, workflow, test results, VLM methodology

## Vision Agent

- Model: Qwen3-VL-32B-Instruct
- Server: http://192.222.54.152:8000/v1 (vLLM)
- Role: Layout verification and feedback

## Code Agent

- Manual iteration with Claude/Augment guidance
- Generates gdsfactory Python code
- Focus: Dimensional accuracy and topological correctness

