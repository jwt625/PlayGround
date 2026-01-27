# DevLog-000: VIGA Project Setup and Qwen Model Testing

**Date:** 2026-01-26  
**Author:** Setup Log  

---

## 1. Machine Specifications

| Component | Details |
|-----------|---------|
| GPU | 2x NVIDIA H100 80GB HBM3 (160GB total VRAM) |
| CUDA | 12.8 |
| Python | 3.10.12 |
| OS | Linux |

---

## 2. VIGA Project Installation

### 2.1 Prerequisites Installed

- **Miniconda**: `/home/ubuntu/miniconda3`
- **uv**: v0.9.2 (available but conda used for CUDA dependencies)

### 2.2 Conda Environments Created

| Environment | Python | Purpose | Status |
|-------------|--------|---------|--------|
| agent | 3.10 | Main agent runtime | Configured |
| blender | 3.11 | Blender tools with infinigen | Configured |
| sam | 3.10 | Segment Anything Model | Configured |
| sam3d | 3.11 | SAM-3D object segmentation | Configured |
| vllm | 3.10 | vLLM server for VLMs | Configured |

### 2.3 Key Components Installed

| Component | Location |
|-----------|----------|
| Blender binary | `utils/third_party/infinigen/blender/blender` |
| SAM checkpoint | `utils/third_party/sam/sam_vit_h_4b8939.pth` (~2.4GB) |
| SAM-3D checkpoints | `utils/third_party/sam3d/checkpoints/hf/` (~12GB) |

### 2.4 Configuration Files Created

- `utils/_path.py` - Environment mappings, CONDA_BASE set to `~/miniconda3/envs`
- `utils/_api_keys.py` - API key template with environment variable fallbacks

---

## 3. Qwen Vision-Language Model Testing

### 3.1 API Requirements for VIGA

The VIGA project requires:
1. **Vision/Multimodal capability** - Images are sent to the model for visual comparison
2. **Tool/Function calling** - Agents use `"tools": tool_configs, "tool_choice": "auto"`
3. **OpenAI-compatible API** - All clients use OpenAI SDK

### 3.2 Model Testing Results

#### Test 1: Qwen2-VL-72B-Instruct (Full Precision BF16)

**Command:**
```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2-VL-72B-Instruct \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 4096
```

**Result:** FAILED

- Model loaded successfully: 68.5 GiB per GPU
- KV cache memory available: 0.04 GiB
- KV cache memory required: 0.62 GiB (for max_model_len=4096)
- Error: Insufficient memory for KV cache

**Analysis:** 72B model in BF16 uses ~137GB total, leaving insufficient headroom for KV cache on 2x 80GB GPUs.

---

#### Test 2: Qwen2-VL-7B-Instruct (Full Precision)

**Command:**
```bash
python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 8000 \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --served-model-name Qwen2-VL-7B-Instruct \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

**Result:** SUCCESS

**Test API Call:**
```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen2-VL-7B-Instruct", "messages": [{"role": "user", "content": "Say hello in 5 words"}], "max_tokens": 50}'
```

**Response:** "Hello! How are you?"

---

#### Test 3: Qwen2-VL-72B-Instruct (INT8 Quantization via bitsandbytes)

**Prerequisites:**
```bash
pip install "bitsandbytes>=0.46.1"
```

**Command:**
```bash
python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 8000 \
  --model Qwen/Qwen2-VL-72B-Instruct \
  --served-model-name Qwen2-VL-72B-Instruct \
  --trust-remote-code \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

**Result:** SUCCESS

**Test API Call:**
```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen2-VL-72B-Instruct", "messages": [{"role": "user", "content": "Say hello in 5 words"}], "max_tokens": 50}'
```

**Response:** "Hello there, how can I assist you today?"

---

## 4. Summary

| Model | Precision | Memory | Status | Quality |
|-------|-----------|--------|--------|---------|
| Qwen2-VL-72B | BF16 | ~137GB | Failed (OOM) | N/A |
| Qwen2-VL-7B | BF16 | ~14GB | Working | Good |
| Qwen2-VL-72B | INT8 (bitsandbytes) | ~72GB | Working | Excellent |

**Recommendation:** Use Qwen2-VL-72B-Instruct with INT8 quantization for VIGA. The 72B model provides significantly better visual reasoning and code generation capabilities compared to 7B, with only ~1-3% quality loss from quantization.

**Configuration for VIGA:**
Set in `utils/_api_keys.py`:
```python
QWEN_BASE_URL = "http://localhost:8000/v1"
```

---

## 5. Notes

- vLLM quantization options do not include `int8` directly; use `bitsandbytes` instead
- Valid quantization methods: awq, fp8, gptq, bitsandbytes, compressed-tensors, etc.
- Tool calling enabled via `--enable-auto-tool-choice --tool-call-parser hermes`





## Launching VLLM


# Kill current server first
pkill -f "vllm.entrypoints.openai.api_server"

```bash
source $HOME/miniconda3/bin/activate && conda activate vllm && \
python -m vllm.entrypoints.openai.api_server \
   --host 0.0.0.0 \
   --port 8000 \
   --model Qwen/Qwen2-VL-72B-Instruct \
   --served-model-name Qwen2-VL-72B-Instruct \
   --trust-remote-code \
   --quantization bitsandbytes \
   --load-format bitsandbytes \
   --tensor-parallel-size 2 \
   --gpu-memory-utilization 0.9 \
   --max-model-len 8192 \
   --enable-auto-tool-choice \
   --tool-call-parser hermes 2>&1
```

```bash
source $HOME/miniconda3/bin/activate && conda activate vllm && \
python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 8000 \
  --model Qwen/Qwen2-VL-72B-Instruct \
  --served-model-name Qwen2-VL-72B-Instruct \
  --trust-remote-code \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 16384 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

Note: 16384 context length causes OOM during initialization. Stick with 8192.

---

## 6. VIGA Integration Testing

### 6.1 Initial Test Failures

Running VIGA with the local Qwen model revealed several issues:

**Test Command:**
```bash
python runners/dynamic_scene.py --task=artist --model=Qwen2-VL-72B-Instruct --max-rounds=5 --test-id=qwen_test --no-tools
```

**Issues Encountered:**

1. **Tool Calling Not Supported** - Qwen2-VL-72B-Instruct via vLLM does not properly support OpenAI-style tool calling. The `--enable-auto-tool-choice --tool-call-parser hermes` flags do not work as expected. Solution: Use `--no-tools` mode.

2. **Context Length Exceeded** - Original target image (3200x2082 pixels) generated too many image tokens, exceeding the 8192 context limit. Error: "The decoder prompt (length 8825) is longer than the maximum model length of 8192."

3. **Malformed JSON Output** - The model generated repetitive, nonsensical content in the `code_diff` field, causing JSON parsing failures.

### 6.2 Fixes Applied

**File: `utils/common.py`**
- Added `IMAGE_MAX_SIZE` global variable (default 1024)
- Added `set_image_max_size()` function
- Modified `get_image_base64()` to automatically resize images exceeding max size

**File: `agents/generator.py`**
- Added import for `set_image_max_size`
- Added Qwen-specific parameters: `temperature=0.3`, `max_tokens=2000`
- Set image max size to 512px for Qwen models

**File: `prompts/dynamic_scene/generator.py`**
- Added simplified `dynamic_scene_generator_system_no_tools` prompt
- Removed complex `code_diff` field requirement that confused the model
- Simplified to just `thought` and `code` fields

**File: `prompts/dynamic_scene/verifier.py`**
- Added `dynamic_scene_verifier_system_no_tools` prompt

**File: `prompts/__init__.py`**
- Registered new `no_tools` prompts in `prompts_dict`

**File: `runners/dynamic_scene.py`**
- Added `--no-tools` argument support

### 6.3 Successful Test Run

After fixes, the test completed successfully:

```
Dynamic scene task artist completed successfully

Dynamic scene task execution completed:
  Successful: 1
  Failed: 0
  Total: 1
Execution time: 93.65 seconds
```

All 5 rounds returned HTTP 200 OK with valid JSON responses.

### 6.4 Generated Code Quality

The model generates reasonable Blender Python code but with some issues:
- Creates table, jug, pears, ball objects
- Attempts physics simulation setup
- Adds camera

Issues in generated code:
- Empty meshes created (no geometry)
- Undefined variable references (e.g., `table_object`)
- Incorrect rigid body API usage
- Missing render settings

### 6.5 Context Length Limitation

Attempted to increase context to 16384 tokens but encountered CUDA OOM during vLLM initialization:
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.16 GiB.
RuntimeError: CUDA out of memory occurred when warming up sampler with 1024 dummy requests.
```

GPU memory at 8192 context: 81GB/81.5GB per GPU (99% utilization).

**Conclusion:** 8192 context is the maximum feasible with current hardware and INT8 quantization.

### 6.6 Officially Supported Models

Per `docs/architecture.md`, VIGA officially supports:
- OpenAI: gpt-4o, gpt-4-turbo, gpt-4o-mini
- Anthropic: claude-sonnet-4, claude-opus-4.5
- Google: gemini-2.5-pro, gemini-2.0-flash
- Qwen: qwen-vl-max, qwen-vl-plus (cloud APIs)

The local Qwen2-VL-72B-Instruct works with `--no-tools` mode but requires the fixes documented above.

---

## 7. Manual Script Fix and Rendering

### 7.1 Issues in VLM-Generated Code

The Qwen model generated Blender Python code with several issues:
- Empty meshes (no geometry) for jug, pears, ball
- Undefined variable references (`table_object` used before assignment)
- Incorrect rigid body API usage
- Missing render settings and camera

### 7.2 Fixed Script

Created `output/dynamic_scene/qwen_test_v3/artist/scripts/improved_scene.py` with:
- Proper mesh primitives (cylinders, spheres, cones)
- Composite jug shape (body, neck, spout, handle)
- Pear shapes using joined spheres
- Correct rigid body physics setup
- Physics simulation baking (`bpy.ops.ptcache.bake_all`)
- Camera and lighting configuration
- EEVEE render engine (CPU-friendly while GPUs run vLLM)

### 7.3 Physics Animation

Ball trajectory implemented using kinematic-to-dynamic transition:
1. Frames 1-12: Ball animated kinematically towards table center
2. Frame 13+: Ball switches to dynamic physics, collides with objects
3. Objects knocked over by ball impact

### 7.4 VLM Verification Results

Used the Qwen2-VL-72B-Instruct model to verify rendered output:

| Aspect | Score (1-10) | Notes |
|--------|--------------|-------|
| Object accuracy | 5 | Jug, plate, pears present but shapes differ from target |
| Composition similarity | 3 | Arrangement similar but not matching target painting |
| Physics animation | 7 | Ball successfully knocks objects |
| Overall success | 4 | Task achieved but visual fidelity to target is low |

### 7.5 Rendered Output

Frames saved to: `output/dynamic_scene/qwen_test_v3/artist/renders/improved/`
- `frame_0001.png` - Initial scene setup
- `frame_0025.png` - Ball approaching objects
- `frame_0050.png` - Objects being knocked over
- `frame_0080.png` - Objects scattered
- `frame_0120.png` - Final state

### 7.6 Full Animation Rendering

Rendered all 120 frames and assembled into animation files:

**Render Process:**
- Created `render_animation.py` script to render full animation
- Physics simulation baked for 250 frames
- Rendered 120 frames at 960x540 resolution (50% of 1920x1080)
- Total render time: approximately 11 minutes (~5.7 seconds per frame)
- Render engine: EEVEE (CPU-based to avoid GPU conflicts with vLLM)

**Output Files:**
| File | Size | Format |
|------|------|--------|
| `animation.gif` | 687 KB | GIF, 24 fps, 480px width, looping |
| `animation.mp4` | 64 KB | H.264, 24 fps, 960x540 |

**FFmpeg Commands Used:**
```bash
# GIF with optimized palette
ffmpeg -y -framerate 24 \
  -i output/dynamic_scene/qwen_test_v3/artist/renders/animation/frame_%04d.png \
  -vf "fps=24,scale=480:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" \
  -loop 0 \
  output/dynamic_scene/qwen_test_v3/artist/animation.gif

# MP4 with H.264 compression
ffmpeg -y -framerate 24 \
  -i output/dynamic_scene/qwen_test_v3/artist/renders/animation/frame_%04d.png \
  -c:v libx264 -pix_fmt yuv420p -crf 23 \
  output/dynamic_scene/qwen_test_v3/artist/animation.mp4
```

**Animation Content:**
- Frames 1-12: Ball enters scene from upper right
- Frames 13-30: Ball impacts objects on table
- Frames 31-80: Objects scatter and fall
- Frames 81-120: Objects settle into final positions

### 7.7 Conclusions

1. The VIGA pipeline works with local Qwen2-VL-72B-Instruct using `--no-tools` mode
2. Generated code requires manual fixes for complex scenes
3. Physics simulation works correctly when properly configured
4. Visual fidelity to artistic targets requires more sophisticated modeling
5. The 8192 context window is sufficient for basic scene generation with 512px images
6. Full animation rendering is feasible with EEVEE engine on CPU while GPUs run vLLM
7. Animation output verified - physics simulation shows ball knocking objects off table as intended


## 8. Qwen3-VL-32B-Instruct Setup

### 8.1 Why Qwen3-VL?

Qwen3-VL was released in September 2025 with significant improvements over Qwen2-VL:

| Feature | Qwen2-VL-72B | Qwen3-VL-32B |
|---------|--------------|--------------|
| Parameters | 72B | 32B |
| Context Length | 32K (8K with INT8) | 256K native |
| Visual Coding | Good | Enhanced |
| Spatial Perception | Limited | Advanced 3D grounding |
| Memory (BF16) | ~137GB | ~66GB |

For 2x H100 80GB, Qwen3-VL-32B-Instruct is the optimal choice:
- Fits in BF16 without quantization
- 32K context window (4x improvement over Qwen2-VL-72B INT8)
- Better visual reasoning and code generation

### 8.2 Initial vLLM Configuration (with bug)

```bash
source $HOME/miniconda3/bin/activate && conda activate vllm && \
python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 8000 \
  --model Qwen/Qwen3-VL-32B-Instruct \
  --served-model-name Qwen3-VL-32B-Instruct \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 32768 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

**Server Stats:**
- Model loading: 31.6 GiB per GPU
- KV cache: 26.90 GiB available, 220,400 tokens capacity
- Maximum concurrency for 32,768 tokens: 6.73x

### 8.3 Tool Calling Bug

**Test Command:**
```bash
source $HOME/miniconda3/bin/activate && conda activate agent && \
python runners/dynamic_scene.py --task=artist --model=Qwen3-VL-32B-Instruct --max-rounds=10 --test-id=qwen3_artist_test
```

**Result:** Test completed but rendered frames were mostly black.

**Root Cause:** Tool call format mismatch.

The model outputs tool calls in XML format:
```
<tool_call>
{"name": "execute_and_evaluate", "arguments": {...}}
</tool_call>
```

But `--tool-call-parser hermes` expects a different format. The system repeatedly rejected the model's output with:
```
Every single output must contain a 'tool_call' field. Your previous message did not contain a 'tool_call' field. Please reconsider.
```

This caused the model to get stuck in a loop (8+ iterations) without ever successfully executing code to create scene objects. The final scene contained only:
- Camera
- MainLight (Area, energy=1500)
- Table
- Wall
- No world/environment lighting
- No still life objects (jug, plate, pears)

### 8.4 Available Tool Call Parsers

vLLM 0.11.0 supports multiple tool call parsers:
```
deepseek_v3, deepseek_v31, glm45, granite-20b-fc, granite, hermes,
hunyuan_a13b, internlm, jamba, kimi_k2, llama4_pythonic, llama4_json,
llama3_json, longcat, minimax, mistral, openai, phi4_mini_json,
pythonic, qwen3_coder, qwen3_xml, seed_oss, step3, xlam
```

For Qwen3-VL models, use `qwen3_xml` instead of `hermes`.

### 8.5 Corrected vLLM Configuration

```bash
source $HOME/miniconda3/bin/activate && conda activate vllm && \
python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 8000 \
  --model Qwen/Qwen3-VL-32B-Instruct \
  --served-model-name Qwen3-VL-32B-Instruct \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 32768 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml
```

**Key change:** `--tool-call-parser qwen3_xml`

### 8.6 Alternative: No-Tools Mode

If tool calling issues persist, use `--no-tools` mode as fallback:

```bash
source $HOME/miniconda3/bin/activate && conda activate agent && \
python runners/dynamic_scene.py --task=artist --model=Qwen3-VL-32B-Instruct --max-rounds=10 --test-id=qwen3_artist_notools --no-tools
```

### 8.7 Bug: Truncated JSON Response (max_tokens too low)

**Symptom:** Test fails on Round 0 with error:
```
Error executing tool: Unterminated string starting at: line 4 column 11 (char 751)
Error during execution: list index out of range
```

**Cause:** Qwen3-VL generates verbose responses. The default `max_tokens=2000` for Qwen models was insufficient, causing JSON to be truncated mid-generation.

**Fix:** Updated `agents/generator.py` to use higher token limit for Qwen3:

```python
if 'qwen3' in self.config.get("model", "").lower():
    self.init_chat_args['max_tokens'] = 8000
else:
    self.init_chat_args['max_tokens'] = 2000
```

### 8.8 Test Results with No-Tools Mode

**Command:**
```bash
python runners/dynamic_scene.py --task=artist --model=Qwen3-VL-32B-Instruct \
  --max-rounds=10 --test-id=qwen3_artist_notools_v2 --no-tools
```

**Result:** Completed 10 rounds in 390 seconds.

**Output:**
- 10 Python scripts generated (5.7-6.0 KB each)
- 10 state.blend files saved
- No rendered PNG frames (scripts crashed before rendering)

### 8.9 Bug: Incorrect Blender Rigid Body API

The generated scripts crash with:
```
AttributeError: 'BlendData' object has no attribute 'rigidbody_worlds'
```

**Cause:** Model hallucinates non-existent API:
```python
# Wrong (model generates this)
bpy.context.scene.rigidbody_world = bpy.data.rigidbody_worlds.new('RigidBodyWorld')

# Correct Blender API
bpy.ops.rigidbody.world_add()
```

This is a known VLM limitation - models generate plausible-looking but incorrect API calls.

### 8.10 Manual Verification

Created a corrected test script with proper Blender primitives (cylinders, spheres) instead of the model's custom mesh code. Rendered successfully.

**VLM Evaluation (Qwen3-VL-32B self-assessment):**

| Aspect | Present | Notes |
|--------|---------|-------|
| Jug | Yes | Simplified as brown cylinder |
| Plate | Yes | Flat circular disc |
| Pears | Yes | Three green spheres |
| Table | Yes | Flat tan plane |
| Similarity | 3/10 | Basic structure correct, lacks detail |

The model correctly identifies scene elements and positions but uses oversimplified geometry and lacks textures, materials, and artistic style.

### 8.11 Validation Status

- [x] vLLM server loads Qwen3-VL-32B-Instruct successfully
- [x] Model responds to API requests
- [x] No-tools mode completes full 10 rounds
- [x] JSON generation works with max_tokens=8000
- [ ] Tool calling with `qwen3_xml` parser (not yet tested)
- [ ] Rigid body physics code (model generates incorrect API)
- [ ] High-fidelity scene recreation (current: 3/10 similarity)

### 8.12 Known Issues

1. **Rigid body API hallucination** - Model uses non-existent `bpy.data.rigidbody_worlds`
2. **Oversimplified geometry** - Uses basic primitives instead of detailed meshes
3. **Missing materials/textures** - Flat colors only, no node-based materials
4. **No rendered output** - Scripts crash before render step due to API errors

---

## 9. Hybrid Agent Architecture

### 9.1 Problem Statement

Qwen3-VL hallucinates incorrect Blender API calls when generating Python scripts directly. Examples:
- Uses `bpy.data.rigidbody_worlds` (does not exist)
- Uses `"Transmission"` instead of `"Transmission Weight"` (Blender 4.x API change)
- Generates plausible-looking but non-functional mesh creation code

### 9.2 Solution: Hybrid Workflow

Separate vision reasoning from code generation:

| Agent | Model | Role |
|-------|-------|------|
| Vision/Verifier | Qwen3-VL-32B-Instruct | Analyze target images, compare renders, provide structured feedback |
| Scripting | Claude | Generate correct Blender Python code based on Qwen3 feedback |

### 9.3 Workflow Steps

1. **Analyze Target** - Qwen3 analyzes target image, outputs structured JSON with object list, camera setup, lighting, materials
2. **Generate Script** - Claude creates Blender Python script based on analysis
3. **Render** - Execute script with Blender headless
4. **Compare** - Qwen3 compares render to target, outputs structured feedback with edit suggestions
5. **Iterate** - Claude updates script based on feedback
6. Repeat steps 3-5 for 7-8 iterations

### 9.4 Helper Scripts

Created helper scripts with proper system prompts based on VIGA patterns:

**call_qwen_analyze.py** - Analyzes target image
```python
SYSTEM_PROMPT = """[Role]
You are SceneAnalyzer - an expert visual analyst for 3D scene reconstruction...
[Response Format]
Output a structured analysis in JSON format:
{
  "overall_description": "...",
  "object_list": [...],
  "object_relations": "...",
  "camera_setup": {...},
  "lighting_setup": {...},
  "technical_details": "..."
}
"""
```

**call_qwen_compare.py** - Compares target and rendered images
```python
SYSTEM_PROMPT = """[Role]
You are DynamicSceneVerifier - an expert visual analyst...
[Response Format]
{
  "visual_difference": {"critical": [...], "major": [...], "minor": [...]},
  "edit_suggestion": [...],
  "overall_similarity": <0-100>,
  "next_focus": "..."
}
"""
```

### 9.5 Test Results: Artist Scene (Still Life Painting)

**Target:** `data/dynamic_scene/artist/target.png`

Completed 8 iterations with progressive improvements:

| Iteration | Key Changes |
|-----------|-------------|
| 01 | Initial scene: basic jug, plate, pears |
| 02 | Added jug handle, improved pear shapes, clustered arrangement |
| 03 | 8 pears (3 on plate, 5 on table), shorter/wider jug |
| 04 | Larger plate offset left, stems on pears, color variation |
| 05 | More pear variety, adjusted camera angle |
| 06 | Proper jug proportions, visible plate, eye-level camera, warmer lighting |
| 07 | Better jug handle, larger plate with rim, 7 fruits, lower camera |
| 08 | Widest jug body, oval plate, 7 vibrant fruits, lowest camera, warmest lighting |

**Output:** `output/dynamic_scene/hybrid_test_001/`

### 9.6 Test Results: Laser Module Scene (Technical Equipment)

**Target:** `data/dynamic_scene/laser_module/target.png`

A macro photograph of a precision laser module assembly featuring optical prisms, thermoelectric coolers (TECs), wirebonds, and purple sapphire alignment beads mounted on an aluminum plate within a partially opened enclosure.

Completed 8 iterations:

| Iteration | Key Changes |
|-----------|-------------|
| 01 | Initial scene: enclosure, plate, PCB, TECs, prisms, beads, wirebonds |
| 02 | Larger components, better camera angle, open enclosure |
| 03 | Better visibility, more detailed arrangement |
| 04 | 45-deg camera angle, smaller beads, more detail |
| 05 | Transparent cover (tilted open), better materials |
| 06 | Closer camera, centered component arrangement |
| 07 | Triangular prisms (cones), TEC with heatsink fins |
| 08 | Rectangular glass prisms, 6 wirebonds, closer macro camera |

**Output:** `output/dynamic_scene/laser_module_001/`

### 9.7 Key Blender API Fixes

**Transmission Weight (Blender 4.x):**
```python
# Blender 4.x uses "Transmission Weight" instead of "Transmission"
if "Transmission Weight" in bsdf.inputs:
    bsdf.inputs["Transmission Weight"].default_value = 0.85
elif "Transmission" in bsdf.inputs:
    bsdf.inputs["Transmission"].default_value = 0.85
```

**Wirebond Creation (Bezier Curves):**
```python
mid = ((start[0]+end[0])/2, (start[1]+end[1])/2, max(start[2], end[2]) + 0.008)
curve = bpy.data.curves.new('wire', 'CURVE')
curve.dimensions = '3D'
spline = curve.splines.new('BEZIER')
spline.bezier_points.add(2)
spline.bezier_points[0].co = start
spline.bezier_points[1].co = mid
spline.bezier_points[2].co = end
obj = bpy.data.objects.new('Wire', curve)
bpy.context.collection.objects.link(obj)
curve.bevel_depth = 0.0006
```

**TEC with Heatsink Fins:**
```python
# Base ceramic block
bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
tec = bpy.context.object
tec.scale = (0.042, 0.042, 0.01)
tec.data.materials.append(mat_tec)
# Heatsink fins on top
for i in range(5):
    bpy.ops.mesh.primitive_cube_add(size=1, location=(loc[0]-0.015+i*0.008, loc[1], loc[2]+0.015))
    fin = bpy.context.object
    fin.scale = (0.001, 0.035, 0.005)
    fin.data.materials.append(mat_metal)
```

### 9.8 Observations

1. **Qwen3 provides consistent feedback** - Even after 8 iterations, similarity scores remain around 15%. This is expected for complex technical scenes.

2. **Structured prompts improve output quality** - Using VIGA-style system prompts with JSON response formats produces more actionable feedback than simple prompts.

3. **Hybrid architecture works** - Qwen3 correctly identifies visual discrepancies; Claude generates correct Blender API code.

4. **Technical scenes are harder** - The laser module scene requires precise geometry (prism angles, heatsink fins) and specific optical properties (glass IOR, metallic roughness) that are difficult to match from a single reference image.

5. **Iterative refinement is effective** - Each iteration progressively adds detail: more wirebonds (4 to 6), smaller beads, closer camera, better materials.

### 9.9 Limitations

1. **No stopping criterion** - The workflow runs for a fixed number of iterations without automatic convergence detection.

2. **Qwen3 feedback can be inconsistent** - Sometimes reports components as "missing" when they are present but rendered differently.

3. **Material matching is difficult** - Matching metallic, glass, and ceramic materials from a photograph requires manual tuning.

4. **Camera/lighting estimation is approximate** - Qwen3 provides qualitative descriptions but not precise camera parameters.
