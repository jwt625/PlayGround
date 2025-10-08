# DevLog-003: OOM Debug and Server Testing

**Date**: 2025-10-08  
**Status**: ✅ Resolved  
**Author**: Development Team

## Problem Statement

GLM-4.5-Air-FP8 server was experiencing CUDA Out of Memory (OOM) errors during initialization on 2x H100 GPUs (2x 80GB), preventing the server from starting.

## Initial Error

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 60.00 MiB. 
GPU 1 has a total capacity of 79.19 GiB of which 39.06 MiB is free. 
Including non-PyTorch memory, this process has 79.14 GiB memory in use. 
Of the allocated memory 72.36 GiB is allocated by PyTorch, 
with 436.00 MiB allocated in private pools (e.g., CUDA Graphs), 
and 71.59 MiB is reserved by PyTorch but unallocated.
```

## Root Cause Analysis

### Memory Breakdown (per GPU with TP=2)

1. **Model Size**: 105 GB on disk (FP8 quantized)
2. **Per GPU with Tensor Parallelism**: ~52.5 GB model weights
3. **KV Cache**: 15-25 GB (with `max_num_seqs=512`, `max_model_len=16384`)
4. **CUDA Graphs**: 0.5-2 GB during capture
5. **Activations & Overhead**: 3-7 GB

**Total Required**: ~71-86 GB per GPU

### Initial Configuration Issues

```bash
--gpu-memory-utilization 0.8  # Only 64.8 GB allocated
--max-model-len 16384
--max-num-seqs 512  # Default from config
```

**Problem**: 64.8 GB < 71 GB minimum required

### Critical Bug Discovered

The `--enforce-eager` flag was not being passed through to the vLLM engine:

1. `start_server.py` created config with `enforce_eager=True`
2. `run_server()` didn't accept config parameter
3. `api_server.py` created fresh config, ignoring CLI arguments
4. CUDA graphs were still being captured, consuming extra memory

## Solution Steps

### Step 1: Add `--enforce-eager` CLI Flag

**File**: `scripts/start_server.py`

```python
parser.add_argument(
    "--enforce-eager",
    action="store_true",
    help="Disable CUDA graphs (use eager mode). Use this if you get OOM during initialization"
)

# Pass to config
config.enforce_eager = args.enforce_eager
```

### Step 2: Fix Config Pass-Through

**File**: `src/glm_server/api_server.py`

```python
def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    model_path: str = "models/GLM-4.5-Air-FP8",
    log_level: str = "info",
    server_config: InferenceConfig | None = None  # Added
):
    global config
    
    # Use provided config if available
    if server_config:
        config = server_config
```

**File**: `scripts/start_server.py`

```python
run_server(
    host=config.host,
    port=config.port,
    model_path=config.model_path,
    log_level=args.log_level,
    server_config=config  # Added
)
```

### Step 3: Reduce Memory Footprint

**Key insight**: `max_num_seqs` has massive impact on KV cache memory.

```
KV cache memory ≈ max_num_seqs × max_model_len × (model dimensions)
```

**Memory savings from reducing `max_num_seqs`**:
- `512 → 8`: **64x reduction** = ~18-22 GB saved per GPU
- `256 → 8`: **32x reduction** = ~9-12 GB saved per GPU

### Step 4: Final Working Configuration

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run python scripts/start_server.py \
    --gpu-memory-utilization 0.85 \
    --max-model-len 65536 \
    --max-num-seqs 8 \
    --enforce-eager \
    --api-key "glm-4pZbgPw71IKknGxeCbT3znqKzNscqgAnQNUdFPE99Lw"
```

**Trade-offs**:
- ✅ Server starts successfully
- ✅ 65K context length (4x original 16K)
- ✅ 8 concurrent requests (sufficient for most use cases)
- ⚠️ No CUDA graphs (5-15% slower inference)

## Additional Bug Fixes

### Bug: `top_k` Parameter Handling

**Error**: `'<' not supported between instances of 'NoneType' and 'int'`

**Root Cause**: vLLM's `SamplingParams` doesn't accept `None` for `top_k`

**Fix**: Only pass `top_k` if it's a positive value

**File**: `src/glm_server/vllm_server.py`

```python
# Build kwargs for SamplingParams
sampling_kwargs = {
    "max_tokens": max_tokens,
    "temperature": temperature,
    "top_p": top_p,
    "stop": stop or [],
    **kwargs
}

# Only add top_k if it's a positive value
if top_k and top_k > 0:
    sampling_kwargs["top_k"] = top_k

sampling_params = SamplingParams(**sampling_kwargs)
```

## Server Testing Results

### Test 1: Authentication - Without Token

**Request**:
```bash
curl -X POST "http://localhost:8000/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how are you?", "max_tokens": 50}'
```

**Response**: ✅ PASS
```json
{"detail":"Not authenticated"}
```

### Test 2: Authentication - With Valid Token

**Request**:
```bash
curl -X POST "http://localhost:8000/v1/completions" \
  -H "Authorization: Bearer glm-4pZbgPw71IKknGxeCbT3znqKzNscqgAnQNUdFPE99Lw" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is 2+2?", "max_tokens": 50}'
```

**Response**: ✅ PASS
```json
{
  "id": "92510954cce14d0291ef8ddfdef84019",
  "object": "text_completion",
  "created": 1759886655,
  "model": "/home/ubuntu/.../GLM-4.5-Air-FP8",
  "choices": [{
    "text": " The answer may seem obvious...",
    "index": 0,
    "finish_reason": "length"
  }],
  "usage": {
    "prompt_tokens": 7,
    "completion_tokens": 50,
    "total_tokens": 57
  },
  "generation_time": 3.699
}
```

**Performance**: ~13.5 tokens/sec

### Test 3: Proper GLM Chat Format

**Request**:
```bash
curl -X POST "http://localhost:8000/v1/completions" \
  -H "Authorization: Bearer glm-4pZbgPw71IKknGxeCbT3znqKzNscqgAnQNUdFPE99Lw" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "[gMASK]<sop><|system|>\nYou are a helpful AI assistant.<|user|>\nWhat is 2+2?<|assistant|>\n",
    "max_tokens": 50,
    "temperature": 0.3
  }'
```

**Response**: ✅ PASS
```json
{
  "choices": [{
    "text": "<think>The question is asking for the result of a simple arithmetic operation: 2 + 2.\n\nThis is a basic addition problem. 2 + 2 equals 4.\n\nI'll provide a direct and concise answer as requested.</think>4"
  }]
}
```

**Note**: GLM-4.5's thinking mode shows reasoning in `<think>` tags before answering.

### Test 4: Health Check

**Request**:
```bash
curl http://localhost:8000/health
```

**Response**: ✅ PASS
```json
{
  "status": "healthy",
  "model": "/home/ubuntu/.../GLM-4.5-Air-FP8",
  "tensor_parallel_size": 2,
  "startup_time": 41.84,
  "gpu_memory_usage": null
}
```

## Performance Metrics

- **Startup Time**: 41.8 seconds
- **Throughput**: ~13.5 tokens/sec (with `--enforce-eager`)
- **Expected with CUDA graphs**: ~15-16 tokens/sec (10-15% faster)
- **Latency**: ~3.7 seconds for 50 tokens
- **Memory Usage**: ~79 GB per GPU (near capacity)

## Lessons Learned

1. **KV cache dominates memory** for large `max_num_seqs` values
2. **CUDA graphs require extra memory** during initialization
3. **Config pass-through is critical** - CLI args must reach the engine
4. **Tensor parallelism splits model** but each GPU still needs full KV cache
5. **FP8 quantization helps** but 105GB model still needs ~52.5GB per GPU
6. **`max_num_seqs=8` is often sufficient** for development and moderate traffic

## Recommendations

### For Development
```bash
--gpu-memory-utilization 0.85
--max-model-len 65536
--max-num-seqs 8
--enforce-eager
```

### For Production (if memory allows)
Try removing `--enforce-eager` to enable CUDA graphs for better performance. If OOM occurs, reduce `max_model_len` or `max_num_seqs`.

### For High Traffic
Consider:
- Reducing `max_model_len` to 32768 or 16384
- Increasing `max_num_seqs` to 32-64
- Using `--enforce-eager` if needed

## GLM Chat Format

For best results, use proper GLM-4.5 chat format:

```
[gMASK]<sop><|system|>
{system_prompt}
<|user|>
{user_message}
<|assistant|>
```

This ensures the model understands the conversation structure and provides better responses.

## Status

✅ **Server is operational** with authentication and text generation working correctly.

