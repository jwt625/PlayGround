# Kimi-K2-Thinking Deployment on 8× H100

## Overview
This deployment runs the Kimi-K2-Thinking model (1T params, 32B active) on 8× NVIDIA H100 80GB GPUs using vLLM with native INT4 quantization.

**Current Configuration:**
- **Model:** moonshotai/Kimi-K2-Thinking
- **Hardware:** 8× H100 80GB HBM3 (HGX system with NVLink)
- **Inference Engine:** vLLM nightly (v0.11.1rc6.dev211+g934a9c3b7)
- **Quantization:** Native INT4 (compressed-tensors)
- **Context Length:** 12k tokens (maximum supported with current memory constraints)
- **Optimization:** Memory-efficient with `--enforce-eager`
- **Status:** 🟢 **RUNNING** on port 8000

## Research Summary: vLLM INT4 Support

✅ **Confirmed:** vLLM supports Kimi-K2-Thinking with compressed-tensors INT4 format

**Evidence:**
1. **Day 0 Support:** vLLM announced day-0 support for Kimi-K2 in partnership with Moonshot AI (source: X/Twitter @mgoin_)
2. **Compressed-Tensors:** vLLM has native support for compressed-tensors format (used by Kimi-K2's INT4 weights)
3. **Model Card:** HuggingFace model card explicitly lists vLLM as a supported inference engine
4. **Version:** vLLM 0.8.5+ includes Kimi-K2 parser support (`--reasoning-parser kimi_k2`)

## Quick Start

### 1. Check Model Download Status
```bash
du -sh /home/ubuntu/models/Kimi-K2-Thinking
# Expected: ~594 GB when complete
```

### 2. Launch Server
```bash
./launch_kimi_k2.sh
```

The server will start on `http://0.0.0.0:8000` with OpenAI-compatible API.

### 3. Test API
In a new terminal:
```bash
source .venv/bin/activate
python test_api.py
```

## API Usage

### Python (OpenAI SDK)
```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy-key"
)

response = client.chat.completions.create(
    model="kimi-k2-thinking",
    messages=[
        {"role": "system", "content": "You are Kimi, an AI assistant created by Moonshot AI."},
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    temperature=1.0,
    max_tokens=4096
)

print(response.choices[0].message.content)

# Access reasoning content (thinking process)
if hasattr(response.choices[0].message, 'reasoning_content'):
    print("Reasoning:", response.choices[0].message.reasoning_content)
```

### cURL
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kimi-k2-thinking",
    "messages": [
      {"role": "system", "content": "You are Kimi, an AI assistant created by Moonshot AI."},
      {"role": "user", "content": "What is 84 * 3 / 2?"}
    ],
    "temperature": 1.0,
    "max_tokens": 2048
  }'
```

## Configuration Details

### Launch Script Parameters
- `--tensor-parallel-size 8`: Distribute model across all 8 H100 GPUs
- `--max-model-len 262144`: Support full 256k context window
- `--max-num-batched-tokens 32768`: Batch size optimized for long contexts
- `--max-num-seqs 4`: Low concurrency (4 parallel requests) for low-throughput use case
- `--gpu-memory-utilization 0.95`: Use 95% of 640GB total VRAM
- `--enable-chunked-prefill`: Essential for processing long contexts efficiently
- `--enable-prefix-caching`: Cache common prefixes to speed up repeated queries
- `--load-format compressed-tensors`: Load native INT4 quantized weights

### Memory Footprint
- **Model weights:** ~594 GB (INT4 quantized)
- **Available VRAM:** 640 GB (8 × 80 GB)
- **KV cache + activations:** ~46 GB (with 95% utilization)
- **Max context per request:** 256k tokens (with chunked prefill)

### Performance Expectations
- **Prefill (long context):** ~500-1000 tokens/sec (with chunked prefill)
- **Decode:** ~40-60 tokens/sec per request
- **Concurrency:** 1-4 parallel requests recommended
- **Latency:** Higher due to thinking/reasoning overhead (expected behavior)

## Monitoring

### Check GPU Usage
```bash
watch -n 1 nvidia-smi
```

### Check Server Logs
The launch script outputs logs to stdout. Look for:
- Model loading progress
- GPU memory allocation
- Request processing times

### Health Check
```bash
curl http://localhost:8000/health
```

## Troubleshooting

### Model Not Loading
- **Issue:** OOM during model load
- **Solution:** Reduce `--gpu-memory-utilization` to 0.90 or lower

### Slow Inference
- **Issue:** Very slow token generation
- **Solution:** This is expected for thinking models. Kimi-K2 generates reasoning tokens before the final answer.

### Context Length Errors
- **Issue:** "Context length exceeded" errors
- **Solution:** Reduce `--max-model-len` to 131072 (128k) or enable more aggressive chunking

### Connection Refused
- **Issue:** Cannot connect to API
- **Solution:** Check if server is running: `ps aux | grep vllm`

## Memory Optimization Guide

### Understanding `gpu_memory_utilization`
- **What it controls:** Total memory budget (model weights + KV cache + activations)
- **Model weights:** ~74GB per GPU (fixed)
- **Available for KV cache:** `(gpu_memory_utilization × 79GB) - 74GB`
- **Example:** With 0.95 → `(0.95 × 79) - 74 = 1.05GB` for KV cache

### The `--enforce-eager` Flag (CRITICAL for this model)
**Problem:** CUDA graph capture causes memory spike during initialization → OOM
**Solution:** Add `--enforce-eager` to disable CUDA graphs

**Trade-offs:**
- ✅ Eliminates memory spike during startup
- ✅ Allows higher `gpu_memory_utilization` (0.95)
- ❌ ~10-20% slower inference (no CUDA graph optimization)

**When to use:**
- ✅ **Always use for Kimi-K2-Thinking** - model is too large for CUDA graphs on H100 80GB
- ✅ When getting OOM during "determine_available_memory" phase
- ❌ Don't use if you have excess memory and want maximum performance

### Current Working Configuration (12k context - MAXIMUM)
```bash
--max-model-len 12288             # 12k tokens (max supported)
--max-num-batched-tokens 4096     # Conservative batch size
--max-num-seqs 1                  # Single request
--gpu-memory-utilization 0.95     # Use 95% of GPU memory
--enforce-eager                   # Disable CUDA graphs (required)
```

### Context Length Limitations

**With `--enforce-eager` (current setup):**
- **Maximum**: ~13k tokens
- **Recommended**: 12k tokens
- **Available KV cache**: 0.86 GiB
- **Limitation**: Model weights (~74GB) consume most of GPU memory

**Tested configurations:**
- ✅ **8k tokens**: Works perfectly
- ✅ **12k tokens**: Works (near maximum)
- ❌ **32k tokens**: Fails - needs 2.14 GiB KV cache, only 0.86 GiB available

**Why can't we go higher?**
1. Model weights: ~74 GB per GPU (fixed)
2. Total GPU memory: 79 GB per GPU
3. Available after model: ~5 GB per GPU
4. With `gpu_memory_utilization=0.95`: Only 0.86 GiB allocated for KV cache
5. KV cache scales linearly with context length

**To support longer contexts (experimental - may not work):**
- Remove `--enforce-eager` (but this causes OOM during initialization)
- Use smaller model or wait for vLLM optimizations
- Use multiple nodes with pipeline parallelism (not currently supported for this model)

## Files
- `launch_kimi_k2.sh` - Main launch script
- `test_api.py` - API test suite
- `.venv/` - Python virtual environment (uv-managed)
- `/home/ubuntu/models/Kimi-K2-Thinking/` - Model weights

## References
- [Kimi-K2-Thinking Model Card](https://huggingface.co/moonshotai/Kimi-K2-Thinking)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Compressed-Tensors Format](https://github.com/vllm-project/llm-compressor)
- [OpenAI API Compatibility](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)

