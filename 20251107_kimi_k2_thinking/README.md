# Kimi-K2-Thinking Deployment on 8× H100

## Overview
This deployment runs the Kimi-K2-Thinking model (1T params, 32B active) on 8× NVIDIA H100 80GB GPUs using vLLM with native INT4 quantization.

**Configuration:**
- **Model:** moonshotai/Kimi-K2-Thinking
- **Hardware:** 8× H100 80GB HBM3 (HGX system with NVLink)
- **Inference Engine:** vLLM 0.8.5.post1
- **Quantization:** Native INT4 (compressed-tensors)
- **Context Length:** 256k tokens
- **Optimization:** Long-context, low-throughput

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

## Advanced Configuration

### Increase Throughput (Trade-off: Shorter Context)
Edit `launch_kimi_k2.sh`:
```bash
--max-model-len 131072 \        # Reduce to 128k
--max-num-batched-tokens 65536 \ # Increase batch size
--max-num-seqs 8 \               # Allow more concurrent requests
```

### Maximize Context Length (Trade-off: Lower Throughput)
```bash
--max-model-len 262144 \         # Full 256k
--max-num-batched-tokens 16384 \ # Smaller batches
--max-num-seqs 2 \               # Fewer concurrent requests
```

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

