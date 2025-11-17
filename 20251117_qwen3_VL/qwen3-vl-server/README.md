# Qwen3-VL-32B-Instruct vLLM Server

A production-ready vLLM server for Qwen3-VL-32B-Instruct with token-based authentication, optimized for NVIDIA H100 GPUs.

## Features

- Token-based authentication for secure API access
- OpenAI-compatible API endpoints
- Optimized for H100 GPUs with configurable settings
- Async scheduling for better performance
- Support for both image and video inputs
- Pre-cached model weights for fast startup

## Quick Start

### 1. Start the Server

```bash
cd qwen3-vl-server
uv run python server.py
```

The server will start on `http://0.0.0.0:8000` by default.

### 2. Test the Server

In a new terminal:

```bash
cd qwen3-vl-server
uv run python test_client.py
```

## Configuration

All configuration is done via the `.env` file. Key settings:

- `API_KEY`: Authentication token (required for all requests)
- `MODEL_NAME`: Model to serve (default: Qwen/Qwen3-VL-32B-Instruct)
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)
- `TENSOR_PARALLEL_SIZE`: Number of GPUs to use (1 or 2)
- `GPU_MEMORY_UTILIZATION`: GPU memory fraction (0.0-1.0, default: 0.95)
- `MAX_MODEL_LEN`: Maximum context length (default: 128000)
- `DTYPE`: Data type (bfloat16, float16, auto)
- `ASYNC_SCHEDULING`: Enable async scheduling (true/false)
- `DISABLE_VIDEO`: Disable video inputs if only using images (true/false)

See `.env.example` for all available options.

## API Usage

### Using OpenAI Python Client

```python
from openai import OpenAI

client = OpenAI(
    api_key="qwen3vl-sk-7f8a9b2c4d6e1f3a5b7c9d2e4f6a8b0c1d3e5f7a9b2c4d6e8f0a2c4e6a8b0c2d4",
    base_url="http://localhost:8000/v1"
)

# Text completion
response = client.chat.completions.create(
    model="Qwen/Qwen3-VL-32B-Instruct",
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ],
    max_tokens=100
)

print(response.choices[0].message.content)
```

### Using cURL

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer qwen3vl-sk-7f8a9b2c4d6e1f3a5b7c9d2e4f6a8b0c1d3e5f7a9b2c4d6e8f0a2c4e6a8b0c2d4" \
  -d '{
    "model": "Qwen/Qwen3-VL-32B-Instruct",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 100
  }'
```

### Image Understanding

```python
response = client.chat.completions.create(
    model="Qwen/Qwen3-VL-32B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image.jpg"}
                },
                {
                    "type": "text",
                    "text": "What do you see in this image?"
                }
            ]
        }
    ],
    max_tokens=200
)
```

## Performance Tuning

### Single GPU (H100 80GB)

The default configuration uses a single GPU and should work well for most use cases:

```env
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.95
MAX_MODEL_LEN=128000
```

### Dual GPU (2x H100 80GB)

To use both H100 GPUs for higher throughput:

```env
TENSOR_PARALLEL_SIZE=2
GPU_MEMORY_UTILIZATION=0.95
MAX_MODEL_LEN=128000
```

### Memory Optimization

If you encounter OOM errors, reduce:

```env
GPU_MEMORY_UTILIZATION=0.85
MAX_MODEL_LEN=65536
```

### Image-Only Workloads

If you only process images (no video):

```env
DISABLE_VIDEO=true
```

## Monitoring

Check GPU usage:

```bash
nvidia-smi
```

Check server logs for performance metrics and errors.

## Troubleshooting

### Server won't start

1. Check that the API_KEY is set in `.env`
2. Verify GPU availability: `nvidia-smi`
3. Check port availability: `lsof -i :8000`

### Out of Memory

1. Reduce `GPU_MEMORY_UTILIZATION` to 0.85 or 0.80
2. Reduce `MAX_MODEL_LEN` to 65536 or 32768
3. Use both GPUs with `TENSOR_PARALLEL_SIZE=2`

### Slow inference

1. Enable `ASYNC_SCHEDULING=true`
2. Use both GPUs with `TENSOR_PARALLEL_SIZE=2`
3. Ensure `DTYPE=bfloat16` for H100 optimization

## API Endpoints

The server provides OpenAI-compatible endpoints:

- `POST /v1/chat/completions` - Chat completions
- `GET /v1/models` - List available models
- `GET /health` - Health check
- `GET /version` - Server version

## Security

- The API key is required for all requests
- Keep your `.env` file secure and never commit it to version control
- The `.env` file is already in `.gitignore`
- Consider using a reverse proxy (nginx) with SSL for production deployments

## Model Information

- **Model**: Qwen3-VL-32B-Instruct
- **Parameters**: 32B (dense)
- **Context Length**: Up to 262K tokens (configurable)
- **Modalities**: Text, Image, Video
- **Recommended Hardware**: NVIDIA H100 80GB (1-2 GPUs)

## License

This server implementation is provided as-is. The Qwen3-VL model has its own license terms from Alibaba Cloud.
