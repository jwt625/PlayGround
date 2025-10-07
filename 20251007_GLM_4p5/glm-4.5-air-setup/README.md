# GLM-4.5-Air Server Setup

Production-ready GLM-4.5-Air inference server optimized for 2x H100 GPUs.

## Quick Start

### 1. Install Dependencies

```bash
# Install all dependencies including dev tools
uv sync --extra dev
```

### 2. Download Model (if not already done)

```bash
uv run python scripts/download_model.py
```

### 3. Start the Server

**Option A: Using the Python script (recommended)**
```bash
uv run python scripts/start_server.py
```

**Option B: With custom configuration**
```bash
uv run python scripts/start_server.py \
    --host 0.0.0.0 \
    --port 8000 \
    --model-path models/GLM-4.5-Air-FP8 \
    --tensor-parallel-size 2 \
    --max-model-len 32768
```

**Option C: Direct vLLM command**
```bash
uv run vllm serve models/GLM-4.5-Air-FP8 \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 32768 \
    --trust-remote-code \
    --api-key "your-api-key-here"
```

### 4. Test the Server

Once the server is running, test it:

```bash
# Run comprehensive vLLM tests
uv run python scripts/test_vllm.py

# Or run specific tests
uv run python scripts/test_vllm.py --test basic
uv run python scripts/test_vllm.py --test streaming
uv run python scripts/test_vllm.py --test benchmark
```

## Configuration

### Optimal H100 Settings

The server is pre-configured with optimal settings for 2x H100 GPUs:

- **Tensor Parallelism**: 2 GPUs
- **GPU Memory**: 95% utilization (~76GB per GPU)
- **Context Length**: 32,768 tokens
- **Max Sequences**: 512 concurrent requests
- **Quantization**: FP8 compressed-tensors
- **Optimizations**: Prefix caching, chunked prefill, CUDA graphs

### API Endpoints

- **Health Check**: `GET /health`
- **Completions**: `POST /v1/completions`
- **Chat**: `POST /v1/chat/completions`
- **Models**: `GET /v1/models`

### Authentication

All endpoints require Bearer token authentication:
```bash
Authorization: Bearer glm-your-api-key-here
```

## Example Usage

### Basic Completion
```bash
curl -X POST "http://localhost:8000/v1/completions" \
  -H "Authorization: Bearer glm-your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-4.5-air",
    "prompt": "Explain quantum computing:",
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

### Chat Completion
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Authorization: Bearer glm-your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-4.5-air",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "max_tokens": 150
  }'
```

## Performance

Expected performance on 2x H100:
- **Throughput**: 70-90 tokens/second
- **Latency**: ~1-2 seconds for 100 tokens
- **Memory Usage**: ~101GB GPU memory total
- **Context**: Up to 32K tokens
- **Concurrent Users**: 512 sequences

## Project Structure

```
glm-4.5-air-setup/
├── scripts/
│   ├── download_model.py    # Model download utility
│   ├── start_server.py      # Production server launcher
│   └── test_vllm.py         # Comprehensive vLLM tests
├── src/
│   └── glm_server/
│       ├── config.py        # Configuration management
│       ├── main.py          # Main entry point
│       ├── api_server.py    # FastAPI server implementation
│       ├── vllm_server.py   # vLLM inference engine wrapper
│       └── model_downloader.py  # Model download utilities
├── tests/
│   ├── test_config.py       # Configuration tests
│   └── test_server.py       # Server tests
├── models/
│   └── GLM-4.5-Air-FP8/     # Model files (104.85 GB)
├── pyproject.toml           # Project configuration
└── README.md                # This file
```

## Development

### Code Quality

This project uses strict type checking and code quality tools:

```bash
# Run linting
uv run ruff check .

# Run formatting
uv run ruff format .

# Run type checking
uv run mypy src/

# Run tests
uv run pytest tests/

# Run all quality checks
uv run ruff check . && uv run ruff format . && uv run mypy src/ && uv run pytest tests/
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test file
uv run pytest tests/test_config.py

# Run vLLM integration tests
uv run python scripts/test_vllm.py --test all
```