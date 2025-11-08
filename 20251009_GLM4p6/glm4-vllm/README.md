# GLM-4.6 vLLM API Server

A high-performance API server for the GLM-4.6 (357B parameter) language model using vLLM inference engine with secure authentication and multi-GPU support.

## Features

- **High-Performance Inference**: Powered by vLLM with tensor parallelism across multiple GPUs
- **Secure Authentication**: Bearer token authentication with configurable tokens
- **RESTful API**: FastAPI-based endpoints with automatic documentation
- **Multi-GPU Support**: Distributed inference across 8x NVIDIA B200 GPUs
- **Production Ready**: Background execution with logging and process management
- **Flexible Parameters**: Configurable temperature, top-p, top-k, and stop sequences

## Requirements

### Hardware
- **GPUs**: 8x NVIDIA B200 (or equivalent high-memory GPUs)
- **Memory**: ~660GB GPU memory total (82.5GB per GPU)
- **CPU**: Multi-core processor (104+ cores recommended)
- **RAM**: 32GB+ system memory

### Software
- **OS**: Linux (Ubuntu 20.04+ recommended)
- **Python**: 3.10+
- **CUDA**: 12.8+
- **Driver**: NVIDIA 570.148.08+

## Installation

1. **Install uv package manager**:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   export PATH="$HOME/.local/bin:$PATH"
   ```

2. **Clone and setup project**:
   ```bash
   git clone <repository-url>
   cd glm4-vllm
   uv init
   ```

3. **Install dependencies**:
   ```bash
   uv add vllm transformers torch huggingface-hub fastapi uvicorn python-dotenv
   ```

## Configuration

1. **Create environment file**:
   ```bash
   cp .env.example .env
   ```

2. **Set authentication token** in `.env`:
   ```env
   GLM_AUTH_TOKEN=your_secure_random_token_here
   ```

   Generate a secure token:
   ```bash
   python -c "import secrets; print('GLM_AUTH_TOKEN=' + secrets.token_urlsafe(64))"
   ```

## Usage

### Start Server (Foreground)
```bash
export PATH="$HOME/.local/bin:$PATH"
uv run python main.py
```

### Start Server (Background with Logging)
```bash
cd /path/to/glm4-vllm && \
export PATH="$HOME/.local/bin:$PATH" && \
nohup uv run python main.py > glm4_server.log 2>&1 &
echo $! > glm4_server.pid
```

### Example API Requests

**Health Check**:
```bash
curl -X GET "http://localhost:8000/health"
```

**Text Generation**:
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Authorization: Bearer your_auth_token_here" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum computing in simple terms.",
    "max_tokens": 200,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50
  }'
```

### Stop Server
```bash
# If running in background
kill $(cat glm4_server.pid) && rm glm4_server.pid

# Or find and kill process
ps aux | grep "python main.py" | grep -v grep | awk '{print $2}' | xargs kill
```

## API Endpoints

### GET `/health`
Health check endpoint (no authentication required).

**Response**:
```json
{
  "status": "healthy",
  "model": "GLM-4.6"
}
```

### POST `/generate`
Generate text using GLM-4.6 (authentication required).

**Request Headers**:
- `Authorization: Bearer <token>`
- `Content-Type: application/json`

**Request Body**:
```json
{
  "prompt": "string",
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "stop": ["optional", "stop", "sequences"]
}
```

**Response**:
```json
{
  "text": "Generated text response",
  "prompt": "Original prompt",
  "finish_reason": "length|stop"
}
```

## Management Commands

### View Real-time Logs
```bash
tail -f glm4_server.log
```

### Check Server Status
```bash
curl -s http://localhost:8000/health
```

### Check GPU Usage
```bash
nvidia-smi
```

### View Process Status
```bash
ps aux | grep "python main.py" | grep -v grep
```

### View Last 50 Log Lines
```bash
tail -50 glm4_server.log
```

## Performance

- **Model Size**: 357B parameters (Mixture of Experts)
- **Inference Speed**: ~4+ tokens/second
- **Memory Usage**: 82.5GB per GPU
- **Max Sequence Length**: 202,752 tokens
- **Concurrent Requests**: Up to 8x batch processing

## Security

- Authentication tokens are stored in `.env` file (excluded from git)
- Use strong, randomly generated tokens for production
- Consider implementing rate limiting for production deployments
- Run behind a reverse proxy (nginx/Apache) for additional security

## Troubleshooting

**GPU Memory Issues**:
```bash
# Clear GPU memory
nvidia-smi --gpu-reset

# Check for zombie processes
ps aux | grep -i vllm
```

**Port Already in Use**:
```bash
# Find process using port 8000
lsof -i :8000
kill <PID>
```

**Model Loading Errors**:
- Ensure sufficient GPU memory (660GB+ total)
- Check CUDA compatibility
- Verify model cache in `./models/` directory