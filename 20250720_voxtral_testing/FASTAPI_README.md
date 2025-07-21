# Voxtral FastAPI Server

A production-ready FastAPI server that provides audio transcription and understanding capabilities using Mistral AI's Voxtral Mini 3B model.

## Features

- **Audio Transcription**: Convert speech to text with high accuracy
- **Audio Understanding**: Ask questions about audio content  
- **Batch Processing**: Process multiple audio files efficiently
- **Streaming Support**: Real-time transcription via WebSocket
- **Multiple Formats**: Support for MP3, WAV, FLAC, M4A, OGG
- **Multilingual**: Support for 8 languages (EN, ES, FR, PT, HI, DE, NL, IT)
- **Rate Limiting**: Built-in rate limiting for production use
- **Auto Documentation**: Interactive OpenAPI/Swagger documentation

## Quick Start

### Prerequisites

1. **Voxtral Backend Running**: Ensure you have a Voxtral Mini 3B model running via vLLM on port 8000
2. **Dependencies Installed**: Install the required dependencies

```bash
# Install dependencies (if not already done)
pip install -e .
```

### Starting the Server

```bash
# Basic startup
python run_server.py

# With custom configuration
python run_server.py --host 0.0.0.0 --port 8080 --voxtral-host localhost --voxtral-port 8000

# Development mode with auto-reload
python run_server.py --reload --debug

# Production mode with multiple workers
python run_server.py --workers 4
```

### Server Options

```bash
python run_server.py --help
```

- `--host`: Host to bind the server to (default: 0.0.0.0)
- `--port`: Port to bind the server to (default: 8080)
- `--voxtral-host`: Voxtral backend host (default: localhost)
- `--voxtral-port`: Voxtral backend port (default: 8000)
- `--reload`: Enable auto-reload for development
- `--debug`: Enable debug mode
- `--workers`: Number of worker processes (default: 1)

## API Endpoints

### Health Check

```bash
GET /health/
```

Check server and model availability.

### Audio Transcription

```bash
POST /transcribe/
```

Transcribe a single audio file to text.

**Request Body:**
```json
{
  "audio_file": "base64_encoded_audio_data",
  "format": "mp3",
  "language": "en",
  "temperature": 0.0
}
```

**Response:**
```json
{
  "transcription": "The transcribed text content",
  "language_detected": "en",
  "confidence": 0.95,
  "processing_time_ms": 1500,
  "audio_duration_seconds": 30.5
}
```

### Batch Transcription

```bash
POST /transcribe/batch
```

Transcribe multiple audio files in a single request.

**Request Body:**
```json
{
  "audio_files": [
    {
      "data": "base64_encoded_audio_1",
      "format": "mp3",
      "id": "file1"
    },
    {
      "data": "base64_encoded_audio_2", 
      "format": "wav",
      "id": "file2"
    }
  ],
  "language": "en",
  "temperature": 0.0
}
```

### Audio Understanding

```bash
POST /understand/
```

Answer questions about audio content.

**Request Body:**
```json
{
  "audio_files": [
    {
      "data": "base64_encoded_audio_data",
      "format": "mp3"
    }
  ],
  "question": "What is this audio about?",
  "temperature": 0.2,
  "max_tokens": 500,
  "top_p": 0.95
}
```

### Streaming Transcription

```bash
WebSocket /stream/transcribe
```

Real-time audio transcription via WebSocket connection.

**Message Format:**
```json
{
  "format": "mp3",
  "language": "en", 
  "chunk_id": 1,
  "is_final": false,
  "audio_data": "base64_encoded_audio_chunk"
}
```

## Testing the Server

### Automated Testing

```bash
# Test all endpoints
python test_fastapi_server.py

# Test against custom server URL
python test_fastapi_server.py --url http://localhost:8080
```

### Manual Testing with curl

```bash
# Health check
curl http://localhost:8080/health/

# Transcription (requires base64 encoded audio)
curl -X POST http://localhost:8080/transcribe/ \
  -H "Content-Type: application/json" \
  -d '{
    "audio_file": "base64_encoded_audio_data",
    "format": "mp3",
    "language": "en",
    "temperature": 0.0
  }'
```

### Interactive Documentation

Visit `http://localhost:8080/docs` for interactive Swagger UI documentation where you can test all endpoints directly in your browser.

## Configuration

### Environment Variables

```bash
export HOST="0.0.0.0"
export PORT="8080"
export DEBUG="false"
export VOXTRAL_HOST="localhost"
export VOXTRAL_PORT="8000"
export RATE_LIMIT_REQUESTS="100"
export MAX_FILE_SIZE_MB="100"
export MAX_AUDIO_DURATION_SECONDS="1800"
```

### Rate Limits

- **General**: 100 requests/minute
- **Transcription**: 50 requests/minute
- **Understanding**: 20 requests/minute  
- **Batch**: 10 requests/minute

### Audio Constraints

- **Max file size**: 100MB per audio file
- **Max duration**: 30 minutes (transcription), 40 minutes (understanding)
- **Supported formats**: MP3, WAV, FLAC, M4A, OGG
- **Sample rates**: 8kHz to 48kHz
- **Channels**: Mono or stereo

## Architecture

```
[Client] --HTTP--> [FastAPI Server] ---> [Voxtral Backend] ---> [Voxtral Mini 3B]
                        ^                      ^
                   REST API endpoints    OpenAI-compatible
                   Custom validation     Internal calls
```

### Components

- **`app/main.py`**: Main FastAPI application
- **`app/routers/`**: API route handlers
- **`app/models/`**: Pydantic request/response models
- **`app/services/`**: Business logic and Voxtral integration
- **`app/core/`**: Configuration, logging, and exception handling

## Production Deployment

### Docker (Recommended)

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .
RUN pip install -e .

EXPOSE 8080
CMD ["python", "run_server.py", "--workers", "4"]
```

### systemd Service

```ini
[Unit]
Description=Voxtral FastAPI Server
After=network.target

[Service]
Type=exec
User=voxtral
WorkingDirectory=/opt/voxtral
ExecStart=/opt/voxtral/venv/bin/python run_server.py --workers 4
Restart=always

[Install]
WantedBy=multi-user.target
```

### Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
    
    location /stream/ {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

## Monitoring and Logging

The server includes structured logging and can be monitored using:

- **Health endpoint**: `/health/` for uptime monitoring
- **Metrics**: Processing times included in all responses
- **Logs**: Structured JSON logs for analysis
- **Rate limiting**: Built-in protection against abuse

## Troubleshooting

### Common Issues

1. **"Model unavailable"**: Ensure Voxtral backend is running on the configured port
2. **"Audio processing error"**: Check audio format and file size limits
3. **Rate limit exceeded**: Reduce request frequency or increase limits
4. **Timeout errors**: Increase timeout settings for large audio files

### Debug Mode

```bash
python run_server.py --debug --reload
```

This enables:
- Detailed error messages
- Auto-reload on code changes
- Debug-level logging
- Interactive error pages

## Performance

Based on testing with Voxtral Mini 3B:

- **Transcription**: ~27x real-time performance
- **Understanding**: ~15x real-time performance  
- **Concurrent requests**: Supports multiple simultaneous requests
- **Memory usage**: ~9GB GPU RAM for the model

## License

MIT License - see LICENSE file for details.
