# Voxtral Mini 3B Testing Environment

A comprehensive testing and development environment for Mistral AI's Voxtral Mini 3B model, featuring type-safe Python code, proper error handling, and extensive testing infrastructure.

## Features

- 🎯 **Type-safe implementation** with mypy validation
- 🔧 **Comprehensive configuration management** with environment variable support
- 🎵 **Audio processing utilities** for multiple formats (MP3, WAV, FLAC, M4A, OGG)
- 📝 **Transcription service** with automatic language detection
- 🧠 **Audio understanding** for Q&A and analysis
- 🧪 **Extensive testing suite** with unit and integration tests
- 📚 **Example scripts** for common use cases
- 🛠️ **Server management tools** for vLLM server lifecycle

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment with uv
uv venv .venv
source .venv/bin/activate

# Install dependencies
uv pip install -U "vllm[audio]" --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
uv pip install mypy black ruff pytest pytest-asyncio types-requests
```

### 2. Start vLLM Server

```bash
# Start the server
python examples/server_manager.py start

# Check server status
python examples/server_manager.py status

# Test server
python examples/server_manager.py test
```

### 3. Run Examples

```bash
# Transcription example
python examples/transcription_example.py

# Audio understanding example
python examples/audio_understanding_example.py
```

## Project Structure

```
20250720_voxtral_testing/
├── src/voxtral/           # Main package
│   ├── __init__.py        # Package exports
│   ├── client.py          # Main client implementation
│   ├── config.py          # Configuration management
│   ├── types.py           # Type definitions
│   └── exceptions.py      # Custom exceptions
├── tests/                 # Test suite
│   ├── test_config.py     # Configuration tests
│   ├── test_types.py      # Type validation tests
│   └── test_integration.py # Integration tests
├── examples/              # Example scripts
│   ├── transcription_example.py
│   ├── audio_understanding_example.py
│   └── server_manager.py
├── docs/                  # Documentation
│   └── RFD-000-voxtral-setup.md
├── config/                # Configuration files
├── pyproject.toml         # Project configuration
└── README.md             # This file
```

## Configuration

### Environment Variables

```bash
export VOXTRAL_SERVER_HOST="localhost"
export VOXTRAL_SERVER_PORT="8000"
export VOXTRAL_API_KEY="EMPTY"
export VOXTRAL_MODEL_ID="mistralai/Voxtral-Mini-3B-2507"
export VOXTRAL_REQUEST_TIMEOUT="300.0"
export VOXTRAL_CONNECTION_TIMEOUT="30.0"
```

### Programmatic Configuration

```python
from voxtral import VoxtralConfig, VoxtralClient

# Custom configuration
config = VoxtralConfig(
    server_host="localhost",
    server_port=8000,
    request_timeout=300.0,
)

# Create client
client = VoxtralClient(config)
```

## Usage Examples

### Audio Transcription

```python
import asyncio
from voxtral import VoxtralClient
from voxtral.types import AudioInput, TranscriptionRequest

async def transcribe_audio():
    client = VoxtralClient()
    
    audio = AudioInput(
        path="https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3",
        language="en"
    )
    
    request = TranscriptionRequest(
        audio=audio,
        language="en",
        temperature=0.0
    )
    
    response = await client.transcribe(request)
    print(f"Transcription: {response.content}")

asyncio.run(transcribe_audio())
```

### Audio Understanding

```python
import asyncio
from voxtral import VoxtralClient
from voxtral.types import AudioInput, AudioUnderstandingRequest

async def understand_audio():
    client = VoxtralClient()
    
    audio_files = [
        AudioInput(path="https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3"),
        AudioInput(path="https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/bcn_weather.mp3"),
    ]
    
    request = AudioUnderstandingRequest(
        audio_files=audio_files,
        question="Compare these two audio files. What are the main differences?",
        temperature=0.2,
        max_tokens=500
    )
    
    response = await client.understand_audio(request)
    print(f"Analysis: {response.content}")

asyncio.run(understand_audio())
```

## Testing

### Run Unit Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_config.py -v

# Run with coverage
pytest tests/ --cov=src/voxtral
```

### Run Integration Tests

```bash
# Make sure server is running first
python examples/server_manager.py start

# Run integration tests
pytest tests/test_integration.py -v

# Run simple integration test
python tests/test_integration.py
```

### Code Quality

```bash
# Type checking
mypy src/

# Code formatting
black src/ tests/ examples/

# Linting
ruff check src/ tests/ examples/
```

## Model Requirements

- **GPU Memory**: ~9.5 GB (bf16/fp16)
- **Context Length**: 32k tokens
- **Audio Duration**: Up to 30-40 minutes
- **Supported Languages**: English, Spanish, French, Portuguese, Hindi, German, Dutch, Italian
- **Audio Formats**: MP3, WAV, FLAC, M4A, OGG

## Server Management

### Start Server

```bash
vllm serve mistralai/Voxtral-Mini-3B-2507 \
  --tokenizer_mode mistral \
  --config_format mistral \
  --load_format mistral \
  --port 8000 \
  --host localhost
```

### Using Server Manager

```bash
# Start server
python examples/server_manager.py start

# Stop server
python examples/server_manager.py stop

# Restart server
python examples/server_manager.py restart

# Check status
python examples/server_manager.py status

# Test server
python examples/server_manager.py test
```

## Error Handling

The package provides comprehensive error handling:

- `VoxtralError`: Base exception for all Voxtral-related errors
- `VoxtralServerError`: Server communication errors
- `VoxtralConfigError`: Configuration validation errors
- `VoxtralAudioError`: Audio processing errors
- `VoxtralTimeoutError`: Request timeout errors

## Contributing

1. Follow type hints and use mypy for validation
2. Format code with black
3. Lint with ruff
4. Write tests for new functionality
5. Update documentation

## License

MIT License - see LICENSE file for details.

## References

- [Voxtral Mini 3B Model](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507)
- [vLLM Documentation](https://github.com/vllm-project/vllm)
- [Mistral Common Library](https://github.com/mistralai/mistral-common)
- [Mistral AI Blog](https://mistral.ai/news/voxtral)
