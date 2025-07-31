# RFD-000: Voxtral Mini 3B Setup and Testing

## Metadata
- **Title**: Voxtral Mini 3B Setup and Testing
- **Author**: Augment Agent
- **Date**: 2025-07-21 03:00:02 UTC
- **Status**: In Progress
- **Type**: Implementation
- **Related**: [Voxtral Mini 3B Model](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507)

## Abstract

This RFD documents the complete setup and testing process for Mistral AI's Voxtral Mini 3B model - a 3B parameter multimodal AI model that processes both text and audio inputs.

**Primary Goal**: Establish a production-ready development environment that can:
1. Run Voxtral Mini 3B via vLLM server locally
2. Test audio transcription capabilities (speech-to-text)
3. Test audio understanding capabilities (audio Q&A, analysis)
4. Provide type-safe Python client code for integration
5. Validate performance and accuracy with real audio samples

**Success Criteria**:
- vLLM server running Voxtral Mini 3B successfully
- Transcription working with <5% word error rate on clear English audio
- Audio understanding providing coherent responses to questions about audio content
- All code passing mypy type checking and automated tests
- Complete documentation for reproduction and extension

## Background

Voxtral Mini 3B is an enhancement of Ministral 3B that incorporates state-of-the-art audio input capabilities while retaining best-in-class text performance. Key features include:

- **Dedicated transcription mode**: Pure speech transcription with automatic language detection
- **Long-form context**: 32k token context length, handles up to 30 minutes for transcription, 40 minutes for understanding
- **Built-in Q&A and summarization**: Direct audio analysis without separate ASR models
- **Natively multilingual**: Support for 8 languages (English, Spanish, French, Portuguese, Hindi, German, Dutch, Italian)
- **Function-calling from voice**: Direct triggering of backend functions based on spoken intents
- **Text capabilities**: Retains full text understanding capabilities of Ministral-3B

## Requirements

### Hardware Requirements
- GPU with ~9.5 GB of GPU RAM (bf16 or fp16)
- CUDA-compatible GPU for optimal performance

### Software Requirements
- Python 3.10+
- vLLM with audio support (nightly build)
- mistral-common >= 1.8.1 with audio dependencies
- Development tools: mypy, black, ruff, pytest

## Implementation Plan

### Phase 1: Environment Setup ✅
- [x] Create virtual environment with uv
- [x] Install vLLM with audio support from nightly builds
- [x] Install mistral-common with audio dependencies
- [x] Install development dependencies (mypy, black, ruff, pytest)
- [x] Verify installation and versions

### Phase 2: Project Structure ✅
- [x] Create src/voxtral directory for main code
- [x] Create tests/ directory for test files
- [x] Create docs/ directory for documentation
- [x] Create examples/ directory for usage examples
- [x] Create config/ directory for configuration files

### Phase 3: FastAPI Server Implementation 🔄
- [ ] Create FastAPI application structure
- [ ] Implement Pydantic models for requests/responses
- [ ] Build transcription endpoint (`POST /transcribe`)
- [ ] Build audio understanding endpoint (`POST /understand`)
- [ ] Build batch transcription endpoint (`POST /transcribe/batch`)
- [ ] Add health check endpoint (`GET /health`)
- [ ] Implement error handling and validation
- [ ] Add rate limiting and middleware
- [ ] Set up structured logging

### Phase 4: Testing and Validation 🔄
- [ ] Create unit tests for all endpoints
- [ ] Implement integration tests with real audio files
- [ ] Test error handling and edge cases
- [ ] Load testing for concurrent requests
- [ ] Validate OpenAPI documentation generation
- [ ] Test rate limiting functionality
- [ ] Validate type safety with mypy
- [ ] Ensure code quality with black and ruff

### Phase 5: Documentation and Deployment 🔄
- [ ] Generate comprehensive OpenAPI documentation
- [ ] Create client usage examples (curl, Python, JavaScript)
- [ ] Add performance benchmarking results
- [ ] Create deployment guide (Docker, systemd)
- [ ] Document rate limits and constraints
- [ ] Create troubleshooting guide

## Technical Architecture

### FastAPI Server Design

**Architecture**: FastAPI web server that wraps vLLM backend
```
[Client] --HTTP--> [FastAPI Server] ---> [vLLM Backend] ---> [Voxtral Mini 3B]
                        ^                      ^
                   REST API endpoints    OpenAI-compatible
                   Custom validation     Internal calls
```

**Why FastAPI?**
- Automatic OpenAPI documentation generation
- Built-in request/response validation with Pydantic
- Async support for handling multiple concurrent requests
- Easy integration with rate limiting and middleware
- Type hints throughout for better development experience

### API Endpoints

#### 1. Health Check
- **Endpoint**: `GET /health`
- **Purpose**: Check server and model availability
- **Response**:
  ```json
  {
    "status": "healthy",
    "model": "voxtral-mini-3b",
    "timestamp": "2025-07-21T03:00:02Z",
    "version": "0.1.0"
  }
  ```

#### 2. Audio Transcription
- **Endpoint**: `POST /transcribe`
- **Purpose**: Convert audio to text
- **Request Body**:
  ```json
  {
    "audio_file": "base64_encoded_audio_data",
    "language": "en",
    "temperature": 0.0,
    "format": "mp3"
  }
  ```
- **Response**:
  ```json
  {
    "transcription": "The transcribed text content",
    "language_detected": "en",
    "confidence": 0.95,
    "processing_time_ms": 1500,
    "audio_duration_seconds": 30.5
  }
  ```

#### 3. Audio Understanding
- **Endpoint**: `POST /understand`
- **Purpose**: Answer questions about audio content
- **Request Body**:
  ```json
  {
    "audio_files": [
      {
        "data": "base64_encoded_audio_1",
        "format": "mp3"
      },
      {
        "data": "base64_encoded_audio_2",
        "format": "wav"
      }
    ],
    "question": "What are the main differences between these audio files?",
    "temperature": 0.2,
    "max_tokens": 500,
    "top_p": 0.95
  }
  ```
- **Response**:
  ```json
  {
    "answer": "The first audio contains a political speech while the second...",
    "audio_count": 2,
    "processing_time_ms": 3200,
    "token_usage": {
      "prompt_tokens": 150,
      "completion_tokens": 85,
      "total_tokens": 235
    }
  }
  ```

#### 4. Batch Transcription
- **Endpoint**: `POST /transcribe/batch`
- **Purpose**: Transcribe multiple audio files efficiently
- **Request Body**:
  ```json
  {
    "audio_files": [
      {"data": "base64_1", "format": "mp3", "id": "file1"},
      {"data": "base64_2", "format": "wav", "id": "file2"}
    ],
    "language": "en",
    "temperature": 0.0
  }
  ```
- **Response**:
  ```json
  {
    "results": [
      {
        "id": "file1",
        "transcription": "First audio transcription",
        "success": true
      },
      {
        "id": "file2",
        "error": "Audio format not supported",
        "success": false
      }
    ],
    "total_processing_time_ms": 2800
  }
  ```

### Error Handling

#### Standard Error Response Format
```json
{
  "error": {
    "code": "AUDIO_PROCESSING_ERROR",
    "message": "Failed to process audio file",
    "details": "Unsupported audio format: .xyz",
    "timestamp": "2025-07-21T03:00:02Z"
  }
}
```

#### Error Codes
- `INVALID_AUDIO_FORMAT`: Unsupported audio format
- `AUDIO_TOO_LONG`: Audio exceeds maximum duration (30 minutes)
- `MODEL_UNAVAILABLE`: Voxtral model not loaded
- `PROCESSING_TIMEOUT`: Request exceeded timeout limit
- `INVALID_REQUEST`: Malformed request body
- `SERVER_ERROR`: Internal server error

### Request Validation

#### Audio File Constraints
- **Max file size**: 100MB per audio file
- **Max duration**: 30 minutes for transcription, 40 minutes for understanding
- **Supported formats**: MP3, WAV, FLAC, M4A, OGG
- **Sample rates**: 8kHz to 48kHz
- **Channels**: Mono or stereo

#### Rate Limiting
- **Per IP**: 100 requests per minute
- **Per endpoint**:
  - `/transcribe`: 50 requests per minute
  - `/understand`: 20 requests per minute (more compute intensive)
  - `/transcribe/batch`: 10 requests per minute

### FastAPI Components

#### 1. Main Application (`app/main.py`)
- FastAPI app initialization
- Middleware setup (CORS, rate limiting, logging)
- Router registration
- Startup/shutdown events

#### 2. API Routes (`app/routers/`)
- `transcription.py`: Transcription endpoints
- `understanding.py`: Audio understanding endpoints
- `health.py`: Health check endpoints

#### 3. Models (`app/models/`)
- Pydantic request/response models
- Validation schemas
- Error response models

#### 4. Services (`app/services/`)
- `voxtral_service.py`: Interface to vLLM backend
- `audio_processor.py`: Audio file validation and processing
- `rate_limiter.py`: Request rate limiting logic

#### 5. Core (`app/core/`)
- `config.py`: Application configuration
- `exceptions.py`: Custom exception classes
- `logging.py`: Structured logging setup

### Dependencies
- **FastAPI**: Web framework with automatic OpenAPI docs
- **vLLM**: Backend inference engine
- **mistral-common**: Audio processing utilities
- **Pydantic**: Request/response validation
- **uvicorn**: ASGI server for production
- **python-multipart**: File upload handling
- **slowapi**: Rate limiting middleware

## Progress Log

### 2025-07-21 03:00:02 UTC - Initial Setup
- ✅ Created virtual environment with uv
- ✅ Installed vLLM 0.10.0rc2.dev3+g7ba34b124 with audio support
- ✅ Installed mistral-common 1.8.1 with audio dependencies
- ✅ Installed development dependencies (mypy, black, ruff, pytest)
- ✅ Created project directory structure
- ✅ Created RFD documentation

### Next Steps
1. Implement type-safe client wrapper
2. Create audio processing utilities
3. Set up testing infrastructure
4. Implement example use cases

## Configuration

### Model Information
- **Model ID**: mistralai/Voxtral-Mini-3B-2507
- **Memory Requirements**: ~9.5 GB GPU RAM
- **Context Length**: 32k tokens
- **Supported Audio**: Up to 30-40 minutes

### Recommended Settings
- **Temperature**: 0.2 for chat completion, 0.0 for transcription
- **Top-p**: 0.95 for chat completion
- **Tokenizer Mode**: mistral
- **Config Format**: mistral
- **Load Format**: mistral

## References
- [Voxtral Mini 3B Model Card](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507)
- [vLLM Documentation](https://github.com/vllm-project/vllm)
- [Mistral Common Library](https://github.com/mistralai/mistral-common)
- [Mistral AI Blog Post](https://mistral.ai/news/voxtral)
