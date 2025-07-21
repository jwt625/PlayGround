# RFD-000: Voxtral Mini 3B Setup and Testing

## Metadata
- **Title**: Voxtral Mini 3B Setup and Testing
- **Author**: Augment Agent
- **Date**: 2025-07-21 03:00:02 UTC
- **Status**: In Progress
- **Type**: Implementation
- **Related**: [Voxtral Mini 3B Model](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507)

## Abstract

This RFD documents the setup and testing of Mistral AI's Voxtral Mini 3B model, a multimodal AI model that combines text and audio capabilities. The goal is to establish a working development environment with proper type safety, testing infrastructure, and example implementations for both audio transcription and audio understanding use cases.

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

### Phase 3: Core Implementation 🔄
- [ ] Implement type-safe client wrapper for vLLM server
- [ ] Create audio processing utilities
- [ ] Implement transcription service
- [ ] Implement audio understanding service
- [ ] Add proper error handling and logging

### Phase 4: Testing and Validation 🔄
- [ ] Create unit tests for all components
- [ ] Implement integration tests with sample audio files
- [ ] Test both transcription and audio understanding modes
- [ ] Validate type safety with mypy
- [ ] Ensure code quality with black and ruff

### Phase 5: Examples and Documentation 🔄
- [ ] Create example scripts for common use cases
- [ ] Document API usage and configuration
- [ ] Add performance benchmarking
- [ ] Create troubleshooting guide

## Technical Architecture

### Components
1. **VoxtralClient**: Main client class for interacting with vLLM server
2. **AudioProcessor**: Utilities for audio file handling and validation
3. **TranscriptionService**: Dedicated transcription functionality
4. **AudioUnderstandingService**: Audio Q&A and analysis functionality
5. **ConfigManager**: Configuration and settings management

### Dependencies
- **vLLM**: Core inference engine with audio support
- **mistral-common**: Audio processing and protocol handling
- **OpenAI**: Client library for API compatibility
- **huggingface-hub**: Model and sample audio file access

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
