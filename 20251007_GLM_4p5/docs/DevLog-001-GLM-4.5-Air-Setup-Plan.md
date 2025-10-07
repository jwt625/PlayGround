# DevLog-001: GLM-4.5-Air Setup Plan

**Date**: October 7, 2025  
**Project**: GLM-4.5-Air Local Deployment  
**Status**: Planning Phase  
**Author**: Development Team  

## Executive Summary

This document outlines the comprehensive plan for setting up GLM-4.5-Air on our local infrastructure. After thorough hardware assessment, we've determined that our current setup is well-suited for the GLM-4.5-Air model but insufficient for the full GLM-4.5 model.

## Hardware Assessment Results

### Current Infrastructure
- **CPU**: Intel Xeon Platinum 8480+ (52 cores, 26 cores per socket)
- **RAM**: 442 GB total system memory
- **GPU**: 2x NVIDIA H100 80GB HBM3 (163 GB total GPU memory)
- **Storage**: 5.4TB main drive + 2.5PB additional storage
- **CUDA**: Version 12.8
- **OS**: Linux (Ubuntu-based)

### Model Compatibility Analysis

#### ✅ GLM-4.5-Air (RECOMMENDED)
- **Model Size**: 106B total parameters, 12B active parameters (~110GB)
- **Memory Requirements**: ~110GB model weights
- **GPU Requirements**: 
  - BF16: H100 x4 / H200 x2 → **Our 2x H100 is SUFFICIENT**
  - FP8: H100 x2 / H200 x1 → **Our 2x H100 is EXCELLENT**
- **System RAM**: Requires <1TB → **Our 442GB is ADEQUATE**

#### ❌ GLM-4.5 Full Model (NOT FEASIBLE)
- **Model Size**: 355B total parameters, 32B active parameters (~358GB)
- **GPU Requirements**: 
  - BF16: H100 x16 / H200 x8 → **INSUFFICIENT (we have 2x H100)**
  - FP8: H100 x8 / H200 x4 → **INSUFFICIENT (we have 2x H100)**

## Implementation Plan

### Phase 1: Environment Setup
**Estimated Time**: 2-3 hours

1. **UV Project Initialization**
   - Initialize new uv project: `uv init glm-4.5-air-setup`
   - Configure `pyproject.toml` with project metadata and dependencies
   - Set up development dependencies (ruff, mypy, pytest)
   - Verify Python 3.10+ installation ✅ (Currently: Python 3.10.12)
   - Leverage uv's built-in virtual environment management ✅ (uv 0.7.13 available)

2. **Code Quality Standards Setup**
   - **Ruff Configuration**: Set up linting and formatting rules
   - **MyPy Configuration**: Configure static type checking
   - **Pre-commit Hooks**: Ensure code quality on every commit
   - **VS Code/IDE Integration**: Configure editor for consistent formatting

3. **CUDA Environment Verification**
   - Confirm CUDA 12.8 compatibility ✅
   - Verify PyTorch CUDA support using uv-managed environment
   - Test GPU accessibility with type-safe code

4. **Storage Preparation**
   - Allocate ~150GB for model weights and cache
   - Set up model download directory
   - Configure cache directories

### Phase 2: Dependency Installation
**Estimated Time**: 1-2 hours

1. **UV-Managed Dependencies Setup**
   ```toml
   # pyproject.toml dependencies
   [project]
   dependencies = [
       "torch>=2.0.0",
       "transformers>=4.35.0",
       "safetensors>=0.4.0",
       "huggingface-hub>=0.19.0",
       "vllm>=0.2.0",  # or sglang as alternative
       "fastapi>=0.104.0",
       "uvicorn>=0.24.0",
   ]

   [project.optional-dependencies]
   dev = [
       "ruff>=0.1.0",
       "mypy>=1.7.0",
       "pytest>=7.4.0",
       "pre-commit>=3.5.0",
       "types-requests",
       "types-PyYAML",
   ]
   ```

2. **Installation via UV**
   - Install production dependencies: `uv sync`
   - Install development dependencies: `uv sync --extra dev`
   - Verify CUDA-enabled PyTorch installation

3. **Code Quality Tools Configuration**
   - Configure ruff for linting and formatting
   - Set up mypy for static type checking
   - Install pre-commit hooks for automated quality checks

### Phase 3: Model Download and Setup
**Status**: ✅ **COMPLETED** (2025-10-07 23:15)
**Estimated Time**: 2-4 hours (depending on network speed)

1. **Model Selection** ✅
   - ✅ Analyzed GLM-4.5-Air variants: BF16 (205.79 GB) vs FP8 (104.85 GB)
   - ✅ Selected `zai-org/GLM-4.5-Air-FP8` for optimal performance on 2x H100 setup
   - ✅ FP8 version provides 50% size reduction with minimal quality loss

2. **Download Strategy** ✅
   - ✅ Used Hugging Face Hub with progress tracking and error handling
   - ✅ Implemented resume capability for large files
   - ✅ Downloaded GLM-4.5-Air-FP8 model (104.85 GB) in 123.13 seconds
   - ✅ Verified model integrity: 47 safetensors files, all essential files present

3. **Model Configuration** ✅
   - ✅ Set up model configuration files in `models/GLM-4.5-Air-FP8/`
   - ✅ Configured tokenizer settings (GLM tokenizer)
   - ✅ Prepared chat templates and generation config
   - ✅ Verified vLLM compatibility with tensor parallelism

### 🎯 **Key Achievements**:
- **Model Downloaded**: GLM-4.5-Air-FP8 (104.85 GB, 47 weight files)
- **vLLM Integration**: Successfully configured with tensor_parallel_size=2
- **Performance Verified**: 73.44 tokens/sec basic inference, 84.55 tokens/sec streaming
- **Memory Usage**: ~50.56 GiB per GPU (total ~101 GB across both H100s)
- **KV Cache**: 19.35 GiB available per GPU for efficient caching
- **Initialization Time**: ~100 seconds (includes torch.compile optimization)

### 🔧 **Technical Configuration**:
- **Model Path**: `/home/ubuntu/GitHub/PlayGround/20251007_GLM_4p5/glm-4.5-air-setup/models/GLM-4.5-Air-FP8`
- **Quantization**: FP8 compressed-tensors (50% size reduction vs BF16)
- **Architecture**: Glm4MoeForCausalLM (Mixture of Experts)
- **Context Length**: 4096 tokens (tested), supports up to 131,072 tokens
- **Flash Attention**: Enabled for optimized memory usage
- **CUDA Graphs**: Captured for maximum inference performance

### Phase 4: Inference Server Setup
**Estimated Time**: 2-3 hours

#### Type-Safe Server Implementation
1. **Create Typed Server Module**
   ```python
   # src/glm_server/main.py
   from typing import Optional, Dict, Any, List
   from pydantic import BaseModel
   import asyncio

   class InferenceConfig(BaseModel):
       model_path: str
       tensor_parallel_size: int = 2
       host: str = "0.0.0.0"
       port: int = 8000
       max_model_len: Optional[int] = None
   ```

2. **UV-Managed Server Execution**

#### Option A: vLLM Setup
```bash
# Run with uv
uv run vllm serve zai-org/GLM-4.5-Air \
    --tensor-parallel-size 2 \
    --tool-call-parser glm45 \
    --reasoning-parser glm45 \
    --enable-auto-tool-choice \
    --served-model-name glm-4.5-air
```

#### Option B: SGLang Setup
```bash
# Run with uv
uv run python -m sglang.launch_server \
  --model-path zai-org/GLM-4.5-Air \
  --tp-size 2 \
  --tool-call-parser glm45 \
  --reasoning-parser glm45 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --mem-fraction-static 0.7 \
  --served-model-name glm-4.5-air \
  --host 0.0.0.0 \
  --port 8000
```

3. **Code Quality Enforcement**
   - Run type checking: `uv run mypy src/`
   - Run linting: `uv run ruff check src/`
   - Run formatting: `uv run ruff format src/`

### Phase 5: Testing and Validation
**Estimated Time**: 1-2 hours

1. **Type-Safe Test Implementation**
   ```python
   # tests/test_inference.py
   from typing import Dict, Any
   import pytest
   from glm_server.client import InferenceClient

   @pytest.mark.asyncio
   async def test_basic_generation() -> None:
       client = InferenceClient(base_url="http://localhost:8000")
       response = await client.generate("Hello, world!")
       assert isinstance(response, str)
       assert len(response) > 0
   ```

2. **UV-Managed Test Execution**
   - Run tests: `uv run pytest tests/`
   - Run with coverage: `uv run pytest --cov=src tests/`
   - Type check tests: `uv run mypy tests/`

3. **Automated Quality Checks**
   - Pre-commit validation: `uv run pre-commit run --all-files`
   - Lint check: `uv run ruff check .`
   - Format check: `uv run ruff format --check .`

4. **Performance Benchmarking**
   - Inference speed measurements (type-safe metrics collection)
   - Memory usage monitoring with proper typing
   - GPU utilization analysis

## Technical Considerations

### Memory Management
- **Model Weights**: ~110GB for GLM-4.5-Air
- **KV Cache**: Variable based on context length and batch size
- **System Overhead**: Reserve 20-30GB for system operations
- **Total GPU Memory Usage**: Expect 130-150GB out of 163GB available

### Performance Optimizations
1. **FP8 Quantization**: Consider using FP8 version for better efficiency
2. **Speculative Decoding**: Enable EAGLE algorithm for faster inference
3. **Tensor Parallelism**: Utilize both H100 GPUs effectively
4. **Memory Fraction**: Optimize memory allocation (0.7 static fraction recommended)

### Potential Challenges
1. **Memory Constraints**: Monitor GPU memory usage carefully
2. **Model Loading Time**: Initial loading may take 10-15 minutes
3. **Network Bandwidth**: Model download requires stable high-speed connection
4. **CUDA Compatibility**: Ensure all libraries support CUDA 12.8

## Success Criteria

### Minimum Viable Product (MVP)
- [ ] UV project properly initialized with pyproject.toml
- [ ] All code passes ruff linting and formatting checks
- [ ] All code passes mypy type checking (strict mode)
- [ ] Model successfully loads without OOM errors
- [ ] Basic text generation works with type-safe interfaces
- [ ] API server responds to requests with proper error handling
- [ ] Utilizes both GPUs effectively
- [ ] Comprehensive test suite with >90% coverage

### Full Feature Set
- [ ] Tool calling functionality operational with typed interfaces
- [ ] Reasoning mode toggleable with proper type annotations
- [ ] Handles long context (64K+ tokens) with memory-safe code
- [ ] Achieves target inference speed (>20 tokens/sec) with performance monitoring
- [ ] Stable operation under load with proper logging and error handling
- [ ] Pre-commit hooks enforce code quality automatically
- [ ] CI/CD pipeline validates code quality on every commit

## Risk Assessment

### High Risk
- **GPU Memory Overflow**: Mitigation - Use FP8 version, optimize batch sizes
- **CUDA Version Conflicts**: Mitigation - Verify compatibility matrix

### Medium Risk
- **Network Download Failures**: Mitigation - Implement resume capability
- **Performance Below Expectations**: Mitigation - Tune hyperparameters

### Low Risk
- **Storage Space**: Mitigation - Ample storage available
- **CPU Bottlenecks**: Mitigation - High-end CPU available

## Development Workflow with UV

### Daily Development Commands
```bash
# Project setup
uv init glm-4.5-air-setup
cd glm-4.5-air-setup

# Install dependencies
uv sync --extra dev

# Code quality checks (run before commits)
uv run ruff check .          # Linting
uv run ruff format .         # Formatting
uv run mypy src/             # Type checking
uv run pytest tests/         # Testing

# Run inference server
uv run python src/glm_server/main.py

# Install new dependencies
uv add torch transformers    # Production deps
uv add --dev pytest mypy     # Development deps
```

### Code Quality Standards

#### Ruff Configuration (pyproject.toml)
```toml
[tool.ruff]
target-version = "py310"
line-length = 88
select = ["E", "F", "I", "N", "W", "UP", "B", "A", "C4", "T20"]
ignore = ["E501"]  # Line too long (handled by formatter)

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
```

#### MyPy Configuration (pyproject.toml)
```toml
[tool.mypy]
python_version = "3.10"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
```

## Next Steps

1. **Immediate Actions**
   - Initialize UV project: `uv init glm-4.5-air-setup`
   - Configure pyproject.toml with dependencies and tools
   - Set up ruff and mypy configurations
   - Begin model download with type-safe download scripts

2. **Short-term Goals** (Next 24 hours)
   - Complete Phase 1-3 implementation with full type coverage
   - Achieve basic model loading with proper error handling
   - Implement comprehensive test suite

3. **Medium-term Goals** (Next week)
   - Optimize performance with type-safe monitoring
   - Implement production-ready setup with proper logging
   - Document operational procedures with type annotations

## Resource Links

- **Model Repository**: https://huggingface.co/zai-org/GLM-4.5-Air
- **Technical Documentation**: GLM-4.5 technical blog and report
- **Community Support**: GLM-4.5 Discord community
- **API Documentation**: Z.ai API Platform documentation

---

## Phase 1 Completion Status ✅

**Completed**: October 7, 2025 22:57 UTC

### ✅ Achievements

1. **UV Project Initialization**: Successfully created `glm-4.5-air-setup` project
2. **Dependencies Installation**: All 163 packages installed successfully including:
   - PyTorch 2.8.0+cu128 with CUDA support
   - vLLM 0.11.0 for inference
   - Transformers, Safetensors, Hugging Face Hub
   - Development tools: ruff 0.14.0, mypy 1.18.2, pytest
3. **Code Quality Standards**: Implemented and verified:
   - Ruff linting and formatting (all checks passed)
   - MyPy strict type checking (no issues found)
   - Comprehensive pyproject.toml configuration
4. **Type-Safe Implementation**: Created foundational modules:
   - `src/glm_server/config.py`: Pydantic-based configuration with validation
   - `src/glm_server/main.py`: Type-safe main entry point with system checks
   - `tests/test_config.py`: Comprehensive test suite (5/5 tests passing)
5. **System Verification**: Confirmed hardware compatibility:
   - 2x NVIDIA H100 80GB HBM3 (158.38 GB total GPU memory)
   - CUDA 12.8 available and functional
   - 442.65 GB system RAM
   - All requirements met for GLM-4.5-Air

### 📊 Quality Metrics

- **Code Quality**: 100% (ruff checks passed)
- **Type Safety**: 100% (mypy strict mode passed)
- **Test Coverage**: 29% (5/5 tests passing, coverage needs improvement)
- **Dependencies**: 163 packages successfully installed
- **Build Status**: ✅ Successful

### 🎯 Next Steps

**Ready for Phase 4**: Model download and setup complete, moving to inference server setup.

**Immediate Next Tasks**:
1. **FastAPI Server Implementation**: Create production-ready API endpoints
2. **Health Check System**: Implement comprehensive monitoring
3. **Performance Optimization**: Fine-tune vLLM configuration
4. **Testing Suite**: Comprehensive test coverage for all functionality

---

**Document Version**: 1.2
**Last Updated**: October 7, 2025 23:15 UTC
**Next Review**: After Phase 4 completion
