# Models

Local OpenAI-compatible server for self-hosted vision-language models.

## Overview

This module launches a vLLM-powered HTTP server that serves Qwen2-VL models with an OpenAI-compatible API. Useful for running VIGA with local models instead of cloud APIs.

**Recommended**: Qwen2-VL-72B-Instruct with INT8 quantization provides excellent visual reasoning while fitting on 2x80GB GPUs.

## Files

| File | Description |
|------|-------------|
| `server.py` | vLLM server launcher |
| `client_chat.py` | Example chat client |
| `client_vision.py` | Example vision client |
| `requirements.txt` | Python dependencies |

## Prerequisites

- Linux with NVIDIA GPU (2x80GB for 72B model, 1x24GB+ for 7B)
- CUDA/cuDNN compatible with PyTorch
- Python 3.10+

## Setup

```bash
# Use the vllm conda environment
conda activate vllm

# Or install dependencies manually
pip install --upgrade pip
pip install -r models/requirements.txt
pip install "bitsandbytes>=0.46.1"  # Required for INT8 quantization
```

If you encounter CUDA/Torch issues:

```bash
pip install --upgrade --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
pip install --upgrade vllm
```

## Usage

### Start Server (72B INT8 - Recommended)

```bash
python models/server.py
```

This uses the default settings optimized for 72B INT8:
- Model: `Qwen/Qwen2-VL-72B-Instruct`
- Quantization: `bitsandbytes` (INT8)
- Tensor parallel: 2 GPUs
- Max context: 8192 tokens

### Start Server (7B Full Precision)

For single-GPU or testing:

```bash
python models/server.py \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --served-model-name Qwen2-VL-7B-Instruct \
  --tensor-parallel-size 1 \
  --quantization none \
  --max-model-len 32768
```

Server exposes OpenAI-compatible endpoints at `http://<host>:<port>/v1`.

### Test Chat

```bash
python models/client_chat.py --prompt "Describe the Eiffel Tower"
```

### Test Vision

```bash
python models/client_vision.py \
  --image-url "https://example.com/image.jpg" \
  --prompt "What is in this image?"
```

## Using with VIGA

Set the model in your VIGA run command:

```bash
python main.py --mode static_scene \
  --model Qwen2-VL-72B-Instruct \
  --api-base-url http://localhost:8000/v1 \
  ...
```

Or set environment variables:

```bash
export QWEN_BASE_URL="http://localhost:8000/v1"
```

## Notes

- **Multi-GPU**: Use `--tensor-parallel-size 2` for 72B model on 2 GPUs
- **Memory**: 72B INT8 uses ~72GB VRAM total, leaving headroom for KV cache
- **Disk space**: First run downloads model weights to HuggingFace cache (~140GB for 72B)
- **Tool calling**: Enabled by default with `--enable-auto-tool-choice --tool-call-parser hermes`
- **Quantization options**: `bitsandbytes` (INT8), `awq`, `fp8`, `gptq`, or `none`
