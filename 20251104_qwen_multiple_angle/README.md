# Qwen-Image-Edit-2509 Setup

Multi-image editing inference server using Qwen-Image-Edit-2509 model.

## System Specifications

- **CPU**: Intel Xeon Platinum 8480+ (52 cores)
- **RAM**: 442GB total
- **GPU**: 2x NVIDIA H100 80GB HBM3
- **Model Memory**: ~54GB GPU memory

## Quick Start

### 1. Test the Model

```bash
uv run python test_model.py
```

This will:
- Create test images
- Run single-image and multi-image editing tests
- Save outputs to `test_output/` directory

### 2. Start the Inference Server

```bash
uv run python server.py
```

Server will start on `http://0.0.0.0:8000`

### 3. Use the API

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Single Image Edit:**
```bash
curl -X POST http://localhost:8000/edit \
  -F "images=@input.png" \
  -F "prompt=A magical wizard bear with a purple hat" \
  -F "num_inference_steps=40" \
  -F "seed=42"
```

**Multi-Image Edit (2-3 images):**
```bash
curl -X POST http://localhost:8000/edit \
  -F "images=@input1.png" \
  -F "images=@input2.png" \
  -F "prompt=Two bears having a picnic in a park" \
  -F "num_inference_steps=40"
```

## API Parameters

- `images`: 1-3 input images (required)
- `prompt`: Text description of desired edit (required)
- `negative_prompt`: Negative prompt (default: " ")
- `num_inference_steps`: Number of steps (default: 40)
- `guidance_scale`: Guidance scale (default: 1.0)
- `true_cfg_scale`: True CFG scale (default: 4.0)
- `seed`: Random seed for reproducibility (optional)

## Files

- `server.py`: FastAPI inference server
- `test_model.py`: Test script with example usage
- `download_model.py`: Model download and caching script
- `pyproject.toml`: Project dependencies

## Model Information

- **Model**: Qwen/Qwen-Image-Edit-2509
- **Type**: Multi-image editing diffusion model
- **Input**: 1-3 images + text prompt
- **Output**: 1024x1024 edited image
- **Features**:
  - Multi-image composition
  - Person/product/text editing
  - ControlNet support
  - High consistency preservation

## Performance

- Single image inference: ~35 seconds (40 steps)
- Multi-image inference: ~63 seconds (40 steps)
- GPU utilization: ~54GB on primary GPU

## Environment Variables

- `PORT`: Server port (default: 8000)
- `HOST`: Server host (default: 0.0.0.0)
- `HF_HOME`: Hugging Face cache directory

