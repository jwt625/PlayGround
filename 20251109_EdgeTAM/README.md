# EdgeTAM Setup and Testing

## Overview

This project sets up and tests the EdgeTAM (Edge Track Anything Model) from Hugging Face on Apple Silicon hardware. EdgeTAM is a lightweight variant of SAM 2 optimized for on-device execution with only 13.9M parameters.

## Setup

### Environment

- **Hardware**: Mac mini with Apple M4 Pro (12 cores, 24 GB RAM)
- **OS**: macOS 15.5
- **Python**: 3.9.6+
- **Package Manager**: uv

### Dependencies

```toml
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.40.0
pillow>=10.0.0
requests>=2.31.0
numpy>=1.24.0
timm>=0.9.0
```

Install with:
```bash
uv sync
```

## Model Download Challenge

### Issue

Direct download from Hugging Face failed with 401 Unauthorized errors despite the model being publicly available. The error persisted even when explicitly disabling authentication tokens:

```
401 Client Error: Unauthorized for url: https://huggingface.co/yonigozlan/EdgeTAM-hf/...
Invalid credentials in Authorization header
```

This issue affected both the main EdgeTAM model and its dependency (timm/repvit_m1.dist_in1k).

### Root Cause

The local environment had cached invalid credentials that were automatically included in all Hugging Face API requests, preventing anonymous access to public models.

### Solution

Models were downloaded using a remote host with clean credentials, then transferred to the local machine:

1. SSH to remote host with clean environment
2. Download models using `huggingface_hub.snapshot_download()`
3. Create tarballs of cached models
4. Transfer via SCP to local machine
5. Extract to local Hugging Face cache directory (`~/.cache/huggingface/hub/`)

Backup tarballs are stored in `model_backups/`:
- `edgetam_model.tar.gz` (50 MB)
- `repvit_model.tar.gz` (39 MB)

## Test Results

### Configuration

- **Device**: MPS (Metal Performance Shaders - Apple Silicon GPU)
- **PyTorch Version**: 2.9.0
- **Model Parameters**: 9.12M
- **Model Load Time**: 1.05 seconds
- **Test Image**: 1800x1200 pixels

### Performance Benchmarks

| Test Case | Input | Inference Time | IoU Scores |
|-----------|-------|----------------|------------|
| Single point segmentation | Point: [500, 375] | 8491 ms (first run) | 0.046, 0.486, 0.769 |
| Multiple points refinement | Points: [500, 375], [1125, 625] | 737 ms | 0.838, 0.690, 0.210 |
| Bounding box segmentation | Box: [75, 275, 1725, 850] | 42 ms | 0.947, 0.978, 0.945 |
| Multiple objects | 2 objects | 1679 ms | N/A |

### Observations

- First inference includes model warmup overhead (8.5 seconds)
- Subsequent inferences are significantly faster (42-1679 ms)
- Bounding box segmentation shows best performance (42 ms)
- Model generates 3 masks per inference with varying IoU scores
- MPS acceleration working correctly on Apple Silicon

## Usage

### Basic Segmentation

```python
from transformers import EdgeTamModel, Sam2Processor
from PIL import Image

# Load model (local files only)
model = EdgeTamModel.from_pretrained(
    "yonigozlan/EdgeTAM-hf", 
    local_files_only=True
).to("mps")

processor = Sam2Processor.from_pretrained(
    "yonigozlan/EdgeTAM-hf",
    local_files_only=True
)

# Load and process image
image = Image.open("test_images/truck.jpg").convert("RGB")
inputs = processor(image, input_points=[[[500, 375]]], return_tensors="pt").to("mps")

# Run inference
outputs = model(**inputs)
masks = processor.post_process_masks(
    outputs.pred_masks,
    original_sizes=inputs.original_sizes,
    reshaped_input_sizes=inputs.reshaped_input_sizes
)
```

### Running Tests

```bash
uv run python test_edgetam.py
```

## Files

- `pyproject.toml` - Project configuration and dependencies
- `download_model.py` - Model download script
- `test_edgetam.py` - Comprehensive test suite with 4 test cases
- `model_backups/` - Model tarball backups for recovery
- `test_images/` - Test images for segmentation

## Notes

- All model loading uses `local_files_only=True` to prevent network requests
- Models are cached in `~/.cache/huggingface/hub/`
- Test script configured for MPS device but falls back to CPU if unavailable
- Warning about model type mismatch (edgetam_video vs edgetam) can be safely ignored
