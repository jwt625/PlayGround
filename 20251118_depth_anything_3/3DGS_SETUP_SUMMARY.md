# 3D Gaussian Splatting Setup and Execution Summary

## Overview

This document describes the setup and execution of 3D Gaussian Splatting (3DGS) training using the original graphdeco-inria implementation on a system with NVIDIA H100 GPUs.

## Repository Setup

### Clone Location
The gaussian-splatting repository was cloned to:
```
/home/ubuntu/GitHub/PlayGround/gaussian-splatting
```

This is a sibling directory to the working directory:
```
/home/ubuntu/GitHub/PlayGround/20251118_depth_anything_3
```

### Clone Command
```bash
cd /home/ubuntu/GitHub/PlayGround
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
```

The `--recursive` flag was essential to clone all submodules, including:
- `submodules/diff-gaussian-rasterization` (CUDA rasterization)
- `submodules/simple-knn` (KNN operations)
- `submodules/fused-ssim` (SSIM loss)

## Environment Setup

### Python Environment
The existing virtual environment was reused:
```
/home/ubuntu/GitHub/PlayGround/20251118_depth_anything_3/.venv
```

Python version: 3.10.12

### Package Installation

#### Core Dependencies
```bash
cd /home/ubuntu/GitHub/PlayGround/gaussian-splatting
source ../20251118_depth_anything_3/.venv/bin/activate
uv pip install torch torchvision torchaudio plyfile tqdm opencv-python joblib
```

Installed versions:
- PyTorch: 2.9.0 with CUDA 12.8 support
- torchvision, torchaudio (compatible versions)
- plyfile, tqdm, opencv-python, joblib

#### CUDA Extensions
All CUDA extensions were built from source using the `--no-build-isolation` flag to ensure compatibility with PyTorch 2.9.0:

```bash
# diff-gaussian-rasterization
cd submodules/diff-gaussian-rasterization
uv pip install --no-build-isolation -e .

# simple-knn
cd ../simple-knn
uv pip install --no-build-isolation -e .

# fused-ssim
cd ../fused-ssim
uv pip install --no-build-isolation .
```

The `--no-build-isolation` flag was critical because the setup.py files required torch to be available during the build process.

### Build Configuration
- CUDA version: 12.8
- GPU compute capability: 9.0 (H100)
- Compiler: nvcc with PyTorch 2.9.0 headers

## Code Modifications

### 1. Missing __init__.py for simple-knn
Created `/home/ubuntu/GitHub/PlayGround/gaussian-splatting/submodules/simple-knn/simple_knn/__init__.py`:
```python
from ._C import *
```

### 2. Empty Point Cloud Handling
Modified `gaussian-splatting/scene/dataset_readers.py` to handle empty COLMAP point clouds by generating random initialization points:

```python
try:
    pcd = fetchPly(ply_path)
except:
    pcd = None

# If point cloud is empty or None, generate random points
if pcd is None or len(pcd.points) == 0:
    num_pts = 100_000
    print(f"Generating random point cloud ({num_pts})...")
    
    xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
    shs = np.random.random((num_pts, 3)) / 255.0
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
```

This was necessary because the COLMAP output had an empty `points3D.bin` file (only 8 bytes).

### 3. PyTorch Checkpoint Loading
Modified `gaussian-splatting/train.py` line 54 to handle PyTorch 2.9.0's new default behavior:
```python
(model_params, first_iter) = torch.load(checkpoint, weights_only=False)
```

## Training Execution

### Input Data
COLMAP output directory:
```
/home/ubuntu/GitHub/PlayGround/20251118_depth_anything_3/colmap_output
```

Contents:
- `sparse/0/cameras.bin` - Camera intrinsics
- `sparse/0/images.bin` - Camera poses (154 images)
- `sparse/0/points3D.bin` - Empty sparse point cloud
- `images/` - Symlink to input images

### Training Command
```bash
cd /home/ubuntu/GitHub/PlayGround/gaussian-splatting
source ../20251118_depth_anything_3/.venv/bin/activate

LD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH \
python train.py \
  -s ../20251118_depth_anything_3/colmap_output \
  --iterations 30000 \
  --save_iterations 5000 10000 15000 20000 25000 30000 \
  --test_iterations 5000 10000 15000 20000 25000 30000 \
  2>&1 | tee training_30k.log
```

### Environment Variables
The `LD_LIBRARY_PATH` setting was required to ensure the CUDA extensions could find PyTorch's shared libraries (libc10.so, etc.) at runtime.

## Training Configuration

### Default Parameters
- Total iterations: 30,000
- SH degree: 3 (16 coefficients per color channel)
- Densification: iterations 500-15,000, every 100 iterations
- Opacity reset: every 3,000 iterations
- Initial points: 100,000 (randomly generated)

### GPU Utilization
- GPU: NVIDIA H100 (GPU 0)
- Utilization: 98%
- Memory usage: ~6.7 GB
- Training speed: ~30-40 iterations/second

## Output

### Directory Structure
```
/home/ubuntu/GitHub/PlayGround/gaussian-splatting/output/1ca35990-0/
├── point_cloud/
│   ├── iteration_5000/point_cloud.ply (34 MB, 143,578 Gaussians)
│   ├── iteration_10000/point_cloud.ply (61 MB)
│   ├── iteration_15000/point_cloud.ply (93 MB)
│   ├── iteration_20000/point_cloud.ply (93 MB)
│   ├── iteration_25000/point_cloud.ply (93 MB)
│   └── iteration_30000/point_cloud.ply (93 MB, ~300k Gaussians)
└── training_30k.log
```

### Training Metrics
| Iteration | PSNR (dB) | L1 Loss |
|-----------|-----------|---------|
| 5,000     | 15.76     | 0.1128  |
| 10,000    | 16.55     | 0.0990  |
| 15,000    | 17.25     | 0.0882  |
| 20,000    | 17.73     | 0.0822  |
| 25,000    | 17.98     | 0.0791  |
| 30,000    | 18.19     | 0.0769  |

Total training time: Approximately 15 minutes

## Key Issues Resolved

1. **PyTorch Version Compatibility**: Original code was designed for PyTorch 1.12.1, but system had PyTorch 2.9.0. Resolved by rebuilding all CUDA extensions from source.

2. **Build Isolation**: Standard pip install failed because setup.py needed torch at build time. Resolved with `--no-build-isolation` flag.

3. **Missing Module Init**: simple-knn package was missing `__init__.py`. Created manually.

4. **Library Path**: CUDA extensions couldn't find PyTorch libraries at runtime. Resolved by setting `LD_LIBRARY_PATH`.

5. **Empty Point Cloud**: COLMAP output had no 3D points. Resolved by implementing random point cloud initialization.

6. **Checkpoint Loading**: PyTorch 2.9.0 changed default `weights_only` parameter. Resolved by explicitly setting `weights_only=False`.

