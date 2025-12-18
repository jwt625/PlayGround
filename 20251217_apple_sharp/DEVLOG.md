# Development Log - Apple ML-SHARP

## Project Overview
SHARP (Sharp Monocular View Synthesis) is Apple's research project for generating 3D Gaussian representations from single images in under one second. This log documents the setup and experimentation process.

**Repository**: https://github.com/apple/ml-sharp  
**Paper**: Sharp Monocular View Synthesis in Less Than a Second (arXiv:2512.10685)  
**Authors**: Lars Mescheder, Wei Dong, Shiwei Li, Xuyang Bai, Marcel Santos, Peiyun Hu, Bruno Lecouat, Mingmin Zhen, Amaël Delaunoy, Tian Fang, Yanghai Tsin, Stephan Richter, Vladlen Koltun

## 2025-12-18

### Initial Setup

**Environment Setup**
- Cloned repository from https://github.com/apple/ml-sharp
- Created virtual environment using `uv` with Python 3.13
- Location: `./20251217_apple_sharp/ml-sharp/.venv`
- Installation command: `uv venv --python 3.13`

**Dependencies Installed**
- PyTorch 2.8.0 (with CUDA support)
- gsplat 1.5.3 (Gaussian splatting renderer)
- torchvision 0.23.0
- timm 1.0.20
- Additional dependencies: numpy, scipy, matplotlib, pillow, imageio, etc.
- Total of 64 packages installed via `uv pip install -r requirements.txt`

**Model Checkpoints Downloaded**
All checkpoints cached at `~/.cache/torch/hub/checkpoints/`:
- sharp_2572gikvuh.pt: 2.7 GB (main SHARP model)
- dinov2_vitl14_reg4_pretrain.pth: 1.2 GB (DINOv2 ViT-L/14 encoder)
- resnet18-f37072fd.pth: 45 MB (ResNet18)
- Total storage: ~3.95 GB

**Installation Verification**
- CLI tool `sharp` successfully installed and accessible
- Commands available: `sharp predict`, `sharp render`
- Model architecture: Uses DINOv2 ViT-L/14 (384px) as encoder preset

### Test Runs

**Test 1: Dummy Image**
- Created synthetic 512x512 RGB image with random noise
- Input: `test_input/dummy.jpg`
- Output: `test_output/dummy.ply` (64 MB)
- Processing time: ~4 seconds
- Status: Successful
- Note: Model warned about missing focal length EXIF data, defaulted to 30mm

**Test 2: Teaser Image**
- Input: `data/teaser.jpg` (repository sample image)
- Output: `test_output_real/teaser.ply`
- Processing time: ~4 seconds
- Status: Successful
- Warning: Missing focal length in EXIF, defaulted to 30mm
- Additional warning: Received 1 reflection matrix from SVD, flipped to rotation

**Test 3: Real Image (IMG_3753.JPG)**
- Input: `test_input/IMG_3753.JPG`
- Output: `test_output_img3753/IMG_3753.ply` (64 MB)
- Processing time: ~4 seconds
- Status: Successful
- Warning: Missing focal length in EXIF, defaulted to 30mm

### Technical Notes

**Processing Pipeline**
1. Preprocessing: Image loading and preparation
2. Inference: Single feedforward pass through neural network
3. Postprocessing: Gaussian splat generation and optimization

**Output Format**
- 3D Gaussian Splat (.ply files)
- Compatible with standard 3DGS renderers
- Coordinate convention: OpenCV (x right, y down, z forward)
- Scene center: approximately at (0, 0, +z)

**Performance Observations**
- Inference time: <1 second on CUDA GPU
- Total processing time: ~4 seconds (including I/O and postprocessing)
- Output file size: Consistent at ~64 MB per image
- Device: CUDA GPU utilized for acceleration

**Known Limitations**
- Rendering trajectories (`--render` flag) requires CUDA GPU only
- CPU and MPS support available for Gaussian prediction only
- Focal length must be provided or defaults to 30mm if missing from EXIF
- First launch of gsplat renderer has initialization overhead

### Next Steps
- Test with images containing proper EXIF focal length data
- Experiment with rendering trajectories using `--render` flag
- Evaluate output quality with different 3DGS viewers
- Test performance with batch processing of multiple images
- Investigate impact of focal length accuracy on output quality

### References
- Project page: https://apple.github.io/ml-sharp/
- Paper: https://arxiv.org/abs/2512.10685
- Model download: https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt

