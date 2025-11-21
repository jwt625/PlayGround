# DevLog: 3D Gaussian Splatting Training Progress

**Date**: 2025-11-21
**Status**: 🔄 IN PROGRESS - Iterating to improve quality
**Agent**: Augment Agent
**Task**: Run 3DGS training on COLMAP reconstruction from 20251119_SAM3

---

## Executive Summary

### Attempt 1: FAILED ❌ (COLMAP v6)
3DGS training completed (30,000 iterations in ~89 seconds) but produced an **invalid/corrupted model** with only 7,612 Gaussians containing mostly NaN values. Root cause: **poor quality COLMAP reconstruction** (7,495 points, 2.57px error).

### Attempt 2: IMPROVED ✅ (COLMAP v5)
3DGS training completed successfully (30,000 iterations in ~37 minutes) with **much better results**:
- **3.28 million Gaussians** (vs 7,612 in v6)
- **PSNR: 33.29 dB** (vs 7.43 dB in v6)
- **777 MB PLY file** with valid data (vs 1.9 MB corrupted file)

**Quality Assessment**: Still not great, but dramatically improved. Continuing with cropped images to reduce lens distortion.

### Next Steps
- ✅ Cropped images to center 70% (2998×3998) to reduce edge distortion
- 🔄 Will run COLMAP on cropped images
- 🔄 Will run 3DGS on improved COLMAP reconstruction

---

## Environment Setup

### Repository & Virtual Environment
- **3DGS Repository**: `/home/ubuntu/GitHub/PlayGround/gaussian-splatting`
- **Virtual Environment**: `/home/ubuntu/GitHub/PlayGround/20251118_depth_anything_3/.venv`
- **Python Version**: 3.10.12
- **PyTorch Version**: 2.9.0 with CUDA 12.8
- **GPU**: NVIDIA H100

### Environment Variables
```bash
LD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
```

### Pre-installed Dependencies
- `diff-gaussian-rasterization` (custom CUDA extension)
- `simple-knn` (custom CUDA extension with manual `__init__.py`)
- `fused-ssim` (CUDA extension)

---

## Attempt 1: Training with COLMAP v6 (FAILED ❌)

### Input Data

#### COLMAP Reconstruction v6
- **Location**: `/home/ubuntu/GitHub/PlayGround/20251119_SAM3/3dgs_data/sparse/0/`
- **Original Camera Model**: OPENCV (with distortion)
- **Undistorted Data**: `/home/ubuntu/GitHub/PlayGround/20251119_SAM3/3dgs_data_undistorted/`
- **Final Camera Model**: PINHOLE (undistorted)

#### COLMAP Statistics (v6 - Poor Quality)
```
Cameras: 2
Images: 32
Registered images: 32 (100%)
Points: 7,495 ⚠️ VERY SPARSE
Observations: 18,334
Mean track length: 2.446 ⚠️ LOW
Mean observations per image: 572.9 ⚠️ LOW
Mean reprojection error: 2.57 pixels ⚠️ HIGH
Error range: 0.0004 - 10.8 pixels
```

#### Camera Models (After Undistortion)
```
Camera 1: PINHOLE 4093 5016 5078.36 4930.99 1910.18 4172.59
Camera 2: PINHOLE 2822 3658 3428.14 3159.17 1246.15 2478.39
```

#### Image Data
- **Count**: 32 images
- **Format**: JPEG
- **Naming**: IMG_0085.jpeg to IMG_0229.jpeg (non-sequential)
- **Total Size**: ~128 MB

### Training Execution (v6)

#### Command
```bash
cd /home/ubuntu/GitHub/PlayGround/gaussian-splatting && \
source /home/ubuntu/GitHub/PlayGround/20251118_depth_anything_3/.venv/bin/activate && \
LD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH \
python train.py \
  -s /home/ubuntu/GitHub/PlayGround/20251119_SAM3/3dgs_data_undistorted \
  --iterations 30000 \
  --save_iterations 5000 10000 15000 20000 25000 30000 \
  --test_iterations 5000 10000 15000 20000 25000 30000 \
  2>&1 | tee /home/ubuntu/GitHub/PlayGround/20251119_SAM3/training_3dgs.log
```

#### Training Parameters
- **Source directory**: `3dgs_data_undistorted/`
- **Total iterations**: 30,000
- **Save checkpoints**: Every 5,000 iterations
- **Test iterations**: Every 5,000 iterations
- **Default 3DGS parameters**: All other parameters at default values

#### Training Performance
- **Duration**: ~89 seconds (1 min 29 sec)
- **Speed**: ~337 iterations/second
- **Terminal ID**: 23
- **Output directory**: `/home/ubuntu/GitHub/PlayGround/gaussian-splatting/output/6645d667-1/`

### Training Results (v6 - FAILED)

#### Final Metrics (Iteration 30,000)
```
L1 Loss: 0.365
PSNR: 7.43 dB  ⚠️ EXTREMELY LOW (should be >25 dB)
Loss: ~0.48-0.52 (stabilized)
```

#### Gaussian Count Evolution
- **Initial**: 7,495 Gaussians (from COLMAP points)
- **Final**: 7,612 Gaussians
- **Growth**: Only +117 Gaussians (+1.6%)
- **Expected**: Should grow to 100K-500K+ Gaussians

#### Checkpoint Files
All saved to: `/home/ubuntu/GitHub/PlayGround/gaussian-splatting/output/6645d667-1/point_cloud/`
```
iteration_5000/point_cloud.ply   (~1.9 MB)
iteration_10000/point_cloud.ply  (~1.9 MB)
iteration_15000/point_cloud.ply  (~1.9 MB)
iteration_20000/point_cloud.ply  (~1.9 MB)
iteration_25000/point_cloud.ply  (~1.9 MB)
iteration_30000/point_cloud.ply  (~1.9 MB)
```

#### Training Log
- **Location**: `/home/ubuntu/GitHub/PlayGround/20251119_SAM3/training_3dgs.log`
- **Size**: Full training output captured

### Problem Analysis (v6)

#### Issue 1: Corrupted Output File
**File**: `iteration_30000/point_cloud.ply` (1.9 MB)

**PLY Header**:
```
ply
format binary_little_endian 1.0
element vertex 7612
property float x
property float y
property float z
...
end_header
```

**Binary Data Inspection**:
```
Offset 0x000005f0: ff ff ff 7f ff ff ff 7f ff ff ff 7f ...
```
- Data contains mostly `ff ff ff 7f` (NaN/infinity in IEEE 754 float)
- Invalid Gaussian parameters throughout the file
- **This is why macOS cannot preview the file**

#### Issue 2: Training Didn't Converge
- Metrics **identical** from iteration 5,000 to 30,000
- No improvement in PSNR or L1 loss
- Suggests training stopped learning very early

#### Issue 3: No Densification
- Gaussian count barely increased (7,495 → 7,612)
- Normal 3DGS training should densify to 100K-500K+ Gaussians
- Indicates adaptive densification failed

### Root Cause: Poor COLMAP Reconstruction (v6)

#### Critical Issues with COLMAP v6 Data

1. **Extremely Sparse Point Cloud**
   - Only 7,495 points from 32 high-resolution images
   - Should have 50K-500K+ points for good reconstruction
   - Only ~0.01% of image pixels have 3D points

2. **High Reprojection Error**
   - Mean: 2.57 pixels (should be <1 pixel)
   - Max: 10.8 pixels (indicates outliers)
   - Noisy geometry → noisy gradients → NaN values

3. **Low Track Length**
   - Mean: 2.45 (each point seen in ~2.4 images)
   - Should be 4-10+ for robust reconstruction
   - Indicates poor image overlap

4. **Low Feature Matching**
   - Only 573 observations per image on average
   - With 4-5 MP images, this is extremely low
   - Suggests insufficient overlap or difficult scene

#### Likely Causes
- **Insufficient image overlap**: Images don't share enough common features
- **Scene characteristics**: Textureless, reflective, or repetitive patterns
- **Camera motion**: Images too far apart or insufficient baseline
- **Feature detection failure**: COLMAP couldn't find/match enough features

---

## Attempt 2: Training with COLMAP v5 (IMPROVED ✅)

### Input Data

#### COLMAP Reconstruction v5 (Better Quality)
- **Location**: `/home/ubuntu/GitHub/PlayGround/20251119_SAM3/colmap_output_v5/0/`
- **Undistorted Data**: `/home/ubuntu/GitHub/PlayGround/20251119_SAM3/3dgs_data_v5_undistorted/`
- **Camera Model**: PINHOLE (undistorted)

#### COLMAP Statistics (v5 - Much Better)
```
Cameras: 2
Images: 30
Registered images: 30 (100%)
Points: 12,651 ✅ 69% MORE than v6
Observations: 36,691
Mean track length: 2.90 ✅ Better than v6
Mean observations per image: 1,223 ✅ 2x better than v6
Mean reprojection error: 1.81 pixels ✅ 30% lower than v6
Error range: 0.0003 - 9.99 pixels
```

**Comparison with v6**:
- 69% more 3D points (12,651 vs 7,495)
- 30% lower reprojection error (1.81px vs 2.57px)
- 2x more observations per image (1,223 vs 573)
- Better track lengths (2.90 vs 2.45)

### Training Execution (v5)

#### Command
```bash
cd /home/ubuntu/GitHub/PlayGround/gaussian-splatting && \
source /home/ubuntu/GitHub/PlayGround/20251118_depth_anything_3/.venv/bin/activate && \
LD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH \
python train.py \
  -s /home/ubuntu/GitHub/PlayGround/20251119_SAM3/3dgs_data_v5_undistorted \
  --iterations 30000 \
  --save_iterations 5000 10000 15000 20000 25000 30000 \
  --test_iterations 5000 10000 15000 20000 25000 30000 \
  2>&1 | tee /home/ubuntu/GitHub/PlayGround/20251119_SAM3/training_3dgs_v5.log
```

#### Training Parameters
- **Source directory**: `3dgs_data_v5_undistorted/`
- **Total iterations**: 30,000
- **Save checkpoints**: Every 5,000 iterations
- **Test iterations**: Every 5,000 iterations
- **Default 3DGS parameters**: All other parameters at default values

#### Training Performance
- **Duration**: ~37 minutes (2,273 seconds)
- **Speed**: ~13-16 iterations/second
- **Terminal ID**: 23
- **Output directory**: `/home/ubuntu/GitHub/PlayGround/gaussian-splatting/output/8135f594-6/`

### Training Results (v5 - SUCCESS ✅)

#### Final Metrics (Iteration 30,000)
```
L1 Loss: 0.0126
PSNR: 33.29 dB ✅ MUCH BETTER (vs 7.43 dB in v6)
```

#### Metrics Evolution
| Iteration | L1 Loss | PSNR (dB) |
|-----------|---------|-----------|
| 5,000     | 0.0400  | 22.98     |
| 10,000    | 0.0264  | 26.35     |
| 15,000    | 0.0183  | 29.43     |
| 20,000    | 0.0149  | 31.43     |
| 25,000    | 0.0140  | 32.29     |
| 30,000    | 0.0126  | 33.29     |

**Observations**:
- ✅ Steady improvement throughout training
- ✅ No NaN values or divergence
- ✅ PSNR increased from 22.98 to 33.29 dB
- ✅ L1 loss decreased from 0.04 to 0.0126

#### Gaussian Count Evolution
- **Initial**: 12,651 Gaussians (from COLMAP v5 points)
- **Final**: 3,282,382 Gaussians ✅
- **Growth**: 259x increase (vs 1.6% in v6)
- **File Size**: 777 MB (vs 1.9 MB in v6)

#### Checkpoint Files
All saved to: `/home/ubuntu/GitHub/PlayGround/gaussian-splatting/output/8135f594-6/point_cloud/`
```
iteration_30000/point_cloud.ply  (777 MB) ✅ Valid data
```

**PLY Header**:
```
ply
format binary_little_endian 1.0
element vertex 3282382 ✅ 3.28 million Gaussians
property float x
property float y
property float z
...
end_header
```

**Binary Data**: Valid floating-point values, no NaN corruption

#### Training Log
- **Location**: `/home/ubuntu/GitHub/PlayGround/20251119_SAM3/training_3dgs_v5.log`
- **Size**: Full training output captured

### Quality Assessment (v5)

**Improvements over v6**:
- ✅ 431x more Gaussians (3.28M vs 7,612)
- ✅ 4.5x better PSNR (33.29 dB vs 7.43 dB)
- ✅ 410x larger file (777 MB vs 1.9 MB)
- ✅ No NaN corruption
- ✅ Proper densification occurred

**Current Limitations**:
- ⚠️ PSNR 33.29 dB is acceptable but not great (good quality is >35 dB)
- ⚠️ Likely still affected by lens distortion at image edges
- ⚠️ May have artifacts from distorted features

**Conclusion**: Training succeeded and produced a valid model, but quality can be improved further.

---

## Attempt 3: Cropped Images (IN PROGRESS 🔄)

### Motivation
- Lens distortion is worst at image edges
- Cropping to center 70% eliminates edge distortion
- Allows simpler PINHOLE camera model
- Should improve feature matching quality

### Image Preprocessing

#### Cropping Script
Created `crop_images_center.py` to crop center portion of images:
```bash
python3 crop_images_center.py images images_cropped --crop-fraction 0.7
```

#### Cropping Results
- **Input**: 154 images (4284×5712 and 3024×4032)
- **Output**: 154 cropped images
  - 4284×5712 → 2998×3998 (70% of dimensions)
  - 3024×4032 → 2116×2822 (70% of dimensions)
- **Area retained**: 49% (70% × 70%)
- **Output directory**: `images_cropped/`

### Next Steps
1. 🔄 Run COLMAP on cropped images
2. 🔄 Analyze COLMAP reconstruction quality
3. 🔄 Run 3DGS training on improved reconstruction
4. 🔄 Compare results with v5 (uncropped)

---

## Debugging Steps Taken

1. ✅ Analyzed COLMAP reconstruction quality (v6 vs v5)
2. ✅ Identified better COLMAP reconstruction (v5)
3. ✅ Re-ran 3DGS training with v5 data
4. ✅ Verified successful training (3.28M Gaussians, 33.29 dB PSNR)
5. ✅ Created image cropping script to reduce lens distortion
6. ✅ Cropped all images to center 70%

---

## Files Generated

### Attempt 1 (v6 - Failed)
- `training_3dgs.log` - Training output (failed)
- `output/6645d667-1/` - Training output directory (corrupted)
- `3dgs_data_undistorted/` - Undistorted COLMAP v6 reconstruction

### Attempt 2 (v5 - Success)
- `training_3dgs_v5.log` - Training output (successful)
- `output/8135f594-6/` - Training output directory (valid)
- `3dgs_data_v5_undistorted/` - Undistorted COLMAP v5 reconstruction
- `output/8135f594-6/point_cloud/iteration_30000/point_cloud.ply` - 777 MB, 3.28M Gaussians

### Attempt 3 (Cropped - In Progress)
- `crop_images_center.py` - Image cropping script
- `images_cropped/` - Cropped images (154 images, center 70%)

---

## Lessons Learned

1. **COLMAP quality is critical** - Better COLMAP reconstruction (v5 vs v6) made the difference between complete failure and success
2. **Check multiple COLMAP runs** - Don't assume the latest run is the best
3. **Metrics matter** - 69% more points and 30% lower error translated to 431x more Gaussians
4. **Lens distortion** - Edge distortion likely still affecting quality; cropping should help
5. **Patience** - v5 training took 37 minutes vs 89 seconds for v6, but produced valid results

---

## Conclusion

**Status**: Significant progress made. Training now works but quality needs improvement.

**Next Actions**:
1. Run COLMAP on cropped images to reduce distortion effects
2. Compare COLMAP quality metrics (cropped vs uncropped)
3. Run 3DGS training on best COLMAP reconstruction
4. Evaluate final quality and iterate if needed

