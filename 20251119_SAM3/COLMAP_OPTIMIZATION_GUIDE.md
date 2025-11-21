# COLMAP Optimization Guide

## Results Summary (154 iPhone 15 Pro images, 5712x4284)

| Version | Database | Matcher | Images | 3D Points | Repr. Error | Notes |
|---------|----------|---------|--------|-----------|-------------|-------|
| v3 | database.db | Sequential (overlap=15) | 22 | 5,531 | 1.99px | Single model |
| **v5** | **database_v5.db** | **Exhaustive** | **30** | **12,651** | **1.81px** | **Best result** |
| v6 | database_v5.db | Exhaustive (reused) | 32 | 7,495 | 2.57px | More permissive mapper |

**Key Finding**: Only 21% of images registered due to low overlap between shots. Exhaustive matching found significantly more connections than sequential matching.

---

## v5 Configuration (Best Result - 30 images, 12,651 points)

### Feature Extraction
```bash
colmap feature_extractor \
    --database_path database_v5.db \
    --image_path ./images \
    --ImageReader.camera_model OPENCV \
    --ImageReader.single_camera 0 \
    --SiftExtraction.max_image_size 4800 \
    --SiftExtraction.max_num_features 16384 \
    --SiftExtraction.first_octave -1 \
    --SiftExtraction.num_threads -1 \
    --SiftExtraction.use_gpu 0
```

### Feature Matching (Exhaustive)
```bash
colmap exhaustive_matcher \
    --database_path database_v5.db \
    --SiftMatching.guided_matching 1 \
    --SiftMatching.max_ratio 0.8 \
    --SiftMatching.max_distance 0.7 \
    --SiftMatching.cross_check 1 \
    --SiftMatching.max_num_matches 65536 \
    --SiftMatching.max_error 4.0 \
    --SiftMatching.confidence 0.999 \
    --SiftMatching.min_num_inliers 15 \
    --SiftMatching.use_gpu 0
```

### Sparse Reconstruction (Mapper)
```bash
colmap mapper \
    --database_path database_v5.db \
    --image_path ./images \
    --output_path ./colmap_output_v5 \
    --Mapper.ba_refine_focal_length 1 \
    --Mapper.ba_refine_principal_point 1 \
    --Mapper.ba_refine_extra_params 1 \
    --Mapper.min_num_matches 8 \
    --Mapper.init_min_num_inliers 30 \
    --Mapper.abs_pose_min_num_inliers 10 \
    --Mapper.abs_pose_min_inlier_ratio 0.10 \
    --Mapper.ba_local_num_images 10 \
    --Mapper.ba_local_max_num_iterations 50 \
    --Mapper.ba_global_max_num_iterations 100 \
    --Mapper.ba_global_images_ratio 1.2 \
    --Mapper.ba_global_points_ratio 1.2 \
    --Mapper.ba_global_max_refinements 10 \
    --Mapper.filter_max_reproj_error 8.0 \
    --Mapper.filter_min_tri_angle 0.5 \
    --Mapper.max_reg_trials 10 \
    --Mapper.multiple_models 0 \
    --Mapper.num_threads -1
```

**Key Differences from Default:**
- **Exhaustive matcher** instead of sequential (critical for low-overlap images)
- **16384 features** (4x default) for high-res images
- **4800px max image size** (near full resolution)
- **Very permissive mapper thresholds** to register as many images as possible
- **Multiple models disabled** to force single reconstruction

---

## Problem Diagnosis

Original run: **154 images registered, 0 3D points** → Feature matching failed completely.

## Key Parameters to Optimize

### 1. Feature Extraction (`colmap feature_extractor`)

| Parameter | Default | Recommended | Why |
|-----------|---------|-------------|-----|
| `--ImageReader.camera_model` | SIMPLE_RADIAL | **OPENCV** | iPhone lenses have distortion that needs proper modeling |
| `--SiftExtraction.max_num_features` | 4096 | **8192-16384** | More features = better matching (your images are high-res) |
| `--SiftExtraction.max_image_size` | 3200 | **3200-4800** | Your images are 5712×4284, downscaling helps speed |
| `--SiftExtraction.first_octave` | 0 | **-1** | Detects finer features in high-resolution images |
| `--ImageReader.single_camera` | 0 | **0** | Auto-detect per image (iPhone EXIF has camera info) |

### 2. Feature Matching

| Parameter | Default | Recommended | Why |
|-----------|---------|-------------|-----|
| `--SiftMatching.guided_matching` | 0 | **1** | Use geometric verification during matching |
| `--SiftMatching.max_ratio` | 0.8 | **0.7-0.8** | Lowe's ratio test (lower = stricter) |
| `--SiftMatching.cross_check` | 1 | **1** | Bidirectional matching (keep enabled) |
| `--SiftMatching.min_num_inliers` | 15 | **15-30** | Minimum matches to accept pair |

**Matcher Choice:**
- **Sequential matcher**: For ordered images (like yours: IMG_0085-0238) - FASTER
- **Exhaustive matcher**: For unordered images - SLOWER but more thorough

### 3. Sparse Reconstruction (`colmap mapper`)

| Parameter | Default | Recommended | Why |
|-----------|---------|-------------|-----|
| `--Mapper.init_min_num_inliers` | 100 | **50-100** | Lower if reconstruction fails to start |
| `--Mapper.min_num_matches` | 15 | **15-30** | Minimum matches to register image |
| `--Mapper.abs_pose_min_num_inliers` | 30 | **20-30** | Minimum inliers for pose estimation |
| `--Mapper.filter_max_reproj_error` | 4.0 | **2.0-4.0** | Lower = higher quality (but may lose points) |
| `--Mapper.ba_refine_focal_length` | 1 | **1** | Refine focal length during bundle adjustment |
| `--Mapper.ba_refine_principal_point` | 0 | **1** | Refine principal point (important for iPhone) |

## Best Tools to Inspect Camera Poses

### 1. COLMAP GUI (Built-in, Best)
```bash
colmap gui --database_path database.db --import_path colmap_output/0
```
- ✅ Interactive 3D viewer
- ✅ Shows camera frustums, 3D points, images
- ✅ Inspect individual camera parameters
- ✅ Export poses to various formats

### 2. Read Binary Files with Python
```python
# Install: pip install numpy matplotlib
import struct
import numpy as np

def read_images_binary(path):
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("Q", f.read(8))[0]
        for _ in range(num_images):
            props = struct.unpack("idddddddi", f.read(64))
            image_id = props[0]
            qw, qx, qy, qz = props[1:5]  # Rotation quaternion
            tx, ty, tz = props[5:8]      # Translation
            # ... (read image name and points2D)
            images[image_id] = {'qvec': [qw,qx,qy,qz], 't': [tx,ty,tz]}
    return images

poses = read_images_binary('colmap_output/0/images.bin')
```

### 3. Export to Other Formats
```bash
# Export to text format (easier to read)
colmap model_converter \
    --input_path colmap_output/0 \
    --output_path colmap_output/0_text \
    --output_type TXT

# Export point cloud to PLY
colmap model_converter \
    --input_path colmap_output/0 \
    --output_path colmap_output/0/points3D.ply \
    --output_type PLY
```

### 4. Rerun.io (Modern Visualization)
```bash
pip install rerun-sdk
# Then use custom script to log COLMAP data to Rerun
```

## Troubleshooting

### Still Getting 0 Points?

1. **Check database for matches:**
   ```bash
   sqlite3 database.db "SELECT COUNT(*) FROM matches"
   ```
   If 0 → matching failed, try:
   - Increase `--SiftExtraction.max_num_features` to 16384
   - Use exhaustive matcher instead of sequential
   - Lower `--SiftMatching.max_ratio` to 0.7

2. **Check if images have overlap:**
   - Images must share visible content
   - Sequential images should have 60-80% overlap

3. **Try different camera model:**
   - OPENCV (current)
   - OPENCV_FISHEYE (for wide-angle)
   - RADIAL (simpler, faster)

### Reconstruction is Partial?

- Lower `--Mapper.abs_pose_min_inlier_ratio` to 0.2
- Increase `--Mapper.max_reg_trials` to 5
- Check for repetitive textures or reflective surfaces

### Too Slow?

- Reduce `--SiftExtraction.max_image_size` to 2400
- Reduce `--SiftExtraction.max_num_features` to 4096
- Use sequential matcher (already in script)

## Running the Script

```bash
chmod +x run_colmap.sh
./run_colmap.sh
```

Expected output: **154 images registered, >10,000 3D points** (if successful)

