#!/bin/bash

# COLMAP v8 reconstruction with OPENCV camera model on cropped images
# Using center-cropped images (70%) with OPENCV model to handle residual distortion

set -e  # Exit on error

# Disable Qt GUI (run headless with GPU)
export QT_QPA_PLATFORM=offscreen

# Configuration
IMAGE_PATH="./images_cropped"
DATABASE_PATH="./database_v8.db"
OUTPUT_PATH="./colmap_output_v8"

# Clean previous run (optional - comment out to keep old data)
# rm -rf "$DATABASE_PATH" "$OUTPUT_PATH"

# Create output directory
mkdir -p "$OUTPUT_PATH"

echo "========================================="
echo "COLMAP v8 Reconstruction (Cropped + OPENCV)"
echo "========================================="
echo "Images: $IMAGE_PATH (154 cropped iPhone 15 Pro photos)"
echo "Database: $DATABASE_PATH"
echo "Output: $OUTPUT_PATH"
echo ""

# ============================================================================
# STEP 1: Feature Extraction
# ============================================================================
echo "[1/4] Extracting features (MAX QUALITY)..."
echo "  - Using OPENCV camera model (handles residual distortion)"
echo "  - Max features: 16384 (maximum quality)"
echo "  - Max image size: 4800 (near full resolution)"
echo "  - First octave: -1 (finest detail)"
echo "  - CPU: 52 cores (GPU requires display for OpenGL)"
echo ""

colmap feature_extractor \
    --database_path "$DATABASE_PATH" \
    --image_path "$IMAGE_PATH" \
    --ImageReader.camera_model OPENCV \
    --ImageReader.single_camera 0 \
    --SiftExtraction.max_image_size 4800 \
    --SiftExtraction.max_num_features 16384 \
    --SiftExtraction.first_octave -1 \
    --SiftExtraction.num_threads -1 \
    --SiftExtraction.use_gpu 0

echo "✓ Feature extraction complete"
echo ""

# ============================================================================
# STEP 2: Feature Matching
# ============================================================================
echo "[2/4] Matching features (exhaustive matcher)..."
echo "  - Exhaustive matching (all image pairs)"
echo "  - This will take longer but find all connections"
echo ""

colmap exhaustive_matcher \
    --database_path "$DATABASE_PATH" \
    --SiftMatching.guided_matching 1 \
    --SiftMatching.max_ratio 0.8 \
    --SiftMatching.max_distance 0.7 \
    --SiftMatching.cross_check 1 \
    --SiftMatching.max_num_matches 65536 \
    --SiftMatching.max_error 4.0 \
    --SiftMatching.confidence 0.999 \
    --SiftMatching.min_num_inliers 15 \
    --SiftMatching.use_gpu 0

echo "✓ Feature matching complete"
echo ""

# ============================================================================
# STEP 3: Sparse Reconstruction (Mapper)
# ============================================================================
echo "[3/4] Running sparse reconstruction..."
echo "  - Refining focal length, principal point, and distortion"
echo "  - Lowered thresholds to help with difficult scenes"
echo ""

colmap mapper \
    --database_path "$DATABASE_PATH" \
    --image_path "$IMAGE_PATH" \
    --output_path "$OUTPUT_PATH" \
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

echo "✓ Sparse reconstruction complete"
echo ""

# ============================================================================
# STEP 4: Analysis
# ============================================================================
echo "[4/4] Analyzing reconstruction..."
echo ""

for model in "$OUTPUT_PATH"/*; do
    if [ -d "$model" ]; then
        echo "Model: $model"
        colmap model_analyzer --path "$model"
        echo ""
    fi
done

echo "========================================="
echo "✓ Reconstruction Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Visualize in COLMAP GUI:"
echo "     colmap gui --database_path $DATABASE_PATH --import_path $OUTPUT_PATH/0"
echo ""
echo "  2. Export point cloud:"
echo "     colmap model_converter --input_path $OUTPUT_PATH/0 --output_path $OUTPUT_PATH/0/points3D.ply --output_type PLY"
echo ""
echo "  3. Check camera poses:"
echo "     ls -lh $OUTPUT_PATH/0/"
echo ""

