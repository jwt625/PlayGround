# Development Log: SAM2 Interactive Segmentation Interface

**Date:** 2025-11-22  
**Objective:** Create an interactive web-based segmentation interface for SAM2 with positive/negative control points, similar to EdgeTAM

## Background

The existing `run_sam3_web.py` only supported single positive point clicks for segmentation. The goal was to create a more flexible interface allowing:
- Multiple positive points (areas to include)
- Negative points (areas to exclude)
- Real-time point visualization
- Proper handling of various image formats, especially TIFF files

## Implementation

### Created: `run_sam2_interactive.py`

A Gradio-based web interface combining SAM2's segmentation capabilities with EdgeTAM's interactive UI patterns.

**Key Features:**
- Interactive point selection with radio button to toggle between positive (green) and negative (red) points
- Real-time point visualization on image
- Segmentation on demand (click "Segment" button)
- Image upload support with automatic SAM2 predictor reinitialization
- Mask saving (PNG and NPY formats)
- Multi-mask output option

**Usage:**
```bash
source sam-3d-objects/.venv/bin/activate
python run_sam2_interactive.py <image_path> [--port 7861]
```

Example:
```bash
python run_sam2_interactive.py IMG_8020.JPG --port 7861
```

## Technical Challenges and Solutions

### Challenge 1: Image Upload Not Updating Visualization

**Problem:** When uploading a new image, the input panel updated but the "Image with Points" panel showed the old image.

**Root Cause:** Visualization functions were using the original global `image` variable instead of tracking the currently loaded image.

**Solution:** 
- Added `current_image` and `current_image_np` global variables
- Created `handle_image_upload()` function to update these variables and reinitialize SAM2 predictor
- Updated all visualization functions to use `current_image`

### Challenge 2: Slow Point Visualization

**Problem:** Point visualization was slow, taking noticeable time after each click.

**Root Cause:** Every click triggered:
1. Full image copy and drawing operations
2. Excessive logging statements
3. Image data transfer through Gradio

**Solution:**
- Removed excessive logging from `visualize_points()`
- Kept visualization lightweight (shallow copy + draw operations only)
- Points still update in real-time but without unnecessary overhead

### Challenge 3: TIFF Images Appearing as Pure Black and White

**Problem:** TIFF images (especially grayscale) were being binarized - all non-black pixels became pure white, losing all intermediate grayscale values.

**Root Cause:** PIL's `convert("RGB")` on certain TIFF formats was not preserving grayscale values properly. Gradio may also have been pre-processing the images.

**Solution:** Used OpenCV for TIFF file handling:
```python
import cv2

# For TIFF files
img_cv = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

# Normalize 16-bit to 8-bit
if img_cv.dtype == np.uint16:
    img_cv = (img_cv / 256).astype(np.uint8)

# Convert grayscale to RGB
if len(img_cv.shape) == 2:
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB)

image = Image.fromarray(img_cv)
```

This approach:
- Preserves full grayscale dynamic range
- Properly handles 16-bit TIFF files
- Converts to RGB without binarization

## Code Structure

**Global State:**
- `current_mask`: Currently generated mask
- `mask_counter`: Counter for saved masks
- `current_image`: Currently loaded PIL Image
- `current_image_np`: Numpy array of current image

**Key Functions:**
- `preprocess_image()`: Handles image loading with OpenCV for TIFF files
- `visualize_points()`: Draws colored circles on image for point visualization
- `create_overlay()`: Creates mask overlay on original image
- `segment_image()`: Runs SAM2 inference with point prompts
- `handle_image_upload()`: Processes new image uploads and reinitializes SAM2
- `update_points()`: Handles click events to add points
- `save_mask()`: Saves current mask to disk
- `clear_all()`: Resets points and outputs

**Event Handlers:**
- `input_image.upload()`: Calls `handle_image_upload()`
- `input_image.select()`: Calls `update_points()` to add point on click
- `segment_btn.click()`: Calls `segment_image()` to run inference
- `clear_btn.click()`: Calls `clear_all()` to reset
- `save_btn.click()`: Calls `save_mask()` to save result

## Dependencies

Required packages:
- `gradio`: Web interface
- `opencv-python` (`cv2`): TIFF image handling
- `numpy`: Array operations
- `PIL` (Pillow): Image processing
- SAM2 (from sam-3d-objects repository)

## Output Files

Masks are saved to `<image_name>_sam2_interactive/`:
- `mask_000.png`, `mask_001.png`, etc. (PNG format)
- `mask_000.npy`, `mask_001.npy`, etc. (NumPy format)

## Comparison with EdgeTAM Implementation

**Similarities:**
- Point selection with positive/negative labels
- Gradio-based web interface
- Separate panels for input, points visualization, and segmentation result

**Differences:**
- SAM2 backend instead of EdgeTAM model
- OpenCV-based TIFF handling for better grayscale support
- Direct SAM2ImagePredictor API instead of HuggingFace transformers
- Simpler preprocessing (EdgeTAM uses processor, we use direct numpy arrays)

## Script Organization

**Active Scripts:**
- `run_sam2_interactive.py`: New interactive segmentation (this work)
- `run_sam3_web.py`: Single-click segmentation + 3D reconstruction
- `run_sam3_points_cli.py`: CLI with pre-specified points
- `run_sam3d_vehicle.py`: Full-image 3D reconstruction
- `run_sam3_auto_segment.py`: Auto segmentation with mask merging
- `run_sam3_points.py`: Matplotlib-based point selection

## Future Improvements

Potential enhancements:
- Add bounding box mode (like EdgeTAM)
- Support for video/multi-frame segmentation
- Integration with SAM3D Objects for direct 3D reconstruction
- Undo/redo for point selection
- Point editing (remove individual points)
- Mask refinement tools

