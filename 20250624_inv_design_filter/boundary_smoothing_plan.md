# Boundary Smoothing Plan

## Overview
Convert the existing 16x16 binary grid pattern to a smoothed version with better boundary quality using spatial domain processing.

## Current State
- 16x16 binary grid from image processing (main.py)
- Sharp, pixelated boundaries with "staircase" artifacts
- Direct conversion to DXF creates blocky rectangles

## Goal
Create smoother, more organic boundaries while maintaining the overall pattern structure.

## Processing Pipeline

### Step 1: Upsampling (16x16 → 160x160)
- Take existing 16x16 binary grid
- Create higher resolution 160x160 grid (10x finer resolution)
- Each original pixel becomes a 10x10 block of the same binary value
- This provides sufficient resolution for smooth boundary detection

### Step 2: Gaussian Smoothing
- Apply 2D convolution with Gaussian kernel
- **Adjustable parameter**: Gaussian radius/sigma
  - Larger radius = more smoothing
  - Smaller radius = closer to original pattern
- Converts sharp binary transitions to smooth gradients
- Results in continuous values between 0 and 1

### Step 3: Re-thresholding
- Apply threshold (typically 0.5) to convert back to binary
- Creates new binary pattern with smoother boundaries
- Removes staircase artifacts while preserving overall structure

### Step 4: Boundary Detection
- Detect contours/boundaries of the smoothed binary regions
- Extract shape coordinates for further processing
- Prepare for conversion to manufacturing formats (DXF, etc.)

## Parameters to Control
1. **Upsampling factor**: Currently 10x (16x16 → 160x160)
2. **Gaussian radius/sigma**: Adjustable smoothing strength
3. **Threshold value**: For binary conversion (default 0.5)

## Expected Benefits
- Smoother, more manufacturable boundaries
- Reduced staircase artifacts
- Better aesthetic quality
- Maintained overall pattern fidelity
- Adjustable smoothing level

## Implementation Notes
- Reuse image loading and initial grid creation from main.py
- Add scipy for Gaussian filtering
- Use opencv or skimage for boundary detection
- Maintain DXF export capability for manufacturing
