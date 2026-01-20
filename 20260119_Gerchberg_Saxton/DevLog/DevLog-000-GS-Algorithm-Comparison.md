# DevLog-000: Gerchberg-Saxton Algorithm Comparison Demo

**Date:** 2026-01-19  
**Author:** Wentao  
**Status:** In Progress

---

## Context

The Gerchberg-Saxton (GS) algorithm is an iterative phase retrieval method developed in 1972. It recovers phase information from intensity measurements by alternating projections between the object plane and Fourier plane.

**Problem:** Standard GS can produce non-uniform intensity distributions and may stagnate in local minima.

**Solution:** Variants like Weighted GS (WGS) improve uniformity through adaptive weighting.

This demo provides a comparison of GS algorithm variants with clear, readable code and visual outputs.

---

## Algorithms Implemented

1. **Standard GS** — Baseline iterative phase retrieval
2. **Weighted GS (WGS)** — Adaptive weighting for improved spot uniformity
3. **GS with Random Phase Reset** — Periodic phase perturbation to escape local minima

---

## Target Patterns

1. **Multi-spot array (4x4 grid)** — Tests uniformity across discrete spots
2. **Custom shape (letter "A")** — Tests fidelity for complex continuous patterns

---

## Output Visualization

Single figure (`results.png`) with subplots:

```
| Target | Phase Mask | Reconstructed | Error Curve |
|--------|------------|---------------|-------------|
| GS     |    ...     |      ...      |    ...      |
| WGS    |    ...     |      ...      |    ...      |
```

Metrics reported:
- Reconstruction error vs. iteration
- Uniformity metric (coefficient of variation for spot arrays)

---

## File Structure

```
20260119_Gerchberg_Saxton/
├── DevLog-000-GS-Algorithm-Comparison.md   # This file
├── gs_algorithms.py                         # Core algorithm implementations
├── demo.py                                  # Algorithm comparison demo
├── demo_tweezer.py                          # Optical tweezer rearrangement demo
├── results.png                              # Algorithm comparison output
├── tweezer_phase.gif                        # Tweezer phase mask animation
├── tweezer_intensity.gif                    # Tweezer intensity animation
└── .venv/                                   # Python virtual environment
```

---

## Environment Setup

### Requirements
- Python 3.10+
- numpy
- matplotlib
- scipy
- imageio

### Setup Instructions

```bash
# Create virtual environment using uv
uv venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
uv pip install numpy matplotlib scipy imageio
```

### Run Demos

```bash
# Algorithm comparison
python demo.py
# Output: results.png

# Optical tweezer rearrangement
python demo_tweezer.py
# Output: tweezer_phase.gif, tweezer_intensity.gif
```

---

## References

- Gerchberg, R. W., & Saxton, W. O. (1972). A practical algorithm for the determination of phase from image and diffraction plane pictures. Optik, 35, 237-246.
- Di Leonardo, R., Ianni, F., & Ruocco, G. (2007). Computer generation of optimal holograms for optical trap arrays. Optics Express, 15(4), 1913-1922.

---

## Implemented Demos

### Demo 1: Algorithm Comparison (demo.py)

Compares Standard GS, Weighted GS, and GS with Random Phase Reset on two target patterns (4x4 spot array and letter "A"). Outputs a single figure showing phase masks, reconstructed intensities, and convergence curves.

**Status:** Complete

### Demo 2: Optical Tweezer Rearrangement (demo_tweezer.py)

Demonstrates smooth atom rearrangement for optical tweezer applications:
- Start: 60 randomly placed spots in 11x11 grid
- End: Circular compact arrangement around center
- Uses Hungarian algorithm for optimal spot-to-target assignment (minimizes total travel)
- Warm-start WGS: each frame uses previous phase as initial guess for smooth evolution
- Output: 60-frame animation at 15 FPS

**Status:** Complete

---

## Planned: Bad Apple Demo

### Objective

Render the "Bad Apple" shadow art video using GS-computed phase masks. Each frame of the video becomes a target intensity pattern, and the corresponding phase mask is computed via Weighted GS.

### Approach

1. Source video: Bad Apple (black and white silhouette animation)
2. Extract frames, resize to 256x256, threshold to binary
3. Run WGS per frame with warm-start from previous frame
4. Output: `badapple_phase.gif`, `badapple_intensity.gif`

### Parameters

| Setting    | Value       | Rationale                              |
|------------|-------------|----------------------------------------|
| Resolution | 256x256     | Balance of detail vs. computation time |
| FPS        | 15          | Downsample from 30, halves frame count |
| Duration   | 30 seconds  | Demo subset, ~450 frames               |
| GS iters   | 50/frame    | Sufficient with warm-start             |

### Estimated Runtime

- ~450 frames at ~0.1s per frame = ~1-2 minutes total

### Dependencies

- Video source (to be obtained)
- opencv-python or ffmpeg for frame extraction

**Status:** Planned

---

## Log

| Date       | Update                                           |
|------------|--------------------------------------------------|
| 2026-01-19 | Initial plan and structure created               |
| 2026-01-19 | Implemented demo.py, gs_algorithms.py            |
| 2026-01-19 | Implemented demo_tweezer.py (8x8 to 4x8 compact) |
| 2026-01-19 | Updated tweezer to 11x11 circular compaction     |
| 2026-01-19 | Added Bad Apple demo plan                        |
