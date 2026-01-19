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
├── demo.py                                  # Run comparison and generate figures
├── results.png                              # Output visualization
└── .venv/                                   # Python virtual environment
```

---

## Environment Setup

### Requirements
- Python 3.10+
- numpy
- matplotlib
- scipy

### Setup Instructions

```bash
# Create virtual environment using uv
uv venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
uv pip install numpy matplotlib scipy
```

### Run Demo

```bash
python demo.py
```

Output: `results.png`

---

## References

- Gerchberg, R. W., & Saxton, W. O. (1972). A practical algorithm for the determination of phase from image and diffraction plane pictures. Optik, 35, 237-246.
- Di Leonardo, R., Ianni, F., & Ruocco, G. (2007). Computer generation of optimal holograms for optical trap arrays. Optics Express, 15(4), 1913-1922.

---

## Log

| Date       | Update                                      |
|------------|---------------------------------------------|
| 2026-01-19 | Initial plan and structure created          |

