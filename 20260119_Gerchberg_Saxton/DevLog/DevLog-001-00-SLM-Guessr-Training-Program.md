# DevLog-001-00: SLM-Guessr Training Program

**Date:** 2026-01-20  
**Author:** Wentao  
**Status:** Planning

---

## Objective

Build an educational program that trains users to develop intuition for SLM phase masks and 2D FFT behavior. Users view phase-intensity pairs progressing from simple to complex, building pattern recognition skills for holographic optics.

Named "SLM-Guessr" (inspired by GeoGuessr).

---

## Program Modes

| Mode | Description | Implementation Complexity |
|------|-------------|---------------------------|
| Gallery | Browse annotated phase-intensity pairs by concept/level | Low |
| Quiz | Given one side, predict the other (multiple choice) | Medium |
| Interactive | Adjust parameters, see FFT update in real-time | High |
| Challenge | Match target intensity by tweaking phase | High |

**Initial scope:** Gallery and Quiz modes only. Interactive/Challenge deferred due to high degrees of freedom in phase mask parameter space.

---

## Curriculum Structure

### Level 1 - Foundations
- Uniform phase (baseline Gaussian FT)
- Linear phase ramp (x, y, diagonal) - shift theorem
- Quadratic phase (lens) - focus/defocus

### Level 2 - Periodic Structures
- Binary grating (0/pi stripes) - diffraction orders
- Sinusoidal grating - Bessel weights
- Vary period and orientation
- Checkerboard (2D separable)
- Blazed grating (sawtooth)

### Level 3 - Discrete Spots
- 2x2, 3x3 spot arrays (GS-optimized)
- Non-uniform brightness
- Off-center spots
- Random positions

### Level 4 - Special Beams
- Vortex (charge 1, 2, 3, 4) - spiral phase, donut intensity
- Axicon - ring/Bessel beam
- Vortex + lens, vortex + grating

### Level 5 - Compound Patterns
- Grating + lens
- Multiple superimposed gratings
- Random phase (speckle)

### Level 6 - Complex Targets
- Letter shapes
- Geometric shapes
- Images (GS-optimized, phase appears noisy)

### Level 7 - Artifacts and Edge Cases
- GS stagnation
- Uniformity problems
- Phase quantization effects

---

## Technical Architecture

```
slm_guessr/
  core/
    patterns.py         # Analytic pattern generators
    fft_engine.py       # FFT with proper shifting
    gs_algorithms.py    # Reuse from existing project
  curriculum/
    lessons.py          # Lesson definitions and metadata
  modes/
    gallery.py          # Browse mode
    quiz.py             # Assessment mode
  visualization/
    renderer.py         # Side-by-side rendering with annotations
  main.py               # Entry point
  config.py             # Global settings
```

---

## Pattern Generators Required

```python
# Analytic (no optimization needed)
linear_phase(size, kx, ky)              # Tilt/shift
quadratic_phase(size, curvature)        # Lens
vortex_phase(size, charge)              # Spiral
axicon_phase(size, slope)               # Conical
grating_phase(size, period, angle, profile)  # Binary/sine/blaze
random_phase(size)                      # Uniform random

# Compound
superimpose(*phases)                    # Sum mod 2pi

# GS-based (for complex targets)
weighted_gs(input_amp, target, n_iter)  # Existing implementation
```

---

## Visualization Requirements

Each sample displays:
- Phase mask (cyclic colormap: twilight)
- Input amplitude (Gaussian beam)
- Fourier plane intensity (hot colormap)
- Text annotations for key features
- Optional: 1D cross-sections

---

## Output Formats

| Format | Purpose |
|--------|---------|
| PNG gallery | Static reference, printable |
| Animated GIF | Parameter sweeps per concept |
| CLI quiz | Terminal-based assessment |

---

## Implementation Plan

### Phase 1: Core Infrastructure
- Pattern generator module
- Visualization renderer
- Lesson data structure

### Phase 2: Gallery Mode
- Generate all curriculum samples
- Navigation by level/concept
- Annotation overlays

### Phase 3: Quiz Mode
- Question bank from gallery
- Multiple choice interface
- Difficulty by curriculum level

### Phase 4: Polish
- Progress tracking
- Export to PDF
- Additional pattern types

---

## Scope Boundaries

**In scope:**
- Static gallery generation
- CLI-based quiz
- Levels 1-6 curriculum content

**Out of scope (for now):**
- Interactive real-time mode
- Challenge mode with parameter search
- Web interface
- Level 7 edge cases (defer to later)

---

## Dependencies

- numpy, matplotlib, scipy (existing)
- imageio (for GIF export)
- Optional: rich (for CLI quiz formatting)

---

## Next Steps

1. Create `slm_guessr/` directory structure
2. Implement `patterns.py` with all analytic generators
3. Implement `renderer.py` for consistent visualization
4. Define lesson metadata format
5. Generate Level 1 gallery as proof of concept

---

## References

- Goodman, J. W. Introduction to Fourier Optics
- Gerchberg-Saxton algorithm (see DevLog-000)

