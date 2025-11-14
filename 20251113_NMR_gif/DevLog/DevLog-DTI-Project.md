# DevLog: DTI Visualization Project

## Metadata
- **Date**: 2025-11-14
- **Author**: Augment Agent (with Wentao)
- **Status**: Phase 1 Complete
- **Repository**: PlayGround/20251113_NMR_gif

---

## Project Overview

Implement DTI (Diffusion Tensor Imaging) visualization using modular NMR simulation framework.

**Goal:** Demonstrate how diffusion anisotropy affects MRI signal decay in different gradient directions.

---

## Phase 1: Planning & Design

### DTI Scenarios (4 Cases)

| Case | Diffusion Tensor | MD | FA | Principal Direction | Description |
|------|------------------|----|----|---------------------|-------------|
| Isotropic | diag([0.05, 0.05, 0.05]) | 0.050 | 0.00 | N/A | Free water, CSF |
| Z-Fiber | diag([0.01, 0.01, 0.10]) | 0.040 | 0.89 | [0,0,1] | White matter (Z-aligned) |
| X-Fiber | diag([0.10, 0.01, 0.01]) | 0.040 | 0.89 | [1,0,0] | White matter (X-aligned) |
| Tilted | Rotated Z-fiber (45°) | 0.040 | 0.89 | [0.707,0,0.707] | White matter (oblique) |

### Gradient Encoding Scheme

**6-direction scheme** (minimum for tensor fitting):
- Gx, Gy, Gz (orthogonal)
- Gxy, Gxz, Gyz (diagonal)

### Signal Analysis Method

**Stejskal-Tanner Equation:**
```
S(b) / S(0) = exp(-b · ADC)
```

Where:
- **b-value**: γ² G² δ² (Δ - δ/3) = 8.33 (for our parameters)
- **ADC**: g^T · D · g (apparent diffusion coefficient)

### Tensor Fitting

Linear system with 6 unknowns:
```
ADC_i = D_xx·g_ix² + D_yy·g_iy² + D_zz·g_iz² + 2·D_xy·g_ix·g_iy + 2·D_xz·g_ix·g_iz + 2·D_yz·g_iy·g_iz
```

Solve using least squares for [D_xx, D_yy, D_zz, D_xy, D_xz, D_yz].

---

## Phase 2: Implementation

### Modules Created

#### 1. `dti_scenarios.py` (200 lines)
- Rotation matrices (x, y, z axes)
- `DTIScenario` class with automatic eigendecomposition
- `get_standard_scenarios()` - returns 4 predefined cases
- `get_gradient_scheme()` - 6-dir or 12-dir encoding

#### 2. `dti_analysis.py` (220 lines)
- `compute_b_value()` - b-value calculation
- `compute_analytical_echo_decay()` - Stejskal-Tanner equation
- `extract_ADC()` - ADC from measured echo amplitudes
- `fit_diffusion_tensor()` - Linear least squares fitting
- `compute_dti_metrics()` - MD, FA, eigenvalues
- `compare_tensors()` - Validation metrics

### Test Scripts

#### 1. `test_all_dti_cases.py`
Validates all 4 scenarios using analytical method.

**Results:** ✓✓✓ ALL TESTS PASSED ✓✓✓
- Error < 10⁻¹⁰ for all cases
- Perfect tensor reconstruction
- MD and FA exact to machine precision

#### 2. `generate_single_case_gif.py`
Generates animated GIF for any case/gradient combination.

**Usage:**
```bash
python3 generate_single_case_gif.py [case_name] [gradient_direction]
```

---

## Phase 3: Results & Validation

### Analytical Validation

| Case | MD (fitted) | FA (fitted) | Max Error | Status |
|------|-------------|-------------|-----------|--------|
| Isotropic | 0.050000 | 0.000000 | < 10⁻¹⁰ | ✓ PASS |
| Z-Fiber | 0.040000 | 0.891133 | < 10⁻¹⁰ | ✓ PASS |
| X-Fiber | 0.040000 | 0.891133 | < 10⁻¹⁰ | ✓ PASS |
| Tilted | 0.040000 | 0.891133 | < 10⁻¹⁰ | ✓ PASS |

### Generated Visualizations

#### Experiment 1: Same Fiber, Different Gradients
**Z-Fiber with varying gradient directions:**
- `dti_z_fiber_gradz.gif` (10 MB) - Parallel: ADC=0.100, strong decay
- `dti_z_fiber_gradx.gif` (12 MB) - Perpendicular: ADC=0.010, weak decay

**Key Insight:** Same diffusion pattern, different measurements depending on gradient direction.

#### Experiment 2: Same Gradient, Different Fibers
**Z-Gradient with varying fiber orientations:**
- `dti_z_fiber_gradz.gif` (10 MB) - Parallel: ADC=0.100, Z-range=29
- `dti_x_fiber_gradz.gif` (13 MB) - Perpendicular: ADC=0.010, Z-range=9
- `dti_tilted_fiber_gradz.gif` (12 MB) - Oblique (45°): ADC=0.055, Z-range=18

**Key Insight:** 10× ADC variation (0.01→0.10) based on fiber-gradient alignment.

### Physics Validation

**ADC Formula:**
```
ADC(θ) = D_parallel·cos²(θ) + D_perp·sin²(θ)
```

| Fiber-Gradient Angle | Expected ADC | Measured ADC | Match |
|---------------------|--------------|--------------|-------|
| 0° (parallel) | 0.100 | 0.100 | ✓ |
| 90° (perpendicular) | 0.010 | 0.010 | ✓ |
| 45° (oblique) | 0.055 | 0.055 | ✓ |

**Position Spread Validation:**
- Z-fiber: Z-spread / X-spread ≈ 3.6 ≈ √(D_zz/D_xx) = √10 ✓
- X-fiber: X-spread / Z-spread ≈ 3.0 ≈ √(D_xx/D_zz) = √10 ✓

---

## Key Concepts Demonstrated

### 1. Apparent Diffusion Coefficient (ADC)
**Definition:** Effective diffusion measured in a specific gradient direction.

**Formula:** ADC = g^T · D · g

**Physical Meaning:**
- ADC is what we **measure** (direction-dependent)
- D is what tissue **has** (intrinsic property)
- ADC is projection of D onto gradient direction

### 2. Diffusion vs. Gradient Direction
**Critical Insight:** Diffusion direction ≠ Gradient direction
- **Diffusion tensor** determines how molecules move (intrinsic)
- **Gradient direction** determines what we measure (our choice)
- Same diffusion can give different ADC values with different gradients

### 3. DTI Principle
By measuring ADC in 6+ directions, reconstruct full 3D diffusion tensor:
```
[ADC_1]   [g_1^T · D · g_1]
[ADC_2] = [g_2^T · D · g_2]
[ADC_3]   [g_3^T · D · g_3]
  ...           ...
```

Solve for 6 unknowns in symmetric tensor D.

---

## Summary Statistics

**Code Created:**
- 2 core modules (420 lines)
- 3 test/generation scripts (350 lines)
- Total: ~770 lines of new code

**Visualizations Generated:**
- 5 GIF animations (~60 MB total)
- Each: 150 frames, 60 spins, 3-panel layout

**Validation:**
- 4 scenarios × 6 gradients = 24 ADC measurements
- All within machine precision of theoretical values
- Tensor fitting: 0% error

---

## Project Status

✅ **Complete:**
- DTI scenario framework
- Analytical validation
- GIF generation for individual cases
- Physics verification

🔄 **Possible Extensions:**
- Multi-panel comparison GIF (all cases side-by-side)
- Quantitative overlays (ADC values, decay curves)
- Multiple b-values
- Noise simulation
- Interactive visualization

---

## Technical Notes

See `DevLog-Technical-Notes.md` for:
- Framework generalization verification
- ADC measurement error analysis and solution
- 3D visualization bug fix

---

**End of DevLog**

