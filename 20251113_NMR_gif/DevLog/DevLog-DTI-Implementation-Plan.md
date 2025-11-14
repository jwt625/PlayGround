# DevLog: DTI Implementation Plan

## Metadata
- **Date**: 2025-11-14
- **Author**: Augment Agent (with Wentao)
- **Project**: NMR Diffusion Tensor Imaging Simulation
- **Status**: Planning Phase
- **Related Files**: 
  - `diffusion.py`
  - `nmr_simulator.py`
  - `nmr_animation.py`
  - `test_generalization.py`

---

## Background

Successfully modularized the Hahn Echo diffusion simulation code into reusable components. The framework has been verified to support:
- ✅ 3D diffusion with arbitrary tensors
- ✅ Rotated diffusion tensors
- ✅ Gradients in arbitrary directions
- ✅ Proper gradient-position relationships

**Verification**: See `test_generalization.py` - all tests passed.

---

## Objective

Implement Diffusion Tensor Imaging (DTI) visualization and quantitative analysis to demonstrate:
1. Anisotropic diffusion in different tissue types
2. Directional dependence of apparent diffusion coefficient (ADC)
3. Extraction of DTI metrics (FA, MD, eigenvalues, fiber directions)
4. Validation of tensor fitting from simulated data

---

## Part 1: Diffusion Scenarios (4 Cases)

### Case 1: Isotropic Diffusion (Control)
```python
D_iso = np.diag([0.05, 0.05, 0.05])  # mm²/s
```
- **Represents**: Free water, CSF
- **Expected behavior**: Equal signal decay for all gradient directions
- **FA**: ~0.0
- **MD**: 0.05 mm²/s

### Case 2: Anisotropic - Fiber along Z-axis
```python
D_z = np.diag([0.01, 0.01, 0.10])  # mm²/s
```
- **Represents**: White matter tract aligned with z-axis
- **Expected behavior**: 
  - Large decay with Gz (parallel to fiber)
  - Small decay with Gx/Gy (perpendicular to fiber)
- **Eigenvalues**: λ₁=0.10, λ₂=0.01, λ₃=0.01
- **FA**: ~0.82
- **MD**: 0.04 mm²/s
- **Principal direction**: [0, 0, 1]

### Case 3: Anisotropic - Fiber along X-axis
```python
D_x = np.diag([0.10, 0.01, 0.01])  # mm²/s
```
- **Represents**: White matter tract aligned with x-axis
- **Expected behavior**: Large decay with Gx, small with Gy/Gz
- **Eigenvalues**: λ₁=0.10, λ₂=0.01, λ₃=0.01
- **FA**: ~0.82
- **MD**: 0.04 mm²/s
- **Principal direction**: [1, 0, 0]

### Case 4: Anisotropic - Tilted Fiber (45° in XZ plane)
```python
theta = np.pi / 4
R_y = rotation_matrix_y(theta)
D_tilted = R_y @ D_z @ R_y.T
```
- **Represents**: Oblique fiber orientation
- **Expected behavior**: Intermediate decay for Gx and Gz
- **Eigenvalues**: λ₁=0.10, λ₂=0.01, λ₃=0.01 (preserved)
- **FA**: ~0.82
- **MD**: 0.04 mm²/s
- **Principal direction**: [0.707, 0, 0.707]

---

## Part 2: Gradient Encoding Scheme

### Minimum 6 Directions (for tensor fitting)
1. **Gx** = [1, 0, 0]
2. **Gy** = [0, 1, 0]
3. **Gz** = [0, 0, 1]
4. **G_xy** = [1/√2, 1/√2, 0]
5. **G_xz** = [1/√2, 0, 1/√2]
6. **G_yz** = [0, 1/√2, 1/√2]

### Optional: Extended Scheme (12+ directions)
- More directions → better tensor estimation
- Can use standard DTI schemes (e.g., Jones 30-direction)

### b-value Calculation
```python
# Stejskal-Tanner equation parameters
gamma = 1.0  # normalized gyromagnetic ratio
G = gradient_strength  # e.g., 0.01
delta = tau  # gradient duration
Delta = tau  # time between gradient pulses

b = gamma**2 * G**2 * delta**2 * (Delta - delta/3)
```

---

## Part 3: Signal Analysis & ADC Extraction

### Theory: Stejskal-Tanner Equation
```
S(b) / S(0) = exp(-b * ADC)

where ADC = g^T · D · g
```
- **g**: unit gradient direction vector
- **D**: diffusion tensor (3×3 symmetric matrix)
- **ADC**: apparent diffusion coefficient in direction g

### Extraction Method

**Step 1: Measure Echo Amplitude**
```python
# For each gradient direction
M_echo = |M_xy(t = 2*tau)|  # Net magnetization at echo time
```

**Step 2: Compute b-value**
```python
b = gamma**2 * G**2 * delta**2 * (Delta - delta/3)
```

**Step 3: Extract ADC**
```python
# Compare with b=0 (no gradient) reference
S_b0 = echo_amplitude_no_gradient
S_b = echo_amplitude_with_gradient

ADC = -ln(S_b / S_b0) / b
```

**Step 4: Fit Diffusion Tensor**

With N ≥ 6 gradient directions, solve linear system:
```
ADC_i = g_i^T · D · g_i  for i = 1, ..., N
```

Expanded form:
```
ADC_i = D_xx * g_ix² + D_yy * g_iy² + D_zz * g_iz² 
        + 2*D_xy * g_ix * g_iy 
        + 2*D_xz * g_ix * g_iz 
        + 2*D_yz * g_iy * g_iz
```

This is linear in the 6 unknowns: [D_xx, D_yy, D_zz, D_xy, D_xz, D_yz]

**Step 5: Compute DTI Metrics**
```python
# Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eig(D)
λ1, λ2, λ3 = sorted(eigenvalues, reverse=True)

# Mean Diffusivity
MD = (λ1 + λ2 + λ3) / 3

# Fractional Anisotropy
FA = sqrt(3/2) * sqrt((λ1-MD)² + (λ2-MD)² + (λ3-MD)²) / sqrt(λ1² + λ2² + λ3²)

# Principal fiber direction
fiber_direction = eigenvector_of_λ1
```

---

## Part 4: Visualization Design

### Layout Option A: Multi-Panel Comparison
```
┌─────────────────────────────────────────────────────────┐
│  Case 1: Isotropic    │  Case 2: Z-fiber               │
│  ┌─────────────────┐  │  ┌─────────────────┐           │
│  │ Bloch Sphere    │  │  │ Bloch Sphere    │           │
│  │ (Gx, Gy, Gz)    │  │  │ (Gx, Gy, Gz)    │           │
│  └─────────────────┘  │  └─────────────────┘           │
├─────────────────────────────────────────────────────────┤
│  Case 3: X-fiber      │  Case 4: Tilted fiber          │
│  ┌─────────────────┐  │  ┌─────────────────┐           │
│  │ Bloch Sphere    │  │  │ Bloch Sphere    │           │
│  │ (Gx, Gy, Gz)    │  │  │ (Gx, Gy, Gz)    │           │
│  └─────────────────┘  │  └─────────────────┘           │
├─────────────────────────────────────────────────────────┤
│  Signal Decay Curves (all cases, all directions)       │
│  ┌───────────────────────────────────────────────────┐ │
│  │ M(t) vs time, color-coded by gradient direction  │ │
│  └───────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│  ADC Polar Plot (directional dependence)               │
│  ┌───────────────────────────────────────────────────┐ │
│  │ ADC vs gradient angle (shows anisotropy visually) │ │
│  └───────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│  DTI Metrics Table                                      │
│  ┌───────────────────────────────────────────────────┐ │
│  │ Case │  MD   │  FA   │ λ1   │ λ2   │ λ3   │ Dir  │ │
│  │  1   │ 0.050 │ 0.00  │ 0.05 │ 0.05 │ 0.05 │ N/A  │ │
│  │  2   │ 0.040 │ 0.82  │ 0.10 │ 0.01 │ 0.01 │ [001]│ │
│  │  3   │ 0.040 │ 0.82  │ 0.10 │ 0.01 │ 0.01 │ [100]│ │
│  │  4   │ 0.040 │ 0.82  │ 0.10 │ 0.01 │ 0.01 │ [101]│ │
│  └───────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

---

## Part 5: Implementation Steps

### Phase 1: Core DTI Modules
1. **Create `dti_scenarios.py`**
   - Define diffusion tensors for each case
   - Rotation utilities for tilted fibers
   - Gradient direction schemes

2. **Create `dti_analysis.py`**
   - b-value calculator
   - ADC extraction from echo decay
   - Tensor fitting (linear least squares)
   - DTI metrics computation (FA, MD, eigendecomposition)

3. **Create `dti_visualization.py`**
   - Polar plot for directional ADC
   - Multi-case comparison layouts
   - Metrics table rendering

### Phase 2: Validation
4. **Create `validate_dti.py`**
   - Run all 4 cases with 6+ gradient directions
   - Extract ADC for each direction
   - Fit diffusion tensor
   - Compare fitted vs. input tensors
   - Compute error metrics

### Phase 3: Visualization
5. **Create `dti_comparison.py`**
   - Generate multi-panel comparison figure
   - Animate through different gradient directions
   - Export comparison GIF

---

## Part 6: Expected Results

### Quantitative Validation

| Case | Input MD | Input FA | Fitted MD | Fitted FA | Error |
|------|----------|----------|-----------|-----------|-------|
| 1 (Iso) | 0.050 | 0.00 | 0.050±0.001 | 0.00±0.01 | <2% |
| 2 (Z) | 0.040 | 0.82 | 0.040±0.001 | 0.82±0.01 | <2% |
| 3 (X) | 0.040 | 0.82 | 0.040±0.001 | 0.82±0.01 | <2% |
| 4 (45°) | 0.040 | 0.82 | 0.040±0.001 | 0.82±0.01 | <2% |

### Qualitative Observations
- **Isotropic**: Circular ADC polar plot
- **Anisotropic**: Ellipsoidal ADC polar plot aligned with fiber
- **Tilted**: Ellipsoid rotated 45° in XZ plane

---

## Part 7: Open Questions

1. **Gradient scheme**:
   - Minimum 6 directions or extended (12, 30)?
   - **Recommendation**: Start with 6, add option for more

2. **b-values**:
   - Single b-value or multiple?
   - **Recommendation**: Single b-value first, multi-b later

3. **Animation**:
   - Animate through gradient directions or static comparison?
   - **Recommendation**: Both - static for paper, animated for presentation

4. **Noise**:
   - Add Rician noise for realism?
   - **Recommendation**: Optional parameter, default off for validation

5. **Visualization priority**:
   - Which plots are most important?
   - **Recommendation**:
     - Priority 1: Multi-case Bloch sphere comparison
     - Priority 2: Signal decay curves
     - Priority 3: ADC polar plots
     - Priority 4: Metrics table

---

## Part 8: Success Criteria

### Must Have
- ✅ Simulate 4 diffusion scenarios
- ✅ Apply gradients in 6+ directions
- ✅ Extract ADC from echo decay
- ✅ Fit diffusion tensor from ADC measurements
- ✅ Compute FA, MD, eigenvalues
- ✅ Validate: fitted tensor matches input (<2% error)

### Nice to Have
- 🎯 Polar plot visualization
- 🎯 Multi-panel comparison figure
- 🎯 Animated gradient direction sweep
- 🎯 Noise simulation
- 🎯 Extended gradient schemes (30+ directions)

---

## Next Actions

1. Review this plan with user
2. Get feedback on visualization priorities
3. Implement Phase 1 (Core DTI Modules)
4. Validate with Phase 2
5. Create visualizations in Phase 3

---

## References

- Stejskal & Tanner (1965): Spin diffusion measurements
- Basser et al. (1994): MR diffusion tensor spectroscopy and imaging
- Jones & Leemans (2011): Diffusion tensor imaging

---

**End of DevLog**

