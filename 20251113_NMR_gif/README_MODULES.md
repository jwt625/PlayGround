# NMR Simulation Modular Framework

This directory contains a modular framework for NMR simulations, refactored from the original `hahn_echo_diffusion.py` script.

## Module Structure

### 1. **nmr_physics.py** - Core Physics
Core physics functions for NMR simulations:
- `rotate_x()`, `rotate_y()`, `rotate_z()` - Rotation operations on Bloch sphere
- `integrate_phase()` - Phase integration with time-varying frequencies
- `PulseSequence` class - Defines pulse sequence timing and parameters

### 2. **diffusion.py** - Diffusion Simulation
Handles molecular diffusion and gradient effects:
- `DiffusionSimulator` class:
  - `generate_random_walk_1d()` - 1D Brownian motion (isotropic)
  - `generate_random_walk_3d()` - 3D Brownian motion (supports anisotropic diffusion tensors)
  - `set_base_frequencies()` - Chemical shift and B0 inhomogeneity
  - `compute_frequencies_with_gradient()` - Gradient effects on spin frequencies

### 3. **nmr_visualization.py** - Visualization
Visualization functions for NMR data:
- `draw_bloch_sphere()` - Render Bloch sphere wireframe
- `draw_axes()` - Draw coordinate axes
- `plot_spins()` - Plot spin vectors with customizable origins
- `setup_bloch_axis()` - Configure 3D axis settings
- `plot_magnetization_history()` - Time-domain magnetization plots

### 4. **nmr_simulator.py** - Simulation Engine
Main simulation engine:
- `HahnEchoSimulator` class:
  - `setup_time_array()` - Initialize time grid
  - `evolve_spins()` - Calculate spin evolution through pulse sequence
  - `compute_net_magnetization()` - Calculate observable signal

### 5. **nmr_animation.py** - Animation
Animation creation and export:
- `HahnEchoAnimator` class:
  - `create_three_panel_figure()` - Multi-panel layout
  - `animate_frame()` - Frame-by-frame animation
  - `save_animation()` - Export to GIF

## Usage Example

```python
from nmr_physics import PulseSequence
from diffusion import DiffusionSimulator
from nmr_simulator import HahnEchoSimulator
from nmr_animation import HahnEchoAnimator

# 1. Define pulse sequence
pulse_seq = PulseSequence(
    pulse_90_duration=5,
    pulse_180_duration=6,
    tau=50,
    extra_time=30
)

# 2. Setup diffusion
diffusion_sim = DiffusionSimulator(
    n_spins=60,
    n_frames=150,
    dt=0.946,
    diffusion_coefficient=0.05,
    gradient_strength=0.01
)
diffusion_sim.generate_random_walk_1d()
diffusion_sim.set_base_frequencies()
diffusion_sim.compute_frequencies_with_gradient()

# 3. Create simulator
simulator = HahnEchoSimulator(
    n_spins=60,
    pulse_sequence=pulse_seq,
    diffusion_sim=diffusion_sim
)
simulator.setup_time_array(150)

# 4. Animate and save
animator = HahnEchoAnimator(simulator)
animator.save_animation('output.gif', fps=15)
```

## Validation

Run `validate_modular.py` to verify that the modular version reproduces the original results:

```bash
python3 validate_modular.py
```

This creates `hahn_echo_D0.050_G0.010_modular.gif` which should be equivalent to the original.

## Benefits of Modular Design

1. **Reusability**: Each module can be used independently
2. **Extensibility**: Easy to add new pulse sequences, diffusion models, or visualizations
3. **Maintainability**: Changes isolated to specific modules
4. **Testing**: Individual components can be unit tested
5. **DTI Ready**: Framework supports anisotropic diffusion tensors for DTI simulations

## Next Steps: DTI Implementation

The modular framework is ready for Diffusion Tensor Imaging (DTI) simulations:

1. Use `generate_random_walk_3d()` with custom diffusion tensors
2. Apply gradients in different directions (x, y, z)
3. Compare signal decay for isotropic vs. anisotropic diffusion
4. Compute DTI metrics (FA, MD, principal eigenvectors)

See the diffusion module's `generate_random_walk_3d()` method which already supports anisotropic diffusion tensors.

