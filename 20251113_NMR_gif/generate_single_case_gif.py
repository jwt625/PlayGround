"""
Generate GIF for a single DTI case

Creates animated visualization showing diffusion with a specific gradient direction
"""

import numpy as np
from nmr_physics import PulseSequence
from diffusion import DiffusionSimulator
from nmr_simulator import HahnEchoSimulator
from nmr_animation import HahnEchoAnimator
from dti_scenarios import get_standard_scenarios


def generate_case_gif(case_name='z_fiber', gradient_direction='z'):
    """
    Generate GIF for a specific DTI case
    
    Parameters:
    -----------
    case_name : str
        'isotropic', 'z_fiber', 'x_fiber', or 'tilted_fiber'
    gradient_direction : str or array
        Gradient direction ('x', 'y', 'z' or 3D vector)
    """
    print("=" * 70)
    print(f"Generating GIF for DTI Case: {case_name}")
    print(f"Gradient direction: {gradient_direction}")
    print("=" * 70)
    
    # Get scenario
    scenarios = get_standard_scenarios()
    scenario = scenarios[case_name]
    print(f"\n{scenario}")
    
    # Simulation parameters
    n_spins = 60
    n_frames = 150
    pulse_90_duration = 5
    pulse_180_duration = 6
    tau = 50
    extra_time = 30
    gradient_strength = 0.01
    
    # Create pulse sequence
    print("\n[1/4] Creating pulse sequence...")
    pulse_sequence = PulseSequence(
        pulse_90_duration=pulse_90_duration,
        pulse_180_duration=pulse_180_duration,
        tau=tau,
        extra_time=extra_time
    )
    print(f"  Total time: {pulse_sequence.t_total}")
    print(f"  Echo time: {pulse_sequence.t4_end}")
    
    # Setup time array
    t_total = pulse_sequence.t_total
    t = np.linspace(0, t_total, n_frames)
    dt = t[1] - t[0]
    
    # Create diffusion simulator
    print(f"\n[2/4] Creating diffusion simulator...")
    print(f"  Diffusion tensor:")
    print(f"  {scenario.D}")
    
    diffusion_sim = DiffusionSimulator(
        n_spins=n_spins,
        n_frames=n_frames,
        dt=dt,
        diffusion_coefficient=0.05,  # Will be overridden by tensor
        gradient_strength=gradient_strength,
        seed=42
    )
    
    # Generate 3D random walk with anisotropic tensor
    print(f"  Generating 3D random walk...")
    diffusion_sim.generate_random_walk_3d(diffusion_tensor=scenario.D)
    diffusion_sim.set_base_frequencies(frequency_spread=0.25)
    diffusion_sim.compute_frequencies_with_gradient(gradient_direction=gradient_direction)
    
    print(f"  Position range:")
    print(f"    X: [{diffusion_sim.positions[:,:,0].min():.2f}, {diffusion_sim.positions[:,:,0].max():.2f}]")
    print(f"    Y: [{diffusion_sim.positions[:,:,1].min():.2f}, {diffusion_sim.positions[:,:,1].max():.2f}]")
    print(f"    Z: [{diffusion_sim.positions[:,:,2].min():.2f}, {diffusion_sim.positions[:,:,2].max():.2f}]")
    
    # Create Hahn Echo simulator
    print(f"\n[3/4] Creating Hahn Echo simulator...")
    simulator = HahnEchoSimulator(
        n_spins=n_spins,
        pulse_sequence=pulse_sequence,
        diffusion_sim=diffusion_sim
    )
    simulator.setup_time_array(n_frames)
    
    # Create animation
    print(f"\n[4/4] Creating and saving animation...")
    animator = HahnEchoAnimator(simulator, figsize=(12, 8), dpi=80)
    
    # Generate filename
    grad_str = gradient_direction if isinstance(gradient_direction, str) else 'custom'
    filename = f'dti_{case_name}_grad{grad_str}.gif'
    
    animator.save_animation(filename, fps=15)
    
    print("\n" + "=" * 70)
    print(f"✓ GIF generated: {filename}")
    print("=" * 70)
    
    return filename


if __name__ == "__main__":
    import sys
    
    # Default: Z-fiber with Z-gradient (parallel - high diffusion)
    case_name = sys.argv[1] if len(sys.argv) > 1 else 'z_fiber'
    gradient_dir = sys.argv[2] if len(sys.argv) > 2 else 'z'
    
    print(f"\nUsage: python3 {sys.argv[0]} [case_name] [gradient_direction]")
    print(f"  case_name: isotropic, z_fiber, x_fiber, tilted_fiber")
    print(f"  gradient_direction: x, y, z")
    print(f"\nRunning with: case={case_name}, gradient={gradient_dir}\n")
    
    filename = generate_case_gif(case_name, gradient_dir)

