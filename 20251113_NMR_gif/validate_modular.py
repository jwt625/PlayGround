"""
Validation Script

Reproduce the same GIF as hahn_echo_diffusion.py using the modular components.
This validates that the refactoring preserves the original functionality.
"""

import numpy as np
from nmr_physics import PulseSequence
from diffusion import DiffusionSimulator
from nmr_simulator import HahnEchoSimulator
from nmr_animation import HahnEchoAnimator


def main():
    """
    Reproduce hahn_echo_diffusion.py using modular components
    """
    print("=" * 60)
    print("Validation: Reproducing hahn_echo_diffusion.py with modules")
    print("=" * 60)
    
    # Parameters (matching original script)
    n_spins = 60
    n_frames = 150
    pulse_90_duration = 5
    pulse_180_duration = 6
    tau = 50
    extra_time = 30
    
    diffusion_coefficient = 0.05
    gradient_strength = 0.01
    
    print(f"\nParameters:")
    print(f"  n_spins: {n_spins}")
    print(f"  n_frames: {n_frames}")
    print(f"  pulse_90_duration: {pulse_90_duration}")
    print(f"  pulse_180_duration: {pulse_180_duration}")
    print(f"  tau: {tau}")
    print(f"  extra_time: {extra_time}")
    print(f"  diffusion_coefficient: {diffusion_coefficient}")
    print(f"  gradient_strength: {gradient_strength}")
    
    # Step 1: Create pulse sequence
    print("\n[1/6] Creating pulse sequence...")
    pulse_sequence = PulseSequence(
        pulse_90_duration=pulse_90_duration,
        pulse_180_duration=pulse_180_duration,
        tau=tau,
        extra_time=extra_time
    )
    print(f"  Total time: {pulse_sequence.t_total}")
    print(f"  Echo time: {pulse_sequence.t4_end}")
    
    # Step 2: Create diffusion simulator
    print("\n[2/6] Creating diffusion simulator...")
    t_total = pulse_sequence.t_total
    t = np.linspace(0, t_total, n_frames)
    dt = t[1] - t[0] if len(t) > 1 else 1
    
    diffusion_sim = DiffusionSimulator(
        n_spins=n_spins,
        n_frames=n_frames,
        dt=dt,
        diffusion_coefficient=diffusion_coefficient,
        gradient_strength=gradient_strength,
        seed=42
    )
    
    # Step 3: Generate diffusion trajectories
    print("\n[3/6] Generating diffusion trajectories...")
    positions = diffusion_sim.generate_random_walk_1d(initial_spread=0.1)
    print(f"  Position array shape: {positions.shape}")
    print(f"  Position range: [{positions.min():.3f}, {positions.max():.3f}]")
    
    # Step 4: Set base frequencies and compute gradient effects
    print("\n[4/6] Computing frequencies with gradient effects...")
    base_frequencies = diffusion_sim.set_base_frequencies(frequency_spread=0.25)
    frequencies_array = diffusion_sim.compute_frequencies_with_gradient(gradient_direction='z')
    print(f"  Base frequency range: [{base_frequencies.min():.3f}, {base_frequencies.max():.3f}]")
    print(f"  Frequency array shape: {frequencies_array.shape}")
    
    # Step 5: Create Hahn Echo simulator
    print("\n[5/6] Creating Hahn Echo simulator...")
    simulator = HahnEchoSimulator(
        n_spins=n_spins,
        pulse_sequence=pulse_sequence,
        diffusion_sim=diffusion_sim
    )
    simulator.setup_time_array(n_frames)
    print(f"  Time array shape: {simulator.t.shape}")
    print(f"  dt: {simulator.dt:.4f}")
    
    # Step 6: Create animation
    print("\n[6/6] Creating and saving animation...")
    animator = HahnEchoAnimator(simulator, figsize=(12, 8), dpi=80)
    
    # Generate filename matching original
    filename = f'hahn_echo_D{diffusion_coefficient:.3f}_G{gradient_strength:.3f}_modular.gif'
    
    animator.save_animation(filename, fps=15)
    
    print("\n" + "=" * 60)
    print("Validation complete!")
    print(f"Output file: {filename}")
    print("Compare this with the original hahn_echo_diffusion.gif")
    print("=" * 60)


if __name__ == "__main__":
    main()

