"""
Test Case 1: Isotropic Diffusion

Generate simulation for isotropic diffusion and verify ADC extraction
"""

import numpy as np
from nmr_physics import PulseSequence
from diffusion import DiffusionSimulator
from nmr_simulator import HahnEchoSimulator
from nmr_animation import HahnEchoAnimator
from dti_scenarios import get_standard_scenarios, get_gradient_scheme
from dti_analysis import compute_b_value, extract_ADC, fit_diffusion_tensor, compute_dti_metrics, compare_tensors


def run_case1_isotropic():
    """
    Run isotropic diffusion case
    """
    print("=" * 70)
    print("DTI Test Case 1: Isotropic Diffusion")
    print("=" * 70)
    
    # Get scenario
    scenarios = get_standard_scenarios()
    scenario = scenarios['isotropic']
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
    print("\n[1/6] Creating pulse sequence...")
    pulse_sequence = PulseSequence(
        pulse_90_duration=pulse_90_duration,
        pulse_180_duration=pulse_180_duration,
        tau=tau,
        extra_time=extra_time
    )
    
    # Compute b-value
    b_value = compute_b_value(gradient_strength, tau)
    print(f"  b-value: {b_value:.6f}")
    
    # Get gradient scheme
    gradients, labels = get_gradient_scheme('6dir')
    print(f"\n[2/6] Gradient scheme: 6 directions")
    for i, (g, label) in enumerate(zip(gradients, labels)):
        print(f"  {label}: {g}")
    
    # Setup time array
    t_total = pulse_sequence.t_total
    t = np.linspace(0, t_total, n_frames)
    dt = t[1] - t[0]
    
    # Reference case: no gradient (b=0)
    # For b=0, we still have diffusion but no gradient, so spins don't dephase from diffusion
    # The echo should be perfect (amplitude = 1.0) with only base frequency dephasing
    print(f"\n[3/6] Reference echo amplitude (b=0)...")
    # For b=0 reference, we use the fact that without gradient, echo refocuses perfectly
    # (ignoring T2 relaxation which we don't model)
    echo_amplitude_b0 = 1.0
    echo_frame = int(pulse_sequence.t4_end / dt)
    print(f"  Echo amplitude (b=0): {echo_amplitude_b0:.6f} (perfect refocusing)")
    print(f"  Echo frame: {echo_frame} (time = {pulse_sequence.t4_end:.1f})")

    # Run simulations for each gradient direction
    print(f"\n[4/6] Running simulations for each gradient direction...")
    ADC_measured = []
    ADC_theoretical = []
    
    for i, (gradient, label) in enumerate(zip(gradients, labels)):
        print(f"\n  Direction {i+1}/{len(gradients)}: {label} = {gradient}")
        
        # Create diffusion simulator with gradient
        diffusion_sim = DiffusionSimulator(
            n_spins=n_spins,
            n_frames=n_frames,
            dt=dt,
            diffusion_coefficient=0.05,  # Will be overridden by tensor
            gradient_strength=gradient_strength,
            seed=42 + i
        )
        
        # Generate 3D random walk with isotropic tensor
        diffusion_sim.generate_random_walk_3d(diffusion_tensor=scenario.D)
        diffusion_sim.set_base_frequencies(frequency_spread=0.25)
        diffusion_sim.compute_frequencies_with_gradient(gradient_direction=gradient)
        
        # Create simulator
        simulator = HahnEchoSimulator(n_spins, pulse_sequence, diffusion_sim)
        simulator.setup_time_array(n_frames)
        
        # Get echo amplitude
        x, y, z, _ = simulator.evolve_spins(echo_frame, with_diffusion=True)
        echo_amplitude = np.sqrt(np.sum(x)**2 + np.sum(y)**2) / n_spins
        
        # Extract ADC
        ADC = extract_ADC(echo_amplitude_b0, echo_amplitude, b_value)
        ADC_measured.append(ADC)
        
        # Theoretical ADC
        ADC_theory = scenario.get_ADC(gradient)
        ADC_theoretical.append(ADC_theory)
        
        print(f"    Echo amplitude: {echo_amplitude:.6f}")
        print(f"    ADC measured: {ADC:.6f}")
        print(f"    ADC theoretical: {ADC_theory:.6f}")
        print(f"    Error: {abs(ADC - ADC_theory):.6f} ({abs(ADC - ADC_theory)/ADC_theory*100:.2f}%)")
    
    ADC_measured = np.array(ADC_measured)
    ADC_theoretical = np.array(ADC_theoretical)
    
    # Fit diffusion tensor
    print(f"\n[5/6] Fitting diffusion tensor...")
    D_fitted, residual = fit_diffusion_tensor(gradients, ADC_measured)
    print(f"  Fitted tensor:")
    print(D_fitted)
    print(f"  Fitting residual: {residual:.6f}")
    
    # Compare with true tensor
    print(f"\n[6/6] Comparing fitted vs. true tensor...")
    comparison = compare_tensors(scenario.D, D_fitted)
    print(f"  True tensor:")
    print(scenario.D)
    print(f"\n  Comparison:")
    print(f"    Max element error: {comparison['max_element_error']:.6f}")
    print(f"    Mean element error: {comparison['mean_element_error']:.6f}")
    print(f"    Relative Frobenius error: {comparison['relative_frobenius_error']*100:.2f}%")
    print(f"    MD true: {comparison['MD_true']:.6f}, fitted: {comparison['MD_fitted']:.6f}, error: {comparison['MD_error']:.6f}")
    print(f"    FA true: {comparison['FA_true']:.6f}, fitted: {comparison['FA_fitted']:.6f}, error: {comparison['FA_error']:.6f}")
    
    print("\n" + "=" * 70)
    print("Case 1 Complete!")
    print("=" * 70)
    
    return {
        'scenario': scenario,
        'ADC_measured': ADC_measured,
        'ADC_theoretical': ADC_theoretical,
        'D_fitted': D_fitted,
        'comparison': comparison
    }


if __name__ == "__main__":
    results = run_case1_isotropic()

