"""
Test Case 1: Isotropic Diffusion (Analytical Method)

Use analytical Stejskal-Tanner equation for accurate ADC measurement
"""

import numpy as np
from dti_scenarios import get_standard_scenarios, get_gradient_scheme
from dti_analysis import (compute_b_value, compute_analytical_echo_decay, 
                          fit_diffusion_tensor, compare_tensors)


def run_case1_analytical():
    """
    Run isotropic diffusion case with analytical method
    """
    print("=" * 70)
    print("DTI Test Case 1: Isotropic Diffusion (Analytical)")
    print("=" * 70)
    
    # Get scenario
    scenarios = get_standard_scenarios()
    scenario = scenarios['isotropic']
    print(f"\n{scenario}")
    
    # Simulation parameters
    tau = 50
    gradient_strength = 0.01
    
    # Compute b-value
    b_value = compute_b_value(gradient_strength, tau)
    print(f"b-value: {b_value:.6f}")
    
    # Get gradient scheme
    gradients, labels = get_gradient_scheme('6dir')
    print(f"\nGradient scheme: 6 directions")
    for i, (g, label) in enumerate(zip(gradients, labels)):
        print(f"  {label}: {g}")
    
    # Compute ADC for each gradient direction using analytical method
    print(f"\n[1/3] Computing ADC using analytical Stejskal-Tanner equation...")
    ADC_measured = []
    echo_decays = []
    
    for i, (gradient, label) in enumerate(zip(gradients, labels)):
        # Analytical echo decay
        decay_ratio, ADC = compute_analytical_echo_decay(b_value, gradient, scenario.D)
        ADC_measured.append(ADC)
        echo_decays.append(decay_ratio)
        
        # Theoretical ADC (should match exactly)
        ADC_theory = scenario.get_ADC(gradient)
        
        print(f"\n  {label}: {gradient}")
        print(f"    Decay ratio S/S0: {decay_ratio:.6f}")
        print(f"    ADC measured: {ADC:.6f}")
        print(f"    ADC theoretical: {ADC_theory:.6f}")
        print(f"    Error: {abs(ADC - ADC_theory):.10f}")
    
    ADC_measured = np.array(ADC_measured)
    
    # Fit diffusion tensor
    print(f"\n[2/3] Fitting diffusion tensor...")
    D_fitted, residual = fit_diffusion_tensor(gradients, ADC_measured)
    print(f"  Fitted tensor:")
    print(D_fitted)
    print(f"  Fitting residual: {residual:.10f}")
    
    # Compare with true tensor
    print(f"\n[3/3] Comparing fitted vs. true tensor...")
    comparison = compare_tensors(scenario.D, D_fitted)
    print(f"  True tensor:")
    print(scenario.D)
    print(f"\n  Comparison:")
    print(f"    Max element error: {comparison['max_element_error']:.10f}")
    print(f"    Mean element error: {comparison['mean_element_error']:.10f}")
    print(f"    Relative Frobenius error: {comparison['relative_frobenius_error']*100:.6f}%")
    print(f"    MD true: {comparison['MD_true']:.6f}, fitted: {comparison['MD_fitted']:.6f}")
    print(f"    MD error: {comparison['MD_error']:.10f}")
    print(f"    FA true: {comparison['FA_true']:.6f}, fitted: {comparison['FA_fitted']:.6f}")
    print(f"    FA error: {comparison['FA_error']:.10f}")
    
    # Check if errors are within tolerance
    tolerance = 1e-6
    success = (comparison['max_element_error'] < tolerance and 
               comparison['MD_error'] < tolerance and 
               comparison['FA_error'] < tolerance)
    
    print("\n" + "=" * 70)
    if success:
        print("✓ Case 1 PASSED - Analytical method works perfectly!")
    else:
        print("✗ Case 1 FAILED - Errors exceed tolerance")
    print("=" * 70)
    
    return {
        'scenario': scenario,
        'ADC_measured': ADC_measured,
        'echo_decays': echo_decays,
        'D_fitted': D_fitted,
        'comparison': comparison,
        'success': success
    }


if __name__ == "__main__":
    results = run_case1_analytical()

