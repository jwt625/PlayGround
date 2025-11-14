"""
Test All DTI Cases

Test all 4 DTI scenarios with analytical method
"""

import numpy as np
from dti_scenarios import get_standard_scenarios, get_gradient_scheme
from dti_analysis import (compute_b_value, compute_analytical_echo_decay, 
                          fit_diffusion_tensor, compare_tensors)


def test_scenario(scenario_name, scenario, b_value, gradients, labels):
    """Test a single DTI scenario"""
    print("\n" + "=" * 70)
    print(f"Testing: {scenario.name}")
    print("=" * 70)
    print(f"Description: {scenario.description}")
    print(f"True MD: {scenario.MD:.6f}, True FA: {scenario.FA:.6f}")
    print(f"Principal direction: {scenario.principal_direction}")
    
    # Compute ADC for each gradient direction
    ADC_measured = []
    
    for gradient, label in zip(gradients, labels):
        decay_ratio, ADC = compute_analytical_echo_decay(b_value, gradient, scenario.D)
        ADC_measured.append(ADC)
        ADC_theory = scenario.get_ADC(gradient)
        print(f"  {label}: ADC={ADC:.6f} (theory={ADC_theory:.6f})")
    
    ADC_measured = np.array(ADC_measured)
    
    # Fit diffusion tensor
    D_fitted, residual = fit_diffusion_tensor(gradients, ADC_measured)
    
    # Compare
    comparison = compare_tensors(scenario.D, D_fitted)
    
    print(f"\nFitted tensor:")
    print(D_fitted)
    print(f"\nComparison:")
    print(f"  MD: true={comparison['MD_true']:.6f}, fitted={comparison['MD_fitted']:.6f}, error={comparison['MD_error']:.8f}")
    print(f"  FA: true={comparison['FA_true']:.6f}, fitted={comparison['FA_fitted']:.6f}, error={comparison['FA_error']:.8f}")
    print(f"  Relative Frobenius error: {comparison['relative_frobenius_error']*100:.6f}%")
    
    # Check success
    tolerance = 1e-6
    success = (comparison['max_element_error'] < tolerance and 
               comparison['MD_error'] < tolerance and 
               comparison['FA_error'] < tolerance)
    
    if success:
        print(f"✓ {scenario.name} PASSED")
    else:
        print(f"✗ {scenario.name} FAILED")
    
    return success, comparison


def main():
    """Test all scenarios"""
    print("=" * 70)
    print("DTI Test Suite - All Cases (Analytical Method)")
    print("=" * 70)
    
    # Parameters
    tau = 50
    gradient_strength = 0.01
    b_value = compute_b_value(gradient_strength, tau)
    
    print(f"\nParameters:")
    print(f"  tau: {tau}")
    print(f"  gradient_strength: {gradient_strength}")
    print(f"  b-value: {b_value:.6f}")
    
    # Get gradient scheme
    gradients, labels = get_gradient_scheme('6dir')
    print(f"\nGradient scheme: 6 directions")
    
    # Get all scenarios
    scenarios = get_standard_scenarios()
    
    # Test each scenario
    results = {}
    all_passed = True
    
    for scenario_name in ['isotropic', 'z_fiber', 'x_fiber', 'tilted_fiber']:
        scenario = scenarios[scenario_name]
        success, comparison = test_scenario(scenario_name, scenario, b_value, gradients, labels)
        results[scenario_name] = {'success': success, 'comparison': comparison}
        all_passed = all_passed and success
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for scenario_name, result in results.items():
        status = "✓ PASS" if result['success'] else "✗ FAIL"
        comp = result['comparison']
        print(f"{status} | {scenario_name:15s} | MD={comp['MD_fitted']:.6f} | FA={comp['FA_fitted']:.6f}")
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
    else:
        print("✗✗✗ SOME TESTS FAILED ✗✗✗")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    results = main()

