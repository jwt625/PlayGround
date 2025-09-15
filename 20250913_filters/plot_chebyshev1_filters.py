#!/usr/bin/env python3
"""
Plot Chebyshev Type I low-pass filters of orders N = 2 to 5.

Chebyshev Type I filters are characterized by having equiripple behavior in the
passband and a monotonic response in the stopband. They provide a sharper
transition from passband to stopband compared to Butterworth filters, but at
the cost of passband ripple.

The magnitude response of a Chebyshev Type I filter is:
|H(jω)| = 1 / sqrt(1 + ε²T_N²(ω/ωc))

where:
- N is the filter order
- ε is the ripple factor (determines passband ripple)
- T_N is the Chebyshev polynomial of the first kind of order N
- ωc is the cutoff frequency
- ω is the angular frequency
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib.patches as mpatches

def chebyshev1_magnitude(omega, omega_c, N, rp):
    """
    Calculate the magnitude response of a Chebyshev Type I filter.
    
    Parameters:
    omega: Angular frequency array
    omega_c: Cutoff frequency
    N: Filter order
    rp: Passband ripple in dB
    
    Returns:
    Magnitude response
    """
    # Convert ripple from dB to linear scale
    epsilon = np.sqrt(10**(rp/10) - 1)
    
    # Normalized frequency
    omega_norm = omega / omega_c
    
    # Chebyshev polynomial of the first kind
    # For |x| <= 1: T_N(x) = cos(N * arccos(x))
    # For |x| > 1: T_N(x) = cosh(N * arccosh(x))
    def chebyshev_poly(x, n):
        x = np.asarray(x)
        result = np.zeros_like(x)
        
        # For |x| <= 1
        mask1 = np.abs(x) <= 1
        result[mask1] = np.cos(n * np.arccos(x[mask1]))
        
        # For |x| > 1
        mask2 = np.abs(x) > 1
        result[mask2] = np.cosh(n * np.arccosh(np.abs(x[mask2]))) * np.sign(x[mask2])**n
        
        return result
    
    T_N = chebyshev_poly(omega_norm, N)
    
    return 1 / np.sqrt(1 + epsilon**2 * T_N**2)

def plot_chebyshev1_filters():
    """Plot Chebyshev Type I filters of orders 2 to 5."""
    
    # Frequency range (normalized)
    omega = np.logspace(-2, 2, 1000)  # 0.01 to 100 rad/s
    omega_c = 1.0  # Normalized cutoff frequency
    rp = 1.0  # Passband ripple in dB
    
    # Filter orders to plot
    orders = [2, 3, 4, 5]
    
    # Colors for different orders
    colors = ['blue', 'red', 'green', 'purple']
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(f'Chebyshev Type I Low-Pass Filters (Rp = {rp} dB)\nOrders N = 2, 3, 4, 5', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Magnitude response (linear scale)
    ax1.set_title('Magnitude Response (Linear Scale)')
    for i, N in enumerate(orders):
        H_mag = chebyshev1_magnitude(omega, omega_c, N, rp)
        ax1.plot(omega, H_mag, color=colors[i], linewidth=2, label=f'N = {N}')
    
    # Add ripple bounds
    ripple_upper = 1.0
    ripple_lower = 1.0 / np.sqrt(1 + (10**(rp/10) - 1))
    ax1.axhline(y=ripple_upper, color='gray', linestyle=':', alpha=0.7, label='Ripple bounds')
    ax1.axhline(y=ripple_lower, color='gray', linestyle=':', alpha=0.7)
    ax1.axvline(x=omega_c, color='black', linestyle='--', alpha=0.7, label='Cutoff frequency')
    
    ax1.set_xlabel('Normalized Frequency (ω/ωc)')
    ax1.set_ylabel('|H(jω)|')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim([0.01, 100])
    ax1.set_ylim([0, 1.1])
    
    # Plot 2: Magnitude response (dB scale)
    ax2.set_title('Magnitude Response (dB Scale)')
    for i, N in enumerate(orders):
        H_mag = chebyshev1_magnitude(omega, omega_c, N, rp)
        H_dB = 20 * np.log10(H_mag)
        ax2.plot(omega, H_dB, color=colors[i], linewidth=2, label=f'N = {N}')
    
    # Add ripple bounds in dB
    ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.7, label='Ripple bounds')
    ax2.axhline(y=-rp, color='gray', linestyle=':', alpha=0.7)
    ax2.axvline(x=omega_c, color='black', linestyle='--', alpha=0.7, label='Cutoff frequency')
    
    ax2.set_xlabel('Normalized Frequency (ω/ωc)')
    ax2.set_ylabel('|H(jω)| (dB)')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim([0.01, 100])
    ax2.set_ylim([-80, 5])
    
    # Plot 3: Phase response
    ax3.set_title('Phase Response')
    for i, N in enumerate(orders):
        # Create Chebyshev Type I filter using scipy
        b, a = signal.cheby1(N, rp, omega_c, 'low', analog=True)
        w, h = signal.freqs(b, a, worN=omega)
        phase = np.angle(h, deg=True)
        ax3.plot(omega, phase, color=colors[i], linewidth=2, label=f'N = {N}')
    
    ax3.set_xlabel('Normalized Frequency (ω/ωc)')
    ax3.set_ylabel('Phase (degrees)')
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_xlim([0.01, 100])
    
    plt.tight_layout()
    plt.savefig('chebyshev1_filters_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_ripple_comparison():
    """Plot Chebyshev Type I filters with different ripple values."""
    
    omega = np.logspace(-2, 1, 1000)  # Focus on transition region
    omega_c = 1.0
    N = 4  # Fixed order for comparison
    
    # Different ripple values
    ripples = [0.5, 1.0, 2.0, 3.0]
    colors = ['blue', 'red', 'green', 'orange']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(f'Chebyshev Type I Filter (N = {N}) - Ripple Comparison', 
                 fontsize=16, fontweight='bold')
    
    # Linear scale
    ax1.set_title('Magnitude Response (Linear Scale)')
    for i, rp in enumerate(ripples):
        H_mag = chebyshev1_magnitude(omega, omega_c, N, rp)
        ax1.plot(omega, H_mag, color=colors[i], linewidth=2, label=f'Rp = {rp} dB')
        
        # Add ripple bounds for each
        ripple_lower = 1.0 / np.sqrt(1 + (10**(rp/10) - 1))
        ax1.axhline(y=ripple_lower, color=colors[i], linestyle=':', alpha=0.5)
    
    ax1.axvline(x=omega_c, color='black', linestyle='--', alpha=0.7, label='Cutoff frequency')
    ax1.set_xlabel('Normalized Frequency (ω/ωc)')
    ax1.set_ylabel('|H(jω)|')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim([0.01, 10])
    ax1.set_ylim([0, 1.1])
    
    # dB scale
    ax2.set_title('Magnitude Response (dB Scale)')
    for i, rp in enumerate(ripples):
        H_mag = chebyshev1_magnitude(omega, omega_c, N, rp)
        H_dB = 20 * np.log10(H_mag)
        ax2.plot(omega, H_dB, color=colors[i], linewidth=2, label=f'Rp = {rp} dB')
        
        # Add ripple bounds in dB
        ax2.axhline(y=-rp, color=colors[i], linestyle=':', alpha=0.5)
    
    ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.7)
    ax2.axvline(x=omega_c, color='black', linestyle='--', alpha=0.7, label='Cutoff frequency')
    ax2.set_xlabel('Normalized Frequency (ω/ωc)')
    ax2.set_ylabel('|H(jω)| (dB)')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim([0.01, 10])
    ax2.set_ylim([-60, 5])
    
    plt.tight_layout()
    plt.savefig('chebyshev1_ripple_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_filter_characteristics():
    """Print key characteristics of Chebyshev Type I filters."""
    
    print("Chebyshev Type I Filter Characteristics:")
    print("=" * 50)
    print("• Equiripple behavior in the passband")
    print("• Monotonic response in the stopband")
    print("• Sharper transition than Butterworth filters")
    print("• Passband ripple determined by ripple parameter (Rp)")
    print("• Cutoff frequency defined at ripple level, not -3dB")
    print("• All poles lie on an ellipse in the s-plane")
    print("• Steeper rolloff rate than Butterworth for same order")
    print("• Phase response is nonlinear")
    print()
    
    orders = [2, 3, 4, 5]
    print("Order-specific characteristics:")
    print("-" * 30)
    for N in orders:
        print(f"Order {N}: {N} poles, {N} passband ripple peaks")
    print()
    
    print("Trade-offs:")
    print("-" * 15)
    print("• Advantage: Sharper cutoff than Butterworth")
    print("• Disadvantage: Passband ripple")
    print("• Use when: Sharp transition is more important than flat passband")
    print()

if __name__ == "__main__":
    print("Plotting Chebyshev Type I Low-Pass Filters")
    print("Orders N = 2, 3, 4, 5")
    print("=" * 60)
    
    # Print characteristics
    print_filter_characteristics()
    
    # Generate plots
    plot_chebyshev1_filters()
    plot_ripple_comparison()
    
    print("Plots saved as:")
    print("• chebyshev1_filters_comparison.png")
    print("• chebyshev1_ripple_comparison.png")
