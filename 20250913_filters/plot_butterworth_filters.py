#!/usr/bin/env python3
"""
Plot maximally flat low-pass filters (Butterworth filters) of orders N = 2 to 5.

Butterworth filters are characterized by having a maximally flat magnitude response
in the passband, meaning the first 2N-1 derivatives of the magnitude response are
zero at DC (ω = 0).

The magnitude response of a Butterworth filter is:
|H(jω)| = 1 / sqrt(1 + (ω/ωc)^(2N))

where:
- N is the filter order
- ωc is the cutoff frequency (3dB point)
- ω is the angular frequency
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib.patches as mpatches

def butterworth_magnitude(omega, omega_c, N):
    """
    Calculate the magnitude response of a Butterworth filter.
    
    Parameters:
    omega: Angular frequency array
    omega_c: Cutoff frequency (3dB point)
    N: Filter order
    
    Returns:
    Magnitude response
    """
    return 1 / np.sqrt(1 + (omega / omega_c)**(2*N))

def plot_butterworth_filters():
    """Plot Butterworth filters of orders 2 to 5."""
    
    # Frequency range (normalized)
    omega = np.logspace(-2, 2, 1000)  # 0.01 to 100 rad/s
    omega_c = 1.0  # Normalized cutoff frequency
    
    # Filter orders to plot
    orders = [2, 3, 4, 5]
    
    # Colors for different orders
    colors = ['blue', 'red', 'green', 'purple']
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('Maximally Flat Low-Pass Filters (Butterworth)\nOrders N = 2, 3, 4, 5', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Magnitude response (linear scale)
    ax1.set_title('Magnitude Response (Linear Scale)')
    for i, N in enumerate(orders):
        H_mag = butterworth_magnitude(omega, omega_c, N)
        ax1.plot(omega, H_mag, color=colors[i], linewidth=2, label=f'N = {N}')
    
    ax1.axhline(y=1/np.sqrt(2), color='black', linestyle='--', alpha=0.7, label='3dB line')
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
        H_mag = butterworth_magnitude(omega, omega_c, N)
        H_dB = 20 * np.log10(H_mag)
        ax2.plot(omega, H_dB, color=colors[i], linewidth=2, label=f'N = {N}')
    
    ax2.axhline(y=-3, color='black', linestyle='--', alpha=0.7, label='-3dB line')
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
        # Create Butterworth filter using scipy
        b, a = signal.butter(N, omega_c, 'low', analog=True)
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
    plt.savefig('butterworth_filters_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_individual_filters():
    """Plot each filter order individually with more detail."""
    
    orders = [2, 3, 4, 5]
    omega = np.logspace(-2, 2, 1000)
    omega_c = 1.0
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Individual Butterworth Filter Responses', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    colors = ['blue', 'red', 'green', 'purple']
    
    for i, N in enumerate(orders):
        ax = axes[i]
        
        # Magnitude response
        H_mag = butterworth_magnitude(omega, omega_c, N)
        H_dB = 20 * np.log10(H_mag)
        
        ax.plot(omega, H_dB, color=colors[i], linewidth=2, label=f'N = {N}')
        ax.axhline(y=-3, color='black', linestyle='--', alpha=0.7, label='-3dB')
        ax.axvline(x=omega_c, color='black', linestyle='--', alpha=0.7, label='ωc')
        
        # Add rolloff rate annotation
        rolloff_rate = -20 * N  # dB/decade
        ax.text(0.02, -20, f'Rolloff: {rolloff_rate} dB/decade', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax.set_title(f'Order N = {N}')
        ax.set_xlabel('Normalized Frequency (ω/ωc)')
        ax.set_ylabel('Magnitude (dB)')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim([0.01, 100])
        ax.set_ylim([-100, 5])
    
    plt.tight_layout()
    plt.savefig('butterworth_individual_filters.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_pole_zero_diagram():
    """Plot pole-zero diagrams for Butterworth filters."""
    
    orders = [2, 3, 4, 5]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Pole-Zero Diagrams for Butterworth Filters', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for i, N in enumerate(orders):
        ax = axes[i]
        
        # Generate Butterworth filter
        b, a = signal.butter(N, 1, 'low', analog=True)
        
        # Find poles and zeros
        poles = np.roots(a)
        zeros = np.roots(b) if len(b) > 1 else []
        
        # Plot unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, label='Unit circle')
        
        # Plot poles
        ax.scatter(poles.real, poles.imag, marker='x', s=100, c='red', linewidth=3, label='Poles')
        
        # Plot zeros (if any)
        if len(zeros) > 0:
            ax.scatter(zeros.real, zeros.imag, marker='o', s=100, c='blue', 
                      facecolors='none', edgecolors='blue', linewidth=2, label='Zeros')
        
        ax.set_title(f'Order N = {N}')
        ax.set_xlabel('Real Part')
        ax.set_ylabel('Imaginary Part')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.legend()
        
        # Set axis limits
        max_val = max(1.5, np.max(np.abs(poles)) * 1.2)
        ax.set_xlim([-max_val, max_val])
        ax.set_ylim([-max_val, max_val])
    
    plt.tight_layout()
    plt.savefig('butterworth_pole_zero.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_filter_characteristics():
    """Print key characteristics of Butterworth filters."""
    
    print("Butterworth Filter Characteristics:")
    print("=" * 50)
    print("• Maximally flat magnitude response in passband")
    print("• No ripple in passband or stopband")
    print("• Monotonic magnitude response")
    print("• 3dB attenuation at cutoff frequency")
    print("• Rolloff rate: -20N dB/decade (where N is the order)")
    print("• All poles lie on a circle in the s-plane")
    print("• Phase response is not linear")
    print()
    
    orders = [2, 3, 4, 5]
    print("Order-specific characteristics:")
    print("-" * 30)
    for N in orders:
        rolloff = -20 * N
        print(f"Order {N}: {rolloff} dB/decade rolloff, {N} poles")
    print()

if __name__ == "__main__":
    print("Plotting Maximally Flat Low-Pass Filters (Butterworth)")
    print("Orders N = 2, 3, 4, 5")
    print("=" * 60)
    
    # Print characteristics
    print_filter_characteristics()
    
    # Generate plots
    plot_butterworth_filters()
    plot_individual_filters()
    plot_pole_zero_diagram()
    
    print("Plots saved as:")
    print("• butterworth_filters_comparison.png")
    print("• butterworth_individual_filters.png") 
    print("• butterworth_pole_zero.png")
