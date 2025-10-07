#!/usr/bin/env python3
"""
Bandpass filter transmission using coupled mode theory and input-output theory.

This script calculates the transmission of a series of coupled resonators
where input couples to the first mode, modes are coupled sequentially,
and the last mode couples to the output.

For N resonators: Input -> Mode1 <-> Mode2 <-> ... <-> ModeN -> Output

The coupled mode equations are:
da_i/dt = (iω_i - γ_i/2) a_i - ig Σ(adjacent modes) - √(γ_ext) s_in (for input mode)

Where:
- a_i: amplitude of mode i
- ω_i: resonance frequency of mode i
- γ_i: intrinsic loss rate (related to Q factor)
- g: coupling rate between adjacent modes
- γ_ext: external coupling rate
- s_in: input field amplitude
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

def coupled_mode_transmission(omega, N, Q=1000, g_norm=0.1, gamma_ext_norm=0.2):
    """
    Calculate transmission through N coupled resonators using coupled mode theory.

    Parameters:
    omega: frequency array (normalized by linewidth)
    N: number of resonators
    Q: quality factor of each resonator
    g_norm: inter-resonator coupling rate (normalized by linewidth)
    gamma_ext_norm: external coupling rate (normalized by linewidth)

    Returns:
    transmission: |t|^2 transmission coefficient
    reflection: |r|^2 reflection coefficient (S11)
    phase: unwrapped phase of transmission coefficient (radians)
    """
    # Convert Q to loss rate (normalized by linewidth)
    gamma_i = 2.0 / Q  # intrinsic loss rate normalized by linewidth

    # External coupling rates
    gamma_in = gamma_ext_norm   # input coupling
    gamma_out = gamma_ext_norm  # output coupling

    # Inter-resonator coupling
    g = g_norm

    transmission = np.zeros_like(omega)
    reflection = np.zeros_like(omega)
    t_complex = np.zeros_like(omega, dtype=complex)

    for idx, w in enumerate(omega):
        # Build the system matrix for steady state: M * a = b
        # where a = [a1, a2, ..., aN] and b is the driving vector

        M = np.zeros((N, N), dtype=complex)
        b = np.zeros(N, dtype=complex)

        # Fill the matrix
        for i in range(N):
            # Diagonal terms: (iω - iω_res - γ_i/2 - γ_ext/2)
            # Assuming all resonators are on resonance (ω_res = 0 in normalized units)
            M[i, i] = 1j * w - gamma_i/2

            # Add external coupling losses
            if i == 0:  # First resonator (input coupling)
                M[i, i] -= gamma_in/2
                b[i] = -1j * np.sqrt(gamma_in)  # Input drive
            if i == N-1:  # Last resonator (output coupling)
                M[i, i] -= gamma_out/2

            # Off-diagonal terms: inter-resonator coupling
            if i < N-1:  # Coupling to next resonator
                M[i, i+1] = -1j * g
                M[i+1, i] = -1j * g

        # Solve for mode amplitudes
        try:
            a = solve(M, b)

            # Calculate transmission using input-output theory
            # t = s_out = -√(γ_out) * a_N (for last resonator)
            t = -1j * np.sqrt(gamma_out) * a[N-1]

            # Calculate reflection using input-output theory (corrected)
            # r = 1 + i√(γ_in) * a_1 (for terminated input port)
            r = 1 + 1j * np.sqrt(gamma_in) * a[0]

            transmission[idx] = np.abs(t)**2
            reflection[idx] = np.abs(r)**2
            t_complex[idx] = t

        except np.linalg.LinAlgError:
            transmission[idx] = 0
            reflection[idx] = 0
            t_complex[idx] = 0

    # Calculate unwrapped phase
    phase = np.unwrap(np.angle(t_complex))

    return transmission, reflection, phase

def plot_coupled_mode_filters():
    """Plot transmission for different numbers of coupled resonators."""
    
    # Frequency range: ±10 linewidths
    omega = np.linspace(-3, 3, 1000)
    
    # Filter parameters
    Q = 1000
    g_norm = 0.2      # Inter-resonator coupling (normalized by linewidth)
    gamma_ext_norm = 0.3  # External coupling (normalized by linewidth)
    
    # Number of resonators to compare
    N_values = [2, 3, 4, 5]
    colors = ['blue', 'red', 'green', 'purple']
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14))
    fig.suptitle(f'Coupled Mode Bandpass Filters\nQ = {Q}, g = {g_norm}γ, γ_ext = {gamma_ext_norm}γ',
                 fontsize=16, fontweight='bold')

    # Plot 1: Linear scale
    ax1.set_title('Transmission (Linear Scale)', fontsize=14)
    for i, N in enumerate(N_values):
        transmission, reflection, phase = coupled_mode_transmission(omega, N, Q, g_norm, gamma_ext_norm)
        ax1.plot(omega, transmission, color=colors[i], linewidth=2, label=f'N = {N}')

    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5, label='Resonance')
    ax1.set_xlabel('Normalized Frequency (ω/γ)', fontsize=14)
    ax1.set_ylabel('Transmission |t|²', fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    ax1.set_xlim([-3, 3])
    ax1.set_ylim([0, 1.1])

    # Plot 2: dB scale with S11
    ax2.set_title('Transmission & Reflection (dB Scale)', fontsize=14)
    for i, N in enumerate(N_values):
        transmission, reflection, phase = coupled_mode_transmission(omega, N, Q, g_norm, gamma_ext_norm)
        transmission_dB = 10 * np.log10(np.maximum(transmission, 1e-10))  # Avoid log(0)
        reflection_dB = 10 * np.log10(np.maximum(reflection, 1e-10))  # Avoid log(0)

        # Transmission (solid lines)
        ax2.plot(omega, transmission_dB, color=colors[i], linewidth=2, label=f'S21, N = {N}')
        # Reflection (dashed lines)
        ax2.plot(omega, reflection_dB, color=colors[i], linewidth=2, linestyle='--', label=f'S11, N = {N}')

    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5, label='Resonance')
    ax2.set_xlabel('Normalized Frequency (ω/γ)', fontsize=14)
    ax2.set_ylabel('S-Parameters (dB)', fontsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10, ncol=2)  # Smaller font and 2 columns for more entries
    ax2.set_xlim([-3, 3])
    ax2.set_ylim([-60, 5])

    # Plot 3: Phase response
    ax3.set_title('Phase Response (Unwrapped)', fontsize=14)
    for i, N in enumerate(N_values):
        transmission, reflection, phase = coupled_mode_transmission(omega, N, Q, g_norm, gamma_ext_norm)
        ax3.plot(omega, phase, color=colors[i], linewidth=2, label=f'N = {N}')

    ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5, label='Resonance')
    ax3.set_xlabel('Normalized Frequency (ω/γ)', fontsize=14)
    ax3.set_ylabel('Phase (radians)', fontsize=14)
    ax3.tick_params(axis='both', which='major', labelsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=12)
    ax3.set_xlim([-3, 3])
    
    plt.tight_layout()
    plt.savefig('coupled_mode_bandpass_filters.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_coupling_strength_comparison():
    """Plot effect of different coupling strengths for N=3 resonators."""
    
    omega = np.linspace(-10, 10, 1000)
    N = 3
    Q = 1000
    gamma_ext_norm = 0.3
    
    # Different coupling strengths
    g_values = [0.2, 0.5, 1.0, 2.0]
    colors = ['blue', 'red', 'green', 'orange']
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
    fig.suptitle(f'Effect of Coupling Strength (N = {N}, Q = {Q})',
                 fontsize=16, fontweight='bold')

    # Linear scale
    ax1.set_title('Transmission vs Coupling Strength (Linear Scale)', fontsize=14)
    for i, g_norm in enumerate(g_values):
        transmission, reflection, phase = coupled_mode_transmission(omega, N, Q, g_norm, gamma_ext_norm)
        ax1.plot(omega, transmission, color=colors[i], linewidth=2, label=f'g = {g_norm}γ')

    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Normalized Frequency (ω/γ)', fontsize=14)
    ax1.set_ylabel('Transmission |t|²', fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    ax1.set_xlim([-8, 8])
    ax1.set_ylim([0, 1.1])

    # dB scale with S11
    ax2.set_title('Transmission & Reflection vs Coupling Strength (dB Scale)', fontsize=14)
    for i, g_norm in enumerate(g_values):
        transmission, reflection, phase = coupled_mode_transmission(omega, N, Q, g_norm, gamma_ext_norm)
        transmission_dB = 10 * np.log10(np.maximum(transmission, 1e-10))
        reflection_dB = 10 * np.log10(np.maximum(reflection, 1e-10))

        # Transmission (solid lines)
        ax2.plot(omega, transmission_dB, color=colors[i], linewidth=2, label=f'S21, g = {g_norm}γ')
        # Reflection (dashed lines)
        ax2.plot(omega, reflection_dB, color=colors[i], linewidth=2, linestyle='--', label=f'S11, g = {g_norm}γ')

    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Normalized Frequency (ω/γ)', fontsize=14)
    ax2.set_ylabel('S-Parameters (dB)', fontsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10, ncol=2)  # Smaller font and 2 columns for more entries
    ax2.set_xlim([-8, 8])
    ax2.set_ylim([-60, 5])

    # Phase response
    ax3.set_title('Phase Response vs Coupling Strength (Unwrapped)', fontsize=14)
    for i, g_norm in enumerate(g_values):
        transmission, reflection, phase = coupled_mode_transmission(omega, N, Q, g_norm, gamma_ext_norm)
        ax3.plot(omega, phase, color=colors[i], linewidth=2, label=f'g = {g_norm}γ')

    ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Normalized Frequency (ω/γ)', fontsize=14)
    ax3.set_ylabel('Phase (radians)', fontsize=14)
    ax3.tick_params(axis='both', which='major', labelsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=12)
    ax3.set_xlim([-8, 8])
    
    plt.tight_layout()
    plt.savefig('coupling_strength_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_theory_summary():
    """Print summary of coupled mode theory for bandpass filters."""
    
    print("Coupled Mode Theory for Bandpass Filters")
    print("=" * 50)
    print("Configuration: Series coupled resonators")
    print("• Input waveguide couples to first resonator")
    print("• Adjacent resonators are coupled with rate g")
    print("• Last resonator couples to output waveguide")
    print()
    print("Key Parameters:")
    print("• Q: Quality factor (intrinsic losses)")
    print("• g: Inter-resonator coupling rate")
    print("• γ_ext: External coupling rate")
    print("• γ_i = ω_res/Q: Intrinsic loss rate")
    print()
    print("Effects of increasing N (number of resonators):")
    print("• Sharper filter response")
    print("• Higher out-of-band rejection")
    print("• More complex passband structure")
    print("• Potential for multiple transmission peaks")
    print()
    print("Effects of coupling strength g:")
    print("• Weak coupling: Individual resonator peaks")
    print("• Strong coupling: Broadened, merged response")
    print("• Critical coupling: Optimized transmission")
    print()

if __name__ == "__main__":
    print("Plotting Coupled Mode Bandpass Filters")
    print("Series configuration: Input -> Mode1 <-> Mode2 <-> ... <-> ModeN -> Output")
    print("=" * 80)
    
    # Print theory summary
    print_theory_summary()
    
    # Generate plots
    plot_coupled_mode_filters()
    plot_coupling_strength_comparison()
    
    print("Plots saved as:")
    print("• coupled_mode_bandpass_filters.png")
    print("• coupling_strength_comparison.png")
