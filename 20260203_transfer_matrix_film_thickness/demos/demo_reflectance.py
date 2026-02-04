"""
Demo script: Calculate and visualize reflectance spectra for BTO/Si stack.

This demonstrates the transfer matrix model for thin film interference.
Configured for 56° incident angle (unpolarized light).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import matplotlib.pyplot as plt
from transfer_matrix import Layer, reflectance_spectrum, calculate_reflectance
from materials import n_air, n_bto, n_si

# Configuration
INCIDENT_ANGLE_DEG = 56.0
INCIDENT_ANGLE_RAD = np.radians(INCIDENT_ANGLE_DEG)


def create_stack(bto_thickness_nm: float):
    """Create a BTO/Si stack (MBE-grown, no oxide layer).

    Args:
        bto_thickness_nm: BTO film thickness in nm

    Returns:
        List of Layer objects
    """
    layers = [
        Layer(n=n_air, d=np.inf, name="Air"),
        Layer(n=n_bto, d=bto_thickness_nm, name="BTO"),
        Layer(n=n_si, d=np.inf, name="Si substrate"),
    ]
    return layers


def plot_reflectance_vs_wavelength():
    """Plot reflectance spectrum for different BTO thicknesses at 56° incidence."""
    wavelengths = np.linspace(380, 780, 200)  # Visible range
    thicknesses = [50, 100, 150, 200, 250, 300]  # nm

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Reflectance vs wavelength for different thicknesses at 56°
    ax1 = axes[0]
    for d in thicknesses:
        layers = create_stack(d)
        R = reflectance_spectrum(layers, wavelengths, theta_inc=INCIDENT_ANGLE_RAD,
                                  polarization='unpolarized')
        ax1.plot(wavelengths, R, label=f'{d} nm')

    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Reflectance')
    ax1.set_title(f'BTO/Si Reflectance Spectrum (θ = {INCIDENT_ANGLE_DEG}°, unpolarized)')
    ax1.legend(title='BTO thickness')
    ax1.set_xlim(380, 780)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Compare normal incidence vs 56° incidence
    ax2 = axes[1]
    d_bto = 200  # nm
    layers = create_stack(d_bto)

    # Normal incidence
    R_normal = reflectance_spectrum(layers, wavelengths, theta_inc=0, polarization='unpolarized')
    ax2.plot(wavelengths, R_normal, 'b-', label='0° (normal)', linewidth=1.5)

    # 56° incidence - show s, p, and average
    R_s = reflectance_spectrum(layers, wavelengths, theta_inc=INCIDENT_ANGLE_RAD, polarization='s')
    R_p = reflectance_spectrum(layers, wavelengths, theta_inc=INCIDENT_ANGLE_RAD, polarization='p')
    R_avg = (R_s + R_p) / 2
    ax2.plot(wavelengths, R_s, 'r--', label=f'{INCIDENT_ANGLE_DEG}° s-pol', linewidth=1, alpha=0.7)
    ax2.plot(wavelengths, R_p, 'g--', label=f'{INCIDENT_ANGLE_DEG}° p-pol', linewidth=1, alpha=0.7)
    ax2.plot(wavelengths, R_avg, 'k-', label=f'{INCIDENT_ANGLE_DEG}° unpolarized', linewidth=2)

    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_ylabel('Reflectance')
    ax2.set_title(f'Normal vs {INCIDENT_ANGLE_DEG}° Incidence (BTO = {d_bto} nm)')
    ax2.legend()
    ax2.set_xlim(380, 780)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'reflectance_spectra_{int(INCIDENT_ANGLE_DEG)}deg.png', dpi=150)
    plt.close()
    print(f"Saved: reflectance_spectra_{int(INCIDENT_ANGLE_DEG)}deg.png")


def plot_reflectance_vs_thickness():
    """Plot reflectance at specific wavelengths vs thickness at 56° incidence."""
    thicknesses = np.linspace(0, 500, 250)  # nm
    wavelengths_of_interest = [450, 532, 633, 700]  # Blue, green, red, deep red
    colors = ['blue', 'green', 'red', 'darkred']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Reflectance vs thickness at specific wavelengths (at 56°)
    ax1 = axes[0]
    for wl, color in zip(wavelengths_of_interest, colors):
        R_values = []
        for d in thicknesses:
            layers = create_stack(d)
            R = calculate_reflectance(layers, wl, theta_inc=INCIDENT_ANGLE_RAD,
                                      polarization='unpolarized')
            R_values.append(R)
        ax1.plot(thicknesses, R_values, color=color, label=f'{wl} nm')

    ax1.set_xlabel('BTO Thickness (nm)')
    ax1.set_ylabel('Reflectance')
    ax1.set_title(f'Reflectance vs BTO Thickness (θ = {INCIDENT_ANGLE_DEG}°)')
    ax1.legend(title='Wavelength')
    ax1.grid(True, alpha=0.3)

    # Plot 2: s vs p polarization comparison at 56°
    ax2 = axes[1]
    d_fixed = 200  # nm
    angles = np.linspace(0, 80, 80)
    wl = 550  # nm (green)

    R_s_vals = []
    R_p_vals = []
    for angle in angles:
        layers = create_stack(d_fixed)
        theta = np.radians(angle)
        R_s = calculate_reflectance(layers, wl, theta_inc=theta, polarization='s')
        R_p = calculate_reflectance(layers, wl, theta_inc=theta, polarization='p')
        R_s_vals.append(R_s)
        R_p_vals.append(R_p)

    ax2.plot(angles, R_s_vals, 'b-', label='s-polarization', linewidth=2)
    ax2.plot(angles, R_p_vals, 'r-', label='p-polarization', linewidth=2)
    ax2.plot(angles, (np.array(R_s_vals) + np.array(R_p_vals))/2, 'k--',
             label='Unpolarized', linewidth=1.5)
    ax2.axvline(INCIDENT_ANGLE_DEG, color='orange', linestyle=':',
                label=f'Current angle ({INCIDENT_ANGLE_DEG}°)', linewidth=2)

    ax2.set_xlabel('Incidence Angle (degrees)')
    ax2.set_ylabel('Reflectance')
    ax2.set_title(f'Polarization Dependence (λ={wl}nm, d={d_fixed}nm)')
    ax2.legend()
    ax2.set_xlim(0, 80)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'reflectance_vs_thickness_{int(INCIDENT_ANGLE_DEG)}deg.png', dpi=150)
    plt.close()
    print(f"Saved: reflectance_vs_thickness_{int(INCIDENT_ANGLE_DEG)}deg.png")


if __name__ == "__main__":
    print("Transfer Matrix Thin Film Model - Demo")
    print("=" * 50)
    print("\nStack: Air / BTO / Si (MBE-grown, no oxide)")
    print(f"Incident angle: {INCIDENT_ANGLE_DEG}° (unpolarized)")
    print("\nGenerating reflectance plots...")

    plot_reflectance_vs_wavelength()
    plot_reflectance_vs_thickness()

    print("\nDone!")

