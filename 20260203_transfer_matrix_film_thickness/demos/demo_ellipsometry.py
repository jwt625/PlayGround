"""
Demo: Ellipsometry calculations for BTO/Si stack.

Calculates Psi (amplitude ratio) and Delta (phase difference) for
s and p polarization reflections.

Configured for near-Brewster angle (~67° for BTO with n≈2.4).
Uses three angles: Brewster ± 5°.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import matplotlib.pyplot as plt
from transfer_matrix import (Layer, calculate_ellipsometry,
                             ellipsometry_spectrum, ellipsometry_vs_thickness)
from materials import n_air, n_bto, n_si

# Configuration - Near Brewster angle for Air/BTO interface
# Brewster angle = arctan(n_BTO/n_air) ≈ arctan(2.4) ≈ 67.4°
BREWSTER_ANGLE_DEG = 67.0
ANGLES_DEG = [BREWSTER_ANGLE_DEG - 5, BREWSTER_ANGLE_DEG, BREWSTER_ANGLE_DEG + 5]  # 62°, 67°, 72°
ANGLES_RAD = [np.radians(a) for a in ANGLES_DEG]

# Fixed BTO thickness for spectroscopic ellipsometry
BTO_THICKNESS_NM = 300.0

# Wavelength range (broader for ellipsometry)
WAVELENGTH_MIN = 250  # nm
WAVELENGTH_MAX = 900  # nm


def create_stack(bto_thickness_nm: float):
    """Create a BTO/Si stack (MBE-grown, no oxide layer)."""
    return [
        Layer(n=n_air, d=np.inf, name="Air"),
        Layer(n=n_bto, d=bto_thickness_nm, name="BTO"),
        Layer(n=n_si, d=np.inf, name="Si substrate"),
    ]


def plot_ellipsometry_vs_wavelength():
    """Plot Psi and Delta vs wavelength at three angles near Brewster."""
    wavelengths = np.linspace(WAVELENGTH_MIN, WAVELENGTH_MAX, 300)
    layers = create_stack(BTO_THICKNESS_NM)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['blue', 'green', 'red']
    linestyles = ['-', '-', '-']

    # Plot Psi vs wavelength for each angle
    ax1 = axes[0]
    for angle_deg, angle_rad, color in zip(ANGLES_DEG, ANGLES_RAD, colors):
        psi, delta = ellipsometry_spectrum(layers, wavelengths, angle_rad)
        ax1.plot(wavelengths, psi, color=color, label=f'{angle_deg}°')

    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Ψ (degrees)')
    ax1.set_title(f'Ellipsometry Ψ vs Wavelength (BTO = {BTO_THICKNESS_NM} nm)')
    ax1.legend(title='Incidence angle')
    ax1.set_xlim(WAVELENGTH_MIN, WAVELENGTH_MAX)
    ax1.set_ylim(0, 90)
    ax1.grid(True, alpha=0.3)

    # Plot Delta vs wavelength for each angle
    ax2 = axes[1]
    for angle_deg, angle_rad, color in zip(ANGLES_DEG, ANGLES_RAD, colors):
        psi, delta = ellipsometry_spectrum(layers, wavelengths, angle_rad)
        ax2.plot(wavelengths, delta, color=color, label=f'{angle_deg}°')

    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_ylabel('Δ (degrees)')
    ax2.set_title(f'Ellipsometry Δ vs Wavelength (BTO = {BTO_THICKNESS_NM} nm)')
    ax2.legend(title='Incidence angle')
    ax2.set_xlim(WAVELENGTH_MIN, WAVELENGTH_MAX)
    ax2.set_ylim(-180, 180)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    angles_str = '_'.join([str(int(a)) for a in ANGLES_DEG])
    plt.savefig(f'ellipsometry_vs_wavelength_{angles_str}deg.png', dpi=150)
    plt.close()
    print(f"Saved: ellipsometry_vs_wavelength_{angles_str}deg.png")


def plot_ellipsometry_vs_thickness():
    """Plot Psi and Delta vs BTO thickness at Brewster angle for different wavelengths."""
    thicknesses = np.linspace(0, 500, 250)
    wavelengths_of_interest = [400, 550, 750]  # UV-Vis-NIR
    colors = ['purple', 'green', 'darkred']

    # Use Brewster angle
    angle_rad = np.radians(BREWSTER_ANGLE_DEG)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot Psi vs thickness
    ax1 = axes[0]
    for wl, color in zip(wavelengths_of_interest, colors):
        psi, delta = ellipsometry_vs_thickness(
            wl, angle_rad, thicknesses, create_stack)
        ax1.plot(thicknesses, psi, color=color, label=f'{wl} nm')

    # Mark the 300nm thickness
    ax1.axvline(BTO_THICKNESS_NM, color='gray', linestyle='--', alpha=0.5, label=f'{BTO_THICKNESS_NM} nm')

    ax1.set_xlabel('BTO Thickness (nm)')
    ax1.set_ylabel('Ψ (degrees)')
    ax1.set_title(f'Ellipsometry Ψ vs Thickness (θ = {BREWSTER_ANGLE_DEG}°)')
    ax1.legend(title='Wavelength')
    ax1.set_ylim(0, 90)
    ax1.grid(True, alpha=0.3)

    # Plot Delta vs thickness
    ax2 = axes[1]
    for wl, color in zip(wavelengths_of_interest, colors):
        psi, delta = ellipsometry_vs_thickness(
            wl, angle_rad, thicknesses, create_stack)
        ax2.plot(thicknesses, delta, color=color, label=f'{wl} nm')

    ax2.axvline(BTO_THICKNESS_NM, color='gray', linestyle='--', alpha=0.5, label=f'{BTO_THICKNESS_NM} nm')

    ax2.set_xlabel('BTO Thickness (nm)')
    ax2.set_ylabel('Δ (degrees)')
    ax2.set_title(f'Ellipsometry Δ vs Thickness (θ = {BREWSTER_ANGLE_DEG}°)')
    ax2.legend(title='Wavelength')
    ax2.set_ylim(-180, 180)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'ellipsometry_vs_thickness_{int(BREWSTER_ANGLE_DEG)}deg.png', dpi=150)
    plt.close()
    print(f"Saved: ellipsometry_vs_thickness_{int(BREWSTER_ANGLE_DEG)}deg.png")


def plot_psi_delta_trajectory():
    """Plot Psi-Delta trajectory as thickness varies at three angles."""
    thicknesses = np.linspace(0, 500, 500)
    wavelength = 633  # HeNe laser wavelength
    colors = ['blue', 'green', 'red']

    fig, ax = plt.subplots(figsize=(10, 8))

    for angle_deg, angle_rad, color in zip(ANGLES_DEG, ANGLES_RAD, colors):
        psi, delta = ellipsometry_vs_thickness(
            wavelength, angle_rad, thicknesses, create_stack)

        # Plot trajectory
        ax.plot(delta, psi, color=color, alpha=0.7, linewidth=1.5, label=f'{angle_deg}°')

        # Mark specific thicknesses
        for d_mark in [0, 100, 200, 300, 400, 500]:
            idx = np.argmin(np.abs(thicknesses - d_mark))
            ax.plot(delta[idx], psi[idx], 'o', color=color, markersize=5)
            if angle_deg == BREWSTER_ANGLE_DEG:  # Only label for middle angle
                ax.annotate(f'{d_mark}', (delta[idx], psi[idx]),
                            fontsize=8, ha='left', va='bottom')

    ax.set_xlabel('Δ (degrees)')
    ax.set_ylabel('Ψ (degrees)')
    ax.set_title(f'Ψ-Δ Trajectory (λ={wavelength}nm, angles: {ANGLES_DEG}°)')
    ax.legend(title='Incidence angle')
    ax.set_xlim(-180, 180)
    ax.set_ylim(0, 90)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    angles_str = '_'.join([str(int(a)) for a in ANGLES_DEG])
    plt.savefig(f'psi_delta_trajectory_{angles_str}deg.png', dpi=150)
    plt.close()
    print(f"Saved: psi_delta_trajectory_{angles_str}deg.png")


def print_ellipsometry_table():
    """Print ellipsometry values at key wavelengths for all three angles."""
    wavelengths = [300, 400, 500, 600, 700, 800]
    layers = create_stack(BTO_THICKNESS_NM)

    print(f"\nEllipsometry values for BTO = {BTO_THICKNESS_NM} nm")
    print("=" * 70)
    print(f"{'λ (nm)':<10}", end="")
    for angle in ANGLES_DEG:
        print(f"{'Ψ@'+str(int(angle))+'°':<10} {'Δ@'+str(int(angle))+'°':<10}", end="")
    print()
    print("-" * 70)

    for wl in wavelengths:
        print(f"{wl:<10}", end="")
        for angle_rad in ANGLES_RAD:
            psi, delta = calculate_ellipsometry(layers, wl, angle_rad)
            print(f"{psi:<10.2f} {delta:<10.2f}", end="")
        print()


if __name__ == "__main__":
    print("Ellipsometry Calculator")
    print("=" * 50)
    print("\nStack: Air / BTO / Si (MBE-grown)")
    print(f"BTO thickness: {BTO_THICKNESS_NM} nm")
    print(f"Incidence angles: {ANGLES_DEG}° (near Brewster ~67°)")
    print(f"Wavelength range: {WAVELENGTH_MIN}-{WAVELENGTH_MAX} nm")
    print("\nGenerating ellipsometry plots...")

    plot_ellipsometry_vs_wavelength()
    plot_ellipsometry_vs_thickness()
    plot_psi_delta_trajectory()
    print_ellipsometry_table()

    print("\nDone!")

