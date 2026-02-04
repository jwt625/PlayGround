"""
Demo: Convert thin film reflectance spectra to visible colors.

Shows how BTO film thickness maps to perceived color.
Configured for 56° incident angle (unpolarized light).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from transfer_matrix import Layer, reflectance_spectrum
from materials import n_air, n_bto, n_si
from color_conversion import spectrum_to_srgb, spectrum_to_rgb_uint8

# Configuration
INCIDENT_ANGLE_DEG = 56.0
INCIDENT_ANGLE_RAD = np.radians(INCIDENT_ANGLE_DEG)


def create_stack(bto_thickness_nm: float):
    """Create a BTO/Si stack (MBE-grown, no oxide layer)."""
    return [
        Layer(n=n_air, d=np.inf, name="Air"),
        Layer(n=n_bto, d=bto_thickness_nm, name="BTO"),
        Layer(n=n_si, d=np.inf, name="Si substrate"),
    ]


def thickness_to_color(thickness_nm: float, wavelengths: np.ndarray,
                       illuminant: str = 'D65') -> tuple:
    """Calculate the perceived color for a given BTO thickness at 56° incidence."""
    layers = create_stack(thickness_nm)
    R = reflectance_spectrum(layers, wavelengths, theta_inc=INCIDENT_ANGLE_RAD,
                              polarization='unpolarized')
    return spectrum_to_srgb(wavelengths, R, illuminant)


def plot_color_vs_thickness():
    """Create a color chart showing thickness-to-color mapping."""
    wavelengths = np.linspace(380, 780, 81)  # 5nm steps
    thicknesses = np.linspace(0, 600, 300)  # 0-600nm in 2nm steps
    
    # Calculate colors for each thickness
    colors = []
    for d in thicknesses:
        rgb = thickness_to_color(d, wavelengths)
        colors.append(rgb)
    colors = np.array(colors)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Color bar
    ax1 = axes[0, 0]
    for i, (d, color) in enumerate(zip(thicknesses, colors)):
        ax1.axvline(d, color=color, linewidth=2)
    ax1.set_xlim(0, 600)
    ax1.set_xlabel('BTO Thickness (nm)')
    ax1.set_title(f'Perceived Color vs BTO Thickness (θ={INCIDENT_ANGLE_DEG}°, D65)')
    ax1.set_yticks([])
    
    # Plot 2: RGB channels vs thickness
    ax2 = axes[0, 1]
    ax2.plot(thicknesses, colors[:, 0], 'r-', label='R', linewidth=1.5)
    ax2.plot(thicknesses, colors[:, 1], 'g-', label='G', linewidth=1.5)
    ax2.plot(thicknesses, colors[:, 2], 'b-', label='B', linewidth=1.5)
    ax2.set_xlabel('BTO Thickness (nm)')
    ax2.set_ylabel('sRGB Value (0-1)')
    ax2.set_title('RGB Channel Values vs Thickness')
    ax2.legend()
    ax2.set_xlim(0, 600)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Color swatches at specific thicknesses
    ax3 = axes[1, 0]
    swatch_thicknesses = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550]
    n_swatches = len(swatch_thicknesses)
    
    for i, d in enumerate(swatch_thicknesses):
        rgb = thickness_to_color(d, wavelengths)
        rect = Rectangle((i, 0), 0.9, 1, facecolor=rgb, edgecolor='black')
        ax3.add_patch(rect)
        ax3.text(i + 0.45, -0.15, f'{d}', ha='center', fontsize=9)
    
    ax3.set_xlim(-0.2, n_swatches)
    ax3.set_ylim(-0.3, 1.2)
    ax3.set_aspect('equal')
    ax3.axis('off')
    ax3.set_title('Color Swatches at Specific Thicknesses (nm)')
    
    # Plot 4: Reflectance spectra for selected thicknesses
    ax4 = axes[1, 1]
    selected_thicknesses = [100, 200, 300, 400, 500]

    for d in selected_thicknesses:
        layers = create_stack(d)
        R = reflectance_spectrum(layers, wavelengths, theta_inc=INCIDENT_ANGLE_RAD,
                                  polarization='unpolarized')
        color = thickness_to_color(d, wavelengths)
        ax4.plot(wavelengths, R, color=color, label=f'{d} nm', linewidth=2)

    ax4.set_xlabel('Wavelength (nm)')
    ax4.set_ylabel('Reflectance')
    ax4.set_title(f'Reflectance Spectra at θ={INCIDENT_ANGLE_DEG}°')
    ax4.legend()
    ax4.set_xlim(380, 780)
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'color_vs_thickness_{int(INCIDENT_ANGLE_DEG)}deg.png', dpi=150)
    plt.close()
    print(f"Saved: color_vs_thickness_{int(INCIDENT_ANGLE_DEG)}deg.png")


def print_color_table():
    """Print a table of thickness to RGB values at 56° incidence."""
    wavelengths = np.linspace(380, 780, 81)

    print(f"\nBTO Thickness to RGB Color Mapping (θ={INCIDENT_ANGLE_DEG}°, D65 illuminant)")
    print("=" * 60)
    print(f"{'Thickness (nm)':<15} {'R':<6} {'G':<6} {'B':<6} {'Hex':<10}")
    print("-" * 60)

    for d in range(0, 601, 25):
        r, g, b = spectrum_to_rgb_uint8(
            wavelengths,
            reflectance_spectrum(create_stack(d), wavelengths,
                                 INCIDENT_ANGLE_RAD, 'unpolarized')
        )
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        print(f"{d:<15} {r:<6} {g:<6} {b:<6} {hex_color:<10}")


if __name__ == "__main__":
    print("Thin Film Color Calculator")
    print("=" * 50)
    print("\nStack: Air / BTO / Si (MBE-grown, no oxide)")
    print(f"Incident angle: {INCIDENT_ANGLE_DEG}° (unpolarized)")
    print("Illuminant: D65 (standard daylight)")
    print("\nGenerating color plots...")

    plot_color_vs_thickness()
    print_color_table()

    print("\nDone!")

