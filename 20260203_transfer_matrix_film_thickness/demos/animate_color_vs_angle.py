"""
Animate perceived color vs BTO thickness as a function of incident angle.

Generates a GIF showing how the color-thickness relationship changes with angle.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from transfer_matrix import Layer, reflectance_spectrum
from materials import n_air, n_bto, n_si
from color_conversion import spectrum_to_srgb


# Configuration
WAVELENGTHS = np.linspace(380, 780, 81)  # 5nm steps
THICKNESSES = np.linspace(0, 600, 300)   # 0-600nm in 2nm steps
ANGLES = np.linspace(0, 80, 81)          # 0-80 degrees in 1 degree steps


def create_stack(bto_thickness_nm: float):
    """Create a BTO/Si stack (MBE-grown, no oxide layer)."""
    return [
        Layer(n=n_air, d=np.inf, name="Air"),
        Layer(n=n_bto, d=bto_thickness_nm, name="BTO"),
        Layer(n=n_si, d=np.inf, name="Si substrate"),
    ]


def thickness_to_color(thickness_nm: float, angle_rad: float,
                       illuminant: str = 'D65') -> tuple:
    """Calculate the perceived color for a given BTO thickness and angle."""
    layers = create_stack(thickness_nm)
    R = reflectance_spectrum(layers, WAVELENGTHS, theta_inc=angle_rad,
                             polarization='unpolarized')
    return spectrum_to_srgb(WAVELENGTHS, R, illuminant)


def compute_colors_for_angle(angle_deg: float) -> np.ndarray:
    """Compute colors for all thicknesses at a given angle."""
    angle_rad = np.radians(angle_deg)
    colors = []
    for d in THICKNESSES:
        rgb = thickness_to_color(d, angle_rad)
        colors.append(rgb)
    return np.array(colors)


def create_animation():
    """Create animated GIF of color vs thickness for varying angles."""
    print("Precomputing colors for all angles...")
    
    # Precompute all colors
    all_colors = {}
    for i, angle in enumerate(ANGLES):
        all_colors[angle] = compute_colors_for_angle(angle)
        if (i + 1) % 10 == 0:
            print(f"  Computed {i + 1}/{len(ANGLES)} angles")
    
    print("Creating animation...")

    # Create figure - match original subplot dimensions (14x10 figure, 2x2 grid = 7x5 per subplot)
    fig, ax = plt.subplots(figsize=(7, 5))

    # Initialize plot
    lines = []
    for d, color in zip(THICKNESSES, all_colors[ANGLES[0]]):
        line = ax.axvline(d, color=color, linewidth=2)
        lines.append(line)

    ax.set_xlim(0, 600)
    ax.set_xlabel('BTO Thickness (nm)')
    ax.set_yticks([])
    title = ax.set_title(f'Perceived Color vs BTO Thickness (θ=0°, D65)')
    
    plt.tight_layout()
    
    def update(frame):
        angle = ANGLES[frame]
        colors = all_colors[angle]
        
        # Update line colors
        for line, color in zip(lines, colors):
            line.set_color(color)
        
        title.set_text(f'Perceived Color vs BTO Thickness (θ={angle:.0f}°, D65)')
        return lines + [title]
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(ANGLES),
                         interval=100, blit=True)
    
    # Save as GIF
    output_file = 'color_vs_thickness_angle_animation.gif'
    print(f"Saving animation to {output_file}...")
    anim.save(output_file, writer=PillowWriter(fps=10))
    plt.close()
    
    print(f"Done! Saved: {output_file}")


if __name__ == "__main__":
    print("Color vs Thickness Animation Generator")
    print("=" * 50)
    print("\nStack: Air / BTO / Si (MBE-grown, no oxide)")
    print("Illuminant: D65 (standard daylight)")
    print(f"Angles: {ANGLES[0]:.0f}° to {ANGLES[-1]:.0f}°")
    print(f"Thicknesses: {THICKNESSES[0]:.0f} to {THICKNESSES[-1]:.0f} nm")
    print()
    
    create_animation()

