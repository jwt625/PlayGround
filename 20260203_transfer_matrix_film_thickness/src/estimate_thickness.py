"""
Estimate thin film thickness from RGB color using transfer matrix model.

Approach:
1. Build a look-up table (LUT) of RGB values for thickness range 0-600nm
2. For each pixel, find the thickness whose predicted RGB is closest to observed

The forward model: thickness → reflectance_spectrum(λ) → RGB
We invert by minimizing |RGB_observed - RGB_predicted(thickness)|

Note: Multiple thicknesses can produce similar colors due to interference periodicity.
This gives the "best" match but may have ambiguity.
"""

import numpy as np
import json
import os
import cv2
from scipy.spatial import cKDTree
from transfer_matrix import Layer, reflectance_spectrum
from materials import n_air, n_bto, n_si, n_sio2
from color_conversion import spectrum_to_srgb
from ellipse_cache import get_ellipse_params, create_ellipse_mask
from paths import THICKNESS_LUT_PATH, THICKNESS_MAP_PATH, IMAGE_PATH, OUTPUT_THICKNESS


# Configuration
LUT_CACHE_FILE = str(THICKNESS_LUT_PATH)
WAVELENGTHS = np.linspace(380, 780, 81)  # 5nm steps, visible range


def create_bto_si_stack(bto_thickness_nm: float):
    """Create Air/BTO/Si stack (MBE-grown, no native oxide)."""
    return [
        Layer(n=n_air, d=np.inf, name="Air"),
        Layer(n=n_bto, d=bto_thickness_nm, name="BTO"),
        Layer(n=n_si, d=np.inf, name="Si substrate"),
    ]


def create_sio2_si_stack(sio2_thickness_nm: float):
    """Create Air/SiO2/Si stack (thermal oxide on silicon)."""
    return [
        Layer(n=n_air, d=np.inf, name="Air"),
        Layer(n=n_sio2, d=sio2_thickness_nm, name="SiO2"),
        Layer(n=n_si, d=np.inf, name="Si substrate"),
    ]


def thickness_to_rgb(thickness_nm: float, angle_rad: float,
                     stack_builder=create_sio2_si_stack) -> np.ndarray:
    """Calculate RGB for a given thickness at specified angle."""
    layers = stack_builder(thickness_nm)
    R_spectrum = reflectance_spectrum(layers, WAVELENGTHS,
                                      theta_inc=angle_rad,
                                      polarization='unpolarized')
    r, g, b = spectrum_to_srgb(WAVELENGTHS, R_spectrum, illuminant='D65')
    return np.array([r, g, b])


def build_thickness_lut(angle_deg: float, thickness_range=(0, 600),
                        thickness_step=1.0, stack_builder=create_sio2_si_stack,
                        force_rebuild=False):
    """
    Build look-up table mapping thickness to RGB.

    Args:
        angle_deg: Viewing angle in degrees
        thickness_range: (min, max) thickness in nm
        thickness_step: Step size in nm
        stack_builder: Function to create layer stack
        force_rebuild: Rebuild even if cache exists

    Returns:
        thicknesses: Array of thickness values
        rgb_lut: Array of RGB values, shape (N, 3)
    """
    stack_name = stack_builder.__name__
    cache_key = f"{angle_deg:.1f}_{thickness_range}_{thickness_step}_{stack_name}"

    if os.path.exists(LUT_CACHE_FILE) and not force_rebuild:
        data = np.load(LUT_CACHE_FILE, allow_pickle=True)
        if 'cache_key' in data and data['cache_key'] == cache_key:
            print(f"Loading cached LUT from {LUT_CACHE_FILE}")
            return data['thicknesses'], data['rgb_lut']

    print(f"Building thickness LUT for angle={angle_deg}°...")
    angle_rad = np.radians(angle_deg)

    thicknesses = np.arange(thickness_range[0], thickness_range[1] + thickness_step,
                            thickness_step)
    rgb_lut = np.zeros((len(thicknesses), 3))

    for i, d in enumerate(thicknesses):
        rgb_lut[i] = thickness_to_rgb(d, angle_rad, stack_builder)
        if i % 100 == 0:
            print(f"  {i}/{len(thicknesses)} ({d:.0f} nm)")

    # Cache the LUT
    np.savez(LUT_CACHE_FILE, thicknesses=thicknesses, rgb_lut=rgb_lut,
             cache_key=cache_key, angle_deg=angle_deg)
    print(f"Cached LUT to {LUT_CACHE_FILE}")

    return thicknesses, rgb_lut


def estimate_thickness_from_rgb(observed_rgb, thicknesses, rgb_lut):
    """
    Find the thickness that best matches the observed RGB.

    Uses nearest neighbor search in RGB space.

    Args:
        observed_rgb: (N, 3) array of observed RGB values (0-1 scale)
        thicknesses: Array of thickness values in LUT
        rgb_lut: (M, 3) array of RGB values in LUT

    Returns:
        estimated_thickness: Array of thickness values
        rgb_error: Distance in RGB space (quality metric)
    """
    # Build KD-tree for fast nearest neighbor search
    tree = cKDTree(rgb_lut)

    # Find nearest neighbor for each observed RGB
    distances, indices = tree.query(observed_rgb)

    estimated_thickness = thicknesses[indices]
    rgb_error = distances

    return estimated_thickness, rgb_error


def process_wafer_image(image_path, angle_deg, red_threshold=100,
                        stack_type='sio2'):
    """
    Process wafer image to estimate thickness map.

    Args:
        image_path: Path to wafer image
        angle_deg: Viewing angle
        red_threshold: Max red value to include (filter glare)
        stack_type: 'sio2' for SiO2/Si or 'bto' for BTO/Si

    Returns:
        thickness_map: 2D array of thickness estimates
        valid_mask: Boolean mask of valid pixels
    """
    # Select stack builder
    stack_builder = create_sio2_si_stack if stack_type == 'sio2' else create_bto_si_stack

    # Build or load LUT
    thicknesses, rgb_lut = build_thickness_lut(
        angle_deg, stack_builder=stack_builder)

    # Load image
    print("Loading image...")
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Load ellipse mask
    params = get_ellipse_params(image_path)
    wafer_mask = create_ellipse_mask(img.shape, params)

    # Apply red threshold for glare
    r_channel = img_rgb[:, :, 0]
    valid_mask = (wafer_mask == 1) & (r_channel < red_threshold)

    print(f"Valid pixels: {np.sum(valid_mask):,} / {np.sum(wafer_mask):,}")

    # Normalize RGB to 0-1 range
    img_normalized = img_rgb.astype(np.float32) / 255.0

    # Get valid pixels
    valid_pixels = img_normalized[valid_mask]  # Shape: (N, 3)

    print(f"Estimating thickness for {len(valid_pixels):,} pixels...")

    # Estimate thickness
    estimated_d, rgb_error = estimate_thickness_from_rgb(
        valid_pixels, thicknesses, rgb_lut)

    # Create thickness map
    thickness_map = np.full(img.shape[:2], np.nan, dtype=np.float32)
    thickness_map[valid_mask] = estimated_d

    # Create error map
    error_map = np.full(img.shape[:2], np.nan, dtype=np.float32)
    error_map[valid_mask] = rgb_error

    return thickness_map, error_map, valid_mask, thicknesses, rgb_lut


def visualize_results(thickness_map, error_map, valid_mask, output_prefix):
    """Create visualization of thickness estimation results."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Thickness map
    ax1 = axes[0, 0]
    im1 = ax1.imshow(thickness_map, cmap='viridis')
    ax1.set_title('Estimated Film Thickness (nm)')
    plt.colorbar(im1, ax=ax1, label='Thickness (nm)')
    ax1.axis('off')

    # Error map
    ax2 = axes[0, 1]
    im2 = ax2.imshow(error_map, cmap='hot', vmin=0, vmax=0.3)
    ax2.set_title('RGB Matching Error (lower = better fit)')
    plt.colorbar(im2, ax=ax2, label='RGB Distance')
    ax2.axis('off')

    # Histogram of thickness
    ax3 = axes[1, 0]
    valid_thickness = thickness_map[valid_mask]
    ax3.hist(valid_thickness, bins=100, edgecolor='black', alpha=0.7)
    ax3.set_xlabel('Thickness (nm)')
    ax3.set_ylabel('Pixel Count')
    ax3.set_title(f'Thickness Distribution\n'
                  f'Mean: {np.nanmean(valid_thickness):.1f} nm, '
                  f'Std: {np.nanstd(valid_thickness):.1f} nm')
    ax3.axvline(np.nanmean(valid_thickness), color='r', linestyle='--',
                label=f'Mean: {np.nanmean(valid_thickness):.1f} nm')
    ax3.legend()

    # Histogram of error
    ax4 = axes[1, 1]
    valid_error = error_map[valid_mask]
    ax4.hist(valid_error, bins=100, edgecolor='black', alpha=0.7)
    ax4.set_xlabel('RGB Distance')
    ax4.set_ylabel('Pixel Count')
    ax4.set_title(f'Fit Quality Distribution\n'
                  f'Mean Error: {np.nanmean(valid_error):.3f}')

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_prefix}_analysis.png")


def plot_lut_rgb_vs_thickness(thicknesses, rgb_lut, output_path):
    """Plot the RGB vs thickness relationship from the LUT."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(thicknesses, rgb_lut[:, 0], 'r-', label='R', linewidth=1.5)
    ax.plot(thicknesses, rgb_lut[:, 1], 'g-', label='G', linewidth=1.5)
    ax.plot(thicknesses, rgb_lut[:, 2], 'b-', label='B', linewidth=1.5)

    ax.set_xlabel('Film Thickness (nm)')
    ax.set_ylabel('sRGB Value (0-1)')
    ax.set_title('Predicted RGB vs Film Thickness (Transfer Matrix Model)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, thicknesses[-1])
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    image_path = str(IMAGE_PATH)

    # Load cached viewing angle
    params = get_ellipse_params(image_path)
    angle_deg = params['viewing_angle_deg']

    print("=" * 60)
    print("Thin Film Thickness Estimation from RGB")
    print("=" * 60)
    print(f"Viewing angle: {angle_deg:.1f}°")
    print(f"Stack: Air / BTO / Si")
    print(f"Glare filter: R < 100")

    # Process image
    thickness_map, error_map, valid_mask, thicknesses, rgb_lut = \
        process_wafer_image(image_path, angle_deg, red_threshold=100,
                            stack_type='bto')

    # Plot LUT
    plot_lut_rgb_vs_thickness(thicknesses, rgb_lut, str(OUTPUT_THICKNESS / 'rgb_vs_thickness_lut.png'))

    # Visualize results
    visualize_results(thickness_map, error_map, valid_mask, str(OUTPUT_THICKNESS / 'thickness'))

    # Save thickness map as numpy array
    np.save(str(THICKNESS_MAP_PATH), thickness_map)
    print(f"Saved: {THICKNESS_MAP_PATH}")

    # Print statistics
    valid_thickness = thickness_map[valid_mask]
    print(f"\n{'=' * 60}")
    print("Thickness Statistics")
    print("=" * 60)
    print(f"Mean thickness: {np.nanmean(valid_thickness):.1f} nm")
    print(f"Std deviation: {np.nanstd(valid_thickness):.1f} nm")
    print(f"Min: {np.nanmin(valid_thickness):.1f} nm")
    print(f"Max: {np.nanmax(valid_thickness):.1f} nm")


if __name__ == '__main__':
    main()
