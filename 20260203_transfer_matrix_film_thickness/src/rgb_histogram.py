"""
Generate RGB histograms for pixels within the wafer region.
Uses the ellipse parameters from the viewing angle measurement.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from measure_viewing_angle import detect_ellipse_and_viewing_angle
from paths import IMAGE_PATH, OUTPUT_ANALYSIS


def create_ellipse_mask(img_shape, ellipse_params):
    """
    Create a binary mask for pixels inside the ellipse.

    Parameters:
        img_shape: (height, width) of the image
        ellipse_params: [cx, cy, a, b, phi] ellipse parameters

    Returns:
        Binary mask where 1 = inside ellipse
    """
    cx, cy, a, b, phi = ellipse_params
    height, width = img_shape[:2]

    # Create coordinate grids
    y, x = np.ogrid[:height, :width]

    # Transform to ellipse-centered coordinates
    x_centered = x - cx
    y_centered = y - cy

    # Rotate to align with ellipse axes
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)
    x_rot = x_centered * cos_phi + y_centered * sin_phi
    y_rot = -x_centered * sin_phi + y_centered * cos_phi

    # Check if inside ellipse: (x/a)² + (y/b)² <= 1
    ellipse_eq = (x_rot / a) ** 2 + (y_rot / b) ** 2
    mask = ellipse_eq <= 1

    return mask.astype(np.uint8)


def compute_rgb_histograms(image_path):
    """
    Compute RGB histograms for pixels inside the wafer.
    """
    # First, detect the ellipse
    print("Detecting wafer ellipse...")
    results = detect_ellipse_and_viewing_angle(image_path)

    cx, cy = results['center']
    a = results['semi_major_axis']
    b = results['semi_minor_axis']
    phi = np.radians(results['rotation_angle_deg'])
    ellipse_params = [cx, cy, a, b, phi]

    # Load image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create ellipse mask
    print("Creating ellipse mask...")
    mask = create_ellipse_mask(img.shape, ellipse_params)

    # Extract pixels inside the ellipse
    pixels_inside = img_rgb[mask == 1]
    n_pixels = len(pixels_inside)
    print(f"Pixels inside wafer: {n_pixels:,}")

    # Separate RGB channels
    r_channel = pixels_inside[:, 0]
    g_channel = pixels_inside[:, 1]
    b_channel = pixels_inside[:, 2]

    # Compute histograms
    bins = np.arange(257)  # 0 to 256 for bin edges
    r_hist, _ = np.histogram(r_channel, bins=bins)
    g_hist, _ = np.histogram(g_channel, bins=bins)
    b_hist, _ = np.histogram(b_channel, bins=bins)

    # Compute statistics
    stats = {
        'R': {'mean': r_channel.mean(), 'std': r_channel.std(), 'min': r_channel.min(), 'max': r_channel.max()},
        'G': {'mean': g_channel.mean(), 'std': g_channel.std(), 'min': g_channel.min(), 'max': g_channel.max()},
        'B': {'mean': b_channel.mean(), 'std': b_channel.std(), 'min': b_channel.min(), 'max': b_channel.max()},
    }

    return r_hist, g_hist, b_hist, stats, n_pixels, mask


def plot_rgb_histograms(r_hist, g_hist, b_hist, stats, n_pixels, output_path):
    """
    Create a figure with three RGB histogram subplots.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    fig.suptitle(f'RGB Histograms for Wafer Region\n({n_pixels:,} pixels)', fontsize=14)

    bin_centers = np.arange(256)

    # Red channel
    axes[0].bar(bin_centers, r_hist, color='red', alpha=0.7, width=1.0)
    axes[0].set_xlim(0, 255)
    axes[0].set_ylabel('Count')
    axes[0].set_title(f"Red Channel (mean={stats['R']['mean']:.1f}, std={stats['R']['std']:.1f})")
    axes[0].axvline(stats['R']['mean'], color='darkred', linestyle='--', linewidth=2, label=f"Mean: {stats['R']['mean']:.1f}")
    axes[0].legend()

    # Green channel
    axes[1].bar(bin_centers, g_hist, color='green', alpha=0.7, width=1.0)
    axes[1].set_xlim(0, 255)
    axes[1].set_ylabel('Count')
    axes[1].set_title(f"Green Channel (mean={stats['G']['mean']:.1f}, std={stats['G']['std']:.1f})")
    axes[1].axvline(stats['G']['mean'], color='darkgreen', linestyle='--', linewidth=2, label=f"Mean: {stats['G']['mean']:.1f}")
    axes[1].legend()

    # Blue channel
    axes[2].bar(bin_centers, b_hist, color='blue', alpha=0.7, width=1.0)
    axes[2].set_xlim(0, 255)
    axes[2].set_xlabel('Pixel Value')
    axes[2].set_ylabel('Count')
    axes[2].set_title(f"Blue Channel (mean={stats['B']['mean']:.1f}, std={stats['B']['std']:.1f})")
    axes[2].axvline(stats['B']['mean'], color='darkblue', linestyle='--', linewidth=2, label=f"Mean: {stats['B']['mean']:.1f}")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Histogram saved to: {output_path}")


def plot_combined_histogram(r_hist, g_hist, b_hist, stats, n_pixels, output_path):
    """
    Create a single plot with all three channels overlaid.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    bin_centers = np.arange(256)

    ax.plot(bin_centers, r_hist, color='red', alpha=0.8, linewidth=1.5, label=f"R (μ={stats['R']['mean']:.1f})")
    ax.plot(bin_centers, g_hist, color='green', alpha=0.8, linewidth=1.5, label=f"G (μ={stats['G']['mean']:.1f})")
    ax.plot(bin_centers, b_hist, color='blue', alpha=0.8, linewidth=1.5, label=f"B (μ={stats['B']['mean']:.1f})")

    ax.fill_between(bin_centers, r_hist, alpha=0.2, color='red')
    ax.fill_between(bin_centers, g_hist, alpha=0.2, color='green')
    ax.fill_between(bin_centers, b_hist, alpha=0.2, color='blue')

    ax.set_xlim(0, 255)
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Count')
    ax.set_title(f'RGB Histogram Overlay - Wafer Region ({n_pixels:,} pixels)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Combined histogram saved to: {output_path}")


def main():
    image_path = str(IMAGE_PATH)

    print("=" * 60)
    print("RGB Histogram Analysis for Wafer Region")
    print("=" * 60)

    # Compute histograms
    r_hist, g_hist, b_hist, stats, n_pixels, mask = compute_rgb_histograms(image_path)

    # Print statistics
    print(f"\n{'=' * 60}")
    print("RGB Statistics")
    print("=" * 60)
    print(f"\nTotal pixels in wafer: {n_pixels:,}")
    print(f"\n{'Channel':<10} {'Mean':>8} {'Std':>8} {'Min':>6} {'Max':>6}")
    print("-" * 40)
    for channel in ['R', 'G', 'B']:
        s = stats[channel]
        print(f"{channel:<10} {s['mean']:>8.1f} {s['std']:>8.1f} {s['min']:>6} {s['max']:>6}")

    # Generate plots
    print("\nGenerating histograms...")
    plot_rgb_histograms(r_hist, g_hist, b_hist, stats, n_pixels, str(OUTPUT_ANALYSIS / 'rgb_histogram.png'))
    plot_combined_histogram(r_hist, g_hist, b_hist, stats, n_pixels, str(OUTPUT_ANALYSIS / 'rgb_histogram_combined.png'))

    # Save mask visualization
    img = cv2.imread(image_path)
    mask_vis = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(OUTPUT_ANALYSIS / 'wafer_mask.png'), mask_vis)
    print(f"Wafer mask saved to: wafer_mask.png")


if __name__ == '__main__':
    main()
