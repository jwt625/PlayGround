"""
Analyze and remove glare from wafer image for better color/thickness estimation.

Glare detection strategies:
1. Saturation-based: Glare is white/gray = low saturation in HSV
2. Brightness-based: Glare regions are bright (high V in HSV)
3. RGB uniformity: Glare has R ≈ G ≈ B (white/gray)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from measure_viewing_angle import detect_ellipse_and_viewing_angle
from paths import IMAGE_PATH, OUTPUT_ANALYSIS


def create_ellipse_mask(img_shape, ellipse_params):
    """Create a binary mask for pixels inside the ellipse."""
    cx, cy, a, b, phi = ellipse_params
    height, width = img_shape[:2]
    y, x = np.ogrid[:height, :width]
    x_centered = x - cx
    y_centered = y - cy
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)
    x_rot = x_centered * cos_phi + y_centered * sin_phi
    y_rot = -x_centered * sin_phi + y_centered * cos_phi
    ellipse_eq = (x_rot / a) ** 2 + (y_rot / b) ** 2
    return (ellipse_eq <= 1).astype(np.uint8)


def detect_glare(img, saturation_threshold=0.3, brightness_threshold=200):
    """
    Detect glare regions using HSV color space.

    Glare characteristics:
    - Low saturation (S < threshold) - appears white/gray
    - High brightness (V > threshold) - bright regions

    Parameters:
        img: BGR image
        saturation_threshold: Max saturation (0-1) to be considered glare
        brightness_threshold: Min brightness (0-255) to be considered glare

    Returns:
        glare_mask: Binary mask where 1 = glare
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # H: 0-179, S: 0-255, V: 0-255 in OpenCV
    h, s, v = cv2.split(hsv)

    # Normalize saturation to 0-1
    s_normalized = s / 255.0

    # Glare = low saturation AND high brightness
    glare_mask = (s_normalized < saturation_threshold) & (v > brightness_threshold)

    # Also consider pixels where all RGB channels are high and similar (white)
    b_ch, g_ch, r_ch = cv2.split(img)
    rgb_min = np.minimum(np.minimum(r_ch, g_ch), b_ch)
    rgb_max = np.maximum(np.maximum(r_ch, g_ch), b_ch)
    rgb_range = rgb_max - rgb_min

    # White-ish: small range between RGB and high overall brightness
    white_mask = (rgb_range < 50) & (rgb_min > 180)

    # Combine masks
    combined_glare = glare_mask | white_mask

    # Clean up with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    combined_glare = cv2.morphologyEx(combined_glare.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    combined_glare = cv2.morphologyEx(combined_glare, cv2.MORPH_OPEN, kernel)

    return combined_glare, s_normalized, v


def analyze_glare(image_path, output_dir=None):
    """
    Analyze glare in the wafer image and show histograms with/without glare.

    Parameters:
        image_path: Path to the input image
        output_dir: Directory to save output files (default: current directory)
    """
    import os
    if output_dir is None:
        output_dir = '.'
    # Detect ellipse
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

    # Create masks
    print("Creating masks...")
    wafer_mask = create_ellipse_mask(img.shape, ellipse_params)
    glare_mask, saturation, brightness = detect_glare(img)

    # Combined mask: inside wafer AND not glare
    valid_mask = wafer_mask & (~glare_mask.astype(bool))
    glare_in_wafer = wafer_mask & glare_mask.astype(bool)

    # Statistics
    total_wafer_pixels = np.sum(wafer_mask)
    glare_pixels = np.sum(glare_in_wafer)
    valid_pixels = np.sum(valid_mask)
    glare_percentage = 100 * glare_pixels / total_wafer_pixels

    print(f"\n{'=' * 60}")
    print("Glare Analysis Results")
    print("=" * 60)
    print(f"Total wafer pixels: {total_wafer_pixels:,}")
    print(f"Glare pixels: {glare_pixels:,} ({glare_percentage:.1f}%)")
    print(f"Valid pixels: {valid_pixels:,} ({100-glare_percentage:.1f}%)")

    # Extract pixels
    wafer_pixels_all = img_rgb[wafer_mask == 1]
    valid_pixels_rgb = img_rgb[valid_mask]
    glare_pixels_rgb = img_rgb[glare_in_wafer]

    # Compute histograms
    bins = np.arange(257)

    # All wafer pixels
    r_hist_all, _ = np.histogram(wafer_pixels_all[:, 0], bins=bins)
    g_hist_all, _ = np.histogram(wafer_pixels_all[:, 1], bins=bins)
    b_hist_all, _ = np.histogram(wafer_pixels_all[:, 2], bins=bins)

    # Valid (no glare) pixels
    r_hist_valid, _ = np.histogram(valid_pixels_rgb[:, 0], bins=bins)
    g_hist_valid, _ = np.histogram(valid_pixels_rgb[:, 1], bins=bins)
    b_hist_valid, _ = np.histogram(valid_pixels_rgb[:, 2], bins=bins)

    # Statistics for valid pixels
    print(f"\nRGB Statistics (glare removed):")
    print(f"{'Channel':<10} {'Mean':>8} {'Std':>8}")
    print("-" * 30)
    for i, ch in enumerate(['R', 'G', 'B']):
        mean = valid_pixels_rgb[:, i].mean()
        std = valid_pixels_rgb[:, i].std()
        print(f"{ch:<10} {mean:>8.1f} {std:>8.1f}")

    # Create visualization
    fig = plt.figure(figsize=(14, 10))

    # Original image with glare overlay
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(img_rgb)
    ax1.set_title('Original Image')
    ax1.axis('off')

    # Glare mask overlay
    ax2 = fig.add_subplot(2, 3, 2)
    overlay = img_rgb.copy()
    overlay[glare_in_wafer] = [255, 0, 0]  # Red for glare
    ax2.imshow(overlay)
    ax2.set_title(f'Glare Regions (red)\n{glare_percentage:.1f}% of wafer')
    ax2.axis('off')

    # Valid regions
    ax3 = fig.add_subplot(2, 3, 3)
    valid_img = img_rgb.copy()
    valid_img[~valid_mask] = [50, 50, 50]  # Darken invalid regions
    ax3.imshow(valid_img)
    ax3.set_title(f'Valid Regions for Analysis\n{100-glare_percentage:.1f}% of wafer')
    ax3.axis('off')

    # Histogram comparison - Red
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(np.arange(256), r_hist_all, 'r-', alpha=0.5, label='All pixels')
    ax4.plot(np.arange(256), r_hist_valid, 'r-', linewidth=2, label='Glare removed')
    ax4.set_xlim(0, 255)
    ax4.set_xlabel('Pixel Value')
    ax4.set_ylabel('Count')
    ax4.set_title('Red Channel')
    ax4.legend()

    # Histogram comparison - Green
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(np.arange(256), g_hist_all, 'g-', alpha=0.5, label='All pixels')
    ax5.plot(np.arange(256), g_hist_valid, 'g-', linewidth=2, label='Glare removed')
    ax5.set_xlim(0, 255)
    ax5.set_xlabel('Pixel Value')
    ax5.set_ylabel('Count')
    ax5.set_title('Green Channel')
    ax5.legend()

    # Histogram comparison - Blue
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(np.arange(256), b_hist_all, 'b-', alpha=0.5, label='All pixels')
    ax6.plot(np.arange(256), b_hist_valid, 'b-', linewidth=2, label='Glare removed')
    ax6.set_xlim(0, 255)
    ax6.set_xlabel('Pixel Value')
    ax6.set_ylabel('Count')
    ax6.set_title('Blue Channel')
    ax6.legend()

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'glare_analysis.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nVisualization saved to: {out_path}")

    # Save the valid mask
    valid_mask_vis = (valid_mask * 255).astype(np.uint8)
    mask_path = os.path.join(output_dir, 'valid_region_mask.png')
    cv2.imwrite(mask_path, valid_mask_vis)
    print(f"Valid region mask saved to: {mask_path}")

    # Create a clean image with only valid regions
    clean_img = img_rgb.copy()
    clean_img[~valid_mask] = [0, 0, 0]
    clean_path = os.path.join(output_dir, 'wafer_no_glare.png')
    cv2.imwrite(clean_path, cv2.cvtColor(clean_img, cv2.COLOR_RGB2BGR))
    print(f"Glare-removed image saved to: {clean_path}")

    return valid_mask, glare_percentage


def main():
    image_path = str(IMAGE_PATH)

    print("=" * 60)
    print("Glare Detection and Removal Analysis")
    print("=" * 60)

    valid_mask, glare_pct = analyze_glare(image_path, output_dir=str(OUTPUT_ANALYSIS))

    print(f"\n{'=' * 60}")
    print("Recommendation")
    print("=" * 60)
    if glare_pct > 30:
        print("High glare coverage. Consider:")
        print("  - Using diffuse lighting")
        print("  - Cross-polarizer setup")
        print("  - Multiple angle captures")
    else:
        print(f"Glare affects {glare_pct:.1f}% of the wafer.")
        print("The remaining pixels should be suitable for thickness estimation.")
        print("Use the valid_region_mask.png to filter pixels for analysis.")


if __name__ == '__main__':
    main()
