"""
Compare actual wafer colors vs predicted colors from thickness estimation.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from ellipse_cache import get_ellipse_params, create_ellipse_mask
from paths import THICKNESS_LUT_PATH, THICKNESS_MAP_PATH, IMAGE_PATH, OUTPUT_THICKNESS


def load_lut():
    """Load the cached LUT."""
    data = np.load(str(THICKNESS_LUT_PATH), allow_pickle=True)
    return data['thicknesses'], data['rgb_lut']


def create_comparison(image_path, red_threshold=100):
    """Create side-by-side comparison of actual vs predicted colors."""

    # Load data
    print("Loading data...")
    thickness_map = np.load(str(THICKNESS_MAP_PATH))
    thicknesses, rgb_lut = load_lut()

    # Load image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Load masks
    params = get_ellipse_params(image_path)
    wafer_mask = create_ellipse_mask(img.shape, params)
    r_channel = img_rgb[:, :, 0]
    valid_mask = (wafer_mask == 1) & (r_channel < red_threshold)

    # Create predicted image
    print("Creating predicted color image...")
    predicted_rgb = np.zeros_like(img_rgb, dtype=np.float32)

    # For each valid pixel, look up the predicted RGB based on thickness
    valid_thickness = thickness_map[valid_mask]

    # Find indices in LUT for each thickness
    # thickness_map values are already from the LUT, so we can directly index
    thickness_indices = np.round(valid_thickness - thicknesses[0]).astype(int)
    thickness_indices = np.clip(thickness_indices, 0, len(thicknesses) - 1)

    # Get predicted RGB values
    predicted_values = rgb_lut[thickness_indices]  # Shape: (N, 3)

    # Fill in the predicted image
    predicted_rgb[valid_mask] = predicted_values

    # Create comparison figure
    print("Creating comparison figure...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Row 1: Full images
    # Actual image (masked)
    actual_masked = img_rgb.copy().astype(np.float32) / 255.0
    actual_masked[~valid_mask] = 0.2  # Darken invalid regions

    axes[0, 0].imshow(actual_masked)
    axes[0, 0].set_title('Actual Wafer (R < 100 filter)')
    axes[0, 0].axis('off')

    # Predicted image
    predicted_display = predicted_rgb.copy()
    predicted_display[~valid_mask] = 0.2

    axes[0, 1].imshow(predicted_display)
    axes[0, 1].set_title('Predicted from Thickness')
    axes[0, 1].axis('off')

    # Difference image
    actual_norm = img_rgb.astype(np.float32) / 255.0
    diff = np.abs(actual_norm - predicted_rgb)
    diff_magnitude = np.sqrt(np.sum(diff**2, axis=2))  # Euclidean distance in RGB
    diff_display = np.zeros_like(img_rgb, dtype=np.float32)
    diff_display[valid_mask] = diff[valid_mask]

    # Show difference as heatmap
    diff_mag_display = np.zeros(img.shape[:2])
    diff_mag_display[valid_mask] = diff_magnitude[valid_mask]
    diff_mag_display[~valid_mask] = np.nan

    im = axes[0, 2].imshow(diff_mag_display, cmap='hot', vmin=0, vmax=0.5)
    axes[0, 2].set_title('RGB Difference (Euclidean distance)')
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)

    # Row 2: Zoomed regions or histograms
    # RGB channel comparison
    actual_valid = actual_norm[valid_mask]
    predicted_valid = predicted_rgb[valid_mask]

    # R channel
    axes[1, 0].hist(actual_valid[:, 0], bins=50, alpha=0.5, label='Actual', color='red')
    axes[1, 0].hist(predicted_valid[:, 0], bins=50, alpha=0.5, label='Predicted', color='darkred')
    axes[1, 0].set_xlabel('Red Value')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Red Channel Distribution')
    axes[1, 0].legend()

    # G channel
    axes[1, 1].hist(actual_valid[:, 1], bins=50, alpha=0.5, label='Actual', color='green')
    axes[1, 1].hist(predicted_valid[:, 1], bins=50, alpha=0.5, label='Predicted', color='darkgreen')
    axes[1, 1].set_xlabel('Green Value')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Green Channel Distribution')
    axes[1, 1].legend()

    # B channel
    axes[1, 2].hist(actual_valid[:, 2], bins=50, alpha=0.5, label='Actual', color='blue')
    axes[1, 2].hist(predicted_valid[:, 2], bins=50, alpha=0.5, label='Predicted', color='darkblue')
    axes[1, 2].set_xlabel('Blue Value')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].set_title('Blue Channel Distribution')
    axes[1, 2].legend()

    plt.tight_layout()
    plt.savefig(str(OUTPUT_THICKNESS / 'actual_vs_predicted_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: actual_vs_predicted_comparison.png")

    # Also create a simple side-by-side image
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 8))

    axes2[0].imshow(actual_masked)
    axes2[0].set_title('Actual', fontsize=16)
    axes2[0].axis('off')

    axes2[1].imshow(predicted_display)
    axes2[1].set_title('Predicted (BTO/Si model)', fontsize=16)
    axes2[1].axis('off')

    plt.tight_layout()
    plt.savefig(str(OUTPUT_THICKNESS / 'actual_vs_predicted_sidebyside.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: actual_vs_predicted_sidebyside.png")

    # Print statistics
    print(f"\n{'='*60}")
    print("Color Comparison Statistics")
    print("="*60)
    print(f"Mean RGB difference: {np.mean(diff_magnitude[valid_mask]):.3f}")
    print(f"Max RGB difference: {np.max(diff_magnitude[valid_mask]):.3f}")

    print(f"\nPer-channel mean absolute error:")
    print(f"  R: {np.mean(np.abs(actual_valid[:, 0] - predicted_valid[:, 0])):.3f}")
    print(f"  G: {np.mean(np.abs(actual_valid[:, 1] - predicted_valid[:, 1])):.3f}")
    print(f"  B: {np.mean(np.abs(actual_valid[:, 2] - predicted_valid[:, 2])):.3f}")


if __name__ == '__main__':
    create_comparison(str(IMAGE_PATH), red_threshold=100)
