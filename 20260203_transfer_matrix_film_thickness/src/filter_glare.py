"""
Filter glare using simple R < 10 threshold.
Show excluded regions with transparency overlay.
"""
import cv2
import numpy as np
from ellipse_cache import get_ellipse_params, create_ellipse_mask
from paths import IMAGE_PATH, OUTPUT_ANALYSIS


def filter_and_visualize(image_path, red_threshold=10):
    # Load cached ellipse params
    params = get_ellipse_params(image_path)

    # Load image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create ellipse mask
    wafer_mask = create_ellipse_mask(img.shape, params)

    # Get red channel
    r_channel = img_rgb[:, :, 0]

    # Valid pixels: inside wafer AND red < threshold
    valid_mask = (wafer_mask == 1) & (r_channel < red_threshold)

    # Excluded pixels: inside wafer but red >= threshold (glare)
    excluded_mask = (wafer_mask == 1) & (r_channel >= red_threshold)

    # Stats
    total_wafer = np.sum(wafer_mask)
    valid_count = np.sum(valid_mask)
    excluded_count = np.sum(excluded_mask)

    print(f"Total wafer pixels: {total_wafer:,}")
    print(f"Valid (R < {red_threshold}): {valid_count:,} ({100*valid_count/total_wafer:.1f}%)")
    print(f"Excluded (R >= {red_threshold}): {excluded_count:,} ({100*excluded_count/total_wafer:.1f}%)")

    # Create visualization: overlay excluded regions in red with 50% transparency
    output = img_rgb.copy()

    # Blend excluded regions with red
    alpha = 0.5
    red_overlay = np.array([255, 0, 0], dtype=np.uint8)
    output[excluded_mask] = (alpha * red_overlay + (1 - alpha) * output[excluded_mask]).astype(np.uint8)

    # Darken outside wafer
    output[wafer_mask == 0] = (output[wafer_mask == 0] * 0.3).astype(np.uint8)

    # Save
    overlay_path = str(OUTPUT_ANALYSIS / f'glare_mask_overlay_r{red_threshold}.png')
    cv2.imwrite(overlay_path, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
    print(f"\nSaved to: {overlay_path}")


if __name__ == '__main__':
    filter_and_visualize(str(IMAGE_PATH), red_threshold=100)
