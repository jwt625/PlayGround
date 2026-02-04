"""
Cache ellipse detection results to avoid re-running expensive detection.
"""
import json
import os
import numpy as np
from measure_viewing_angle import detect_ellipse_and_viewing_angle
from paths import ELLIPSE_PARAMS_PATH


CACHE_FILE = str(ELLIPSE_PARAMS_PATH)


def get_ellipse_params(image_path, force_recompute=False):
    """Get ellipse parameters, using cache if available."""

    if os.path.exists(CACHE_FILE) and not force_recompute:
        print(f"Loading cached ellipse parameters from {CACHE_FILE}")
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)

    print("Computing ellipse parameters (this will be cached)...")
    results = detect_ellipse_and_viewing_angle(image_path)

    # Convert to serializable format
    cache_data = {
        'center_x': results['center'][0],
        'center_y': results['center'][1],
        'semi_major_axis': results['semi_major_axis'],
        'semi_minor_axis': results['semi_minor_axis'],
        'rotation_angle_deg': results['rotation_angle_deg'],
        'viewing_angle_deg': results['viewing_angle_deg'],
    }

    with open(CACHE_FILE, 'w') as f:
        json.dump(cache_data, f, indent=2)
    print(f"Cached to {CACHE_FILE}")

    return cache_data


def create_ellipse_mask(img_shape, params):
    """Create ellipse mask from cached parameters."""
    cx = params['center_x']
    cy = params['center_y']
    a = params['semi_major_axis']
    b = params['semi_minor_axis']
    phi = np.radians(params['rotation_angle_deg'])

    height, width = img_shape[:2]
    y, x = np.ogrid[:height, :width]

    x_centered = x - cx
    y_centered = y - cy
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)
    x_rot = x_centered * cos_phi + y_centered * sin_phi
    y_rot = -x_centered * sin_phi + y_centered * cos_phi

    ellipse_eq = (x_rot / a) ** 2 + (y_rot / b) ** 2
    return (ellipse_eq <= 1).astype(np.uint8)


if __name__ == '__main__':
    from paths import IMAGE_PATH
    # Pre-compute and cache
    params = get_ellipse_params(str(IMAGE_PATH), force_recompute=True)
    print(f"\nCached parameters:")
    for k, v in params.items():
        print(f"  {k}: {v:.2f}")
