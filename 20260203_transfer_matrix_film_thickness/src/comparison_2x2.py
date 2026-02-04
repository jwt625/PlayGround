"""
Create 2x2 comparison image:
- Actual wafer
- Predicted wafer (from thickness)
- Thickness map
- RGB difference
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from ellipse_cache import get_ellipse_params, create_ellipse_mask
from paths import IMAGE_PATH, THICKNESS_MAP_PATH, THICKNESS_LUT_PATH, OUTPUT_THICKNESS


def create_2x2_comparison(red_threshold=100):
    # Load data
    thickness_map = np.load(THICKNESS_MAP_PATH)
    data = np.load(THICKNESS_LUT_PATH, allow_pickle=True)
    thicknesses, rgb_lut = data['thicknesses'], data['rgb_lut']

    # Load image
    img = cv2.imread(str(IMAGE_PATH))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # Load masks
    params = get_ellipse_params(str(IMAGE_PATH))
    wafer_mask = create_ellipse_mask(img.shape, params)
    r_channel = img_rgb[:, :, 0]
    valid_mask = (wafer_mask == 1) & (r_channel < red_threshold)

    # Crop to wafer bounding box with small padding
    rows = np.any(wafer_mask, axis=1)
    cols = np.any(wafer_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    pad = 20
    rmin, rmax = max(0, rmin - pad), min(h, rmax + pad)
    cmin, cmax = max(0, cmin - pad), min(w, cmax + pad)

    # Crop all arrays
    img_rgb = img_rgb[rmin:rmax, cmin:cmax]
    wafer_mask = wafer_mask[rmin:rmax, cmin:cmax]
    valid_mask = valid_mask[rmin:rmax, cmin:cmax]
    thickness_map = thickness_map[rmin:rmax, cmin:cmax]
    h, w = img_rgb.shape[:2]

    # Create predicted RGB image
    predicted_rgb = np.zeros((h, w, 3), dtype=np.float32)
    valid_thickness = thickness_map[valid_mask]
    thickness_indices = np.round(valid_thickness - thicknesses[0]).astype(int)
    thickness_indices = np.clip(thickness_indices, 0, len(thicknesses) - 1)
    predicted_rgb[valid_mask] = rgb_lut[thickness_indices]

    # Normalize actual image
    actual_norm = img_rgb.astype(np.float32) / 255.0

    # Calculate RGB difference
    diff = np.abs(actual_norm - predicted_rgb)
    diff_magnitude = np.sqrt(np.sum(diff**2, axis=2))

    # Calculate figure size to match image aspect ratio (2 cols x 2 rows)
    img_aspect = w / h
    fig_width = 14
    fig_height = fig_width / img_aspect

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor='#1a1a1a')

    # GridSpec: 2 rows, 4 cols (col 1,3 for images, col 2,4 for colorbars)
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 0.03, 1, 0.03],
                          left=0.01, right=0.99, top=0.95, bottom=0.02,
                          wspace=0.02, hspace=0.08)

    # Create axes for images
    ax_actual = fig.add_subplot(gs[0, 0])
    ax_pred = fig.add_subplot(gs[0, 2])
    ax_thick = fig.add_subplot(gs[1, 0])
    ax_err = fig.add_subplot(gs[1, 2])

    for ax in [ax_actual, ax_pred, ax_thick, ax_err]:
        ax.set_facecolor('#1a1a1a')
        ax.axis('off')

    # 1. Actual wafer
    actual_display = actual_norm.copy()
    actual_display[~valid_mask] = 0.1
    ax_actual.imshow(actual_display)
    ax_actual.set_title('Actual (R < 100)', color='white', fontsize=11, pad=3)

    # 2. Predicted wafer
    predicted_display = predicted_rgb.copy()
    predicted_display[~valid_mask] = 0.1
    ax_pred.imshow(predicted_display)
    ax_pred.set_title('Predicted (BTO/Si)', color='white', fontsize=11, pad=3)

    # 3. Thickness map
    thickness_display = np.full((h, w, 3), 0.1, dtype=np.float32)
    valid_d = thickness_map[valid_mask]

    # Use 5th-95th percentile for colorbar range (avoid outlier stretching)
    d_p5, d_p95 = np.percentile(valid_d, [5, 95])
    d_min_actual, d_max_actual = np.nanmin(valid_d), np.nanmax(valid_d)
    print(f"Thickness distribution:")
    print(f"  Min: {d_min_actual:.1f}, Max: {d_max_actual:.1f}")
    print(f"  5th percentile: {d_p5:.1f}, 95th percentile: {d_p95:.1f}")

    cmap_thick = plt.cm.viridis
    norm_thick = Normalize(vmin=d_p5, vmax=d_p95)
    thickness_normalized = (thickness_map - d_p5) / (d_p95 - d_p5)
    thickness_normalized = np.clip(thickness_normalized, 0, 1)
    thickness_colored = cmap_thick(thickness_normalized)[:, :, :3]
    thickness_display[valid_mask] = thickness_colored[valid_mask]
    ax_thick.imshow(thickness_display)
    ax_thick.set_title(f'Thickness ({d_p5:.0f}-{d_p95:.0f} nm, 90%)', color='white', fontsize=11, pad=3)

    # Colorbar for thickness
    cax_thick = fig.add_subplot(gs[1, 1])
    sm_thick = plt.cm.ScalarMappable(cmap=cmap_thick, norm=norm_thick)
    cbar_thick = plt.colorbar(sm_thick, cax=cax_thick)
    cbar_thick.set_label('nm', color='white', fontsize=9)
    cbar_thick.ax.yaxis.set_tick_params(color='white', labelsize=7)
    plt.setp(plt.getp(cbar_thick.ax.axes, 'yticklabels'), color='white')

    # 4. RGB difference
    diff_display = np.full((h, w, 3), 0.1, dtype=np.float32)
    valid_err = diff_magnitude[valid_mask]

    # Use 5th-95th percentile for RGB error too
    err_p5, err_p95 = np.percentile(valid_err, [5, 95])
    print(f"RGB error distribution:")
    print(f"  Min: {np.min(valid_err):.3f}, Max: {np.max(valid_err):.3f}")
    print(f"  5th percentile: {err_p5:.3f}, 95th percentile: {err_p95:.3f}")

    cmap_err = plt.cm.hot
    norm_err = Normalize(vmin=err_p5, vmax=err_p95)
    diff_normalized = (diff_magnitude - err_p5) / (err_p95 - err_p5)
    diff_normalized = np.clip(diff_normalized, 0, 1)
    diff_colored = cmap_err(diff_normalized)[:, :, :3]
    diff_display[valid_mask] = diff_colored[valid_mask]
    ax_err.imshow(diff_display)
    ax_err.set_title(f'RGB Error (mean={np.mean(valid_err):.2f})',
                     color='white', fontsize=11, pad=3)

    # Colorbar for RGB error
    cax_err = fig.add_subplot(gs[1, 3])
    sm_err = plt.cm.ScalarMappable(cmap=cmap_err, norm=norm_err)
    cbar_err = plt.colorbar(sm_err, cax=cax_err)
    cbar_err.set_label('RGB dist', color='white', fontsize=9)
    cbar_err.ax.yaxis.set_tick_params(color='white', labelsize=7)
    plt.setp(plt.getp(cbar_err.ax.axes, 'yticklabels'), color='white')

    output_path = OUTPUT_THICKNESS / 'comparison_2x2.png'
    plt.savefig(output_path, dpi=150, facecolor='#1a1a1a', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    create_2x2_comparison(red_threshold=100)
