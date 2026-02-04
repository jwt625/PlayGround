"""
Measure the viewing angle of a circular wafer from its elliptical appearance in an image.

Uses edge detection and robust ellipse fitting with 5 DOF:
- Center (cx, cy)
- Semi-major axis (a)
- Semi-minor axis (b)
- Rotation angle (phi)

When a circle is viewed at angle θ from normal: aspect_ratio = b/a = cos(θ)
"""

import cv2
import numpy as np
from scipy.optimize import least_squares
from paths import IMAGE_PATH, OUTPUT_ANALYSIS


def ellipse_points(params, n_points=360):
    """Generate points on an ellipse given parameters."""
    cx, cy, a, b, phi = params
    t = np.linspace(0, 2 * np.pi, n_points)
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)
    x = cx + a * np.cos(t) * cos_phi - b * np.sin(t) * sin_phi
    y = cy + a * np.cos(t) * sin_phi + b * np.sin(t) * cos_phi
    return x, y


def ellipse_distance(params, edge_points):
    """
    Calculate algebraic distance from points to ellipse.
    Uses the general conic equation: Ax² + Bxy + Cy² + Dx + Ey + F = 0
    """
    cx, cy, a, b, phi = params

    # Ensure a >= b (a is semi-major axis)
    if a < b:
        a, b = b, a
        phi = phi + np.pi / 2

    cos_phi, sin_phi = np.cos(phi), np.sin(phi)

    # Transform points to ellipse-centered coordinates
    x_centered = edge_points[:, 0] - cx
    y_centered = edge_points[:, 1] - cy

    # Rotate to align with ellipse axes
    x_rot = x_centered * cos_phi + y_centered * sin_phi
    y_rot = -x_centered * sin_phi + y_centered * cos_phi

    # Distance in normalized ellipse space: (x/a)² + (y/b)² = 1
    # Residual is deviation from 1
    normalized_dist = (x_rot / a) ** 2 + (y_rot / b) ** 2
    residuals = normalized_dist - 1

    return residuals


def fit_ellipse_least_squares(edge_points, initial_guess=None):
    """
    Fit an ellipse to edge points using nonlinear least squares.

    Parameters:
        edge_points: Nx2 array of (x, y) coordinates
        initial_guess: Optional initial parameters [cx, cy, a, b, phi]

    Returns:
        Optimized parameters [cx, cy, a, b, phi]
    """
    if initial_guess is None:
        # Use OpenCV's fitEllipse for initial guess
        ellipse = cv2.fitEllipse(edge_points.reshape(-1, 1, 2).astype(np.float32))
        (cx, cy), (w, h), angle = ellipse
        a, b = max(w, h) / 2, min(w, h) / 2
        phi = np.radians(angle)
        initial_guess = [cx, cy, a, b, phi]

    # Bounds for parameters
    img_max = max(edge_points.max(), 5000)
    bounds = (
        [0, 0, 10, 10, -np.pi],  # Lower bounds
        [img_max, img_max, img_max, img_max, np.pi]  # Upper bounds
    )

    result = least_squares(
        ellipse_distance,
        initial_guess,
        args=(edge_points,),
        bounds=bounds,
        method='trf',
        ftol=1e-10,
        xtol=1e-10
    )

    params = result.x
    # Ensure a >= b
    if params[2] < params[3]:
        params[2], params[3] = params[3], params[2]
        params[4] = params[4] + np.pi / 2

    # Normalize angle to [-pi/2, pi/2]
    params[4] = ((params[4] + np.pi / 2) % np.pi) - np.pi / 2

    return params, result


def detect_wafer_edges(img):
    """
    Detect the wafer edges using Canny edge detection.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)

    # Use Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Also use color information to focus on the wafer region
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Blue color range for the wafer
    lower_blue = np.array([90, 30, 30])
    upper_blue = np.array([140, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Dilate the mask to include edge pixels
    kernel = np.ones((15, 15), np.uint8)
    dilated_mask = cv2.dilate(blue_mask, kernel, iterations=2)

    # Combine edge detection with color mask
    # Keep edges that are near the blue region
    masked_edges = cv2.bitwise_and(edges, dilated_mask)

    # Find edge points
    edge_points = np.column_stack(np.where(masked_edges > 0))
    # Swap to (x, y) format
    edge_points = edge_points[:, ::-1]

    return edge_points, masked_edges


def filter_outliers_ransac(edge_points, n_iterations=100, threshold_percentile=90):
    """
    Use RANSAC-like approach to filter outlier edge points.
    """
    best_inliers = None
    best_params = None
    best_score = 0

    n_points = len(edge_points)
    sample_size = min(50, n_points // 2)

    for _ in range(n_iterations):
        # Random sample
        idx = np.random.choice(n_points, sample_size, replace=False)
        sample = edge_points[idx]

        try:
            # Fit ellipse to sample
            if len(sample) < 5:
                continue
            ellipse = cv2.fitEllipse(sample.reshape(-1, 1, 2).astype(np.float32))
            (cx, cy), (w, h), angle = ellipse
            a, b = max(w, h) / 2, min(w, h) / 2
            phi = np.radians(angle)
            params = [cx, cy, a, b, phi]

            # Calculate residuals for all points
            residuals = np.abs(ellipse_distance(params, edge_points))

            # Count inliers
            threshold = np.percentile(residuals, threshold_percentile)
            inlier_mask = residuals < threshold
            score = np.sum(inlier_mask)

            if score > best_score:
                best_score = score
                best_inliers = inlier_mask
                best_params = params

        except cv2.error:
            continue

    if best_inliers is None:
        return edge_points, np.ones(len(edge_points), dtype=bool)

    return edge_points[best_inliers], best_inliers


def detect_ellipse_and_viewing_angle(image_path, output_dir=None):
    """
    Detect the ellipse in the image and calculate the viewing angle.

    Parameters:
        image_path: Path to the input image
        output_dir: Directory to save output files (default: same as input image)
    """
    import os
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    print("Step 1: Detecting edges...")
    edge_points, edge_image = detect_wafer_edges(img)
    print(f"  Found {len(edge_points)} edge points")

    print("Step 2: Filtering outliers with RANSAC...")
    filtered_points, inlier_mask = filter_outliers_ransac(edge_points)
    print(f"  Kept {len(filtered_points)} inlier points")

    print("Step 3: Fitting ellipse with least squares optimization...")
    params, result = fit_ellipse_least_squares(filtered_points)
    cx, cy, a, b, phi = params
    print(f"  Optimization converged: {result.success}")
    print(f"  Final cost: {result.cost:.6f}")

    # Calculate aspect ratio and viewing angle
    aspect_ratio = b / a
    viewing_angle_rad = np.arccos(aspect_ratio)
    viewing_angle_deg = np.degrees(viewing_angle_rad)

    # Create visualization
    output_img = img.copy()

    # Draw the fitted ellipse
    ellipse_x, ellipse_y = ellipse_points(params, n_points=500)
    for i in range(len(ellipse_x) - 1):
        pt1 = (int(ellipse_x[i]), int(ellipse_y[i]))
        pt2 = (int(ellipse_x[i + 1]), int(ellipse_y[i + 1]))
        cv2.line(output_img, pt1, pt2, (0, 255, 0), 3)

    # Draw center
    cv2.circle(output_img, (int(cx), int(cy)), 8, (0, 0, 255), -1)

    # Draw axes
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)
    # Major axis
    pt1 = (int(cx - a * cos_phi), int(cy - a * sin_phi))
    pt2 = (int(cx + a * cos_phi), int(cy + a * sin_phi))
    cv2.line(output_img, pt1, pt2, (255, 0, 0), 2)
    # Minor axis
    pt1 = (int(cx + b * sin_phi), int(cy - b * cos_phi))
    pt2 = (int(cx - b * sin_phi), int(cy + b * cos_phi))
    cv2.line(output_img, pt1, pt2, (255, 0, 255), 2)

    # Save outputs
    if output_dir is None:
        output_path = image_path.replace('.png', '_detected.png')
        edge_output_path = image_path.replace('.png', '_edges.png')
    else:
        basename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, basename.replace('.png', '_detected.png'))
        edge_output_path = os.path.join(output_dir, basename.replace('.png', '_edges.png'))

    cv2.imwrite(output_path, output_img)
    cv2.imwrite(edge_output_path, edge_image)

    results = {
        'center': (cx, cy),
        'semi_major_axis': a,
        'semi_minor_axis': b,
        'major_axis': 2 * a,
        'minor_axis': 2 * b,
        'aspect_ratio': aspect_ratio,
        'rotation_angle_deg': np.degrees(phi),
        'viewing_angle_deg': viewing_angle_deg,
        'viewing_angle_rad': viewing_angle_rad,
        'n_edge_points': len(filtered_points),
        'fit_cost': result.cost,
        'output_image': output_path,
        'edge_image': edge_output_path
    }

    return results


def main():
    image_path = str(IMAGE_PATH)

    print("=" * 60)
    print("Viewing Angle Measurement from Ellipse Aspect Ratio")
    print("Edge Detection + Least Squares Ellipse Fitting")
    print("=" * 60)

    results = detect_ellipse_and_viewing_angle(image_path, output_dir=str(OUTPUT_ANALYSIS))

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print("=" * 60)

    print(f"\nEllipse Parameters (5 DOF):")
    print(f"  Center: ({results['center'][0]:.1f}, {results['center'][1]:.1f}) pixels")
    print(f"  Semi-major axis (a): {results['semi_major_axis']:.1f} pixels")
    print(f"  Semi-minor axis (b): {results['semi_minor_axis']:.1f} pixels")
    print(f"  Rotation angle: {results['rotation_angle_deg']:.2f}°")

    print(f"\nFit Quality:")
    print(f"  Edge points used: {results['n_edge_points']}")
    print(f"  Fit cost (sum of squared residuals): {results['fit_cost']:.4f}")

    print(f"\nAspect Ratio Analysis:")
    print(f"  Aspect ratio (b/a): {results['aspect_ratio']:.4f}")

    print(f"\nViewing Angle:")
    print(f"  θ = arccos({results['aspect_ratio']:.4f})")
    print(f"  θ = {results['viewing_angle_deg']:.2f}° from normal")
    print(f"  θ = {results['viewing_angle_rad']:.4f} radians")

    print(f"\nOutput files:")
    print(f"  Fitted ellipse: {results['output_image']}")
    print(f"  Edge detection: {results['edge_image']}")


if __name__ == '__main__':
    main()
