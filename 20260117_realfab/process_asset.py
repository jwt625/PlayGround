#!/usr/bin/env python3
"""
Process pyramid asset images by:
1. Cropping to center third horizontally (optional)
2. Sampling green background color from edges
3. Removing green background with tolerance
4. Removing isolated pixels not connected to main object
5. Converting to WebP with transparency
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage


def sample_edge_colors(img, sample_size=20):
    """Sample colors from the four corners and edges of the image."""
    width, height = img.size
    pixels = img.load()
    
    samples = []
    
    # Sample from corners (sample_size x sample_size squares)
    corners = [
        (0, 0),  # Top-left
        (width - sample_size, 0),  # Top-right
        (0, height - sample_size),  # Bottom-left
        (width - sample_size, height - sample_size),  # Bottom-right
    ]
    
    for corner_x, corner_y in corners:
        for x in range(corner_x, min(corner_x + sample_size, width)):
            for y in range(corner_y, min(corner_y + sample_size, height)):
                samples.append(pixels[x, y][:3])  # RGB only
    
    # Sample from edges (middle sections)
    # Top edge
    for x in range(width // 2 - sample_size, width // 2 + sample_size):
        for y in range(sample_size):
            if 0 <= x < width and 0 <= y < height:
                samples.append(pixels[x, y][:3])
    
    # Bottom edge
    for x in range(width // 2 - sample_size, width // 2 + sample_size):
        for y in range(height - sample_size, height):
            if 0 <= x < width and 0 <= y < height:
                samples.append(pixels[x, y][:3])
    
    # Left edge
    for x in range(sample_size):
        for y in range(height // 2 - sample_size, height // 2 + sample_size):
            if 0 <= x < width and 0 <= y < height:
                samples.append(pixels[x, y][:3])
    
    # Right edge
    for x in range(width - sample_size, width):
        for y in range(height // 2 - sample_size, height // 2 + sample_size):
            if 0 <= x < width and 0 <= y < height:
                samples.append(pixels[x, y][:3])
    
    return samples


def get_dominant_green(samples):
    """Find the dominant green color from samples."""
    # Filter for greenish colors (G > R and G > B)
    green_samples = [s for s in samples if s[1] > s[0] and s[1] > s[2]]
    
    if not green_samples:
        print("Warning: No green colors found in samples, using default bright green")
        return (0, 255, 0)
    
    # Calculate median of green samples
    green_array = np.array(green_samples)
    median_color = np.median(green_array, axis=0).astype(int)
    
    print(f"Detected background green: RGB{tuple(median_color)}")
    return tuple(median_color)


def remove_isolated_pixels(img_rgba, keep_multiple=False):
    """Remove pixels not connected to the main object(s).

    Args:
        img_rgba: RGBA PIL Image
        keep_multiple: If True, keep all objects that overlap with the middle 50% of the image.
                      If False, keep only the largest connected component.
    """
    data = np.array(img_rgba)
    alpha = data[:, :, 3]

    # Create binary mask of non-transparent pixels
    binary_mask = alpha > 0

    # Label connected components
    labeled, num_features = ndimage.label(binary_mask)

    if num_features == 0:
        print("No objects found in image")
        return img_rgba

    if keep_multiple:
        # Keep all objects that overlap with the middle 50% of the image
        height, width = alpha.shape

        # Define middle region (middle 50% horizontally and vertically)
        middle_x_start = width // 4
        middle_x_end = 3 * width // 4
        middle_y_start = height // 4
        middle_y_end = 3 * height // 4

        # Create mask for middle region
        middle_region = np.zeros_like(binary_mask, dtype=bool)
        middle_region[middle_y_start:middle_y_end, middle_x_start:middle_x_end] = True

        # Find which components overlap with the middle region
        components_to_keep = set()
        for component_id in range(1, num_features + 1):
            component_mask = labeled == component_id
            if np.any(component_mask & middle_region):
                components_to_keep.add(component_id)

        # Create mask for all components to keep
        main_object_mask = np.zeros_like(binary_mask, dtype=bool)
        for component_id in components_to_keep:
            main_object_mask |= (labeled == component_id)

        # Count isolated pixels to be removed
        isolated_pixels = np.sum(binary_mask & ~main_object_mask)

        if isolated_pixels > 0:
            kept_components = len(components_to_keep)
            removed_components = num_features - kept_components
            print(f"Removing {isolated_pixels:,} isolated pixels ({removed_components} small blobs)")
            print(f"Keeping {kept_components} objects overlapping with center region")

            # Set alpha to 0 for all pixels not in kept objects
            alpha[~main_object_mask] = 0
            data[:, :, 3] = alpha

            return Image.fromarray(data, 'RGBA')
        else:
            print(f"Keeping all {num_features} objects (all overlap with center)")
            return img_rgba
    else:
        # Original behavior: keep only the largest connected component
        component_sizes = ndimage.sum(binary_mask, labeled, range(num_features + 1))
        largest_component = np.argmax(component_sizes[1:]) + 1  # Skip background (0)

        # Create mask for largest component only
        main_object_mask = labeled == largest_component

        # Count isolated pixels to be removed
        isolated_pixels = np.sum(binary_mask & ~main_object_mask)

        if isolated_pixels > 0:
            print(f"Removing {isolated_pixels:,} isolated pixels ({num_features - 1} small blobs)")

            # Set alpha to 0 for all pixels not in main object
            alpha[~main_object_mask] = 0
            data[:, :, 3] = alpha

            return Image.fromarray(data, 'RGBA')
        else:
            print("No isolated pixels found")
            return img_rgba


def remove_green_background(img, target_green, tolerance=50, aggressive=True):
    """Remove green background with tolerance, making it transparent."""
    img_rgba = img.convert('RGBA')
    data = np.array(img_rgba)

    # Extract RGB channels
    r, g, b, a = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]

    # Calculate color distance from target green
    target_r, target_g, target_b = target_green
    distance = np.sqrt(
        (r.astype(float) - target_r) ** 2 +
        (g.astype(float) - target_g) ** 2 +
        (b.astype(float) - target_b) ** 2
    )

    # Create mask: pixels within tolerance of target green
    mask = distance <= tolerance

    if aggressive:
        # Additional aggressive removal: any pixel where green is dominant
        # Target greenish halo pixels using ratio-based detection
        # Avoid removing white/gray pixels (where R≈G≈B)

        # Calculate how much green dominates over red and blue
        r_float = r.astype(float) + 1  # Add 1 to avoid division by zero
        b_float = b.astype(float) + 1
        g_float = g.astype(float) + 1

        # Green is significantly higher than both R and B (ratio-based)
        green_ratio_r = g_float / r_float
        green_ratio_b = g_float / b_float

        # Check if colors are balanced (white/gray pixels)
        max_channel = np.maximum(np.maximum(r_float, g_float), b_float)
        min_channel = np.minimum(np.minimum(r_float, g_float), b_float)
        is_balanced = (max_channel - min_channel) < 30  # RGB values are close

        # Green halo: G is 2x or more than R, 1.7x or more than B, not balanced, and G > 80
        green_dominant = (green_ratio_r >= 2.0) & (green_ratio_b >= 1.7) & ~is_balanced & (g > 80)
        mask = mask | green_dominant

        print(f"Removed {np.sum(mask):,} green pixels ({np.sum(mask)/mask.size*100:.1f}%)")
        print(f"  - Distance-based: {np.sum(distance <= tolerance):,}")
        print(f"  - Green-dominant: {np.sum(green_dominant):,}")
    else:
        pixels_removed = np.sum(mask)
        total_pixels = mask.size
        print(f"Removed {pixels_removed:,} green pixels ({pixels_removed/total_pixels*100:.1f}%)")

    # Set alpha to 0 for masked pixels
    a[mask] = 0

    # Update alpha channel
    data[:, :, 3] = a

    return Image.fromarray(data, 'RGBA')


def inspect_vertical_cutline(img, output_path):
    """Inspect pixels along vertical center line from bottom up."""
    width, height = img.size
    center_x = width // 2

    print(f"\n--- Inspecting vertical cutline at x={center_x} ---")
    pixels = img.load()

    # Find first non-transparent pixel from bottom
    first_opaque_y = None
    for y in range(height - 1, -1, -1):
        pixel = pixels[center_x, y]
        if len(pixel) == 4 and pixel[3] > 0:  # Has alpha and not transparent
            first_opaque_y = y
            break

    if first_opaque_y is None:
        print("No opaque pixels found on center line")
        return

    print(f"First opaque pixel from bottom at y={first_opaque_y}")
    print(f"\nFirst 20 pixels from bottom (y={first_opaque_y} upward):")
    print("Y-pos | R   G   B   A   | RGB Distance from detected green")
    print("-" * 60)

    for i in range(20):
        y = first_opaque_y - i
        if y < 0:
            break
        pixel = pixels[center_x, y]
        if len(pixel) == 4:
            r, g, b, a = pixel
            # Calculate distance from detected green (will be shown in main processing)
            print(f"{y:5d} | {r:3d} {g:3d} {b:3d} {a:3d} |")

    print("-" * 60)


def process_image(input_path, output_path, crop_center_third=False, inspect=False, keep_multiple=False):
    """Process a single image."""
    print(f"\nProcessing: {input_path}")

    # Load image
    img = Image.open(input_path)
    print(f"Original size: {img.size}")

    # Step 1: Crop to center third horizontally if requested
    if crop_center_third:
        width, height = img.size
        third_width = width // 3
        left = third_width
        right = 2 * third_width
        img = img.crop((left, 0, right, height))
        print(f"Cropped to center third: {img.size}")

    # Step 2: Sample edge colors
    print("Sampling edge colors...")
    samples = sample_edge_colors(img)
    print(f"Collected {len(samples)} color samples")

    # Step 3: Detect dominant green
    target_green = get_dominant_green(samples)

    # Step 4: Remove green background
    print("Removing green background...")
    img_transparent = remove_green_background(img, target_green, tolerance=50)

    # Inspection mode: check vertical cutline before cleanup
    if inspect:
        inspect_vertical_cutline(img_transparent, output_path)

    # Step 5: Remove isolated pixels
    print("Removing isolated pixels...")
    img_clean = remove_isolated_pixels(img_transparent, keep_multiple=keep_multiple)

    # Step 6: Save as WebP
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img_clean.save(output_path, 'WEBP', quality=90)
    print(f"Saved to: {output_path}")
    print(f"Output size: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python process_asset.py <input_image> <output_image.webp> [--crop] [--inspect] [--keep-multiple]")
        print("\nExample:")
        print("  python process_asset.py assets/raw/MetalPowerContainer.png assets/eggs.webp")
        print("  python process_asset.py input.png output.webp --crop           # Crop to center third")
        print("  python process_asset.py input.png output.webp --inspect        # Show pixel inspection")
        print("  python process_asset.py input.png output.webp --keep-multiple  # Keep multiple objects in center")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    crop = '--crop' in sys.argv
    inspect = '--inspect' in sys.argv
    keep_multiple = '--keep-multiple' in sys.argv

    process_image(input_file, output_file, crop_center_third=crop, inspect=inspect, keep_multiple=keep_multiple)
    print("\nDone!")

