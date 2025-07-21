import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import ezdxf
from scipy import ndimage
from skimage import measure
import cv2

def load_and_create_binary_grid(image_path, grid_size=16):
    """
    Load image and create binary grid (reused from main.py)
    
    Args:
        image_path: Path to input image
        grid_size: Size of the binary grid (default 16x16)
    
    Returns:
        binary_grid: 2D numpy array with 0s and 1s
    """
    # Load and process image
    img = Image.open(image_path).convert("L")
    img_array = np.array(img)
    
    # Determine grid resolution
    height, width = img_array.shape
    x_step = width / grid_size
    y_step = height / grid_size
    
    # Sample at cell centers
    binary_grid = np.zeros((grid_size, grid_size), dtype=int)
    for j in range(grid_size):
        for i in range(grid_size):
            x = int((i + 0.5) * x_step)
            y = int((j + 0.5) * y_step)
            pixel = img_array[y, x]
            binary_grid[j, i] = 1 if pixel > 128 else 0  # 1: bright, 0: dark
    
    return binary_grid

def upsample_binary_grid(binary_grid, upsample_factor=10):
    """
    Upsample binary grid by repeating each pixel
    
    Args:
        binary_grid: Original binary grid
        upsample_factor: Factor to upsample by (default 10)
    
    Returns:
        upsampled_grid: Higher resolution binary grid
    """
    # Use numpy repeat to create blocks
    upsampled = np.repeat(np.repeat(binary_grid, upsample_factor, axis=0), upsample_factor, axis=1)
    return upsampled.astype(np.float64)

def apply_gaussian_smoothing(grid, sigma):
    """
    Apply 2D Gaussian smoothing to the grid
    
    Args:
        grid: Input grid (can be binary or continuous)
        sigma: Standard deviation of Gaussian kernel
    
    Returns:
        smoothed_grid: Smoothed continuous-valued grid
    """
    return ndimage.gaussian_filter(grid, sigma=sigma)

def find_area_preserving_threshold(smoothed_grid, target_area, tolerance=1, debug=True):
    """
    Find threshold that preserves the original area using binary search

    Args:
        smoothed_grid: Continuous-valued smoothed grid
        target_area: Target number of pixels that should be 1
        tolerance: Acceptable difference in area (default: 1 pixel)
        debug: Print debug information

    Returns:
        optimal_threshold: Threshold value that gives closest to target area
    """
    # Binary search for the optimal threshold
    low = smoothed_grid.min()
    high = smoothed_grid.max()
    best_threshold = 0.5
    best_diff = float('inf')

    if debug:
        print(f"Binary search range: [{low:.6f}, {high:.6f}]")
        print(f"Target area: {target_area}")

    target_area = target_area * 1.1
    # Binary search with precision
    for iteration in range(50):  # 50 iterations gives very high precision
        mid = (low + high) / 2.0

        # Calculate area with current threshold
        binary_result = (smoothed_grid > mid).astype(int)
        current_area = np.sum(binary_result)
        diff = current_area - target_area

        # Track best result
        if abs(diff) < abs(best_diff):
            best_diff = diff
            best_threshold = mid

        if debug and iteration < 10:  # Show first 10 iterations
            print(f"Iter {iteration:2d}: thresh={mid:.6f}, area={int(current_area):5d}, diff={int(diff):5d}, range=[{low:.6f}, {high:.6f}]")

        # If we're within tolerance, we can stop
        if abs(diff) <= tolerance:
            if debug:
                print(f"Converged at iteration {iteration} with diff={diff}")
            break

        # Adjust search range
        if current_area > target_area:
            # Too many 1s, need higher threshold
            low = mid
        else:
            # Too few 1s, need lower threshold
            high = mid

    if debug:
        final_area = np.sum((smoothed_grid > best_threshold).astype(int))
        print(f"Final: thresh={best_threshold:.6f}, area={int(final_area)}, diff={int(final_area - target_area)}")

    return best_threshold

def threshold_to_binary(grid, threshold=None, target_area=None):
    """
    Convert continuous grid back to binary using threshold

    Args:
        grid: Continuous-valued grid
        threshold: Threshold value (if None, will find area-preserving threshold)
        target_area: Target area to preserve (number of 1s)

    Returns:
        binary_grid: Thresholded binary grid
        actual_threshold: The threshold value used
    """
    if threshold is None and target_area is not None:
        threshold = find_area_preserving_threshold(grid, target_area)
        print(f"Found area-preserving threshold: {threshold:.3f}")
    elif threshold is None:
        threshold = 0.5
        print("Using default threshold: 0.5")

    binary_grid = (grid > threshold).astype(int)
    return binary_grid, threshold

def detect_boundaries_from_continuous(smoothed_grid, threshold=0.5):
    """
    Detect smooth boundaries/contours directly from continuous smoothed data

    Args:
        smoothed_grid: Continuous-valued smoothed grid
        threshold: Contour level to extract (default 0.5)

    Returns:
        contours: List of smooth contour coordinates
    """
    from skimage import measure

    # Use marching squares to find smooth contours at the threshold level
    contours = measure.find_contours(smoothed_grid, threshold)

    # Convert to OpenCV-compatible format
    opencv_contours = []
    for contour in contours:
        # Swap x,y coordinates and convert to integer (but keep smooth shape)
        opencv_contour = np.array([[int(point[1]), int(point[0])] for point in contour])
        opencv_contours.append(opencv_contour.reshape(-1, 1, 2))

    return opencv_contours

def detect_boundaries(binary_grid):
    """
    Detect boundaries/contours from binary grid (legacy function)

    Args:
        binary_grid: Binary grid

    Returns:
        contours: List of contour coordinates
    """
    # Convert to uint8 for OpenCV
    binary_uint8 = (binary_grid * 255).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(binary_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours

def create_dxf_from_contours(contours, scale_factor, total_width_mm=2.0, filename="smoothed_output.dxf"):
    """
    Create DXF file from contours
    
    Args:
        contours: List of contour coordinates
        scale_factor: Factor to convert grid coordinates to mm
        total_width_mm: Total width in millimeters
        filename: Output DXF filename
    
    Returns:
        filename: The saved DXF filename
    """
    # Create a new DXF document
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    
    # Calculate scaling
    mm_per_pixel = total_width_mm / (160)  # Assuming 160x160 final grid
    
    contour_count = 0
    for contour in contours:
        if len(contour) > 2:  # Only process contours with enough points
            # Convert contour points to mm coordinates
            points = []
            for point in contour:
                x = point[0][0] * mm_per_pixel
                y = -point[0][1] * mm_per_pixel  # Flip Y axis for DXF
                points.append((x, y))
            
            # Create polyline from contour
            msp.add_lwpolyline(points, close=True)
            contour_count += 1
    
    print(f"Created {contour_count} contours in DXF")
    
    # Save the DXF file
    doc.saveas(filename)
    print(f"DXF file saved as: {filename}")
    
    return filename

def smooth_boundaries_pipeline(image_path, gaussian_sigma=2.0, upsample_factor=10, threshold=0.5):
    """
    Complete boundary smoothing pipeline
    
    Args:
        image_path: Path to input image
        gaussian_sigma: Standard deviation for Gaussian smoothing
        upsample_factor: Factor for upsampling (default 10)
        threshold: Threshold for binary conversion (default 0.5)
    
    Returns:
        Dictionary with all intermediate and final results
    """
    print("=== Boundary Smoothing Pipeline ===")
    
    # Step 1: Load and create initial binary grid
    print("Step 1: Loading image and creating 16x16 binary grid...")
    binary_grid_16x16 = load_and_create_binary_grid(image_path)
    print(f"Original grid shape: {binary_grid_16x16.shape}")
    
    # Step 2: Upsample to higher resolution
    print(f"Step 2: Upsampling by factor {upsample_factor}...")
    upsampled_grid = upsample_binary_grid(binary_grid_16x16, upsample_factor)
    print(f"Upsampled grid shape: {upsampled_grid.shape}")
    
    # Step 3: Apply Gaussian smoothing
    print(f"Step 3: Applying Gaussian smoothing (sigma={gaussian_sigma})...")
    smoothed_grid = apply_gaussian_smoothing(upsampled_grid, gaussian_sigma)
    
    # Step 4: Threshold back to binary using area-preserving threshold
    original_area = np.sum(upsampled_grid)
    print(f"Original area (number of 1s): {original_area}")
    print(f"Step 4: Finding area-preserving threshold...")
    final_binary_grid, actual_threshold = threshold_to_binary(smoothed_grid, target_area=original_area)
    final_area = np.sum(final_binary_grid)
    print(f"Final area: {final_area} (difference: {final_area - original_area})")
    
    # Step 5: Detect smooth boundaries directly from continuous data
    print("Step 5: Detecting smooth boundaries from continuous data...")
    smooth_contours = detect_boundaries_from_continuous(smoothed_grid, threshold)
    print(f"Found {len(smooth_contours)} smooth contours")

    # Also detect boundaries from binary for comparison
    binary_contours = detect_boundaries(final_binary_grid)
    print(f"Found {len(binary_contours)} binary contours (for comparison)")
    
    # Step 6: Create DXF files
    print("Step 6: Creating DXF files...")
    smooth_dxf_filename = create_dxf_from_contours(smooth_contours, upsample_factor, filename="smooth_boundaries.dxf")
    binary_dxf_filename = create_dxf_from_contours(binary_contours, upsample_factor, filename="binary_boundaries.dxf")

    return {
        'original_grid': binary_grid_16x16,
        'upsampled_grid': upsampled_grid,
        'smoothed_grid': smoothed_grid,
        'final_binary_grid': final_binary_grid,
        'smooth_contours': smooth_contours,
        'binary_contours': binary_contours,
        'smooth_dxf_filename': smooth_dxf_filename,
        'binary_dxf_filename': binary_dxf_filename
    }

def visualize_results(results, save_plots=True):
    """
    Visualize the results of the smoothing pipeline
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original 16x16 grid
    axes[0, 0].imshow(results['original_grid'], cmap='gray', interpolation='nearest')
    axes[0, 0].set_title('Original 16x16 Grid')
    axes[0, 0].axis('off')
    
    # Upsampled grid
    axes[0, 1].imshow(results['upsampled_grid'], cmap='gray', interpolation='nearest')
    axes[0, 1].set_title('Upsampled 160x160 Grid')
    axes[0, 1].axis('off')
    
    # Smoothed (continuous) grid
    axes[0, 2].imshow(results['smoothed_grid'], cmap='gray', interpolation='nearest')
    axes[0, 2].set_title('Gaussian Smoothed')
    axes[0, 2].axis('off')
    
    # Final binary grid
    axes[1, 0].imshow(results['final_binary_grid'], cmap='gray', interpolation='nearest')
    axes[1, 0].set_title('Final Binary Grid')
    axes[1, 0].axis('off')
    
    # Contours overlay - show both smooth and binary contours
    axes[1, 1].imshow(results['final_binary_grid'], cmap='gray', interpolation='nearest')
    # Plot smooth contours in red
    for contour in results['smooth_contours']:
        if len(contour) > 2:
            contour_points = contour.reshape(-1, 2)
            axes[1, 1].plot(contour_points[:, 0], contour_points[:, 1], 'r-', linewidth=2, label='Smooth' if contour is results['smooth_contours'][0] else "")
    # Plot binary contours in blue
    for contour in results['binary_contours']:
        if len(contour) > 2:
            contour_points = contour.reshape(-1, 2)
            axes[1, 1].plot(contour_points[:, 0], contour_points[:, 1], 'b-', linewidth=1, label='Binary' if contour is results['binary_contours'][0] else "")
    axes[1, 1].set_title('Smooth (Red) vs Binary (Blue) Boundaries')
    axes[1, 1].legend()
    axes[1, 1].axis('off')
    
    # Comparison: original vs final
    axes[1, 2].imshow(results['original_grid'], cmap='Reds', alpha=0.7, interpolation='nearest')
    axes[1, 2].imshow(cv2.resize(results['final_binary_grid'].astype(np.uint8), (16, 16), interpolation=cv2.INTER_NEAREST), 
                     cmap='Blues', alpha=0.7, interpolation='nearest')
    axes[1, 2].set_title('Original (Red) vs Smoothed (Blue)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_plots:
        import time
        timestamp = int(time.time())
        filename = f'boundary_smoothing_results_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Visualization saved as: {filename}")

        # Also save with the standard name for consistency
        plt.savefig('boundary_smoothing_results.png', dpi=300, bbox_inches='tight')
        print("Also saved as: boundary_smoothing_results.png")
    
    # plt.show()  # Commented out to avoid GUI issues in headless environments

def debug_smoothing_pipeline():
    """
    Debug version to examine each step carefully
    """
    print("=== DEBUG: Examining smoothing pipeline ===")

    # Step 1: Load and create initial binary grid
    binary_grid_16x16 = load_and_create_binary_grid("in.png")
    print(f"Original 16x16 grid:\n{binary_grid_16x16}")

    # Step 2: Upsample
    upsampled_grid = upsample_binary_grid(binary_grid_16x16, 10)
    print(f"Upsampled grid shape: {upsampled_grid.shape}")
    print(f"Upsampled grid unique values: {np.unique(upsampled_grid)}")

    # Step 3: Apply Gaussian smoothing
    smoothed_grid = apply_gaussian_smoothing(upsampled_grid, sigma=3.0)
    print(f"Smoothed grid shape: {smoothed_grid.shape}")
    print(f"Smoothed grid min/max: {smoothed_grid.min():.3f} / {smoothed_grid.max():.3f}")
    print(f"Smoothed grid unique values (first 10): {np.unique(smoothed_grid)[:10]}")

    # Step 4: Threshold back to binary using area-preserving threshold
    original_area = np.sum(upsampled_grid)
    print(f"Original area (number of 1s): {original_area}")
    final_binary_grid, actual_threshold = threshold_to_binary(smoothed_grid, target_area=original_area)
    final_area = np.sum(final_binary_grid)
    print(f"Final binary grid shape: {final_binary_grid.shape}")
    print(f"Final binary grid unique values: {np.unique(final_binary_grid)}")
    print(f"Area preservation: original={original_area}, final={final_area}, diff={final_area - original_area}")

    # Check a larger region to see the transition across boundaries
    print("\n=== Examining a larger region (70:100, 70:100) ===")
    print("Original upsampled (should be all 0s or 1s):")
    print(upsampled_grid[70:100, 70:100])
    print("\nAfter Gaussian smoothing (should have gradual transitions):")
    print(smoothed_grid[70:100, 70:100])
    print("\nAfter thresholding (should be 0s and 1s, but with smoother boundaries):")
    print(final_binary_grid[70:100, 70:100])

    # Let's also check if we have any boundary regions (where smoothed values are around 0.5)
    boundary_mask = (smoothed_grid > 0.3) & (smoothed_grid < 0.7)
    print(f"\nNumber of boundary pixels (0.3 < value < 0.7): {np.sum(boundary_mask)}")
    print(f"This should be > 0 if we have smooth transitions")

    # Find a region with boundaries
    boundary_indices = np.where(boundary_mask)
    if len(boundary_indices[0]) > 0:
        # Pick a boundary region
        by, bx = boundary_indices[0][0], boundary_indices[1][0]
        print(f"\nExamining boundary region around ({by}, {bx}):")
        y_start, y_end = max(0, by-5), min(160, by+6)
        x_start, x_end = max(0, bx-5), min(160, bx+6)
        print("Smoothed values:")
        print(smoothed_grid[y_start:y_end, x_start:x_end])
        print("Thresholded values:")
        print(final_binary_grid[y_start:y_end, x_start:x_end])

    # Save intermediate results for inspection
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(upsampled_grid, cmap='gray', interpolation='nearest')
    plt.title('Upsampled (should be blocky)')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(smoothed_grid, cmap='gray', interpolation='nearest')
    plt.title('Gaussian Smoothed (should be smooth)')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(final_binary_grid, cmap='gray', interpolation='nearest')
    plt.title('Thresholded (should have smooth boundaries)')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig('debug_smoothing_steps.png', dpi=300, bbox_inches='tight')
    print("\nDebug visualization saved as: debug_smoothing_steps.png")

    return {
        'original': binary_grid_16x16,
        'upsampled': upsampled_grid,
        'smoothed': smoothed_grid,
        'final_binary': final_binary_grid
    }

if __name__ == "__main__":
    # Debug the smoothing pipeline first
    debug_results = debug_smoothing_pipeline()

    print("\n" + "="*50)
    print("Now running full pipeline...")

    # Parameters
    image_path = "in.png"
    gaussian_sigma = 6.0  # Adjustable smoothing parameter

    # Run the pipeline
    results = smooth_boundaries_pipeline(
        image_path=image_path,
        gaussian_sigma=gaussian_sigma,
        upsample_factor=10,
        threshold=0.5
    )

    # Visualize results
    visualize_results(results)

    print(f"\n=== Pipeline Complete ===")
    print(f"Gaussian sigma used: {gaussian_sigma}")
    print(f"Smooth DXF file created: {results['smooth_dxf_filename']}")
    print(f"Binary DXF file created: {results['binary_dxf_filename']}")
