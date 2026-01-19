"""
Optical Tweezer Array Rearrangement Demo

Demonstrates smooth interpolation between two spot configurations:
- Start: Randomly half-filled 8x8 grid (32 spots)
- End: Compact 4x8 array in center

Uses Hungarian algorithm for optimal spot assignment and
Weighted GS with warm-start for smooth phase evolution.

Output: tweezer_phase.gif, tweezer_intensity.gif
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
import imageio
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for rendering
import matplotlib.pyplot as plt
from gs_algorithms import weighted_gs


# --- Configuration ---
GRID_SIZE = 256          # SLM resolution
N_FRAMES = 60            # Animation frames
GS_ITERATIONS = 50       # GS iterations per frame
SPOT_RADIUS = 3          # Spot size in pixels
SEED = 42
FPS = 15                 # GIF frame rate


def create_gaussian_input(size: int, sigma: float = None) -> np.ndarray:
    """Create Gaussian input beam amplitude."""
    if sigma is None:
        sigma = size / 4
    x = np.linspace(-size // 2, size // 2, size)
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X**2 + Y**2) / (2 * sigma**2))


def grid_to_pixel(row: int, col: int, grid_rows: int, grid_cols: int, size: int) -> tuple[int, int]:
    """Convert grid position to pixel coordinates (centered)."""
    spacing_y = size // (grid_rows + 1)
    spacing_x = size // (grid_cols + 1)
    px = spacing_x * (col + 1)
    py = spacing_y * (row + 1)
    return py, px


def create_spot_pattern(positions: list[tuple[float, float]], size: int, radius: int) -> np.ndarray:
    """Create target amplitude from list of (y, x) positions."""
    target = np.zeros((size, size))
    y_grid, x_grid = np.ogrid[:size, :size]
    for py, px in positions:
        mask = (x_grid - px)**2 + (y_grid - py)**2 <= radius**2
        target[mask] = 1.0
    return target


def generate_start_end_positions(seed: int = 42) -> tuple[list, list]:
    """
    Generate start (random half-filled 11x11) and end (circular compact) positions.
    Returns positions in pixel coordinates.
    """
    np.random.seed(seed)

    grid_size = 11
    center = grid_size // 2  # Center of 11x11 grid is (5, 5)

    # Start: randomly select ~half positions from 11x11 grid
    all_positions = [(r, c) for r in range(grid_size) for c in range(grid_size)]
    n_spots = len(all_positions) // 2  # 60 spots
    selected_indices = np.random.choice(len(all_positions), n_spots, replace=False)
    start_grid = [all_positions[i] for i in selected_indices]

    # End: circular fill from center outward
    # Sort all grid positions by distance from center
    def dist_from_center(pos):
        return (pos[0] - center) ** 2 + (pos[1] - center) ** 2

    all_sorted = sorted(all_positions, key=dist_from_center)
    end_grid = all_sorted[:n_spots]  # Take the n_spots closest to center

    # Convert to pixel coordinates
    start_pixels = [grid_to_pixel(r, c, grid_size, grid_size, GRID_SIZE) for r, c in start_grid]
    end_pixels = [grid_to_pixel(r, c, grid_size, grid_size, GRID_SIZE) for r, c in end_grid]

    return start_pixels, end_pixels


def solve_assignment(start: list, end: list) -> list[int]:
    """
    Solve optimal assignment using Hungarian algorithm.
    Returns permutation: end[i] should receive spot from start[assignment[i]].
    """
    n = len(start)
    cost_matrix = np.zeros((n, n))
    for i, (sy, sx) in enumerate(start):
        for j, (ey, ex) in enumerate(end):
            cost_matrix[i, j] = (sy - ey)**2 + (sx - ex)**2
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # Return mapping: for each end position, which start position feeds it
    assignment = [0] * n
    for r, c in zip(row_ind, col_ind):
        assignment[c] = r
    return assignment


def interpolate_positions(start: list, end: list, assignment: list, t: float) -> list:
    """Linearly interpolate positions. t in [0, 1]."""
    positions = []
    for i, end_pos in enumerate(end):
        start_pos = start[assignment[i]]
        y = start_pos[0] + t * (end_pos[0] - start_pos[0])
        x = start_pos[1] + t * (end_pos[1] - start_pos[1])
        positions.append((y, x))
    return positions


def render_frame(data: np.ndarray, title: str, cmap: str, vmin=None, vmax=None) -> np.ndarray:
    """Render a single frame as RGB array."""
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')
    fig.tight_layout()

    # Convert to RGB array using buffer_rgba
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    img = buf[:, :, :3]  # Drop alpha channel
    plt.close(fig)
    return img.copy()


def main():
    """Run tweezer rearrangement demo."""
    print("Optical Tweezer Rearrangement Demo")
    print("=" * 40)

    # Setup
    input_amp = create_gaussian_input(GRID_SIZE)
    start_pos, end_pos = generate_start_end_positions(SEED)
    assignment = solve_assignment(start_pos, end_pos)

    print(f"Start: {len(start_pos)} random spots in 11x11 grid")
    print(f"End: Circular compact around center")
    print(f"Frames: {N_FRAMES}")
    print(f"GS iterations per frame: {GS_ITERATIONS}")

    # Storage for frames
    phase_frames = []
    intensity_frames = []

    # Initial phase (random)
    current_phase = np.random.uniform(-np.pi, np.pi, (GRID_SIZE, GRID_SIZE))

    # Generate frames
    for frame in range(N_FRAMES):
        t = frame / (N_FRAMES - 1)  # 0 to 1

        # Interpolate positions
        positions = interpolate_positions(start_pos, end_pos, assignment, t)
        target_amp = create_spot_pattern(positions, GRID_SIZE, SPOT_RADIUS)

        # Run WGS with warm start
        result = weighted_gs(input_amp, target_amp, GS_ITERATIONS, current_phase)
        current_phase = result.phase_mask  # Warm start for next frame

        # Render frames
        phase_img = render_frame(
            result.phase_mask,
            f"Phase Mask (t={t:.2f})",
            'twilight',
            vmin=-np.pi, vmax=np.pi
        )
        intensity_img = render_frame(
            result.reconstructed,
            f"Intensity (t={t:.2f})",
            'hot'
        )

        phase_frames.append(phase_img)
        intensity_frames.append(intensity_img)

        if (frame + 1) % 10 == 0:
            print(f"  Frame {frame + 1}/{N_FRAMES} complete")

    # Save GIFs
    print("\nSaving GIFs...")
    imageio.mimsave('tweezer_phase.gif', phase_frames, fps=FPS, loop=0)
    imageio.mimsave('tweezer_intensity.gif', intensity_frames, fps=FPS, loop=0)

    print("Saved: tweezer_phase.gif")
    print("Saved: tweezer_intensity.gif")


if __name__ == "__main__":
    main()

