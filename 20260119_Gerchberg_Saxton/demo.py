"""
GS Algorithm Comparison Demo

Generates side-by-side comparison of:
1. Standard GS
2. Weighted GS (WGS)
3. GS with Random Phase Reset

For two target patterns:
- 4x4 spot array
- Letter "A" shape
"""

import numpy as np
import matplotlib.pyplot as plt
from gs_algorithms import standard_gs, weighted_gs, gs_random_reset, GSResult


# --- Configuration ---
GRID_SIZE = 256
N_ITERATIONS = 100
SEED = 42


def create_gaussian_input(size: int, sigma: float = None) -> np.ndarray:
    """Create Gaussian input beam amplitude."""
    if sigma is None:
        sigma = size / 4
    x = np.linspace(-size // 2, size // 2, size)
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X**2 + Y**2) / (2 * sigma**2))


def create_spot_array(size: int, n_spots: int = 4, spot_radius: int = 3) -> np.ndarray:
    """Create n_spots x n_spots grid of spots."""
    target = np.zeros((size, size))
    spacing = size // (n_spots + 1)
    for i in range(n_spots):
        for j in range(n_spots):
            cx = spacing * (i + 1)
            cy = spacing * (j + 1)
            y, x = np.ogrid[:size, :size]
            mask = (x - cx)**2 + (y - cy)**2 <= spot_radius**2
            target[mask] = 1.0
    return target


def create_letter_a(size: int) -> np.ndarray:
    """Create letter A pattern."""
    target = np.zeros((size, size))
    center = size // 2
    height = size // 3
    width = size // 5
    thickness = max(3, size // 40)
    
    # Left leg
    for i in range(height):
        x = center - width // 2 + (i * width) // (2 * height)
        y = center - height // 2 + i
        target[max(0,y-thickness):min(size,y+thickness), 
               max(0,x-thickness):min(size,x+thickness)] = 1.0
    
    # Right leg
    for i in range(height):
        x = center + width // 2 - (i * width) // (2 * height)
        y = center - height // 2 + i
        target[max(0,y-thickness):min(size,y+thickness), 
               max(0,x-thickness):min(size,x+thickness)] = 1.0
    
    # Crossbar
    crossbar_y = center
    target[crossbar_y-thickness:crossbar_y+thickness,
           center-width//3:center+width//3] = 1.0
    
    return target


def compute_uniformity(intensity: np.ndarray, target: np.ndarray) -> float:
    """Compute coefficient of variation (lower is more uniform)."""
    mask = target > 0
    if mask.sum() == 0:
        return 0.0
    values = intensity[mask]
    if values.mean() == 0:
        return float('inf')
    return values.std() / values.mean()


def run_comparison(
    input_amp: np.ndarray,
    target_amp: np.ndarray,
    target_name: str,
    initial_phase: np.ndarray,
) -> dict[str, GSResult]:
    """Run all algorithms on the same target."""
    results = {}
    
    print(f"  Running Standard GS...")
    results["Standard GS"] = standard_gs(
        input_amp, target_amp, N_ITERATIONS, initial_phase.copy()
    )
    
    print(f"  Running Weighted GS...")
    results["Weighted GS"] = weighted_gs(
        input_amp, target_amp, N_ITERATIONS, initial_phase.copy()
    )
    
    print(f"  Running GS + Reset...")
    results["GS + Reset"] = gs_random_reset(
        input_amp, target_amp, N_ITERATIONS, initial_phase.copy()
    )

    return results


def plot_results(
    target_amp: np.ndarray,
    results: dict[str, GSResult],
    target_name: str,
    axes_row: list,
):
    """Plot results for one target pattern across one row of axes."""
    algo_names = list(results.keys())

    # Column 0: Target
    axes_row[0].imshow(target_amp**2, cmap='hot')
    axes_row[0].set_title(f"Target\n({target_name})")
    axes_row[0].axis('off')

    # Columns 1-3: Phase masks
    for i, name in enumerate(algo_names):
        axes_row[1 + i].imshow(results[name].phase_mask, cmap='twilight', vmin=-np.pi, vmax=np.pi)
        axes_row[1 + i].set_title(f"Phase\n{name}")
        axes_row[1 + i].axis('off')

    # Columns 4-6: Reconstructed intensity
    for i, name in enumerate(algo_names):
        axes_row[4 + i].imshow(results[name].reconstructed, cmap='hot')
        uniformity = compute_uniformity(results[name].reconstructed, target_amp)
        axes_row[4 + i].set_title(f"Recon\n{name}\nCV={uniformity:.3f}")
        axes_row[4 + i].axis('off')

    # Column 7: Error curves
    for name in algo_names:
        axes_row[7].plot(results[name].errors, label=name)
    axes_row[7].set_xlabel("Iteration")
    axes_row[7].set_ylabel("Error")
    axes_row[7].set_title("Convergence")
    axes_row[7].legend(fontsize=6)
    axes_row[7].grid(True, alpha=0.3)


def main():
    """Run full comparison demo."""
    np.random.seed(SEED)

    print("GS Algorithm Comparison Demo")
    print("=" * 40)

    # Create input beam
    input_amp = create_gaussian_input(GRID_SIZE)

    # Create targets
    targets = {
        "4x4 Spots": create_spot_array(GRID_SIZE),
        "Letter A": create_letter_a(GRID_SIZE),
    }

    # Create figure
    fig, axes = plt.subplots(2, 8, figsize=(20, 6))
    fig.suptitle("Gerchberg-Saxton Algorithm Comparison", fontsize=14, fontweight='bold')

    # Run comparison for each target
    for row_idx, (target_name, target_amp) in enumerate(targets.items()):
        print(f"\nTarget: {target_name}")

        # Same initial phase for fair comparison
        initial_phase = np.random.uniform(-np.pi, np.pi, (GRID_SIZE, GRID_SIZE))

        results = run_comparison(input_amp, target_amp, target_name, initial_phase)
        plot_results(target_amp, results, target_name, axes[row_idx])

        # Print metrics
        print(f"  Final errors:")
        for name, result in results.items():
            uniformity = compute_uniformity(result.reconstructed, target_amp)
            print(f"    {name}: error={result.errors[-1]:.4f}, CV={uniformity:.4f}")

    plt.tight_layout()
    plt.savefig("results.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved: results.png")
    plt.close()


if __name__ == "__main__":
    main()

