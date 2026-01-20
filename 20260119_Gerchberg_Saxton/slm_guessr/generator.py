"""
Sample Generator for SLM-Guessr

Generates animated GIFs and samples.json manifest for the training gallery.
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Callable
import sys

# Add parent dir for gs_algorithms import
sys.path.insert(0, str(Path(__file__).parent.parent))
from gs_algorithms import standard_gs

from .patterns import (
    create_gaussian_input,
    create_uniform_phase,
    create_linear_ramp,
    create_quadratic_phase,
    create_cubic_phase,
    create_spot_target,
    create_gaussian_spot_target,
    create_rectangular_slab_target,
    compute_intensity,
)

# Use PIL for GIF generation
from PIL import Image


GRID_SIZE = 256
GS_ITERATIONS = 50
GIF_DURATION_MS = 100  # ms per frame


@dataclass
class SampleConfig:
    """Configuration for a training sample."""
    id: str
    level: int
    category: str
    name: str
    description: str
    generator: Callable  # Function that returns list of (phase, intensity) frames
    parameters: dict = None


def normalize_for_image(arr: np.ndarray, is_phase: bool = False) -> np.ndarray:
    """Normalize array to 0-255 uint8 for image saving."""
    if is_phase:
        # Phase: map [-pi, pi] to [0, 255]
        normalized = (arr + np.pi) / (2 * np.pi)
    else:
        # Intensity: normalize to [0, 1] then scale
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            normalized = (arr - arr_min) / (arr_max - arr_min)
        else:
            normalized = np.zeros_like(arr)
    return (normalized * 255).astype(np.uint8)


def save_gif(frames: List[np.ndarray], path: Path, is_phase: bool = False):
    """Save list of arrays as animated GIF."""
    images = []
    for frame in frames:
        img_data = normalize_for_image(frame, is_phase)
        if is_phase:
            # Use twilight-like colormap for phase
            img = Image.fromarray(img_data, mode='L')
            img = img.convert('P')
            # Apply custom phase colormap (twilight-inspired)
            palette = []
            for i in range(256):
                t = i / 255.0
                # Twilight-like: purple -> blue -> white -> orange -> purple
                if t < 0.25:
                    r = int(100 + 155 * (t / 0.25))
                    g = int(50 * (t / 0.25))
                    b = int(150 + 105 * (t / 0.25))
                elif t < 0.5:
                    r = 255
                    g = int(50 + 205 * ((t - 0.25) / 0.25))
                    b = int(255 - 100 * ((t - 0.25) / 0.25))
                elif t < 0.75:
                    r = int(255 - 50 * ((t - 0.5) / 0.25))
                    g = int(255 - 50 * ((t - 0.5) / 0.25))
                    b = int(155 - 55 * ((t - 0.5) / 0.25))
                else:
                    r = int(205 - 105 * ((t - 0.75) / 0.25))
                    g = int(205 - 155 * ((t - 0.75) / 0.25))
                    b = int(100 + 50 * ((t - 0.75) / 0.25))
                palette.extend([r, g, b])
            img.putpalette(palette)
        else:
            # Hot colormap for intensity
            img = Image.fromarray(img_data, mode='L')
            img = img.convert('P')
            palette = []
            for i in range(256):
                t = i / 255.0
                r = int(min(255, t * 3 * 255))
                g = int(min(255, max(0, (t - 0.33) * 3 * 255)))
                b = int(min(255, max(0, (t - 0.67) * 3 * 255)))
                palette.extend([r, g, b])
            img.putpalette(palette)
        images.append(img)

    if len(images) == 1:
        images[0].save(path, save_all=False)
    else:
        images[0].save(
            path,
            save_all=True,
            append_images=images[1:],
            duration=GIF_DURATION_MS,
            loop=0
        )


def generate_sample(
    config: SampleConfig,
    output_dir: Path,
    input_amp: np.ndarray
) -> dict:
    """
    Generate a single sample with phase and intensity GIFs.

    Returns:
        Sample manifest entry dict
    """
    level_dir = output_dir / f"L{config.level}"
    level_dir.mkdir(parents=True, exist_ok=True)

    # Generate frames
    frames = config.generator(input_amp)
    phase_frames = [f[0] for f in frames]
    intensity_frames = [f[1] for f in frames]

    # Save GIFs
    phase_path = level_dir / f"{config.id}_phase.gif"
    intensity_path = level_dir / f"{config.id}_intensity.gif"

    save_gif(phase_frames, phase_path, is_phase=True)
    save_gif(intensity_frames, intensity_path, is_phase=False)

    return {
        "id": config.id,
        "level": config.level,
        "category": config.category,
        "name": config.name,
        "description": config.description,
        "phase_gif": f"assets/L{config.level}/{config.id}_phase.gif",
        "intensity_gif": f"assets/L{config.level}/{config.id}_intensity.gif",
        "parameters": config.parameters or {},
    }


# =============================================================================
# Level 1: Foundations - Sample Generators
# =============================================================================

def gen_uniform_phase(input_amp: np.ndarray):
    """Uniform (zero) phase - static baseline."""
    phase = create_uniform_phase(GRID_SIZE)
    intensity = compute_intensity(input_amp, phase)
    return [(phase, intensity)]


def gen_spot_sweep_x(input_amp: np.ndarray):
    """Single spot sweeping left to right."""
    frames = []
    n_frames = 16
    max_offset = GRID_SIZE // 6
    for i in range(n_frames):
        cx = -max_offset + (2 * max_offset * i) / (n_frames - 1)
        target = create_spot_target(GRID_SIZE, cx, 0, radius=4)
        result = standard_gs(input_amp, target, GS_ITERATIONS)
        frames.append((result.phase_mask, result.reconstructed))
    return frames


def gen_spot_sweep_y(input_amp: np.ndarray):
    """Single spot sweeping bottom to top."""
    frames = []
    n_frames = 16
    max_offset = GRID_SIZE // 6
    for i in range(n_frames):
        cy = -max_offset + (2 * max_offset * i) / (n_frames - 1)
        target = create_spot_target(GRID_SIZE, 0, cy, radius=4)
        result = standard_gs(input_amp, target, GS_ITERATIONS)
        frames.append((result.phase_mask, result.reconstructed))
    return frames


def gen_spot_circular(input_amp: np.ndarray):
    """Single spot moving in circle."""
    frames = []
    n_frames = 24
    radius = GRID_SIZE // 8
    for i in range(n_frames):
        angle = 2 * np.pi * i / n_frames
        cx = radius * np.cos(angle)
        cy = radius * np.sin(angle)
        target = create_spot_target(GRID_SIZE, cx, cy, radius=4)
        result = standard_gs(input_amp, target, GS_ITERATIONS)
        frames.append((result.phase_mask, result.reconstructed))
    return frames


def gen_gaussian_spot_sweep(input_amp: np.ndarray):
    """Gaussian spot (soft edges) sweeping."""
    frames = []
    n_frames = 16
    max_offset = GRID_SIZE // 6
    for i in range(n_frames):
        cx = -max_offset + (2 * max_offset * i) / (n_frames - 1)
        target = create_gaussian_spot_target(GRID_SIZE, cx, 0, sigma=6)
        result = standard_gs(input_amp, target, GS_ITERATIONS)
        frames.append((result.phase_mask, result.reconstructed))
    return frames


def gen_slab_sweep_x(input_amp: np.ndarray):
    """Rectangular slab sweeping horizontally."""
    frames = []
    n_frames = 16
    max_offset = GRID_SIZE // 6
    for i in range(n_frames):
        cx = -max_offset + (2 * max_offset * i) / (n_frames - 1)
        target = create_rectangular_slab_target(GRID_SIZE, cx, 0, width=15, height=30)
        result = standard_gs(input_amp, target, GS_ITERATIONS)
        frames.append((result.phase_mask, result.reconstructed))
    return frames


def gen_slab_sweep_y(input_amp: np.ndarray):
    """Rectangular slab sweeping vertically."""
    frames = []
    n_frames = 16
    max_offset = GRID_SIZE // 6
    for i in range(n_frames):
        cy = -max_offset + (2 * max_offset * i) / (n_frames - 1)
        target = create_rectangular_slab_target(GRID_SIZE, 0, cy, width=15, height=30)
        result = standard_gs(input_amp, target, GS_ITERATIONS)
        frames.append((result.phase_mask, result.reconstructed))
    return frames


def gen_linear_ramp_x(input_amp: np.ndarray):
    """Linear phase ramp in X, kx sweep."""
    frames = []
    n_frames = 16
    for i in range(n_frames):
        kx = 2 * np.pi * i / (n_frames - 1) / (GRID_SIZE / 8)
        phase = create_linear_ramp(GRID_SIZE, kx=kx, ky=0)
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


def gen_linear_ramp_y(input_amp: np.ndarray):
    """Linear phase ramp in Y, ky sweep."""
    frames = []
    n_frames = 16
    for i in range(n_frames):
        ky = 2 * np.pi * i / (n_frames - 1) / (GRID_SIZE / 8)
        phase = create_linear_ramp(GRID_SIZE, kx=0, ky=ky)
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


def gen_linear_ramp_diagonal(input_amp: np.ndarray):
    """Linear phase ramp with rotating direction."""
    frames = []
    n_frames = 16
    k_mag = 2 * np.pi / (GRID_SIZE / 8)
    for i in range(n_frames):
        angle = np.pi / 4 * i / (n_frames - 1)  # 0 to 45 deg
        kx = k_mag * np.cos(angle)
        ky = k_mag * np.sin(angle)
        phase = create_linear_ramp(GRID_SIZE, kx=kx, ky=ky)
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


def gen_quadratic_positive(input_amp: np.ndarray):
    """Quadratic phase (positive curvature) sweep."""
    frames = []
    n_frames = 16
    for i in range(n_frames):
        curvature = 0.5 + 3.0 * i / (n_frames - 1)
        phase = create_quadratic_phase(GRID_SIZE, curvature)
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


def gen_quadratic_negative(input_amp: np.ndarray):
    """Quadratic phase (negative curvature) sweep."""
    frames = []
    n_frames = 16
    for i in range(n_frames):
        curvature = -0.5 - 3.0 * i / (n_frames - 1)
        phase = create_quadratic_phase(GRID_SIZE, curvature)
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


def gen_cubic_x(input_amp: np.ndarray):
    """Cubic phase in X sweep."""
    frames = []
    n_frames = 16
    for i in range(n_frames):
        coeff = 1.0 + 4.0 * i / (n_frames - 1)
        phase = create_cubic_phase(GRID_SIZE, coeff_x=coeff, coeff_y=0)
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


def gen_cubic_y(input_amp: np.ndarray):
    """Cubic phase in Y sweep."""
    frames = []
    n_frames = 16
    for i in range(n_frames):
        coeff = 1.0 + 4.0 * i / (n_frames - 1)
        phase = create_cubic_phase(GRID_SIZE, coeff_x=0, coeff_y=coeff)
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames



# =============================================================================
# Sample Configuration Registry
# =============================================================================

L1_SAMPLES = [
    SampleConfig(
        id="uniform_phase",
        level=1,
        category="foundations",
        name="Uniform Phase",
        description="Zero phase everywhere - baseline reference showing unmodified Gaussian",
        generator=gen_uniform_phase,
    ),
    SampleConfig(
        id="spot_sweep_x",
        level=1,
        category="foundations",
        name="Single Spot Sweep (X)",
        description="Spot moves left to right - observe phase ramp tilting",
        generator=gen_spot_sweep_x,
    ),
    SampleConfig(
        id="spot_sweep_y",
        level=1,
        category="foundations",
        name="Single Spot Sweep (Y)",
        description="Spot moves bottom to top - vertical phase ramp",
        generator=gen_spot_sweep_y,
    ),
    SampleConfig(
        id="spot_circular",
        level=1,
        category="foundations",
        name="Single Spot Circular",
        description="Spot moves in circle - phase ramp rotates",
        generator=gen_spot_circular,
    ),
    SampleConfig(
        id="gaussian_spot_sweep",
        level=1,
        category="foundations",
        name="Gaussian Spot Sweep",
        description="Soft Gaussian spot - no sinc ringing due to smooth edges",
        generator=gen_gaussian_spot_sweep,
    ),
    SampleConfig(
        id="slab_sweep_x",
        level=1,
        category="foundations",
        name="Rectangular Slab Sweep (X)",
        description="Rectangle moves horizontally - observe sinc envelope",
        generator=gen_slab_sweep_x,
    ),
    SampleConfig(
        id="slab_sweep_y",
        level=1,
        category="foundations",
        name="Rectangular Slab Sweep (Y)",
        description="Rectangle moves vertically - sinc in orthogonal direction",
        generator=gen_slab_sweep_y,
    ),
    SampleConfig(
        id="linear_ramp_x",
        level=1,
        category="foundations",
        name="Linear Ramp (X)",
        description="Phase gradient in X shifts beam horizontally in Fourier plane",
        generator=gen_linear_ramp_x,
    ),
    SampleConfig(
        id="linear_ramp_y",
        level=1,
        category="foundations",
        name="Linear Ramp (Y)",
        description="Phase gradient in Y shifts beam vertically in Fourier plane",
        generator=gen_linear_ramp_y,
    ),
    SampleConfig(
        id="linear_ramp_diagonal",
        level=1,
        category="foundations",
        name="Linear Ramp (Diagonal)",
        description="Phase ramp direction rotates from X toward diagonal",
        generator=gen_linear_ramp_diagonal,
    ),
    SampleConfig(
        id="quadratic_positive",
        level=1,
        category="foundations",
        name="Quadratic Phase (+)",
        description="Positive curvature (converging lens) - ring pattern expands",
        generator=gen_quadratic_positive,
    ),
    SampleConfig(
        id="quadratic_negative",
        level=1,
        category="foundations",
        name="Quadratic Phase (-)",
        description="Negative curvature (diverging lens) - ring pattern",
        generator=gen_quadratic_negative,
    ),
    SampleConfig(
        id="cubic_x",
        level=1,
        category="foundations",
        name="Cubic Phase (X)",
        description="Cubic phase in X - Airy-like asymmetric pattern",
        generator=gen_cubic_x,
    ),
    SampleConfig(
        id="cubic_y",
        level=1,
        category="foundations",
        name="Cubic Phase (Y)",
        description="Cubic phase in Y - rotated Airy pattern",
        generator=gen_cubic_y,
    ),
]


def get_all_samples() -> List[SampleConfig]:
    """Get all sample configurations."""
    return L1_SAMPLES  # Add more levels here as implemented


def generate_all_samples(output_dir: Path) -> dict:
    """
    Generate all training samples.

    Args:
        output_dir: Base output directory for assets

    Returns:
        Complete manifest dict
    """
    input_amp = create_gaussian_input(GRID_SIZE)
    samples = get_all_samples()
    manifest_entries = []

    print(f"Generating {len(samples)} samples...")

    for i, config in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] {config.name}...")
        entry = generate_sample(config, output_dir, input_amp)
        manifest_entries.append(entry)

    manifest = {
        "samples": manifest_entries,
        "generated_at": datetime.now().isoformat(),
        "version": "1.0.0",
    }

    # Save manifest
    manifest_path = output_dir.parent / "samples.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved manifest: {manifest_path}")
    return manifest
