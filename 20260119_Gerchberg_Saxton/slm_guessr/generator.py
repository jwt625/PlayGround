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
    create_binary_grating,
    create_sinusoidal_grating,
    create_blazed_grating,
    create_checkerboard,
    create_crossed_gratings,
    create_multi_frequency_grating,
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
    duration_ms: int = None  # Override GIF frame duration (default: GIF_DURATION_MS)


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
    """Linear phase ramp in X, kx sweep. 2x faster, 2x longer, 4x smaller end pitch."""
    frames = []
    n_frames = 32  # 2x more frames
    for i in range(n_frames):
        # End at 4x smaller pitch (4x larger k)
        kx = 2 * np.pi * i / (n_frames - 1) / (GRID_SIZE / 32)
        phase = create_linear_ramp(GRID_SIZE, kx=kx, ky=0)
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


def gen_linear_ramp_y(input_amp: np.ndarray):
    """Linear phase ramp in Y, ky sweep. 2x faster, 2x longer, 4x smaller end pitch."""
    frames = []
    n_frames = 32  # 2x more frames
    for i in range(n_frames):
        # End at 4x smaller pitch (4x larger k)
        ky = 2 * np.pi * i / (n_frames - 1) / (GRID_SIZE / 32)
        phase = create_linear_ramp(GRID_SIZE, kx=0, ky=ky)
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


def gen_linear_ramp_diagonal(input_amp: np.ndarray):
    """Linear phase ramp with rotating direction. 3x smaller pitch."""
    frames = []
    n_frames = 16
    # 3x smaller pitch = 3x larger k_mag
    k_mag = 2 * np.pi / (GRID_SIZE / 24)
    for i in range(n_frames):
        angle = np.pi / 4 * i / (n_frames - 1)  # 0 to 45 deg
        kx = k_mag * np.cos(angle)
        ky = k_mag * np.sin(angle)
        phase = create_linear_ramp(GRID_SIZE, kx=kx, ky=ky)
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


def gen_quadratic_positive(input_amp: np.ndarray):
    """Quadratic phase (positive curvature) sweep. 4x longer."""
    frames = []
    n_frames = 64  # 4x more frames
    for i in range(n_frames):
        curvature = 0.5 + 3.0 * i / (n_frames - 1)
        phase = create_quadratic_phase(GRID_SIZE, curvature)
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


def gen_quadratic_negative(input_amp: np.ndarray):
    """Quadratic phase (negative curvature) sweep. 4x longer."""
    frames = []
    n_frames = 64  # 4x more frames
    for i in range(n_frames):
        curvature = -0.5 - 3.0 * i / (n_frames - 1)
        phase = create_quadratic_phase(GRID_SIZE, curvature)
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


def gen_cubic_x(input_amp: np.ndarray):
    """Cubic phase in X sweep. 4x longer."""
    frames = []
    n_frames = 64  # 4x more frames
    for i in range(n_frames):
        coeff = 1.0 + 4.0 * i / (n_frames - 1)
        phase = create_cubic_phase(GRID_SIZE, coeff_x=coeff, coeff_y=0)
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


def gen_cubic_y(input_amp: np.ndarray):
    """Cubic phase in Y sweep. 4x longer."""
    frames = []
    n_frames = 64  # 4x more frames
    for i in range(n_frames):
        coeff = 1.0 + 4.0 * i / (n_frames - 1)
        phase = create_cubic_phase(GRID_SIZE, coeff_x=0, coeff_y=coeff)
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


def gen_coherent_aperture(input_amp: np.ndarray):
    """Variable coherent aperture - demonstrates spot size vs aperture."""
    frames = []
    n_frames = 16
    x = np.linspace(-GRID_SIZE // 2, GRID_SIZE // 2, GRID_SIZE)
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X**2 + Y**2)
    # Fixed random phase for outer region (seeded for consistency)
    rng = np.random.RandomState(42)
    random_phase = rng.uniform(-np.pi, np.pi, (GRID_SIZE, GRID_SIZE))

    for i in range(n_frames):
        # Radius from small to large (small aperture = large spot)
        # Start at 10px (2x smaller than before) up to 100px
        radius = 10 + 90 * i / (n_frames - 1)  # 10 to 100 pixels
        # Flat phase inside, random outside
        phase = np.where(R <= radius, 0.0, random_phase)
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


def gen_soft_aperture(input_amp: np.ndarray):
    """Soft-edged aperture - demonstrates edge smoothness effect."""
    frames = []
    n_frames = 16
    x = np.linspace(-GRID_SIZE // 2, GRID_SIZE // 2, GRID_SIZE)
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X**2 + Y**2)
    rng = np.random.RandomState(42)
    random_phase = rng.uniform(-np.pi, np.pi, (GRID_SIZE, GRID_SIZE))

    radius = 30  # Fixed radius (2x smaller than before)
    for i in range(n_frames):
        # Edge softness from sharp to very soft
        softness = 1 + 30 * i / (n_frames - 1)  # 1 to 31 pixels
        # Soft transition using sigmoid-like function
        mask = 1.0 / (1.0 + np.exp((R - radius) / softness))
        # Blend flat phase (inside) with random phase (outside)
        phase = mask * 0.0 + (1 - mask) * random_phase
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


# =============================================================================
# L2: Periodic Structures Frame Generators
# =============================================================================

def gen_binary_grating_vertical(input_amp: np.ndarray):
    """Binary grating vertical, period sweep."""
    frames = []
    n_frames = 16
    for i in range(n_frames):
        period = 64 - 56 * i / (n_frames - 1)  # 64 to 8 pixels
        phase = create_binary_grating(GRID_SIZE, period=period, angle=0)
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


def gen_binary_grating_horizontal(input_amp: np.ndarray):
    """Binary grating horizontal, period sweep."""
    frames = []
    n_frames = 16
    for i in range(n_frames):
        period = 64 - 56 * i / (n_frames - 1)  # 64 to 8 pixels
        phase = create_binary_grating(GRID_SIZE, period=period, angle=np.pi/2)
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


def gen_binary_grating_rotated(input_amp: np.ndarray):
    """Binary grating with rotating angle."""
    frames = []
    n_frames = 16
    period = 24  # Smaller pitch for rotation demo
    for i in range(n_frames):
        angle = np.pi / 2 * i / (n_frames - 1)  # 0 to 90 deg
        phase = create_binary_grating(GRID_SIZE, period=period, angle=angle)
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


def gen_sinusoidal_grating(input_amp: np.ndarray):
    """Sinusoidal grating, period sweep."""
    frames = []
    n_frames = 16
    for i in range(n_frames):
        period = 64 - 56 * i / (n_frames - 1)  # 64 to 8 pixels
        phase = create_sinusoidal_grating(GRID_SIZE, period=period, angle=0)
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


def gen_blazed_grating(input_amp: np.ndarray):
    """Blazed grating, period sweep."""
    frames = []
    n_frames = 16
    for i in range(n_frames):
        period = 64 - 56 * i / (n_frames - 1)  # 64 to 8 pixels
        phase = create_blazed_grating(GRID_SIZE, period=period, angle=0)
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


def gen_checkerboard(input_amp: np.ndarray):
    """Checkerboard pattern, period sweep."""
    frames = []
    n_frames = 16
    for i in range(n_frames):
        period = 64 - 56 * i / (n_frames - 1)  # 64 to 8 pixels
        phase = create_checkerboard(GRID_SIZE, period=period)
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


def gen_crossed_gratings(input_amp: np.ndarray):
    """Crossed gratings with rotating angle."""
    frames = []
    n_frames = 16
    period = 16  # 2x smaller pitch
    for i in range(n_frames):
        angle = np.pi / 4 * i / (n_frames - 1)  # 0 to 45 deg
        phase = create_crossed_gratings(GRID_SIZE, period=period, angle=angle)
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


def gen_multi_frequency_grating(input_amp: np.ndarray):
    """Multi-frequency grating, ratio sweep."""
    frames = []
    n_frames = 16
    period1 = 32
    for i in range(n_frames):
        ratio = 2 + 3 * i / (n_frames - 1)  # period ratio from 2 to 5
        period2 = period1 / ratio
        phase = create_multi_frequency_grating(GRID_SIZE, period1=period1, period2=period2)
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
    SampleConfig(
        id="coherent_aperture",
        level=1,
        category="foundations",
        name="Coherent Aperture Size",
        description="Varying coherent region size - smaller aperture = larger spot (Fourier relationship)",
        generator=gen_coherent_aperture,
    ),
    SampleConfig(
        id="soft_aperture",
        level=1,
        category="foundations",
        name="Soft Aperture Edge",
        description="Edge smoothness sweep - softer edges reduce ringing/sidelobes",
        generator=gen_soft_aperture,
    ),
]

L2_SAMPLES = [
    SampleConfig(
        id="binary_grating_vertical",
        level=2,
        category="periodic",
        name="Binary Grating (Vertical)",
        description="Vertical binary grating - period decreases, diffraction orders separate",
        generator=gen_binary_grating_vertical,
    ),
    SampleConfig(
        id="binary_grating_horizontal",
        level=2,
        category="periodic",
        name="Binary Grating (Horizontal)",
        description="Horizontal binary grating - diffraction in vertical direction",
        generator=gen_binary_grating_horizontal,
    ),
    SampleConfig(
        id="binary_grating_rotated",
        level=2,
        category="periodic",
        name="Binary Grating (Rotating)",
        description="Binary grating rotates 0° to 90° - watch diffraction spots rotate",
        generator=gen_binary_grating_rotated,
    ),
    SampleConfig(
        id="sinusoidal_grating",
        level=2,
        category="periodic",
        name="Sinusoidal Grating",
        description="Sinusoidal phase grating - only ±1 orders, no higher harmonics",
        generator=gen_sinusoidal_grating,
    ),
    SampleConfig(
        id="blazed_grating",
        level=2,
        category="periodic",
        name="Blazed Grating",
        description="Sawtooth phase grating - asymmetric diffraction, power in +1 order",
        generator=gen_blazed_grating,
    ),
    SampleConfig(
        id="checkerboard",
        level=2,
        category="periodic",
        name="Checkerboard",
        description="2D periodic pattern - 4-spot diffraction pattern",
        generator=gen_checkerboard,
    ),
    SampleConfig(
        id="crossed_gratings",
        level=2,
        category="periodic",
        name="Crossed Gratings",
        description="Two perpendicular gratings - XOR pattern creates grid diffraction",
        generator=gen_crossed_gratings,
    ),
    SampleConfig(
        id="multi_frequency_grating",
        level=2,
        category="periodic",
        name="Multi-Frequency Grating",
        description="Superposition of two frequencies - multiple diffraction orders",
        generator=gen_multi_frequency_grating,
    ),
]


def get_all_samples() -> List[SampleConfig]:
    """Get all sample configurations."""
    return L1_SAMPLES + L2_SAMPLES


def generate_selected_samples(output_dir: Path, sample_ids: List[str]) -> dict:
    """
    Generate selected training samples by ID.

    Args:
        output_dir: Base output directory for assets
        sample_ids: List of sample IDs to generate

    Returns:
        Manifest dict with generated samples
    """
    input_amp = create_gaussian_input(GRID_SIZE)
    all_samples = get_all_samples()
    samples_by_id = {s.id: s for s in all_samples}

    # Validate IDs
    invalid_ids = [sid for sid in sample_ids if sid not in samples_by_id]
    if invalid_ids:
        print(f"Warning: Unknown sample IDs: {invalid_ids}")

    samples_to_gen = [samples_by_id[sid] for sid in sample_ids if sid in samples_by_id]
    manifest_entries = []

    print(f"Generating {len(samples_to_gen)} samples...")

    for i, config in enumerate(samples_to_gen):
        print(f"  [{i+1}/{len(samples_to_gen)}] {config.name}...")
        entry = generate_sample(config, output_dir, input_amp)
        manifest_entries.append(entry)

    manifest = {
        "samples": manifest_entries,
        "generated_at": datetime.now().isoformat(),
        "version": "1.0.0",
    }

    # Note: Don't overwrite full manifest when generating subset
    print(f"Generated {len(manifest_entries)} samples (manifest not updated)")
    return manifest


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
