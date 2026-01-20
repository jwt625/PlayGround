"""
Pattern Generators for SLM-Guessr

Phase mask and target intensity pattern generators for training samples.
"""

import numpy as np
from typing import Tuple


def create_gaussian_input(size: int, sigma: float = None) -> np.ndarray:
    """Create Gaussian input beam amplitude."""
    if sigma is None:
        sigma = size / 4
    x = np.linspace(-size // 2, size // 2, size)
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X**2 + Y**2) / (2 * sigma**2))


def create_uniform_phase(size: int) -> np.ndarray:
    """Create uniform (zero) phase mask."""
    return np.zeros((size, size))


def create_linear_ramp(
    size: int, kx: float = 0, ky: float = 0
) -> np.ndarray:
    """
    Create linear phase ramp.
    
    Args:
        size: Grid size
        kx: Spatial frequency in x (radians per pixel)
        ky: Spatial frequency in y (radians per pixel)
    
    Returns:
        Phase mask wrapped to [-pi, pi]
    """
    x = np.linspace(-size // 2, size // 2, size)
    X, Y = np.meshgrid(x, x)
    phase = kx * X + ky * Y
    return np.mod(phase + np.pi, 2 * np.pi) - np.pi


def create_quadratic_phase(size: int, curvature: float) -> np.ndarray:
    """
    Create quadratic (lens-like) phase.
    
    Args:
        size: Grid size
        curvature: Curvature coefficient (positive = converging, negative = diverging)
    
    Returns:
        Phase mask wrapped to [-pi, pi]
    """
    x = np.linspace(-size // 2, size // 2, size)
    X, Y = np.meshgrid(x, x)
    r2 = X**2 + Y**2
    phase = curvature * r2 / (size**2) * 4 * np.pi
    return np.mod(phase + np.pi, 2 * np.pi) - np.pi


def create_cubic_phase(
    size: int, coeff_x: float = 0, coeff_y: float = 0
) -> np.ndarray:
    """
    Create cubic phase pattern.
    
    Args:
        size: Grid size
        coeff_x: Cubic coefficient in x
        coeff_y: Cubic coefficient in y
    
    Returns:
        Phase mask wrapped to [-pi, pi]
    """
    x = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, x)
    phase = coeff_x * X**3 + coeff_y * Y**3
    phase = phase * np.pi
    return np.mod(phase + np.pi, 2 * np.pi) - np.pi


def create_spot_target(
    size: int, cx: float, cy: float, radius: int = 3
) -> np.ndarray:
    """
    Create single spot target intensity.
    
    Args:
        size: Grid size
        cx: Center x position (pixels from center)
        cy: Center y position (pixels from center)
        radius: Spot radius in pixels
    
    Returns:
        Target amplitude (sqrt of intensity)
    """
    x = np.arange(size) - size // 2
    X, Y = np.meshgrid(x, x)
    r2 = (X - cx)**2 + (Y - cy)**2
    target = np.zeros((size, size))
    target[r2 <= radius**2] = 1.0
    return target


def create_gaussian_spot_target(
    size: int, cx: float, cy: float, sigma: float = 5.0
) -> np.ndarray:
    """
    Create Gaussian spot target (soft edges, no sinc ringing).
    
    Args:
        size: Grid size
        cx: Center x position (pixels from center)
        cy: Center y position (pixels from center)
        sigma: Gaussian width in pixels
    
    Returns:
        Target amplitude
    """
    x = np.arange(size) - size // 2
    X, Y = np.meshgrid(x, x)
    r2 = (X - cx)**2 + (Y - cy)**2
    return np.exp(-r2 / (2 * sigma**2))


def create_rectangular_slab_target(
    size: int, cx: float, cy: float, width: int = 20, height: int = 40
) -> np.ndarray:
    """
    Create rectangular slab target.
    
    Args:
        size: Grid size
        cx: Center x position (pixels from center)
        cy: Center y position (pixels from center)
        width: Rectangle width in pixels
        height: Rectangle height in pixels
    
    Returns:
        Target amplitude
    """
    x = np.arange(size) - size // 2
    X, Y = np.meshgrid(x, x)
    target = np.zeros((size, size))
    mask = (np.abs(X - cx) <= width // 2) & (np.abs(Y - cy) <= height // 2)
    target[mask] = 1.0
    return target


def compute_intensity(
    input_amplitude: np.ndarray, phase_mask: np.ndarray
) -> np.ndarray:
    """
    Compute Fourier plane intensity from phase mask.
    
    Args:
        input_amplitude: Input beam amplitude
        phase_mask: Phase mask in radians
    
    Returns:
        Intensity at Fourier plane
    """
    field = input_amplitude * np.exp(1j * phase_mask)
    fourier_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
    return np.abs(fourier_field) ** 2

