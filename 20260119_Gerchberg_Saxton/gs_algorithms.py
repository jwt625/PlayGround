"""
Gerchberg-Saxton Algorithm Implementations

This module provides implementations of:
1. Standard GS
2. Weighted GS (WGS)
3. GS with Random Phase Reset

All algorithms compute a phase mask to produce a target intensity distribution.
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable


@dataclass
class GSResult:
    """Result container for GS algorithm output."""
    phase_mask: np.ndarray      # Final phase mask (radians, range [-pi, pi])
    reconstructed: np.ndarray   # Reconstructed intensity at target plane
    errors: list[float]         # Error metric per iteration
    iterations: int             # Number of iterations run


def _compute_error(target: np.ndarray, achieved: np.ndarray, mask: np.ndarray = None) -> float:
    """Compute normalized RMS error between target and achieved intensity."""
    if mask is None:
        mask = target > 0
    if mask.sum() == 0:
        return 0.0
    target_norm = target[mask] / target[mask].max()
    achieved_norm = achieved[mask] / (achieved[mask].max() + 1e-10)
    return np.sqrt(np.mean((target_norm - achieved_norm) ** 2))


def standard_gs(
    input_amplitude: np.ndarray,
    target_amplitude: np.ndarray,
    n_iterations: int = 100,
    initial_phase: np.ndarray = None,
) -> GSResult:
    """
    Standard Gerchberg-Saxton algorithm.
    
    Args:
        input_amplitude: Amplitude at input plane (e.g., Gaussian beam)
        target_amplitude: Desired amplitude at Fourier plane
        n_iterations: Number of iterations
        initial_phase: Initial phase guess (random if None)
    
    Returns:
        GSResult with phase mask and reconstruction
    """
    if initial_phase is None:
        initial_phase = np.random.uniform(-np.pi, np.pi, input_amplitude.shape)
    
    # Initialize field at input plane
    field = input_amplitude * np.exp(1j * initial_phase)
    errors = []
    
    for _ in range(n_iterations):
        # Forward propagate to Fourier plane
        fourier_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
        
        # Record error
        achieved_intensity = np.abs(fourier_field) ** 2
        errors.append(_compute_error(target_amplitude ** 2, achieved_intensity))
        
        # Apply Fourier plane constraint: replace amplitude, keep phase
        fourier_phase = np.angle(fourier_field)
        fourier_field = target_amplitude * np.exp(1j * fourier_phase)
        
        # Inverse propagate to input plane
        field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(fourier_field)))
        
        # Apply input plane constraint: replace amplitude, keep phase
        input_phase = np.angle(field)
        field = input_amplitude * np.exp(1j * input_phase)
    
    # Final forward propagate for reconstruction
    fourier_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
    reconstructed = np.abs(fourier_field) ** 2
    
    return GSResult(
        phase_mask=np.angle(field),
        reconstructed=reconstructed,
        errors=errors,
        iterations=n_iterations,
    )


def weighted_gs(
    input_amplitude: np.ndarray,
    target_amplitude: np.ndarray,
    n_iterations: int = 100,
    initial_phase: np.ndarray = None,
) -> GSResult:
    """
    Weighted Gerchberg-Saxton algorithm for improved uniformity.
    
    Weights are updated each iteration to compensate for intensity variations.
    """
    if initial_phase is None:
        initial_phase = np.random.uniform(-np.pi, np.pi, input_amplitude.shape)
    
    field = input_amplitude * np.exp(1j * initial_phase)
    weights = np.ones_like(target_amplitude)
    errors = []
    
    # Mask for target regions
    target_mask = target_amplitude > 0
    
    for _ in range(n_iterations):
        # Forward propagate
        fourier_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
        achieved_amplitude = np.abs(fourier_field)
        
        # Record error
        errors.append(_compute_error(target_amplitude ** 2, achieved_amplitude ** 2))
        
        # Update weights where target is nonzero
        with np.errstate(divide='ignore', invalid='ignore'):
            weight_update = target_amplitude / (achieved_amplitude + 1e-10)
            weight_update = np.clip(weight_update, 0.5, 2.0)  # Stability clamp
        weights = np.where(target_mask, weights * weight_update, weights)
        
        # Apply weighted Fourier constraint
        weighted_target = weights * target_amplitude
        fourier_phase = np.angle(fourier_field)
        fourier_field = weighted_target * np.exp(1j * fourier_phase)
        
        # Inverse propagate
        field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(fourier_field)))
        
        # Apply input constraint
        input_phase = np.angle(field)
        field = input_amplitude * np.exp(1j * input_phase)
    
    # Final reconstruction
    fourier_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
    reconstructed = np.abs(fourier_field) ** 2

    return GSResult(
        phase_mask=np.angle(field),
        reconstructed=reconstructed,
        errors=errors,
        iterations=n_iterations,
    )


def gs_random_reset(
    input_amplitude: np.ndarray,
    target_amplitude: np.ndarray,
    n_iterations: int = 100,
    initial_phase: np.ndarray = None,
    reset_interval: int = 20,
    reset_fraction: float = 0.1,
) -> GSResult:
    """
    GS with periodic random phase reset to escape local minima.

    Every reset_interval iterations, a fraction of the phase is randomly perturbed.

    Args:
        reset_interval: Iterations between resets
        reset_fraction: Fraction of phase perturbation (0 to 1)
    """
    if initial_phase is None:
        initial_phase = np.random.uniform(-np.pi, np.pi, input_amplitude.shape)

    field = input_amplitude * np.exp(1j * initial_phase)
    errors = []
    best_phase = initial_phase.copy()
    best_error = float('inf')

    for i in range(n_iterations):
        # Forward propagate
        fourier_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
        achieved_intensity = np.abs(fourier_field) ** 2

        # Record error
        error = _compute_error(target_amplitude ** 2, achieved_intensity)
        errors.append(error)

        # Track best result
        if error < best_error:
            best_error = error
            best_phase = np.angle(field).copy()

        # Apply Fourier constraint
        fourier_phase = np.angle(fourier_field)
        fourier_field = target_amplitude * np.exp(1j * fourier_phase)

        # Inverse propagate
        field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(fourier_field)))

        # Apply input constraint
        input_phase = np.angle(field)
        field = input_amplitude * np.exp(1j * input_phase)

        # Random phase reset
        if (i + 1) % reset_interval == 0 and i < n_iterations - 1:
            perturbation = np.random.uniform(-np.pi, np.pi, input_amplitude.shape)
            current_phase = np.angle(field)
            new_phase = current_phase + reset_fraction * perturbation
            field = input_amplitude * np.exp(1j * new_phase)

    # Use best phase for final reconstruction
    field = input_amplitude * np.exp(1j * best_phase)
    fourier_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
    reconstructed = np.abs(fourier_field) ** 2

    return GSResult(
        phase_mask=best_phase,
        reconstructed=reconstructed,
        errors=errors,
        iterations=n_iterations,
    )

