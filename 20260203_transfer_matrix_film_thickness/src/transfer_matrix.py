"""
Transfer Matrix Method for Multilayer Thin Film Optics

Calculates reflection and transmission for arbitrary multilayer stacks
with support for s and p polarizations at any angle of incidence.

Physics Reference:
- Fresnel equations for interface reflection/transmission
- Transfer matrix formalism for coherent multilayer interference
"""

import numpy as np
from typing import Tuple, List, Union, Optional
from dataclasses import dataclass


@dataclass
class Layer:
    """Represents a layer in the thin film stack.
    
    Attributes:
        n: Complex refractive index (can be callable n(wavelength_nm))
        d: Thickness in nm (use np.inf for semi-infinite substrate/superstrate)
        name: Optional layer name for identification
    """
    n: Union[complex, float, callable]
    d: float
    name: str = ""
    
    def get_n(self, wavelength_nm: float) -> complex:
        """Get refractive index at given wavelength."""
        if callable(self.n):
            return self.n(wavelength_nm)
        return complex(self.n)


def snell_angle(n1: complex, n2: complex, theta1: float) -> complex:
    """Calculate refracted angle using Snell's law.
    
    Args:
        n1: Refractive index of incident medium
        n2: Refractive index of transmitted medium  
        theta1: Incident angle in radians
        
    Returns:
        Complex angle in medium 2 (can be complex for evanescent waves)
    """
    sin_theta2 = (n1 / n2) * np.sin(theta1)
    # Use complex sqrt to handle total internal reflection
    cos_theta2 = np.sqrt(1 - sin_theta2**2 + 0j)
    # Choose correct branch for absorbing media
    if cos_theta2.imag < 0:
        cos_theta2 = -cos_theta2
    return np.arcsin(sin_theta2)


def fresnel_coefficients(n1: complex, n2: complex, theta1: float, 
                         polarization: str = 's') -> Tuple[complex, complex]:
    """Calculate Fresnel reflection and transmission coefficients.
    
    Args:
        n1: Refractive index of incident medium
        n2: Refractive index of transmitted medium
        theta1: Incident angle in radians
        polarization: 's' (TE) or 'p' (TM)
        
    Returns:
        (r, t): Fresnel reflection and transmission amplitude coefficients
    """
    cos1 = np.cos(theta1)
    sin1 = np.sin(theta1)
    
    # Calculate cos(theta2) using Snell's law
    sin2 = (n1 / n2) * sin1
    cos2 = np.sqrt(1 - sin2**2 + 0j)
    if cos2.imag < 0:
        cos2 = -cos2
    
    if polarization.lower() == 's':
        # s-polarization (TE): E-field perpendicular to plane of incidence
        r = (n1 * cos1 - n2 * cos2) / (n1 * cos1 + n2 * cos2)
        t = (2 * n1 * cos1) / (n1 * cos1 + n2 * cos2)
    elif polarization.lower() == 'p':
        # p-polarization (TM): E-field in plane of incidence
        r = (n2 * cos1 - n1 * cos2) / (n2 * cos1 + n1 * cos2)
        t = (2 * n1 * cos1) / (n2 * cos1 + n1 * cos2)
    else:
        raise ValueError(f"Polarization must be 's' or 'p', got '{polarization}'")
    
    return r, t


def interface_matrix(n1: complex, n2: complex, theta1: float,
                     polarization: str = 's') -> np.ndarray:
    """Compute interface transfer matrix between two media.
    
    Args:
        n1, n2: Refractive indices
        theta1: Incident angle in radians
        polarization: 's' or 'p'
        
    Returns:
        2x2 complex transfer matrix
    """
    r, t = fresnel_coefficients(n1, n2, theta1, polarization)
    
    # Interface matrix: relates fields on both sides
    M = (1 / t) * np.array([
        [1, r],
        [r, 1]
    ], dtype=complex)
    
    return M


def propagation_matrix(n: complex, d: float, wavelength_nm: float,
                       theta: float) -> np.ndarray:
    """Compute propagation matrix through a layer.
    
    Args:
        n: Complex refractive index of the layer
        d: Layer thickness in nm
        wavelength_nm: Wavelength in nm
        theta: Angle of propagation in the layer (radians)
        
    Returns:
        2x2 complex propagation matrix
    """
    # Phase accumulated during propagation
    cos_theta = np.cos(theta)
    if hasattr(cos_theta, 'imag') and cos_theta.imag < 0:
        cos_theta = -cos_theta
    
    # k_z * d = (2*pi*n/lambda) * cos(theta) * d
    delta = (2 * np.pi * n * d * np.cos(theta)) / wavelength_nm
    
    P = np.array([
        [np.exp(-1j * delta), 0],
        [0, np.exp(1j * delta)]
    ], dtype=complex)
    
    return P


def transfer_matrix_stack(layers: List[Layer], wavelength_nm: float,
                          theta_inc: float = 0.0, 
                          polarization: str = 's') -> np.ndarray:
    """Compute total transfer matrix for a multilayer stack.
    
    Args:
        layers: List of Layer objects (first is incident medium, last is substrate)
        wavelength_nm: Wavelength in nm
        theta_inc: Incident angle in radians
        polarization: 's' or 'p'
        
    Returns:
        2x2 total transfer matrix
    """
    if len(layers) < 2:
        raise ValueError("Need at least 2 layers (incident medium and substrate)")
    
    # Get refractive indices at this wavelength
    n_list = [layer.get_n(wavelength_nm) for layer in layers]
    
    # Calculate angles in each layer using Snell's law
    theta_list = [complex(theta_inc)]
    for i in range(len(n_list) - 1):
        sin_next = (n_list[i] / n_list[i+1]) * np.sin(theta_list[i])
        cos_next = np.sqrt(1 - sin_next**2 + 0j)
        if cos_next.imag < 0:
            cos_next = -cos_next
        theta_list.append(np.arccos(cos_next))

    # Build total transfer matrix
    # Start with identity
    M_total = np.eye(2, dtype=complex)

    for i in range(len(layers) - 1):
        # Interface matrix from layer i to layer i+1
        M_int = interface_matrix(n_list[i], n_list[i+1], theta_list[i], polarization)
        M_total = M_total @ M_int

        # Propagation matrix through layer i+1 (skip if substrate)
        if i + 1 < len(layers) - 1:
            d = layers[i + 1].d
            if np.isfinite(d) and d > 0:
                M_prop = propagation_matrix(n_list[i+1], d, wavelength_nm, theta_list[i+1])
                M_total = M_total @ M_prop

    return M_total


def calculate_reflectance(layers: List[Layer], wavelength_nm: float,
                          theta_inc: float = 0.0,
                          polarization: str = 's') -> float:
    """Calculate reflectance for a multilayer stack.

    Args:
        layers: List of Layer objects
        wavelength_nm: Wavelength in nm
        theta_inc: Incident angle in radians
        polarization: 's', 'p', or 'unpolarized'

    Returns:
        Reflectance (power reflection coefficient, 0 to 1)
    """
    if polarization.lower() == 'unpolarized':
        R_s = calculate_reflectance(layers, wavelength_nm, theta_inc, 's')
        R_p = calculate_reflectance(layers, wavelength_nm, theta_inc, 'p')
        return (R_s + R_p) / 2

    M = transfer_matrix_stack(layers, wavelength_nm, theta_inc, polarization)

    # Reflection amplitude coefficient
    r = M[1, 0] / M[0, 0]

    # Reflectance is |r|^2
    R = np.abs(r)**2

    return R


def calculate_transmittance(layers: List[Layer], wavelength_nm: float,
                            theta_inc: float = 0.0,
                            polarization: str = 's') -> float:
    """Calculate transmittance for a multilayer stack.

    Args:
        layers: List of Layer objects
        wavelength_nm: Wavelength in nm
        theta_inc: Incident angle in radians
        polarization: 's' or 'p'

    Returns:
        Transmittance (power transmission coefficient)
    """
    M = transfer_matrix_stack(layers, wavelength_nm, theta_inc, polarization)

    # Get refractive indices of first and last layers
    n_inc = layers[0].get_n(wavelength_nm)
    n_sub = layers[-1].get_n(wavelength_nm)

    # Calculate angles
    sin_sub = (n_inc / n_sub) * np.sin(theta_inc)
    cos_sub = np.sqrt(1 - sin_sub**2 + 0j)
    if cos_sub.imag < 0:
        cos_sub = -cos_sub
    cos_inc = np.cos(theta_inc)

    # Transmission amplitude coefficient
    t = 1 / M[0, 0]

    # Transmittance includes index and angle correction
    if polarization.lower() == 's':
        T = np.real(n_sub * cos_sub) / np.real(n_inc * cos_inc) * np.abs(t)**2
    else:  # p-polarization
        T = np.real(n_sub * np.conj(cos_sub)) / np.real(n_inc * np.conj(cos_inc)) * np.abs(t)**2

    return np.real(T)


def reflectance_spectrum(layers: List[Layer],
                         wavelengths_nm: np.ndarray,
                         theta_inc: float = 0.0,
                         polarization: str = 's') -> np.ndarray:
    """Calculate reflectance spectrum over a range of wavelengths.

    Args:
        layers: List of Layer objects
        wavelengths_nm: Array of wavelengths in nm
        theta_inc: Incident angle in radians
        polarization: 's', 'p', or 'unpolarized'

    Returns:
        Array of reflectance values
    """
    if polarization.lower() == 'unpolarized':
        R_s = np.array([calculate_reflectance(layers, wl, theta_inc, 's')
                        for wl in wavelengths_nm])
        R_p = np.array([calculate_reflectance(layers, wl, theta_inc, 'p')
                        for wl in wavelengths_nm])
        return (R_s + R_p) / 2
    else:
        return np.array([calculate_reflectance(layers, wl, theta_inc, polarization)
                         for wl in wavelengths_nm])


# =============================================================================
# Ellipsometry Functions
# =============================================================================

def reflection_coefficient(layers: List[Layer], wavelength_nm: float,
                           theta_inc: float, polarization: str) -> complex:
    """Calculate complex reflection amplitude coefficient.

    Args:
        layers: List of Layer objects
        wavelength_nm: Wavelength in nm
        theta_inc: Incident angle in radians
        polarization: 's' or 'p'

    Returns:
        Complex reflection coefficient r (amplitude, not intensity)
    """
    M = transfer_matrix_stack(layers, wavelength_nm, theta_inc, polarization)
    r = M[1, 0] / M[0, 0]
    return r


def calculate_ellipsometry(layers: List[Layer], wavelength_nm: float,
                           theta_inc: float) -> Tuple[float, float]:
    """Calculate ellipsometry parameters Psi and Delta.

    Ellipsometry measures the ratio of p to s reflection:
        rho = rp/rs = tan(Psi) * exp(i*Delta)

    Args:
        layers: List of Layer objects
        wavelength_nm: Wavelength in nm
        theta_inc: Incident angle in radians

    Returns:
        (Psi, Delta) in degrees
        - Psi: amplitude ratio angle, tan(Psi) = |rp|/|rs|, range [0, 90]
        - Delta: phase difference, Delta = arg(rp) - arg(rs), range [-180, 180]
    """
    r_s = reflection_coefficient(layers, wavelength_nm, theta_inc, 's')
    r_p = reflection_coefficient(layers, wavelength_nm, theta_inc, 'p')

    # rho = rp / rs
    rho = r_p / r_s

    # Psi from amplitude ratio
    psi_rad = np.arctan(np.abs(rho))
    psi_deg = np.degrees(psi_rad)

    # Delta from phase difference
    delta_rad = np.angle(rho)
    delta_deg = np.degrees(delta_rad)

    return psi_deg, delta_deg


def ellipsometry_spectrum(layers: List[Layer],
                          wavelengths_nm: np.ndarray,
                          theta_inc: float) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate ellipsometry parameters over a range of wavelengths.

    Args:
        layers: List of Layer objects
        wavelengths_nm: Array of wavelengths in nm
        theta_inc: Incident angle in radians

    Returns:
        (Psi_array, Delta_array) in degrees
    """
    psi_vals = []
    delta_vals = []

    for wl in wavelengths_nm:
        psi, delta = calculate_ellipsometry(layers, wl, theta_inc)
        psi_vals.append(psi)
        delta_vals.append(delta)

    return np.array(psi_vals), np.array(delta_vals)


def ellipsometry_vs_thickness(wavelength_nm: float, theta_inc: float,
                              thicknesses_nm: np.ndarray,
                              layer_builder: callable) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate ellipsometry parameters vs film thickness.

    Args:
        wavelength_nm: Wavelength in nm
        theta_inc: Incident angle in radians
        thicknesses_nm: Array of film thicknesses to calculate
        layer_builder: Function that takes thickness and returns layer list

    Returns:
        (Psi_array, Delta_array) in degrees
    """
    psi_vals = []
    delta_vals = []

    for d in thicknesses_nm:
        layers = layer_builder(d)
        psi, delta = calculate_ellipsometry(layers, wavelength_nm, theta_inc)
        psi_vals.append(psi)
        delta_vals.append(delta)

    return np.array(psi_vals), np.array(delta_vals)

