"""
DTI Analysis Module

Functions for:
- b-value calculation
- ADC extraction from echo decay
- Diffusion tensor fitting
- DTI metrics computation
"""

import numpy as np


def compute_b_value(gradient_strength, tau, gamma=1.0):
    """
    Compute b-value for Hahn Echo sequence with diffusion gradients
    
    For Hahn Echo: b = γ² G² δ² (Δ - δ/3)
    where:
    - γ = gyromagnetic ratio
    - G = gradient strength
    - δ = gradient duration (≈ tau in our case)
    - Δ = time between gradient pulses (≈ tau)
    
    Parameters:
    -----------
    gradient_strength : float
        Gradient strength G
    tau : float
        Time between pulses
    gamma : float
        Gyromagnetic ratio (default 1.0 for normalized units)
    
    Returns:
    --------
    b : float
        b-value
    """
    G = gradient_strength
    delta = tau  # Gradient on during entire tau period
    Delta = tau  # Time between gradients
    
    b = gamma**2 * G**2 * delta**2 * (Delta - delta/3)
    return b


def compute_analytical_echo_decay(b_value, gradient_direction, diffusion_tensor):
    """
    Compute echo decay using analytical Stejskal-Tanner equation

    S(b) / S(0) = exp(-b * ADC)
    where ADC = g^T * D * g

    Parameters:
    -----------
    b_value : float
        b-value
    gradient_direction : array_like
        Gradient direction (will be normalized)
    diffusion_tensor : ndarray
        3x3 diffusion tensor

    Returns:
    --------
    decay_ratio : float
        S(b) / S(0) ratio
    ADC : float
        Apparent diffusion coefficient
    """
    g = np.array(gradient_direction)
    g = g / np.linalg.norm(g)

    D = np.array(diffusion_tensor)
    ADC = g @ D @ g

    decay_ratio = np.exp(-b_value * ADC)

    return decay_ratio, ADC


def extract_ADC(echo_amplitude_b0, echo_amplitude_b, b_value):
    """
    Extract apparent diffusion coefficient from echo decay

    S(b) / S(0) = exp(-b * ADC)
    ADC = -ln(S(b) / S(0)) / b

    Parameters:
    -----------
    echo_amplitude_b0 : float
        Echo amplitude with no gradient (b=0)
    echo_amplitude_b : float
        Echo amplitude with gradient (b>0)
    b_value : float
        b-value

    Returns:
    --------
    ADC : float
        Apparent diffusion coefficient
    """
    if echo_amplitude_b0 <= 0 or echo_amplitude_b <= 0:
        return np.nan

    ratio = echo_amplitude_b / echo_amplitude_b0

    # Avoid log of zero or negative
    if ratio <= 0:
        return np.nan

    ADC = -np.log(ratio) / b_value
    return ADC


def fit_diffusion_tensor(gradient_directions, ADC_measurements):
    """
    Fit diffusion tensor from ADC measurements in multiple directions
    
    For each direction i: ADC_i = g_i^T * D * g_i
    
    Expanded: ADC_i = D_xx*g_ix² + D_yy*g_iy² + D_zz*g_iz² 
                      + 2*D_xy*g_ix*g_iy + 2*D_xz*g_ix*g_iz + 2*D_yz*g_iy*g_iz
    
    This is linear in the 6 unknowns: [D_xx, D_yy, D_zz, D_xy, D_xz, D_yz]
    
    Parameters:
    -----------
    gradient_directions : ndarray
        Array of shape (N, 3) with N gradient directions
    ADC_measurements : ndarray
        Array of shape (N,) with ADC measurements
    
    Returns:
    --------
    D : ndarray
        Fitted 3x3 diffusion tensor
    residual : float
        Fitting residual (RMS error)
    """
    N = len(gradient_directions)
    
    # Build design matrix A
    # Each row: [g_x², g_y², g_z², 2*g_x*g_y, 2*g_x*g_z, 2*g_y*g_z]
    A = np.zeros((N, 6))
    for i, g in enumerate(gradient_directions):
        gx, gy, gz = g
        A[i, :] = [gx**2, gy**2, gz**2, 2*gx*gy, 2*gx*gz, 2*gy*gz]
    
    # Solve: A * x = b, where x = [D_xx, D_yy, D_zz, D_xy, D_xz, D_yz]
    b = ADC_measurements
    
    # Least squares solution
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    
    # Reconstruct symmetric tensor
    D = np.array([
        [x[0], x[3], x[4]],
        [x[3], x[1], x[5]],
        [x[4], x[5], x[2]]
    ])
    
    # Compute residual
    ADC_fitted = A @ x
    residual = np.sqrt(np.mean((ADC_measurements - ADC_fitted)**2))
    
    return D, residual


def compute_dti_metrics(D):
    """
    Compute DTI metrics from diffusion tensor
    
    Parameters:
    -----------
    D : ndarray
        3x3 diffusion tensor
    
    Returns:
    --------
    metrics : dict
        Dictionary with:
        - 'eigenvalues': [λ1, λ2, λ3] (sorted descending)
        - 'eigenvectors': 3x3 array (columns are eigenvectors)
        - 'MD': mean diffusivity
        - 'FA': fractional anisotropy
        - 'principal_direction': principal eigenvector
    """
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(D)
    
    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    lambda1, lambda2, lambda3 = eigenvalues
    
    # Mean Diffusivity
    MD = np.mean(eigenvalues)
    
    # Fractional Anisotropy
    numerator = np.sqrt(((lambda1 - MD)**2 + (lambda2 - MD)**2 + (lambda3 - MD)**2))
    denominator = np.sqrt(lambda1**2 + lambda2**2 + lambda3**2)
    if denominator == 0:
        FA = 0.0
    else:
        FA = np.sqrt(3/2) * numerator / denominator
    
    metrics = {
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'MD': MD,
        'FA': FA,
        'principal_direction': eigenvectors[:, 0]
    }
    
    return metrics


def compare_tensors(D_true, D_fitted):
    """
    Compare true and fitted diffusion tensors
    
    Parameters:
    -----------
    D_true : ndarray
        True diffusion tensor
    D_fitted : ndarray
        Fitted diffusion tensor
    
    Returns:
    --------
    comparison : dict
        Dictionary with error metrics
    """
    # Element-wise error
    element_error = np.abs(D_true - D_fitted)
    max_element_error = np.max(element_error)
    mean_element_error = np.mean(element_error)
    
    # Frobenius norm error
    frobenius_error = np.linalg.norm(D_true - D_fitted, 'fro')
    frobenius_norm_true = np.linalg.norm(D_true, 'fro')
    relative_frobenius_error = frobenius_error / frobenius_norm_true if frobenius_norm_true > 0 else 0
    
    # Metrics comparison
    metrics_true = compute_dti_metrics(D_true)
    metrics_fitted = compute_dti_metrics(D_fitted)
    
    MD_error = abs(metrics_true['MD'] - metrics_fitted['MD'])
    FA_error = abs(metrics_true['FA'] - metrics_fitted['FA'])
    
    comparison = {
        'max_element_error': max_element_error,
        'mean_element_error': mean_element_error,
        'frobenius_error': frobenius_error,
        'relative_frobenius_error': relative_frobenius_error,
        'MD_error': MD_error,
        'FA_error': FA_error,
        'MD_true': metrics_true['MD'],
        'MD_fitted': metrics_fitted['MD'],
        'FA_true': metrics_true['FA'],
        'FA_fitted': metrics_fitted['FA']
    }
    
    return comparison

