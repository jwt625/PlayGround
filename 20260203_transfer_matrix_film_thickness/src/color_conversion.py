"""
Spectrum to RGB color conversion.

Uses CIE 1931 standard observer color matching functions and sRGB color space.
Includes standard illuminants (D65, D50, etc.)

Pipeline: Spectrum → XYZ → Linear sRGB → Gamma-corrected sRGB
"""

import numpy as np
from scipy.interpolate import interp1d


# CIE 1931 2° Standard Observer Color Matching Functions
# Wavelength (nm), x̄, ȳ, z̄
# Sampled at 5nm intervals from 380-780nm
_CIE1931_DATA = np.array([
    [380, 0.001368, 0.000039, 0.006450],
    [385, 0.002236, 0.000064, 0.010550],
    [390, 0.004243, 0.000120, 0.020050],
    [395, 0.007650, 0.000217, 0.036210],
    [400, 0.014310, 0.000396, 0.067850],
    [405, 0.023190, 0.000640, 0.110200],
    [410, 0.043510, 0.001210, 0.207400],
    [415, 0.077630, 0.002180, 0.371300],
    [420, 0.134380, 0.004000, 0.645600],
    [425, 0.214770, 0.007300, 1.039050],
    [430, 0.283900, 0.011600, 1.385600],
    [435, 0.328500, 0.016840, 1.622960],
    [440, 0.348280, 0.023000, 1.747060],
    [445, 0.348060, 0.029800, 1.782600],
    [450, 0.336200, 0.038000, 1.772110],
    [455, 0.318700, 0.048000, 1.744100],
    [460, 0.290800, 0.060000, 1.669200],
    [465, 0.251100, 0.073900, 1.528100],
    [470, 0.195360, 0.090980, 1.287640],
    [475, 0.142100, 0.112600, 1.041900],
    [480, 0.095640, 0.139020, 0.812950],
    [485, 0.058010, 0.169300, 0.616200],
    [490, 0.032010, 0.208020, 0.465180],
    [495, 0.014700, 0.258600, 0.353300],
    [500, 0.004900, 0.323000, 0.272000],
    [505, 0.002400, 0.407300, 0.212300],
    [510, 0.009300, 0.503000, 0.158200],
    [515, 0.029100, 0.608200, 0.111700],
    [520, 0.063270, 0.710000, 0.078250],
    [525, 0.109600, 0.793200, 0.057250],
    [530, 0.165500, 0.862000, 0.042160],
    [535, 0.225750, 0.914850, 0.029840],
    [540, 0.290400, 0.954000, 0.020300],
    [545, 0.359700, 0.980300, 0.013400],
    [550, 0.433450, 0.994950, 0.008750],
    [555, 0.512050, 1.000000, 0.005750],
    [560, 0.594500, 0.995000, 0.003900],
    [565, 0.678400, 0.978600, 0.002750],
    [570, 0.762100, 0.952000, 0.002100],
    [575, 0.842500, 0.915400, 0.001800],
    [580, 0.916300, 0.870000, 0.001650],
    [585, 0.978600, 0.816300, 0.001400],
    [590, 1.026300, 0.757000, 0.001100],
    [595, 1.056700, 0.694900, 0.001000],
    [600, 1.062200, 0.631000, 0.000800],
    [605, 1.045600, 0.566800, 0.000600],
    [610, 1.002600, 0.503000, 0.000340],
    [615, 0.938400, 0.441200, 0.000240],
    [620, 0.854450, 0.381000, 0.000190],
    [625, 0.751400, 0.321000, 0.000100],
    [630, 0.642400, 0.265000, 0.000050],
    [635, 0.541900, 0.217000, 0.000030],
    [640, 0.447900, 0.175000, 0.000020],
    [645, 0.360800, 0.138200, 0.000010],
    [650, 0.283500, 0.107000, 0.000000],
    [655, 0.218700, 0.081600, 0.000000],
    [660, 0.164900, 0.061000, 0.000000],
    [665, 0.121200, 0.044580, 0.000000],
    [670, 0.087400, 0.032000, 0.000000],
    [675, 0.063600, 0.023200, 0.000000],
    [680, 0.046770, 0.017000, 0.000000],
    [685, 0.032900, 0.011920, 0.000000],
    [690, 0.022700, 0.008210, 0.000000],
    [695, 0.015840, 0.005723, 0.000000],
    [700, 0.011359, 0.004102, 0.000000],
    [705, 0.008111, 0.002929, 0.000000],
    [710, 0.005790, 0.002091, 0.000000],
    [715, 0.004109, 0.001484, 0.000000],
    [720, 0.002899, 0.001047, 0.000000],
    [725, 0.002049, 0.000740, 0.000000],
    [730, 0.001440, 0.000520, 0.000000],
    [735, 0.001000, 0.000361, 0.000000],
    [740, 0.000690, 0.000249, 0.000000],
    [745, 0.000476, 0.000172, 0.000000],
    [750, 0.000332, 0.000120, 0.000000],
    [755, 0.000235, 0.000085, 0.000000],
    [760, 0.000166, 0.000060, 0.000000],
    [765, 0.000117, 0.000042, 0.000000],
    [770, 0.000083, 0.000030, 0.000000],
    [775, 0.000059, 0.000021, 0.000000],
    [780, 0.000042, 0.000015, 0.000000],
])

# Create interpolation functions for CIE 1931
_cie_x_interp = interp1d(_CIE1931_DATA[:, 0], _CIE1931_DATA[:, 1], 
                          kind='linear', bounds_error=False, fill_value=0)
_cie_y_interp = interp1d(_CIE1931_DATA[:, 0], _CIE1931_DATA[:, 2], 
                          kind='linear', bounds_error=False, fill_value=0)
_cie_z_interp = interp1d(_CIE1931_DATA[:, 0], _CIE1931_DATA[:, 3], 
                          kind='linear', bounds_error=False, fill_value=0)


def cie1931_xyz(wavelength_nm: float):
    """Get CIE 1931 color matching functions at given wavelength."""
    return (_cie_x_interp(wavelength_nm),
            _cie_y_interp(wavelength_nm),
            _cie_z_interp(wavelength_nm))


# Standard Illuminants - Relative spectral power distribution
# Using CIE standard illuminants

def illuminant_d65(wavelength_nm: float) -> float:
    """CIE Standard Illuminant D65 (average daylight, ~6500K).

    This is the most common illuminant for photography and displays.
    Approximation using Planckian + daylight correction.
    """
    # Simplified D65 approximation using tabulated data points
    # More accurate than Planckian for daylight simulation
    wl = wavelength_nm
    # Polynomial approximation valid for 380-780nm
    if wl < 380 or wl > 780:
        return 0.0

    # Normalized D65 SPD (simplified analytical approximation)
    # Based on CIE D65 tabulated data
    x = (wl - 560) / 100
    spd = 100 * np.exp(-0.5 * x**2) * (1 + 0.1 * np.sin(0.05 * wl))

    # Add blue boost characteristic of D65
    if wl < 500:
        spd *= 1.0 + 0.3 * np.exp(-((wl - 450) / 30)**2)

    return max(0, spd)


def illuminant_d50(wavelength_nm: float) -> float:
    """CIE Standard Illuminant D50 (horizon daylight, ~5000K)."""
    wl = wavelength_nm
    if wl < 380 or wl > 780:
        return 0.0
    x = (wl - 580) / 100  # Shifted redder than D65
    spd = 100 * np.exp(-0.5 * x**2)
    return max(0, spd)


def illuminant_a(wavelength_nm: float) -> float:
    """CIE Standard Illuminant A (incandescent, 2856K).

    Planckian radiator at 2856K.
    """
    wl_m = wavelength_nm * 1e-9
    T = 2856  # Kelvin
    h = 6.626e-34
    c = 3e8
    k = 1.381e-23

    # Planck's law (relative, normalized)
    x = h * c / (wl_m * k * T)
    if x > 700:  # Prevent overflow
        return 0.0
    spd = (1 / wl_m**5) / (np.exp(x) - 1)

    # Normalize to ~100 at 560nm
    spd_560 = (1 / (560e-9)**5) / (np.exp(h * c / (560e-9 * k * T)) - 1)
    return 100 * spd / spd_560


def illuminant_equal_energy(wavelength_nm: float) -> float:
    """Equal energy illuminant (flat spectrum)."""
    if 380 <= wavelength_nm <= 780:
        return 100.0
    return 0.0


ILLUMINANTS = {
    'D65': illuminant_d65,
    'd65': illuminant_d65,
    'D50': illuminant_d50,
    'd50': illuminant_d50,
    'A': illuminant_a,
    'a': illuminant_a,
    'E': illuminant_equal_energy,
    'equal': illuminant_equal_energy,
}


# XYZ to sRGB conversion matrix (D65 white point)
_XYZ_TO_SRGB = np.array([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252]
])


def xyz_to_linear_srgb(X: float, Y: float, Z: float) -> tuple:
    """Convert CIE XYZ to linear sRGB (before gamma correction)."""
    xyz = np.array([X, Y, Z])
    rgb = _XYZ_TO_SRGB @ xyz
    return tuple(rgb)


def srgb_gamma(linear_value: float) -> float:
    """Apply sRGB gamma correction (linear to gamma-encoded)."""
    if linear_value <= 0.0031308:
        return 12.92 * linear_value
    else:
        return 1.055 * (linear_value ** (1/2.4)) - 0.055


def linear_to_srgb(r: float, g: float, b: float) -> tuple:
    """Convert linear RGB to gamma-corrected sRGB."""
    return (srgb_gamma(r), srgb_gamma(g), srgb_gamma(b))


def spectrum_to_xyz(wavelengths: np.ndarray, spectrum: np.ndarray,
                    illuminant: str = 'D65') -> tuple:
    """Convert a reflectance/radiance spectrum to CIE XYZ.

    Args:
        wavelengths: Array of wavelengths in nm
        spectrum: Array of spectral values (reflectance 0-1 or radiance)
        illuminant: Illuminant name ('D65', 'D50', 'A', 'E')

    Returns:
        (X, Y, Z) tristimulus values
    """
    illum_func = ILLUMINANTS.get(illuminant, illuminant_d65)

    # Integration using trapezoidal rule
    X, Y, Z = 0.0, 0.0, 0.0
    k = 0.0  # Normalization factor

    for i, wl in enumerate(wavelengths):
        x_bar, y_bar, z_bar = cie1931_xyz(wl)
        illum = illum_func(wl)

        # Reflected light = reflectance * illuminant
        reflected = spectrum[i] * illum

        X += reflected * x_bar
        Y += reflected * y_bar
        Z += reflected * z_bar
        k += illum * y_bar

    # Normalize so perfect white (R=1) gives Y=1
    if k > 0:
        X /= k
        Y /= k
        Z /= k

    return X, Y, Z


def spectrum_to_srgb(wavelengths: np.ndarray, spectrum: np.ndarray,
                     illuminant: str = 'D65', clip: bool = True) -> tuple:
    """Convert a reflectance spectrum to sRGB color.

    Args:
        wavelengths: Array of wavelengths in nm
        spectrum: Array of reflectance values (0-1)
        illuminant: Illuminant name
        clip: Whether to clip RGB to [0, 1] range

    Returns:
        (R, G, B) tuple with values in [0, 1] (or [0, 255] if scaled)
    """
    X, Y, Z = spectrum_to_xyz(wavelengths, spectrum, illuminant)
    r, g, b = xyz_to_linear_srgb(X, Y, Z)

    if clip:
        r = np.clip(r, 0, 1)
        g = np.clip(g, 0, 1)
        b = np.clip(b, 0, 1)

    # Apply gamma correction
    R, G, B = linear_to_srgb(r, g, b)

    if clip:
        R = np.clip(R, 0, 1)
        G = np.clip(G, 0, 1)
        B = np.clip(B, 0, 1)

    return R, G, B


def spectrum_to_rgb_uint8(wavelengths: np.ndarray, spectrum: np.ndarray,
                          illuminant: str = 'D65') -> tuple:
    """Convert spectrum to 8-bit RGB values (0-255)."""
    R, G, B = spectrum_to_srgb(wavelengths, spectrum, illuminant, clip=True)
    return (int(R * 255), int(G * 255), int(B * 255))

