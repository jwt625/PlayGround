"""Semi-analytic thermal model and coupled-mode FBG transfer matrix.

Coordinates are SI internally. Wavelength-facing functions use nm for convenience.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ThermalParams:
    length_m: float = 10e-3
    diameter_m: float = 125e-6
    silica_k_w_mk: float = 1.38
    absorption_db_cm: float = 32.43
    ambient_c: float = 23.0
    effective_h_w_m2k: float = 1451.0

    @property
    def absorption_np_m(self) -> float:
        """Power attenuation coefficient: dB/cm -> Np/m."""
        return self.absorption_db_cm * 100.0 * np.log(10.0) / 10.0

    @property
    def area_m2(self) -> float:
        return np.pi * (self.diameter_m / 2.0) ** 2

    @property
    def perimeter_m(self) -> float:
        return np.pi * self.diameter_m

    @property
    def fin_m_inv(self) -> float:
        return np.sqrt(
            self.effective_h_w_m2k
            * self.perimeter_m
            / (self.silica_k_w_mk * self.area_m2)
        )


@dataclass(frozen=True)
class FBGParams:
    length_m: float = 10e-3
    lambda0_nm: float = 1547.493
    n_eff: float = 1.447
    kappa0_m_inv: float = 560.0
    apodization: str = "gaussian"
    apod_sigma_fraction: float = 0.30
    apod_floor: float = 0.0
    segments: int = 800


# Values read from Figure 5(b). They also provide a convenient monotonic P->T map.
POWER_MW = np.array([0.0, 17.0, 73.0, 131.0, 188.0, 244.0, 296.0, 344.0, 367.0])
PEAK_TEMP_C = np.array([23.0, 66.0, 171.0, 251.0, 315.0, 371.0, 415.0, 459.0, 479.0])

# Labeled hot-side reflection peaks in Figure 5(a); these are not the spectral edges.
LABELED_PEAK_NM = np.array(
    [1547.493, 1547.792, 1548.831, 1549.790, 1550.589, 1551.329, 1552.068, 1552.647, 1552.947]
)


def peak_temperature_c(power_mw: float | np.ndarray) -> np.ndarray:
    """Shape-preserving interpolation is supplied in the driver; linear is dependency-light."""
    return np.interp(power_mw, POWER_MW, PEAK_TEMP_C)


def fbg_shift_nm(temperature_c: np.ndarray | float, ambient_c: float = 23.0) -> np.ndarray:
    """Integrate the paper's piecewise high-temperature FBG sensitivity.

    Sensitivities are 10, 11.8, 13.3, 14.4 and 15.1 pm/K over boundaries
    23, 100, 200, 300, 400 and 500 degC, respectively.
    """
    t = np.asarray(temperature_c, dtype=float)
    boundaries = np.array([ambient_c, 100.0, 200.0, 300.0, 400.0, 500.0])
    slopes_nm_k = np.array([0.0100, 0.0118, 0.0133, 0.0144, 0.0151])
    shift = np.zeros_like(t)
    for lo, hi, slope in zip(boundaries[:-1], boundaries[1:], slopes_nm_k):
        shift += np.clip(t - lo, 0.0, hi - lo) * slope
    # Harmless linear extension for exploratory powers above the paper's range.
    shift += np.maximum(t - boundaries[-1], 0.0) * slopes_nm_k[-1]
    return shift


def exponential_temperature_c(
    z_m: np.ndarray,
    peak_c: float,
    ambient_c: float = 23.0,
    decay_m_inv: float = 247.55,
) -> np.ndarray:
    """Effective one-sided thermal profile.

    decay_m_inv=ln(2)/2.8 mm matches MDF03's measured thermal FWHM. The nominal
    Beer-Lambert coefficient is 746.7 1/m and can be selected for comparison.
    """
    return ambient_c + (peak_c - ambient_c) * np.exp(-decay_m_inv * z_m)


def hotspot_temperature_c(
    z_m: np.ndarray,
    peak_c: float,
    ambient_c: float = 23.0,
    full_fwhm_m: float = 2.8e-3,
    shape_order: float = 2.0,
    center_m: float = 5.0e-3,
) -> np.ndarray:
    """Effective compact hotspot inferred jointly from thermal and spectral figures.

    The camera-reported FWHM is treated as the full width of a symmetric hotspot.
    Fitting the relative cold/hot peak heights favors aligning the centers of the
    10 mm heater and grating. The default shape_order=2 is Gaussian.
    """
    z = np.asarray(z_m, dtype=float)
    half_width = full_fwhm_m / 2.0
    scale = half_width / np.log(2.0) ** (1.0 / shape_order)
    return ambient_c + (peak_c - ambient_c) * np.exp(
        -((np.abs(z - center_m) / scale) ** shape_order)
    )


def recovered_hotspot_temperature_c(
    z_m: np.ndarray,
    peak_c: float,
    ambient_c: float = 23.0,
    center_m: float = 2.06563e-3,
    full_fwhm_m: float = 2.74276e-3,
    left_fraction: float = 0.15,
    left_order: float = 4.42495,
    right_order: float = 2.06024,
) -> np.ndarray:
    """Spectrum-recovered asymmetric hotspot in the physical heater orientation.

    Power-reflection spectra have a z->L-z ambiguity. The inverse fit returned the
    mirror profile centered at 7.934 mm; this orientation places the sharp heater edge
    near z=2.066 mm and the broad thermal tail toward increasing z.
    """
    z = np.asarray(z_m, dtype=float)
    left_half = full_fwhm_m * left_fraction
    right_half = full_fwhm_m * (1.0 - left_fraction)
    left_scale = left_half / np.log(2.0) ** (1.0 / left_order)
    right_scale = right_half / np.log(2.0) ** (1.0 / right_order)
    distance = np.abs(z - center_m)
    normalized = np.where(
        z <= center_m,
        np.exp(-((distance / left_scale) ** left_order)),
        np.exp(-((distance / right_scale) ** right_order)),
    )
    return ambient_c + (peak_c - ambient_c) * normalized


def fin_temperature_rise_k(
    z_m: np.ndarray,
    optical_power_w: float,
    thermal: ThermalParams = ThermalParams(),
) -> np.ndarray:
    """Exact steady 1-D cylindrical-fin solution for an exponential line source.

    Solves theta'' - m^2 theta = -alpha*P0*exp(-alpha*z)/(k*A), with
    zero axial heat flux at both ends. This is a compact semi-analytic benchmark;
    contact to the second fiber/paste is intentionally not folded into h_eff.
    """
    z = np.asarray(z_m, dtype=float)
    length = thermal.length_m
    alpha = thermal.absorption_np_m
    m = thermal.fin_m_inv
    source = alpha * optical_power_w / (thermal.silica_k_w_mk * thermal.area_m2)
    particular = source / (m * m - alpha * alpha)

    # theta = A*cosh(mz) + B*sinh(mz) + C*exp(-alpha*z)
    # theta'(0)=theta'(L)=0. Stable here because mL~60, but express the second
    # boundary in a form that avoids solving a poorly scaled 2x2 system.
    c = particular
    b = alpha * c / m
    a = (
        alpha * c * np.exp(-alpha * length) / m
        - b * np.cosh(m * length)
    ) / np.sinh(m * length)
    return a * np.cosh(m * z) + b * np.sinh(m * z) + c * np.exp(-alpha * z)


def _apodization(z_m: np.ndarray, fbg: FBGParams) -> np.ndarray:
    if fbg.apodization == "uniform":
        return np.ones_like(z_m)
    if fbg.apodization == "gaussian":
        x = (z_m - fbg.length_m / 2.0) / (fbg.apod_sigma_fraction * fbg.length_m)
        return fbg.apod_floor + (1.0 - fbg.apod_floor) * np.exp(-0.5 * x * x)
    raise ValueError(f"Unknown apodization: {fbg.apodization}")


def reflection_spectrum(
    wavelength_nm: np.ndarray,
    local_bragg_nm: np.ndarray,
    fbg: FBGParams = FBGParams(),
    reverse_grating: bool = False,
) -> np.ndarray:
    """Return power reflectivity using a piecewise-uniform CMT transfer matrix."""
    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    if local_bragg_nm.size != fbg.segments:
        raise ValueError("local_bragg_nm must have fbg.segments elements")
    if reverse_grating:
        local_bragg_nm = local_bragg_nm[::-1]

    dz = fbg.length_m / fbg.segments
    z_mid = (np.arange(fbg.segments) + 0.5) * dz
    kappa = fbg.kappa0_m_inv * _apodization(z_mid, fbg)
    lam_m = wavelength_nm * 1e-9

    # Total matrix maps fields at z=0 to z=L. Each scalar below is an array over lambda.
    t11 = np.ones_like(lam_m, dtype=complex)
    t12 = np.zeros_like(lam_m, dtype=complex)
    t21 = np.zeros_like(lam_m, dtype=complex)
    t22 = np.ones_like(lam_m, dtype=complex)
    for kb, lb_nm in zip(kappa, local_bragg_nm):
        delta = 2.0 * np.pi * fbg.n_eff * (1.0 / lam_m - 1.0 / (lb_nm * 1e-9))
        gamma = np.sqrt((kb * kb - delta * delta) + 0j)
        # sinh(gamma*dz)/gamma has a finite dz limit.
        s_over_g = np.where(np.abs(gamma) > 1e-14, np.sinh(gamma * dz) / gamma, dz)
        c = np.cosh(gamma * dz)
        m11 = c - 1j * delta * s_over_g
        m12 = -1j * kb * s_over_g
        m21 = 1j * kb * s_over_g
        m22 = c + 1j * delta * s_over_g
        n11 = m11 * t11 + m12 * t21
        n12 = m11 * t12 + m12 * t22
        n21 = m21 * t11 + m22 * t21
        n22 = m21 * t12 + m22 * t22
        t11, t12, t21, t22 = n11, n12, n21, n22
    return np.abs(-t21 / t22) ** 2


def spectrum_at_power(
    wavelength_nm: np.ndarray,
    power_mw: float,
    fbg: FBGParams = FBGParams(),
    hot_zone_fwhm_m: float = 2.74276e-3,
    hotspot_center_m: float = 2.06563e-3,
    hotspot_left_fraction: float = 0.15,
    hotspot_left_order: float = 4.42495,
    hotspot_right_order: float = 2.06024,
    reverse_grating: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convenience wrapper returning reflectivity, z and temperature."""
    dz = fbg.length_m / fbg.segments
    z = (np.arange(fbg.segments) + 0.5) * dz
    temp = recovered_hotspot_temperature_c(
        z,
        float(peak_temperature_c(power_mw)),
        full_fwhm_m=hot_zone_fwhm_m,
        center_m=hotspot_center_m,
        left_fraction=hotspot_left_fraction,
        left_order=hotspot_left_order,
        right_order=hotspot_right_order,
    )
    local_bragg = fbg.lambda0_nm + fbg_shift_nm(temp)
    return reflection_spectrum(wavelength_nm, local_bragg, fbg, reverse_grating), z, temp
