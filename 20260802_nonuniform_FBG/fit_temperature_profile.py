"""Recover an asymmetric axial temperature shape from the digitized FBG spectra.

The inversion fits smoothed spectral envelopes first; every forward evaluation remains a
fully coherent coupled-mode transfer-matrix calculation. Fine ripple phase is intentionally
not used to infer temperature because it is non-unique without the complex spectrum.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import differential_evolution

from digitize_spectra import RGB, digitize
from fbg_model import FBGParams, PEAK_TEMP_C, POWER_MW, fbg_shift_nm, reflection_spectrum
from reproduce_spectra import reflected_power_dbm


OUT = Path("outputs")
FIT_POWERS = (73, 131, 188, 244, 296, 367)
OSA_FWHM_NM = 0.020
COMPARISON_FWHM_NM = 0.060


def asymmetric_hotspot(
    z_m: np.ndarray,
    center_mm: float,
    total_fwhm_mm: float,
    left_fraction: float,
    left_order: float,
    right_order: float,
) -> np.ndarray:
    """Unit-peak, continuous asymmetric generalized-Gaussian temperature shape."""
    center = center_mm * 1e-3
    left_half = total_fwhm_mm * left_fraction * 1e-3
    right_half = total_fwhm_mm * (1.0 - left_fraction) * 1e-3
    left_scale = left_half / np.log(2.0) ** (1.0 / left_order)
    right_scale = right_half / np.log(2.0) ** (1.0 / right_order)
    distance = np.abs(z_m - center)
    return np.where(
        z_m <= center,
        np.exp(-((distance / left_scale) ** left_order)),
        np.exp(-((distance / right_scale) ** right_order)),
    )


def prepare_targets(step_nm: float = 0.018):
    traces = digitize()
    targets = {}
    for power in FIT_POWERS:
        wavelength_raw, dbm_raw = traces[power]
        wavelength = np.arange(wavelength_raw.min(), wavelength_raw.max(), step_nm)
        dbm = np.interp(wavelength, wavelength_raw, dbm_raw)
        # Suppress screenshot pixel noise and fine phase-sensitive fringes for inversion.
        sigma = COMPARISON_FWHM_NM / 2.355 / step_nm
        linear = gaussian_filter1d(10.0 ** (dbm / 10.0), sigma)
        dbm = 10.0 * np.log10(linear)
        valid = dbm > -43.25
        targets[power] = (wavelength[valid], dbm[valid])
    return targets


def temperature_for_power(power: float, shape: np.ndarray) -> np.ndarray:
    peak = np.interp(power, POWER_MW, PEAK_TEMP_C)
    return 23.0 + (peak - 23.0) * shape


def forward_dbm(wavelength: np.ndarray, power: float, shape: np.ndarray, fbg: FBGParams):
    local_bragg = fbg.lambda0_nm + fbg_shift_nm(temperature_for_power(power, shape))
    reflected = reflection_spectrum(wavelength, local_bragg, fbg)
    dw = np.median(np.diff(wavelength))
    reflected = gaussian_filter1d(reflected, OSA_FWHM_NM / 2.355 / dw)
    reflected = gaussian_filter1d(reflected, COMPARISON_FWHM_NM / 2.355 / dw)
    return reflected_power_dbm(reflected, launched_dbm=-31.05)


def fit(maxiter: int = 32, popsize: int = 8, seed: int = 7):
    targets = prepare_targets()
    fbg = replace(FBGParams(), segments=260)
    dz = fbg.length_m / fbg.segments
    z = (np.arange(fbg.segments) + 0.5) * dz

    def objective(x):
        center, fwhm, left_fraction, left_order, right_order = x
        shape = asymmetric_hotspot(z, center, fwhm, left_fraction, left_order, right_order)
        losses = []
        for power, (wavelength, measured) in targets.items():
            modeled = forward_dbm(wavelength, power, shape, fbg)
            # Eliminate one absolute-height nuisance parameter per trace. It cannot alter
            # spectral width, peak locations, envelope, or ripple spacing.
            offset = np.median(measured - modeled)
            residual = (modeled + offset - measured) / 0.55
            losses.append(np.mean(2.0 * (np.sqrt(1.0 + residual * residual) - 1.0)))
        # Use the independent camera FWHM as a soft, not exact, constraint.
        fwhm_penalty = 0.35 * ((fwhm - 2.8) / 0.35) ** 2
        return float(np.mean(losses) + fwhm_penalty)

    bounds = [
        (1.0, 9.0),   # hotspot center, mm
        (2.0, 4.2),   # total FWHM, mm
        (0.15, 0.85), # fraction of FWHM on left
        (0.65, 4.5),  # left generalized-Gaussian order
        (0.65, 4.5),  # right generalized-Gaussian order
    ]
    result = differential_evolution(
        objective,
        bounds,
        maxiter=maxiter,
        popsize=popsize,
        seed=seed,
        polish=True,
        workers=1,
        updating="immediate",
        tol=2e-3,
    )
    return result, targets, fbg, z


def save_result(result, targets, fbg, z):
    center, fwhm, left_fraction, left_order, right_order = result.x
    shape = asymmetric_hotspot(z, center, fwhm, left_fraction, left_order, right_order)
    params = {
        "objective": float(result.fun),
        "success": bool(result.success),
        "message": str(result.message),
        "hotspot_center_mm": float(center),
        "total_fwhm_mm": float(fwhm),
        "left_halfwidth_mm": float(fwhm * left_fraction),
        "right_halfwidth_mm": float(fwhm * (1.0 - left_fraction)),
        "left_shape_order": float(left_order),
        "right_shape_order": float(right_order),
        "osa_fwhm_nm": OSA_FWHM_NM,
        "comparison_smoothing_fwhm_nm": COMPARISON_FWHM_NM,
    }
    (OUT / "recovered_temperature_parameters.json").write_text(json.dumps(params, indent=2) + "\n")

    z_plot = np.linspace(0.0, fbg.length_m, 1001)
    shape_plot = asymmetric_hotspot(z_plot, center, fwhm, left_fraction, left_order, right_order)
    np.savetxt(
        OUT / "recovered_temperature_shape.csv",
        np.column_stack((z_plot * 1e3, shape_plot)),
        delimiter=",",
        header="z_mm,normalized_temperature_rise",
        comments="",
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    axes[0].plot(z_plot * 1e3, shape_plot, label="recovered asymmetric profile")
    axes[0].axhline(0.5, color="0.5", ls="--", lw=1)
    axes[0].set(xlabel="FBG position (mm)", ylabel="Normalized temperature rise", xlim=(0, 10), ylim=(0, 1.05))
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False)

    color_map = {int(p): c / 255.0 for p, c in zip([17, 73, 131, 188, 244, 296, 344, 367], RGB)}
    for power, (wavelength, measured) in targets.items():
        modeled = forward_dbm(wavelength, power, shape, fbg)
        offset = np.median(measured - modeled)
        axes[1].plot(wavelength, measured, lw=1.4, color=color_map[power], alpha=0.55)
        axes[1].plot(wavelength, modeled + offset, lw=1.4, color=color_map[power], ls="--")
    axes[1].set(xlabel="Wavelength (nm)", ylabel="Reflection (dBm)", xlim=(1547, 1554.2), ylim=(-44, -33))
    axes[1].grid(alpha=0.2)
    axes[1].text(0.02, 0.03, "solid: digitized; dashed: recovered-profile model", transform=axes[1].transAxes)
    fig.tight_layout()
    fig.savefig(OUT / "recovered_temperature_fit.png", dpi=180)
    print(json.dumps(params, indent=2))


def main():
    OUT.mkdir(exist_ok=True)
    result, targets, fbg, z = fit()
    save_result(result, targets, fbg, z)


if __name__ == "__main__":
    main()
