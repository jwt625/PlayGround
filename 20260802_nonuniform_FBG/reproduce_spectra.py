"""Generate the reconstructed spectra and thermal-model diagnostic plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

from fbg_model import (
    FBGParams,
    LABELED_PEAK_NM,
    PEAK_TEMP_C,
    POWER_MW,
    ThermalParams,
    exponential_temperature_c,
    fbg_shift_nm,
    fin_temperature_rise_k,
    hotspot_temperature_c,
    recovered_hotspot_temperature_c,
    spectrum_at_power,
)


OUT = Path("outputs")
COLORS = ["#333333", "#f33", "#1675d1", "#25a95a", "#a66bd4", "#cf8b00", "#13bfc3", "#87504b", "#929000"]


def measurement_scale_db(power_mw: float) -> float:
    """Empirical peak-height correction digitized from the heated traces.

    This is deliberately a measurement/coherence nuisance term, not heat or FBG loss.
    A(P)=-4.50[1-exp(-P/66.87)] dB fits the cold-envelope peak reduction to ~0.2 dB RMS.
    """
    return -4.5001 * (1.0 - np.exp(-power_mw / 66.866))


def reflected_power_dbm(
    reflectivity: np.ndarray,
    launched_dbm: float = -31.05,
    scale_db: float = 0.0,
) -> np.ndarray:
    """Map reflectivity to the paper's absolute vertical scale and -44 dBm floor."""
    signal_mw = 10.0 ** ((launched_dbm + scale_db) / 10.0) * reflectivity
    floor_mw = 10.0 ** (-44.0 / 10.0)
    return 10.0 * np.log10(signal_mw + floor_mw)


def plot_spectra() -> None:
    wavelength = np.linspace(1546.0, 1556.0, 3201)
    fbg = FBGParams()
    fig, ax = plt.subplots(figsize=(9.0, 7.0))
    rows = []
    exported = [wavelength]
    exported_header = ["wavelength_nm"]
    for power, target_peak, color in zip(POWER_MW, LABELED_PEAK_NM, COLORS):
        reflectivity, _, temp = spectrum_at_power(wavelength, power, fbg=fbg)
        # A representative 20 pm OSA resolution is a stated assumption; raw metadata
        # are unavailable. Convolution is performed in linear reflected power.
        sigma_samples = 0.020 / 2.355 / (wavelength[1] - wavelength[0])
        reflectivity = gaussian_filter1d(reflectivity, sigma_samples)
        dbm = reflected_power_dbm(
            reflectivity, scale_db=measurement_scale_db(float(power))
        )
        exported.append(dbm)
        exported_header.append(f"power_{power:g}_mW_dbm")
        ax.plot(wavelength, dbm, lw=2.0, color=color, label=f"{power:g} mW")
        # Compare the matching modeled local maximum with the labeled detected peak.
        candidates = np.flatnonzero((dbm[1:-1] > dbm[:-2]) & (dbm[1:-1] >= dbm[2:]) & (dbm[1:-1] > -43.0)) + 1
        nearby = candidates[np.abs(wavelength[candidates] - target_peak) < 0.30]
        simulated_peak = (
            wavelength[nearby[np.argmin(np.abs(wavelength[nearby] - target_peak))]]
            if len(nearby)
            else np.nan
        )
        rows.append((power, temp.max(), target_peak, simulated_peak))
    ax.set(xlim=(1546, 1556), ylim=(-44, -30), xlabel="Wavelength (nm)", ylabel="Reflection power (dBm)")
    ax.grid(alpha=0.18)
    ax.legend(ncol=1, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "reconstructed_spectra.png", dpi=180)
    np.savetxt(
        OUT / "peak_comparison.csv",
        np.asarray(rows),
        delimiter=",",
        header="power_mW,model_peak_temperature_C,labeled_peak_nm,modeled_matching_peak_nm",
        comments="",
        fmt="%.6g",
    )
    np.savetxt(
        OUT / "reconstructed_spectra.csv",
        np.column_stack(exported),
        delimiter=",",
        header=",".join(exported_header),
        comments="",
        fmt="%.8g",
    )


def plot_thermal_models() -> None:
    thermal = ThermalParams()
    z = np.linspace(0.0, thermal.length_m, 1200)
    exact = fin_temperature_rise_k(z, 0.367, thermal)
    exact *= (PEAK_TEMP_C[-1] - thermal.ambient_c) / exact.max()  # compare shapes, not h calibration
    effective = recovered_hotspot_temperature_c(z, PEAK_TEMP_C[-1], thermal.ambient_c) - thermal.ambient_c
    beer = exponential_temperature_c(
        z, PEAK_TEMP_C[-1], thermal.ambient_c, thermal.absorption_np_m
    ) - thermal.ambient_c

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(z * 1e3, exact, label="exact fin + Beer–Lambert (shape-normalized)")
    ax.plot(z * 1e3, beer, "--", label=f"local equilibrium, alpha={thermal.absorption_np_m/1e3:.3f} mm^-1")
    ax.plot(z * 1e3, effective, label="spectrum-recovered asymmetric hotspot, FWHM=2.74 mm")
    ax.set(xlabel="Position from heater input (mm)", ylabel="Temperature rise (K)", xlim=(0, 10), ylim=(0, 480))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "thermal_profile_models.png", dpi=180)
    np.savetxt(
        OUT / "thermal_profile_models.csv",
        np.column_stack((z * 1e3, exact, beer, effective)),
        delimiter=",",
        header="z_mm,exact_fin_beer_lambert_rise_K,local_beer_lambert_rise_K,effective_hotspot_rise_K",
        comments="",
        fmt="%.8g",
    )


def plot_power_calibration() -> None:
    p_dense = np.linspace(0.0, POWER_MW[-1], 400)
    # Compact quadratic regression is included only as a smooth engineering surrogate.
    coeff = np.polyfit(POWER_MW, PEAK_TEMP_C, 2)
    fit = np.polyval(coeff, p_dense)
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.plot(POWER_MW, PEAK_TEMP_C, "s", label="FBG points digitized from Fig. 5(b)")
    ax.plot(p_dense, fit, label=f"quadratic: T={coeff[0]:.4g}P^2+{coeff[1]:.4g}P+{coeff[2]:.3g}")
    ax.set(xlabel="Optical power (mW)", ylabel="Peak temperature (degC)", xlim=(0, 390), ylim=(0, 510))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "power_temperature_fit.png", dpi=180)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    plot_spectra()
    plot_thermal_models()
    plot_power_calibration()
    print(f"Wrote figures and CSV to {OUT.resolve()}")


if __name__ == "__main__":
    main()
