"""Digitize the colored traces in FBG-spectrum.png using the published plot axes."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


POWERS = np.array([17, 73, 131, 188, 244, 296, 344, 367], dtype=float)
RGB = np.array(
    [
        [238, 52, 55],
        [24, 94, 211],
        [48, 161, 96],
        [166, 103, 211],
        [198, 141, 35],
        [25, 196, 197],
        [106, 60, 60],
        [131, 131, 33],
    ],
    dtype=float,
)
# Manual exclusion bounds remove legend samples and colored annotation artifacts.
MAX_WAVELENGTH_NM = np.array([1548.35, 1549.55, 1550.55, 1551.35, 1551.85, 1552.95, 1553.75, 1554.10])
MAX_DBM = np.array([-32.0, -34.0, -35.0, -35.8, -36.3, -36.5, -37.0, -37.0])

# Plot-spine calibration in the 1436x1188 supplied PNG.
X_LEFT, X_RIGHT = 201, 1357
Y_TOP, Y_BOTTOM = 63, 994


def x_to_nm(x: np.ndarray) -> np.ndarray:
    return 1546.0 + (x - X_LEFT) * 10.0 / (X_RIGHT - X_LEFT)


def y_to_dbm(y: np.ndarray) -> np.ndarray:
    return -30.0 - (y - Y_TOP) * 14.0 / (Y_BOTTOM - Y_TOP)


def digitize(path: str | Path = "FBG-spectrum.png", rgb_distance: float = 42.0):
    image = np.asarray(Image.open(path).convert("RGB"), dtype=float)
    traces = {}
    for power, color, max_nm, max_dbm in zip(POWERS, RGB, MAX_WAVELENGTH_NM, MAX_DBM):
        mask = np.linalg.norm(image - color, axis=2) < rgb_distance
        wavelength, power_dbm = [], []
        for x in range(X_LEFT + 2, X_RIGHT - 1):
            yy = np.flatnonzero(mask[Y_TOP + 1 : Y_BOTTOM, x]) + Y_TOP + 1
            if yy.size == 0:
                continue
            dbm = y_to_dbm(yy)
            yy = yy[dbm < max_dbm]
            if yy.size:
                wavelength.append(x_to_nm(x))
                power_dbm.append(y_to_dbm(np.median(yy)))
        wavelength = np.asarray(wavelength)
        power_dbm = np.asarray(power_dbm)
        valid = (
            (wavelength >= 1546.75)
            & (wavelength <= max_nm)
            & (power_dbm >= -43.65)
            & (power_dbm <= max_dbm)
        )
        traces[int(power)] = (wavelength[valid], power_dbm[valid])
    return traces


def main() -> None:
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    traces = digitize()
    all_wavelength = np.unique(np.concatenate([v[0] for v in traces.values()]))
    columns = [all_wavelength]
    header = ["wavelength_nm"]
    for power, (wavelength, dbm) in traces.items():
        values = np.full(all_wavelength.shape, np.nan)
        indices = np.searchsorted(all_wavelength, wavelength)
        values[indices] = dbm
        columns.append(values)
        header.append(f"power_{power}_mW_dbm")
    np.savetxt(
        out / "digitized_heated_spectra.csv",
        np.column_stack(columns),
        delimiter=",",
        header=",".join(header),
        comments="",
        fmt="%.8g",
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    for power, color, (wavelength, dbm) in zip(POWERS, RGB / 255.0, traces.values()):
        ax.plot(wavelength, dbm, ".", ms=2, color=color, label=f"{power:g} mW")
    ax.set(xlim=(1546, 1556), ylim=(-44, -30), xlabel="Wavelength (nm)", ylabel="Digitized reflection (dBm)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "digitized_heated_spectra.png", dpi=180)


if __name__ == "__main__":
    main()
