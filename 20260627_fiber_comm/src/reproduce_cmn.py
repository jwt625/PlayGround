from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


@dataclass(frozen=True)
class FiberParams:
    wavelength_nm: float = 1550.0
    dispersion_ps_nm_km: float = 17.0
    length_km: float = 3600.0
    symbol_rate_gbaud: float = 16.0
    rrc_rolloff: float = 0.1


@dataclass(frozen=True)
class GridParams:
    max_index: int = 45
    samples_per_symbol: int = 8
    fft_symbols: int = 1024
    z_steps: int = 81


def beta2_ps2_per_km(params: FiberParams) -> float:
    """Convert D [ps/(nm km)] to beta2 [ps^2/km]."""
    c_nm_per_ps = 299_792.458
    lam = params.wavelength_nm
    return -(params.dispersion_ps_nm_km * lam * lam) / (2.0 * np.pi * c_nm_per_ps)


def rrc_frequency_response(freq_cycles_per_symbol: np.ndarray, rolloff: float) -> np.ndarray:
    """Unitless root-raised-cosine amplitude response versus f*T."""
    f = np.abs(freq_cycles_per_symbol)
    h_rc = np.zeros_like(f, dtype=float)

    if rolloff <= 0:
        h_rc[f <= 0.5] = 1.0
        return np.sqrt(h_rc)

    f1 = (1.0 - rolloff) / 2.0
    f2 = (1.0 + rolloff) / 2.0

    h_rc[f <= f1] = 1.0
    transition = (f > f1) & (f <= f2)
    h_rc[transition] = 0.5 * (
        1.0
        + np.cos(
            np.pi
            / rolloff
            * (f[transition] - f1)
        )
    )
    return np.sqrt(h_rc)


def dispersed_rrc_pulse(params: FiberParams, grid: GridParams, z_km: float) -> np.ndarray:
    n = grid.samples_per_symbol * grid.fft_symbols
    dt_ps = 1.0e3 / params.symbol_rate_gbaud / grid.samples_per_symbol
    freqs_per_ps = np.fft.fftfreq(n, d=dt_ps)
    freqs_per_symbol = freqs_per_ps * (1.0e3 / params.symbol_rate_gbaud)

    h_rrc = rrc_frequency_response(freqs_per_symbol, params.rrc_rolloff)
    omega_rad_per_ps = 2.0 * np.pi * freqs_per_ps
    phase = np.exp(-0.5j * beta2_ps2_per_km(params) * omega_rad_per_ps**2 * z_km)
    pulse = np.fft.ifft(h_rrc * phase)

    # Center the peak near t = 0 and normalize energy. Absolute scale cancels in the final dB plot.
    pulse = np.fft.fftshift(pulse)
    energy = np.sum(np.abs(pulse) ** 2) * dt_ps
    return pulse / np.sqrt(energy)


def shifted_bank(pulse: np.ndarray, max_shift_symbols: int, samples_per_symbol: int) -> dict[int, np.ndarray]:
    return {
        shift: np.roll(pulse, shift * samples_per_symbol)
        for shift in range(-max_shift_symbols, max_shift_symbols + 1)
    }


def overlap_coefficients(params: FiberParams, grid: GridParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute normalized C_m,n coefficients for symmetric EDC.

    The implementation follows the form used in Cartledge et al. Eq. (5)-(7):
    C_m,n is proportional to the integral over z of
    int conj(u_z(t)) u_z(t-nT) u_z(t-mT) conj(u_z(t-(m+n)T)) dt.
    Constant prefactors and the z-independent nonlinear coefficient cancel after
    normalization to C_0,0.
    """
    m_values = np.arange(-grid.max_index, grid.max_index + 1)
    n_values = np.arange(-grid.max_index, grid.max_index + 1)
    coeff = np.zeros((len(n_values), len(m_values)), dtype=np.complex128)

    z_values = np.linspace(0.0, params.length_km / 2.0, grid.z_steps)
    weights = np.ones_like(z_values)
    weights[0] = 0.5
    weights[-1] = 0.5
    dz = z_values[1] - z_values[0] if len(z_values) > 1 else 1.0

    dt_ps = 1.0e3 / params.symbol_rate_gbaud / grid.samples_per_symbol
    max_combined_shift = 2 * grid.max_index

    for zi, (z_km, weight) in enumerate(zip(z_values, weights), start=1):
        pulse = dispersed_rrc_pulse(params, grid, z_km)
        conj_pulse = np.conjugate(pulse)
        shifts = shifted_bank(pulse, max_combined_shift, grid.samples_per_symbol)
        conj_shifts = {key: np.conjugate(value) for key, value in shifts.items()}

        for mi, m in enumerate(m_values):
            u_m = shifts[int(m)]
            for ni, n in enumerate(n_values):
                integrand = conj_pulse * shifts[int(n)] * u_m * conj_shifts[int(m + n)]
                coeff[ni, mi] += weight * np.sum(integrand) * dt_ps

        print(f"z step {zi:03d}/{len(z_values)}: {z_km:8.2f} km", flush=True)

    coeff *= dz
    magnitude = np.abs(coeff)
    c00 = magnitude[grid.max_index, grid.max_index]
    db = 20.0 * np.log10(np.maximum(magnitude / c00, 1.0e-12))
    return m_values, n_values, np.clip(db, -40.0, 0.0)


def osa_like_colorscale() -> list[list[float | str]]:
    colors = [
        "#000000",
        "#7f007f",
        "#5d00ff",
        "#19b5e5",
        "#00f000",
        "#caff00",
        "#ffb800",
        "#ff1111",
        "#d80000",
        "#8c0000",
    ]
    stops = np.linspace(0.0, 1.0, len(colors))
    return [[float(stop), color] for stop, color in zip(stops, colors)]


def build_figure(m_values: np.ndarray, n_values: np.ndarray, db: np.ndarray) -> go.Figure:
    fig = go.Figure(
        data=go.Contour(
            z=db,
            x=m_values,
            y=n_values,
            zmin=-40,
            zmax=0,
            contours=dict(
                coloring="fill",
                start=-35,
                end=0,
                size=3.5,
                showlines=True,
            ),
            colorscale=osa_like_colorscale(),
            colorbar=dict(
                title=dict(text="C<sub>m,n</sub> (dB)", side="top"),
                tickmode="array",
                tickvals=[0, -3.5, -7, -10.5, -14, -17.5, -21, -24.5, -28, -31.5, -35],
            ),
            hovertemplate="m=%{x}<br>n=%{y}<br>C=%{z:.2f} dB<extra></extra>",
        )
    )
    fig.update_layout(
        title="Normalized C<sub>m,n</sub> coefficients for 3600 km standard SMF",
        width=980,
        height=760,
        plot_bgcolor="black",
        paper_bgcolor="white",
        xaxis=dict(title="Index <i>m</i>", range=[-45, 45], zeroline=False, showgrid=False, dtick=20),
        yaxis=dict(
            title="Index <i>n</i>",
            range=[-45, 45],
            zeroline=False,
            showgrid=False,
            dtick=20,
            scaleanchor="x",
        ),
        margin=dict(l=80, r=110, t=80, b=80),
        font=dict(size=18),
    )
    return fig


def make_plot(m_values: np.ndarray, n_values: np.ndarray, db: np.ndarray, out_html: Path) -> None:
    fig = build_figure(m_values, n_values, db)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="cdn", full_html=True)


def save_artifacts(m_values: np.ndarray, n_values: np.ndarray, db: np.ndarray, npz_path: Path) -> None:
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, m=m_values, n=n_values, cmn_db=db)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-html", default="outputs/cmn_3600km_interactive.html")
    parser.add_argument("--out-npz", default="outputs/cmn_3600km_data.npz")
    parser.add_argument("--from-npz", help="reuse a previously computed coefficient NPZ")
    parser.add_argument("--max-index", type=int, default=45)
    parser.add_argument("--sps", type=int, default=8)
    parser.add_argument("--fft-symbols", type=int, default=1024)
    parser.add_argument("--z-steps", type=int, default=81)
    parser.add_argument("--rolloff", type=float, default=0.1)
    parser.add_argument("--symbol-rate-gbaud", type=float, default=16.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.from_npz:
        data = np.load(args.from_npz)
        make_plot(data["m"], data["n"], data["cmn_db"], Path(args.out_html))
        print(f"read {args.from_npz}")
        print(f"wrote {args.out_html}")
        return

    params = FiberParams(
        rrc_rolloff=args.rolloff,
        symbol_rate_gbaud=args.symbol_rate_gbaud,
    )
    grid = GridParams(
        max_index=args.max_index,
        samples_per_symbol=args.sps,
        fft_symbols=args.fft_symbols,
        z_steps=args.z_steps,
    )
    m_values, n_values, db = overlap_coefficients(params, grid)
    save_artifacts(m_values, n_values, db, Path(args.out_npz))
    make_plot(m_values, n_values, db, Path(args.out_html))
    print(f"wrote {args.out_npz}")
    print(f"wrote {args.out_html}")


if __name__ == "__main__":
    main()
