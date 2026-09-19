"""Run orchestration and immutable run records (TODO T01/T07, R4)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import meep as mp
import numpy as np

from .config import CouplerConfig
from .geometry import Plan, build_geometry, make_plan
from .ports import (
    add_oblique_mode_monitor,
    add_output_dft,
    extraction_line,
    freqs_to_nm,
    line_flux,
    oblique_modal_powers,
    run_incident_reference,
)
from .sources import build_source
from .units import freq_from_wavelength_nm


DF_FREQ_WIDTH = 0.05
DF_NFREQ = 51


def _freq_grid(cfg: CouplerConfig) -> np.ndarray:
    fcen = freq_from_wavelength_nm(cfg.source.wavelength_nm)
    return np.linspace(fcen - DF_FREQ_WIDTH / 2, fcen + DF_FREQ_WIDTH / 2, DF_NFREQ)


def _stop_condition(cfg: CouplerConfig):
    comp = mp.Ez if cfg.source.branch == "ez" else mp.Hz
    pt = mp.Vector3(*[0.0, 0.0, 0.0])
    return mp.stop_when_fields_decayed(50, comp, pt, cfg.numerics.decay)


def run_coupler(cfg: CouplerConfig, *, verbose: bool = False, compute_reference: bool = True) -> dict:
    errors = cfg.validate()
    if errors:
        raise ValueError("; ".join(errors))
    plan = make_plan(cfg)
    geom, plan, idx = build_geometry(cfg)
    src = build_source(cfg, plan)
    sim = mp.Simulation(
        cell_size=mp.Vector3(plan.domain["sx"], plan.domain["sy"], 0),
        resolution=cfg.numerics.resolution,
        geometry=geom,
        sources=src,
        boundary_layers=[mp.PML(cfg.numerics.pml_um)],
        default_material=idx.medium("clad"),
        dimensions=2,
        Courant=cfg.numerics.cour,
    )
    g = cfg.geometry
    s_mon = g.stack_start_um + g.stack_length_um + min(
        cfg.monitor.output_offset_um, g.guide_ext_down_um
    )
    mmon, mcenter = add_oblique_mode_monitor(sim, cfg, plan, s_mon)
    line = extraction_line(cfg, plan)
    fmon, fcenter, fsize = add_output_dft(sim, cfg, line)
    freqs = _freq_grid(cfg)
    if verbose:
        mp.verbosity(1)

    if cfg.numerics.stop_on_decay:
        comp = mp.Ez if cfg.source.branch == "ez" else mp.Hz
        stop = mp.stop_when_fields_decayed(50, comp, mp.Vector3(), cfg.numerics.decay)
        sim.run(until_after_sources=stop)
    else:
        sim.run(until_after_sources=cfg.numerics.until_after_sources)

    p_out = np.array([line_flux(sim, fmon, fcenter, fsize, line, i) for i in range(len(freqs))])
    modal = oblique_modal_powers(sim, mmon, cfg, plan)

    result = {
        "freqs": freqs,
        "wavelength_nm": freqs_to_nm(freqs),
        "P_out": p_out,
        "modal": modal,
        "line": {
            "center": line.center_xy.tolist(),
            "s_ext_um": line.s_ext_um,
            "q_lo": line.q_lo,
            "q_hi": line.q_hi,
        },
        "monitor": {
            "s_mon_um": s_mon,
            "center": [mcenter.x, mcenter.y, mcenter.z],
        },
        "domain": plan.domain,
    }
    if compute_reference:
        ref = run_incident_reference(cfg, plan)
        result["P_inc"] = np.array(ref["flux"])
        p_inc = np.where(np.abs(result["P_inc"]) > 0, result["P_inc"], np.nan)
        eta = p_out.real / p_inc
        result["eta"] = eta
        result["eta_dB"] = np.where(eta > 0, 10 * np.log10(np.abs(eta)), np.nan)
    return result


# --------------------------------------------------------------------------
# Run records
# --------------------------------------------------------------------------
def run_dir(cfg: CouplerConfig, root: str | None = None) -> Path:
    base = Path(root or cfg.run.root)
    return base / f"{cfg.run.name}_{cfg.config_hash()}"


def save_run(cfg: CouplerConfig, result: dict, *, status: str = "complete", extra: dict | None = None) -> Path:
    d = run_dir(cfg)
    if d.exists():
        d = Path(str(d) + "_" + time.strftime("%Y%m%dT%H%M%S"))
    d.mkdir(parents=True, exist_ok=True)
    cfg.save(d / "config.effective.yaml")
    data = {
        "wavelength_nm": result["wavelength_nm"],
        "P_out_real": result["P_out"].real,
        "P_out_imag": result["P_out"].imag,
    }
    for band, payload in result.get("modal", {}).items():
        data[f"modal_fwd_b{band}"] = payload["forward"]
        data[f"modal_bwd_b{band}"] = payload["backward"]
        data[f"modal_neff_b{band}"] = payload["neff"]
    if "P_inc" in result:
        data["P_inc"] = result["P_inc"]
        data["eta"] = result["eta"]
        data["eta_dB"] = result["eta_dB"]
    np.savez(d / "spectrum.npz", **data)
    meta = {
        "status": status,
        "config_hash": cfg.config_hash(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "solver": {"meep": mp.__version__},
        "domain": result.get("domain"),
        "line": result.get("line"),
        "monitor": result.get("monitor"),
        "warnings": [],
    }
    if extra:
        meta.update(extra)
    (d / "status.json").write_text(json.dumps(meta, indent=2, default=str))
    return d
