"""Source construction.

Primary source: an in-plane 2D Gaussian beam in homogeneous silicon, matching
the paper's ``w0 = 5 um`` 1/e amplitude radius and the SMF-28 fundamental mode.
Meep's ``GaussianBeam2DSource`` is a true 2D beam solution, preferred over the
3D cross-section approximation (SIMULATION_SPEC.md s3).
"""

from __future__ import annotations

import math

import meep as mp

from .config import CouplerConfig
from .geometry import Plan
from .units import freq_from_wavelength_nm

SOURCE_HALF_SPAN_W0 = 3.5  # source line half-length in units of w0


def branch_vector(branch: str) -> mp.Vector3:
    if branch == "ez":
        return mp.Vector3(0, 0, 1)  # E out of the incidence plane (s polarization)
    if branch == "hz":
        return mp.Vector3(0, 1, 0)  # E in the incidence plane (p polarization)
    raise ValueError(f"unknown branch {branch!r}")


def build_source(cfg: CouplerConfig, plan: Plan, *, cw: bool = False) -> list[mp.Source]:
    s = cfg.source
    freq0 = freq_from_wavelength_nm(s.wavelength_nm)
    if cw:
        time = mp.ContinuousSource(freq0, width=20.0)
    else:
        time = mp.GaussianSource(freq0, fwidth=s.fwidth)

    center = plan.meep_xy(s.x_source_um, s.y_source_um)
    focus = plan.points["focus_xy"]
    beam_x0 = mp.Vector3(focus[0] - s.x_source_um, focus[1] - s.y_source_um, 0)
    tilt = mp.Vector3(1, 0, 0).rotate(mp.Vector3(0, 0, 1), math.radians(s.tilt_deg))

    return [
        mp.GaussianBeam2DSource(
            src=time,
            center=center,
            size=mp.Vector3(0, SOURCE_HALF_SPAN_W0 * 2 * s.w0_um, 0),
            beam_x0=beam_x0,
            beam_kdir=tilt,
            beam_w0=s.w0_um,
            beam_E0=branch_vector(s.branch),
        )
    ]
