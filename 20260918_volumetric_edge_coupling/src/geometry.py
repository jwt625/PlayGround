"""Geometry contract for the volumetric edge coupler (SIMULATION_SPEC.md s2).

Paper frame
-----------
* ``x``: nominal incident-beam direction (horizontal).
* ``y``: height above the cavity base.
* ``z``: finite width / out-of-plane.

Sidewall coordinates
--------------------
``u = (cos a, -sin a)`` runs down the sidewall, ``n = (sin a, cos a)`` points
from the substrate toward the oxide.  ``r(s) = r_top + s u`` with ``s = 0`` at
the top of the oxide/guide stack.  The substrate-facing guide face is
``q = g(s) = g_ref + (s - s_ref) tan(a - theta_wg)``; the guide's other face is
offset by the physical normal thickness ``twg`` along ``m``.  The substrate is
the half-space ``q < 0`` (an infinite inclined interface) in the primary
reduced model; the finite cavity floor and wafer surface are documented
secondary hypotheses.

All coordinates are in micrometres.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import meep as mp
import numpy as np

from .config import CouplerConfig
from .materials import IndexSet


@dataclass
class Plan:
    cfg: CouplerConfig
    u: np.ndarray
    n: np.ndarray
    d: np.ndarray
    m: np.ndarray
    r_top: np.ndarray
    twg_um: float
    s_guide_lo: float
    s_guide_hi: float
    oxide_poly: np.ndarray
    guide_poly: np.ndarray
    substrate: dict
    domain: dict
    points: dict
    shift: np.ndarray

    def meep_xy(self, x: float, y: float) -> mp.Vector3:
        """Map paper-frame (x, y) to the Meep cell centred at the origin."""
        return mp.Vector3(float(x) + self.shift[0], float(y) + self.shift[1], 0)

    def boundary(self) -> dict:
        return {
            "alpha_deg": self.cfg.geometry.alpha_deg,
            "theta_wg_deg": self.cfg.geometry.theta_wg_deg,
            "twg_nm": self.cfg.geometry.twg_nm,
            "depth_um": self.cfg.geometry.depth_um,
            "g_ref_nm": self.cfg.geometry.g_ref_nm,
            "s_ref_um": self.cfg.geometry.s_ref_um,
            "gap_slope": self.cfg.geometry.gap_slope,
            "s_impact_um": self.cfg.geometry.s_impact(self.cfg.source.y_source_um),
            "gap_at_impact_um": self.cfg.geometry.gap_at(
                self.cfg.geometry.s_impact(self.cfg.source.y_source_um)
            ),
            "s_guide": [self.s_guide_lo, self.s_guide_hi],
            "gap_lo_um": self.cfg.geometry.gap_at(self.s_guide_lo),
            "gap_hi_um": self.cfg.geometry.gap_at(self.s_guide_hi),
            "r_top": self.r_top.tolist(),
            "guide_poly": self.guide_poly.tolist(),
            "oxide_poly": self.oxide_poly.tolist(),
            "substrate": self.substrate,
            "domain": self.domain,
            "points": self.points,
        }


def _r(plan_params, s):
    return plan_params


def make_plan(cfg: CouplerConfig) -> Plan:
    g = cfg.geometry
    a = g.alpha_rad
    th = g.theta_rad
    u = np.array([math.cos(a), -math.sin(a)])
    n = np.array([math.sin(a), math.cos(a)])
    d = np.array([math.cos(th), -math.sin(th)])
    m = np.array([math.sin(th), math.cos(th)])
    twg = g.twg_nm / 1000.0
    r_top = np.array([g.x_top_um, g.depth_um])

    def r_of(s):
        return r_top + s * u

    def p_lo(s):
        return r_of(s) + g.gap_at(s) * n

    beta = a - th
    cos_beta = math.cos(beta)

    def q_sub(s):
        """Substrate/oxide boundary offset along n (isolated-output-port-v1).

        C1 Hermite blend from q_sub=0 (sidewall) at the splice to the line
        parallel to the guide at normal clearance D.  Downstream this equals
        ``g(s) - D/cos(beta)``, which has the same slope as ``g(s)``.
        """
        if not g.isolated_port:
            return 0.0
        s0 = g.port_start_s_um
        s1 = s0 + g.port_transition_um
        if s <= s0:
            return 0.0
        if s >= s1:
            return g.gap_at(s) - g.oxide_clearance_um / cos_beta
        L = s1 - s0
        q1 = g.gap_at(s1) - g.oxide_clearance_um / cos_beta
        m1 = math.tan(beta)
        t = (s - s0) / L
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2
        return float(h01 * q1 + h11 * L * m1)

    def b_sub(s):
        return r_of(s) + q_sub(s) * n

    s_lo = g.stack_start_um - g.guide_ext_up_um
    s_hi = g.stack_start_um + g.stack_length_um + g.guide_ext_down_um
    if g.isolated_port:
        s_port_end = g.port_start_s_um + g.port_transition_um + g.port_length_um
    else:
        s_port_end = s_hi
    # uniform materials extend through the PML; domain is sized only to s_port_end
    s_absorb = s_port_end + g.guide_absorb_um
    oxide_lo = g.stack_start_um

    # single master vertex sequence so oxide and substrate share identical
    # vertices on their common boundary (no overlap/air slivers)
    s_shared = np.linspace(oxide_lo, s_absorb, 400)

    def band_polys(s_end, s_ox=None):
        s_ox = s_shared if s_ox is None else s_ox
        Bnd = np.array([b_sub(s) for s in s_ox])
        P = np.array([p_lo(s) for s in s_ox])
        oxide = np.vstack([Bnd, P[::-1]])
        guide = np.array(
            [p_lo(s_lo), p_lo(s_end), p_lo(s_end) + twg * m, p_lo(s_lo) + twg * m]
        )
        return oxide, guide

    s_ox_dom = np.linspace(oxide_lo, s_port_end, 300)
    oxide_poly, guide_poly = band_polys(s_absorb)
    oxide_poly_dom, guide_poly_dom = band_polys(s_port_end, s_ox_dom)

    # substrate: half-space q < q_sub(s); a polygon when the port is active
    B = 120.0
    substrate = {
        "center": (r_top - 0.5 * B * n).tolist(),
        "size": [2.0 * B, B, float("inf")],
        "e1": u.tolist(),
        "e2": n.tolist(),
        "poly": None,
    }
    if g.isolated_port:
        # include the oxide's shared vertices exactly, plus extensions both ends
        s_sub = np.unique(
            np.concatenate(
                [np.linspace(-80.0, oxide_lo, 100), s_shared, np.linspace(s_absorb, s_absorb + 20.0, 50)]
            )
        )
        bnd = np.array([b_sub(s) for s in s_sub])
        substrate["poly"] = np.vstack([bnd, bnd[-1] - 200.0 * n, bnd[0] - 200.0 * n])

    y_src = cfg.source.y_source_um
    x_src = cfg.source.x_source_um
    w0 = cfg.source.w0_um
    s_imp = g.s_impact(y_src)
    x_imp = r_of(s_imp)[0]
    k_ref = np.array([math.cos(2 * a), -math.sin(2 * a)])  # reflected TIR direction
    # sample reflected ray until it is far outside
    refl_pts = [r_of(s_imp)]
    for t in np.linspace(0, 20, 9):
        refl_pts.append(r_of(s_imp) + t * k_ref)
    refl_pts = np.array(refl_pts)

    s_ext = g.stack_start_um + g.stack_length_um + min(
        cfg.monitor.output_offset_um, g.guide_ext_down_um
    )
    r_ext = r_of(s_ext)
    ext_pts = np.array(
        [r_ext + cfg.monitor.aperture_um * m, r_ext, r_ext + m]
    )
    # domain sized from the port end (materials continue past it into the PML)
    feature_pts = np.vstack([oxide_poly_dom, guide_poly_dom, refl_pts, ext_pts])
    src_span = 4.0 * w0
    # physical region must contain features, the beam, and its reflected ray
    x_lo = min(feature_pts[:, 0].min(), x_src - src_span, x_imp - 2 * w0)
    x_hi = max(feature_pts[:, 0].max(), x_imp + 4 * w0, x_src + src_span)
    y_lo = min(feature_pts[:, 1].min(), y_src - src_span)
    y_hi = max(feature_pts[:, 1].max(), y_src + src_span)
    margin = 3.0  # clearance between features and PML (in addition to beam span)
    pml = cfg.numerics.pml_um
    x_lo -= margin + pml
    x_hi += margin + pml
    y_lo -= margin + pml
    y_hi += margin + pml
    domain = {
        "x_lo": x_lo,
        "x_hi": x_hi,
        "y_lo": y_lo,
        "y_hi": y_hi,
        "sx": x_hi - x_lo,
        "sy": y_hi - y_lo,
        "pml_um": pml,
        "cell_center": [0.5 * (x_lo + x_hi), 0.5 * (y_lo + y_hi)],
        "cell_size": [x_hi - x_lo, y_hi - y_lo],
    }

    focus = r_of(s_imp) if cfg.source.focus == "impact" else np.array([x_src, y_src])
    points = {
        "x_source": x_src,
        "y_source": y_src,
        "s_impact_um": s_imp,
        "impact_xy": r_of(s_imp).tolist(),
        "x_impact": x_imp,
        "focus_xy": focus.tolist(),
    }

    # optional common cell/grid override for controlled comparisons
    nc = cfg.numerics
    if nc.cell_sx_um and nc.cell_sy_um:
        cx = nc.cell_center_x_um if nc.cell_center_x_um is not None else 0.5 * (domain["x_lo"] + domain["x_hi"])
        cy = nc.cell_center_y_um if nc.cell_center_y_um is not None else 0.5 * (domain["y_lo"] + domain["y_hi"])
        domain["sx"], domain["sy"] = float(nc.cell_sx_um), float(nc.cell_sy_um)
        domain["cell_center"] = [float(cx), float(cy)]
        domain["x_lo"], domain["x_hi"] = cx - nc.cell_sx_um / 2, cx + nc.cell_sx_um / 2
        domain["y_lo"], domain["y_hi"] = cy - nc.cell_sy_um / 2, cy + nc.cell_sy_um / 2
        domain["cell_size"] = [float(nc.cell_sx_um), float(nc.cell_sy_um)]

    shift = -np.array(domain["cell_center"], dtype=float)
    return Plan(
        cfg=cfg,
        u=u,
        n=n,
        d=d,
        m=m,
        r_top=r_top,
        twg_um=twg,
        s_guide_lo=s_lo,
        s_guide_hi=s_hi,
        oxide_poly=oxide_poly,
        guide_poly=guide_poly,
        substrate=substrate,
        domain=domain,
        points=points,
        shift=shift,
    )


def build_geometry(cfg: CouplerConfig) -> tuple[list, Plan, IndexSet]:
    plan = make_plan(cfg)
    indices = IndexSet(
        n_si=cfg.materials.n_si,
        n_oxide=cfg.materials.n_oxide,
        cladding=cfg.materials.cladding,
    )
    si = indices.medium("si")
    oxide = indices.medium("oxide")
    clad = indices.medium("clad")

    sub_center = plan.substrate["center"]
    if plan.substrate.get("poly") is not None:
        substrate_obj = mp.Prism(
            vertices=[plan.meep_xy(x, y) for x, y in plan.substrate["poly"]],
            height=mp.inf,
            material=si,
        )
    else:
        substrate_obj = mp.Block(
            center=plan.meep_xy(sub_center[0], sub_center[1]),
            size=mp.Vector3(*plan.substrate["size"]),
            e1=mp.Vector3(*plan.substrate["e1"]),
            e2=mp.Vector3(*plan.substrate["e2"]),
            material=si,
        )
    geom = [
        substrate_obj,
        mp.Prism(
            vertices=[plan.meep_xy(x, y) for x, y in plan.oxide_poly],
            height=mp.inf,
            material=oxide,
        ),
        mp.Prism(
            vertices=[plan.meep_xy(x, y) for x, y in plan.guide_poly],
            height=mp.inf,
            material=si,
        ),
    ]
    _ = clad  # background is cladding
    return geom, plan, indices


def plot_geometry(cfg: CouplerConfig, path, sampled: dict | None = None) -> None:
    """Material-map / axes review artifact (matplotlib, no solver needed)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    plan = make_plan(cfg)
    d = plan.domain
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.add_patch(
        Polygon(plan.oxide_poly, closed=True, facecolor="#8ecae6", edgecolor="k", lw=1.2, label="oxide gap")
    )
    ax.add_patch(
        Polygon(plan.guide_poly, closed=True, facecolor="#ffb703", edgecolor="k", lw=1.2, label="Si guide")
    )
    # interface line across domain
    a = cfg.geometry.alpha_rad
    r_top = plan.r_top
    ss = np.linspace(-40, 40, 2)
    ax.plot(
        [r_top[0] + s * math.cos(a) for s in ss],
        [r_top[1] - s * math.sin(a) for s in ss],
        color="crimson",
        lw=1.0,
        ls="--",
        label="Si/oxide sidewall",
    )
    src = plan.points
    ax.plot(src["x_source"], src["y_source"], "kv", ms=8, label="source")
    ax.plot(*src["impact_xy"], "ro", ms=6, label="beam impact")
    # beam axis + envelope
    xs = np.linspace(src["x_source"], src["impact_xy"][0], 40)
    ax.plot(xs, np.full_like(xs, src["y_source"]), color="k", lw=0.8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    ax.set_xlim(d["x_lo"], d["x_hi"])
    ax.set_ylim(d["y_lo"], d["y_hi"])
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um, above cavity base)")
    ax.set_title(f"geometry: {cfg.run.name}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
