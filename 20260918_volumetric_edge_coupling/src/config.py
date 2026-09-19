"""Typed configuration schema with units and provenance labels.

Every physical field is labelled by provenance:

- ``paper``   : explicitly reported by the paper (PAPER_REVIEW.md ledger).
- ``assumed`` : missing input given a documented provisional value.
- ``proposed``: reproduction/acceptance choice made by this project.
- ``fitted``  : arrived at by simulation-based search; must log the baseline.

The configuration hash includes every physical and numerical field, so changing
any of them changes the run identity (TODO T01).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# --------------------------------------------------------------------------
# Provenance helper
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Tagged:
    value: Any
    provenance: str

    def __post_init__(self):
        if self.provenance not in {"paper", "assumed", "proposed", "fitted"}:
            raise ValueError(f"bad provenance {self.provenance!r}")


# --------------------------------------------------------------------------
# Section configs
# --------------------------------------------------------------------------
@dataclass
class MaterialsConfig:
    n_si: float = 3.48
    n_oxide: float = 1.444
    cladding: str = "air"  # "air" | "oxide"
    model: str = "constant"  # "constant" | "dispersive"
    provenance: str = "assumed"


@dataclass
class GeometryConfig:
    # paper-reported angles / thickness
    alpha_deg: float = 54.74          # {111} KOH sidewall (paper)
    theta_wg_deg: float = 53.50       # receiving-guide inclination (paper)
    twg_nm: float = 262.0             # physical (normal) guide thickness (paper)
    # assumed absolute geometry
    depth_um: float = 15.0            # stack-top height above cavity base
    x_top_um: float = 0.0             # x of the stack-top endpoint
    g_ref_nm: float = 50.0            # oxide gap at the stack top (s = stack_start)
    s_ref_um: float | None = None     # sidewall coordinate of g_ref (defaults to stack_start)
    stack_start_um: float = 0.0       # sidewall coordinate of the guide top
    stack_length_um: float = 20.0     # oxide+guide extent along the sidewall
    guide_ext_down_um: float = 12.0   # straight inclined guide past the stack
    guide_ext_up_um: float = 0.0      # extension above the stack
    guide_absorb_um: float = 12.0     # continuation into the absorbing PML
    # isolated-output-port-v1 (named numerical continuation; R6 handoff s4)
    isolated_port: bool = False
    port_start_s_um: float = 33.0
    port_transition_um: float = 8.0
    oxide_clearance_um: float = 3.0
    provenance: str = "assumed"

    # derived
    @property
    def alpha_rad(self) -> float:
        import math

        return math.radians(self.alpha_deg)

    @property
    def theta_rad(self) -> float:
        import math

        return math.radians(self.theta_wg_deg)

    @property
    def gap_slope(self) -> float:
        """dq/ds of the substrate-facing guide face (dimensionless).

        Sign is set by the guide/sidewall angle difference: a guide shallower
        than the sidewall (theta_wg < alpha) opens the gap downward; a steeper
        guide (theta_wg > alpha) closes it.  This keeps the polygon and the
        guide direction self-consistent.
        """
        import math

        return math.tan(math.radians(self.alpha_deg - self.theta_wg_deg))

    def s_impact(self, y_source_um: float) -> float:
        """Sidewall coordinate where the horizontal beam axis meets q=0."""
        import math

        return (self.depth_um - y_source_um) / math.sin(self.alpha_rad)

    @property
    def s_ref(self) -> float:
        return self.stack_start_um if self.s_ref_um is None else self.s_ref_um

    def gap_at(self, s_um: float) -> float:
        """Gap in um at sidewall coordinate ``s`` (linear guide)."""
        return (self.g_ref_nm / 1000.0) + (s_um - self.s_ref) * self.gap_slope


@dataclass
class SourceConfig:
    w0_um: float = 5.0                # 1/e amplitude radius (paper)
    y_source_um: float = 11.0         # nominal axis height above cavity base (paper)
    x_source_um: float = -12.0        # launch plane (assumed/proposed)
    branch: str = "ez"                # "ez" (s, out-of-plane E) | "hz" (p)
    focus: str = "impact"             # "impact" | "plane"
    tilt_deg: float = 0.0             # source rotation (paper nominal 0)
    wavelength_nm: float = 1550.0
    fwidth: float = 0.05              # GaussianSource frequency half-width
    provenance: str = "assumed"


@dataclass
class MonitorConfig:
    # monitor offset along guide from the down end of the stack (positive into ext)
    output_offset_um: float = 8.0
    aperture_um: float = 12.0
    n_modes: int = 3
    provenance: str = "proposed"


@dataclass
class NumericsConfig:
    resolution: float = 50.0          # pixels per um
    pml_um: float = 1.5
    until_after_sources: float = 200.0
    decay: float = 1e-7
    stop_on_decay: bool = False
    cour: float = 0.5
    provenance: str = "proposed"


@dataclass
class RunConfig:
    name: str = "nominal"
    root: str = "results"
    provenance: str = "proposed"


@dataclass
class CouplerConfig:
    materials: MaterialsConfig = field(default_factory=MaterialsConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    numerics: NumericsConfig = field(default_factory=NumericsConfig)
    run: RunConfig = field(default_factory=RunConfig)

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()[:16]

    def save(self, path: Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=True))

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict) -> "CouplerConfig":
        def build(dc, payload):
            if payload is None:
                return dc()
            fields = {f.name for f in dataclasses.fields(dc)}
            unknown = set(payload) - fields
            if unknown:
                raise ValueError(f"unknown config keys for {dc.__name__}: {sorted(unknown)}")
            return dc(**payload)

        return cls(
            materials=build(MaterialsConfig, data.get("materials")),
            geometry=build(GeometryConfig, data.get("geometry")),
            source=build(SourceConfig, data.get("source")),
            monitor=build(MonitorConfig, data.get("monitor")),
            numerics=build(NumericsConfig, data.get("numerics")),
            run=build(RunConfig, data.get("run")),
        )

    @classmethod
    def load(cls, path: str | Path) -> "CouplerConfig":
        payload = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_dict(payload)

    # ------------------------------------------------------------------
    def validate(self) -> list[str]:
        """Return a list of fatal preflight errors (empty means valid)."""
        errors: list[str] = []
        g = self.geometry
        if not (0.0 < g.alpha_deg < 90.0):
            errors.append("alpha_deg must be in (0,90)")
        if not (0.0 < g.theta_wg_deg < 90.0):
            errors.append("theta_wg_deg must be in (0,90)")
        if g.twg_nm <= 0:
            errors.append("twg_nm must be positive")
        if g.depth_um <= self.source.y_source_um:
            errors.append("wafer top must lie above the source axis")
        s_imp = g.s_impact(self.source.y_source_um)
        s_lo = g.stack_start_um - g.guide_ext_up_um
        s_hi = g.stack_start_um + g.stack_length_um + g.guide_ext_down_um
        if not (g.stack_start_um <= s_imp <= g.stack_start_um + g.stack_length_um):
            errors.append(
                f"beam impacts s={s_imp:.2f} um outside stack "
                f"[{g.stack_start_um},{g.stack_start_um + g.stack_length_um}]"
            )
        for s in (s_lo, s_hi):
            if g.gap_at(s) <= 0.0:
                errors.append(f"non-positive gap {g.gap_at(s)*1000:.1f} nm at s={s:.2f} um")
        if self.numerics.resolution <= 0:
            errors.append("resolution must be positive")
        if self.source.w0_um <= 0:
            errors.append("w0_um must be positive")
        return errors


def load_config(path: str | Path) -> CouplerConfig:
    cfg = CouplerConfig.load(path)
    errors = cfg.validate()
    if errors:
        raise ValueError("; ".join(errors))
    return cfg
