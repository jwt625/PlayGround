from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import sqrt
from typing import Any

from .constants import EPS0, capacitance_units
from .geometry import Domain, Electrode, parse_domain, parse_electrodes
from .materials import (
    Material,
    epsilon_tensor_at,
    parse_materials,
    uses_spatial_permittivity,
    uses_tensor_permittivity,
)


@dataclass
class Mesh:
    domain: Domain
    nx: int
    ny: int
    nodes: list[tuple[float, float]]
    triangles: list[tuple[int, int, int]]


@dataclass
class SolveResult:
    capacitance_energy: float
    capacitance_charge: float
    energy_per_length: float
    potential: list[float]
    mesh: Mesh
    iterations: int
    residual: float
    units: dict[str, float]
    reference: dict[str, float | str] | None = None
    permittivity_model: str = "homogeneous scalar eps_r"


def solve_config(config: dict[str, Any]) -> SolveResult:
    sim = config.get("Simulation", {})
    nx = int(sim.get("mesh_nx", 81))
    ny = int(sim.get("mesh_ny", 61))
    domain = parse_domain(config)
    eps_r = float(config.get("Materials", {}).get("background", {}).get("eps_r", 1.0))
    materials = parse_materials(config)
    electrodes = parse_electrodes(config)
    mesh = make_structured_tri_mesh(domain, nx, ny)
    k_rows = assemble_stiffness(mesh, materials)
    dirichlet, labels = dirichlet_nodes(mesh, electrodes)
    phi, iterations, residual = solve_dirichlet(k_rows, dirichlet)
    energy = field_energy(k_rows, phi)
    signal_name = str(sim.get("signal_electrode", "signal"))
    signal_nodes = [i for i, label in labels.items() if label == signal_name]
    q_signal = electrode_charge(k_rows, phi, signal_nodes)
    v_signal = _electrode_potential(electrodes, signal_name)
    c_energy = 2.0 * energy / (v_signal * v_signal)
    c_charge = abs(q_signal / v_signal)
    reference = reference_capacitance(config, eps_r)
    return SolveResult(
        capacitance_energy=c_energy,
        capacitance_charge=c_charge,
        energy_per_length=energy,
        potential=phi,
        mesh=mesh,
        iterations=iterations,
        residual=residual,
        units=capacitance_units(c_energy),
        reference=reference,
        permittivity_model=permittivity_model_description(materials, eps_r),
    )


def make_structured_tri_mesh(domain: Domain, nx: int, ny: int) -> Mesh:
    if nx < 2 or ny < 2:
        raise ValueError("mesh_nx and mesh_ny must be at least 2")
    nodes = []
    for j in range(ny):
        y = domain.y_min + (domain.y_max - domain.y_min) * j / (ny - 1)
        for i in range(nx):
            x = domain.x_min + (domain.x_max - domain.x_min) * i / (nx - 1)
            nodes.append((x, y))
    triangles = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n00 = j * nx + i
            n10 = n00 + 1
            n01 = n00 + nx
            n11 = n01 + 1
            triangles.append((n00, n10, n11))
            triangles.append((n00, n11, n01))
    return Mesh(domain=domain, nx=nx, ny=ny, nodes=nodes, triangles=triangles)


def assemble_stiffness(mesh: Mesh, materials_or_eps_r: list[Material] | float) -> list[dict[int, float]]:
    rows: list[dict[int, float]] = [dict() for _ in mesh.nodes]
    for tri in mesh.triangles:
        pts = [mesh.nodes[i] for i in tri]
        area2 = (
            (pts[1][0] - pts[0][0]) * (pts[2][1] - pts[0][1])
            - (pts[2][0] - pts[0][0]) * (pts[1][1] - pts[0][1])
        )
        area = abs(area2) / 2.0
        if area == 0.0:
            continue
        b = [pts[1][1] - pts[2][1], pts[2][1] - pts[0][1], pts[0][1] - pts[1][1]]
        c = [pts[2][0] - pts[1][0], pts[0][0] - pts[2][0], pts[1][0] - pts[0][0]]
        if isinstance(materials_or_eps_r, list):
            eps_xx, eps_yy, eps_xy = epsilon_tensor_at(
                materials_or_eps_r,
                (pts[0][0] + pts[1][0] + pts[2][0]) / 3.0,
                (pts[0][1] + pts[1][1] + pts[2][1]) / 3.0,
            )
        else:
            eps_r = float(materials_or_eps_r)
            eps_xx, eps_yy, eps_xy = eps_r, eps_r, 0.0
        for a in range(3):
            ia = tri[a]
            for d in range(3):
                value = (
                    EPS0
                    * (
                        eps_xx * b[a] * b[d]
                        + eps_xy * b[a] * c[d]
                        + eps_xy * c[a] * b[d]
                        + eps_yy * c[a] * c[d]
                    )
                    / (4.0 * area)
                )
                rows[ia][tri[d]] = rows[ia].get(tri[d], 0.0) + value
    return rows


def permittivity_model_description(materials: list[Material], background_eps_r: float) -> str:
    spatial = uses_spatial_permittivity(materials)
    tensor = uses_tensor_permittivity(materials)
    if tensor and spatial:
        return "spatial anisotropic eps_r tensor (triangle centroid)"
    if tensor:
        return "homogeneous anisotropic eps_r tensor"
    if spatial:
        return "spatial scalar eps_r (triangle centroid)"
    return f"homogeneous scalar eps_r={background_eps_r:g}"


def dirichlet_nodes(
    mesh: Mesh, electrodes: list[Electrode]
) -> tuple[dict[int, float], dict[int, str]]:
    values: dict[int, float] = {}
    labels: dict[int, str] = {}
    for i, (x, y) in enumerate(mesh.nodes):
        for electrode in electrodes:
            if electrode.contains(x, y):
                values[i] = electrode.potential
                labels[i] = electrode.name
                break
    if len(set(labels.values())) < 2:
        raise ValueError("at least two electrodes must intersect mesh nodes")
    return values, labels


def solve_dirichlet(
    rows: list[dict[int, float]], dirichlet: dict[int, float], tol: float = 1e-6, maxiter: int = 20_000
) -> tuple[list[float], int, float]:
    n = len(rows)
    free = [i for i in range(n) if i not in dirichlet]
    free_set = set(free)
    phi = [0.0] * n
    for i, value in dirichlet.items():
        phi[i] = value
    rhs = []
    for i in free:
        rhs.append(-sum(v * phi[j] for j, v in rows[i].items() if j in dirichlet))

    def matvec(vec: list[float]) -> list[float]:
        full = [0.0] * n
        for idx, node in enumerate(free):
            full[node] = vec[idx]
        out = []
        for node in free:
            out.append(sum(v * full[j] for j, v in rows[node].items() if j in free_set))
        return out

    inv_diag = [1.0 / rows[node].get(node, 1.0) for node in free]
    x, iterations, residual = conjugate_gradient(matvec, rhs, inv_diag, tol=tol, maxiter=maxiter)
    for idx, node in enumerate(free):
        phi[node] = x[idx]
    return phi, iterations, residual


def conjugate_gradient(
    matvec: Callable[[list[float]], list[float]],
    rhs: list[float],
    inv_diag: list[float] | None = None,
    tol: float = 1e-6,
    maxiter: int = 20_000,
) -> tuple[list[float], int, float]:
    x = [0.0] * len(rhs)
    r = rhs[:]
    rsold = dot(r, r)
    rhs_norm = sqrt(rsold) or 1.0
    if rhs_norm == 0.0:
        return x, 0, 0.0
    z = apply_jacobi(inv_diag, r)
    p = z[:]
    rzold = dot(r, z)
    iteration = 0
    for iteration in range(1, maxiter + 1):
        ap = matvec(p)
        denom = dot(p, ap)
        if denom == 0.0:
            break
        alpha = rzold / denom
        x = [xi + alpha * pi for xi, pi in zip(x, p, strict=True)]
        r = [ri - alpha * api for ri, api in zip(r, ap, strict=True)]
        rsnew = dot(r, r)
        residual = sqrt(rsnew) / rhs_norm
        if residual < tol:
            return x, iteration, residual
        z = apply_jacobi(inv_diag, r)
        rznew = dot(r, z)
        beta = rznew / rzold
        p = [zi + beta * pi for zi, pi in zip(z, p, strict=True)]
        rsold = rsnew
        rzold = rznew
    return x, iteration, sqrt(rsold) / rhs_norm


def apply_jacobi(inv_diag: list[float] | None, values: list[float]) -> list[float]:
    if inv_diag is None:
        return values[:]
    return [diagonal * value for diagonal, value in zip(inv_diag, values, strict=True)]


def field_energy(rows: list[dict[int, float]], phi: list[float]) -> float:
    kphi = [sum(v * phi[j] for j, v in row.items()) for row in rows]
    return 0.5 * dot(phi, kphi)


def electrode_charge(rows: list[dict[int, float]], phi: list[float], nodes: list[int]) -> float:
    return sum(sum(v * phi[j] for j, v in rows[i].items()) for i in nodes)


def reference_capacitance(config: dict[str, Any], eps_r: float) -> dict[str, float | str] | None:
    from .analytic import parallel_plate_capacitance, two_cylinder_capacitance

    outputs = config.get("Outputs", {})
    reference = outputs.get("reference")
    if reference == "parallel_plate":
        c = parallel_plate_capacitance(
            eps_r=eps_r,
            width=float(outputs["plate_width"]),
            gap=float(outputs["plate_gap"]),
        )
    elif reference == "two_cylinders":
        c = two_cylinder_capacitance(
            eps_r=eps_r,
            radius=float(outputs["radius"]),
            center_distance=float(outputs["center_distance"]),
        )
    else:
        return None
    return {"name": str(reference), "capacitance": c, **capacitance_units(c)}


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _electrode_potential(electrodes: list[Electrode], name: str) -> float:
    for electrode in electrodes:
        if electrode.name == name:
            if electrode.potential == 0.0:
                raise ValueError("signal electrode potential cannot be zero")
            return electrode.potential
    raise ValueError(f"signal electrode {name!r} not found")
