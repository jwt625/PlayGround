from __future__ import annotations

import argparse
import json
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy.special import sph_harm_y


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
SOURCE_URL = "https://raw.githubusercontent.com/libigl/libigl-tutorial-data/master/cow.off"
TARGET_BBOX = np.array([1.0440, 0.6397, 0.3403], dtype=float)
MAX_DEGREE = 24


@dataclass(frozen=True)
class HarmonicTerm:
    l: int
    m: int


def ensure_directories() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)


def download_mesh() -> Path:
    off_path = DATA_DIR / "cow.off"
    if not off_path.exists():
        print(f"Downloading canonical libigl tutorial mesh to {off_path}...")
        urllib.request.urlretrieve(SOURCE_URL, off_path)
    return off_path


def load_and_prepare_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(path, file_type="off", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    mesh = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy(), process=True)

    vertices = mesh.vertices.copy()
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    extents = bbox_max - bbox_min

    order = np.argsort(extents)[::-1]
    vertices = vertices[:, order]

    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    extents = bbox_max - bbox_min
    scale = TARGET_BBOX[0] / extents[0]
    vertices *= scale

    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    extents = bbox_max - bbox_min
    axis_scales = TARGET_BBOX / extents
    if np.max(np.abs(axis_scales - 1.0)) > 0.03:
        vertices *= axis_scales

    vertices -= 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    mesh = trimesh.Trimesh(vertices=vertices, faces=mesh.faces.copy(), process=True)
    (DATA_DIR / "cow.obj").write_text(trimesh.exchange.obj.export_obj(mesh))
    return mesh


def spherical_from_cartesian(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    r = np.linalg.norm(points, axis=1)
    r_safe = np.maximum(r, 1e-12)
    theta = np.arccos(np.clip(y / r_safe, -1.0, 1.0))
    phi = np.arctan2(z, x)
    return r, theta, phi


def cartesian_from_spherical(r: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    sin_theta = np.sin(theta)
    x = r * sin_theta * np.cos(phi)
    y = r * np.cos(theta)
    z = r * sin_theta * np.sin(phi)
    return np.column_stack([x, y, z])


def fibonacci_sphere(samples: int) -> np.ndarray:
    i = np.arange(samples, dtype=float)
    phi = math.pi * (3.0 - math.sqrt(5.0))
    y = 1.0 - (2.0 * i + 1.0) / samples
    radius = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = phi * i
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius
    return np.column_stack([x, y, z])


def count_hits_per_ray(mesh: trimesh.Trimesh, directions: np.ndarray) -> np.ndarray:
    origins = np.zeros_like(directions)
    locations, ray_ids, _ = mesh.ray.intersects_location(
        ray_origins=origins, ray_directions=directions, multiple_hits=True
    )
    if len(ray_ids) == 0:
        return np.zeros(len(directions), dtype=int)
    distances = np.linalg.norm(locations, axis=1)
    positive = distances > 1e-8
    counts = np.bincount(ray_ids[positive], minlength=len(directions))
    return counts


def find_star_shell(mesh: trimesh.Trimesh) -> tuple[np.ndarray, float, dict[str, float]]:
    directions = fibonacci_sphere(3000)
    normals = mesh.vertex_normals
    bbox_diag = np.linalg.norm(mesh.bounding_box.extents)

    baseline_counts = count_hits_per_ray(mesh, directions)
    best_vertices = mesh.vertices.copy()
    best_offset = 0.0
    best_rate = float(np.mean(baseline_counts == 1))
    best_multi = int(np.max(baseline_counts)) if len(baseline_counts) else 0

    for fraction in np.linspace(0.0, 0.6, 31):
        offset = fraction * bbox_diag
        candidate_vertices = mesh.vertices + offset * normals
        candidate = trimesh.Trimesh(vertices=candidate_vertices, faces=mesh.faces, process=False)
        counts = count_hits_per_ray(candidate, directions)
        one_hit_rate = float(np.mean(counts == 1))
        max_hits = int(np.max(counts)) if len(counts) else 0
        if one_hit_rate > best_rate or (math.isclose(one_hit_rate, best_rate) and max_hits < best_multi):
            best_vertices = candidate_vertices
            best_offset = offset
            best_rate = one_hit_rate
            best_multi = max_hits
        if one_hit_rate > 0.995 and max_hits <= 1:
            break

    diagnostics = {
        "star_shell_offset": best_offset,
        "single_hit_fraction": best_rate,
        "max_hits_along_any_ray": float(best_multi),
    }
    return best_vertices, best_offset, diagnostics


def vertex_areas(mesh: trimesh.Trimesh) -> np.ndarray:
    areas = np.zeros(len(mesh.vertices), dtype=float)
    face_areas = mesh.area_faces
    for corner in range(3):
        np.add.at(areas, mesh.faces[:, corner], face_areas / 3.0)
    return areas


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def harmonic_terms(max_degree: int) -> list[HarmonicTerm]:
    return [HarmonicTerm(l, m) for l in range(max_degree + 1) for m in range(-l, l + 1)]


def build_design_matrix(theta: np.ndarray, phi: np.ndarray, max_degree: int) -> tuple[np.ndarray, list[HarmonicTerm]]:
    terms = harmonic_terms(max_degree)
    basis = np.empty((len(theta), len(terms)), dtype=complex)
    for idx, term in enumerate(terms):
        basis[:, idx] = sph_harm_y(term.l, term.m, theta, phi)
    return basis, terms


def weighted_complex_fit(
    basis: np.ndarray, values: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    w = np.sqrt(np.maximum(weights, 1e-15))
    lhs = basis * w[:, None]
    rhs = values * w
    coeffs, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
    return coeffs


def evaluate_harmonics(
    theta: np.ndarray,
    phi: np.ndarray,
    coeffs: np.ndarray,
    terms: list[HarmonicTerm],
    max_degree: int,
) -> np.ndarray:
    values = np.zeros(len(theta), dtype=complex)
    for coeff, term in zip(coeffs, terms):
        if term.l > max_degree:
            continue
        values += coeff * sph_harm_y(term.l, term.m, theta, phi)
    return values.real


def orient_faces_outward(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if mesh.volume < 0:
        return faces[:, [0, 2, 1]]
    return faces


def build_uv_sphere(theta_count: int = 96, phi_count: int = 192) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, np.pi, theta_count)
    phi = np.linspace(-np.pi, np.pi, phi_count, endpoint=False)

    vertices = [[0.0, 0.0]]
    for theta_value in theta[1:-1]:
        for phi_value in phi:
            vertices.append([theta_value, phi_value])
    vertices.append([np.pi, 0.0])

    north_index = 0
    south_index = len(vertices) - 1
    ring_count = theta_count - 2

    def ring_vertex(ring: int, column: int) -> int:
        return 1 + ring * phi_count + (column % phi_count)

    faces: list[list[int]] = []

    if ring_count > 0:
        for j in range(phi_count):
            j_next = (j + 1) % phi_count
            faces.append([north_index, ring_vertex(0, j), ring_vertex(0, j_next)])

        for ring in range(ring_count - 1):
            for j in range(phi_count):
                j_next = (j + 1) % phi_count
                a = ring_vertex(ring, j)
                b = ring_vertex(ring, j_next)
                c = ring_vertex(ring + 1, j)
                d = ring_vertex(ring + 1, j_next)
                faces.append([a, d, b])
                faces.append([a, c, d])

        last_ring = ring_count - 1
        for j in range(phi_count):
            j_next = (j + 1) % phi_count
            faces.append([ring_vertex(last_ring, j), south_index, ring_vertex(last_ring, j_next)])

    angle_vertices = np.asarray(vertices, dtype=float)
    unit_vertices = cartesian_from_spherical(
        np.ones(len(angle_vertices)),
        angle_vertices[:, 0],
        angle_vertices[:, 1],
    )
    face_array = orient_faces_outward(unit_vertices, np.asarray(faces, dtype=int))
    return angle_vertices, face_array


def plot_reconstructions(original: trimesh.Trimesh, reconstructed: dict[int, trimesh.Trimesh]) -> None:
    entries = [("Original", original)] + [(f"l <= {order}", reconstructed[order]) for order in sorted(reconstructed)]
    count = len(entries)
    cols = min(4, count)
    rows = math.ceil(count / cols)
    fig = plt.figure(figsize=(4.8 * cols, 3.2 * rows))
    for index, (title, mesh) in enumerate(entries, start=1):
        ax = fig.add_subplot(rows, cols, index, projection="3d")
        tris = mesh.faces
        verts = mesh.vertices
        ax.plot_trisurf(
            verts[:, 0],
            verts[:, 2],
            verts[:, 1],
            triangles=tris,
            linewidth=0.05,
            antialiased=True,
            color="#d7ddd1",
            edgecolor="#40513b",
            shade=True,
            alpha=1.0,
        )
        ax.set_title(title)
        ax.set_box_aspect(tuple(original.bounding_box.extents))
        ax.view_init(elev=18, azim=-62)
        ax.set_axis_off()
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "reconstruction_grid.png", dpi=220)
    plt.close(fig)


def save_coefficients(
    coeffs_r: np.ndarray,
    coeffs_theta: np.ndarray,
    coeffs_phi: np.ndarray,
    terms: list[HarmonicTerm],
) -> None:
    rows = []
    for index, term in enumerate(terms):
        rows.append(
            {
                "l": term.l,
                "m": term.m,
                "f_r_real": float(coeffs_r[index].real),
                "f_r_imag": float(coeffs_r[index].imag),
                "f_dtheta_real": float(coeffs_theta[index].real),
                "f_dtheta_imag": float(coeffs_theta[index].imag),
                "f_dphi_real": float(coeffs_phi[index].real),
                "f_dphi_imag": float(coeffs_phi[index].imag),
            }
        )
    (OUTPUT_DIR / "coefficients.json").write_text(json.dumps(rows, indent=2))


def compute_surface_rmse(reference: trimesh.Trimesh, candidate: trimesh.Trimesh) -> float:
    _, distances, _ = trimesh.proximity.closest_point(candidate, reference.vertices)
    return float(np.sqrt(np.mean(distances**2)))


def fit_order(
    theta_star: np.ndarray,
    phi_star: np.ndarray,
    r_values: np.ndarray,
    dtheta: np.ndarray,
    dphi: np.ndarray,
    areas: np.ndarray,
    order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[HarmonicTerm]]:
    basis, terms = build_design_matrix(theta_star, phi_star, max_degree=order)
    coeffs_r = weighted_complex_fit(basis, r_values.astype(complex), areas)
    coeffs_theta = weighted_complex_fit(basis, dtheta.astype(complex), areas)
    coeffs_phi = weighted_complex_fit(basis, dphi.astype(complex), areas)
    coeffs_theta[0] = 0.0
    coeffs_phi[0] = 0.0
    return coeffs_r, coeffs_theta, coeffs_phi, terms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct the benchmark cow with spherical harmonics.")
    parser.add_argument("--max-degree", type=int, default=MAX_DEGREE, help="Maximum harmonic order to export.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing OBJ outputs when present instead of overwriting them.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip reconstruction_grid.png generation.",
    )
    return parser.parse_args()


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    return mesh


def load_existing_metrics() -> dict[str, object] | None:
    metrics_path = OUTPUT_DIR / "metrics.json"
    if not metrics_path.exists():
        return None
    return json.loads(metrics_path.read_text())


def write_metrics(metrics: dict[str, object]) -> None:
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))


def main() -> None:
    args = parse_args()
    ensure_directories()
    mesh_path = download_mesh()
    original = load_and_prepare_mesh(mesh_path)

    star_vertices, _, diagnostics = find_star_shell(original)
    r_values, theta_values, phi_values = spherical_from_cartesian(original.vertices)
    _, theta_star, phi_star = spherical_from_cartesian(star_vertices)

    dtheta = theta_values - theta_star
    dphi = wrap_angle(phi_values - phi_star)

    areas = vertex_areas(original)
    sample_angles, faces = build_uv_sphere(theta_count=100, phi_count=200)
    theta_sample = sample_angles[:, 0]
    phi_sample = sample_angles[:, 1]

    reconstructed_meshes: dict[int, trimesh.Trimesh] = {}
    existing_metrics = load_existing_metrics() if args.skip_existing else None
    metrics: dict[str, object] = {
        "source_url": SOURCE_URL,
        "target_bbox": TARGET_BBOX.tolist(),
        "prepared_bbox": original.bounding_box.extents.tolist(),
        "star_shell_diagnostics": diagnostics,
        "orders": {},
    }

    for order in range(args.max_degree + 1):
        output_path = OUTPUT_DIR / f"cow_reconstruction_l{order}.obj"
        existing_order = None if existing_metrics is None else existing_metrics.get("orders", {}).get(str(order))

        if args.skip_existing and output_path.exists():
            recon = load_mesh(output_path)
            reconstructed_meshes[order] = recon
            rmse = (
                float(existing_order["surface_rmse"])
                if existing_order is not None and "surface_rmse" in existing_order
                else compute_surface_rmse(original, recon)
            )
            metrics["orders"][str(order)] = {
                "surface_rmse": rmse,
                "vertex_count": int(len(recon.vertices)),
                "face_count": int(len(recon.faces)),
            }
            write_metrics(metrics)
            print(f"l <= {order}: reused existing mesh")
            continue

        coeffs_r, coeffs_theta, coeffs_phi, terms = fit_order(
            theta_star=theta_star,
            phi_star=phi_star,
            r_values=r_values,
            dtheta=dtheta,
            dphi=dphi,
            areas=areas,
            order=order,
        )

        radius = evaluate_harmonics(theta_sample, phi_sample, coeffs_r, terms, order)
        delta_theta = evaluate_harmonics(theta_sample, phi_sample, coeffs_theta, terms, order)
        delta_phi = evaluate_harmonics(theta_sample, phi_sample, coeffs_phi, terms, order)

        radius = np.maximum(radius, 1e-4)
        theta_recon = np.clip(theta_sample + delta_theta, 1e-4, np.pi - 1e-4)
        phi_recon = wrap_angle(phi_sample + delta_phi)
        vertices = cartesian_from_spherical(radius, theta_recon, phi_recon)

        faces_oriented = orient_faces_outward(vertices, faces)
        recon = trimesh.Trimesh(vertices=vertices, faces=faces_oriented, process=True)
        reconstructed_meshes[order] = recon
        recon.export(output_path)
        rmse = compute_surface_rmse(original, recon)
        metrics["orders"][str(order)] = {
            "surface_rmse": rmse,
            "vertex_count": int(len(recon.vertices)),
            "face_count": int(len(recon.faces)),
        }
        if order == args.max_degree:
            save_coefficients(coeffs_r, coeffs_theta, coeffs_phi, terms)
        write_metrics(metrics)
        print(f"l <= {order}: surface RMSE = {rmse:.6f}")

    if not args.no_plot:
        plot_reconstructions(original, reconstructed_meshes)

    print("Prepared bbox:", np.round(original.bounding_box.extents, 6).tolist())
    print("Star-shell diagnostics:", json.dumps(diagnostics, indent=2))
    print(f"Wrote outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
