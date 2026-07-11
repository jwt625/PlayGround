from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

from eo_fem.backends.gmsh_fields import compile_mesh_sequence
from eo_fem.backends.mesh_sequence import MeshOperationSpec, MeshSequenceSpec


def generate_3d_validation_examples(output_dir: str | Path) -> list[Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    return [
        _generate_sphere_in_box(root / "sphere_in_box"),
        _generate_swept_capacitor(root / "swept_capacitor"),
        _generate_coaxial_segment(root / "coaxial_segment"),
    ]


def _generate_sphere_in_box(output_dir: Path) -> Path:
    gmsh = importlib.import_module("gmsh")
    output_dir.mkdir(parents=True, exist_ok=True)
    gmsh.initialize(interruptible=False)
    try:
        gmsh.model.add("sphere_in_box")
        box = gmsh.model.occ.addBox(-1.0, -1.0, -1.0, 2.0, 2.0, 2.0)
        sphere = gmsh.model.occ.addSphere(0.0, 0.0, 0.0, 0.45)
        volumes, _ = gmsh.model.occ.fragment([(3, box)], [(3, sphere)])
        gmsh.model.occ.synchronize()
        volume_tags = [tag for dim, tag in volumes if dim == 3]
        inclusion = min(volume_tags, key=lambda tag: gmsh.model.occ.getMass(3, tag))
        matrix = next(tag for tag in volume_tags if tag != inclusion)
        inclusion_surfaces = _boundary_tags(gmsh, 3, inclusion)
        matrix_surfaces = _boundary_tags(gmsh, 3, matrix)
        outer_surfaces = sorted(set(matrix_surfaces) - set(inclusion_surfaces))
        groups = {
            "matrix_volume": _physical(gmsh, 3, [matrix], "matrix_volume"),
            "inclusion_volume": _physical(gmsh, 3, [inclusion], "inclusion_volume"),
            "inclusion_interface": _physical(gmsh, 2, inclusion_surfaces, "inclusion_interface"),
            "outer_boundary": _physical(gmsh, 2, outer_surfaces, "outer_boundary"),
        }
        sequence = MeshSequenceSpec(
            name="mesh1",
            dimension=3,
            sequence_type="user_controlled",
            operations=(
                MeshOperationSpec("global_size", "size", "entire_geometry", True, {"h_min": 0.08, "h_max": 0.35}),
                MeshOperationSpec("inclusion_size", "size", "inclusion_volume", True, {"h_max": 0.10}),
                MeshOperationSpec(
                    "interface_proximity",
                    "boundary_proximity",
                    "inclusion_interface",
                    True,
                    {"h_min": 0.06, "h_max": 0.35, "distance_min": 0.04, "distance_max": 0.45, "sampling": 80},
                ),
                MeshOperationSpec(
                    "free_tetrahedral",
                    "free_tetrahedral",
                    "all_domains",
                    True,
                    {"algorithm": "delaunay"},
                ),
            ),
        )
        selections = {
            "entire_geometry": {"dim": 3, "entity_tags": volume_tags},
            "all_domains": {"dim": 3, "entity_tags": volume_tags},
            "inclusion_volume": {"dim": 3, "entity_tags": [inclusion]},
            "inclusion_interface": {"dim": 2, "entity_tags": inclusion_surfaces},
        }
        groups["mesh_sequence"] = compile_mesh_sequence(gmsh, sequence, selections)
        gmsh.model.mesh.generate(3)
        return _write_artifacts(gmsh, output_dir, "sphere_in_box", groups)
    finally:
        gmsh.finalize()


def _generate_swept_capacitor(output_dir: Path) -> Path:
    gmsh = importlib.import_module("gmsh")
    output_dir.mkdir(parents=True, exist_ok=True)
    gmsh.initialize(interruptible=False)
    try:
        gmsh.model.add("swept_capacitor")
        source = gmsh.model.occ.addRectangle(-1.5, -0.75, 0.0, 3.0, 1.5)
        extrusion = gmsh.model.occ.extrude([(2, source)], 0.0, 0.0, 2.0, numElements=[12], recombine=True)
        gmsh.model.occ.synchronize()
        volume = next(tag for dim, tag in extrusion if dim == 3)
        surfaces = _boundary_tags(gmsh, 3, volume)
        top = max(surfaces, key=lambda tag: gmsh.model.occ.getCenterOfMass(2, tag)[2])
        bottom = min(surfaces, key=lambda tag: gmsh.model.occ.getCenterOfMass(2, tag)[2])
        sides = sorted(set(surfaces) - {top, bottom})
        groups = {
            "dielectric_volume": _physical(gmsh, 3, [volume], "dielectric_volume"),
            "signal_face": _physical(gmsh, 2, [top], "signal_face"),
            "ground_face": _physical(gmsh, 2, [bottom], "ground_face"),
            "side_boundaries": _physical(gmsh, 2, sides, "side_boundaries"),
            "mesh_sequence": {
                "name": "mesh1",
                "dimension": 3,
                "sequence_type": "user_controlled",
                "operations": [
                    {"name": "mapped_source", "type": "mapped", "selection": "ground_face"},
                    {"name": "sweep_distribution", "type": "distribution", "layers": 12},
                    {"name": "swept", "type": "swept", "selection": "dielectric_volume", "recombine": True},
                ],
            },
        }
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.18)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.28)
        gmsh.model.mesh.generate(3)
        return _write_artifacts(gmsh, output_dir, "swept_capacitor", groups)
    finally:
        gmsh.finalize()


def _generate_coaxial_segment(output_dir: Path) -> Path:
    gmsh = importlib.import_module("gmsh")
    output_dir.mkdir(parents=True, exist_ok=True)
    gmsh.initialize(interruptible=False)
    try:
        gmsh.model.add("coaxial_segment")
        outer = gmsh.model.occ.addCylinder(0.0, 0.0, -1.0, 0.0, 0.0, 2.0, 1.0)
        inner = gmsh.model.occ.addCylinder(0.0, 0.0, -1.0, 0.0, 0.0, 2.0, 0.30)
        volumes, _ = gmsh.model.occ.cut([(3, outer)], [(3, inner)], removeObject=True, removeTool=True)
        gmsh.model.occ.synchronize()
        volume = next(tag for dim, tag in volumes if dim == 3)
        surfaces = _boundary_tags(gmsh, 3, volume)
        cylinders = [
            tag
            for tag in surfaces
            if gmsh.model.getBoundingBox(2, tag)[5] - gmsh.model.getBoundingBox(2, tag)[2] > 1.9
        ]
        inner_surface = min(cylinders, key=lambda tag: gmsh.model.occ.getMass(2, tag))
        outer_surface = max(cylinders, key=lambda tag: gmsh.model.occ.getMass(2, tag))
        end_faces = sorted(set(surfaces) - set(cylinders))
        groups = {
            "dielectric_volume": _physical(gmsh, 3, [volume], "dielectric_volume"),
            "signal_boundary": _physical(gmsh, 2, [inner_surface], "signal_boundary"),
            "ground_boundary": _physical(gmsh, 2, [outer_surface], "ground_boundary"),
            "end_faces": _physical(gmsh, 2, end_faces, "end_faces"),
        }
        sequence = MeshSequenceSpec(
            name="mesh1",
            dimension=3,
            sequence_type="user_controlled",
            operations=(
                MeshOperationSpec("global_size", "size", "entire_geometry", True, {"h_min": 0.05, "h_max": 0.30}),
                MeshOperationSpec(
                    "signal_proximity",
                    "boundary_proximity",
                    "signal_boundary",
                    True,
                    {"h_min": 0.045, "h_max": 0.30, "distance_min": 0.03, "distance_max": 0.40, "sampling": 100},
                ),
                MeshOperationSpec(
                    "free_tetrahedral",
                    "free_tetrahedral",
                    "all_domains",
                    True,
                    {"algorithm": "delaunay"},
                ),
            ),
        )
        selections = {
            "entire_geometry": {"dim": 3, "entity_tags": [volume]},
            "all_domains": {"dim": 3, "entity_tags": [volume]},
            "signal_boundary": {"dim": 2, "entity_tags": [inner_surface]},
        }
        groups["mesh_sequence"] = compile_mesh_sequence(gmsh, sequence, selections)
        gmsh.model.mesh.generate(3)
        return _write_artifacts(gmsh, output_dir, "coaxial_segment", groups)
    finally:
        gmsh.finalize()


def _write_artifacts(gmsh: Any, output_dir: Path, name: str, groups: dict[str, Any]) -> Path:
    mesh_path = output_dir / f"{name}.msh"
    manifest_path = output_dir / f"{name}.tags.json"
    gmsh.write(str(mesh_path))
    groups["summary"] = _mesh_summary(gmsh)
    manifest_path.write_text(json.dumps(groups, indent=2, sort_keys=True), encoding="utf-8")
    return mesh_path


def _mesh_summary(gmsh: Any) -> dict[str, Any]:
    node_tags, _, _ = gmsh.model.mesh.getNodes()
    element_types, element_tags, _ = gmsh.model.mesh.getElements(3)
    type_counts: dict[str, int] = {}
    all_tags: list[int] = []
    for element_type, tags in zip(element_types, element_tags, strict=True):
        name = str(gmsh.model.mesh.getElementProperties(element_type)[0])
        type_counts[name] = int(len(tags))
        all_tags.extend(int(tag) for tag in tags)
    quality = gmsh.model.mesh.getElementQualities(all_tags, "minSICN") if all_tags else []
    return {
        "node_count": int(len(node_tags)),
        "volume_element_count": len(all_tags),
        "element_types": type_counts,
        "quality_min_sicn": float(min(quality)) if len(quality) else None,
        "quality_mean_sicn": float(sum(quality) / len(quality)) if len(quality) else None,
    }


def _boundary_tags(gmsh: Any, dim: int, tag: int) -> list[int]:
    return sorted(
        boundary_tag
        for boundary_dim, boundary_tag in gmsh.model.getBoundary([(dim, tag)])
        if boundary_dim == dim - 1
    )


def _physical(gmsh: Any, dim: int, entity_tags: list[int], name: str) -> dict[str, Any]:
    physical_tag = int(gmsh.model.addPhysicalGroup(dim, entity_tags))
    gmsh.model.setPhysicalName(dim, physical_tag, name)
    return {"dim": dim, "physical_tag": physical_tag, "entity_tags": entity_tags}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate advanced 3D Gmsh validation artifacts.")
    parser.add_argument("--out", default="artifacts/mesh_controls_3d")
    args = parser.parse_args()
    for path in generate_3d_validation_examples(args.out):
        print(path)


if __name__ == "__main__":
    main()
