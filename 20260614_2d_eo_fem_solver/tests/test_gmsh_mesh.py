from __future__ import annotations

import json
import math

import pytest

from eo_fem.backends.gmsh_mesh import generate_legacy_gmsh_mesh
from eo_fem.config import load_config


def test_parallel_plate_gmsh_mesh_has_domain_and_electrode_tags(tmp_path) -> None:
    pytest.importorskip("gmsh")

    artifact = generate_legacy_gmsh_mesh(
        load_config("examples/parallel_plate.yaml"),
        tmp_path,
        model_name="parallel_plate",
    )

    assert artifact.mesh_path.exists()
    assert artifact.tag_map_path.exists()
    assert artifact.node_count > 0
    assert artifact.element_count > 0

    tag_map = json.loads(artifact.tag_map_path.read_text(encoding="utf-8"))
    assert tag_map["background_domain"]["dim"] == 2
    assert tag_map["signal_boundary"]["dim"] == 1
    assert tag_map["ground_boundary"]["dim"] == 1
    assert len(tag_map["signal_boundary"]["entity_tags"]) == 4
    assert len(tag_map["ground_boundary"]["entity_tags"]) == 4


def test_boundary_proximity_sequence_refines_electrode_neighborhood(tmp_path) -> None:
    gmsh = pytest.importorskip("gmsh")
    artifact = generate_legacy_gmsh_mesh(
        load_config("examples/mesh_controls/parallel_plate_proximity.yaml"),
        tmp_path,
        model_name="parallel_plate_proximity",
    )
    tag_map = json.loads(artifact.tag_map_path.read_text(encoding="utf-8"))
    assert [operation["name"] for operation in tag_map["mesh_sequence"]["operations"]] == [
        "global_size",
        "signal_proximity",
        "ground_proximity",
        "free_triangular",
    ]

    gmsh.initialize()
    try:
        gmsh.open(str(artifact.mesh_path))
        near_edges, far_edges = _triangle_edge_lengths_by_region(gmsh)
    finally:
        gmsh.finalize()
    assert sum(near_edges) / len(near_edges) < 0.55 * (sum(far_edges) / len(far_edges))


def _triangle_edge_lengths_by_region(gmsh: object) -> tuple[list[float], list[float]]:
    node_tags, coordinates, _ = gmsh.model.mesh.getNodes()  # type: ignore[attr-defined]
    points = {
        int(tag): (float(coordinates[3 * index]), float(coordinates[3 * index + 1]))
        for index, tag in enumerate(node_tags)
    }
    near: list[float] = []
    far: list[float] = []
    element_types, _, node_blocks = gmsh.model.mesh.getElements(2)  # type: ignore[attr-defined]
    for element_type, flat_nodes in zip(element_types, node_blocks, strict=True):
        _, _, _, node_count, _, _ = gmsh.model.mesh.getElementProperties(element_type)  # type: ignore[attr-defined]
        if int(node_count) < 3:
            continue
        for start in range(0, len(flat_nodes), int(node_count)):
            triangle = [points[int(tag)] for tag in flat_nodes[start : start + 3]]
            centroid_y = sum(point[1] for point in triangle) / 3
            lengths = [
                math.dist(triangle[0], triangle[1]),
                math.dist(triangle[1], triangle[2]),
                math.dist(triangle[2], triangle[0]),
            ]
            if abs(abs(centroid_y) - 1.2e-6) < 0.6e-6:
                near.extend(lengths)
            elif abs(centroid_y) > 5e-6:
                far.extend(lengths)
    assert near and far
    return near, far
