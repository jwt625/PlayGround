from __future__ import annotations

import json

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
