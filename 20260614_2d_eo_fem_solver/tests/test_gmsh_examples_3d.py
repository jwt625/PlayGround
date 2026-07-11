from __future__ import annotations

import json

import pytest

from eo_fem.backends.gmsh_examples_3d import generate_3d_validation_examples


def test_3d_validation_examples_have_tagged_volumes_and_expected_elements(tmp_path) -> None:
    pytest.importorskip("gmsh")
    mesh_paths = generate_3d_validation_examples(tmp_path)
    assert [path.name for path in mesh_paths] == [
        "sphere_in_box.msh",
        "swept_capacitor.msh",
        "coaxial_segment.msh",
    ]

    sphere = json.loads((tmp_path / "sphere_in_box" / "sphere_in_box.tags.json").read_text(encoding="utf-8"))
    assert sphere["matrix_volume"]["dim"] == 3
    assert sphere["inclusion_volume"]["dim"] == 3
    assert sphere["inclusion_interface"]["dim"] == 2
    assert sphere["summary"]["element_types"]["Tetrahedron 4"] > 0
    assert sphere["summary"]["quality_mean_sicn"] > 0.7

    swept = json.loads((tmp_path / "swept_capacitor" / "swept_capacitor.tags.json").read_text(encoding="utf-8"))
    assert swept["dielectric_volume"]["dim"] == 3
    assert swept["signal_face"]["dim"] == 2
    assert swept["ground_face"]["dim"] == 2
    assert swept["summary"]["element_types"]["Prism 6"] > 0
    assert swept["summary"]["quality_min_sicn"] > 0.8

    coax = json.loads((tmp_path / "coaxial_segment" / "coaxial_segment.tags.json").read_text(encoding="utf-8"))
    assert coax["dielectric_volume"]["dim"] == 3
    assert coax["signal_boundary"]["dim"] == 2
    assert coax["ground_boundary"]["dim"] == 2
    assert coax["summary"]["element_types"]["Tetrahedron 4"] > 0
    assert coax["summary"]["quality_mean_sicn"] > 0.7
