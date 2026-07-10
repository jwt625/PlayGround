from __future__ import annotations

from datetime import UTC, datetime

from eo_fem.backends.artifacts import create_artifact_run, simulation_name


def test_create_artifact_run_uses_stable_directory_layout(tmp_path) -> None:
    run = create_artifact_run(
        tmp_path,
        "parallel plate / gmsh",
        timestamp=datetime(2026, 7, 10, 12, 30, tzinfo=UTC),
    )

    assert run.root.name == "20260710T123000Z_parallel_plate___gmsh"
    assert run.mesh_dir.is_dir()
    assert run.results_dir.is_dir()
    assert run.logs_dir.is_dir()


def test_simulation_name_reads_current_config_shape() -> None:
    assert simulation_name({"Simulation": {"name": "demo"}}) == "demo"
    assert simulation_name({}) == "simulation"
