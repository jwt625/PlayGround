from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArtifactRun:
    root: Path
    mesh_dir: Path
    results_dir: Path
    logs_dir: Path


def create_artifact_run(base_dir: str | Path, name: str, *, timestamp: datetime | None = None) -> ArtifactRun:
    now = timestamp or datetime.now(UTC)
    safe_name = _safe_path_component(name)
    run_id = f"{now.strftime('%Y%m%dT%H%M%SZ')}_{safe_name}"
    root = Path(base_dir) / run_id
    mesh_dir = root / "mesh"
    results_dir = root / "results"
    logs_dir = root / "logs"
    for path in (mesh_dir, results_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=False)
    return ArtifactRun(root=root, mesh_dir=mesh_dir, results_dir=results_dir, logs_dir=logs_dir)


def simulation_name(config: dict[str, Any], fallback: str = "simulation") -> str:
    return str(config.get("Simulation", {}).get("name", fallback))


def _safe_path_component(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.strip())
    return safe or "simulation"
