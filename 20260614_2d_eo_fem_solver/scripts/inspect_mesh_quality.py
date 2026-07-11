from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any


def inspect_mesh(mesh_path: str | Path) -> dict[str, Any]:
    gmsh = importlib.import_module("gmsh")
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(mesh_path))
        _, element_blocks, _ = gmsh.model.mesh.getElements(3)
        element_tags = [int(tag) for block in element_blocks for tag in block]
        qualities = [float(value) for value in gmsh.model.mesh.getElementQualities(element_tags, "minSICN")]
        ordered = sorted(zip(qualities, element_tags, strict=True))
        bins = [0] * 10
        for quality in qualities:
            bins[min(9, max(0, int(quality * 10)))] += 1
        return {
            "mesh": str(mesh_path),
            "metric": "minSICN",
            "element_count": len(element_tags),
            "minimum": min(qualities),
            "mean": sum(qualities) / len(qualities),
            "percentiles": {
                "p01": _percentile(qualities, 0.01),
                "p05": _percentile(qualities, 0.05),
                "p50": _percentile(qualities, 0.50),
                "p95": _percentile(qualities, 0.95),
            },
            "histogram": [
                {"low": index / 10, "high": (index + 1) / 10, "count": count}
                for index, count in enumerate(bins)
            ],
            "worst_elements": [
                {"element_tag": tag, "quality": quality, "centroid": _element_centroid(gmsh, tag)}
                for quality, tag in ordered[:10]
            ],
        }
    finally:
        gmsh.finalize()


def _element_centroid(gmsh: Any, element_tag: int) -> list[float]:
    _, node_tags, _, _ = gmsh.model.mesh.getElement(element_tag)
    points = [gmsh.model.mesh.getNode(int(tag))[0] for tag in node_tags]
    return [float(sum(point[axis] for point in points) / len(points)) for axis in range(3)]


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Write Gmsh volume-element quality evidence.")
    parser.add_argument("mesh")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = inspect_mesh(args.mesh)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
