from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eo_fem.backends.gmsh_fields import compile_mesh_sequence
from eo_fem.backends.mesh_sequence import MeshSequenceSpec, parse_mesh_sequence
from eo_fem.backends.schema import BoundarySpec, ExplicitModel, ShapeSpec, legacy_config_to_explicit_model


@dataclass(frozen=True)
class GmshMeshArtifact:
    mesh_path: Path
    tag_map_path: Path
    node_count: int
    element_count: int
    physical_groups: dict[str, Any]


def generate_legacy_gmsh_mesh(
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    model_name: str = "eo_fem",
) -> GmshMeshArtifact:
    model = legacy_config_to_explicit_model(config)
    mesh_size = _mesh_size_from_legacy_config(config)
    mesh_sequence = parse_mesh_sequence(config, default_size=mesh_size)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return generate_gmsh_mesh(
        model,
        output_path / f"{model_name}.msh",
        output_path / f"{model_name}.tags.json",
        mesh_size=mesh_size,
        model_name=model_name,
        mesh_sequence=mesh_sequence,
    )


def generate_gmsh_mesh(
    model: ExplicitModel,
    mesh_path: str | Path,
    tag_map_path: str | Path,
    *,
    mesh_size: float,
    model_name: str = "eo_fem",
    mesh_sequence: MeshSequenceSpec | None = None,
) -> GmshMeshArtifact:
    gmsh = importlib.import_module("gmsh")
    mesh_file = Path(mesh_path)
    tag_file = Path(tag_map_path)
    gmsh.initialize(interruptible=False)
    try:
        gmsh.model.add(model_name)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)

        background = model.domains["background"]
        domain_surface = _add_surface(gmsh, background.shape)
        tool_surfaces = [_add_surface(gmsh, boundary.shape) for boundary in model.boundaries.values()]

        if tool_surfaces:
            cut_surfaces, _ = gmsh.model.occ.cut(
                [(2, domain_surface)],
                [(2, tag) for tag in tool_surfaces],
                removeObject=True,
                removeTool=True,
            )
            domain_surfaces = [tag for dim, tag in cut_surfaces if dim == 2]
        else:
            domain_surfaces = [domain_surface]

        gmsh.model.occ.synchronize()
        _set_uniform_mesh_size(gmsh, mesh_size)

        physical_groups: dict[str, Any] = {}
        physical_groups["background_domain"] = _add_physical_group(
            gmsh,
            dim=2,
            tags=domain_surfaces,
            name="background_domain",
        )
        for boundary in model.boundaries.values():
            curve_tags = _matching_boundary_curves(gmsh, boundary)
            physical_groups[boundary.name] = _add_physical_group(
                gmsh,
                dim=1,
                tags=curve_tags,
                name=boundary.name,
            )

        if mesh_sequence is not None:
            selections = _mesh_selections(model, physical_groups)
            physical_groups["mesh_sequence"] = compile_mesh_sequence(gmsh, mesh_sequence, selections)

        gmsh.model.mesh.generate(2)
        if mesh_sequence is not None:
            for operation in mesh_sequence.operations:
                if operation.enabled and operation.type == "uniform_refine":
                    for _ in range(int(operation.parameters["levels"])):
                        gmsh.model.mesh.refine()
        gmsh.write(str(mesh_file))

        node_tags, _, _ = gmsh.model.mesh.getNodes()
        element_types, element_tags, _ = gmsh.model.mesh.getElements(2)
        element_count = sum(len(tags) for tags in element_tags)
        physical_groups["summary"] = {
            "node_count": int(len(node_tags)),
            "element_count": int(element_count),
            "surface_element_type_count": int(len(element_types)),
        }
        tag_file.write_text(json.dumps(physical_groups, indent=2, sort_keys=True), encoding="utf-8")
        return GmshMeshArtifact(
            mesh_path=mesh_file,
            tag_map_path=tag_file,
            node_count=int(len(node_tags)),
            element_count=int(element_count),
            physical_groups=physical_groups,
        )
    finally:
        gmsh.finalize()


def _add_surface(gmsh: Any, shape: ShapeSpec) -> int:
    if shape.kind == "rectangle":
        return int(
            gmsh.model.occ.addRectangle(
                shape.params["x_min"],
                shape.params["y_min"],
                0.0,
                shape.params["x_max"] - shape.params["x_min"],
                shape.params["y_max"] - shape.params["y_min"],
            )
        )
    if shape.kind == "circle":
        return int(
            gmsh.model.occ.addDisk(
                shape.params["x"],
                shape.params["y"],
                0.0,
                shape.params["radius"],
                shape.params["radius"],
            )
        )
    raise ValueError(f"unsupported Gmsh surface shape: {shape.kind}")


def _set_uniform_mesh_size(gmsh: Any, mesh_size: float) -> None:
    points = gmsh.model.getEntities(0)
    if points:
        gmsh.model.mesh.setSize(points, mesh_size)


def _add_physical_group(gmsh: Any, *, dim: int, tags: list[int], name: str) -> dict[str, int | list[int]]:
    if not tags:
        raise ValueError(f"physical group {name} has no entity tags")
    group_tag = int(gmsh.model.addPhysicalGroup(dim, tags))
    gmsh.model.setPhysicalName(dim, group_tag, name)
    return {"dim": dim, "physical_tag": group_tag, "entity_tags": tags}


def _matching_boundary_curves(gmsh: Any, boundary: BoundarySpec) -> list[int]:
    curves = [tag for _, tag in gmsh.model.getEntities(1)]
    if boundary.shape.kind == "rectangle":
        matches = [tag for tag in curves if _curve_matches_rectangle(gmsh, tag, boundary.shape.params)]
    elif boundary.shape.kind == "circle":
        matches = [tag for tag in curves if _curve_matches_circle(gmsh, tag, boundary.shape.params)]
    else:
        raise ValueError(f"unsupported boundary shape: {boundary.shape.kind}")
    if not matches:
        raise ValueError(f"no Gmsh curves matched boundary {boundary.name}")
    return sorted(matches)


def _curve_matches_rectangle(gmsh: Any, tag: int, params: dict[str, float]) -> bool:
    xmin, ymin, _, xmax, ymax, _ = _entity_bbox(gmsh, 1, tag)
    tol = _bbox_tolerance(params)
    on_vertical_edge = (
        _close(xmin, xmax, tol)
        and (_close(xmin, params["x_min"], tol) or _close(xmin, params["x_max"], tol))
        and ymin >= params["y_min"] - tol
        and ymax <= params["y_max"] + tol
    )
    on_horizontal_edge = (
        _close(ymin, ymax, tol)
        and (_close(ymin, params["y_min"], tol) or _close(ymin, params["y_max"], tol))
        and xmin >= params["x_min"] - tol
        and xmax <= params["x_max"] + tol
    )
    return on_vertical_edge or on_horizontal_edge


def _curve_matches_circle(gmsh: Any, tag: int, params: dict[str, float]) -> bool:
    xmin, ymin, _, xmax, ymax, _ = _entity_bbox(gmsh, 1, tag)
    radius = params["radius"]
    tol = max(radius * 0.25, 1e-12)
    return (
        xmin >= params["x"] - radius - tol
        and xmax <= params["x"] + radius + tol
        and ymin >= params["y"] - radius - tol
        and ymax <= params["y"] + radius + tol
    )


def _bbox_tolerance(params: dict[str, float]) -> float:
    span = max(params.get("x_max", 0.0) - params.get("x_min", 0.0), params.get("y_max", 0.0) - params.get("y_min", 0.0))
    return max(abs(span) * 0.05, 1e-12)


def _close(left: float, right: float, tol: float) -> bool:
    return abs(left - right) <= tol


def _entity_bbox(gmsh: Any, dim: int, tag: int) -> tuple[float, float, float, float, float, float]:
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
    return float(xmin), float(ymin), float(zmin), float(xmax), float(ymax), float(zmax)


def _mesh_size_from_legacy_config(config: dict[str, Any]) -> float:
    domain = config["Domain"]
    sim = config.get("Simulation", {})
    dx = (float(domain["x_max"]) - float(domain["x_min"])) / max(int(sim.get("mesh_nx", 81)) - 1, 1)
    dy = (float(domain["y_max"]) - float(domain["y_min"])) / max(int(sim.get("mesh_ny", 61)) - 1, 1)
    return max(min(dx, dy), 1e-15)


def _mesh_selections(model: ExplicitModel, physical_groups: dict[str, Any]) -> dict[str, dict[str, Any]]:
    domain = physical_groups["background_domain"]
    selections: dict[str, dict[str, Any]] = {
        "entire_geometry": {"dim": domain["dim"], "entity_tags": domain["entity_tags"]},
        "all_domains": {"dim": domain["dim"], "entity_tags": domain["entity_tags"]},
        "background_domain": {"dim": domain["dim"], "entity_tags": domain["entity_tags"]},
    }
    for boundary in model.boundaries.values():
        group = physical_groups[boundary.name]
        value = {"dim": group["dim"], "entity_tags": group["entity_tags"]}
        selections[boundary.name] = value
        selections[f"{boundary.source}_boundary_selection"] = value
    return selections
