from __future__ import annotations

from dataclasses import asdict
from typing import Any

from eo_fem.backends.mesh_sequence import MeshOperationSpec, MeshSequenceSpec


def compile_mesh_sequence(
    gmsh: Any,
    sequence: MeshSequenceSpec,
    selections: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    field_tags: list[int] = []
    compiled_operations: list[dict[str, Any]] = []
    h_mins: list[float] = []
    h_maxs: list[float] = []
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

    for operation in sequence.operations:
        compiled = {**asdict(operation), "field_tags": []}
        if operation.enabled:
            tags = _compile_operation(gmsh, operation, selections)
            compiled["field_tags"] = tags
            field_tags.extend(tags[-1:] if operation.type in {"size", "boundary_proximity", "box_size"} else [])
            _collect_size_limits(operation, h_mins, h_maxs)
        compiled_operations.append(compiled)

    if field_tags:
        background = field_tags[0]
        if len(field_tags) > 1:
            background = int(gmsh.model.mesh.field.add("Min"))
            gmsh.model.mesh.field.setNumbers(background, "FieldsList", field_tags)
        gmsh.model.mesh.field.setAsBackgroundMesh(background)
    else:
        background = None
    if h_mins:
        gmsh.option.setNumber("Mesh.MeshSizeMin", min(h_mins))
    if h_maxs:
        gmsh.option.setNumber("Mesh.MeshSizeMax", max(h_maxs))
    return {
        "name": sequence.name,
        "dimension": sequence.dimension,
        "sequence_type": sequence.sequence_type,
        "operations": compiled_operations,
        "background_field": background,
    }


def _compile_operation(
    gmsh: Any,
    operation: MeshOperationSpec,
    selections: dict[str, dict[str, Any]],
) -> list[int]:
    params = operation.parameters
    if operation.type == "size":
        field = int(gmsh.model.mesh.field.add("Constant"))
        gmsh.model.mesh.field.setNumber(field, "VIn", float(params["h_max"]))
        outside_size = float(params["h_max"]) if operation.selection == "entire_geometry" else 1e22
        gmsh.model.mesh.field.setNumber(field, "VOut", outside_size)
        if operation.selection != "entire_geometry":
            selection = _selection(selections, operation.selection)
            gmsh.model.mesh.field.setNumbers(field, _entity_list_option(selection["dim"]), selection["entity_tags"])
        return [field]
    if operation.type == "boundary_proximity":
        selection = _selection(selections, operation.selection)
        if selection["dim"] not in {1, 2}:
            raise ValueError(f"boundary_proximity selection {operation.selection} must contain curves or surfaces")
        distance = int(gmsh.model.mesh.field.add("Distance"))
        option = "CurvesList" if selection["dim"] == 1 else "SurfacesList"
        gmsh.model.mesh.field.setNumbers(distance, option, selection["entity_tags"])
        gmsh.model.mesh.field.setNumber(distance, "Sampling", int(params.get("sampling", 100)))
        threshold = int(gmsh.model.mesh.field.add("Threshold"))
        gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
        gmsh.model.mesh.field.setNumber(threshold, "SizeMin", float(params["h_min"]))
        gmsh.model.mesh.field.setNumber(threshold, "SizeMax", float(params["h_max"]))
        gmsh.model.mesh.field.setNumber(threshold, "DistMin", float(params["distance_min"]))
        gmsh.model.mesh.field.setNumber(threshold, "DistMax", float(params["distance_max"]))
        return [distance, threshold]
    if operation.type == "box_size":
        field = int(gmsh.model.mesh.field.add("Box"))
        option_map = {
            "h_in": "VIn",
            "h_out": "VOut",
            "x_min": "XMin",
            "x_max": "XMax",
            "y_min": "YMin",
            "y_max": "YMax",
            "z_min": "ZMin",
            "z_max": "ZMax",
            "thickness": "Thickness",
        }
        for key, option in option_map.items():
            if key in params:
                gmsh.model.mesh.field.setNumber(field, option, float(params[key]))
        return [field]
    if operation.type == "curvature":
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", float(params["elements_per_2pi"]))
        return []
    if operation.type == "free_triangular":
        algorithm = str(params.get("algorithm", "frontal_delaunay"))
        gmsh.option.setNumber("Mesh.Algorithm", {"delaunay": 5, "frontal_delaunay": 6, "bamg": 7}[algorithm])
        return []
    if operation.type == "free_tetrahedral":
        algorithm = str(params.get("algorithm", "delaunay"))
        gmsh.option.setNumber("Mesh.Algorithm3D", {"delaunay": 1, "frontal": 4, "hxt": 10}[algorithm])
        return []
    if operation.type == "uniform_refine":
        return []
    raise ValueError(f"unsupported mesh operation: {operation.type}")


def _selection(selections: dict[str, dict[str, Any]], name: str) -> dict[str, Any]:
    try:
        return selections[name]
    except KeyError as error:
        raise ValueError(f"mesh operation references unknown selection: {name}") from error


def _entity_list_option(dim: int) -> str:
    return {0: "PointsList", 1: "CurvesList", 2: "SurfacesList", 3: "VolumesList"}[dim]


def _collect_size_limits(operation: MeshOperationSpec, h_mins: list[float], h_maxs: list[float]) -> None:
    params = operation.parameters
    for key in ("h_min", "h_in"):
        if key in params:
            h_mins.append(float(params[key]))
    for key in ("h_max", "h_out"):
        if key in params:
            h_maxs.append(float(params[key]))
