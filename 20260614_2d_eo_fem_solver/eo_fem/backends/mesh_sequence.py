from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

MeshOperationType = Literal[
    "size",
    "boundary_proximity",
    "box_size",
    "curvature",
    "free_triangular",
    "free_tetrahedral",
    "uniform_refine",
]

_ALLOWED_OPERATION_TYPES = {
    "size",
    "boundary_proximity",
    "box_size",
    "curvature",
    "free_triangular",
    "free_tetrahedral",
    "uniform_refine",
}


@dataclass(frozen=True)
class MeshOperationSpec:
    name: str
    type: MeshOperationType
    selection: str
    enabled: bool
    parameters: dict[str, float | int | str | bool]


@dataclass(frozen=True)
class MeshSequenceSpec:
    name: str
    dimension: int
    sequence_type: Literal["physics_controlled", "user_controlled"]
    operations: tuple[MeshOperationSpec, ...]


def parse_mesh_sequence(config: dict[str, Any], *, default_size: float) -> MeshSequenceSpec:
    block = config.get("Mesh")
    if block is None:
        return MeshSequenceSpec(
            name="mesh1",
            dimension=2,
            sequence_type="user_controlled",
            operations=(
                MeshOperationSpec(
                    name="global_size",
                    type="size",
                    selection="entire_geometry",
                    enabled=True,
                    parameters={"h_max": default_size, "h_min": default_size},
                ),
                MeshOperationSpec(
                    name="free_triangular",
                    type="free_triangular",
                    selection="all_domains",
                    enabled=True,
                    parameters={},
                ),
            ),
        )
    if not isinstance(block, dict):
        raise ValueError("Mesh must be a mapping")
    dimension = int(block.get("dimension", 2))
    if dimension not in {2, 3}:
        raise ValueError("Mesh.dimension must be 2 or 3")
    sequence_type = str(block.get("sequence_type", "user_controlled"))
    if sequence_type not in {"physics_controlled", "user_controlled"}:
        raise ValueError("Mesh.sequence_type must be physics_controlled or user_controlled")
    raw_operations = block.get("operations", {})
    if not isinstance(raw_operations, dict) or not raw_operations:
        raise ValueError("Mesh.operations must be a nonempty mapping")
    operations = tuple(_parse_operation(name, raw) for name, raw in raw_operations.items())
    _validate_generator(dimension, operations)
    return MeshSequenceSpec(
        name=str(block.get("name", "mesh1")),
        dimension=dimension,
        sequence_type=sequence_type,  # type: ignore[arg-type]
        operations=operations,
    )


def _parse_operation(name: str, raw: Any) -> MeshOperationSpec:
    if not isinstance(raw, dict):
        raise ValueError(f"Mesh operation {name} must be a mapping")
    operation_type = str(raw.get("type", ""))
    if operation_type not in _ALLOWED_OPERATION_TYPES:
        raise ValueError(f"Mesh operation {name} has unsupported type: {operation_type}")
    selection = str(raw.get("selection", _default_selection(operation_type)))
    enabled = bool(raw.get("enabled", True))
    parameters = {
        key: value
        for key, value in raw.items()
        if key not in {"type", "selection", "enabled"}
    }
    _validate_parameters(name, operation_type, parameters)
    return MeshOperationSpec(
        name=name,
        type=operation_type,  # type: ignore[arg-type]
        selection=selection,
        enabled=enabled,
        parameters=parameters,
    )


def _validate_parameters(name: str, operation_type: str, parameters: dict[str, Any]) -> None:
    required: dict[str, tuple[str, ...]] = {
        "size": ("h_max",),
        "boundary_proximity": ("h_min", "h_max", "distance_min", "distance_max"),
        "box_size": ("h_in", "h_out", "x_min", "x_max", "y_min", "y_max"),
        "curvature": ("elements_per_2pi",),
        "uniform_refine": ("levels",),
    }
    missing = [key for key in required.get(operation_type, ()) if key not in parameters]
    if missing:
        raise ValueError(f"Mesh operation {name} is missing: {', '.join(missing)}")
    positive_keys = {
        "h_min",
        "h_max",
        "h_in",
        "h_out",
        "distance_min",
        "distance_max",
        "elements_per_2pi",
        "levels",
    }
    for key in positive_keys.intersection(parameters):
        if float(parameters[key]) <= 0:
            raise ValueError(f"Mesh operation {name}.{key} must be positive")
    if "h_min" in parameters and "h_max" in parameters and float(parameters["h_min"]) > float(parameters["h_max"]):
        raise ValueError(f"Mesh operation {name} requires h_min <= h_max")
    inverted_distance = (
        "distance_min" in parameters
        and "distance_max" in parameters
        and float(parameters["distance_min"]) > float(parameters["distance_max"])
    )
    if inverted_distance:
        raise ValueError(f"Mesh operation {name} requires distance_min <= distance_max")


def _validate_generator(dimension: int, operations: tuple[MeshOperationSpec, ...]) -> None:
    generators = [
        operation.type for operation in operations if operation.enabled and operation.type.startswith("free_")
    ]
    expected = "free_triangular" if dimension == 2 else "free_tetrahedral"
    if expected not in generators:
        raise ValueError(f"Mesh dimension {dimension} requires an enabled {expected} operation")


def _default_selection(operation_type: str) -> str:
    if operation_type in {"size", "curvature"}:
        return "entire_geometry"
    if operation_type.startswith("free_"):
        return "all_domains"
    return ""
