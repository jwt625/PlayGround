from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from eo_fem.geometry import parse_domain, parse_electrodes
from eo_fem.materials import parse_materials

ShapeKind = Literal["rectangle", "circle"]
EntityKind = Literal["domain", "boundary", "vertex"]
SelectionMethod = Literal["explicit", "box", "derived"]


@dataclass(frozen=True)
class ShapeSpec:
    kind: ShapeKind
    params: dict[str, float]


@dataclass(frozen=True)
class DomainSpec:
    name: str
    shape: ShapeSpec


@dataclass(frozen=True)
class BoundarySpec:
    name: str
    shape: ShapeSpec
    source: str | None = None


@dataclass(frozen=True)
class VertexSpec:
    name: str
    x: float
    y: float
    source: str | None = None


@dataclass(frozen=True)
class SelectionSpec:
    name: str
    entity_kind: EntityKind
    method: SelectionMethod
    entities: tuple[str, ...]


@dataclass(frozen=True)
class MaterialAssignment:
    name: str
    selection: str
    properties: dict[str, float]


@dataclass(frozen=True)
class DirichletBoundaryCondition:
    name: str
    selection: str
    potential: float


@dataclass(frozen=True)
class ExplicitModel:
    domains: dict[str, DomainSpec]
    boundaries: dict[str, BoundarySpec]
    vertices: dict[str, VertexSpec]
    selections: dict[str, SelectionSpec]
    materials: dict[str, MaterialAssignment]
    dirichlet: dict[str, DirichletBoundaryCondition]


def legacy_config_to_explicit_model(config: dict[str, Any]) -> ExplicitModel:
    """Translate the current compact YAML format into explicit backend entities.

    The legacy browser solver resolves material regions by shape overlap. The
    backend model makes each region an explicit domain and assigns materials to
    named selections. Electrode shapes become boundary selections; the Gmsh
    backend can later decide whether to mesh them as internal conductor holes or
    tagged embedded boundaries.
    """

    domain = parse_domain(config)
    domains = {
        "background": DomainSpec(
            name="background",
            shape=ShapeSpec(
                kind="rectangle",
                params={
                    "x_min": domain.x_min,
                    "x_max": domain.x_max,
                    "y_min": domain.y_min,
                    "y_max": domain.y_max,
                },
            ),
        )
    }
    boundaries: dict[str, BoundarySpec] = {}
    vertices: dict[str, VertexSpec] = {}
    selections = {
        "background_domain": SelectionSpec(
            name="background_domain",
            entity_kind="domain",
            method="explicit",
            entities=("background",),
        )
    }
    material_assignments: dict[str, MaterialAssignment] = {}
    for material in parse_materials(config):
        if material.shape == "background":
            selection_name = "background_domain"
        else:
            domains[material.name] = DomainSpec(
                name=material.name,
                shape=ShapeSpec(kind=_shape_kind(material.shape), params=material.params),
            )
            selection_name = f"{material.name}_domain"
            selections[selection_name] = SelectionSpec(
                name=selection_name,
                entity_kind="domain",
                method="explicit",
                entities=(material.name,),
            )
        material_assignments[material.name] = MaterialAssignment(
            name=material.name,
            selection=selection_name,
            properties=material.properties,
        )

    dirichlet: dict[str, DirichletBoundaryCondition] = {}
    for electrode in parse_electrodes(config):
        boundary_name = f"{electrode.name}_boundary"
        selection_name = f"{electrode.name}_boundary_selection"
        boundaries[boundary_name] = BoundarySpec(
            name=boundary_name,
            shape=ShapeSpec(kind=_shape_kind(electrode.shape), params=electrode.params),
            source=electrode.name,
        )
        selections[selection_name] = SelectionSpec(
            name=selection_name,
            entity_kind="boundary",
            method="explicit",
            entities=(boundary_name,),
        )
        dirichlet[electrode.name] = DirichletBoundaryCondition(
            name=electrode.name,
            selection=selection_name,
            potential=electrode.potential,
        )

    return ExplicitModel(
        domains=domains,
        boundaries=boundaries,
        vertices=vertices,
        selections=selections,
        materials=material_assignments,
        dirichlet=dirichlet,
    )


def _shape_kind(shape: str) -> ShapeKind:
    if shape == "rectangle":
        return "rectangle"
    if shape == "circle":
        return "circle"
    raise ValueError(f"unsupported backend shape: {shape}")
