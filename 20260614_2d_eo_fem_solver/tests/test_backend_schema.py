from __future__ import annotations

from eo_fem.backends.schema import legacy_config_to_explicit_model
from eo_fem.config import load_config


def test_parallel_plate_legacy_config_translates_to_explicit_domains_and_boundaries() -> None:
    model = legacy_config_to_explicit_model(load_config("examples/parallel_plate.yaml"))

    assert "background" in model.domains
    assert model.materials["background"].selection == "background_domain"
    assert model.selections["background_domain"].entities == ("background",)

    assert model.boundaries["signal_boundary"].source == "signal"
    assert model.boundaries["ground_boundary"].source == "ground"
    assert model.dirichlet["signal"].selection == "signal_boundary_selection"
    assert model.dirichlet["signal"].potential == 1.0
    assert model.dirichlet["ground"].potential == 0.0


def test_material_regions_become_explicit_domain_assignments() -> None:
    model = legacy_config_to_explicit_model(load_config("examples/material_stack.yaml"))

    assert "oxide" in model.domains
    assert "tfln" in model.domains
    assert model.materials["oxide"].selection == "oxide_domain"
    assert model.materials["tfln"].selection == "tfln_domain"
    assert model.selections["oxide_domain"].entity_kind == "domain"
    assert model.selections["tfln_domain"].entities == ("tfln",)


def test_electrode_shapes_become_boundary_selections() -> None:
    model = legacy_config_to_explicit_model(load_config("examples/two_cylinders.yaml"))

    assert model.boundaries["signal_boundary"].shape.kind == "circle"
    assert model.boundaries["ground_boundary"].shape.kind == "circle"
    assert model.selections["signal_boundary_selection"].entity_kind == "boundary"
    assert model.dirichlet["signal"].selection == "signal_boundary_selection"
