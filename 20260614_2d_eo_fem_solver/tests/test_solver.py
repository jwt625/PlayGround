from copy import deepcopy

from eo_fem.config import load_config
from eo_fem.solver import solve_config


def test_parallel_plate_is_close_to_analytic():
    cfg = load_config("examples/parallel_plate.yaml")
    result = solve_config(cfg)
    rel_err = abs(result.capacitance_energy - result.reference["capacitance"]) / result.reference[
        "capacitance"
    ]
    assert rel_err < 0.35


def test_parallel_plate_width_reduces_relative_error():
    narrow = load_config("examples/parallel_plate.yaml")
    wide = deepcopy(narrow)
    narrow["Electrodes"]["signal"]["x_min"] = -2e-6
    narrow["Electrodes"]["signal"]["x_max"] = 2e-6
    narrow["Electrodes"]["ground"]["x_min"] = -2e-6
    narrow["Electrodes"]["ground"]["x_max"] = 2e-6
    narrow["Outputs"]["plate_width"] = 4e-6

    narrow_result = solve_config(narrow)
    wide_result = solve_config(wide)
    narrow_err = abs(narrow_result.capacitance_energy - narrow_result.reference["capacitance"])
    wide_err = abs(wide_result.capacitance_energy - wide_result.reference["capacitance"])
    assert wide_err / wide_result.reference["capacitance"] < narrow_err / narrow_result.reference[
        "capacitance"
    ]


def test_two_cylinders_same_order_as_analytic():
    cfg = load_config("examples/two_cylinders.yaml")
    result = solve_config(cfg)
    ratio = result.capacitance_energy / result.reference["capacitance"]
    assert 0.45 < ratio < 1.8


def test_spatial_scalar_epsilon_increases_capacitance_for_high_k_slab():
    low = slab_config(1.0)
    high = slab_config(30.0)
    low_result = solve_config(low)
    high_result = solve_config(high)
    assert high_result.capacitance_energy > 3 * low_result.capacitance_energy
    assert "spatial scalar eps_r" in high_result.permittivity_model


def test_overlapping_material_regions_use_later_non_background_material_in_assembly():
    high_last = overlapping_slab_config(2.0, 20.0)
    low_last = overlapping_slab_config(20.0, 2.0)
    high_last_result = solve_config(high_last)
    low_last_result = solve_config(low_last)
    assert high_last_result.capacitance_energy > 2 * low_last_result.capacitance_energy


def slab_config(eps_r):
    cfg = load_config("examples/parallel_plate.yaml")
    cfg["Simulation"]["mesh_nx"] = 61
    cfg["Simulation"]["mesh_ny"] = 61
    cfg["Domain"] = {
        "x_min": -6e-6,
        "x_max": 6e-6,
        "y_min": -4e-6,
        "y_max": 4e-6,
    }
    cfg["Materials"] = {
        "background": {"eps_r": 1.0},
        "slab": {
            "shape": "rectangle",
            "eps_r": eps_r,
            "x_min": -6e-6,
            "x_max": 6e-6,
            "y_min": -1e-6,
            "y_max": 1e-6,
        },
    }
    cfg["Electrodes"]["signal"]["y_min"] = 1e-6
    cfg["Electrodes"]["signal"]["y_max"] = 1.2e-6
    cfg["Electrodes"]["ground"]["y_min"] = -1.2e-6
    cfg["Electrodes"]["ground"]["y_max"] = -1e-6
    cfg.pop("Outputs", None)
    return cfg


def overlapping_slab_config(first_eps_r, second_eps_r):
    cfg = slab_config(1.0)
    cfg["Materials"] = {
        "background": {"eps_r": 1.0},
        "first": {
            "shape": "rectangle",
            "eps_r": first_eps_r,
            "x_min": -6e-6,
            "x_max": 6e-6,
            "y_min": -1e-6,
            "y_max": 1e-6,
        },
        "second": {
            "shape": "rectangle",
            "eps_r": second_eps_r,
            "x_min": -6e-6,
            "x_max": 6e-6,
            "y_min": -1e-6,
            "y_max": 1e-6,
        },
    }
    return cfg
