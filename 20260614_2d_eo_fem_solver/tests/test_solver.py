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
