import math

from eo_fem.analytic import parallel_plate_capacitance, two_cylinder_capacitance
from eo_fem.constants import EPS0


def test_parallel_plate_reference():
    assert parallel_plate_capacitance(2.0, 4.0, 8.0) == EPS0


def test_two_cylinder_reference():
    c = two_cylinder_capacitance(1.0, radius=0.5, center_distance=3.0)
    assert math.isclose(c, math.pi * EPS0 / math.acosh(3.0), rel_tol=1e-15)
