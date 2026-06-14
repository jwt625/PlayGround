EPS0 = 8.8541878128e-12


def capacitance_units(c_per_m: float) -> dict[str, float]:
    return {
        "F_per_m": c_per_m,
        "fF_per_mm": c_per_m * 1e12 / 1e3,
        "pF_per_cm": c_per_m * 1e12 / 100.0,
    }
