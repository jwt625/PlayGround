from __future__ import annotations

import argparse

from .config import load_config
from .solver import solve_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a 2D EO electrostatic solve.")
    parser.add_argument("config", help="Path to a MOOSE-like YAML config.")
    args = parser.parse_args()
    result = solve_config(load_config(args.config))
    print(f"C_energy = {result.capacitance_energy:.6e} F/m")
    print(f"C_energy = {result.units['fF_per_mm']:.3f} fF/mm")
    print(f"C_energy = {result.units['pF_per_cm']:.3f} pF/cm")
    print(f"C_charge = {result.capacitance_charge:.6e} F/m")
    print(f"Permittivity model = {result.permittivity_model}")
    print(f"CG iterations = {result.iterations}, residual = {result.residual:.3e}")
    if result.reference:
        ref = result.reference
        reference_capacitance = float(ref["capacitance"])
        err = (result.capacitance_energy - reference_capacitance) / reference_capacitance
        print(f"Reference({ref['name']}) = {reference_capacitance:.6e} F/m")
        print(f"Relative error = {err:.3%}")
