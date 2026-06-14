from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Electrode:
    name: str
    shape: str
    potential: float
    params: dict[str, float]

    def contains(self, x: float, y: float) -> bool:
        if self.shape == "rectangle":
            return (
                self.params["x_min"] <= x <= self.params["x_max"]
                and self.params["y_min"] <= y <= self.params["y_max"]
            )
        if self.shape == "circle":
            dx = x - self.params["x"]
            dy = y - self.params["y"]
            return dx * dx + dy * dy <= self.params["radius"] ** 2
        raise ValueError(f"unsupported electrode shape: {self.shape}")


@dataclass(frozen=True)
class Domain:
    x_min: float
    x_max: float
    y_min: float
    y_max: float


def parse_domain(config: dict[str, Any]) -> Domain:
    block = config["Domain"]
    return Domain(
        x_min=float(block["x_min"]),
        x_max=float(block["x_max"]),
        y_min=float(block["y_min"]),
        y_max=float(block["y_max"]),
    )


def parse_electrodes(config: dict[str, Any]) -> list[Electrode]:
    electrodes = []
    for name, block in config["Electrodes"].items():
        shape = str(block["shape"])
        potential = float(block["potential"])
        params = {k: float(v) for k, v in block.items() if k not in {"shape", "potential"}}
        electrodes.append(Electrode(name=name, shape=shape, potential=potential, params=params))
    return electrodes
