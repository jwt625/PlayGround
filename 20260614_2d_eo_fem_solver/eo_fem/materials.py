from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MATERIAL_PROPERTY_KEYS = {
    "eps_r",
    "eps_r_xx",
    "eps_r_yy",
    "eps_r_xy",
    "r13",
    "r33",
    "r22",
    "r_eff",
}


@dataclass(frozen=True)
class Material:
    name: str
    shape: str
    properties: dict[str, float]
    params: dict[str, float]

    def contains(self, x: float, y: float) -> bool:
        if self.shape == "background":
            return True
        if self.shape == "rectangle":
            return (
                self.params["x_min"] <= x <= self.params["x_max"]
                and self.params["y_min"] <= y <= self.params["y_max"]
            )
        if self.shape == "circle":
            dx = x - self.params["x"]
            dy = y - self.params["y"]
            return dx * dx + dy * dy <= self.params["radius"] ** 2
        raise ValueError(f"unsupported material shape: {self.shape}")


def parse_materials(config: dict[str, Any]) -> list[Material]:
    materials = []
    for name, block in config.get("Materials", {}).items():
        shape = str(block.get("shape", "background"))
        properties = {
            key: float(value) for key, value in block.items() if key in MATERIAL_PROPERTY_KEYS
        }
        params = {
            key: float(value)
            for key, value in block.items()
            if key not in MATERIAL_PROPERTY_KEYS and key != "shape"
        }
        materials.append(Material(name=name, shape=shape, properties=properties, params=params))
    if not any(material.shape == "background" for material in materials):
        materials.insert(
            0,
            Material(
                name="background",
                shape="background",
                properties={"eps_r": 1.0},
                params={},
            ),
        )
    return materials


def material_at(materials: list[Material], x: float, y: float) -> Material:
    selected = next((material for material in materials if material.shape == "background"), materials[0])
    for material in materials:
        if material.shape == "background":
            continue
        if material.contains(x, y):
            selected = material
    return selected


def scalar_eps_r_at(materials: list[Material], x: float, y: float) -> float:
    material = material_at(materials, x, y)
    return material.properties.get("eps_r", material.properties.get("eps_r_xx", 1.0))


def epsilon_tensor_at(materials: list[Material], x: float, y: float) -> tuple[float, float, float]:
    material = material_at(materials, x, y)
    scalar = material.properties.get("eps_r", material.properties.get("eps_r_xx", 1.0))
    return (
        material.properties.get("eps_r_xx", scalar),
        material.properties.get("eps_r_yy", scalar),
        material.properties.get("eps_r_xy", 0.0),
    )


def uses_spatial_permittivity(materials: list[Material]) -> bool:
    return any(material.shape != "background" for material in materials)


def uses_tensor_permittivity(materials: list[Material]) -> bool:
    return any(
        "eps_r_xx" in material.properties
        or "eps_r_yy" in material.properties
        or "eps_r_xy" in material.properties
        for material in materials
    )
