"""Shared paths and config loading for this project."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"


def load_yaml(name: str) -> dict:
    with open(CONFIG / name) as f:
        return yaml.safe_load(f)


def scene_cfg() -> dict:
    cfg = load_yaml("scene.yaml")
    cfg["source_dir"] = (ROOT / Path(cfg["source_dir"]).expanduser()).resolve()
    cfg["sparse_dir"] = (ROOT / cfg["sparse_model"]).resolve()
    return cfg
