"""Environment group: the silicone mat as a plane textured from the photos (corners from the texture spec)."""

import json

import lib


def build(P: dict, coll) -> None:
    spec = json.loads((lib.ROOT / "config" / "model" / "env_mat_texture_spec.json").read_text())
    c = [[x, y, P["mat"]["z"]] for x, y, _ in spec["surface"]["corners_mm"]]
    mat = lib.mat_image("env_mat_tex", P["mat"]["texture"], roughness=P["mat"]["roughness"])
    lib.label_quad("_env.mat", c, coll, mat, offset_mm=0.0)
