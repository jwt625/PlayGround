"""Environment group (coordinator): table, plinth, ground and green backdrop as helper objects ("_env.*":
rendered in RGB, excluded from ID/eval/points). Built in the scene (gravity) frame from config/model/env.toml and
moved into the tray (world) frame with lib.apply_frame(objs, "scene")."""

import lib


def _box(name, xr, yr, zr, coll, mat):
    size = (xr[1] - xr[0], yr[1] - yr[0], zr[1] - zr[0])
    ctr = ((xr[0] + xr[1]) / 2, (yr[0] + yr[1]) / 2, (zr[0] + zr[1]) / 2)
    return lib.box(name, size, ctr, coll, mat)


def build(P: dict, coll) -> None:
    sp = P.get("specular", 0.2)

    def mat(key):
        m = lib.mat_pbr(f"env_{key}", color=tuple(P[key]["color"]), roughness=0.8)
        m.node_tree.nodes["Principled BSDF"].inputs["Specular IOR Level"].default_value = sp
        return m

    t, pl, g, b = P["table"], P["plinth"], P["ground"], P["backdrop"]
    objs = [
        _box("_env.table", t["x"], t["y"], (t["z"] - t["thickness"], t["z"]), coll, mat("table")),
        _box("_env.plinth", pl["x"], pl["y"], pl["z"], coll, mat("plinth")),
        _box("_env.ground", (-g["half_size"], g["half_size"]), (-g["half_size"], g["half_size"]), (g["z"] - 10, g["z"]),
             coll, mat("ground")),
        _box("_env.backdrop", b["x"], (b["y"], b["y"] + 10), b["z"], coll, mat("backdrop")),
    ]
    lib.apply_frame(objs, "scene")
