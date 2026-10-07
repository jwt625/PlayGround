"""Render a model at the COLMAP views (runs inside Blender).

Usage (always through the slot wrapper):
  scripts/bslot.sh -b [model.blend] --python scripts/blender/render_views.py -- \
      --out outputs/runs/<run> [--build scripts/model/build_all.py] [--views probe|holdout|train|all|IMG_1518,...]
      [--scale 4] [--passes id,rgb,pts] [--samples 16]

Passes:
  id   Workbench flat object colors, no AA; decode with id_map.json (object name -> 8-bit sRGB color).
       Background alpha 0. Exact per-object coverage for silhouettes, edges and per-part error attribution.
  rgb  EEVEE render under a uniform white world (strength 1), no lights: photo-textured materials (no added
       specular) reproduce photo colors; eval also normalizes color per view.
  pts  Distance from every sparse point (data/points_world.npy) to the nearest model surface, with the
       nearest object's name: points.npz (dist_m, obj_index, names).

Model input: either a .blend opened by Blender, or --build <script> executed in a fresh scene (the script must
create the model's objects; nothing else). Objects with names starting "_" are helpers: excluded from the ID
pass and point distances, but rendered in RGB (e.g. the photo-textured mat "_env.mat").
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix
from mathutils.bvhtree import BVHTree

ROOT = Path(__file__).resolve().parents[2]


def parse():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--build", default=None)
    ap.add_argument("--views", default="probe")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--passes", default="id,rgb,pts")
    ap.add_argument("--samples", type=int, default=16)
    return ap.parse_args(argv)


def srgb8(c: float) -> int:
    c = max(0.0, min(1.0, c))
    s = 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055
    return int(round(255 * s))


def palette(n: int) -> list[tuple[float, float, float]]:
    """n well-separated linear colors whose 8-bit sRGB encodings are unique (golden-ratio hue walk)."""
    import colorsys
    out, seen, i = [], set(), 0
    while len(out) < n:
        h = (i * 0.61803398875) % 1.0
        s = 0.55 + 0.45 * ((i * 7) % 5) / 4
        v = 0.45 + 0.55 * ((i * 3) % 4) / 3
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        key = (srgb8(r), srgb8(g), srgb8(b))
        i += 1
        if key in seen or key == (0, 0, 0):
            continue
        seen.add(key)
        out.append((r, g, b))
    return out


def model_objects():
    return [o for o in bpy.context.scene.objects
            if o.type in {"MESH", "CURVE", "SURFACE", "META", "FONT"} and not o.name.startswith("_")
            and not o.hide_render]


def make_camera(scene):
    cam_data = bpy.data.cameras.new("_eval_cam")
    cam = bpy.data.objects.new("_eval_cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    cam_data.clip_start = 0.005
    cam_data.clip_end = 20.0
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width = 36.0
    return cam


def set_view(scene, cam, v: dict):
    W, H = v["width"], v["height"]
    K = np.array(v["K"])
    R = np.array(v["R"])
    t = np.array(v["t"])
    scene.render.resolution_x, scene.render.resolution_y = W, H
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1.0
    cd = cam.data
    cd.sensor_fit = "HORIZONTAL" if W >= H else "VERTICAL"
    big = max(W, H)
    if W >= H:
        cd.lens = K[0, 0] * cd.sensor_width / W
    else:
        cd.sensor_height = 36.0
        cd.lens = K[1, 1] * cd.sensor_height / H
    cd.shift_x = -(K[0, 2] - (W - 1) / 2) / big
    cd.shift_y = (K[1, 2] - (H - 1) / 2) / big
    c2w = R.T
    C = -R.T @ t
    M = np.eye(4)
    M[:3, :3] = c2w @ np.diag([1.0, -1.0, -1.0])
    M[:3, 3] = C
    cam.matrix_world = Matrix(M.tolist())


def setup_world(scene):
    if scene.world is None:
        scene.world = bpy.data.worlds.new("_eval_world")
    w = scene.world
    w.use_nodes = True
    nt = w.node_tree
    bg = nt.nodes.get("Background")
    # Uniform white environment of strength 1 for diffuse light: a diffuse surface textured with photo colors
    # renders at (about) the photo's own values. Glossy rays see a dim environment instead (the real room seen in
    # reflections is mostly dark), so glossy black plastic reads black, not gray. Env var EVAL_GLOSSY_ENV sets it.
    import os
    bg.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
    bg.inputs[1].default_value = 1.0
    if "_glossy_mix" not in nt.nodes:
        out = nt.nodes.get("World Output")
        lp = nt.nodes.new("ShaderNodeLightPath")
        dim = nt.nodes.new("ShaderNodeBackground")
        g = float(os.environ.get("EVAL_GLOSSY_ENV", "1.0"))  # 0.12 tested 2026-10-05: EEVEE darkens all world light
        dim.inputs[0].default_value = (g, g, g, 1.0)
        mix = nt.nodes.new("ShaderNodeMixShader")
        mix.name = "_glossy_mix"
        nt.links.new(lp.outputs["Is Glossy Ray"], mix.inputs[0])
        nt.links.new(bg.outputs[0], mix.inputs[1])
        nt.links.new(dim.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], out.inputs["Surface"])
    if "_eval_key" not in bpy.data.objects:
        ld = bpy.data.lights.new("_eval_key", "AREA")
        ld.energy = 0.0  # calibration: world only (a key light brightened textured surfaces 1.4-1.9x)
        ld.size = 0.6
        lo = bpy.data.objects.new("_eval_key", ld)
        lo.location = (0.0, -0.1, 0.6)
        scene.collection.objects.link(lo)


def helper_geometry(scene):
    """Renderable helper objects (names starting "_", e.g. the textured mat "_env.mat"): visible in RGB only."""
    return [o for o in scene.objects if o.type in {"MESH", "CURVE", "SURFACE", "META", "FONT"}
            and o.name.startswith("_") and not o.name.startswith("_eval")]


def render_id(scene, cam, views, sel, out: Path, objs):
    hidden = [o for o in helper_geometry(scene) if not o.hide_render]
    for o in hidden:
        o.hide_render = True
    cols = palette(len(objs))
    id_map = {}
    for o, c in zip(objs, cols):
        o.color = (*c, 1.0)
        id_map[o.name] = [srgb8(c[0]), srgb8(c[1]), srgb8(c[2])]
    (out / "id_map.json").write_text(json.dumps(id_map, indent=0))
    scene.render.engine = "BLENDER_WORKBENCH"
    sh = scene.display.shading
    sh.light = "FLAT"
    sh.color_type = "OBJECT"
    sh.show_shadows = False
    sh.show_cavity = False
    sh.show_object_outline = False
    sh.show_specular_highlight = False
    scene.display.render_aa = "OFF"
    scene.render.film_transparent = True
    scene.render.dither_intensity = 0.0
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    (out / "id").mkdir(parents=True, exist_ok=True)
    for name in sel:
        set_view(scene, cam, views[name])
        scene.render.filepath = str(out / "id" / f"{name}.png")
        bpy.ops.render.render(write_still=True)
    for o in hidden:
        o.hide_render = False


def render_rgb(scene, cam, views, sel, out: Path, samples: int):
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.eevee.taa_render_samples = samples
    import os
    if os.environ.get("EVAL_RAYTRACE", "1") == "1":  # default on (+0.46 dB holdout, 2026-10-05); screen-space reflections/refraction (glossy walls mirror the mat)
        scene.eevee.use_raytracing = True
        scene.eevee.ray_tracing_method = "SCREEN"
    scene.render.film_transparent = True
    scene.render.dither_intensity = 0.0
    scene.view_settings.view_transform = "Standard"
    setup_world(scene)
    (out / "rgb").mkdir(parents=True, exist_ok=True)
    for name in sel:
        set_view(scene, cam, views[name])
        scene.render.filepath = str(out / "rgb" / f"{name}.png")
        bpy.ops.render.render(write_still=True)


def point_distances(out: Path, objs):
    X = np.load(ROOT / "data" / "points_world.npy")
    dg = bpy.context.evaluated_depsgraph_get()
    verts, polys, poly_obj, names = [], [], [], []
    for o in objs:
        oe = o.evaluated_get(dg)
        try:
            me = oe.to_mesh()
        except RuntimeError:
            continue
        if me is None or len(me.polygons) == 0:
            oe.to_mesh_clear()
            continue
        mw = oe.matrix_world
        base = len(verts)
        verts.extend(mw @ v.co for v in me.vertices)
        polys.extend(tuple(base + i for i in p.vertices) for p in me.polygons)
        poly_obj.extend([len(names)] * len(me.polygons))
        names.append(o.name)
        oe.to_mesh_clear()
    dist = np.full(len(X), np.inf)
    idx = np.full(len(X), -1)
    if polys:
        tree = BVHTree.FromPolygons(verts, polys)
        poly_obj = np.array(poly_obj)
        for i, p in enumerate(X):
            hit = tree.find_nearest(p, 0.05)
            if hit[0] is not None:
                dist[i] = hit[3]
                idx[i] = poly_obj[hit[2]]
    np.savez(out / "points.npz", dist_m=dist, obj_index=idx, names=np.array(names))


def main():
    a = parse()
    t0 = time.time()
    scene = bpy.context.scene
    if a.build:
        ns = {"__name__": "__build__", "__file__": str(Path(a.build).resolve())}
        exec(compile(Path(a.build).read_text(), a.build, "exec"), ns)
    t_build = time.time() - t0
    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    cams = json.loads((ROOT / "data" / f"cameras_s{a.scale}.json").read_text())["views"]
    split = json.loads((ROOT / "data" / "split.json").read_text())
    sel = split[a.views] if a.views in split else (list(cams) if a.views == "all" else a.views.split(","))
    if set(sel) & set(split["holdout"]) and os.environ.get("EVAL_ALLOW_HOLDOUT") != "1":
        # holdout renders only at milestones (scripts/final_eval.sh sets EVAL_ALLOW_HOLDOUT=1); audit DevLog-005
        raise SystemExit("render_views: holdout views are rendered only by scripts/final_eval.sh milestones")
    objs = model_objects()
    cam = make_camera(scene)
    passes = a.passes.split(",")
    timing = {"build_s": round(t_build, 2), "n_objects": len(objs), "n_views": len(sel)}
    if "id" in passes:
        t = time.time()
        render_id(scene, cam, cams, sel, out, objs)
        timing["id_s"] = round(time.time() - t, 2)
    if "rgb" in passes:
        t = time.time()
        render_rgb(scene, cam, cams, sel, out, a.samples)
        timing["rgb_s"] = round(time.time() - t, 2)
    if "pts" in passes:
        t = time.time()
        point_distances(out, objs)
        timing["pts_s"] = round(time.time() - t, 2)
    meta = {"views": sel, "scale": a.scale, "passes": passes, "timing": timing,
            "build": a.build, "blend": bpy.data.filepath or None}
    (out / "render_meta.json").write_text(json.dumps(meta, indent=1))
    print("RENDER_DONE", json.dumps(timing))


if __name__ == "__main__":
    main()
