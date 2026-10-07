"""Fit a group's parameters to the photos by rendering inside one Blender process (pattern search).

Usage (through the slot wrapper):
  scripts/bslot.sh -b --factory-startup --python scripts/blender/fit_params.py -- \
      --group case --params body.length:1,body.width:1,screen.height:0.5 [--views probe] \
      [--targets case.] [--res 50] [--max-evals 120] [--w-fp 10] [--w-fn 0] [--out outputs/fits/<name>]

--params   comma list of dotted keys in config/model/<group>.toml with initial step sizes (key:step, mm/deg).
--targets  object-name prefixes whose pixels/edges enter the loss (default "<group>.").
Loss per view (mean over views):
  edge = mean over target edge pixels (silhouette boundary of each target object + shading creases inside
         it, Workbench studio lighting) of the distance to the nearest photo edge (px at this resolution,
         clipped at 10)
  fp   = fraction of target pixels where the photo mask says mat (excess geometry)
  fn   = photo-object pixels within 12 px of the target that no model object covers, over target pixels
         (pulls geometry outward; keep 0 unless neighbors are modeled)
  loss = edge + w_fp * fp + w_fn * fn
All other groups are built once and stay in the scene (occluders). Writes <out>/fit.jsonl (every evaluation)
and <out>/best.json; it never edits the config. Copy accepted values into the TOML by hand.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "model"))
sys.path.insert(0, str(ROOT / "scripts" / "blender"))
import build_all  # noqa: E402
import lib  # noqa: E402


def parse():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--views", default="probe")
    ap.add_argument("--targets", default=None)
    ap.add_argument("--res", type=int, default=50)
    ap.add_argument("--max-evals", type=int, default=120)
    ap.add_argument("--min-step-frac", type=float, default=0.05)
    ap.add_argument("--w-fp", type=float, default=10.0)
    ap.add_argument("--w-fn", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    return ap.parse_args(argv)


def get_key(P, k):
    for kk in k.split("."):
        P = P[kk]
    return P


def setup_render(scene, res_pct):
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.render_aa = "OFF"
    scene.render.film_transparent = True
    scene.render.dither_intensity = 0.0
    scene.view_settings.view_transform = "Standard"
    scene.render.resolution_percentage = res_pct
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    sh = scene.display.shading
    sh.color_type = "OBJECT"
    sh.show_shadows = False
    sh.show_cavity = False
    sh.show_object_outline = False
    sh.show_specular_highlight = False


def read_png(path: Path) -> np.ndarray:
    img = bpy.data.images.load(str(path))
    img.colorspace_settings.name = "Non-Color"
    w, h = img.size
    a = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(a)
    bpy.data.images.remove(img)
    return a.reshape(h, w, 4)[::-1]


def main():
    a = parse()
    t0 = time.time()
    scene = bpy.context.scene
    build_all.build_all()
    targets = tuple((a.targets or f"{a.group}.").split(","))
    P0 = lib.load_params(a.group)
    keys, steps = [], []
    for item in a.params.split(","):
        k, s = item.split(":")
        keys.append(k)
        steps.append(float(s))
    x0 = np.array([float(get_key(P0, k)) for k in keys])
    steps = np.array(steps)

    import render_views as rv  # camera helpers (module runs main() only via its own entry)
    cams = json.loads((ROOT / "data" / "cameras_s4.json").read_text())["views"]
    split = json.loads((ROOT / "data" / "split.json").read_text())
    sel = split[a.views] if a.views in split else a.views.split(",")
    if set(sel) & set(split["holdout"]):  # holdout views are for evaluation only (audit 2026-10-05)
        raise SystemExit("fit_params: holdout views are not allowed for fitting; use probe/train views")
    f = 100 // a.res
    edt = {v: np.load(ROOT / "data" / "edt_s4" / f"{v}.npy")[::f, ::f].astype(np.float32) / 10.0 / f for v in sel}
    masks = {}
    for v in sel:
        m = read_png(ROOT / "data" / "masks_s4" / f"{v}.png")[..., 0]
        masks[v] = np.round(m[::f, ::f] * 255).astype(np.uint8)
    cam = rv.make_camera(scene)
    setup_render(scene, a.res)
    tmp = Path(bpy.app.tempdir) / "fit"
    tmp.mkdir(parents=True, exist_ok=True)
    out = Path(a.out) if a.out else ROOT / "outputs" / "fits" / f"{a.group}_{time.strftime('%Y%m%d_%H%M%S')}"
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "fit.jsonl", "w")

    def evaluate(x):
        build_all.build_group(a.group, {k: float(v) for k, v in zip(keys, x)})
        objs = [o for o in scene.objects if o.type in {"MESH", "CURVE"} and not o.name.startswith("_")]
        for o in scene.objects:
            if o.type in {"MESH", "CURVE"} and o.name.startswith("_") and not o.name.startswith("_eval"):
                o.hide_render = True
        tobj = [o for o in objs if o.name.startswith(targets)]
        if not tobj:
            return 1e9, {}
        for i, o in enumerate(objs):
            o.color = (0, 0, 0, 1)
        for i, o in enumerate(tobj):
            o.color = (((i * 37) % 200 + 40) / 255, ((i * 91) % 200 + 40) / 255, 0.8, 1)
        tot = {"edge": 0.0, "fp": 0.0, "fn": 0.0}
        for v in sel:
            rv.set_view(scene, cam, cams[v])
            scene.render.resolution_percentage = a.res
            scene.display.shading.light = "FLAT"
            scene.render.filepath = str(tmp / "id.png")
            bpy.ops.render.render(write_still=True)
            idm = read_png(tmp / "id.png")
            scene.display.shading.light = "STUDIO"
            scene.render.filepath = str(tmp / "sh.png")
            bpy.ops.render.render(write_still=True)
            sh = read_png(tmp / "sh.png")
            alpha = idm[..., 3] > 0.5
            key = (idm[..., 0] * 255).round().astype(np.int32) * 256 + (idm[..., 1] * 255).round().astype(np.int32)
            tgt = alpha & (idm[..., 2] > 0.5)
            key = np.where(alpha, key, -1)
            e = np.zeros_like(tgt)
            e[:, 1:] |= (key[:, 1:] != key[:, :-1]) & (tgt[:, 1:] | tgt[:, :-1])
            e[1:, :] |= (key[1:, :] != key[:-1, :]) & (tgt[1:, :] | tgt[:-1, :])
            g = sh[..., :3].mean(2)
            gx = np.zeros_like(g)
            gy = np.zeros_like(g)
            gx[:, 1:] = np.abs(g[:, 1:] - g[:, :-1])
            gy[1:, :] = np.abs(g[1:, :] - g[:-1, :])
            e |= ((gx > 0.06) | (gy > 0.06)) & tgt
            hh, ww = min(tgt.shape[0], edt[v].shape[0]), min(tgt.shape[1], edt[v].shape[1])
            d, m = edt[v][:hh, :ww], masks[v][:hh, :ww]
            tgt, e, alpha = tgt[:hh, :ww], e[:hh, :ww], alpha[:hh, :ww]
            nt = max(int(tgt.sum()), 1)
            tot["edge"] += float(np.minimum(d[e], 10.0).mean()) if e.any() else 10.0
            tot["fp"] += float((tgt & (m == 0)).sum()) / nt
            if a.w_fn > 0:
                near = tgt.copy()
                for _ in range(12):
                    near[1:, :] |= near[:-1, :]
                    near[:-1, :] |= near[1:, :]
                    near[:, 1:] |= near[:, :-1]
                    near[:, :-1] |= near[:, 1:]
                tot["fn"] += float((near & ~alpha & (m == 255)).sum()) / nt
        n = len(sel)
        comp = {k: v / n for k, v in tot.items()}
        loss = comp["edge"] + a.w_fp * comp["fp"] + a.w_fn * comp["fn"]
        return loss, comp

    nev = 0
    best_x = x0.copy()
    best_l, best_c = evaluate(best_x)
    nev += 1
    log.write(json.dumps({"eval": nev, "x": best_x.tolist(), "loss": best_l, **best_c}) + "\n")
    print(f"FIT start loss {best_l:.4f} {best_c}")
    step = steps.copy()
    while nev < a.max_evals and np.any(step > steps * a.min_step_frac):
        improved = False
        for i in range(len(keys)):
            if step[i] <= steps[i] * a.min_step_frac:
                continue
            for sgn in (+1, -1):
                x = best_x.copy()
                x[i] += sgn * step[i]
                loss, comp = evaluate(x)
                nev += 1
                log.write(json.dumps({"eval": nev, "x": x.tolist(), "loss": loss, **comp}) + "\n")
                log.flush()
                if loss < best_l - 1e-6:
                    best_l, best_c, best_x = loss, comp, x
                    improved = True
                    break
                if nev >= a.max_evals:
                    break
            if nev >= a.max_evals:
                break
        if not improved:
            step = step / 2
    res = {"group": a.group, "views": sel, "res_pct": a.res, "evals": nev, "seconds": round(time.time() - t0, 1),
           "loss": best_l, "components": best_c,
           "params": {k: round(float(v), 3) for k, v in zip(keys, best_x)},
           "initial": {k: float(v) for k, v in zip(keys, x0)}}
    (out / "best.json").write_text(json.dumps(res, indent=1))
    print("FIT_DONE", json.dumps(res))


main()
