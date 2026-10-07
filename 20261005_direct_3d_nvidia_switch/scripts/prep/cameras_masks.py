"""Blender camera JSON and 3-level evaluation masks (run after world_frame.py, split.py and masks_sam.py).

Outputs (data/):
  cameras_s{N}.json     per view: K (index-pixel convention), R, t (world-to-camera, meters), size
  masks_s4/<stem>.png   255 = object (tray or package, union of the SAM masks), 0 = background,
                        128 = unknown (ignored by silhouette metrics): holes fully enclosed by the object
                        (dark interior seen through gaps is more likely object than background), background
                        inside the projected hull of each hull_unknown box (only the outer outline is scored;
                        SAM drops dark cables and black parts inside), object pixels outside the projected
                        outside_unknown hulls (SAM spill onto the plinth/hardware), SAM masks of occluder objects (sign; the package stand, black on black),
                        a band of
                        --band px around every object boundary (SAM edge noise), and regions listed in
                        config/mask_prompts.yaml views.<stem>.unknown_px (polygons, 1/2-scale px: out-of-scope
                        occluders such as the acrylic signs and the second package)
Views without any SAM mask (or excluded views) get no mask file. Skips nothing: masks are cheap, always rewritten.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CONFIG, DATA, scene_cfg  # noqa: E402
from tools.geom import load_views  # noqa: E402

OBJECTS = ("tray", "package", "uqd_px", "uqd_nx")  # uqd_*: rear fittings SAM drops from the tray mask


def frame_M(frames: dict, name: str) -> np.ndarray:
    M = np.eye(4)
    if name != "world":
        M[:3, :3] = np.array(frames[name]["R"], float)
        M[:3, 3] = np.array(frames[name]["t_mm"], float)
    return M


def hull_px(v, box_mm, M) -> np.ndarray | None:
    x0, x1, y0, y1, z0, z1 = box_mm
    C = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], float)
    C = (M[:3, :3] @ C.T).T + M[:3, 3]
    uv, z = v.project(C / 1e3)
    if (z <= 0).any():
        return None
    return cv2.convexHull(uv.astype(np.float32)).astype(np.int32)


def main() -> None:
    band = 2
    if "--band" in sys.argv:
        band = int(sys.argv[sys.argv.index("--band") + 1])
    cfg = scene_cfg()
    for s in cfg["work_scales"]:
        V = load_views(s)
        cams = {n: {"K": v.K.tolist(), "R": v.R.tolist(), "t": v.t.tolist(), "width": v.width, "height": v.height,
                    "camera_id": v.camera_id, "image": str(v.path.relative_to(DATA.parent))} for n, v in V.items()}
        (DATA / f"cameras_s{s}.json").write_text(json.dumps({"scale": s, "views": cams}))
    prompts = yaml.safe_load((CONFIG / "mask_prompts.yaml").read_text())
    frames = json.loads((CONFIG / "frames.json").read_text())["frames"]
    V4 = load_views(4)
    out = DATA / "masks_s4"
    out.mkdir(exist_ok=True)
    n_written = 0
    for n, v in V4.items():
        m = np.zeros((v.height, v.width), np.uint8)
        any_obj = False
        for o in OBJECTS:
            p = DATA / "sam_s2" / o / f"{n}.png"
            if p.exists():
                mo = cv2.resize(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), (v.width, v.height),
                                interpolation=cv2.INTER_AREA) > 127
                m |= mo.astype(np.uint8)
                any_obj = True
        if not any_obj:
            continue
        res = np.where(m > 0, 255, 0).astype(np.uint8)
        for hname, hb in (prompts.get("hull_unknown") or {}).items():
            hp = hull_px(v, hb["box_mm"], frame_M(frames, hb["frame"]))
            if hp is not None:
                inside = np.zeros_like(m)
                cv2.fillConvexPoly(inside, hp, 1)
                res[(inside > 0) & (res == 0)] = 128
        # enclosed holes: background components not touching the image border
        k, lab, stats, _ = cv2.connectedComponentsWithStats((m == 0).astype(np.uint8), 4)
        h, w = m.shape
        for i in range(1, k):
            x, y, ww, hh, _ = stats[i]
            if x > 0 and y > 0 and x + ww < w and y + hh < h:
                res[lab == i] = 128
        if band > 0:
            edge = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
            edge = cv2.dilate(edge.astype(np.uint8), np.ones((2 * band - 1, 2 * band - 1), np.uint8)) > 0
            res[edge] = 128
        ou = prompts.get("outside_unknown")
        if ou:
            allowed = np.zeros_like(m)
            for hb in ou["boxes"].values():
                hp = hull_px(v, hb["box_mm"], frame_M(frames, hb["frame"]))
                if hp is not None:
                    cv2.fillConvexPoly(allowed, hp, 1)
            k_ = 2 * ou.get("margin_px", 6) + 1
            allowed = cv2.dilate(allowed, np.ones((k_, k_), np.uint8)) > 0
            res[(res == 255) & ~allowed] = 128
        for on, ob in prompts["objects"].items():
            p = DATA / "sam_s2" / on / f"{n}.png"
            if ob.get("occluder") and p.exists():
                mo = cv2.resize(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), (v.width, v.height),
                                interpolation=cv2.INTER_AREA) > 127
                res[cv2.dilate(mo.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0] = 128
        for poly in ((prompts.get("views") or {}).get(n, {}) or {}).get("unknown_px", []):
            cv2.fillPoly(res, [np.round(np.array(poly, float) / 2).astype(np.int32)], 128)
        cv2.imwrite(str(out / f"{n}.png"), res)
        n_written += 1
    print("cameras written for scales", cfg["work_scales"], "; masks written:", n_written)


if __name__ == "__main__":
    main()
