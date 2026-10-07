"""Per-object photo masks with SAM 2.1 (box + point prompts projected from 3D hypotheses), one PNG per view.

  uv run python scripts/prep/masks_sam.py [--objects tray,package] [--views all|train|IMG_5713,...] [--force]

Prompts come from config/mask_prompts.yaml:
  objects.<name>.box_mm      world-mm box [x0, x1, y0, y1, z0, z1] in the frame named by objects.<name>.frame
                             (world, or a frame in config/frames.json); its 8 projected corners give the 2D
                             box prompt (clipped to the image)
  objects.<name>.points_mm   positive points (same frame), projected; points outside the image or behind the
                             camera are skipped
  objects.<name>.views_only  optional list: the object is segmented only in these views (skip elsewhere)
  objects.<name>.skip_views  optional list: views where the segmentation was reviewed as wrong
  views.<stem>.<name>        optional per-view overrides: box_px / pos_px / neg_px in 1/2-scale pixel coords,
                             or skip: true (object not visible in that view)
Model: SAM 2.1 hiera base_plus, source and checkpoint used in place from ~/Documents/GitHub/sam2 (read-only;
nothing is installed or written there). Device: MPS if available, else CPU. Input: 1/2-scale undistorted
image. Output: data/sam_s2/<name>/<stem>.png (255 = object) and data/sam_s2/<name>/scores.json; skips
existing files unless --force. Sheet for visual checks: outputs/masks/sam_<name>.jpg.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CONFIG, DATA, OUTPUTS  # noqa: E402
from tools.geom import load_views  # noqa: E402

SAM_ROOT = Path("~/Documents/GitHub/sam2").expanduser()
SAM_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
SAM_CKPT = SAM_ROOT / "checkpoints" / "sam2.1_hiera_base_plus.pt"


def frame_matrix(name: str) -> np.ndarray:
    """4x4 transform from the named frame to world (mm); identity for 'world'."""
    if name == "world":
        return np.eye(4)
    fr = json.loads((CONFIG / "frames.json").read_text())["frames"][name]
    M = np.eye(4)
    M[:3, :3] = np.array(fr["R"], float)
    M[:3, 3] = np.array(fr["t_mm"], float)
    return M


def to_world(P: np.ndarray, M: np.ndarray) -> np.ndarray:
    return (M[:3, :3] @ np.atleast_2d(P).T).T + M[:3, 3]


def box_corners(b) -> np.ndarray:
    x0, x1, y0, y1, z0, z1 = b
    return np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], float)


def load_predictor():
    import torch
    sys.path.insert(0, str(SAM_ROOT))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = build_sam2(SAM_CFG, str(SAM_CKPT), device=dev)
    return SAM2ImagePredictor(model), dev


def prompts_for(view, obj: dict, override: dict | None):
    M = frame_matrix(obj.get("frame", "world"))
    W, H = view.width, view.height
    if override and "box_px" in override:
        box = np.array(override["box_px"], float)
    else:
        uv, z = view.project(to_world(box_corners(obj["box_mm"]), M) / 1e3)
        if (z <= 0).any():
            return None
        box = np.array([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()])
        box = np.clip(box, 0, [W - 1, H - 1, W - 1, H - 1])
        if box[2] - box[0] < 20 or box[3] - box[1] < 20:
            return None
    pos, neg = [], []
    if override and "pos_px" in override:
        pos = [list(p) for p in override["pos_px"]]
    elif obj.get("points_mm"):
        uv, z = view.project(to_world(np.array(obj["points_mm"], float), M) / 1e3)
        pos = [list(p) for p, zz in zip(uv, z) if zz > 0 and 0 <= p[0] < W and 0 <= p[1] < H]
    if override and "neg_px" in override:
        neg = [list(p) for p in override["neg_px"]]
    return box, pos, neg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", default=None)
    ap.add_argument("--views", default="all")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load((CONFIG / "mask_prompts.yaml").read_text())
    objs = a.objects.split(",") if a.objects else list(cfg["objects"])
    V = load_views(2)
    split = json.loads((DATA / "split.json").read_text())
    names = list(V) if a.views == "all" else (split[a.views] if a.views in split else a.views.split(","))
    pred, dev = load_predictor()
    print("SAM 2.1 on", dev)
    for on in objs:
        obj = cfg["objects"][on]
        out = DATA / "sam_s2" / on
        out.mkdir(parents=True, exist_ok=True)
        sp = out / "scores.json"
        scores = json.loads(sp.read_text()) if sp.exists() else {}
        for n in names:
            p = out / f"{n}.png"
            ov = (cfg.get("views") or {}).get(n, {}).get(on)
            if ("views_only" in obj and n not in obj["views_only"]) or n in obj.get("skip_views", []):
                ov = {"skip": True}
            if ov and ov.get("skip"):
                if p.exists():
                    p.unlink()
                scores[n] = None
                continue
            if p.exists() and not a.force:
                continue
            v = V[n]
            pr = prompts_for(v, obj, ov)
            if pr is None:
                scores[n] = None
                continue
            box, pos, neg = pr
            img = cv2.cvtColor(v.image(), cv2.COLOR_BGR2RGB)
            pred.set_image(img)
            pts = pos + neg
            kw = {}
            if pts:
                kw = {"point_coords": np.array(pts, np.float32),
                      "point_labels": np.array([1] * len(pos) + [0] * len(neg), np.int32)}
            masks, sc, _ = pred.predict(box=box[None].astype(np.float32), multimask_output=False, **kw)
            m = (masks[0] > 0).astype(np.uint8) * 255
            cv2.imwrite(str(p), m)
            scores[n] = {"score": float(sc[0]), "box": box.round(1).tolist(), "n_pos": len(pos), "n_neg": len(neg),
                         "frac": float((m > 0).mean())}
            print(on, n, round(float(sc[0]), 3), round(float((m > 0).mean()), 3))
        sp.write_text(json.dumps(scores, indent=1))
        sheet(on, names, V, scores)


def sheet(on: str, names: list[str], V, scores: dict, w: int = 357) -> None:
    tiles = []
    for n in names:
        p = DATA / "sam_s2" / on / f"{n}.png"
        img = V[n].image()
        if p.exists():
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 0
            tint = img.copy()
            tint[m] = (0.5 * tint[m] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
            cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(tint, cnt, -1, (0, 255, 255), 3)
            s = scores.get(n)
            if s:
                b = np.round(s["box"]).astype(int)
                cv2.rectangle(tint, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 3)
            img = tint
        t = cv2.resize(img, (w, int(w * img.shape[0] / img.shape[1])))
        cv2.putText(t, n[4:], (5, 22), 0, 0.7, (0, 255, 255), 2)
        tiles.append(t)
    cols = 6
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    (OUTPUTS / "masks").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUTS / "masks" / f"sam_{on}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    main()
