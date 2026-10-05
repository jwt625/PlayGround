"""Evaluate a render run against the photos and steer visual inspection.

Usage: uv run python scripts/eval/evaluate.py outputs/runs/<run> [--top 8] [--parts crt.] [--mode geom|full]
  --parts  restrict inspection regions to tiles dominated by objects with these name prefixes (comma list),
           plus unmodeled tiles touching them; metrics tables still cover everything.
  --mode   geom: region ranking uses silhouette + edges only (use while blocking out shapes);
           full (default): adds color/structure residual (use once materials/textures exist).

Reads <run>/{id,rgb}/<view>.png, id_map.json, points.npz, render_meta.json; data/undistorted_s4, masks_s4.
Writes <run>/eval/:
  summary.json     global metrics
  parts.json       per-object metrics (sorted worst first) -> feedback to the owner of each part
  views.json       per-view metrics
  orphans.json     clusters of sparse points far from any model surface (missing geometry), world coords
  regions.json     top-K error regions (view, bbox, dominant part, error mix)
  regions.png      inspection sheet: per region, photo | render | edge overlay | error heat (from 1/2-scale photo)
  views.png        thumbnail sheet: per view, photo with model outline + error heat
  report.md        human/agent-readable summary of all of the above

Metrics (all at 1/4 scale, 1428x1071):
  sil_iou          silhouette IoU inside the known region (mask != 128), ignoring a 2 px band at the photo
                   mask boundary (mask noise). fp = model where the photo shows mat, fn = object where the
                   model shows nothing (attributed to the nearest model part if within 20 px, else "unmodeled").
  edge_chamfer     for model edge pixels (ID boundaries + creases in the RGB render), distance (px) to the
                   nearest photo edge (Canny), truncated at 10 px; reported as mean and fraction <= 2 px.
  color_res        mean abs residual (0-255) of photo vs per-view affine-color-fitted render inside model and
                   photo masks; ssim = structural similarity of gray images in the same region.
  per-part pts     all points whose nearest surface (within 50 mm) is that part: untruncated median and the
                   fraction beyond 4 mm (a part can look tight on its near points while missing others)
  pts              sparse points on the object (z > 2.5 mm, inside the object box; outside the case footprint
                   mat-blue points are dropped as mat relief): distance to the model
                   surface; per part: n, median mm, frac <= 1 mm; orphans: points > 4 mm from any surface.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA  # noqa: E402

OBJ_BOX = ((-0.24, 0.24), (-0.20, 0.20), (0.0025, 0.12))  # z > 2.5 mm: skip mat relief points  # world meters: region containing the CRT unit
TILE = 48


def load_run(run: Path):
    meta = json.loads((run / "render_meta.json").read_text())
    id_map = json.loads((run / "id_map.json").read_text())
    names = list(id_map)
    keys = np.array([(c[0] << 16) | (c[1] << 8) | c[2] for c in id_map.values()])
    return meta, names, keys


def decode_id(path: Path, keys: np.ndarray, tol: int = 6) -> np.ndarray:
    """Map rendered ID colors to object indices: nearest palette color within an L1 tolerance (8-bit encoding of
    Workbench flat colors can be off by one level), else -1."""
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    b, g, r, a = (im[..., i].astype(np.int64) for i in range(4))
    k = (r << 16) | (g << 8) | b
    out = np.full(k.shape, -1, np.int32)
    pal = np.stack([(keys >> 16) & 255, (keys >> 8) & 255, keys & 255], 1)
    uniq, inv = np.unique(np.where(a > 0, k, -1), return_inverse=True)
    lut = np.full(len(uniq), -1, np.int32)
    for j, kk in enumerate(uniq):
        if kk < 0:
            continue
        c = np.array([(kk >> 16) & 255, (kk >> 8) & 255, kk & 255])
        d = np.abs(pal - c).sum(1)
        i = int(np.argmin(d))
        if d[i] <= tol:
            lut[j] = i
    out = lut[inv.reshape(k.shape)]
    return out


def photo_edges(gray: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(gray, (0, 0), 1.0)
    return cv2.Canny(g, 40, 100) > 0


def model_edges(idm: np.ndarray, rgb: np.ndarray | None) -> np.ndarray:
    e = np.zeros(idm.shape, bool)
    e[:, 1:] |= idm[:, 1:] != idm[:, :-1]
    e[1:, :] |= idm[1:, :] != idm[:-1, :]
    if rgb is not None:
        g = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        c = cv2.Canny(cv2.GaussianBlur(g, (0, 0), 1.0), 30, 80) > 0
        e |= c & (idm >= 0)
    return e


def ssim_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a.astype(np.float32), b.astype(np.float32)
    k = (0, 0)
    mu_a, mu_b = cv2.GaussianBlur(a, k, 2), cv2.GaussianBlur(b, k, 2)
    va = cv2.GaussianBlur(a * a, k, 2) - mu_a ** 2
    vb = cv2.GaussianBlur(b * b, k, 2) - mu_b ** 2
    cov = cv2.GaussianBlur(a * b, k, 2) - mu_a * mu_b
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))


def eval_view(run: Path, name: str, keys, n_parts: int, mode: str = "full"):
    photo = cv2.imread(str(DATA / "undistorted_s4" / f"{name}.jpg"), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    pm = cv2.imread(str(DATA / "masks_s4" / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
    idm = decode_id(run / "id" / f"{name}.png", keys)
    rgb_path = run / "rgb" / f"{name}.png"
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_UNCHANGED) if rgb_path.exists() else None
    H, W = idm.shape
    mr = idm >= 0
    po = pm == 255
    known = pm != 128
    band = cv2.dilate(cv2.morphologyEx(po.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)),
                      np.ones((3, 3), np.uint8)) > 0
    valid = known & ~band
    fp = mr & ~po & valid
    fn = po & ~mr & valid
    inter = (mr & po & valid).sum()
    union = ((mr | po) & valid).sum()
    iou = inter / max(union, 1)
    iou_noband = (mr & po & known).sum() / max(((mr | po) & known).sum(), 1)
    unscored = float((mr & ~known).sum() / max(mr.sum(), 1))  # model pixels outside the known region
    # attribute fn to nearest model part
    fn_part = np.full(fn.shape, -1, np.int32)
    if mr.any():
        dist, lab = cv2.distanceTransformWithLabels((~mr).astype(np.uint8), cv2.DIST_L2, 5,
                                                    labelType=cv2.DIST_LABEL_PIXEL)
        ys, xs = np.nonzero(mr)
        lab_to_part = np.full(lab.max() + 1, -1, np.int32)
        lab_to_part[lab[ys, xs]] = idm[ys, xs]
        near = lab_to_part[lab]
        fn_part = np.where(fn & (dist <= 20), near, -1)
    # edges
    gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    pe = photo_edges(gray)
    me = model_edges(idm, rgb[..., :3] if rgb is not None else None)
    dpe = np.minimum(cv2.distanceTransform((~pe).astype(np.uint8), cv2.DIST_L2, 5), 10.0)
    me_valid = me & (cv2.dilate(mr.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    edge_d = np.where(me_valid, dpe, 0.0)
    # color / structure
    both = mr & po
    col_res = np.zeros((H, W), np.float32)
    ssim = np.ones((H, W), np.float32)
    color_ok = rgb is not None and both.sum() > 500
    if color_ok:
        R = rgb[..., :3].reshape(-1, 3)[both.ravel()].astype(np.float64)
        P = photo.reshape(-1, 3)[both.ravel()].astype(np.float64)
        A = np.hstack([R, np.ones((len(R), 1))])
        M = np.linalg.lstsq(A, P, rcond=None)[0]
        fit = np.clip(np.hstack([rgb[..., :3].reshape(-1, 3), np.ones((H * W, 1))]) @ M, 0, 255).reshape(H, W, 3)
        col_res = np.abs(fit - photo).mean(2).astype(np.float32) * both
        ssim = ssim_map(gray, cv2.cvtColor(fit.astype(np.uint8), cv2.COLOR_BGR2GRAY))
    per_part = []
    for k in range(n_parts):
        pk = idm == k
        npx = int(pk.sum())
        if npx == 0:
            per_part.append(None)
            continue
        ek = me_valid & pk
        per_part.append({
            "px": npx, "fp": int((fp & pk).sum()), "fn": int((fn_part == k).sum()),
            "edge_n": int(ek.sum()), "edge_mean": float(dpe[ek].mean()) if ek.any() else None,
            "edge_le2": float((dpe[ek] <= 2).mean()) if ek.any() else None,
            "color_res": float(col_res[pk & both].mean()) if color_ok and (pk & both).any() else None,
            "ssim": float(ssim[pk & both].mean()) if color_ok and (pk & both).any() else None,
        })
    v = {"view": name, "sil_iou": float(iou), "sil_iou_noband": float(iou_noband), "unscored_frac": unscored,
         "fp": int(fp.sum()), "fn": int(fn.sum()),
         "fn_unmodeled": int((fn & (fn_part < 0)).sum()),
         "edge_mean": float(dpe[me_valid].mean()) if me_valid.any() else None,
         "edge_le2": float((dpe[me_valid] <= 2).mean()) if me_valid.any() else None,
         "color_res": float(col_res[both].mean()) if color_ok else None,
         "ssim": float(ssim[both].mean()) if color_ok else None}
    # combined error map for region ranking (each term roughly 0..1)
    err = (fp | fn).astype(np.float32) * 1.0
    err += cv2.dilate((edge_d / 10.0).astype(np.float32), np.ones((3, 3), np.uint8)) * 0.6
    if color_ok and mode == "full":
        err += np.clip(col_res / 60.0, 0, 1) * 0.5 * both
        err += np.clip((1 - ssim) / 1.0, 0, 1) * 0.3 * both
    maps = {"photo": photo, "rgb": rgb, "idm": idm, "pe": pe, "me": me_valid, "fp": fp, "fn": fn,
            "fn_part": fn_part, "err": err, "pm": pm}
    return v, per_part, maps


def tiles(err: np.ndarray, idm: np.ndarray, fn_part: np.ndarray, view: str):
    H, W = err.shape
    out = []
    for y in range(0, H - TILE + 1, TILE // 2):
        for x in range(0, W - TILE + 1, TILE // 2):
            e = err[y:y + TILE, x:x + TILE]
            s = float(e.mean())
            if s <= 0.02:
                continue
            ids = idm[y:y + TILE, x:x + TILE]
            ids = np.concatenate([ids[ids >= 0], fn_part[y:y + TILE, x:x + TILE][fn_part[y:y + TILE, x:x + TILE] >= 0]])
            part = int(np.bincount(ids).argmax()) if ids.size else -1
            out.append({"view": view, "x": x, "y": y, "score": s, "part": part})
    return out


def nms(cands, k, max_per_view=2, max_per_part=2):
    cands = sorted(cands, key=lambda c: -c["score"])
    sel, per_view, per_part = [], {}, {}
    for c in cands:
        if per_view.get(c["view"], 0) >= max_per_view or per_part.get(c["part"], 0) >= max_per_part:
            continue
        if any(s["view"] == c["view"] and abs(s["x"] - c["x"]) < TILE * 2 and abs(s["y"] - c["y"]) < TILE * 2
               for s in sel):
            continue
        sel.append(c)
        per_view[c["view"]] = per_view.get(c["view"], 0) + 1
        per_part[c["part"]] = per_part.get(c["part"], 0) + 1
        if len(sel) >= k:
            break
    return sel


def region_panel(run: Path, r: dict, maps: dict, names: list[str], crop_scale: int = 2, size: int = 300):
    """photo (1/2-scale crop) | render | photo edges green + model edges red | error heat."""
    x, y = r["x"], r["y"]
    pad = TILE  # show context around the tile
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + TILE + pad, maps["err"].shape[1]), min(y + TILE + pad, maps["err"].shape[0])
    hi = cv2.imread(str(DATA / f"undistorted_s{4 // crop_scale * 0 + 2}" / f"{r['view']}.jpg"),
                    cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    f = 4 // 2
    photo = hi[y0 * f:y1 * f, x0 * f:x1 * f]
    rgb = maps["rgb"]
    if rgb is not None:
        a = rgb[y0:y1, x0:x1, 3:4] / 255.0
        rend = (rgb[y0:y1, x0:x1, :3] * a + 40 * (1 - a)).astype(np.uint8)
    else:
        rend = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
    ov = maps["photo"][y0:y1, x0:x1].copy() // 2
    ov[maps["pe"][y0:y1, x0:x1]] = (0, 255, 0)
    ov[maps["me"][y0:y1, x0:x1]] = (0, 0, 255)
    ov[maps["pe"][y0:y1, x0:x1] & maps["me"][y0:y1, x0:x1]] = (0, 255, 255)
    e = maps["err"][y0:y1, x0:x1]
    heat = cv2.applyColorMap(np.clip(e / 1.5 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    heat[maps["fp"][y0:y1, x0:x1]] = (255, 0, 255)  # excess geometry: magenta
    heat[maps["fn"][y0:y1, x0:x1]] = (255, 255, 0)  # missing geometry: cyan
    tiles_ = [cv2.resize(t, (size, size * t.shape[0] // max(t.shape[1], 1)), interpolation=cv2.INTER_NEAREST
                         if t is not photo else cv2.INTER_AREA) for t in (photo, rend, ov, heat)]
    hmax = max(t.shape[0] for t in tiles_)
    tiles_ = [cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, 4, cv2.BORDER_CONSTANT, value=(30, 30, 30))
              for t in tiles_]
    row = np.hstack(tiles_)
    label = f"#{r['rank']} {r['view']} x{x0}-{x1} y{y0}-{y1} part={names[r['part']] if r['part'] >= 0 else 'unmodeled'}"
    lab = np.full((22, row.shape[1], 3), 20, np.uint8)
    cv2.putText(lab, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
    return np.vstack([lab, row])


def view_thumb(name, maps, v, w=356):
    ov = maps["photo"].copy()
    m = (maps["idm"] >= 0).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(ov, cs, -1, (0, 0, 255), 2)
    heat = cv2.applyColorMap(np.clip(maps["err"] / 1.5 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    heat[maps["fp"]] = (255, 0, 255)
    heat[maps["fn"]] = (255, 255, 0)
    h = w * ov.shape[0] // ov.shape[1]
    row = np.hstack([cv2.resize(ov, (w, h)), cv2.resize(heat, (w, h))])
    cv2.putText(row, f"{name} IoU {v['sil_iou']:.3f} edge {v['edge_mean'] or 0:.2f}px", (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return row


def points_eval(run: Path, names: list[str]):
    p = run / "points.npz"
    if not p.exists():
        return None, None, []
    z = np.load(p)
    d, oi = z["dist_m"], z["obj_index"]
    pnames = list(z["names"])
    X = np.load(DATA / "points_world.npy")
    err = np.load(DATA / "points_err.npy")
    (x0, x1), (y0, y1), (z0, z1) = OBJ_BOX
    sel = (X[:, 0] > x0) & (X[:, 0] < x1) & (X[:, 1] > y0) & (X[:, 1] < y1) & (X[:, 2] > z0) & (X[:, 2] < z1)
    sel &= err < 2.0
    # outside the case footprint, mat-blue points are the mat's own raised ridges/compartments, not the object
    rgb = np.load(DATA / "points_rgb.npy")
    hsv = cv2.cvtColor(rgb.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(int)
    matblue = (hsv[:, 0] >= 95) & (hsv[:, 0] <= 120) & (hsv[:, 1] >= 110)
    outside = (np.abs(X[:, 0]) > 0.107) | (np.abs(X[:, 1]) > 0.055)
    sel &= ~(matblue & outside)
    dd = np.where(np.isfinite(d), d, 0.05)[sel]
    glob = {"n": int(sel.sum()), "median_mm": float(np.median(dd) * 1e3), "frac_le1mm": float((dd <= 1e-3).mean()),
            "frac_le2mm": float((dd <= 2e-3).mean()), "frac_gt4mm": float((dd > 4e-3).mean())}
    per = {}
    dd_all = np.where(np.isfinite(d), d, 0.05)
    for k, nm in enumerate(pnames):
        m = sel & (oi == k)  # every point whose nearest surface is this part (search radius 50 mm), untruncated
        if m.any():
            per[nm] = {"n": int(m.sum()), "median_mm": float(np.median(dd_all[m]) * 1e3),
                       "frac_le1mm": float((dd_all[m] <= 1e-3).mean()),
                       "frac_gt4mm": float((dd_all[m] > 4e-3).mean())}
    orphan = sel & ~(d <= 4e-3)
    Xo = X[orphan]
    clusters = []
    if len(Xo):
        # greedy 3D clustering on a 6 mm grid
        cell = np.floor(Xo / 0.006).astype(int)
        uniq, inv, cnt = np.unique(cell, axis=0, return_inverse=True, return_counts=True)
        order = np.argsort(-cnt)
        for j in order[:25]:
            if cnt[j] < 4:
                break
            pts = Xo[inv.ravel() == j]
            near = oi[orphan][inv.ravel() == j]
            clusters.append({"center_mm": (pts.mean(0) * 1e3).round(1).tolist(), "n": int(cnt[j]),
                             "nearest_part": pnames[int(np.bincount(near[near >= 0]).argmax())]
                             if (near >= 0).any() else None,
                             "median_dist_mm": float(np.median(np.where(np.isfinite(d[orphan][inv.ravel() == j]),
                                                                         d[orphan][inv.ravel() == j], 0.05)) * 1e3)})
    return glob, per, clusters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--parts", default=None)
    ap.add_argument("--mode", default="full", choices=["geom", "full"])
    a = ap.parse_args()
    run = Path(a.run)
    meta, names, keys = load_run(run)
    out = run / "eval"
    out.mkdir(exist_ok=True)
    views, parts_acc, cands, thumbs, maps_by_view = [], {}, [], [], {}
    for name in meta["views"]:
        v, per_part, maps = eval_view(run, name, keys, len(names), a.mode)
        views.append(v)
        for k, pp in enumerate(per_part):
            if pp is None:
                continue
            acc = parts_acc.setdefault(names[k], {"views": 0, "px": 0, "fp": 0, "fn": 0, "edge_n": 0, "edge_sum": 0.0,
                                                  "edge_le2_n": 0.0, "color_sum": 0.0, "color_n": 0,
                                                  "ssim_sum": 0.0})
            acc["views"] += 1
            for key in ("px", "fp", "fn", "edge_n"):
                acc[key] += pp[key]
            if pp["edge_mean"] is not None:
                acc["edge_sum"] += pp["edge_mean"] * pp["edge_n"]
                acc["edge_le2_n"] += pp["edge_le2"] * pp["edge_n"]
            if pp["color_res"] is not None:
                acc["color_sum"] += pp["color_res"] * pp["px"]
                acc["ssim_sum"] += pp["ssim"] * pp["px"]
                acc["color_n"] += pp["px"]
        cands += tiles(maps["err"], maps["idm"], maps["fn_part"], name)
        thumbs.append(view_thumb(name, maps, v))
        maps_by_view[name] = maps
    pts_glob, pts_part, orphans = points_eval(run, names)
    parts = []
    for nm, acc in parts_acc.items():
        row = {"part": nm, "views": acc["views"], "px": acc["px"],
               "fp_frac": acc["fp"] / max(acc["px"], 1), "fn_px": acc["fn"],
               "edge_mean": acc["edge_sum"] / acc["edge_n"] if acc["edge_n"] else None,
               "edge_le2": acc["edge_le2_n"] / acc["edge_n"] if acc["edge_n"] else None,
               "color_res": acc["color_sum"] / acc["color_n"] if acc["color_n"] else None,
               "ssim": acc["ssim_sum"] / acc["color_n"] if acc["color_n"] else None}
        if pts_part and nm in pts_part:
            row.update({f"pts_{k}": v for k, v in pts_part[nm].items()})
        row["badness"] = (row["fp_frac"] + row["fn_px"] / max(acc["px"], 1) + (row["edge_mean"] or 0) / 10
                          + (1 - (row["ssim"] if row["ssim"] is not None else 1)) * 0.5)
        parts.append(row)
    parts.sort(key=lambda r: -r["badness"])
    unmod = sum(v["fn_unmodeled"] for v in views)
    summary = {"run": run.name, "n_views": len(views),
               "sil_iou_mean": float(np.mean([v["sil_iou"] for v in views])),
               "sil_iou_min": float(np.min([v["sil_iou"] for v in views])),
               "sil_iou_noband_mean": float(np.mean([v["sil_iou_noband"] for v in views])),
               "unscored_frac_mean": float(np.mean([v["unscored_frac"] for v in views])),
               "unscored_frac_max": float(np.max([v["unscored_frac"] for v in views])),
               "fn_unmodeled_px": int(unmod),
               "edge_mean_px": float(np.nanmean([v["edge_mean"] or np.nan for v in views])),
               "edge_le2": float(np.nanmean([v["edge_le2"] if v["edge_le2"] is not None else np.nan for v in views])),
               "color_res": float(np.nanmean([v["color_res"] if v["color_res"] is not None else np.nan
                                              for v in views])),
               "ssim": float(np.nanmean([v["ssim"] if v["ssim"] is not None else np.nan for v in views])),
               "points": pts_glob, "n_orphan_clusters": len(orphans), "timing_render": meta.get("timing")}
    if a.parts:
        pref = tuple(a.parts.split(","))
        cands = [c for c in cands if (c["part"] >= 0 and names[c["part"]].startswith(pref))]
    regions = nms(cands, a.top, max_per_part=3 if a.parts else 2)
    for i, r in enumerate(regions):
        r["rank"] = i + 1
        r["part_name"] = names[r["part"]] if r["part"] >= 0 else "unmodeled"
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "parts.json").write_text(json.dumps(parts, indent=1))
    (out / "views.json").write_text(json.dumps(views, indent=1))
    (out / "orphans.json").write_text(json.dumps(orphans, indent=1))
    (out / "regions.json").write_text(json.dumps(regions, indent=1))
    if regions:
        panels = [region_panel(run, r, maps_by_view[r["view"]], names) for r in regions]
        wmax = max(p.shape[1] for p in panels)
        panels = [cv2.copyMakeBorder(p, 0, 4, 0, wmax - p.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
                  for p in panels]
        cv2.imwrite(str(out / "regions.png"), np.vstack(panels))
    cols = 2
    rows = [np.hstack(thumbs[i:i + cols] + [np.zeros_like(thumbs[0])] * (cols - len(thumbs[i:i + cols])))
            for i in range(0, len(thumbs), cols)]
    cv2.imwrite(str(out / "views.png"), np.vstack(rows), [cv2.IMWRITE_PNG_COMPRESSION, 3])
    write_report(out, summary, parts, regions, orphans)
    print(json.dumps({k: v for k, v in summary.items() if k != "timing_render"}, indent=1))


def fmt(x, f="{:.3f}"):
    return "-" if x is None else f.format(x)


def write_report(out: Path, s: dict, parts: list, regions: list, orphans: list):
    L = [f"# Eval {s['run']}", "",
         f"- views {s['n_views']}; silhouette IoU mean {s['sil_iou_mean']:.3f} (min {s['sil_iou_min']:.3f}; without "
         f"the 2 px band {s['sil_iou_noband_mean']:.3f}); unmodeled object px {s['fn_unmodeled_px']}",
         f"- model pixels not scored (ray leaves the mat): mean {s['unscored_frac_mean']:.3f}, max "
         f"{s['unscored_frac_max']:.3f}",
         f"- model edges: mean dist to photo edge {s['edge_mean_px']:.2f} px, within 2 px {s['edge_le2']:.3f}",
         f"- color residual {fmt(s['color_res'], '{:.1f}')}, SSIM {fmt(s['ssim'])}"]
    if s["points"]:
        p = s["points"]
        L.append(f"- sparse points on object: {p['n']}, median {p['median_mm']:.2f} mm to surface, "
                 f"<=1 mm {p['frac_le1mm']:.3f}, <=2 mm {p['frac_le2mm']:.3f}, >4 mm (orphans) {p['frac_gt4mm']:.3f}")
    L += ["", "## Parts (worst first)", "",
          "| part | views | fp frac | fn px | edge mean px | edge<=2px | color res | ssim | pts n | pts med mm | pts >4mm |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in parts:
        L.append(f"| {r['part']} | {r['views']} | {r['fp_frac']:.3f} | {r['fn_px']} | {fmt(r['edge_mean'], '{:.2f}')} | "
                 f"{fmt(r['edge_le2'])} | {fmt(r['color_res'], '{:.1f}')} | {fmt(r['ssim'])} | "
                 f"{r.get('pts_n', '-')} | {fmt(r.get('pts_median_mm'), '{:.2f}')} | {fmt(r.get('pts_frac_gt4mm'), '{:.2f}')} |")
    L += ["", "## Orphan point clusters (missing geometry candidates, world mm)", ""]
    for c in orphans[:15]:
        L.append(f"- center {c['center_mm']} n {c['n']} nearest {c['nearest_part']} dist {c['median_dist_mm']:.1f} mm")
    L += ["", "## Inspection regions (regions.png, top to bottom)", ""]
    for r in regions:
        L.append(f"- #{r['rank']} {r['view']} tile x{r['x']} y{r['y']} (1/4 scale) part {r['part_name']} "
                 f"score {r['score']:.2f}")
    L += ["", "Legend regions.png: photo | render | edges (green photo, red model, yellow both) | "
          "error heat (magenta excess geometry, cyan missing geometry)."]
    (out / "report.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
