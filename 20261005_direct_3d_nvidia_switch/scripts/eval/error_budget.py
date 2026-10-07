"""Photometric error budget: share of squared error (after the per-view color fit, sigma 4 px blur) by model
part and group, inside the photo object region. Tells where photometric effort pays off.
Usage: uv run python scripts/eval/error_budget.py outputs/runs/<run> [--top 15]"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA  # noqa: E402
from eval.compare_photometric import affine_fit  # noqa: E402
from eval.evaluate import decode_id, load_run  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--top", type=int, default=15)
a = ap.parse_args()
run = Path(a.run)
meta, names, keys = load_run(run)
part, px = defaultdict(float), defaultdict(int)
tot = 0.0
for n in meta["views"]:
    photo = cv2.imread(str(DATA / "undistorted_s4" / f"{n}.jpg"), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    pm = cv2.imread(str(DATA / "masks_s4" / f"{n}.png"), cv2.IMREAD_GRAYSCALE)
    m = cv2.erode((pm == 255).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    rgba = cv2.imread(str(run / "rgb" / f"{n}.png"), cv2.IMREAD_UNCHANGED)
    model = (rgba[..., :3] * (rgba[..., 3:4] / 255.0)).astype(np.float64)
    known = (pm != 128) & (rgba[..., 3] > 0)
    fit = affine_fit(model, photo, known if known.sum() > 2 * m.sum() else m)
    d = cv2.GaussianBlur(fit.astype(np.float32), (0, 0), 4) - cv2.GaussianBlur(photo.astype(np.float32), (0, 0), 4)
    e = (d.astype(np.float64) ** 2).sum(2)
    idm = decode_id(run / "id" / f"{n}.png", keys)
    for k in np.unique(idm[m]):
        sel = m & (idm == k)
        nm = names[k] if k >= 0 else "(no model: missing geometry)"
        part[nm] += float(e[sel].sum())
        px[nm] += int(sel.sum())
    tot += float(e[m].sum())
grp = defaultdict(float)
for nm, v in part.items():
    grp[nm.split(".")[0] if not nm.startswith("(") else nm] += v
print(json.dumps({"by_group_percent": {g: round(100 * v / tot, 1) for g, v in sorted(grp.items(), key=lambda t: -t[1])},
                  "top_parts": [(nm, round(100 * v / tot, 1), round(10 * np.log10(255 ** 2 * 3 * px[nm] / max(v, 1e-9)), 1))
                                for nm, v in sorted(part.items(), key=lambda t: -t[1])[: a.top]]}, indent=1))
