"""sse.py RUN_A RUN_B [...]: absolute photometric squared error (as scripts/eval/error_budget.py: per-view color
fit, blur 4 px, photo object region) summed per chassis surface family, plus chassis total and all-model total.
Family = part name without the "tex_" prefix and the side/bay suffix, so a box part and its textured quad compare
as one. Units: 1e6 (8-bit levels squared). Per-view results cached in <run>/eval/sse_cache.json."""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
from common import DATA  # noqa: E402
from eval.compare_photometric import affine_fit  # noqa: E402
from eval.evaluate import decode_id, load_run  # noqa: E402

FAM = [(r"floor", "floor"), (r"rear_(wall|in|out)", "rear"), (r"divider", "divider"), (r"crossbar_top|crossbar_ext", "crossbar_top"),
       (r"crossbar_fl|crossbar_ch", "crossbar_fl"), (r"duct_front", "ducts"), (r"wall_m?p?x(_out|_in)?$|wall_[mp]x_(in|out)", "walls"),
       (r"flange|rim_", "rims"), (r"front_plate", "front_plate"), (r"lip", "lip"), (r"grip|wing", "wing"),
       (r"mmc", "mmc"), (r"lever", "levers"), (r"clip", "clips"), (r"shelf", "shelf"), (r"rail_rear", "rail_rear")]


def fam(nm):
    if not nm.startswith("chassis."):
        return None
    for pat, f in FAM:
        if re.search(pat, nm):
            return f
    return "other"


def run_sse(run: Path):
    cache = run / "eval" / "sse_cache.json"
    if cache.exists() and cache.stat().st_mtime > (run / "eval" / "report.md").stat().st_mtime:
        return json.loads(cache.read_text())
    meta, names, keys = load_run(run)
    part = defaultdict(float)
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
            nm = names[k] if k >= 0 else "(none)"
            part[nm] += float(e[m & (idm == k)].sum()) / 1e6
        part["(all)"] += float(e[m].sum()) / 1e6
    cache.write_text(json.dumps(part))
    return part


if __name__ == "__main__":
    runs = sys.argv[1:]
    R = [run_sse(ROOT / "outputs" / "runs" / r) for r in runs]
    F = []
    for p in R:
        f = defaultdict(float)
        for nm, v in p.items():
            g = fam(nm)
            if g:
                f[g] += v
                f["(chassis)"] += v
        f["(all)"] = p["(all)"]
        F.append(f)
    keys = sorted(set().union(*F), key=lambda k: -F[0].get(k, 0))
    print("family".ljust(14) + "".join(f"{r[-7:]:>10}" for r in runs))
    for k in keys:
        print(k.ljust(14) + "".join(f"{f.get(k, 0):10.1f}" for f in F))
