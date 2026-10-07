"""partcrop.py RUN PART [--top N]: per view SSE of PART (as sse.py) and a crop sheet of the worst views
(photo | render after color fit | ID mask of the part) -> outputs/scratch/chassis/partcrop_<part>.jpg."""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
from common import DATA  # noqa: E402
from eval.compare_photometric import affine_fit  # noqa: E402
from eval.evaluate import decode_id, load_run  # noqa: E402

run, part = ROOT / "outputs" / "runs" / sys.argv[1], sys.argv[2]
top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 3
meta, names, keys = load_run(run)
k = names.index(part)
res = []
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
    sel = m & (decode_id(run / "id" / f"{n}.png", keys) == k)
    if sel.sum():
        res.append((float(e[sel].sum()) / 1e6, n, photo, np.clip(fit, 0, 255).astype(np.uint8), sel))
res.sort(key=lambda t: -t[0])
print({n: round(s, 1) for s, n, *_ in res})
rows = []
for s, n, photo, fit, sel in res[:top]:
    ys, xs = np.nonzero(sel)
    y0, y1, x0, x1 = max(ys.min() - 20, 0), ys.max() + 20, max(xs.min() - 20, 0), xs.max() + 20
    msk = np.zeros_like(photo); msk[sel] = (255, 0, 255)
    tiles = [cv2.resize(im[y0:y1, x0:x1], None, fx=1, fy=1) for im in (photo, fit, cv2.addWeighted(photo, 0.6, msk, 0.4, 0))]
    h = 360; tiles = [cv2.resize(t, (int(t.shape[1] * h / t.shape[0]), h)) for t in tiles]
    row = np.hstack(tiles)
    cv2.putText(row, f"{n} sse {s:.0f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    rows.append(row)
w = max(r.shape[1] for r in rows)
rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
out = ROOT / "outputs" / "scratch" / "chassis" / f"partcrop_{part.split('.')[-1]}.jpg"
sheet = np.vstack(rows)
if sheet.shape[1] > 1800:
    sheet = cv2.resize(sheet, (1800, int(sheet.shape[0] * 1800 / sheet.shape[1])))
cv2.imwrite(str(out), sheet)
print(out)
