"""Draw world-mm wireframe hypotheses on a photo (1/2-scale undistorted) with a labeled pixel grid.

  uv run python scripts/tools/overlay.py VIEW SEGS [--out path.jpg] [--step 50] [--pad 60]

SEGS is a JSON file or an inline JSON string: a list of items, each one of
  {"box": [x0, x1, y0, y1, z0, z1], "c": [b, g, r]}      12 edges of an axis-aligned box
  {"poly": [[x, y, z], ...], "c": [b, g, r]}             closed polyline (e.g. a rim outline or a card)
  {"p": [x, y, z], "q": [x, y, z], "c": [b, g, r]}       one segment
Colors are BGR (default red). The crop covers the projected geometry plus --pad px and is scaled to at most
1400 px; the grid labels are 1/2-scale pixel coords (the same coords pick.py crop/tri/tri2 use).
Use it to check a dimension hypothesis against several views before editing the TOML, and to read the
pixel of a feature for tri2. Output defaults to outputs/picks/overlay_<VIEW>.jpg.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import OUTPUTS  # noqa: E402
from tools.geom import load_views  # noqa: E402


def box_edges(x0, x1, y0, y1, z0, z1):
    P = [(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]
    return [(P[i], P[j]) for i in range(8) for j in range(i + 1, 8)
            if sum(a != b for a, b in zip(P[i], P[j])) == 1]


def segments(items):
    out = []
    for s in items:
        c = tuple(s.get("c", [0, 0, 255]))
        if "box" in s:
            out += [(c, p, q) for p, q in box_edges(*s["box"])]
        elif "poly" in s:
            pts = s["poly"]
            out += [(c, pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
        else:
            out.append((c, s["p"], s["q"]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("view")
    ap.add_argument("segs")
    ap.add_argument("--out", default=None)
    ap.add_argument("--step", type=int, default=50)
    ap.add_argument("--pad", type=int, default=60)
    a = ap.parse_args()
    src = Path(a.segs)
    items = json.loads(src.read_text() if src.suffix == ".json" and src.exists() else a.segs)
    v = load_views(2)[a.view]
    img = v.image().copy()
    pts = []
    for c, p, q in segments(items):
        t = np.linspace(0, 1, 40)[:, None]
        uv, z = v.project((np.array(p) * (1 - t) + np.array(q) * t) / 1e3)
        uv = uv[z > 0]
        pts += uv.tolist()
        for i in range(len(uv) - 1):
            cv2.line(img, tuple(int(x) for x in uv[i]), tuple(int(x) for x in uv[i + 1]), c, 1, cv2.LINE_AA)
    pts = np.array(pts)
    pts = pts[(pts[:, 0] > -500) & (pts[:, 0] < v.width + 500) & (pts[:, 1] > -500) & (pts[:, 1] < v.height + 500)]
    x0, y0 = np.clip(pts.min(0) - a.pad, 0, None).astype(int)
    x1, y1 = np.minimum(pts.max(0) + a.pad, [v.width, v.height]).astype(int)
    for g in range((x0 // a.step + 1) * a.step, x1, a.step):
        cv2.line(img, (g, y0), (g, y1), (0, 200, 200), 1)
        cv2.putText(img, str(g), (g + 2, y0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    for g in range((y0 // a.step + 1) * a.step, y1, a.step):
        cv2.line(img, (x0, g), (x1, g), (0, 200, 200), 1)
        cv2.putText(img, str(g), (x0 + 2, g - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    crop = img[y0:y1, x0:x1]
    s = min(1.0, 1400 / max(crop.shape[:2]))
    crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    out = Path(a.out) if a.out else OUTPUTS / "picks" / f"overlay_{a.view}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), crop)
    print(out, "crop origin", x0, y0, "scale", round(s, 3))


if __name__ == "__main__":
    main()
