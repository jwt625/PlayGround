"""Photo edge maps and their distance transforms at 1/4 scale, for the in-Blender fit loss.

Output: data/edt_s4/<stem>.npy (uint8 = round(10 x px distance to the nearest photo edge), clipped at 20 px).
Edges: Canny(40, 100) after a 1 px Gaussian blur on gray (same as scripts/eval/evaluate.py).
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA  # noqa: E402
from eval.evaluate import photo_edges  # noqa: E402

out = DATA / "edt_s4"
out.mkdir(exist_ok=True)
n = 0
for p in sorted((DATA / "undistorted_s4").glob("*.jpg")):
    q = out / f"{p.stem}.npy"
    if q.exists():
        continue
    g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
    e = photo_edges(g)
    d = np.round(np.minimum(cv2.distanceTransform((~e).astype(np.uint8), cv2.DIST_L2, 5), 20.0) * 10).astype(np.uint8)
    np.save(q, d)
    n += 1
print("edt written", n)
