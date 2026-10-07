"""Intersect pixel rays of one view with an axis-aligned plane (world mm; 1/2-scale pixel coords).

  uv run python scripts/tools/rayplane.py VIEW AXIS VALUE U1 V1 [U2 V2 ...]
  e.g.  ... rayplane.py IMG_1513 y -49.75 1060 618 1112 712      picks on the near case wall (plane y = -49.75)
        ... rayplane.py IMG_1518 z 37 1410 620                   a point known to lie at z = 37

AXIS is x, y or z. Prints U V -> [x, y, z] per pick. Use it for silhouette or textureless points on a known
plane (a wall face, a rim height, the mat), where tri/tri2 cannot match. A plane offset of d mm moves the
result along the ray, so check with a second view when the plane value is uncertain.
Negative VALUE works directly (arguments are parsed positionally, not by argparse).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.geom import load_views, refuse_holdout  # noqa: E402


def main():
    a = [x for x in sys.argv[1:] if x != "--allow-holdout"]
    if len(a) < 5 or len(a[3:]) % 2:
        raise SystemExit(__doc__)
    refuse_holdout([a[0]], allow="--allow-holdout" in sys.argv)
    v = load_views(2)[a[0]]
    ax = "xyz".index(a[1])
    val = float(a[2]) / 1e3
    c = v.center
    for i in range(3, len(a), 2):
        d = v.ray(np.array([float(a[i]), float(a[i + 1])]))
        s = (val - c[ax]) / d[ax]
        print(a[i], a[i + 1], "->", ((c + s * d) * 1e3).round(1).tolist(), "range %.0f mm" % (s * 1e3))


if __name__ == "__main__":
    main()
