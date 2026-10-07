"""epi.py VIEW u v VIEW2 [VIEW3..]: s2 pixel ray -> points at depths, in package-local mm and projected into other views"""
import json, sys
import numpy as np
sys.path.insert(0, "scripts")
from tools.geom import load_views
V = load_views(2)
fr = json.load(open("config/frames.json"))["frames"]["package"]; R = np.array(fr["R"]); t = np.array(fr["t_mm"])
v = V[sys.argv[1]]; uv = np.array([float(sys.argv[2]), float(sys.argv[3])])
d = v.ray(uv); c = v.center
for dep in range(150, 501, 25):
    X = c + d * dep / 1e3
    loc = (X * 1e3 - t) @ R + [4, 3.4, 0]
    s = f"depth {dep}: local {loc.round(0)}"
    for o in sys.argv[4:]:
        p, z = V[o].project(X); s += f"  {o[-4:]} ({p[0][0]:.0f},{p[0][1]:.0f})"
    print(s)
