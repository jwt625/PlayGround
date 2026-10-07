"""pproj.py VIEW x y z [x y z ...]: package-local mm -> s2 pixel"""
import json, sys
import numpy as np
sys.path.insert(0, "scripts")
from tools.geom import load_views
V = load_views(2)
fr = json.load(open("config/frames.json"))["frames"]["package"]; R = np.array(fr["R"]); t = np.array(fr["t_mm"])
a = [float(x) for x in sys.argv[2:]]
for i in range(0, len(a), 3):
    uv, _ = V[sys.argv[1]].project((R @ np.array(a[i:i + 3]) + t) / 1e3)
    print(a[i:i + 3], uv[0].round(0).tolist())
