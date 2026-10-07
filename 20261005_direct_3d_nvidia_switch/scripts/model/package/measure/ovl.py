"""ovl.py out.jpg VIEW,... : draw local-frame (corrected) polylines from a JSON on stdin onto brightened s4 views"""
import json, sys
import numpy as np, cv2
sys.path.insert(0, "scripts")
from tools.geom import load_views
V = load_views(4)
fr = json.load(open("config/frames.json"))["frames"]["package"]; R = np.array(fr["R"]); t = np.array(fr["t_mm"])
off = np.array([0.0, 0.0, 0.0])
polys = json.loads(sys.stdin.read())
tiles = []
for vn in sys.argv[2].split(","):
    im = cv2.imread(f"data/undistorted_s4/{vn}.jpg", cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION).astype(np.float32) / 255
    im = np.clip((im ** 0.4) * 255, 0, 255).astype(np.uint8)
    for i, pl in enumerate(polys):
        P = np.array(pl["pts"], float) + off
        uv, z = V[vn].project((P @ R.T + t) / 1e3)
        col = [(0, 0, 255), (255, 0, 255), (0, 255, 0), (255, 255, 0), (0, 165, 255)][i % 5]
        cv2.polylines(im, [uv.astype(np.int32)], pl.get("closed", True), col, 2)
    cv2.putText(im, vn, (10, 30), 0, 1, (0, 255, 255), 2)
    tiles.append(cv2.resize(im, (714, 535)))
rows = [np.hstack(tiles[i:i + 2]) if i + 1 < len(tiles) else np.hstack([tiles[i], np.zeros_like(tiles[i])]) for i in range(0, len(tiles), 2)]
cv2.imwrite(sys.argv[1], np.vstack(rows)); print(sys.argv[1])
