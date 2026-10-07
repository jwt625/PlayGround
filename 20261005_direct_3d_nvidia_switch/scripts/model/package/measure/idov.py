"""idov.py RUN VIEWS crop(x0,y0,x1,y1 s4): contours of package.* ID regions on brightened photos"""
import json, sys, cv2, numpy as np
run, views = sys.argv[1], sys.argv[2].split(",")
x0, y0, x1, y1 = map(int, sys.argv[3].split(","))
idm = json.load(open(f"outputs/runs/{run}/id_map.json"))
cols = {"stand": (0, 0, 255), "ring": (255, 0, 255), "substrate": (0, 255, 0), "oe_gold": (255, 255, 0), "oe_chip": (0, 255, 255), "lid": (0, 165, 255), "field": (255, 128, 0)}
tiles = []
for v in views:
    ph = cv2.imread(f"data/undistorted_s4/{v}.jpg", cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION).astype(np.float32) / 255
    ph = np.clip(ph ** 0.4 * 255, 0, 255).astype(np.uint8)
    idi = cv2.imread(f"outputs/runs/{run}/id/{v}.png")[:, :, ::-1].astype(int)
    for name, rgb in idm.items():
        if not name.startswith("package."): continue
        m = (np.abs(idi - np.array(rgb)).sum(2) <= 3).astype(np.uint8)
        if m.sum() == 0: continue
        key = next((k for k in cols if k in name), "ring")
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(ph, cs, -1, cols[key], 1)
    c = ph[y0:y1, x0:x1]; cv2.putText(c, v, (5, 20), 0, 0.6, (0, 255, 255), 2); tiles.append(c)
o = f"outputs/scratch/package/idov_{run}.jpg"; cv2.imwrite(o, np.hstack(tiles)); print(o)
