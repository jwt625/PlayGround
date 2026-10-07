"""edges.py RUN VIEW [x0 y0 x1 y1 at s4]: photo with render edges (red) + model silhouette of chassis parts (from id pass)."""
import sys, json, cv2, numpy as np
run, s = sys.argv[1], sys.argv[2]
im = cv2.imread(f'data/undistorted_s4/{s}.jpg'); r = cv2.imread(f'outputs/runs/{run}/rgb/{s}.png')
r = cv2.resize(r, (im.shape[1], im.shape[0]))
e = cv2.Canny(cv2.cvtColor(r, cv2.COLOR_BGR2GRAY), 15, 45)
o = im.copy(); o[e > 0] = (0, 0, 255)
if len(sys.argv) > 6:
    x0, y0, x1, y1 = map(int, sys.argv[3:7]); o = o[y0:y1, x0:x1]
cv2.imwrite(f'outputs/scratch/chassis/edge_{run}_{s}.jpg', o)
