"""bright gridded s2 crop: bcrop.py VIEW u v half step"""
import sys, cv2, numpy as np
v, u0, v0, half, step = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
im = cv2.imread(f"data/undistorted_s2/{v}.jpg", cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION).astype(np.float32) / 255
x0, y0 = u0 - half, v0 - half
c = np.clip((im[y0:v0 + half, x0:u0 + half] ** 0.35) * 255, 0, 255).astype(np.uint8).copy()
for k in range((-x0) % step, c.shape[1], step):
    cv2.line(c, (k, 0), (k, c.shape[0]), (0, 255, 255), 1); cv2.putText(c, str(x0 + k), (k + 2, 12), 0, 0.35, (0, 255, 0), 1)
for k in range((-y0) % step, c.shape[0], step):
    cv2.line(c, (0, k), (c.shape[1], k), (0, 255, 255), 1); cv2.putText(c, str(y0 + k), (2, k + 12), 0, 0.35, (0, 255, 0), 1)
o = f"outputs/scratch/package/bc_{v}_{u0}_{v0}.jpg"; cv2.imwrite(o, c); print(o)
