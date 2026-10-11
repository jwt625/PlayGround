"""Stylized PAM4 eye-diagram screen texture for the test-station monitor (workstream M, DevLog-019).
Synthetic: random 4-level symbols, raised-cosine transitions, Gaussian noise, accumulated as a persistence image.
No measured data; display only. Usage: uv run --with numpy --with pillow python scripts/assembly/make_eye_texture.py <out.png>"""
import sys

import numpy as np
from PIL import Image

W, H = 768, 512
rng = np.random.default_rng(7)
acc = np.zeros((H, W))
levels = np.array([-1.0, -1 / 3, 1 / 3, 1.0])
n_sym, spu = 3000, 480                                  # symbols, samples per unit interval
sym = levels[rng.integers(0, 4, n_sym)]
t = np.arange(spu) / spu
ramp = 0.5 - 0.5 * np.cos(np.pi * np.clip(t / 0.55, 0, 1))   # transition over 55 % of the UI
for k in range(1, n_sym - 2):
    a, b = sym[k], sym[k + 1]
    seg = a + (b - a) * ramp                            # one UI from symbol k to k+1
    seg = seg + rng.normal(0, 0.035, spu) + rng.normal(0, 0.02)
    # draw 2 UI wide window: x in [0, W) covers 2 UI; place this UI at phase 0 and 1
    for ph in (0, 1):
        x = (((ph + t + 0.25 + rng.normal(0, 0.012)) % 2) / 2 * W).astype(int) % W
        y = ((0.5 - 0.42 * seg) * H).astype(int)
        ok = (y >= 0) & (y < H)
        np.add.at(acc, (y[ok], x[ok]), 1.0)
acc = np.log1p(acc)
acc /= acc.max()
# blur slightly
k = np.array([1, 4, 6, 4, 1], float); k /= k.sum()
acc = np.apply_along_axis(lambda r: np.convolve(r, k, "same"), 1, acc)
acc = np.apply_along_axis(lambda c: np.convolve(c, k, "same"), 0, acc)
acc = np.clip(acc / acc.max() * 1.3, 0, 1)
img = np.zeros((H, W, 3))
img[..., 0] = acc ** 0.8 * 1.0                          # amber persistence
img[..., 1] = acc ** 1.1 * 0.72
img[..., 2] = acc ** 2.0 * 0.25
# graticule
for gx in range(0, W, W // 10):
    img[:, gx, :] = np.maximum(img[:, gx, :], 0.10)
for gy in range(0, H, H // 8):
    img[gy, :, :] = np.maximum(img[gy, :, :], 0.10)
Image.fromarray((img * 255).astype(np.uint8)).save(sys.argv[1])
print("wrote", sys.argv[1], W, H)
