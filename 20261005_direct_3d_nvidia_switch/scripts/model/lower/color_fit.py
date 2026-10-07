"""Suggest flat base colors for the lower group's photo-sampled materials from a full-mode run on train views:
per material, ratio of photo to render medians (linear RGB) over pixels the material's objects own (ID pass,
eroded 1 px); new color = old color x ratio. Usage: uv run python scripts/model/lower/color_fit.py RUN"""
import json
import re
import sys
import tomllib
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
GROUPS = {  # color key -> object name regex (lower.*)
    "copper": r"lower\.(finger_\d|loops_|tube_f)", "oe": r"lower\.oe_\d", "base": r"lower\.oe_base",
    "org": r"lower\.org_", "strip": r"lower\.strip_", "fiber": r"lower\.fibers", "braid": r"lower\.braid_a",
    "braid_b": r"lower\.braid_b", "ribbon": r"lower\.ribbon", "conn": r"lower\.connector",
    "white": r"lower\.(plug|clip_)", "metal": r"lower\.(fitting|manifold$|plate_l$|board_bracket)",
    "silver": r"lower\.(rod_l|finger_straps)", "cable_blk": r"lower\.cable_loop", "board": r"lower\.(board$|smd)"}


def lin(c):
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


run = ROOT / sys.argv[1]
hold = set(json.loads((ROOT / "data/split.json").read_text())["holdout"])
P = tomllib.loads((ROOT / "config/model/lower.toml").read_text())
idm = json.loads((run / "id_map.json").read_text())
acc = {k: ([], []) for k in GROUPS}
for f in sorted((run / "id").glob("*.png")):
    if f.stem in hold:
        continue
    ID = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
    R = cv2.imread(str(run / "rgb" / f.name))
    Ph = cv2.imread(str(ROOT / "data/undistorted_s4" / f"{f.stem}.jpg"))
    if R is None or Ph is None:
        continue
    for k, rx in GROUPS.items():
        m = np.zeros(ID.shape[:2], np.uint8)
        for n, c in idm.items():
            if re.match(rx, n):
                m |= (np.abs(ID[..., :3].astype(int) - np.array(c[::-1])).sum(2) <= 6).astype(np.uint8)
        m = cv2.erode(m, np.ones((3, 3), np.uint8)) > 0
        if m.sum():
            acc[k][0].append(Ph[m])
            acc[k][1].append(R[m])
for k, (ph, re_) in acc.items():
    if not ph:
        continue
    ph, re_ = np.concatenate(ph), np.concatenate(re_)
    p, r = lin(np.median(ph, 0)[::-1]), lin(np.median(re_, 0)[::-1])
    old = np.array(P["colors"][k])
    new = np.clip(old * p / np.maximum(r, 1e-4), 0, 0.9)
    print(f"{k:10s} n {len(ph):7d} photo {np.round(p, 3)} render {np.round(r, 3)} old {old} new {np.round(new, 3).tolist()}")
