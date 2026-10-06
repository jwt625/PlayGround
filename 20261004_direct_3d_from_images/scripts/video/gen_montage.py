"""Assets for the 30 s build montage (two 1:1 panels, 5 images/s): viewer plan, artifact and code cards, and
3DGS "training" stages as .splat files for the Spark viewer.

Top panel: the explicit model in the glb viewer, revealed group by group (case, crt, pcb, wires), flat first,
photo textures later, on an orbit; interleaved with the agents' real artifacts (worst-region sheets, view
sheets, pick crops, texture sheets, comparison sheet) and code flashes.
Bottom panel (illustration, not a real training run): Gaussians interpolated from a random initialization to the
final Brush ply along a typical 3DGS schedule: count grows by densification until iteration 12k, each Gaussian
converges exponentially after its birth (time constants 300-1500 iterations), exact final state at 30k.

Usage: uv run python scripts/video/gen_montage.py   -> outputs/video/montage/{plan.json, art/*.jpg, splat/*.splat}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import OUTPUTS, ROOT, scene_cfg  # noqa: E402
from eval.render_3dgs import load_ply  # noqa: E402

N_FRAMES, N_STAGES, S, H = 150, 30, 1080, 810  # panels 4:3, stacked 2:3
OUT = OUTPUTS / "video" / "montage"
SH_C0 = 0.28209479177387814
A = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0.0]])  # world (Z up) -> glTF (Y up)


def letterbox(img, caption):
    h, w = img.shape[:2]
    sc = min((S - 40) / w, (H - 100) / h)
    img = cv2.resize(img, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
    can = np.full((H, S, 3), 18, np.uint8)
    y0 = (H - 70 - img.shape[0]) // 2 + 10
    x0 = (S - img.shape[1]) // 2
    can[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
    cv2.putText(can, caption, (28, H - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (235, 235, 235), 2, cv2.LINE_AA)
    return can


def code_card(path, start, n, caption):
    lines = path.read_text().splitlines()[start - 1:start - 1 + n]
    img = Image.new("RGB", (S, H), (22, 24, 28))
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 20)
    d.text((28, 22), str(path.relative_to(ROOT)), font=font, fill=(120, 180, 255))
    for i, ln in enumerate(lines):
        col = (150, 150, 150) if ln.strip().startswith(("#", '"""')) else (225, 225, 225)
        d.text((28, 70 + i * 27), ln[:84], font=font, fill=col)
    d.text((28, H - 48), caption, font=ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28),
           fill=(235, 235, 235))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def find_line(path, needle):
    for i, ln in enumerate(path.read_text().splitlines(), 1):
        if needle in ln:
            return i
    return 1


def artifacts():
    R = OUTPUTS / "runs"
    T = OUTPUTS / "textures"
    items = []

    def latest(group):  # newest surviving run of a group that has an eval sheet
        runs = sorted(R.glob(f"{group}_[0-9]*/eval/regions.png"), key=lambda q: q.stat().st_mtime)
        return runs[-1] if runs else R / "missing"

    def img(p, cap, crop=None):
        if p.exists():
            key = f"{p.relative_to(OUTPUTS)}|{cap}"
            items.append((key, lambda p=p, cap=cap, crop=crop: letterbox(
                cv2.imread(str(p))[crop[0]:crop[1]] if crop else cv2.imread(str(p)), cap)))

    def code(rel, needle, n, cap):
        p = ROOT / rel
        items.append((f"{rel}|{cap}", lambda: code_card(p, max(find_line(p, needle) - 2, 1), n, cap)))

    img(OUTPUTS / "picks" / "IMG_1560_1230_420.jpg", "measure: grid crop for picking (flyback label)")
    code("scripts/tools/geom.py", "def epipolar_match", 25, "code: epipolar NCC match with affine-warped patches")
    img(latest("case"), "eval: worst regions (case agent)", (0, 1300))
    img(T / "pcb_flyback_top_views.jpg", "texture: WARNING label from 5 train views")
    img(latest("crt"), "eval: worst regions (crt agent)", (0, 1300))
    code("scripts/blender/fit_params.py", "def evaluate(x)", 25, "code: in-Blender fit loss (edges + excess)")
    img(latest("pcb"), "eval: worst regions (pcb agent)", (0, 1300))
    img(T / "crt_yoke_views.jpg", "texture: yoke tape per-texel median")
    img(latest("wires"), "eval: worst regions (wires agent)", (0, 1300))
    code("scripts/eval/evaluate.py", "def tiles(", 25, "code: rank error tiles for visual inspection")
    img(R / "v3_probe" / "eval" / "views.png", "eval: all probe views, error heat", (0, 1200))
    img(T / "pcb_parts_sheet.jpg", "texture: 47 pcb component atlases (per-texel median)")
    img(R / "pcb_025" / "pcb_components_sheet.jpg", "photo | render: textured components", (0, 1540))
    img(OUTPUTS / "snapshots" / "v3" / "compare_holdout" / "compare.png", "holdout: photo | model | 3DGS", (0, 1100))
    return items


def splat_stages():
    xyz, scale, q, op, sh = load_ply(next(scene_cfg()["source_dir"].glob("*.ply")))
    n = len(xyz)
    rng = np.random.default_rng(0)
    lo, hi = np.percentile(xyz, 15, 0), np.percentile(xyz, 85, 0)
    x0 = lo + rng.random((n, 3)) * (hi - lo)
    s0 = np.full((n, 3), np.exp(np.mean(np.log(scale))) * 6)
    c_fin = np.clip(0.5 + SH_C0 * sh[:, 0], 0, 1)
    c0 = np.clip(0.5 + 0.15 * rng.standard_normal((n, 3)), 0, 1)
    a_fin = op
    a0 = np.full(n, 0.1)
    order = rng.permutation(n)
    n0 = 60000
    birth = np.zeros(n)
    # count N(it) = n0 + (n - n0) * smoothstep(it / 15000): invert for the birth iteration of each rank
    ranks = np.empty(n)
    ranks[order] = np.arange(n)
    frac = np.clip((ranks - n0) / (n - n0), 0, 1)
    u = np.linspace(0, 1, 2001)
    sm = u * u * (3 - 2 * u)
    birth = np.interp(frac, sm, u) * 12000
    birth[ranks < n0] = 0
    tau = np.exp(rng.uniform(np.log(300), np.log(1500), n))
    pend = 1 - np.exp(-(30000 - birth) / tau)
    (OUT / "splat").mkdir(parents=True, exist_ok=True)
    stages = []
    for k in range(N_STAGES):
        it = 30000 * (k / (N_STAGES - 1)) ** 1.6
        alive = birth <= it
        p = np.clip((1 - np.exp(-np.maximum(it - birth, 0) / tau)) / pend, 0, 1)[alive, None]
        X = x0[alive] * (1 - p) + xyz[alive] * p
        Sc = np.exp(np.log(s0[alive]) * (1 - p) + np.log(scale[alive]) * p)
        C = c0[alive] * (1 - p) + c_fin[alive] * p
        Al = a0[alive] * (1 - p[:, 0]) + a_fin[alive] * p[:, 0]
        Q = q[alive]
        buf = np.zeros(alive.sum(), dtype=[("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("r", "u1", 4)])
        buf["p"], buf["s"] = X, Sc
        buf["c"] = np.clip(np.concatenate([C, Al[:, None]], 1) * 255, 0, 255).astype(np.uint8)
        buf["r"] = np.clip(Q * 128 + 128, 0, 255).astype(np.uint8)
        f = OUT / "splat" / f"stage_{k:02d}.splat"
        buf.tofile(f)
        stages.append({"file": f"splat/{f.name}", "iter": int(it), "count": int(alive.sum())})
    return stages


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "art").mkdir(exist_ok=True)
    # Card registry: cards are rendered once into montage/cards/ and kept forever (sources such as agent runs may be
    # deleted later); each rebuild keeps every registered card in its order and only appends new ones.
    reg_p = OUT / "cards.json"
    reg = json.loads(reg_p.read_text()) if reg_p.exists() else []
    if not reg and (OUT / "art").exists():  # seed from the cards of the previous build (kept as is)
        for f in sorted((OUT / "art").glob("a_*.jpg")):
            reg.append({"key": f"seed:{f.name}", "file": f"art/{f.name}"})
    known = {r["key"] for r in reg}
    (OUT / "cards").mkdir(exist_ok=True)
    for key, make in artifacts():
        if key in known or any(r["key"].startswith("seed:") and r.get("src_key") == key for r in reg):
            continue
        p = OUT / "cards" / f"c_{len(reg):03d}.jpg"
        cv2.imwrite(str(p), make(), [cv2.IMWRITE_JPEG_QUALITY, 92])
        reg.append({"key": key, "file": f"cards/{p.name}"})
        known.add(key)
    reg_p.write_text(json.dumps(reg, indent=1))
    art_files = [r["file"] for r in reg]
    stages = splat_stages()
    # artifact slots: 3 frames each, spread over the montage
    slots = {}
    starts = np.linspace(9, N_FRAMES - 14, len(art_files)).astype(int)
    for a, s0 in zip(art_files, starts):
        for j in range(3):
            slots[int(s0 + j)] = a
    frames = []
    for k in range(N_FRAMES):
        t = k / (N_FRAMES - 1)
        az = np.radians(-90 + 360 * t)
        el = np.radians(40 + 12 * np.sin(2 * np.pi * t))
        d = 0.30
        T = np.array([0, 0, 0.015])
        C = T + d * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        frames.append({"C": (A @ C).tolist(), "T": (A @ T).tolist(), "top_img": slots.get(k),
                       "build": t, "stage": min(int(k * N_STAGES / N_FRAMES), N_STAGES - 1)})
    cfg = json.loads((OUTPUTS / "video" / "viewer" / "cfg.json").read_text())
    plan = {"size": S, "w": S, "h": H, "fps": 5, "fov_deg": cfg["fov_deg"], "splat_xform": cfg["splat"], "stages": stages,
            "frames": frames}
    (OUT / "plan.json").write_text(json.dumps(plan))
    print("art", len(art_files), "stages", [(s["iter"], s["count"]) for s in stages[::6]])


if __name__ == "__main__":
    main()
