"""CRT slice of scripts/eval/error_budget.py (same metric: per-view color fit, sigma 4 px blur, photo object region),
reported as absolute numbers so runs compare even while other groups change the total.
Usage: uv run python scripts/model/crt/crt_budget.py outputs/runs/<run> [outputs/runs/<run2> ...]"""
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import DATA  # noqa: E402
from eval.compare_photometric import affine_fit  # noqa: E402
from eval.evaluate import decode_id, load_run  # noqa: E402

for r in sys.argv[1:]:
    run = Path(r)
    meta, names, keys = load_run(run)
    sse, px = defaultdict(float), defaultdict(int)
    tot = 0.0
    for n in meta["views"]:
        photo = cv2.imread(str(DATA / "undistorted_s4" / f"{n}.jpg"), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        pm = cv2.imread(str(DATA / "masks_s4" / f"{n}.png"), cv2.IMREAD_GRAYSCALE)
        m = cv2.erode((pm == 255).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        rgba = cv2.imread(str(run / "rgb" / f"{n}.png"), cv2.IMREAD_UNCHANGED)
        model = (rgba[..., :3] * (rgba[..., 3:4] / 255.0)).astype(np.float64)
        known = (pm != 128) & (rgba[..., 3] > 0)
        fit = affine_fit(model, photo, known if known.sum() > 2 * m.sum() else m)
        d = cv2.GaussianBlur(fit.astype(np.float32), (0, 0), 4) - cv2.GaussianBlur(photo.astype(np.float32), (0, 0), 4)
        e = (d.astype(np.float64) ** 2).sum(2)
        idm = decode_id(run / "id" / f"{n}.png", keys)
        tot += float(e[m].sum())
        for k in np.unique(idm[m]):
            if k < 0 or not names[k].startswith("crt."):
                continue
            sel = m & (idm == k)
            sse[names[k]] += float(e[sel].sum())
            px[names[k]] += int(sel.sum())
    S, P = sum(sse.values()), sum(px.values())
    db = lambda s, p: 10 * np.log10(255 ** 2 * 3 * p / max(s, 1e-9))  # noqa: E731
    print(f"{run.name}: crt share {100 * S / tot:.1f}%  crt px {P}  crt SSE {S / 1e9:.3f}e9  crt dB {db(S, P):.2f}")
    for nm in sorted(sse, key=lambda k: -sse[k])[:8]:
        print(f"   {nm:24s} px {px[nm]:7d}  SSE {sse[nm] / 1e9:.3f}e9  dB {db(sse[nm], px[nm]):.1f}")
