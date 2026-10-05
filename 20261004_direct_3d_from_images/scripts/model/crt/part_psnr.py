"""PSNR inside crt.* pixels (ID pass, eroded 2 px, photo mask == object): raw, and after the same per-view affine
color fit as compare_photometric.py (fit over object + mat known region). Also 3DGS on the same pixels.
Usage: uv run python scripts/model/crt/part_psnr.py outputs/runs/<run> [prefix=crt.]"""
import json, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, 'scripts')
from eval.compare_photometric import affine_fit
run = Path(sys.argv[1]); pre = sys.argv[2] if len(sys.argv) > 2 else 'crt.'
idm = json.loads((run / 'id_map.json').read_text())
keys = np.array([(c[0] << 16) | (c[1] << 8) | c[2] for n, c in idm.items() if n.startswith(pre)])
acc = {k: [0.0, 0] for k in ('raw', 'fit', 'fit_local', 'gs', 'gs_fit', 'gs_fit_local')}
for f in sorted((run / 'rgb').glob('*.png')):
    n = f.stem
    rgba = cv2.imread(str(f), cv2.IMREAD_UNCHANGED); model = (rgba[..., :3] * (rgba[..., 3:4] / 255.0))
    photo = cv2.imread(f'data/undistorted_s4/{n}.jpg', cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
    pm = cv2.imread(f'data/masks_s4/{n}.png', 0)
    i = cv2.imread(str(run / 'id' / f.name), cv2.IMREAD_UNCHANGED).astype(np.int64)
    k = (i[..., 2] << 16) | (i[..., 1] << 8) | i[..., 0]
    m = np.isin(k, keys) & (i[..., 3] > 0)
    m = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if m.sum() < 200: continue
    known = (pm != 128) & (rgba[..., 3] > 0)
    gs_p = Path(f'outputs/baseline_3dgs/s4/{n}.png')
    imgs = {'raw': model, 'fit': affine_fit(model, photo, known), 'fit_local': affine_fit(model, photo, m)}
    if gs_p.exists():
        gs = cv2.imread(str(gs_p)).astype(float); imgs['gs'] = gs; imgs['gs_fit'] = affine_fit(gs, photo, known)
        imgs['gs_fit_local'] = affine_fit(gs, photo, m)
    row = [n, int(m.sum())]
    for key, img in imgs.items():
        se = ((img - photo.astype(float))[m] ** 2).mean(); acc[key][0] += se * m.sum(); acc[key][1] += m.sum()
        row.append(f'{key} {10 * np.log10(255 ** 2 / se):.2f}')
    print(*row)
print('ALL', {k: round(10 * np.log10(255 ** 2 / (s / c)), 2) for k, (s, c) in acc.items() if c})
