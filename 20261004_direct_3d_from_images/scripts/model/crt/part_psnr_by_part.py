"""Raw PSNR per crt part over all rendered views (ID-owned pixels eroded 2 px) and pixel share. Usage: uv run python scripts/model/crt/part_psnr_by_part.py outputs/runs/<run>"""
import json, sys, cv2, numpy as np
from pathlib import Path
run = Path(sys.argv[1]); idm = json.loads((run / 'id_map.json').read_text())
parts = {n: c for n, c in idm.items() if n.startswith('crt.')}
acc = {n: [0.0, 0] for n in parts}
for f in sorted((run / 'rgb').glob('*.png')):
    r = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)[..., :3].astype(float)
    p = cv2.imread(f'data/undistorted_s4/{f.stem}.jpg', cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR).astype(float)
    i = cv2.imread(str(run / 'id' / f.name), cv2.IMREAD_UNCHANGED).astype(int)
    for n, c in parts.items():
        m = (np.abs(i[..., 2] - c[0]) + np.abs(i[..., 1] - c[1]) + np.abs(i[..., 0] - c[2])) <= 3
        m = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        if m.sum():
            acc[n][0] += ((r[m] - p[m]) ** 2).sum() / 3; acc[n][1] += m.sum()
tot = sum(c for _, c in acc.values())
for n, (s, c) in sorted(acc.items(), key=lambda t: -t[1][0]):
    if c: print(f'{n:28s} px {c:7d} ({100*c/tot:4.1f}%)  psnr {10*np.log10(255**2/(s/c)):5.2f}  sse share {100*s/sum(a for a,_ in acc.values()):4.1f}%')
