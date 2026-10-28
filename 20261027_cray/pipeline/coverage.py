"""Coverage check: how much scan ink the trace misses, and how much trace ink lies off the scan.

Usage: python3 -I pipeline/coverage.py <pn> <fig-id> [--tol 4] [--minpix 60]
Renders the trace at scale 1 (via overlay.py), compares against the crop with a `tol` px tolerance,
prints coverage percentages and the largest MISSING / EXTRA clusters (bbox in crop px, ready for
`overlay.py --region`), and writes work/<pn>/check/fig-<id>.diff.png (grey scan, blue missing, red extra).
Pattern fills (hatch/stipple/dashes) show as residual MISSING by design.
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pn")
    ap.add_argument("fig")
    ap.add_argument("--tol", type=int, default=4)
    ap.add_argument("--minpix", type=int, default=60)
    a = ap.parse_args()
    subprocess.run(["python3", "-I", str(ROOT / "pipeline" / "overlay.py"), a.pn, a.fig, "--scale", "1"],
                   check=True, capture_output=True)
    scan = np.array(Image.open(ROOT / "work" / a.pn / "figs" / f"fig-{a.fig}.png").convert("L")) < 128
    tr = np.array(Image.open(ROOT / "work" / a.pn / "check" / f"fig-{a.fig}.trace.png").convert("L"))
    tr = tr[: scan.shape[0], : scan.shape[1]] < 128
    st = np.ones((2 * a.tol + 1, 2 * a.tol + 1), bool)
    miss = scan & ~ndi.binary_dilation(tr, st)
    extra = tr & ~ndi.binary_dilation(scan, st)
    print(f"scan ink covered: {100 * (1 - miss.sum() / max(scan.sum(), 1)):.1f}%   "
          f"trace ink off scan: {100 * extra.sum() / max(tr.sum(), 1):.1f}%")
    for name, m in (("MISSING", miss), ("EXTRA", extra)):
        lab, _ = ndi.label(ndi.binary_closing(m, np.ones((5, 5))), structure=np.ones((3, 3)))
        rows = []
        for i, sl in enumerate(ndi.find_objects(lab)):
            c = int((m[sl] & (lab[sl] == i + 1)).sum())
            if c >= a.minpix:  # ignore isolated specks
                rows.append((c, sl[1].start, sl[0].start, sl[1].stop, sl[0].stop))
        rows.sort(reverse=True)
        print(f"{name}: {len(rows)} clusters >= {a.minpix} px, {int(m.sum())} px total")
        for r in rows[:40]:
            print(f"  {r[0]:6d} px  region {r[1]},{r[2]},{r[3]},{r[4]}")
    out = np.full(scan.shape + (3,), 255, np.uint8)
    out[scan] = (200, 200, 200)
    out[miss] = (0, 0, 255)
    out[extra] = (255, 0, 0)
    dst = ROOT / "work" / a.pn / "check" / f"fig-{a.fig}.diff.png"
    Image.fromarray(out).save(dst)
    print(dst)


if __name__ == "__main__":
    main()
