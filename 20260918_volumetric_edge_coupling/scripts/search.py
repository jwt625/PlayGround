"""Quick simulation-based search for the unknown absolute geometry.

The paper omits the oxide gap, cavity depth and interaction endpoints
(PAPER_REVIEW s5).  This scans a bounded family at 1550 nm on a coarse mesh to
locate a physically sensible nominal, then refines.  It is explicitly a fit:
the pre-fit baseline and all candidates are logged.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

from src.config import CouplerConfig
from src.geometry import make_plan
from src.ports import run_incident_reference
from src.run import run_coupler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/nominal.yaml")
    ap.add_argument("--res", type=float, default=20.0)
    ap.add_argument("--until", type=float, default=40.0)
    ap.add_argument("--out", default="results/search/scan.csv")
    ap.add_argument("--depth", type=float, nargs="+", default=[12.0, 13.0, 14.0, 15.0])
    ap.add_argument("--gref", type=float, nargs="+", default=[10.0, 30.0, 60.0, 100.0])
    ap.add_argument("--length", type=float, nargs="+", default=[12.0])
    args = ap.parse_args()

    base = CouplerConfig.load(args.config)
    base.numerics.resolution = args.res
    base.numerics.until_after_sources = args.until

    # incident reference computed once (geometry-independent enough for search)
    ref_cfg = CouplerConfig.load(args.config)
    ref_cfg.numerics.resolution = args.res
    ref_cfg.numerics.until_after_sources = args.until
    ref = run_incident_reference(ref_cfg, make_plan(ref_cfg))
    p_inc = ref["flux"][25]
    print(f"reference P_inc(1550) = {p_inc:.4g}")

    rows = []
    combos = list(
        itertools.product(args.depth, args.gref, args.length)
    )
    for depth, gref, length in combos:
        cfg = CouplerConfig.load(args.config)
        cfg.numerics.resolution = args.res
        cfg.numerics.until_after_sources = args.until
        cfg.geometry.depth_um = depth
        cfg.geometry.g_ref_nm = gref
        cfg.geometry.stack_length_um = length
        errors = cfg.validate()
        if errors:
            rows.append({"depth": depth, "g_ref": gref,
                         "length": length, "eta": None, "error": "; ".join(errors)})
            print(f"d={depth} g={gref} L={length}: invalid {errors}")
            continue
        t0 = time.time()
        try:
            res = run_coupler(cfg, compute_reference=False)
            eta = float(np.real(res["P_out"][25]) / p_inc)
            rows.append({"depth": depth, "g_ref": gref,
                         "length": length, "eta": eta, "error": None,
                         "gap_impact_nm": cfg.geometry.gap_at(
                             cfg.geometry.s_impact(cfg.source.y_source_um)) * 1000,
                         "runtime_s": time.time() - t0})
            print(f"d={depth} g={gref} L={length}: eta={eta:.4f} "
                  f"gap_impact={rows[-1]['gap_impact_nm']:.0f}nm ({time.time()-t0:.0f}s)")
        except Exception as exc:  # noqa: BLE001
            rows.append({"depth": depth, "g_ref": gref,
                         "length": length, "eta": None, "error": str(exc)})
            print(f"d={depth} g={gref} L={length}: FAILED {exc}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    import csv

    keys = sorted({k for r in rows for k in r})
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    best = max(rows, key=lambda r: (r["eta"] is not None, r["eta"] or -1))
    print("best:", json.dumps(best))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
