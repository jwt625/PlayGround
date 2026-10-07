"""Holdout / train / probe split, constrained by capture session and subject (run before any measuring).

Holdout: farthest-point sampling of view directions and centers (as in the CRT run), drawn per session with
quotas from config/scene.yaml (split.holdout_quota), plus at least split.min_package_holdout views from the
package view list. Views with fewer than split.min_obs sparse observations are never held out (they are weakly
registered and would make the holdout noisy). Probe: the same sampling over train views with quotas
(split.probe_quota) and at least split.min_package_probe package views.
Output: data/split.json {"holdout", "train", "probe", "excluded", "sessions"}. Views in split.exclude_views (bad poses)
are in no set. Refuses to overwrite (holdout must not move
once measuring has started) unless --force.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, scene_cfg  # noqa: E402
from prep.colmap_io import read_model  # noqa: E402


def session_of(stem: str, sessions: dict) -> str:
    k = int(stem[4:8])
    for s, (a, b) in sessions.items():
        if int(a[4:8]) <= k <= int(b[4:8]):
            return s
    raise ValueError(stem)


def fps(feat: dict[str, np.ndarray], cand: list[str], k: int, chosen: list[str]) -> list[str]:
    """Add k views from cand, each farthest (feature space) from all chosen so far."""
    out = []
    for _ in range(k):
        pool = [c for c in cand if c not in chosen + out]
        if not pool:
            break
        ref = chosen + out
        if not ref:
            c = np.mean([feat[n] for n in pool], 0)
            out.append(min(pool, key=lambda n: np.linalg.norm(feat[n] - c)))
            continue
        out.append(max(pool, key=lambda n: min(np.linalg.norm(feat[n] - feat[r]) for r in ref)))
    return out


def main() -> None:
    cfg = scene_cfg()
    sp = cfg["split"]
    out = DATA / "split.json"
    if out.exists() and "--force" not in sys.argv:
        raise SystemExit(f"{out} exists; the holdout must not change after measuring started (--force to redo)")
    _, imgs, pts = read_model(cfg["sparse_dir"])
    nobs = {Path(im.name).stem: int((im.point3d_ids >= 0).sum()) for im in imgs.values()}
    C = {Path(im.name).stem: -im.R.T @ im.tvec for im in imgs.values()}
    D = {Path(im.name).stem: im.R[2] for im in imgs.values()}  # optical axis in the model frame
    cmed = np.median(np.array(list(C.values())), 0)
    rad = np.median([np.linalg.norm(c - cmed) for c in C.values()])
    feat = {n: np.hstack([D[n], (C[n] - cmed) / rad]) for n in C}
    names = sorted(n for n in C if n not in set(sp.get("exclude_views", [])))
    sess = {n: session_of(n, cfg["sessions"]) for n in names}
    pkg = set(sp["package_views"])
    eligible = [n for n in names if nobs[n] >= sp["min_obs"]]

    hold: list[str] = fps(feat, [n for n in eligible if n in pkg], sp["min_package_holdout"], [])
    for s, q in sp["holdout_quota"].items():
        have = sum(sess[h] == s for h in hold)
        hold += fps(feat, [n for n in eligible if sess[n] == s], max(0, q - have), hold)
    train = [n for n in names if n not in hold]
    ptrain = [n for n in train if nobs[n] >= sp["min_obs"]]
    probe: list[str] = fps(feat, [n for n in ptrain if n in pkg], sp["min_package_probe"], [])
    for s, q in sp["probe_quota"].items():
        have = sum(sess[p] == s for p in probe)
        probe += fps(feat, [n for n in ptrain if sess[n] == s], max(0, q - have), probe)
    res = {"holdout": sorted(hold), "train": train, "probe": sorted(probe), "excluded": sp.get("exclude_views", []),
           "sessions": {s: [n for n in names if sess[n] == s] for s in cfg["sessions"]}}
    out.write_text(json.dumps(res, indent=1))
    for k in ("holdout", "probe"):
        print(k, [(n, sess[n], "pkg" if n in pkg else "", nobs[n]) for n in res[k]])
    print(f"{len(hold)} holdout, {len(train)} train, {len(probe)} probe")


if __name__ == "__main__":
    main()
