"""Interactive HTML report from outputs/runs. Writes outputs/report/grin_report.html (data embedded, no build step).

Re-run any time; it embeds whatever results exist and shows per-stage job completion.
Usage: uv run python scripts/make_report.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grinopt import material  # noqa: E402
from grinopt.evaluate import Numerics, Source, etendue, evaluate  # noqa: E402
from scripts.run_study import RUNS, job_list, load_cfg, shape_from  # noqa: E402

OUT = ROOT / "outputs" / "report" / "grin_report.html"
TEMPLATE = ROOT / "scripts" / "report_template.html"


def sig(x: float, n: int = 4) -> float:
    return float(f"{x:.{n}g}") if np.isfinite(x) else None  # type: ignore[return-value]


def parse_gid(gid: str) -> dict:
    src, cons, kind = gid.split("__")[:3]
    obj = "robust" if kind.endswith("_robust") else "strict"
    kind = kind.removesuffix("_robust")
    n = int(kind[5:]) if kind.startswith("stair") else None
    return {"src": src, "cons": cons, "obj": obj, "kind": "stair" if n else kind, "n": n}


def label(p: dict) -> str:
    if p["kind"] == "stair":
        return f"staircase N={p['n']}"
    return {"alpha": "alpha", "bernstein": "Bernstein continuous", "bernsteinstep": "Bernstein edge step",
            "step": "step index"}[p["kind"]]


def stage_status(cfg: dict) -> list[dict]:
    out = []
    for st in ("1", "2", "1r", "2r"):
        ids = [j["id"] for j in job_list(cfg, st)]
        out.append({"stage": st, "done": sum((RUNS / f"{i}.json").exists() for i in ids), "total": len(ids)})
    fin = len(list(RUNS.glob("final__*.json")))
    out.append({"stage": "3 final", "done": fin, "total": None})
    return out


def num(cfg: dict) -> Numerics:
    return Numerics(**{k: cfg["numerics"][k] for k in ("dr_um", "clad_margin_um", "oversample")})


def display_eval(d: dict, lam: float, cfg: dict) -> dict:
    """Profile and center-wavelength mode delays for plotting."""
    a, x0, shape = d["a_um"], d["x0"], shape_from(d["design"])
    r = np.linspace(0.0, a + 6.0, 700)
    x = x0 * (1 - shape(np.clip(r / a, 0, 1)))
    x[r >= a] = 0.0
    dn = (material.index(lam, np.clip(x, 0, None)) - material.index(lam, 0.0)) * 1e3
    res = evaluate(a, x0, shape, Source(lam, 0.0, 1), num(cfg))
    m = res.modes_center
    t = m.tau_ps_per_m
    t0 = t[(m.l == 0) & (m.m == 1)][0]
    return {"r": [round(v, 3) for v in r.tolist()], "dn": [round(v, 5) for v in dn.tolist()],
            "p": m.principal.tolist(), "l": m.l.tolist(), "dtau": [round(v, 5) for v in (t - t0).tolist()],
            "sigma_center": sig(res.sigma_modal_ps_per_m), "modes": res.mode_count_center,
            "etendue": sig(res.etendue_um2_sr)}


def main() -> None:
    cfg = load_cfg()
    runs = {p.stem: json.loads(p.read_text()) for p in sorted(RUNS.glob("*.json")) if not p.stem.startswith("final__")}
    finals = {p.stem.removeprefix("final__"): json.loads(p.read_text()) for p in sorted(RUNS.glob("final__*.json"))}

    # alpha scans (fixed constraints)
    scans = []
    for k, r in runs.items():
        if k.endswith("__alpha") or k.endswith("__alpha_robust"):
            p = parse_gid(k)
            if p["cons"].startswith("C_"):
                continue
            sc = r["alpha_scan"]
            scans.append({**p, "alpha": sc["alpha"], "sigma": [sig(v) for v in sc["sigma_ps_per_m"]],
                          "best_alpha": sig(r["design"]["alpha"], 5), "best_sigma": sig(r["sigma_opt_ps_per_m"])})

    # group optimization runs by design (seeds)
    groups: dict[str, list[dict]] = {}
    for k, r in runs.items():
        p = parse_gid(k) if not k.split("__")[1].startswith("C_") else None
        if p is None:
            continue
        groups.setdefault(re.sub(r"__s\d+$", "", k), []).append(r)
    layers = []
    for gid, rs in groups.items():
        p = parse_gid(gid)
        layers.append({**p, "gid": gid, "seeds": [sig(r["sigma_opt_ps_per_m"]) for r in rs],
                       "seed_ids": [r["id"] for r in rs]})

    # pareto
    pareto = []
    for k, r in runs.items():
        mt = re.match(r"(\w+?)__C_pareto__d([\d.]+)_na([\d.]+)__alpha(_robust)?$", k)
        if not mt:
            continue
        lam = cfg["sources"][mt.group(1)]["lam0_um"]
        d, na = float(mt.group(2)), float(mt.group(3))
        g = etendue(d / 2, r["x0"], shape_from(r["design"]), lam, num(cfg))
        pareto.append({"src": mt.group(1), "obj": "robust" if mt.group(4) else "strict", "d": d, "na": na,
                       "alpha": sig(r["design"]["alpha"], 5), "sigma": sig(r["sigma_opt_ps_per_m"]),
                       "etendue": sig(g)})

    # designs: best seed per group + step-index baselines
    designs = []
    for gid, rs in groups.items():
        best = min(rs, key=lambda r: r["sigma_opt_ps_per_m"])
        p = parse_gid(gid)
        lam = cfg["sources"][p["src"]]["lam0_um"]
        de = display_eval(best, lam, cfg)
        fin = finals.get(gid)
        designs.append({**p, "gid": gid, "label": label(p), "a": best["a_um"], "na": best["na"], "x0": best["x0"],
                        "design": best["design"], "objective_value": sig(best["sigma_opt_ps_per_m"]),
                        "n_seeds": len(rs), **de, "final": slim_final(fin)})
    for src in cfg["sources"]:
        for cons, c in cfg["constraints"].items():
            if c["mode"] != "fixed" or not c.get("enabled", True):
                continue
            lam = cfg["sources"][src]["lam0_um"]
            st = {"a_um": c["core_diameter_um"] / 2, "x0": material.x_for_na(lam, c["na"]), "design": {"kind": "step"}}
            p = {"src": src, "cons": cons, "obj": "strict", "kind": "step", "n": None}
            gid = f"{src}__{cons}__step"
            designs.append({**p, "gid": gid, "label": "step index", "a": st["a_um"], "na": c["na"], "x0": st["x0"],
                            "design": st["design"], "objective_value": None, "n_seeds": 0,
                            **display_eval(st, lam, cfg), "final": slim_final(finals.get(gid))})

    # sigma vs wavelength, 50 um / NA 0.2 designs
    lams = np.round(np.arange(0.40, 0.9501, 0.001), 4)
    svl = {"lam_nm": (lams * 1e3).round(1).tolist(), "series": []}
    cache_p = OUT.parent / "cache_sigma_vs_lambda.json"
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    for dsg in designs:
        if dsg["cons"] == "A_om3" and dsg["kind"] in ("alpha", "bernsteinstep", "step"):
            if dsg["kind"] == "step" and dsg["src"] != "uled":
                continue
            key = json.dumps([dsg["a"], dsg["x0"], dsg["design"], lams[0], lams[-1], len(lams)], sort_keys=True)
            if key not in cache:
                cache[key] = [sig(evaluate(dsg["a"], dsg["x0"], shape_from(dsg["design"]), Source(float(lm), 0.0, 1),
                                           num(cfg)).sigma_modal_ps_per_m) for lm in lams]
                cache_p.parent.mkdir(parents=True, exist_ok=True)
                cache_p.write_text(json.dumps(cache))
            s = cache[key]
            svl["series"].append({"gid": dsg["gid"], "src": dsg["src"], "obj": dsg["obj"], "kind": dsg["kind"],
                                  "label": dsg["label"], "sigma": s})

    val = sorted((ROOT / "outputs" / "validation").glob("alpha_opt_*.json"))
    validation = json.loads(val[-1].read_text()) if val else []
    tests = subprocess.run(["uv", "run", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True).stdout
    tests_line = [ln for ln in tests.splitlines() if "passed" in ln or "failed" in ln][-1:] or ["not run"]

    data = {"generated": datetime.now().isoformat(timespec="seconds"), "status": stage_status(cfg), "config": cfg,
            "scans": scans, "layers": layers, "pareto": pareto, "designs": designs, "svl": svl,
            "validation": validation, "tests": tests_line[0]}
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB), designs={len(designs)}")


def slim_final(fin: dict | None) -> dict | None:
    if not fin:
        return None
    out = {"seed_sigmas": fin.get("seed_sigmas")}
    for s, e in fin["eval"].items():
        out[s] = {k: (sig(v) if isinstance(v, float) else v) for k, v in e.items()
                  if k not in ("modes_center", "robust_members")}
        out[s]["robust_members"] = e.get("robust_members")
    return out


if __name__ == "__main__":
    main()
