"""Batch driver: alpha / Bernstein free-form / staircase optimizations for uLED and uVCSEL sources.

Stages (each job writes outputs/runs/<job_id>.json; existing results are skipped):
  1  alpha: alpha-profile optimum per (source, fixed constraint) and per Pareto grid point
  2  bernstein, staircase N: CMA-ES from the stage-1 alpha optimum
  3  final: re-evaluate every design for both sources (cross-evaluation), n_nodes_eval, plus dr/2 check

  1r, 2r: same as 1, 2 with the robust objective sqrt(mean sigma^2) over the config `robust` ensemble

Usage: uv run python scripts/run_study.py [--stage 1|2|1r|2r|3|robust|all] [--only SUBSTR] [--dry-run] [--workers N]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grinopt import material  # noqa: E402
from grinopt.evaluate import Numerics, Source, bandwidth_length_mhz_km, etendue, evaluate  # noqa: E402
from grinopt.profiles import (  # noqa: E402
    alpha_shape,
    bernstein_coeffs,
    bernstein_shape,
    fit_bernstein_params,
    staircase_init_from_shape,
    staircase_layers,
    staircase_shape,
)

RUNS = ROOT / "outputs" / "runs"
LOGS = ROOT / "logs"


def load_cfg() -> dict:
    return yaml.safe_load((ROOT / "config" / "study.yaml").read_text())


def numerics(cfg: dict, scale: float = 1.0) -> Numerics:
    n = cfg["numerics"]
    return Numerics(n["dr_um"] * scale, n["clad_margin_um"], n["oversample"])


def source(cfg: dict, name: str, phase: str) -> Source:
    s = cfg["sources"][name]
    return Source(s["lam0_um"], s["fwhm_nm"], s["n_nodes_opt"] if phase == "opt" else s["n_nodes_eval"])


def shape_from(design: dict):
    kind = design["kind"]
    if kind == "alpha":
        al = design["alpha"]
        return lambda q: alpha_shape(q, al)
    if kind == "bernstein":
        c = np.array(design["coeffs"])
        return lambda q: bernstein_shape(q, c)
    if kind == "staircase":
        radii, levels = np.array(design["radii"]), np.array(design["levels"])
        return lambda q: staircase_shape(q, radii, levels)
    if kind == "step":
        return lambda q: np.zeros_like(q)
    raise ValueError(kind)


# ---------------------------------------------------------------- jobs

def job_list(cfg: dict, stage: str) -> list[dict]:
    """Stages 1/2 strict objective, 1r/2r robust objective (suffix _robust on the kind), 3 final evaluation."""
    jobs: list[dict] = []
    robust = stage.endswith("r")
    sfx = "_robust" if robust else ""
    obj = {"objective": "robust" if robust else "strict"}
    stage = stage.rstrip("r") if stage != "3" else stage
    for src in cfg["sources"]:
        for cname, c in cfg["constraints"].items():
            if not c.get("enabled", True):
                continue
            if c["mode"] == "fixed":
                a = c["core_diameter_um"] / 2
                base = {"source": src, "constraint": cname, "a_um": a, "na": c["na"], **obj}
                if stage == "1":
                    jobs.append({**base, "id": f"{src}__{cname}__alpha{sfx}", "kind": "alpha"})
                if stage == "2":
                    for seed in range(1, cfg["optimizer"]["n_starts"] + 1):
                        sd = {**base, "seed": seed}
                        jobs.append({**sd, "id": f"{src}__{cname}__bernstein{sfx}__s{seed}", "kind": "bernstein"})
                        jobs.append({**sd, "id": f"{src}__{cname}__bernsteinstep{sfx}__s{seed}", "kind": "bernstein",
                                     "edge_step": True})
                        for nl in cfg["parameterizations"]["staircase"]["n_layers"]:
                            jobs.append({**sd, "id": f"{src}__{cname}__stair{nl:02d}{sfx}__s{seed}",
                                         "kind": "staircase", "n_layers": nl})
            elif c["mode"] == "pareto" and stage == "1":
                for d in c["core_diameter_um"]:
                    for na in c["na"]:
                        jobs.append({"source": src, "constraint": cname, "a_um": d / 2, "na": na, "kind": "alpha",
                                     **obj, "id": f"{src}__{cname}__d{d:05.1f}_na{na:.2f}__alpha{sfx}"})
    if stage == "3":
        groups: dict[str, list[dict]] = {}
        for p in sorted(RUNS.glob("*.json")):
            if p.name.startswith("final__"):
                continue
            r = json.loads(p.read_text())
            if r.get("constraint", "").startswith("C_"):
                continue
            groups.setdefault(re.sub(r"__s\d+$", "", r["id"]), []).append(r)
        for gid, rs in groups.items():
            best = min(rs, key=lambda r: r["sigma_opt_ps_per_m"])
            jobs.append({"id": f"final__{gid}", "kind": "final", "design_id": best["id"],
                         "seed_sigmas": {r["id"]: r["sigma_opt_ps_per_m"] for r in rs}})
        for src in cfg["sources"]:
            for cname, c in cfg["constraints"].items():
                if c.get("enabled", True) and c["mode"] == "fixed":
                    jobs.append({"id": f"final__{src}__{cname}__step", "kind": "final", "design_id": None,
                                 "step": {"source": src, "constraint": cname, "a_um": c["core_diameter_um"] / 2,
                                          "na": c["na"]}})
    # longest first: uled before uvcsel, larger cores and more layers first
    def cost(j: dict) -> float:
        a = j.get("a_um", 30.0)
        lam = 0.43 if j.get("source", "uled") == "uled" else 0.85
        dim = {"alpha": 1, "bernstein": 8}.get(j["kind"], 2 * j.get("n_layers", 4))
        return (a * j.get("na", 0.2) / lam) ** 2 * dim * (3 if j.get("objective") == "robust" else 1)

    return sorted(jobs, key=cost, reverse=True)


def ensemble(cfg: dict, src_name: str, src: Source) -> list[tuple[float, Source]]:
    """Robust-objective members: (x0 scale, source with shifted center wavelength)."""
    rb = cfg["robust"]
    return [(sc, Source(src.lam0_um + dl * 1e-3, src.fwhm_nm, src.n_nodes))
            for sc in rb["delta_scale"] for dl in rb["lam_shift_nm"][src_name]]


def sigma_robust(a: float, x0: float, shape, members: list[tuple[float, Source]], num: Numerics
                 ) -> tuple[float, list[float], float]:
    """sqrt(mean sigma^2) over members; also per-member sigmas and min mode count."""
    sig, mc = [], np.inf
    for sc, s in members:
        r = evaluate(a, x0 * sc, shape, s, num)
        sig.append(r.sigma_modal_ps_per_m)
        mc = min(mc, r.mode_count_center)
    return float(np.sqrt(np.mean(np.square(sig)))), [float(v) for v in sig], float(mc)


def objective_factory(a: float, x0: float, src: Source, num: Numerics, cfg: dict, ref_alpha: float | None = None,
                      members: list[tuple[float, Source]] | None = None):
    """sigma_modal (strict) or sqrt(mean sigma^2) over robust members; if ref_alpha is given and the floor is enabled,
    add a penalty when the nominal etendue falls below the alpha = ref_alpha profile (coupling floor)."""
    cpl = cfg["coupling_floor"]
    use_floor = cpl["enabled"] and ref_alpha is not None
    g_ref = etendue(a, x0, lambda q: alpha_shape(q, ref_alpha), src.lam0_um, num) if use_floor else 0.0

    def f(shape) -> float:
        if use_floor:
            deficit = max(0.0, 1.0 - cpl["tolerance"] - etendue(a, x0, shape, src.lam0_um, num) / g_ref)
            if deficit > 0.05:
                return BAD + cpl["penalty_ps_per_m"] * deficit
        else:
            deficit = 0.0
        if members:
            sig, _, mc = sigma_robust(a, x0, shape, members, num)
        else:
            r = evaluate(a, x0, shape, src, num)
            sig, mc = r.sigma_modal_ps_per_m, r.mode_count_center
        if not np.isfinite(sig) or mc < 2:
            return BAD
        return sig + cpl["penalty_ps_per_m"] * deficit
    return f


BAD = 1e3


def run_alpha(job: dict, cfg: dict) -> dict:
    from scipy.optimize import minimize_scalar

    src = source(cfg, job["source"], "opt")
    x0 = material.x_for_na(src.lam0_um, job["na"])
    members = ensemble(cfg, job["source"], src) if job.get("objective") == "robust" else None
    f = objective_factory(job["a_um"], x0, src, numerics(cfg), cfg, members=members)
    lo, hi = cfg["parameterizations"]["alpha"]["bounds"]
    grid = np.round(np.arange(lo, hi + 1e-9, 0.01), 4)
    sig = [f(lambda q, al=al: alpha_shape(q, al)) for al in grid]
    i = int(np.argmin(sig))
    a0, a1 = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    res = minimize_scalar(lambda al: f(lambda q: alpha_shape(q, al)), bounds=(a0, a1), method="bounded",
                          options={"xatol": 1e-4})
    best_al, best = (float(res.x), float(res.fun)) if res.fun < sig[i] else (float(grid[i]), float(sig[i]))
    return {"design": {"kind": "alpha", "alpha": best_al}, "sigma_opt_ps_per_m": best,
            "alpha_scan": {"alpha": grid.tolist(), "sigma_ps_per_m": [float(s) for s in sig]}, "x0": x0}


def run_cma(job: dict, cfg: dict) -> dict:
    import cma

    src = source(cfg, job["source"], "opt")
    x0 = material.x_for_na(src.lam0_um, job["na"])
    robust = job.get("objective") == "robust"
    sfx = "_robust" if robust else ""
    alpha_run = json.loads((RUNS / f"{job['source']}__{job['constraint']}__alpha{sfx}.json").read_text())
    al = alpha_run["design"]["alpha"]
    members = ensemble(cfg, job["source"], src) if robust else None
    f = objective_factory(job["a_um"], x0, src, numerics(cfg), cfg, ref_alpha=al, members=members)
    pz = cfg["parameterizations"]
    if job["kind"] == "bernstein":
        rho = np.linspace(0, 1, 400)
        p0 = fit_bernstein_params(alpha_shape(rho, al), rho, pz["bernstein"]["degree"])
        edge = bool(job.get("edge_step", False))
        if edge:
            p0 = np.concatenate([p0, [pz["bernstein"]["edge_logit_init"]]])
        maxfev = pz["bernstein"]["maxfev"]

        def decode(p):
            return {"kind": "bernstein", "edge_step": edge, "coeffs": bernstein_coeffs(p, edge_step=edge).tolist()}
    else:
        nl = job["n_layers"]
        lev0, rad0 = staircase_init_from_shape(lambda q: alpha_shape(q, al), nl)
        free = pz["staircase"]["free_radii"]
        p0 = np.concatenate([lev0, rad0]) if free else lev0
        maxfev = pz["staircase"]["maxfev_per_param"] * p0.size

        def decode(p):
            radii, levels = staircase_layers(p[:nl], p[nl:] if free else None, nl)
            return {"kind": "staircase", "n_layers": nl, "radii": radii.tolist(), "levels": levels.tolist()}

    def obj(p):
        return f(shape_from(decode(np.asarray(p))))

    from scipy.optimize import minimize

    best = {"f": np.inf, "p": p0, "n": 0}

    def tracked(p):
        v = obj(p)
        best["n"] += 1
        if v < best["f"]:
            best["f"], best["p"] = v, np.array(p, dtype=float)
        return v

    seed = int(job.get("seed", 1))
    ch = cfg["optimizer"]
    if seed > 1:  # perturbed start for restarts
        p0 = p0 + np.random.default_rng(seed).normal(0.0, ch["restart_perturb"], p0.size)
    f_init = tracked(p0)
    es = cma.CMAEvolutionStrategy(p0, ch["cma_sigma0"], {"maxfevals": int(ch["cma_frac"] * maxfev), "verbose": -9,
                                                          "seed": seed, "tolfun": 1e-7})
    es.optimize(tracked)
    n_cma = best["n"]
    minimize(tracked, best["p"], method="Powell",
             options={"maxfev": int(ch["powell_frac"] * maxfev), "xtol": 1e-4, "ftol": 1e-7})
    n_pow = best["n"]
    minimize(tracked, best["p"], method="Nelder-Mead",
             options={"maxfev": max(maxfev - best["n"], 50), "xatol": 1e-5, "fatol": 1e-7, "adaptive": True})
    pb, fb = best["p"], best["f"]
    return {"design": decode(np.asarray(pb)), "sigma_opt_ps_per_m": float(fb), "sigma_init_ps_per_m": float(f_init),
            "alpha_init": al, "nfev": best["n"], "nfev_stages": [n_cma, n_pow - n_cma, best["n"] - n_pow],
            "params": np.asarray(pb).tolist(), "x0": x0}


def metrics_dict(r, length_m: float) -> dict:
    bl = bandwidth_length_mhz_km(r.tau_by_lam, r.w_by_lam, r.s_by_lam)
    m = r.modes_center
    bl_modal = bandwidth_length_mhz_km([m.tau_ps_per_m], [m.degeneracy], np.array([1.0]))
    inner = m.principal <= 0.8 * m.principal.max()
    wi, ti = m.degeneracy[inner], m.tau_ps_per_m[inner]
    sig_inner = float(np.sqrt(np.sum(wi * (ti - np.sum(wi * ti) / wi.sum()) ** 2) / wi.sum()))
    return {"sigma_modal_ps_per_m": r.sigma_modal_ps_per_m,
            "sigma_modal_center_ps_per_m": r.sigma_modal_center_ps_per_m,
            "sigma_total_ps_per_m": r.sigma_total_ps_per_m,
            "max_dmd_center_ps_per_m": r.max_dmd_center_ps_per_m,
            "mode_count_center": r.mode_count_center, "etendue_um2_sr": r.etendue_um2_sr,
            "sigma_inner80_center_ps_per_m": sig_inner,
            "bl_total_mhz_km": bl, "bl_modal_center_mhz_km": bl_modal,
            "bw_total_ghz_at_L": bl / (length_m * 1e-3) * 1e-3,
            "bw_modal_ghz_at_L": bl_modal / (length_m * 1e-3) * 1e-3}


def run_final(job: dict, cfg: dict) -> dict:
    if job["design_id"] is None:
        st = job["step"]
        design = {"kind": "step"}
        a, x0, home = st["a_um"], material.x_for_na(cfg["sources"][st["source"]]["lam0_um"], st["na"]), st["source"]
        meta = {"source": home, "constraint": st["constraint"], "a_um": a, "na": st["na"]}
    else:
        d = json.loads((RUNS / f"{job['design_id']}.json").read_text())
        design, a, x0, home = d["design"], d["a_um"], d["x0"], d["source"]
        meta = {k: d[k] for k in ("source", "constraint", "a_um", "na")}
    length = cfg["link"]["length_m"]
    out: dict = {"design": design, "x0": x0, **meta, "home_source": home, "eval": {}}
    for sname in cfg["sources"]:
        src = source(cfg, sname, "eval")
        r = evaluate(a, x0, shape_from(design), src, numerics(cfg))
        md = metrics_dict(r, length)
        rf = evaluate(a, x0, shape_from(design), src, numerics(cfg, 0.5))
        md["sigma_modal_ps_per_m_dr_half"] = rf.sigma_modal_ps_per_m
        mem = ensemble(cfg, sname, src)
        srob, smem, _ = sigma_robust(a, x0, shape_from(design), mem, numerics(cfg))
        md["sigma_robust_ps_per_m"] = srob
        md["robust_members"] = [{"delta_scale": sc, "lam_um": s_.lam0_um, "sigma_ps_per_m": v}
                                for (sc, s_), v in zip(mem, smem, strict=True)]
        m = r.modes_center
        md["modes_center"] = {"l": m.l.tolist(), "m": m.m.tolist(), "tau_ps_per_m": m.tau_ps_per_m.tolist()}
        out["eval"][sname] = md
    return out


def run_job(job: dict) -> dict:
    cfg = load_cfg()
    t = time.time()
    fn = {"alpha": run_alpha, "bernstein": run_cma, "staircase": run_cma, "final": run_final}[job["kind"]]
    res = fn(job, cfg)
    res = {**{k: v for k, v in job.items() if k not in ("step",)}, **res, "wall_s": time.time() - t,
           "finished": datetime.now().isoformat(timespec="seconds")}
    (RUNS / f"{job['id']}.json").write_text(json.dumps(res))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    ap.add_argument("--only", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    cfg = load_cfg()
    RUNS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = (LOGS / f"run_study_{stamp}.jsonl").open("a")

    def note(**kw):
        kw["ts"] = datetime.now().isoformat(timespec="seconds")
        log.write(json.dumps(kw) + "\n")
        log.flush()
        print(json.dumps(kw), flush=True)

    stages = {"all": ["1", "2", "1r", "2r", "3"], "robust": ["1r", "2r", "3"]}.get(args.stage, [args.stage])
    workers = args.workers or cfg["run"]["workers"]
    for st in stages:
        jobs = [j for j in job_list(cfg, st) if args.only in j["id"]]
        todo = [j for j in jobs if not (RUNS / f"{j['id']}.json").exists()]
        note(event="stage", stage=st, total=len(jobs), todo=len(todo))
        if args.dry_run:
            for j in todo:
                note(event="would_run", id=j["id"])
            continue
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(run_job, j): j for j in todo}
            for fu in as_completed(futs):
                j = futs[fu]
                try:
                    r = fu.result()
                    note(event="done", id=j["id"], wall_s=round(r["wall_s"], 1),
                         sigma=r.get("sigma_opt_ps_per_m"))
                except Exception:
                    note(event="failed", id=j["id"], error=traceback.format_exc()[-800:])


if __name__ == "__main__":
    main()
