"""Figures and summary table from outputs/runs (stage 1-3). Writes outputs/figures/*.png and outputs/summary.md."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grinopt import material  # noqa: E402
from grinopt.evaluate import Numerics, Source, etendue, evaluate  # noqa: E402
from scripts.run_study import RUNS, load_cfg, shape_from  # noqa: E402

FIG = ROOT / "outputs" / "figures"
plt.style.use("dark_background")
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150,
                     "savefig.bbox": "tight", "axes.prop_cycle": plt.cycler(color=[
                         "#4ea1ff", "#ff8c42", "#5fd068", "#e05a5a", "#b48cff", "#e6c84f", "#4fd1c5", "#c0c0c0"])})
SRC_LABEL = {"uled": "uLED 430 nm", "uvcsel": "uVCSEL 850 nm"}
CONS = ("A_om3", "B_large")
CONS_LABEL = {"A_om3": "50 um / NA 0.20", "B_large": "80 um / NA 0.28"}
OBJ = {"strict": "", "robust": "_robust"}


def load(prefix: str = "") -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted(RUNS.glob(f"{prefix}*.json"))}


def finals() -> dict[str, dict]:
    return {k.removeprefix("final__"): v for k, v in load("final__").items()}


def dn_profile(d: dict, lam: float, r: np.ndarray) -> np.ndarray:
    a, x0 = d["a_um"], d["x0"]
    x = x0 * (1 - shape_from(d["design"])(np.clip(r / a, 0, 1)))
    x[r >= a] = 0
    return material.index(lam, np.clip(x, 0, None)) - material.index(lam, 0.0)


def design_label(gid: str) -> str:
    kind = gid.split("__")[2].removesuffix("_robust")
    lab = {"alpha": "alpha", "bernstein": "Bernstein (continuous)", "bernsteinstep": "Bernstein (edge step)",
           "step": "step index"}.get(kind)
    if lab is None:
        lab = f"staircase N={int(kind.removeprefix('stair'))}"
    return lab


def fig_profiles(fin: dict, cfg: dict) -> None:
    for src in cfg["sources"]:
        for cons in CONS:
            fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.0), sharey=True)
            drawn = False
            for ax, (obj, sfx) in zip(axs, OBJ.items(), strict=True):
                keys = [f"{src}__{cons}__step"] + [f"{src}__{cons}__{k}{sfx}" for k in
                                                   ("alpha", "bernstein", "bernsteinstep", "stair04", "stair08",
                                                    "stair16")]
                keys = [k for k in keys if k in fin]
                if not keys:
                    continue
                drawn = True
                lam = cfg["sources"][src]["lam0_um"]
                a = fin[keys[0]]["a_um"]
                r = np.linspace(0, a + 5, 2000)
                for k in keys:
                    ax.plot(r, dn_profile(fin[k], lam, r) * 1e3, lw=1.1, label=design_label(k))
                ax.set_xlabel(f"r (um), {obj} objective")
                ax.legend(fontsize=6, frameon=False)
            axs[0].set_ylabel(f"n - n_SiO2 at {cfg['sources'][src]['lam0_um'] * 1e3:.0f} nm (1e-3)")
            if drawn:
                fig.savefig(FIG / f"profiles_{src}_{cons}.png")
            plt.close(fig)


def fig_alpha_scan(runs: dict, cfg: dict) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.0), sharey=True)
    for ax, src in zip(axs, cfg["sources"], strict=True):
        for ci, cons in enumerate(CONS):
            for obj, sfx in OBJ.items():
                k = f"{src}__{cons}__alpha{sfx}"
                if k not in runs:
                    continue
                sc = runs[k]["alpha_scan"]
                ax.plot(sc["alpha"], sc["sigma_ps_per_m"], lw=1.1, color=f"C{ci}", ls="-" if obj == "strict" else "--",
                        label=f"{CONS_LABEL[cons]}, {obj}")
        ax.set_yscale("log")
        ax.set_xlabel(f"alpha, {SRC_LABEL[src]}")
        ax.legend(fontsize=6, frameon=False)
    axs[0].set_ylabel("OFL RMS modal delay (ps/m)")
    fig.savefig(FIG / "alpha_scan.png")
    plt.close(fig)


def fig_layers(runs: dict, fin: dict, cfg: dict) -> None:
    ns = cfg["parameterizations"]["staircase"]["n_layers"]
    fig, axs = plt.subplots(2, 2, figsize=(8.4, 6.0), sharex=True)
    for row, (obj, sfx) in enumerate(OBJ.items()):
        for col, src in enumerate(cfg["sources"]):
            ax = axs[row, col]
            for ci, cons in enumerate(CONS):
                c = f"C{ci}"
                nn, lo, hi = [], [], []
                for n in ns:
                    seeds = [v["sigma_opt_ps_per_m"] for k, v in runs.items()
                             if k.startswith(f"{src}__{cons}__stair{n:02d}{sfx}__s")]
                    if seeds:
                        nn.append(n)
                        lo.append(min(seeds))
                        hi.append(max(seeds))
                if nn:
                    ax.fill_between(nn, lo, hi, color=c, alpha=0.2, lw=0)
                    ax.plot(nn, lo, "o-", color=c, ms=3, lw=1.1, label=f"{CONS_LABEL[cons]} staircase")
                for kind, ls in (("alpha", ":"), ("bernstein", "--"), ("bernsteinstep", "-.")):
                    seeds = [v["sigma_opt_ps_per_m"] for k, v in runs.items()
                             if k == f"{src}__{cons}__{kind}{sfx}" or k.startswith(f"{src}__{cons}__{kind}{sfx}__s")]
                    if seeds:
                        ax.axhline(min(seeds), color=c, ls=ls, lw=0.9, label=f"{CONS_LABEL[cons]} {kind}")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks(ns)
            ax.set_xticklabels([str(n) for n in ns])
            ax.minorticks_off()
            if row == 1:
                ax.set_xlabel(f"layers N, {SRC_LABEL[src]}")
            if col == 0:
                ax.set_ylabel(f"{obj} objective (ps/m)")
    axs[0, 1].legend(fontsize=5.5, frameon=False, loc="upper right")
    fig.savefig(FIG / "sigma_vs_layers.png")
    plt.close(fig)


def fig_delays(fin: dict, cfg: dict) -> None:
    for src in cfg["sources"]:
        for cons in CONS:
            fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.0), sharey=True)
            drawn = False
            for ax, (obj, sfx) in zip(axs, OBJ.items(), strict=True):
                for k in (f"{src}__{cons}__alpha{sfx}", f"{src}__{cons}__bernsteinstep{sfx}",
                          f"{src}__{cons}__stair16{sfx}"):
                    if k not in fin:
                        continue
                    drawn = True
                    mc = fin[k]["eval"][src]["modes_center"]
                    lv, mv, tv = np.array(mc["l"]), np.array(mc["m"]), np.array(mc["tau_ps_per_m"])
                    ax.plot(2 * mv + lv - 1, tv - tv[(lv == 0) & (mv == 1)][0], ".", ms=2.5, label=design_label(k))
                ax.set_xlabel(f"principal mode number 2m + l - 1, {obj}")
                ax.legend(fontsize=6, frameon=False, markerscale=3)
            axs[0].set_ylabel("group delay - LP01 (ps/m)")
            if drawn:
                fig.savefig(FIG / f"delays_{src}_{cons}.png")
            plt.close(fig)


def fig_pareto(runs: dict, cfg: dict) -> None:
    c = cfg["constraints"]["C_pareto"]
    fig, axs = plt.subplots(2, 2, figsize=(8.4, 6.0), sharex="col")
    for row, (obj, sfx) in enumerate(OBJ.items()):
        for col, src in enumerate(cfg["sources"]):
            ax = axs[row, col]
            lam = cfg["sources"][src]["lam0_um"]
            for j, na in enumerate(c["na"]):
                pts = []
                for d in c["core_diameter_um"]:
                    k = f"{src}__C_pareto__d{d:05.1f}_na{na:.2f}__alpha{sfx}"
                    if k in runs:
                        r = runs[k]
                        pts.append((etendue(d / 2, r["x0"], shape_from(r["design"]), lam), r["sigma_opt_ps_per_m"]))
                if pts:
                    g, s = map(np.array, zip(*pts, strict=True))
                    ax.plot(g, s, "o-", ms=3, lw=1, color=f"C{j}", label=f"NA {na:.2f}")
            ax.set_xscale("log")
            ax.set_yscale("log")
            if row == 1:
                ax.set_xlabel(f"fiber etendue (um^2 sr), {SRC_LABEL[src]}")
            if col == 0:
                ax.set_ylabel(f"alpha optimum, {obj} (ps/m)")
    axs[0, 1].legend(fontsize=6, frameon=False)
    fig.savefig(FIG / "pareto_alpha.png")
    plt.close(fig)


def fig_sigma_vs_lambda(fin: dict, cfg: dict) -> dict:
    lams = np.round(np.arange(0.40, 0.951, 0.025), 4)
    keys = [f"{s}__A_om3__{k}{sfx}" for s in cfg["sources"] for k in ("alpha", "bernsteinstep")
            for sfx in OBJ.values()]
    keys = [k for k in keys if k in fin]
    out: dict = {"lam_um": lams.tolist()}
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    num = Numerics(**{k: cfg["numerics"][k] for k in ("dr_um", "clad_margin_um", "oversample")})
    for i, k in enumerate(keys):
        d = fin[k]
        s = [evaluate(d["a_um"], d["x0"], shape_from(d["design"]), Source(float(lam), 0.0, 1), num)
             .sigma_modal_ps_per_m for lam in lams]
        out[k] = s
        obj = "robust" if k.endswith("_robust") else "strict"
        ax.plot(lams * 1e3, s, lw=1.1, color=f"C{i // 2}", ls="-" if obj == "strict" else "--",
                label=f"{design_label(k)}, {obj}, for {SRC_LABEL[k.split('__')[0]]}")
    ax.set_yscale("log")
    ax.set_xlabel("wavelength (nm), 50 um / NA 0.20")
    ax.set_ylabel("OFL RMS modal delay (ps/m)")
    ax.legend(fontsize=6, frameon=False)
    fig.savefig(FIG / "sigma_vs_wavelength_A_om3.png")
    plt.close(fig)
    return out


def summary(fin: dict, cfg: dict) -> str:
    L = cfg["link"]["length_m"]
    lines = ["# Summary (generated by scripts/make_figures.py)", "",
             "sigma = OFL RMS intermodal delay (ps/m), spectrally averaged over the source; robust = sqrt(mean "
             "sigma^2) over the config `robust` ensemble (x0 x 0.98/1/1.02, VCSEL also +-10 nm); dr/2 = grid check; "
             "inner 80% = center-wavelength sigma over mode groups with principal number <= 0.8 p_max (diagnostic); "
             f"BW = -3 dB optical bandwidth at L = {L:g} m (GHz), modal = center wavelength intermodal only, total = "
             "modal + chromatic over the spectrum; G/G_a = etendue relative to the alpha optimum of the same case and "
             "objective; seeds = min / max objective over restarts; other src = strict sigma of the same fiber with "
             "the other source.", ""]
    for cons in CONS:
        for home in cfg["sources"]:
            other = [s for s in cfg["sources"] if s != home][0]
            lines += [f"## {CONS_LABEL[cons]}, {SRC_LABEL[home]}", "",
                      "| design | objective | sigma | robust | dr/2 | inner 80% | max DMD | BW modal | BW total | "
                      "modes | G/G_a | seeds | other src |",
                      "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for gid, d in fin.items():
                if not gid.startswith(f"{home}__{cons}__"):
                    continue
                obj = "robust" if gid.endswith("_robust") else ("-" if gid.endswith("__step") else "strict")
                ka = f"{home}__{cons}__alpha{'_robust' if obj == 'robust' else ''}"
                ga = fin[ka]["eval"][home]["etendue_um2_sr"] if ka in fin else None
                e, eo = d["eval"][home], d["eval"][other]
                seeds = d.get("seed_sigmas") or {}
                sd = f"{min(seeds.values()):.3g} / {max(seeds.values()):.3g}" if len(seeds) > 1 else "-"
                g = f"{e['etendue_um2_sr'] / ga:.3f}" if ga else "-"
                lines.append(
                    f"| {design_label(gid)} | {obj} | {e['sigma_modal_ps_per_m']:.3g} | "
                    f"{e['sigma_robust_ps_per_m']:.3g} | {e['sigma_modal_ps_per_m_dr_half']:.3g} | "
                    f"{e['sigma_inner80_center_ps_per_m']:.3g} | {e['max_dmd_center_ps_per_m']:.3g} | "
                    f"{e['bw_modal_ghz_at_L']:.3g} | {e['bw_total_ghz_at_L']:.3g} | {e['mode_count_center']:.0f} | "
                    f"{g} | {sd} | {eo['sigma_modal_ps_per_m']:.3g} |")
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg()
    runs = {k: v for k, v in load().items() if not k.startswith("final__")}
    fin = finals()
    fig_alpha_scan(runs, cfg)
    fig_layers(runs, fin, cfg)
    fig_pareto(runs, cfg)
    if fin:
        fig_profiles(fin, cfg)
        fig_delays(fin, cfg)
        sl = fig_sigma_vs_lambda(fin, cfg)
        (ROOT / "outputs" / "sigma_vs_wavelength_A_om3.json").write_text(json.dumps(sl))
        (ROOT / "outputs" / "summary.md").write_text(summary(fin, cfg))


if __name__ == "__main__":
    main()
