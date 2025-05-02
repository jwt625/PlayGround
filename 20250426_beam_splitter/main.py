#!/usr/bin/env python3
"""
Wigner–function evolution of a Fock state |n⟩ repeatedly sent through a
50:50 beam-splitter and traced over the unused port.

Requirements
------------
pip install qutip matplotlib numpy
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
from qutip import (basis, ket2dm, destroy, qeye, tensor,
                   wigner, displace, squeeze, coherent, expect)
import matplotlib.colors as mcolors      
import imageio.v2 as imageio  
import mplcursors                             # NEW

# ---------- helpers --------------------------------------------------------- #

def beam_splitter(cutoff, theta=np.pi / 4):
    """
    Return the 2-mode unitary for a loss-less beam splitter with mixing
    angle θ (θ = π/4 → 50:50).
    """
    a  = destroy(cutoff)
    a1 = tensor(a, qeye(cutoff))
    a2 = tensor(qeye(cutoff), a)
    return (-1j * theta * (a1.dag() * a2 - a1 * a2.dag())).expm()

def apply_bs_once(rho_single, U, ket0):
    """Embed ρ (mode-1) ⊗ |0⟩⟨0| (mode-2), apply U, trace out mode-2."""
    rho_two = tensor(rho_single, ket0)
    rho_two = U * rho_two * U.dag()
    return rho_two.ptrace(0)              # reduced state of port-1

def wigner_panel(rho, xvec, ax, title):
    W = wigner(rho, xvec, xvec, g=2)      # g=2 → rotate axes into x-p
    Wmax = np.max(np.abs(W))              # Get maximum absolute value
    norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-Wmax, vmax=Wmax)  # Symmetric norm
    cs = ax.contourf(xvec, xvec, W, 120, cmap='RdBu_r',  # diverging colormap
                     norm=norm)
    ax.set_title(title)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$p$')
    x_range = 10
    ax.set_xlim(-x_range, x_range)  # Set fixed x range
    ax.set_ylim(-x_range, x_range)  # Set fixed y range
    ax.set_aspect('equal', adjustable='box')
    
    # Add colorbar
    cbar = plt.colorbar(cs, ax=ax)
    cbar.set_label('W(x,p)')
    
    return cs                                    # NEW


# ---------- animation ------------------------------------------------------ #

def save_wigner_gif(states, xvec, filename="wigner.gif", title_prefix="k="):
    """Save an animated GIF of the Wigner function for every state in *states*."""
    W_all   = [wigner(r, xvec, xvec, g=2) for r in states]
    Wmax    = max(np.max(np.abs(W)) for W in W_all)
    norm    = mcolors.TwoSlopeNorm(vcenter=0, vmin=-Wmax, vmax=+Wmax)
    W_all   = [wigner(r, xvec, xvec, g=2) for r in states]   # pre-compute
    frames  = []
    for k, W in enumerate(W_all):
        Wmax = np.max(np.abs(W))                              # per-frame scale
        norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-Wmax, vmax=+Wmax)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.contourf(xvec, xvec, W, 120, cmap='RdBu_r', norm=norm)
        ax.set_axis_off()
        ax.set_title(f"{title_prefix}{k}")
        # fig.canvas.draw()                                        # grab RGBA buffer
        # w, h   = fig.canvas.get_width_height()
        # frame  = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        # frames.append(frame.reshape((h, w, 3)))
        fig.canvas.draw()                                        # render the figure
        frame = np.asarray(fig.canvas.buffer_rgba())            # RGBA array
        frames.append(frame[:, :, :3])                           # use RGB, drop α

        plt.close(fig)
    imageio.mimsave(filename, frames, duration=0.6)


# ===== Gaussian / non‑Gaussian initial states ============================

def squeezed_coherent_state(cutoff, *, n, alpha, r, phi=0):
    """D(α) S(r e^{iφ}) |0⟩ as a density matrix."""
    z = r * np.exp(1j * phi)
    ket = displace(cutoff, alpha) * (squeeze(cutoff, z) * basis(cutoff, 0))
    return ket2dm(ket)


def cat_state(cutoff, *, n, alpha, theta=0):
    """Even/odd cat ∝ |α⟩ + e^{iθ}|−α⟩ (normalised)."""
    ket = coherent(cutoff, alpha) + np.exp(1j * theta) * coherent(cutoff, -alpha)
    # ket = ket / np.sqrt((ket.dag() * ket).full()[0, 0])
    ket = ket.unit()
    return ket2dm(ket)


def displaced_fock_state(cutoff, *, n, alpha):
    """D(α)|n⟩."""
    ket = displace(cutoff, alpha) * basis(cutoff, n)
    return ket2dm(ket)

def initial_state(state_type, cutoff, **kw):
    """Return (ρ₀, label) for the chosen `state_type`."""
    if state_type == "fock":
        n = kw.get("n", 0)
        return ket2dm(basis(cutoff, n)), fr"|{n}⟩"
    if state_type == "squeezed_coh":
        rho = squeezed_coherent_state(cutoff, **kw)
        return rho, "D(α)S(r)|0⟩"
    if state_type == "cat":
        rho = cat_state(cutoff, **kw)
        return rho, "cat"
    if state_type == "displaced_fock":
        rho = displaced_fock_state(cutoff, **kw)
        return rho, fr"D(α)|{kw['n']}⟩"
    if state_type == "random_ripple":
        rho = random_ripple_state(cutoff, **kw)
        return rho, "rand"
    raise ValueError("unknown state_type")



def random_ripple_state(cutoff, *, n_modes=30, sigma=1.5, seed=None, **_):
    """Random super-position of coherent states → Gaussian envelope + ripples."""
    rng = np.random.default_rng(seed)
    ket = 0
    for _ in range(n_modes):
        alpha  = rng.normal(scale=sigma) + 1j * rng.normal(scale=sigma)
        phase  = rng.random() * 2 * np.pi
        ket   += np.exp(1j * phase) * coherent(cutoff, alpha)
    ket = ket.unit()
    return ket2dm(ket)
# -------------------------------------------------------------------------

# ---------- diagnostics ---------------------------------------------------- #

TOL_TRACE  = 1e-8        # allowed loss of trace in Fock truncation
TOL_INTEG  = 1e-3        # allowed error on ∫W dx dp / (2π)

def check_trace(rho):
    tr = rho.tr()
    if abs(tr - 1) > TOL_TRACE:
        print(f"[warn] truncated Trρ = {tr:.6g} (loss {(1-tr):.3g})")
    return tr

def integrate_wigner(W, dx, dp):
    return (dx*dp)/(2*np.pi) * W.sum()

def adaptive_grid(rho, base_range=5.0, points=201):
    """Enlarge phase-space window until Husimi Q captures ≥0.9999 mass."""
    R = base_range
    while True:
        xvec = np.linspace(-R, R, points)
        Q0   = wigner(rho, xvec, xvec, g=1)          # Husimi-like positive
        dx   = xvec[1]-xvec[0]
        mass = (dx*dx)/(2*np.pi) * Q0.sum()
        if mass > 0.9999:
            return xvec
        R *= 1.5

# ---------- main ------------------------------------------------------------ #

def main(state_type="fock", n=5, n_bs=2, cutoff=None, *, gif_file=None, **state_kw):

    """
    Parameters
    ----------
    n      : photon number of the initial Fock state.
    n_bs   : how many consecutive BS applications to perform.
    cutoff : Fock-space dimension.  Default: 5 n + 1 (≥ 2 n is OK).
    """
    cutoff = cutoff or 5 * n + 1
    ket0   = ket2dm(basis(cutoff, 0))
    U_bs   = beam_splitter(cutoff)        # fixed 50:50 BS

    rho0, label0 = initial_state(state_type, cutoff, n=n, **state_kw)
    check_trace(rho0)
    # evolution list: ρ₀, ρ₁, …, ρ_{n_bs}
    states = [rho0]
    for k in range(n_bs):
        rho_next = apply_bs_once(states[-1], U_bs, ket0)
        states.append(rho_next)

    # plotting ----------------------------------------------------------------
    xvec = np.linspace(-10, 10, 1001)
    # choose phase-space window adaptively -------------------------------
    # xvec = adaptive_grid(rho0)
    n_plots = n_bs + 1
    n_cols  = min(2, n_plots)             # up to three per row
    n_rows  = int(np.ceil(n_plots / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows),
                             squeeze=False)
    cs_all = []                       
    for idx, rho in enumerate(states):
        r, c = divmod(idx, n_cols)
        if idx == 0:
            title = fr'$W_{{{label0}}}(x,p)$'
        else:
            # W_{k=1}, W_{k=2}, …
            title = fr'$W_{{k={idx}}}(x,p)$'
        cs = wigner_panel(rho, xvec, axes[r][c], title)   # MOD
        cs_all.append(cs)                                 # NEW

        # --- Wigner integral check -------------------------------------
        if idx == 0:        # only need once – same grid for all
            W0  = wigner(rho, xvec, xvec, g=2)
            dx  = xvec[1]-xvec[0]
            integ = integrate_wigner(W0, dx, dx)
            if abs(integ-1) > TOL_INTEG:
                print(f"[warn] ∫W dx dp /(2π) = {integ:.5f}")
                if state_kw.get("renorm", False):
                    print("        renormalising Wigner arrays")
                    rho0  /= rho0.tr()

    # hide unused axes
    for ax in axes.ravel()[n_plots:]:
        ax.axis('off')

    fig.suptitle(
        fr"Wigner evolution of $|{n}\rangle$ through {n_bs} beam-splitter pass(es)",
        y=0.95, fontsize=14)
    # fig.colorbar(axes[0][0].collections[0], ax=axes.ravel().tolist(),
    #           location='right')

    fig.tight_layout()
    # enable hover tooltips showing (x, p, W) ------------------------------
    # robustly collect artists from any Matplotlib version
    artists = []
    for cs in cs_all:
        artists.extend(getattr(cs, "collections", [cs]))

    mplcursors.cursor(
        artists,
        hover=True).connect(
        "add", lambda sel: sel.annotation.set_text(
            f"x={sel.target[0]:.2f}\np={sel.target[1]:.2f}\nW={sel.target[2]:.3g}"))

    plt.show()
    # optional GIF ----------------------------------------------------------
    if gif_file:
        save_wigner_gif(states, xvec, filename=gif_file)

# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # examples: choose one
    # main(state_type="fock", n=5, n_bs=25, gif_file="fock_evolution.gif")
    # main(state_type="squeezed_coh", n_bs=99, alpha=2+0j, r=0.8, phi=0, gif_file="squeezed_coh_evolution.gif")
    # main(state_type="cat", n_bs=99, alpha=2, theta=np.pi, gif_file="cat_evolution.gif")
    # main(state_type="displaced_fock", n=4, n_bs=99, alpha=1.5, gif_file="displaced_fock_evolution.gif")
    # one-liner example
    main(state_type="random_ripple",
        n=0,                    # ignored for this state type
        n_bs=2,                # number of beam-splitter passes
        cutoff=40,              # Fock-space truncation
        n_modes=10,             # how many random coherent components
        sigma=1.5,              # envelope width in phase space
        seed=323,               # RNG seed for reproducibility (optional)
        gif_file="ripple_evolution_fast.gif")


# %%
