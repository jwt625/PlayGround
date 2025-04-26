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
from qutip import basis, ket2dm, destroy, qeye, tensor, wigner

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
    ax.contourf(xvec, xvec, W, 120)
    ax.set_title(title)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$p$')
    ax.set_aspect('equal', adjustable='box')

# ---------- main ------------------------------------------------------------ #

def main(n=3, n_bs=2, cutoff=None):
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

    # evolution list: ρ₀, ρ₁, …, ρ_{n_bs}
    states = [ket2dm(basis(cutoff, n))]   # ρ₀   (single-mode)
    for k in range(n_bs):
        rho_next = apply_bs_once(states[-1], U_bs, ket0)
        states.append(rho_next)

    # plotting ----------------------------------------------------------------
    xvec = np.linspace(-5, 5, 201)
    n_plots = n_bs + 1
    n_cols  = min(2, n_plots)             # up to three per row
    n_rows  = int(np.ceil(n_plots / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows),
                             squeeze=False)
    for idx, rho in enumerate(states):
        r, c = divmod(idx, n_cols)
        if idx == 0:
            # W_{|n⟩}(x,p)
            title = fr'$W_{{|{n}\rangle}}(x,p)$'
        else:
            # W_{k=1}, W_{k=2}, …
            title = fr'$W_{{k={idx}}}(x,p)$'
        wigner_panel(rho, xvec, axes[r][c], title)

    # hide unused axes
    for ax in axes.ravel()[n_plots:]:
        ax.axis('off')

    fig.suptitle(
        fr"Wigner evolution of $|{n}\rangle$ through {n_bs} beam-splitter pass(es)",
        y=0.95, fontsize=14)
    fig.tight_layout()
    plt.show()

# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    main(n=6, n_bs=3)     # change parameters as desired

# %%
