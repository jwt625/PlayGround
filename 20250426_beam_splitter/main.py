#!/usr/bin/env python3
"""
Plot the Wigner function of a photon-number state |n⟩ before and after a
50:50 beam-splitter, tracing out the unused output port.
"""


#%%
import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, destroy, qeye, tensor, ket2dm, wigner

def wigner_contour(rho, xvec, ax, title=""):
    """Helper: filled-contour Wigner plot."""
    W = wigner(rho, xvec, xvec)
    ax.contourf(xvec, xvec, W, 100)
    ax.set_title(title)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$p$")
    ax.set_aspect("equal")

def bs_output_state(n: int, cutoff: int):
    """Return reduced state ρ₁ after a 50:50 beam splitter."""
    a = destroy(cutoff)
    a1 = tensor(a, qeye(cutoff))
    a2 = tensor(qeye(cutoff), a)

    psi_in = tensor(basis(cutoff, n), basis(cutoff, 0))     # |n⟩|0⟩
    rho_in = ket2dm(psi_in)

    theta = np.pi / 4                                        # 50:50 BS
    U = (-1j * theta * (a1.dag() * a2 - a1 * a2.dag())).expm()
    rho_out = U * rho_in * U.dag()                           # two-mode state
    return rho_out.ptrace(0)                                 # trace over mode 2

def main(n=3, cutoff=None):
    cutoff = cutoff or 5 * n + 1
    xvec = np.linspace(-5, 5, 201)

    rho_initial = ket2dm(basis(cutoff, n))
    rho_after   = bs_output_state(n, cutoff)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    wigner_contour(rho_initial, xvec, axes[0], fr"$W_{{|{n}\rangle}}(x,p)$")
    wigner_contour(rho_after,   xvec, axes[1],
                   fr"$W_{{\mathrm{{BS}}\,|{n}\rangle}}(x,p)$ (mode 1)")
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main(n=6)          # change n as desired

# %%
