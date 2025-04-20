"""
Generate output.gif – electric‑ and magnetic‑field profiles inside a
parallel‑plate capacitor over one full period.

Parameters
----------
R      : plate radius (m)          – 0.05 m
d      : plate separation (m)      – 0.002 m  (not used here)
I0     : current amplitude (A)     – 1 A
omega  : angular frequency (rad/s) – 2π·1e5 rad/s
frames : animation frames          – 200
fps    : frames‑per‑second in GIF  – 30
"""

#%%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from math import pi, sin, cos

# --- physical constants -----------------------------------------------------
eps0 = 8.8541878128e-12        # F m‑1
mu0  = 4 * pi * 1e-7             # H m‑1

# --- problem parameters -----------------------------------------------------
R     = 0.05                      # m
d     = 0.002                     # m (for completeness)
I0    = 1.0                       # A
omega = 2 * pi * 1e5              # rad s‑1
T     = 2 * pi / omega            # one period (s)

frames = 200
fps     = 30                      # GIF frame‑rate

# --- derived amplitudes -----------------------------------------------------
E0 = I0 / (pi * R**2 * eps0 * omega)       # |E|
B0 = mu0 * I0 / (2 * pi * R)               # |B| at r = R

# --- radial grid ------------------------------------------------------------
r_max = 0.1                                # 2 R, to show zero field outside
r      = np.linspace(0.0, r_max, 400)
B_factor = mu0 * I0 * r / (2 * pi * R**2)  # r‑dependent prefactor

# --- figure setup -----------------------------------------------------------
fig, (axE, axB) = plt.subplots(1, 2, figsize=(8, 4))
for ax in (axE, axB):
    ax.set_xlim(0, r_max)
    ax.set_xlabel("r (m)")
axE.set_ylim(-1.1 * E0, 1.1 * E0)
axE.set_ylabel("E (V m⁻¹)")
axE.set_title("Electric Field")
axB.set_ylim(-1.1 * B0, 1.1 * B0)
axB.set_ylabel("B (T)")
axB.set_title("Magnetic Field")
lineE, = axE.plot([], [], lw=2)
lineB, = axB.plot([], [], lw=2)

# --- helper functions -------------------------------------------------------
def E_profile(t):
    """Electric field profile at time t."""
    val = E0 * sin(omega * t)
    return np.where(r <= R, val, 0.0)

def B_profile(t):
    """Magnetic field profile at time t."""
    return np.where(r <= R, B_factor * cos(omega * t), 0.0)

# --- animation callback -----------------------------------------------------
def update(frame):
    t = frame * T / frames
    lineE.set_data(r, E_profile(t))
    lineB.set_data(r, B_profile(t))
    return lineE, lineB

# --- build and save ---------------------------------------------------------
ani = FuncAnimation(fig, update, frames=frames, blit=True)
ani.save("output.gif", writer=PillowWriter(fps=fps))
plt.close(fig)
print("output.gif saved.")

# %%
