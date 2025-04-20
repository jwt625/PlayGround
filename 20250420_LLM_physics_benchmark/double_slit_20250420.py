#%%

#!/usr/bin/env python3
# Electron double‑slit wave‑packet, split‑operator TDSE  (with cubic‑profile PML)
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as iio
import matplotlib.cm as cm
import os

# ── physical constants ────────────────────────────────────────────────────────
hbar = 1.054_571_817e-34          # J·s
m_e  = 9.109_383_7015e-31         # kg
eV   = 1.602_176_634e-19          # J

# ── user‑defined parameters ───────────────────────────────────────────────────
sigma   = 0.5e-6                  # initial rms width
k0      = 10.0e6                   # centre wave‑number
x0      = -6e-6                   # launch position
V0      = 10.0 * eV               # barrier height
slit_a  = 0.3e-6                  # slit width
slit_d  = 3.0e-6                  # slit separation (centre‑to‑centre)
det_x   = 8e-6                   # detection‑plane position
use_log_scale = False              # True: dB scale, False: linear

NX, NY  = 512, 256                # grid points (x, y)
x_span  = 20e-6                   # −8 µm … +12 µm
y_span  = 12e-6                   # −6 µm … +6 µm
dt      = 2e-11                   # time step
t_final = 5e-8                    # total time  ≈27 ns

# ── numerical grid ────────────────────────────────────────────────────────────
dx      = x_span / NX
dy      = y_span / NY
x       = np.linspace(-x_span/2,  x_span/2,  NX)
y       = np.linspace(-y_span/2,  y_span/2,  NY)
X, Y    = np.meshgrid(x, y)

# ── initial wave‑packet ───────────────────────────────────────────────────────
psi = np.exp(-((X-x0)**2 + Y**2)/(4*sigma**2) + 1j*k0*X)
psi /= np.sqrt((np.abs(psi)**2).sum()*dx*dy)

# ── potential: barrier with two slits at x = 0 ────────────────────────────────
V  = np.zeros_like(X)
bx = np.argmin(np.abs(x))               # grid column nearest x = 0
for j, yv in enumerate(y):
    if not ((abs(yv-1e-6) < slit_a/2) or (abs(yv+1e-6) < slit_a/2)):
        V[j, bx] = V0

# ── perfectly‑matched layer (cubic complex absorbing potential) ───────────────
pml_w      = 32            # PML thickness (cells)
sigma_max  = 5e11          # damping rate (1/s)
sigma_x = np.zeros_like(X)
sigma_y = np.zeros_like(X)

for i in range(pml_w):
    u = (pml_w - i) / pml_w               # 1 → 0 from boundary inward
    sigma_x[:,  i]   = sigma_max * u**3
    sigma_x[:, -i-1] = sigma_max * u**3
for j in range(pml_w):
    u = (pml_w - j) / pml_w
    sigma_y[  j, :]  = sigma_max * u**3
    sigma_y[-j-1, :] = sigma_max * u**3

sigma_tot  = sigma_x + sigma_y
pml_factor = np.exp(-sigma_tot * dt)      # multiplicative damping each step

# ── Fourier‑space operators ───────────────────────────────────────────────────
kx, ky   = 2*np.pi*np.fft.fftfreq(NX, d=dx), 2*np.pi*np.fft.fftfreq(NY, d=dy)
KX, KY   = np.meshgrid(kx, ky)
kin_phase   = np.exp(-1j*hbar*(KX**2 + KY**2)*dt/(2*m_e))
pot_phase_h = np.exp(-1j*V*dt/(2*hbar))

# ── storage ───────────────────────────────────────────────────────────────────
nsteps     = int(t_final/dt)
frame_step = max(1, nsteps//160)          # ≈160 animation frames
det_idx    = np.argmin(np.abs(x-det_x))
frames, Iacc = [], np.zeros(NY)
cmap = cm.get_cmap('viridis')

# ── time evolution (split‑operator with PML) ──────────────────────────────────
for s in range(nsteps):
    psi *= pot_phase_h
    psi  = np.fft.ifft2(np.fft.fft2(psi)*kin_phase)
    psi *= pot_phase_h
    psi *= pml_factor                     # absorb outgoing components

    Iacc += np.abs(psi[:, det_idx])**2
    if s % frame_step == 0:
        frames.append(np.abs(psi)**2)

# ── build GIF ─────────────────────────────────────────────────────────────────
rgb_frames = []
for fr in frames:
    if use_log_scale:
        fr_db = 10*np.log10(fr + 1e-12)
        frn   = (fr_db - fr_db.min()) / (fr_db.max() - fr_db.min() + 1e-20)
    else:
        frn   = (fr - fr.min()) / (fr.max() - fr.min() + 1e-20)
    rgb_frames.append((cmap(frn)[..., :3]*255).astype(np.uint8))

gif_name = 'double_slit.gif'
iio.mimsave(gif_name, rgb_frames, fps=20)

# ── plot PML factor for inspection ─────────────────────────────────────────────
plt.figure(figsize=(8, 6), dpi=110)
plt.imshow(pml_factor, cmap='viridis', origin='lower', 
           extent=[-x_span/2*1e6, x_span/2*1e6, -y_span/2*1e6, y_span/2*1e6])
plt.colorbar(label='PML damping factor')
plt.xlabel('x (µm)')
plt.ylabel('y (µm)')
plt.title('Perfectly Matched Layer (PML) Damping Factor')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ── detection‑plane intensity vs Fraunhofer theory ────────────────────────────
I_sim = Iacc / Iacc.max()
lam   = 2*np.pi/k0
beta  = np.pi*slit_d*y/(lam*det_x)
alpha = np.pi*slit_a*y/(lam*det_x)
I_th  = np.cos(beta)**2 * np.sinc(alpha/np.pi)**2
I_th /= I_th.max()

plt.figure(figsize=(6,3.5), dpi=110)
if use_log_scale:
    I_sim_db = 10*np.log10(I_sim + 1e-12)
    I_th_db  = 10*np.log10(I_th  + 1e-12)
    plt.plot(y*1e6, I_sim_db, lw=2, label='TDSE time‑average')
    plt.plot(y*1e6, I_th_db,  '--', lw=1.5, label='Fraunhofer prediction')
    plt.ylabel('normalised intensity (dB)')
    plt.title('Double‑slit detection‑plane profile (dB scale)')
else:
    plt.plot(y*1e6, I_sim, lw=2, label='TDSE time‑average')
    plt.plot(y*1e6, I_th,  '--', lw=1.5, label='Fraunhofer prediction')
    plt.ylabel('normalised intensity')
    plt.title('Double‑slit detection‑plane profile (linear scale)')

plt.xlabel('y (µm)')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

print(f'Animation saved → {os.path.abspath(gif_name)}')

# %%
