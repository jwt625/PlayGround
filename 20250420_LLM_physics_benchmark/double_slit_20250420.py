#%%
#!/usr/bin/env python3
# Electron double‑slit wave‑packet, split‑operator TDSE
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as iio
import matplotlib.cm as cm
import os

# ── physical constants ────────────────────────────────────────────────────────
hbar = 1.054_571_817e-34          # J·s
m_e  = 9.109_383_7015e-31         # kg
eV   = 1.602_176_634e-19          # J

# ── user‑defined parameters (all SI unless noted) ─────────────────────────────
sigma   = 0.5e-6                  # initial rms width
k0      = 5.0e6                  # centre wave‑number
x0      = -6e-6                   # launch position
V0      = 10.0 * eV               # barrier height
slit_a  = 0.3e-6                  # slit width
slit_d  = 2.0e-6                  # slit separation (centre‑to‑centre)
det_x   = 10e-6                   # detection‑plane position
use_log_scale = True             # True for dB scale, False for linear scale

NX, NY  = 256, 128                # grid points (x, y)
x_span  = 20e-6                   # −8 µm … +12 µm
y_span  = 12e-6                   # −6 µm … +6 µm
dt      = 2e-11                   # time step
t_final = 3e-8                    # total time  ≈27 ns

# ── numerical grid ────────────────────────────────────────────────────────────
dx      = x_span / NX
dy      = y_span / NY
x       = np.linspace(-x_span/2,  x_span/2,  NX)
y       = np.linspace(-y_span/2,  y_span/2,  NY)
X, Y    = np.meshgrid(x, y)

# ── initial wave‑packet ───────────────────────────────────────────────────────
psi = np.exp(-( (X-x0)**2 + Y**2 )/(4*sigma**2) + 1j*k0*X)
psi /= np.sqrt((np.abs(psi)**2).sum()*dx*dy)

# ── potential: barrier with two slits at x=0 ──────────────────────────────────
V  = np.zeros_like(X)
bx = np.argmin(np.abs(x))               # grid column nearest x = 0
for j, yv in enumerate(y):
    if not ((abs(yv-1e-6) < slit_a/2) or (abs(yv+1e-6) < slit_a/2)):
        V[j, bx] = V0

# ── absorbing mask (cos² taper, 20 cells all around) ──────────────────────────
absorb_w = 20
mask     = np.ones_like(X)
for n in range(absorb_w):
    f = np.exp(-4*(n/absorb_w)**2)
    mask[:, n]       *= f
    mask[:, -n-1]    *= f
    mask[n, :]       *= f
    mask[-n-1, :]    *= f

# ── Fourier‑space operators ───────────────────────────────────────────────────
kx, ky   = 2*np.pi*np.fft.fftfreq(NX, d=dx), 2*np.pi*np.fft.fftfreq(NY, d=dy)
KX, KY   = np.meshgrid(kx, ky)
kin_phase   = np.exp(-1j*hbar*(KX**2+KY**2)*dt/(2*m_e))
pot_phase_h = np.exp(-1j*V*dt/(2*hbar))

# ── storage ───────────────────────────────────────────────────────────────────
nsteps       = int(t_final/dt)
frame_step   = max(1, nsteps//160)     # ≈160 animation frames
det_idx      = np.argmin(np.abs(x-det_x))
frames, Iacc = [], np.zeros(NY)

cmap = cm.get_cmap('viridis')

# ── time evolution (split‑operator) ───────────────────────────────────────────
for s in range(nsteps):
    psi *= pot_phase_h
    psi  = np.fft.ifft2(np.fft.fft2(psi)*kin_phase)
    psi *= pot_phase_h
    psi *= mask

    Iacc += np.abs(psi[:, det_idx])**2
    if s % frame_step == 0:
        frames.append(np.abs(psi)**2)

# ── build GIF directly from data ──────────────────────────────────────────────
rgb_frames = []
for fr in frames:
    if use_log_scale:
        # Apply logarithmic scaling to the frames
        db_offset = 1e-10
        fr_db = 10 * np.log10(fr + db_offset)
        # Normalize to [0, 1] range for colormap
        frn = (fr_db - fr_db.min()) / (fr_db.max() - fr_db.min() + 1e-20)
    else:
        # Original linear scaling
        frn = (fr - fr.min()) / (fr.max() - fr.min() + 1e-20)
    
    rgb = (cmap(frn)[..., :3]*255).astype(np.uint8)
    rgb_frames.append(rgb)
gif_name = 'double_slit.gif'
iio.mimsave(gif_name, rgb_frames, fps=20)

# ── detection‑plane intensity vs Fraunhofer theory ────────────────────────────
I_sim = Iacc / Iacc.max()
lam   = 2*np.pi/k0
beta  = np.pi*slit_d*y/(lam*det_x)
alpha = np.pi*slit_a*y/(lam*det_x)
I_th  = np.cos(beta)**2 * np.sinc(alpha/np.pi)**2
I_th /= I_th.max()

plt.figure(figsize=(6,3.5), dpi=110)

if use_log_scale:
    # Convert to dB scale (with small offset to avoid log(0))
    db_offset = 1e-10
    I_sim_db = 10 * np.log10(I_sim + db_offset)
    I_th_db = 10 * np.log10(I_th + db_offset)
    
    plt.plot(y*1e6, I_sim_db, label='TDSE time‑average', lw=2)
    plt.plot(y*1e6, I_th_db, '--', lw=1.5, label='Fraunhofer prediction')
    plt.ylabel('normalised intensity (dB)')
    plt.title('Double‑slit detection‑plane profile (dB scale)')
else:
    plt.plot(y*1e6, I_sim, label='TDSE time‑average', lw=2)
    plt.plot(y*1e6, I_th, '--', lw=1.5, label='Fraunhofer prediction')
    plt.ylabel('normalised intensity')
    plt.title('Double‑slit detection‑plane profile (linear scale)')

plt.xlabel('y (µm)')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

print(f'Animation saved → {os.path.abspath(gif_name)}')

# %%
