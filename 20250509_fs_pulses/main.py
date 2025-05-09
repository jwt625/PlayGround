

#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants
c = 3e8  # speed of light (m/s)
lambda0 = 1e-6  # central wavelength (1 µm)
omega0 = 2 * np.pi * c / lambda0  # angular frequency
tau_fwhm = 5e-15  # FWHM of pulse (5 fs)

# Gaussian envelope std dev
sigma_t = tau_fwhm / (2 * np.sqrt(2 * np.log(2)))

# Time vector
t = np.linspace(-20e-15, 20e-15, 200)
E_t = np.exp(-t**2 / (2 * sigma_t**2)) * np.cos(omega0 * t)

# Set up plot
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.set_xlim(t[0]*1e15, t[-1]*1e15)
ax.set_ylim(-1.1, 1.1)
ax.set_xlabel('Time (fs)')
ax.set_ylabel('E-field amplitude (arb. units)')
ax.set_title('E-field of 5 fs Laser Pulse at 1 µm')

# Animation functions
def init():
    line.set_data([], [])
    return line,

def update(frame):
    window = 100
    start = max(0, frame - window//2)
    end = min(len(t), frame + window//2)
    line.set_data(t[start:end]*1e15, E_t[start:end])
    return line,

ani = animation.FuncAnimation(
    fig, update, frames=len(t), init_func=init, blit=True, interval=20
)

# Save as GIF
ani.save("efield_pulse.gif", writer='pillow', fps=60)

# %%
