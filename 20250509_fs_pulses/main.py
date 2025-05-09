#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# Constants
c = 3e8
lambda0 = 1e-6
omega0 = 2 * np.pi * c / lambda0
tau_fwhm = 5e-15
sigma_t = tau_fwhm / (2 * np.sqrt(2 * np.log(2)))

# Time vector
t = np.linspace(-20e-15, 20e-15, 300)

# Circular polarization: x and y components with 90° phase shift
envelope = np.exp(-t**2 / (2 * sigma_t**2))
Ex = envelope * np.cos(omega0 * t)
Ey = envelope * np.sin(omega0 * t)

# Set up 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_zlim(t[0]*1e15, t[-1]*1e15)
ax.set_xlabel('Ex')
ax.set_ylabel('Ey')
ax.set_zlabel('Time (fs)')
ax.set_title('Circularly Polarized 5 fs Pulse (λ = 1 µm)')

line, = ax.plot([], [], [], lw=2)

def init():
    line.set_data([], [])
    line.set_3d_properties([])
    return line,

def update(frame):
    line.set_data(Ex[:frame], Ey[:frame])
    line.set_3d_properties(t[:frame]*1e15)
    return line,

ani = animation.FuncAnimation(
    fig, update, frames=len(t), init_func=init, blit=True, interval=20
)

ani.save("circular_pulse.gif", writer='pillow', fps=60)
# %%


#%%
import numpy as np
import matplotlib.pyplot as plt

# Parameters
c = 3e8
lambda0 = 1e-6
omega0 = 2 * np.pi * c / lambda0
tau_fwhm = 5e-15
sigma_t = tau_fwhm / (2 * np.sqrt(2 * np.log(2)))
phi = np.pi / 3  # adjustable phase shift in radians

# Time vector
t = np.linspace(-20e-15, 20e-15, 1000)
envelope = np.exp(-t**2 / (2 * sigma_t**2))

# Ex: x-polarized pulse
Ex = envelope * np.cos(omega0 * t)
# Ey: y-polarized pulse with phase shift
Ey = envelope * np.cos(omega0 * t + phi)

# Plot vector trajectory in Ex-Ey space
plt.figure(figsize=(6, 6))
plt.plot(Ex, Ey, linewidth=1)
plt.xlabel('Ex (arb. units)')
plt.ylabel('Ey (arb. units)')
plt.title(f'Polarization Ellipse (φ = {phi:.2f} rad)')
plt.axis('equal')
plt.grid(True)
plt
