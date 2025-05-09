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
#%%
#%%
import numpy as np
import plotly.graph_objects as go

# Parameters
c = 3e8
lambda0 = 1e-6
omega0 = 2 * np.pi * c / lambda0
tau_fwhm = 5e-15
sigma_t = tau_fwhm / (2 * np.sqrt(2 * np.log(2)))
phi = np.pi / 3  # adjustable phase shift

# Time vector
t = np.linspace(-20e-15, 20e-15, 1000)
envelope = np.exp(-t**2 / (2 * sigma_t**2))

# Electric fields
Ex = envelope * np.cos(omega0 * t)
Ey = envelope * np.cos(omega0 * t + phi)
time_fs = t * 1e15  # convert to fs

# Create 3D plot
fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=Ex, y=Ey, z=time_fs,
    mode='lines',
    line=dict(width=3, color=time_fs, colorscale='Viridis'),
    name='E-field Trajectory'
))

fig.update_layout(
    scene=dict(
        xaxis_title='Ex (arb. units)',
        yaxis_title='Ey (arb. units)',
        zaxis_title='Time (fs)',
        xaxis=dict(range=[-1.1, 1.1]),
        yaxis=dict(range=[-1.1, 1.1]),
        zaxis=dict(range=[time_fs[0], time_fs[-1]])
    ),
    title=f"Circular/Elliptical Polarization (φ = {phi:.2f} rad)",
    margin=dict(l=0, r=0, b=0, t=40)
)

fig.show()


# %%
#%%
import numpy as np
import plotly.graph_objects as go

# Parameters
c = 3e8
lambda0 = 1e-6
omega0 = 2 * np.pi * c / lambda0
tau_fwhm = 5e-15
sigma_t = tau_fwhm / (2 * np.sqrt(2 * np.log(2)))

# Time vector
t = np.linspace(-25e-15, 25e-15, 1000)
time_fs = t * 1e15

# Gaussian pulse envelope
envelope = np.exp(-t**2 / (2 * sigma_t**2))

# Time-varying phase shift for polarization gating
# Goes from ~π/2 in wings to 0 at center
delta_phi = (np.pi / 2) * np.tanh(t / (5e-15))

# Two orthogonal components with varying phase shift
Ex = envelope * np.cos(omega0 * t)
Ey = envelope * np.cos(omega0 * t + delta_phi)

# Plot with Plotly
fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=Ex, y=Ey, z=time_fs,
    mode='lines',
    line=dict(width=3, color=time_fs, colorscale='Plasma'),
    name='PG-modulated E-field'
))

fig.update_layout(
    scene=dict(
        xaxis_title='Ex (arb. units)',
        yaxis_title='Ey (arb. units)',
        zaxis_title='Time (fs)',
        xaxis=dict(range=[-1.1, 1.1]),
        yaxis=dict(range=[-1.1, 1.1]),
        zaxis=dict(range=[time_fs[0], time_fs[-1]])
    ),
    title="Polarization Gating: Linear Center, Circular Wings",
    margin=dict(l=0, r=0, b=0, t=40)
)

fig.show()
# %%
