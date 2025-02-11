

#%%

import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx, ny = 200, 200  # Grid size
dx = dy = 1e-9      # Grid spacing (e.g., 1 nm)
dt = dx * 0.99 / (np.sqrt(2) * 3e8)  # Time step (Courant)
n = 1.5             # Refractive index
v = 0.7 * 3e8       # Particle velocity (> c/n)
q = 1e-19           # Charge
epsilon_r = 1.5

# Initialize fields
Ez = np.zeros((nx, ny))
Hx = np.zeros((nx, ny))
Hy = np.zeros((nx, ny))
Jz = np.zeros((nx, ny))

# Track particle position
x_part = nx//2
y_part = ny//2

for t in range(300):
    # Update particle position
    x_part = int(x_part + v * dt / dx)
    
    # Inject current (Gaussian pulse)
    sigma = 2  # Width of Gaussian
    for i in range(-3*sigma, 3*sigma+1):
        for j in range(-3*sigma, 3*sigma+1):
            xi = x_part + i
            yj = y_part + j
            if 0 <= xi < nx and 0 <= yj < ny:
                Jz[xi, yj] += q * v * np.exp(-(i**2 + j**2)/(2*sigma**2))
    
    # Update H fields
    Hx[:, :-1] += (Ez[:, 1:] - Ez[:, :-1]) * dt / (dy * 120 * np.pi)
    Hy[:-1, :] -= (Ez[1:, :] - Ez[:-1, :]) * dt / (dx * 120 * np.pi)
    
    # Update E field with J
    Ez[1:-1, 1:-1] += ( (Hy[1:-1, 1:-1] - Hy[:-2, 1:-1]) / dx
                      - (Hx[1:-1, 1:-1] - Hx[1:-1, :-2]) / dy
                      - Jz[1:-1, 1:-1]/epsilon_r ) * dt
    
    Jz.fill(0)  # Reset current

# Visualize
plt.imshow(Ez, cmap='RdBu')
plt.show()


# %%
