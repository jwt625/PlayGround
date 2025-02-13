

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


# %% 20250212, naive calculation, written by mistral
# % did not work. Switched to chatGPT pro I got free from the training project
import numpy as np
import matplotlib.pyplot as plt

# Define parameters
wavelength = 1.0  # Wavelength of the emitted waves
c = 1.0  # Speed of light in medium (normalized to 1)
v_particle = 1.2 * c  # Particle speed (greater than c to induce Cherenkov radiation)
k = 2 * np.pi / wavelength  # Wavenumber
omega = k * c  # Angular frequency
period = 2 * np.pi / omega  # Wave period
dt = period / 10  # Time step

domain_x = 100 * wavelength

# Set domain and grid
nx, ny = 500, 100  # Resolution
x = np.linspace(0, domain_x, nx)
y = np.linspace(-10 * wavelength, 10 * wavelength, ny)
X, Y = np.meshgrid(x, y)

# Time evolution
final_time = domain_x / v_particle * 0.8  # Final time when particle is at 80% of domain_x
n_steps = int(final_time / dt)

# Initialize wave field
wave_field = np.zeros((ny, nx))

# Compute the superposition of waves
for step in range(n_steps):
    t = step * dt
    x_source = v_particle * t  # Particle position at time t
    r = np.sqrt((X - x_source)**2 + Y**2)  # Distance to current wavefront
    valid_mask = r > 0  # Avoid singularity at r=0
    wave_field[valid_mask] += (1 / r[valid_mask]**2) * np.cos(k * r[valid_mask] - omega * t)

# Normalize for visualization
wave_field /= np.max(np.abs(wave_field))

# Plot the result
plt.figure(figsize=(10, 5))
plt.imshow(wave_field, extent=[0, domain_x, -10 * wavelength, 10 * wavelength], cmap='bwr', origin='lower')
plt.colorbar(label='Wave Amplitude')
plt.xlabel('x (Wavelengths)')
plt.ylabel('y (Wavelengths)')
plt.title('Cherenkov Radiation Pattern')
plt.show()



# %%
