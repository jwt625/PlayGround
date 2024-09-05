

#%%

import numpy as np
import matplotlib.pyplot as plt

# Parameters
n_grid = 500

# Create the meshgrid
xs, ys = np.meshgrid(np.arange(1, n_grid+1), np.arange(1, n_grid+1))

# Reshape the grids
xs = xs.reshape(1, n_grid**2)
ys = ys.reshape(1, n_grid**2)

# Offsets
offset_x = 0
offset_y = 0

# Calculate angles (in radians)
thetas = np.arctan((ys + offset_y) / (xs + offset_x))
thetas = thetas[thetas < np.pi / 4]

# Convert to degrees
thetas_deg = thetas / np.pi * 180

# Plot histogram of angles
N, edges = np.histogram(thetas_deg, bins=100000)

# Plot using tangent of edges and stem plot
plt.figure(figsize=(12, 8))
plt.stem(np.tan(np.radians(edges[:-1])), N, markerfmt=".")
plt.show()


# %%
