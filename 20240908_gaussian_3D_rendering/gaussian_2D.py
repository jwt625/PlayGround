
#%%

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyvista as pv
# import trimesh

# Parameters for the 2D Gaussian
A = 1  # Amplitude
sigma = 1  # Standard deviation

# Create a grid of points
x = np.linspace(-5, 5, 200)
y = np.linspace(-5, 5, 200)
x, y = np.meshgrid(x, y)

# 2D Gaussian formula
z = A * np.exp(-(x**2 + y**2) / (2 * sigma**2))

# Plot the surface for visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='viridis')

plt.show()

# Save the surface data as an OBJ or STL file (use another library like trimesh for export)

# Create a structured grid
grid = pv.StructuredGrid(x, y, z)

# Export to STL or OBJ
grid.save('gaussian_surface.stl')  # For STL



#%%
import numpy as np
import trimesh

# Parameters for the 2D Gaussian
A = 1  # Amplitude
sigma = 1  # Standard deviation

# Create a grid of points
x = np.linspace(-5, 5, 200)
y = np.linspace(-5, 5, 200)
x, y = np.meshgrid(x, y)

# 2D Gaussian formula
z = A * np.exp(-(x**2 + y**2) / (2 * sigma**2))

# Create the vertices for the mesh
vertices = np.column_stack([x.ravel(), y.ravel(), z.ravel()])

# Create faces (triangles) for the mesh
faces = []
for i in range(len(x) - 1):
    for j in range(len(y) - 1):
        idx1 = i * len(y) + j
        idx2 = i * len(y) + (j + 1)
        idx3 = (i + 1) * len(y) + j
        idx4 = (i + 1) * len(y) + (j + 1)
        # Two triangles per grid square
        faces.append([idx1, idx2, idx3])
        faces.append([idx2, idx4, idx3])

faces = np.array(faces)

# Create a trimesh object
mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

# Export to STL or OBJ
mesh.export('gaussian_surface.stl')  # For STL
# mesh.export('gaussian_surface.obj')  # For OBJ
