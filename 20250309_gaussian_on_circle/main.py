
#%% 2D
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go

# --- Parameters ---
sigma = 1.0  # circle radius equals sigma
x = np.linspace(-7*sigma, 7*sigma, 1000)
pdf = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-0.5*(x/sigma)**2)

# --- Stereographic Projection ---
# Inverse stereographic projection (from the north pole) mapping x in R to a point on the unit circle:
# u = 2x/(1+x^2),   v = (x^2 - 1)/(1+x^2)
u = 2*x/(1+x**2)
v = (x**2 - 1)/(1+x**2)

# --- Matplotlib Figures ---

# Figure 1: Gaussian on the real line
plt.figure(figsize=(7, 4))
plt.plot(x, pdf, label='Gaussian PDF', lw=2)
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Gaussian Distribution on the Real Line')
plt.legend()
plt.grid(True)
plt.show()

# Figure 2: Wrapped Gaussian on a circle (points colored by the density)
fig, ax = plt.subplots(figsize=(6, 6))
scatter = ax.scatter(u, v, c=pdf, cmap='viridis', s=20)
ax.set_aspect('equal', 'box')
ax.set_title('Gaussian Distribution Wrapped on a Circle\n(Inverse Stereographic Projection)', fontsize=12)
plt.colorbar(scatter, ax=ax, label='PDF value')
# Optionally, draw the unit circle for reference:
theta = np.linspace(0, 2*np.pi, 200)
ax.plot(np.cos(theta), np.sin(theta), 'k--', lw=1)
ax.set_xlabel('u')
ax.set_ylabel('v')
plt.show()

# --- Plotly Figures ---

# Plotly Figure A: Gaussian on the real line
fig_real = go.Figure()
fig_real.add_trace(go.Scatter(x=x, y=pdf,
                              mode='lines',
                              name='Gaussian PDF'))
fig_real.update_layout(title='Gaussian Distribution on the Real Line (Plotly)',
                       xaxis_title='x',
                       yaxis_title='Density')
fig_real.show()

# Plotly Figure B: Wrapped Gaussian on a circle
# We'll show both the projection curve (colored by the pdf) and the unit circle for reference.
fig_circle = go.Figure()

# The projection curve: using lines+markers so hovering shows x and pdf info.
fig_circle.add_trace(go.Scatter(x=u, y=v,
                                mode='lines+markers',
                                marker=dict(color=pdf, colorscale='Viridis', colorbar=dict(title='PDF value')),
                                text=[f"x={xi:.2f}<br>pdf={pi:.3f}" for xi,pi in zip(x, pdf)],
                                hoverinfo='text',
                                name='Wrapped Gaussian'))

# Draw the unit circle (dashed line) for reference.
theta_circle = np.linspace(0, 2*np.pi, 200)
circle_x = np.cos(theta_circle)
circle_y = np.sin(theta_circle)
fig_circle.add_trace(go.Scatter(x=circle_x, y=circle_y,
                                mode='lines',
                                line=dict(dash='dash'),
                                name='Unit Circle'))

# Ensure the axes are equal and set an appropriate range.
fig_circle.update_layout(title='Gaussian Distribution Wrapped on a Circle (Plotly)',
                         xaxis=dict(scaleanchor="y", scaleratio=1, range=[-1.5, 1.5]),
                         yaxis=dict(constrain='domain', range=[-1.5, 1.5]),
                         xaxis_title='u',
                         yaxis_title='v')
fig_circle.show()

# %% 3D

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # This import registers the 3D projection
import plotly.graph_objs as go

# --- Parameters ---
sigma = 1.0/3  # sigma is equal to the circle radius
x = np.linspace(-7*sigma, 7*sigma, 1000)
pdf = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-0.5*(x/sigma)**2)

# --- Stereographic Projection ---
# Mapping from the real line to the unit circle:
# u(x) = 2x/(1+x^2) and v(x) = (x^2-1)/(1+x^2)
u = 2*x/(1+x**2)
v = (x**2 - 1)/(1+x**2)

# --- Matplotlib 3D Figure ---
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
# Plot the Gaussian as a 3D curve: (u, v, pdf)
ax.plot(u, v, pdf, lw=2, label='Gaussian Wrapped on a Circle')
ax.set_xlabel('u')
ax.set_ylabel('v')
ax.set_zlabel('PDF')
ax.set_title('3D Gaussian Distribution Wrapped on a Circle')
ax.legend()

# Add a reference unit circle in the u-v plane at z=0
theta = np.linspace(0, 2*np.pi, 200)
circle_u = np.cos(theta)
circle_v = np.sin(theta)
circle_z = np.zeros_like(theta)
ax.plot(circle_u, circle_v, circle_z, 'k--', lw=1, label='Unit Circle (z=0)')
plt.show()

# --- Plotly 3D Figure ---
fig_3d = go.Figure()

# Add the Gaussian curve (wrapped onto the circle) as a 3D scatter trace
fig_3d.add_trace(go.Scatter3d(
    x=u,
    y=v,
    z=pdf,
    mode='lines+markers',
    marker=dict(
        size=4,
        color=pdf,
        colorscale='Viridis',
        colorbar=dict(title='PDF')
    ),
    line=dict(width=2),
    name='Gaussian Wrapped on a Circle'
))

# Add the reference unit circle (at z=0)
fig_3d.add_trace(go.Scatter3d(
    x=circle_u,
    y=circle_v,
    z=circle_z,
    mode='lines',
    line=dict(dash='dash', width=2),
    name='Unit Circle (z=0)'
))

fig_3d.update_layout(
    title='3D Gaussian Distribution Wrapped on a Circle (Plotly)',
    scene=dict(
        xaxis_title='u',
        yaxis_title='v',
        zaxis_title='PDF'
    )
)

fig_3d.show()



























# %% shaded 3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import plotly.graph_objs as go

# --- Parameters ---
sigma = 1.0/3  # sigma equals the circle radius
x = np.linspace(-20*sigma, 20*sigma, 5000)
pdf = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-0.5*(x/sigma)**2)

# --- Stereographic Projection ---
# Mapping from the real line to the unit circle:
# u(x) = 2x/(1+x^2) and v(x) = (x^2 - 1)/(1+x^2)
u = 2*x/(1+x**2)
v = (x**2 - 1)/(1+x**2)

# --- Matplotlib 3D Plot with Shaded Surface ---
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D Gaussian curve (wrapped on the circle)
ax.plot(u, v, pdf, lw=2, label='Gaussian Curve')

# Create polygons (quadrilaterals) for the surface between the curve and z=0
polys = []
for i in range(len(u) - 1):
    poly = [
        (u[i],   v[i],   0),       # bottom current point
        (u[i+1], v[i+1], 0),        # bottom next point
        (u[i+1], v[i+1], pdf[i+1]),  # top next point
        (u[i],   v[i],   pdf[i])     # top current point
    ]
    polys.append(poly)

# Create a Poly3DCollection with transparency (alpha)
collection = Poly3DCollection(polys, facecolors='cyan', edgecolors='none', alpha=0.5)
ax.add_collection3d(collection)

# Optionally, add the projection curve on the uv plane (z=0)
ax.plot(u, v, np.zeros_like(pdf), 'k--', lw=1, label='Projection (z=0)')

# Set labels and title
ax.set_xlabel('u')
ax.set_ylabel('v')
ax.set_zlabel('PDF')
ax.set_title('3D Gaussian on a Ring with Shaded Surface (Matplotlib)')
ax.legend()
plt.show()

# --- Plotly 3D Plot with Shaded Surface ---
# For Plotly, we'll build a mesh that fills between the Gaussian curve and its projection.

n = len(u)
# Create vertices: first half are the top (curve) and second half are the bottom (projection)
x_vertices = np.concatenate([u, u])
y_vertices = np.concatenate([v, v])
z_vertices = np.concatenate([pdf, np.zeros_like(pdf)])

# Build faces (two triangles per segment)
faces_i = []
faces_j = []
faces_k = []
for i_pt in range(n - 1):
    # Indices: top: i_pt and i_pt+1, bottom: i_pt+n and i_pt+1+n
    # Triangle 1: (top[i], top[i+1], bottom[i])
    faces_i.append(i_pt)
    faces_j.append(i_pt + 1)
    faces_k.append(i_pt + n)
    
    # Triangle 2: (top[i+1], bottom[i+1], bottom[i])
    faces_i.append(i_pt + 1)
    faces_j.append(i_pt + 1 + n)
    faces_k.append(i_pt + n)

mesh_trace = go.Mesh3d(
    x=x_vertices,
    y=y_vertices,
    z=z_vertices,
    i=faces_i,
    j=faces_j,
    k=faces_k,
    opacity=0.5,
    color='cyan',
    name='Shaded Surface'
)

# Add the top Gaussian curve as a line for clarity
curve_trace = go.Scatter3d(
    x=u,
    y=v,
    z=pdf,
    mode='lines',
    line=dict(width=4),
    name='Gaussian Curve'
)

# Also add the projection (curve on the uv plane)
proj_trace = go.Scatter3d(
    x=u,
    y=v,
    z=np.zeros_like(pdf),
    mode='lines',
    line=dict(dash='dash', width=2),
    name='Projection (z=0)'
)

fig_plotly = go.Figure(data=[mesh_trace, curve_trace, proj_trace])
fig_plotly.update_layout(
    title='3D Gaussian on a Ring with Shaded Surface (Plotly)',
    scene=dict(
        xaxis_title='u',
        yaxis_title='v',
        zaxis_title='PDF'
    )
)
fig_plotly.show()

# %%
