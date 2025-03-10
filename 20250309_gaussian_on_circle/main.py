
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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import plotly.graph_objs as go

# --- Parameters ---
sigma = 0.2  # sigma equals the circle radius
x = np.linspace(-10*sigma, 10*sigma, 1000)
x_extend = np.logspace(np.log10(10*sigma), np.log10(1000*sigma), 100)
x = np.concatenate((-np.flip(x_extend), x, x_extend))
pdf = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-0.5*(x/sigma)**2)

# --- Stereographic Projection ---
# Mapping from the real line to the unit circle:
# u(x) = 2x/(1+x^2) and v(x) = (x^2 - 1)/(1+x^2)
u = 2*x/(1+x**2)
v = (x**2 - 1)/(1+x**2)

# ------------------------------
# Matplotlib 3D Plot with Colored & Transparent Surface (for reference)
# ------------------------------
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Create colored segments for the 3D curve
segments = [((u[i], v[i], pdf[i]), (u[i+1], v[i+1], pdf[i+1])) for i in range(len(u)-1)]
norm = plt.Normalize(pdf.min(), pdf.max())
segment_colors = [plt.cm.viridis(norm((pdf[i] + pdf[i+1]) / 2)) for i in range(len(u)-1)]
line_collection = Line3DCollection(segments, colors=segment_colors, linewidths=2)
ax.add_collection3d(line_collection)

# Markers colored by PDF
ax.scatter(u, v, pdf, c=pdf, cmap='viridis', s=20)

# Create polygons for the surface between the curve and its projection (z=0)
polys = []
for i in range(len(u) - 1):
    poly = [
        (u[i],   v[i],   0),       # bottom current point
        (u[i+1], v[i+1], 0),        # bottom next point
        (u[i+1], v[i+1], pdf[i+1]),  # top next point
        (u[i],   v[i],   pdf[i])     # top current point
    ]
    polys.append(poly)
facecolors = [plt.cm.viridis(norm((pdf[i] + pdf[i+1]) / 2)) for i in range(len(u)-1)]
surface = Poly3DCollection(polys, facecolors=facecolors, edgecolors='none', alpha=0.5)
ax.add_collection3d(surface)

# Add reference unit circle in the uv plane at z=0 (dashed)
theta = np.linspace(0, 2*np.pi, 200)
circle_u = np.cos(theta)
circle_v = np.sin(theta)
circle_z = np.zeros_like(theta)
ax.plot(circle_u, circle_v, circle_z, 'k--', lw=1, label='Unit Circle (z=0)')

ax.set_xlabel('u')
ax.set_ylabel('v')
ax.set_zlabel('PDF')
ax.set_title('3D Gaussian on a Ring with Colored, Transparent Surface (Matplotlib)')
# ax.legend()
plt.show()

# ------------------------------
# Plotly 3D Plot with Colored & Transparent Surface and Arrow Annotation
# ------------------------------
# Build a mesh that fills between the top (curve) and its projection (z=0)
n = len(u)
x_vertices = np.concatenate([u, u])
y_vertices = np.concatenate([v, v])
z_vertices = np.concatenate([pdf, np.zeros_like(pdf)])
intensity = np.concatenate([pdf, pdf])

# Build faces (two triangles per segment)
faces_i = []
faces_j = []
faces_k = []
for i_pt in range(n - 1):
    faces_i.append(i_pt)
    faces_j.append(i_pt + 1)
    faces_k.append(i_pt + n)
    
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
    intensity=intensity,
    colorscale='Viridis',
    opacity=0.5,
    showscale=False,
    name='Shaded Surface'
)

curve_trace_line = go.Scatter3d(
    x=u,
    y=v,
    z=pdf,
    mode='lines',
    line=dict(width=2, color='darkblue'),
    name='Gaussian Curve Line'
)

curve_trace_markers = go.Scatter3d(
    x=u,
    y=v,
    z=pdf,
    mode='markers',
    marker=dict(
        size=4,
        color=pdf,
        colorscale='Viridis'
    ),
    name='Gaussian Curve Markers'
)

proj_trace = go.Scatter3d(
    x=u,
    y=v,
    z=np.zeros_like(pdf),
    mode='lines',
    line=dict(dash='dash', width=2, color='black'),
    name='Projection (z=0)'
)

fig_plotly = go.Figure(data=[mesh_trace, curve_trace_line, curve_trace_markers, proj_trace])

# Add annotation with arrow at the top of the circle (point (0,1,0))
fig_plotly.update_layout(
    width=600, height=600, margin=dict(l=0, r=0, b=0, t=0),
    showlegend=False,
    scene=dict(
        xaxis=dict(title='', showticklabels=False, ticks='', showgrid=True),
        yaxis=dict(title='', showticklabels=False, ticks='', showgrid=True),
        zaxis=dict(title='', showticklabels=False, ticks='', showgrid=True),
        annotations=[
            dict(
                x=0,
                y=1,
                z=0,
                text="you are here",
                showarrow=True,
                arrowhead=2,
                arrowcolor="red",
                ax=0,   # horizontal offset (in pixels)
                ay=-80, # vertical offset (in pixels)
                font=dict(color="black", size=20)
            )
        ]
    )
)

fig_plotly.show()

#%% additional code for gif
import plotly.io as pio
import imageio
import io

frames = []
num_frames = 72  # number of frames in the orbit shot
d = 2.5        # distance of the camera from the center
cam_z = 0.5    # fixed z coordinate for the camera

for angle in np.linspace(0, 360, num_frames, endpoint=False):
    print(angle)
    angle_rad = np.deg2rad(angle)
    cam_x = d * np.cos(angle_rad)
    cam_y = d * np.sin(angle_rad)
    
    # Update the camera view for the 3D scene.
    fig_plotly.update_layout(
        scene_camera=dict(
            eye=dict(x=cam_x, y=cam_y, z=cam_z)
        )
    )
    
    # Export the current view as a PNG image (requires kaleido).
    img_bytes = pio.to_image(fig_plotly, format='png')
    image = imageio.imread(io.BytesIO(img_bytes))
    frames.append(image)

gif_filename = 'orbit_plotly.gif'
imageio.mimsave(gif_filename, frames, fps=20)
print(f"GIF saved as {gif_filename}")


























# %% generate gif (no labeling. Run the one above)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import imageio

# --- Parameters ---
sigma = 0.2  # sigma equals the circle radius
x = np.linspace(-10*sigma, 10*sigma, 1000)
x_extend = np.logspace(np.log10(10*sigma), np.log10(1000*sigma), 100)
x = np.concatenate((-np.flip(x_extend), x, x_extend))
pdf = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-0.5*(x/sigma)**2)

# --- Stereographic Projection ---
# u(x) = 2x/(1+x^2) and v(x) = (x^2 - 1)/(1+x^2)
u = 2*x/(1+x**2)
v = (x**2 - 1)/(1+x**2)

# ------------------------------
# Matplotlib 3D Plot with Colored & Transparent Surface
# ------------------------------
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Create colored segments for the 3D curve
segments = [((u[i], v[i], pdf[i]), (u[i+1], v[i+1], pdf[i+1])) for i in range(len(u)-1)]
norm = plt.Normalize(pdf.min(), pdf.max())
segment_colors = [plt.cm.viridis(norm((pdf[i] + pdf[i+1]) / 2)) for i in range(len(u)-1)]
line_collection = Line3DCollection(segments, colors=segment_colors, linewidths=2)
ax.add_collection3d(line_collection)

# Optionally, add markers for clarity (colored by the PDF)
ax.scatter(u, v, pdf, c=pdf, cmap='viridis', s=20)

# Create polygons for the surface between the curve and its projection (z=0)
polys = []
for i in range(len(u) - 1):
    poly = [
        (u[i],   v[i],   0),       # bottom current point
        (u[i+1], v[i+1], 0),        # bottom next point
        (u[i+1], v[i+1], pdf[i+1]),  # top next point
        (u[i],   v[i],   pdf[i])     # top current point
    ]
    polys.append(poly)
# Compute face colors based on the average PDF on the top edge of each quadrilateral
facecolors = [plt.cm.viridis(norm((pdf[i] + pdf[i+1]) / 2)) for i in range(len(u)-1)]
surface = Poly3DCollection(polys, facecolors=facecolors, edgecolors='none', alpha=0.5)
ax.add_collection3d(surface)

# Add a reference unit circle in the uv plane at z=0 (dashed)
theta = np.linspace(0, 2*np.pi, 200)
circle_u = np.cos(theta)
circle_v = np.sin(theta)
circle_z = np.zeros_like(theta)
ax.plot(circle_u, circle_v, circle_z, 'k--', lw=1, label='Unit Circle (z=0)')

ax.set_xlabel('u')
ax.set_ylabel('v')
ax.set_zlabel('PDF')
# ax.set_title('3D Gaussian on a Ring with Colored, Transparent Surface (Matplotlib)')
# ax.legend()

# ------------------------------
# Generate GIF by Rotating the View
# ------------------------------
frames = []
num_frames = 72  # e.g., 72 frames for a full 360° rotation (~5° per frame)
for angle in np.linspace(0, 360, num_frames):
    ax.view_init(elev=30, azim=angle)
    fig.canvas.draw()  # update the figure
    # Extract the image from the figure canvas as a numpy array
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    frames.append(image)

gif_filename = 'orbit.gif'
imageio.mimsave(gif_filename, frames, fps=20)
print(f"GIF saved as {gif_filename}")

plt.show()


# %%
