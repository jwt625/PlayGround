
#%%
import numpy as np
import matplotlib.pyplot as plt


def generate_2d_mesh(x_min, x_max, y_min, y_max, x_points, y_points):
    x = np.linspace(x_min, x_max, x_points)
    y = np.linspace(y_min, y_max, y_points)
    xx, yy = np.meshgrid(x, y)
    return xx, yy

def calculate_distance(xx, yy, x0, y0, z0):
    distances = np.sqrt((xx - x0)**2 + (yy - y0)**2 + z0**2)
    return distances

def calculate_distance_and_azimuth(xx, yy, x0, y0, z0):
    distances = np.sqrt((xx - x0)**2 + (yy - y0)**2 + z0**2)
    azimuth = np.arctan2(yy - y0, xx - x0)
    return distances, azimuth



def create_frame(xx, yy, distances, point, frame_number):
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, distances, cmap='viridis')
    plt.colorbar(label='noise (dB)')
    plt.scatter(point[0], point[1], color='red', label=f'Point {point}')
    plt.title(f'2D Color Plot of noise (Frame {frame_number})')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.tight_layout()
    filename = f'frame_{frame_number:03d}.png'
    plt.savefig(filename)
    plt.close()
    return filename


#%%

# Parameters for the 2D mesh
x_min, x_max, y_min, y_max = -10, 10, -10, 10
x_points, y_points = 1000, 1000

# Generate the 2D mesh
xx, yy = generate_2d_mesh(x_min, x_max, y_min, y_max, x_points, y_points)

# Coordinates of the given point in space
x0, y0, z0 = 0, 0, 1

# Calculate distances
distances, thetas = calculate_distance_and_azimuth(xx, yy, x0, y0, z0)

rhos = 1.5 + 0.5*np.cos(thetas) + 0.1*np.cos(2*thetas + np.pi) + 0.2*np.cos(3*thetas)

# print(distances)

N0 = 1
alpha = 1   # Neper per km
N_atten = rhos*N0*np.exp(-alpha*distances)

N_atten_dB = 10*np.log10(N_atten) + 100




# %%

# Create a 2D color plot
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, N_atten_dB, cmap='viridis')
plt.colorbar(label='noise (dB)')
plt.scatter(x0, y0, color='red', label=f'Point ({x0}, {y0}, {z0})')
plt.title('noise (dB)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()






# %% polar plot
import numpy as np
import matplotlib.pyplot as plt

# Generate theta values
theta = np.linspace(0, 2 * np.pi, 1000)

# Compute rho values
# rho = 1.5 + 0.5*np.cos(theta)
rho = 1.5 + 0.5*np.cos(theta) + 0.1*np.cos(2*theta + np.pi) + 0.2*np.cos(3*theta)
# rho = np.cos(1*theta)

# Create a polar plot
plt.figure(figsize=(8, 6))
ax = plt.subplot(111, polar=True)
ax.plot(theta, rho)

# Add title and grid
ax.set_title(r'$\rho = 1 + \cos(\theta)$', va='bottom')
ax.grid(True)

# Show the plot
plt.show()


# %%
