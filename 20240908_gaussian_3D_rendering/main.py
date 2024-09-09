

#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def gaussian_3d(x, y, z, mu, sigma):
    """3D Gaussian function"""
    return np.exp(-((x-mu[0])**2 + (y-mu[1])**2 + (z-mu[2])**2) / (2*sigma**2))

def ray_march(start, direction, steps, step_size, light_pos, density_func):
    """Perform ray marching and return color"""
    transmittance = 1.0
    total_light = 0.0
    
    for _ in range(steps):
        current_pos = start + direction * step_size * _
        density = density_func(*current_pos)
        
        # Light direction
        light_dir = light_pos - current_pos
        light_dir /= np.linalg.norm(light_dir)
        
        # Simple lighting model
        light_intensity = max(0, np.dot(light_dir, direction))
        
        absorption = density * step_size
        total_light += transmittance * absorption * light_intensity
        transmittance *= np.exp(-absorption)
        
        if transmittance < 0.01:
            break
    
    return total_light

def render_gaussian(resolution=100, view_distance=5):
    x = y = z = np.linspace(-3, 3, resolution)
    X, Y, Z = np.meshgrid(x, y, z)
    
    mu = np.array([0, 0, 0])
    sigma = 1
    
    density_func = lambda x, y, z: gaussian_3d(x, y, z, mu, sigma)
    
    view_pos = np.array([view_distance, view_distance, view_distance])
    light_pos = np.array([5, 5, 5])
    
    image = np.zeros((resolution, resolution))
    
    for i in range(resolution):
        for j in range(resolution):
            ray_origin = view_pos
            ray_target = np.array([X[i,j,0], Y[i,j,0], 0])
            ray_direction = ray_target - ray_origin
            ray_direction /= np.linalg.norm(ray_direction)
            
            color = ray_march(ray_origin, ray_direction, 100, 0.1, light_pos, density_func)
            image[i,j] = color
    
    return image

# Render the Gaussian
image = render_gaussian()

# Display the result
plt.imshow(image, cmap='viridis')
plt.colorbar()
plt.title('3D Gaussian Distribution Rendering')
plt.show()


