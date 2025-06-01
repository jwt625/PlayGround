#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

# Create x values for positive and negative ranges
x_pos = np.linspace(0.01, 5, 1000)  # Avoid x=0 to prevent singularity
x_neg = np.linspace(-6, -0.01, 1000)  # Avoid x=0 to prevent singularity

# Calculate y values
y_pos = gamma(x_pos + 1)**2
y_neg = 1 / gamma(1 - x_neg)**2

# Create the plot
plt.figure(figsize=(10, 8))

# Plot both parts
plt.plot(x_pos, y_pos, 'b-')
plt.plot(x_neg, y_neg, 'r-')

# Set axis limits
plt.xlim(-6, 5)
plt.ylim(-2, 5)

# Set x-axis ticks at every integer
plt.xticks(np.arange(-6, 6))

# Set equal aspect ratio for x and y axes
plt.gca().set_aspect('equal')

# Add labels and title
plt.xlabel('x')
plt.ylabel('y')
# plt.title('Gamma Function Plot')
plt.grid(True)

# Show the plot
plt.show() 
# %%
