

#%% chatGPT 4o, trash:
import numpy as np
import matplotlib.pyplot as plt

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 5))

# Define key points
input_light = (1, 2)
bs1 = (2, 2)
mirror1 = (2, 3)
mirror2 = (3, 1)
bs2 = (4, 2)
detector = (5, 2)

# Draw optical components
ax.plot([bs1[0], mirror1[0]], [bs1[1], mirror1[1]], 'k', lw=3)  # Beam splitter 1 to mirror 1
ax.plot([bs1[0], mirror2[0]], [bs1[1], mirror2[1]], 'k', lw=3)  # Beam splitter 1 to mirror 2
ax.plot([mirror1[0], bs2[0]], [mirror1[1], bs2[1]], 'k', lw=3)  # Mirror 1 to BS2
ax.plot([mirror2[0], bs2[0]], [mirror2[1], bs2[1]], 'k', lw=3)  # Mirror 2 to BS2
ax.plot([bs2[0], detector[0]], [bs2[1], detector[1]], 'k', lw=3)  # BS2 to detector

# Draw light paths (red)
ax.arrow(input_light[0], input_light[1], 0.8, 0, color='red', width=0.05, head_width=0.15)  # Input light
ax.arrow(bs1[0], bs1[1], 0, 0.8, color='red', width=0.05, head_width=0.15)  # Upwards path
ax.arrow(bs1[0], bs1[1], 0.8, -0.8, color='red', width=0.05, head_width=0.15)  # Diagonal path
ax.arrow(mirror1[0], mirror1[1], 0.8, -0.8, color='red', width=0.05, head_width=0.15)  # Back to BS2
ax.arrow(mirror2[0], mirror2[1], 0.8, 0.8, color='red', width=0.05, head_width=0.15)  # Back to BS2
ax.arrow(bs2[0], bs2[1], 0.8, 0, color='red', width=0.05, head_width=0.15)  # Output path

# Draw phase delay label
ax.text(2.8, 2.8, r'$\phi = k\Delta L$', fontsize=12, color='black')

# Draw detector signal (blue arrow)
ax.arrow(detector[0], detector[1], 0.5, 0.5, color='blue', width=0.05, head_width=0.15)
ax.text(5.3, 2.5, 'Electrical Signal', fontsize=10, color='blue')

# Draw power vs wavelength plot
x = np.linspace(0, 10, 100)
y = 0.5 * (1 + np.cos(x))  # Simplified interference pattern
ax.plot(5.5 + 0.5 * x / 10, 1.5 + 0.5 * y, 'k', lw=2)  # Normalized plot near detector
ax.text(6, 1.5, r'$P(\lambda)$', fontsize=12)

# Labels
ax.text(0.5, 2, 'Input Light', fontsize=12, color='red')
ax.text(4.5, 2.2, 'Beam Splitter', fontsize=10, color='black')
ax.text(5.2, 2, 'Detector', fontsize=12, color='black')

# Hide axes
ax.set_xticks([])
ax.set_yticks([])
ax.set_frame_on(False)

# Show the figure
plt.show()

# %%











#%% Le Chat:
import matplotlib.pyplot as plt
import numpy as np

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 5))

# Draw the components
# Input light
ax.arrow(0.5, 2.5, 0.5, 0, head_width=0.1, head_length=0.1, fc='red', ec='red')
ax.text(0.5, 2.6, 'Input Light', ha='center', va='bottom', color='black')

# First beam splitter
ax.plot([1, 1], [2, 3], color='black', lw=2)
ax.plot([0.9, 1.1], [2.5, 2.5], color='black', lw=2)
ax.text(1, 2.6, 'BS1', ha='center', va='bottom', color='black')

# Light paths
ax.arrow(1, 2.5, 2, 0, head_width=0.1, head_length=0.1, fc='red', ec='red')
ax.arrow(1, 2.5, 2, 0.5, head_width=0.1, head_length=0.1, fc='red', ec='red')

# Phase delay
ax.text(2, 2.8, r'$k\Delta L$', ha='center', va='bottom', color='black')

# Second beam splitter
ax.plot([3, 3], [2, 3], color='black', lw=2)
ax.plot([2.9, 3.1], [2.5, 2.5], color='black', lw=2)
ax.text(3, 2.6, 'BS2', ha='center', va='bottom', color='black')

# Output light to detector
ax.arrow(3, 2.5, 1, 0, head_width=0.1, head_length=0.1, fc='red', ec='red')

# Photodetector
ax.plot([4, 4.5], [2.4, 2.4], color='black', lw=2)
ax.plot([4, 4.5], [2.6, 2.6], color='black', lw=2)
ax.plot([4, 4], [2.4, 2.6], color='black', lw=2)
ax.plot([4.5, 4.5], [2.4, 2.6], color='black', lw=2)
ax.text(4.25, 2.7, 'Detector', ha='center', va='bottom', color='black')

# Electrical signal
ax.arrow(4.5, 2.5, 0.5, 0, head_width=0.1, head_length=0.1, fc='blue', ec='blue')

# Power vs wavelength plot
ax_power = fig.add_axes([0.6, 0.1, 0.2, 0.2])
wavelengths = np.linspace(0, 10, 100)
power = 0.5 * np.sin(wavelengths) + 0.5
ax_power.plot(wavelengths, power, color='black')
ax_power.set_xticks([])
ax_power.set_yticks([])
ax_power.set_title('Power vs Wavelength', fontsize=10)
ax_power.text(5, 1.1, '$I_{max}$', ha='center', va='bottom', color='black', fontsize=8)
ax_power.text(5, 0, '0', ha='center', va='top', color='black', fontsize=8)

# Set limits and display the plot
ax.set_xlim(0, 5)
ax.set_ylim(1.5, 3.5)
ax.set_aspect('equal')
ax.axis('off')
plt.show()













# %% Claude
import matplotlib.pyplot as plt
import numpy as np

# Create figure and axis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={'width_ratios': [2, 1]})

# MZI Setup (left plot)
# Beam splitters
ax1.plot([2, 3], [2, 3], 'k-', linewidth=2)  # BS1
ax1.plot([2, 3], [3, 2], 'k-', linewidth=2)
ax1.plot([6, 7], [2, 3], 'k-', linewidth=2)  # BS2
ax1.plot([6, 7], [3, 2], 'k-', linewidth=2)

# Light path arrows
# Input
ax1.arrow(0, 2.5, 1.8, 0, head_width=0.1, head_length=0.2, fc='r', ec='r')
# After BS1
ax1.arrow(3.2, 3, 2.6, 0, head_width=0.1, head_length=0.2, fc='r', ec='r')  # Upper path
ax1.arrow(3.2, 2, 2.6, 0, head_width=0.1, head_length=0.2, fc='r', ec='r')  # Lower path
# After BS2
ax1.arrow(7.2, 2.5, 1.8, 0, head_width=0.1, head_length=0.2, fc='r', ec='r')

# Phase delay text
ax1.text(4.5, 3.2, r'k$\Delta$L', fontsize=12)

# Detector
detector = plt.Rectangle((9.2, 2), 0.5, 1, fc='k')
ax1.add_patch(detector)

# Electrical signal from detector
ax1.arrow(9.9, 2.5, 1, 0, head_width=0.1, head_length=0.2, fc='b', ec='b')

# Add labels
ax1.text(0.5, 2.8, 'Input', fontsize=10)
ax1.text(2.2, 3.3, 'BS1', fontsize=10)
ax1.text(6.2, 3.3, 'BS2', fontsize=10)
ax1.text(9.2, 1.7, 'PD', fontsize=10)

# Power vs Wavelength plot (right plot)
wavelength = np.linspace(0, 10, 1000)
power = 0.5 * (1 + np.cos(2 * np.pi * wavelength))

ax2.plot(wavelength, power, 'b-')
ax2.set_ylim([-0.1, 1.1])
ax2.set_title('Power vs Wavelength')
ax2.text(-2, 0.5, 'Power', rotation=90, ha='center', va='center')
ax2.text(5, -0.2, 'Wavelength', ha='center', va='center')
ax2.text(-1, 1, 'I_max', ha='right', va='center')
ax2.text(-1, 0, '0', ha='right', va='center')

# Remove ticks from both plots
ax1.set_xticks([])
ax1.set_yticks([])
ax2.set_xticks([])
ax2.set_yticks([])

# Set equal aspect ratio for MZI plot
ax1.set_aspect('equal')

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()
















# %% Gemini
import matplotlib.pyplot as plt
import numpy as np

# Define component positions
input_x = 0
splitter1_x = 1
mirror1_x = 2
mirror2_x = 2
splitter2_x = 3
output_x = 4
detector_x = 5

# Define y positions
input_y = 0
splitter1_y = 0
mirror1_y = 1
mirror2_y = -1
splitter2_y = 0
output_y = 0
detector_y = 0
plot_x_start = 6
plot_y_start = -1
plot_width = 2
plot_height = 2

# Create the figure and axes
fig, ax = plt.subplots(figsize=(10, 4))

# Plot the components
ax.plot([input_x, input_x], [-0.5, 0.5], 'k-', linewidth=2)  # Input light source
ax.plot([splitter1_x, splitter1_x], [-0.5, 0.5], 'k-', linewidth=2)  # Beam splitter 1
ax.plot([mirror1_x, mirror1_x], [0.5, 1.5], 'k-', linewidth=2)  # Mirror 1
ax.plot([mirror2_x, mirror2_x], [-1.5, -0.5], 'k-', linewidth=2)  # Mirror 2
ax.plot([splitter2_x, splitter2_x], [-0.5, 0.5], 'k-', linewidth=2)  # Beam splitter 2
ax.plot([detector_x, detector_x], [-0.5, 0.5], 'k-', linewidth=2)  # Detector

# Plot the light paths
ax.arrow(input_x, 0, splitter1_x - input_x, 0, head_width=0.2, head_length=0.3, fc='r', ec='r', linewidth=2)  # Input light
ax.arrow(splitter1_x, 0, mirror1_x - splitter1_x, mirror1_y - splitter1_y, head_width=0.2, head_length=0.3, fc='r', ec='r', linewidth=2) # path to mirror 1
ax.arrow(mirror1_x, mirror1_y, splitter2_x - mirror1_x, splitter2_y - mirror1_y, head_width=0.2, head_length=0.3, fc='r', ec='r', linewidth=2, linestyle = '--') # path from mirror 1 to splitter 2
ax.arrow(splitter1_x, 0, mirror2_x - splitter1_x, mirror2_y - splitter1_y, head_width=0.2, head_length=0.3, fc='r', ec='r', linewidth=2) # path to mirror 2
ax.arrow(mirror2_x, mirror2_y, splitter2_x - mirror2_x, splitter2_y - mirror2_y, head_width=0.2, head_length=0.3, fc='r', ec='r', linewidth=2, linestyle = '--') # path from mirror 2 to splitter 2
ax.arrow(splitter2_x, 0, detector_x - splitter2_x, 0, head_width=0.2, head_length=0.3, fc='r', ec='r', linewidth=2)  # Output light

# Plot the electrical signal
ax.arrow(detector_x, 0, 1, 0, head_width=0.2, head_length=0.3, fc='b', ec='b', linewidth=2, linestyle='--')  # Electrical signal

# Add labels and title
ax.text(input_x -0.2, input_y, 'Input', ha='right', va='center')
ax.text(splitter1_x -0.2, splitter1_y, 'BS1', ha='right', va='center')
ax.text(mirror1_x -0.2, mirror1_y + 0.2, 'M1', ha='right', va='bottom')
ax.text(mirror2_x -0.2, mirror2_y - 0.2, 'M2', ha='right', va='top')
ax.text(splitter2_x -0.2, splitter2_y, 'BS2', ha='right', va='center')
ax.text(detector_x -0.2, detector_y, 'PD', ha='right', va='center')
ax.text((mirror1_x + splitter2_x)/2, (mirror1_y + splitter2_y)/2 + 0.2, r'$k\Delta L$', ha='center', va='bottom')

# Power vs. wavelength plot
x = np.linspace(0, 1, 100)
y = 0.5 * (1 + np.cos(2 * np.pi * x)) # Example fluctuating power
ax.plot(plot_x_start + x * plot_width, plot_y_start + y * plot_height, 'k-', linewidth=1)
ax.text(plot_x_start + plot_width/2, plot_y_start + plot_height + 0.2, 'Power vs. Wavelength', ha='center', va='bottom')


# Set axis limits and remove ticks
ax.set_xlim(-1, plot_x_start + plot_width + 1)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Show the plot
plt.title('Mach-Zehnder Interferometer')
plt.tight_layout()
plt.show()

# %%
