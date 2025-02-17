
#%% chatGPT-4o
import matplotlib.pyplot as plt
import numpy as np

def draw_mzi(ax):
    # Define component positions
    source = (0, 1)
    bs1 = (1, 1)
    mirror1 = (1, 2)
    mirror2 = (3, 0)
    bs2 = (3, 1)
    detector1 = (3.5, 2)
    detector2 = (4, 1)
    
    # Draw optical paths
    paths = [
        (source, bs1),
        (bs1, mirror1),
        (mirror1, bs2),
        (bs1, mirror2),
        (mirror2, bs2),
        (bs2, detector1),
        (bs2, detector2)
    ]
    
    for (start, end) in paths:
        ax.plot([start[0], end[0]], [start[1], end[1]], 'r-', lw=2)
    
    # Draw components
    ax.scatter(*zip(*[bs1, bs2]), marker='s', color='black', label='Beam Splitter')
    ax.scatter(*zip(*[mirror1, mirror2]), marker='|', color='black', s=100, label='Mirror')
    ax.scatter(*zip(*[detector1, detector2]), marker='o', color='blue', label='Detector')
    ax.text(*source, ' Source', verticalalignment='center', fontsize=10)
    ax.text(*bs1, ' BS1', verticalalignment='bottom', fontsize=10)
    ax.text(*mirror1, ' M', verticalalignment='bottom', fontsize=10)
    ax.text(*mirror2, ' M', verticalalignment='bottom', fontsize=10)
    ax.text(*bs2, ' BS2', verticalalignment='bottom', fontsize=10)
    ax.text(*detector1, ' D1', verticalalignment='bottom', fontsize=10)
    ax.text(*detector2, ' D2', verticalalignment='bottom', fontsize=10)
    
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 2.5)
    ax.axis('off')

def draw_interference(ax):
    # Generate a sine wave to simulate interference pattern
    wavelengths = np.linspace(0, 10, 500)
    intensity = (np.cos(2 * np.pi * wavelengths) + 1) / 2  # Normalize between 0 and 1
    
    ax.plot(intensity, wavelengths, 'b-', lw=2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('Power')
    ax.set_ylabel('Wavelength')
    ax.set_title('Interference Pattern')

# Create figure and axes
fig, ax = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [2, 1]})
draw_mzi(ax[0])
draw_interference(ax[1])

plt.tight_layout()
plt.show()
