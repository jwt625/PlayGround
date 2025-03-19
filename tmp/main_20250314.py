

#%%

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Define the categories (the 5 axes)
# ---------------------------------------------------------
categories = [
    "Young's modulus (GPa)",
    "Hardness (GPa)",
    "Thermal conductivity (W/cm·K)",
    "1/(Friction coefficient)",
    "1/(Thermal expansion coefficient)"
]
N = len(categories)

# ---------------------------------------------------------
# 2. Define raw data for each material
#    (Approximate values gleaned from the example figure)
# ---------------------------------------------------------
materials = {
    'Diamond': [1100, 100, 30, 30, 14],
    'Si':       [190,  10,  1.5, 5,  3],
    'SiC':      [450,  28,  5,   15, 5],
    'GaAs':     [85,   7,   0.55,4,  2],
    'GaN':      [300,  15,  1.3, 3,  6]
}

# ---------------------------------------------------------
# 3. Normalize each axis so all data fits on [0, 1]
#    (This lets us share a single radial scale.)
# ---------------------------------------------------------
# Find max of each property across all materials
max_per_category = [0]*N
for i in range(N):
    max_per_category[i] = max(materials[m][i] for m in materials)

# Convert raw data to normalized values
normalized_data = {}
for m in materials:
    raw_values = materials[m]
    norm_values = [raw_values[i]/max_per_category[i] for i in range(N)]
    normalized_data[m] = norm_values

# ---------------------------------------------------------
# 4. Prepare the angles for the radar chart
# ---------------------------------------------------------
# Angles equally spaced around a circle
angles = [n / float(N) * 2 * np.pi for n in range(N)]
# Ensure the polygon is closed by appending the start angle at the end
angles += angles[:1]

# ---------------------------------------------------------
# 5. Create the radar chart
# ---------------------------------------------------------
fig = plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)

# Make the first axis start at the top (90 degrees) and go clockwise
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# Draw one axis per category + add labels
plt.xticks(angles[:-1], categories, fontsize=10)

# Optionally fix the radial range to [0, 1] since we're normalizing
ax.set_rlabel_position(0)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2","0.4","0.6","0.8","1.0"], color="gray", size=8)
plt.ylim(0, 1)

# ---------------------------------------------------------
# 6. Plot each material's polygon
# ---------------------------------------------------------
colors = {
    'Diamond': 'red',
    'Si': 'blue',
    'SiC': 'purple',
    'GaAs': 'orange',
    'GaN': 'green'
}

for material, norm_values in normalized_data.items():
    # Close the polygon by repeating the first value at the end
    values = norm_values + norm_values[:1]
    ax.plot(angles, values, color=colors[material], linewidth=2, label=material)
    ax.fill(angles, values, color=colors[material], alpha=0.1)

plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.title("Approximate Radar Chart for Semiconductor Properties", y=1.08)
plt.tight_layout()
plt.show()


# %% v2

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Define the categories (the 5 axes)
# ---------------------------------------------------------
categories = [
    "Young's modulus (GPa)",
    "Hardness (GPa)",
    "Thermal conductivity (W/m·K)",
    "1/(Friction coefficient)",
    "1/(Thermal expansion coefficient)"
]
N = len(categories)

# ---------------------------------------------------------
# 2. Define raw data for each material
#    (Average values taken where ranges are provided)
# ---------------------------------------------------------
materials = {
    'Diamond': [1125, 85, 2000, 1/0.04, 1/1.1],
    'Si':      [157.5, 10, 150, 1/0.25, 1/2.95],
    'SiC':     [555, 26.5, 420, 1/0.25, 1/4.15],
    'GaAs':    [85, 7, 55, 1/0.15, 1/5.8],
    'GaN':     [300, 15, 130, 1/0.3, 1/5.6]
}

# ---------------------------------------------------------
# 3. Normalize each axis so all data fits on [0, 1]
# ---------------------------------------------------------
# Find max of each property across all materials
# max_per_category = [max(materials[m][i] for m in materials) for i in range(N)]
max_per_category = [1125, 100, 3000, 30, 1]

# Convert raw data to normalized values
normalized_data = {
    m: [materials[m][i] / max_per_category[i] for i in range(N)]
    for m in materials
}

# ---------------------------------------------------------
# 4. Prepare the angles for the radar chart
# ---------------------------------------------------------
# Angles equally spaced around a circle
angles = [n / float(N) * 2 * np.pi for n in range(N)]
# Ensure the polygon is closed by appending the start angle at the end
angles += angles[:1]

# ---------------------------------------------------------
# 5. Create the radar chart
# ---------------------------------------------------------
fig = plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)

# Make the first axis start at the top (90 degrees) and go clockwise
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# Draw one axis per category + add labels
plt.xticks(angles[:-1], categories, fontsize=10)

# Optionally fix the radial range to [0, 1] since we're normalizing
ax.set_rlabel_position(0)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2","0.4","0.6","0.8","1.0"], color="gray", size=8)
plt.ylim(0, 1)

# ---------------------------------------------------------
# 6. Plot each material's polygon
# ---------------------------------------------------------
colors = {
    'Diamond': 'red',
    'Si': 'blue',
    'SiC': 'green',
    'GaAs': 'purple',
    'GaN': 'orange'
}

for material, norm_values in normalized_data.items():
    # Close the polygon by repeating the first value at the end
    values = norm_values + norm_values[:1]
    ax.plot(angles, values, color=colors[material], linewidth=2, label=material)
    ax.fill(angles, values, color=colors[material], alpha=0.1)

plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.title("Radar Chart for Semiconductor Properties", y=1.08)
plt.tight_layout()
plt.show()




# %%
