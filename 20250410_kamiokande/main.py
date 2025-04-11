#%%
import cv2
import numpy as np
import os

# Check current working directory
current_path = os.getcwd()
print("Current working directory:", current_path)

#%% convert gif to png

from PIL import Image

# Convert and save as PNG
image_path = '20250410T1118.gif'
image = Image.open(image_path)
image.save('image_converted.png')

#%%

# Load the image
image_path = 'image_converted.png'  # Update if path differs
image = cv2.imread(image_path)
# Convert from BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Create a mask to filter out black background
# A pixel is considered "non-black" if any channel is above a threshold
threshold = 10
mask = np.any(image_rgb > threshold, axis=-1)

# Get coordinates of non-black pixels
coords = np.column_stack(np.where(mask))

# Get the RGB values at those coordinates
colors = image_rgb[mask]

# Combine coordinates and colors
detector_pixels = [
    {'x': int(x), 'y': int(y), 'color': (int(r), int(g), int(b))}
    for (y, x), (r, g, b) in zip(coords, colors)
]

# Example: print first 10 results
for entry in detector_pixels[:10]:
    print(entry)

# %%
