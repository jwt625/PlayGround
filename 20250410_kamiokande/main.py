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
# image_path = 'image_converted.png'  # Update if path differs
# image = cv2.imread(image_path)

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN

# Load and prepare the image
image_path = 'image_converted.png' 
image = Image.open(image_path).convert('RGB')
image_np = np.array(image)

# Threshold out background
threshold = 50
mask = np.any(image_np > threshold, axis=-1)
coords = np.column_stack(np.where(mask))
colors = image_np[mask]

# Cluster based on spatial location
# eps = radius in pixels, min_samples = 1 to include single pixels
clustering = DBSCAN(eps=2, min_samples=11).fit(coords)

# Aggregate by cluster
detectors = []
labels = clustering.labels_
unique_labels = set(labels)
for label in unique_labels:
    if label == -1:
        continue  # skip noise

    indices = np.where(labels == label)[0]
    cluster_coords = coords[indices]
    cluster_colors = colors[indices]

    # Compute centroid and average color
    y_mean, x_mean = np.mean(cluster_coords, axis=0)
    avg_color = np.mean(cluster_colors, axis=0).astype(int)

    detectors.append({
        'x': int(round(x_mean)),
        'y': int(round(y_mean)),
        'color': tuple(avg_color)
    })

# Preview results
for d in detectors[:10]:
    print(d)
print(len(detectors))
# %%
# import matplotlib.pyplot as plt

# # Plot clustered detector positions with their average colors
# plt.figure(figsize=(10, 10))
# for det in detectors:
#     plt.scatter(det['x'], det['y'], color=np.array(det['color']) / 255.0, s=30)

# plt.gca().invert_yaxis()  # Match image coordinates
# plt.axis('equal')
# plt.grid(True)
# plt.title('Detected Detectors by Clustered Color + Position')
# plt.show()

# %% interactive
import plotly.express as px
import pandas as pd

# Convert to DataFrame for Plotly
df = pd.DataFrame(detectors)

# Normalize color values to 0–1 range for Plotly
df['r'] = df['color'].apply(lambda c: c[0] / 255.0)
df['g'] = df['color'].apply(lambda c: c[1] / 255.0)
df['b'] = df['color'].apply(lambda c: c[2] / 255.0)

# Build color strings in rgba format
df['rgba'] = df.apply(lambda row: f'rgba({row["color"][0]},{row["color"][1]},{row["color"][2]},1.0)', axis=1)

# Create interactive scatter plot
fig = px.scatter(
    df, x='x', y='y',
    color='rgba',
    color_discrete_map='identity',
    hover_data=['x', 'y', 'color'],
    title='Clustered Detector Pixels'
)

fig.update_traces(marker=dict(size=6))
fig.update_layout(
    yaxis=dict(autorange='reversed'),
    xaxis=dict(scaleanchor='y'),
    width=700,
    height=700
)

fig.show()

# %%
