
#%%
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.onnx import SamOnnxModel

import onnxruntime
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic
# %%
def show_mask(mask, ax):
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))   



# %% example from meta/github readme, did not work
# checkpoint = "sam_vit_h_4b8939.pth"
# model_type = "vit_h"

# from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
# sam = sam_model_registry[model_type](checkpoint=checkpoint)
# mask_generator = SamAutomaticMaskGenerator(sam)
# masks = mask_generator.generate("signal-2024-07-02-214900_002.jpeg")


# %%
import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

sam = sam_model_registry[model_type](checkpoint=checkpoint)
mask_generator = SamAutomaticMaskGenerator(sam)

#%%
# Load image using OpenCV
# image_path = "signal-2024-07-02-214900_002.jpeg"
image_path = "signal-2024-06-30-172053_002.jpeg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

# Generate masks
masks = mask_generator.generate(image)

# Print or use the masks as needed
print(masks)

# %%

# randomize color:
def show_mask(mask, ax):
    color = np.concatenate([np.random.rand(3), [0.6]])  # Generate random color with alpha 0.6
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

# Plot the image and the masks
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(image)

# Iterate through the masks and plot each one
# for mask_dict in masks[0:100]:
for mask_dict in masks[0:20]:
    mask_array = mask_dict['segmentation']  # Access the mask array using the 'segmentation' key
    show_mask(mask_array, ax)

plt.axis('off')
plt.show()


# %%
