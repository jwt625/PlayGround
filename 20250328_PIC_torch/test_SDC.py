#%% test segmented directional coupler

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from DC_CMT import DirectionalCouplerCoupledMode, SegmentedDirectionalCoupler


#%%

# Define wavelength points (e.g., from 1.5 μm to 1.6 μm)
wavelength_points = np.linspace(1.3e-6, 1.7e-6, 2000)

# Create an instance of the updated directional coupler
sdc = SegmentedDirectionalCoupler(wavelength_points, total_length=0.00000195, num_segments=10)

# Create an example input:
# Assume batch_size=1, 4 channels (real/imag for both waveguides), N wavelength points.
batch_size = 1
N = len(wavelength_points)
input_field = torch.zeros(batch_size, 4, N, dtype=torch.float32)
input_field[:, 0, :] = 1.0  # set the real part of waveguide 1 to 1

# Forward pass through the directional coupler
output_field = sdc(input_field)

print("Input shape:", input_field.shape)
print("Output shape:", output_field.shape)
print("Learnable parameters:")
for name, param in sdc.named_parameters():
    print(f"  {name}: {param.shape}")




# %%
