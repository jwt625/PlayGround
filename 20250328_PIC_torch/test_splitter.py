#%%
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from BasicComponents import YSplitter, YCombiner, DelayLine

class MZITestNetwork(nn.Module):
    def __init__(self, wavelength_points, delta_L=100e-6, n_eff=1.5):
        super(MZITestNetwork, self).__init__()
        self.splitter = YSplitter()
        self.delay_line = DelayLine(wavelength_points, delta_L, n_eff)
        self.combiner = YCombiner()
        self.register_buffer('wavelength_points', torch.tensor(wavelength_points, dtype=torch.float32))
        
    def forward(self, x):
        # Split the input tensor
        x_split = self.splitter(x)

        # Apply delay line to the second path
        x_delay = self.delay_line(x_split[:, 2:, :])  # Take the second path

        # Combine the outputs
        x_combined = torch.cat((x_split[:, :2, :], x_delay), dim=1)  # Combine first path and delayed second path
        output = self.combiner(x_combined)

        return output

# Example usage
#%%
# Define wavelength points
wavelength_points = np.linspace(1.5e-6, 1.6e-6, 5000)  # 5000 points from 1.5 to 1.6 μm

# Create MZI test network
mzi_test = MZITestNetwork(wavelength_points, delta_L=100e-6)

# Create example input (batch_size=1, 4 channels, N wavelength points)
batch_size = 1
N = len(wavelength_points)
input_field = torch.zeros(batch_size, 4, N)
input_field[:, 0, :] = 1.0  # Set real part of waveguide 1 to 1

# Forward pass
output_field = mzi_test(input_field)

# Convert to complex representation for visualization
output_complex_1 = torch.complex(output_field[0, 0], output_field[0, 1])

# Compute power (magnitude squared) for each wavelength point
power_output_1 = torch.abs(output_complex_1)**2

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(wavelength_points * 1e9, power_output_1.detach().numpy(), label='Output WG1 Power')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Power')
plt.title('MZI Output Power')
plt.legend()
plt.grid(True)
plt.show()
# %%
