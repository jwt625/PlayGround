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


class NestedMZINetwork(nn.Module):
    def __init__(self, wavelength_points, dL1=100e-6, dL2=100e-6, dL3=100e-6, n_eff=1.5):
        super(NestedMZINetwork, self).__init__()
        self.splitter = YSplitter()
        self.mzi1 = MZITestNetwork(wavelength_points, dL1, n_eff)
        self.mzi2 = MZITestNetwork(wavelength_points, dL2, n_eff)
        self.delay_line = DelayLine(wavelength_points, dL3, n_eff)
        self.combiner = YCombiner()  # Combiner for the final output
        self.register_buffer('wavelength_points', torch.tensor(wavelength_points, dtype=torch.float32))
        
    def forward(self, x):
        # Split the input tensor
        x_split = self.splitter(x)
        # Pass through the first MZI
        output1 = self.mzi1(x_split[:, :2, :])  # Apply mzi1 to the first path
        # Pass through the second MZI
        output2 = self.mzi2(x_split[:, 2:, :])  # Apply mzi2 to the second path
        # apply extra delay to the second path
        output2 = self.delay_line(output2)
        # Combine the outputs from both MZIs
        combined_output = torch.cat((output1, output2), dim=1)  # Combine outputs from both MZIs
        final_output = self.combiner(combined_output)  # Final combiner
        
        return final_output


# Example usage
#%%
# Define wavelength points
wavelength_points = np.linspace(1.5e-6, 1.6e-6, 5000)  # 5000 points from 1.5 to 1.6 μm

# Create MZI test network
# mzi_test = MZITestNetwork(wavelength_points, delta_L=100e-6)
mzi_test = NestedMZINetwork(wavelength_points, dL1=100e-6, dL2=50e-6, dL3=100e-6)

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

# %% test MZI in MZI


