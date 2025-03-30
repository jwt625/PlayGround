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

# create a class to nest the nested MZI network again
class NestedNestedMZINetwork(nn.Module):
    def __init__(self, wavelength_points, dL1=100e-6, dL2=50e-6, dL3=100e-6, n_eff=1.5):
        super(NestedNestedMZINetwork, self).__init__()
        self.splitter = YSplitter()
        self.nmzi1 = NestedMZINetwork(wavelength_points, dL1, dL2, dL3, n_eff)
        self.nmzi2 = NestedMZINetwork(wavelength_points, dL1, dL2, dL3, n_eff)
        self.delay_line = DelayLine(wavelength_points, dL1, n_eff)
        self.combiner = YCombiner()
        
    def forward(self, x):
        x_split = self.splitter(x)
        output1 = self.nmzi1(x_split[:, :2, :])
        output2 = self.nmzi2(x_split[:, 2:, :])
        output2 = self.delay_line(output2)
        combined_output = torch.cat((output1, output2), dim=1)
        final_output = self.combiner(combined_output)
        return final_output


def compute_cost(output_field, wavelength_points):
    """
    Compute the cost function based on the output power.
    The cost function penalizes the output power outside the range [1540 nm, 1560 nm]
    and encourages it to be 1 within that range.
    """
    # Convert wavelength points to nm for easier comparison
    wavelength_nm = wavelength_points * 1e9

    # Calculate output power
    output_complex_1 = torch.complex(output_field[0, 0], output_field[0, 1])
    power_output_1 = torch.abs(output_complex_1)**2


    # Create a mask for the desired wavelength range
    mask = (wavelength_nm >= 1540) & (wavelength_nm <= 1560)

    # Calculate the cost
    cost = torch.mean((power_output_1[mask] - 1) ** 2)  # Encourage power to be 1 in the range
    cost += torch.mean(power_output_1[~mask] ** 2)  # Encourage power to be 0 outside the range

    return cost

# Example usage
#%%
# Define wavelength points
wavelength_points = np.linspace(1.5e-6, 1.6e-6, 5000)  # 5000 points from 1.5 to 1.6 μm

# Create nested MZI test network
nested_mzi_test = NestedNestedMZINetwork(wavelength_points, dL1=100e-6, dL2=50e-6, dL3=100e-6)

# Create example input (batch_size=1, 4 channels, N wavelength points)
batch_size = 1
N = len(wavelength_points)
input_field = torch.zeros(batch_size, 4, N)
input_field[:, 0, :] = 1.0  # Set real part of waveguide 1 to 1

# Set up optimizer
optimizer = torch.optim.Adam([nested_mzi_test.nmzi1.mzi1.delay_line.delta_L, 
                            nested_mzi_test.nmzi1.mzi2.delay_line.delta_L, 
                            nested_mzi_test.nmzi1.delay_line.delta_L,
                            nested_mzi_test.nmzi2.mzi1.delay_line.delta_L,
                            nested_mzi_test.nmzi2.mzi2.delay_line.delta_L,
                            nested_mzi_test.nmzi2.delay_line.delta_L,
                            nested_mzi_test.delay_line.delta_L], lr=1e-6)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # Forward pass
    output_field = nested_mzi_test(input_field)
    
    # Compute cost
    cost = compute_cost(output_field, wavelength_points)
    
    # Backward pass
    cost.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Cost: {cost.item():.6e}")
        # print all parameter values with names
        for name, param in nested_mzi_test.named_parameters():
            print(f"{name}: {param.data}")

# Final output power plot
output_field = nested_mzi_test(input_field)
output_complex_1 = torch.complex(output_field[0, 0], output_field[0, 1])
power_output_1 = torch.abs(output_complex_1)**2

plt.figure(figsize=(10, 6))
plt.plot(wavelength_points * 1e9, power_output_1.detach().numpy(), label='Output WG1 Power')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Power')
plt.title('Nested MZI Output Power After Optimization')
plt.legend()
plt.grid(True)
plt.show()

#%%
from torch.utils.tensorboard import SummaryWriter

# Visualize network structure using TensorBoard
writer = SummaryWriter('runs/nested_mzi_network')

# Create a dummy input tensor for visualization
dummy_input = torch.zeros(1, 4, N)  # batch_size=1, 4 channels, N wavelength points
dummy_input[:, 0, :] = 1.0  # Set real part of waveguide 1 to 1

# Add the network graph to TensorBoard
writer.add_graph(nested_mzi_test, dummy_input)
writer.close()

# To view the network structure, run:
# tensorboard --logdir=runs/mzi_network
# Then open your browser and go to http://localhost:6006

# Log the network parameters
for name, param in nested_mzi_test.named_parameters():
    writer.add_histogram(f'{name}_values', param.data)
    if param.grad is not None:
        writer.add_histogram(f'{name}_grads', param.grad)
writer.close()

# %%
