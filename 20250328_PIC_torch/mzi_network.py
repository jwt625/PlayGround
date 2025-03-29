#%%
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class DirectionalCoupler(nn.Module):
    def __init__(self, coupling_ratio=0.5):
        """
        Directional coupler layer
        coupling_ratio: power coupling ratio (0 to 1)
        """
        super(DirectionalCoupler, self).__init__()
        self.coupling_ratio = coupling_ratio
        # Create the coupling matrix
        k = np.sqrt(coupling_ratio)
        t = np.sqrt(1 - coupling_ratio)
        self.coupling_matrix = torch.tensor([
            [t, -1j*k],
            [-1j*k, t]
        ], dtype=torch.complex64)
        
    def forward(self, x):
        """
        x: input tensor of shape (batch_size, 4, N)
           where 4 represents [real1, imag1, real2, imag2]
           and N is the number of wavelength points
        """
        # Convert to complex representation
        x_complex = torch.complex(x[:, 0], x[:, 1])  # waveguide 1
        y_complex = torch.complex(x[:, 2], x[:, 3])  # waveguide 2
        
        # Stack complex numbers
        z = torch.stack([x_complex, y_complex], dim=1)  # shape: (batch_size, 2, N)
        
        # Apply coupling matrix to each wavelength point
        # Reshape for batch matrix multiplication
        z_reshaped = z.permute(0, 2, 1)  # shape: (batch_size, N, 2)
        z_out = torch.matmul(z_reshaped, self.coupling_matrix)  # shape: (batch_size, N, 2)
        z_out = z_out.permute(0, 2, 1)  # shape: (batch_size, 2, N)
        
        # Convert back to real/imag representation
        out = torch.stack([
            z_out[:, 0].real,  # real part of waveguide 1
            z_out[:, 0].imag,  # imag part of waveguide 1
            z_out[:, 1].real,  # real part of waveguide 2
            z_out[:, 1].imag   # imag part of waveguide 2
        ], dim=1)
        
        return out

class DelayLine(nn.Module):
    def __init__(self, wavelength_points, delta_L=1e-6, n_eff=1.5):
        """
        Delay line layer
        wavelength_points: array of wavelength points
        delta_L: length difference between arms (m)
        n_eff: effective refractive index
        """
        super(DelayLine, self).__init__()
        self.wavelength_points = wavelength_points
        self.delta_L = delta_L
        self.n_eff = n_eff
        
        # Calculate phase difference for each wavelength
        k = 2 * np.pi * n_eff / wavelength_points
        self.phase_diff = torch.tensor(k * delta_L, dtype=torch.float32)
        
    def forward(self, x):
        """
        x: input tensor of shape (batch_size, 4, N)
        """
        # Convert to complex representation
        x_complex = torch.complex(x[:, 0], x[:, 1])  # waveguide 1
        y_complex = torch.complex(x[:, 2], x[:, 3])  # waveguide 2
        
        # Apply phase difference to waveguide 2
        y_complex = y_complex * torch.exp(1j * self.phase_diff)
        
        # Convert back to real/imag representation
        out = torch.stack([
            x_complex.real,    # real part of waveguide 1
            x_complex.imag,    # imag part of waveguide 1
            y_complex.real,    # real part of waveguide 2
            y_complex.imag     # imag part of waveguide 2
        ], dim=1)
        
        return out

class MZINetwork(nn.Module):
    def __init__(self, wavelength_points, coupling_ratio=0.5, delta_L=1e-6, n_eff=1.5):
        """
        Complete MZI network
        wavelength_points: array of wavelength points
        coupling_ratio: power coupling ratio of directional couplers
        delta_L: length difference between arms (m)
        n_eff: effective refractive index
        """
        super(MZINetwork, self).__init__()
        self.dc1 = DirectionalCoupler(coupling_ratio)
        self.delay = DelayLine(wavelength_points, delta_L, n_eff)
        self.dc2 = DirectionalCoupler(coupling_ratio)
        
    def forward(self, x):
        """
        x: input tensor of shape (batch_size, 4, N)
           where 4 represents [real1, imag1, real2, imag2]
           and N is the number of wavelength points
        """
        x = self.dc1(x)      # First directional coupler
        x = self.delay(x)    # Delay line
        x = self.dc2(x)      # Second directional coupler
        return x

# Example usage
#%%
# if __name__ == "__main__":
# Define wavelength points
wavelength_points = np.linspace(1.5e-6, 1.6e-6, 1000)  # 100 points from 1.5 to 1.6 μm

# Create network
mzi = MZINetwork(wavelength_points, delta_L=100e-6)

# Create example input (batch_size=1, 4 channels, N wavelength points)
batch_size = 1
N = len(wavelength_points)
input_field = torch.zeros(batch_size, 4, N)
input_field[:, 0, :] = 1.0  # Set real part of waveguide 1 to 1

# Forward pass
output_field = mzi(input_field)

# Print shapes
print(f"Input shape: {input_field.shape}")
print(f"Output shape: {output_field.shape}")
# Convert to complex representation for visualization
input_complex_1 = torch.complex(input_field[0, 0], input_field[0, 1])
input_complex_2 = torch.complex(input_field[0, 2], input_field[0, 3])
output_complex_1 = torch.complex(output_field[0, 0], output_field[0, 1])
output_complex_2 = torch.complex(output_field[0, 2], output_field[0, 3])

# Plot results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(wavelength_points * 1e9, (torch.abs(input_complex_1).numpy())**2, label='Input WG1')
plt.plot(wavelength_points * 1e9, (torch.abs(input_complex_2).numpy())**2, label='Input WG2')
plt.plot(wavelength_points * 1e9, (torch.abs(output_complex_1).numpy())**2, label='Output WG1')
plt.plot(wavelength_points * 1e9, (torch.abs(output_complex_2).numpy())**2, label='Output WG2')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Power')
plt.title('MZI Response')
plt.legend()
plt.grid(True)
plt.show()


#%%
from torch.utils.tensorboard import SummaryWriter

# Visualize network structure using TensorBoard
writer = SummaryWriter('runs/mzi_network')

# Create a dummy input tensor for visualization
dummy_input = torch.zeros(1, 4, N)  # batch_size=1, 4 channels, N wavelength points
dummy_input[:, 0, :] = 1.0  # Set real part of waveguide 1 to 1

# Add the network graph to TensorBoard
writer.add_graph(mzi, dummy_input)
writer.close()

# To view the network structure, run:
# tensorboard --logdir=runs/mzi_network
# Then open your browser and go to http://localhost:6006

# Also log the network parameters
for name, param in mzi.named_parameters():
    writer.add_histogram(f'{name}_values', param.data)
    if param.grad is not None:
        writer.add_histogram(f'{name}_grads', param.grad)
writer.close()
# %%
