#%%
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from DC_CMT import DirectionalCouplerCoupledMode

class DirectionalCoupler(nn.Module):
    def __init__(self, coupling_ratio=0.5):
        """
        Directional coupler layer using nn.Linear
        coupling_ratio: initial power coupling ratio (0 to 1)
        """
        super(DirectionalCoupler, self).__init__()
        # Initialize coupling matrix as learnable parameters
        k = np.sqrt(coupling_ratio)
        t = np.sqrt(1 - coupling_ratio)
        self.coupling_matrix = nn.Parameter(torch.tensor([
            [t, -1j*k],
            [-1j*k, t]
        ], dtype=torch.complex64))
        
        # Add learnable phase parameters
        self.phase = nn.Parameter(torch.zeros(2, dtype=torch.float32))
        
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
        z_reshaped = z.permute(0, 2, 1)  # shape: (batch_size, N, 2)
        z_out = torch.matmul(z_reshaped, self.coupling_matrix)  # shape: (batch_size, N, 2)
        
        # Apply learnable phase shifts
        phase_matrix = torch.exp(1j * self.phase)
        z_out = z_out * phase_matrix
        
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
        Delay line layer using nn.Linear
        wavelength_points: array of wavelength points
        delta_L: initial length difference between arms (m)
        n_eff: initial effective refractive index
        """
        super(DelayLine, self).__init__()
        # Convert wavelength points to tensor and make it a buffer (not learnable)
        self.register_buffer('wavelength_points', torch.tensor(wavelength_points, dtype=torch.float32))
        
        # Make parameters learnable
        self.delta_L = nn.Parameter(torch.tensor(delta_L, dtype=torch.float32))
        self.n_eff = nn.Parameter(torch.tensor(n_eff, dtype=torch.float32))
        
        # Add learnable phase parameters
        self.phase = nn.Parameter(torch.zeros(2, dtype=torch.float32))
        
    def forward(self, x):
        """
        x: input tensor of shape (batch_size, 4, N)
        """
        # Convert to complex representation
        x_complex = torch.complex(x[:, 0], x[:, 1])  # waveguide 1
        y_complex = torch.complex(x[:, 2], x[:, 3])  # waveguide 2
        
        # Calculate phase difference using learnable parameters
        k = 2 * torch.pi * self.n_eff / self.wavelength_points
        phase_diff = k * self.delta_L
        
        # Apply phase difference to waveguide 2
        y_complex = y_complex * torch.exp(1j * phase_diff)
        
        # Apply learnable phase shifts
        phase_matrix = torch.exp(1j * self.phase)
        x_complex = x_complex * phase_matrix[0]
        y_complex = y_complex * phase_matrix[1]
        
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
        Complete MZI network with learnable parameters
        """
        super(MZINetwork, self).__init__()
        # self.dc1 = DirectionalCoupler(coupling_ratio)
        self.dc1 = DirectionalCouplerCoupledMode(wavelength_points, L=0.00000195)
        self.delay = DelayLine(wavelength_points, delta_L, n_eff)
        # self.dc2 = DirectionalCoupler(coupling_ratio)
        self.dc2 = DirectionalCouplerCoupledMode(wavelength_points, L=0.00000195)
        
    def forward(self, x):
        """
        x: input tensor of shape (batch_size, 4, N)
        """
        x = self.dc1(x)      # First directional coupler
        x = self.delay(x)    # Delay line
        x = self.dc2(x)      # Second directional coupler
        return x

# Example usage
#%%
# Define wavelength points
wavelength_points = np.linspace(1.5e-6, 1.6e-6, 5000)  # 1000 points from 1.5 to 1.6 μm

# Create network
mzi = MZINetwork(wavelength_points, delta_L=300e-6)

# Create example input (batch_size=1, 4 channels, N wavelength points)
batch_size = 1
N = len(wavelength_points)
input_field = torch.zeros(batch_size, 4, N)
input_field[:, 0, :] = 1.0  # Set real part of waveguide 1 to 1

# Forward pass
output_field = mzi(input_field)

# Print shapes and parameters
print(f"Input shape: {input_field.shape}")
print(f"Output shape: {output_field.shape}")
print("\nLearnable parameters:")
for name, param in mzi.named_parameters():
    print(f"{name}: {param.shape}")

# Convert to complex representation for visualization
input_complex_1 = torch.complex(input_field[0, 0], input_field[0, 1])
input_complex_2 = torch.complex(input_field[0, 2], input_field[0, 3])
output_complex_1 = torch.complex(output_field[0, 0], output_field[0, 1])
output_complex_2 = torch.complex(output_field[0, 2], output_field[0, 3])

# Plot results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.semilogy(wavelength_points * 1e9, (torch.abs(input_complex_1).detach().numpy())**2, label='Input WG1')
# plt.semilogy(wavelength_points * 1e9, (torch.abs(input_complex_2).detach().numpy())**2, label='Input WG2')
plt.semilogy(wavelength_points * 1e9, (torch.abs(output_complex_1).detach().numpy())**2, label='Output WG1')
plt.semilogy(wavelength_points * 1e9, (torch.abs(output_complex_2).detach().numpy())**2, label='Output WG2')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Power')
plt.title('MZI Response')
plt.legend()
plt.grid(True)
plt.show()

#%%
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

# Log the network parameters
for name, param in mzi.named_parameters():
    writer.add_histogram(f'{name}_values', param.data)
    if param.grad is not None:
        writer.add_histogram(f'{name}_grads', param.grad)
writer.close()

#%%