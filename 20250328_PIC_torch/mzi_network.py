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
# Accessing Layer Weights and Gradients
print("\nAccessing Layer Weights and Gradients:")
print("-" * 50)

# 1. Access weights and gradients for each layer
for name, module in mzi.named_modules():
    if isinstance(module, (DirectionalCoupler, DelayLine)):
        print(f"\nLayer: {name}")
        print("Parameters:")
        for param_name, param in module.named_parameters():
            print(f"  {param_name}:")
            print(f"    Shape: {param.shape}")
            print(f"    Values: {param.data}")
            print(f"    Gradients: {param.grad if param.grad is not None else 'None'}")

# 2. Create a simple training loop to see gradients
print("\nRunning a training loop to generate gradients:")
print("-" * 50)

# Create a target output (example: we want output power in waveguide 1 to be 1)
target_output = torch.zeros_like(output_field)
target_output[:, 0, :] = 1.0  # Set real part of waveguide 1 to 1

# Create optimizer
optimizer = torch.optim.Adam(mzi.parameters(), lr=0.001)

# One training step
optimizer.zero_grad()
output = mzi(input_field)
loss = torch.mean((output - target_output) ** 2)
loss.backward()

# Now print gradients after backpropagation
print("\nGradients after backpropagation:")
print("-" * 50)
for name, param in mzi.named_parameters():
    print(f"\n{name}:")
    print(f"  Gradient: {param.grad}")

# 3. Access specific layer parameters
print("\nAccessing specific layer parameters:")
print("-" * 50)

# Directional Coupler 1
print("\nDirectional Coupler 1:")
print(f"Coupling Matrix:\n{mzi.dc1.coupling_matrix.data}")
print(f"Phase Parameters:\n{mzi.dc1.phase.data}")

# Delay Line
print("\nDelay Line:")
print(f"Delta L: {mzi.delay.delta_L.data}")
print(f"Effective Index: {mzi.delay.n_eff.data}")
print(f"Phase Parameters:\n{mzi.delay.phase.data}")

# Directional Coupler 2
print("\nDirectional Coupler 2:")
print(f"Coupling Matrix:\n{mzi.dc2.coupling_matrix.data}")
print(f"Phase Parameters:\n{mzi.dc2.phase.data}")

# 4. Visualize parameter distributions
plt.figure(figsize=(15, 10))

# Plot coupling matrices
plt.subplot(2, 2, 1)
plt.imshow(torch.abs(mzi.dc1.coupling_matrix.data).numpy())
plt.colorbar()
plt.title('DC1 Coupling Matrix Magnitude')

plt.subplot(2, 2, 2)
plt.imshow(torch.abs(mzi.dc2.coupling_matrix.data).numpy())
plt.colorbar()
plt.title('DC2 Coupling Matrix Magnitude')

# Plot phase parameters
plt.subplot(2, 2, 3)
plt.bar(['DC1 Phase 1', 'DC1 Phase 2'], mzi.dc1.phase.data.numpy())
plt.title('DC1 Phase Parameters')

plt.subplot(2, 2, 4)
plt.bar(['Delay Phase 1', 'Delay Phase 2'], mzi.delay.phase.data.numpy())
plt.title('Delay Line Phase Parameters')

plt.tight_layout()
plt.show()

# 5. Log parameter changes to TensorBoard
for name, param in mzi.named_parameters():
    if torch.is_complex(param.data):
        # For complex parameters (coupling matrices), log magnitude
        writer.add_scalar(f'parameters/{name}_magnitude', torch.abs(param.data).mean().item(), 0)
        writer.add_scalar(f'parameters/{name}_phase', torch.angle(param.data).mean().item(), 0)
    else:
        # For real parameters (phases, delta_L, n_eff), log directly
        writer.add_scalar(f'parameters/{name}_value', param.data.mean().item(), 0)
    
    if param.grad is not None:
        if torch.is_complex(param.grad):
            # For complex gradients, log magnitude
            writer.add_scalar(f'parameters/{name}_grad_magnitude', torch.abs(param.grad).mean().item(), 0)
            writer.add_scalar(f'parameters/{name}_grad_phase', torch.angle(param.grad).mean().item(), 0)
        else:
            # For real gradients, log directly
            writer.add_scalar(f'parameters/{name}_gradient', param.grad.mean().item(), 0)
writer.close()

#%%
# Accessing MZI Network Weights
print("\nAccessing MZI Network Weights:")
print("-" * 50)

# Access weights for each layer
print("\nDirectional Coupler 1:")
print("Coupling Matrix:\n", mzi.dc1.coupling_matrix.data)
print("Phase Parameters:\n", mzi.dc1.phase.data)

print("\nDelay Line:")
print("Delta L:", mzi.delay.delta_L.data)
print("Effective Index:", mzi.delay.n_eff.data)
print("Phase Parameters:\n", mzi.delay.phase.data)

print("\nDirectional Coupler 2:")
print("Coupling Matrix:\n", mzi.dc2.coupling_matrix.data)
print("Phase Parameters:\n", mzi.dc2.phase.data)

# Visualize the weights
plt.figure(figsize=(15, 5))

# Plot coupling matrices
plt.subplot(1, 3, 1)
plt.imshow(torch.abs(mzi.dc1.coupling_matrix.data).numpy(), cmap='viridis')
plt.colorbar()
plt.title('DC1 Coupling Matrix')

plt.subplot(1, 3, 2)
plt.imshow(torch.abs(mzi.dc2.coupling_matrix.data).numpy(), cmap='viridis')
plt.colorbar()
plt.title('DC2 Coupling Matrix')

# Plot phase parameters
plt.subplot(1, 3, 3)
plt.bar(['DC1 Phase 1', 'DC1 Phase 2', 'Delay Phase 1', 'Delay Phase 2'],
        torch.cat([mzi.dc1.phase.data, mzi.delay.phase.data]).numpy())
plt.title('Phase Parameters')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# %%
